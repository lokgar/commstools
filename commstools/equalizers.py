"""
Adaptive and block equalization algorithms.

This module provides production-grade equalizer implementations for
compensating inter-symbol interference (ISI) and channel distortion
in digital communication systems.

Adaptive algorithms (LMS, RLS, CMA) support two execution backends:

* **Numba**: compiles the sequential loop to native LLVM via ``@njit``.
  Eliminates Python interpreter overhead and enables SIMD vectorisation of
  inner tap loops. Typically 2-5x faster than JAX on CPU for SISO/small-MIMO
  sequences where scan serialisation dominates.
* **JAX**: uses ``jax.lax.scan`` for compiled sequential weight
  updates on both CPU and GPU. Enables end-to-end automatic differentiation
  through the equalizer for gradient-based learning pipelines.

Block algorithms (ZF/MMSE) use NumPy/CuPy dispatch for vectorized
frequency-domain Overlap-Save processing.

All adaptive equalizers support a **butterfly MIMO** topology: for an input
of shape ``(C, N)``, the equalizer maintains a ``(C, C, num_taps)`` weight
matrix so that each output stream is a filtered combination of *all* input
streams. This enables cross-channel interference cancellation (e.g.
dual-polarization demultiplexing in coherent optical, spatial MIMO demux).

Input SPS Convention
--------------------
Adaptive equalizers require T/2-spaced input (2 samples/symbol), the
industry standard for coherent optical and many wireless systems. The
equalizer outputs one symbol per 2 input samples, decimating to symbol rate.
A ``ValueError`` is raised if ``sps != 2``.

Functions
---------
lms :
    Least Mean Squares / Normalized LMS adaptive equalizer.
rls :
    Recursive Least Squares adaptive equalizer.
cma :
    Constant Modulus Algorithm blind equalizer.
zf_equalizer :
    Zero-Forcing / MMSE frequency-domain block equalizer.
"""

import functools
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .backend import ArrayType, _get_jax, dispatch, from_jax, to_jax, to_device
from .logger import logger


# ============================================================================
# RESULT CONTAINER
# ============================================================================


@dataclass
class EqualizerResult:
    """Container for equalizer outputs.

    Attributes
    ----------
    y_hat : ArrayType
        Equalized output symbols. Shape: ``(N_sym,)`` or ``(C, N_sym)``.
    weights : ArrayType
        Final tap weight vector. Shape: ``(num_taps,)`` for SISO
        or ``(C, C, num_taps)`` for MIMO butterfly.
    error : ArrayType
        Error signal history. Shape: ``(N_sym,)`` or ``(C, N_sym)``.
    weights_history : ArrayType or None
        Tap weight evolution over time. Only populated when
        ``store_weights=True``.
    num_train_symbols : int
        Number of data-aided training symbols actually consumed by the equalizer.
        This value is used to discard transients (the training portion) before
        computing steady-state metrics like EVM, SNR, and BER on the blind
        decision-directed portion of the output.
    """

    y_hat: ArrayType
    weights: ArrayType
    error: ArrayType
    weights_history: Optional[ArrayType] = None
    num_train_symbols: int = 0


# ============================================================================
# NUMBA LAZY LOADER
# ============================================================================

_NUMBA_CACHE: dict = {}


def _get_numba():
    """Lazy loader for Numba.

    Returns the ``numba`` module if installed, else ``None``.
    """
    if "numba" not in _NUMBA_CACHE:
        try:
            import numba  # noqa: PLC0415

            _NUMBA_CACHE["numba"] = numba
        except ImportError:
            _NUMBA_CACHE["numba"] = None
    return _NUMBA_CACHE.get("numba")


# ============================================================================
# NUMBA KERNELS — ADAPTIVE EQUALIZER LOOPS
# ============================================================================
#
# Each factory lazily compiles a @njit kernel on first call and caches it in
# _NUMBA_KERNELS.  Numba specialises over argument *types* (all complex64/
# float32/int32/bool), so a single compiled version handles any (C, T, M).
#
# All kernels share the same calling conventions:
#   • x_padded   : (C, N_pad) complex64, C-contiguous
#   • W          : (C, C, T)  complex64, modified **in-place** — caller owns
#   • y_out      : (N_sym, C) complex64, pre-allocated output
#   • e_out      : (N_sym, C) complex64, pre-allocated output
#   • w_hist_out : (N_sym, C, C, T) if store_weights=True, else (1, C, C, T)
#
# Performance tricks:
#   • fastmath=True   — LLVM unsafe-math: enables SIMD vectorisation of
#                       inner tap loops (AVX-512 on modern CPUs)
#   • cache=True      — compiled LLVM IR cached to __pycache__ (*.nbi/*.nbc)
#   • Pre-allocated temporaries (X_wins, y, e, x_bar, Px …) — no heap
#                       allocation inside the hot symbol loop
#   • Manual argmin slicer — explicit loop over M; LLVM can SIMD-vectorise
#                       the distance computation for small constellations
#   • In-place W update — no copy per step, minimal working-set pressure
#   • np.conj() on scalars — valid in Numba 0.64+

_NUMBA_KERNELS: dict = {}


def _get_numba_lms():
    """JIT-compile and cache the Numba NLMS butterfly loop kernel.

    Returns
    -------
    lms_loop : numba-compiled callable
        See kernel source for argument shapes and semantics.
    """
    if "lms" not in _NUMBA_KERNELS:
        numba_mod = _get_numba()
        if numba_mod is None:
            raise ImportError("Numba is required for backend='numba'.")

        @numba_mod.njit(cache=True, fastmath=True, nogil=True)
        def lms_loop(
            x_padded,
            training,
            constellation,
            W,
            step_size,
            n_train,
            stride,
            store_weights,
            y_out,
            e_out,
            w_hist_out,
        ):
            # x_padded      : (C, N_pad)          complex64
            # training      : (C, N_sym)           complex64
            # constellation : (M,)                 complex64
            # W             : (C, C, num_taps)      complex64 — modified in-place
            # step_size     : float32              — NLMS mu ∈ (0, 2)
            # n_train       : int32                — training/DD boundary
            # stride        : int                  — sps (== 2 for T/2-spaced)
            # store_weights : bool
            # y_out         : (N_sym, C)            complex64 — pre-allocated
            # e_out         : (N_sym, C)            complex64 — pre-allocated
            # w_hist_out    : (N_sym or 1, C, C, T) complex64 — pre-allocated
            C = W.shape[0]
            num_taps = W.shape[2]
            n_sym = y_out.shape[0]
            M = len(constellation)

            X_wins = np.empty((C, num_taps), dtype=np.complex64)
            y = np.empty(C, dtype=np.complex64)
            e = np.empty(C, dtype=np.complex64)

            for idx in range(n_sym):
                sample_idx = idx * stride

                # Window extraction
                for c in range(C):
                    for t in range(num_taps):
                        X_wins[c, t] = x_padded[c, sample_idx + t]

                # Butterfly forward pass: y[i] = Σ_j conj(W[i,j]) · X_wins[j]
                for i in range(C):
                    acc = np.complex64(0.0)
                    for j in range(C):
                        for t in range(num_taps):
                            acc = acc + np.conj(W[i, j, t]) * X_wins[j, t]
                    y[i] = acc
                    y_out[idx, i] = acc

                # Desired symbol: training or decision-directed slicer
                for i in range(C):
                    if idx < n_train:
                        d_i = training[i, idx]
                    else:
                        min_dist = np.float32(1e38)
                        min_idx = 0
                        for k in range(M):
                            diff = y[i] - constellation[k]
                            dist = diff.real * diff.real + diff.imag * diff.imag
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = k
                        d_i = constellation[min_idx]
                    e[i] = d_i - y[i]
                    e_out[idx, i] = e[i]

                # NLMS: normalise by instantaneous input power
                power = np.float32(1e-10)
                for j in range(C):
                    for t in range(num_taps):
                        v = X_wins[j, t]
                        power = power + v.real * v.real + v.imag * v.imag
                mu_eff = step_size / power

                # Weight update: W[i,j,t] += mu_eff * conj(e[i]) * X_wins[j,t]
                for i in range(C):
                    ce_i = np.conj(e[i])
                    for j in range(C):
                        for t in range(num_taps):
                            W[i, j, t] = W[i, j, t] + mu_eff * ce_i * X_wins[j, t]

                if store_weights:
                    for i in range(C):
                        for j in range(C):
                            for t in range(num_taps):
                                w_hist_out[idx, i, j, t] = W[i, j, t]

        _NUMBA_KERNELS["lms"] = lms_loop
    return _NUMBA_KERNELS["lms"]


def _get_numba_rls():
    """JIT-compile and cache the Numba Leaky-RLS butterfly loop kernel.

    Returns
    -------
    rls_loop : numba-compiled callable
        See kernel source for argument shapes and semantics.
    """
    if "rls" not in _NUMBA_KERNELS:
        numba_mod = _get_numba()
        if numba_mod is None:
            raise ImportError("Numba is required for backend='numba'.")

        @numba_mod.njit(cache=True, fastmath=True, nogil=True)
        def rls_loop(
            x_padded,
            training,
            constellation,
            W,
            P,
            lam,
            leakage,
            n_train,
            n_update_halt,
            stride,
            store_weights,
            y_out,
            e_out,
            w_hist_out,
        ):
            # x_padded      : (C, N_pad)                  complex64
            # training      : (C, N_sym)                   complex64
            # constellation : (M,)                         complex64
            # W             : (C, C, num_taps)              complex64 — in-place
            # P             : (C*num_taps, C*num_taps)      complex64 — in-place
            # lam           : float32  — forgetting factor λ ∈ (0, 1]
            # leakage       : float32  — weight-decay coefficient γ ∈ [0, 1)
            # n_train       : int32    — training/DD boundary
            # n_update_halt : int32    — freeze W,P beyond this index
            # stride        : int      — sps
            # store_weights : bool
            # y_out         : (N_sym, C)                   complex64
            # e_out         : (N_sym, C)                   complex64
            # w_hist_out    : (N_sym or 1, C, C, T)        complex64
            C = W.shape[0]
            num_taps = W.shape[2]
            n_sym = y_out.shape[0]
            N = C * num_taps  # regressor dimension
            M = len(constellation)

            x_bar = np.empty(N, dtype=np.complex64)
            Px = np.empty(N, dtype=np.complex64)
            xH_P = np.empty(N, dtype=np.complex64)
            k = np.empty(N, dtype=np.complex64)
            y = np.empty(C, dtype=np.complex64)
            e = np.empty(C, dtype=np.complex64)

            lam_f32 = np.float32(lam)
            leak_term = np.float32(1.0) - np.float32(leakage)

            for idx in range(n_sym):
                sample_idx = idx * stride

                # Flatten regressor directly from input
                for j in range(C):
                    for t in range(num_taps):
                        x_bar[j * num_taps + t] = x_padded[j, sample_idx + t]

                # Butterfly forward pass
                for i in range(C):
                    acc = np.complex64(0.0)
                    for j in range(C):
                        for t in range(num_taps):
                            acc = acc + np.conj(W[i, j, t]) * x_bar[j * num_taps + t]
                    y[i] = acc
                    y_out[idx, i] = acc

                # Desired and error
                for i in range(C):
                    if idx < n_train:
                        d_i = training[i, idx]
                    else:
                        min_dist = np.float32(1e38)
                        min_idx = 0
                        for kk in range(M):
                            diff = y[i] - constellation[kk]
                            dist = diff.real * diff.real + diff.imag * diff.imag
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = kk
                        d_i = constellation[min_idx]
                    e[i] = d_i - y[i]
                    e_out[idx, i] = e[i]

                # Kalman gain: Px = P @ x_bar
                Px = np.dot(P, x_bar)

                # denom = λ + real(conj(x_bar) · Px)
                denom = lam_f32
                for jj in range(N):
                    denom = denom + (np.conj(x_bar[jj]) * Px[jj]).real

                # k = Px / denom
                inv_denom = np.float32(1.0) / denom
                for ii in range(N):
                    k[ii] = Px[ii] * inv_denom

                if idx < n_update_halt:
                    # Leaky W update: W[i,j,t] = (1−γ)W[i,j,t] + k[j*T+t]*conj(e[i])
                    for i in range(C):
                        ce_i = np.conj(e[i])
                        for j in range(C):
                            for t in range(num_taps):
                                W[i, j, t] = (
                                    leak_term * W[i, j, t] + k[j * num_taps + t] * ce_i
                                )

                    # Hermitian symmetry reduction: P is Hermitian, so x^H P = (P^H x)^H = (P x)^H
                    # We already computed Px = P @ x_bar. Thus xH_P is simply conj(Px),
                    # reducing the Riccati rank-1 update complexity by O(N^2).
                    for jj in range(N):
                        xH_P[jj] = np.conj(Px[jj])

                    # Riccati: P = (P - outer(k, xH_P)) / λ
                    for ii in range(N):
                        for jj in range(N):
                            P[ii, jj] = (P[ii, jj] - k[ii] * xH_P[jj]) / lam_f32

                    # Hermitian re-symmetrization: P ← (P + Pᴴ)/2
                    # The 1/λ division amplifies asymmetry from FP rounding each
                    # step. Without re-symmetrization, accumulated asymmetry
                    # causes P eigenvalues to diverge in float32 for λ < 1.
                    for ii in range(N):
                        for jj in range(ii + 1, N):
                            avg = (P[ii, jj] + np.conj(P[jj, ii])) * np.float32(0.5)
                            P[ii, jj] = avg
                            P[jj, ii] = np.conj(avg)
                        P[ii, ii] = np.complex64(P[ii, ii].real)

                if store_weights:
                    for i in range(C):
                        for j in range(C):
                            for t in range(num_taps):
                                w_hist_out[idx, i, j, t] = W[i, j, t]

        _NUMBA_KERNELS["rls"] = rls_loop
    return _NUMBA_KERNELS["rls"]


def _get_numba_cma():
    """JIT-compile and cache the Numba CMA butterfly loop kernel.

    Returns
    -------
    cma_loop : numba-compiled callable
        See kernel source for argument shapes and semantics.
    """
    if "cma" not in _NUMBA_KERNELS:
        numba_mod = _get_numba()
        if numba_mod is None:
            raise ImportError("Numba is required for backend='numba'.")

        @numba_mod.njit(cache=True, fastmath=True, nogil=True)
        def cma_loop(
            x_padded,
            W,
            step_size,
            r2,
            stride,
            store_weights,
            y_out,
            e_out,
            w_hist_out,
        ):
            # x_padded   : (C, N_pad)          complex64
            # W          : (C, C, num_taps)     complex64 — modified in-place
            # step_size  : float32             — fixed CMA gradient step μ
            # r2         : float32             — Godard radius R² = E[|s|⁴]/E[|s|²]
            # stride     : int                 — sps
            # store_weights : bool
            # y_out      : (N_sym, C)           complex64
            # e_out      : (N_sym, C)           complex64
            # w_hist_out : (N_sym or 1, C, C, T) complex64
            C = W.shape[0]
            num_taps = W.shape[2]
            n_sym = y_out.shape[0]

            X_wins = np.empty((C, num_taps), dtype=np.complex64)
            y = np.empty(C, dtype=np.complex64)
            e = np.empty(C, dtype=np.complex64)

            for idx in range(n_sym):
                sample_idx = idx * stride

                # Window extraction
                for c in range(C):
                    for t in range(num_taps):
                        X_wins[c, t] = x_padded[c, sample_idx + t]

                # Butterfly forward pass
                for i in range(C):
                    acc = np.complex64(0.0)
                    for j in range(C):
                        for t in range(num_taps):
                            acc = acc + np.conj(W[i, j, t]) * X_wins[j, t]
                    y[i] = acc
                    y_out[idx, i] = acc

                # CMA error: e[i] = y[i] * (|y[i]|² − R²)
                # Real-valued |y|² prevents imaginary leakage from FP noise
                # from causing a parasitic phase rotation via multiplication by y.
                for i in range(C):
                    mod2 = y[i].real * y[i].real + y[i].imag * y[i].imag
                    e[i] = y[i] * np.float32(mod2 - r2)
                    e_out[idx, i] = e[i]

                # Weight update (gradient descent): W -= mu * conj(e[i]) * X_wins[j,t]
                for i in range(C):
                    ce_i = np.conj(e[i])
                    for j in range(C):
                        for t in range(num_taps):
                            W[i, j, t] = W[i, j, t] - step_size * ce_i * X_wins[j, t]

                if store_weights:
                    for i in range(C):
                        for j in range(C):
                            for t in range(num_taps):
                                w_hist_out[idx, i, j, t] = W[i, j, t]

        _NUMBA_KERNELS["cma"] = cma_loop
    return _NUMBA_KERNELS["cma"]


# ============================================================================
# JAX KERNELS — ADAPTIVE EQUALIZER SCANS
# ============================================================================
#
# Each factory JIT-compiles a jax.lax.scan kernel on first call and caches
# it in _JITTED_EQ.  Static closure variables are baked into the compiled XLA
# program — changing any of them produces a cache miss and triggers a new
# trace+compile (not a retrace of an existing graph).
#
# Static variables (shared by all three kernels):
#   num_taps : FIR length per polyphase arm — fixes XLA buffer allocation
#   stride   : decimation factor (== sps, typically 2 for T/2-spaced input)
#   num_ch   : MIMO butterfly width C — fixes matrix shapes in XLA IR
#
# Kernel-specific static variable:
#   const_size : constellation size M for LMS/RLS — fixes the slicer argmin
#                shape at trace time so XLA can compile the min-distance
#                search without dynamic dispatch.  CMA is blind (no slicer),
#                so this variable is omitted from its key.
#
# All kernels share the same output convention:
#   y_hat  : (N_sym, C)              complex64 — equalized symbols
#              (transposed by _unpack_result_jax to (C, N_sym) before return)
#   errors : (N_sym, C)              complex64 — complex errors d − y
#   w_hist : (N_sym, C, C, num_taps) complex64 — weight snapshots per symbol
#
# Performance notes:
#   • jax.jit + XLA ahead-of-time compilation eliminates Python overhead for
#     the scan loop; each symbol step is a single XLA op dispatch.
#   • jax.vmap(get_win) vectorises window extraction across C channels.
#   • jnp.einsum('ijt,jt->i') compiles to an optimised GEMV via OpenBLAS
#     on CPU or cuBLAS on GPU.
#   • lax.dynamic_slice emits a single gather op with runtime offset and
#     compile-time slice size — no Python-level indexing overhead.
#   • For GPU: lax.scan serialises the symbol loop on a single CUDA stream,
#     limiting parallelism to the per-step GEMV.  GPU is only beneficial
#     for large MIMO widths (C >> 4) that saturate the cuBLAS kernel.
#     Use backend='numba' for CPU-optimal throughput on typical SISO/2x2.

_JITTED_EQ = {}


def _get_jax_lms(num_taps, stride, const_size, num_ch):
    """JIT-compile and cache the sample-by-sample NLMS butterfly scan.

    Static closure variables (baked into the compiled kernel; a new cache
    entry is created — not a retrace — when any of these change):

    num_taps   : FIR filter length per polyphase arm.
    stride     : decimation factor (== sps, typically 2 for T/2-spaced input).
    const_size : constellation size M — fixes the slicer ``argmin`` shape at
                 trace time so XLA can compile it without dynamic dispatch.
    num_ch     : MIMO butterfly width C (number of input/output channels).

    Returns
    -------
    lms_scan : JIT-compiled callable
        See the inner function for the call signature.
    """
    key = ("lms", num_taps, stride, const_size, num_ch)
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()

        @jax.jit
        def lms_scan(
            x_input, training_padded, constellation, w_init, step_size, n_train
        ):
            # Argument shapes and semantics
            # ------------------------------
            # x_input         : (C, N_pad)        complex64 — zero-padded received samples
            #                    N_pad = n_sym_padded * stride + num_taps - 1
            # training_padded : (C, N_sym)         complex64 — reference symbols,
            #                    zeros beyond column n_train (DD region)
            # constellation   : (M,)               complex64 — slicer lookup table
            # w_init          : (C, C, num_taps)   complex64 — initial butterfly filter
            # step_size       : scalar float32     — NLMS mu, stable in (0, 2)
            # n_train         : scalar int32       — training/DD boundary (dynamic)
            #
            # lax.scan carry  : W  (C, C, num_taps) — butterfly weight matrix
            # lax.scan xs     : jnp.arange(N_sym)   — symbol indices 0..N_sym-1
            # lax.scan output : y_hat   (N_sym, C)              equalized symbols
            #                   errors  (N_sym, C)              complex errors d - y
            #                   w_hist  (N_sym, C, C, num_taps) weight snapshots

            def step(W, idx):
                sample_idx = idx * stride

                X_wins = jax.lax.dynamic_slice(
                    x_input, (0, sample_idx), (num_ch, num_taps)
                )

                # Butterfly: y_i = sum_j conj(W[i,j]) . X_wins[j]
                y = jnp.einsum("ijt,jt->i", jnp.conj(W), X_wins)  # (C,)

                # Training or decision-directed
                def slicer(ch_y):
                    return constellation[jnp.argmin(jnp.abs(ch_y - constellation) ** 2)]

                dd = jax.vmap(slicer)(y)  # (C,)
                d = jnp.where(idx < n_train, training_padded[:, idx], dd)

                e = d - y  # (C,)

                # NLMS: normalize by instantaneous input power for convergence
                # robustness across input-power variations. step_size interpreted
                # as normalized mu in (0, 2) — typical range: 0.01 to 0.1.
                power = jnp.real(jnp.sum(X_wins * jnp.conj(X_wins))) + 1e-10
                mu_eff = step_size / power

                W_new = W + mu_eff * jnp.einsum("i,jt->ijt", jnp.conj(e), X_wins)
                return W_new, (y, e, W_new)

            n_sym = training_padded.shape[1]
            W_final, (y_hat, errors, w_hist) = jax.lax.scan(
                step, w_init, jnp.arange(n_sym)
            )
            return y_hat, errors, W_final, w_hist

        _JITTED_EQ[key] = lms_scan
    return _JITTED_EQ[key]


def _get_jax_rls(num_taps, stride, const_size, num_ch):
    """JIT-compile and cache the sample-by-sample Leaky-RLS butterfly scan.

    Static closure variables (same semantics as ``_get_jitted_lms``):
    num_taps, stride, const_size, num_ch.

    Returns
    -------
    rls_scan : JIT-compiled callable
        See the inner function for the call signature.
    """
    key = ("rls", num_taps, stride, const_size, num_ch)
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()

        @jax.jit
        def rls_scan(
            x_input,
            training_padded,
            constellation,
            w_init,
            P_init,
            lam,
            n_train,
            leakage,
            n_update_halt,
        ):
            # Argument shapes and semantics
            # ------------------------------
            # x_input         : (C, N_pad)              complex64 — padded received samples
            # training_padded : (C, N_sym)               complex64 — reference symbols
            # constellation   : (M,)                    complex64 — slicer lookup table
            # w_init          : (C, C, num_taps)         complex64 — initial butterfly weights
            # P_init          : (C*num_taps, C*num_taps) complex64 — initial inverse
            #                    correlation matrix P = (1/delta) * I
            # lam             : scalar float32           — forgetting factor λ ∈ (0,1]
            # n_train         : scalar int32             — training/DD boundary (dynamic)
            # leakage         : scalar float32           — weight-decay coefficient γ ∈ [0,1):
            #                    W ← (1−γ)W + k⊗ē  each step; P update is unchanged.
            #                    Decays null-subspace weights without inflating P eigenvalues.
            # n_update_halt   : scalar int32             — freeze W and P beyond this index;
            #                    = n_sym - num_taps//2 to avoid zero-padding contamination
            #
            # lax.scan carry  : (W, P)
            #   W : (C, C, num_taps)         butterfly weight matrix
            #   P : (C*num_taps, C*num_taps) inverse input correlation matrix
            # lax.scan xs     : jnp.arange(N_sym)
            # lax.scan output : y_hat   (N_sym, C)              equalized symbols
            #                   errors  (N_sym, C)              complex errors d - y
            #                   w_hist  (N_sym, C, C, num_taps) weight snapshots
            #
            # Per-step equations (idx is the current symbol index):
            #   X_wins = [x_input[c, idx*stride : idx*stride+num_taps] for c]  (C, T)
            #   x_bar  = X_wins.flatten()                                        (C*T,)
            #   y      = einsum('ijt,jt->i', conj(W), X_wins)                   (C,)
            #   d      = training[:, idx] if idx < n_train else slicer(y)        (C,)
            #   e      = d - y                                                   (C,)
            #   Px     = P @ x_bar                                               (C*T,)
            #   k      = Px / (λ + x_bar^H @ Px)                                (C*T,) Kalman gain
            #   W     ← (1−γ)W + k ⊗ conj(e)              (if idx < n_update_halt)
            #   P      = (P - outer(k, x_bar^H P)) / λ     (if idx < n_update_halt)

            def step(carry, idx):
                W, P = carry
                sample_idx = idx * stride

                X_wins = jax.lax.dynamic_slice(
                    x_input, (0, sample_idx), (num_ch, num_taps)
                )

                y = jnp.einsum("ijt,jt->i", jnp.conj(W), X_wins)

                def slicer(ch_y):
                    return constellation[jnp.argmin(jnp.abs(ch_y - constellation) ** 2)]

                dd = jax.vmap(slicer)(y)
                d = jnp.where(idx < n_train, training_padded[:, idx], dd)
                e = d - y

                x_bar = X_wins.flatten()  # (C * num_taps,)

                Px = P @ x_bar
                denom = lam + jnp.real(jnp.dot(jnp.conj(x_bar), Px))
                k = Px / denom

                def w_update(w_row, err_val):
                    w_flat = w_row.flatten()
                    # Weight decay: suppress null-subspace taps exponentially.
                    # Adding γI directly to P would inflate P eigenvalues, making
                    # R_xx more singular and worsening AWGN amplification.
                    w_flat_new = (1.0 - leakage) * w_flat + k * jnp.conj(err_val)
                    return w_flat_new.reshape(num_ch, num_taps)

                W_upd = jax.vmap(w_update)(W, e)
                # Standard Riccati update — no diagonal loading on P.
                # Exploit Hermitian symmetry: x^H P = (P^H x)^H = (P x)^H = conj(Px)^T
                # Reduces Riccati update by an O(N^2) mat-vec multiplication.
                P_upd = (P - jnp.outer(k, jnp.conj(Px))) / lam
                # Hermitian re-symmetrization: P ← (P + Pᴴ)/2
                # Prevents asymmetry drift from the 1/λ amplification.
                P_upd = jnp.float32(0.5) * (P_upd + jnp.conj(P_upd).T)

                # Early halt: freeze W and P once the sliding window begins
                # overlapping the right zero-padding (last num_taps//2 symbols).
                # The forward pass (y) continues so all output symbols are produced.
                update_ok = idx < n_update_halt
                W_new = jnp.where(update_ok, W_upd, W)
                P_new = jnp.where(update_ok, P_upd, P)

                return (W_new, P_new), (y, e, W_new)

            n_sym = training_padded.shape[1]
            (W_final, _), (y_hat, errors, w_hist) = jax.lax.scan(
                step, (w_init, P_init), jnp.arange(n_sym)
            )
            return y_hat, errors, W_final, w_hist

        _JITTED_EQ[key] = rls_scan
    return _JITTED_EQ[key]


def _get_jax_cma(num_taps, stride, num_ch):
    """JIT-compile and cache the sample-by-sample CMA butterfly scan.

    Static closure variables: num_taps, stride, num_ch (same as LMS/RLS).
    No constellation required — CMA is a blind algorithm.

    Returns
    -------
    cma_scan : JIT-compiled callable
        See the inner function for the call signature.
    """
    key = ("cma", num_taps, stride, num_ch)
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()

        # n_sym (arg 4) is static: lax.scan requires a compile-time iteration
        # count.  JAX's bounded LRU cache retraces when n_sym changes, keeping
        # _JITTED_EQ from growing without bound.
        @functools.partial(jax.jit, static_argnums=(4,))
        def cma_scan(x_input, w_init, step_size, r2, n_sym):
            # Argument shapes and semantics
            # ------------------------------
            # x_input   : (C, N_pad)       complex64 — padded received samples
            # w_init    : (C, C, num_taps) complex64 — initial butterfly weights
            # step_size : scalar float32   — fixed gradient step μ (no NLMS; non-convex surface)
            # r2        : scalar float32   — Godard dispersion radius R² = E[|s|⁴] / E[|s|²]
            # n_sym     : int (static)     — total symbol count; fixes scan iteration count
            #
            # lax.scan carry  : W  (C, C, num_taps)
            # lax.scan xs     : jnp.arange(n_sym)
            # lax.scan output : y_hat   (n_sym, C)
            #                   errors  (n_sym, C)   Godard errors y*(|y|²−R²)
            #                   w_hist  (n_sym, C, C, num_taps)
            #
            # Per-step gradient descent (Godard criterion):
            #   y = einsum('ijt,jt->i', conj(W), X_wins)            (C,)
            #   e = y * (real(y * conj(y)) - R²)                    (C,)  CMA error
            #   W -= μ * einsum('i,jt->ijt', conj(e), X_wins)               gradient step
            # Note: real() is required to prevent imaginary leakage from
            # floating-point noise in |y|² from causing parasitic phase rotation.
            def step(W, idx):
                sample_idx = idx * stride

                X_wins = jax.lax.dynamic_slice(
                    x_input, (0, sample_idx), (num_ch, num_taps)
                )
                y = jnp.einsum("ijt,jt->i", jnp.conj(W), X_wins)

                # CMA error: e_i = y_i * (|y_i|^2 - R2)
                # jnp.real enforces strict real-valued modulus: floating-point noise
                # in y*conj(y) would otherwise inject imaginary components, causing
                # a parasitic phase rotation into the gradient via multiplication by y.
                e = y * (jnp.real(y * jnp.conj(y)) - r2)

                W_new = W - step_size * jnp.einsum("i,jt->ijt", jnp.conj(e), X_wins)
                return W_new, (y, e, W_new)

            W_final, (y_hat, errors, w_hist) = jax.lax.scan(
                step, w_init, jnp.arange(n_sym)
            )
            return y_hat, errors, W_final, w_hist

        _JITTED_EQ[key] = cma_scan
    return _JITTED_EQ[key]


# ============================================================================
# SHARED HELPERS
# ============================================================================


def _normalize_inputs_jax(samples, training_symbols, sps, xp):
    """Scale samples and training symbols to a common unit-power reference.

    For fractionally-spaced equalizers (sps=2) the fractional timing phase
    is unknown.  Strided power measurement ``samples[..., ::sps]`` is unsafe
    because it can land on zero-crossings of the Nyquist pulse, severely
    underestimating signal power and destabilising adaptation.

    Instead we use the *wideband* power.  For a unit-energy Nyquist pulse
    the total discrete power is ``E[|s|²] / sps``, so the symbol-rate RMS
    is ``global_rms * sqrt(sps)``.  This estimate is phase-invariant.

    Parameters
    ----------
    samples         : (C, N) or (N,)  complex on any backend (NumPy / CuPy)
    training_symbols: (C, K) or (K,)  or None
    sps             : int — samples per symbol (stride)
    xp              : array module (np or cp)

    Returns
    -------
    samples          : unit symbol-power, same shape/backend
    training_symbols : unit average-power, same shape/backend (or None)
    """
    from commstools.helpers import normalize as c_normalize, rms

    # Robust symbol-power estimate invariant to arbitrary fractional delays
    global_rms = rms(samples, axis=-1, keepdims=True)
    sym_rms = global_rms * xp.sqrt(xp.asarray(sps, dtype=global_rms.dtype))

    # Avoid div by 0 just in case
    sym_rms = xp.where(sym_rms == 0, 1.0, sym_rms)
    samples = samples / sym_rms

    if training_symbols is not None:
        training_symbols = c_normalize(training_symbols, "average_power", axis=-1)

    return samples, training_symbols


def _normalize_inputs_numpy(samples, training_symbols, sps):
    """Scale samples and training symbols to unit power using plain NumPy.

    NumPy counterpart of ``_normalize_inputs_jax`` for use with the Numba
    backend.  No ``xp`` dispatch, no helper imports — operates strictly on
    NumPy arrays.  Uses the same wideband-power estimate as the JAX variant
    to ensure both backends produce identical normalization results.

    Parameters
    ----------
    samples          : (C, N) or (N,)  complex64 NumPy array
    training_symbols : (C, K) or (K,)  complex64 NumPy array, or None
    sps              : int — samples per symbol (stride)

    Returns
    -------
    samples          : unit symbol-power NumPy array, same shape
    training_symbols : unit average-power NumPy array (or None)
    """
    # Robust symbol-power estimate: global RMS * sqrt(sps) — phase-invariant
    global_rms = np.sqrt(np.mean(np.abs(samples) ** 2, axis=-1, keepdims=True))
    sym_rms = global_rms * np.sqrt(np.float32(sps))
    sym_rms = np.where(sym_rms == 0, np.float32(1.0), sym_rms)
    samples = samples / sym_rms

    if training_symbols is not None:
        avg_pwr = np.mean(np.abs(training_symbols) ** 2, axis=-1, keepdims=True)
        scale = np.sqrt(np.where(avg_pwr == 0, np.float32(1.0), avg_pwr))
        training_symbols = training_symbols / scale

    return samples, training_symbols


def _init_butterfly_weights_jax(num_ch, num_taps, jnp, sps=2, center_tap=None):
    """Build center-tap identity butterfly weight matrix as a JAX array.

    Initializes a ``(C, C, num_taps)`` complex64 array where
    ``W[i, i, center] = 1+0j`` for each channel ``i`` and all other entries
    are zero.  This is the canonical identity starting point: at time 0 the
    equalizer passes each channel straight through with unit gain and zero
    delay relative to the center tap.

    Parameters
    ----------
    num_ch     : int — number of input/output channels C
    num_taps   : int — FIR filter length T
    jnp        : JAX numpy module (passed as argument to avoid importing at
                 module level when JAX is unavailable)
    sps        : int — unused; retained for call-site symmetry
    center_tap : int or None — tap index for unit initialization;
                 defaults to ``num_taps // 2``

    Returns
    -------
    W : (C, C, num_taps) complex64 JAX array
    """
    W = jnp.zeros((num_ch, num_ch, num_taps), dtype="complex64")
    center = center_tap if center_tap is not None else num_taps // 2
    W = W.at[jnp.arange(num_ch), jnp.arange(num_ch), center].set(1.0 + 0j)
    return W


def _init_butterfly_weights_numpy(num_ch, num_taps, center_tap=None):
    """Build center-tap identity butterfly weight matrix as a NumPy array.

    NumPy counterpart of ``_init_butterfly_weights_jax`` for use with the
    Numba backend.  Same semantics and output shape; no JAX dependency.

    Parameters
    ----------
    num_ch     : int — number of input/output channels C
    num_taps   : int — FIR filter length T
    center_tap : int or None — tap index for unit initialization;
                 defaults to ``num_taps // 2``

    Returns
    -------
    W : (C, C, num_taps) complex64 NumPy array
    """
    W = np.zeros((num_ch, num_ch, num_taps), dtype=np.complex64)
    center = center_tap if center_tap is not None else num_taps // 2
    for i in range(num_ch):
        W[i, i, center] = 1.0 + 0j
    return W


def _prepare_training_jax(
    training_symbols,
    num_ch,
    n_sym,
    num_train_symbols=None,
):
    """Build the zero-padded training array expected by the JAX scan kernels.

    The kernels index ``training_padded[:, sym_idx]`` at every symbol,
    conditioned on ``sym_idx < n_train``.  Symbols beyond ``n_train_aligned``
    are zero — the kernel ignores them (DD slicer is used instead).

    If ``training_symbols`` is 1-D it is broadcast to all ``num_ch`` channels.
    Any extra training symbols beyond ``n_sym`` are silently clamped.
    The array is kept on the same device as ``training_symbols`` to avoid
    unnecessary CPU round-trips before the ``to_jax()`` transfer.

    Note: callers should pre-slice ``training_symbols`` to ``num_train_symbols``
    *before* calling this function (done in ``lms()``/``rls()``); the
    ``num_train_symbols`` argument here only provides a secondary safety cap.

    Parameters
    ----------
    training_symbols : array or None — (K,) or (C, K), any backend
    num_ch           : int — C
    n_sym            : int — padded symbol count (columns of output array)
    num_train_symbols: int or None — secondary cap on n_train_aligned

    Returns
    -------
    train_full      : (C, n_sym) complex64 on same backend as input (or NumPy)
    n_train_aligned : int — effective number of data-aided symbols
    """
    if training_symbols is not None:
        # Keep training data on its original device
        train_arr, xp, _ = dispatch(training_symbols)
        train_arr = train_arr.astype("complex64")
        if train_arr.ndim == 1:
            train_arr = (
                xp.tile(train_arr[None, :], (num_ch, 1))
                if num_ch > 1
                else train_arr[None, :]
            )
        n_raw = train_arr.shape[1]
        n_train_aligned = max(0, min(n_raw, n_sym))
        if num_train_symbols is not None:
            n_train_aligned = min(n_train_aligned, num_train_symbols)

        train_full = xp.zeros((num_ch, n_sym), dtype="complex64")
        if n_train_aligned > 0:
            train_full[:, :n_train_aligned] = train_arr[:, :n_train_aligned]
    else:
        n_train_aligned = 0
        train_full = np.zeros((num_ch, n_sym), dtype="complex64")

    return train_full, n_train_aligned


def _prepare_training_numpy(
    training_symbols,
    num_ch,
    n_sym,
    num_train_symbols=None,
):
    """Build the zero-padded training array for the Numba scan kernels.

    Pure NumPy implementation — no JAX, CuPy, or ``dispatch`` dependencies.
    The caller must ensure ``training_symbols`` is already a NumPy array
    (or any object accepted by ``np.asarray``).

    Callers should pre-slice ``training_symbols`` to ``num_train_symbols``
    *before* calling this function; the argument here is a secondary safety cap.

    Parameters
    ----------
    training_symbols : (K,) or (C, K) complex64 NumPy array, or None
    num_ch           : int — C
    n_sym            : int — symbol count (columns of output array)
    num_train_symbols: int or None — secondary cap on n_train_aligned

    Returns
    -------
    train_full      : (C, n_sym) complex64 NumPy array
    n_train_aligned : int — effective number of data-aided symbols
    """
    if training_symbols is not None:
        train_arr = np.asarray(training_symbols, dtype=np.complex64)
        if train_arr.ndim == 1:
            train_arr = (
                np.tile(train_arr[None, :], (num_ch, 1))
                if num_ch > 1
                else train_arr[None, :]
            )
        n_raw = train_arr.shape[1]
        n_train_aligned = max(0, min(n_raw, n_sym))
        if num_train_symbols is not None:
            n_train_aligned = min(n_train_aligned, num_train_symbols)

        train_full = np.zeros((num_ch, n_sym), dtype=np.complex64)
        if n_train_aligned > 0:
            train_full[:, :n_train_aligned] = train_arr[:, :n_train_aligned]
    else:
        n_train_aligned = 0
        train_full = np.zeros((num_ch, n_sym), dtype=np.complex64)

    return train_full, n_train_aligned


def _unpack_result_jax(
    y_hat_jax,
    errors_jax,
    W_final_jax,
    w_hist_jax,
    was_1d,
    store_weights,
    n_sym=None,
    xp=np,
    num_train_symbols=0,
):
    """Convert JAX scan outputs into an ``EqualizerResult``.

    Transfers JAX device arrays back to NumPy/CuPy via ``from_jax``, then:
      1. Transposes from ``(N_sym, C)`` scan layout to ``(C, N_sym)`` convention.
      2. Truncates to ``n_sym`` when provided (e.g. RLS early-halt boundary).
      3. Squeezes the channel dimension for 1-D SISO inputs (``was_1d=True``).
      4. Optionally keeps the weight-trajectory array.

    Parameters
    ----------
    y_hat_jax       : (N_sym, C) JAX array — equalized symbols
    errors_jax      : (N_sym, C) JAX array — complex errors
    W_final_jax     : (C, C, num_taps) JAX array — final weights
    w_hist_jax      : (N_sym, C, C, num_taps) JAX array — weight history
    was_1d          : bool — squeeze C=1 dimension for SISO inputs
    store_weights   : bool — if False, ``weights_history`` is None
    n_sym           : int or None — truncation length (None = no truncation)
    xp              : output array module (np or cp)
    num_train_symbols: int — stored in the result for caller reference

    Returns
    -------
    EqualizerResult
    """
    y_hat = xp.asarray(from_jax(y_hat_jax).T)  # (N_sym, C) -> (C, N_sym)
    errors = xp.asarray(from_jax(errors_jax).T)
    W_final = xp.asarray(from_jax(W_final_jax))

    if n_sym is not None:
        y_hat = y_hat[..., :n_sym]
        errors = errors[..., :n_sym]

    if was_1d:
        y_hat = y_hat[0]
        errors = errors[0]
        W_final = W_final[0, 0]

    w_history = None
    if store_weights:
        w_history = xp.asarray(from_jax(w_hist_jax))
        if was_1d:
            w_history = w_history[:, 0, 0, :]

    return EqualizerResult(
        y_hat=y_hat,
        weights=W_final,
        error=errors,
        weights_history=w_history,
        num_train_symbols=num_train_symbols,
    )


def _unpack_result_numpy(
    y_out,
    e_out,
    W_final,
    w_hist,
    was_1d,
    store_weights,
    n_sym=None,
    xp=np,
    num_train_symbols=0,
):
    """Convert Numba kernel outputs (plain NumPy) into an ``EqualizerResult``.

    No ``from_jax`` calls — all inputs are already NumPy arrays produced by
    the Numba kernels.  Same post-processing as ``_unpack_result_jax`` but
    operates directly on NumPy memory without any device transfer overhead.

    Parameters
    ----------
    y_out           : (N_sym, C) complex64 NumPy array — equalized symbols
    e_out           : (N_sym, C) complex64 NumPy array — complex errors
    W_final         : (C, C, num_taps) complex64 NumPy array — final weights
    w_hist          : (N_sym or 1, C, C, num_taps) NumPy — weight history
    was_1d          : bool — squeeze C=1 dimension for SISO inputs
    store_weights   : bool — if False, ``weights_history`` is None
    n_sym           : int or None — truncation length (None = no truncation)
    xp              : output array module (np or cp)
    num_train_symbols: int — stored in result for caller reference

    Returns
    -------
    EqualizerResult
    """
    y_hat = xp.asarray(y_out.T)  # (N_sym, C) -> (C, N_sym)
    errors = xp.asarray(e_out.T)
    W = xp.asarray(W_final)

    if n_sym is not None:
        y_hat = y_hat[..., :n_sym]
        errors = errors[..., :n_sym]

    if was_1d:
        y_hat = y_hat[0]
        errors = errors[0]
        W = W[0, 0]

    w_history = None
    if store_weights:
        w_history = xp.asarray(w_hist)
        if was_1d:
            w_history = w_history[:, 0, 0, :]

    return EqualizerResult(
        y_hat=y_hat,
        weights=W,
        error=errors,
        weights_history=w_history,
        num_train_symbols=num_train_symbols,
    )


def _validate_sps(sps, num_taps):
    """Raise if sps != 2 (T/2-spaced only); warn if num_taps is too small."""
    if sps != 2:
        raise ValueError(
            f"Adaptive equalizers require 2 samples/symbol "
            f"(T/2-spaced input). Got sps={sps}."
        )
    if num_taps < 2 * sps:
        logger.warning(
            f"num_taps={num_taps} is small for sps={sps}. "
            f"Recommend num_taps >= {4 * sps + 1} for fractionally-spaced equalization."
        )


# ============================================================================
# ADAPTIVE EQUALIZERS
# ============================================================================


def lms(
    samples: ArrayType,
    training_symbols: Optional[ArrayType] = None,
    num_taps: int = 21,
    step_size: float = 0.01,
    modulation: Optional[str] = None,
    order: Optional[int] = None,
    unipolar: bool = False,
    sps: int = 2,
    store_weights: bool = False,
    num_train_symbols: Optional[int] = None,
    device: Optional[str] = "cpu",
    center_tap: Optional[int] = None,
    backend: str = "jax",
) -> EqualizerResult:
    """
    Least Mean Squares adaptive equalizer with butterfly MIMO support.

    Supports data-aided (training) and decision-directed (DD) modes.
    When ``training_symbols`` are provided, the equalizer uses them for the
    initial convergence phase, then switches to DD mode using the
    ``modulation`` and ``order`` parameters, or a constellation auto-inferred
    from the training symbols for hard-decision slicing.

    For MIMO inputs ``(C, N)``, a butterfly ``(C, C, num_taps)`` filter
    structure is used so each output is a weighted sum of all input streams,
    enabling cross-channel interference cancellation.

    Parameters
    ----------
    samples : array_like
        Input signal samples. Shape: ``(N_samples,)`` for SISO or
        ``(C, N_samples)`` for MIMO butterfly equalization.
        Typically at 2 samples/symbol for fractionally-spaced equalization.
    training_symbols : array_like, optional
        Known transmitted symbols (at symbol rate, 1 SPS).
        Shape: ``(N_train,)`` for SISO or ``(C, N_train)`` for MIMO.
        If None, pure DD mode (requires ``modulation`` and ``order``).
    num_taps : int, default 21
        Number of equalizer taps per FIR filter. For fractionally-spaced
        equalization (sps > 1), use at least ``4 * sps`` taps.
    step_size : float, default 0.01
        NLMS normalized step size (mu). After dividing the gradient by
        instantaneous input power ``||x||^2``, the effective correlation matrix
        has unit eigenvalues, so the stability interval collapses to the fixed
        range ``(0, 2)`` regardless of signal power. Values near 2 converge
        fast but with high steady-state misadjustment; values near 0 converge
        slowly but reach lower residual MSE. Typical: 0.01-0.1.
    modulation : str, optional
        Modulation scheme (e.g., 'psk', 'qam', 'pam') for DD slicing.
        Required if ``training_symbols`` is None.
    order : int, optional
        Modulation order (e.g., 4, 16).
    unipolar : bool, default False
        If True, indicates the modulation is unipolar (e.g., unipolar PAM).
    sps : int, default 2
        Samples per symbol at the input. Must be 2 (T/2-spaced).
        The equalizer natively filters intermediate transition points by
        decimating to symbol rate (shifting the mathematical window by
        exactly ``sps`` samples) to compute robust DD slicing errors.
    store_weights : bool, default False
        If True, stores weight trajectory in ``weights_history``.
    num_train_symbols : int, optional
        Limits the number of training symbols used. If provided, the
        equalizer will forcefully switch to blind Decision-Directed (DD)
        mode after this many symbols, even if more training symbols
        are available in the array.
    device : str, optional
        Target device for JAX computations (e.g., 'cpu', 'gpu', 'tpu').
        Default is 'cpu'. Ignored when ``backend='numba'``.
    center_tap : int, optional
        Index of the center tap. If None, defaults to ``num_taps // 2``.
    backend : str, default 'numba'
        Execution backend. ``'numba'`` compiles the sequential loop with LLVM
        via Numba ``@njit``; typically 2-5x faster than JAX on CPU for
        SISO/small-MIMO signals (no scan serialization overhead).
        ``'jax'`` uses ``jax.lax.scan`` and supports GPU placement and
        automatic differentiation through the equalizer.

    Returns
    -------
    EqualizerResult
        Equalized symbols, final weights, error history, and optionally
        weight trajectory. Arrays reside on the same backend as input.

    Warnings
    --------
    **JAX GPU mode is typically slower than CPU for adaptive equalizers.**
    LMS is inherently sequential: each symbol's weight update depends on the
    previous weights, so ``lax.scan`` serializes execution even on GPU.
    The per-step arithmetic (a ``num_taps``-length dot product) is far too
    small to saturate GPU compute units, while kernel-launch and
    device-memory-transfer overhead dominate.  In practice, ``device='cpu'``
    is 2-10x faster for typical SISO sequences up to ~100 k symbols.  Use
    ``device='gpu'`` only when the number of MIMO channels (``num_ch``) is
    large enough to amortize GPU launch costs, or when batching many
    independent signals externally.  For CPU-optimal throughput use
    ``backend='numba'`` (the default).
    """
    logger.info(
        f"LMS equalizer: num_taps={num_taps}, mu={step_size}, sps={sps}, "
        f"backend={backend}, num_train_symbols={num_train_symbols}"
    )

    samples, xp, _ = dispatch(samples)
    stride = int(sps)
    _validate_sps(sps, num_taps)

    # Clip training to num_train_symbols (on original backend; no copy)
    if training_symbols is not None:
        training_symbols, _, _ = dispatch(training_symbols)
        if num_train_symbols is not None:
            training_symbols = training_symbols[..., :num_train_symbols]

    # Shape calcs — independent of normalization
    was_1d = samples.ndim == 1
    num_ch = 1 if was_1d else samples.shape[0]
    n_samples = samples.shape[0] if was_1d else samples.shape[1]
    n_sym = n_samples // stride

    c_tap = center_tap if center_tap is not None else num_taps // 2
    pad_left = c_tap
    pad_right = n_sym * stride - n_samples + num_taps - 1 - pad_left

    if backend == "numba":
        # Convert to plain NumPy (no-op for CPU NumPy; downloads for CuPy)
        samples_np = np.ascontiguousarray(
            samples.get() if hasattr(samples, "get") else np.asarray(samples),
            dtype=np.complex64,
        )
        training_np = (
            np.asarray(
                training_symbols.get()
                if hasattr(training_symbols, "get")
                else training_symbols,
                dtype=np.complex64,
            )
            if training_symbols is not None
            else None
        )
        samples_np, training_np = _normalize_inputs_numpy(samples_np, training_np, sps)
        # Pad (NumPy)
        if was_1d:
            samples_padded = np.pad(samples_np, (pad_left, pad_right))[np.newaxis, :]
        else:
            samples_padded = np.pad(samples_np, ((0, 0), (pad_left, pad_right)))
        # Constellation (NumPy)
        if modulation is not None and order is not None:
            from .mapping import gray_constellation

            reference_constellation = gray_constellation(
                modulation, order, unipolar=unipolar
            )
            constellation_np = (
                np.asarray(reference_constellation).flatten().astype(np.complex64)
            )
        elif training_np is not None:
            train_flat = training_np.reshape(-1)
            constellation_np = np.unique(np.round(train_flat, decimals=8))
        else:
            raise ValueError("modulation and order must be provided for DD mode.")
        train_full, n_train_aligned = _prepare_training_numpy(
            training_np,
            num_ch,
            n_sym,
            num_train_symbols=num_train_symbols,
        )
        W = _init_butterfly_weights_numpy(num_ch, num_taps, center_tap=center_tap)
        y_out = np.empty((n_sym, num_ch), dtype=np.complex64)
        e_out = np.empty((n_sym, num_ch), dtype=np.complex64)
        w_hist_buf = (
            np.empty((n_sym, num_ch, num_ch, num_taps), dtype=np.complex64)
            if store_weights
            else np.empty((1, num_ch, num_ch, num_taps), dtype=np.complex64)
        )
        _get_numba_lms()(
            samples_padded,
            train_full,
            constellation_np,
            W,
            np.float32(step_size),
            np.int32(n_train_aligned),
            stride,
            store_weights,
            y_out,
            e_out,
            w_hist_buf,
        )
        return _unpack_result_numpy(
            y_out,
            e_out,
            W,
            w_hist_buf,
            was_1d,
            store_weights,
            n_sym=None,
            xp=xp,
            num_train_symbols=int(n_train_aligned),
        )

    # JAX backend
    jax, jnp, _ = _get_jax()
    if jax is None:
        raise ImportError("JAX is required for backend='jax'.")

    samples, training_symbols = _normalize_inputs_jax(
        samples, training_symbols, sps, xp
    )
    # Pad (backend-agnostic via xp)
    samples_padded = (
        xp.pad(samples, ((0, 0), (pad_left, pad_right)))
        if not was_1d
        else xp.pad(samples, (pad_left, pad_right))
    )
    # Constellation
    if modulation is not None and order is not None:
        from .mapping import gray_constellation

        reference_constellation = gray_constellation(
            modulation, order, unipolar=unipolar
        )
    elif training_symbols is not None:
        _, _xp, _ = dispatch(training_symbols)
        train_flat = _xp.reshape(training_symbols, (-1,))
        reference_constellation = _xp.unique(_xp.round(train_flat, decimals=8))
    else:
        raise ValueError("modulation and order must be provided for DD mode.")
    constellation_np = (
        np.asarray(to_device(reference_constellation, "cpu"))
        .flatten()
        .astype("complex64")
    )
    train_full, n_train_aligned = _prepare_training_jax(
        training_symbols,
        num_ch,
        n_sym,
        num_train_symbols=num_train_symbols,
    )
    x_jax = to_jax(samples_padded, device=device)
    if was_1d:
        x_jax = x_jax[None, :]

    try:
        platform = (
            device.lower()
            if device is not None
            else (
                x_jax.device.platform
                if hasattr(x_jax, "device")
                else list(x_jax.devices())[0].platform
            )
        )
    except Exception:
        platform = "cpu"

    train_jax = to_jax(train_full, device=platform)
    const_jax = to_jax(constellation_np, device=platform)
    w_init = _init_butterfly_weights_jax(
        num_ch, num_taps, jnp, sps=sps, center_tap=center_tap
    )
    w_init = to_jax(w_init, device=platform)
    mu_jax = to_jax(jnp.float32(step_size), device=platform)
    n_train_jax = to_jax(jnp.int32(n_train_aligned), device=platform)

    scan_fn = _get_jax_lms(num_taps, stride, len(constellation_np), num_ch)
    y_jax, e_jax, W_jax, wh_jax = scan_fn(
        x_jax, train_jax, const_jax, w_init, mu_jax, n_train_jax
    )
    return _unpack_result_jax(
        y_jax,
        e_jax,
        W_jax,
        wh_jax,
        was_1d,
        store_weights,
        n_sym=None,
        xp=xp,
        num_train_symbols=int(n_train_aligned),
    )


def rls(
    samples: ArrayType,
    training_symbols: Optional[ArrayType] = None,
    num_taps: int = 21,
    forgetting_factor: float = 0.99,
    delta: float = 0.01,
    leakage: float = 0.0,
    modulation: Optional[str] = None,
    order: Optional[int] = None,
    unipolar: bool = False,
    sps: int = 1,
    store_weights: bool = False,
    num_train_symbols: Optional[int] = None,
    device: Optional[str] = "cpu",
    center_tap: Optional[int] = None,
    backend: str = "numba",
) -> EqualizerResult:
    """
    Recursive Least Squares adaptive equalizer with butterfly MIMO support.

    RLS converges faster than LMS at the cost of higher per-symbol
    complexity (O(num_taps²) for the rank-1 Riccati update vs O(num_taps)
    for LMS).  It maintains an inverse correlation matrix P per output stream.

    Parameters
    ----------
    samples : array_like
        Input signal samples. Shape: ``(N_samples,)`` or ``(C, N_samples)``.
    training_symbols : array_like, optional
        Known symbols for data-aided adaptation (at symbol rate, 1 SPS).
    num_taps : int, default 21
        Number of equalizer taps per FIR filter.
    forgetting_factor : float, default 0.99
        RLS forgetting factor (lambda). Range: (0, 1].
        Values close to 1 give longer memory.
    delta : float, default 0.01
        Tikhonov regularisation coefficient that seeds the inverse correlation
        matrix as ``P₀ = (1/delta) · I``.

        **Physical interpretation.**  RLS recursively refines a running estimate
        of ``Rxx⁻¹``, where ``Rxx = E[x xᴴ]`` is the input auto-correlation
        matrix.  Before any data have been observed, ``P`` must be initialised
        to some positive-definite matrix.  Choosing ``P₀ = (1/delta) · I``
        is equivalent to assuming a fictitious prior with `delta` units of
        regularisation energy per tap — a textbook Tikhonov (ridge) prior on
        the tap vector with regularisation parameter ``delta``.

        **Effect on convergence.**

        * **Large delta** (e.g. 1.0): ``P₀`` is small → the first Kalman gain
          vectors ``k = P x / (λ + xᴴ P x)`` are small → the equalizer adapts
          sluggishly over the first tens of symbols.  Once enough data are seen
          the bias disappears, so this is safe when a long training sequence is
          available and numerical robustness is the priority.
        * **Small delta** (e.g. 1e-4): ``P₀ = (1/delta) · I`` is a large matrix
          → ``k`` is large for the first symbols → the equalizer converges in
          very few symbols but the weight update is dominated by noise on those
          first few observations, potentially requiring more symbols to settle.
          Extremely small values (< 1e-5) risk numerical overflow of ``P``
          before the Riccati update can contract it.

        **Sensitivity to signal power.**  The code normalises input samples to
        unit symbol-rate power before running the Riccati recursion, so
        ``delta`` is expressed in normalised units (≈ noise variance scale) and
        is not sensitive to the raw signal amplitude.

        **Practical guidelines.**

        * **Training-aided mode** (``training_symbols`` provided): the default
          ``delta=0.01`` works well for most symbol rates and SNR regimes.
          Increase toward 1.0 if tap weights oscillate wildly during the first
          training symbols; decrease toward 1e-3 if convergence is slow and
          your training block is short.
        * **Decision-directed (DD) only**: prefer ``delta=1.0`` and rely on the
          forgetting factor to drive convergence, keeping ``P`` bounded.
        * **Fractionally-spaced signals** (``sps=2``, with ``leakage > 0``):
          larger ``delta`` (0.1–1.0) helps counteract the positive-feedback
          tendency of the unbounded ``P`` eigenvalues in the null sub-space.
          Pair with ``leakage=1e-4`` for structural stability.
    leakage : float, default 0.0
        Weight-decay coefficient (γ) applied to the tap vector at every step::

            W ← (1 - γ)·W + k·ē         # leaky weight update
            P ← (P - k·x̄ᴴP) / λ         # standard Riccati (unchanged)

        Weight decay exponentially suppresses tap weights in the null subspace —
        noise-only frequency bands that arise in T/2-spaced (sps=2) signals —
        without inflating ``P``'s eigenvalues.
        A value of ``0.0`` (default) gives standard RLS.
        For fractionally-spaced equalizers start with ``leakage=1e-4`` and
        increase if steady-state EVM remains high.
    modulation : str, optional
        Modulation scheme (e.g., 'psk', 'qam', 'pam') for DD slicing.
        Required if ``training_symbols`` is None.
    order : int, optional
        Modulation order (e.g., 4, 16).
    unipolar : bool, default False
        If True, indicates the modulation is unipolar (e.g., unipolar PAM).
    sps : int, default 1
        Samples per symbol at the input.
    store_weights : bool, default False
        If True, stores weight trajectory.
    num_train_symbols : int, optional
        Limits the number of training symbols used. If provided, the
        equalizer will forcefully switch to blind Decision-Directed (DD)
        mode after this many symbols.
    device : str, optional
        Target device for JAX computations (e.g., 'cpu', 'gpu', 'tpu').
        Default is 'cpu'. Ignored when ``backend='numba'``.
    center_tap : int, optional
        Index of the center tap. If None, defaults to ``num_taps // 2``.
    backend : str, default 'numba'
        Execution backend. ``'numba'`` uses Numba ``@njit``; LLVM-compiled,
        typically fastest on CPU, particularly for the O(num_taps²) Riccati
        update. ``'jax'`` uses ``jax.lax.scan`` (XLA-compiled, GPU-capable).

    Returns
    -------
    EqualizerResult
        Equalized symbols, final weights, error history, and optionally
        weight trajectory.

    Warnings
    --------
    **Fractional Spacing Singularity:**
    Applying RLS to fractionally-spaced signals (sps > 1) is not recommended.
    Fractional spacing bounds the signal energy within a subset of the Nyquist
    bandwidth. The unexcited frequency bands contain strictly thermal noise,
    rendering the input correlation matrix mathematically singular. RLS
    attempts to invert these near-zero eigenvalues, exponentially amplifying
    high-frequency noise and causing severe tap weight bloat.  Normalized LMS
    is the structurally stable alternative.
    """
    if sps > 1:
        logger.warning(
            f"RLS is mathematically ill-conditioned for fractionally-spaced signals (sps={sps}). "
            "The noise-only null-subspace creates a singular correlation matrix, causing tap bloat. "
            "Use LMS for fractionally-spaced equalization unless heavy Tikhonov regularization is applied."
        )

    logger.info(
        f"RLS equalizer: num_taps={num_taps}, forgetting_factor={forgetting_factor}, "
        f"delta={delta:.2e}, leakage={leakage:.2e}, sps={sps}, "
        f"backend={backend}, num_train_symbols={num_train_symbols}"
    )

    samples, xp, _ = dispatch(samples)
    stride = int(sps)

    if training_symbols is not None:
        training_symbols, _, _ = dispatch(training_symbols)
        if num_train_symbols is not None:
            training_symbols = training_symbols[..., :num_train_symbols]

    was_1d = samples.ndim == 1
    if was_1d:
        num_ch = 1
        n_samples = samples.shape[0]
    else:
        num_ch, n_samples = samples.shape

    n_sym = n_samples // stride
    # Early-halt boundary: freeze W and P once the sliding window reaches the
    # right zero-padding (last num_taps//2 symbols have contaminated windows).
    n_update_halt = max(0, n_sym - num_taps // 2)

    c_tap = center_tap if center_tap is not None else num_taps // 2
    pad_left = c_tap
    pad_right = n_sym * stride - n_samples + num_taps - 1 - pad_left

    if backend == "numba":
        numba = _get_numba()
        if numba is None:
            raise ImportError("Numba is required for backend='numba'.")

        samples_np = np.ascontiguousarray(
            samples.get() if hasattr(samples, "get") else np.asarray(samples),
            dtype=np.complex64,
        )
        training_np = (
            np.asarray(
                training_symbols.get()
                if hasattr(training_symbols, "get")
                else training_symbols,
                dtype=np.complex64,
            )
            if training_symbols is not None
            else None
        )
        samples_np, training_np = _normalize_inputs_numpy(samples_np, training_np, sps)

        x_np = (
            np.pad(samples_np, ((0, 0), (pad_left, pad_right)))
            if not was_1d
            else np.pad(samples_np, (pad_left, pad_right))
        )
        x_np = np.ascontiguousarray(x_np)
        if was_1d:
            x_np = x_np[np.newaxis, :]

        if modulation is not None and order is not None:
            from .mapping import gray_constellation

            reference_constellation = gray_constellation(
                modulation, order, unipolar=unipolar
            )
            constellation_np = (
                np.asarray(reference_constellation).flatten().astype("complex64")
            )
        elif training_np is not None:
            train_flat = training_np.reshape(-1)
            constellation_np = np.unique(np.round(train_flat, decimals=8)).astype(
                "complex64"
            )
        else:
            raise ValueError("modulation and order must be provided for DD mode.")

        train_full, n_train_aligned = _prepare_training_numpy(
            training_np,
            num_ch,
            n_sym,
            num_train_symbols=num_train_symbols,
        )
        W = _init_butterfly_weights_numpy(num_ch, num_taps, center_tap=center_tap)
        regressor_dim = num_ch * num_taps
        P = np.eye(regressor_dim, dtype=np.complex64) / np.float32(delta)
        y_out = np.empty((n_sym, num_ch), dtype=np.complex64)
        e_out = np.empty((n_sym, num_ch), dtype=np.complex64)
        w_hist_buf = (
            np.empty((n_sym, num_ch, num_ch, num_taps), dtype=np.complex64)
            if store_weights
            else np.empty((1, num_ch, num_ch, num_taps), dtype=np.complex64)
        )
        _get_numba_rls()(
            x_np,
            train_full,
            constellation_np,
            W,
            P,
            np.float32(forgetting_factor),
            np.float32(leakage),
            np.int32(n_train_aligned),
            np.int32(n_update_halt),
            stride,
            store_weights,
            y_out,
            e_out,
            w_hist_buf,
        )
        # Truncate last num_taps//2 symbols: those windows overlap the right
        # zero-padding, producing near-zero y that the slicer maps to ~1.0
        # magnitude, creating a spurious MSE/EVM spike.
        return _unpack_result_numpy(
            y_out,
            e_out,
            W,
            w_hist_buf,
            was_1d,
            store_weights,
            n_sym=n_update_halt,
            xp=xp,
            num_train_symbols=int(n_train_aligned),
        )

    # JAX backend
    jax, jnp, _ = _get_jax()
    if jax is None:
        raise ImportError("JAX is required for backend='jax'.")

    samples, training_symbols = _normalize_inputs_jax(
        samples, training_symbols, sps, xp
    )

    samples_padded = (
        xp.pad(samples, ((0, 0), (pad_left, pad_right)))
        if not was_1d
        else xp.pad(samples, (pad_left, pad_right))
    )

    if modulation is not None and order is not None:
        from .mapping import gray_constellation

        reference_constellation = gray_constellation(
            modulation, order, unipolar=unipolar
        )
    elif training_symbols is not None:
        _, _xp, _ = dispatch(training_symbols)
        train_flat = _xp.reshape(training_symbols, (-1,))
        reference_constellation = _xp.unique(_xp.round(train_flat, decimals=8))
    else:
        raise ValueError("modulation and order must be provided for DD mode.")

    constellation_np = (
        np.asarray(to_device(reference_constellation, "cpu"))
        .flatten()
        .astype("complex64")
    )

    logger.debug(
        f"RLS internals: n_sym={n_sym}, n_train={num_train_symbols}, "
        f"n_update_halt={n_update_halt}, leakage={leakage:.2e}, delta={delta:.2e}"
    )

    train_full, n_train_aligned = _prepare_training_jax(
        training_symbols,
        num_ch,
        n_sym,
        num_train_symbols=num_train_symbols,
    )
    x_jax = to_jax(samples_padded, device=device)
    if was_1d:
        x_jax = x_jax[None, :]

    try:
        platform = (
            device.lower()
            if device is not None
            else (
                x_jax.device.platform
                if hasattr(x_jax, "device")
                else list(x_jax.devices())[0].platform
            )
        )
    except Exception:
        platform = "cpu"

    train_jax = to_jax(train_full, device=platform)
    const_jax = to_jax(constellation_np, device=platform)
    w_init = _init_butterfly_weights_jax(
        num_ch, num_taps, jnp, sps=sps, center_tap=center_tap
    )
    w_init = to_jax(w_init, device=platform)

    regressor_dim = num_ch * num_taps
    P_init = jnp.eye(regressor_dim, dtype="complex64") / delta
    P_init = to_jax(P_init, device=platform)
    lam_jax = to_jax(jnp.float32(forgetting_factor), device=platform)
    n_train_jax = to_jax(jnp.int32(n_train_aligned), device=platform)
    leakage_jax = to_jax(jnp.float32(leakage), device=platform)
    n_update_halt_jax = to_jax(jnp.int32(n_update_halt), device=platform)

    scan_fn = _get_jax_rls(num_taps, stride, len(constellation_np), num_ch)
    y_jax, e_jax, W_jax, wh_jax = scan_fn(
        x_jax,
        train_jax,
        const_jax,
        w_init,
        P_init,
        lam_jax,
        n_train_jax,
        leakage_jax,
        n_update_halt_jax,
    )
    # Truncate last num_taps//2 symbols (zero-padding contamination).
    return _unpack_result_jax(
        y_jax,
        e_jax,
        W_jax,
        wh_jax,
        was_1d,
        store_weights,
        n_sym=n_update_halt,
        xp=xp,
        num_train_symbols=int(n_train_aligned),
    )


def cma(
    samples: ArrayType,
    num_taps: int = 21,
    step_size: float = 1e-3,
    modulation: Optional[str] = None,
    order: Optional[int] = None,
    unipolar: bool = False,
    sps: int = 2,
    store_weights: bool = False,
    device: Optional[str] = "cpu",
    center_tap: Optional[int] = None,
    backend: str = "numba",
) -> EqualizerResult:
    """
    Constant Modulus Algorithm blind equalizer with butterfly MIMO support.

    CMA minimizes the Godard dispersion criterion and requires no training
    symbols. It is the standard blind equalizer for constant-modulus signals
    (PSK) and near-constant-modulus signals (low-order QAM).

    CMA recovers the signal up to a phase ambiguity. A phase recovery step
    (e.g. Viterbi-Viterbi, pilot-aided) is typically needed after CMA.

    Parameters
    ----------
    samples : array_like
        Input signal samples. Shape: ``(N_samples,)`` or ``(C, N_samples)``.
        Typically at 2 samples/symbol for fractionally-spaced equalization.
    num_taps : int, default 21
        Number of equalizer taps per FIR filter.
    step_size : float, default 1e-3
        CMA step size (mu). Unlike LMS, CMA's cost surface is non-convex and
        higher-order, so input-power normalization distorts the gradient geometry.
        Use a fixed step size in the range 1e-5 to 1e-3 for stability.
    modulation : str, optional
        Modulation type for auto-computing Godard radius R2 (e.g. ``"psk"``, ``"qam"``).
        If None, defaults to R2=1.0.
    order : int, optional
        Modulation order for auto-computing R2.
    unipolar : bool, default False
        Use unipolar constellation for auto-computing R2.
    sps : int, default 2
        Samples per symbol at the input. Must be 2 (T/2-spaced).
        The equalizer natively filters intermediate transition points by
        decimating to symbol rate (shifting the mathematical window by
        exactly ``sps`` samples) to compute robust blind errors.
    store_weights : bool, default False
        If True, stores weight trajectory.
    device : str, optional
        Target device for JAX computations (e.g., 'cpu', 'gpu', 'tpu').
        Default is 'cpu'. Ignored when ``backend='numba'``.
    center_tap : int, optional
        Index of the center tap. If None, defaults to ``num_taps // 2``.
    backend : str, default 'numba'
        Execution backend. ``'numba'`` uses Numba ``@njit``; LLVM-compiled,
        typically fastest on CPU. ``'jax'`` uses ``jax.lax.scan``
        (XLA-compiled, GPU-capable).

    Returns
    -------
    EqualizerResult
        Equalized symbols, final weights, CMA error history, and optionally
        weight trajectory.

    Warnings
    --------
    **JAX GPU mode is typically slower than CPU for adaptive equalizers.**
    CMA is inherently sequential: each weight update depends on the previous
    weights, so ``lax.scan`` serializes execution even on GPU.  Use
    ``device='cpu'`` for typical SISO sequences, or ``backend='numba'`` for
    CPU-optimal throughput.
    """
    logger.info(
        f"CMA equalizer: num_taps={num_taps}, mu={step_size}, sps={sps}, "
        f"backend={backend}"
    )

    samples, xp, _ = dispatch(samples)
    stride = int(sps)
    _validate_sps(sps, num_taps)

    was_1d = samples.ndim == 1
    if was_1d:
        num_ch = 1
        n_samples = samples.shape[0]
    else:
        num_ch, n_samples = samples.shape

    # Compute R2 from the Godard constellation (constant for a given modulation)
    if modulation is not None and order is not None:
        from .mapping import gray_constellation

        const = gray_constellation(modulation, order, unipolar=unipolar)
        r2 = float(np.mean(np.abs(const) ** 4) / np.mean(np.abs(const) ** 2))
        logger.debug(f"CMA R2 from {modulation.upper()}-{order}: {r2:.4f}")
    else:
        r2 = 1.0

    n_sym = n_samples // stride

    c_tap = center_tap if center_tap is not None else num_taps // 2
    pad_left = c_tap
    pad_right = n_sym * stride - n_samples + num_taps - 1 - pad_left

    if backend == "numba":
        numba = _get_numba()
        if numba is None:
            raise ImportError("Numba is required for backend='numba'.")

        samples_np = np.ascontiguousarray(
            samples.get() if hasattr(samples, "get") else np.asarray(samples),
            dtype=np.complex64,
        )
        # RMS-normalize samples to unit symbol-rate power (CMA has no training)
        samples_np, _ = _normalize_inputs_numpy(samples_np, None, sps)

        x_np = (
            np.pad(samples_np, ((0, 0), (pad_left, pad_right)))
            if not was_1d
            else np.pad(samples_np, (pad_left, pad_right))
        )
        x_np = np.ascontiguousarray(x_np)
        if was_1d:
            x_np = x_np[np.newaxis, :]

        W = _init_butterfly_weights_numpy(num_ch, num_taps, center_tap=center_tap)
        y_out = np.empty((n_sym, num_ch), dtype=np.complex64)
        e_out = np.empty((n_sym, num_ch), dtype=np.complex64)
        w_hist_buf = (
            np.empty((n_sym, num_ch, num_ch, num_taps), dtype=np.complex64)
            if store_weights
            else np.empty((1, num_ch, num_ch, num_taps), dtype=np.complex64)
        )
        _get_numba_cma()(
            x_np,
            W,
            np.float32(step_size),
            np.float32(r2),
            stride,
            store_weights,
            y_out,
            e_out,
            w_hist_buf,
        )
        return _unpack_result_numpy(
            y_out,
            e_out,
            W,
            w_hist_buf,
            was_1d,
            store_weights,
            n_sym=None,
            xp=xp,
        )

    # JAX backend
    jax, jnp, _ = _get_jax()
    if jax is None:
        raise ImportError("JAX is required for backend='jax'.")

    # RMS-normalize samples to unit symbol-rate power (CMA has no training)
    samples, _ = _normalize_inputs_jax(samples, None, sps, xp)

    samples_padded = (
        xp.pad(samples, ((0, 0), (pad_left, pad_right)))
        if not was_1d
        else xp.pad(samples, (pad_left, pad_right))
    )

    x_jax = to_jax(samples_padded, device=device)
    if was_1d:
        x_jax = x_jax[None, :]

    try:
        platform = (
            device.lower()
            if device is not None
            else (
                x_jax.device.platform
                if hasattr(x_jax, "device")
                else list(x_jax.devices())[0].platform
            )
        )
    except Exception:
        platform = "cpu"

    w_init = _init_butterfly_weights_jax(
        num_ch, num_taps, jnp, sps=sps, center_tap=center_tap
    )
    w_init = to_jax(w_init, device=platform)
    mu_jax = to_jax(jnp.float32(step_size), device=platform)
    r2_jax = to_jax(jnp.float32(r2), device=platform)

    scan_fn = _get_jax_cma(num_taps, stride, num_ch)
    y_jax, e_jax, W_jax, wh_jax = scan_fn(x_jax, w_init, mu_jax, r2_jax, n_sym)
    return _unpack_result_jax(
        y_jax,
        e_jax,
        W_jax,
        wh_jax,
        was_1d,
        store_weights,
        n_sym=None,
        xp=xp,
    )


# ============================================================================
# BLOCK EQUALIZERS
# ============================================================================


def zf_equalizer(
    samples: ArrayType,
    channel_estimate: ArrayType,
    noise_variance: float = 0.0,
) -> ArrayType:
    """
    Zero-Forcing / MMSE frequency-domain block equalizer.

    Given a channel impulse response estimate, equalizes the received signal
    in the frequency domain. Supports both SISO and MIMO channels.

    When ``noise_variance=0``, this is a pure Zero-Forcing equalizer.
    When ``noise_variance > 0``, this becomes an MMSE equalizer that
    regularizes spectral nulls to avoid noise enhancement.

    Parameters
    ----------
    samples : array_like
        Received samples. Shape: ``(N,)`` for SISO or ``(C, N)`` for MIMO.
    channel_estimate : array_like
        Channel impulse response. Shape: ``(L,)`` for SISO or
        ``(C, C, L)`` for MIMO (each ``[i, j]`` is the FIR from input j
        to output i).
    noise_variance : float, default 0.0
        Noise variance (sigma^2) for MMSE regularization.
        ``0.0`` gives pure ZF.

    Returns
    -------
    array_like
        Equalized samples. Same shape and backend as input.
    """
    logger.info(f"ZF/MMSE equalizer: noise_variance={noise_variance:.2e}")
    samples, xp, _ = dispatch(samples)
    channel_estimate = xp.asarray(channel_estimate)

    import math

    # Capture BEFORE any reshape so the later frequency-domain branch
    # can still take the efficient scalar path for SISO channels.
    siso_channel = channel_estimate.ndim == 1

    was_1d = samples.ndim == 1
    if was_1d:
        samples = samples[None, :]
        if siso_channel:
            channel_estimate = channel_estimate[None, None, :]

    num_ch, N = samples.shape
    L = channel_estimate.shape[-1] if channel_estimate.ndim > 0 else 1
    reg = noise_variance if noise_variance > 0 else 1e-12

    # N_fft must satisfy discard = N_fft//4 >= L so the IIR filter's causal
    # and anti-causal tails both decay within the discarded guard regions.
    N_fft = max(1024, 2 ** math.ceil(math.log2(max(1, 4 * L))))
    while N_fft < 4 * L:
        N_fft *= 2
    B = N_fft // 2  # 50% overlap hop
    discard = N_fft // 4  # symmetric guard: reject causal and anti-causal transients
    num_blocks = math.ceil(N / B)

    logger.debug(
        f"ZF/MMSE internals: N={N}, L={L}, num_ch={num_ch}, "
        f"N_fft={N_fft}, B={B}, discard={discard}, num_blocks={num_blocks}"
    )

    # Pre-pad by discard: shifts the first valid output to align with sample 0.
    # Post-pad by (num_blocks*B - N + discard): fills the last block window.
    pad_left = discard
    pad_right = num_blocks * B - N + discard
    samples_padded = xp.pad(samples, ((0, 0), (pad_left, pad_right)))

    # Vectorize window extraction using as_strided to avoid memory copies
    stride = samples_padded.strides
    windows = xp.lib.stride_tricks.as_strided(
        samples_padded,
        shape=(num_ch, num_blocks, N_fft),
        strides=(stride[0], B * stride[1], stride[1]),
    )

    Y = xp.fft.fft(windows, n=N_fft, axis=-1)

    if siso_channel:
        # channel_estimate is (L,) for MIMO input or (1,1,L) for SISO input
        # after the was_1d reshape; flatten to 1D for scalar frequency-domain division.
        H = xp.fft.fft(channel_estimate.reshape(-1), n=N_fft)
        W = xp.conj(H) / (xp.abs(H) ** 2 + reg)
        X_hat_f = Y * W
    else:
        H_f = xp.fft.fft(channel_estimate, n=N_fft, axis=-1)
        Hk = xp.transpose(H_f, (2, 0, 1))
        Hk_H = xp.conj(xp.transpose(Hk, (0, 2, 1)))
        HHh = Hk @ Hk_H

        eye = xp.eye(num_ch, dtype=samples.dtype)[None, :, :]
        if num_ch == 2:
            # Fast-path: Explicit 2x2 matrix inversion via Cramer's Rule avoiding
            # GPU LAPACK solver overheads (e.g. cuSOLVER batch inversion).
            H_reg = HHh + reg * eye
            a = H_reg[:, 0, 0]
            b = H_reg[:, 0, 1]
            c = H_reg[:, 1, 0]
            d = H_reg[:, 1, 1]
            det = a * d - b * c

            inv_term = xp.empty_like(H_reg)
            inv_term[:, 0, 0] = d / det
            inv_term[:, 0, 1] = -b / det
            inv_term[:, 1, 0] = -c / det
            inv_term[:, 1, 1] = a / det
        else:
            inv_term = xp.linalg.inv(HHh + reg * eye)

        Wk = Hk_H @ inv_term

        # Wk: (N_fft, num_ch, num_ch), Y: (num_ch, num_blocks, N_fft)
        # Vectorized batch matrix multiplication across frequency bins and blocks
        # using explicit batched GEMM instead of einsum for better GPU utilization
        Y_t = xp.transpose(Y, (2, 0, 1))  # (N_fft, num_ch, num_blocks)
        X_hat_k = (
            Wk @ Y_t
        )  # (N_fft, num_ch, num_ch) @ (N_fft, num_ch, num_blocks) -> (N_fft, num_ch, num_blocks)
        X_hat_f = xp.transpose(X_hat_k, (1, 2, 0))  # -> (num_ch, num_blocks, N_fft)

    # IFFT back to time domain
    x_hat = xp.fft.ifft(X_hat_f, n=N_fft, axis=-1)

    # Symmetric discard: keep the center N_fft//2 samples of each block.
    # The first discard samples contain causal circular transient; the last
    # discard samples contain anti-causal (IIR precursor) circular transient.
    valid = x_hat[:, :, discard : discard + B]
    out = valid.reshape(num_ch, -1)[:, :N]
    return out[0] if was_1d else out
