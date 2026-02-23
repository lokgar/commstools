"""
Adaptive and block equalization algorithms.

This module provides production-grade equalizer implementations for
compensating inter-symbol interference (ISI) and channel distortion
in digital communication systems.

Adaptive algorithms (LMS, RLS, CMA) use JAX ``lax.scan`` for compiled
sequential weight updates on both CPU and GPU. Block algorithms (ZF/MMSE)
use NumPy/CuPy dispatch for vectorized frequency-domain processing.

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
# JAX JIT CACHE — ADAPTIVE EQUALIZER KERNELS
# ============================================================================
#
# Alignment convention (center-tap):
#   At scan index k, the input window spans samples[k*sps : k*sps + num_taps].
#   The center-tap at index num_taps//2 references sample k*sps + num_taps//2.
#   Training symbols are pre-shifted by (num_taps//2) in the public functions
#   so that desired[k] matches the symbol at the center of the window.

_JITTED_EQ = {}


def _get_jitted_lms(num_taps, stride, const_size, num_ch):
    """Factory: JIT-compile and cache the sample-by-sample NLMS butterfly scan.

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

                def get_win(ch):
                    return jax.lax.dynamic_slice(ch, (sample_idx,), (num_taps,))

                X_wins = jax.vmap(get_win)(x_input)  # (C, num_taps)

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


def _get_jitted_rls(num_taps, stride, const_size, num_ch):
    """Factory: JIT-compile and cache the sample-by-sample Leaky-RLS butterfly scan.

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

                def get_win(ch):
                    return jax.lax.dynamic_slice(ch, (sample_idx,), (num_taps,))

                X_wins = jax.vmap(get_win)(x_input)  # (C, num_taps)

                y = jnp.einsum("ijt,jt->i", jnp.conj(W), X_wins)

                def slicer(ch_y):
                    return constellation[jnp.argmin(jnp.abs(ch_y - constellation) ** 2)]

                dd = jax.vmap(slicer)(y)
                d = jnp.where(idx < n_train, training_padded[:, idx], dd)
                e = d - y

                x_bar = X_wins.flatten()  # (C * num_taps,)

                Px = P @ x_bar
                denom = lam + jnp.dot(jnp.conj(x_bar), Px)
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
                P_upd = (P - jnp.outer(k, jnp.conj(x_bar) @ P)) / lam

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


def _get_jitted_cma(num_taps, stride, num_ch):
    """Factory: JIT-compile and cache the sample-by-sample CMA butterfly scan.

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

                def get_win(ch):
                    return jax.lax.dynamic_slice(ch, (sample_idx,), (num_taps,))

                X_wins = jax.vmap(get_win)(x_input)  # (C, num_taps)
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
# BLOCK-ADAPTIVE KERNELS (GPU-optimized)
# ============================================================================
#
# Instead of lax.scan over individual symbols, these kernels scan over
# *blocks* of B symbols.  Within each block:
#   1. Parallel forward pass  (vmap) — apply frozen weights to B symbols
#   2. Parallel error compute (vmap) — B errors in one shot
#   3. Single weight update          — average gradient across the block
#
# When block_size=1 this is mathematically identical to the sample-by-sample
# kernel, just with higher per-step overhead.  For block_size >= 8 the
# GPU gets enough parallel work per step to amortise kernel-launch latency.


def _get_jitted_block_lms(num_taps, stride, const_size, num_ch, block_size):
    """Factory: JIT-compile and cache the block-NLMS butterfly scan.

    Processes B symbols in parallel per weight update step rather than one at
    a time.  The outer ``lax.scan`` iterates over blocks; within each block a
    ``vmap`` executes the forward pass over B symbols simultaneously.

    Additional static closure variable:
    B (== block_size) : number of symbols processed per weight update.

    Returns
    -------
    block_lms_scan : JIT-compiled callable
        See the inner function for the call signature.
    """
    key = ("block_lms", num_taps, stride, const_size, num_ch, block_size)
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()
        B = block_size

        @functools.partial(jax.jit, static_argnums=(6,))
        def block_lms_scan(
            x_input,
            training_padded,
            constellation,
            w_init,
            step_size,
            n_train,
            n_blocks,
            n_sym,
        ):
            # Argument shapes and semantics
            # ------------------------------
            # x_input         : (C, N_pad)            complex64
            # training_padded : (C, n_sym_padded)      complex64
            # constellation   : (M,)                  complex64
            # w_init          : (C, C, num_taps)       complex64
            # step_size       : scalar float32         — NLMS mu in (0, 2)
            # n_train         : scalar int32           — training/DD boundary (dynamic)
            # n_blocks        : int (static)           — ceil(n_sym_padded / B)
            # n_sym           : scalar int32           — original (unpadded) symbol count;
            #                    used to zero-mask the final partial block's ghost symbols
            #
            # lax.scan carry  : W  (C, C, num_taps)
            # lax.scan xs     : jnp.arange(n_blocks)
            # lax.scan output : y_all   (n_blocks, B, C)              → reshaped to (N, C)
            #                   e_all   (n_blocks, B, C)              → reshaped to (N, C)
            #                   w_hist  (n_blocks, C, C, num_taps)    one snapshot per block
            #
            # Per-block algorithm:
            #   sym_indices = block_idx*B + [0..B-1]                        (B,)
            #   X_block = vmap(row_extract)(sym_indices)                    (B, C, num_taps)
            #   y_block = vmap(forward_one)(sym_indices)                    (B, C)
            #   d_block = training_padded[:, base:base+B].T                 (B, C)
            #     masked by sym_idx < n_train (training) / slicer (DD)
            #   e_block = d_block - y_block                                 (B, C)
            #     zero-masked for ghost symbols: sym_idx >= n_sym
            #   avg_grad = sum(grads, axis=0) / num_valid                   (C, C, num_taps)
            #   avg_power = sum(valid_powers) / num_valid                   scalar
            #   W += (mu / avg_power) * avg_grad                            NLMS update

            def extract_window(ch, sample_idx):
                return jax.lax.dynamic_slice(ch, (sample_idx,), (num_taps,))

            def forward_one(W, sym_idx):
                """Forward pass + error for a single symbol with frozen W."""
                sample_idx = sym_idx * stride
                X_wins = jax.vmap(extract_window, in_axes=(0, None))(
                    x_input, sample_idx
                )  # (C, num_taps)
                y = jnp.einsum("ijt,jt->i", jnp.conj(W), X_wins)  # (C,)
                return y, X_wins

            def slicer(ch_y):
                return constellation[jnp.argmin(jnp.abs(ch_y - constellation) ** 2)]

            def block_step(W, block_idx):
                # Symbol indices for this block
                base = block_idx * B
                sym_indices = base + jnp.arange(B)  # (B,)

                # Parallel forward pass over B symbols
                y_block, X_block = jax.vmap(lambda idx: forward_one(W, idx))(
                    sym_indices
                )  # y_block: (B, C), X_block: (B, C, num_taps)

                # Desired symbols: training or DD
                d_train = jax.lax.dynamic_slice(
                    training_padded, (0, base), (num_ch, B)
                ).T  # (B, C)
                dd = jax.vmap(lambda y_i: jax.vmap(slicer)(y_i))(y_block)  # (B, C)
                # Per-symbol mask: use training if sym_idx < n_train
                mask = (sym_indices < n_train)[:, None]  # (B, 1)
                d_block = jnp.where(mask, d_train, dd)  # (B, C)

                e_block = d_block - y_block  # (B, C)

                # Mask out ghost errors from out-of-bounds padding
                valid_mask = (sym_indices < n_sym)[:, None]
                e_block = jnp.where(valid_mask, e_block, 0.0)

                # Sum gradient across the block, normalise by valid count only.
                # Dividing by the fixed block size B would dilute gradients in
                # the final partial block where some symbols are zero-masked.
                grads = jax.vmap(
                    lambda e_i, X_i: jnp.einsum("i,jt->ijt", jnp.conj(e_i), X_i)
                )(e_block, X_block)  # (B, C, C, num_taps)
                num_valid = jnp.sum(
                    valid_mask[..., 0], dtype=jnp.float32
                ) + jnp.float32(1e-10)
                avg_grad = jnp.sum(grads, axis=0) / num_valid  # (C, C, num_taps)

                # NLMS: normalize by average power of valid windows only
                powers = jax.vmap(lambda X_i: jnp.real(jnp.sum(X_i * jnp.conj(X_i))))(
                    X_block
                )  # (B,)
                avg_power = (
                    jnp.sum(jnp.where(valid_mask[..., 0], powers, jnp.float32(0.0)))
                    / num_valid
                )
                mu_eff = step_size / (avg_power + jnp.float32(1e-10))

                W_new = W + mu_eff * avg_grad
                return W_new, (y_block, e_block, W_new)

            W_final, (y_all, e_all, w_hist) = jax.lax.scan(
                block_step, w_init, jnp.arange(n_blocks)
            )
            # y_all: (n_blocks, B, C) → reshape to (n_blocks*B, C)
            y_flat = y_all.reshape(-1, num_ch)
            e_flat = e_all.reshape(-1, num_ch)
            # w_hist: (n_blocks, C, C, num_taps) — one per block
            return y_flat, e_flat, W_final, w_hist

        _JITTED_EQ[key] = block_lms_scan
    return _JITTED_EQ[key]


def _get_jitted_block_cma(num_taps, stride, num_ch, block_size):
    """Factory: JIT-compile and cache the block-CMA butterfly scan.

    Block-parallel version of CMA; no training symbols, no constellation.
    ``n_blocks`` is static (arg 4) because it is the ``lax.scan`` trip count.

    Returns
    -------
    block_cma_scan : JIT-compiled callable
        See the inner function for the call signature.
    """
    key = ("block_cma", num_taps, stride, num_ch, block_size)
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()
        B = block_size

        @functools.partial(jax.jit, static_argnums=(4,))
        def block_cma_scan(x_input, w_init, step_size, r2, n_blocks, n_sym):
            # Argument shapes and semantics
            # ------------------------------
            # x_input   : (C, N_pad)       complex64
            # w_init    : (C, C, num_taps) complex64
            # step_size : scalar float32   — fixed CMA gradient step μ
            # r2        : scalar float32   — Godard radius R²
            # n_blocks  : int (static)     — scan trip count
            # n_sym     : scalar int32     — original symbol count for ghost masking
            #
            # lax.scan carry  : W  (C, C, num_taps)
            # lax.scan xs     : jnp.arange(n_blocks)
            # lax.scan output : y_flat  (n_blocks*B, C)
            #                   e_flat  (n_blocks*B, C)   Godard errors
            #                   w_hist  (n_blocks, C, C, num_taps)
            #
            # Per-block:
            #   e_block = y_block * (real(y_block * conj(y_block)) - R²)   (B, C)
            #     zero-masked for sym_idx >= n_sym
            #   avg_grad = sum(grads) / num_valid                           (C, C, num_taps)
            #   W -= μ * avg_grad
            def extract_window(ch, sample_idx):
                return jax.lax.dynamic_slice(ch, (sample_idx,), (num_taps,))

            def forward_one(W, sym_idx):
                sample_idx = sym_idx * stride
                X_wins = jax.vmap(extract_window, in_axes=(0, None))(
                    x_input, sample_idx
                )
                y = jnp.einsum("ijt,jt->i", jnp.conj(W), X_wins)
                return y, X_wins

            def block_step(W, block_idx):
                base = block_idx * B
                sym_indices = base + jnp.arange(B)

                y_block, X_block = jax.vmap(lambda idx: forward_one(W, idx))(
                    sym_indices
                )  # (B, C), (B, C, num_taps)

                # CMA error: e = y * (|y|^2 - R2)
                # jnp.real enforces strict real-valued modulus — see sample-by-sample
                # kernel for rationale.
                e_block = y_block * (
                    jnp.real(y_block * jnp.conj(y_block)) - r2
                )  # (B, C)

                # Mask out ghost errors from out-of-bounds padding
                valid_mask = (sym_indices < n_sym)[:, None]
                e_block = jnp.where(valid_mask, e_block, 0.0)

                grads = jax.vmap(
                    lambda e_i, X_i: jnp.einsum("i,jt->ijt", jnp.conj(e_i), X_i)
                )(e_block, X_block)

                # Normalize by valid symbol count, not block size B, so the
                # final (partial) block is not gradient-diluted by OOB zeros.
                num_valid = jnp.sum(
                    valid_mask[..., 0], dtype=jnp.float32
                ) + jnp.float32(1e-10)
                avg_grad = jnp.sum(grads, axis=0) / num_valid

                W_new = W - step_size * avg_grad  # CMA uses gradient descent (minus)
                return W_new, (y_block, e_block, W_new)

            W_final, (y_all, e_all, w_hist) = jax.lax.scan(
                block_step, w_init, jnp.arange(n_blocks)
            )
            y_flat = y_all.reshape(-1, num_ch)
            e_flat = e_all.reshape(-1, num_ch)
            return y_flat, e_flat, W_final, w_hist

        _JITTED_EQ[key] = block_cma_scan
    return _JITTED_EQ[key]


def _get_jitted_block_rls(num_taps, stride, const_size, num_ch, block_size):
    """Factory: JIT-compile and cache the hybrid block-RLS butterfly scan.

    Hybrid two-phase design per block:

    Phase 1 (parallel ``vmap``):   apply frozen W to B symbols → X_block, y_block
    Phase 2 (sequential ``lax.scan``): update (W, P) symbol-by-symbol within the block

    This amortises the outer ``lax.scan`` loop overhead (B× fewer outer steps)
    while keeping P's sequential rank-1 update mathematically correct.

    Additional static closure variable: B (== block_size).

    Returns
    -------
    block_rls_scan : JIT-compiled callable
        See the inner function for the call signature.
    """
    key = ("block_rls", num_taps, stride, const_size, num_ch, block_size)
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()
        B = block_size

        @functools.partial(jax.jit, static_argnums=(7,))
        def block_rls_scan(
            x_input,
            training_padded,
            constellation,
            w_init,
            P_init,
            lam,
            n_train,
            n_blocks,
            leakage,
            n_update_halt,
        ):
            # Argument shapes and semantics
            # ------------------------------
            # x_input         : (C, N_pad)              complex64 — padded received samples
            # training_padded : (C, n_sym_padded)        complex64 — reference symbols
            # constellation   : (M,)                    complex64 — slicer lookup table
            # w_init          : (C, C, num_taps)         complex64 — initial butterfly weights
            # P_init          : (C*num_taps, C*num_taps) complex64 — initial inv. corr. matrix
            # lam             : scalar float32           — forgetting factor λ
            # n_train         : scalar int32             — training/DD boundary (dynamic)
            # n_blocks        : int (static)             — outer scan trip count
            # leakage         : scalar float32           — weight-decay coefficient γ ∈ [0,1):
            #                    W ← (1−γ)W + k⊗ē  each step; P update is unchanged.
            # n_update_halt   : scalar int32             — last symbol index for W/P update;
            #                    = n_sym - num_taps//2 prevents zero-padding contamination
            #
            # Outer lax.scan carry : (W, P)  — butterfly weights + inv. correlation matrix
            # Outer lax.scan xs    : jnp.arange(n_blocks)
            # Outer lax.scan out   : (y_all, e_all, W_new) each (n_blocks, B, ...)
            #
            # Inner lax.scan (phase 2) carry : (W_i, P_i)
            # Inner lax.scan xs              : jnp.arange(B)  — local symbol index in block
            #
            # Per-inner-step equations (sym_idx = base + b_idx):
            #   X_i   = row_extract(sym_idx)                         (C, num_taps)
            #   x_bar = X_i.flatten()                                (C*num_taps,)
            #   y_i   = einsum('ijt,jt->i', conj(W_i), X_i)         (C,)
            #   d_i   = training[:, sym_idx] if sym_idx < n_train else slicer(y_i)
            #   e_i   = d_i - y_i                                    (C,)
            #   Px    = P_i @ x_bar                                  (C*T,)
            #   k     = Px / (λ + x_bar^H Px)                       Kalman gain (C*T,)
            #   W_i  ← (1−γ)W_i + k ⊗ conj(e_i)  if sym_idx < n_update_halt  (else frozen)
            #   P_i   = (P_i - outer(k, x_bar^H P_i)) / λ          if sym_idx < n_update_halt

            def extract_window(ch, sample_idx):
                return jax.lax.dynamic_slice(ch, (sample_idx,), (num_taps,))

            def row_extract(sym_idx):
                sample_idx = sym_idx * stride
                return jax.vmap(extract_window, in_axes=(0, None))(x_input, sample_idx)

            def slicer(ch_y):
                return constellation[jnp.argmin(jnp.abs(ch_y - constellation) ** 2)]

            def block_step(carry, block_idx):
                W, P = carry
                base = block_idx * B
                sym_indices = base + jnp.arange(B)

                # Phase 1: parallel window extraction
                X_block = jax.vmap(row_extract)(sym_indices)  # (B, C, num_taps)

                # Desired symbols from training (parallel slice)
                d_train = jax.lax.dynamic_slice(
                    training_padded, (0, base), (num_ch, B)
                ).T

                is_train = (sym_indices < n_train)[:, None]  # (B, 1)
                # can_update: True only while the sliding window hasn't reached
                # the right zero-padding AND we're still within valid signal.
                # n_update_halt = n_sym - num_taps//2, so this already implies
                # sym_idx < n_sym, making the old is_valid gate redundant.
                can_update = (sym_indices < n_update_halt)[:, None]  # (B, 1)

                # Phase 2: sequential RLS weight update over B symbols
                def rls_inner_step(carry_inner, b_idx):
                    W_i, P_i = carry_inner
                    X_i = X_block[b_idx]  # (C, num_taps)

                    # Compute y (always — output for every symbol)
                    y_i = jnp.einsum("ijt,jt->i", jnp.conj(W_i), X_i)

                    # Slicer
                    dd_i = jax.vmap(slicer)(y_i)
                    d_i = jnp.where(is_train[b_idx], d_train[b_idx], dd_i)
                    e_i = d_i - y_i

                    x_bar = X_i.flatten()  # (C * num_taps,)
                    Px = P_i @ x_bar
                    denom = lam + jnp.dot(jnp.conj(x_bar), Px)
                    k = Px / denom

                    def w_update(w_row, err_val):
                        w_flat = w_row.flatten()
                        # Weight decay: suppress null-subspace taps exponentially.
                        w_flat_new = (1.0 - leakage) * w_flat + k * jnp.conj(err_val)
                        return w_flat_new.reshape(num_ch, num_taps)

                    W_upd = jax.vmap(w_update)(W_i, e_i)
                    # Standard Riccati update — no diagonal loading on P.
                    P_upd = (P_i - jnp.outer(k, jnp.conj(x_bar) @ P_i)) / lam

                    # Gate both updates: freeze W and P for padding-contaminated
                    # windows (early halt) and for true OOB symbols.
                    gate = can_update[b_idx, 0]
                    W_new = jnp.where(gate, W_upd, W_i)
                    P_new = jnp.where(gate, P_upd, P_i)

                    # Report the unmasked error for diagnostics (caller can
                    # slice to n_sym to discard padded output).
                    return (W_new, P_new), (y_i, e_i, W_new)

                (W_new, P_new), (y_block, e_block, w_hist) = jax.lax.scan(
                    rls_inner_step, (W, P), jnp.arange(B)
                )

                return (W_new, P_new), (y_block, e_block, W_new)

            (W_final, _), (y_all, e_all, w_hist) = jax.lax.scan(
                block_step, (w_init, P_init), jnp.arange(n_blocks)
            )
            y_flat = y_all.reshape(-1, num_ch)
            e_flat = e_all.reshape(-1, num_ch)
            return y_flat, e_flat, W_final, w_hist

        _JITTED_EQ[key] = block_rls_scan
    return _JITTED_EQ[key]


# ============================================================================
# SHARED HELPERS
# ============================================================================


def _normalize_inputs(samples, training_symbols, sps, xp):
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


def _pad_to_block(n_sym, block_size):
    """Round ``n_sym`` up to the next exact multiple of ``block_size``.

    Block-mode kernels iterate over exactly ``n_blocks`` blocks of size B.
    Ghost symbols at the tail (indices ``n_sym..n_sym_padded-1``) are
    zero-masked inside the kernel via the ``valid_mask`` / ``can_update``
    guards so they do not affect weights or outputs.

    Parameters
    ----------
    n_sym      : int — actual symbol count
    block_size : int — B, number of symbols per block

    Returns
    -------
    n_blocks     : int — ``ceil(n_sym / block_size)``
    n_sym_padded : int — ``n_blocks * block_size``
    """
    n_blocks = (n_sym + block_size - 1) // block_size
    return n_blocks, n_blocks * block_size


def _init_butterfly_weights(num_ch, num_taps, jnp, sps=2, center_tap=None):
    """Build center-tap identity butterfly weight matrix.

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


def _prepare_training(
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


def _unpack_result(
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
    """Convert raw JAX scan outputs into an ``EqualizerResult``.

    The scan kernels emit arrays shaped for the *padded* symbol count.
    This function:
      1. Transfers JAX arrays back to NumPy/CuPy (via ``from_jax``).
      2. Transposes from ``(N_sym, C)`` scan layout to ``(C, N_sym)`` convention.
      3. Truncates to ``n_sym`` (block mode only) to strip zero-padded tail symbols.
      4. Squeezes the channel dimension for 1-D SISO inputs (``was_1d=True``).
      5. Optionally keeps the weight-trajectory array.

    Parameters
    ----------
    y_hat_jax       : (N_sym, C) JAX array — equalized symbols
    errors_jax      : (N_sym, C) JAX array — complex errors
    W_final_jax     : (C, C, num_taps) JAX array — final weights
    w_hist_jax      : (N_sym or n_blocks, C, C, num_taps) — weight history
    was_1d          : bool — if True, squeeze C=1 dimension from outputs
    store_weights   : bool — if False, ``weights_history`` is None
    n_sym           : int or None — truncation length (None = no truncation)
    xp              : array module for the output (np or cp)
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
    block_size: int = 1,
    num_train_symbols: Optional[int] = None,
    device: Optional[str] = "cpu",
    center_tap: Optional[int] = None,
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
        If None, pure DD mode (requires ``reference_constellation``).
    num_taps : int, default 21
        Number of equalizer taps per FIR filter. For fractionally-spaced
        equalization (sps > 1), use at least ``4 * sps`` taps.
    step_size : float, default 0.01
        NLMS normalized step size (mu). After dividing the gradient by
        instantaneous input power ``||x||^2``, the effective correlation matrix
        has unit eigenvalues, so the stability interval collapses to the fixed
        range ``(0, 2)`` regardless of signal power. Values near 2 converge
        fast but with high steady-state misadjustment; values near 0 converge
        slowly but reach lower residual MSE. Typical: 0.01–0.1.
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
    block_size : int, default 1
        Number of symbols processed in parallel per weight update.
        ``block_size=1`` is the classical sample-by-sample algorithm.
        Larger values (8–64) improve GPU throughput by reducing
        ``lax.scan`` iterations at the cost of slower adaptation.
    num_train_symbols : int, optional
        Limits the number of training symbols used. If provided, the
        equalizer will forcefully switch to blind Decision-Directed (DD)
        mode after this many symbols, even if more training symbols
        are available in the array.
    device : str, optional
        Target device for JAX computations (e.g., 'cpu', 'gpu', 'tpu').
        Default is 'cpu'. JAX will automatically use the specified accelerator if available.
    center_tap : int, optional
        Index of the center tap. If None, defaults to ``num_taps // 2``.

    Returns
    -------
    EqualizerResult
        Equalized symbols, final weights, error history, and optionally
        weight trajectory. Arrays reside on the same backend as input.

    Warnings
    --------
    **GPU mode is typically slower than CPU for adaptive equalizers.**
    LMS is inherently sequential: each symbol's weight update depends on the
    previous weights, so ``lax.scan`` serializes execution even on GPU.
    The per-step arithmetic (a ``num_taps``-length dot product) is far too
    small to saturate GPU compute units, while kernel-launch and
    device-memory-transfer overhead dominate.  In practice, ``device='cpu'``
    is 2–10× faster for typical SISO sequences up to ~100 k symbols.  Use
    ``device='gpu'`` only when the number of MIMO channels (``num_ch``) is
    large enough to amortize GPU launch costs, or when batching many
    independent signals externally.

    **Block mode (``block_size > 1``) trades convergence quality for
    throughput.**  All B symbols in a block are filtered with the same frozen
    weights from the block boundary, producing a stale gradient: the filter
    takes one composite step derived from an outdated state.  This is
    equivalent to multiplying the effective step size by B, which causes:

    * Higher steady-state MSE floor (excess misadjustment).
    * Overshooting and weight oscillation in rapidly varying channels.
    * Tracking lag of up to B symbols behind channel variations.

    Use ``block_size=1`` for best equalization quality.  Larger values are
    only beneficial when JAX compilation overhead dominates runtime (e.g.,
    very short signals evaluated repeatedly) or for GPU MIMO with many
    channels where B is kept small (8–16).
    """
    logger.info(
        f"LMS equalizer: num_taps={num_taps}, mu={step_size}, sps={sps}, "
        f"block_size={block_size}, num_train_symbols={num_train_symbols}"
    )
    jax, jnp, _ = _get_jax()
    if jax is None:
        raise ImportError("JAX is required for adaptive equalizers.")

    samples, xp, _ = dispatch(samples)
    stride = int(sps)
    _validate_sps(sps, num_taps)

    # RMS-normalize samples; scale training to match symbol-rate power
    if training_symbols is not None:
        training_symbols, _, _ = dispatch(training_symbols)
        if num_train_symbols is not None:
            training_symbols = training_symbols[..., :num_train_symbols]
    samples, training_symbols = _normalize_inputs(samples, training_symbols, sps, xp)

    was_1d = samples.ndim == 1
    if was_1d:
        num_ch = 1
        n_samples = samples.shape[0]
    else:
        num_ch, n_samples = samples.shape

    n_sym = n_samples // stride

    # Compute block dimensions first so pad_right can cover the scan's full
    # window reach. The last dynamic_slice starts at (n_sym_padded-1)*stride,
    # so samples_padded must have length >= n_sym_padded*stride + num_taps - 1.
    if block_size > 1:
        n_blocks, n_sym_padded = _pad_to_block(n_sym, block_size)
    else:
        n_sym_padded = n_sym

    c_tap = center_tap if center_tap is not None else num_taps // 2
    pad_left = c_tap
    pad_right = n_sym_padded * stride - n_samples + num_taps - 1 - pad_left
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
        # Infer constellation from unique training values, handling on the current backend
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

    train_full, n_train_aligned = _prepare_training(
        training_symbols,
        num_ch,
        n_sym_padded,
        num_train_symbols=num_train_symbols,
    )

    # Convert to JAX — preserves device or overrides if `device` is given
    x_jax = to_jax(samples_padded, device=device)
    if was_1d:
        x_jax = x_jax[None, :]

    # Ensure all JAX arrays are on the target device
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
    w_init = _init_butterfly_weights(
        num_ch, num_taps, jnp, sps=sps, center_tap=center_tap
    )
    w_init = to_jax(w_init, device=platform)
    mu_jax = to_jax(jnp.float32(step_size), device=platform)
    n_train_jax = to_jax(jnp.int32(n_train_aligned), device=platform)
    n_sym_jax = to_jax(jnp.int32(n_sym), device=platform)

    if block_size > 1:
        scan_fn = _get_jitted_block_lms(
            num_taps,
            stride,
            len(constellation_np),
            num_ch,
            block_size,
        )
        y_jax, e_jax, W_jax, wh_jax = scan_fn(
            x_jax,
            train_jax,
            const_jax,
            w_init,
            mu_jax,
            n_train_jax,
            n_blocks,  # static arg — must be Python int
            n_sym_jax,
        )
    else:
        scan_fn = _get_jitted_lms(num_taps, stride, len(constellation_np), num_ch)
        y_jax, e_jax, W_jax, wh_jax = scan_fn(
            x_jax, train_jax, const_jax, w_init, mu_jax, n_train_jax
        )

    return _unpack_result(
        y_jax,
        e_jax,
        W_jax,
        wh_jax,
        was_1d,
        store_weights,
        n_sym=n_sym if block_size > 1 else None,
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
    sps: int = 2,
    store_weights: bool = False,
    block_size: int = 1,
    num_train_symbols: Optional[int] = None,
    device: Optional[str] = "cpu",
    center_tap: Optional[int] = None,
) -> EqualizerResult:
    """
    Recursive Least Squares adaptive equalizer with butterfly MIMO support.

    RLS converges faster than LMS at the cost of higher per-symbol
    complexity (maintains an inverse correlation matrix per output stream).

    Parameters
    ----------
    samples : array_like
        Input signal samples. Shape: ``(N_samples,)`` or ``(C, N_samples)``.
        Typically at 2 samples/symbol for fractionally-spaced equalization.
    training_symbols : array_like, optional
        Known symbols for data-aided adaptation (at symbol rate, 1 SPS).
    num_taps : int, default 21
        Number of equalizer taps per FIR filter.
    forgetting_factor : float, default 0.99
        RLS forgetting factor (lambda). Range: (0, 1].
        Values close to 1 give longer memory.
    delta : float, default 0.01
        Regularization for initial inverse correlation matrix
        ``P = (1/delta) * I``.
    leakage : float, default 0.0
        Weight-decay coefficient (γ) applied to the tap vector at every step::

            W ← (1 − γ)·W + k·ē         # leaky weight update
            P ← (P − k·x̄ᴴP) / λ         # standard Riccati (unchanged)

        Weight decay exponentially suppresses tap weights in the null subspace —
        noise-only frequency bands that arise in T/2-spaced (sps=2) signals —
        without inflating ``P``'s eigenvalues.  Adding ``γI`` directly to ``P``
        would increase P's eigenvalues, effectively shrinking R_xx eigenvalues
        toward zero and worsening null-subspace amplification.
        A value of ``0.0`` (default) gives standard RLS.
        For fractionally-spaced equalizers start with ``leakage=1e-4`` and
        increase if steady-state EVM remains high.  Values above ~``1e-2``
        noticeably slow convergence.
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
        If True, stores weight trajectory.
    block_size : int, default 1
        Number of symbols per weight update block. ``block_size=1`` is
        standard sample-by-sample RLS. Larger values use a hybrid
        approach: parallel forward pass + sequential P/W update within
        each block, reducing outer-scan overhead.
    num_train_symbols : int, optional
        Limits the number of training symbols used. If provided, the
        equalizer will forcefully switch to blind Decision-Directed (DD)
        mode after this many symbols, even if more training symbols
        are available in the array.
    device : str, optional
        Target device for JAX computations ("e.g., 'cpu', 'gpu', 'tpu'").
        Default is 'cpu'. JAX will automatically use the specified accelerator if available.
    center_tap : int, optional
        Index of the center tap. If None, defaults to ``num_taps // 2``.

    Returns
    -------
    EqualizerResult
        Equalized symbols, final weights, error history, and optionally
        weight trajectory.

    Warnings
    --------
    **GPU mode is typically slower than CPU for adaptive equalizers.**
    RLS is inherently sequential: each symbol's ``(W, P)`` update depends on
    the previous inverse correlation matrix ``P``, so ``lax.scan`` serializes
    execution even on GPU.  RLS has higher per-step cost than LMS
    (O(num_taps²) for the rank-1 P update vs. O(num_taps) for LMS), making
    the compute-to-launch-overhead ratio even worse on GPU.  Use
    ``device='cpu'`` unless you have a very large MIMO configuration.

    **Block mode (``block_size > 1``) trades convergence quality for
    throughput.**  The block-RLS implementation uses a hybrid scheme: the
    forward pass over B symbols is parallelized with frozen weights, then W
    and P are updated sequentially within the block.  Despite the sequential
    inner update, the forward-pass weights are already stale by up to B
    symbols at the start of each block, which:

    * Increases effective step size and steady-state misadjustment.
    * Causes the P matrix to accumulate error from outputs computed with
      outdated weights, corrupting the covariance estimate.
    * Degrades tracking in non-stationary channels.

    RLS already converges faster than LMS in sample-by-sample mode; the
    throughput gains of block mode rarely justify the convergence loss.
    Use ``block_size=1`` unless compile-time overhead is a bottleneck.
    """
    logger.info(
        f"RLS equalizer: num_taps={num_taps}, lambda={forgetting_factor}, "
        f"delta={delta:.2e}, leakage={leakage:.2e}, sps={sps}, "
        f"block_size={block_size}, num_train_symbols={num_train_symbols}"
    )
    jax, jnp, _ = _get_jax()
    if jax is None:
        raise ImportError("JAX is required for adaptive equalizers.")

    samples, xp, _ = dispatch(samples)
    stride = int(sps)
    _validate_sps(sps, num_taps)

    # RMS-normalize samples; scale training to match symbol-rate power
    if training_symbols is not None:
        training_symbols, _, _ = dispatch(training_symbols)
        if num_train_symbols is not None:
            training_symbols = training_symbols[..., :num_train_symbols]
    samples, training_symbols = _normalize_inputs(samples, training_symbols, sps, xp)

    was_1d = samples.ndim == 1
    if was_1d:
        num_ch = 1
        n_samples = samples.shape[0]
    else:
        num_ch, n_samples = samples.shape

    n_sym = n_samples // stride

    # Compute block dimensions first — same reasoning as lms().
    if block_size > 1:
        n_blocks, n_sym_padded = _pad_to_block(n_sym, block_size)
    else:
        n_sym_padded = n_sym

    c_tap = center_tap if center_tap is not None else num_taps // 2
    pad_left = c_tap
    pad_right = n_sym_padded * stride - n_samples + num_taps - 1 - pad_left
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
        # Infer constellation from unique training values, handling on the current backend
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

    train_full, n_train_aligned = _prepare_training(
        training_symbols,
        num_ch,
        n_sym_padded,
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
    w_init = _init_butterfly_weights(
        num_ch, num_taps, jnp, sps=sps, center_tap=center_tap
    )
    w_init = to_jax(w_init, device=platform)

    regressor_dim = num_ch * num_taps
    P_init = jnp.eye(regressor_dim, dtype="complex64") / delta
    P_init = to_jax(P_init, device=platform)
    lam_jax = to_jax(jnp.float32(forgetting_factor), device=platform)
    n_train_jax = to_jax(jnp.int32(n_train_aligned), device=platform)

    # Early-halt boundary: freeze W and P once the sliding window reaches the
    # right zero-padding (last num_taps//2 symbols have contaminated windows).
    n_update_halt = max(0, n_sym - num_taps // 2)
    leakage_jax = to_jax(jnp.float32(leakage), device=platform)
    n_update_halt_jax = to_jax(jnp.int32(n_update_halt), device=platform)

    logger.debug(
        f"RLS internals: n_sym={n_sym}, n_train={n_train_aligned}, "
        f"n_update_halt={n_update_halt}, leakage={leakage:.2e}, delta={delta:.2e}"
    )

    if block_size > 1:
        scan_fn = _get_jitted_block_rls(
            num_taps, stride, len(constellation_np), num_ch, block_size
        )
        y_jax, e_jax, W_jax, wh_jax = scan_fn(
            x_jax,
            train_jax,
            const_jax,
            w_init,
            P_init,
            lam_jax,
            n_train_jax,
            n_blocks,  # static arg — must be Python int
            leakage_jax,
            n_update_halt_jax,
        )
    else:
        scan_fn = _get_jitted_rls(num_taps, stride, len(constellation_np), num_ch)
        y_jax, e_jax, W_jax, wh_jax = scan_fn(
            x_jax, train_jax, const_jax, w_init, P_init, lam_jax, n_train_jax,
            leakage_jax, n_update_halt_jax,
        )

    # Truncate last num_taps//2 symbols: those windows overlap the right zero-padding,
    # producing near-zero y that the slicer maps to ~1.0 magnitude, creating a spurious
    # MSE/EVM spike. n_update_halt = n_sym - num_taps//2 is the correct output length.
    return _unpack_result(
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
    block_size: int = 1,
    device: Optional[str] = "cpu",
    center_tap: Optional[int] = None,
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
    block_size : int, default 1
        Number of symbols processed in parallel per weight update.
        ``block_size=1`` is the classical sample-by-sample algorithm.
        Larger values (8–64) improve GPU throughput. CMA is more
        sensitive to block size than LMS due to its non-convex cost
        surface — recommend ``block_size <= 32``.
    device : str, optional
        Target device for JAX computations (e.g., 'cpu', 'gpu', 'tpu').
        Default is 'cpu'. JAX will automatically use the specified accelerator if available.
    center_tap : int, optional
        Index of the center tap. If None, defaults to ``num_taps // 2``.

    Returns
    -------
    EqualizerResult
        Equalized symbols, final weights, CMA error history, and optionally
        weight trajectory.

    Warnings
    --------
    **GPU mode is typically slower than CPU for adaptive equalizers.**
    CMA is inherently sequential: each weight update depends on the previous
    weights, so ``lax.scan`` serializes execution even on GPU.  The per-step
    arithmetic (a ``num_taps``-length dot product) is too small to saturate
    GPU compute units, and kernel-launch overhead dominates.  Use
    ``device='cpu'`` for typical SISO sequences; ``device='gpu'`` is only
    beneficial for large MIMO configurations or externally batched signals.

    **Block mode (``block_size > 1``) trades convergence quality for
    throughput, with greater risk for CMA than for LMS.**  All B symbols in
    a block are filtered with the same frozen weights, producing a stale
    gradient.  Because CMA's Godard cost surface is non-convex and
    higher-order, the stale-gradient step can push the filter toward a
    spurious local minimum or cause divergence — especially at large block
    sizes.  Concretely:

    * The effective gradient magnitude grows with B, increasing the risk
      of escaping the basin of attraction around the correct solution.
    * Convergence to a phase-rotated or permuted solution becomes more
      likely as B increases.
    * Steady-state excess MSE scales approximately with block size.

    Keep ``block_size <= 16`` and prefer ``block_size=1`` whenever signal
    quality matters more than throughput.
    """
    logger.info(
        f"CMA equalizer: num_taps={num_taps}, mu={step_size}, sps={sps}, "
        f"block_size={block_size}"
    )
    jax, jnp, _ = _get_jax()
    if jax is None:
        raise ImportError("JAX is required for adaptive equalizers.")

    samples, xp, _ = dispatch(samples)
    stride = int(sps)
    _validate_sps(sps, num_taps)

    # RMS-normalize samples to unit symbol-rate power (CMA has no training)
    samples, _ = _normalize_inputs(samples, None, sps, xp)

    was_1d = samples.ndim == 1
    if was_1d:
        num_ch = 1
        n_samples = samples.shape[0]
    else:
        num_ch, n_samples = samples.shape

    # Compute R2 from the (now unit-power) constellation
    if modulation is not None and order is not None:
        from .mapping import gray_constellation

        const = gray_constellation(modulation, order, unipolar=unipolar)
        r2 = float(np.mean(np.abs(const) ** 4) / np.mean(np.abs(const) ** 2))
        logger.debug(f"CMA R2 from {modulation.upper()}-{order}: {r2:.4f}")
    else:
        r2 = 1.0

    n_sym = n_samples // stride

    # Compute block dimensions first — same reasoning as lms().
    if block_size > 1:
        n_blocks, n_sym_padded = _pad_to_block(n_sym, block_size)
    else:
        n_sym_padded = n_sym

    c_tap = center_tap if center_tap is not None else num_taps // 2
    pad_left = c_tap
    pad_right = n_sym_padded * stride - n_samples + num_taps - 1 - pad_left
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

    w_init = _init_butterfly_weights(
        num_ch, num_taps, jnp, sps=sps, center_tap=center_tap
    )
    w_init = to_jax(w_init, device=platform)
    mu_jax = to_jax(jnp.float32(step_size), device=platform)
    r2_jax = to_jax(jnp.float32(r2), device=platform)
    n_sym_jax = to_jax(jnp.int32(n_sym), device=platform)

    if block_size > 1:
        scan_fn = _get_jitted_block_cma(num_taps, stride, num_ch, block_size)
        y_jax, e_jax, W_jax, wh_jax = scan_fn(
            x_jax,
            w_init,
            mu_jax,
            r2_jax,
            n_blocks,  # static arg
            n_sym_jax,
        )
    else:
        scan_fn = _get_jitted_cma(num_taps, stride, num_ch)
        y_jax, e_jax, W_jax, wh_jax = scan_fn(x_jax, w_init, mu_jax, r2_jax, n_sym)

    return _unpack_result(
        y_jax,
        e_jax,
        W_jax,
        wh_jax,
        was_1d,
        store_weights,
        n_sym=n_sym if block_size > 1 else None,
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
        inv_term = xp.linalg.inv(HHh + reg * eye)
        Wk = Hk_H @ inv_term

        # Wk: (N_fft, num_ch, num_ch), Y: (num_ch, num_blocks, N_fft)
        # Vectorized batch matrix multiplication across frequency bins and blocks
        X_hat_f = xp.einsum("k c j, j b k -> c b k", Wk, Y)

    # IFFT back to time domain
    x_hat = xp.fft.ifft(X_hat_f, n=N_fft, axis=-1)

    # Symmetric discard: keep the center N_fft//2 samples of each block.
    # The first discard samples contain causal circular transient; the last
    # discard samples contain anti-causal (IIR precursor) circular transient.
    valid = x_hat[:, :, discard : discard + B]
    out = valid.reshape(num_ch, -1)[:, :N]
    return out[0] if was_1d else out
