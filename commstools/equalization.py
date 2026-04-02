"""
Adaptive and block equalization algorithms.

This module provides equalizer implementations for compensating
inter-symbol interference (ISI) and channel distortion in digital
communication systems.

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

All adaptive equalization support a **butterfly MIMO** topology: for an input
of shape ``(C, N)``, the equalizer maintains a ``(C, C, num_taps)`` weight
matrix so that each output stream is a filtered combination of *all* input
streams. This enables cross-channel interference cancellation (e.g.
dual-polarization demultiplexing in coherent optical, spatial MIMO demux).

Input SPS Convention
--------------------
By default, adaptive equalizers expect T/2-spaced input (2 samples/symbol,
``sps=2``), the industry standard for coherent optical and many wireless
systems.  The equalizer decimates by ``sps``, producing one output symbol
per ``sps`` input samples.

**sps=1 (symbol-spaced mode)** is accepted for multi-stage pipelines where
a second equalization pass is needed *after* a prior FSE stage + FOE + CPR.
At that point the signal is already at 1 SPS and upsampling before
re-equalization would be unnecessary.  Use a short filter (3-11 taps) for
residual ISI cleanup in this mode.

Functions
---------
lms :
    Least Mean Squares / Normalized LMS adaptive equalizer.
rls :
    Recursive Least Squares adaptive equalizer.
cma :
    Constant Modulus Algorithm blind equalizer (optionally pilot-aided).
rde :
    Radius Directed Equalizer blind equalizer (optionally pilot-aided).
build_pilot_ref :
    Build dense pilot reference arrays for passing to ``cma`` / ``rde``.
zf_equalizer :
    Zero-Forcing / MMSE frequency-domain block equalizer.
block_lms :
    Block LMS equalizer with frequency-domain gradient accumulation (GPU-optimised).
apply_taps :
    Apply frozen equalizer taps to a new signal (no weight updates).
"""

import functools
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np

from .backend import ArrayType, _get_jax, dispatch, from_jax, to_jax, to_device
from .filtering import _ols_backward, _ols_forward
from .logger import logger


# -----------------------------------------------------------------------------
# RESULT CONTAINER
# -----------------------------------------------------------------------------


@dataclass
class EqualizerResult:
    """Container for equalizer outputs.

    Attributes
    ----------
    y_hat : ArrayType
        Equalized symbol sequence.  Shape: ``(N_sym,)`` SISO or
        ``(C, N_sym)`` MIMO.
    weights : ArrayType
        Final tap weight vector. Shape: ``(num_taps,)`` for SISO
        or ``(C, C, num_taps)`` for MIMO butterfly.
    error : ArrayType
        Error signal history. Shape: ``(N_sym,)`` or ``(C, N_sym)``.
    weights_history : ArrayType or None
        Tap weight evolution over time. Only populated when
        ``store_weights=True``.
    num_train_symbols : int
        Number of data-aided training symbols consumed (LMS/RLS).  Used to
        discard the DA-trained transient before computing steady-state metrics
        like EVM, SNR, and BER.
    input_norm_factor : float or np.ndarray
        Normalization factor(s) ``rms(samples, axis=-1) * sqrt(sps)`` applied
        per-channel by ``_normalize_inputs`` before equalization.  For SISO
        this is a plain ``float``; for MIMO it is a 1-D ``np.ndarray`` of
        shape ``(C,)`` — one factor per input stream.
        Stored so callers can apply the post-hoc power correction needed when
        passing a different capture (e.g. vacuum noise) through the same
        frozen taps without disturbing the per-channel ratio:

        .. code-block:: python

            α = signal_result.input_norm_factor  # float or (C,) array
            β = noise_result.input_norm_factor
            P_noise_corrected = (
                np.mean(np.abs(noise_result.y_hat) ** 2, axis=-1) * (β / α) ** 2
            )

        ``P_signal / P_noise_corrected`` is then the physically meaningful
        per-channel signal-to-noise power ratio preserved through the DSP chain.
        At ``sps=1`` this factor equals the plain per-channel RMS of the input symbols.
    tail_trim : int
        Number of symbols trimmed from the tail of ``y_hat`` to remove the
        zero-padding contamination zone.  Non-zero only for RLS (equals
        ``num_taps // 2``).  If non-zero, trim reference arrays to match::

            source_symbols = source_symbols[..., :-result.tail_trim]
            source_bits    = source_bits[..., :-result.tail_trim * bits_per_symbol]
    phase_trajectory : np.ndarray or None
        Per-symbol phase estimates produced by the inline CPR stage, in
        radians.  ``None`` when ``cpr_type=None``.

        Shape: ``(N_sym,)`` for SISO, ``(C, N_sym)`` for MIMO butterfly.

        The values are the instantaneous phase corrections *applied* to each
        symbol before the hard decision and weight update, i.e.
        ``y_hat[n] = (W^H x[n]) · exp(-j · phase_trajectory[n])``.
        Useful for post-hoc phase-noise analysis, cycle-slip diagnostics, and
        as a warm-start phase estimate for a subsequent CPR stage.
    """

    y_hat: ArrayType
    weights: ArrayType
    error: ArrayType
    weights_history: Optional[ArrayType] = None
    num_train_symbols: int = 0
    input_norm_factor: Union[float, np.ndarray] = 1.0
    tail_trim: int = 0
    phase_trajectory: Optional[ArrayType] = None


def _log_equalizer_exit(
    result: "EqualizerResult",
    name: str,
    debug_plot: bool = False,
    check_convergence: bool = False,
    plot_smoothing: int = 50,
) -> "EqualizerResult":
    """Log exit MSE and optionally show a debug plot for an EqualizerResult."""
    if result.error is not None:
        err = to_device(result.error, "cpu")
        n_sym = err.shape[-1]  # time axis; correct for (N_sym,) and (C, N_sym)
        window = max(1, min(100, n_sym))

        if err.ndim == 1:
            # SISO
            mse_final = float(np.mean(np.abs(err[-window:]) ** 2))
            mse_db = 10.0 * np.log10(mse_final + 1e-30)
            logger.info(f"{name}: exit MSE={mse_db:.1f} dB (final {window} symbols)")
        else:
            # MIMO: log per-channel MSE; keep mean for convergence check
            per_ch_mse = [
                float(np.mean(np.abs(err[c, -window:]) ** 2))
                for c in range(err.shape[0])
            ]
            parts = ", ".join(
                f"ch{c}={10.0 * np.log10(m + 1e-30):.1f}"
                for c, m in enumerate(per_ch_mse)
            )
            logger.info(f"{name}: exit MSE (final {window} symbols): {parts} dB")
            mse_final = float(np.mean(per_ch_mse))
            mse_db = 10.0 * np.log10(mse_final + 1e-30)

        if check_convergence and n_sym >= 20:
            init_window = max(1, min(100, n_sym // 10))
            mse_init = float(np.mean(np.abs(err[..., :init_window]) ** 2))
            if mse_init > 0 and mse_final > mse_init * 0.9:
                logger.warning(
                    f"{name}: convergence may be poor — "
                    f"final MSE ({mse_db:.1f} dB) not significantly below "
                    f"initial MSE ({10.0 * np.log10(mse_init + 1e-30):.1f} dB). "
                    "Consider reducing step_size or increasing signal length."
                )

    if debug_plot:
        from . import plotting as _plotting  # lazy import avoids circular dep

        _plotting.equalizer_result(result, smoothing=plot_smoothing)

    return result


# -----------------------------------------------------------------------------
# NUMBA LAZY LOADER
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# NUMBA KERNELS — ADAPTIVE EQUALIZER LOOPS
# -----------------------------------------------------------------------------
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
    """JIT-compile and cache the Numba LMS butterfly loop kernel.

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
            # step_size     : float32              — LMS step size μ; stable when 0 < μ < 2/(C·num_taps·P_x)
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

                # Weight update: W[i,j,t] += μ * conj(e[i]) * X_wins[j,t]
                for i in range(C):
                    ce_i = np.conj(e[i])
                    for j in range(C):
                        for t in range(num_taps):
                            W[i, j, t] = W[i, j, t] + step_size * ce_i * X_wins[j, t]

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

                    # Exploit Hermitian symmetry P = P^H: for each j, (x^H P)[j] = conj((Px)[j]).
                    # Reuse the already-computed Px to avoid a second O(N²) mat-vec.
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


def _get_numba_lms_cpr():
    """JIT-compile and cache the Numba LMS+CPR butterfly loop kernel.

    Extends the baseline LMS kernel with an inline carrier phase tracker
    (PLL or single-shot BPS) and a causal cycle-slip corrector.

    cpr_mode 1 = PLL (DD second-order PI loop).
    cpr_mode 2 = BPS (block-averaged Blind Phase Search, window=bps_block_size).

    Returns
    -------
    lms_cpr_loop : numba-compiled callable
    """
    if "lms_cpr" not in _NUMBA_KERNELS:
        numba_mod = _get_numba()
        if numba_mod is None:
            raise ImportError("Numba is required for backend='numba'.")

        @numba_mod.njit(cache=True, fastmath=True, nogil=True)
        def lms_cpr_loop(
            x_padded,
            training,
            constellation,
            bps_phases_neg,
            bps_angles,
            bps_block_size,
            bps_joint_channels,
            W,
            step_size,
            n_train,
            stride,
            store_weights,
            cpr_mode,
            pll_mu,
            pll_beta,
            symmetry,
            cs_enabled,
            cs_threshold,
            pll_phi,
            pll_freq,
            cs_buf_x,
            cs_buf_y,
            cs_buf_ptr,
            cs_buf_n,
            cs_stats,
            y_out,
            e_out,
            phase_out,
            w_hist_out,
        ):
            # x_padded          : (C, N_pad)        complex64
            # training          : (C, N_sym)         complex64
            # constellation     : (M,)               complex64
            # bps_phases_neg    : (B,)               complex64  exp(-j*theta_k)
            # bps_angles        : (B,)               float32    theta_k
            # bps_block_size    : int32              BPS averaging window (≥1)
            # bps_joint_channels: bool               True → joint metric across C channels
            # W                 : (C, C, T)          complex64  — in-place
            # step_size     : float32
            # n_train       : int32
            # stride        : int
            # store_weights : bool
            # cpr_mode      : int32  1=pll 2=bps
            # pll_mu        : float32
            # pll_beta      : float32
            # symmetry      : int32  — cycle-slip quantum = 2π/symmetry
            # cs_enabled    : bool
            # cs_threshold  : float32
            # pll_phi       : (C,) float32          — in-place
            # pll_freq      : (C,) float32          — in-place
            # cs_buf_x      : (C, H) float64        — in-place
            # cs_buf_y      : (C, H) float64        — in-place
            # cs_buf_ptr    : (C,) int64            — in-place
            # cs_buf_n      : (C,) int64            — in-place
            # cs_stats      : (C, 4) float64  [Sx,Sy,Sxx,Sxy] — in-place
            # y_out         : (N_sym, C) complex64
            # e_out         : (N_sym, C) complex64
            # phase_out     : (N_sym, C) float32
            # w_hist_out    : (N_sym or 1, C, C, T) complex64
            C = W.shape[0]
            num_taps = W.shape[2]
            n_sym = y_out.shape[0]
            M = len(constellation)
            B = len(bps_phases_neg)
            H = cs_buf_x.shape[1]
            two_pi = np.float64(2.0 * np.pi)
            quantum = two_pi / np.float64(symmetry)

            X_wins = np.empty((C, num_taps), dtype=np.complex64)
            y_raw = np.empty(C, dtype=np.complex64)
            y_fin = np.empty(C, dtype=np.complex64)
            phi_c = np.empty(C, dtype=np.float32)
            d_sym = np.empty(C, dtype=np.complex64)
            e_clean = np.empty(C, dtype=np.complex64)
            e_eq = np.empty(C, dtype=np.complex64)
            phi_hat_bps = np.zeros(C, dtype=np.float32)
            bps_buf = np.zeros((C, bps_block_size), dtype=np.complex64)
            bps_buf_ptr = np.int64(0)

            for idx in range(n_sym):
                sample_idx = idx * stride

                # Window extraction
                for c in range(C):
                    for t in range(num_taps):
                        X_wins[c, t] = x_padded[c, sample_idx + t]

                # Butterfly forward pass: y_raw[i] = Σ_j conj(W[i,j]) · X
                for i in range(C):
                    acc = np.complex64(0.0)
                    for j in range(C):
                        for t in range(num_taps):
                            acc = acc + np.conj(W[i, j, t]) * X_wins[j, t]
                    y_raw[i] = acc

                # ── CPR: Phase estimation ──────────────────────────────────

                # BPS: fill circular buffer slot with current y_raw
                if cpr_mode == 2:
                    slot = bps_buf_ptr % np.int64(bps_block_size)
                    for i in range(C):
                        bps_buf[i, slot] = y_raw[i]
                    bps_buf_ptr = bps_buf_ptr + np.int64(1)
                    fill = (
                        np.int32(bps_buf_ptr)
                        if bps_buf_ptr < np.int64(bps_block_size)
                        else bps_block_size
                    )

                    if bps_joint_channels:
                        # Accumulate min-dist metric jointly across all channels
                        best_k_joint = np.int32(0)
                        min_tot_joint = np.float32(1e38)
                        for k in range(B):
                            metric_k = np.float32(0.0)
                            for n in range(fill):
                                for i in range(C):
                                    y_rot = bps_buf[i, n] * bps_phases_neg[k]
                                    d2_min = np.float32(1e38)
                                    for m in range(M):
                                        dv = y_rot - constellation[m]
                                        d2 = dv.real * dv.real + dv.imag * dv.imag
                                        if d2 < d2_min:
                                            d2_min = d2
                                    metric_k = metric_k + d2_min
                            if metric_k < min_tot_joint:
                                min_tot_joint = metric_k
                                best_k_joint = k
                        for i in range(C):
                            phi_hat_bps[i] = bps_angles[best_k_joint]
                    else:
                        # Independent BPS per channel
                        for i in range(C):
                            best_k = np.int32(0)
                            min_tot = np.float32(1e38)
                            for k in range(B):
                                metric_k = np.float32(0.0)
                                for n in range(fill):
                                    y_rot = bps_buf[i, n] * bps_phases_neg[k]
                                    d2_min = np.float32(1e38)
                                    for m in range(M):
                                        dv = y_rot - constellation[m]
                                        d2 = dv.real * dv.real + dv.imag * dv.imag
                                        if d2 < d2_min:
                                            d2_min = d2
                                    metric_k = metric_k + d2_min
                                if metric_k < min_tot:
                                    min_tot = metric_k
                                    best_k = k
                            phi_hat_bps[i] = bps_angles[best_k]

                for i in range(C):
                    if cpr_mode == 1:  # PLL: read current integrator state
                        phi_hat = pll_phi[i]
                    else:  # BPS: use block-averaged estimate
                        phi_hat = phi_hat_bps[i]

                    # ── Cycle-slip correction ──────────────────────────────
                    if cs_enabled:
                        n_b = cs_buf_n[i]
                        ptr = cs_buf_ptr[i]
                        x_b = np.float64(idx)
                        y_b = np.float64(phi_hat)

                        if n_b == 0:
                            phi_corr = phi_hat
                        else:
                            if n_b < np.int64(10):
                                last_pos = (ptr - np.int64(1) + np.int64(H)) % np.int64(
                                    H
                                )
                                phi_expected = cs_buf_y[i, last_pos]
                            else:
                                Sx = cs_stats[i, 0]
                                Sy = cs_stats[i, 1]
                                Sxx = cs_stats[i, 2]
                                Sxy = cs_stats[i, 3]
                                n_f = np.float64(n_b)
                                denom = n_f * Sxx - Sx * Sx
                                if np.abs(denom) > np.float64(1e-30):
                                    slope = (n_f * Sxy - Sx * Sy) / denom
                                    intercept = (Sy - slope * Sx) / n_f
                                else:
                                    slope = np.float64(0.0)
                                    intercept = Sy / n_f
                                phi_expected = slope * x_b + intercept

                            diff = y_b - phi_expected
                            k_slip = np.int64(np.round(diff / quantum))
                            if np.abs(diff) > np.float64(
                                cs_threshold
                            ) and k_slip != np.int64(0):
                                y_b = y_b - np.float64(k_slip) * quantum
                            phi_corr = np.float32(y_b)

                        # Update circular buffer and incremental stats
                        write_pos = ptr % np.int64(H)
                        if n_b == np.int64(H):
                            old_x = cs_buf_x[i, write_pos]
                            old_y = cs_buf_y[i, write_pos]
                            cs_stats[i, 0] -= old_x
                            cs_stats[i, 1] -= old_y
                            cs_stats[i, 2] -= old_x * old_x
                            cs_stats[i, 3] -= old_x * old_y
                        new_y = np.float64(phi_corr)
                        cs_buf_x[i, write_pos] = x_b
                        cs_buf_y[i, write_pos] = new_y
                        cs_stats[i, 0] += x_b
                        cs_stats[i, 1] += new_y
                        cs_stats[i, 2] += x_b * x_b
                        cs_stats[i, 3] += x_b * new_y
                        cs_buf_ptr[i] = ptr + np.int64(1)
                        if n_b < np.int64(H):
                            cs_buf_n[i] = n_b + np.int64(1)
                    else:
                        phi_corr = phi_hat

                    phi_c[i] = phi_corr
                    phase_out[idx, i] = phi_corr

                    # y_final = y_raw * exp(-j * phi_corr)
                    cos_p = np.cos(np.float64(phi_corr))
                    sin_p = np.sin(np.float64(phi_corr))
                    yr = y_raw[i].real * np.float32(cos_p) + y_raw[i].imag * np.float32(
                        sin_p
                    )
                    yi = -y_raw[i].real * np.float32(sin_p) + y_raw[
                        i
                    ].imag * np.float32(cos_p)
                    y_fin[i] = np.complex64(yr + yi * np.complex64(1j))
                    y_out[idx, i] = y_fin[i]

                # ── Slicer and error ───────────────────────────────────────
                for i in range(C):
                    if idx < n_train:
                        d_i = training[i, idx]
                    else:
                        min_dist = np.float32(1e38)
                        min_idx = 0
                        for k in range(M):
                            dv = y_fin[i] - constellation[k]
                            dist = dv.real * dv.real + dv.imag * dv.imag
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = k
                        d_i = constellation[min_idx]
                    d_sym[i] = d_i
                    e_clean[i] = d_i - y_fin[i]
                    e_out[idx, i] = e_clean[i]

                # ── PLL state update ───────────────────────────────────────
                if cpr_mode == 1:
                    for i in range(C):
                        # Cross-product phase detector: Im(y_fin · conj(d))
                        e_ph = (
                            y_fin[i].imag * d_sym[i].real
                            - y_fin[i].real * d_sym[i].imag
                        )
                        pll_phi[i] = (
                            pll_phi[i] + pll_mu * np.float32(e_ph) + pll_freq[i]
                        )
                        pll_freq[i] = pll_freq[i] + pll_beta * np.float32(e_ph)

                # ── De-rotate error and weight update ──────────────────────
                for i in range(C):
                    # e_eq = e_clean * exp(+j * phi_corr)  (back to pre-rotation plane)
                    cos_p = np.cos(np.float64(phi_c[i]))
                    sin_p = np.sin(np.float64(phi_c[i]))
                    er = e_clean[i].real * np.float32(cos_p) - e_clean[
                        i
                    ].imag * np.float32(sin_p)
                    ei = e_clean[i].real * np.float32(sin_p) + e_clean[
                        i
                    ].imag * np.float32(cos_p)
                    e_eq[i] = np.complex64(er + ei * np.complex64(1j))

                for i in range(C):
                    ce_i = np.conj(e_eq[i])
                    for j in range(C):
                        for t in range(num_taps):
                            W[i, j, t] = W[i, j, t] + step_size * ce_i * X_wins[j, t]

                if store_weights:
                    for i in range(C):
                        for j in range(C):
                            for t in range(num_taps):
                                w_hist_out[idx, i, j, t] = W[i, j, t]

        _NUMBA_KERNELS["lms_cpr"] = lms_cpr_loop
    return _NUMBA_KERNELS["lms_cpr"]


def _get_numba_rls_cpr():
    """JIT-compile and cache the Numba RLS+CPR butterfly loop kernel.

    Combines the Leaky-RLS Riccati update with the same inline CPR tracker
    used by the LMS CPR kernel.

    Returns
    -------
    rls_cpr_loop : numba-compiled callable
    """
    if "rls_cpr" not in _NUMBA_KERNELS:
        numba_mod = _get_numba()
        if numba_mod is None:
            raise ImportError("Numba is required for backend='numba'.")

        @numba_mod.njit(cache=True, fastmath=True, nogil=True)
        def rls_cpr_loop(
            x_padded,
            training,
            constellation,
            bps_phases_neg,
            bps_angles,
            bps_block_size,
            bps_joint_channels,
            W,
            P,
            lam,
            leakage,
            n_train,
            n_update_halt,
            stride,
            store_weights,
            cpr_mode,
            pll_mu,
            pll_beta,
            symmetry,
            cs_enabled,
            cs_threshold,
            pll_phi,
            pll_freq,
            cs_buf_x,
            cs_buf_y,
            cs_buf_ptr,
            cs_buf_n,
            cs_stats,
            y_out,
            e_out,
            phase_out,
            w_hist_out,
        ):
            # Same shapes as rls_loop plus CPR state — see lms_cpr_loop header.
            C = W.shape[0]
            num_taps = W.shape[2]
            n_sym = y_out.shape[0]
            N = C * num_taps
            M = len(constellation)
            B = len(bps_phases_neg)
            H = cs_buf_x.shape[1]
            two_pi = np.float64(2.0 * np.pi)
            quantum = two_pi / np.float64(symmetry)

            x_bar = np.empty(N, dtype=np.complex64)
            Px = np.empty(N, dtype=np.complex64)
            xH_P = np.empty(N, dtype=np.complex64)
            k_gain = np.empty(N, dtype=np.complex64)
            y_raw = np.empty(C, dtype=np.complex64)
            y_fin = np.empty(C, dtype=np.complex64)
            phi_c = np.empty(C, dtype=np.float32)
            d_sym = np.empty(C, dtype=np.complex64)
            e_clean = np.empty(C, dtype=np.complex64)
            e_eq = np.empty(C, dtype=np.complex64)
            phi_hat_bps = np.zeros(C, dtype=np.float32)
            bps_buf = np.zeros((C, bps_block_size), dtype=np.complex64)
            bps_buf_ptr = np.int64(0)

            lam_f32 = np.float32(lam)
            leak_term = np.float32(1.0) - np.float32(leakage)

            for idx in range(n_sym):
                sample_idx = idx * stride

                for j in range(C):
                    for t in range(num_taps):
                        x_bar[j * num_taps + t] = x_padded[j, sample_idx + t]

                # Butterfly forward pass
                for i in range(C):
                    acc = np.complex64(0.0)
                    for j in range(C):
                        for t in range(num_taps):
                            acc = acc + np.conj(W[i, j, t]) * x_bar[j * num_taps + t]
                    y_raw[i] = acc

                # ── CPR: Phase estimation ──────────────────────────────────

                # BPS: fill circular buffer slot with current y_raw
                if cpr_mode == 2:
                    slot = bps_buf_ptr % np.int64(bps_block_size)
                    for i in range(C):
                        bps_buf[i, slot] = y_raw[i]
                    bps_buf_ptr = bps_buf_ptr + np.int64(1)
                    fill = (
                        np.int32(bps_buf_ptr)
                        if bps_buf_ptr < np.int64(bps_block_size)
                        else bps_block_size
                    )

                    if bps_joint_channels:
                        best_k_joint = np.int32(0)
                        min_tot_joint = np.float32(1e38)
                        for k in range(B):
                            metric_k = np.float32(0.0)
                            for n in range(fill):
                                for i in range(C):
                                    y_rot = bps_buf[i, n] * bps_phases_neg[k]
                                    d2_min = np.float32(1e38)
                                    for m in range(M):
                                        dv = y_rot - constellation[m]
                                        d2 = dv.real * dv.real + dv.imag * dv.imag
                                        if d2 < d2_min:
                                            d2_min = d2
                                    metric_k = metric_k + d2_min
                            if metric_k < min_tot_joint:
                                min_tot_joint = metric_k
                                best_k_joint = k
                        for i in range(C):
                            phi_hat_bps[i] = bps_angles[best_k_joint]
                    else:
                        for i in range(C):
                            best_k = np.int32(0)
                            min_tot = np.float32(1e38)
                            for k in range(B):
                                metric_k = np.float32(0.0)
                                for n in range(fill):
                                    y_rot = bps_buf[i, n] * bps_phases_neg[k]
                                    d2_min = np.float32(1e38)
                                    for m in range(M):
                                        dv = y_rot - constellation[m]
                                        d2 = dv.real * dv.real + dv.imag * dv.imag
                                        if d2 < d2_min:
                                            d2_min = d2
                                    metric_k = metric_k + d2_min
                                if metric_k < min_tot:
                                    min_tot = metric_k
                                    best_k = k
                            phi_hat_bps[i] = bps_angles[best_k]

                for i in range(C):
                    if cpr_mode == 1:
                        phi_hat = pll_phi[i]
                    else:
                        phi_hat = phi_hat_bps[i]

                    # ── Cycle-slip correction ──────────────────────────────
                    if cs_enabled:
                        n_b = cs_buf_n[i]
                        ptr = cs_buf_ptr[i]
                        x_b = np.float64(idx)
                        y_b = np.float64(phi_hat)

                        if n_b == 0:
                            phi_corr = phi_hat
                        else:
                            if n_b < np.int64(10):
                                last_pos = (ptr - np.int64(1) + np.int64(H)) % np.int64(
                                    H
                                )
                                phi_expected = cs_buf_y[i, last_pos]
                            else:
                                Sx = cs_stats[i, 0]
                                Sy = cs_stats[i, 1]
                                Sxx = cs_stats[i, 2]
                                Sxy = cs_stats[i, 3]
                                n_f = np.float64(n_b)
                                denom = n_f * Sxx - Sx * Sx
                                if np.abs(denom) > np.float64(1e-30):
                                    slope = (n_f * Sxy - Sx * Sy) / denom
                                    intercept = (Sy - slope * Sx) / n_f
                                else:
                                    slope = np.float64(0.0)
                                    intercept = Sy / n_f
                                phi_expected = slope * x_b + intercept

                            diff = y_b - phi_expected
                            k_slip = np.int64(np.round(diff / quantum))
                            if np.abs(diff) > np.float64(
                                cs_threshold
                            ) and k_slip != np.int64(0):
                                y_b = y_b - np.float64(k_slip) * quantum
                            phi_corr = np.float32(y_b)

                        write_pos = ptr % np.int64(H)
                        if n_b == np.int64(H):
                            old_x = cs_buf_x[i, write_pos]
                            old_y = cs_buf_y[i, write_pos]
                            cs_stats[i, 0] -= old_x
                            cs_stats[i, 1] -= old_y
                            cs_stats[i, 2] -= old_x * old_x
                            cs_stats[i, 3] -= old_x * old_y
                        new_y = np.float64(phi_corr)
                        cs_buf_x[i, write_pos] = x_b
                        cs_buf_y[i, write_pos] = new_y
                        cs_stats[i, 0] += x_b
                        cs_stats[i, 1] += new_y
                        cs_stats[i, 2] += x_b * x_b
                        cs_stats[i, 3] += x_b * new_y
                        cs_buf_ptr[i] = ptr + np.int64(1)
                        if n_b < np.int64(H):
                            cs_buf_n[i] = n_b + np.int64(1)
                    else:
                        phi_corr = phi_hat

                    phi_c[i] = phi_corr
                    phase_out[idx, i] = phi_corr

                    cos_p = np.cos(np.float64(phi_corr))
                    sin_p = np.sin(np.float64(phi_corr))
                    yr = y_raw[i].real * np.float32(cos_p) + y_raw[i].imag * np.float32(
                        sin_p
                    )
                    yi = -y_raw[i].real * np.float32(sin_p) + y_raw[
                        i
                    ].imag * np.float32(cos_p)
                    y_fin[i] = np.complex64(yr + yi * np.complex64(1j))
                    y_out[idx, i] = y_fin[i]

                # ── Slicer and error ───────────────────────────────────────
                for i in range(C):
                    if idx < n_train:
                        d_i = training[i, idx]
                    else:
                        min_dist = np.float32(1e38)
                        min_idx = 0
                        for kk in range(M):
                            dv = y_fin[i] - constellation[kk]
                            dist = dv.real * dv.real + dv.imag * dv.imag
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = kk
                        d_i = constellation[min_idx]
                    d_sym[i] = d_i
                    e_clean[i] = d_i - y_fin[i]
                    e_out[idx, i] = e_clean[i]

                # ── PLL state update ───────────────────────────────────────
                if cpr_mode == 1:
                    for i in range(C):
                        e_ph = (
                            y_fin[i].imag * d_sym[i].real
                            - y_fin[i].real * d_sym[i].imag
                        )
                        pll_phi[i] = (
                            pll_phi[i] + pll_mu * np.float32(e_ph) + pll_freq[i]
                        )
                        pll_freq[i] = pll_freq[i] + pll_beta * np.float32(e_ph)

                # ── De-rotate error ────────────────────────────────────────
                for i in range(C):
                    cos_p = np.cos(np.float64(phi_c[i]))
                    sin_p = np.sin(np.float64(phi_c[i]))
                    er = e_clean[i].real * np.float32(cos_p) - e_clean[
                        i
                    ].imag * np.float32(sin_p)
                    ei = e_clean[i].real * np.float32(sin_p) + e_clean[
                        i
                    ].imag * np.float32(cos_p)
                    e_eq[i] = np.complex64(er + ei * np.complex64(1j))

                # ── Kalman gain ────────────────────────────────────────────
                Px = np.dot(P, x_bar)
                denom_k = lam_f32
                for jj in range(N):
                    denom_k = denom_k + (np.conj(x_bar[jj]) * Px[jj]).real
                inv_denom = np.float32(1.0) / denom_k
                for ii in range(N):
                    k_gain[ii] = Px[ii] * inv_denom

                if idx < n_update_halt:
                    for i in range(C):
                        ce_i = np.conj(e_eq[i])
                        for j in range(C):
                            for t in range(num_taps):
                                W[i, j, t] = (
                                    leak_term * W[i, j, t]
                                    + k_gain[j * num_taps + t] * ce_i
                                )
                    for jj in range(N):
                        xH_P[jj] = np.conj(Px[jj])
                    for ii in range(N):
                        for jj in range(N):
                            P[ii, jj] = (P[ii, jj] - k_gain[ii] * xH_P[jj]) / lam_f32
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

        _NUMBA_KERNELS["rls_cpr"] = rls_cpr_loop
    return _NUMBA_KERNELS["rls_cpr"]


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


def _get_numba_rde():
    """JIT-compile and cache the Numba RDE butterfly loop kernel.

    RDE (Radius Directed Equalizer) is a CMA variant that selects a
    per-symbol target radius from the set of unique constellation radii
    rather than using a single fixed Godard radius.  This makes it
    converge correctly on multi-ring constellations (16-QAM, 64-QAM).

    Returns
    -------
    rde_loop : numba-compiled callable
        See kernel source for argument shapes and semantics.
    """
    if "rde" not in _NUMBA_KERNELS:
        numba_mod = _get_numba()
        if numba_mod is None:
            raise ImportError("Numba is required for backend='numba'.")

        @numba_mod.njit(cache=True, fastmath=True, nogil=True)
        def rde_loop(
            x_padded,
            W,
            step_size,
            radii,
            stride,
            store_weights,
            y_out,
            e_out,
            w_hist_out,
        ):
            # x_padded    : (C, N_pad)            complex64
            # W           : (C, C, num_taps)       complex64 — modified in-place
            # step_size   : float32               — fixed gradient step μ
            # radii       : (K,)                  float32   — unique |c| radii, sorted ascending
            # stride      : int                   — sps
            # store_weights : bool
            # y_out       : (N_sym, C)             complex64
            # e_out       : (N_sym, C)             complex64
            # w_hist_out  : (N_sym or 1, C, C, T)  complex64
            C = W.shape[0]
            num_taps = W.shape[2]
            n_sym = y_out.shape[0]
            K = radii.shape[0]

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

                # RDE error: e[i] = y[i] * (|y[i]|² − R_d²)
                # R_d is the magnitude of the nearest constellation ring.
                # Linear scan over K unique radii is cheap (K ≤ 8 for typical QAM).
                for i in range(C):
                    mod2 = y[i].real * y[i].real + y[i].imag * y[i].imag
                    abs_y = mod2 ** np.float32(0.5)

                    best_rd = radii[0]
                    best_dist = abs(abs_y - radii[0])
                    for k in range(1, K):
                        dist_k = abs(abs_y - radii[k])
                        if dist_k < best_dist:
                            best_dist = dist_k
                            best_rd = radii[k]

                    e[i] = y[i] * np.float32(mod2 - best_rd * best_rd)
                    e_out[idx, i] = e[i]

                # Weight update: W -= mu * conj(e[i]) * X_wins[j,t]
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

        _NUMBA_KERNELS["rde"] = rde_loop
    return _NUMBA_KERNELS["rde"]


# -----------------------------------------------------------------------------
# JAX KERNELS — ADAPTIVE EQUALIZER SCANS
# -----------------------------------------------------------------------------
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
    """JIT-compile and cache the sample-by-sample LMS butterfly scan.

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
            # step_size       : scalar float32     — LMS mu
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

                W_new = W + step_size * jnp.einsum("i,jt->ijt", jnp.conj(e), X_wins)
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
                # Riccati: exploit Hermitian symmetry P = P^H so (x^H P)[j] = conj((Px)[j]).
                # outer(k, conj(Px)) == k ⊗ (x^H P); reuses Px to avoid a second O(N²) mat-vec.
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


def _get_jax_lms_cpr(
    num_taps,
    stride,
    const_size,
    num_ch,
    cpr_type,
    bps_n,
    bps_block_size,
    bps_joint_channels,
    cs_history_len,
):
    """JIT-compile and cache the LMS+CPR butterfly scan.

    All CPR parameters are static closure variables (baked into the XLA graph
    at trace time).  A separate cache entry is created for each distinct
    combination of these parameters.

    cpr_type : "pll" or "bps"
    bps_n    : number of BPS test phases (ignored for cpr_type="pll")
    cs_history_len : int — circular buffer depth for cycle-slip correction
    """
    key = (
        "lms_cpr",
        num_taps,
        stride,
        const_size,
        num_ch,
        cpr_type,
        bps_n,
        bps_block_size,
        bps_joint_channels,
        cs_history_len,
    )
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()

        H = cs_history_len
        KB = bps_block_size  # static closure: BPS window length

        import math as _math  # noqa: PLC0415

        _quantum_static = jnp.float32(_math.pi / 2.0)  # symmetry=4 default

        @jax.jit
        def lms_cpr_scan(
            x_input,
            training_padded,
            constellation,
            bps_phases_neg,
            bps_angles,
            w_init,
            step_size,
            n_train,
            pll_mu,
            pll_beta,
            cs_threshold,
            cs_enabled,
        ):
            # x_input         : (C, N_pad)       complex64
            # training_padded : (C, N_sym)        complex64
            # constellation   : (M,)              complex64
            # bps_phases_neg  : (B,)              complex64  exp(-j*theta_k)
            # bps_angles      : (B,)              float32    theta_k
            # w_init          : (C, C, T)         complex64
            # step_size       : scalar float32
            # n_train         : scalar int32
            # pll_mu, pll_beta: scalar float32
            # cs_threshold    : scalar float32
            # cs_enabled      : scalar bool
            #
            # lax.scan carry:
            #   W             : (C, C, T)         complex64
            #   pll_phi       : (C,)              float32
            #   pll_freq      : (C,)              float32
            #   bps_buf       : (C, KB)           complex64  — y_raw circular buffer
            #   bps_buf_ptr   : scalar int32
            #   cs_buf_x      : (C, H)            float32  — symbol index
            #   cs_buf_y      : (C, H)            float32  — phase value
            #   cs_buf_ptr    : (C,)              int32    — write pointer

            def step(carry, idx):
                (
                    W,
                    pll_phi,
                    pll_freq,
                    bps_buf,
                    bps_buf_ptr,
                    cs_buf_x,
                    cs_buf_y,
                    cs_buf_ptr,
                ) = carry
                sample_idx = idx * stride

                X_wins = jax.lax.dynamic_slice(
                    x_input, (0, sample_idx), (num_ch, num_taps)
                )  # (C, T)

                y_raw = jnp.einsum("ijt,jt->i", jnp.conj(W), X_wins)  # (C,)

                # ── Phase estimation (static branch at trace time) ──
                if cpr_type == "pll":
                    phi_hat = pll_phi  # (C,)
                    bps_buf_new = bps_buf
                    bps_buf_ptr_new = bps_buf_ptr
                else:
                    # Fill BPS circular buffer with current y_raw
                    slot = bps_buf_ptr % KB
                    bps_buf_new = jax.lax.dynamic_update_slice(
                        bps_buf,
                        y_raw[None, :],  # (1, C) — update one row
                        (slot, 0),
                    )  # broadcast doesn't work for transpose, use (C, KB) layout instead
                    bps_buf_ptr_new = bps_buf_ptr + 1
                    fill = jnp.minimum(bps_buf_ptr_new, KB)

                    # rotated: (B, KB, C)
                    rotated = (
                        bps_phases_neg[:, None, None] * bps_buf_new[None, :, :]
                    )  # (B, KB, C)
                    # min-dist per candidate per slot per channel: (B, KB, C)
                    d2_all = jnp.min(
                        jnp.abs(
                            rotated[:, :, :, None] - constellation[None, None, None, :]
                        )
                        ** 2,
                        axis=-1,
                    )
                    # Mask slots beyond fill
                    slot_mask = jnp.arange(KB)[None, :, None] < fill  # (1, KB, 1)
                    d2_masked = jnp.where(slot_mask, d2_all, 0.0)
                    # Sum over buffer slots: (B, C)
                    metric = d2_masked.sum(axis=1)

                    if bps_joint_channels:
                        # Sum over channels too → (B,); broadcast winner to all C
                        best_k = jnp.argmin(metric.sum(axis=-1))  # scalar
                        phi_hat = jnp.full(num_ch, bps_angles[best_k])  # (C,)
                    else:
                        best_k = jnp.argmin(metric, axis=0)  # (C,)
                        phi_hat = bps_angles[best_k]  # (C,)

                # ── Cycle-slip correction ────────────────────────────
                def correct_slip_ch(phi_h, buf_x_ch, buf_y_ch, ptr_ch):
                    fill_cs = jnp.minimum(ptr_ch, H)
                    x_b = idx.astype(jnp.float32)
                    y_b = phi_h

                    mask = jnp.arange(H) < fill_cs
                    n_f = fill_cs.astype(jnp.float32)
                    Sx = jnp.where(mask, buf_x_ch, 0.0).sum()
                    Sy = jnp.where(mask, buf_y_ch, 0.0).sum()
                    Sxx = jnp.where(mask, buf_x_ch * buf_x_ch, 0.0).sum()
                    Sxy = jnp.where(mask, buf_x_ch * buf_y_ch, 0.0).sum()

                    denom = n_f * Sxx - Sx * Sx
                    safe_denom = jnp.where(
                        jnp.abs(denom) > 1e-20, denom, jnp.float32(1.0)
                    )
                    slope = jnp.where(
                        fill_cs >= 10,
                        (n_f * Sxy - Sx * Sy) / safe_denom,
                        jnp.float32(0.0),
                    )
                    intercept = jnp.where(
                        fill_cs >= 10,
                        (Sy - slope * Sx) / jnp.maximum(n_f, jnp.float32(1.0)),
                        Sy / jnp.maximum(n_f, jnp.float32(1.0)),
                    )
                    phi_exp_lin = slope * x_b + intercept

                    last_pos = (ptr_ch - 1 + H) % H
                    phi_last = jax.lax.dynamic_index_in_dim(
                        buf_y_ch, last_pos, keepdims=False
                    )
                    phi_expected = jnp.where(fill_cs >= 10, phi_exp_lin, phi_last)
                    phi_expected = jnp.where(fill_cs == 0, y_b, phi_expected)

                    diff = y_b - phi_expected
                    k_slip = jnp.round(diff / _quantum_static)
                    should_correct = (
                        cs_enabled & (jnp.abs(diff) > cs_threshold) & (k_slip != 0)
                    )
                    phi_corr = jnp.where(
                        should_correct, y_b - k_slip * _quantum_static, y_b
                    )

                    write_pos = ptr_ch % H
                    buf_x_new = jax.lax.dynamic_update_slice(
                        buf_x_ch, x_b[None], [write_pos]
                    )
                    buf_y_new = jax.lax.dynamic_update_slice(
                        buf_y_ch, phi_corr[None], [write_pos]
                    )
                    ptr_new = ptr_ch + 1
                    return phi_corr, buf_x_new, buf_y_new, ptr_new

                phi_corr, cs_buf_x_new, cs_buf_y_new, cs_buf_ptr_new = jax.vmap(
                    correct_slip_ch
                )(phi_hat, cs_buf_x, cs_buf_y, cs_buf_ptr)

                phasor = jnp.exp(-1j * phi_corr.astype(jnp.float32))
                y_fin = y_raw * phasor  # (C,)

                def slicer(ch_y):
                    return constellation[jnp.argmin(jnp.abs(ch_y - constellation) ** 2)]

                dd = jax.vmap(slicer)(y_fin)
                d = jnp.where(idx < n_train, training_padded[:, idx], dd)
                e_clean = d - y_fin  # (C,)

                if cpr_type == "pll":
                    e_ph = y_fin.imag * d.real - y_fin.real * d.imag  # (C,)
                    pll_phi_new = pll_phi + pll_mu * e_ph + pll_freq
                    pll_freq_new = pll_freq + pll_beta * e_ph
                else:
                    pll_phi_new = pll_phi
                    pll_freq_new = pll_freq

                phasor_inv = jnp.exp(1j * phi_corr.astype(jnp.float32))
                e_eq = e_clean * phasor_inv  # (C,)

                W_new = W + step_size * jnp.einsum("i,jt->ijt", jnp.conj(e_eq), X_wins)

                carry_new = (
                    W_new,
                    pll_phi_new,
                    pll_freq_new,
                    bps_buf_new,
                    bps_buf_ptr_new,
                    cs_buf_x_new,
                    cs_buf_y_new,
                    cs_buf_ptr_new,
                )
                return carry_new, (y_fin, e_clean, W_new, phi_corr)

            n_sym = training_padded.shape[1]
            init_carry = (
                w_init,
                jnp.zeros(num_ch, dtype=jnp.float32),  # pll_phi
                jnp.zeros(num_ch, dtype=jnp.float32),  # pll_freq
                jnp.zeros((KB, num_ch), dtype=jnp.complex64),  # bps_buf (KB, C)
                jnp.int32(0),  # bps_buf_ptr
                jnp.zeros((num_ch, H), dtype=jnp.float32),  # cs_buf_x
                jnp.zeros((num_ch, H), dtype=jnp.float32),  # cs_buf_y
                jnp.zeros(num_ch, dtype=jnp.int32),  # cs_buf_ptr
            )
            (W_final, _, _, _, _, _, _, _), (y_hat, errors, w_hist, phi_traj) = (
                jax.lax.scan(step, init_carry, jnp.arange(n_sym))
            )
            return y_hat, errors, W_final, w_hist, phi_traj

        _JITTED_EQ[key] = lms_cpr_scan
    return _JITTED_EQ[key]


def _get_jax_rls_cpr(
    num_taps,
    stride,
    const_size,
    num_ch,
    cpr_type,
    bps_n,
    bps_block_size,
    bps_joint_channels,
    cs_history_len,
):
    """JIT-compile and cache the RLS+CPR butterfly scan.

    Combines the Leaky-RLS Riccati update with an inline CPR tracker.
    Static parameters are identical to ``_get_jax_lms_cpr``.
    """
    key = (
        "rls_cpr",
        num_taps,
        stride,
        const_size,
        num_ch,
        cpr_type,
        bps_n,
        bps_block_size,
        bps_joint_channels,
        cs_history_len,
    )
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()

        H = cs_history_len
        KB = bps_block_size
        import math as _math  # noqa: PLC0415

        _quantum_static = jnp.float32(_math.pi / 2.0)

        @jax.jit
        def rls_cpr_scan(
            x_input,
            training_padded,
            constellation,
            bps_phases_neg,
            bps_angles,
            w_init,
            P_init,
            lam,
            n_train,
            leakage,
            n_update_halt,
            pll_mu,
            pll_beta,
            cs_threshold,
            cs_enabled,
        ):
            def step(carry, idx):
                (
                    W,
                    P,
                    pll_phi,
                    pll_freq,
                    bps_buf,
                    bps_buf_ptr,
                    cs_buf_x,
                    cs_buf_y,
                    cs_buf_ptr,
                ) = carry
                sample_idx = idx * stride

                X_wins = jax.lax.dynamic_slice(
                    x_input, (0, sample_idx), (num_ch, num_taps)
                )
                y_raw = jnp.einsum("ijt,jt->i", jnp.conj(W), X_wins)

                if cpr_type == "pll":
                    phi_hat = pll_phi
                    bps_buf_new = bps_buf
                    bps_buf_ptr_new = bps_buf_ptr
                else:
                    slot = bps_buf_ptr % KB
                    bps_buf_new = jax.lax.dynamic_update_slice(
                        bps_buf,
                        y_raw[None, :],
                        (slot, 0),
                    )
                    bps_buf_ptr_new = bps_buf_ptr + 1
                    fill = jnp.minimum(bps_buf_ptr_new, KB)

                    rotated = (
                        bps_phases_neg[:, None, None] * bps_buf_new[None, :, :]
                    )  # (B, KB, C)
                    d2_all = jnp.min(
                        jnp.abs(
                            rotated[:, :, :, None] - constellation[None, None, None, :]
                        )
                        ** 2,
                        axis=-1,
                    )
                    slot_mask = jnp.arange(KB)[None, :, None] < fill
                    metric = jnp.where(slot_mask, d2_all, 0.0).sum(axis=1)  # (B, C)

                    if bps_joint_channels:
                        best_k = jnp.argmin(metric.sum(axis=-1))
                        phi_hat = jnp.full(num_ch, bps_angles[best_k])
                    else:
                        best_k = jnp.argmin(metric, axis=0)
                        phi_hat = bps_angles[best_k]

                def correct_slip_ch(phi_h, buf_x_ch, buf_y_ch, ptr_ch):
                    fill_cs = jnp.minimum(ptr_ch, H)
                    x_b = idx.astype(jnp.float32)
                    y_b = phi_h
                    mask = jnp.arange(H) < fill_cs
                    n_f = fill_cs.astype(jnp.float32)
                    Sx = jnp.where(mask, buf_x_ch, 0.0).sum()
                    Sy = jnp.where(mask, buf_y_ch, 0.0).sum()
                    Sxx = jnp.where(mask, buf_x_ch * buf_x_ch, 0.0).sum()
                    Sxy = jnp.where(mask, buf_x_ch * buf_y_ch, 0.0).sum()
                    denom = n_f * Sxx - Sx * Sx
                    safe_denom = jnp.where(
                        jnp.abs(denom) > 1e-20, denom, jnp.float32(1.0)
                    )
                    slope = jnp.where(
                        fill_cs >= 10,
                        (n_f * Sxy - Sx * Sy) / safe_denom,
                        jnp.float32(0.0),
                    )
                    intercept = jnp.where(
                        fill_cs >= 10,
                        (Sy - slope * Sx) / jnp.maximum(n_f, jnp.float32(1.0)),
                        Sy / jnp.maximum(n_f, jnp.float32(1.0)),
                    )
                    phi_exp_lin = slope * x_b + intercept
                    last_pos = (ptr_ch - 1 + H) % H
                    phi_last = jax.lax.dynamic_index_in_dim(
                        buf_y_ch, last_pos, keepdims=False
                    )
                    phi_expected = jnp.where(fill_cs >= 10, phi_exp_lin, phi_last)
                    phi_expected = jnp.where(fill_cs == 0, y_b, phi_expected)
                    diff = y_b - phi_expected
                    k_slip = jnp.round(diff / _quantum_static)
                    should_correct = (
                        cs_enabled & (jnp.abs(diff) > cs_threshold) & (k_slip != 0)
                    )
                    phi_corr = jnp.where(
                        should_correct, y_b - k_slip * _quantum_static, y_b
                    )
                    write_pos = ptr_ch % H
                    buf_x_new = jax.lax.dynamic_update_slice(
                        buf_x_ch, x_b[None], [write_pos]
                    )
                    buf_y_new = jax.lax.dynamic_update_slice(
                        buf_y_ch, phi_corr[None], [write_pos]
                    )
                    ptr_new = ptr_ch + 1
                    return phi_corr, buf_x_new, buf_y_new, ptr_new

                phi_corr, cs_buf_x_new, cs_buf_y_new, cs_buf_ptr_new = jax.vmap(
                    correct_slip_ch
                )(phi_hat, cs_buf_x, cs_buf_y, cs_buf_ptr)

                phasor = jnp.exp(-1j * phi_corr.astype(jnp.float32))
                y_fin = y_raw * phasor

                def slicer(ch_y):
                    return constellation[jnp.argmin(jnp.abs(ch_y - constellation) ** 2)]

                dd = jax.vmap(slicer)(y_fin)
                d = jnp.where(idx < n_train, training_padded[:, idx], dd)
                e_clean = d - y_fin

                if cpr_type == "pll":
                    e_ph = y_fin.imag * d.real - y_fin.real * d.imag
                    pll_phi_new = pll_phi + pll_mu * e_ph + pll_freq
                    pll_freq_new = pll_freq + pll_beta * e_ph
                else:
                    pll_phi_new = pll_phi
                    pll_freq_new = pll_freq

                phasor_inv = jnp.exp(1j * phi_corr.astype(jnp.float32))
                e_eq = e_clean * phasor_inv

                x_bar = X_wins.flatten()
                Px = P @ x_bar
                denom_k = lam + jnp.real(jnp.dot(jnp.conj(x_bar), Px))
                k_gain = Px / denom_k

                def w_update(w_row, err_val):
                    w_flat = w_row.flatten()
                    w_flat_new = (1.0 - leakage) * w_flat + k_gain * jnp.conj(err_val)
                    return w_flat_new.reshape(num_ch, num_taps)

                W_upd = jax.vmap(w_update)(W, e_eq)
                P_upd = (P - jnp.outer(k_gain, jnp.conj(Px))) / lam
                P_upd = jnp.float32(0.5) * (P_upd + jnp.conj(P_upd).T)

                update_ok = idx < n_update_halt
                W_new = jnp.where(update_ok, W_upd, W)
                P_new = jnp.where(update_ok, P_upd, P)

                carry_new = (
                    W_new,
                    P_new,
                    pll_phi_new,
                    pll_freq_new,
                    bps_buf_new,
                    bps_buf_ptr_new,
                    cs_buf_x_new,
                    cs_buf_y_new,
                    cs_buf_ptr_new,
                )
                return carry_new, (y_fin, e_clean, W_new, phi_corr)

            n_sym = training_padded.shape[1]
            init_carry = (
                w_init,
                P_init,
                jnp.zeros(num_ch, dtype=jnp.float32),
                jnp.zeros(num_ch, dtype=jnp.float32),
                jnp.zeros((KB, num_ch), dtype=jnp.complex64),  # bps_buf
                jnp.int32(0),  # bps_buf_ptr
                jnp.zeros((num_ch, H), dtype=jnp.float32),
                jnp.zeros((num_ch, H), dtype=jnp.float32),
                jnp.zeros(num_ch, dtype=jnp.int32),
            )
            (W_final, _, _, _, _, _, _, _, _), (y_hat, errors, w_hist, phi_traj) = (
                jax.lax.scan(step, init_carry, jnp.arange(n_sym))
            )
            return y_hat, errors, W_final, w_hist, phi_traj

        _JITTED_EQ[key] = rls_cpr_scan
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


def _get_jax_rde(num_taps, stride, num_radii, num_ch):
    """JIT-compile and cache the sample-by-sample RDE butterfly scan.

    RDE (Radius Directed Equalizer) extends CMA by replacing the single
    Godard radius with per-symbol radius selection from a precomputed set
    of unique constellation magnitudes.  This provides correct blind
    convergence on multi-ring constellations such as 16-QAM and 64-QAM.

    Static closure variables (baked into the compiled kernel):

    num_taps   : FIR filter length per polyphase arm.
    stride     : decimation factor (sps, typically 2 for T/2-spaced input).
    num_radii  : number of unique radii K — fixes the argmin shape at trace
                 time so XLA can compile without dynamic dispatch.
    num_ch     : MIMO butterfly width C.

    Returns
    -------
    rde_scan : JIT-compiled callable
        See the inner function for the call signature.
    """
    key = ("rde", num_taps, stride, num_radii, num_ch)
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()

        @functools.partial(jax.jit, static_argnums=(4,))
        def rde_scan(x_input, w_init, step_size, radii, n_sym):
            # Argument shapes and semantics
            # ------------------------------
            # x_input   : (C, N_pad)       complex64 — padded received samples
            # w_init    : (C, C, num_taps) complex64 — initial butterfly weights
            # step_size : scalar float32   — fixed gradient step μ
            # radii     : (K,)             float32   — unique constellation radii, sorted
            # n_sym     : int (static)     — total symbol count; fixes scan iteration count
            #
            # lax.scan carry  : W  (C, C, num_taps)
            # lax.scan xs     : jnp.arange(n_sym)
            # lax.scan output : y_hat   (n_sym, C)
            #                   errors  (n_sym, C)   RDE errors y*(|y|²−R_d²)
            #                   w_hist  (n_sym, C, C, num_taps)
            #
            # Per-step RDE gradient:
            #   y     = einsum('ijt,jt->i', conj(W), X_wins)    (C,)
            #   abs_y = sqrt(real(y*conj(y)))                   (C,)
            #   R_d   = radii[argmin(|radii−abs_y|)]            (C,)  nearest ring
            #   e     = y * (real(y*conj(y)) − R_d²)            (C,)
            #   W    -= μ * einsum('i,jt->ijt', conj(e), X_wins)
            def step(W, idx):
                sample_idx = idx * stride

                X_wins = jax.lax.dynamic_slice(
                    x_input, (0, sample_idx), (num_ch, num_taps)
                )
                y = jnp.einsum("ijt,jt->i", jnp.conj(W), X_wins)

                abs_y2 = jnp.real(y * jnp.conj(y))  # (C,) strict real |y|²
                abs_y = jnp.sqrt(abs_y2)  # (C,) |y|

                # (C, K) distance table; argmin over K gives nearest radius index
                dist = jnp.abs(abs_y[:, None] - radii[None, :])  # (C, K)
                rd = radii[jnp.argmin(dist, axis=1)]  # (C,)

                e = y * (abs_y2 - rd**2)

                W_new = W - step_size * jnp.einsum("i,jt->ijt", jnp.conj(e), X_wins)
                return W_new, (y, e, W_new)

            W_final, (y_hat, errors, w_hist) = jax.lax.scan(
                step, w_init, jnp.arange(n_sym)
            )
            return y_hat, errors, W_final, w_hist

        _JITTED_EQ[key] = rde_scan
    return _JITTED_EQ[key]


# -----------------------------------------------------------------------------
# HYBRID DA / BLIND KERNELS  (internal — used by cma/rde when pilot_ref supplied)
# -----------------------------------------------------------------------------
#
# These kernels process every symbol with a per-step error-function switch:
#
#   pilot_mask[idx] == 1  →  DA-LMS error:   e[i] = pilot_ref[i, idx] - y[i]
#   pilot_mask[idx] == 0  →  blind error:    e[i] = CMA or RDE formula
#
# The weight update rule is identical to the blind case.  This achieves
# phase-resolved convergence at pilot/preamble positions while preserving
# correct blind tracking at data positions — all in a single kernel pass.


def _get_numba_pa_cma():
    """JIT-compile and cache the Numba pilot-aided CMA butterfly loop kernel.

    Hybrid CMA: LMS error at pilot positions (pilot_mask==1), standard
    Godard CMA error at data positions (pilot_mask==0).  "Pilot-Aided" (PA)
    is the standard telecom term for this mixed DA/blind adaptation strategy.

    Returns
    -------
    pa_cma_loop : numba-compiled callable
    """
    if "pa_cma" not in _NUMBA_KERNELS:
        numba_mod = _get_numba()
        if numba_mod is None:
            raise ImportError("Numba is required for backend='numba'.")

        @numba_mod.njit(cache=True, fastmath=True, nogil=True)
        def pa_cma_loop(
            x_padded,
            W,
            step_size,
            r2,
            stride,
            store_weights,
            y_out,
            e_out,
            w_hist_out,
            pilot_ref,
            pilot_mask,
        ):
            # x_padded   : (C, N_pad)          complex64
            # W          : (C, C, num_taps)     complex64 — modified in-place
            # step_size  : float32
            # r2         : float32             — Godard radius R²
            # stride     : int
            # store_weights : bool
            # y_out      : (N_sym, C)           complex64
            # e_out      : (N_sym, C)           complex64
            # w_hist_out : (N_sym or 1, C, C, T) complex64
            # pilot_ref  : (C, N_sym)           complex64 — known symbol; 0 at data positions
            # pilot_mask : (N_sym,)             uint8     — 1 at pilot/preamble, 0 at data
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

                # Error: DA-LMS at pilots, CMA Godard at data
                for i in range(C):
                    if pilot_mask[idx]:
                        e[i] = (
                            y[i] - pilot_ref[i, idx]
                        )  # inverted to match blind subtractive update
                    else:
                        mod2 = y[i].real * y[i].real + y[i].imag * y[i].imag
                        e[i] = y[i] * np.float32(mod2 - r2)
                    e_out[idx, i] = e[i]

                # Weight update
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

        _NUMBA_KERNELS["pa_cma"] = pa_cma_loop
    return _NUMBA_KERNELS["pa_cma"]


def _get_numba_pa_rde():
    """JIT-compile and cache the Numba pilot-aided RDE butterfly loop kernel.

    Hybrid RDE: LMS error at pilot positions (pilot_mask==1), standard
    ring-directed RDE error at data positions (pilot_mask==0).  "Pilot-Aided"
    (PA) is the standard telecom term for this mixed DA/blind adaptation strategy.

    Returns
    -------
    pa_rde_loop : numba-compiled callable
    """
    if "pa_rde" not in _NUMBA_KERNELS:
        numba_mod = _get_numba()
        if numba_mod is None:
            raise ImportError("Numba is required for backend='numba'.")

        @numba_mod.njit(cache=True, fastmath=True, nogil=True)
        def pa_rde_loop(
            x_padded,
            W,
            step_size,
            radii,
            stride,
            store_weights,
            y_out,
            e_out,
            w_hist_out,
            pilot_ref,
            pilot_mask,
        ):
            # x_padded   : (C, N_pad)            complex64
            # W          : (C, C, num_taps)       complex64 — modified in-place
            # step_size  : float32
            # radii      : (K,)                  float32   — unique |c| radii, sorted
            # stride     : int
            # store_weights : bool
            # y_out      : (N_sym, C)             complex64
            # e_out      : (N_sym, C)             complex64
            # w_hist_out : (N_sym or 1, C, C, T)  complex64
            # pilot_ref  : (C, N_sym)             complex64 — 0 at data positions
            # pilot_mask : (N_sym,)               uint8     — 1 at pilot/preamble
            C = W.shape[0]
            num_taps = W.shape[2]
            n_sym = y_out.shape[0]
            K = radii.shape[0]

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

                # Error: DA-LMS at pilots, RDE ring-directed at data
                for i in range(C):
                    if pilot_mask[idx]:
                        e[i] = (
                            y[i] - pilot_ref[i, idx]
                        )  # inverted to match blind subtractive update
                    else:
                        mod2 = y[i].real * y[i].real + y[i].imag * y[i].imag
                        abs_y = mod2 ** np.float32(0.5)
                        best_rd = radii[0]
                        best_dist = abs(abs_y - radii[0])
                        for k in range(1, K):
                            dist_k = abs(abs_y - radii[k])
                            if dist_k < best_dist:
                                best_dist = dist_k
                                best_rd = radii[k]
                        e[i] = y[i] * np.float32(mod2 - best_rd * best_rd)
                    e_out[idx, i] = e[i]

                # Weight update
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

        _NUMBA_KERNELS["pa_rde"] = pa_rde_loop
    return _NUMBA_KERNELS["pa_rde"]


def _get_jax_pa_cma(num_taps: int, stride: int, num_ch: int):
    """JIT-compile and cache the JAX pilot-aided CMA butterfly scan.

    Hybrid CMA scan using ``jax.lax.scan``.  At pilot positions
    (``pilot_mask[t] == True``) the error is the standard LMS residual
    ``pilot_ref[t] - y``; at data positions the Godard CMA error
    ``y * (|y|² - R²)`` is used.  The switch is XLA-branchless via
    ``jnp.where``, keeping the scan body shape-static for efficient
    compilation.

    Parameters
    ----------
    num_taps : int
    stride   : int — samples per symbol (sps)
    num_ch   : int — number of MIMO channels C

    Returns
    -------
    pa_cma_scan : jax.jit-compiled callable
        ``(x_input, w_init, step_size, r2, pilot_ref, pilot_mask, n_sym)
        → (y_all, e_all, W_final, w_hist)``
        where ``pilot_ref`` has shape ``(n_sym, C)`` and ``pilot_mask``
        has shape ``(n_sym,)`` bool.
    """
    key = ("pa_cma", num_taps, stride, num_ch)
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()

        @functools.partial(jax.jit, static_argnums=(6,))
        def pa_cma_scan(
            x_input,  # (C, N_pad)  complex64 — padded received samples
            w_init,  # (C, C, T)   complex64 — initial butterfly weights
            step_size,  # ()          float32
            r2,  # ()          float32 — Godard R² = E[|s|⁴]/E[|s|²]
            pilot_ref,  # (n_sym, C)  complex64 — known symbols; 0 at data
            pilot_mask,  # (n_sym,)    bool — True at pilot/preamble positions
            n_sym,  # int (static) — total symbol count
        ):
            def step(W, xs_t):
                idx, p_ref, p_mask = xs_t  # (): int, (C,): cplx, (): bool
                X_wins = jax.lax.dynamic_slice(
                    x_input, (0, idx * stride), (num_ch, num_taps)
                )  # (C, T)
                y = jnp.einsum("ijt,jt->i", jnp.conj(W), X_wins)  # (C,)

                abs_y2 = jnp.real(y * jnp.conj(y))  # strict real |y|²
                e_blind = y * (abs_y2 - r2)  # Godard CMA
                e_da = (
                    y - p_ref
                )  # pilot LMS (inverted to match blind subtractive update)
                e = jnp.where(p_mask, e_da, e_blind)  # (C,) branchless

                W_new = W - step_size * jnp.einsum("i,jt->ijt", jnp.conj(e), X_wins)
                return W_new, (y, e, W_new)

            xs = (jnp.arange(n_sym), pilot_ref, pilot_mask)
            W_final, (y_all, e_all, wh_all) = jax.lax.scan(step, w_init, xs)
            return y_all, e_all, W_final, wh_all

        _JITTED_EQ[key] = pa_cma_scan
    return _JITTED_EQ[key]


def _get_jax_pa_rde(num_taps: int, stride: int, num_radii: int, num_ch: int):
    """JIT-compile and cache the JAX pilot-aided RDE butterfly scan.

    Hybrid RDE scan using ``jax.lax.scan``.  At pilot positions the error
    is the LMS residual ``pilot_ref[t] - y``; at data positions the
    ring-directed RDE error ``y * (|y|² - R_d²)`` is used, where ``R_d``
    is the nearest constellation ring radius.  The switch is branchless.

    Parameters
    ----------
    num_taps  : int
    stride    : int — samples per symbol (sps)
    num_radii : int — number of unique constellation ring radii K
    num_ch    : int — number of MIMO channels C

    Returns
    -------
    pa_rde_scan : jax.jit-compiled callable
        ``(x_input, w_init, step_size, radii, pilot_ref, pilot_mask, n_sym)
        → (y_all, e_all, W_final, w_hist)``
        where ``pilot_ref`` has shape ``(n_sym, C)`` and ``pilot_mask``
        has shape ``(n_sym,)`` bool.
    """
    key = ("pa_rde", num_taps, stride, num_radii, num_ch)
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()

        @functools.partial(jax.jit, static_argnums=(6,))
        def pa_rde_scan(
            x_input,  # (C, N_pad)  complex64 — padded received samples
            w_init,  # (C, C, T)   complex64 — initial butterfly weights
            step_size,  # ()          float32
            radii,  # (K,)        float32 — unique |c| constellation radii, sorted
            pilot_ref,  # (n_sym, C)  complex64 — known symbols; 0 at data
            pilot_mask,  # (n_sym,)    bool — True at pilot/preamble positions
            n_sym,  # int (static) — total symbol count
        ):
            def step(W, xs_t):
                idx, p_ref, p_mask = xs_t  # (): int, (C,): cplx, (): bool
                X_wins = jax.lax.dynamic_slice(
                    x_input, (0, idx * stride), (num_ch, num_taps)
                )  # (C, T)
                y = jnp.einsum("ijt,jt->i", jnp.conj(W), X_wins)  # (C,)

                abs_y2 = jnp.real(y * jnp.conj(y))  # strict real |y|²
                abs_y = jnp.sqrt(abs_y2)  # (C,)
                dist = jnp.abs(abs_y[:, None] - radii[None, :])  # (C, K) broadcast
                rd = radii[jnp.argmin(dist, axis=1)]  # (C,) nearest radius

                e_blind = y * (abs_y2 - rd**2)  # RDE ring-directed
                e_da = (
                    y - p_ref
                )  # pilot LMS (inverted to match blind subtractive update)
                e = jnp.where(p_mask, e_da, e_blind)  # (C,) branchless

                W_new = W - step_size * jnp.einsum("i,jt->ijt", jnp.conj(e), X_wins)
                return W_new, (y, e, W_new)

            xs = (jnp.arange(n_sym), pilot_ref, pilot_mask)
            W_final, (y_all, e_all, wh_all) = jax.lax.scan(step, w_init, xs)
            return y_all, e_all, W_final, wh_all

        _JITTED_EQ[key] = pa_rde_scan
    return _JITTED_EQ[key]


# -----------------------------------------------------------------------------
# SHARED HELPERS
# -----------------------------------------------------------------------------


def _normalize_inputs(samples, training_symbols, sps):
    """Scale samples and training symbols to a common unit symbol-power reference.

    For fractionally-spaced equalization (sps > 1) the fractional timing phase
    is unknown.  Strided power measurement ``samples[..., ::sps]`` is unsafe
    because it can land on zero-crossings of the Nyquist pulse, severely
    underestimating signal power and destabilising adaptation.

    Instead the *wideband* (all-sample) power is used via
    ``normalize(..., "symbol_power", sps=sps)``, which divides by
    ``rms(samples) * √sps``.  This estimate is phase-invariant and gives
    unit symbol energy (Es = 1) for any pulse shape with unit-energy taps.
    Works transparently on NumPy and CuPy arrays via the helper dispatch.

    Parameters
    ----------
    samples          : (C, N) or (N,)  complex, any backend (NumPy / CuPy)
    training_symbols : (C, K) or (K,)  or None — always at 1 sps
    sps              : int — samples per symbol

    Returns
    -------
    samples          : unit symbol-power, same shape/backend
    training_symbols : unit average-power, same shape/backend (or None)
    input_norm_factor : float or np.ndarray
        Per-channel normalization factor(s) ``rms(ch) * sqrt(sps)`` applied
        to *samples* before this function returned.  ``float`` for SISO,
        ``np.ndarray`` of shape ``(C,)`` for MIMO.  Stored in
        ``EqualizerResult.input_norm_factor`` so callers can reconstruct
        the physical power scale of a different capture (e.g. vacuum noise)
        passed through the same frozen taps.  See ``EqualizerResult`` docs.
    """
    from commstools.helpers import normalize as c_normalize, rms as _rms

    # Per-channel norm: rms(ch) * sqrt(sps) — matches the per-channel scaling
    # that c_normalize("symbol_power", axis=-1) applies below.
    # For SISO (N,): _rms(axis=-1) is a 0-d scalar → stored as float.
    # For MIMO (C, N): _rms(axis=-1) is (C,) → stored as np.ndarray on CPU.
    norm_vec = _rms(samples, axis=-1) * (sps**0.5)
    if samples.ndim == 1:
        input_norm_factor = float(norm_vec)
    else:
        input_norm_factor = to_device(norm_vec, "cpu")
    samples = c_normalize(samples, "symbol_power", sps=sps, axis=-1)

    if training_symbols is not None:
        # Training symbols are at 1 sps; "average_power" == "symbol_power" at sps=1.
        training_symbols = c_normalize(training_symbols, "average_power", axis=-1)

    return samples, training_symbols, input_norm_factor


def _init_butterfly_weights_jax(num_ch, num_taps, jnp, center_tap=None):
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


def _validate_w_init(w: np.ndarray, num_ch: int, num_taps: int) -> np.ndarray:
    """Validate w_init shape and return it in butterfly layout ``(C, C, T)``.

    The library's unpack helpers squeeze SISO weights from ``(1, 1, T)`` to
    ``(T,)`` in ``EqualizerResult.weights`` for user convenience.  This means
    a weight array produced by one SISO equalizer stage and passed as ``w_init``
    to the next stage arrives here as ``(T,)``; that shape must be accepted.

    Parameters
    ----------
    w        : np.ndarray — candidate w_init array (already cast to NumPy)
    num_ch   : int        — expected number of channels C
    num_taps : int        — expected number of FIR taps T

    Returns
    -------
    np.ndarray — w reshaped to ``(num_ch, num_ch, num_taps)`` if needed.

    Raises
    ------
    ValueError if the shape cannot be mapped to ``(num_ch, num_ch, num_taps)``.
    """
    expected = (num_ch, num_ch, num_taps)
    if w.shape == expected:
        return w
    # SISO: accept the squeezed shapes emitted by _unpack_result_*:
    #   (T,)    — both channel dims collapsed  (_unpack: W[0, 0])
    #   (1, T)  — one channel dim collapsed
    if num_ch == 1 and w.shape in ((num_taps,), (1, num_taps)):
        return w.reshape(1, 1, num_taps)
    raise ValueError(
        f"w_init shape {tuple(w.shape)} does not match expected "
        f"(num_ch={num_ch}, num_ch={num_ch}, num_taps={num_taps}) = {expected}."
    )


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
    (use ``to_device(training_symbols, "cpu")`` before calling).

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
    input_norm_factor=1.0,
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
        input_norm_factor=input_norm_factor,
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
    input_norm_factor=1.0,
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
        input_norm_factor=input_norm_factor,
    )


def _cpr_pll_gains(bandwidth: float):
    """Convert normalised loop bandwidth to PI gains (mu, beta).

    Uses the standard 2nd-order loop approximation for a critically-damped
    (ζ = 1/√2) PI loop:  μ ≈ 4·B_L,  β ≈ 4·B_L².

    Parameters
    ----------
    bandwidth : float
        Normalised one-sided loop bandwidth as a fraction of the symbol rate,
        e.g. ``1e-3`` for a narrow loop.

    Returns
    -------
    mu, beta : float32
    """
    mu = np.float32(4.0 * bandwidth)
    beta = np.float32(4.0 * bandwidth**2)
    return mu, beta


def _cpr_symmetry(modulation: Optional[str], order: Optional[int]) -> int:
    """Return the rotational symmetry order used for cycle-slip correction.

    QAM and most practical CPR algorithms exploit 4-fold (π/2) symmetry.
    BPSK is the only exception (2-fold).

    Parameters
    ----------
    modulation : str or None
    order : int or None

    Returns
    -------
    int — 4 (default/QAM/PSK M≥4) or 2 (BPSK/PAM)
    """
    if modulation is None:
        return 4
    m = modulation.lower().strip()
    if m in ("pam",):
        return 2
    if m in ("psk", "bpsk") and order == 2:
        return 2
    return 4


def _validate_sps(sps, num_taps):
    """Validate sps; warn about unusual values, check tap count minimum."""
    if sps < 1:
        raise ValueError(f"sps must be >= 1. Got sps={sps}.")
    if sps == 1:
        logger.info(
            "sps=1: symbol-spaced equalizer mode. "
            "No fractional-spacing benefit; suitable for residual ISI correction "
            "after a prior FSE stage + FOE + CPR."
        )
    elif sps > 2:
        logger.warning(
            f"sps={sps}: non-standard oversampling ratio for adaptive equalization. "
            "T/2-spaced (sps=2) is the industry standard. Higher values give "
            "marginal benefit but require proportionally more taps."
        )
    if num_taps < 2 * sps:
        logger.warning(
            f"num_taps={num_taps} is small for sps={sps}. "
            f"Recommend num_taps >= {4 * sps + 1} for fractionally-spaced equalization."
        )


# -----------------------------------------------------------------------------
# ADAPTIVE equalization
# -----------------------------------------------------------------------------


def lms(
    samples: ArrayType,
    training_symbols: Optional[ArrayType] = None,
    num_taps: int = 21,
    sps: int = 2,
    step_size: float = 0.01,
    modulation: Optional[str] = None,
    order: Optional[int] = None,
    unipolar: bool = False,
    store_weights: bool = False,
    num_train_symbols: Optional[int] = None,
    device: Optional[str] = "cpu",
    center_tap: Optional[int] = None,
    backend: str = "numba",
    w_init: Optional[ArrayType] = None,
    pmf: Optional[Any] = None,
    cpr_type: Optional[str] = None,
    cpr_pll_bandwidth: float = 1e-3,
    cpr_bps_test_phases: int = 64,
    cpr_bps_block_size: int = 32,
    cpr_bps_joint_channels: bool = False,
    cpr_cycle_slip_correction: bool = True,
    cpr_cycle_slip_history: int = 1000,
    cpr_cycle_slip_threshold: float = np.pi / 4,
    debug_plot: bool = False,
    plot_smoothing: int = 50,
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
    sps : int, default 2
        Samples per symbol at the input.  Use ``sps=2`` (T/2-spaced, default)
        for the first equalization stage.  ``sps=1`` is valid for a second
        symbol-spaced stage after FOE + CPR; use a short filter (3-11 taps)
        for residual ISI cleanup.  ``sps > 2`` is accepted but uncommon.
        The equalizer decimates by ``sps`` to produce one output symbol per
        input stride.
    step_size : float, default 0.01
        Plain LMS step size (mu). The gradient is applied directly without
        input-power normalization, matching the convention in Haykin's
        *Adaptive Filter Theory* and most published papers.  Stability
        requires ``0 < mu < 2 / (C * num_taps * P_x)`` where ``P_x`` is the
        mean per-tap input power.  Because inputs are normalized to unit
        symbol-rate power by default, a safe starting range for typical
        settings is ``1e-4`` to ``1e-2``.  Values closer to the upper bound
        converge faster but produce higher steady-state misadjustment.
    modulation : str, optional
        Modulation scheme (e.g., 'psk', 'qam', 'pam') for DD slicing.
        Required if ``training_symbols`` is None.
    order : int, optional
        Modulation order (e.g., 4, 16).
    unipolar : bool, default False
        If True, indicates the modulation is unipolar (e.g., unipolar PAM).
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
    w_init : array_like, optional
        Initial tap weights. Shape: ``(C, C, num_taps)`` complex64, or the
        SISO short-hand ``(num_taps,)`` / ``(1, num_taps)`` as returned by
        ``EqualizerResult.weights`` for single-channel equalizers.
        If provided, the equalizer warm-starts from these weights instead of
        the default center-tap identity matrix.  Useful for weight handoff from
        a prior stage (e.g. preamble LMS → payload LMS).
        Raises ``ValueError`` if the shape does not match.
    pmf : array_like of float, optional
        Probability mass function over the constellation for probabilistically
        shaped QAM (PS-QAM).  When provided together with ``modulation`` and
        ``order``, the DD slicer constellation is scaled by ``1/sqrt(E_PS)``
        (where ``E_PS = sum_m P(s_m)|s_m|^2`` on the normalised grid) so it
        matches the unit-power normalised equaliser input.  Training symbols
        are left untouched — ``_normalize_inputs`` already brings them to unit
        average power.  Has no effect for uniform modulations.
    cpr_type : {'pll', 'bps', None}, default None
        Inline carrier phase recovery algorithm applied jointly with weight
        updates at every symbol.  ``None`` disables CPR (default, bit-exact
        with the legacy behaviour).

        * ``'pll'`` — 2nd-order decision-directed phase-locked loop.  The
          cross-product phase detector ``Im(y · conj(d))`` drives a PI loop
          with gains derived from ``cpr_pll_bandwidth``.  Low noise floor;
          recommended for QPSK through 64-QAM.
        * ``'bps'`` — Blind Phase Search over ``cpr_bps_test_phases`` candidate
          angles in ``[0, π/2)`` (exploiting 4-fold QAM symmetry), averaged
          over a causal window of ``cpr_bps_block_size`` past y_raw samples.
          Preferred for burst/packet modes where PLL pull-in is impractical.
    cpr_pll_bandwidth : float, default 1e-3
        Normalised loop bandwidth ``B_L · T_s`` for the PLL.  Gains are
        computed as ``K_p = 4 B_L``, ``K_i = 4 B_L²`` (critically-damped
        approximation).  Typical range: ``5e-4`` (low phase noise) to
        ``5e-3`` (high phase noise / fast drift).  Ignored when
        ``cpr_type != 'pll'``.
    cpr_bps_test_phases : int, default 64
        Number of candidate phase angles for the BPS search in ``[0, π/2)``.
        Higher values improve phase resolution at the cost of ``B`` extra
        distance evaluations per symbol.  32-64 is sufficient for ≤ 16-QAM;
        use 64-128 for 64-QAM.  Ignored when ``cpr_type != 'bps'``.
    cpr_bps_block_size : int, default 32
        Number of past y_raw samples whose min-distance metrics are summed
        before the BPS ``argmin``.  Larger values reduce noise on the phase
        estimate at the cost of increased latency (``K-1`` symbols).
        ``cpr_bps_block_size=1`` recovers the degenerate single-symbol BPS.
        Ignored when ``cpr_type != 'bps'``.
    cpr_bps_joint_channels : bool, default False
        For MIMO inputs (C > 1): if ``True``, sum the BPS distance metric
        across all C channels before ``argmin``, producing one shared phase
        estimate broadcast to all channels.  Reduces estimation variance by
        ~√C for shared-LO systems.  If ``False``, each channel estimates its
        phase independently.  Ignored when ``cpr_type != 'bps'`` or C == 1.
    cpr_cycle_slip_correction : bool, default True
        Enable causal cycle-slip detection and correction.  A circular buffer
        of ``cpr_cycle_slip_history`` past phase estimates is maintained per
        channel; a linear trend is extrapolated to predict the next phase.
        If the new estimate deviates by more than ``cpr_cycle_slip_threshold``
        from the prediction, it is snapped to the nearest ``2π/symmetry``
        quantum.  Disable for parity checks or when the channel is known to
        be slip-free.
    cpr_cycle_slip_history : int, default 1000
        Length of the phase-history buffer used for cycle-slip extrapolation.
        Longer buffers give a more accurate linear-trend estimate but are
        slower to adapt to genuine frequency steps.  Ignored when
        ``cpr_cycle_slip_correction=False``.
    cpr_cycle_slip_threshold : float, default π/4
        Maximum tolerated deviation (radians) between the predicted and
        observed phase before a slip is declared.  Should be set to half the
        constellation's angular symmetry quantum (``π/4`` for QPSK/QAM).
        Ignored when ``cpr_cycle_slip_correction=False``.
    debug_plot : bool, default False
        If True, display a convergence + tap-weight diagnostic plot on exit.
    plot_smoothing : int, default 50
        Moving-average window (symbols) for the MSE convergence curve in the
        debug plot.

    Returns
    -------
    EqualizerResult
        Equalized symbols, final weights, error history, and optionally
        weight trajectory and phase trajectory.  Arrays reside on the same
        backend as input.

    Warnings
    --------
    **JAX GPU mode is typically slower than CPU for adaptive equalization.**
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
    if cpr_type is not None and cpr_type not in ("pll", "bps"):
        raise ValueError(f"cpr_type must be 'pll', 'bps', or None. Got {cpr_type!r}.")

    logger.info(
        f"LMS equalizer: num_taps={num_taps}, mu={step_size}, sps={sps}, "
        f"backend={backend}, num_train_symbols={num_train_symbols}"
        + (f", cpr={cpr_type}" if cpr_type else "")
    )
    if sps > 1:
        logger.warning(
            "LMS output y_hat is at 1 SPS (symbol rate). "
            "Update sampling_rate = symbol_rate after applying this equalizer."
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

    if training_symbols is not None and training_symbols.shape[-1] > n_sym:
        logger.warning(
            f"training_symbols length ({training_symbols.shape[-1]}) exceeds "
            f"available symbol count ({n_sym}); excess training symbols will be ignored."
        )
    if num_train_symbols is not None and num_train_symbols > n_sym:
        logger.warning(
            f"num_train_symbols={num_train_symbols} exceeds "
            f"available symbol count ({n_sym}); effective training count clamped to {n_sym}."
        )

    c_tap = center_tap if center_tap is not None else num_taps // 2
    pad_total = max(0, n_sym * stride - n_samples + num_taps - 1)
    pad_left = min(c_tap, pad_total)
    pad_right = pad_total - pad_left

    if backend == "numba":
        # Convert to plain NumPy (no-op for CPU NumPy; downloads for CuPy)
        samples_np = np.ascontiguousarray(to_device(samples, "cpu"), dtype=np.complex64)
        training_np = (
            to_device(training_symbols, "cpu").astype(np.complex64)
            if training_symbols is not None
            else None
        )
        samples_np, training_np, eq_norm = _normalize_inputs(
            samples_np, training_np, sps
        )
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
                to_device(reference_constellation, "cpu").flatten().astype(np.complex64)
            )
        elif training_np is not None:
            train_flat = training_np.reshape(-1)
            constellation_np = np.unique(np.round(train_flat, decimals=8))
        else:
            raise ValueError("modulation and order must be provided for DD mode.")
        # PS-QAM: scale slicer constellation to unit-power {s_m/sqrt(E_PS)} so it
        # matches the normalised equaliser input.  Training is already at unit power
        # after _normalize_inputs — only the constellation reference needs scaling.
        if pmf is not None and modulation is not None and order is not None:
            _pmf_arr = np.asarray(pmf, dtype=np.float64)
            _e_ps = float(
                np.dot(_pmf_arr, np.abs(constellation_np).astype(np.float64) ** 2)
            )
            if _e_ps < 1.0 - 1e-6:
                _c_ps = np.float32(1.0 / np.sqrt(_e_ps))
                constellation_np = (constellation_np * _c_ps).astype(np.complex64)
        train_full, n_train_aligned = _prepare_training_numpy(
            training_np,
            num_ch,
            n_sym,
            num_train_symbols=num_train_symbols,
        )
        if w_init is not None:
            w_arr = np.ascontiguousarray(to_device(w_init, "cpu"), dtype=np.complex64)
            w_arr = _validate_w_init(w_arr, num_ch, num_taps)
            W = w_arr.copy()
        else:
            W = _init_butterfly_weights_numpy(num_ch, num_taps, center_tap=center_tap)
        y_out = np.empty((n_sym, num_ch), dtype=np.complex64)
        e_out = np.empty((n_sym, num_ch), dtype=np.complex64)
        w_hist_buf = (
            np.empty((n_sym, num_ch, num_ch, num_taps), dtype=np.complex64)
            if store_weights
            else np.empty((1, num_ch, num_ch, num_taps), dtype=np.complex64)
        )
        if cpr_type is None:
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
            result = _unpack_result_numpy(
                y_out,
                e_out,
                W,
                w_hist_buf,
                was_1d,
                store_weights,
                n_sym=None,
                xp=xp,
                num_train_symbols=int(n_train_aligned),
                input_norm_factor=eq_norm,
            )
        else:
            pll_mu, pll_beta = _cpr_pll_gains(cpr_pll_bandwidth)
            symmetry = _cpr_symmetry(modulation, order)
            B = int(cpr_bps_test_phases)
            bps_angles_np = np.linspace(
                0.0, np.pi / 2.0, B, endpoint=False, dtype=np.float32
            )
            bps_phases_neg_np = np.exp(-1j * bps_angles_np).astype(np.complex64)
            H = int(cpr_cycle_slip_history)
            pll_phi = np.zeros(num_ch, dtype=np.float32)
            pll_freq = np.zeros(num_ch, dtype=np.float32)
            cs_buf_x = np.zeros((num_ch, H), dtype=np.float64)
            cs_buf_y = np.zeros((num_ch, H), dtype=np.float64)
            cs_buf_ptr = np.zeros(num_ch, dtype=np.int64)
            cs_buf_n = np.zeros(num_ch, dtype=np.int64)
            cs_stats = np.zeros((num_ch, 4), dtype=np.float64)
            phase_out = np.empty((n_sym, num_ch), dtype=np.float32)
            cpr_mode_int = np.int32(1 if cpr_type == "pll" else 2)
            _get_numba_lms_cpr()(
                samples_padded,
                train_full,
                constellation_np,
                bps_phases_neg_np,
                bps_angles_np,
                np.int32(cpr_bps_block_size),
                bool(cpr_bps_joint_channels),
                W,
                np.float32(step_size),
                np.int32(n_train_aligned),
                stride,
                store_weights,
                cpr_mode_int,
                pll_mu,
                pll_beta,
                np.int32(symmetry),
                bool(cpr_cycle_slip_correction),
                np.float32(cpr_cycle_slip_threshold),
                pll_phi,
                pll_freq,
                cs_buf_x,
                cs_buf_y,
                cs_buf_ptr,
                cs_buf_n,
                cs_stats,
                y_out,
                e_out,
                phase_out,
                w_hist_buf,
            )
            result = _unpack_result_numpy(
                y_out,
                e_out,
                W,
                w_hist_buf,
                was_1d,
                store_weights,
                n_sym=None,
                xp=xp,
                num_train_symbols=int(n_train_aligned),
                input_norm_factor=eq_norm,
            )
            phi_t = xp.asarray(phase_out.T)  # (C, N_sym)
            result.phase_trajectory = phi_t[0] if was_1d else phi_t
        return _log_equalizer_exit(
            result, name="LMS", debug_plot=debug_plot, plot_smoothing=plot_smoothing
        )

    # JAX backend
    jax, jnp, _ = _get_jax()
    if jax is None:
        raise ImportError("JAX is required for backend='jax'.")

    samples, training_symbols, eq_norm = _normalize_inputs(
        samples, training_symbols, sps
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
        to_device(reference_constellation, "cpu").flatten().astype("complex64")
    )
    # PS-QAM: scale slicer constellation to unit-power {s_m/sqrt(E_PS)}.
    if pmf is not None and modulation is not None and order is not None:
        _pmf_arr = np.asarray(pmf, dtype=np.float64)
        _e_ps = float(
            np.dot(_pmf_arr, np.abs(constellation_np).astype(np.float64) ** 2)
        )
        if _e_ps < 1.0 - 1e-6:
            constellation_np = (
                constellation_np * np.float32(1.0 / np.sqrt(_e_ps))
            ).astype(np.complex64)
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
    if w_init is not None:
        w_arr = np.ascontiguousarray(to_device(w_init, "cpu"), dtype=np.complex64)
        w_arr = _validate_w_init(w_arr, num_ch, num_taps)
        W_jax = to_jax(w_arr, device=platform)
    else:
        W_jax = _init_butterfly_weights_jax(
            num_ch, num_taps, jnp, center_tap=center_tap
        )
        W_jax = to_jax(W_jax, device=platform)
    mu_jax = to_jax(jnp.float32(step_size), device=platform)
    n_train_jax = to_jax(jnp.int32(n_train_aligned), device=platform)

    if cpr_type is None:
        scan_fn = _get_jax_lms(num_taps, stride, len(constellation_np), num_ch)
        y_jax, e_jax, W_jax, wh_jax = scan_fn(
            x_jax, train_jax, const_jax, W_jax, mu_jax, n_train_jax
        )
        result = _unpack_result_jax(
            y_jax,
            e_jax,
            W_jax,
            wh_jax,
            was_1d,
            store_weights,
            n_sym=None,
            xp=xp,
            num_train_symbols=int(n_train_aligned),
            input_norm_factor=eq_norm,
        )
    else:
        pll_mu, pll_beta = _cpr_pll_gains(cpr_pll_bandwidth)
        B = int(cpr_bps_test_phases)
        H = int(cpr_cycle_slip_history)
        bps_angles_np = np.linspace(
            0.0, np.pi / 2.0, B, endpoint=False, dtype=np.float32
        )
        bps_phases_neg_np = np.exp(-1j * bps_angles_np).astype(np.complex64)
        bps_pn_jax = to_jax(bps_phases_neg_np, device=platform)
        bps_ang_jax = to_jax(bps_angles_np, device=platform)
        scan_fn = _get_jax_lms_cpr(
            num_taps,
            stride,
            len(constellation_np),
            num_ch,
            cpr_type,
            B,
            int(cpr_bps_block_size),
            bool(cpr_bps_joint_channels),
            H,
        )
        y_jax, e_jax, W_jax, wh_jax, phi_jax = scan_fn(
            x_jax,
            train_jax,
            const_jax,
            bps_pn_jax,
            bps_ang_jax,
            W_jax,
            mu_jax,
            n_train_jax,
            to_jax(jnp.float32(pll_mu), device=platform),
            to_jax(jnp.float32(pll_beta), device=platform),
            to_jax(jnp.float32(cpr_cycle_slip_threshold), device=platform),
            to_jax(jnp.bool_(cpr_cycle_slip_correction), device=platform),
        )
        result = _unpack_result_jax(
            y_jax,
            e_jax,
            W_jax,
            wh_jax,
            was_1d,
            store_weights,
            n_sym=None,
            xp=xp,
            num_train_symbols=int(n_train_aligned),
            input_norm_factor=eq_norm,
        )
        phi_np = np.asarray(from_jax(phi_jax))  # (N_sym, C)
        phi_t = xp.asarray(phi_np.T)  # (C, N_sym)
        result.phase_trajectory = phi_t[0] if was_1d else phi_t
    return _log_equalizer_exit(result, name="LMS", debug_plot=debug_plot)


def rls(
    samples: ArrayType,
    training_symbols: Optional[ArrayType] = None,
    num_taps: int = 21,
    sps: int = 1,
    forgetting_factor: float = 0.99,
    delta: float = 0.01,
    leakage: float = 0.0,
    modulation: Optional[str] = None,
    order: Optional[int] = None,
    unipolar: bool = False,
    store_weights: bool = False,
    num_train_symbols: Optional[int] = None,
    device: Optional[str] = "cpu",
    center_tap: Optional[int] = None,
    backend: str = "numba",
    w_init: Optional[ArrayType] = None,
    pmf: Optional[Any] = None,
    cpr_type: Optional[str] = None,
    cpr_pll_bandwidth: float = 1e-3,
    cpr_bps_test_phases: int = 64,
    cpr_bps_block_size: int = 32,
    cpr_bps_joint_channels: bool = False,
    cpr_cycle_slip_correction: bool = True,
    cpr_cycle_slip_history: int = 1000,
    cpr_cycle_slip_threshold: float = np.pi / 4,
    debug_plot: bool = False,
    plot_smoothing: int = 50,
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
    sps : int, default 1
        Samples per symbol at the input.
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
          larger ``delta`` (0.1-1.0) helps counteract the positive-feedback
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
        For fractionally-spaced equalization start with ``leakage=1e-4`` and
        increase if steady-state EVM remains high.
    modulation : str, optional
        Modulation scheme (e.g., 'psk', 'qam', 'pam') for DD slicing.
        Required if ``training_symbols`` is None.
    order : int, optional
        Modulation order (e.g., 4, 16).
    unipolar : bool, default False
        If True, indicates the modulation is unipolar (e.g., unipolar PAM).
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
    pmf : array_like of float, optional
        Probability mass function for PS-QAM.  Scales the DD slicer
        constellation by ``1/sqrt(E_PS)`` to match the unit-power normalised
        equaliser input.  Requires ``modulation`` and ``order``.
    cpr_type : {'pll', 'bps', None}, default None
        Inline carrier phase recovery algorithm.  See :func:`lms` for full
        parameter documentation; behaviour is identical.
    cpr_pll_bandwidth : float, default 1e-3
        Normalised PLL loop bandwidth ``B_L · T_s``.  Ignored when
        ``cpr_type != 'pll'``.  See :func:`lms` for details.
    cpr_bps_test_phases : int, default 64
        Number of BPS candidate angles.  Ignored when ``cpr_type != 'bps'``.
        See :func:`lms` for details.
    cpr_bps_block_size : int, default 32
        BPS averaging window length.  See :func:`lms` for details.
    cpr_bps_joint_channels : bool, default False
        Joint MIMO BPS metric.  See :func:`lms` for details.
    cpr_cycle_slip_correction : bool, default True
        Enable causal cycle-slip detection.  See :func:`lms` for details.
    cpr_cycle_slip_history : int, default 1000
        Phase-history buffer length for slip extrapolation.  See :func:`lms`.
    cpr_cycle_slip_threshold : float, default π/4
        Slip detection threshold in radians.  See :func:`lms` for details.
    debug_plot : bool, default False
        Display convergence + tap-weight diagnostic plot on exit.
    plot_smoothing : int, default 50
        MSE moving-average window for the debug plot.

    Returns
    -------
    EqualizerResult
        Equalized symbols, final weights, error history, and optionally
        weight trajectory and phase trajectory.

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

    ``w_init`` warms-start the tap weights; the inverse correlation matrix ``P``
    always begins at ``(1/delta) · I`` regardless of ``w_init``.
    """
    if sps > 1:
        logger.warning(
            f"RLS is mathematically ill-conditioned for fractionally-spaced signals (sps={sps}). "
            "The noise-only null-subspace creates a singular correlation matrix, causing tap bloat. "
            "Use LMS for fractionally-spaced equalization unless heavy Tikhonov regularization is applied."
        )

    if cpr_type is not None and cpr_type not in ("pll", "bps"):
        raise ValueError(f"cpr_type must be 'pll', 'bps', or None. Got {cpr_type!r}.")

    logger.info(
        f"RLS equalizer: num_taps={num_taps}, forgetting_factor={forgetting_factor}, "
        f"delta={delta:.2e}, leakage={leakage:.2e}, sps={sps}, "
        f"backend={backend}, num_train_symbols={num_train_symbols}"
        + (f", cpr={cpr_type}" if cpr_type else "")
    )
    if sps > 1:
        logger.warning(
            "RLS output y_hat is at 1 SPS (symbol rate). "
            "Update sampling_rate = symbol_rate after applying this equalizer."
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

    if training_symbols is not None and training_symbols.shape[-1] > n_sym:
        logger.warning(
            f"training_symbols length ({training_symbols.shape[-1]}) exceeds "
            f"available symbol count ({n_sym}); excess training symbols will be ignored."
        )
    if num_train_symbols is not None and num_train_symbols > n_sym:
        logger.warning(
            f"num_train_symbols={num_train_symbols} exceeds "
            f"available symbol count ({n_sym}); effective training count clamped to {n_sym}."
        )

    # Early-halt boundary: freeze W and P once the sliding window reaches the
    # right zero-padding (last num_taps//2 symbols have contaminated windows).
    n_update_halt = max(0, n_sym - num_taps // 2)
    tail_trim = num_taps // 2
    if tail_trim > 0:
        logger.warning(
            f"RLS tail trim: last {tail_trim} symbols removed from y_hat "
            "(zero-padding contamination zone). Trim reference arrays to match: "
            "source_symbols = source_symbols[..., :-result.tail_trim], "
            "source_bits = source_bits[..., :-result.tail_trim * bits_per_symbol]."
        )

    c_tap = center_tap if center_tap is not None else num_taps // 2
    pad_total = max(0, n_sym * stride - n_samples + num_taps - 1)
    pad_left = min(c_tap, pad_total)
    pad_right = pad_total - pad_left

    if backend == "numba":
        numba = _get_numba()
        if numba is None:
            raise ImportError("Numba is required for backend='numba'.")

        samples_np = np.ascontiguousarray(to_device(samples, "cpu"), dtype=np.complex64)
        training_np = (
            to_device(training_symbols, "cpu").astype(np.complex64)
            if training_symbols is not None
            else None
        )
        samples_np, training_np, eq_norm = _normalize_inputs(
            samples_np, training_np, sps
        )

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
                to_device(reference_constellation, "cpu").flatten().astype(np.complex64)
            )
        elif training_np is not None:
            train_flat = training_np.reshape(-1)
            constellation_np = np.unique(np.round(train_flat, decimals=8)).astype(
                "complex64"
            )
        else:
            raise ValueError("modulation and order must be provided for DD mode.")
        # PS-QAM: scale slicer constellation to unit-power {s_m/sqrt(E_PS)}.
        if pmf is not None and modulation is not None and order is not None:
            _pmf_arr = np.asarray(pmf, dtype=np.float64)
            _e_ps = float(
                np.dot(_pmf_arr, np.abs(constellation_np).astype(np.float64) ** 2)
            )
            if _e_ps < 1.0 - 1e-6:
                _c_ps = np.float32(1.0 / np.sqrt(_e_ps))
                constellation_np = (constellation_np * _c_ps).astype(np.complex64)

        train_full, n_train_aligned = _prepare_training_numpy(
            training_np,
            num_ch,
            n_sym,
            num_train_symbols=num_train_symbols,
        )
        if w_init is not None:
            w_arr = np.ascontiguousarray(to_device(w_init, "cpu"), dtype=np.complex64)
            w_arr = _validate_w_init(w_arr, num_ch, num_taps)
            W = w_arr.copy()
        else:
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
        if cpr_type is None:
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
            result = _unpack_result_numpy(
                y_out,
                e_out,
                W,
                w_hist_buf,
                was_1d,
                store_weights,
                n_sym=n_update_halt,
                xp=xp,
                num_train_symbols=int(n_train_aligned),
                input_norm_factor=eq_norm,
            )
        else:
            pll_mu, pll_beta = _cpr_pll_gains(cpr_pll_bandwidth)
            symmetry = _cpr_symmetry(modulation, order)
            B = int(cpr_bps_test_phases)
            bps_angles_np = np.linspace(
                0.0, np.pi / 2.0, B, endpoint=False, dtype=np.float32
            )
            bps_phases_neg_np = np.exp(-1j * bps_angles_np).astype(np.complex64)
            H = int(cpr_cycle_slip_history)
            pll_phi = np.zeros(num_ch, dtype=np.float32)
            pll_freq = np.zeros(num_ch, dtype=np.float32)
            cs_buf_x = np.zeros((num_ch, H), dtype=np.float64)
            cs_buf_y = np.zeros((num_ch, H), dtype=np.float64)
            cs_buf_ptr = np.zeros(num_ch, dtype=np.int64)
            cs_buf_n = np.zeros(num_ch, dtype=np.int64)
            cs_stats = np.zeros((num_ch, 4), dtype=np.float64)
            phase_out = np.empty((n_sym, num_ch), dtype=np.float32)
            cpr_mode_int = np.int32(1 if cpr_type == "pll" else 2)
            _get_numba_rls_cpr()(
                x_np,
                train_full,
                constellation_np,
                bps_phases_neg_np,
                bps_angles_np,
                np.int32(cpr_bps_block_size),
                bool(cpr_bps_joint_channels),
                W,
                P,
                np.float32(forgetting_factor),
                np.float32(leakage),
                np.int32(n_train_aligned),
                np.int32(n_update_halt),
                stride,
                store_weights,
                cpr_mode_int,
                pll_mu,
                pll_beta,
                np.int32(symmetry),
                bool(cpr_cycle_slip_correction),
                np.float32(cpr_cycle_slip_threshold),
                pll_phi,
                pll_freq,
                cs_buf_x,
                cs_buf_y,
                cs_buf_ptr,
                cs_buf_n,
                cs_stats,
                y_out,
                e_out,
                phase_out,
                w_hist_buf,
            )
            result = _unpack_result_numpy(
                y_out,
                e_out,
                W,
                w_hist_buf,
                was_1d,
                store_weights,
                n_sym=n_update_halt,
                xp=xp,
                num_train_symbols=int(n_train_aligned),
                input_norm_factor=eq_norm,
            )
            phi_t = xp.asarray(phase_out[:n_update_halt].T)  # (C, n_update_halt)
            result.phase_trajectory = phi_t[0] if was_1d else phi_t
        # Truncate last num_taps//2 symbols (zero-padding contamination).
        result = _log_equalizer_exit(
            result, name="RLS", debug_plot=debug_plot, plot_smoothing=plot_smoothing
        )
        result.tail_trim = tail_trim
        return result

    # JAX backend
    jax, jnp, _ = _get_jax()
    if jax is None:
        raise ImportError("JAX is required for backend='jax'.")

    samples, training_symbols, eq_norm = _normalize_inputs(
        samples, training_symbols, sps
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
        to_device(reference_constellation, "cpu").flatten().astype("complex64")
    )
    # PS-QAM: scale slicer constellation to unit-power {s_m/sqrt(E_PS)}.
    if pmf is not None and modulation is not None and order is not None:
        _pmf_arr = np.asarray(pmf, dtype=np.float64)
        _e_ps = float(
            np.dot(_pmf_arr, np.abs(constellation_np).astype(np.float64) ** 2)
        )
        if _e_ps < 1.0 - 1e-6:
            constellation_np = (
                constellation_np * np.float32(1.0 / np.sqrt(_e_ps))
            ).astype(np.complex64)

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
    if w_init is not None:
        w_arr = np.ascontiguousarray(to_device(w_init, "cpu"), dtype=np.complex64)
        w_arr = _validate_w_init(w_arr, num_ch, num_taps)
        W_jax = to_jax(w_arr, device=platform)
    else:
        W_jax = _init_butterfly_weights_jax(
            num_ch, num_taps, jnp, center_tap=center_tap
        )
        W_jax = to_jax(W_jax, device=platform)

    regressor_dim = num_ch * num_taps
    P_init = jnp.eye(regressor_dim, dtype="complex64") / delta
    P_init = to_jax(P_init, device=platform)
    lam_jax = to_jax(jnp.float32(forgetting_factor), device=platform)
    n_train_jax = to_jax(jnp.int32(n_train_aligned), device=platform)
    leakage_jax = to_jax(jnp.float32(leakage), device=platform)
    n_update_halt_jax = to_jax(jnp.int32(n_update_halt), device=platform)

    if cpr_type is None:
        scan_fn = _get_jax_rls(num_taps, stride, len(constellation_np), num_ch)
        y_jax, e_jax, W_jax, wh_jax = scan_fn(
            x_jax,
            train_jax,
            const_jax,
            W_jax,
            P_init,
            lam_jax,
            n_train_jax,
            leakage_jax,
            n_update_halt_jax,
        )
        result = _unpack_result_jax(
            y_jax,
            e_jax,
            W_jax,
            wh_jax,
            was_1d,
            store_weights,
            n_sym=n_update_halt,
            xp=xp,
            num_train_symbols=int(n_train_aligned),
            input_norm_factor=eq_norm,
        )
    else:
        pll_mu, pll_beta = _cpr_pll_gains(cpr_pll_bandwidth)
        B = int(cpr_bps_test_phases)
        H = int(cpr_cycle_slip_history)
        bps_angles_np = np.linspace(
            0.0, np.pi / 2.0, B, endpoint=False, dtype=np.float32
        )
        bps_phases_neg_np = np.exp(-1j * bps_angles_np).astype(np.complex64)
        bps_pn_jax = to_jax(bps_phases_neg_np, device=platform)
        bps_ang_jax = to_jax(bps_angles_np, device=platform)
        scan_fn = _get_jax_rls_cpr(
            num_taps,
            stride,
            len(constellation_np),
            num_ch,
            cpr_type,
            B,
            int(cpr_bps_block_size),
            bool(cpr_bps_joint_channels),
            H,
        )
        y_jax, e_jax, W_jax, wh_jax, phi_jax = scan_fn(
            x_jax,
            train_jax,
            const_jax,
            bps_pn_jax,
            bps_ang_jax,
            W_jax,
            P_init,
            lam_jax,
            n_train_jax,
            leakage_jax,
            n_update_halt_jax,
            to_jax(jnp.float32(pll_mu), device=platform),
            to_jax(jnp.float32(pll_beta), device=platform),
            to_jax(jnp.float32(cpr_cycle_slip_threshold), device=platform),
            to_jax(jnp.bool_(cpr_cycle_slip_correction), device=platform),
        )
        result = _unpack_result_jax(
            y_jax,
            e_jax,
            W_jax,
            wh_jax,
            was_1d,
            store_weights,
            n_sym=n_update_halt,
            xp=xp,
            num_train_symbols=int(n_train_aligned),
            input_norm_factor=eq_norm,
        )
        phi_np = np.asarray(from_jax(phi_jax))  # (N_sym, C)
        phi_t = xp.asarray(phi_np[:n_update_halt].T)  # (C, n_update_halt)
        result.phase_trajectory = phi_t[0] if was_1d else phi_t
    # Truncate last num_taps//2 symbols (zero-padding contamination).
    result = _log_equalizer_exit(result, name="RLS", debug_plot=debug_plot)
    result.tail_trim = tail_trim
    return result


def block_lms(
    samples: ArrayType,
    training_symbols: Optional[ArrayType] = None,
    num_taps: int = 21,
    sps: int = 2,
    step_size: float = 0.01,
    block_size: int = 256,
    modulation: Optional[str] = None,
    order: Optional[int] = None,
    unipolar: bool = False,
    store_weights: bool = False,
    num_train_symbols: Optional[int] = None,
    w_init: Optional[ArrayType] = None,
    pmf: Optional[Any] = None,
    cpr_type: Optional[str] = None,
    cpr_bps_test_phases: int = 64,
    cpr_bps_block_size: int = 32,
    cpr_bps_joint_channels: bool = False,
    cpr_cycle_slip_correction: bool = True,
    cpr_cycle_slip_history: int = 1000,
    cpr_cycle_slip_threshold: float = np.pi / 4,
    debug_plot: bool = False,
    plot_smoothing: int = 50,
) -> EqualizerResult:
    """Block LMS equalizer with frequency-domain gradient accumulation.

    Processes the signal in fixed-size blocks of ``block_size`` symbols.
    Within each block the filter is held frozen, all ``block_size`` errors are
    accumulated into a single frequency-domain gradient, and the weights are
    updated once per block.  This amortises the FFT overhead over many symbols,
    making it significantly more efficient than per-symbol LMS on GPU for large
    MIMO configurations (C ≥ 4) or long sequences.

    The primary target is **GPU** via CuPy.  On CPU, per-symbol LMS with the
    Numba backend (``lms(..., backend='numba')``) is typically faster because
    the block-FFT overhead outweighs the gradient-accumulation saving for
    small channel counts.

    Algorithm (per block b)
    -----------------------
    1. **Forward pass** — frequency-domain butterfly filter:

       .. math::

           Y_{\\text{fd}}[i] = \\sum_j \\overline{H_{\\text{fd}}[i,j]} \\cdot X_{\\text{fd}}[j]

       where :math:`H_{\\text{fd}} = \\mathrm{FFT}(h, n=F)` and
       :math:`X_{\\text{fd}} = \\mathrm{FFT}(x_{\\text{block}}, n=F)`.
       Output symbols are extracted at decimated positions ``y[n] = y_time[n·sps]``.

    2. **BPS phase recovery** (if ``cpr_type='bps'``) — for each symbol in the
       block, averages the min-distance metric over a causal trailing window of
       ``cpr_bps_block_size`` symbols and picks the minimum-metric candidate
       rotation.  This produces one phase estimate per symbol (not one per
       block), so ``cpr_bps_block_size`` and ``block_size`` are independent
       parameters: ``block_size`` controls FFT/gradient efficiency while
       ``cpr_bps_block_size`` controls phase noise suppression.

    3. **Error** — training or DD slicer; back-rotated to the tap plane:

       .. math::

           e_{\\text{taps}}[n] = e_{\\text{clean}}[n] \\cdot e^{+j\\varphi_b}

    4. **Gradient** — scatter ``e_taps`` to sample positions, then:

       .. math::

           \\Delta H_{\\text{fd}}[i,j] = \\overline{E_{\\text{fd}}[i]} \\cdot X_{\\text{fd}}[j]

           h \\mathrel{+}= \\frac{\\mu}{B} \\cdot \\mathrm{IFFT}(\\Delta H_{\\text{fd}})[\\ldots:T]

       Dividing by the actual block length *B* normalises the accumulated
       gradient to the per-symbol equivalent, so the same ``step_size`` value
       gives similar convergence behaviour to :func:`lms`.

    Parameters
    ----------
    samples : array_like
        Input signal samples.  Shape: ``(N_samples,)`` for SISO or
        ``(C, N_samples)`` for MIMO butterfly equalization.
        Typically at 2 samples/symbol for fractionally-spaced equalization.
    training_symbols : array_like, optional
        Known transmitted symbols at 1 SPS.
        Shape: ``(N_train,)`` for SISO or ``(C, N_train)`` for MIMO.
    num_taps : int, default 21
        Number of taps per FIR filter (tap count in samples).
    sps : int, default 2
        Samples per symbol.  ``sps=2`` (T/2-spaced) is the default.
    step_size : float, default 0.01
        LMS step size μ.  The accumulated block gradient is normalised by the
        block length before applying μ, so the same value is appropriate as
        for per-symbol :func:`lms`.  Stability requires
        ``0 < μ < 2/(C·T·P_x)`` — the same bound as per-symbol LMS.
    block_size : int, default 256
        Number of output symbols per LMS gradient accumulation block.  Larger
        values increase GPU efficiency but reduce adaptation speed.  Independent
        of the BPS averaging window (see ``cpr_bps_block_size``).
    modulation : str, optional
        Modulation scheme (e.g., ``'qam'``, ``'psk'``).  Required when
        ``training_symbols`` is ``None``.
    order : int, optional
        Modulation order (e.g., 16, 64).
    unipolar : bool, default False
        Unipolar PAM flag.
    store_weights : bool, default False
        If ``True``, stores the weight tensor at every block start in
        ``EqualizerResult.weights_history``.
    num_train_symbols : int, optional
        Clip training to this many symbols.
    w_init : array_like, optional
        Initial tap weights, shape ``(C, C, T)`` or SISO short-hands.
    pmf : array_like, optional
        Probability mass function for PS-QAM constellation scaling.
    cpr_type : {'bps', None}, default None
        Inline carrier phase recovery.  Only ``'bps'`` is supported for
        block LMS; PLL is not available because per-symbol integration does
        not fit the block processing model.
    cpr_bps_test_phases : int, default 64
        Number of BPS candidate angles in ``[0, π/2)``.
    cpr_bps_block_size : int, default 32
        Trailing-window length (symbols) for BPS metric averaging.  At each
        symbol position the min-distance metric is summed over the last
        ``cpr_bps_block_size`` symbols before the argmin.  Independent of
        ``block_size``; matches the semantics of the same parameter in
        :func:`lms`.  Larger values reduce phase-noise variance at the cost
        of slower tracking of rapid phase changes.
    cpr_bps_joint_channels : bool, default False
        Sum BPS metric across all C channels before argmin (shared LO).
    cpr_cycle_slip_correction : bool, default True
        Enable per-block cycle-slip detection.  An online linear regression
        predictor is maintained over the past ``cpr_cycle_slip_history``
        blocks; each block's midpoint phase is compared against the
        prediction, and if the deviation exceeds ``cpr_cycle_slip_threshold``
        the entire block's phase trajectory is shifted by the nearest
        ``2π/symmetry`` quantum.  When disabled, no device↔host transfer
        of the phase tensor occurs.
    cpr_cycle_slip_history : int, default 1000
        Number of past symbols used for the linear-trend predictor.
    cpr_cycle_slip_threshold : float, default π/4
        Phase deviation that triggers a slip correction (radians).
    debug_plot : bool, default False
        Show a convergence + phase diagnostic plot on exit.
    plot_smoothing : int, default 50
        Moving-average window for the MSE curve in the debug plot.

    Returns
    -------
    EqualizerResult
        Same fields as :func:`lms`.  ``phase_trajectory`` is populated when
        ``cpr_type='bps'``; shape ``(N_sym,)`` SISO or ``(C, N_sym)`` MIMO,
        with one phase estimate per output symbol.

    Warnings
    --------
    On CPU (NumPy backend) block LMS is typically **slower** than
    ``lms(..., backend='numba')``.  Use this function primarily with CuPy
    (GPU) arrays for large MIMO configurations or long sequences.
    """
    if cpr_type is not None and cpr_type != "bps":
        raise ValueError(
            f"block_lms only supports cpr_type='bps' or None. Got {cpr_type!r}. "
            "PLL is not available for block processing."
        )

    logger.info(
        f"Block-LMS: num_taps={num_taps}, block_size={block_size}, "
        f"mu={step_size}, sps={sps}" + (f", cpr={cpr_type}" if cpr_type else "")
    )
    _validate_sps(sps, num_taps)

    samples, xp, _ = dispatch(samples)
    if xp is np:
        logger.warning(
            "block_lms is running on CPU (NumPy). "
            "For CPU workloads lms(..., backend='numba') is typically 2-10x faster. "
            "Move samples to GPU (CuPy) to benefit from block-FFT acceleration."
        )

    was_1d = samples.ndim == 1
    if was_1d:
        samples = samples[np.newaxis, :]

    C = samples.shape[0]
    N = samples.shape[1]
    n_sym = N // sps

    if training_symbols is not None:
        training_symbols, _, _ = dispatch(training_symbols)
        if training_symbols.ndim == 1:
            training_symbols = training_symbols[np.newaxis, :]
        if num_train_symbols is not None:
            training_symbols = training_symbols[..., :num_train_symbols]

    samples, training_symbols, eq_norm = _normalize_inputs(
        samples, training_symbols, sps
    )

    # ── Constellation ─────────────────────────────────────────────────────────
    if modulation is not None and order is not None:
        from .mapping import gray_constellation

        reference_constellation = gray_constellation(
            modulation, order, unipolar=unipolar
        )
        constellation_np = (
            to_device(reference_constellation, "cpu").flatten().astype(np.complex64)
        )
    elif training_symbols is not None:
        train_flat = to_device(training_symbols, "cpu").reshape(-1)
        constellation_np = np.unique(np.round(train_flat, decimals=8)).astype(
            np.complex64
        )
    else:
        raise ValueError("Provide modulation+order or training_symbols for DD slicer.")

    if pmf is not None and modulation is not None and order is not None:
        _pmf_arr = np.asarray(pmf, dtype=np.float64)
        _e_ps = float(
            np.dot(_pmf_arr, np.abs(constellation_np).astype(np.float64) ** 2)
        )
        if _e_ps < 1.0 - 1e-6:
            constellation_np = (constellation_np / np.sqrt(_e_ps)).astype(np.complex64)

    constellation = xp.asarray(constellation_np)  # (M,) on device
    M = len(constellation_np)

    # ── Training alignment ────────────────────────────────────────────────────
    if training_symbols is not None:
        n_train_aligned = min(int(training_symbols.shape[-1]), n_sym)
        if num_train_symbols is not None:
            n_train_aligned = min(n_train_aligned, int(num_train_symbols))
    else:
        n_train_aligned = 0

    # ── Weight initialisation ─────────────────────────────────────────────────
    if w_init is not None:
        w_arr = np.ascontiguousarray(to_device(w_init, "cpu"), dtype=np.complex64)
        w_arr = _validate_w_init(w_arr, C, num_taps)
        h = xp.asarray(w_arr.copy())
    else:
        h = xp.asarray(_init_butterfly_weights_numpy(C, num_taps))  # (C, C, T)

    # ── BPS setup ─────────────────────────────────────────────────────────────
    if cpr_type == "bps":
        symmetry = _cpr_symmetry(modulation, order)
        P = int(cpr_bps_test_phases)
        bps_angles_np = np.linspace(
            0.0, np.pi / 2.0, P, endpoint=False, dtype=np.float32
        )
        bps_phases_neg = xp.asarray(
            np.exp(-1j * bps_angles_np).astype(np.complex64)
        )  # (P,)
        bps_angles = xp.asarray(bps_angles_np)  # (P,)
        quantum = np.float64(2.0 * np.pi / symmetry)
        # Cycle-slip state (CPU scalars — negligible overhead vs GPU compute)
        # Uses global symbol index as x_b so the regression is consistent with
        # the per-symbol Numba kernel and across blocks of varying B.
        _cs_H = min(int(cpr_cycle_slip_history), n_sym)
        cs_buf_x = np.zeros((C, _cs_H), dtype=np.float64)
        cs_buf_y = np.zeros((C, _cs_H), dtype=np.float64)
        cs_buf_ptr = np.zeros(C, dtype=np.int64)
        cs_buf_n = np.zeros(C, dtype=np.int64)
        cs_stats = np.zeros((C, 4), dtype=np.float64)  # Sx, Sy, Sxx, Sxy per channel

    # ── OLS block size ────────────────────────────────────────────────────────
    # fftsize must be >= block_size * sps + num_taps - 1 (linear OLS condition)
    _ols_min = int(block_size) * int(sps) + int(num_taps) - 1
    fftsize = 1 << (_ols_min - 1).bit_length()  # next power of 2

    # ── Padding — matches lms() convention ───────────────────────────────────
    c_tap = num_taps // 2
    pad_total = max(0, n_sym * sps - N + num_taps - 1)
    pad_left = min(c_tap, pad_total)
    pad_right = pad_total - pad_left
    x_padded = xp.pad(samples, ((0, 0), (pad_left, pad_right)))  # (C, N_pad)
    N_padded = x_padded.shape[1]

    # ── Output buffers ────────────────────────────────────────────────────────
    y_all = xp.empty((C, n_sym), dtype=xp.complex64)
    e_all = xp.empty((C, n_sym), dtype=xp.complex64)
    w_hist = (
        xp.empty((n_sym, C, C, num_taps), dtype=xp.complex64) if store_weights else None
    )
    phi_all = xp.zeros((C, n_sym), dtype=xp.float32) if cpr_type == "bps" else None

    n_blocks = (n_sym + block_size - 1) // block_size

    # ── Block loop ────────────────────────────────────────────────────────────
    for b in range(n_blocks):
        b_start = b * block_size
        b_end = min(b_start + block_size, n_sym)
        B = b_end - b_start  # symbols in this block (may be < block_size for last)

        # Input window: x_padded[:, b_start*sps : b_start*sps + fftsize]
        x_start = b_start * sps
        x_win = xp.zeros((C, fftsize), dtype=xp.complex64)
        available = min(fftsize, N_padded - x_start)
        if available > 0:
            x_win[:, :available] = x_padded[:, x_start : x_start + available]

        # ── Forward pass (frequency-domain butterfly) ─────────────────────
        X_fd = xp.fft.fft(x_win, axis=-1)  # (C, F)
        H_fd = xp.fft.fft(h, n=fftsize, axis=-1)  # (C, C, F)
        Y_fd = xp.einsum("ijk,jk->ik", xp.conj(H_fd), X_fd)  # (C, F)
        y_time = xp.fft.ifft(Y_fd, axis=-1)  # (C, F)
        y_block = y_time[:, : B * sps : sps].astype(xp.complex64)  # (C, B)

        # ── BPS phase recovery ────────────────────────────────────────────
        if cpr_type == "bps":
            # rotated: (P, C, B) — all candidate rotations for all block symbols
            rotated = bps_phases_neg[:, None, None] * y_block[None, :, :]
            # min_d2: (P, C, B) — min squared distance to constellation, loop
            # over M to keep peak memory at O(P·C·B) not O(P·C·B·M).
            min_d2 = xp.full((P, C, B), xp.inf, dtype=xp.float32)
            for _m in range(M):
                d2_m = (xp.abs(rotated - constellation[_m]) ** 2).real
                min_d2 = xp.minimum(min_d2, d2_m.astype(xp.float32))

            # Causal sliding-window average of width K along the B (symbol) axis.
            # win_sum[:,:,n] = sum of min_d2[:,:, max(0,n-K+1)..n].
            K = min(int(cpr_bps_block_size), B)
            pad = xp.zeros((P, C, K), dtype=xp.float32)
            cs_d2 = xp.concatenate([pad, min_d2.cumsum(axis=2)], axis=2)
            win_sum = cs_d2[:, :, K:] - cs_d2[:, :, :-K]  # (P, C, B)
            counts = xp.minimum(
                xp.arange(1, B + 1, dtype=xp.float32), xp.float32(K)
            )  # (B,) — denominator handles warmup
            metric = win_sum / counts[None, None, :]  # (P, C, B)

            # Per-symbol argmin over P phases → phi_c (C, B)
            if cpr_bps_joint_channels and C > 1:
                best_k = xp.argmin(metric.sum(axis=1), axis=0)  # (B,)
                phi_c_dev = xp.broadcast_to(
                    bps_angles[best_k][None, :], (C, B)
                ).copy()  # (C, B)
            else:
                best_k = xp.argmin(metric, axis=0)  # (C, B)
                phi_c_dev = bps_angles[best_k]  # (C, B)

            # ── Per-block cycle-slip correction ───────────────────────────
            # A cycle slip is a block-level event (discrete quantum jump).
            # Detecting it once per block and broadcasting the correction to
            # all B symbols avoids a full D→H→D transfer of (C, B) every
            # block: we transfer only C scalars (block-midpoint phase per
            # channel), run C Python iterations, then apply any quantum
            # offset on-device as a vectorized broadcast.
            # When CS is disabled no device transfer occurs at all.
            if cpr_cycle_slip_correction:
                x_b = float(b_start + B // 2)  # global index of block midpoint
                # Transfer only (C,) float scalars from device
                phi_mid_np = to_device(phi_c_dev[:, B // 2], "cpu").astype(np.float64)
                offsets = np.zeros(C, dtype=np.float64)

                for ci in range(C):
                    y_b = phi_mid_np[ci]
                    n_b = int(cs_buf_n[ci])
                    ptr = int(cs_buf_ptr[ci])

                    if n_b == 0:
                        phi_expected = y_b
                    elif n_b < 10:
                        last_pos = (ptr - 1 + _cs_H) % _cs_H
                        phi_expected = cs_buf_y[ci, last_pos]
                    else:
                        sx, sy, sxx, sxy = cs_stats[ci]
                        denom = n_b * sxx - sx * sx
                        if abs(denom) > 1e-30:
                            slope = (n_b * sxy - sx * sy) / denom
                            intercept = (sy - slope * sx) / n_b
                        else:
                            slope = 0.0
                            intercept = sy / n_b
                        phi_expected = slope * x_b + intercept

                    diff = y_b - phi_expected
                    k_slip = int(round(diff / quantum))
                    if abs(diff) > float(cpr_cycle_slip_threshold) and k_slip != 0:
                        offsets[ci] = float(k_slip) * quantum
                        y_b -= offsets[ci]

                    # Update rolling regression buffer with corrected midpoint
                    write_pos = ptr % _cs_H
                    if n_b == _cs_H:
                        ox = cs_buf_x[ci, write_pos]
                        oy = cs_buf_y[ci, write_pos]
                        cs_stats[ci, 0] -= ox
                        cs_stats[ci, 1] -= oy
                        cs_stats[ci, 2] -= ox * ox
                        cs_stats[ci, 3] -= ox * oy
                    cs_buf_x[ci, write_pos] = x_b
                    cs_buf_y[ci, write_pos] = y_b
                    cs_stats[ci, 0] += x_b
                    cs_stats[ci, 1] += y_b
                    cs_stats[ci, 2] += x_b * x_b
                    cs_stats[ci, 3] += x_b * y_b
                    cs_buf_ptr[ci] = ptr + 1
                    if n_b < _cs_H:
                        cs_buf_n[ci] = n_b + 1

                # Apply quantum corrections on device (broadcast over B symbols)
                if np.any(offsets != 0.0):
                    phi_c_dev = (
                        phi_c_dev - xp.asarray(offsets.astype(np.float32))[:, None]
                    )

            phi_c = phi_c_dev  # already on device, float32
            y_rot = y_block * xp.exp(-1j * phi_c.astype(xp.complex64))  # (C, B)
            phi_all[:, b_start:b_end] = phi_c
        else:
            y_rot = y_block

        # ── Error computation (training or DD slicer) ─────────────────────
        e_clean = xp.empty((C, B), dtype=xp.complex64)
        n_train_blk = max(0, min(n_train_aligned - b_start, B))

        if n_train_blk > 0:
            d_train = training_symbols[:, b_start : b_start + n_train_blk]
            e_clean[:, :n_train_blk] = d_train - y_rot[:, :n_train_blk]

        if n_train_blk < B:
            y_dd = y_rot[:, n_train_blk:]
            # Slicer: (C, B-n_train, M) → argmin → (C, B-n_train)
            d2_sl = (xp.abs(y_dd[:, :, None] - constellation[None, None, :]) ** 2).real
            d_dd = constellation[xp.argmin(d2_sl, axis=-1)]
            e_clean[:, n_train_blk:] = d_dd - y_dd

        # ── Store per-symbol outputs ──────────────────────────────────────
        y_all[:, b_start:b_end] = y_rot
        e_all[:, b_start:b_end] = e_clean
        if store_weights:
            w_hist[b_start:b_end] = h[None, :, :, :]

        # ── Back-rotate error to tap plane and compute gradient ───────────
        if cpr_type == "bps":
            e_taps = e_clean * xp.exp(1j * phi_c.astype(xp.complex64))  # (C, B) ✓
        else:
            e_taps = e_clean

        # Scatter e_taps to sample positions within the block window
        e_scatter = xp.zeros((C, fftsize), dtype=xp.complex64)
        e_scatter[:, : B * sps : sps] = e_taps

        # Frequency-domain gradient: dH_fd[i,j,k] = conj(E_fd[i,k]) * X_fd[j,k]
        E_fd = xp.fft.fft(e_scatter, axis=-1)  # (C, F)
        dH_fd = xp.einsum("ik,jk->ijk", xp.conj(E_fd), X_fd)  # (C, C, F)
        dh = xp.fft.ifft(dH_fd, axis=-1)[:, :, :num_taps]  # (C, C, T)

        # Normalise by B so the effective per-symbol step matches step_size
        h = h + (xp.float32(step_size) / xp.float32(B)) * dh

    # ── Pack result ───────────────────────────────────────────────────────────
    if was_1d:
        y_out = y_all[0]
        e_out = e_all[0]
        W_out = h[0, 0]
        w_history = w_hist[:, 0, 0, :] if store_weights else None
        phase_traj = phi_all[0] if cpr_type == "bps" else None
    else:
        y_out = y_all
        e_out = e_all
        W_out = h
        w_history = w_hist if store_weights else None
        phase_traj = phi_all if cpr_type == "bps" else None

    result = EqualizerResult(
        y_hat=y_out,
        weights=W_out,
        error=e_out,
        weights_history=w_history,
        num_train_symbols=n_train_aligned,
        input_norm_factor=eq_norm,
        phase_trajectory=phase_traj,
    )
    return _log_equalizer_exit(
        result, name="Block-LMS", debug_plot=debug_plot, plot_smoothing=plot_smoothing
    )


def build_pilot_ref(
    pilot_symbols: np.ndarray,
    pilot_mask: np.ndarray,
    n_sym: int,
    num_ch: int,
) -> tuple:
    """Build dense pilot reference array and uint8 mask for the hybrid PA kernel.

    Packs sparse pilot symbols into a dense ``(C, n_sym)`` array suitable for
    passing to :func:`cma` or :func:`rde` as ``pilot_ref`` / ``pilot_mask``.
    Data positions are filled with zeros; the mask marks which positions carry
    known reference symbols.

    Parameters
    ----------
    pilot_symbols : (K,) or (C, K) complex64 ndarray
        Known pilot symbols in transmission order (only the K pilot positions,
        no data symbols).  A 1-D array is broadcast to all C channels.
    pilot_mask : (n_sym,) bool ndarray
        ``True`` at the K pilot positions within the equalized body region.
    n_sym : int
        Total number of output symbols (payload + pilots).
    num_ch : int
        Number of receive channels C.

    Returns
    -------
    pilot_ref : (C, n_sym) complex64 ndarray
        Dense reference array — zeros at data positions, pilot symbols at
        pilot positions.
    pilot_mask_u8 : (n_sym,) uint8 ndarray
        ``1`` at pilot positions, ``0`` elsewhere.

    Examples
    --------
    Build the reference from a :class:`~commstools.core.SingleCarrierFrame` and
    pass it directly to :func:`rde`:

    .. code-block:: python

        struct = frame.get_structure_map(unit="symbols", sps=1, include_preamble=False)
        pilot_ref, pilot_mask_u8 = build_pilot_ref(
            pilot_symbols=frame.pilot_symbols,
            pilot_mask=np.asarray(struct["pilots"]),
            n_sym=n_body_symbols,
            num_ch=num_ch,
        )
        result = rde(body_samples, ..., pilot_ref=pilot_ref, pilot_mask=pilot_mask_u8)
    """
    pilot_ref_arr = np.zeros((num_ch, n_sym), dtype=np.complex64)
    mask_uint8 = np.zeros(n_sym, dtype=np.uint8)

    pilot_positions = np.where(pilot_mask)[0]
    n_pilots = len(pilot_positions)

    if pilot_symbols.ndim == 1:
        pilot_symbols = np.broadcast_to(
            pilot_symbols[np.newaxis, :], (num_ch, n_pilots)
        )
    else:
        pilot_symbols = np.asarray(pilot_symbols, dtype=np.complex64)

    for ch in range(num_ch):
        pilot_ref_arr[ch, pilot_positions] = pilot_symbols[ch, :n_pilots]
    mask_uint8[pilot_positions] = 1

    return pilot_ref_arr, mask_uint8


def cma(
    samples: ArrayType,
    num_taps: int = 21,
    sps: int = 2,
    step_size: float = 1e-3,
    modulation: Optional[str] = None,
    order: Optional[int] = None,
    unipolar: bool = False,
    store_weights: bool = False,
    device: Optional[str] = "cpu",
    center_tap: Optional[int] = None,
    backend: str = "numba",
    w_init: Optional[ArrayType] = None,
    pilot_ref: Optional[ArrayType] = None,
    pilot_mask: Optional[np.ndarray] = None,
    pilot_gain_db: float = 0.0,
    pmf: Optional[Any] = None,
    debug_plot: bool = False,
    plot_smoothing: int = 50,
) -> EqualizerResult:
    """
    Constant Modulus Algorithm blind equalizer with butterfly MIMO support.

    CMA minimizes the Godard dispersion criterion and requires no training
    symbols. It is the standard blind equalizer for constant-modulus signals
    (PSK) and near-constant-modulus signals (low-order QAM).

    CMA recovers the signal up to a phase ambiguity. A phase recovery step
    (e.g. Viterbi-Viterbi, pilot-aided) is typically needed after CMA.

    When ``pilot_ref`` and ``pilot_mask`` are both supplied the equalizer
    switches to a **pilot-aided hybrid** mode: the standard Godard CMA error
    is used at data positions while an LMS residual error
    (``pilot_ref - y``) is used at every pilot position.  This resolves the
    phase ambiguity at pilot locations while preserving blind adaptation
    elsewhere.  Build the dense arrays with :func:`build_pilot_ref`.

    Parameters
    ----------
    samples : array_like
        Input signal samples. Shape: ``(N_samples,)`` or ``(C, N_samples)``.
        Typically at 2 samples/symbol for fractionally-spaced equalization.
    num_taps : int, default 21
        Number of equalizer taps per FIR filter.
    sps : int, default 2
        Samples per symbol at the input.  Use ``sps=2`` (T/2-spaced, default)
        for the standard first-stage blind equalization.  ``sps=1`` enables
        symbol-spaced CMA, useful when input is already decimated but phase
        ambiguity resolution is still needed.
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
    w_init : array_like, optional
        Initial tap weights. Shape: ``(C, C, num_taps)`` complex64, or the
        SISO short-hand ``(num_taps,)`` / ``(1, num_taps)`` as returned by
        ``EqualizerResult.weights`` for single-channel equalizers.
        Warm-starts blind equalization from pre-converged weights (e.g. from
        a prior ``lms()`` call on the preamble). Raises ``ValueError`` on
        shape mismatch.
    pilot_ref : (C, N_sym) complex64 array, optional
        Dense pilot reference array — zeros at data positions, known symbols
        at pilot positions.  Build with :func:`build_pilot_ref`.
        Must be provided together with ``pilot_mask``.
    pilot_mask : (N_sym,) uint8 array, optional
        Pilot position mask — ``1`` at pilot positions, ``0`` elsewhere.
        Build with :func:`build_pilot_ref`.
    pilot_gain_db : float, default 0.0
        Pilot boosting in dB relative to payload power, matching
        ``SingleCarrierFrame.pilot_gain_db``.  When non-zero, the received
        signal at pilot positions is attenuated by the inverse of the boost
        factor before the global RMS normalisation.  This prevents boosted
        pilots from inflating the RMS estimate and biasing the Godard
        convergence target at data positions.  Set to ``0.0`` when pilots
        are not boosted.
    pmf : array_like of float, optional
        Probability mass function for PS-QAM.  When provided with ``modulation``
        and ``order``, the Godard R2 is computed for the unit-power PS
        distribution ``{s_m/sqrt(E_PS)}``:
        ``R2 = E_PS[|s_m|^4] / E_PS^2``.  Pilot references are also scaled
        by ``1/sqrt(E_PS)`` so pilot-aided and blind sections converge to the
        same unit-power target.

    Returns
    -------
    EqualizerResult
        Equalized symbols, final weights, CMA error history, and optionally
        weight trajectory.

    Warnings
    --------
    **JAX GPU mode is typically slower than CPU for adaptive equalization.**
    CMA is inherently sequential: each weight update depends on the previous
    weights, so ``lax.scan`` serializes execution even on GPU.  Use
    ``device='cpu'`` for typical SISO sequences, or ``backend='numba'`` for
    CPU-optimal throughput.
    """
    use_pilots = pilot_ref is not None and pilot_mask is not None
    logger.info(
        f"CMA equalizer: num_taps={num_taps}, mu={step_size}, sps={sps}, "
        f"backend={backend}, pilot_aided={use_pilots}, pilot_gain_db={pilot_gain_db}"
    )
    if sps > 1:
        logger.warning(
            "CMA output y_hat is at 1 SPS (symbol rate). "
            "Update sampling_rate = symbol_rate after applying this equalizer."
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

    # Compute R2 and PS-QAM scale factor from the Godard constellation.
    _c_ps = None  # 1/sqrt(E_PS) scale factor; None for uniform modulation
    if modulation is not None and order is not None:
        from .mapping import gray_constellation

        const = gray_constellation(modulation, order, unipolar=unipolar)
        if pmf is not None:
            # PS-QAM: R2 for the unit-power distribution {s_m/sqrt(E_PS)}:
            #   R2 = E_PS[|s_m/sqrt(E_PS)|^4] / E_PS[|s_m/sqrt(E_PS)|^2]
            #      = (E_PS[|s_m|^4] / E_PS^2) / 1
            #      = E_PS[|s_m|^4] / E_PS^2
            _pmf_arr = np.asarray(pmf, dtype=np.float64)
            _abs2 = np.abs(const) ** 2
            _e_ps = float(np.dot(_pmf_arr, _abs2))
            r2 = float(np.dot(_pmf_arr, np.abs(const) ** 4)) / (_e_ps**2)
            if _e_ps < 1.0 - 1e-6:
                _c_ps = np.float32(1.0 / np.sqrt(_e_ps))
            logger.debug(
                f"CMA R2 (PS-QAM pmf-weighted, {modulation.upper()}-{order}): {r2:.4f}"
            )
        else:
            r2 = float(np.mean(np.abs(const) ** 4) / np.mean(np.abs(const) ** 2))
            logger.debug(f"CMA R2 from {modulation.upper()}-{order}: {r2:.4f}")
    else:
        r2 = 1.0

    n_sym = n_samples // stride

    c_tap = center_tap if center_tap is not None else num_taps // 2
    pad_total = max(0, n_sym * stride - n_samples + num_taps - 1)
    pad_left = min(c_tap, pad_total)
    pad_right = pad_total - pad_left

    if backend == "numba":
        numba = _get_numba()
        if numba is None:
            raise ImportError("Numba is required for backend='numba'.")

        samples_np = np.ascontiguousarray(to_device(samples, "cpu"), dtype=np.complex64)
        # Deboost pilot positions before global normalisation so boosted pilots
        # don't inflate the RMS estimate and bias the Godard convergence target.
        if use_pilots and pilot_gain_db != 0.0:
            _amp = np.float32(10.0 ** (pilot_gain_db / 20.0))
            _smask = np.repeat(pilot_mask.astype(bool), stride)  # (N_samples,)
            samples_np[..., _smask] /= _amp
        # RMS-normalize samples to unit symbol-rate power (CMA has no training)
        samples_np, _, eq_norm = _normalize_inputs(samples_np, None, sps)

        x_np = (
            np.pad(samples_np, ((0, 0), (pad_left, pad_right)))
            if not was_1d
            else np.pad(samples_np, (pad_left, pad_right))
        )
        x_np = np.ascontiguousarray(x_np)
        if was_1d:
            x_np = x_np[np.newaxis, :]

        if w_init is not None:
            w_arr = np.ascontiguousarray(to_device(w_init, "cpu"), dtype=np.complex64)
            w_arr = _validate_w_init(w_arr, num_ch, num_taps)
            W = w_arr.copy()
        else:
            W = _init_butterfly_weights_numpy(num_ch, num_taps, center_tap=center_tap)
        y_out = np.empty((n_sym, num_ch), dtype=np.complex64)
        e_out = np.empty((n_sym, num_ch), dtype=np.complex64)
        w_hist_buf = (
            np.empty((n_sym, num_ch, num_ch, num_taps), dtype=np.complex64)
            if store_weights
            else np.empty((1, num_ch, num_ch, num_taps), dtype=np.complex64)
        )
        if use_pilots:
            pref = np.ascontiguousarray(to_device(pilot_ref, "cpu"), dtype=np.complex64)
            if _c_ps is not None:
                pref = (pref * _c_ps).astype(np.complex64)
            pmask = np.ascontiguousarray(pilot_mask, dtype=np.uint8)
            _get_numba_pa_cma()(
                x_np,
                W,
                np.float32(step_size),
                np.float32(r2),
                stride,
                store_weights,
                y_out,
                e_out,
                w_hist_buf,
                pref,
                pmask,
            )
        else:
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
        return _log_equalizer_exit(
            _unpack_result_numpy(
                y_out,
                e_out,
                W,
                w_hist_buf,
                was_1d,
                store_weights,
                n_sym=None,
                xp=xp,
                input_norm_factor=eq_norm,
            ),
            name="CMA" if not use_pilots else "CMA(PA)",
            debug_plot=debug_plot,
            check_convergence=True,
            plot_smoothing=plot_smoothing,
        )

    # JAX backend
    jax, jnp, _ = _get_jax()
    if jax is None:
        raise ImportError("JAX is required for backend='jax'.")

    # Deboost pilot positions before global normalisation so boosted pilots
    # don't inflate the RMS estimate and bias the Godard convergence target.
    if use_pilots and pilot_gain_db != 0.0:
        _amp = float(10.0 ** (pilot_gain_db / 20.0))
        _smask = xp.asarray(np.repeat(pilot_mask.astype(bool), stride))
        samples = samples.copy()
        samples[..., _smask] /= xp.float32(_amp)
    # RMS-normalize samples to unit symbol-rate power (CMA has no training)
    samples, _, eq_norm = _normalize_inputs(samples, None, sps)

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

    if w_init is not None:
        w_arr = np.ascontiguousarray(to_device(w_init, "cpu"), dtype=np.complex64)
        w_arr = _validate_w_init(w_arr, num_ch, num_taps)
        W_jax = to_jax(w_arr, device=platform)
    else:
        W_jax = _init_butterfly_weights_jax(
            num_ch, num_taps, jnp, center_tap=center_tap
        )
        W_jax = to_jax(W_jax, device=platform)
    mu_jax = to_jax(jnp.float32(step_size), device=platform)
    r2_jax = to_jax(jnp.float32(r2), device=platform)

    if use_pilots:
        pref_np = np.ascontiguousarray(to_device(pilot_ref, "cpu"), dtype=np.complex64)
        if _c_ps is not None:
            pref_np = (pref_np * _c_ps).astype(np.complex64)
        pmask_np = np.ascontiguousarray(pilot_mask, dtype=np.uint8)
        # pilot_ref: (C, n_sym) → (n_sym, C) for scan xs
        pref_jax = to_jax(pref_np.T, device=platform)
        pmask_jax = to_jax(pmask_np.astype(bool), device=platform)
        scan_fn = _get_jax_pa_cma(num_taps, stride, num_ch)
        y_jax, e_jax, W_jax, wh_jax = scan_fn(
            x_jax, W_jax, mu_jax, r2_jax, pref_jax, pmask_jax, n_sym
        )
    else:
        scan_fn = _get_jax_cma(num_taps, stride, num_ch)
        y_jax, e_jax, W_jax, wh_jax = scan_fn(x_jax, W_jax, mu_jax, r2_jax, n_sym)
    return _log_equalizer_exit(
        _unpack_result_jax(
            y_jax,
            e_jax,
            W_jax,
            wh_jax,
            was_1d,
            store_weights,
            n_sym=None,
            xp=xp,
            input_norm_factor=eq_norm,
        ),
        name="CMA" if not use_pilots else "CMA(PA)",
        debug_plot=debug_plot,
        check_convergence=True,
    )


def rde(
    samples: ArrayType,
    num_taps: int = 21,
    sps: int = 2,
    step_size: float = 1e-3,
    modulation: Optional[str] = None,
    order: Optional[int] = None,
    unipolar: bool = False,
    store_weights: bool = False,
    device: Optional[str] = "cpu",
    center_tap: Optional[int] = None,
    backend: str = "numba",
    w_init: Optional[ArrayType] = None,
    pilot_ref: Optional[ArrayType] = None,
    pilot_mask: Optional[np.ndarray] = None,
    pilot_gain_db: float = 0.0,
    pmf: Optional[Any] = None,
    debug_plot: bool = False,
    plot_smoothing: int = 50,
) -> EqualizerResult:
    """
    Radius Directed Equalizer (RDE) — blind equalizer for multi-ring constellations.

    RDE is a CMA variant that replaces the single Godard dispersion radius with
    per-symbol radius selection from the set of unique constellation ring radii.
    For each output sample ``y[n]``, the target radius ``R_d`` is chosen as the
    magnitude of the nearest constellation ring::

        R_d[n]  = argmin_r  | r − |y[n]| |     r ∈ { |c| : c ∈ constellation }
        e[n]    = y[n] * ( |y[n]|² − R_d[n]² )
        W      -= μ · conj(e[n]) ⊗ x[n]

    This corrects CMA's fundamental weakness on higher-order QAM: CMA forces
    all symbols toward a single average circle, severely degrading convergence
    when the constellation spans multiple rings (e.g. inner, middle, outer rings
    of 16-QAM).  RDE instead drives each symbol toward its *nearest* ring,
    producing a signal-quality surface that matches the true constellation
    geometry.

    Like CMA, RDE is fully blind (no training symbols) and recovers the channel
    up to a **phase ambiguity**.  A carrier-phase recovery step (Viterbi-Viterbi,
    pilot-aided rotation, etc.) is typically needed afterwards.

    Parameters
    ----------
    samples : array_like
        Input signal samples. Shape: ``(N_samples,)`` or ``(C, N_samples)``.
        Typically at 2 samples/symbol for fractionally-spaced equalization.
    num_taps : int, default 21
        Number of equalizer taps per FIR filter.
    sps : int, default 2
        Samples per symbol at the input.  Use ``sps=2`` (T/2-spaced, default)
        for standard blind equalization.  ``sps=1`` is accepted.
    step_size : float, default 1e-3
        RDE step size (mu). Same non-convex gradient geometry as CMA; use a
        fixed step in the range 1e-5 to 1e-3 for stability.
    modulation : str, optional
        Modulation type for constellation construction (``"psk"``, ``"qam"``).
        Required to extract unique ring radii.  If ``None``, falls back to a
        single unit radius (identical to CMA with ``R²=1``).
    order : int, optional
        Modulation order (e.g. 4, 16, 64).
    unipolar : bool, default False
        Use unipolar constellation for radius extraction.
    store_weights : bool, default False
        If True, stores weight trajectory in ``result.weights_history``.
    device : str, optional
        Target JAX device (``'cpu'``, ``'gpu'``). Ignored for ``backend='numba'``.
    center_tap : int, optional
        Index of the center tap. Defaults to ``num_taps // 2``.
    backend : str, default 'numba'
        ``'numba'`` uses Numba ``@njit``; ``'jax'`` uses ``jax.lax.scan``.
    w_init : array_like, optional
        Initial tap weights. Shape: ``(C, C, num_taps)`` complex64, or the
        SISO short-hand ``(num_taps,)`` / ``(1, num_taps)`` as returned by
        ``EqualizerResult.weights`` for single-channel equalizers.
        Warm-starts blind equalization from pre-converged weights (e.g. from
        a prior ``lms()`` or ``cma()`` call). Raises ``ValueError`` on shape
        mismatch.
    pilot_ref : (C, N_sym) complex64 array, optional
        Dense pilot reference array — zeros at data positions, known symbols
        at pilot positions.  Build with :func:`build_pilot_ref`.
        Must be provided together with ``pilot_mask``.
    pilot_mask : (N_sym,) uint8 array, optional
        Pilot position mask — ``1`` at pilot positions, ``0`` elsewhere.
        Build with :func:`build_pilot_ref`.
    pilot_gain_db : float, default 0.0
        Pilot boosting in dB relative to payload power, matching
        ``SingleCarrierFrame.pilot_gain_db``.  When non-zero, the received
        signal at pilot positions is attenuated by the inverse of the boost
        factor before the global RMS normalisation.  This prevents boosted
        pilots from inflating the RMS estimate and biasing the ring-radius
        convergence targets at data positions.  Set to ``0.0`` when pilots
        are not boosted.
    pmf : array_like of float, optional
        Probability mass function for PS-QAM.  When provided with ``modulation``
        and ``order``, the ring radii are scaled by ``1/sqrt(E_PS)`` to target
        the unit-power constellation ``{|s_m|/sqrt(E_PS)}``.  Pilot references
        are also scaled accordingly.  Requires ``modulation`` and ``order``.

    Returns
    -------
    EqualizerResult
        Equalized symbols, final weights, RDE error history, and optionally
        weight trajectory.

    Notes
    -----
    **Why RDE outperforms CMA on high-order QAM:**

    For 16-QAM the Godard radius ``R² = E[|s|⁴]/E[|s|²] ≈ 1.32`` (normalized).
    This single target is a poor proxy for the three distinct rings at
    ``|c| ≈ {0.45, 1.00, 1.34}`` (normalized unit-average-power 16-QAM).
    CMA pulls inner-ring symbols outward and outer-ring symbols inward,
    creating a persistent gradient that opposes correct convergence.
    RDE eliminates this bias entirely: each symbol is only attracted to its
    own ring, so the steady-state gradient vanishes at the correct solution.

    **Phase ambiguity:** Both CMA and RDE share the same 90°-symmetric cost
    surface for QAM/PSK.  Use a phase recovery algorithm after blind equalization.

    **GPU note:** RDE is inherently sequential (each weight update depends on
    previous weights), so ``lax.scan`` serializes execution even on GPU.
    Use ``device='cpu'`` for typical SISO sequences, or ``backend='numba'``
    for CPU-optimal throughput.
    """
    use_pilots = pilot_ref is not None and pilot_mask is not None
    logger.info(
        f"RDE equalizer: num_taps={num_taps}, mu={step_size}, sps={sps}, "
        f"backend={backend}, pilot_aided={use_pilots}, pilot_gain_db={pilot_gain_db}"
    )
    if sps > 1:
        logger.warning(
            "RDE output y_hat is at 1 SPS (symbol rate). "
            "Update sampling_rate = symbol_rate after applying this equalizer."
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

    # Compute unique ring radii from constellation.
    # For constant-modulus signals (PSK) this degenerates to a single radius,
    # making RDE identical to CMA.
    _c_ps = None  # 1/sqrt(E_PS) scale factor; None for uniform modulation
    if modulation is not None and order is not None:
        from .mapping import gray_constellation

        const = gray_constellation(modulation, order, unipolar=unipolar)
        raw_radii = np.abs(const).astype(np.float32)
        if pmf is not None:
            # PS-QAM: scale radii to unit-power targets {|s_m|/sqrt(E_PS)}
            _pmf_arr = np.asarray(pmf, dtype=np.float64)
            _e_ps = float(np.dot(_pmf_arr, raw_radii.astype(np.float64) ** 2))
            if _e_ps < 1.0 - 1e-6:
                _c_ps = np.float32(1.0 / np.sqrt(_e_ps))
                raw_radii = (raw_radii * _c_ps).astype(np.float32)
        # Round to 6 significant digits to merge numerically identical radii
        radii = np.unique(np.round(raw_radii, 6))
        logger.debug(
            f"RDE radii from {modulation.upper()}-{order}: "
            + ", ".join(f"{r:.4f}" for r in radii)
        )
    else:
        radii = np.array([1.0], dtype=np.float32)
        logger.debug("RDE: no modulation provided, using single unit radius (≡ CMA)")

    n_sym = n_samples // stride
    num_radii = len(radii)

    c_tap = center_tap if center_tap is not None else num_taps // 2
    pad_total = max(0, n_sym * stride - n_samples + num_taps - 1)
    pad_left = min(c_tap, pad_total)
    pad_right = pad_total - pad_left

    if backend == "numba":
        numba = _get_numba()
        if numba is None:
            raise ImportError("Numba is required for backend='numba'.")

        samples_np = np.ascontiguousarray(to_device(samples, "cpu"), dtype=np.complex64)
        # Deboost pilot positions before global normalisation so boosted pilots
        # don't inflate the RMS estimate and bias the ring-radius convergence targets.
        if use_pilots and pilot_gain_db != 0.0:
            _amp = np.float32(10.0 ** (pilot_gain_db / 20.0))
            _smask = np.repeat(pilot_mask.astype(bool), stride)  # (N_samples,)
            samples_np[..., _smask] /= _amp
        samples_np, _, eq_norm = _normalize_inputs(samples_np, None, sps)

        x_np = (
            np.pad(samples_np, ((0, 0), (pad_left, pad_right)))
            if not was_1d
            else np.pad(samples_np, (pad_left, pad_right))
        )
        x_np = np.ascontiguousarray(x_np)
        if was_1d:
            x_np = x_np[np.newaxis, :]

        # Normalize radii to match the unit-power-normalized samples
        # (constellation is unit-average-power after gray_constellation)
        radii_np = np.ascontiguousarray(radii, dtype=np.float32)

        if w_init is not None:
            w_arr = np.ascontiguousarray(to_device(w_init, "cpu"), dtype=np.complex64)
            w_arr = _validate_w_init(w_arr, num_ch, num_taps)
            W = w_arr.copy()
        else:
            W = _init_butterfly_weights_numpy(num_ch, num_taps, center_tap=center_tap)
        y_out = np.empty((n_sym, num_ch), dtype=np.complex64)
        e_out = np.empty((n_sym, num_ch), dtype=np.complex64)
        w_hist_buf = (
            np.empty((n_sym, num_ch, num_ch, num_taps), dtype=np.complex64)
            if store_weights
            else np.empty((1, num_ch, num_ch, num_taps), dtype=np.complex64)
        )
        if use_pilots:
            pref = np.ascontiguousarray(to_device(pilot_ref, "cpu"), dtype=np.complex64)
            if _c_ps is not None:
                pref = (pref * _c_ps).astype(np.complex64)
            pmask = np.ascontiguousarray(pilot_mask, dtype=np.uint8)
            _get_numba_pa_rde()(
                x_np,
                W,
                np.float32(step_size),
                radii_np,
                stride,
                store_weights,
                y_out,
                e_out,
                w_hist_buf,
                pref,
                pmask,
            )
        else:
            _get_numba_rde()(
                x_np,
                W,
                np.float32(step_size),
                radii_np,
                stride,
                store_weights,
                y_out,
                e_out,
                w_hist_buf,
            )
        return _log_equalizer_exit(
            _unpack_result_numpy(
                y_out,
                e_out,
                W,
                w_hist_buf,
                was_1d,
                store_weights,
                n_sym=None,
                xp=xp,
                input_norm_factor=eq_norm,
            ),
            name="RDE" if not use_pilots else "RDE(PA)",
            debug_plot=debug_plot,
            check_convergence=True,
            plot_smoothing=plot_smoothing,
        )

    # JAX backend
    jax, jnp, _ = _get_jax()
    if jax is None:
        raise ImportError("JAX is required for backend='jax'.")

    # Deboost pilot positions before global normalisation so boosted pilots
    # don't inflate the RMS estimate and bias the ring-radius convergence targets.
    if use_pilots and pilot_gain_db != 0.0:
        _amp = float(10.0 ** (pilot_gain_db / 20.0))
        _smask = xp.asarray(np.repeat(pilot_mask.astype(bool), stride))
        samples = samples.copy()
        samples[..., _smask] /= xp.float32(_amp)
    samples, _, eq_norm = _normalize_inputs(samples, None, sps)

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

    if w_init is not None:
        w_arr = np.ascontiguousarray(to_device(w_init, "cpu"), dtype=np.complex64)
        w_arr = _validate_w_init(w_arr, num_ch, num_taps)
        W_jax = to_jax(w_arr, device=platform)
    else:
        W_jax = _init_butterfly_weights_jax(
            num_ch, num_taps, jnp, center_tap=center_tap
        )
        W_jax = to_jax(W_jax, device=platform)
    mu_jax = to_jax(jnp.float32(step_size), device=platform)
    radii_jax = to_jax(jnp.asarray(radii, dtype=jnp.float32), device=platform)

    if use_pilots:
        pref_np = np.ascontiguousarray(to_device(pilot_ref, "cpu"), dtype=np.complex64)
        if _c_ps is not None:
            pref_np = (pref_np * _c_ps).astype(np.complex64)
        pmask_np = np.ascontiguousarray(pilot_mask, dtype=np.uint8)
        # pilot_ref: (C, n_sym) → (n_sym, C) for scan xs
        pref_jax = to_jax(pref_np.T, device=platform)
        pmask_jax = to_jax(pmask_np.astype(bool), device=platform)
        scan_fn = _get_jax_pa_rde(num_taps, stride, num_radii, num_ch)
        y_jax, e_jax, W_jax, wh_jax = scan_fn(
            x_jax, W_jax, mu_jax, radii_jax, pref_jax, pmask_jax, n_sym
        )
    else:
        scan_fn = _get_jax_rde(num_taps, stride, num_radii, num_ch)
        y_jax, e_jax, W_jax, wh_jax = scan_fn(x_jax, W_jax, mu_jax, radii_jax, n_sym)
    return _log_equalizer_exit(
        _unpack_result_jax(
            y_jax,
            e_jax,
            W_jax,
            wh_jax,
            was_1d,
            store_weights,
            n_sym=None,
            xp=xp,
            input_norm_factor=eq_norm,
        ),
        name="RDE" if not use_pilots else "RDE(PA)",
        debug_plot=debug_plot,
        check_convergence=True,
    )


# -----------------------------------------------------------------------------
# BLOCK equalization
# -----------------------------------------------------------------------------


def zf_equalizer(
    samples: ArrayType,
    channel_estimate: ArrayType,
    noise_variance: float = 0.0,
    debug_plot: bool = False,
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
    N_fft = max(1024, 1 << (max(1, 4 * L) - 1).bit_length())

    logger.debug(f"ZF/MMSE internals: N={N}, L={L}, num_ch={num_ch}, N_fft={N_fft}")

    # --- Shared OLS forward pass: pad → stride_tricks → batch FFT ---
    Y, meta = _ols_forward(samples, N_fft)  # Y: (num_ch, num_blocks, N_fft)

    if siso_channel:
        # SISO: scalar frequency-domain ZF/MMSE inversion.
        # channel_estimate was reshaped to (1,1,L) for was_1d; flatten to 1D.
        H = xp.fft.fft(channel_estimate.reshape(-1), n=N_fft)
        W = xp.conj(H) / (xp.abs(H) ** 2 + reg)
        X_hat_f = Y * W
    else:
        # MIMO: per-bin (C×C) matrix inversion — cannot reduce to scalar multiply.
        H_f = xp.fft.fft(channel_estimate, n=N_fft, axis=-1)
        Hk = xp.transpose(H_f, (2, 0, 1))  # (N_fft, C_rx, C_tx)
        Hk_H = xp.conj(xp.transpose(Hk, (0, 2, 1)))  # (N_fft, C_tx, C_rx)
        HHh = Hk @ Hk_H  # (N_fft, C_rx, C_rx)

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

        Wk = Hk_H @ inv_term  # (N_fft, C_tx, C_rx)

        # Vectorized batch matrix multiplication across frequency bins and blocks.
        # Uses explicit batched GEMM instead of einsum for better GPU utilization.
        Y_t = xp.transpose(Y, (2, 0, 1))  # (N_fft, num_ch, num_blocks)
        X_hat_k = Wk @ Y_t  # (N_fft, C_tx, C_rx) @ (N_fft, C_rx, num_blocks)
        X_hat_f = xp.transpose(X_hat_k, (1, 2, 0))  # (num_ch, num_blocks, N_fft)

    # --- Shared OLS backward pass: batch IFFT → symmetric discard → reshape ---
    out = _ols_backward(X_hat_f, meta)

    if debug_plot:
        from . import plotting as _plotting  # lazy import avoids circular dep

        _plotting.zf_equalizer_response(
            channel_estimate=to_device(channel_estimate, "cpu"),
            noise_variance=noise_variance,
            show=True,
        )

    return out[0] if was_1d else out


def apply_taps(
    samples: ArrayType,
    weights: ArrayType,
    sps: int = 2,
    normalize: bool = True,
) -> ArrayType:
    """Apply frozen equalizer taps to a signal (inference pass, no weight updates).

    Performs the butterfly FIR forward pass using pre-converged weights,
    decimating by ``sps`` to produce one output symbol per ``sps`` input samples.
    Suitable for reusing frozen taps from a prior equalizer run on a new signal
    without re-running adaptation.

    The forward pass implements:

    .. math::

        y[i, n] = \\sum_j \\sum_t W^*[i,j,t] \\cdot x[j,\\, n \\cdot sps + t]

    which is the same inner computation as the Numba/JAX adaptive-equalizer
    kernels, fully vectorized over ``n`` via a single batched ``einsum``.

    Parameters
    ----------
    samples : array_like
        Input samples. Shape: ``(N_samples,)`` for SISO or
        ``(C, N_samples)`` for MIMO. Typically at ``sps`` samples/symbol.
    weights : array_like
        Frozen tap weights, typically ``EqualizerResult.weights`` from a
        prior equalizer run. Shape: ``(num_taps,)`` for SISO or
        ``(C, C, num_taps)`` for MIMO butterfly.
    sps : int, default 2
        Samples per symbol. Output length is ``N_samples // sps``.
        Unlike the adaptive equalizers, any ``sps >= 1`` is accepted.
    normalize : bool, default True
        If ``True``, normalize ``samples`` to unit symbol power before
        filtering (same pre-processing as the adaptive equalizers via
        ``_normalize_inputs``). Set to ``False`` if the caller has already
        scaled the input consistently with the original training run.

    Returns
    -------
    array_like
        Equalized symbols. Shape: ``(N_sym,)`` for SISO or
        ``(C, N_sym)`` for MIMO, where ``N_sym = N_samples // sps``.
        Resides on the same backend as ``samples``.

    Examples
    --------
    Freeze taps from a training run and apply to a new capture::

        result = equalization.lms(train_signal, training_symbols, num_taps=31)
        y = equalization.apply_taps(new_signal, result.weights)
    """
    samples, xp, _ = dispatch(samples)
    weights = xp.asarray(weights)

    was_1d = samples.ndim == 1

    # Promote SISO → MIMO shapes for uniform butterfly code path
    if was_1d:
        samples = samples[None, :]  # (N,) → (1, N)
    if weights.ndim == 1:
        weights = weights[None, None, :]  # (T,) → (1, 1, T)

    C, N = samples.shape
    num_taps = weights.shape[-1]
    n_sym = N // sps

    if normalize:
        samples, _, _ = _normalize_inputs(samples, None, sps)

    # Pad left by center tap so window[0] is center-aligned with sample 0,
    # pad right to fill the last window completely — mirrors LMS pre-processing.
    c_tap = num_taps // 2
    pad_left = c_tap
    pad_right = n_sym * sps - N + num_taps - 1 - pad_left
    samples_padded = xp.pad(samples, ((0, 0), (pad_left, pad_right)))

    # Zero-copy sliding-window view: (C, N_sym, num_taps)
    # windows[c, n, t] = samples_padded[c, n*sps + t]
    s0, s1 = samples_padded.strides
    windows = xp.lib.stride_tricks.as_strided(
        samples_padded,
        shape=(C, n_sym, num_taps),
        strides=(s0, sps * s1, s1),
    )

    # Butterfly forward pass — single batched GEMM (dispatches to cuBLAS on GPU)
    # y[i, n] = Σ_j Σ_t conj(W[i,j,t]) * windows[j, n, t]
    y = xp.einsum("ijt,jnt->in", xp.conj(weights), windows)  # (C, N_sym)

    return y[0] if was_1d else y
