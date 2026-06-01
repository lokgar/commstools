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
class CPRState:
    """Carrier-phase-recovery state for warm-starting across equalizer calls.

    All arrays are CPU NumPy regardless of the equalizer backend — CPR state is
    small and must survive device resets.  Do not convert to CuPy.

    Used/produced by ``lms()``, ``rls()``, and ``block_lms()`` when
    ``cpr_type`` is not None.  Pass as ``cpr_state=result.cpr_state`` to the
    next call to continue phase tracking without a re-lock transient.
    """

    # PLL state (lms / rls with cpr_type='pll' or 'bps')
    pll_phi: Optional[np.ndarray] = None  # (C,) float64
    pll_freq: Optional[np.ndarray] = None  # (C,) float64

    # BPS cross-block unwrap state (block_lms with cpr_type='bps')
    bps_prev4: Optional[np.ndarray] = None  # (C,) float64
    bps_offset4: Optional[np.ndarray] = None  # (C,) float64
    bps_d2_hist: Optional[np.ndarray] = None  # (P, C, K-1) float32 — CPU copy

    # Cycle-slip regression state (all CPR modes)
    cs_buf_x: Optional[np.ndarray] = None  # (C, H) float64
    cs_buf_y: Optional[np.ndarray] = None  # (C, H) float64
    cs_buf_ptr: Optional[np.ndarray] = None  # (C,) int64
    cs_buf_n: Optional[np.ndarray] = None  # (C,) int64
    cs_stats: Optional[np.ndarray] = None  # (C, 4) float64

    # JAX-specific BPS buffer state (JAX backend only; None for Numba)
    jax_bps_buf: Optional[np.ndarray] = None  # (KB, C) complex64
    jax_bps_buf_ptr: Optional[int] = None  # scalar int32

    # Identity tags — used to validate shape compatibility on warm-start
    cpr_type: Optional[str] = None
    num_ch: int = 0
    symmetry: int = 4
    bps_P: int = 0  # number of BPS test phases
    bps_K: int = 0  # BPS block size
    cs_H: int = 0  # cycle-slip history length


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
    cpr_state: Optional["CPRState"] = None


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


def _sq_qam_slicer_params(constellation_np):
    """Return (sq_side, lev_min_f32, d_grid_f32) for O(1) square-QAM slicing.

    For a square M-QAM constellation the nearest symbol can be found by
    independently snapping the real and imaginary components to the nearest
    level — no O(M) search required.  Returns sq_side=0 for non-square
    constellations (PSK, PAM, shaped QAM) to indicate the O(M) fallback
    should be used.
    """
    M = len(constellation_np)
    sq_side = int(M**0.5)
    if sq_side * sq_side != M:
        return 0, np.float32(0.0), np.float32(1.0)
    levels = np.unique(np.round(constellation_np.real, 6))
    if len(levels) != sq_side:
        return 0, np.float32(0.0), np.float32(1.0)
    lev_min = np.float32(levels[0])
    d_grid = np.float32(levels[1] - levels[0]) if sq_side > 1 else np.float32(1.0)
    return sq_side, lev_min, d_grid


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
            sq_lev_min,
            sq_d_grid,
            sq_side,
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
                    acc_re = 0.0
                    acc_im = 0.0
                    for j in range(C):
                        for t in range(num_taps):
                            w = W[i, j, t]
                            x = X_wins[j, t]
                            acc_re += np.float64(w.real) * np.float64(
                                x.real
                            ) + np.float64(w.imag) * np.float64(x.imag)
                            acc_im += np.float64(w.real) * np.float64(
                                x.imag
                            ) - np.float64(w.imag) * np.float64(x.real)
                    y[i] = acc_re + 1j * acc_im
                    y_out[idx, i] = acc_re + 1j * acc_im

                # Desired symbol: training or decision-directed slicer
                for i in range(C):
                    if idx < n_train:
                        d_i = training[i, idx]
                    elif sq_side > np.int32(0):
                        ir = np.int32(np.round((y[i].real - sq_lev_min) / sq_d_grid))
                        if ir < np.int32(0):
                            ir = np.int32(0)
                        if ir >= sq_side:
                            ir = sq_side - np.int32(1)
                        ii = np.int32(np.round((y[i].imag - sq_lev_min) / sq_d_grid))
                        if ii < np.int32(0):
                            ii = np.int32(0)
                        if ii >= sq_side:
                            ii = sq_side - np.int32(1)
                        nr = sq_lev_min + np.float32(ir) * sq_d_grid
                        ni = sq_lev_min + np.float32(ii) * sq_d_grid
                        d_i = np.complex64(nr + ni * np.complex64(1j))
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
            sq_lev_min,
            sq_d_grid,
            sq_side,
        ):
            # x_padded      : (C, N_pad)                  complex64
            # training      : (C, N_sym)                   complex64
            # constellation : (M,)                         complex64
            # W             : (C, C, num_taps)              complex64 — in-place
            # P             : (C*num_taps, C*num_taps)      complex128 — in-place
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

            x_bar = np.empty(N, dtype=np.complex128)
            Px = np.empty(N, dtype=np.complex128)
            xH_P = np.empty(N, dtype=np.complex128)
            k = np.empty(N, dtype=np.complex128)
            y = np.empty(C, dtype=np.complex64)
            e = np.empty(C, dtype=np.complex64)

            lam_f64 = np.float64(lam)
            leak_term = np.float32(1.0) - np.float32(leakage)

            for idx in range(n_sym):
                sample_idx = idx * stride

                # Flatten regressor directly from input
                for j in range(C):
                    for t in range(num_taps):
                        x_bar[j * num_taps + t] = x_padded[j, sample_idx + t]

                # Butterfly forward pass
                for i in range(C):
                    acc_re = 0.0
                    acc_im = 0.0
                    for j in range(C):
                        for t in range(num_taps):
                            w = W[i, j, t]
                            x = x_bar[j * num_taps + t]
                            acc_re += np.float64(w.real) * np.float64(
                                x.real
                            ) + np.float64(w.imag) * np.float64(x.imag)
                            acc_im += np.float64(w.real) * np.float64(
                                x.imag
                            ) - np.float64(w.imag) * np.float64(x.real)
                    y[i] = acc_re + 1j * acc_im
                    y_out[idx, i] = acc_re + 1j * acc_im

                # Desired and error
                for i in range(C):
                    if idx < n_train:
                        d_i = training[i, idx]
                    elif sq_side > np.int32(0):
                        ir = np.int32(np.round((y[i].real - sq_lev_min) / sq_d_grid))
                        if ir < np.int32(0):
                            ir = np.int32(0)
                        if ir >= sq_side:
                            ir = sq_side - np.int32(1)
                        ii = np.int32(np.round((y[i].imag - sq_lev_min) / sq_d_grid))
                        if ii < np.int32(0):
                            ii = np.int32(0)
                        if ii >= sq_side:
                            ii = sq_side - np.int32(1)
                        nr = sq_lev_min + np.float32(ir) * sq_d_grid
                        ni = sq_lev_min + np.float32(ii) * sq_d_grid
                        d_i = np.complex64(nr + ni * np.complex64(1j))
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

                # Kalman gain: Px = P @ x_bar  (in-place, no heap allocation)
                for ii in range(N):
                    acc_re = 0.0
                    acc_im = 0.0
                    for jj in range(N):
                        p_val = P[ii, jj]
                        x_val = x_bar[jj]
                        acc_re += p_val.real * x_val.real - p_val.imag * x_val.imag
                        acc_im += p_val.real * x_val.imag + p_val.imag * x_val.real
                    Px[ii] = acc_re + 1j * acc_im

                # denom = λ + real(conj(x_bar) · Px)
                denom = lam_f64
                for jj in range(N):
                    denom = denom + (np.conj(x_bar[jj]) * Px[jj]).real

                # k = Px / denom
                inv_denom = np.float64(1.0) / denom
                for ii in range(N):
                    k[ii] = Px[ii] * inv_denom

                if idx < n_update_halt:
                    # Leaky W update: W[i,j,t] = (1−γ)W[i,j,t] + k[j*T+t]*conj(e[i])
                    for i in range(C):
                        ce_i = np.conj(e[i])
                        for j in range(C):
                            for t in range(num_taps):
                                W[i, j, t] = np.complex64(
                                    leak_term * W[i, j, t] + k[j * num_taps + t] * ce_i
                                )

                    # Exploit Hermitian symmetry P = P^H: for each j, (x^H P)[j] = conj((Px)[j]).
                    # Reuse the already-computed Px to avoid a second O(N²) mat-vec.
                    for jj in range(N):
                        xH_P[jj] = np.conj(Px[jj])

                    # Riccati + Hermitian re-symmetrization in one pass (upper
                    # triangle only). Halves memory traffic vs. two separate
                    # O(N²) loops; correctness is identical.
                    for ii in range(N):
                        for jj in range(ii, N):
                            val = (P[ii, jj] - k[ii] * xH_P[jj]) / lam_f64
                            if ii == jj:
                                P[ii, ii] = np.complex128(val.real)
                            else:
                                conj_val = (P[jj, ii] - k[jj] * xH_P[ii]) / lam_f64
                                avg = (val + np.conj(conj_val)) * np.float64(0.5)
                                P[ii, jj] = avg
                                P[jj, ii] = np.conj(avg)

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

        @numba_mod.njit(cache=True, fastmath=False, nogil=True)
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
            bps_prev4,
            y_out,
            e_out,
            phase_out,
            w_hist_out,
            sq_lev_min,
            sq_d_grid,
            sq_side,
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
            # pll_phi       : (C,) float64          — in-place
            # pll_freq      : (C,) float64          — in-place
            # cs_buf_x      : (C, H) float64        — in-place
            # cs_buf_y      : (C, H) float64        — in-place
            # cs_buf_ptr    : (C,) int64            — in-place
            # cs_buf_n      : (C,) int64            — in-place
            # cs_stats      : (C, 4) float64  [Sx,Sy,Sxx,Sxy] — in-place
            # bps_prev4     : (C,) float64    — causal 4-fold unwrap state — in-place
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
            phi_c = np.empty(C, dtype=np.float64)
            d_sym = np.empty(C, dtype=np.complex64)
            e_clean = np.empty(C, dtype=np.complex64)
            e_eq = np.empty(C, dtype=np.complex64)
            phi_hat_bps = np.zeros(C, dtype=np.float64)
            # BPS: incremental running-sum (avoids O(K·B·M) per-symbol rescan)
            bps_dist_buf = np.zeros((C, bps_block_size, B), dtype=np.float32)
            bps_running_sum = np.zeros((C, B), dtype=np.float64)
            bps_dist_ptr = np.int64(0)

            for idx in range(n_sym):
                sample_idx = idx * stride

                # Window extraction
                for c in range(C):
                    for t in range(num_taps):
                        X_wins[c, t] = x_padded[c, sample_idx + t]

                # Butterfly forward pass: y_raw[i] = Σ_j conj(W[i,j]) · X
                for i in range(C):
                    acc_re = 0.0
                    acc_im = 0.0
                    for j in range(C):
                        for t in range(num_taps):
                            w = W[i, j, t]
                            x = X_wins[j, t]
                            acc_re += np.float64(w.real) * np.float64(
                                x.real
                            ) + np.float64(w.imag) * np.float64(x.imag)
                            acc_im += np.float64(w.real) * np.float64(
                                x.imag
                            ) - np.float64(w.imag) * np.float64(x.real)
                    y_raw[i] = acc_re + 1j * acc_im

                # ── CPR: Phase estimation ──────────────────────────────────

                # BPS: incremental running-sum over a causal window of K symbols.
                # For each new symbol, compute min-dist for all B candidates,
                # subtract the oldest slot from the running sum and add the new
                # one — O(B·M·C) per symbol, independent of window size K.
                # O(1) square-QAM path: snap real/imag to nearest level grid point.
                if cpr_mode == 2:
                    slot = bps_dist_ptr % np.int64(bps_block_size)
                    for i in range(C):
                        for k in range(B):
                            y_rot = y_raw[i] * bps_phases_neg[k]
                            if sq_side > np.int32(0):
                                ir = np.int32(
                                    np.round((y_rot.real - sq_lev_min) / sq_d_grid)
                                )
                                if ir < np.int32(0):
                                    ir = np.int32(0)
                                if ir >= sq_side:
                                    ir = sq_side - np.int32(1)
                                ii = np.int32(
                                    np.round((y_rot.imag - sq_lev_min) / sq_d_grid)
                                )
                                if ii < np.int32(0):
                                    ii = np.int32(0)
                                if ii >= sq_side:
                                    ii = sq_side - np.int32(1)
                                nr = sq_lev_min + np.float32(ir) * sq_d_grid
                                ni = sq_lev_min + np.float32(ii) * sq_d_grid
                                d2_min = (y_rot.real - nr) ** 2 + (y_rot.imag - ni) ** 2
                            else:
                                d2_min = np.float32(1e38)
                                for m in range(M):
                                    dv = y_rot - constellation[m]
                                    d2 = dv.real * dv.real + dv.imag * dv.imag
                                    if d2 < d2_min:
                                        d2_min = d2
                            # Subtract evicted slot, add new slot
                            bps_running_sum[i, k] = (
                                bps_running_sum[i, k]
                                - bps_dist_buf[i, slot, k]
                                + d2_min
                            )
                            bps_dist_buf[i, slot, k] = d2_min
                    bps_dist_ptr = bps_dist_ptr + np.int64(1)

                    if bps_joint_channels:
                        best_k_joint = np.int32(0)
                        min_tot_joint = np.float64(1e300)
                        for k in range(B):
                            metric_k = np.float64(0.0)
                            for i in range(C):
                                metric_k = metric_k + bps_running_sum[i, k]
                            if metric_k < min_tot_joint:
                                min_tot_joint = metric_k
                                best_k_joint = k
                        # Raw argmin in [0, π/2); apply 4-fold causal unwrap
                        raw4 = np.float64(bps_angles[best_k_joint]) * np.float64(4.0)
                        for i in range(C):
                            diff4 = raw4 - bps_prev4[i]
                            diff4 = diff4 - np.float64(2.0 * np.pi) * np.round(
                                diff4 / (np.float64(2.0 * np.pi))
                            )
                            bps_prev4[i] = bps_prev4[i] + diff4
                            phi_hat_bps[i] = bps_prev4[i] / np.float64(4.0)
                    else:
                        for i in range(C):
                            best_k = np.int32(0)
                            min_tot = np.float64(1e300)
                            for k in range(B):
                                if bps_running_sum[i, k] < min_tot:
                                    min_tot = bps_running_sum[i, k]
                                    best_k = k
                            # Raw argmin in [0, π/2); apply 4-fold causal unwrap
                            raw4 = np.float64(bps_angles[best_k]) * np.float64(4.0)
                            diff4 = raw4 - bps_prev4[i]
                            diff4 = diff4 - np.float64(2.0 * np.pi) * np.round(
                                diff4 / (np.float64(2.0 * np.pi))
                            )
                            bps_prev4[i] = bps_prev4[i] + diff4
                            phi_hat_bps[i] = bps_prev4[i] / np.float64(4.0)

                for i in range(C):
                    if cpr_mode == 1:  # PLL: read current integrator state
                        phi_hat = pll_phi[i]
                    else:  # BPS: unwrapped phase estimate
                        phi_hat = phi_hat_bps[i]

                    # ── Cycle-slip correction ──────────────────────────────
                    if cs_enabled:
                        n_b = cs_buf_n[i]
                        ptr = cs_buf_ptr[i]
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
                                # cs_stats layout: [0]=Sy, [1]=Sxy (relative coords)
                                Sy = cs_stats[i, 0]
                                Sxy = cs_stats[i, 1]
                                n_f = np.float64(n_b)
                                if n_b < np.int64(H):
                                    Sx_c = (
                                        n_f * (n_f - np.float64(1.0)) / np.float64(2.0)
                                    )
                                    Sxx_c = (
                                        n_f
                                        * (n_f - np.float64(1.0))
                                        * (np.float64(2.0) * n_f - np.float64(1.0))
                                        / np.float64(6.0)
                                    )
                                    denom = n_f * Sxx_c - Sx_c * Sx_c
                                else:
                                    H_f = np.float64(H)
                                    Sx_c = (
                                        H_f * (H_f - np.float64(1.0)) / np.float64(2.0)
                                    )
                                    Sxx_c = (
                                        H_f
                                        * (H_f - np.float64(1.0))
                                        * (np.float64(2.0) * H_f - np.float64(1.0))
                                        / np.float64(6.0)
                                    )
                                    denom = H_f * Sxx_c - Sx_c * Sx_c
                                if np.abs(denom) > np.float64(1e-30):
                                    slope = (n_f * Sxy - Sx_c * Sy) / denom
                                    intercept = (Sy - slope * Sx_c) / n_f
                                else:
                                    slope = np.float64(0.0)
                                    intercept = Sy / n_f
                                phi_expected = slope * n_f + intercept

                            diff = y_b - phi_expected
                            k_slip = np.int64(np.round(diff / quantum))
                            if np.abs(diff) > np.float64(
                                cs_threshold
                            ) and k_slip != np.int64(0):
                                y_b = y_b - np.float64(k_slip) * quantum
                            phi_corr = y_b

                        # Update circular buffer — relative coords, only y needed
                        write_pos = ptr % np.int64(H)
                        if n_b == np.int64(H):
                            old_y = cs_buf_y[i, write_pos]
                            H_f = np.float64(H)
                            old_Sy = cs_stats[i, 0]
                            # Sxy_new = Sxy_old - Sy_old + y_old + (H-1)*y_new
                            cs_stats[i, 1] = (
                                cs_stats[i, 1]
                                - old_Sy
                                + old_y
                                + (H_f - np.float64(1.0)) * phi_corr
                            )
                            cs_stats[i, 0] = old_Sy - old_y + phi_corr
                        else:
                            n_b_f = np.float64(n_b)
                            cs_stats[i, 1] += n_b_f * phi_corr
                            cs_stats[i, 0] += phi_corr
                        cs_buf_y[i, write_pos] = phi_corr
                        cs_buf_ptr[i] = ptr + np.int64(1)
                        if n_b < np.int64(H):
                            cs_buf_n[i] = n_b + np.int64(1)
                    else:
                        phi_corr = np.float64(phi_hat)

                    phi_c[i] = phi_corr
                    phase_out[idx, i] = phi_corr

                    # y_final = y_raw * exp(-j * phi_corr)
                    cos_p = np.cos(phi_corr)
                    sin_p = np.sin(phi_corr)
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
                    elif sq_side > np.int32(0):
                        ir = np.int32(
                            np.round((y_fin[i].real - sq_lev_min) / sq_d_grid)
                        )
                        if ir < np.int32(0):
                            ir = np.int32(0)
                        if ir >= sq_side:
                            ir = sq_side - np.int32(1)
                        ii = np.int32(
                            np.round((y_fin[i].imag - sq_lev_min) / sq_d_grid)
                        )
                        if ii < np.int32(0):
                            ii = np.int32(0)
                        if ii >= sq_side:
                            ii = sq_side - np.int32(1)
                        nr = sq_lev_min + np.float32(ir) * sq_d_grid
                        ni = sq_lev_min + np.float32(ii) * sq_d_grid
                        d_i = np.complex64(nr + ni * np.complex64(1j))
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
                    if bps_joint_channels:
                        # Average phase error across channels (shared LO)
                        e_ph_sum = np.float32(0.0)
                        for i in range(C):
                            e_ph_sum = e_ph_sum + (
                                y_fin[i].imag * d_sym[i].real
                                - y_fin[i].real * d_sym[i].imag
                            )
                        e_ph_avg = e_ph_sum / np.float32(C)
                        for i in range(C):
                            pll_phi[i] = pll_phi[i] + pll_mu * e_ph_avg + pll_freq[i]
                            pll_freq[i] = pll_freq[i] + pll_beta * e_ph_avg
                    else:
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
                    cos_p = np.cos(phi_c[i])
                    sin_p = np.sin(phi_c[i])
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

        @numba_mod.njit(cache=True, fastmath=False, nogil=True)
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
            bps_prev4,
            y_out,
            e_out,
            phase_out,
            w_hist_out,
            sq_lev_min,
            sq_d_grid,
            sq_side,
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

            x_bar = np.empty(N, dtype=np.complex128)
            Px = np.empty(N, dtype=np.complex128)
            xH_P = np.empty(N, dtype=np.complex128)
            k_gain = np.empty(N, dtype=np.complex128)
            y_raw = np.empty(C, dtype=np.complex64)
            y_fin = np.empty(C, dtype=np.complex64)
            phi_c = np.empty(C, dtype=np.float64)
            d_sym = np.empty(C, dtype=np.complex64)
            e_clean = np.empty(C, dtype=np.complex64)
            e_eq = np.empty(C, dtype=np.complex64)
            phi_hat_bps = np.zeros(C, dtype=np.float64)
            bps_dist_buf = np.zeros((C, bps_block_size, B), dtype=np.float32)
            bps_running_sum = np.zeros((C, B), dtype=np.float64)
            bps_dist_ptr = np.int64(0)

            lam_f64 = np.float64(lam)
            leak_term = np.float32(1.0) - np.float32(leakage)

            for idx in range(n_sym):
                sample_idx = idx * stride

                for j in range(C):
                    for t in range(num_taps):
                        x_bar[j * num_taps + t] = x_padded[j, sample_idx + t]

                # Butterfly forward pass
                for i in range(C):
                    acc_re = 0.0
                    acc_im = 0.0
                    for j in range(C):
                        for t in range(num_taps):
                            w = W[i, j, t]
                            x = x_bar[j * num_taps + t]
                            acc_re += np.float64(w.real) * np.float64(
                                x.real
                            ) + np.float64(w.imag) * np.float64(x.imag)
                            acc_im += np.float64(w.real) * np.float64(
                                x.imag
                            ) - np.float64(w.imag) * np.float64(x.real)
                    y_raw[i] = acc_re + 1j * acc_im

                # ── CPR: Phase estimation ──────────────────────────────────

                # BPS: incremental running-sum (O(B·M·C) per symbol, independent of K)
                # O(1) square-QAM path: snap real/imag to nearest level grid point.
                if cpr_mode == 2:
                    slot = bps_dist_ptr % np.int64(bps_block_size)
                    for i in range(C):
                        for k in range(B):
                            y_rot = y_raw[i] * bps_phases_neg[k]
                            if sq_side > np.int32(0):
                                ir = np.int32(
                                    np.round((y_rot.real - sq_lev_min) / sq_d_grid)
                                )
                                if ir < np.int32(0):
                                    ir = np.int32(0)
                                if ir >= sq_side:
                                    ir = sq_side - np.int32(1)
                                ii = np.int32(
                                    np.round((y_rot.imag - sq_lev_min) / sq_d_grid)
                                )
                                if ii < np.int32(0):
                                    ii = np.int32(0)
                                if ii >= sq_side:
                                    ii = sq_side - np.int32(1)
                                nr = sq_lev_min + np.float32(ir) * sq_d_grid
                                ni = sq_lev_min + np.float32(ii) * sq_d_grid
                                d2_min = (y_rot.real - nr) ** 2 + (y_rot.imag - ni) ** 2
                            else:
                                d2_min = np.float32(1e38)
                                for m in range(M):
                                    dv = y_rot - constellation[m]
                                    d2 = dv.real * dv.real + dv.imag * dv.imag
                                    if d2 < d2_min:
                                        d2_min = d2
                            bps_running_sum[i, k] = (
                                bps_running_sum[i, k]
                                - bps_dist_buf[i, slot, k]
                                + d2_min
                            )
                            bps_dist_buf[i, slot, k] = d2_min
                    bps_dist_ptr = bps_dist_ptr + np.int64(1)

                    if bps_joint_channels:
                        best_k_joint = np.int32(0)
                        min_tot_joint = np.float64(1e300)
                        for k in range(B):
                            metric_k = np.float64(0.0)
                            for i in range(C):
                                metric_k = metric_k + bps_running_sum[i, k]
                            if metric_k < min_tot_joint:
                                min_tot_joint = metric_k
                                best_k_joint = k
                        raw4 = np.float64(bps_angles[best_k_joint]) * np.float64(4.0)
                        for i in range(C):
                            diff4 = raw4 - bps_prev4[i]
                            diff4 = diff4 - np.float64(2.0 * np.pi) * np.round(
                                diff4 / (np.float64(2.0 * np.pi))
                            )
                            bps_prev4[i] = bps_prev4[i] + diff4
                            phi_hat_bps[i] = bps_prev4[i] / np.float64(4.0)
                    else:
                        for i in range(C):
                            best_k = np.int32(0)
                            min_tot = np.float64(1e300)
                            for k in range(B):
                                if bps_running_sum[i, k] < min_tot:
                                    min_tot = bps_running_sum[i, k]
                                    best_k = k
                            raw4 = np.float64(bps_angles[best_k]) * np.float64(4.0)
                            diff4 = raw4 - bps_prev4[i]
                            diff4 = diff4 - np.float64(2.0 * np.pi) * np.round(
                                diff4 / (np.float64(2.0 * np.pi))
                            )
                            bps_prev4[i] = bps_prev4[i] + diff4
                            phi_hat_bps[i] = bps_prev4[i] / np.float64(4.0)

                for i in range(C):
                    if cpr_mode == 1:
                        phi_hat = pll_phi[i]
                    else:
                        phi_hat = phi_hat_bps[i]

                    # ── Cycle-slip correction ──────────────────────────────
                    if cs_enabled:
                        n_b = cs_buf_n[i]
                        ptr = cs_buf_ptr[i]
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
                                # cs_stats layout: [0]=Sy, [1]=Sxy (relative coords)
                                Sy = cs_stats[i, 0]
                                Sxy = cs_stats[i, 1]
                                n_f = np.float64(n_b)
                                if n_b < np.int64(H):
                                    Sx_c = (
                                        n_f * (n_f - np.float64(1.0)) / np.float64(2.0)
                                    )
                                    Sxx_c = (
                                        n_f
                                        * (n_f - np.float64(1.0))
                                        * (np.float64(2.0) * n_f - np.float64(1.0))
                                        / np.float64(6.0)
                                    )
                                    denom = n_f * Sxx_c - Sx_c * Sx_c
                                else:
                                    H_f = np.float64(H)
                                    Sx_c = (
                                        H_f * (H_f - np.float64(1.0)) / np.float64(2.0)
                                    )
                                    Sxx_c = (
                                        H_f
                                        * (H_f - np.float64(1.0))
                                        * (np.float64(2.0) * H_f - np.float64(1.0))
                                        / np.float64(6.0)
                                    )
                                    denom = H_f * Sxx_c - Sx_c * Sx_c
                                if np.abs(denom) > np.float64(1e-30):
                                    slope = (n_f * Sxy - Sx_c * Sy) / denom
                                    intercept = (Sy - slope * Sx_c) / n_f
                                else:
                                    slope = np.float64(0.0)
                                    intercept = Sy / n_f
                                phi_expected = slope * n_f + intercept

                            diff = y_b - phi_expected
                            k_slip = np.int64(np.round(diff / quantum))
                            if np.abs(diff) > np.float64(
                                cs_threshold
                            ) and k_slip != np.int64(0):
                                y_b = y_b - np.float64(k_slip) * quantum
                            phi_corr = y_b

                        # Update circular buffer — relative coords, only y needed
                        write_pos = ptr % np.int64(H)
                        if n_b == np.int64(H):
                            old_y = cs_buf_y[i, write_pos]
                            H_f = np.float64(H)
                            old_Sy = cs_stats[i, 0]
                            # Sxy_new = Sxy_old - Sy_old + y_old + (H-1)*y_new
                            cs_stats[i, 1] = (
                                cs_stats[i, 1]
                                - old_Sy
                                + old_y
                                + (H_f - np.float64(1.0)) * phi_corr
                            )
                            cs_stats[i, 0] = old_Sy - old_y + phi_corr
                        else:
                            n_b_f = np.float64(n_b)
                            cs_stats[i, 1] += n_b_f * phi_corr
                            cs_stats[i, 0] += phi_corr
                        cs_buf_y[i, write_pos] = phi_corr
                        cs_buf_ptr[i] = ptr + np.int64(1)
                        if n_b < np.int64(H):
                            cs_buf_n[i] = n_b + np.int64(1)
                    else:
                        phi_corr = np.float64(phi_hat)

                    phi_c[i] = phi_corr
                    phase_out[idx, i] = phi_corr

                    cos_p = np.cos(phi_corr)
                    sin_p = np.sin(phi_corr)
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
                    elif sq_side > np.int32(0):
                        ir = np.int32(
                            np.round((y_fin[i].real - sq_lev_min) / sq_d_grid)
                        )
                        if ir < np.int32(0):
                            ir = np.int32(0)
                        if ir >= sq_side:
                            ir = sq_side - np.int32(1)
                        ii = np.int32(
                            np.round((y_fin[i].imag - sq_lev_min) / sq_d_grid)
                        )
                        if ii < np.int32(0):
                            ii = np.int32(0)
                        if ii >= sq_side:
                            ii = sq_side - np.int32(1)
                        nr = sq_lev_min + np.float32(ir) * sq_d_grid
                        ni = sq_lev_min + np.float32(ii) * sq_d_grid
                        d_i = np.complex64(nr + ni * np.complex64(1j))
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
                    if bps_joint_channels:
                        e_ph_sum = np.float32(0.0)
                        for i in range(C):
                            e_ph_sum = e_ph_sum + (
                                y_fin[i].imag * d_sym[i].real
                                - y_fin[i].real * d_sym[i].imag
                            )
                        e_ph_avg = e_ph_sum / np.float32(C)
                        for i in range(C):
                            pll_phi[i] = pll_phi[i] + pll_mu * e_ph_avg + pll_freq[i]
                            pll_freq[i] = pll_freq[i] + pll_beta * e_ph_avg
                    else:
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
                    cos_p = np.cos(phi_c[i])
                    sin_p = np.sin(phi_c[i])
                    er = e_clean[i].real * np.float32(cos_p) - e_clean[
                        i
                    ].imag * np.float32(sin_p)
                    ei = e_clean[i].real * np.float32(sin_p) + e_clean[
                        i
                    ].imag * np.float32(cos_p)
                    e_eq[i] = np.complex64(er + ei * np.complex64(1j))

                # ── Kalman gain ────────────────────────────────────────────
                for ii in range(N):
                    acc_re = 0.0
                    acc_im = 0.0
                    for jj in range(N):
                        p_val = P[ii, jj]
                        x_val = x_bar[jj]
                        acc_re += p_val.real * x_val.real - p_val.imag * x_val.imag
                        acc_im += p_val.real * x_val.imag + p_val.imag * x_val.real
                    Px[ii] = acc_re + 1j * acc_im
                denom_k = lam_f64
                for jj in range(N):
                    denom_k = denom_k + (np.conj(x_bar[jj]) * Px[jj]).real
                inv_denom = np.float64(1.0) / denom_k
                for ii in range(N):
                    k_gain[ii] = Px[ii] * inv_denom

                if idx < n_update_halt:
                    for i in range(C):
                        ce_i = np.conj(e_eq[i])
                        for j in range(C):
                            for t in range(num_taps):
                                W[i, j, t] = np.complex64(
                                    leak_term * W[i, j, t]
                                    + k_gain[j * num_taps + t] * ce_i
                                )
                    for jj in range(N):
                        xH_P[jj] = np.conj(Px[jj])
                    for ii in range(N):
                        for jj in range(ii, N):
                            val = (P[ii, jj] - k_gain[ii] * xH_P[jj]) / lam_f64
                            if ii == jj:
                                P[ii, ii] = np.complex128(val.real)
                            else:
                                conj_val = (P[jj, ii] - k_gain[jj] * xH_P[ii]) / lam_f64
                                avg = (val + np.conj(conj_val)) * np.float64(0.5)
                                P[ii, jj] = avg
                                P[jj, ii] = np.conj(avg)

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
                    acc_re = 0.0
                    acc_im = 0.0
                    for j in range(C):
                        for t in range(num_taps):
                            w = W[i, j, t]
                            x = X_wins[j, t]
                            acc_re += np.float64(w.real) * np.float64(
                                x.real
                            ) + np.float64(w.imag) * np.float64(x.imag)
                            acc_im += np.float64(w.real) * np.float64(
                                x.imag
                            ) - np.float64(w.imag) * np.float64(x.real)
                    y[i] = acc_re + 1j * acc_im
                    y_out[idx, i] = acc_re + 1j * acc_im

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
                    acc_re = 0.0
                    acc_im = 0.0
                    for j in range(C):
                        for t in range(num_taps):
                            w = W[i, j, t]
                            x = X_wins[j, t]
                            acc_re += np.float64(w.real) * np.float64(
                                x.real
                            ) + np.float64(w.imag) * np.float64(x.imag)
                            acc_im += np.float64(w.real) * np.float64(
                                x.imag
                            ) - np.float64(w.imag) * np.float64(x.real)
                    y[i] = acc_re + 1j * acc_im
                    y_out[idx, i] = acc_re + 1j * acc_im

                # RDE error: e[i] = y[i] * (|y[i]|² − R_d²)
                # R_d is the magnitude of the nearest constellation ring.
                # Linear scan over K unique radii is cheap (K ≤ 8 for typical QAM).
                for i in range(C):
                    mod2 = y[i].real * y[i].real + y[i].imag * y[i].imag
                    abs_y = np.sqrt(mod2)

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


def _cpr_state_to_jax_inits(state: "CPRState", num_ch: int, KB: int, H: int):
    """Extract CPR carry init arrays from a CPRState for JAX warm-start.

    Returns CPU NumPy arrays with correct dtypes/shapes for the JAX carry.
    Missing fields are zero-initialised.  Caller converts to JAX arrays via
    ``to_jax(..., device=platform)``.

    Returns (pll_phi, pll_freq, bps_buf, bps_buf_ptr, bps_prev4,
             cs_buf_x, cs_buf_y, cs_buf_ptr) — all NumPy.
    """

    def _get(val, shape, dtype):
        return (
            np.asarray(val, dtype=dtype)
            if val is not None
            else np.zeros(shape, dtype=dtype)
        )

    pll_phi = _get(state.pll_phi, (num_ch,), np.float64)
    pll_freq = _get(state.pll_freq, (num_ch,), np.float64)
    bps_buf = _get(state.jax_bps_buf, (KB, num_ch), np.complex64)
    bps_buf_ptr = np.int32(
        state.jax_bps_buf_ptr if state.jax_bps_buf_ptr is not None else 0
    )
    bps_prev4 = _get(state.bps_prev4, (num_ch,), np.float64)
    cs_buf_x = _get(state.cs_buf_x, (num_ch, H), np.float64)
    cs_buf_y = _get(state.cs_buf_y, (num_ch, H), np.float64)
    cs_buf_ptr = _get(state.cs_buf_ptr, (num_ch,), np.int32)
    return (
        pll_phi,
        pll_freq,
        bps_buf,
        bps_buf_ptr,
        bps_prev4,
        cs_buf_x,
        cs_buf_y,
        cs_buf_ptr,
    )


def _get_jax_lms(
    num_taps, stride, const_size, num_ch, sq_side=0, sq_lev_min=0.0, sq_d_grid=1.0
):
    """JIT-compile and cache the sample-by-sample LMS butterfly scan.

    Static closure variables (baked into the compiled kernel; a new cache
    entry is created — not a retrace — when any of these change):

    num_taps   : FIR filter length per polyphase arm.
    stride     : decimation factor (== sps, typically 2 for T/2-spaced input).
    const_size : constellation size M — fixes the slicer ``argmin`` shape at
                 trace time so XLA can compile it without dynamic dispatch.
    num_ch     : MIMO butterfly width C (number of input/output channels).
    sq_side    : int — 0 means O(M) slicer; >0 enables O(1) square-QAM slicer.
    sq_lev_min, sq_d_grid : float — constellation level grid parameters.

    Returns
    -------
    lms_scan : JIT-compiled callable
        See the inner function for the call signature.
    """
    key = (
        "lms",
        num_taps,
        stride,
        const_size,
        num_ch,
        sq_side,
        float(sq_lev_min),
        float(sq_d_grid),
    )
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()

        @jax.jit
        def lms_scan(
            x_input, training_padded, constellation, w_init, step_size, n_train
        ):
            # x_input         : (C, N_pad)        complex64
            # training_padded : (C, N_sym)         complex64
            # constellation   : (M,)               complex64 — slicer lookup table
            # w_init          : (C, C, num_taps)   complex64
            # step_size       : scalar float32
            # n_train         : scalar int32

            def step(W, idx):
                sample_idx = idx * stride

                X_wins = jax.lax.dynamic_slice(
                    x_input, (0, sample_idx), (num_ch, num_taps)
                )

                _P = jax.lax.Precision.HIGHEST
                y = jnp.einsum("ijt,jt->i", jnp.conj(W), X_wins, precision=_P)  # (C,)

                def slicer(ch_y):
                    if sq_side > 0:  # static branch at trace time
                        ir = jnp.clip(
                            jnp.round((ch_y.real - sq_lev_min) / sq_d_grid).astype(
                                jnp.int32
                            ),
                            0,
                            sq_side - 1,
                        )
                        ii = jnp.clip(
                            jnp.round((ch_y.imag - sq_lev_min) / sq_d_grid).astype(
                                jnp.int32
                            ),
                            0,
                            sq_side - 1,
                        )
                        nr = sq_lev_min + ir.astype(jnp.float32) * jnp.float32(
                            sq_d_grid
                        )
                        ni = sq_lev_min + ii.astype(jnp.float32) * jnp.float32(
                            sq_d_grid
                        )
                        return jax.lax.complex(nr, ni)
                    else:
                        return constellation[
                            jnp.argmin(jnp.abs(ch_y - constellation) ** 2)
                        ]

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


def _get_jax_rls(
    num_taps, stride, const_size, num_ch, sq_side=0, sq_lev_min=0.0, sq_d_grid=1.0
):
    """JIT-compile and cache the sample-by-sample Leaky-RLS butterfly scan.

    Static closure variables (same semantics as ``_get_jax_lms``):
    num_taps, stride, const_size, num_ch.
    sq_side, sq_lev_min, sq_d_grid: O(1) square-QAM slicer parameters.

    Returns
    -------
    rls_scan : JIT-compiled callable
        See the inner function for the call signature.
    """
    key = (
        "rls",
        num_taps,
        stride,
        const_size,
        num_ch,
        sq_side,
        float(sq_lev_min),
        float(sq_d_grid),
    )
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

            _P = jax.lax.Precision.HIGHEST

            def step(carry, idx):
                W, P = carry
                sample_idx = idx * stride

                X_wins = jax.lax.dynamic_slice(
                    x_input, (0, sample_idx), (num_ch, num_taps)
                )

                y = jnp.einsum("ijt,jt->i", jnp.conj(W), X_wins, precision=_P)

                def slicer(ch_y):
                    if sq_side > 0:  # static branch at trace time
                        ir = jnp.clip(
                            jnp.round((ch_y.real - sq_lev_min) / sq_d_grid).astype(
                                jnp.int32
                            ),
                            0,
                            sq_side - 1,
                        )
                        ii = jnp.clip(
                            jnp.round((ch_y.imag - sq_lev_min) / sq_d_grid).astype(
                                jnp.int32
                            ),
                            0,
                            sq_side - 1,
                        )
                        nr = sq_lev_min + ir.astype(jnp.float32) * jnp.float32(
                            sq_d_grid
                        )
                        ni = sq_lev_min + ii.astype(jnp.float32) * jnp.float32(
                            sq_d_grid
                        )
                        return jax.lax.complex(nr, ni)
                    else:
                        return constellation[
                            jnp.argmin(jnp.abs(ch_y - constellation) ** 2)
                        ]

                dd = jax.vmap(slicer)(y)
                d = jnp.where(idx < n_train, training_padded[:, idx], dd)
                e = d - y

                x_bar = X_wins.flatten()  # (C * num_taps,)

                Px = jnp.matmul(P, x_bar, precision=_P)
                denom = lam + jnp.real(jnp.dot(jnp.conj(x_bar), Px, precision=_P))
                k = Px / denom

                def w_update(w_row, err_val):
                    w_flat = w_row.flatten()
                    # Weight decay: suppress null-subspace taps exponentially.
                    # Adding γI directly to P would inflate P eigenvalues, making
                    # R_xx more singular and worsening AWGN amplification.
                    w_flat_new = (1.0 - leakage) * w_flat + k * jnp.conj(err_val)
                    return w_flat_new.reshape(num_ch, num_taps)

                W_upd = jax.vmap(w_update)(W, e).astype(jnp.complex64)
                # Riccati: exploit Hermitian symmetry P = P^H so (x^H P)[j] = conj((Px)[j]).
                # outer(k, conj(Px)) == k ⊗ (x^H P); reuses Px to avoid a second O(N²) mat-vec.
                P_upd = (P - jnp.outer(k, jnp.conj(Px))) / lam
                # Hermitian re-symmetrization: P ← (P + Pᴴ)/2
                # Prevents asymmetry drift from the 1/λ amplification.
                P_upd = 0.5 * (P_upd + jnp.conj(P_upd).T)

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
    symmetry=4,
    sq_side=0,
    sq_lev_min=0.0,
    sq_d_grid=1.0,
):
    """JIT-compile and cache the LMS+CPR butterfly scan.

    All CPR parameters are static closure variables (baked into the XLA graph
    at trace time).  A separate cache entry is created for each distinct
    combination of these parameters.

    cpr_type : "pll" or "bps"
    bps_n    : number of BPS test phases (ignored for cpr_type="pll")
    cs_history_len : int — circular buffer depth for cycle-slip correction
    sq_side, sq_lev_min, sq_d_grid : O(1) square-QAM slicer parameters
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
        int(symmetry),
        sq_side,
        float(sq_lev_min),
        float(sq_d_grid),
    )
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()

        H = cs_history_len
        KB = bps_block_size  # static closure: BPS window length

        import math as _math  # noqa: PLC0415

        _quantum_static = jnp.float64(2.0 * _math.pi / symmetry)

        _PREC = jax.lax.Precision.HIGHEST

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
            pll_phi_init,
            pll_freq_init,
            bps_buf_init,
            bps_buf_ptr_init,
            bps_prev4_init,
            cs_buf_x_init,
            cs_buf_y_init,
            cs_buf_ptr_init,
        ):
            # x_input         : (C, N_pad)       complex64
            # training_padded : (C, N_sym)        complex64
            # constellation   : (M,)              complex64
            # bps_phases_neg  : (B,)              complex64  exp(-j*theta_k)
            # bps_angles      : (B,)              float32    theta_k
            # w_init          : (C, C, T)         complex64
            # step_size       : scalar float32
            # n_train         : scalar int32
            # pll_mu, pll_beta: scalar float64
            # cs_threshold    : scalar float64
            # cs_enabled      : scalar bool
            # pll_phi_init    : (C,) float64  — warm-start PLL integrator (zeros → cold)
            # pll_freq_init   : (C,) float64  — warm-start PLL frequency  (zeros → cold)
            # bps_buf_init    : (KB, C) complex64 — warm-start BPS buffer (zeros → cold)
            # bps_buf_ptr_init: scalar int32  — warm-start BPS buffer pointer
            # bps_prev4_init  : (C,) float64  — warm-start 4-fold unwrap state
            # cs_buf_x_init   : (C, H) float64 — warm-start cycle-slip symbol index
            # cs_buf_y_init   : (C, H) float64 — warm-start cycle-slip phase value
            # cs_buf_ptr_init : (C,)   int32   — warm-start cycle-slip write pointer
            #
            # lax.scan carry:
            #   W             : (C, C, T)         complex64
            #   pll_phi       : (C,)              float64
            #   pll_freq      : (C,)              float64
            #   bps_buf       : (KB, C)           complex64  — y_raw circular buffer
            #   bps_buf_ptr   : scalar int32
            #   bps_prev4     : (C,)              float64   — causal 4-fold unwrap state
            #   cs_buf_x      : (C, H)            float64  — symbol index
            #   cs_buf_y      : (C, H)            float64  — phase value
            #   cs_buf_ptr    : (C,)              int32    — write pointer

            def step(carry, idx):
                (
                    W,
                    pll_phi,
                    pll_freq,
                    bps_buf,
                    bps_buf_ptr,
                    bps_prev4,
                    cs_buf_x,
                    cs_buf_y,
                    cs_buf_ptr,
                ) = carry
                sample_idx = idx * stride

                X_wins = jax.lax.dynamic_slice(
                    x_input, (0, sample_idx), (num_ch, num_taps)
                )  # (C, T)

                y_raw = jnp.einsum(
                    "ijt,jt->i",
                    jnp.conj(W).astype(jnp.complex128),
                    X_wins.astype(jnp.complex128),
                    precision=_PREC,
                ).astype(
                    jnp.complex64
                )  # float64 accumulation → complex64, matches Numba

                # ── Phase estimation (static branch at trace time) ──
                if cpr_type == "pll":
                    phi_hat = pll_phi  # (C,)
                    bps_buf_new = bps_buf
                    bps_buf_ptr_new = bps_buf_ptr
                    bps_prev4_new = bps_prev4
                else:
                    # Fill BPS circular buffer with current y_raw
                    slot = jnp.int32(bps_buf_ptr % KB)
                    bps_buf_new = jax.lax.dynamic_update_slice(
                        bps_buf,
                        y_raw[None, :],  # (1, C) — update one row
                        (slot, jnp.int32(0)),
                    )  # broadcast doesn't work for transpose, use (C, KB) layout instead
                    bps_buf_ptr_new = bps_buf_ptr + 1
                    fill = jnp.minimum(bps_buf_ptr_new, KB)

                    # rotated: (B, KB, C)
                    rotated = (
                        bps_phases_neg[:, None, None] * bps_buf_new[None, :, :]
                    )  # (B, KB, C)
                    # min-dist per candidate per slot per channel: (B, KB, C)
                    if sq_side > 0:  # static branch at trace time
                        r_idx = jnp.clip(
                            jnp.round((rotated.real - sq_lev_min) / sq_d_grid).astype(
                                jnp.int32
                            ),
                            0,
                            sq_side - 1,
                        )
                        i_idx = jnp.clip(
                            jnp.round((rotated.imag - sq_lev_min) / sq_d_grid).astype(
                                jnp.int32
                            ),
                            0,
                            sq_side - 1,
                        )
                        r_near = sq_lev_min + r_idx.astype(jnp.float32) * jnp.float32(
                            sq_d_grid
                        )
                        i_near = sq_lev_min + i_idx.astype(jnp.float32) * jnp.float32(
                            sq_d_grid
                        )
                        d2_all = (rotated.real - r_near) ** 2 + (
                            rotated.imag - i_near
                        ) ** 2
                    else:
                        d2_all = jnp.min(
                            jnp.abs(
                                rotated[:, :, :, None]
                                - constellation[None, None, None, :]
                            )
                            ** 2,
                            axis=-1,
                        )
                    # Mask slots beyond fill
                    slot_mask = jnp.arange(KB)[None, :, None] < fill  # (1, KB, 1)
                    d2_masked = jnp.where(
                        slot_mask, d2_all.astype(jnp.float64), jnp.float64(0.0)
                    )
                    metric = d2_masked.sum(
                        axis=1
                    )  # (B, C) float64 — matches Numba bps_running_sum

                    if bps_joint_channels:
                        # Sum over channels too → (B,); broadcast winner to all C
                        best_k = jnp.argmin(metric.sum(axis=-1))  # scalar
                        phi_raw = jnp.full(num_ch, bps_angles[best_k])  # (C,)
                    else:
                        best_k = jnp.argmin(metric, axis=0)  # (C,)
                        phi_raw = bps_angles[best_k]  # (C,)

                    # Causal 4-fold phase unwrap: equivalent to np.unwrap(phi*4)/4
                    raw4 = phi_raw.astype(jnp.float64) * jnp.float64(4.0)
                    diff4 = raw4 - bps_prev4
                    two_pi = jnp.float64(2.0 * 3.141592653589793)
                    diff4 = diff4 - jnp.round(diff4 / two_pi) * two_pi
                    bps_prev4_new = bps_prev4 + diff4
                    phi_hat = bps_prev4_new / jnp.float64(4.0)  # (C,) unwrapped float64

                # ── Cycle-slip correction ────────────────────────────
                def correct_slip_ch(phi_h, buf_x_ch, buf_y_ch, ptr_ch):
                    # buf_x_ch is retained for call-site compat but unused here.
                    fill_cs = jnp.minimum(ptr_ch, H)
                    y_b = phi_h

                    mask = jnp.arange(H) < fill_cs
                    n_f = fill_cs.astype(jnp.float64)

                    # Sx and Sxx are exact closed-form constants — no cancellation.
                    Sx = n_f * (n_f - jnp.float64(1.0)) / jnp.float64(2.0)
                    Sxx = (
                        n_f
                        * (n_f - jnp.float64(1.0))
                        * (jnp.float64(2.0) * n_f - jnp.float64(1.0))
                        / jnp.float64(6.0)
                    )
                    denom = n_f * Sxx - Sx * Sx

                    # Sxy uses relative positions [0, fill_cs-1].  In the circular
                    # buffer the oldest entry is at slot ptr_ch%H and has position 0;
                    # slot j has relative position (j - ptr_ch%H + H) % H for a full
                    # window, or just j for a partial window (entries written 0..fill-1).
                    oldest_slot = ptr_ch % H
                    slots = jnp.arange(H)
                    rel_pos_full = (slots - oldest_slot + H) % H
                    rel_pos_partial = slots
                    rel_pos = jnp.where(
                        fill_cs >= H, rel_pos_full, rel_pos_partial
                    ).astype(jnp.float64)
                    Sy = jnp.where(mask, buf_y_ch, jnp.float64(0.0)).sum()
                    Sxy = jnp.where(mask, rel_pos * buf_y_ch, jnp.float64(0.0)).sum()

                    # Prediction target: one step past the newest entry (relative pos = fill_cs).
                    x_pred = n_f

                    safe_denom = jnp.where(
                        jnp.abs(denom) > 1e-20, denom, jnp.float64(1.0)
                    )
                    slope = jnp.where(
                        fill_cs >= 10,
                        (n_f * Sxy - Sx * Sy) / safe_denom,
                        jnp.float64(0.0),
                    )
                    intercept = jnp.where(
                        fill_cs >= 10,
                        (Sy - slope * Sx) / jnp.maximum(n_f, jnp.float64(1.0)),
                        Sy / jnp.maximum(n_f, jnp.float64(1.0)),
                    )
                    phi_exp_lin = slope * x_pred + intercept

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
                    buf_y_new = jax.lax.dynamic_update_slice(
                        buf_y_ch, phi_corr[None], [write_pos]
                    )
                    ptr_new = ptr_ch + 1
                    return phi_corr, buf_x_ch, buf_y_new, ptr_new

                phi_corr, cs_buf_x_new, cs_buf_y_new, cs_buf_ptr_new = jax.vmap(
                    correct_slip_ch
                )(phi_hat, cs_buf_x, cs_buf_y, cs_buf_ptr)

                # Wrap to [-π, π] before casting to float32 for fast GPU exp
                _two_pi_f64 = jnp.float64(2.0 * 3.141592653589793)
                phi_wrapped = phi_corr - jnp.round(phi_corr / _two_pi_f64) * _two_pi_f64
                _phi32 = phi_wrapped.astype(jnp.float32)
                phasor = jnp.exp(_phi32 * jnp.array(-1j, dtype=jnp.complex64))
                y_fin = y_raw * phasor  # (C,)

                def slicer(ch_y):
                    if sq_side > 0:  # static branch at trace time
                        ir = jnp.clip(
                            jnp.round((ch_y.real - sq_lev_min) / sq_d_grid).astype(
                                jnp.int32
                            ),
                            0,
                            sq_side - 1,
                        )
                        ii = jnp.clip(
                            jnp.round((ch_y.imag - sq_lev_min) / sq_d_grid).astype(
                                jnp.int32
                            ),
                            0,
                            sq_side - 1,
                        )
                        nr = sq_lev_min + ir.astype(jnp.float32) * jnp.float32(
                            sq_d_grid
                        )
                        ni = sq_lev_min + ii.astype(jnp.float32) * jnp.float32(
                            sq_d_grid
                        )
                        return jax.lax.complex(nr, ni)
                    else:
                        return constellation[
                            jnp.argmin(jnp.abs(ch_y - constellation) ** 2)
                        ]

                dd = jax.vmap(slicer)(y_fin)
                d = jnp.where(idx < n_train, training_padded[:, idx], dd)
                e_clean = d - y_fin  # (C,)

                if cpr_type == "pll":
                    e_ph = y_fin.imag * d.real - y_fin.real * d.imag  # (C,)
                    if bps_joint_channels:  # static branch — shared LO
                        e_ph = jnp.full(num_ch, e_ph.mean())
                    pll_phi_new = pll_phi + pll_mu * e_ph + pll_freq
                    pll_freq_new = pll_freq + pll_beta * e_ph
                else:
                    pll_phi_new = pll_phi
                    pll_freq_new = pll_freq

                phasor_inv = jnp.exp(_phi32 * jnp.array(1j, dtype=jnp.complex64))
                e_eq = e_clean * phasor_inv  # (C,)

                W_new = W + step_size * jnp.einsum("i,jt->ijt", jnp.conj(e_eq), X_wins)

                carry_new = (
                    W_new,
                    pll_phi_new,
                    pll_freq_new,
                    bps_buf_new,
                    bps_buf_ptr_new,
                    bps_prev4_new,
                    cs_buf_x_new,
                    cs_buf_y_new,
                    cs_buf_ptr_new,
                )
                return carry_new, (y_fin, e_clean, W_new, phi_corr)

            n_sym = training_padded.shape[1]
            init_carry = (
                w_init,
                pll_phi_init,
                pll_freq_init,
                bps_buf_init,
                bps_buf_ptr_init,
                bps_prev4_init,
                cs_buf_x_init,
                cs_buf_y_init,
                cs_buf_ptr_init,
            )
            (
                (
                    W_final,
                    pll_phi_f,
                    pll_freq_f,
                    bps_buf_f,
                    bps_buf_ptr_f,
                    bps_prev4_f,
                    cs_buf_x_f,
                    cs_buf_y_f,
                    cs_buf_ptr_f,
                ),
                (y_hat, errors, w_hist, phi_traj),
            ) = jax.lax.scan(step, init_carry, jnp.arange(n_sym))
            return (
                y_hat,
                errors,
                W_final,
                w_hist,
                phi_traj,
                pll_phi_f,
                pll_freq_f,
                bps_buf_f,
                bps_buf_ptr_f,
                bps_prev4_f,
                cs_buf_x_f,
                cs_buf_y_f,
                cs_buf_ptr_f,
            )

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
    symmetry=4,
    sq_side=0,
    sq_lev_min=0.0,
    sq_d_grid=1.0,
):
    """JIT-compile and cache the RLS+CPR butterfly scan.

    Combines the Leaky-RLS Riccati update with an inline CPR tracker.
    Static parameters are identical to ``_get_jax_lms_cpr``.
    sq_side, sq_lev_min, sq_d_grid : O(1) square-QAM slicer parameters.
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
        int(symmetry),
        sq_side,
        float(sq_lev_min),
        float(sq_d_grid),
    )
    if key not in _JITTED_EQ:
        jax, jnp, _ = _get_jax()

        H = cs_history_len
        KB = bps_block_size
        import math as _math  # noqa: PLC0415

        _quantum_static = jnp.float64(2.0 * _math.pi / symmetry)

        _PREC = jax.lax.Precision.HIGHEST

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
            pll_phi_init,
            pll_freq_init,
            bps_buf_init,
            bps_buf_ptr_init,
            bps_prev4_init,
            cs_buf_x_init,
            cs_buf_y_init,
            cs_buf_ptr_init,
        ):
            def step(carry, idx):
                (
                    W,
                    P,
                    pll_phi,
                    pll_freq,
                    bps_buf,
                    bps_buf_ptr,
                    bps_prev4,
                    cs_buf_x,
                    cs_buf_y,
                    cs_buf_ptr,
                ) = carry
                sample_idx = idx * stride

                X_wins = jax.lax.dynamic_slice(
                    x_input, (0, sample_idx), (num_ch, num_taps)
                )
                y_raw = jnp.einsum(
                    "ijt,jt->i",
                    jnp.conj(W).astype(jnp.complex128),
                    X_wins.astype(jnp.complex128),
                    precision=_PREC,
                ).astype(
                    jnp.complex64
                )  # float64 accumulation → complex64, matches Numba

                if cpr_type == "pll":
                    phi_hat = pll_phi
                    bps_buf_new = bps_buf
                    bps_buf_ptr_new = bps_buf_ptr
                    bps_prev4_new = bps_prev4
                else:
                    slot = jnp.int32(bps_buf_ptr % KB)
                    bps_buf_new = jax.lax.dynamic_update_slice(
                        bps_buf,
                        y_raw[None, :],
                        (slot, jnp.int32(0)),
                    )
                    bps_buf_ptr_new = bps_buf_ptr + 1
                    fill = jnp.minimum(bps_buf_ptr_new, KB)

                    rotated = (
                        bps_phases_neg[:, None, None] * bps_buf_new[None, :, :]
                    )  # (B, KB, C)
                    if sq_side > 0:  # static branch at trace time
                        r_idx = jnp.clip(
                            jnp.round((rotated.real - sq_lev_min) / sq_d_grid).astype(
                                jnp.int32
                            ),
                            0,
                            sq_side - 1,
                        )
                        i_idx = jnp.clip(
                            jnp.round((rotated.imag - sq_lev_min) / sq_d_grid).astype(
                                jnp.int32
                            ),
                            0,
                            sq_side - 1,
                        )
                        r_near = sq_lev_min + r_idx.astype(jnp.float32) * jnp.float32(
                            sq_d_grid
                        )
                        i_near = sq_lev_min + i_idx.astype(jnp.float32) * jnp.float32(
                            sq_d_grid
                        )
                        d2_all = (rotated.real - r_near) ** 2 + (
                            rotated.imag - i_near
                        ) ** 2
                    else:
                        d2_all = jnp.min(
                            jnp.abs(
                                rotated[:, :, :, None]
                                - constellation[None, None, None, :]
                            )
                            ** 2,
                            axis=-1,
                        )
                    slot_mask = jnp.arange(KB)[None, :, None] < fill
                    metric = jnp.where(
                        slot_mask, d2_all.astype(jnp.float64), jnp.float64(0.0)
                    ).sum(axis=1)  # (B, C) float64

                    if bps_joint_channels:
                        best_k = jnp.argmin(metric.sum(axis=-1))
                        phi_raw = jnp.full(num_ch, bps_angles[best_k])
                    else:
                        best_k = jnp.argmin(metric, axis=0)
                        phi_raw = bps_angles[best_k]

                    # Causal 4-fold phase unwrap
                    raw4 = phi_raw.astype(jnp.float64) * jnp.float64(4.0)
                    diff4 = raw4 - bps_prev4
                    two_pi = jnp.float64(2.0 * 3.141592653589793)
                    diff4 = diff4 - jnp.round(diff4 / two_pi) * two_pi
                    bps_prev4_new = bps_prev4 + diff4
                    phi_hat = bps_prev4_new / jnp.float64(4.0)

                def correct_slip_ch(phi_h, buf_x_ch, buf_y_ch, ptr_ch):
                    # buf_x_ch is retained for call-site compat but unused here.
                    fill_cs = jnp.minimum(ptr_ch, H)
                    y_b = phi_h
                    mask = jnp.arange(H) < fill_cs
                    n_f = fill_cs.astype(jnp.float64)

                    # Sx and Sxx are exact closed-form constants — no cancellation.
                    Sx = n_f * (n_f - jnp.float64(1.0)) / jnp.float64(2.0)
                    Sxx = (
                        n_f
                        * (n_f - jnp.float64(1.0))
                        * (jnp.float64(2.0) * n_f - jnp.float64(1.0))
                        / jnp.float64(6.0)
                    )
                    denom = n_f * Sxx - Sx * Sx

                    # Sxy uses relative positions [0, fill_cs-1] derived from the
                    # circular buffer layout (oldest slot = ptr_ch % H → position 0).
                    oldest_slot = ptr_ch % H
                    slots = jnp.arange(H)
                    rel_pos_full = (slots - oldest_slot + H) % H
                    rel_pos = jnp.where(fill_cs >= H, rel_pos_full, slots).astype(
                        jnp.float64
                    )
                    Sy = jnp.where(mask, buf_y_ch, jnp.float64(0.0)).sum()
                    Sxy = jnp.where(mask, rel_pos * buf_y_ch, jnp.float64(0.0)).sum()

                    # Prediction target: one step past newest (relative pos = fill_cs).
                    x_pred = n_f

                    safe_denom = jnp.where(
                        jnp.abs(denom) > 1e-20, denom, jnp.float64(1.0)
                    )
                    slope = jnp.where(
                        fill_cs >= 10,
                        (n_f * Sxy - Sx * Sy) / safe_denom,
                        jnp.float64(0.0),
                    )
                    intercept = jnp.where(
                        fill_cs >= 10,
                        (Sy - slope * Sx) / jnp.maximum(n_f, jnp.float64(1.0)),
                        Sy / jnp.maximum(n_f, jnp.float64(1.0)),
                    )
                    phi_exp_lin = slope * x_pred + intercept
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
                    buf_y_new = jax.lax.dynamic_update_slice(
                        buf_y_ch, phi_corr[None], [write_pos]
                    )
                    ptr_new = ptr_ch + 1
                    return phi_corr, buf_x_ch, buf_y_new, ptr_new

                phi_corr, cs_buf_x_new, cs_buf_y_new, cs_buf_ptr_new = jax.vmap(
                    correct_slip_ch
                )(phi_hat, cs_buf_x, cs_buf_y, cs_buf_ptr)

                # Wrap to [-π, π] before casting to float32 for fast GPU exp
                _two_pi_f64 = jnp.float64(2.0 * 3.141592653589793)
                phi_wrapped = phi_corr - jnp.round(phi_corr / _two_pi_f64) * _two_pi_f64
                _phi32 = phi_wrapped.astype(jnp.float32)
                phasor = jnp.exp(_phi32 * jnp.array(-1j, dtype=jnp.complex64))
                y_fin = y_raw * phasor

                def slicer(ch_y):
                    if sq_side > 0:  # static branch at trace time
                        ir = jnp.clip(
                            jnp.round((ch_y.real - sq_lev_min) / sq_d_grid).astype(
                                jnp.int32
                            ),
                            0,
                            sq_side - 1,
                        )
                        ii = jnp.clip(
                            jnp.round((ch_y.imag - sq_lev_min) / sq_d_grid).astype(
                                jnp.int32
                            ),
                            0,
                            sq_side - 1,
                        )
                        nr = sq_lev_min + ir.astype(jnp.float32) * jnp.float32(
                            sq_d_grid
                        )
                        ni = sq_lev_min + ii.astype(jnp.float32) * jnp.float32(
                            sq_d_grid
                        )
                        return jax.lax.complex(nr, ni)
                    else:
                        return constellation[
                            jnp.argmin(jnp.abs(ch_y - constellation) ** 2)
                        ]

                dd = jax.vmap(slicer)(y_fin)
                d = jnp.where(idx < n_train, training_padded[:, idx], dd)
                e_clean = d - y_fin

                if cpr_type == "pll":
                    e_ph = y_fin.imag * d.real - y_fin.real * d.imag
                    if bps_joint_channels:  # static branch — shared LO
                        e_ph = jnp.full(num_ch, e_ph.mean())
                    pll_phi_new = pll_phi + pll_mu * e_ph + pll_freq
                    pll_freq_new = pll_freq + pll_beta * e_ph
                else:
                    pll_phi_new = pll_phi
                    pll_freq_new = pll_freq

                phasor_inv = jnp.exp(_phi32 * jnp.array(1j, dtype=jnp.complex64))
                e_eq = e_clean * phasor_inv

                x_bar = X_wins.flatten()
                Px = jnp.matmul(P, x_bar, precision=_PREC)
                denom_k = lam + jnp.real(jnp.dot(jnp.conj(x_bar), Px, precision=_PREC))
                k_gain = Px / denom_k

                def w_update(w_row, err_val):
                    w_flat = w_row.flatten()
                    w_flat_new = (1.0 - leakage) * w_flat + k_gain * jnp.conj(err_val)
                    return w_flat_new.reshape(num_ch, num_taps)

                W_upd = jax.vmap(w_update)(W, e_eq).astype(jnp.complex64)
                P_upd = (P - jnp.outer(k_gain, jnp.conj(Px))) / lam
                P_upd = 0.5 * (P_upd + jnp.conj(P_upd).T)

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
                    bps_prev4_new,
                    cs_buf_x_new,
                    cs_buf_y_new,
                    cs_buf_ptr_new,
                )
                return carry_new, (y_fin, e_clean, W_new, phi_corr)

            n_sym = training_padded.shape[1]
            init_carry = (
                w_init,
                P_init,
                pll_phi_init,
                pll_freq_init,
                bps_buf_init,
                bps_buf_ptr_init,
                bps_prev4_init,
                cs_buf_x_init,
                cs_buf_y_init,
                cs_buf_ptr_init,
            )
            (
                (
                    W_final,
                    _,
                    pll_phi_f,
                    pll_freq_f,
                    bps_buf_f,
                    bps_buf_ptr_f,
                    bps_prev4_f,
                    cs_buf_x_f,
                    cs_buf_y_f,
                    cs_buf_ptr_f,
                ),
                (y_hat, errors, w_hist, phi_traj),
            ) = jax.lax.scan(step, init_carry, jnp.arange(n_sym))
            return (
                y_hat,
                errors,
                W_final,
                w_hist,
                phi_traj,
                pll_phi_f,
                pll_freq_f,
                bps_buf_f,
                bps_buf_ptr_f,
                bps_prev4_f,
                cs_buf_x_f,
                cs_buf_y_f,
                cs_buf_ptr_f,
            )

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
            _P = jax.lax.Precision.HIGHEST

            def step(W, idx):
                sample_idx = idx * stride

                X_wins = jax.lax.dynamic_slice(
                    x_input, (0, sample_idx), (num_ch, num_taps)
                )
                y = jnp.einsum("ijt,jt->i", jnp.conj(W), X_wins, precision=_P)

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
            _P = jax.lax.Precision.HIGHEST

            def step(W, idx):
                sample_idx = idx * stride

                X_wins = jax.lax.dynamic_slice(
                    x_input, (0, sample_idx), (num_ch, num_taps)
                )
                y = jnp.einsum("ijt,jt->i", jnp.conj(W), X_wins, precision=_P)

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
                    acc_re = 0.0
                    acc_im = 0.0
                    for j in range(C):
                        for t in range(num_taps):
                            w = W[i, j, t]
                            x = X_wins[j, t]
                            acc_re += np.float64(w.real) * np.float64(
                                x.real
                            ) + np.float64(w.imag) * np.float64(x.imag)
                            acc_im += np.float64(w.real) * np.float64(
                                x.imag
                            ) - np.float64(w.imag) * np.float64(x.real)
                    y[i] = acc_re + 1j * acc_im
                    y_out[idx, i] = acc_re + 1j * acc_im

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
                    acc_re = 0.0
                    acc_im = 0.0
                    for j in range(C):
                        for t in range(num_taps):
                            w = W[i, j, t]
                            x = X_wins[j, t]
                            acc_re += np.float64(w.real) * np.float64(
                                x.real
                            ) + np.float64(w.imag) * np.float64(x.imag)
                            acc_im += np.float64(w.real) * np.float64(
                                x.imag
                            ) - np.float64(w.imag) * np.float64(x.real)
                    y[i] = acc_re + 1j * acc_im
                    y_out[idx, i] = acc_re + 1j * acc_im

                # Error: DA-LMS at pilots, RDE ring-directed at data
                for i in range(C):
                    if pilot_mask[idx]:
                        e[i] = (
                            y[i] - pilot_ref[i, idx]
                        )  # inverted to match blind subtractive update
                    else:
                        mod2 = y[i].real * y[i].real + y[i].imag * y[i].imag
                        abs_y = np.sqrt(mod2)
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
            _P = jax.lax.Precision.HIGHEST

            def step(W, xs_t):
                idx, p_ref, p_mask = xs_t  # (): int, (C,): cplx, (): bool
                X_wins = jax.lax.dynamic_slice(
                    x_input, (0, idx * stride), (num_ch, num_taps)
                )  # (C, T)
                y = jnp.einsum("ijt,jt->i", jnp.conj(W), X_wins, precision=_P)  # (C,)

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
            _P = jax.lax.Precision.HIGHEST

            def step(W, xs_t):
                idx, p_ref, p_mask = xs_t  # (): int, (C,): cplx, (): bool
                X_wins = jax.lax.dynamic_slice(
                    x_input, (0, idx * stride), (num_ch, num_taps)
                )  # (C, T)
                y = jnp.einsum("ijt,jt->i", jnp.conj(W), X_wins, precision=_P)  # (C,)

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


def _normalize_inputs(samples, training_symbols, sps, input_norm_factor=None):
    """Scale samples and training symbols to a common unit symbol-power reference.

    For fractionally-spaced equalization (sps > 1) the fractional timing phase
    is unknown.  Strided power measurement ``samples[..., ::sps]`` is unsafe
    because it can land on zero-crossings of the Nyquist pulse, severely
    underestimating signal power and destabilising adaptation.

    Instead the *wideband* (all-sample) power over the full signal is used.
    This is the most accurate power estimate and is independent of whether
    training symbols are provided.

    Parameters
    ----------
    samples          : (C, N) or (N,)  complex, any backend (NumPy / CuPy)
    training_symbols : (C, K) or (K,)  or None — always at 1 sps
    sps              : int — samples per symbol
    input_norm_factor : float or np.ndarray, optional
        When supplied, skip the RMS computation and divide samples by this
        factor instead.  Pass ``EqualizerResult.input_norm_factor`` from a
        previous call to keep successive blocks on the same power scale.
        ``float`` for SISO, ``(C,)`` array for MIMO.

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
    if input_norm_factor is not None:
        # Caller supplies scale — skip RMS and just apply it.
        nf = input_norm_factor
        if samples.ndim == 1:
            samples = samples / float(nf)
        else:
            nf_arr = np.asarray(nf, dtype=np.float64).ravel()
            if nf_arr.shape[0] != samples.shape[0]:
                raise ValueError(
                    f"input_norm_factor shape {nf_arr.shape} does not match "
                    f"samples channel count {samples.shape[0]}."
                )
            _, xp_loc, _ = dispatch(samples)
            nf_dev = xp_loc.asarray(nf_arr)[..., None]  # (C, 1) on same device
            samples = samples / nf_dev
        if training_symbols is not None:
            from commstools.helpers import normalize as c_normalize

            training_symbols = c_normalize(training_symbols, "average_power", axis=-1)
        return samples, training_symbols, input_norm_factor

    from commstools.helpers import rms as _rms

    ref_samples = samples

    norm_vec = _rms(ref_samples, axis=-1) * (sps**0.5)
    if samples.ndim == 1:
        input_norm_factor = float(norm_vec)
        samples = samples / float(norm_vec)
    else:
        input_norm_factor = to_device(norm_vec, "cpu")
        # Broadcast (C,) divisor over last axis
        samples = samples / norm_vec[..., None]

    if training_symbols is not None:
        from commstools.helpers import normalize as c_normalize

        # Training symbols are at 1 sps; "average_power" == "symbol_power" at sps=1.
        training_symbols = c_normalize(training_symbols, "average_power", axis=-1)

    return samples, training_symbols, input_norm_factor


def _build_padded_samples(
    samples_np, pad_left, pad_right, samples_prefix, pad_mode, eq_norm, sps
):
    """Construct the padded input array for the equalizer.

    When ``samples_prefix`` is supplied its last ``pad_left`` samples replace the
    leading zero-pad, eliminating the warm-start transient.  Otherwise the
    leading edge is filled according to ``pad_mode``.
    """
    if samples_prefix is not None:
        # Normalize prefix by the same factor used for the main block.
        prefix_np = np.ascontiguousarray(
            to_device(samples_prefix, "cpu"), dtype=np.complex64
        )
        if prefix_np.ndim == 1:
            prefix_np = prefix_np[np.newaxis, :]
        if prefix_np.shape[-1] < pad_left:
            raise ValueError(
                f"samples_prefix last axis length {prefix_np.shape[-1]} is less than "
                f"pad_left={pad_left}. Provide at least pad_left samples."
            )
        if eq_norm is not None:
            nf = eq_norm
            if prefix_np.shape[0] == 1:
                prefix_np = (
                    prefix_np / float(nf)
                    if np.ndim(nf) == 0
                    else prefix_np / float(np.asarray(nf).ravel()[0])
                )
            else:
                nf_arr = np.asarray(nf, dtype=np.float64).ravel()
                prefix_np = prefix_np / nf_arr[:, None]
        left_pad = prefix_np[:, -pad_left:]
        if samples_np.ndim == 1:
            samples_2d = samples_np[np.newaxis, :]
        else:
            samples_2d = samples_np
        right_zero = np.zeros((samples_2d.shape[0], pad_right), dtype=np.complex64)
        padded = np.concatenate([left_pad, samples_2d, right_zero], axis=-1)
        return padded
    if pad_mode == "zeros":
        if samples_np.ndim == 1:
            return np.pad(samples_np, (pad_left, pad_right))[np.newaxis, :]
        return np.pad(samples_np, ((0, 0), (pad_left, pad_right)))
    if pad_mode in ("edge", "reflect"):
        np_mode = pad_mode
        if samples_np.ndim == 1:
            return np.pad(samples_np, (pad_left, pad_right), mode=np_mode)[
                np.newaxis, :
            ]
        return np.pad(samples_np, ((0, 0), (pad_left, pad_right)), mode=np_mode)
    raise ValueError(
        f"pad_mode must be 'zeros', 'edge', or 'reflect'. Got {pad_mode!r}."
    )


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
):
    """Build the zero-padded training array expected by the JAX scan kernels.

    The kernels index ``training_padded[:, sym_idx]`` at every symbol,
    conditioned on ``sym_idx < n_train``.  Symbols beyond ``n_train_aligned``
    are zero — the kernel ignores them (DD slicer is used instead).

    If ``training_symbols`` is 1-D it is broadcast to all ``num_ch`` channels.
    Any extra training symbols beyond ``n_sym`` are silently clamped.
    The array is kept on the same device as ``training_symbols`` to avoid
    unnecessary CPU round-trips before the ``to_jax()`` transfer.

    Parameters
    ----------
    training_symbols : array or None — (K,) or (C, K), any backend
    num_ch           : int — C
    n_sym            : int — padded symbol count (columns of output array)

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
):
    """Build the zero-padded training array for the Numba scan kernels.

    Pure NumPy implementation — no JAX, CuPy, or ``dispatch`` dependencies.
    The caller must ensure ``training_symbols`` is already a NumPy array
    (use ``to_device(training_symbols, "cpu")`` before calling).

    Parameters
    ----------
    training_symbols : (K,) or (C, K) complex64 NumPy array, or None
    num_ch           : int — C
    n_sym            : int — symbol count (columns of output array)

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
    device: Optional[str] = "cpu",
    center_tap: Optional[int] = None,
    backend: str = "numba",
    w_init: Optional[ArrayType] = None,
    pmf: Optional[Any] = None,
    cpr_type: Optional[str] = None,
    cpr_pll_bandwidth: float = 1e-3,
    cpr_bps_test_phases: int = 64,
    cpr_bps_block_size: int = 32,
    cpr_joint_channels: bool = False,
    cpr_cycle_slip_correction: bool = False,
    cpr_cycle_slip_history: int = 100,
    cpr_cycle_slip_threshold: float = np.pi / 4,
    debug_plot: bool = False,
    plot_smoothing: int = 50,
    cpr_state: Optional["CPRState"] = None,
    input_norm_factor: Optional[Union[float, np.ndarray]] = None,
    samples_prefix: Optional[ArrayType] = None,
    pad_mode: str = "zeros",
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

    Algorithm (per symbol n)
    ------------------------
    1. **Sliding input window** — the length-T tap vector for output channel c
       is drawn from the padded input at the strided sample position:

       .. math::

           \\mathbf{x}_{c',n} = \\bigl[x_{c'}[n{\\cdot}\\mathrm{sps} - T_c],\\;
           \\ldots,\\; x_{c'}[n{\\cdot}\\mathrm{sps} - T_c + T - 1]\\bigr]

       where :math:`T_c` = ``center_tap`` (default :math:`T // 2`).  The
       causal delay :math:`T_c` is absorbed into the tap vector so the filter
       can model both pre- and post-cursor ISI.

    2. **Butterfly filter** — cross-correlate conjugate weights with the input
       across all C input channels:

       .. math::

           y_c^{\\mathrm{raw}}[n] = \\sum_{c'} \\mathbf{w}_{c,c'}^H \\mathbf{x}_{c',n}

    3. **Carrier phase recovery** (if ``cpr_type`` is set):

       * **PLL** — cross-product phase detector
         :math:`\\varphi_{\\mathrm{err}} = \\mathrm{Im}(y^{\\mathrm{raw}} \\cdot \\bar{d}_{\\mathrm{prev}})`
         drives a PI integrator with gains ``K_p``, ``K_i``; accumulated
         phase :math:`\\varphi_n` is applied as
         :math:`y[n] = y^{\\mathrm{raw}} \\cdot e^{-j\\varphi_n}`.
       * **BPS** — :math:`B` candidate rotations :math:`e^{-jk\\pi/(2B)}` are
         tested; the one minimising the summed nearest-constellation distance
         over the trailing :math:`K` = ``cpr_bps_block_size`` symbols is chosen.
         A causal 4-fold unwrap converts the ``[0, π/2)`` argmin to full-range
         :math:`\\varphi_n` stored in a float64 accumulator.

    4. **Decision** — training symbol ``d[n]`` (DA phase, while
       ``n < len(training_symbols)``) or nearest-constellation hard decision on
       ``y[n]`` (DD phase thereafter).

    5. **Error and tap-plane back-rotation**:

       .. math::

           e_{\\mathrm{clean}}[n] = d[n] - y[n], \\qquad
           e_{\\mathrm{taps}}[n] = e_{\\mathrm{clean}}[n] \\cdot e^{+j\\varphi_n}

       The back-rotation undoes the CPR correction so the gradient operates in
       the original tap space.

    6. **LMS weight update** (plain, no input-power normalisation):

       .. math::

           \\mathbf{w}_{c,c'} \\mathrel{+}= \\mu \\cdot
           \\overline{e_{\\mathrm{taps},c}[n]} \\cdot \\mathbf{x}_{c',n}

       Stability bound: :math:`0 < \\mu < 2 / (C \\cdot T \\cdot P_x)` where
       :math:`P_x` is the mean per-tap input power.  The equalizer normalises
       inputs to unit symbol-rate power before adaptation, so :math:`P_x \\approx 1`.

    7. **Cycle-slip correction** (if ``cpr_cycle_slip_correction=True``) — a
       circular buffer of ``cpr_cycle_slip_history`` past phase values is
       maintained per channel.  An online least-squares linear fit over the
       buffer predicts :math:`\\hat{\\varphi}_n`.  If
       :math:`|\\varphi_n - \\hat{\\varphi}_n| > ` ``cpr_cycle_slip_threshold``,
       :math:`\\varphi_n` is snapped to the nearest :math:`2\\pi/\\mathrm{sym}`
       multiple (``sym`` = constellation symmetry order, 4 for QAM/QPSK); the
       corrected value replaces :math:`\\varphi_n` in steps 5 and 6 and is
       written into the history buffer.

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

          **Dual-path output:** the wrapped float32 phase estimate rotates
          ``y_raw`` for the weight update, while the unwrapped float64
          accumulator is stored in ``phase_trajectory``.  The two paths are
          kept separate to prevent float32 rounding errors from accumulating
          in the trajectory over thousands of symbols.

          **4-fold causal unwrap:** the raw BPS ``argmin`` lives in
          ``[0, π/2)``.  A causal unwrap tracks the argmin evolution symbol
          by symbol, adding or subtracting ``π/2`` multiples as needed to
          keep the estimate continuous, then converting to full-range radians.
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
    cpr_joint_channels : bool, default False
        For MIMO inputs (C > 1): if ``True``, the per-symbol phase estimate
        is computed jointly across all C channels before being applied.

        * **BPS** — the per-candidate distance metrics are summed across
          channels before ``argmin``, giving one shared estimate broadcast
          to all channels.  Reduces estimation variance by ~√C for
          shared-LO transmitter setups.
        * **PLL** — the cross-product phase-error signal
          ``Im(y · conj(d))`` is averaged across channels before the PI
          integrator, so all channels share one phase trajectory.

        When ``False``, each channel runs its own independent estimator.
        Ignored for SISO inputs (C == 1).
    cpr_cycle_slip_correction : bool, default False
        Enable causal cycle-slip detection and correction.  A circular
        buffer of ``cpr_cycle_slip_history`` past phase estimates is
        maintained per channel.  Before each symbol, an online linear
        regression over the buffer predicts the current phase; if the new
        estimate deviates from the prediction by more than
        ``cpr_cycle_slip_threshold``, the estimate is snapped to the
        nearest ``2π/symmetry`` quantum (e.g. π/2 for QPSK/QAM) and the
        buffer is updated with the corrected value.  Disable for
        deterministic parity checks or when the channel is known to be
        slip-free.
    cpr_cycle_slip_history : int, default 100
        Length of the phase-history buffer used for cycle-slip prediction.
        Longer buffers produce a more stable linear-trend fit but are
        slower to adapt when the true carrier frequency drifts.  Ignored
        when ``cpr_cycle_slip_correction=False``.
    cpr_cycle_slip_threshold : float, default π/4
        Maximum tolerated deviation (radians) between the predicted and
        observed phase before a slip is declared.  Should be set to half
        the constellation's angular symmetry quantum (``π/4`` for
        QPSK/QAM).  Ignored when ``cpr_cycle_slip_correction=False``.
    debug_plot : bool, default False
        If True, display a convergence + tap-weight diagnostic plot on exit.
    plot_smoothing : int, default 50
        Moving-average window (symbols) for the MSE convergence curve in the
        debug plot.
    cpr_state : CPRState, optional
        Warm-start CPR state from a previous ``lms()`` call (obtained via
        ``EqualizerResult.cpr_state``).  When provided and the CPR type and
        channel count match, the PLL integrators, BPS unwrap accumulators,
        and cycle-slip buffers are pre-loaded rather than zero-initialized.
        This eliminates the ~5-10 k symbol CPR convergence transient that
        occurs at every block boundary in streaming pipelines.  Pass
        ``None`` (default) to cold-start the CPR from zero.  Ignored when
        ``cpr_type=None`` or when the stored state is incompatible (mismatched
        ``cpr_type``, channel count, or history depth), in which case the
        equalizer falls back to cold-start silently.

        .. note::
            JAX backend: ``cpr_state`` warm-start is not yet supported;
            passing a non-``None`` value raises ``NotImplementedError``.
    input_norm_factor : float or ndarray, optional
        Pre-computed RMS normalization factor from a previous call (obtained
        via ``EqualizerResult.input_norm_factor``).  When provided, the
        ``_normalize_inputs`` step is skipped and this value is used directly
        to scale the input samples and training symbols.  This ensures that
        warm-started weight vectors see the same amplitude regime as the
        block on which they were trained, preventing a gradient scale mismatch
        when signal power drifts slowly between blocks.
        Pass ``None`` (default) to recompute the RMS from the current block.
    samples_prefix : array_like, optional
        Signal history from the end of the previous block, used to eliminate
        the zero-padded leading transient at each block boundary.  Shape:
        ``(≥ pad_left,)`` SISO or ``(C, ≥ pad_left)`` MIMO, where
        ``pad_left = min(center_tap, max(0, num_taps - 1))``.  The last
        ``pad_left`` samples of ``samples_prefix`` replace the leading zeros
        in the tap window so that the first output symbol sees a fully
        populated, real-signal tap vector.  The prefix is normalized by the
        same ``input_norm_factor`` as the main block before being prepended.
        Pass ``None`` (default) for standard zero-padding.  Raises
        ``ValueError`` if the prefix length is less than ``pad_left``.
    pad_mode : {'zeros', 'edge'}, default 'zeros'
        Padding strategy for the leading tap window when ``samples_prefix``
        is ``None``.  ``'zeros'`` (default) prepends ``pad_left`` complex
        zeros, which is the standard causal initialisation.  ``'edge'``
        replicates the first sample of the current block, which can reduce
        the initial amplitude jump at cold start.  Has no effect when
        ``samples_prefix`` is provided.

    Returns
    -------
    EqualizerResult
        Result container with the following fields:

        * ``y_hat`` — equalized symbol estimates, shape ``(N_sym,)`` SISO
          or ``(C, N_sym)`` MIMO, at 1 SPS (symbol rate).
        * ``weights`` — final tap-weight tensor, shape ``(num_taps,)`` SISO
          or ``(C, C, num_taps)`` MIMO.
        * ``error`` — complex error signal ``e[n] = d[n] - y[n]``, same
          shape as ``y_hat``.
        * ``weights_history`` — tap weights recorded at each symbol (only
          when ``store_weights=True``); shape ``(N_sym, num_taps)`` SISO or
          ``(N_sym, C, C, num_taps)`` MIMO.  ``None`` otherwise.
        * ``phase_trajectory`` — accumulated per-symbol phase estimates,
          shape ``(N_sym,)`` SISO or ``(C, N_sym)`` MIMO.  For BPS, this
          is the causal 4-fold-unwrapped float64 phase.  For PLL, it is
          the PI integrator state accumulated over all symbols.  ``None``
          when ``cpr_type=None``.
        * ``num_train_symbols`` — number of training symbols consumed
          (data-aided phase).
        * ``input_norm_factor`` — the RMS factor used to normalize inputs
          (float).  Store and pass as ``input_norm_factor`` on the next call
          to keep weight magnitudes consistent across block boundaries.
        * ``cpr_state`` — ``CPRState`` snapshot of PLL/BPS/cycle-slip
          integrators after the last symbol.  Pass as ``cpr_state`` on the
          next call to resume CPR without a re-convergence transient.
          ``None`` when ``cpr_type=None``.

        Arrays reside on the same device as the input (NumPy CPU or CuPy
        GPU).

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

    n_train_log = training_symbols.shape[-1] if training_symbols is not None else 0
    logger.info(
        f"LMS equalizer: num_taps={num_taps}, mu={step_size}, sps={sps}, "
        f"backend={backend}, n_train={n_train_log}"
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

    if training_symbols is not None:
        training_symbols, _, _ = dispatch(training_symbols)

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
            samples_np, training_np, sps, input_norm_factor=input_norm_factor
        )
        # Pad (NumPy)
        samples_padded = _build_padded_samples(
            samples_np, pad_left, pad_right, samples_prefix, pad_mode, eq_norm, sps
        )
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
        )
        _sq_side, _sq_lev_min, _sq_d_grid = _sq_qam_slicer_params(constellation_np)
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
                _sq_lev_min,
                _sq_d_grid,
                np.int32(_sq_side),
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
            _st = cpr_state
            _st_ok = (
                _st is not None
                and _st.cpr_type == cpr_type
                and _st.num_ch == num_ch
                and _st.cs_H == H
                and _st.pll_phi is not None
            )
            pll_phi = (
                _st.pll_phi.copy() if _st_ok else np.zeros(num_ch, dtype=np.float64)
            )
            pll_freq = (
                _st.pll_freq.copy() if _st_ok else np.zeros(num_ch, dtype=np.float64)
            )
            cs_buf_x = (
                _st.cs_buf_x.copy()
                if _st_ok
                else np.zeros((num_ch, H), dtype=np.float64)
            )
            cs_buf_y = (
                _st.cs_buf_y.copy()
                if _st_ok
                else np.zeros((num_ch, H), dtype=np.float64)
            )
            cs_buf_ptr = (
                _st.cs_buf_ptr.copy() if _st_ok else np.zeros(num_ch, dtype=np.int64)
            )
            cs_buf_n = (
                _st.cs_buf_n.copy() if _st_ok else np.zeros(num_ch, dtype=np.int64)
            )
            cs_stats = (
                _st.cs_stats.copy()
                if _st_ok
                else np.zeros((num_ch, 4), dtype=np.float64)
            )
            bps_prev4 = (
                _st.bps_prev4.copy()
                if (_st_ok and _st.bps_prev4 is not None)
                else np.zeros(num_ch, dtype=np.float64)
            )
            phase_out = np.empty((n_sym, num_ch), dtype=np.float64)
            cpr_mode_int = np.int32(1 if cpr_type == "pll" else 2)
            _get_numba_lms_cpr()(
                samples_padded,
                train_full,
                constellation_np,
                bps_phases_neg_np,
                bps_angles_np,
                np.int32(cpr_bps_block_size),
                bool(cpr_joint_channels),
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
                bps_prev4,
                y_out,
                e_out,
                phase_out,
                w_hist_buf,
                _sq_lev_min,
                _sq_d_grid,
                np.int32(_sq_side),
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
            result.cpr_state = CPRState(
                pll_phi=pll_phi.copy(),
                pll_freq=pll_freq.copy(),
                bps_prev4=bps_prev4.copy(),
                cs_buf_x=cs_buf_x.copy(),
                cs_buf_y=cs_buf_y.copy(),
                cs_buf_ptr=cs_buf_ptr.copy(),
                cs_buf_n=cs_buf_n.copy(),
                cs_stats=cs_stats.copy(),
                cpr_type=cpr_type,
                num_ch=num_ch,
                symmetry=symmetry,
                bps_P=B,
                bps_K=int(cpr_bps_block_size),
                cs_H=H,
            )
        return _log_equalizer_exit(
            result, name="LMS", debug_plot=debug_plot, plot_smoothing=plot_smoothing
        )

    # JAX backend
    jax, jnp, _ = _get_jax()
    if jax is None:
        raise ImportError("JAX is required for backend='jax'.")

    samples, training_symbols, eq_norm = _normalize_inputs(
        samples, training_symbols, sps, input_norm_factor=input_norm_factor
    )
    # Pad — use _build_padded_samples (returns CPU NumPy); convert to xp array after
    _samp_cpu = to_device(samples, "cpu").astype(np.complex64)
    samples_padded_np = _build_padded_samples(
        _samp_cpu, pad_left, pad_right, samples_prefix, pad_mode, eq_norm, sps
    )
    samples_padded = (
        xp.asarray(samples_padded_np)
        if not was_1d
        else xp.asarray(samples_padded_np[0])
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
        _sq_side_j, _sq_lev_min_j, _sq_d_grid_j = _sq_qam_slicer_params(
            constellation_np
        )
        scan_fn = _get_jax_lms(
            num_taps,
            stride,
            len(constellation_np),
            num_ch,
            int(_sq_side_j),
            float(_sq_lev_min_j),
            float(_sq_d_grid_j),
        )
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
        x64_enabled = (
            jax.config.jax_enable_x64
            if hasattr(jax.config, "jax_enable_x64")
            else jax.config.read("jax_enable_x64")
        )
        if not x64_enabled:
            raise RuntimeError(
                "JAX x64 mode must be enabled for CPR phase tracking: "
                "call jax.config.update('jax_enable_x64', True) before using "
                "backend='jax' with cpr_type set."
            )
        pll_mu, pll_beta = _cpr_pll_gains(cpr_pll_bandwidth)
        symmetry = _cpr_symmetry(modulation, order)
        B = int(cpr_bps_test_phases)
        H = int(cpr_cycle_slip_history)
        bps_angles_np = np.linspace(
            0.0, np.pi / 2.0, B, endpoint=False, dtype=np.float32
        )
        bps_phases_neg_np = np.exp(-1j * bps_angles_np).astype(np.complex64)
        bps_pn_jax = to_jax(bps_phases_neg_np, device=platform)
        bps_ang_jax = to_jax(bps_angles_np, device=platform)
        _sq_side_j, _sq_lev_min_j, _sq_d_grid_j = _sq_qam_slicer_params(
            constellation_np
        )
        scan_fn = _get_jax_lms_cpr(
            num_taps,
            stride,
            len(constellation_np),
            num_ch,
            cpr_type,
            B,
            int(cpr_bps_block_size),
            bool(cpr_joint_channels),
            H,
            int(symmetry),
            int(_sq_side_j),
            float(_sq_lev_min_j),
            float(_sq_d_grid_j),
        )
        KB = int(cpr_bps_block_size)
        if cpr_state is not None:
            _pi, _pf, _bb, _bbp, _bp4, _cx, _cy, _cp = _cpr_state_to_jax_inits(
                cpr_state, num_ch, KB, H
            )
        else:
            _pi = np.zeros(num_ch, dtype=np.float64)
            _pf = np.zeros(num_ch, dtype=np.float64)
            _bb = np.zeros((KB, num_ch), dtype=np.complex64)
            _bbp = np.int32(0)
            _bp4 = np.zeros(num_ch, dtype=np.float64)
            _cx = np.zeros((num_ch, H), dtype=np.float64)
            _cy = np.zeros((num_ch, H), dtype=np.float64)
            _cp = np.zeros(num_ch, dtype=np.int32)
        (
            y_jax,
            e_jax,
            W_jax,
            wh_jax,
            phi_jax,
            pll_phi_f,
            pll_freq_f,
            bps_buf_f,
            bps_buf_ptr_f,
            bps_prev4_f,
            cs_buf_x_f,
            cs_buf_y_f,
            cs_buf_ptr_f,
        ) = scan_fn(
            x_jax,
            train_jax,
            const_jax,
            bps_pn_jax,
            bps_ang_jax,
            W_jax,
            mu_jax,
            n_train_jax,
            to_jax(jnp.float64(pll_mu), device=platform),
            to_jax(jnp.float64(pll_beta), device=platform),
            to_jax(jnp.float64(cpr_cycle_slip_threshold), device=platform),
            to_jax(jnp.bool_(cpr_cycle_slip_correction), device=platform),
            to_jax(_pi, device=platform),
            to_jax(_pf, device=platform),
            to_jax(_bb, device=platform),
            to_jax(_bbp, device=platform),
            to_jax(_bp4, device=platform),
            to_jax(_cx, device=platform),
            to_jax(_cy, device=platform),
            to_jax(_cp, device=platform),
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
        result.cpr_state = CPRState(
            pll_phi=np.asarray(from_jax(pll_phi_f)),
            pll_freq=np.asarray(from_jax(pll_freq_f)),
            bps_prev4=np.asarray(from_jax(bps_prev4_f)),
            jax_bps_buf=np.asarray(from_jax(bps_buf_f)),
            jax_bps_buf_ptr=int(np.asarray(from_jax(bps_buf_ptr_f))),
            cs_buf_x=np.asarray(from_jax(cs_buf_x_f)),
            cs_buf_y=np.asarray(from_jax(cs_buf_y_f)),
            cs_buf_ptr=np.asarray(from_jax(cs_buf_ptr_f)),
            cpr_type=cpr_type,
            num_ch=num_ch,
            symmetry=symmetry,
            bps_P=B,
            bps_K=KB,
            cs_H=H,
        )
    return _log_equalizer_exit(result, name="LMS", debug_plot=debug_plot)


def _check_rls_divergence(weights, xp, forgetting_factor, delta):
    """Raise if RLS produced non-finite weights (silent divergence guard).

    Mirrors the ``_div_flag`` check in ``block_lms``: a single device→host sync on
    the assembled weights catches loss of positive-definiteness in the inverse
    correlation matrix P (which surfaces as NaN/Inf taps) and converts it into an
    actionable error instead of returning garbage weights.
    """
    if not bool(xp.isfinite(weights).all()):
        raise RuntimeError(
            f"RLS equalizer diverged (forgetting_factor={forgetting_factor}, "
            f"delta={delta}). RLS requires a positive-definite correlation matrix. "
            "Try increasing regularization 'delta', reducing 'forgetting_factor', "
            "or adding 'leakage' (e.g. 1e-4) to stabilize fractionally-spaced inputs."
        )


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
    device: Optional[str] = "cpu",
    center_tap: Optional[int] = None,
    backend: str = "numba",
    w_init: Optional[ArrayType] = None,
    pmf: Optional[Any] = None,
    cpr_type: Optional[str] = None,
    cpr_pll_bandwidth: float = 1e-3,
    cpr_bps_test_phases: int = 64,
    cpr_bps_block_size: int = 32,
    cpr_joint_channels: bool = False,
    cpr_cycle_slip_correction: bool = False,
    cpr_cycle_slip_history: int = 100,
    cpr_cycle_slip_threshold: float = np.pi / 4,
    debug_plot: bool = False,
    plot_smoothing: int = 50,
    cpr_state: Optional["CPRState"] = None,
    input_norm_factor: Optional[Union[float, np.ndarray]] = None,
    samples_prefix: Optional[ArrayType] = None,
    pad_mode: str = "zeros",
) -> EqualizerResult:
    """
    Recursive Least Squares adaptive equalizer with butterfly MIMO support.

    RLS converges faster than LMS at the cost of higher per-symbol
    complexity (O(num_taps²) for the rank-1 Riccati update vs O(num_taps)
    for LMS).  It maintains an inverse correlation matrix P per output stream.

    Algorithm (per symbol n)
    ------------------------
    Steps 1–5 and 7 are identical to :func:`lms` (input windowing, butterfly
    filter output, carrier phase recovery, decision, error + tap-plane
    back-rotation, and cycle-slip correction).  Step 6 replaces the plain LMS
    gradient with a rank-1 Riccati update:

    6. **RLS weight update** — for each output channel c, maintaining the
       inverse input auto-correlation matrix
       :math:`P_c \\in \\mathbb{C}^{T \\times T}`:

       .. math::

           \\mathbf{k}_c &= \\frac{P_c\\,\\mathbf{x}_{c,n}}{\\lambda +
           \\mathbf{x}_{c,n}^H P_c\\,\\mathbf{x}_{c,n}}

           P_c &\\leftarrow \\frac{P_c - \\mathbf{k}_c\\,\\mathbf{x}_{c,n}^H P_c}{\\lambda}

           \\mathbf{w}_{c,c'} &\\mathrel{+}= \\mathbf{k}_c \\cdot
           \\overline{e_{\\mathrm{taps},c}[n]}

       where :math:`\\lambda` = ``forgetting_factor``.  :math:`P_c` is
       initialised to :math:`(1/\\delta)\\,\\mathbf{I}` (``delta`` parameter).
       With ``leakage`` :math:`\\gamma > 0` the weight update becomes
       :math:`\\mathbf{w} \\leftarrow (1-\\gamma)\\mathbf{w} +
       \\mathbf{k}\\cdot\\bar{e}_{\\mathrm{taps}}`, which exponentially suppresses
       tap energy in frequency-null subspaces and prevents the
       eigenvalue blow-up that afflicts :math:`P` for fractionally-spaced
       (sps > 1) inputs.

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
        Inline carrier phase recovery applied jointly with weight updates at
        every symbol.  ``None`` disables CPR (default).

        * ``'pll'`` — 2nd-order decision-directed PLL.  The cross-product
          detector ``Im(y · conj(d))`` drives a PI loop with gains from
          ``cpr_pll_bandwidth``.  Low noise floor; suited to QPSK–64-QAM.
        * ``'bps'`` — Blind Phase Search over ``cpr_bps_test_phases``
          candidate angles in ``[0, π/2)``, averaged over a causal window
          of ``cpr_bps_block_size`` samples.  Uses a **dual-path** design:
          the wrapped float32 estimate rotates ``y_raw`` for the Riccati
          update; the unwrapped float64 accumulator populates
          ``phase_trajectory``.  A **causal 4-fold unwrap** converts the
          ``[0, π/2)`` argmin to full-range radians symbol by symbol.
    cpr_pll_bandwidth : float, default 1e-3
        Normalised loop bandwidth ``B_L · T_s`` for the PLL.  Gains are
        ``K_p = 4 B_L``, ``K_i = 4 B_L²`` (critically-damped).  Typical
        range: ``5e-4`` (low phase noise) to ``5e-3`` (high drift).
        Ignored when ``cpr_type != 'pll'``.
    cpr_bps_test_phases : int, default 64
        Number of BPS candidate phase angles in ``[0, π/2)``.  32-64 is
        sufficient for ≤ 16-QAM; use 64-128 for 64-QAM.  Ignored when
        ``cpr_type != 'bps'``.
    cpr_bps_block_size : int, default 32
        Trailing-window length (symbols) summed before the BPS ``argmin``.
        Larger values reduce phase-noise variance at the cost of latency.
        ``cpr_bps_block_size=1`` gives single-symbol BPS.  Ignored when
        ``cpr_type != 'bps'``.
    cpr_joint_channels : bool, default False
        For MIMO inputs (C > 1): share the phase estimate across channels.

        * **BPS** — distance metrics are summed across channels before
          ``argmin``.  Reduces estimation variance by ~√C for shared-LO
          systems.
        * **PLL** — the phase-error signal is averaged across channels
          before the PI integrator.

        When ``False``, each channel has an independent estimator.
        Ignored for SISO inputs.
    cpr_cycle_slip_correction : bool, default False
        Enable causal cycle-slip detection and correction.  A circular
        buffer of ``cpr_cycle_slip_history`` past phase estimates is kept
        per channel.  An online linear regression predicts the next phase;
        if the new estimate deviates by more than
        ``cpr_cycle_slip_threshold``, it is snapped to the nearest
        ``2π/symmetry`` quantum and the buffer is updated.  Disable for
        parity checks or slip-free channels.
    cpr_cycle_slip_history : int, default 100
        Phase-history buffer length for slip prediction.  Longer buffers
        give a more stable trend estimate but adapt more slowly to genuine
        frequency steps.  Ignored when ``cpr_cycle_slip_correction=False``.
    cpr_cycle_slip_threshold : float, default π/4
        Maximum tolerated deviation (radians) before a slip is declared.
        Set to half the constellation's angular quantum (``π/4`` for
        QPSK/QAM).  Ignored when ``cpr_cycle_slip_correction=False``.
    debug_plot : bool, default False
        Display convergence + tap-weight diagnostic plot on exit.
    plot_smoothing : int, default 50
        MSE moving-average window for the debug plot.
    cpr_state : CPRState, optional
        Warm-start CPR state from a previous ``rls()`` call.  See
        ``lms()`` for the full description; behaviour is identical.
    input_norm_factor : float or ndarray, optional
        Pre-computed RMS normalization factor from a previous call.  See
        ``lms()`` for the full description; behaviour is identical.
    samples_prefix : array_like, optional
        Signal history from the end of the previous block.  See ``lms()``
        for the full description; behaviour is identical.
    pad_mode : {'zeros', 'edge'}, default 'zeros'
        Padding strategy when ``samples_prefix`` is ``None``.  See
        ``lms()`` for the full description; behaviour is identical.

    Returns
    -------
    EqualizerResult
        Result container with the following fields:

        * ``y_hat`` — equalized symbol estimates, shape ``(N_sym,)`` SISO
          or ``(C, N_sym)`` MIMO, at 1 SPS.
        * ``weights`` — final tap-weight tensor, shape ``(num_taps,)`` SISO
          or ``(C, C, num_taps)`` MIMO.
        * ``error`` — complex error signal ``e[n] = d[n] − y[n]``, same
          shape as ``y_hat``.
        * ``weights_history`` — tap weights at each symbol when
          ``store_weights=True``; ``None`` otherwise.
        * ``phase_trajectory`` — per-symbol phase estimates, shape
          ``(N_sym,)`` SISO or ``(C, N_sym)`` MIMO.  BPS: causal
          4-fold-unwrapped float64.  PLL: PI integrator state.  ``None``
          when ``cpr_type=None``.
        * ``num_train_symbols`` — number of data-aided training symbols.
        * ``input_norm_factor`` — RMS factor used to normalize inputs.
        * ``cpr_state`` — CPRState snapshot after the last symbol; ``None``
          when ``cpr_type=None``.

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

    n_train_log = training_symbols.shape[-1] if training_symbols is not None else 0
    logger.info(
        f"RLS equalizer: num_taps={num_taps}, forgetting_factor={forgetting_factor}, "
        f"delta={delta:.2e}, leakage={leakage:.2e}, sps={sps}, "
        f"backend={backend}, n_train={n_train_log}"
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
            samples_np, training_np, sps, input_norm_factor=input_norm_factor
        )

        x_np = _build_padded_samples(
            samples_np, pad_left, pad_right, samples_prefix, pad_mode, eq_norm, sps
        )
        x_np = np.ascontiguousarray(x_np)

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
        )
        _sq_side, _sq_lev_min, _sq_d_grid = _sq_qam_slicer_params(constellation_np)
        if w_init is not None:
            w_arr = np.ascontiguousarray(to_device(w_init, "cpu"), dtype=np.complex64)
            w_arr = _validate_w_init(w_arr, num_ch, num_taps)
            W = w_arr.copy()
        else:
            W = _init_butterfly_weights_numpy(num_ch, num_taps, center_tap=center_tap)
        regressor_dim = num_ch * num_taps
        P = np.eye(regressor_dim, dtype=np.complex128) / np.float64(delta)
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
                _sq_lev_min,
                _sq_d_grid,
                np.int32(_sq_side),
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
            _st = cpr_state
            _st_ok = (
                _st is not None
                and _st.cpr_type == cpr_type
                and _st.num_ch == num_ch
                and _st.cs_H == H
                and _st.pll_phi is not None
            )
            pll_phi = (
                _st.pll_phi.copy() if _st_ok else np.zeros(num_ch, dtype=np.float64)
            )
            pll_freq = (
                _st.pll_freq.copy() if _st_ok else np.zeros(num_ch, dtype=np.float64)
            )
            cs_buf_x = (
                _st.cs_buf_x.copy()
                if _st_ok
                else np.zeros((num_ch, H), dtype=np.float64)
            )
            cs_buf_y = (
                _st.cs_buf_y.copy()
                if _st_ok
                else np.zeros((num_ch, H), dtype=np.float64)
            )
            cs_buf_ptr = (
                _st.cs_buf_ptr.copy() if _st_ok else np.zeros(num_ch, dtype=np.int64)
            )
            cs_buf_n = (
                _st.cs_buf_n.copy() if _st_ok else np.zeros(num_ch, dtype=np.int64)
            )
            cs_stats = (
                _st.cs_stats.copy()
                if _st_ok
                else np.zeros((num_ch, 4), dtype=np.float64)
            )
            bps_prev4 = (
                _st.bps_prev4.copy()
                if (_st_ok and _st.bps_prev4 is not None)
                else np.zeros(num_ch, dtype=np.float64)
            )
            phase_out = np.empty((n_sym, num_ch), dtype=np.float64)
            cpr_mode_int = np.int32(1 if cpr_type == "pll" else 2)
            _get_numba_rls_cpr()(
                x_np,
                train_full,
                constellation_np,
                bps_phases_neg_np,
                bps_angles_np,
                np.int32(cpr_bps_block_size),
                bool(cpr_joint_channels),
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
                bps_prev4,
                y_out,
                e_out,
                phase_out,
                w_hist_buf,
                _sq_lev_min,
                _sq_d_grid,
                np.int32(_sq_side),
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
            result.cpr_state = CPRState(
                pll_phi=pll_phi.copy(),
                pll_freq=pll_freq.copy(),
                bps_prev4=bps_prev4.copy(),
                cs_buf_x=cs_buf_x.copy(),
                cs_buf_y=cs_buf_y.copy(),
                cs_buf_ptr=cs_buf_ptr.copy(),
                cs_buf_n=cs_buf_n.copy(),
                cs_stats=cs_stats.copy(),
                cpr_type=cpr_type,
                num_ch=num_ch,
                symmetry=symmetry,
                bps_P=B,
                bps_K=int(cpr_bps_block_size),
                cs_H=H,
            )
        # Truncate last num_taps//2 symbols (zero-padding contamination).
        result = _log_equalizer_exit(
            result, name="RLS", debug_plot=debug_plot, plot_smoothing=plot_smoothing
        )
        result.tail_trim = tail_trim
        _check_rls_divergence(result.weights, xp, forgetting_factor, delta)
        return result

    # JAX backend
    jax, jnp, _ = _get_jax()
    if jax is None:
        raise ImportError("JAX is required for backend='jax'.")
    x64_enabled = (
        jax.config.jax_enable_x64
        if hasattr(jax.config, "jax_enable_x64")
        else jax.config.read("jax_enable_x64")
    )
    if not x64_enabled:
        raise RuntimeError(
            "JAX x64 mode must be enabled for RLS: the P (Riccati) matrix requires "
            "complex128 precision to remain positive-definite. "
            "Call jax.config.update('jax_enable_x64', True) before using backend='jax'."
        )

    samples, training_symbols, eq_norm = _normalize_inputs(
        samples, training_symbols, sps, input_norm_factor=input_norm_factor
    )

    _samp_cpu_rls = to_device(samples, "cpu").astype(np.complex64)
    samples_padded_np_rls = _build_padded_samples(
        _samp_cpu_rls, pad_left, pad_right, samples_prefix, pad_mode, eq_norm, sps
    )
    samples_padded = (
        xp.asarray(samples_padded_np_rls)
        if not was_1d
        else xp.asarray(samples_padded_np_rls[0])
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
        f"RLS internals: n_sym={n_sym}, n_train={n_train_log}, "
        f"n_update_halt={n_update_halt}, leakage={leakage:.2e}, delta={delta:.2e}"
    )

    train_full, n_train_aligned = _prepare_training_jax(
        training_symbols,
        num_ch,
        n_sym,
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
    P_init = jnp.eye(regressor_dim, dtype="complex128") / delta
    P_init = to_jax(P_init, device=platform)
    lam_jax = to_jax(jnp.float32(forgetting_factor), device=platform)
    n_train_jax = to_jax(jnp.int32(n_train_aligned), device=platform)
    leakage_jax = to_jax(jnp.float32(leakage), device=platform)
    n_update_halt_jax = to_jax(jnp.int32(n_update_halt), device=platform)

    if cpr_type is None:
        _sq_side_j, _sq_lev_min_j, _sq_d_grid_j = _sq_qam_slicer_params(
            constellation_np
        )
        scan_fn = _get_jax_rls(
            num_taps,
            stride,
            len(constellation_np),
            num_ch,
            int(_sq_side_j),
            float(_sq_lev_min_j),
            float(_sq_d_grid_j),
        )
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
        symmetry = _cpr_symmetry(modulation, order)
        B = int(cpr_bps_test_phases)
        H = int(cpr_cycle_slip_history)
        bps_angles_np = np.linspace(
            0.0, np.pi / 2.0, B, endpoint=False, dtype=np.float32
        )
        bps_phases_neg_np = np.exp(-1j * bps_angles_np).astype(np.complex64)
        bps_pn_jax = to_jax(bps_phases_neg_np, device=platform)
        bps_ang_jax = to_jax(bps_angles_np, device=platform)
        _sq_side_j, _sq_lev_min_j, _sq_d_grid_j = _sq_qam_slicer_params(
            constellation_np
        )
        scan_fn = _get_jax_rls_cpr(
            num_taps,
            stride,
            len(constellation_np),
            num_ch,
            cpr_type,
            B,
            int(cpr_bps_block_size),
            bool(cpr_joint_channels),
            H,
            int(symmetry),
            int(_sq_side_j),
            float(_sq_lev_min_j),
            float(_sq_d_grid_j),
        )
        KB = int(cpr_bps_block_size)
        if cpr_state is not None:
            _pi, _pf, _bb, _bbp, _bp4, _cx, _cy, _cp = _cpr_state_to_jax_inits(
                cpr_state, num_ch, KB, H
            )
        else:
            _pi = np.zeros(num_ch, dtype=np.float64)
            _pf = np.zeros(num_ch, dtype=np.float64)
            _bb = np.zeros((KB, num_ch), dtype=np.complex64)
            _bbp = np.int32(0)
            _bp4 = np.zeros(num_ch, dtype=np.float64)
            _cx = np.zeros((num_ch, H), dtype=np.float64)
            _cy = np.zeros((num_ch, H), dtype=np.float64)
            _cp = np.zeros(num_ch, dtype=np.int32)
        (
            y_jax,
            e_jax,
            W_jax,
            wh_jax,
            phi_jax,
            pll_phi_f,
            pll_freq_f,
            bps_buf_f,
            bps_buf_ptr_f,
            bps_prev4_f,
            cs_buf_x_f,
            cs_buf_y_f,
            cs_buf_ptr_f,
        ) = scan_fn(
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
            to_jax(jnp.float64(pll_mu), device=platform),
            to_jax(jnp.float64(pll_beta), device=platform),
            to_jax(jnp.float64(cpr_cycle_slip_threshold), device=platform),
            to_jax(jnp.bool_(cpr_cycle_slip_correction), device=platform),
            to_jax(_pi, device=platform),
            to_jax(_pf, device=platform),
            to_jax(_bb, device=platform),
            to_jax(_bbp, device=platform),
            to_jax(_bp4, device=platform),
            to_jax(_cx, device=platform),
            to_jax(_cy, device=platform),
            to_jax(_cp, device=platform),
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
        result.cpr_state = CPRState(
            pll_phi=np.asarray(from_jax(pll_phi_f)),
            pll_freq=np.asarray(from_jax(pll_freq_f)),
            bps_prev4=np.asarray(from_jax(bps_prev4_f)),
            jax_bps_buf=np.asarray(from_jax(bps_buf_f)),
            jax_bps_buf_ptr=int(np.asarray(from_jax(bps_buf_ptr_f))),
            cs_buf_x=np.asarray(from_jax(cs_buf_x_f)),
            cs_buf_y=np.asarray(from_jax(cs_buf_y_f)),
            cs_buf_ptr=np.asarray(from_jax(cs_buf_ptr_f)),
            cpr_type=cpr_type,
            num_ch=num_ch,
            symmetry=symmetry,
            bps_P=B,
            bps_K=KB,
            cs_H=H,
        )
    # Truncate last num_taps//2 symbols (zero-padding contamination).
    result = _log_equalizer_exit(result, name="RLS", debug_plot=debug_plot)
    result.tail_trim = tail_trim
    _check_rls_divergence(result.weights, xp, forgetting_factor, delta)
    return result


def _get_numba_cs_block():
    """Lazy-compile a Numba JIT kernel for the block_lms cycle-slip correction loop.

    Replaces the Python ``for ci in range(C): for i in range(B)`` loop with
    compiled native code.  D→H/H→D transfers (few KB) still happen, but the
    loop body itself runs at native speed, eliminating the dominant overhead
    when ``cpr_cycle_slip_correction=True`` on GPU.
    """
    if "cs_block" not in _NUMBA_KERNELS:
        numba_mod = _get_numba()
        if numba_mod is None:
            return None

        @numba_mod.njit(cache=True, fastmath=True, nogil=True)
        def cs_block(
            phi_blk,
            phi_corr,
            cs_buf_x,
            cs_buf_y,
            cs_buf_ptr,
            cs_buf_n,
            cs_stats,
            b_start,
            quantum,
            threshold,
            cs_H,
        ):
            """Per-symbol cycle-slip correction for one equalizer block.

            Parameters
            ----------
            phi_blk   : (C, B) float64 — BPS phase before correction (input)
            phi_corr  : (C, B) float64 — corrected phase (output, pre-allocated)
            cs_buf_x  : (C, H) float64 — unused (retained for call-site compat)
            cs_buf_y  : (C, H) float64 — circular buffer of past corrected phases
            cs_buf_ptr: (C,)   int64   — write pointer (monotonically increasing)
            cs_buf_n  : (C,)   int64   — number of valid entries (≤ H)
            cs_stats  : (C, 4) float64 — [0]=Sy, [1]=Sxy (relative coords); [2..3] unused
            b_start   : int            — global symbol index of block start (unused)
            quantum   : float64        — slip quantum (2π / symmetry)
            threshold : float64        — |diff| threshold for declaring a slip
            cs_H      : int            — circular buffer length
            """
            C = phi_blk.shape[0]
            B = phi_blk.shape[1]
            H_f = float(cs_H)
            Sx_full = H_f * (H_f - 1.0) / 2.0
            Sxx_full = H_f * (H_f - 1.0) * (2.0 * H_f - 1.0) / 6.0
            denom_full = H_f * Sxx_full - Sx_full * Sx_full
            for ci in range(C):
                for i in range(B):
                    y_b = phi_blk[ci, i]
                    n_b = cs_buf_n[ci]
                    ptr = cs_buf_ptr[ci]

                    if n_b == 0:
                        phi_expected = y_b
                    elif n_b < 10:
                        last_pos = (ptr - 1 + cs_H) % cs_H
                        phi_expected = cs_buf_y[ci, last_pos]
                    else:
                        sy = cs_stats[ci, 0]
                        sxy = cs_stats[ci, 1]
                        n_f = float(n_b)
                        if n_b < cs_H:
                            Sx_c = n_f * (n_f - 1.0) / 2.0
                            Sxx_c = n_f * (n_f - 1.0) * (2.0 * n_f - 1.0) / 6.0
                            denom = n_f * Sxx_c - Sx_c * Sx_c
                        else:
                            Sx_c = Sx_full
                            denom = denom_full
                        if abs(denom) > 1e-30:
                            slope = (n_f * sxy - Sx_c * sy) / denom
                            intercept = (sy - slope * Sx_c) / n_f
                        else:
                            slope = 0.0
                            intercept = sy / n_f
                        phi_expected = slope * n_f + intercept

                    diff = y_b - phi_expected
                    k_slip = int(round(diff / quantum))
                    if abs(diff) > threshold and k_slip != 0:
                        y_b -= float(k_slip) * quantum
                    phi_corr[ci, i] = y_b

                    # Update circular buffer — relative coords, only y needed
                    write_pos = ptr % cs_H
                    if n_b == cs_H:
                        old_y = cs_buf_y[ci, write_pos]
                        old_sy = cs_stats[ci, 0]
                        # Sxy_new = Sxy_old - Sy_old + y_old + (H-1)*y_new
                        cs_stats[ci, 1] = (
                            cs_stats[ci, 1] - old_sy + old_y + (H_f - 1.0) * y_b
                        )
                        cs_stats[ci, 0] = old_sy - old_y + y_b
                    else:
                        cs_stats[ci, 1] += float(n_b) * y_b
                        cs_stats[ci, 0] += y_b
                    cs_buf_y[ci, write_pos] = y_b
                    cs_buf_ptr[ci] = ptr + 1
                    if n_b < cs_H:
                        cs_buf_n[ci] = n_b + 1

        _NUMBA_KERNELS["cs_block"] = cs_block
    return _NUMBA_KERNELS["cs_block"]


def block_lms(
    samples: ArrayType,
    training_symbols: Optional[ArrayType] = None,
    num_taps: int = 21,
    sps: int = 2,
    step_size: float = 2e-4,
    block_size: int = 256,
    modulation: Optional[str] = None,
    order: Optional[int] = None,
    unipolar: bool = False,
    store_weights: bool = False,
    w_init: Optional[ArrayType] = None,
    pmf: Optional[Any] = None,
    cpr_type: Optional[str] = None,
    cpr_bps_test_phases: int = 64,
    cpr_bps_block_size: int = 32,
    cpr_joint_channels: bool = False,
    cpr_cycle_slip_correction: bool = False,
    cpr_cycle_slip_history: int = 100,
    cpr_cycle_slip_threshold: float = np.pi / 4,
    debug_plot: bool = False,
    plot_smoothing: int = 50,
    cpr_state: Optional["CPRState"] = None,
    input_norm_factor: Optional[Union[float, np.ndarray]] = None,
    samples_prefix: Optional[ArrayType] = None,
    pad_mode: str = "zeros",
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
       rotation.  This produces one phase estimate :math:`\\varphi_n` per symbol
       (not one per block), so ``cpr_bps_block_size`` and ``block_size`` are
       independent parameters: ``block_size`` controls FFT/gradient efficiency
       while ``cpr_bps_block_size`` controls phase noise suppression.  The raw
       ``[0, π/2)`` argmin is converted to full-range radians by a causal 4-fold
       unwrap, and stored in a float64 accumulator in ``phase_trajectory``.

    3. **Cycle-slip correction** (if ``cpr_cycle_slip_correction=True``) — the
       full per-symbol BPS phase tensor :math:`\\varphi_n` (shape ``(C, B)``)
       is transferred device→host; for each symbol the phase is compared to a
       linear-regression prediction built from a circular buffer of
       ``cpr_cycle_slip_history`` past corrected phases (identical algorithm to
       :func:`lms` with ``cpr_type='bps'``).  If
       :math:`|\\hat{\\varphi}_n - \\varphi_{\\mathrm{pred}}| >` ``cpr_cycle_slip_threshold``
       the nearest :math:`2\\pi/\\mathrm{symmetry}` quantum is subtracted, the
       corrected value is stored in the history buffer, and the corrected block
       is written back to device.  Cost: one ``(C, B)`` float64 D→H + H→D
       round-trip per block; disable on slip-free channels to avoid this
       transfer.

    4. **Error** — training or DD slicer on CPR-corrected output; back-rotated
       to the tap plane using the block-average phase :math:`\\varphi_b`:

       .. math::

           e_{\\text{taps}}[n] = e_{\\text{clean}}[n] \\cdot e^{+j\\varphi_b}

    5. **Gradient** — scatter ``e_taps`` to sample positions, then:

       .. math::

           \\Delta H_{\\text{fd}}[i,j] = \\overline{E_{\\text{fd}}[i]} \\cdot X_{\\text{fd}}[j]

           h \\mathrel{+}= \\mu \\cdot \\mathrm{IFFT}(\\Delta H_{\\text{fd}})[\\ldots:T]

       :math:`\\mu` is applied to the **summed** block gradient (all B per-symbol
       contributions).  This sum is exactly what a frozen-weight per-symbol LMS
       would accumulate over the same B symbols, so ``step_size`` is on the
       **same scale as** :func:`lms`: the same :math:`\\mu` yields the same
       convergence and steady-state MSE (see ``step_size`` below).  Only the
       *stability ceiling* is B× lower — the operating step that matches
       :func:`lms` is unchanged.

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
    step_size : float, default 2e-4
        LMS step size μ, on the **same scale as** :func:`lms`.  Use the same
        value you would use for :func:`lms`: because the block update is the
        summed gradient over all B symbols — exactly what a frozen-weight
        per-symbol LMS accumulates over those symbols — the same μ produces the
        same convergence speed and steady-state MSE, independent of
        ``block_size``.  **Do not** divide by ``block_size``; doing so
        under-adapts the filter by that factor.

        The only ``block_size`` dependence is the *stability ceiling*: because
        the weights are frozen across the block, the maximum stable μ is
        ``2/(B·C·T·P_x)`` — roughly ``block_size`` times lower than
        :func:`lms`.  Reduce μ below your :func:`lms` value **only if** it
        exceeds this ceiling (i.e. the run raises the divergence error); the
        default ``2e-4`` is conservative and safe for ``block_size`` up to a
        few thousand.
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
    w_init : array_like, optional
        Initial tap weights, shape ``(C, C, T)`` or SISO short-hands.
    pmf : array_like, optional
        Probability mass function for PS-QAM constellation scaling.
    cpr_type : {'bps', None}, default None
        Inline carrier phase recovery.  Only ``'bps'`` is supported; PLL is
        not available because its per-symbol PI integration does not fit the
        block gradient model.

        **BPS dual-path design:** within each equalizer block, the wrapped
        float32 phase estimate rotates the pre-CPR symbol ``y_raw`` to
        produce the weight-update error, while the unwrapped float64
        accumulator is written to ``phase_trajectory``.  This separation
        prevents float32 rounding errors from accumulating over long signals.

        **4-fold causal unwrap:** the BPS ``argmin`` is in ``[0, π/2)``.
        A per-symbol causal tracker adds or subtracts ``π/2`` multiples to
        maintain continuity, then scales to full-range radians.
    cpr_bps_test_phases : int, default 64
        Number of BPS candidate angles in ``[0, π/2)``.  32-64 is
        sufficient for ≤ 16-QAM; use 64-128 for 64-QAM.
    cpr_bps_block_size : int, default 32
        Trailing-window length (symbols) summed before the BPS ``argmin``.
        This is evaluated per symbol (not per equalizer block), so it is
        independent of ``block_size``.  Larger values reduce phase-noise
        variance at the cost of increased tracking latency.
        ``cpr_bps_block_size=1`` gives single-symbol BPS.
    cpr_joint_channels : bool, default False
        For MIMO inputs (C > 1): if ``True``, the BPS distance metrics are
        summed across all C channels before ``argmin``, producing one shared
        phase estimate broadcast to all channels.  Reduces estimation
        variance by ~√C for shared-LO transmitters.  When ``False``, each
        channel estimates its phase independently.  Ignored for SISO inputs.
    cpr_cycle_slip_correction : bool, default False
        Enable per-symbol cycle-slip detection and correction using the same
        algorithm as :func:`lms`.  After each BPS block the full ``(C, B)``
        float64 phase tensor is transferred device→host; every symbol is
        compared to a regression prediction, corrected if a slip is detected,
        and added to the circular history buffer before the corrected block is
        written back.  Disable on slip-free channels to avoid the D→H/H→D
        round-trip per block.
    cpr_cycle_slip_history : int, default 100
        Length of the per-symbol phase history buffer used for the linear
        regression predictor.  Same semantics as in :func:`lms`: one entry
        per symbol, so ``100`` means 100 past corrected symbol phases.
        Ignored when ``cpr_cycle_slip_correction=False``.
    cpr_cycle_slip_threshold : float, default π/4
        Maximum phase step (radians) between adjacent symbols before a
        cycle slip is declared.  Set to half the constellation's angular
        symmetry quantum (``π/4`` for QPSK/QAM).  Ignored when
        ``cpr_cycle_slip_correction=False``.
    debug_plot : bool, default False
        Show a convergence + phase diagnostic plot on exit.
    plot_smoothing : int, default 50
        Moving-average window for the MSE curve in the debug plot.
    cpr_state : CPRState, optional
        Warm-start BPS CPR state from a previous ``block_lms()`` call.
        When provided, the BPS 4-fold unwrap accumulators (``bps_prev4``,
        ``bps_offset4``) and the block-distance history matrix
        (``bps_d2_hist``, shape ``(B, C, K-1)``) are restored from the
        previous block boundary.  This prevents the BPS from re-converging
        its phase estimate at each block boundary, which otherwise causes
        a ~``cpr_bps_block_size``-symbol transient of increased phase error.
        Pass ``None`` (default) to cold-start.  Only BPS state is used;
        PLL/cycle-slip fields are ignored.
    input_norm_factor : float or ndarray, optional
        Pre-computed RMS normalization factor.  See ``lms()`` for the full
        description; behaviour is identical.
    samples_prefix : array_like, optional
        Signal history from the end of the previous block.  See ``lms()``
        for the full description; behaviour is identical.
    pad_mode : {'zeros', 'edge'}, default 'zeros'
        Padding strategy when ``samples_prefix`` is ``None``.  See
        ``lms()`` for the full description; behaviour is identical.

    Returns
    -------
    EqualizerResult
        Same fields as :func:`lms`, plus:

        * ``input_norm_factor`` — RMS factor used to normalize inputs.
        * ``cpr_state`` — ``CPRState`` with BPS accumulators after the last
          block.  ``None`` when ``cpr_type=None``.

        ``phase_trajectory`` is populated when ``cpr_type='bps'``; shape
        ``(N_sym,)`` SISO or ``(C, N_sym)`` MIMO, one estimate per symbol.

    Warnings
    --------
    **GPU throughput — use large block_size:** On GPU (CuPy) each Python
    loop iteration launches ~10–20 CUDA kernels (FFT, einsum, IFFT, BPS
    rotations, …).  At ``block_size=64`` and 100k symbols that is ~1 500
    blocks × kernel-launch overhead; at ``block_size=2048`` it drops to
    ~49 blocks.  Throughput improves markedly once the cuFFT/cuBLAS work
    per block dominates the Python overhead.  On GPU prefer
    ``block_size`` ≥ 512, ideally 1024–4096.

    **BPS cycle-slip correction (``cpr_cycle_slip_correction=True``):**
    Every block transfers the full ``(C, block_size)`` float64 phase
    tensor device→host, runs a Python loop over ``C × block_size``
    symbols, then writes it back host→device.  This synchronous round-trip
    serialises the GPU pipeline.  Keep this disabled if slips are rare or
    absent, and enable only when necessary.

    **CPU (NumPy) backend:** even slower than GPU because the Python loop
    dominates at any practical block size.  Use ``lms(backend='numba')``
    or ``lms(backend='jax')`` for CPU workloads instead.

    **Stability / overflow:** ``step_size`` is applied to the **summed**
    gradient over all ``block_size`` symbols (not averaged).  This keeps μ on
    the same scale as :func:`lms` (same μ → same convergence and steady-state
    MSE), but it also means the *stability ceiling* —
    ``0 < μ < 2/(block_size·C·T·P_x)`` — is roughly ``block_size`` times lower
    than per-symbol LMS, because the weights are frozen across the block.  So
    start from the **same** ``step_size`` you use for :func:`lms`; if a large
    ``block_size`` pushes that value above the ceiling the run diverges (NaN
    weights, detected at end of run), in which case reduce μ until stable —
    do **not** routinely divide by ``block_size`` (that under-adapts the
    filter by the same factor).
    """
    if cpr_type is not None and cpr_type != "bps":
        raise ValueError(
            f"block_lms only supports cpr_type='bps' or None. Got {cpr_type!r}. "
            "PLL is not available for block processing."
        )

    num_taps = int(num_taps)
    sps = int(sps)
    block_size = int(block_size)

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

    samples, training_symbols, eq_norm = _normalize_inputs(
        samples, training_symbols, sps, input_norm_factor=input_norm_factor
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
    _sq_side, _sq_lev_min_f, _sq_d_grid_f = _sq_qam_slicer_params(constellation_np)
    _sq_lev_min = float(_sq_lev_min_f)
    _sq_d_grid = float(_sq_d_grid_f)
    _sq_m1 = _sq_side - 1  # clip upper bound (0 when sq_side==0 — never used)

    # ── Training alignment ────────────────────────────────────────────────────
    if training_symbols is not None:
        n_train_aligned = min(int(training_symbols.shape[-1]), n_sym)
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
        _two_pi = xp.float64(2.0 * np.pi)
        # Cross-block 4-fold unwrap state (CPU, one value per channel)
        _cs_H = min(int(cpr_cycle_slip_history), n_sym)
        _bps_K = int(cpr_bps_block_size)
        _bps_hist_len = max(0, _bps_K - 1)
        _st = cpr_state
        _st_ok = (
            _st is not None
            and _st.cpr_type == cpr_type
            and _st.num_ch == C
            and _st.cs_H == _cs_H
            and _st.bps_P == P
            and _st.bps_K == _bps_K
            and _st.bps_prev4 is not None
        )
        bps_prev4 = _st.bps_prev4.copy() if _st_ok else np.zeros(C, dtype=np.float64)
        bps_offset4 = (
            _st.bps_offset4.copy() if _st_ok else np.zeros(C, dtype=np.float64)
        )
        # Cycle-slip regression state (CPU — negligible overhead vs GPU compute).
        # Tracks last corrected boundary symbol per block; history depth = cpr_cycle_slip_history.
        cs_buf_x = (
            _st.cs_buf_x.copy() if _st_ok else np.zeros((C, _cs_H), dtype=np.float64)
        )
        cs_buf_y = (
            _st.cs_buf_y.copy() if _st_ok else np.zeros((C, _cs_H), dtype=np.float64)
        )
        cs_buf_ptr = _st.cs_buf_ptr.copy() if _st_ok else np.zeros(C, dtype=np.int64)
        cs_buf_n = _st.cs_buf_n.copy() if _st_ok else np.zeros(C, dtype=np.int64)
        cs_stats = _st.cs_stats.copy() if _st_ok else np.zeros((C, 4), dtype=np.float64)
        # Cross-block sliding-window history for BPS metric (K-1 prev samples)
        if _st_ok and _st.bps_d2_hist is not None:
            bps_d2_hist = xp.asarray(_st.bps_d2_hist)
        else:
            bps_d2_hist = xp.zeros((P, C, _bps_hist_len), dtype=xp.float32)

    # ── OLS block size ────────────────────────────────────────────────────────
    # fftsize must be >= block_size * sps + num_taps - 1 (linear OLS condition)
    _ols_min = block_size * sps + num_taps - 1
    fftsize = 1 << (_ols_min - 1).bit_length()  # next power of 2

    _cpr_info = ""
    if cpr_type == "bps":
        _cs = (
            f", cs_corr=True(thr={cpr_cycle_slip_threshold:.3f})"
            if cpr_cycle_slip_correction
            else ", cs_corr=False"
        )
        _joint = ", joint" if cpr_joint_channels and C > 1 else ""
        _cpr_info = (
            f", cpr=bps(P={cpr_bps_test_phases}, K={cpr_bps_block_size}{_joint}{_cs})"
        )
    logger.info(
        f"Block-LMS: C={C}, num_taps={num_taps}, sps={sps}, block_size={block_size}, "
        f"fftsize={fftsize}, mu={step_size}, n_sym={n_sym}{_cpr_info}"
    )

    # ── Padding — matches lms() convention ───────────────────────────────────
    c_tap = num_taps // 2
    pad_total = max(0, n_sym * sps - N + num_taps - 1)
    pad_left = min(c_tap, pad_total)
    pad_right = pad_total - pad_left
    if samples_prefix is not None or xp is np or pad_mode != "zeros":
        _samp_cpu_blms = to_device(samples, "cpu").astype(np.complex64)
        x_padded = xp.asarray(
            _build_padded_samples(
                _samp_cpu_blms,
                pad_left,
                pad_right,
                samples_prefix,
                pad_mode,
                eq_norm,
                sps,
            )
        )  # (C, N_pad)
    else:
        # Fast on-device path: avoid D→H→D round-trip when there is no prefix.
        # _normalize_inputs already normalized samples; eq_norm is only needed
        # by _build_padded_samples to normalize the samples_prefix, which is
        # handled by the branch above.
        _samp_f32 = (
            samples if samples.dtype == xp.complex64 else samples.astype(xp.complex64)
        )
        _left = xp.zeros((C, pad_left), dtype=xp.complex64)
        _right = (
            xp.zeros((C, pad_right), dtype=xp.complex64)
            if pad_right > 0
            else xp.empty((C, 0), dtype=xp.complex64)
        )
        x_padded = xp.concatenate([_left, _samp_f32, _right], axis=1)  # (C, N_pad)
    N_padded = x_padded.shape[1]

    # ── Output buffers ────────────────────────────────────────────────────────
    y_all = xp.empty((C, n_sym), dtype=xp.complex64)
    e_all = xp.empty((C, n_sym), dtype=xp.complex64)
    w_hist = (
        xp.empty((n_sym, C, C, num_taps), dtype=xp.complex64) if store_weights else None
    )
    phi_all = xp.zeros((C, n_sym), dtype=xp.float32) if cpr_type == "bps" else None

    # Pre-allocate scratch buffers — reused every block to avoid per-block heap pressure.
    x_win = xp.zeros((C, fftsize), dtype=xp.complex64)
    e_scatter = xp.zeros((C, fftsize), dtype=xp.complex64)

    n_blocks = (n_sym + block_size - 1) // block_size
    # On-device divergence flag — accumulates with |= inside the loop, zero D→H
    # syncs during iteration.  Checked once with bool() after the loop exits.
    _div_flag = xp.zeros(1, dtype=xp.bool_)

    # ── Block loop ────────────────────────────────────────────────────────────
    for b in range(n_blocks):
        b_start = b * block_size
        b_end = min(b_start + block_size, n_sym)
        B = b_end - b_start  # symbols in this block (may be < block_size for last)

        # Input window: x_padded[:, b_start*sps : b_start*sps + fftsize]
        # fftsize = next_pow2(B*sps + num_taps - 1), so the window includes
        # B*sps new samples plus num_taps-1 anti-causal look-ahead samples
        # needed by the cross-correlation filter y[n]=Σ conj(h[τ])·x[n+τ].
        x_start = b_start * sps
        x_win.fill(0)
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
            # min_d2: (P, C, B) — min squared distance to constellation.
            # O(1) square-QAM: snap I/Q independently to nearest level grid point.
            if _sq_side > 0:
                _nr = (
                    _sq_lev_min
                    + xp.clip(
                        xp.round((rotated.real - _sq_lev_min) / _sq_d_grid), 0, _sq_m1
                    )
                    * _sq_d_grid
                )
                _ni = (
                    _sq_lev_min
                    + xp.clip(
                        xp.round((rotated.imag - _sq_lev_min) / _sq_d_grid), 0, _sq_m1
                    )
                    * _sq_d_grid
                )
                min_d2 = ((rotated.real - _nr) ** 2 + (rotated.imag - _ni) ** 2).astype(
                    xp.float32
                )
            else:
                d2_all = (
                    xp.abs(rotated[..., None] - constellation[None, None, None, :]) ** 2
                ).real
                min_d2 = xp.min(d2_all, axis=-1).astype(xp.float32)

            # Causal sliding-window average of width K along the B (symbol) axis.
            # win_sum[:,:,n] = sum of min_d2[:,:, n-K+1..n] (with K-1 samples from
            # previous block as prefix so the window is full from symbol 0).
            K = min(_bps_K, B)
            hist_prefix = (
                bps_d2_hist[:, :, -(K - 1) :]
                if K > 1
                else xp.empty((P, C, 0), dtype=xp.float32)
            )
            cat_d2 = xp.concatenate([hist_prefix, min_d2], axis=2)  # (P, C, K-1+B)
            cs_d2 = xp.concatenate(
                [xp.zeros((P, C, 1), dtype=xp.float32), cat_d2.cumsum(axis=2)], axis=2
            )  # (P, C, K+B)
            win_sum = cs_d2[:, :, K:] - cs_d2[:, :, :-K]  # (P, C, B)
            metric = win_sum / xp.float32(K)  # (P, C, B) — always full K-sample window

            # Per-symbol argmin over P phases → raw (C, B) in [0, π/2)
            if cpr_joint_channels and C > 1:
                best_k = xp.argmin(metric.sum(axis=1), axis=0)  # (B,)
                phi_raw = xp.broadcast_to(
                    bps_angles[best_k][None, :], (C, B)
                ).copy()  # (C, B)
            else:
                best_k = xp.argmin(metric, axis=0)  # (C, B)
                phi_raw = bps_angles[best_k]  # (C, B)

            # Slide BPS distance history across block boundary (for next block's prefix)
            if _bps_hist_len > 0:
                combined_hist = xp.concatenate([bps_d2_hist, min_d2], axis=2)
                bps_d2_hist = combined_hist[:, :, -_bps_hist_len:]

            # 4-fold causal unwrap: equivalent to np.unwrap(phi*4)/4 via
            # diff→wrap[-π,π]→cumsum.  CPU: np.unwrap directly (no transfer cost).
            # GPU: run arithmetic on-device, sync only (C,) float64 per block.
            if xp is np:
                raw4 = phi_raw.astype(np.float64) * 4.0  # (C, B)
                extended = np.concatenate(
                    [bps_prev4[:, np.newaxis], raw4], axis=1
                )  # (C, B+1)
                unwrapped_ext = np.unwrap(extended, axis=1)  # (C, B+1)
                _cumul_cpu = unwrapped_ext[:, 1:] - unwrapped_ext[:, 0:1]  # (C, B)
                _phi_f64 = (bps_offset4[:, np.newaxis] + _cumul_cpu) / 4.0  # (C, B)
                bps_prev4[:] = unwrapped_ext[:, -1]
                bps_offset4 += _cumul_cpu[:, -1]
            else:
                raw4_dev = phi_raw.astype(xp.float64) * xp.float64(4.0)  # (C, B)
                ext_dev = xp.concatenate(
                    [xp.asarray(bps_prev4)[:, None], raw4_dev], axis=1
                )  # (C, B+1)
                _two_pi_d = xp.float64(2.0 * np.pi)
                d4 = ext_dev[:, 1:] - ext_dev[:, :-1]  # (C, B)
                d4 -= xp.round(d4 / _two_pi_d) * _two_pi_d  # wrap to [-π, π]
                _cumul_dev = xp.cumsum(d4, axis=1)  # (C, B)
                _phi_f64 = (xp.asarray(bps_offset4)[:, None] + _cumul_dev) / xp.float64(
                    4.0
                )
                _delta = to_device(_cumul_dev[:, -1], "cpu")  # (C,) — 16 bytes for C=2
                bps_prev4 += _delta
                bps_offset4 += _delta
            # ── Per-symbol cycle-slip correction ──────────────────────────
            # Identical algorithm to per-symbol lms: for each symbol in the
            # block, compare its BPS phase to the regression prediction, snap
            # to the nearest quantum if |diff| > threshold, then add the
            # corrected (x, y) pair to the circular history buffer.
            # Requires a D→H transfer of the full (C, B) float64 phase block
            # and an H→D write-back; cost is O(C·B) per block.
            if cpr_cycle_slip_correction:
                phi_blk_np = to_device(_phi_f64, "cpu").astype(np.float64)  # (C, B)
                phi_corr_np = phi_blk_np.copy()

                _cs_kernel = _get_numba_cs_block()
                if _cs_kernel is not None:
                    _cs_kernel(
                        phi_blk_np,
                        phi_corr_np,
                        cs_buf_x,
                        cs_buf_y,
                        cs_buf_ptr,
                        cs_buf_n,
                        cs_stats,
                        b_start,
                        float(quantum),
                        float(cpr_cycle_slip_threshold),
                        _cs_H,
                    )
                else:
                    _H_f = float(_cs_H)
                    for ci in range(C):
                        for i in range(B):
                            y_b = phi_blk_np[ci, i]
                            n_b = int(cs_buf_n[ci])
                            ptr = int(cs_buf_ptr[ci])

                            if n_b == 0:
                                phi_expected = y_b
                            elif n_b < 10:
                                last_pos = (ptr - 1 + _cs_H) % _cs_H
                                phi_expected = cs_buf_y[ci, last_pos]
                            else:
                                sy = cs_stats[ci, 0]
                                sxy = cs_stats[ci, 1]
                                n_f = float(n_b)
                                if n_b < _cs_H:
                                    Sx_c = n_f * (n_f - 1.0) / 2.0
                                    Sxx_c = n_f * (n_f - 1.0) * (2.0 * n_f - 1.0) / 6.0
                                    denom = n_f * Sxx_c - Sx_c * Sx_c
                                else:
                                    Sx_c = _H_f * (_H_f - 1.0) / 2.0
                                    Sxx_c = (
                                        _H_f * (_H_f - 1.0) * (2.0 * _H_f - 1.0) / 6.0
                                    )
                                    denom = _H_f * Sxx_c - Sx_c * Sx_c
                                if abs(denom) > 1e-30:
                                    slope = (n_f * sxy - Sx_c * sy) / denom
                                    intercept = (sy - slope * Sx_c) / n_f
                                else:
                                    slope = 0.0
                                    intercept = sy / n_f
                                phi_expected = slope * n_f + intercept

                            diff = y_b - phi_expected
                            k_slip = int(round(diff / quantum))
                            if (
                                abs(diff) > float(cpr_cycle_slip_threshold)
                                and k_slip != 0
                            ):
                                y_b -= float(k_slip) * quantum
                            phi_corr_np[ci, i] = y_b

                            # Update circular buffer — relative coords, only y needed
                            write_pos = ptr % _cs_H
                            if n_b == _cs_H:
                                old_y = cs_buf_y[ci, write_pos]
                                old_sy = cs_stats[ci, 0]
                                # Sxy_new = Sxy_old - Sy_old + y_old + (H-1)*y_new
                                cs_stats[ci, 1] = (
                                    cs_stats[ci, 1]
                                    - old_sy
                                    + old_y
                                    + (_H_f - 1.0) * y_b
                                )
                                cs_stats[ci, 0] = old_sy - old_y + y_b
                            else:
                                cs_stats[ci, 1] += float(n_b) * y_b
                                cs_stats[ci, 0] += y_b
                            cs_buf_y[ci, write_pos] = y_b
                            cs_buf_ptr[ci] = ptr + 1
                            if n_b < _cs_H:
                                cs_buf_n[ci] = n_b + 1

                _phi_f64 = xp.asarray(phi_corr_np)

                # Carry the net slip correction into bps_offset4 so that
                # subsequent blocks do not re-detect the same slip.
                # bps_offset4 tracks the 4x-domain accumulated phase; the slip
                # quantum is π/2 → 2π in 4x, which is a multiple of 2π and
                # therefore transparent to the 4-fold unwrap of bps_prev4.
                for ci in range(C):
                    net4 = (phi_corr_np[ci, -1] - phi_blk_np[ci, -1]) * 4.0
                    if net4 != 0.0:
                        bps_offset4[ci] += net4

            # Wrap unbounded float64 phase to [-π, π] before float32 cast so that
            # the GPU exp() argument is bounded (dual-path: wrapped for rotation,
            # unwrapped for trajectory storage).
            phi_c_dev = (_phi_f64 - xp.round(_phi_f64 / _two_pi) * _two_pi).astype(
                xp.float32
            )
            phi_c_traj = _phi_f64.astype(xp.float32)  # unwrapped, for output trajectory

            phi_c = phi_c_dev  # wrapped float32, for rotation
            y_rot = y_block * xp.exp(-1j * phi_c.astype(xp.complex64))  # (C, B)
            phi_all[:, b_start:b_end] = phi_c_traj  # unwrapped float32, for trajectory
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
            if _sq_side > 0:
                _dd_r = (
                    _sq_lev_min
                    + xp.clip(
                        xp.round((y_dd.real - _sq_lev_min) / _sq_d_grid), 0, _sq_m1
                    )
                    * _sq_d_grid
                )
                _dd_i = (
                    _sq_lev_min
                    + xp.clip(
                        xp.round((y_dd.imag - _sq_lev_min) / _sq_d_grid), 0, _sq_m1
                    )
                    * _sq_d_grid
                )
                d_dd = xp.empty(y_dd.shape, dtype=xp.complex64)
                d_dd.real[:] = _dd_r
                d_dd.imag[:] = _dd_i
            else:
                d2_sl = (
                    xp.abs(y_dd[:, :, None] - constellation[None, None, :]) ** 2
                ).real
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
        e_scatter.fill(0)
        e_scatter[:, : B * sps : sps] = e_taps

        # Frequency-domain gradient: dH_fd[i,j,k] = conj(E_fd[i,k]) * X_fd[j,k]
        E_fd = xp.fft.fft(e_scatter, axis=-1)  # (C, F)
        dH_fd = xp.einsum("ik,jk->ijk", xp.conj(E_fd), X_fd)  # (C, C, F)
        dh = xp.fft.ifft(dH_fd, axis=-1)[:, :, :num_taps]  # (C, C, T)

        h = h + xp.float32(step_size) * dh

        # Accumulate divergence flag on-device — no D→H sync here.
        _div_flag |= ~xp.isfinite(h).all()

    # Single D→H sync after the full loop to check for divergence.
    if bool(_div_flag[0]):
        raise RuntimeError(
            f"block_lms diverged (step_size={step_size}, block_size={block_size}). "
            f"step_size is on the same scale as lms(), but because the weights are "
            f"frozen across the block the stability ceiling is ~{block_size}x lower "
            f"than per-symbol LMS. Reduce step_size until stable (e.g. try "
            f"{step_size / 2:.2e}, then keep halving) rather than dividing by "
            f"block_size, which would under-adapt the filter by that factor."
        )

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
    if cpr_type == "bps":
        result.cpr_state = CPRState(
            bps_prev4=bps_prev4.copy(),
            bps_offset4=bps_offset4.copy(),
            bps_d2_hist=to_device(bps_d2_hist, "cpu"),
            cs_buf_x=cs_buf_x.copy(),
            cs_buf_y=cs_buf_y.copy(),
            cs_buf_ptr=cs_buf_ptr.copy(),
            cs_buf_n=cs_buf_n.copy(),
            cs_stats=cs_stats.copy(),
            cpr_type=cpr_type,
            num_ch=C,
            symmetry=_cpr_symmetry(modulation, order),
            bps_P=P,
            bps_K=_bps_K,
            cs_H=_cs_H,
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
    input_norm_factor: Optional[Union[float, np.ndarray]] = None,
    samples_prefix: Optional[ArrayType] = None,
    pad_mode: str = "zeros",
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

    Algorithm (per symbol n)
    ------------------------
    Steps 1 and 2 are identical to :func:`lms` (sliding input window and
    butterfly filter output :math:`y^{\\mathrm{raw}}[n]`).  There is **no
    CPR step** — CMA's cost surface is phase-invariant; no radial error
    can drive a phase rotator (see Notes below).

    3. **Godard error** — third-order radial gradient of the dispersion
       cost :math:`J = E[(|y|^2 - R^2)^2]`:

       .. math::

           e[n] = \\bigl(|y[n]|^2 - R^2\\bigr) \\cdot y[n]

       The Godard radius :math:`R^2 = E[|s|^4] / E[|s|^2]` is computed
       once from the normalised constellation (defaults to 1 if
       ``modulation`` is not given).  The error is purely radial: any
       constant phase rotation of :math:`y` leaves :math:`|y|^2` and
       therefore :math:`e` unchanged up to the same rotation, so CMA
       cannot resolve the phase ambiguity it introduces.

    4. **Weight update** — steepest descent on the Godard criterion (note
       the minus sign, opposite to LMS):

       .. math::

           \\mathbf{w}_{c,c'} \\mathrel{-}= \\mu \\cdot
           \\overline{e_c[n]} \\cdot \\mathbf{x}_{c',n}

    **Pilot-aided hybrid** (when ``pilot_ref`` and ``pilot_mask`` are set):
    at pilot positions the Godard error is replaced by the LMS pilot error
    :math:`e_p[n] = \\mathrm{pilot\\_ref}[n] - y[n]`, and the weight update
    sign flips to :math:`+\\mu` (standard LMS gradient ascent toward the
    reference).  This resolves the phase ambiguity at pilot locations while
    CMA handles data positions blindly.

    Notes
    -----
    **Why joint CMA + CPR is not supported:**
    PLL requires a phase-coherent decision :math:`d[n]` (nearest
    constellation point) to form the cross-product error
    :math:`\\mathrm{Im}(y \\cdot \\bar{d})`; but CMA output has an unknown
    phase rotation, so the decision is unreliable.  BPS is blind, but CMA
    weights converge to one of four equally-valid 90° rotations and slowly
    drift between them — BPS would track that drift, but the next CMA
    gradient step would fight the correction.  Use the sequential pipeline
    instead: CMA → :func:`~commstools.recovery.correct_carrier_phase` (BPS or
    Viterbi-Viterbi) → optional :func:`lms` fine-tune.

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
    input_norm_factor : float or ndarray, optional
        Pre-computed RMS normalization factor from a previous call.  See
        ``lms()`` for the full description; behaviour is identical.
    samples_prefix : array_like, optional
        Signal history from the end of the previous block.  See ``lms()``
        for the full description; behaviour is identical.
    pad_mode : {'zeros', 'edge'}, default 'zeros'
        Padding strategy when ``samples_prefix`` is ``None``.  See
        ``lms()`` for the full description; behaviour is identical.

    Returns
    -------
    EqualizerResult
        Equalized symbols, final weights, CMA error history, and optionally
        weight trajectory.  ``input_norm_factor`` field is populated.

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
        samples_np, _, eq_norm = _normalize_inputs(
            samples_np, None, sps, input_norm_factor=input_norm_factor
        )

        x_np = _build_padded_samples(
            samples_np, pad_left, pad_right, samples_prefix, pad_mode, eq_norm, sps
        )
        x_np = np.ascontiguousarray(x_np)

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
    samples, _, eq_norm = _normalize_inputs(
        samples, None, sps, input_norm_factor=input_norm_factor
    )

    _samp_cpu_cma = to_device(samples, "cpu").astype(np.complex64)
    samples_padded_np_cma = _build_padded_samples(
        _samp_cpu_cma, pad_left, pad_right, samples_prefix, pad_mode, eq_norm, sps
    )
    samples_padded = (
        xp.asarray(samples_padded_np_cma)
        if not was_1d
        else xp.asarray(samples_padded_np_cma[0])
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
    input_norm_factor: Optional[Union[float, np.ndarray]] = None,
    samples_prefix: Optional[ArrayType] = None,
    pad_mode: str = "zeros",
) -> EqualizerResult:
    """
    Radius Directed Equalizer (RDE) — blind equalizer for multi-ring constellations.

    RDE is a CMA variant that replaces the single Godard dispersion radius with
    per-symbol radius selection from the set of unique constellation ring radii.
    This corrects CMA's fundamental weakness on higher-order QAM: CMA forces
    all symbols toward a single average circle, severely degrading convergence
    when the constellation spans multiple rings (e.g. inner, middle, outer rings
    of 16-QAM).  RDE instead drives each symbol toward its *nearest* ring,
    producing a gradient surface that matches the true constellation geometry.

    Like CMA, RDE is fully blind (no training symbols) and recovers the channel
    up to a **phase ambiguity**.  A carrier-phase recovery step is needed after
    convergence; see :func:`cma` Notes for why joint CPR is not supported.

    Algorithm (per symbol n)
    ------------------------
    Steps 1 and 2 are identical to :func:`lms` (sliding input window and
    butterfly filter output :math:`y[n]`).  Like :func:`cma`, there is no
    CPR step.

    3. **Ring selection** — choose the constellation ring radius closest to
       the current output magnitude:

       .. math::

           R_d[n] = \\operatorname*{argmin}_{r\\,\\in\\,\\mathcal{R}}
           \\bigl|\\,r - |y[n]|\\,\\bigr|, \\qquad
           \\mathcal{R} = \\bigl\\{|c| : c \\in \\text{constellation}\\bigr\\}

       :math:`\\mathcal{R}` is the set of unique ring radii extracted once
       from the normalised Gray constellation.  For 16-QAM this yields
       three radii rather than the single CMA average, eliminating the
       inward/outward pull that degrades CMA convergence on higher-order
       QAM.

    4. **RDE error** — same third-order form as :func:`cma` but using the
       per-symbol ring radius:

       .. math::

           e[n] = \\bigl(|y[n]|^2 - R_d[n]^2\\bigr) \\cdot y[n]

    5. **Weight update** — steepest descent (same sign convention as CMA):

       .. math::

           \\mathbf{w}_{c,c'} \\mathrel{-}= \\mu \\cdot
           \\overline{e_c[n]} \\cdot \\mathbf{x}_{c',n}

    **Pilot-aided hybrid** (when ``pilot_ref`` and ``pilot_mask`` are set):
    identical to :func:`cma` — at pilot positions the RDE error is replaced
    by :math:`e_p[n] = \\mathrm{pilot\\_ref}[n] - y[n]` and the sign flips
    to :math:`+\\mu`, resolving the phase ambiguity at those locations.

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
    input_norm_factor : float or ndarray, optional
        Pre-computed RMS normalization factor from a previous call.  See
        ``lms()`` for the full description; behaviour is identical.
    samples_prefix : array_like, optional
        Signal history from the end of the previous block.  See ``lms()``
        for the full description; behaviour is identical.
    pad_mode : {'zeros', 'edge'}, default 'zeros'
        Padding strategy when ``samples_prefix`` is ``None``.  See
        ``lms()`` for the full description; behaviour is identical.

    Returns
    -------
    EqualizerResult
        Equalized symbols, final weights, RDE error history, and optionally
        weight trajectory.  ``input_norm_factor`` field is populated.

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
        samples_np, _, eq_norm = _normalize_inputs(
            samples_np, None, sps, input_norm_factor=input_norm_factor
        )

        x_np = _build_padded_samples(
            samples_np, pad_left, pad_right, samples_prefix, pad_mode, eq_norm, sps
        )
        x_np = np.ascontiguousarray(x_np)

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
    samples, _, eq_norm = _normalize_inputs(
        samples, None, sps, input_norm_factor=input_norm_factor
    )

    _samp_cpu_rde = to_device(samples, "cpu").astype(np.complex64)
    samples_padded_np_rde = _build_padded_samples(
        _samp_cpu_rde, pad_left, pad_right, samples_prefix, pad_mode, eq_norm, sps
    )
    samples_padded = (
        xp.asarray(samples_padded_np_rde)
        if not was_1d
        else xp.asarray(samples_padded_np_rde[0])
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
        # MIMO: per-bin (CxC) matrix inversion — cannot reduce to scalar multiply.
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
    input_norm_factor: Optional[Union[float, np.ndarray]] = None,
    samples_prefix: Optional[ArrayType] = None,
    pad_mode: str = "zeros",
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
    input_norm_factor : float or ndarray, optional
        Pre-computed RMS normalization factor.  When provided and
        ``normalize=True``, skips RMS recomputation and uses this value
        directly.  Typically ``EqualizerResult.input_norm_factor`` from the
        prior equalizer run that produced ``weights``.
    samples_prefix : array_like, optional
        Signal history from the end of the previous block.  See ``lms()``
        for the full description; behaviour is identical.
    pad_mode : {'zeros', 'edge'}, default 'zeros'
        Padding strategy when ``samples_prefix`` is ``None``.  See
        ``lms()`` for the full description; behaviour is identical.

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
        y = equalization.apply_taps(new_signal, result.weights,
                                    input_norm_factor=result.input_norm_factor)
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

    _at_eq_norm = None
    if normalize:
        samples, _, _at_eq_norm = _normalize_inputs(
            samples, None, sps, input_norm_factor=input_norm_factor
        )

    # Pad left by center tap so window[0] is center-aligned with sample 0,
    # pad right to fill the last window completely — mirrors LMS pre-processing.
    c_tap = num_taps // 2
    pad_left = c_tap
    pad_right = n_sym * sps - N + num_taps - 1 - pad_left
    if samples_prefix is None and pad_mode == "zeros":
        samples_padded = xp.pad(samples, ((0, 0), (pad_left, pad_right)))
    else:
        _samp_cpu_at = to_device(samples, "cpu").astype(np.complex64)
        samples_padded = xp.asarray(
            _build_padded_samples(
                _samp_cpu_at,
                pad_left,
                pad_right,
                samples_prefix,
                pad_mode,
                _at_eq_norm,
                sps,
            )
        )

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
