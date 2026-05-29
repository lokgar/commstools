"""
Carrier phase recovery utilities.

This module provides routines for carrier phase recovery (CPR), including
streaming decision-directed PLL, block-based Viterbi-Viterbi, Blind Phase
Search, MAP Tikhonov-RTS, and pilot-aided methods, along with cycle-slip
correction and phase ambiguity resolution.

Functions
---------
recover_carrier_phase_pll :
    Streaming CPR via Decision-Directed PLL (1st/2nd-order loop); Numba-compiled
    inner loop for CPU performance; GPU-transparent via CPU offload.
recover_carrier_phase_viterbi_viterbi :
    Block-based CPR via M-th power law (Viterbi-Viterbi) for PSK/QAM symbols.
recover_carrier_phase_bps :
    Blind Phase Search CPR for QAM constellations (Pfau et al.).
recover_carrier_phase_tikhonov :
    MAP CPR with Tikhonov/Wiener prior; RTS Kalman smoother on VV block phases.
recover_carrier_phase_pilots :
    Pilot-aided CPR with phase unwrapping and interpolation across the symbol grid.
    Single-carrier only; for OFDM CPE tracking see 5G NR PTRS / DVB-T2.
correct_carrier_phase :
    Applies per-symbol phase correction by complex rotation.
correct_cycle_slips :
    Detects and corrects cycle slips in a block-phase trajectory.
resolve_phase_ambiguity :
    Resolves rotational phase ambiguity after blind carrier phase recovery.
"""

from typing import Optional

import numpy as np

from .backend import ArrayType, dispatch, to_device
from .frequency import _modulation_power_m
from .logger import logger

# Lazy-compiled Numba kernels for sequential CPR algorithms.
_NUMBA_PLL: dict = {}


def _get_numba_dd_pll():
    """JIT-compile and cache the Numba DD-PLL sample-wise loop kernel.

    Returns
    -------
    callable
        Numba-compiled ``_dd_pll_loop``.
    """
    if "dd_pll" not in _NUMBA_PLL:
        import numba  # noqa: PLC0415

        @numba.njit(cache=True, fastmath=True, nogil=True)
        def _dd_pll_loop(
            sym_r,
            sym_i,
            const_r,
            const_i,
            mu,
            beta,
            phi0,
            freq0,
            is_sq_qam,
            levels,
            d_grid,
            lev_min,
            side,
        ):
            """Inner DD-PLL loop compiled to machine code by Numba.

            Parameters
            ----------
            sym_r, sym_i : (N,) float64
                Real and imaginary parts of received symbols.
            const_r, const_i : (M,) float64
                Real and imaginary parts of reference constellation.
            mu : float64
                Proportional (phase) gain — corrects the instantaneous phase error.
            beta : float64
                Integral (frequency) gain — tracks residual frequency drift.
                Set to 0.0 for a 1st-order loop.
            phi0 : float64
                Initial phase state in radians.
            freq0 : float64
                Initial frequency correction state in radians/symbol.
            is_sq_qam : bool
                True when the constellation is a square QAM grid.  Enables the
                O(1) rounding decision path instead of the O(M) linear search.
            levels : (side,) float64
                Sorted unique axis levels for square QAM (ignored when not sq_qam).
            d_grid : float64
                Grid spacing (levels[1] - levels[0]).
            lev_min : float64
                Minimum level value (levels[0]).
            side : int
                Number of points per axis (sqrt of constellation order).

            Returns
            -------
            phase_est : (N,) float64
                Per-symbol phase trajectory φ[n].
            """
            N = len(sym_r)
            M = len(const_r)
            phase_est = np.empty(N, dtype=np.float64)
            phi = phi0
            freq = freq0

            for n in range(N):
                # Rotate received symbol by current phase estimate:
                # y[n] = s[n] · exp(−jφ[n])
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                yr = sym_r[n] * cos_phi + sym_i[n] * sin_phi
                yi = -sym_r[n] * sin_phi + sym_i[n] * cos_phi

                # Hard decision: argmin_{c ∈ C} |y − c|²
                if is_sq_qam:
                    # O(1) grid rounding for square QAM
                    r_idx = int(round((yr - lev_min) / d_grid))
                    if r_idx < 0:
                        r_idx = 0
                    elif r_idx >= side:
                        r_idx = side - 1
                    d_r = levels[r_idx]
                    i_idx = int(round((yi - lev_min) / d_grid))
                    if i_idx < 0:
                        i_idx = 0
                    elif i_idx >= side:
                        i_idx = side - 1
                    d_i = levels[i_idx]
                else:
                    min_d2 = (yr - const_r[0]) ** 2 + (yi - const_i[0]) ** 2
                    d_r = const_r[0]
                    d_i = const_i[0]
                    for k in range(1, M):
                        d2 = (yr - const_r[k]) ** 2 + (yi - const_i[k]) ** 2
                        if d2 < min_d2:
                            min_d2 = d2
                            d_r = const_r[k]
                            d_i = const_i[k]

                # Cross-product phase error:  e = Im(y · d*) = yi·d_r − yr·d_i
                e = yi * d_r - yr * d_i

                # Record the phase used to derotate symbol n — before the update.
                phase_est[n] = phi

                # 2nd-order loop filter (reduces to 1st order when beta=0):
                #   φ[n+1] = φ[n] + μ·e[n] + ν[n]
                #   ν[n]   = ν[n−1] + β·e[n]
                phi = phi + mu * e + freq
                freq = freq + beta * e

            return phase_est

        _NUMBA_PLL["dd_pll"] = _dd_pll_loop

    return _NUMBA_PLL["dd_pll"]


def _get_numba_dd_pll_butterworth():
    """JIT-compile and cache the Numba DD-PLL loop with Butterworth loop filter.

    The Butterworth loop filter replaces the simple PI (proportional-integral)
    structure with a 2nd-order IIR biquad.  This gives a flatter passband and
    sharper roll-off, improving phase noise rejection near the loop bandwidth.

    Returns
    -------
    callable
        Numba-compiled ``_dd_pll_bw_loop``.
    """
    if "dd_pll_bw" not in _NUMBA_PLL:
        import numba  # noqa: PLC0415

        @numba.njit(cache=True, fastmath=True, nogil=True)
        def _dd_pll_bw_loop(
            sym_r,
            sym_i,
            const_r,
            const_i,
            phi0,
            b0,
            b1,
            b2,
            a1,
            a2,
            is_sq_qam,
            levels,
            d_grid,
            lev_min,
            side,
        ):
            """DD-PLL inner loop with a 2nd-order Butterworth loop filter.

            Parameters
            ----------
            sym_r, sym_i : (N,) float64
                Real and imaginary parts of received symbols.
            const_r, const_i : (M,) float64
                Real and imaginary parts of reference constellation.
            phi0 : float64
                Initial phase state in radians.
            b0, b1, b2 : float64
                Numerator coefficients of the 2nd-order Butterworth IIR filter.
            a1, a2 : float64
                Denominator coefficients (a[1], a[2]; a[0] is normalised to 1).
            is_sq_qam : bool
                Enables O(1) rounding decision for square QAM grids.
            levels : (side,) float64
            d_grid, lev_min : float64
            side : int

            Returns
            -------
            phase_est : (N,) float64
                Per-symbol phase trajectory φ[n].
            """
            N = len(sym_r)
            M = len(const_r)
            phase_est = np.empty(N, dtype=np.float64)
            phi = phi0

            # Biquad Direct Form II Transposed state variables
            w1 = 0.0
            w2 = 0.0

            for n in range(N):
                # Rotate received symbol by current phase estimate
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                yr = sym_r[n] * cos_phi + sym_i[n] * sin_phi
                yi = -sym_r[n] * sin_phi + sym_i[n] * cos_phi

                # Hard decision: argmin_{c ∈ C} |y − c|²
                if is_sq_qam:
                    r_idx = int(round((yr - lev_min) / d_grid))
                    if r_idx < 0:
                        r_idx = 0
                    elif r_idx >= side:
                        r_idx = side - 1
                    d_r = levels[r_idx]
                    i_idx = int(round((yi - lev_min) / d_grid))
                    if i_idx < 0:
                        i_idx = 0
                    elif i_idx >= side:
                        i_idx = side - 1
                    d_i = levels[i_idx]
                else:
                    min_d2 = (yr - const_r[0]) ** 2 + (yi - const_i[0]) ** 2
                    d_r = const_r[0]
                    d_i = const_i[0]
                    for k in range(1, M):
                        d2 = (yr - const_r[k]) ** 2 + (yi - const_i[k]) ** 2
                        if d2 < min_d2:
                            min_d2 = d2
                            d_r = const_r[k]
                            d_i = const_i[k]

                # Cross-product phase error: e = Im(y · d*) = yi·d_r − yr·d_i
                e = yi * d_r - yr * d_i

                # Record phase before update
                phase_est[n] = phi

                # Biquad (Direct Form II Transposed):
                #   v[n] = b0·e[n] + w1[n-1]
                #   w1[n] = b1·e[n] - a1·v[n] + w2[n-1]
                #   w2[n] = b2·e[n] - a2·v[n]
                v_out = b0 * e + w1
                w1 = b1 * e - a1 * v_out + w2
                w2 = b2 * e - a2 * v_out

                phi = phi + v_out

            return phase_est

        _NUMBA_PLL["dd_pll_bw"] = _dd_pll_bw_loop

    return _NUMBA_PLL["dd_pll_bw"]


def _get_numba_dd_pll_joint():
    """JIT-compile and cache the joint-channel DD-PLL PI kernel.

    Averages the cross-product phase error across C channels at each symbol
    before updating the single shared phase/frequency state.  This is the
    MVUE joint estimator for shared-LO systems.

    Returns
    -------
    callable
        Numba-compiled ``_dd_pll_joint_loop``.
    """
    if "dd_pll_joint" not in _NUMBA_PLL:
        import numba  # noqa: PLC0415

        @numba.njit(cache=True, fastmath=True, nogil=True)
        def _dd_pll_joint_loop(
            sym_r,
            sym_i,
            const_r,
            const_i,
            mu,
            beta,
            phi0,
            freq0,
            is_sq_qam,
            levels,
            d_grid,
            lev_min,
            side,
        ):
            """Joint-channel DD-PLL with PI loop filter.

            Parameters
            ----------
            sym_r, sym_i : (C, N) float64
                Real and imaginary parts of received symbols, all channels.
            const_r, const_i : (M,) float64
                Reference constellation.
            mu, beta, phi0, freq0 : float64
                Loop parameters — same semantics as ``_dd_pll_loop``.
            is_sq_qam : bool
                Enables O(1) rounding decision for square QAM grids.
            levels : (side,) float64
            d_grid, lev_min : float64
            side : int

            Returns
            -------
            phase_est : (N,) float64
                Single shared phase trajectory (broadcast to all channels by caller).
            """
            C = sym_r.shape[0]
            N = sym_r.shape[1]
            M = len(const_r)
            phase_est = np.empty(N, dtype=np.float64)
            phi = phi0
            freq = freq0

            for n in range(N):
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                e_sum = 0.0
                for c in range(C):
                    yr = sym_r[c, n] * cos_phi + sym_i[c, n] * sin_phi
                    yi = -sym_r[c, n] * sin_phi + sym_i[c, n] * cos_phi
                    if is_sq_qam:
                        r_idx = int(round((yr - lev_min) / d_grid))
                        if r_idx < 0:
                            r_idx = 0
                        elif r_idx >= side:
                            r_idx = side - 1
                        d_r = levels[r_idx]
                        i_idx = int(round((yi - lev_min) / d_grid))
                        if i_idx < 0:
                            i_idx = 0
                        elif i_idx >= side:
                            i_idx = side - 1
                        d_i = levels[i_idx]
                    else:
                        min_d2 = (yr - const_r[0]) ** 2 + (yi - const_i[0]) ** 2
                        d_r = const_r[0]
                        d_i = const_i[0]
                        for k in range(1, M):
                            d2 = (yr - const_r[k]) ** 2 + (yi - const_i[k]) ** 2
                            if d2 < min_d2:
                                min_d2 = d2
                                d_r = const_r[k]
                                d_i = const_i[k]
                    e_sum += yi * d_r - yr * d_i
                # Average error across channels — MVUE for shared LO
                e = e_sum / float(C)
                phase_est[n] = phi
                phi = phi + mu * e + freq
                freq = freq + beta * e

            return phase_est

        _NUMBA_PLL["dd_pll_joint"] = _dd_pll_joint_loop

    return _NUMBA_PLL["dd_pll_joint"]


def _get_numba_dd_pll_joint_butterworth():
    """JIT-compile and cache the joint-channel DD-PLL Butterworth kernel.

    Returns
    -------
    callable
        Numba-compiled ``_dd_pll_joint_bw_loop``.
    """
    if "dd_pll_joint_bw" not in _NUMBA_PLL:
        import numba  # noqa: PLC0415

        @numba.njit(cache=True, fastmath=True, nogil=True)
        def _dd_pll_joint_bw_loop(
            sym_r,
            sym_i,
            const_r,
            const_i,
            phi0,
            b0,
            b1,
            b2,
            a1,
            a2,
            is_sq_qam,
            levels,
            d_grid,
            lev_min,
            side,
        ):
            """Joint-channel DD-PLL with 2nd-order Butterworth loop filter.

            Parameters
            ----------
            sym_r, sym_i : (C, N) float64
            const_r, const_i : (M,) float64
            phi0 : float64
            b0, b1, b2, a1, a2 : float64
                Butterworth biquad coefficients.
            is_sq_qam : bool
                Enables O(1) rounding decision for square QAM grids.
            levels : (side,) float64
            d_grid, lev_min : float64
            side : int

            Returns
            -------
            phase_est : (N,) float64
            """
            C = sym_r.shape[0]
            N = sym_r.shape[1]
            M = len(const_r)
            phase_est = np.empty(N, dtype=np.float64)
            phi = phi0
            w1 = 0.0
            w2 = 0.0

            for n in range(N):
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                e_sum = 0.0
                for c in range(C):
                    yr = sym_r[c, n] * cos_phi + sym_i[c, n] * sin_phi
                    yi = -sym_r[c, n] * sin_phi + sym_i[c, n] * cos_phi
                    if is_sq_qam:
                        r_idx = int(round((yr - lev_min) / d_grid))
                        if r_idx < 0:
                            r_idx = 0
                        elif r_idx >= side:
                            r_idx = side - 1
                        d_r = levels[r_idx]
                        i_idx = int(round((yi - lev_min) / d_grid))
                        if i_idx < 0:
                            i_idx = 0
                        elif i_idx >= side:
                            i_idx = side - 1
                        d_i = levels[i_idx]
                    else:
                        min_d2 = (yr - const_r[0]) ** 2 + (yi - const_i[0]) ** 2
                        d_r = const_r[0]
                        d_i = const_i[0]
                        for k in range(1, M):
                            d2 = (yr - const_r[k]) ** 2 + (yi - const_i[k]) ** 2
                            if d2 < min_d2:
                                min_d2 = d2
                                d_r = const_r[k]
                                d_i = const_i[k]
                    e_sum += yi * d_r - yr * d_i
                e = e_sum / float(C)
                phase_est[n] = phi
                v_out = b0 * e + w1
                w1 = b1 * e - a1 * v_out + w2
                w2 = b2 * e - a2 * v_out
                phi = phi + v_out

            return phase_est

        _NUMBA_PLL["dd_pll_joint_bw"] = _dd_pll_joint_bw_loop

    return _NUMBA_PLL["dd_pll_joint_bw"]


_NUMBA_RTS: dict = {}


def _get_numba_rts_smoother():
    """JIT-compile and cache the Numba RTS-smoother kernel.

    Returns
    -------
    callable
        Numba-compiled ``_rts_loop``.
    """
    if "rts" not in _NUMBA_RTS:
        import numba  # noqa: PLC0415

        @numba.njit(cache=True, fastmath=True, nogil=True)
        def _rts_loop(phi_obs, sigma_p2, sigma_v2):
            """Rauch-Tung-Striebel smoother — Numba inner kernel.

            Parameters
            ----------
            phi_obs : (B,) float64
            sigma_p2 : float64
            sigma_v2 : float64

            Returns
            -------
            (B,) float64
            """
            B = len(phi_obs)
            x_filt = np.empty(B, dtype=np.float64)
            P_filt = np.empty(B, dtype=np.float64)
            x_pred = np.empty(B, dtype=np.float64)
            P_pred = np.empty(B, dtype=np.float64)

            x_filt[0] = phi_obs[0]
            P_filt[0] = sigma_v2

            for k in range(1, B):
                x_pred[k] = x_filt[k - 1]
                P_pred[k] = P_filt[k - 1] + sigma_p2
                K = P_pred[k] / (P_pred[k] + sigma_v2)
                x_filt[k] = x_pred[k] + K * (phi_obs[k] - x_pred[k])
                P_filt[k] = (1.0 - K) * P_pred[k]

            x_smooth = x_filt.copy()
            for k in range(B - 2, -1, -1):
                G = P_filt[k] / P_pred[k + 1]
                x_smooth[k] = x_filt[k] + G * (x_smooth[k + 1] - x_pred[k + 1])

            return x_smooth

        _NUMBA_RTS["rts"] = _rts_loop

    return _NUMBA_RTS["rts"]


def recover_carrier_phase_pll(
    symbols: ArrayType,
    modulation: str,
    order: int,
    mu: float = 1e-2,
    beta: float = 0.0,
    phase_init: float = 0.0,
    loop_filter: str = "pi",
    loop_bandwidth_normalized: float = 1e-3,
    joint_channels: bool = False,
    cycle_slip_correction: bool = False,
    cycle_slip_history: int = 100,
    cycle_slip_threshold: float = np.pi / 4,
    debug_plot: bool = False,
) -> ArrayType:
    r"""
    Carrier phase recovery via a Decision-Directed Phase-Locked Loop (DD-PLL).

    Tracks the carrier phase symbol-by-symbol using hard decisions as phase
    references.  A 1st-order loop (``beta=0``) corrects static or slowly
    varying phase noise; a 2nd-order loop (``beta > 0``) additionally tracks
    a residual frequency offset left over after coarse FOE.

    This is the standard streaming CPR for hardware implementations: it is
    modulation-format agnostic (works for any QAM/PSK order) and converges
    much faster than block-based methods (VV, BPS) after equalizer pull-in.

    .. warning::
        The DD-PLL requires reliable decisions at the input.  For a cold
        start the first ``~1/mu`` symbols may show slow convergence.
        A common strategy is to pre-converge with BPS or a short preamble
        and feed the resulting phase as ``phase_init``.

    Parameters
    ----------
    symbols : array_like
        1-SPS complex symbols after matched filtering and FOE.
        Shape: ``(N,)`` or ``(C, N)``.
    modulation : str
        Modulation scheme (case-insensitive): ``'qam'``, ``'psk'``, etc.
        Used to fetch the reference constellation via
        :func:`~commstools.mapping.gray_constellation`.
    order : int
        Modulation order (4, 16, 64, …).
    mu : float, default 1e-2
        Proportional gain — controls convergence speed and steady-state
        jitter.  Larger ``mu`` converges faster but amplifies noise.
        Typical range: ``1e-3`` (high-SNR, high-order QAM) to ``5e-2``
        (QPSK, low latency).
    beta : float, default 0.0
        Integral gain — enables 2nd-order frequency tracking.
        Set ``beta > 0`` when a residual frequency offset remains after
        FOE (e.g. ``beta ≈ mu² / 4``).  Zero gives a 1st-order loop.
    phase_init : float, default 0.0
        Initial phase state in radians.  Use the last sample of a
        preceding BPS or pilot-aided estimate to warm-start the loop.
    loop_filter : {"pi", "butterworth"}, default "pi"
        Loop filter type.

        * ``"pi"`` (default) — classic proportional-integral filter
          controlled by ``mu`` and ``beta``.
        * ``"butterworth"`` — 2nd-order Butterworth IIR loop filter
          designed via bilinear (Tustin) transform.  Controlled by
          ``loop_bandwidth_normalized`` instead of ``mu``/``beta``.
          Provides a flatter passband and sharper roll-off than PI,
          improving phase noise rejection at the cost of a small
          transient overshoot.

        When ``loop_filter='butterworth'``, ``mu`` and ``beta`` are
        ignored.
    loop_bandwidth_normalized : float, default 1e-3
        Normalised one-sided loop bandwidth as a fraction of the symbol
        rate (i.e. in the range ``(0, 0.5)``).  Only used when
        ``loop_filter='butterworth'``.  Typical values: ``1e-4``
        (narrow, low phase noise) to ``1e-2`` (wide, fast tracking).
    joint_channels : bool, default False
        For MIMO inputs (C > 1): if ``True``, average the cross-product
        phase error across all channels at each symbol before updating the
        shared loop state.  Both polarisations drive a single phase/frequency
        trajectory, giving ~√C variance reduction for shared-LO systems.
        The output ``phi_full[ch]`` rows are all identical.
        Has no effect for SISO (C = 1).
    cycle_slip_correction : bool, default False
        If ``True``, apply :func:`correct_cycle_slips` to the per-symbol
        phase trajectory after the loop, to detect and fix sudden ``π/2``
        jumps caused by incorrect hard decisions near the branch boundary.
    cycle_slip_history : int, default 100
        ``history_length`` passed to :func:`correct_cycle_slips`.
        Default is higher than for block-phase methods because the trajectory
        is per-symbol (not per-block).
    cycle_slip_threshold : float, default π/4
        ``threshold`` passed to :func:`correct_cycle_slips` (radians).

    Returns
    -------
    array_like
        Per-symbol phase estimate φ[n] in radians.
        Shape matches ``symbols``.  Same backend as input.

    Notes
    -----
    **Algorithm** (per sample n):

    .. math::

        y[n]       &= s[n] \cdot e^{-j\hat{\phi}[n]} \\
        \hat{d}[n] &= \operatorname{argmin}_{c \in \mathcal{C}}
                       \lvert y[n] - c \rvert^2 \\
        e[n]       &= \operatorname{Im}\!\bigl(y[n]\,\hat{d}^*[n]\bigr) \\
        \hat{\phi}[n+1] &= \hat{\phi}[n] + \mu e[n] + \nu[n] \\
        \nu[n]     &= \nu[n-1] + \beta e[n]

    where :math:`\nu` is the integral (frequency) state of the loop.

    **Backend notes:** The inner loop is inherently sequential (each sample
    depends on the previous phase state) and is compiled with Numba
    (``@njit``) for CPU performance.  When the input lives on a GPU
    (CuPy), samples are transparently moved to CPU for processing and
    the result is moved back — acceptable because the CPR loop is not
    the throughput bottleneck.

    **M-fold phase ambiguity:** Like VV and BPS, the DD-PLL may converge
    to any of the M constellation-symmetry-equivalent phases.  Resolve
    via a pilot symbol or known reference after CPR.

    References
    ----------
    I. Fatadin, D. Ives, and S. J. Savory, "Blind equalization and
    carrier phase recovery in a 16-QAM optical coherent system," *J.
    Lightw. Technol.*, vol. 27, no. 15, pp. 3042-3049, Aug. 2009.

    Md. S. Faruk and S. J. Savory, "Digital signal processing for coherent
    transceivers employing multilevel formats," *J. Lightw. Technol.*,
    vol. 35, no. 5, pp. 1125-1141, Mar. 2017, Sec. VIII.A, refs [65, 108].

    J. G. Proakis, *Digital Communications*, 4th ed., McGraw-Hill, 2001,
    ch. 6 (carrier phase synchronisation).
    """
    from .helpers import normalize
    from .mapping import gray_constellation

    if loop_filter not in ("pi", "butterworth"):
        raise ValueError(
            f"loop_filter must be 'pi' or 'butterworth', got {loop_filter!r}."
        )
    if loop_filter == "butterworth" and not (0.0 < loop_bandwidth_normalized < 0.5):
        raise ValueError(
            f"loop_bandwidth_normalized must be in (0, 0.5), got {loop_bandwidth_normalized}."
        )
    if loop_filter == "butterworth":
        if mu != 1e-2 or beta != 0.0:
            logger.warning(
                "loop_filter='butterworth': mu and beta are ignored. "
                "Use loop_bandwidth_normalized to control loop bandwidth."
            )

    symbols, xp, _ = dispatch(symbols)
    was_1d = symbols.ndim == 1
    if was_1d:
        symbols = symbols[None, :]
    C, N = symbols.shape

    # Normalise to unit average power so the effective loop gain is mu regardless
    # of input amplitude.  The error signal is e[n] = Im(y[n]*d_hat*), which
    # scales with signal amplitude; without this, the effective gain is mu*A
    # (where A is the RMS amplitude), making loop bandwidth input-dependent.
    symbols = normalize(symbols, mode="average_power", axis=-1)

    # Constellation on CPU (decisions are scalar operations in the loop)
    const_np = gray_constellation(modulation, order).astype(np.complex128)
    const_r = const_np.real.copy()
    const_i = const_np.imag.copy()

    # Square-QAM O(1) decision parameters.  For square QAM (order a perfect
    # square, e.g. 4/16/64/256/1024) the constellation is a uniform grid and
    # the nearest point can be found by rounding to the closest axis level.
    import math as _math  # noqa: PLC0415

    _sq_root = _math.isqrt(order)
    _is_sq_qam = ("qam" in modulation.lower()) and (_sq_root * _sq_root == order)
    if _is_sq_qam:
        _levels = np.unique(const_np.real).astype(np.float64)
        _d_grid = float(_levels[1] - _levels[0]) if len(_levels) > 1 else 1.0
        _lev_min = float(_levels[0])
        _side = _sq_root
    else:
        _levels = np.empty(0, dtype=np.float64)
        _d_grid = 1.0
        _lev_min = 0.0
        _side = 0

    # Move to CPU for sequential processing
    if xp is not np:
        symbols_cpu = to_device(symbols, "cpu")
    else:
        symbols_cpu = symbols

    phi_full = np.zeros((C, N), dtype=np.float64)
    use_joint = joint_channels and C > 1

    # Pre-build (C, N) float64 views used by joint kernels
    if use_joint:
        symbols_np = symbols_cpu.astype(np.complex128)
        sym_r_all = np.ascontiguousarray(symbols_np.real)  # (C, N) float64
        sym_i_all = np.ascontiguousarray(symbols_np.imag)

    if loop_filter == "butterworth":
        import scipy.signal as _ss  # noqa: PLC0415

        # Design 2nd-order Butterworth lowpass at loop_bandwidth_normalized
        # (normalised by Nyquist = 0.5 symbol rate, so Wn = 2 * lbw).
        b_arr, a_arr = _ss.butter(
            2, 2.0 * loop_bandwidth_normalized, btype="low", analog=False
        )
        b0, b1, b2 = float(b_arr[0]), float(b_arr[1]), float(b_arr[2])
        a1, a2 = float(a_arr[1]), float(a_arr[2])

        if use_joint:
            jbw_kernel = _get_numba_dd_pll_joint_butterworth()
            phi_joint = jbw_kernel(
                sym_r_all,
                sym_i_all,
                const_r,
                const_i,
                float(phase_init),
                b0,
                b1,
                b2,
                a1,
                a2,
                _is_sq_qam,
                _levels,
                _d_grid,
                _lev_min,
                _side,
            )
            for ch in range(C):
                phi_full[ch] = phi_joint
        else:
            bw_kernel = _get_numba_dd_pll_butterworth()
            for ch in range(C):
                sym = symbols_cpu[ch].astype(np.complex128)
                phi_full[ch] = bw_kernel(
                    sym.real.copy(),
                    sym.imag.copy(),
                    const_r,
                    const_i,
                    float(phase_init),
                    b0,
                    b1,
                    b2,
                    a1,
                    a2,
                    _is_sq_qam,
                    _levels,
                    _d_grid,
                    _lev_min,
                    _side,
                )
        loop_desc = f"Butterworth, BW={loop_bandwidth_normalized:.2g}"
    else:
        if use_joint:
            j_kernel = _get_numba_dd_pll_joint()
            phi_joint = j_kernel(
                sym_r_all,
                sym_i_all,
                const_r,
                const_i,
                float(mu),
                float(beta),
                float(phase_init),
                0.0,
                _is_sq_qam,
                _levels,
                _d_grid,
                _lev_min,
                _side,
            )
            for ch in range(C):
                phi_full[ch] = phi_joint
        else:
            pi_kernel = _get_numba_dd_pll()
            for ch in range(C):
                sym = symbols_cpu[ch].astype(np.complex128)
                phi_full[ch] = pi_kernel(
                    sym.real.copy(),
                    sym.imag.copy(),
                    const_r,
                    const_i,
                    float(mu),
                    float(beta),
                    float(phase_init),
                    0.0,
                    _is_sq_qam,
                    _levels,
                    _d_grid,
                    _lev_min,
                    _side,
                )
        loop_order = "2nd" if beta > 0.0 else "1st"
        loop_desc = f"PI {loop_order}-order, mu={mu}, beta={beta}"

    if cycle_slip_correction:
        if use_joint:
            # All rows are identical — correct once and broadcast
            phi_full[0] = correct_cycle_slips(
                phi_full[0],
                symmetry=4,
                history_length=cycle_slip_history,
                threshold=cycle_slip_threshold,
            )
            for ch in range(1, C):
                phi_full[ch] = phi_full[0]
        else:
            for ch in range(C):
                phi_full[ch] = correct_cycle_slips(
                    phi_full[ch],
                    symmetry=4,
                    history_length=cycle_slip_history,
                    threshold=cycle_slip_threshold,
                )

    # Move result back to original device
    if xp is not np:
        phi_full = xp.asarray(phi_full)

    phi_mean_deg = float(np.mean(phi_full)) * 180.0 / np.pi
    phi_std_deg = float(np.std(phi_full)) * 180.0 / np.pi
    logger.info(
        f"CPR (DD-PLL, {loop_desc}): phase mean={phi_mean_deg:.2f}°, "
        f"std={phi_std_deg:.2f}° [C={C}]"
    )

    if debug_plot:
        from . import plotting as _plotting

        _plotting.carrier_phase_trajectory(
            phi_full=phi_full if xp is np else to_device(phi_full, "cpu"),
            show=True,
            title=f"CPR — DD-PLL ({loop_desc})",
        )

    if was_1d:
        return phi_full[0]
    return phi_full


def recover_carrier_phase_viterbi_viterbi(
    symbols: ArrayType,
    modulation: str,
    order: int,
    block_size: int = 32,
    joint_channels: bool = False,
    cycle_slip_correction: bool = False,
    cycle_slip_history: int = 100,
    cycle_slip_threshold: float = np.pi / 4,
    debug_plot: bool = False,
) -> ArrayType:
    """
    Carrier phase recovery via the Viterbi-Viterbi (M-th power) algorithm.

    Block-based blind phase estimation for PSK and QAM symbols. Raises each
    block of symbols to the M-th power to remove modulation, extracts the
    block phase, resolves the M-fold ambiguity by unwrapping, then
    interpolates to per-symbol resolution.

    Parameters
    ----------
    symbols : array_like
        1-SPS complex symbols after matched filter. Shape: (N,) or (C, N).
    modulation : str
        Modulation scheme (case-insensitive): 'psk', 'qam', etc.
    order : int
        Modulation order.
    block_size : int, default 32
        Number of symbols per estimation block. Larger blocks reduce
        variance but reduce tracking bandwidth for fast phase noise.
        Typical range: 16-128 for QAM; as low as 1 for PSK (data cancels
        exactly in the M-th power for M-PSK constellations).
    joint_channels : bool, default False
        For MIMO inputs (C > 1): if ``True``, sum the M-th-power block
        phasors ``S_b`` across all channels before phase extraction.
        The resulting single trajectory is broadcast to all C output rows.
        Reduces variance by ~√C for shared-LO systems.  SISO-safe.
    cycle_slip_correction : bool, default False
        If ``True``, apply cycle-slip detection and correction
        (:func:`correct_cycle_slips`) after M-fold unwrap, before
        interpolation.
    cycle_slip_history : int, default 100
        ``history_length`` passed to :func:`correct_cycle_slips`.
    cycle_slip_threshold : float, default π/4
        ``threshold`` passed to :func:`correct_cycle_slips` (radians).
    debug_plot : bool, default False
        If ``True``, opens a diagnostic figure showing the per-symbol phase
        trajectory alongside the block-phase estimates.

    Returns
    -------
    array_like
        Per-symbol phase estimate in radians. Shape matches ``symbols``.
        Same backend as input.

    Notes
    -----
    Algorithm for each block ``b``:

    .. math::
        S_b = \\sum_{n \\in \\text{block } b} s[n]^M, \\quad
        \\hat{\\phi}_b = \\angle(S_b) / M

    **M-fold ambiguity resolution:** block phases are scaled by M,
    unwrapped in the 2π domain, then re-divided by M.  A global ``2π/M``
    phase ambiguity always remains — resolve it via a known pilot or
    preamble reference.

    .. warning::
        **Phase-unwrapping slip risk:** the unwrapper assumes consecutive block
        phases differ by less than ``π/M``.  Two independent effects can
        violate this:

        1. **Phase noise (high linewidth):** for a combined linewidth
           :math:`\\Delta\\nu`, the safety condition is:

           .. math::

               \\Delta\\nu \\cdot T_{\\text{block}} < 0.05 \\cdot f_s

           where :math:`T_{\\text{block}} = \\text{block\\_size} / f_s`.
           For example, 100 kHz linewidth at 32 Gbaud is safe up to
           ``block_size ≈ 16 000``, but 1 MHz linewidth requires
           ``block_size ≤ 1 600``.

        2. **Insufficient averaging for QAM (dominant at small block_size):**
           For PSK constellations every point maps to the *same* M-th power
           value, so the data modulation cancels exactly even with a single
           symbol per block.  For QAM constellations with ``order > 4`` the
           M-th power of individual symbols is **not** constant — it varies
           by constellation point.  Averaging over a block suppresses this
           residual, but a small ``block_size`` leaves significant variance
           that exceeds the unwrap threshold.  The minimum block size for
           reliable unwrapping scales roughly as
           :math:`4\\lceil\\sqrt{\\text{order}}\\rceil`:

           * 16-QAM → ``block_size ≥ 16``
           * 64-QAM → ``block_size ≥ 32``
           * 256-QAM → ``block_size ≥ 64``

           Using a ``block_size`` below this threshold will cause persistent
           ``2π/M`` phase slips regardless of SNR.

        When operating near or above the phase-noise limit, prefer
        :func:`recover_carrier_phase_bps`, which uses a brute-force phase
        search and does not require phase unwrapping.

    References
    ----------
    A. J. Viterbi and A. M. Viterbi, "Nonlinear estimation of PSK-modulated
    carrier phase with application to burst digital transmission," IEEE
    Trans. Inf. Theory, 1983.
    """
    symbols, xp, _ = dispatch(symbols)
    was_1d = symbols.ndim == 1
    if was_1d:
        symbols = symbols[None, :]  # (1, N)
    C, N = symbols.shape

    M = _modulation_power_m(modulation, order)

    N_trunc = (N // block_size) * block_size
    N_blocks = N_trunc // block_size

    if N_blocks == 0:
        raise ValueError(
            f"Signal length {N} is shorter than block_size={block_size}. "
            "Reduce block_size or use a longer symbol sequence."
        )

    # For QAM with order > 4 the M-th power of individual symbols does NOT cancel
    # the data modulation (unlike PSK, where every M-PSK point gives (c/|c|)^M = 1).
    # Sufficient block averaging is required so that the block-phase variance stays
    # below the π/M unwrap threshold.  The practical minimum scales as 4·ceil(√order).
    if "qam" in modulation.lower() and order > 4:
        _min_bs = max(8, 4 * int(np.ceil(order**0.5)))
        if block_size < _min_bs:
            logger.warning(
                f"CPR (VV): block_size={block_size} is too small for {order}-QAM. "
                f"Individual QAM symbols' M-th powers do not cancel the data modulation; "
                f"insufficient averaging causes block-phase variance that exceeds the "
                f"π/M unwrap threshold, producing persistent 2π/M phase slips. "
                f"Recommended minimum for {order}-QAM: block_size ≥ {_min_bs}."
            )

    # Reshape for block processing: (C, N_blocks, block_size).
    # Promote to complex128 for the M-th power — identical to estimate_frequency_offset_mth_power.
    # On GPU, complex64^4 loses precision near the ±π/M unwrap boundary, causing
    # spurious branch flips for high-order QAM with small block sizes.
    blocks = symbols[:, :N_trunc].reshape(C, N_blocks, block_size)
    blocks_c = blocks.astype(
        xp.complex128 if blocks.dtype == xp.complex64 else blocks.dtype
    )

    # For QAM, project to unit circle before the M-th power (normalized VV).
    # This removes outer-ring amplitude dominance and makes the π/M QAM bias
    # correction exact (by the 4-fold rotational symmetry of the constellation).
    # PSK is already constant-modulus; normalization is a no-op.
    if "qam" in modulation.lower():
        mag = xp.abs(blocks_c)
        blocks_c = blocks_c / xp.maximum(mag, 1e-15 * xp.max(mag))

    S_b = xp.sum(blocks_c**M, axis=-1)  # (C, N_blocks)

    # Block centre positions for interpolation (uniform spacing = block_size)
    block_centers = xp.arange(N_blocks, dtype=xp.float64) * block_size + block_size / 2
    all_positions = xp.arange(N, dtype=xp.float64)

    phi_full = xp.zeros((C, N), dtype=xp.float64)
    phi_blocks_out = xp.zeros((C, N_blocks), dtype=xp.float64)

    if joint_channels and C > 1:
        # Sum M-th-power phasors across channels → single block-phase trajectory
        S_b_joint = xp.sum(S_b, axis=0)  # (N_blocks,)
        phi_raw_joint = xp.angle(S_b_joint) / M
        phi_u_joint = xp.unwrap((phi_raw_joint * M).astype(xp.float64)) / M
        if "qam" in modulation.lower():
            phi_u_joint = phi_u_joint - (np.pi / M)
        if cycle_slip_correction:
            phi_u_joint_np = correct_cycle_slips(
                to_device(phi_u_joint, "cpu"),
                4,
                cycle_slip_history,
                cycle_slip_threshold,
            )
            phi_u_joint = xp.asarray(phi_u_joint_np)
        phi_interp = xp.interp(all_positions, block_centers, phi_u_joint)
        for ch in range(C):
            phi_full[ch] = phi_interp
            phi_blocks_out[ch] = phi_u_joint
    else:
        # Raw block phase in [-π/M, π/M)
        phi_raw = xp.angle(S_b) / M  # (C, N_blocks)

        # M-fold unwrap: scale into 2π domain, unwrap, re-scale back.
        # Cast to float64 before unwrap — cp.unwrap preserves input dtype so float32
        # would lose precision during the discontinuity test (diff vs 2π threshold).
        phi_u = (
            xp.unwrap((phi_raw * M).astype(xp.float64), axis=-1) / M
        )  # (C, N_blocks)

        # QAM bias correction.
        if "qam" in modulation.lower():
            phi_u = phi_u - (np.pi / M)

        # MIMO M-fold alignment: align every channel to channel 0's branch.
        # Skipped in joint mode (all channels share the same trajectory).
        if C > 1:
            for ch in range(1, C):
                diff = float(xp.mean(phi_u[ch] - phi_u[0]))
                k = round(diff * M / (2 * np.pi))
                phi_u[ch] = phi_u[ch] - k * (2 * np.pi / M)

        # xp.interp is 1D-only; loop over C channels.
        for ch in range(C):
            phi_u_ch = phi_u[ch]
            if cycle_slip_correction:
                phi_u_ch_np = correct_cycle_slips(
                    to_device(phi_u_ch, "cpu"),
                    4,
                    cycle_slip_history,
                    cycle_slip_threshold,
                )
                phi_u_ch = xp.asarray(phi_u_ch_np)
            phi_full[ch] = xp.interp(all_positions, block_centers, phi_u_ch)
            phi_blocks_out[ch] = phi_u_ch

    phi_full_np = to_device(phi_full, "cpu")
    phi_mean_deg = float(np.mean(phi_full_np)) * 180.0 / np.pi
    phi_std_deg = float(np.std(phi_full_np)) * 180.0 / np.pi
    mode_str = "joint" if (joint_channels and C > 1) else "independent"
    logger.info(
        f"CPR (Viterbi-Viterbi, M={M}, {mode_str}): phase mean={phi_mean_deg:.2f}°, "
        f"std={phi_std_deg:.2f}° [{N_blocks} blocks x {block_size} symbols, C={C}, "
        f"cycle_slip_correction={cycle_slip_correction}]"
    )

    if debug_plot:
        from . import plotting as _plotting

        _plotting.carrier_phase_trajectory(
            phi_full=phi_full_np,
            block_centers=to_device(block_centers, "cpu"),
            phi_blocks=to_device(phi_blocks_out, "cpu"),
            show=True,
            title="CPR — Viterbi-Viterbi",
        )

    if was_1d:
        return phi_full[0]
    return phi_full


def recover_carrier_phase_bps(
    symbols: ArrayType,
    modulation: str,
    order: int,
    num_test_phases: int = 64,
    block_size: int = 32,
    joint_channels: bool = False,
    cycle_slip_correction: bool = False,
    cycle_slip_history: int = 100,
    cycle_slip_threshold: float = np.pi / 4,
    pmf: Optional[np.ndarray] = None,
    debug_plot: bool = False,
) -> ArrayType:
    """
    Carrier phase recovery via Blind Phase Search (BPS).

    Tests ``num_test_phases`` candidate rotation angles over ``[0, π/2)``
    (exploiting 4-fold QAM symmetry), selects the candidate that minimises
    the block-averaged sum of minimum squared distances to the reference
    constellation, and interpolates to per-symbol resolution.

    Parameters
    ----------
    symbols : array_like
        1-SPS complex symbols after matched filter. Shape: (N,) or (C, N).
    modulation : str
        Modulation scheme (case-insensitive). Used to fetch the reference
        constellation via
        :func:`~commstools.mapping.gray_constellation`.
    order : int
        Modulation order.
    num_test_phases : int, default 64
        Number of candidate phase offsets B. Resolution is ``π/(2B)``
        rad per step. More candidates improve accuracy at higher compute cost.
    block_size : int, default 32
        Number of symbols per block for error-metric averaging.
        Very small values (< 4) make the 4-fold phase unwrap unreliable
        because noise on a single-symbol metric causes the best-candidate
        index to jump between non-adjacent phase bins between consecutive
        blocks, triggering false unwrap corrections.  Recommended
        minimum: ``block_size ≥ 4``.
    joint_channels : bool, default False
        For MIMO inputs (C > 1): if ``True``, sum the distance metrics
        across all channels before selecting the best phase candidate.
        The resulting single phase trajectory is broadcast to all C rows
        of the output (all channels identical before ambiguity resolution).
        Reduces phase estimation variance by ~√C for shared-LO systems.
        Has no effect for SISO (C = 1).
    cycle_slip_correction : bool, default False
        If ``True``, apply cycle-slip detection and correction
        (:func:`correct_cycle_slips`) to the block-phase trajectory
        after 4-fold unwrap, before interpolation.
    cycle_slip_history : int, default 100
        ``history_length`` passed to :func:`correct_cycle_slips`.
        Number of past corrected blocks used for linear extrapolation.
    cycle_slip_threshold : float, default π/4
        ``threshold`` passed to :func:`correct_cycle_slips` (radians).
    pmf : np.ndarray, optional
        Symbol PMF of shape ``(order,)`` for PS-QAM.  When provided, the
        reference constellation is scaled by ``1/sqrt(E_PS)`` (where
        ``E_PS = Σ P(s_m) |s_m|²`` on the normalised grid) so the
        nearest-neighbour distance metric matches the scale of the
        unit-avg-power input.  Without this, mid-shell PS points cross
        decision boundaries in the BPS metric and bias the phase estimate.
        No-op for uniform modulations.
    debug_plot : bool, default False
        If ``True``, opens a diagnostic figure showing the per-symbol phase
        trajectory alongside the block-phase estimates.

    Returns
    -------
    array_like
        Per-symbol phase estimate in radians. Shape matches ``symbols``.
        Same backend as input.

    Notes
    -----
    Algorithm:

    1. Candidates: :math:`\\phi_k = k\\,\\pi/(2B)` for :math:`k=0,\\ldots,B-1`.
    2. Rotate: ``x_rot[n, k] = symbols[n] · exp(-j·φ_k)``. Shape: (N, B).
    3. Min dist: ``d²[n,k] = min_c |x_rot[n,k]-c|²``. Shape: (N, B).
    4. Block sum: ``metric[b,k] = Σ d²[n,k]`` over block ``b``.
    5. Best phase: ``φ_b = candidates[argmin(metric[b,:])]``.
    6. 4-fold unwrap, optional cycle-slip correction, per-symbol interpolation.

    .. note::
        The candidate search covers :math:`[0, \\pi/2)` and the unwrap
        exploits the **4-fold** symmetry of square QAM constellations.
        For PSK modulations whose symmetry order differs from 4, the
        candidate range and unwrap fold should be adjusted accordingly.

    **Memory:** The ``(N, B, M_const)`` distance tensor scales as
    ``N·B·M_const·8`` bytes.  For N=10 000, B=64, M=256 → ~1.3 GB.
    Reduce ``num_test_phases`` or process shorter segments for high-order
    constellations.

    **4-fold ambiguity:** A global ``π/2`` phase offset remains after
    unwrapping.  Resolve via a pilot or preamble phase reference.

    References
    ----------
    T. Pfau, S. Hoffmann, and R. Noe, "Hardware-efficient coherent digital
    receiver concept with feedforward carrier recovery for M-QAM
    constellations," J. Lightw. Technol., 2009.
    """
    from .mapping import gray_constellation

    from .helpers import normalize

    symbols, xp, _ = dispatch(symbols)
    was_1d = symbols.ndim == 1
    if was_1d:
        symbols = symbols[None, :]
    C, N = symbols.shape

    # Normalise each channel to unit average power so the metric is computed at
    # the same scale as the reference constellation (gray_constellation returns
    # unit-average-power points).  BPS is a phase estimator; it must be
    # amplitude-agnostic.
    symbols = normalize(symbols, mode="average_power", axis=-1)

    # Reference constellation on the same device
    const_np = gray_constellation(modulation, order)

    # PS-QAM: unit-avg-power input lives on the ``{s_m/sqrt(E_PS)}`` grid.
    # Rescale the comparison constellation to the same grid so the nearest-
    # neighbour distance metric is correct.  Skip on uniform PMF.
    if pmf is not None:
        pmf_arr = np.asarray(pmf, dtype=np.float64)
        e_ps = float(np.dot(pmf_arr, np.abs(const_np) ** 2))
        if e_ps < 1.0 - 1e-6:
            const_np = const_np / np.sqrt(e_ps)

    const_xp = xp.asarray(const_np)  # (M_const,)

    # Candidate test phases over [0, π/2)
    B = num_test_phases
    candidates = xp.arange(B, dtype=symbols.real.dtype) * (np.pi / 2.0 / B)  # (B,)

    N_trunc = (N // block_size) * block_size
    N_blocks = N_trunc // block_size

    if N_blocks == 0:
        raise ValueError(
            f"Signal length {N} is shorter than block_size={block_size}. "
            "Reduce block_size or use a longer symbol sequence."
        )

    # Very small block_size makes the 4-fold phase unwrap unreliable: with only
    # one or two symbols per block the noise on the distance-metric argmin causes
    # large candidate-index jumps between consecutive blocks, triggering false
    # 4-fold unwrap corrections.  Warn early so users diagnose this easily.
    if block_size < 4:
        logger.warning(
            f"CPR (BPS): block_size={block_size} is very small. "
            f"Averaging the distance metric over only {block_size} symbol(s) per block "
            f"makes the 4-fold phase unwrap unreliable. Recommended minimum: block_size ≥ 4."
        )

    # block_centers[b] = b * block_size + block_size/2  (consistent with VV)
    block_centers = xp.arange(N_blocks, dtype=xp.float64) * block_size + block_size / 2

    all_positions = xp.arange(N, dtype=xp.float64)

    # Pre-compute interpolation indices and weights (identical for every channel).
    # block b is "to the left" of position n when its centre b*bs + bs/2 <= n
    #   => b <= (n - bs/2) / bs  => idx_left = floor((n - bs/2) / bs)
    idx_left = xp.clip(
        xp.floor((all_positions - block_size / 2) / block_size).astype(xp.int64),
        0,
        N_blocks - 2,
    )  # (N,)
    idx_right = idx_left + 1  # (N,)
    t_interp = xp.clip(
        (all_positions - block_centers[idx_left]) / block_size, 0.0, 1.0
    )  # (N,)

    # Pre-compute phasors for all B candidates once (avoid redundant exp per channel)
    dtype_c = xp.complex64 if symbols.dtype == xp.complex64 else xp.complex128
    phasors = xp.exp(-1j * candidates.astype(xp.float64)).astype(dtype_c)  # (B,)

    # For square QAM (order a perfect square): the nearest constellation point
    # can be found in O(1) per symbol via per-component rounding, eliminating
    # the (CHUNK, B, M_const) distance tensor entirely.
    side = int(order**0.5)
    is_sq_qam = ("qam" in modulation.lower()) and (side * side == order)
    if is_sq_qam:
        # Sorted unique real levels of the constellation (shape: (side,))
        levels = xp.sort(xp.unique(const_xp.real))
        d_grid = float(levels[1] - levels[0])  # uniform grid spacing
        lev_min = float(levels[0])

    float_dtype = xp.float32 if symbols.dtype == xp.complex64 else xp.float64

    # Chunk size for N axis: bounds peak memory of the distance tensor.
    # Always a multiple of block_size so each chunk covers a whole number of
    # blocks exactly.  Rounded up to the nearest multiple ≥ 1024.
    CHUNK_N = max(block_size, ((1024 + block_size - 1) // block_size) * block_size)

    phi_full = xp.zeros((C, N), dtype=xp.float64)
    phi_blocks = xp.zeros((C, N_blocks), dtype=xp.float64)

    # Accumulate per-channel distance metrics (N_blocks, B) for all channels.
    metrics_all = xp.zeros((C, N_blocks, B), dtype=float_dtype)

    for ch in range(C):
        sym = symbols[ch, :N_trunc]  # (N_trunc,)

        for n0 in range(0, N_trunc, CHUNK_N):
            n1 = min(n0 + CHUNK_N, N_trunc)
            x_rot = sym[n0:n1, None] * phasors[None, :]  # (CHUNK, B)

            if is_sq_qam:
                # O(1) nearest-point: round each component to the nearest grid level
                r_idx = xp.clip(
                    xp.round((x_rot.real - lev_min) / d_grid).astype(xp.int64),
                    0,
                    side - 1,
                )
                i_idx = xp.clip(
                    xp.round((x_rot.imag - lev_min) / d_grid).astype(xp.int64),
                    0,
                    side - 1,
                )
                r_near = levels[r_idx]  # (CHUNK, B)
                i_near = levels[i_idx]  # (CHUNK, B)
                chunk_min_d = (
                    (x_rot.real - r_near) ** 2 + (x_rot.imag - i_near) ** 2
                ).astype(float_dtype)
            else:
                # General: (CHUNK, B, M_const) — bounded by CHUNK_N
                d_sq = xp.abs(x_rot[:, :, None] - const_xp[None, None, :]) ** 2
                chunk_min_d = xp.min(d_sq, axis=-1).astype(float_dtype)

            b0 = n0 // block_size
            n_b = (n1 - n0) // block_size
            metrics_all[ch, b0 : b0 + n_b] = chunk_min_d.reshape(
                n_b, block_size, B
            ).sum(axis=1)

    # Phase estimation: joint (sum metrics across channels) or independent per channel.
    if joint_channels and C > 1:
        metric_joint = xp.sum(metrics_all, axis=0)  # (N_blocks, B)
        best_k_joint = xp.argmin(metric_joint, axis=-1)  # (N_blocks,)
        phi_b_joint = candidates[best_k_joint]  # (N_blocks,)
        phi_u_joint = xp.unwrap(phi_b_joint.astype(xp.float64) * 4, axis=-1) / 4
        if cycle_slip_correction:
            phi_u_joint_np = correct_cycle_slips(
                to_device(phi_u_joint, "cpu"),
                4,
                cycle_slip_history,
                cycle_slip_threshold,
            )
            phi_u_joint = xp.asarray(phi_u_joint_np)
        for ch in range(C):
            phi_full[ch] = (
                phi_u_joint[idx_left] * (1.0 - t_interp)
                + phi_u_joint[idx_right] * t_interp
            )
            phi_blocks[ch] = phi_u_joint
    else:
        for ch in range(C):
            metric = metrics_all[ch]  # (N_blocks, B)
            best_k = xp.argmin(metric, axis=-1)  # (N_blocks,)
            phi_b = candidates[best_k]  # (N_blocks,)
            phi_u = xp.unwrap(phi_b.astype(xp.float64) * 4, axis=-1) / 4
            if cycle_slip_correction:
                phi_u_np = correct_cycle_slips(
                    to_device(phi_u, "cpu"), 4, cycle_slip_history, cycle_slip_threshold
                )
                phi_u = xp.asarray(phi_u_np)
            phi_full[ch] = (
                phi_u[idx_left] * (1.0 - t_interp) + phi_u[idx_right] * t_interp
            )
            phi_blocks[ch] = phi_u

    phi_full_np = to_device(phi_full, "cpu")
    phi_mean_deg = float(np.mean(phi_full_np)) * 180.0 / np.pi
    phi_std_deg = float(np.std(phi_full_np)) * 180.0 / np.pi
    mode_str = "joint" if (joint_channels and C > 1) else "independent"
    logger.info(
        f"CPR (BPS, B={B}, {mode_str}): phase mean={phi_mean_deg:.2f}°, std={phi_std_deg:.2f}° "
        f"[{N_blocks} blocks x {block_size} symbols, C={C}, "
        f"cycle_slip_correction={cycle_slip_correction}]"
    )

    if debug_plot:
        from . import plotting as _plotting

        _plotting.carrier_phase_trajectory(
            phi_full=phi_full_np,
            show=True,
            title="CPR — Blind Phase Search",
        )

    if was_1d:
        return phi_full[0]
    return phi_full


def _rts_smoother_1d(
    phi_obs: np.ndarray,
    sigma_p2: float,
    sigma_v2: float,
) -> np.ndarray:
    """Rauch-Tung-Striebel (RTS) Kalman smoother for a 1-D random-walk state.

    Uses the Numba-compiled kernel (:func:`_get_numba_rts_smoother`) when
    available; falls back to a pure-Python loop otherwise.  Always runs on
    CPU — call with a NumPy array; the caller is responsible for
    ``to_device`` conversion.

    State model  : x[k+1] = x[k] + w[k],   w ~ N(0, sigma_p2)
    Observation  : y[k]   = x[k] + v[k],   v ~ N(0, sigma_v2)

    Parameters
    ----------
    phi_obs : (B,) float64
        Noisy block-phase observations in radians (e.g. from VV).
    sigma_p2 : float
        Process noise variance per block (Wiener phase noise increment).
    sigma_v2 : float
        Observation noise variance (VV estimator variance per block).

    Returns
    -------
    (B,) float64
        MAP-smoothed phase trajectory.
    """
    return _get_numba_rts_smoother()(phi_obs, float(sigma_p2), float(sigma_v2))


def _sskf_smoother_1d(
    phi_obs: ArrayType,
    sigma_p2: float,
    sigma_v2: float,
    sp,
    xp,
) -> ArrayType:
    """Steady-state Kalman smoother via zero-phase IIR filter (filtfilt).

    Approximates the RTS smoother by replacing the sequential Kalman
    recurrence with a 1st-order IIR filter whose gain is solved analytically
    from the discrete algebraic Riccati equation.  The bidirectional
    ``filtfilt`` call makes it equivalent to the RTS smoother in steady state.

    Backend-aware: uses ``sp.signal.filtfilt`` where ``sp`` is
    ``scipy`` (CPU) or ``cupyx.scipy`` (GPU) as returned by
    :func:`~commstools.backend.dispatch`.

    The approximation is excellent when ``B >> 1/K_∞``
    (typically ``B > 20``).  For ``B < 7`` (``filtfilt`` minimum), falls
    back to the exact :func:`_rts_smoother_1d` on CPU.

    Parameters
    ----------
    phi_obs : (B,) float64, on the target device
        Noisy block-phase observations in radians.
    sigma_p2, sigma_v2 : float
        Process and observation noise variances per block.
    sp : module
        ``scipy`` or ``cupyx.scipy``, from :func:`~commstools.backend.dispatch`.
    xp : module
        ``numpy`` or ``cupy``, from :func:`~commstools.backend.dispatch`.

    Returns
    -------
    (B,) float64, same device as ``phi_obs``.
    """
    # filtfilt requires at least padlen * 2 + 1 samples; padlen = 3 * max(len(b), len(a)) = 6
    if len(phi_obs) < 7:
        phi_np = to_device(phi_obs, "cpu")
        return xp.asarray(_rts_smoother_1d(phi_np, sigma_p2, sigma_v2))

    # Steady-state prediction error covariance from discrete Riccati equation:
    #   p² - σ_p²·p - σ_p²·σ_v² = 0  →  p = (σ_p² + √(σ_p⁴ + 4σ_p²σ_v²)) / 2
    p_ss = (sigma_p2 + float(np.sqrt(sigma_p2**2 + 4.0 * sigma_p2 * sigma_v2))) / 2.0
    K_ss = p_ss / (p_ss + sigma_v2)

    # Forward IIR:  y[k] = (1-K)·y[k-1] + K·x[k]
    #   H(z) = K / (1 - (1-K)·z⁻¹)
    # filtfilt applies forward + backward  →  zero-phase, ≡ RTS smoother at
    # steady state.
    b = [K_ss]
    a = [1.0, -(1.0 - K_ss)]
    return sp.signal.filtfilt(b, a, phi_obs)


def recover_carrier_phase_tikhonov(
    symbols: ArrayType,
    modulation: str,
    order: int,
    linewidth_symbol_periods: float,
    block_size: int = 32,
    snr_db: Optional[float] = None,
    method: str = "exact",
    joint_channels: bool = False,
    cycle_slip_correction: bool = False,
    cycle_slip_history: int = 100,
    cycle_slip_threshold: float = np.pi / 4,
    debug_plot: bool = False,
) -> ArrayType:
    r"""
    Carrier phase recovery via MAP estimation with a Tikhonov/Wiener phase
    noise prior (Colavolpe et al., 2005).

    Extends the Viterbi-Viterbi block estimator with a Kalman smoother
    matched to the laser phase noise statistics.  Two smoother backends are
    available via ``method``:

    * ``'exact'`` — full Rauch-Tung-Striebel (RTS) smoother; Numba-compiled.
      Exact for all sequence lengths; runs on CPU.
    * ``'sskf'`` — steady-state Kalman filter approximation via zero-phase
      IIR (``filtfilt``); backend-aware (stays on GPU when input is on GPU).
      Approximation holds for ``N_blocks >> 1/K_∞`` (~20+ blocks typical).

    Parameters
    ----------
    symbols : array_like
        1-SPS complex symbols after matched filter and FOE.
        Shape: ``(N,)`` or ``(C, N)``.
    modulation : str
        Modulation scheme (case-insensitive): ``'psk'``, ``'qam'``, etc.
    order : int
        Modulation order.
    linewidth_symbol_periods : float
        Combined linewidth-symbol-time product :math:`\Delta\nu \cdot T_s`.
        Typical values: ``1e-5`` (narrow laser, 32 GBd), ``5e-4`` (wide
        laser / high baud rate).  Sets the Kalman process noise variance:
        :math:`\sigma_p^2 = 2\pi \cdot \Delta\nu T_s \cdot N_b`.
    block_size : int, default 32
        Symbols per VV estimation block.  Same trade-off as for
        :func:`recover_carrier_phase_viterbi_viterbi`.
    snr_db : float or None, default None
        Per-symbol SNR in dB.  Used to compute the VV observation noise
        variance :math:`\sigma_v^2 \approx 1/(M^2 \cdot \mathrm{SNR} \cdot N_b)`.
        If ``None``, defaults to 20 dB with a warning — provide the actual
        operating SNR for the optimal smoother bandwidth.
    method : {'exact', 'sskf'}, default 'exact'
        Smoother implementation:

        * ``'exact'``: full RTS smoother (:func:`_rts_smoother_1d`); Numba
          kernel when available.  Sequential CPU recurrence; exact for any
          ``N_blocks``.
        * ``'sskf'``: steady-state approximation via ``filtfilt``
          (:func:`_sskf_smoother_1d`); runs on the input device (GPU-native
          when data is on GPU).  Excellent for ``N_blocks ≥ 20``; for
          ``N_blocks < 7`` silently falls back to ``'exact'``.
    joint_channels : bool, default False
        For MIMO inputs (C > 1): if ``True``, sum the M-th-power block
        phasors across all channels before the VV phase extraction and
        Kalman smoother.  The single smoothed trajectory is broadcast to
        all C output rows.  Reduces variance by ~√C for shared-LO systems.
    cycle_slip_correction : bool, default False
        If ``True``, apply cycle-slip detection and correction
        (:func:`correct_cycle_slips`) after the Kalman smoother, before
        interpolation.
    cycle_slip_history : int, default 100
        ``history_length`` passed to :func:`correct_cycle_slips`.
    cycle_slip_threshold : float, default π/4
        ``threshold`` passed to :func:`correct_cycle_slips` (radians).
    debug_plot : bool, default False
        If ``True``, opens a diagnostic figure showing the per-symbol phase
        trajectory with the Kalman-smoothed block phases.

    Returns
    -------
    array_like
        Per-symbol phase estimate in radians.  Shape matches ``symbols``.
        Same backend as input.

    Notes
    -----
    **Algorithm:**

    1. Compute VV block phases using normalized M-th power (unit-circle
       projection before raising to the M-th power removes QAM amplitude
       bias; see :func:`recover_carrier_phase_viterbi_viterbi`).
    2. Apply the Kalman smoother with:

       .. math::

           \sigma_p^2 &= 2\pi \cdot \Delta\nu T_s \cdot N_b \\
           \sigma_v^2 &\approx \frac{1}{M^2 \cdot \mathrm{SNR} \cdot N_b}

       where :math:`N_b` = ``block_size`` and :math:`M` is the modulation
       exponent from :func:`_modulation_power_m`.
    3. Interpolate smoothed block phases to per-symbol resolution (linear,
       consistent with VV).

    **M-fold ambiguity:** same as VV — a residual ``2π/M`` phase offset
    always remains.  Resolve via a pilot or preamble reference.

    References
    ----------
    G. Colavolpe, A. Barbieri, and G. Caire, "Algorithms for iterative
    decoding in the presence of strong phase noise," *IEEE J. Sel. Areas
    Commun.*, vol. 23, no. 9, pp. 1748-1757, Sep. 2005.

    A. J. Viterbi and A. M. Viterbi, "Nonlinear estimation of PSK-modulated
    carrier phase with application to burst digital transmission," *IEEE
    Trans. Inf. Theory*, 1983.
    """
    if method not in ("exact", "sskf"):
        raise ValueError(f"Unknown method {method!r}. Choose 'exact' or 'sskf'.")

    symbols, xp, sp = dispatch(symbols)
    was_1d = symbols.ndim == 1
    if was_1d:
        symbols = symbols[None, :]
    C, N = symbols.shape

    M = _modulation_power_m(modulation, order)

    N_trunc = (N // block_size) * block_size
    N_blocks = N_trunc // block_size

    if N_blocks == 0:
        raise ValueError(
            f"Signal length {N} is shorter than block_size={block_size}. "
            "Reduce block_size or use a longer symbol sequence."
        )

    # Same data-residual constraint as VV: for QAM with order > 4 the M-th power
    # does not cancel per symbol.  Block phase variance can exceed π/M before the
    # Kalman smoother is applied, causing unwrap slips that the smoother cannot fix.
    if "qam" in modulation.lower() and order > 4:
        _min_bs = max(8, 4 * int(np.ceil(order**0.5)))
        if block_size < _min_bs:
            logger.warning(
                f"CPR (Tikhonov): block_size={block_size} is too small for {order}-QAM. "
                f"Block phases are estimated via Viterbi-Viterbi; the data-residual "
                f"constraint is identical — see recover_carrier_phase_viterbi_viterbi. "
                f"Recommended minimum for {order}-QAM: block_size ≥ {_min_bs}."
            )

    # Smoother noise parameters
    if snr_db is None:
        logger.warning(
            "CPR (Tikhonov): snr_db not provided — defaulting to 20 dB. "
            "Pass the operating SNR for the optimal smoother bandwidth."
        )
        snr_lin = 100.0  # 20 dB default
    else:
        snr_lin = 10.0 ** (snr_db / 10.0)

    sigma_p2 = float(2.0 * np.pi * linewidth_symbol_periods * block_size)
    sigma_v2 = float(1.0 / (M**2 * snr_lin * block_size))

    # VV block phase estimation with unit-circle normalisation for QAM
    blocks = symbols[:, :N_trunc].reshape(C, N_blocks, block_size)
    blocks_c = blocks.astype(
        xp.complex128 if blocks.dtype == xp.complex64 else blocks.dtype
    )
    if "qam" in modulation.lower():
        mag = xp.abs(blocks_c)
        blocks_c = blocks_c / xp.maximum(mag, 1e-15 * xp.max(mag))

    S_b = xp.sum(blocks_c**M, axis=-1)  # (C, N_blocks)

    block_centers = xp.arange(N_blocks, dtype=xp.float64) * block_size + block_size / 2
    all_positions = xp.arange(N, dtype=xp.float64)
    phi_full = xp.zeros((C, N), dtype=xp.float64)

    if joint_channels and C > 1:
        # Sum M-th-power phasors → single VV estimate → single Kalman pass
        S_b_joint = xp.sum(S_b, axis=0)  # (N_blocks,)
        phi_raw_joint = xp.angle(S_b_joint) / M
        phi_u_joint = xp.unwrap((phi_raw_joint * M).astype(xp.float64)) / M
        if "qam" in modulation.lower():
            phi_u_joint = phi_u_joint - (np.pi / M)

        # Kalman smoother on the joint trajectory
        if method == "exact":
            phi_u_joint_np = to_device(phi_u_joint, "cpu")
            phi_smooth_joint_np = _rts_smoother_1d(phi_u_joint_np, sigma_p2, sigma_v2)
            phi_smooth_joint = xp.asarray(phi_smooth_joint_np)
        else:
            phi_smooth_joint = _sskf_smoother_1d(
                phi_u_joint, sigma_p2, sigma_v2, sp, xp
            )
            phi_smooth_joint_np = to_device(phi_smooth_joint, "cpu")

        if cycle_slip_correction:
            phi_smooth_joint_np = correct_cycle_slips(
                to_device(phi_smooth_joint, "cpu"),
                4,
                cycle_slip_history,
                cycle_slip_threshold,
            )
            phi_smooth_joint = xp.asarray(phi_smooth_joint_np)

        phi_interp = xp.interp(all_positions, block_centers, phi_smooth_joint)
        for ch in range(C):
            phi_full[ch] = phi_interp
        phi_smooth_np = np.tile(to_device(phi_smooth_joint, "cpu"), (C, 1))
    else:
        phi_raw = xp.angle(S_b) / M
        phi_u = (
            xp.unwrap((phi_raw * M).astype(xp.float64), axis=-1) / M
        )  # (C, N_blocks)

        if "qam" in modulation.lower():
            phi_u = phi_u - (np.pi / M)

        if C > 1:
            for ch in range(1, C):
                diff = float(xp.mean(phi_u[ch] - phi_u[0]))
                k = round(diff * M / (2 * np.pi))
                phi_u[ch] = phi_u[ch] - k * (2 * np.pi / M)

        # Kalman smoother — dispatch on method
        if method == "exact":
            phi_u_np = to_device(phi_u, "cpu")  # (C, N_blocks) float64
            phi_smooth_np = np.empty_like(phi_u_np)
            for ch in range(C):
                phi_smooth_np[ch] = _rts_smoother_1d(phi_u_np[ch], sigma_p2, sigma_v2)
            phi_smooth = xp.asarray(phi_smooth_np)
        else:  # method == "sskf"
            phi_smooth = xp.empty_like(phi_u)
            for ch in range(C):
                phi_smooth[ch] = _sskf_smoother_1d(
                    phi_u[ch], sigma_p2, sigma_v2, sp, xp
                )
            phi_smooth_np = to_device(phi_smooth, "cpu")

        for ch in range(C):
            phi_s_ch = phi_smooth[ch]
            if cycle_slip_correction:
                phi_s_ch_np = correct_cycle_slips(
                    to_device(phi_s_ch, "cpu"),
                    4,
                    cycle_slip_history,
                    cycle_slip_threshold,
                )
                phi_s_ch = xp.asarray(phi_s_ch_np)
                phi_smooth_np[ch] = to_device(phi_s_ch, "cpu")
            phi_full[ch] = xp.interp(all_positions, block_centers, phi_s_ch)

    phi_full_np = to_device(phi_full, "cpu")
    phi_mean_deg = float(np.mean(phi_full_np)) * 180.0 / np.pi
    phi_std_deg = float(np.std(phi_full_np)) * 180.0 / np.pi
    mode_str = "joint" if (joint_channels and C > 1) else "independent"
    logger.info(
        f"CPR (Tikhonov-{method.upper()}, M={M}, {mode_str}): "
        f"phase mean={phi_mean_deg:.2f}°, std={phi_std_deg:.2f}° "
        f"[{N_blocks} blocks x {block_size}, σ_p²={sigma_p2:.2e}, σ_v²={sigma_v2:.2e}, "
        f"C={C}, cycle_slip_correction={cycle_slip_correction}]"
    )

    if debug_plot:
        from . import plotting as _plotting

        _plotting.carrier_phase_trajectory(
            phi_full=phi_full_np,
            block_centers=to_device(block_centers, "cpu"),
            phi_blocks=phi_smooth_np,
            show=True,
            title=f"CPR — Tikhonov-{method.upper()}",
        )

    if was_1d:
        return phi_full[0]
    return phi_full


def recover_carrier_phase_pilots(
    symbols: ArrayType,
    pilot_indices: ArrayType,
    pilot_values: ArrayType,
    interpolation: str = "linear",
    joint_channels: bool = False,
    cycle_slip_correction: bool = False,
    cycle_slip_history: int = 100,
    cycle_slip_threshold: float = np.pi / 4,
    debug_plot: bool = False,
) -> ArrayType:
    """
    Carrier phase recovery using known pilot symbols.

    Computes the phase error at each pilot position, unwraps the pilot
    phase sequence, and interpolates across the full symbol grid.

    Parameters
    ----------
    symbols : array_like
        Received 1-SPS complex symbols. Shape: (N,) or (C, N).
    pilot_indices : array_like of int
        Indices of pilot symbols within the frame, in increasing order.
        Shape: (P,).
    pilot_values : array_like
        Known transmitted pilot constellation points.
        Shape: (P,) for shared pilots (broadcast to all MIMO channels),
        or (C, P) for per-channel pilots.
    interpolation : {'linear', 'cubic'}, default 'linear'
        Interpolation method between pilot positions.  Both modes loop over
        MIMO channels (``xp.interp`` and ``CubicSpline`` are 1D-only);
        C is typically 1-4 so the overhead is negligible.  ``'cubic'`` uses
        :class:`scipy.interpolate.CubicSpline` (CPU) or
        :class:`cupyx.scipy.interpolate.CubicSpline` (GPU) with natural
        boundary conditions (zero second derivative at endpoints) and
        constant-hold extrapolation outside the pilot span.
    joint_channels : bool, default False
        For MIMO inputs (C > 1): if ``True``, perform coherent complex
        averaging of ``r_pilot * conj(s_pilot)`` across all channels before
        calling ``angle()``.  This avoids wrap-around artefacts that arise
        when averaging phases directly, and reduces variance by ~√C for
        shared-LO systems.  The resulting single phase trajectory is broadcast
        to all C output rows.  Has no effect for SISO (C = 1).
    cycle_slip_correction : bool, default False
        If ``True``, apply :func:`correct_cycle_slips` to the unwrapped pilot
        phase sequence before interpolation, with ``symmetry=1`` (correction
        quantum ``2π``) to detect and fix wrap-around errors introduced by
        ``xp.unwrap`` at large inter-pilot gaps.
    cycle_slip_history : int, default 100
        ``history_length`` passed to :func:`correct_cycle_slips`.
    cycle_slip_threshold : float, default π/4
        ``threshold`` passed to :func:`correct_cycle_slips` (radians).
    debug_plot : bool, default False
        If ``True``, opens a diagnostic figure showing the unwrapped pilot
        phase sequence and the interpolated phase trajectory.

    Returns
    -------
    array_like
        Per-symbol phase estimate in radians. Shape matches ``symbols``.
        Same backend as input.

    Notes
    -----
    Phase at pilot position :math:`k`:

    .. math::
        \\hat{\\phi}[k] = \\angle\\!\\left(
            r[\\mathrm{pilot\\_indices}[k]] \\cdot
            s^*[\\mathrm{pilot\\_values}[k]]
        \\right)

    For ``'linear'``, boundary extrapolation holds the first/last pilot value
    (constant hold — identical to ``numpy.interp`` behaviour).  For
    ``'cubic'``, the spline uses natural boundary conditions (zero second
    derivative at endpoints), and symbols before the first pilot or after
    the last pilot are filled with the respective boundary pilot value,
    preventing edge oscillation.

    .. note::
        **Single-carrier use only.**  This function tracks carrier phase across
        a linear symbol stream using scattered pilot positions.  For OFDM
        systems, phase noise is tracked as *common phase error* (CPE) across
        pilot *subcarriers* within each OFDM symbol (e.g. 5G NR PTRS,
        DVB-T2 continual pilots) — a structurally different problem not
        covered here.

    References
    ----------
    S. J. Savory, "Digital filters for coherent optical receivers,"
    Optics Express, 2008.
    """
    symbols, xp, _ = dispatch(symbols)
    was_1d = symbols.ndim == 1
    if was_1d:
        symbols = symbols[None, :]
    C, N = symbols.shape

    pilot_indices_np = to_device(pilot_indices, "cpu").astype(np.intp)
    pilot_indices_xp = xp.asarray(pilot_indices, dtype=xp.float64)
    pilot_values_xp = xp.asarray(pilot_values)
    P = len(pilot_indices_np)

    # Broadcast shared pilots (P,) → (C, P) for all channels
    if pilot_values_xp.ndim == 1:
        pilot_values_xp = xp.broadcast_to(pilot_values_xp[None, :], (C, P))

    # Phase at each pilot position: angle(r_pilot · conj(s_pilot))
    r_pilots = symbols[:, pilot_indices_np]  # (C, P)

    if joint_channels and C > 1:
        # Coherent complex averaging before angle() — avoids wrap-around artefacts
        # that arise from averaging phases directly (e.g. antipodal channels).
        z_joint = xp.mean(r_pilots * xp.conj(pilot_values_xp), axis=0)  # (P,)
        phi_joint_u = xp.unwrap(xp.angle(z_joint).astype(xp.float64))  # (P,)
        if cycle_slip_correction:
            phi_joint_np = to_device(phi_joint_u, "cpu")
            phi_joint_np = correct_cycle_slips(
                phi_joint_np,
                symmetry=1,
                history_length=cycle_slip_history,
                threshold=cycle_slip_threshold,
            )
            phi_joint_u = xp.asarray(phi_joint_np)
        # Broadcast to (C, P) — read-only, downstream code only reads phi_pilots_u[ch]
        phi_pilots_u = xp.broadcast_to(phi_joint_u[None, :], (C, P))
    else:
        phi_pilots = xp.angle(r_pilots * xp.conj(pilot_values_xp))  # (C, P)
        # Unwrap along the pilot axis in float64 (cp.unwrap preserves input dtype;
        # casting before avoids precision loss in the discontinuity test for float32 input)
        phi_pilots_u = xp.unwrap(phi_pilots.astype(xp.float64), axis=-1)  # (C, P)
        if cycle_slip_correction:
            phi_pilots_u_np = to_device(phi_pilots_u, "cpu")
            for ch in range(C):
                phi_pilots_u_np[ch] = correct_cycle_slips(
                    phi_pilots_u_np[ch],
                    symmetry=1,
                    history_length=cycle_slip_history,
                    threshold=cycle_slip_threshold,
                )
            phi_pilots_u = xp.asarray(phi_pilots_u_np)

    all_positions = xp.arange(N, dtype=xp.float64)

    if interpolation == "linear":
        # xp.interp handles non-uniform pilot spacing natively, is boundary-safe
        # (extrapolates with first/last pilot value), and avoids the divide-by-zero
        # guards that the searchsorted form required.  Loop over C channels because
        # xp.interp is 1D-only; overhead is negligible for typical C = 1-4.
        phi_full = xp.zeros((C, N), dtype=xp.float64)
        for ch in range(C):
            phi_full[ch] = xp.interp(all_positions, pilot_indices_xp, phi_pilots_u[ch])

    elif interpolation == "cubic":
        # CubicSpline is inherently per-channel (1D y input); loop is unavoidable.
        # Both scipy (CPU) and cupyx.scipy (GPU) share the same API.
        phi_full = xp.zeros((C, N), dtype=xp.float64)
        if xp is not np:
            from cupyx.scipy.interpolate import CubicSpline
        else:
            from scipy.interpolate import CubicSpline

        for ch in range(C):
            phi_ch = phi_pilots_u[ch]  # already float64
            cs = CubicSpline(pilot_indices_xp, phi_ch, bc_type="natural")
            # Evaluate the spline only within the pilot span; constant-hold outside.
            first_idx = int(pilot_indices_np[0])
            last_idx = int(pilot_indices_np[-1])
            phi_full[ch, first_idx : last_idx + 1] = cs(
                all_positions[first_idx : last_idx + 1]
            )
            if first_idx > 0:
                phi_full[ch, :first_idx] = phi_ch[0]
            if last_idx < N - 1:
                phi_full[ch, last_idx + 1 :] = phi_ch[-1]

    else:
        raise ValueError(
            f"Unknown interpolation method: {interpolation!r}. "
            "Choose 'linear' or 'cubic'."
        )

    phi_full_np = to_device(phi_full, "cpu")
    phi_mean_deg = float(np.mean(phi_full_np)) * 180.0 / np.pi
    phi_std_deg = float(np.std(phi_full_np)) * 180.0 / np.pi
    logger.info(
        f"CPR (pilot-aided, {interpolation}): phase mean={phi_mean_deg:.2f}°, "
        f"std={phi_std_deg:.2f}° [P={P} pilots, C={C}]"
    )

    if debug_plot:
        from . import plotting as _plotting

        phi_pilots_u_np = to_device(phi_pilots_u, "cpu")
        _plotting.pilot_phase_estimate(
            pilot_indices=pilot_indices_np,
            phi_pilots_u=phi_pilots_u_np,
            phi_full=phi_full_np,
            show=True,
            title="CPR — Pilot-Aided Phase",
        )

    if was_1d:
        return phi_full[0]
    return phi_full


def correct_carrier_phase(
    symbols: ArrayType,
    phase_vector: ArrayType,
) -> ArrayType:
    """
    Applies carrier phase correction to a symbol sequence.

    Rotates each symbol by the negative of the estimated phase to cancel
    the carrier phase offset:

    .. math::
        y[n] = s[n] \\cdot e^{-j\\,\\hat{\\phi}[n]}

    Parameters
    ----------
    symbols : array_like
        Complex symbols. Shape: (N,) or (C, N).
    phase_vector : array_like
        Per-symbol phase estimates in radians.  Shape: (N,) for SISO, or
        broadcastable to ``symbols.shape`` for MIMO.

    Returns
    -------
    array_like
        Phase-corrected symbols, same shape and dtype as ``symbols``.
    """
    symbols, xp, _ = dispatch(symbols)
    logger.debug(f"Applying carrier phase correction: shape={symbols.shape}")
    # Wrap to [-π, π] in float64 (handles unbounded phase trajectories from
    # standalone CPR), then cast to float32 for fast GPU exp.
    phase_f64 = xp.asarray(phase_vector, dtype=xp.float64)
    two_pi = 2.0 * np.pi
    phase_wrapped = (phase_f64 - xp.round(phase_f64 / two_pi) * two_pi).astype(
        xp.float32
    )
    phasor = xp.exp(-1j * phase_wrapped)
    if phasor.dtype != symbols.dtype:
        phasor = phasor.astype(symbols.dtype)
    return symbols * phasor


_NUMBA_CYCLE_SLIP: dict = {}


def _get_numba_cycle_slip():
    """JIT-compile and cache the Numba cycle-slip correction kernel.

    Returns
    -------
    callable
        Numba-compiled ``_cycle_slip_loop``.
    """
    if "cs" not in _NUMBA_CYCLE_SLIP:
        import numba  # noqa: PLC0415

        @numba.njit(cache=True, fastmath=True, nogil=True)
        def _cycle_slip_loop(phi_u, symmetry, history_length, threshold):
            """Cycle-slip detection and correction via linear extrapolation.

            Scans the block-phase trajectory ``phi_u`` sequentially.  For each
            block, linearly extrapolates from up to ``history_length`` past
            *corrected* blocks.  When the deviation exceeds ``threshold``, a
            ``π/2`` step correction is applied.

            The linear regression uses relative coordinates [0, W-1] so that
            ``Sx`` and ``Sxx`` are exact compile-time constants.  Only ``Sy``
            and ``Sxy`` are maintained as running state, updated in O(1) per
            step via a closed-form sliding-window identity.

            Parameters
            ----------
            phi_u : (B,) float64
                Block-phase trajectory after M-fold unwrap (modified in place).
            symmetry : int
                Rotational symmetry order; correction quantum = ``2π/symmetry``.
                Pass 4 for QAM (all BPS, VV, Tikhonov use 4-fold symmetry).
            history_length : int
                Maximum number of past corrected blocks used for extrapolation.
                Use ``min(b, history_length)`` at each step.
            threshold : float64
                Deviation from extrapolated value that triggers a correction
                (radians).  Default in the caller: ``π/4``.

            Returns
            -------
            (B,) float64
                Corrected block-phase trajectory (same array, modified in place).
            """
            two_pi = 2.0 * np.pi
            quantum = two_pi / float(symmetry)
            B = len(phi_u)
            W = history_length
            W_f = float(W)

            # Precompute full-window regression constants in relative coords [0, W-1].
            # With relative coords the x-values are always small integers, so Sx and
            # Sxx never grow and there is no catastrophic cancellation regardless of
            # how many total blocks have been processed.
            Sx_full = W_f * (W_f - 1.0) / 2.0
            Sxx_full = W_f * (W_f - 1.0) * (2.0 * W_f - 1.0) / 6.0
            denom_full = W_f * Sxx_full - Sx_full * Sx_full  # W²(W²-1)/12

            # Only Sy and Sxy need to be maintained as running state.
            buf_y = np.empty(W, dtype=np.float64)
            buf_head = 0  # next write slot (circular)
            n_buf = 0  # valid entries currently in buffer

            Sy = 0.0
            Sxy = 0.0

            for b in range(B):
                y_b = phi_u[b]

                if n_buf == 0:
                    # First block: trust it unconditionally (at relative position 0).
                    buf_y[0] = y_b
                    buf_head = 1
                    n_buf = 1
                    Sy = y_b
                    Sxy = 0.0  # 0 * y_b
                    continue

                if n_buf < min(10, W):
                    # Constant extrapolation during warmup to avoid cementing false slips.
                    phi_pred = buf_y[(buf_head - 1) % W]
                else:
                    # Linear extrapolation.  Prediction target is always one step past
                    # the newest buffered entry, i.e. relative coordinate = n_buf.
                    x_pred = float(n_buf)
                    n_f = float(n_buf)
                    if n_buf < W:
                        # Partial window: derive exact Sx/Sxx from closed-form sums.
                        Sx_p = n_f * (n_f - 1.0) / 2.0
                        Sxx_p = n_f * (n_f - 1.0) * (2.0 * n_f - 1.0) / 6.0
                        denom = n_f * Sxx_p - Sx_p * Sx_p
                        if abs(denom) > 1e-30:
                            slope = (n_f * Sxy - Sx_p * Sy) / denom
                            intercept = (Sy - slope * Sx_p) / n_f
                        else:
                            slope = 0.0
                            intercept = Sy / n_f
                    else:
                        # Full window: use precomputed constants (numerically exact).
                        if denom_full > 1e-30:
                            slope = (W_f * Sxy - Sx_full * Sy) / denom_full
                            intercept = (Sy - slope * Sx_full) / W_f
                        else:
                            slope = 0.0
                            intercept = Sy / W_f
                    phi_pred = slope * x_pred + intercept

                diff = y_b - phi_pred
                # Round to nearest correction quantum
                k = round(diff / quantum)
                if abs(diff) > threshold and k != 0:
                    phi_u[b] -= float(k) * quantum
                    y_b = phi_u[b]

                # Update circular buffer using relative coordinates.
                if n_buf == W:
                    # Slide window: evict oldest (relative pos 0), shift all down by 1,
                    # add y_b at relative position W-1.
                    # Sxy update uses the identity:
                    #   Sxy_new = Sxy_old − Sy_old + y_old + (W−1)·y_new
                    # (derived by relabelling positions after eviction)
                    old_idx = buf_head % W
                    y_old = buf_y[old_idx]
                    Sxy = Sxy - Sy + y_old + (W_f - 1.0) * y_b  # must precede Sy update
                    Sy = Sy - y_old + y_b
                    buf_y[old_idx] = y_b
                    buf_head += 1
                else:
                    # Append at relative position n_buf.
                    idx = buf_head % W
                    buf_y[idx] = y_b
                    Sxy += float(n_buf) * y_b
                    Sy += y_b
                    buf_head += 1
                    n_buf += 1

            return phi_u

        _NUMBA_CYCLE_SLIP["cs"] = _cycle_slip_loop

    return _NUMBA_CYCLE_SLIP["cs"]


def correct_cycle_slips(
    phi_u: np.ndarray,
    symmetry: int = 4,
    history_length: int = 1000,
    threshold: float = np.pi / 4,
) -> np.ndarray:
    """
    Detects and corrects cycle slips in a block-phase trajectory.

    After ``xp.unwrap`` resolves the M-fold ambiguity, residual cycle slips
    may remain where the unwrapper chose the wrong quadrant.  This function
    scans the trajectory sequentially: for each block it extrapolates the
    expected phase from up to ``history_length`` past corrected blocks using
    a linear fit.  When the deviation exceeds ``threshold``, the block is
    corrected by the nearest integer multiple of ``2π/symmetry``.

    Algorithm: linear extrapolation from the previous
    ``history_length`` corrected phases; correction quantum = ``π/2`` for
    4-fold QAM symmetry; threshold = ``π/4``.

    Parameters
    ----------
    phi_u : (B,) float64
        Block-phase trajectory on CPU after M-fold unwrap (e.g. output of
        ``xp.unwrap(phi_raw * M) / M``).  **Modified in place.**
    symmetry : int, default 4
        Rotational symmetry order of the constellation.  Correction quantum
        is ``2π/symmetry``.  Use 4 for all square QAM constellations and BPS
        (which always searches over ``[0, π/2)``).  For M-PSK use ``symmetry = M``.
    history_length : int, default 1000
        Number of past corrected blocks used for linear extrapolation.
        Reduce for short bursts.
    threshold : float, default π/4
        Deviation from the extrapolated phase that triggers a correction.
        ``π/4`` is the midpoint between adjacent correction quanta for 4-fold
        symmetry.

    Returns
    -------
    (B,) float64
        Corrected block-phase trajectory (same NumPy array).

    Notes
    -----
    Runs on CPU only (sequential scan; Numba-compiled).
    The caller should transfer ``phi_u`` to CPU before calling and move
    the result back to the device if needed.
    """
    phi_u = np.asarray(phi_u, dtype=np.float64)
    kernel = _get_numba_cycle_slip()
    return kernel(phi_u, int(symmetry), int(history_length), float(threshold))


def resolve_phase_ambiguity(
    symbols: ArrayType,
    ref_symbols: ArrayType,
    modulation: str,
    order: int,
    symmetry_order: Optional[int] = None,
    num_skip_symbols: int = 0,
    pmf: Optional[np.ndarray] = None,
) -> ArrayType:
    """
    Resolves rotational phase ambiguity after blind carrier phase recovery.

    Blind CPR methods (VV, BPS, Tikhonov) cannot distinguish between
    ``symmetry_order`` rotational copies of the constellation.  This function
    tests all candidate rotations, scores each by Symbol Error Rate (SER)
    against the known transmitted symbols, and returns the symbols rotated by
    the best candidate.

    For MIMO inputs each channel is resolved independently — after MIMO
    equalisation the output streams may land on different ambiguity branches.

    Parameters
    ----------
    symbols : array_like
        Received complex symbols after CPR and ``correct_carrier_phase``.
        Shape: ``(N,)`` or ``(C, N)``.
    ref_symbols : array_like
        Known transmitted symbols (unit-average-power normalised).
        Shape: ``(N,)`` or ``(C, N)``.
    modulation : str
        Modulation scheme (case-insensitive): ``'qam'``, ``'psk'``, etc.
    order : int
        Modulation order.
    symmetry_order : int, optional
        Number of rotationally equivalent constellation copies to test.
        Defaults to 4 for QAM (4-fold ``π/2`` symmetry) and ``order`` for
        PSK.  Override for non-standard constellations.
    num_skip_symbols : int, default 0
        Number of leading symbols to exclude from SER scoring.  The applied
        rotation still covers the full input — only the scoring window is
        trimmed.  Useful when the first ``num_skip_symbols`` symbols have not
        yet converged and would bias the rotation choice.  Must be strictly
        less than the total symbol count.
    pmf : np.ndarray, optional
        Symbol PMF of shape ``(order,)`` for PS-QAM.  Forwarded to
        :func:`commstools.metrics.ser` so the diagnostic SER reported in
        the log is unbiased for shaped constellations.  The phase-rotation
        choice itself uses a scale-invariant inner product and does not
        depend on ``pmf``.

    Returns
    -------
    array_like
        Phase-ambiguity-resolved symbols, same shape and dtype as ``symbols``.
    """
    from .metrics import ser as _ser

    symbols, xp, _ = dispatch(symbols)
    was_1d = symbols.ndim == 1
    if was_1d:
        symbols = symbols[None, :]
    C, N = symbols.shape

    if num_skip_symbols >= N:
        raise ValueError(
            f"num_skip_symbols={num_skip_symbols} must be less than the total "
            f"symbol count N={N}."
        )

    ref = xp.asarray(ref_symbols)
    if ref.ndim == 1:
        ref = ref[None, :]
    if ref.shape[0] == 1 and C > 1:
        ref = xp.broadcast_to(ref, (C, N))

    if symmetry_order is None:
        symmetry_order = 4 if "qam" in modulation.lower() else order

    step = 2.0 * np.pi / symmetry_order
    candidates = [
        xp.exp(1j * k * step).astype(symbols.dtype) for k in range(symmetry_order)
    ]

    out = xp.empty_like(symbols)
    for ch in range(C):
        # ML phase ambiguity estimator: the optimal rotation maximises
        # Re(e^{jkθ} · Σ y_n s_n*), which equals choosing k closest to
        # -∠(Σ y_n s_n*) / step.  Single inner product replaces symmetry_order
        # full SER passes.
        seg_y = symbols[ch, num_skip_symbols:]
        seg_r = ref[ch, num_skip_symbols:]
        correlation = xp.sum(seg_y * xp.conj(seg_r))
        theta_est = -float(xp.angle(correlation))
        best_k = int(round(theta_est / step)) % symmetry_order
        out[ch] = symbols[ch] * candidates[best_k]
        best_ser = float(
            xp.mean(
                xp.asarray(
                    _ser(
                        out[ch, num_skip_symbols:],
                        seg_r,
                        modulation,
                        order,
                        pmf=pmf,
                    )
                )
            )
        )
        logger.info(
            f"Phase ambiguity resolution: ch={ch}, best_k={best_k}, "
            f"rotation={best_k * step * 180.0 / np.pi:.1f}°, SER={best_ser:.4f}"
        )

    if was_1d:
        return out[0]
    return out
