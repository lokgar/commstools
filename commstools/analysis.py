"""
Signal analysis and characterization.

Post-processing and diagnostic routines that operate on recovered signals to
quantify their properties, as opposed to the DSP stages that *produce* those
signals (synchronization, equalization, recovery, ...).  Functions here are
grouped by the property they characterize; new analyses can be added as
independent groups without disturbing the others.

Currently provided
------------------
* **Carrier-phase characterization** — separating and quantifying carrier
  frequency drift, phase noise / linewidth, and the AWGN floor from a recovered
  carrier-phase trajectory.  See the section banner below.

Conventions shared by all routines: accept SISO ``(N,)`` or MIMO ``(C, N)``
(time on the last axis); perform phase arithmetic in ``float64`` (per the repo
CPR precision rule); and return scalar estimates as Python ``float`` / NumPy
arrays regardless of the input backend.
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np

from .backend import ArrayType, dispatch, to_device
from .logger import logger
from .spectral import welch_psd

__all__ = [
    "carrier_phase_trajectory",
    "separate_drift_phase_noise",
    "frequency_drift_metrics",
    "linewidth_increment",
    "fm_noise_psd",
    "linewidth_beta_separation",
    "allan_deviation",
    "characterize_carrier_phase",
]

# β-separation-line slope (Di Domenico 2010): S_f(f) = (8 ln2 / π²) · f
_BETA_SLOPE = 8.0 * np.log(2.0) / (np.pi**2)
# FWHM linewidth from the integrated FM-noise area above the β-line:
# Δν = sqrt(8 ln2 · A)
_FWHM_FROM_AREA = 8.0 * np.log(2.0)


def _as_2d(arr):
    """Promote SISO ``(N,)`` to ``(1, N)``; return ``(arr2d, was_1d)``."""
    return (arr[None, :], True) if arr.ndim == 1 else (arr, False)


def _scalar_or_array(values):
    """Collapse a length-1 per-channel result to a Python float."""
    values = np.asarray(values, dtype=np.float64)
    return float(values[0]) if values.size == 1 else values


# =============================================================================
# Carrier-phase characterization: drift, phase noise & linewidth
# -----------------------------------------------------------------------------
# Separates the three additive contributions to a recovered carrier-phase
# trajectory, which live at different timescales:
#
#     φ(t) = 2π∫Δf(t)dt        +  φ_PN(t)          +  n_φ(t)
#            └── drift ──┘         └ phase noise ┘    └ AWGN ┘
#
# The recovered baseband phase is the *beat* of the two free-running sources,
# so every linewidth here is the combined Δν_sig + Δν_LO and every drift is the
# relative wander; per-source attribution needs a changed optical front-end
# (matched pair, known reference, or self-heterodyne bench).
#
# Typical flow: carrier_phase_trajectory → separate_drift_phase_noise →
# frequency_drift_metrics + {linewidth_increment, linewidth_beta_separation} →
# allan_deviation, wrapped by characterize_carrier_phase.
# =============================================================================
def carrier_phase_trajectory(
    y_eq: ArrayType,
    ref_symbols: ArrayType,
    *,
    channel_pairing: str = "auto",
) -> ArrayType:
    r"""Data-aided unwrapped carrier phase from frozen-tap output + known symbols.

    Forms ``angle(y · conj(d))`` (which cancels the data modulation — for QAM
    ``|d|²`` is real-positive so the carrier angle is preserved) and unwraps it
    in ``float64``.  Because the symbols are *known*, the result carries only
    carrier phase + AWGN angle noise and **never cycle-slips** — unlike a
    blind/feed-forward estimate which would add its own estimator noise and
    slips that corrupt a linewidth estimate.  A constant offset or
    +/- pi/2 ambiguity is irrelevant; only the time variation is used.

    Parameters
    ----------
    y_eq : array_like
        Equalized symbols at 1 sps (e.g. ``apply_taps``
        output with the CPR **disabled** so the carrier phase is left intact).
        Shape ``(N,)`` (SISO) or ``(C, N)`` (MIMO, time on last axis).
    ref_symbols : array_like
        Known transmitted symbols, same layout as ``y_eq``.  The two are
        truncated to their common length on the last axis.
    channel_pairing : {"auto", "identity", "swap"}, default "auto"
        For dual-pol (``C == 2``) inputs the equalizer may map pol 0↔1.
        ``"auto"`` picks the pairing (identity vs swapped ``ref``) with the
        lower total phase-error variance; ``"identity"`` / ``"swap"`` force it.
        Ignored for SISO or ``C != 2``.

    Returns
    -------
    array_like
        Unwrapped carrier phase in radians (``float64``), shape matching the
        truncated input, on the same backend as ``y_eq``.
    """
    y, xp, _ = dispatch(y_eq)
    d = xp.asarray(ref_symbols)

    y2, was_1d = _as_2d(y)
    d2, _ = _as_2d(d)

    n = min(y2.shape[-1], d2.shape[-1])
    y2, d2 = y2[:, :n], d2[:, :n]
    c = y2.shape[0]

    if c == 2 and channel_pairing == "auto":
        var_id = _pairing_variance(y2, d2, xp)
        var_sw = _pairing_variance(y2, d2[::-1], xp)
        if var_sw < var_id:
            d2 = d2[::-1]
            logger.info("carrier_phase_trajectory: swapped pol pairing (lower var).")
    elif c == 2 and channel_pairing == "swap":
        d2 = d2[::-1]

    phi = xp.unwrap(xp.angle(y2 * xp.conj(d2)).astype(xp.float64), axis=-1)
    return phi[0] if was_1d else phi


def _pairing_variance(y2, d2, xp):
    """Total wrapped phase-error increment variance for a channel pairing."""
    pe = xp.angle(y2 * xp.conj(d2)).astype(xp.float64)
    return float(xp.sum(xp.var(xp.diff(pe, axis=-1), axis=-1)))


def separate_drift_phase_noise(
    phi: ArrayType,
    symbol_rate: float,
    *,
    cutoff: float,
    method: str = "butterworth",
    order: int = 4,
    debug_plot: bool = False,
) -> Tuple[ArrayType, ArrayType]:
    r"""Split a phase trajectory into slow drift and fast phase-noise residual.

    Applies a **zero-phase** low-pass (default 4th-order Butterworth in
    second-order-sections form via ``sosfiltfilt``, numerically stable at the
    very low normalized cutoffs typical here) at ``cutoff`` to obtain the
    drift; the residual
    ``pn = phi - drift`` carries the phase noise + AWGN.  Zero-phase filtering
    avoids the group-delay bias of a causal filter and the spectral leakage of
    a boxcar moving average.

    The split is a modelling choice: too low a cutoff lets fast drift leak into
    ``pn`` (inflating the linewidth); too high a cutoff absorbs genuine
    low-frequency phase noise into ``drift``.  Because the single-symbol
    increment used downstream is itself a high-pass, the *increment-variance*
    linewidth is only weakly sensitive to this cutoff — but discard the filter
    edge transients (``edge_trim`` on the metric functions) regardless.

    Parameters
    ----------
    phi : array_like
        Unwrapped carrier phase (radians), ``(N,)`` or ``(C, N)``.  Sampled at
        the symbol rate (one value per symbol).
    symbol_rate : float
        Symbol rate in Baud; the effective sampling rate of ``phi``.
    cutoff : float
        Low-pass cutoff in Hz separating drift (below) from phase noise
        (above).  Must satisfy ``0 < cutoff < symbol_rate / 2``.
    method : {"butterworth", "savgol", "boxcar"}, default "butterworth"
        Low-pass implementation.  ``"savgol"`` is a polynomial (Savitzky-Golay)
        detrend; ``"boxcar"`` is the crude moving average (provided for
        comparison only).
    order : int, default 4
        Butterworth order, Savitzky-Golay polynomial order, or — reinterpreted
        — ignored for the boxcar.
    debug_plot : bool, default False
        If True, plot the phase trajectory with the drift overlaid
        (``carrier_phase_decomposition``).

    Returns
    -------
    drift_phase : array_like
        Low-frequency drift component, same shape/backend/dtype as ``phi``.
    pn_phase : array_like
        High-frequency phase-noise + AWGN residual ``phi - drift``.
    """
    phi_arr, xp, sp = dispatch(phi)
    fs = float(symbol_rate)
    nyq = 0.5 * fs
    if not (0.0 < cutoff < nyq):
        raise ValueError(
            f"cutoff={cutoff} must lie in (0, symbol_rate/2={nyq}). "
            "phi is sampled at the symbol rate."
        )

    phi2, was_1d = _as_2d(phi_arr)
    in_dtype = phi2.dtype

    if method == "butterworth":
        # SOS form is numerically stable at the very low normalized cutoffs
        # typical here (cutoff ≪ symbol_rate ⇒ poles bunch near z=1).
        sos = sp.signal.butter(order, cutoff / nyq, btype="low", output="sos")
        if xp.__name__ == "cupy":
            sos = xp.asarray(sos)
        drift = sp.signal.sosfiltfilt(sos, phi2.astype(xp.float64), axis=-1)
    elif method == "savgol":
        # Window ≈ one cutoff period (odd, > polyorder).
        win = int(round(fs / cutoff)) | 1
        win = max(win, order + 2 + (order % 2 == 0))
        win = min(win, phi2.shape[-1] - (1 - phi2.shape[-1] % 2))
        drift = sp.signal.savgol_filter(phi2.astype(xp.float64), win, order, axis=-1)
    elif method == "boxcar":
        w = max(1, int(round(fs / cutoff)))
        kern = xp.ones(w, dtype=xp.float64) / w
        drift = xp.stack(
            [xp.convolve(row, kern, mode="same") for row in phi2.astype(xp.float64)],
            axis=0,
        )
    else:
        raise ValueError(f"Unknown method {method!r}.")

    pn = phi2 - drift
    if was_1d:
        drift, pn = drift[0], pn[0]

    drift = drift.astype(in_dtype, copy=False)
    pn = pn.astype(in_dtype, copy=False)

    if debug_plot:
        from . import plotting as _plotting

        _plotting.carrier_phase_decomposition(
            phi, drift, symbol_rate=symbol_rate, show=True
        )

    return drift, pn


def frequency_drift_metrics(
    drift_phase: ArrayType,
    symbol_rate: float,
    *,
    edge_trim: int = 0,
    amp_ref: Optional[float] = None,
    debug_plot: bool = False,
) -> Dict[str, Union[float, np.ndarray]]:
    r"""Residual frequency-wander statistics from a smoothed phase ramp.

    The instantaneous residual frequency offset is the phase slope
    ``df = diff(drift) / (2π T_sym)`` in Hz.  Report the std (typical wander)
    and the peak-to-peak (worst-case spin the CPR must follow).

    Relate to the BPS tracking limit: a residual ``δf`` rotates the phase by
    ``2π·δf·T_sym`` per symbol, so over a window of ``K`` symbols the
    intra-window rotation must stay below the QAM quarter-symmetry ``π/4`` →
    ``δf_max ≈ 1/(8·K·T_sym)``.  A larger BPS window tracks *less* drift.

    Parameters
    ----------
    drift_phase : array_like
        Drift phase component (radians), ``(N,)`` or ``(C, N)``.
    symbol_rate : float
        Symbol rate in Baud.
    edge_trim : int, default 0
        Number of samples to discard from each end before differencing
        (removes low-pass filter transients).
    amp_ref : float, optional
        Reference wander amplitude (Hz) drawn as ``±amp_ref`` guides when
        ``debug_plot=True`` (e.g. an injected amplitude in a simulation).
    debug_plot : bool, default False
        If True, plot the residual frequency vs time
        (``frequency_drift``).

    Returns
    -------
    dict
        ``{'df', 'std', 'pp', 'max_abs'}``.  ``df`` is the
        per-symbol residual frequency array; the rest are floats (SISO) or
        per-channel arrays (MIMO).
    """
    d, xp, _ = dispatch(drift_phase)
    d2, was_1d = _as_2d(d)
    if edge_trim > 0:
        d2 = d2[:, edge_trim:-edge_trim]

    t_sym = 1.0 / float(symbol_rate)
    df = xp.diff(d2.astype(xp.float64), axis=-1) / (2.0 * np.pi * t_sym)

    std = to_device(xp.std(df, axis=-1), "cpu")
    pp = to_device(xp.max(df, axis=-1) - xp.min(df, axis=-1), "cpu")
    max_abs = to_device(xp.max(xp.abs(df), axis=-1), "cpu")

    df_out = df[0] if was_1d else df

    if debug_plot:
        from . import plotting as _plotting

        _plotting.frequency_drift(
            df_out, symbol_rate=symbol_rate, amp_ref=amp_ref, show=True
        )

    return {
        "df": df_out,
        "std": _scalar_or_array(std),
        "pp": _scalar_or_array(pp),
        "max_abs": _scalar_or_array(max_abs),
    }


def linewidth_increment(
    pn_phase: ArrayType,
    symbol_rate: float,
    *,
    method: str = "slope",
    lags: Tuple[int, ...] = (1, 2, 3, 4, 5),
    noise_var: Optional[float] = None,
    snr_db: Optional[Union[float, np.ndarray]] = None,
    ref_symbols: Optional[ArrayType] = None,
    edge_trim: int = 0,
) -> Dict[str, Union[float, np.ndarray, bool]]:
    r"""Wiener linewidth from the phase-increment variance.

    For a Wiener phase + AWGN angle noise, the variance of the lag-``k``
    increment ``Δφ_k = φ(n) - φ(n-k)`` is **linear in ``k``**:

    Var(delta_phi_k) = slope * k + intercept
    where slope = 2 * pi * linewidth * T_sym and intercept = 2 * noise_var_phi

    because the random-walk variance accumulates with ``k`` while the
    *uncorrelated* AWGN angle noise contributes a fixed ``2σ_φ²`` regardless of
    ``k``.  Two estimators are provided:

    * ``method="slope"`` (default, **rigorous & AWGN-free**): least-squares fit
      of ``Var(Δφ_k)`` vs ``k`` over ``lags``; ``Δν = slope/(2π·T_sym)``.  The
      additive noise (AWGN *and* any residual white error from imperfect
      equalization) cancels into the intercept, so **no noise estimate is
      needed** — the key advantage over single-lag subtraction.
    * ``method="subtract"``: single-lag (``k=1``) variance minus an explicit
      AWGN term.  With ``d`` at unit power, ``σ_n² = 1/ρ``; the flat correction
      subtracts ``σ_n²`` (exact for QPSK, *under*-corrects QAM), while passing
      ``ref_symbols`` applies the amplitude-aware ``σ_n²·E[1/|d|²]`` (rigorous
      for QAM, since inner-ring symbols carry larger angle noise).

    Note: ``method="subtract"`` needs the additive-noise variance only.
    ``metrics.snr`` reports total residual (noise + phase noise + ISI) and
    over-subtracts.  Prefer ``method="slope"``.

    Parameters
    ----------
    pn_phase : array_like
        Phase-noise residual (radians) — the detrended phase — ``(N,)`` or
        ``(C, N)``.  (Use the drift-removed ``pn`` so a residual frequency ramp
        does not add a spurious ``k²`` term to the slope fit.)
    symbol_rate : float
        Symbol rate in Baud.
    method : {"slope", "subtract"}, default "slope"
        Estimator, as above.
    lags : tuple of int, default (1, 2, 3, 4, 5)
        Increment lags ``k`` for the slope fit (``method="slope"``).
    noise_var, snr_db, ref_symbols : optional
        AWGN-correction inputs for ``method="subtract"`` (see above).
    edge_trim : int, default 0
        Samples discarded from each end before differencing.

    Returns
    -------
    dict
        ``{'linewidth', 'dphi_var', 'awgn_var', 'method'}`` — linewidth /
        variances are floats (SISO) or per-channel arrays.  ``dphi_var`` is the
        lag-1 increment variance; ``awgn_var`` is the fitted intercept
        (``slope``) or the subtracted AWGN term (``subtract``).
    """
    p, xp, _ = dispatch(pn_phase)
    p2, was_1d = _as_2d(p)
    n_full = p2.shape[-1]
    sl = slice(edge_trim, n_full - edge_trim) if edge_trim > 0 else slice(None)
    p2 = p2[:, sl].astype(xp.float64)
    c = p2.shape[0]
    t_sym = 1.0 / float(symbol_rate)

    def _var_lag(k):
        dk = p2[:, k:] - p2[:, :-k]
        return xp.var(dk, axis=-1)

    var1 = _var_lag(1)

    if method == "slope":
        ks = np.asarray(sorted(set(int(k) for k in lags if k >= 1)), dtype=np.float64)
        if ks.size < 2:
            raise ValueError("method='slope' needs at least two distinct lags ≥ 1.")
        var_k = xp.stack([_var_lag(int(k)) for k in ks], axis=0)  # (n_lag, C)
        var_k_cpu = np.asarray(to_device(var_k, "cpu"), dtype=np.float64)
        # Per-channel least-squares slope/intercept of var_k vs k.
        coeffs = np.polyfit(ks, var_k_cpu, 1)  # (2, C): [slope, intercept]
        slope, intercept = coeffs[0], coeffs[1]
        linewidth_cpu = np.maximum(slope, 0.0) / (2.0 * np.pi * t_sym)
        awgn_var_cpu = intercept
    elif method == "subtract":
        if noise_var is not None:
            sigma_n2 = xp.full(c, float(noise_var), dtype=xp.float64)
        elif snr_db is not None:
            snr_val = xp.atleast_1d(xp.asarray(snr_db, dtype=xp.float64))
            sigma_n2 = 10.0 ** (-snr_val / 10.0)
            if sigma_n2.size == 1:
                sigma_n2 = xp.full(c, float(sigma_n2[0]))
        else:
            sigma_n2 = xp.zeros(c, dtype=xp.float64)

        if ref_symbols is not None and xp.any(sigma_n2):
            from .helpers import normalize

            d2, _ = _as_2d(xp.asarray(ref_symbols))
            d2 = d2[:, :n_full][:, sl]
            d2 = normalize(d2, mode="average_power", axis=-1)
            inv = 1.0 / xp.maximum(xp.abs(d2) ** 2, 1e-12)
            pair_mean = 0.5 * (inv[:, 1:] + inv[:, :-1])
            e_inv = xp.mean(pair_mean, axis=-1)
            awgn_var = sigma_n2 * e_inv
        else:
            awgn_var = sigma_n2.copy()
        linewidth = xp.maximum(var1 - awgn_var, 0.0) / (2.0 * np.pi * t_sym)

        linewidth_cpu = to_device(linewidth, "cpu")
        awgn_var_cpu = to_device(awgn_var, "cpu")
    else:
        raise ValueError(f"Unknown method {method!r} (use 'slope' or 'subtract').")

    var1_cpu = to_device(var1, "cpu")

    return {
        "linewidth": _scalar_or_array(linewidth_cpu),
        "dphi_var": _scalar_or_array(var1_cpu),
        "awgn_var": _scalar_or_array(awgn_var_cpu),
        "method": method,
    }


def fm_noise_psd(
    phi: ArrayType,
    symbol_rate: float,
    *,
    nperseg: Optional[int] = None,
    detrend: Union[str, bool] = "constant",
    debug_plot: bool = False,
) -> Tuple[ArrayType, ArrayType]:
    r"""One-sided frequency-noise PSD S_f(f) [Hz²/Hz] from the phase.

    Differentiates the phase to the instantaneous frequency
    ``f_inst = diff(phi)/(2π·T_sym)`` (Hz) and estimates its one-sided PSD via
    Welch's method (``welch_psd``).  Distinct
    impairments occupy distinct regions of S_f(f):

    * **white-FM** (linewidth): flat plateau at ``S_f = Δν/π``,
    * **drift / flicker**: steep ``1/f`` (and steeper) rise at low ``f``,
    * **AWGN** angle noise: white phase noise → ``S_f ∝ f²`` rise at high ``f``.

    Parameters
    ----------
    phi : array_like
        Unwrapped carrier phase (radians), ``(N,)`` or ``(C, N)``.
    symbol_rate : float
        Symbol rate in Baud (sampling rate of ``phi``).
    nperseg : int, optional
        Welch segment length.  Defaults to ``min(N//8, 4096)`` (clipped ≥ 256).
    detrend : str or bool, default "constant"
        Per-segment detrend passed to Welch; ``"constant"`` removes the mean
        residual frequency offset.
    debug_plot : bool, default False
        If True, plot the PSD (``frequency_noise_psd``).

    Returns
    -------
    f : array_like
        One-sided frequency axis in Hz (length ``nperseg//2 + 1``).
    S_f : array_like
        Frequency-noise PSD in Hz²/Hz, shape ``(nfreq,)`` or ``(C, nfreq)``.
    """
    p, xp, _ = dispatch(phi)
    p2, was_1d = _as_2d(p)
    t_sym = 1.0 / float(symbol_rate)

    f_inst = xp.diff(p2.astype(xp.float64), axis=-1) / (2.0 * np.pi * t_sym)
    n = f_inst.shape[-1]
    if nperseg is None:
        nperseg = int(min(max(n // 8, 256), 4096))
    nperseg = min(nperseg, n)

    f, S_f = welch_psd(
        f_inst,
        sampling_rate=float(symbol_rate),
        nperseg=nperseg,
        detrend=detrend,
        return_onesided=True,
        axis=-1,
    )
    S_out = S_f[0] if was_1d else S_f

    if debug_plot:
        from . import plotting as _plotting

        _plotting.frequency_noise_psd(f, S_out, show=True)

    return f, S_out


def linewidth_beta_separation(
    phi: ArrayType,
    symbol_rate: float,
    *,
    nperseg: Optional[int] = None,
    f_min: Optional[float] = None,
    f_max: Optional[float] = None,
    debug_plot: bool = False,
) -> Dict[str, Union[float, np.ndarray]]:
    r"""Linewidth via the Di Domenico β-separation line (canonical method).

    Integrates the frequency-noise PSD S_f(f) only over the band where
    it lies **above** the beta-separation line S_f = (8 * ln(2) / pi^2) * f;
    the FWHM linewidth is linewidth = sqrt(8 * ln(2) * A) with A the integrated
    area (Hz²).  This excludes the low-frequency drift (below the line) and —
    with an appropriate f_max — the high-frequency AWGN f^2 tail (which
    eventually climbs back above the line).  A white-FM-floor cross-check
    linewidth = pi * median(S_f) over the integration band is also returned.

    Parameters
    ----------
    phi : array_like
        Unwrapped carrier phase (radians), ``(N,)`` or ``(C, N)``.
    symbol_rate : float
        Symbol rate in Baud.
    nperseg : int, optional
        Welch segment length (see ``fm_noise_psd``).
    f_min : float, optional
        Lower integration bound in Hz (drops the residual-FOE DC region).
        Defaults to the first non-zero Welch bin.
    f_max : float, optional
        Upper integration bound in Hz.  **Set this below the AWGN ``f²`` knee**
        to avoid biasing ``Δν`` high; defaults to the Nyquist bin.
    debug_plot : bool, default False
        If True, plot the PSD with the β-line, white-FM floor, and integration
        band (``frequency_noise_psd``).

    Returns
    -------
    dict
        ``{'linewidth', 'linewidth_floor', 'area_hz2', 'f', 'S_f',
        'beta_line'}`` — linewidths are floats (SISO) / arrays (MIMO);
        ``f``/``S_f``/``beta_line`` are NumPy arrays for plotting.
    """
    f, S_f = fm_noise_psd(phi, symbol_rate, nperseg=nperseg)
    _, xp, _ = dispatch(f)
    S2 = S_f[None, :] if S_f.ndim == 1 else S_f

    beta = _BETA_SLOPE * f
    f_gt_0 = f[f > 0]
    fmin = float(f_gt_0[0]) if f_min is None else float(f_min)
    fmax = float(f[-1]) if f_max is None else float(f_max)
    band = (f >= fmin) & (f <= fmax)

    lw = xp.zeros(S2.shape[0], dtype=xp.float64)
    lw_floor = xp.zeros_like(lw)
    area = xp.zeros_like(lw)
    for ch in range(S2.shape[0]):
        s = S2[ch]
        above = band & (s > beta)
        integrand = xp.where(above, s, 0.0)
        area[ch] = xp.trapezoid(integrand, f)
        lw[ch] = xp.sqrt(_FWHM_FROM_AREA * area[ch])
        in_band = s[band]
        lw_floor[ch] = xp.pi * xp.median(in_band) if in_band.size else 0.0

    lw_cpu = to_device(lw, "cpu")
    lw_floor_cpu = to_device(lw_floor, "cpu")
    area_cpu = to_device(area, "cpu")
    f_cpu = np.asarray(to_device(f, "cpu"), dtype=np.float64)
    S_cpu = np.asarray(to_device(S_f, "cpu"), dtype=np.float64)
    beta_cpu = np.asarray(to_device(beta, "cpu"), dtype=np.float64)

    result = {
        "linewidth": _scalar_or_array(lw_cpu),
        "linewidth_floor": _scalar_or_array(lw_floor_cpu),
        "area_hz2": _scalar_or_array(area_cpu),
        "f": f_cpu,
        "S_f": S_cpu,
        "beta_line": beta_cpu,
    }

    if debug_plot:
        from . import plotting as _plotting

        _plotting.frequency_noise_psd(
            f_cpu,
            S_cpu,
            beta_line=beta_cpu,
            floor=result["linewidth_floor"],
            band=(fmin, fmax),
            show=True,
        )

    return result


def allan_deviation(
    df: ArrayType,
    symbol_rate: float,
    *,
    taus: Optional[np.ndarray] = None,
    n_taus: int = 30,
    debug_plot: bool = False,
) -> Dict[str, np.ndarray]:
    r"""Overlapping Allan deviation of an instantaneous-frequency series.

    The log-log slope of sigma_y(tau) classifies the dominant noise
    process by averaging time: white-FM proportional to tau^(-1/2),
    flicker-FM proportional to tau^0, random-walk-FM proportional to tau^(+1/2),
    linear drift proportional to tau^(+1).

    Parameters
    ----------
    df : array_like
        Instantaneous frequency samples in Hz (e.g. ``frequency_drift_metrics``
        ``df``), ``(N,)`` or ``(C, N)``, sampled at ``symbol_rate``.
    symbol_rate : float
        Sample rate of ``df`` in Hz (``τ_0 = 1/symbol_rate``).
    taus : array_like, optional
        Explicit averaging times in seconds.  Default: ``n_taus`` values
        geometrically spaced from ``τ_0`` to ``N//4·τ_0``.
    n_taus : int, default 30
        Number of log-spaced averaging times when ``taus`` is None.
    debug_plot : bool, default False
        If True, plot the Allan deviation
        (``allan_deviation``).

    Returns
    -------
    dict
        ``{'tau_s', 'adev'}`` where ``adev`` is ``(n_tau,)`` (SISO) or
        ``(C, n_tau)`` (MIMO).  NumPy arrays.
    """
    df_arr, xp, _ = dispatch(df)
    y2, was_1d = _as_2d(df_arr)
    c, n = y2.shape
    tau0 = 1.0 / float(symbol_rate)

    # Cumulative phase (time error) x_i = Σ y · τ0.
    zeros_col = xp.zeros((c, 1), dtype=xp.float64)
    x = xp.concatenate(
        [zeros_col, xp.cumsum(y2.astype(xp.float64), axis=-1) * tau0], axis=-1
    )

    if taus is None:
        m_max = max(1, n // 4)
        ms = np.unique(np.round(np.geomspace(1, m_max, n_taus)).astype(int))
    else:
        taus_cpu = np.asarray(to_device(taus, "cpu"), dtype=np.float64)
        ms = np.unique(np.maximum(1, np.round(taus_cpu / tau0).astype(int)))
        ms = ms[ms <= max(1, (x.shape[-1] - 1) // 2)]

    tau_s = ms * tau0
    adev = xp.full((c, ms.size), xp.nan, dtype=xp.float64)
    for ch in range(c):
        xc = x[ch]
        for j, m in enumerate(ms):
            m = int(m)
            if x.shape[-1] - 2 * m < 1:
                continue
            d2 = xc[2 * m :] - 2.0 * xc[m:-m] + xc[: -2 * m]
            avar = xp.mean(d2**2) / (2.0 * (m * tau0) ** 2)
            adev[ch, j] = xp.sqrt(avar)

    tau_s_cpu = np.asarray(tau_s, dtype=np.float64)
    adev_cpu = to_device(adev, "cpu")
    adev_out = adev_cpu[0] if was_1d else adev_cpu

    if debug_plot:
        from . import plotting as _plotting

        _plotting.allan_deviation(tau_s_cpu, adev_out, show=True)

    return {"tau_s": tau_s_cpu, "adev": adev_out}


def characterize_carrier_phase(
    y_eq: ArrayType,
    ref_symbols: ArrayType,
    symbol_rate: float,
    *,
    drift_cutoff: float,
    noise_var: Optional[float] = None,
    snr_db: Optional[Union[float, np.ndarray]] = None,
    nperseg: Optional[int] = None,
    f_min: Optional[float] = None,
    f_max: Optional[float] = None,
    channel_pairing: str = "auto",
    detrend_method: str = "butterworth",
    increment_method: str = "slope",
    amp_ref: Optional[float] = None,
    debug_plot: bool = False,
) -> Dict[str, object]:
    r"""End-to-end carrier-phase characterization report.

    Runs the full chain — ``carrier_phase_trajectory`` →
    ``separate_drift_phase_noise`` → drift metrics →
    increment-variance **and** β-separation linewidth → ``allan_deviation``
    — and returns a nested report dict.

    Parameters
    ----------
    y_eq, ref_symbols : array_like
        Frozen-tap equalizer output (CPR off) and the known symbols.
    symbol_rate : float
        Symbol rate in Baud.
    drift_cutoff : float
        Drift / phase-noise separation cutoff in Hz.
    noise_var, snr_db : optional
        AWGN-correction inputs for ``linewidth_increment`` (``noise_var``
        preferred; see that function's note).
    nperseg, f_min, f_max : optional
        Passed to the FM-noise-PSD / β-separation estimator.
    channel_pairing : str, default "auto"
        Forwarded to ``carrier_phase_trajectory``.
    detrend_method : str, default "butterworth"
        Forwarded to ``separate_drift_phase_noise``.
    increment_method : {"slope", "subtract"}, default "slope"
        Forwarded to ``linewidth_increment``.
    amp_ref : float, optional
        Wander-amplitude reference for the drift panel when ``debug_plot=True``.
    debug_plot : bool, default False
        If True, render the full 2x2 dashboard
        (``carrier_phase_characterization``).

    Returns
    -------
    dict
        ``{'phi', 'drift', 'pn', 'drift_metrics', 'linewidth_increment',
        'linewidth_beta', 'allan'}``.
    """
    phi = carrier_phase_trajectory(y_eq, ref_symbols, channel_pairing=channel_pairing)
    drift, pn = separate_drift_phase_noise(
        phi, symbol_rate, cutoff=drift_cutoff, method=detrend_method
    )
    # Discard ~one cutoff period of filter transient at each end.
    edge = int(round(0.5 * symbol_rate / drift_cutoff))
    edge = min(edge, max(0, (phi.shape[-1] // 4) - 1))

    drift_metrics = frequency_drift_metrics(drift, symbol_rate, edge_trim=edge)
    lw_inc = linewidth_increment(
        pn,
        symbol_rate,
        method=increment_method,
        noise_var=noise_var,
        snr_db=snr_db,
        ref_symbols=ref_symbols,
        edge_trim=edge,
    )
    lw_beta = linewidth_beta_separation(
        phi, symbol_rate, nperseg=nperseg, f_min=f_min, f_max=f_max
    )
    allan = allan_deviation(drift_metrics["df"], symbol_rate)

    report = {
        "phi": phi,
        "drift": drift,
        "pn": pn,
        "drift_metrics": drift_metrics,
        "linewidth_increment": lw_inc,
        "linewidth_beta": lw_beta,
        "allan": allan,
    }

    if debug_plot:
        from . import plotting as _plotting

        band = None
        if f_min is not None and f_max is not None:
            band = (float(f_min), float(f_max))
        _plotting.carrier_phase_characterization(
            report,
            symbol_rate=symbol_rate,
            drift_cutoff=drift_cutoff,
            band=band,
            amp_ref=amp_ref,
            show=True,
        )

    return report
