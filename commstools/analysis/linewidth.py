"""Linewidth estimators: phase-increment slope, FM-noise PSD, β-separation."""

import numpy as np

from ..backend import ArrayType, dispatch, to_device
from ..spectral import welch_psd
from ._common import _BETA_SLOPE, _FWHM_FROM_AREA, _as_2d, _scalar_or_array

__all__ = ["fm_noise_psd", "linewidth_beta_separation", "linewidth_increment"]


def linewidth_increment(
    pn_phase: ArrayType,
    symbol_rate: float,
    *,
    method: str = "slope",
    lags: tuple[int, ...] = (1, 2, 3, 4, 5),
    noise_var: float | None = None,
    snr_db: float | np.ndarray | None = None,
    ref_symbols: ArrayType | None = None,
    edge_trim: int = 0,
) -> dict[str, float | np.ndarray | bool]:
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
            from ..helpers import normalize

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
    nperseg: int | None = None,
    detrend: str | bool = "constant",
    debug_plot: bool = False,
) -> tuple[ArrayType, ArrayType]:
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
        from .. import plotting as _plotting

        _plotting.plot_frequency_noise_psd(f, S_out, show=True)

    return f, S_out


def linewidth_beta_separation(
    phi: ArrayType,
    symbol_rate: float,
    *,
    nperseg: int | None = None,
    f_min: float | None = None,
    f_max: float | None = None,
    debug_plot: bool = False,
) -> dict[str, float | np.ndarray]:
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

    # Pack the three (C,) metrics into one D2H transfer instead of three.
    lw_cpu, lw_floor_cpu, area_cpu = to_device(xp.stack([lw, lw_floor, area]), "cpu")
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
        from .. import plotting as _plotting

        _plotting.plot_frequency_noise_psd(
            f_cpu,
            S_cpu,
            beta_line=beta_cpu,
            floor=result["linewidth_floor"],
            band=(fmin, fmax),
            show=True,
        )

    return result
