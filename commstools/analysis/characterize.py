"""End-to-end carrier-phase characterization orchestrator."""

import numpy as np

from ..backend import ArrayType
from .allan import allan_deviation
from .drift import frequency_drift_metrics, separate_drift_phase_noise
from .linewidth import linewidth_beta_separation, linewidth_increment
from .trajectory import carrier_phase_trajectory

__all__ = ["characterize_carrier_phase"]


def characterize_carrier_phase(
    y_eq: ArrayType,
    ref_symbols: ArrayType,
    symbol_rate: float,
    *,
    drift_cutoff: float,
    noise_var: float | None = None,
    snr_db: float | np.ndarray | None = None,
    nperseg: int | None = None,
    f_min: float | None = None,
    f_max: float | None = None,
    channel_pairing: str = "auto",
    detrend_method: str = "butterworth",
    increment_method: str = "slope",
    amp_ref: float | None = None,
    debug_plot: bool = False,
) -> dict[str, object]:
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
        from .. import plotting as _plotting

        band = None
        if f_min is not None and f_max is not None:
            band = (float(f_min), float(f_max))
        _plotting.plot_carrier_phase_characterization(
            report,
            symbol_rate=symbol_rate,
            drift_cutoff=drift_cutoff,
            band=band,
            amp_ref=amp_ref,
            show=True,
        )

    return report
