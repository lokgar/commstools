"""Laser/carrier characterization plots (drift, Allan, linewidth)."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..backend import to_device
from .sync import plot_carrier_phase_decomposition
from .theme import (
    _as_channels,
    _set_eng_formatter,
)


def plot_frequency_drift(
    df,
    *,
    symbol_rate: float,
    amp_ref: float | None = None,
    ax=None,
    show: bool = False,
    title: str = "Residual frequency drift",
) -> tuple[Any, Any] | None:
    """
    Plots the instantaneous residual frequency offset vs time.

    ``df`` is the per-symbol frequency wander from
    ``analysis.frequency_drift_metrics`` — the slope of the smoothed (drift)
    phase.  This is the spin the carrier-phase recovery must track.

    Parameters
    ----------
    df : array_like
        Residual frequency in Hz. Shape ``(M,)`` or ``(C, M)``.
    symbol_rate : float
        Symbol rate in Baud (time axis).
    amp_ref : float, optional
        If given, draws dashed ``±amp_ref`` reference lines (e.g. the
        injected wander amplitude in a simulation).
    ax : Axes, optional
    show : bool, default False
    title : str

    Returns
    -------
    (fig, ax) or None
    """
    df_c = _as_channels(df)
    C, M = df_c.shape

    if ax is None:
        fig, axi = plt.subplots(1, 1, figsize=(5, 3.5))
    else:
        axi = ax
        fig = axi.figure

    t = np.arange(M) / float(symbol_rate)
    for i in range(C):
        axi.plot(t, df_c[i], color=f"C{i}", lw=0.8, label=f"pol {i}" if C > 1 else None)

    if amp_ref is not None:
        axi.axhline(amp_ref, color="white", ls="--", lw=0.8, label="±amplitude")
        axi.axhline(-amp_ref, color="white", ls="--", lw=0.8)

    _set_eng_formatter(axi, "x", "s")
    _set_eng_formatter(axi, "y", "Hz")
    axi.set_xlabel("time")
    axi.set_ylabel("Δf")
    axi.set_title(title)
    if C > 1 or amp_ref is not None:
        axi.legend(fontsize=8, loc="best")
    axi.grid(True, alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()
        return None
    return fig, axi


def plot_frequency_noise_psd(
    f,
    S_f,
    *,
    beta_line=None,
    floor=None,
    band: tuple[float, float] | None = None,
    ax=None,
    show: bool = False,
    title: str = "Frequency-noise PSD",
) -> tuple[Any, Any] | None:
    """
    Plots the frequency-noise PSD S_f(f) on log-log axes.

    Overlays the optional Di Domenico β-separation line and the white-FM-noise
    floor, and shades the integration band.  See ``analysis.fm_noise_psd`` and
    ``analysis.linewidth_beta_separation``.

    Parameters
    ----------
    f : array_like
        One-sided frequency axis in Hz, shape ``(nfreq,)``.
    S_f : array_like
        Frequency-noise PSD in Hz²/Hz, shape ``(nfreq,)`` or ``(C, nfreq)``.
    beta_line : array_like, optional
        β-separation line ``(8 ln2/π²)·f``, shape ``(nfreq,)``.  Drawn dashed.
    floor : float or array_like, optional
        White-FM linewidth estimate(s) in Hz; a horizontal guide is drawn at the
        corresponding PSD level ``S_f = Δν/π``.
    band : (float, float), optional
        ``(f_min, f_max)`` integration band, shaded.
    ax : Axes, optional
    show : bool, default False
    title : str

    Returns
    -------
    (fig, ax) or None
    """
    f_c = np.asarray(to_device(f, "cpu"), dtype=np.float64)
    S_c = _as_channels(S_f)
    C = S_c.shape[0]
    pos = f_c > 0

    if ax is None:
        fig, axi = plt.subplots(1, 1, figsize=(5, 3.5))
    else:
        axi = ax
        fig = axi.figure

    for i in range(C):
        axi.loglog(
            f_c[pos],
            S_c[i, pos],
            color=f"C{i}",
            lw=0.9,
            label=f"$S_f$ pol {i}" if C > 1 else "$S_f(f)$",
        )

    if beta_line is not None:
        b_c = np.asarray(to_device(beta_line, "cpu"), dtype=np.float64)
        axi.loglog(
            f_c[pos],
            b_c[pos],
            color="#ff5555",
            ls="--",
            lw=1.2,
            label="β-separation line",
        )

    if floor is not None:
        floors = np.atleast_1d(np.asarray(floor, dtype=np.float64))
        floor_mean = float(np.mean(floors))
        axi.axhline(
            floor_mean / np.pi,
            color="#ffd166",
            ls=":",
            lw=1.6,
            label="white-FM floor  Δν/π",
        )

    if band is not None:
        axi.axvspan(
            band[0],
            band[1],
            color="#06d6a0",
            alpha=0.12,
            label="floor / integration band",
        )

    _set_eng_formatter(axi, "x", "Hz")
    axi.set_xlabel("frequency")
    axi.set_ylabel("$S_f$ (Hz²/Hz)")
    axi.set_title(title)
    axi.legend(fontsize=8, loc="best")
    axi.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()
        return None
    return fig, axi


def plot_allan_deviation(
    tau_s,
    adev,
    *,
    reference_slopes: bool = True,
    ax=None,
    show: bool = False,
    title: str = "Allan deviation",
) -> tuple[Any, Any] | None:
    """
    Plots the (overlapping) Allan deviation vs averaging time on log-log axes.

    The local slope classifies the dominant frequency-noise process:
    white-FM ~ tau^(-1/2), flicker-FM ~ tau^0 (flat), random-walk-FM ~ tau^(+1/2),
    linear drift ~ tau^(+1).  See ``analysis.allan_deviation``.

    Parameters
    ----------
    tau_s : array_like
        Averaging times in seconds, shape ``(n_tau,)``.
    adev : array_like
        Allan deviation in Hz, shape ``(n_tau,)`` or ``(C, n_tau)``.
    reference_slopes : bool, default True
        If True, overlays a faint ``τ^{-1/2}`` (white-FM) guide line.
    ax : Axes, optional
    show : bool, default False
    title : str

    Returns
    -------
    (fig, ax) or None
    """
    tau = np.asarray(to_device(tau_s, "cpu"), dtype=np.float64)
    adv = _as_channels(adev)
    C = adv.shape[0]

    if ax is None:
        fig, axi = plt.subplots(1, 1, figsize=(5, 3.5))
    else:
        axi = ax
        fig = axi.figure

    for i in range(C):
        axi.loglog(
            tau,
            adv[i],
            "o-",
            ms=3,
            lw=0.9,
            color=f"C{i}",
            label=f"pol {i}" if C > 1 else "σ$_y$(τ)",
        )

    if reference_slopes:
        good = np.isfinite(adv[0]) & (adv[0] > 0)
        if np.any(good):
            tau0, a0 = tau[good][0], adv[0][good][0]
            guide = a0 * np.sqrt(tau0 / tau)  # τ^{-1/2} anchored at first point
            axi.loglog(
                tau,
                guide,
                color="gray",
                ls=":",
                lw=1.0,
                label=r"$\tau^{-1/2}$ (white-FM)",
            )

    _set_eng_formatter(axi, "x", "s")
    _set_eng_formatter(axi, "y", "Hz")
    axi.set_xlabel("averaging time τ")
    axi.set_ylabel("Allan deviation σ$_y$(τ)")
    axi.set_title(title)
    axi.legend(fontsize=8, loc="best")
    axi.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    if show:
        plt.show()
        return None
    return fig, axi


def plot_carrier_phase_characterization(
    report: dict,
    *,
    symbol_rate: float,
    drift_cutoff: float | None = None,
    band: tuple[float, float] | None = None,
    floor=None,
    amp_ref: float | None = None,
    show: bool = False,
    title: str | None = None,
) -> tuple[Any, Any] | None:
    """
    Full 2x2 carrier-phase characterization dashboard.

    Combines ``carrier_phase_decomposition``, ``frequency_drift``,
    ``frequency_noise_psd``, and ``allan_deviation`` into one figure
    from the report dict returned by
    ``analysis.characterize_carrier_phase``.

    Parameters
    ----------
    report : dict
        Output of ``analysis.characterize_carrier_phase``.
    symbol_rate : float
        Symbol rate in Baud.
    drift_cutoff : float, optional
        Annotated in the phase-decomposition panel title.
    band : (float, float), optional
        ``(f_min, f_max)`` integration band, shaded on the PSD panel.
    floor : float or array_like, optional
        White-FM floor guide; defaults to the report's estimated floor.
    amp_ref : float, optional
        Injected wander amplitude reference for the drift panel.
    show : bool, default False
    title : str, optional

    Returns
    -------
    (fig, axes) or None
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    lp = f"  (LP {drift_cutoff / 1e6:.1f} MHz)" if drift_cutoff else ""
    plot_carrier_phase_decomposition(
        report["phi"],
        report.get("drift"),
        symbol_rate=symbol_rate,
        ax=axes[0, 0],
        title=f"Recovered carrier phase{lp}",
    )
    plot_frequency_drift(
        report["drift_metrics"]["df"],
        symbol_rate=symbol_rate,
        amp_ref=amp_ref,
        ax=axes[0, 1],
    )

    lw_beta = report["linewidth_beta"]
    if floor is None:
        floor = lw_beta.get("linewidth_floor")
    plot_frequency_noise_psd(
        lw_beta["f"],
        lw_beta["S_f"],
        beta_line=lw_beta.get("beta_line"),
        floor=floor,
        band=band,
        ax=axes[1, 0],
    )
    plot_allan_deviation(
        report["allan"]["tau_s"],
        report["allan"]["adev"],
        ax=axes[1, 1],
    )

    if title:
        fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    if show:
        plt.show()
        return None
    return fig, axes
