from typing import TYPE_CHECKING, Any, Optional, Tuple

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from .core.signal import Signal

try:
    font_prop = fm.FontProperties(family="Roboto", weight="regular")
    fm.findfont(font_prop, fallback_to_default=False)
    font_name = "Roboto"
except ValueError:
    font_name = "sans"
    print("Roboto font not found, falling back to default sans-serif.")

mpl.rcParams.update(
    {
        "figure.figsize": (5, 3.5),
        "font.family": font_name,
        "font.size": 12,
        "lines.linewidth": 2,
        "axes.linewidth": 1,
        "figure.autolayout": True,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1,
        "ytick.major.width": 1,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.width": 1,
        "ytick.minor.width": 1,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.top": True,
        "ytick.right": True,
    }
)

plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = font_name
plt.rcParams["mathtext.it"] = f"{font_name}:italic"
plt.rcParams["mathtext.bf"] = f"{font_name}:bold"


def plot_psd(
    signal: "Signal", NFFT: int = 256, ax: Optional[Any] = None
) -> Tuple[Any, Any]:
    """
    Plots the Power Spectral Density (PSD) of the signal.

    Args:
        signal: The signal to plot.
        NFFT: Number of FFT points.
        ax: Optional matplotlib axis to plot on.

    Returns:
        Tuple of (figure, axis).
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.psd(signal.samples, NFFT=NFFT, Fs=signal.sampling_rate, detrend="none")
    return fig, ax


def plot_signal(signal: "Signal", ax: Optional[Any] = None) -> Tuple[Any, Any]:
    """
    Plots the time-domain representation of the signal.

    Args:
        signal: The signal to plot.
        ax: Optional matplotlib axis to plot on.

    Returns:
        Tuple of (figure, axis).
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(signal.time_axis(), signal.samples)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    return fig, ax
