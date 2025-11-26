from typing import TYPE_CHECKING, Any, Optional, Tuple

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from scipy.ndimage._filters import gaussian_filter

if TYPE_CHECKING:
    from .core.signal import Signal


def apply_default_theme():
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


def psd(signal: "Signal", NFFT: int = 256, ax: Optional[Any] = None) -> Tuple[Any, Any]:
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

    ax.psd(signal.samples, NFFT=NFFT, Fs=signal.sampling_rate, detrend="mean")
    return fig, ax


def signal(signal: "Signal", ax: Optional[Any] = None) -> Tuple[Any, Any]:
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


def eye_diagram(
    signal: "Signal",
    ax: Optional[Any] = None,
    samples_per_symbol: Optional[int] = None,
    num_symbols: int = 2,
    plot_type: str = "line",
) -> Tuple[Any, Any]:
    """
    Plots the eye diagram of the signal.

    Args:
        signal: The signal to plot.
        ax: Optional matplotlib axis to plot on.
        samples_per_symbol: Number of samples per symbol. If None, attempts to fetch from global config.
        num_symbols: Number of symbol periods to display in the eye diagram. Defaults to 2.
        plot_type: Type of plot ('line' or '2d'). 'line' plots overlapping traces, '2d' plots a 2D histogram.

    Returns:
        Tuple of (figure, axis).
    """
    import numpy as np

    from .core.config import get_config

    if samples_per_symbol is None:
        config = get_config()
        if config is not None:
            samples_per_symbol = config.samples_per_symbol
        else:
            raise ValueError(
                "samples_per_symbol must be provided either explicitly or via global config."
            )

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Ensure we have a numpy array for plotting
    samples = np.array(signal.samples)

    # Use real part for plotting if complex
    if np.iscomplexobj(samples):
        samples = samples.real

    # We want to include the endpoint to avoid a gap at the end of the plot
    # So we need one extra sample per trace
    trace_len = int(num_symbols * samples_per_symbol) + 1
    if trace_len > len(samples):
        raise ValueError("Signal is shorter than the required trace length.")

    # Calculate number of traces
    # We slide by 1 symbol period (samples_per_symbol)
    n_samples = len(samples)
    num_traces = (n_samples - trace_len) // int(samples_per_symbol) + 1

    if plot_type == "line":
        # Limit traces for performance/visuals
        max_traces = 5000
        if num_traces > max_traces:
            skip = num_traces // max_traces
            indices = np.arange(0, num_traces, skip)[:max_traces]
        else:
            indices = np.arange(num_traces)

        # Extract traces
        traces_list = []
        for i in indices:
            start = int(i * samples_per_symbol)
            traces_list.append(samples[start : start + trace_len])

        traces = np.stack(traces_list, axis=1)

        # Time axis in symbols
        t = np.linspace(0, num_symbols, trace_len, endpoint=True)

        ax.plot(t, traces, color="C0", alpha=0.2, linewidth=1)

    elif plot_type == "2d":
        # For 2D, we use all traces to build a good histogram
        # We can vectorize the extraction using stride_tricks or just reshaping if possible
        # But since we have overlapping windows, stride_tricks is best or a loop if memory is concern.
        # Given typical signal sizes, a loop or simple list comp is fine for extraction, then flatten.

        # However, for very large signals, we might want to limit or batch.
        # Let's use a reasonable limit for histogram calculation to avoid OOM on huge signals
        max_traces_2d = 20000
        if num_traces > max_traces_2d:
            skip = num_traces // max_traces_2d
            indices = np.arange(0, num_traces, skip)[:max_traces_2d]
        else:
            indices = np.arange(num_traces)

        traces_list = []
        for i in indices:
            start = int(i * samples_per_symbol)
            traces_list.append(samples[start : start + trace_len])

        traces = np.stack(traces_list, axis=0)  # Shape: (num_traces, trace_len)

        # Interpolate traces
        target_width = 300
        if trace_len < target_width:
            from scipy.interpolate import interp1d

            x_old = np.arange(trace_len)
            x_new = np.linspace(0, trace_len - 1, target_width)

            # Interpolate along the last axis (time)
            f = interp1d(x_old, traces, kind="quadratic", axis=1)
            traces = f(x_new)
            trace_len = target_width

        # Create time matrix
        t = np.linspace(0, num_symbols, trace_len, endpoint=True)
        t_matrix = np.tile(t, (traces.shape[0], 1))

        # Flatten for histogram
        t_flat = t_matrix.flatten()
        y_flat = traces.flatten()

        # Compute 2D histogram
        # Bins: Time (x) and Amplitude (y)
        # Time bins: match the sample resolution roughly
        bins_x = trace_len
        bins_y = 300  # Higher resolution for amplitude

        h, xedges, yedges = np.histogram2d(t_flat, y_flat, bins=[bins_x, bins_y])

        h = h.T
        h = gaussian_filter(h, sigma=1)

        # Plot using imshow with LogNorm for better contrast
        # We need to transpose h because imshow expects (rows, cols) -> (y, x)
        # and origin='lower'
        ax.imshow(
            h,
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            aspect="auto",
            cmap="viridis",
            interpolation="bilinear",
            # norm=mpl.colors.LogNorm(vmin=1),
        )

    else:
        raise ValueError(f"Unknown plot_type: {plot_type}. Supported: 'line', '2d'")

    ax.set_xlabel("Time (Symbol Periods)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, num_symbols)

    return fig, ax
