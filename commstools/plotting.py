from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from scipy.ndimage._filters import gaussian_filter

if TYPE_CHECKING:
    from .core.signal import Signal


def apply_default_theme() -> None:
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
            "axes.titleweight": "bold",
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


def psd(
    signal: "Signal",
    nperseg: int = 256,
    detrend: Optional[Union[str, bool]] = False,
    average: Optional[str] = "mean",
    ax: Optional[Any] = None,
) -> Tuple[Any, Any]:
    """
    Plots the Power Spectral Density (PSD) of the signal.

    Args:
        signal: The signal to plot.
        nperseg: Length of each segment.
        detrend: Detrend method.
        average: Averaging method.
        ax: Optional matplotlib axis to plot on.

    Returns:
        Tuple of (figure, axis).
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    import numpy as np

    f, Pxx = signal.welch_psd(nperseg=nperseg, detrend=detrend, average=average)
    ax.plot(f, 10 * np.log10(Pxx))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.set_title("Spectrum")

    return fig, ax


def signal(
    signal: "Signal", num_symbols: int = 2, ax: Optional[Any] = None
) -> Tuple[Any, Any]:
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

    ax.plot(
        signal.time_axis()[: int(num_symbols * signal.sps)],
        signal.samples[: int(num_symbols * signal.sps)],
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")

    return fig, ax


def eye_diagram(
    signal: "Signal",
    ax: Optional[Any] = None,
    sps: Optional[float] = None,
    num_symbols: int = 2,
    plot_type: str = "line",
    **kwargs,
) -> Tuple[Any, Any]:
    """
    Plots the eye diagram of the signal.

    Args:
        signal: The signal to plot.
        ax: Optional matplotlib axis to plot on.
        sps: Samples per symbol.
        num_symbols: Number of symbol periods to display in the eye diagram. Defaults to 2.
        plot_type: Type of plot ('line' or 'hist'). 'line' plots overlapping traces, 'hist' plots a 2D histogram.

    Returns:
        Tuple of (figure, axis).
    """
    import numpy as np

    if sps is None:
        raise ValueError("sps must be provided explicitly.")

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
    trace_len = int(num_symbols * sps) + 1
    if trace_len > len(samples):
        raise ValueError("Signal is shorter than the required trace length.")

    # Calculate number of traces
    # We slide by 1 symbol period (sps)
    num_traces = (len(samples) - trace_len) // int(sps) + 1

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
            start = int(i * sps)
            traces_list.append(samples[start : start + trace_len])

        traces = np.stack(traces_list, axis=1)  # Shape: (trace_len, num_traces)

        # Time axis in symbols
        t = np.linspace(0, num_symbols, trace_len, endpoint=True)

        ax.plot(t, traces, color="C0", alpha=0.2, linewidth=1, **kwargs)

    elif plot_type == "hist":
        # For hist, we use all traces to build a good histogram
        # We can vectorize the extraction using stride_tricks or just reshaping if possible
        # But since we have overlapping windows, stride_tricks is best or a loop if memory is concern.
        # Given typical signal sizes, a loop or simple list comp is fine for extraction, then flatten.

        # However, for very large signals, we might want to limit or batch.
        # Let's use a reasonable limit for histogram calculation to avoid OOM on huge signals
        max_traces_hist = 20000
        if num_traces > max_traces_hist:
            skip = num_traces // max_traces_hist
            indices = np.arange(0, num_traces, skip)[:max_traces_hist]
        else:
            indices = np.arange(num_traces)

        traces_list = []
        for i in indices:
            start = int(i * sps)
            traces_list.append(samples[start : start + trace_len])

        traces = np.stack(traces_list, axis=0)  # Shape: (num_traces, trace_len)

        # Interpolate traces
        target_width = 500
        if trace_len < target_width:
            from scipy.interpolate import interp1d

            x_old = np.arange(trace_len)
            x_new = np.linspace(0, trace_len - 1, target_width)

            # Interpolate along the last axis (time)
            f = interp1d(x_old, traces, kind="linear", axis=1)
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
        bins_y = 500

        h, xedges, yedges = np.histogram2d(t_flat, y_flat, bins=[bins_x, bins_y])

        h = h.T
        h = gaussian_filter(h, sigma=1)

        # Plot using imshow with LogNorm for better contrast
        # We need to transpose h because imshow expects (rows, cols) -> (y, x)
        # and origin='lower'
        # Normalize histogram to [0, 1] for intuitive vmax
        if h.max() > 0:
            h = h / h.max()

        # Plot using imshow
        imshow_kwargs = {
            "origin": "lower",
            "extent": [xedges[0], xedges[-1], yedges[0], yedges[-1]],
            "aspect": "auto",
            "cmap": "copper",
        }
        imshow_kwargs.update(kwargs)

        ax.imshow(h, **imshow_kwargs)  # type: ignore[arg-type]

    else:
        raise ValueError(f"Unknown plot_type: {plot_type}. Supported: 'line', 'hist'")

    ax.set_xlabel("Time (Symbol Periods)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, num_symbols)
    ax.set_title("Eye Diagram")

    return fig, ax


def filter_response(
    taps: Any, sps: float = 1.0, ax: Optional[Any] = None
) -> Tuple[Any, Tuple[Any, Any, Any]]:
    """
    Plots the impulse and frequency response of a filter.

    Args:
        taps: Filter taps.
        sps: Samples per symbol. Used for time axis normalization.
        ax: Optional matplotlib axes. If provided, should be a list/tuple of 3 axes.

    Returns:
        Tuple of (figure, (ax_impulse, ax_mag, ax_phase)).
    """

    import numpy as np
    from scipy import signal
    import matplotlib.ticker as ticker

    # Ensure numpy array
    taps = np.array(taps)

    if ax is None:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 7))
        fig.subplots_adjust(hspace=0.4)
    elif isinstance(ax, (list, tuple, np.ndarray)) and len(ax) == 3:
        fig = ax[0].figure
        ax1, ax2, ax3 = ax
    else:
        print("Warning: filter_response requires 3 axes. Creating new figure.")
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))

    # 1. Impulse Response
    # Center the time axis
    num_taps = len(taps)

    t = (np.arange(num_taps) - (num_taps - 1) / 2) / sps

    if np.iscomplexobj(taps):
        ax1.plot(t, taps.real, label="Real", color="C0")
        ax1.plot(t, taps.imag, label="Imag", color="C1")
        ax1.legend()
    else:
        ax1.plot(t, taps, color="C0")

    ax1.set_title("Impulse Response")
    ax1.set_xlabel("Time (Symbol Periods)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)

    # Set ticks at integer intervals (1T, 2T, etc.)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))

    def t_formatter(x, pos):
        if np.isclose(x, 0):
            return "0"
        return f"{int(x)}T" if float(x).is_integer() else f"{x}T"

    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(t_formatter))

    # 2. Frequency Response
    w, h = signal.freqz(taps, worN=2048)

    # Normalize to Nyquist (0 to 1)
    freqs = w / (2 * np.pi)

    # Magnitude
    ax2.plot(freqs, 20 * np.log10(np.abs(h) + 1e-12), color="C2")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.tick_params(axis="y")
    ax2.grid(True)
    ax2.set_title("Frequency Response (Magnitude)")
    ax2.set_xlabel("Frequency (Cycles/Sample)")
    ax2.set_xlim(0, 0.5)

    # Phase
    angles = np.unwrap(np.angle(h))
    ax3.plot(freqs, angles, color="C3")
    ax3.set_ylabel("Phase (radians)")
    ax3.tick_params(axis="y")
    ax3.set_title("Frequency Response (Phase)")
    ax3.set_xlabel("Frequency (Cycles/Sample)")
    ax3.set_xlim(0, 0.5)
    ax3.grid(True)

    return fig, (ax1, ax2, ax3)
