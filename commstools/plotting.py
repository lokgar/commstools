from typing import Any, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.signal import freqz, welch

from .backend import to_host


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
            "axes.grid": True,
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
    samples: Any,
    sampling_rate: float = 1.0,
    nperseg: int = 128,
    detrend: Optional[Union[str, bool]] = False,
    average: Optional[str] = "mean",
    ax: Optional[Any] = None,
    title: Optional[str] = "Spectrum",
    show: bool = False,
    **kwargs: Any,
) -> Optional[Tuple[Any, Any]]:
    """
    Plots the Power Spectral Density (PSD) of the signal.

    Args:
        samples: The signal samples to plot.
        sampling_rate: Sampling rate in Hz.
        nperseg: Length of each segment.
        detrend: Detrend method.
        average: Averaging method.
        ax: Optional matplotlib axis to plot on.
        title: Title of the plot. Defaults to "Spectrum". If None, no title is set.
        show: Whether to call plt.show() after plotting.
        **kwargs: Additional arguments passed to ax.plot.

    Returns:
        Tuple of (figure, axis) if show is False, else None.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Ensure samples are on host
    samples = to_host(samples)

    # Calculate PSD
    if np.iscomplexobj(samples):
        f, Pxx = welch(
            samples,
            fs=sampling_rate,
            nperseg=nperseg,
            detrend=detrend,
            average=average,
            return_onesided=False,
        )
        # Shift zero frequency to center if complex
        f = np.fft.fftshift(f)
        Pxx = np.fft.fftshift(Pxx)
    else:
        f, Pxx = welch(
            samples,
            fs=sampling_rate,
            nperseg=nperseg,
            detrend=detrend,
            average=average,
        )

    ax.plot(f, 10 * np.log10(Pxx), **kwargs)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [dB/Hz]")
    if title is not None:
        ax.set_title(title)

    if show:
        plt.show()
        return None
    return fig, ax


def time_domain(
    samples: Any,
    sampling_rate: float = 1.0,
    num_symbols: Optional[int] = None,
    sps: Optional[float] = None,
    ax: Optional[Any] = None,
    title: Optional[str] = "Waveform",
    show: bool = False,
    **kwargs: Any,
) -> Optional[Tuple[Any, Any]]:
    """
    Plots the time-domain representation of the signal.

    Args:
        samples: The signal samples to plot.
        sampling_rate: Sampling rate in Hz.
        num_symbols: Number of symbols to plot (requires sps).
        sps: Samples per symbol (required if num_symbols is used).
        ax: Optional matplotlib axis to plot on.
        title: Title of the plot. Defaults to "Waveform". If None, no title is set.
        show: Whether to call plt.show() after plotting.
        **kwargs: Additional arguments passed to ax.plot.

    Returns:
        Tuple of (figure, axis) if show is False, else None.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    import numpy as np
    from .backend import to_host

    samples = to_host(samples)

    if num_symbols is not None and sps is not None:
        limit = int(num_symbols * sps)
        plot_samples = samples[:limit]
    else:
        plot_samples = samples

    time_axis = np.arange(len(plot_samples)) / sampling_rate

    if np.iscomplexobj(plot_samples):
        ax.plot(
            time_axis,
            plot_samples.real,
            label="I",
            **kwargs,
        )
        ax.plot(
            time_axis,
            plot_samples.imag,
            label="Q",
            **kwargs,
        )
        ax.legend()
    else:
        ax.plot(
            time_axis,
            plot_samples,
            **kwargs,
        )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    if title is not None:
        ax.set_title(title)

    if show:
        plt.show()
        return None
    return fig, ax


def _plot_eye_traces(
    samples: Any,
    sps: float,
    num_symbols: int,
    ax: Any,
    type: str,
    title: Optional[str],
    **kwargs,
) -> None:
    """Helper to plot a single eye diagram trace (real values)."""
    samples = to_host(samples)

    # Normalize to max amplitude 1.0
    max_val = np.max(np.abs(samples))
    if max_val > 0:
        samples = samples / max_val

    # We want to include the endpoint to avoid a gap at the end of the plot
    # So we need one extra sample per trace
    trace_len = int(num_symbols * sps) + 1
    if trace_len > samples.shape[0]:
        raise ValueError("Signal is shorter than the required trace length.")

    # Calculate number of traces
    # We slide by 1 symbol period (sps)
    num_traces = (samples.shape[0] - trace_len) // int(sps) + 1

    if type == "line":
        # Limit traces for performance/visuals
        max_traces = 5000
        if num_traces > max_traces:
            skip = num_traces // max_traces
            indices = np.arange(0, num_traces, skip)[:max_traces]
        else:
            indices = np.arange(num_traces)

        # Extract traces
        indices_host = indices

        traces_list = []
        for i in indices_host:
            start = int(i * sps)
            traces_list.append(samples[start : start + trace_len])

        traces = np.stack(traces_list, axis=1)  # Shape: (trace_len, num_traces)

        # Move to host for plotting
        # Move to host for plotting (already on host)

        # Time axis in symbols
        t = np.linspace(0, num_symbols, trace_len, endpoint=True)

        line_kwargs = {"alpha": 0.2, "linewidth": 1}
        line_kwargs.update(kwargs)

        ax.plot(t, traces, color="C0", **line_kwargs)

    elif type == "hist":
        # For hist, we use all traces to build a good histogram
        max_traces_hist = 20000
        if num_traces > max_traces_hist:
            skip = num_traces // max_traces_hist
            indices = np.arange(0, num_traces, skip)[:max_traces_hist]
        else:
            indices = np.arange(num_traces)

        indices_host = indices

        traces_list = []
        for i in indices_host:
            start = int(i * sps)
            traces_list.append(samples[start : start + trace_len])

        traces = np.stack(traces_list, axis=0)  # Shape: (num_traces, trace_len)

        # Interpolate traces
        target_width = 500
        if trace_len < target_width:
            # Interpolate traces using scipy
            x_old = np.arange(trace_len)
            x_new = np.linspace(0, trace_len - 1, target_width)

            # Interpolate along the last axis (time)
            f_interp = interp1d(x_old, traces, axis=1, kind="linear")
            traces = f_interp(x_new)
            trace_len = target_width

        # Create time matrix
        t = np.linspace(0, num_symbols, trace_len, endpoint=True)
        # Use np.tile
        t_matrix = np.tile(t, (traces.shape[0], 1))  # shape: (num_traces, trace_len)

        # Flatten
        t_flat = t_matrix.flatten()
        y_flat = traces.flatten()

        # Compute 2D histogram
        # Bins: Time (x) and Amplitude (y)
        bins_x = trace_len
        bins_y = 500

        # Add padding to Y range
        y_min, y_max = np.min(y_flat), np.max(y_flat)
        y_range = y_max - y_min
        if y_range == 0:
            y_range = 1.0
        y_pad = y_range * 0.1
        range_y = [float(y_min - y_pad), float(y_max + y_pad)]

        # Min/max of t_flat
        t_min, t_max = np.min(t_flat), np.max(t_flat)

        h, xedges, yedges = np.histogram2d(
            t_flat,
            y_flat,
            bins=[bins_x, bins_y],
            range=[[float(t_min), float(t_max)], range_y],
        )

        h = h.T
        h = gaussian_filter(h, sigma=1)

        # Normalize
        h_max = np.max(h)
        if h_max > 0:
            h = h / h_max

        # Plot using imshow
        imshow_kwargs = {
            "origin": "lower",
            "extent": [xedges[0], xedges[-1], yedges[0], yedges[-1]],
            "aspect": "auto",
            "cmap": "inferno",
        }
        imshow_kwargs.update(kwargs)

        ax.imshow(h, **imshow_kwargs)  # type: ignore[arg-type]

    else:
        raise ValueError(f"Unknown type: {type}. Supported: 'line', 'hist'")

    ax.set_xlabel("Time [Symbol Periods]")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, num_symbols)
    if title is not None:
        ax.set_title(title)


def eye_diagram(
    samples: Any,
    sps: float,
    ax: Optional[Any] = None,
    num_symbols: int = 2,
    type: str = "hist",
    title: Optional[str] = "Eye Diagram",
    show: bool = False,
    **kwargs,
) -> Optional[Tuple[Any, Any]]:
    """
    Plots the eye diagram of the signal.

    Args:
        samples: The signal samples to plot.
        sps: Samples per symbol.
        ax: Optional matplotlib axis to plot on.
        num_symbols: Number of symbol periods to display in the eye diagram. Defaults to 2.
        type: Type of plot ('line' or 'hist'). 'line' plots overlapping traces, 'hist' plots a 2D histogram.
        title: Title of the plot. Defaults to "Eye Diagram". If None, no title is set.
        show: Whether to call plt.show() after plotting.

    Returns:
        Tuple of (figure, axis) if show is False, else None.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Ensure backend usage
    # Ensure backend usage
    samples = to_host(samples)
    is_complex = np.iscomplexobj(samples)

    if ax is None:
        if is_complex:
            fig, ax = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
        else:
            fig, ax = plt.subplots(1, 1)
            # Handle the fact that plt.subplots(1, 1) returns a single ax, not a list
    else:
        if isinstance(ax, (list, tuple, np.ndarray)):
            fig = ax[0].figure
        else:
            fig = ax.figure

    if is_complex:
        if not isinstance(ax, (list, tuple, np.ndarray)) or len(ax) < 2:
            raise ValueError(
                "For complex signals, 'ax' must be a list/tuple of at least 2 axes."
            )

        # Plot I
        _plot_eye_traces(
            samples.real,
            sps,
            num_symbols,
            ax[0],
            type,
            title=f"{title} (I)" if title else "I-Channel",
            **kwargs,
        )

        # Plot Q
        _plot_eye_traces(
            samples.imag,
            sps,
            num_symbols,
            ax[1],
            type,
            title=f"{title} (Q)" if title else "Q-Channel",
            **kwargs,
        )
    else:
        # If user passed a list of axes for real signal, use the first one
        target_ax = ax
        if isinstance(ax, (list, tuple, np.ndarray)):
            target_ax = ax[0]

        _plot_eye_traces(
            samples,
            sps,
            num_symbols,
            target_ax,
            type,
            title=title,
            **kwargs,
        )

    if show:
        plt.show()
        return None
    return fig, ax


def filter_response(
    taps: Any, sps: float = 1.0, ax: Optional[Any] = None, show: bool = False
) -> Optional[Tuple[Any, Tuple[Any, Any, Any]]]:
    """
    Plots the impulse and frequency response of a filter.

    Args:
        taps: Filter taps.
        sps: Samples per symbol. Used for time axis normalization.
        ax: Optional matplotlib axes. If provided, should be a list/tuple of 3 axes.
        show: Whether to call plt.show() after plotting.

    Returns:
        Tuple of (figure, (ax_impulse, ax_mag, ax_phase)) if show is False, else None.
    """

    import numpy as np
    import matplotlib.ticker as ticker

    # Ensure numpy array and get backend
    # Ensure numpy array and get backend
    # Keep taps on host
    taps = to_host(taps)

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
    # Move to host for plotting
    taps_host = taps
    num_taps = len(taps_host)

    t = (np.arange(num_taps) - (num_taps - 1) / 2) / sps

    if np.iscomplexobj(taps_host):
        ax1.plot(t, taps_host.real, label="Real", color="C0")
        ax1.plot(t, taps_host.imag, label="Imag", color="C1")
        ax1.legend()
    else:
        ax1.plot(t, taps_host, color="C0")

    ax1.set_title("Impulse Response")
    ax1.set_xlabel("Time [Symbol Periods]")
    ax1.set_ylabel("Amplitude")

    # Set ticks at integer intervals (1T, 2T, etc.)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))

    def t_formatter(x, pos):
        if np.isclose(x, 0):
            return "0"
        return f"{int(x)}T" if float(x).is_integer() else f"{x}T"

    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(t_formatter))

    # 2. Frequency Response
    # Compute frequency response using scipy
    w, h = freqz(taps, worN=2048)

    # Normalize to Nyquist (0 to 1)
    freqs = w / (2 * np.pi)
    mag = 20 * np.log10(np.abs(h) + 1e-12)
    angles = np.unwrap(np.angle(h))

    # Magnitude
    ax2.plot(freqs, mag, color="C2")
    ax2.set_ylabel("Magnitude [dB]")
    ax2.set_title("Frequency Response (Magnitude)")
    ax2.set_xlabel("Frequency [Cycles/Sample]")
    ax2.set_xlim(0, 0.5)

    # Phase
    ax3.plot(freqs, angles, color="C3")
    ax3.set_ylabel("Phase [radians]")
    ax3.set_title("Frequency Response (Phase)")
    ax3.set_xlabel("Frequency [Cycles/Sample]")
    ax3.set_xlim(0, 0.5)

    if show:
        plt.show()
        return None
    return fig, (ax1, ax2, ax3)


def plot_ideal_constellation(
    modulation: str,
    order: int,
    ax: Optional[Any] = None,
    title: Optional[str] = None,
    show: bool = False,
) -> Optional[Tuple[Any, Any]]:
    """
    Plots the ideal constellation diagram for a given modulation and order.

    Args:
        modulation: Modulation type ('psk', 'qam', 'ask').
        order: Modulation order.
        ax: Optional matplotlib axis to plot on.
        title: Title of the plot.
        show: Whether to call plt.show() after plotting.

    Returns:
        Tuple of (figure, axis) if show is False, else None.
    """
    from .mapping import gray_constellation

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure

    try:
        # Generate constellation on backend
        const = gray_constellation(modulation, order)
    except ValueError as e:
        print(f"Error generating constellation: {e}")
        return None

    # Move to host for plotting
    const = to_host(const)

    # Separate real/imag
    if np.iscomplexobj(const):
        real = const.real
        imag = const.imag
    else:
        real = const
        imag = np.zeros_like(const)

    # Plot points
    ax.scatter(real, imag, zorder=10)

    # Annotate points
    n_bits = int(np.log2(order))
    for i, point in enumerate(const):
        if np.iscomplex(point):
            x, y = point.real, point.imag
        else:
            x, y = point, 0

        label = f"{i:0{n_bits}b} ({i})"
        ax.annotate(
            label,
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            # Font settings handled by rcParams
        )

    # Titles and Labels
    if title is None:
        title = f"Constellation: {modulation.upper()} {order}"
    ax.set_title(title)
    ax.set_xlabel("In-Phase (I)")
    ax.set_ylabel("Quadrature (Q)")

    # Center lines
    ax.axhline(0, color="black", linewidth=1, zorder=0)
    ax.axvline(0, color="black", linewidth=1, zorder=0)

    # Limits and Aspect
    max_range = np.max(np.abs(const))
    limit = max_range * 1.1 if max_range > 0 else 1
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")

    ax.grid(False)

    # Draw concentric circles (rings) at point magnitudes
    # Find unique radii from the constellation points
    radii = np.unique(np.round(np.abs(const), 6))

    # Filter out zero radius (origin)
    radii = radii[radii > 1e-6]

    for r in radii:
        circle = plt.Circle(
            (0, 0),
            r,
            fill=False,
            color="gray",
            linestyle="-",
            linewidth=0.5,
            alpha=0.5,
            zorder=-5,
        )
        ax.add_artist(circle)

    if show:
        plt.show()
        return None
    return fig, ax
