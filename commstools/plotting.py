from typing import Any, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import scipy.ndimage


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

    import numpy as np
    from .core.backend import ensure_on_backend, get_backend, to_host

    # Ensure samples are on the global backend
    samples = ensure_on_backend(samples)
    samples -= samples.mean()
    backend = get_backend()

    # If samples are complex, we need to handle that
    # Note: backend.welch handles complex inputs if supported

    # Calculate PSD using the appropriate backend
    if backend.iscomplexobj(samples):
        f, Pxx = backend.welch(
            samples,
            fs=sampling_rate,
            nperseg=nperseg,
            detrend=detrend,
            average=average,
            return_onesided=False,
        )
        # Shift zero frequency to center if complex
        f = backend.fftshift(f)
        Pxx = backend.fftshift(Pxx)
    else:
        f, Pxx = backend.welch(
            samples,
            fs=sampling_rate,
            nperseg=nperseg,
            detrend=detrend,
            average=average,
        )

    # Convert to NumPy for plotting using to_host
    f = to_host(f)
    Pxx = to_host(Pxx)

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
    from .core.backend import to_host

    samples = to_host(samples)

    if num_symbols is not None and sps is not None:
        limit = int(num_symbols * sps)
        plot_samples = samples[:limit]
    else:
        plot_samples = samples

    time_axis = np.arange(len(plot_samples)) / sampling_rate

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


def eye_diagram(
    samples: Any,
    sps: float,
    ax: Optional[Any] = None,
    num_symbols: int = 2,
    plot_type: str = "line",
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
        plot_type: Type of plot ('line' or 'hist'). 'line' plots overlapping traces, 'hist' plots a 2D histogram.
        title: Title of the plot. Defaults to "Eye Diagram". If None, no title is set.
        show: Whether to call plt.show() after plotting.

    Returns:
        Tuple of (figure, axis) if show is False, else None.
    """
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Ensure we have a numpy array for plotting
    from .core.backend import to_host

    samples = to_host(samples)

    # Use real part for plotting if complex
    if np.iscomplexobj(samples):
        samples = samples.real

    # Normalize to max amplitude 1.0
    max_val = np.max(np.abs(samples))
    if max_val > 0:
        samples = samples / max_val

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

        # Add padding to Y range to avoid cutting off edges
        y_min, y_max = y_flat.min(), y_flat.max()
        y_range = y_max - y_min
        if y_range == 0:
            y_range = 1.0
        y_pad = y_range * 0.1
        range_y = [y_min - y_pad, y_max + y_pad]

        h, xedges, yedges = np.histogram2d(
            t_flat,
            y_flat,
            bins=[bins_x, bins_y],
            range=[[t_flat.min(), t_flat.max()], range_y],
        )

        h = h.T
        h = scipy.ndimage.gaussian_filter(h, sigma=1)

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
            "cmap": "inferno",
        }
        imshow_kwargs.update(kwargs)

        ax.imshow(h, **imshow_kwargs)  # type: ignore[arg-type]

    else:
        raise ValueError(f"Unknown plot_type: {plot_type}. Supported: 'line', 'hist'")

    ax.set_xlabel("Time [Symbol Periods]")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, num_symbols)
    if title is not None:
        ax.set_title(title)

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
    from .core.backend import to_host, get_backend

    taps = to_host(taps)
    backend = get_backend()

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
    # Compute frequency response using backend (supports GPU)
    # Note: taps is already on host (numpy), so we need to transfer back to backend
    # But since freqz needs to output arrays anyway, we can work with host directly
    # Actually, let's use backend for computation if it supports it
    from .core.backend import ensure_on_backend

    taps_backend = ensure_on_backend(taps)
    w, h = backend.freqz(taps_backend, worN=2048)
    # Transfer to host for plotting
    w = to_host(w)
    h = to_host(h)

    # Normalize to Nyquist (0 to 1)
    freqs = w / (2 * np.pi)

    # Magnitude
    ax2.plot(freqs, 20 * np.log10(np.abs(h) + 1e-12), color="C2")
    ax2.set_ylabel("Magnitude [dB]")
    ax2.set_title("Frequency Response (Magnitude)")
    ax2.set_xlabel("Frequency [Cycles/Sample]")
    ax2.set_xlim(0, 0.5)

    # Phase
    angles = np.unwrap(np.angle(h))
    ax3.plot(freqs, angles, color="C3")
    ax3.set_ylabel("Phase [radians]")
    ax3.set_title("Frequency Response (Phase)")
    ax3.set_xlabel("Frequency [Cycles/Sample]")
    ax3.set_xlim(0, 0.5)

    if show:
        plt.show()
        return None
    return fig, (ax1, ax2, ax3)
