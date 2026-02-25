"""
Signal visualization and publication-quality plotting tools.

This module provides high-level plotting functions optimized for
communication signals. It leverages Matplotlib to produce high-density,
professional diagrams with automatic SI scaling and backend-agnostic data
handling.

Functions
---------
apply_default_theme :
    Sets the library's visual style (fonts, grid, colors).
psd :
    Plots Power Spectral Density with frequency/wavelength scaling.
time_domain :
    Plots I/Q waveforms or real-valued time-series data.
eye_diagram :
    Visualizes signal quality via vectorized traces or 2D density histograms.
constellation :
    Generates high-definition constellation density diagrams for noisy signals.
ideal_constellation :
    Draws theoretical constellations with Gray-coded bit annotations.
filter_response :
    Analyzes FIR filters in both time and frequency domains.
equalizer_result :
    Convergence curve and tap weight diagnostics for adaptive equalization.
"""

from typing import Any, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

from .backend import dispatch, to_device
from .logger import logger
from . import helpers


def apply_default_theme() -> None:
    """
    Applies the library's default visual theme to Matplotlib.

    This theme configures publication-quality defaults, including:
    - Roboto or standard sans-serif typography.
    - Optimized figure dimensions and DPI.
    - Consistent grid styling and axis formatting.
    - Improved LaTeX math rendering.

    Notes
    -----
    This function modifies `matplotlib.rcParams` globally. It is recommended
    to call this at the start of a script or notebook for consistent styling.
    """
    logger.debug("Applying default plotting theme.")
    try:
        font_prop = fm.FontProperties(family="Roboto", weight="regular")
        fm.findfont(font_prop, fallback_to_default=False)
        font_name = "Roboto"
    except ValueError:
        font_name = "sans"
        logger.warning("Roboto font not found, falling back to default sans-serif.")

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


def _create_subplot_grid(num_axes: int, max_cols: int = 2) -> Tuple[int, int]:
    """
    Computes a grid layout (rows, cols) for a given number of axes.

    Limits the maximum number of columns to prevent excessively wide figures.

    Parameters
    ----------
    num_axes : int
        Total number of subplots required.
    max_cols : int, default 2
        Maximum allowed number of columns.

    Returns
    -------
    nrows : int
        Number of rows in the grid.
    ncols : int
        Number of columns in the grid.
    """
    if num_axes <= max_cols:
        return 1, num_axes
    ncols = max_cols
    nrows = (num_axes + ncols - 1) // ncols  # Ceiling division
    return nrows, ncols


def psd(
    samples: Any,
    sampling_rate: float = 1.0,
    nperseg: int = 256,
    detrend: Optional[Union[str, bool]] = False,
    average: Optional[str] = "mean",
    center_frequency: float = 0.0,
    domain: str = "RF",
    x_axis: str = "frequency",
    ax: Optional[Any] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = "Spectrum",
    show: bool = False,
    **kwargs: Any,
) -> Optional[Tuple[Any, Any]]:
    """
    Plots the Power Spectral Density (PSD) of the signal.

    Supports automatic frequency scaling (Hz, MHz, GHz, etc.) or wavelength
    conversion for optical signals. Handles multidimensional (MIMO) signals
    by generating a grid of subplots.

    Parameters
    ----------
    samples : array_like or Signal
        Input signal samples. Shape: (..., N_samples).
    sampling_rate : float, default 1.0
        Sampling rate in Hz.
    nperseg : int, default 256
        Length of each segment for Welch's method. Higher values provide
        better frequency resolution but more noise.
    detrend : str or bool, default False
        Specifies how to detrend each segment (e.g., 'constant', 'linear').
    average : str, default "mean"
        Method to use for averaging segments ('mean' or 'median').
    center_frequency : float, default 0.0
        Frequency offset to apply to the x-axis in Hz.
    domain : {"RF", "OPT"}, default "RF"
        Signal domain. If "OPT", wavelength scaling is enabled.
    x_axis : {"frequency", "wavelength"}, default "frequency"
        Units for the horizontal axis.
    ax : matplotlib.axes.Axes, optional
        Existing axis to plot on. If `None`, a new figure is created.
    xlim, ylim : tuple of float, optional
        Axis limits for the plot.
    title : str, optional
        Plot title. Defaults to "Spectrum".
    show : bool, default False
        If True, calls `plt.show()` immediately.
    **kwargs : Any
        Additional keyword arguments passed to `ax.plot`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes or ndarray
        The axis or array of axes used for the plot.
    """
    logger.debug(f"Generating PSD plot (sampling_rate={sampling_rate} Hz).")

    samples, xp, _ = dispatch(samples)

    # Handle Multichannel (e.g. Dual-Pol)
    # Convention: (Channels, Time)
    if samples.ndim > 1:
        num_channels = samples.shape[0]

        if ax is None:
            nrows, ncols = _create_subplot_grid(num_channels)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False
            )
        else:
            if not isinstance(ax, (list, tuple, np.ndarray)):
                # If single axis provided but multiple channels, warn and overlay?
                # Or better, just overlay on the same axis for PSD
                # Actually, the user asked for "side-by-side".
                # But if the user provides a single axis, we must respect it.
                # Let's overlay if single axis provided, or fail.
                logger.warning(
                    "Multiple channels detected but single axis provided. Overlaying plots."
                )
                axes = np.array([[ax] * num_channels])
                fig = ax.figure
            else:
                axes = np.atleast_2d(ax)
                fig = axes.flat[0].figure

        for i in range(num_channels):
            # Recursively call psd for each channel
            channel_samples = samples[i]

            # Determine target axis using 2D indexing
            row, col = divmod(i, axes.shape[1])
            target_ax = axes[row, col] if row < axes.shape[0] else axes.flat[-1]

            ch_title = f"{title} (Ch {i})" if title else f"Channel {i}"

            psd(
                channel_samples,
                sampling_rate=sampling_rate,
                nperseg=nperseg,
                detrend=detrend,
                average=average,
                center_frequency=center_frequency,
                domain=domain,
                x_axis=x_axis,
                ax=target_ax,
                xlim=xlim,
                ylim=ylim,
                title=ch_title,
                show=False,
                **kwargs,
            )

        if show:
            plt.show()
            return None
        return fig, axes

    # --- 1D Logic Starts Here ---

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    from . import spectral

    # Calculate PSD
    f, Pxx = spectral.welch_psd(
        samples,
        sampling_rate=sampling_rate,
        nperseg=nperseg,
        detrend=detrend,
        average=average,
    )

    # Move to cpu for plotting
    f = to_device(f, "cpu")
    Pxx = to_device(Pxx, "cpu")

    # Apply center frequency shift
    f = f + center_frequency

    xlabel = "Frequency [Hz]"
    x_values = f

    if x_axis == "wavelength":
        if domain != "OPT":
            logger.warning("Wavelength plotting is typically used for optical signals.")

        # c = 299,792,458 m/s
        c = 299792458.0
        # Avoid division by zero
        # Convert frequency to wavelength: lambda = c / f
        # Result in nanometers (1e9)
        valid_indices = f > 0
        x_values = np.zeros_like(f)
        x_values[valid_indices] = (c / f[valid_indices]) * 1e9
        x_values[~valid_indices] = np.nan  # Handle non-positive frequencies

        xlabel = "Wavelength [nm]"
    else:
        # Auto-scale frequency axis
        max_f = np.max(np.abs(f))
        if max_f >= 1e12:
            scale_factor = 1e12
            unit = "THz"
        elif max_f >= 1e9:
            scale_factor = 1e9
            unit = "GHz"
        elif max_f >= 1e6:
            scale_factor = 1e6
            unit = "MHz"
        elif max_f >= 1e3:
            scale_factor = 1e3
            unit = "kHz"
        else:
            scale_factor = 1.0
            unit = "Hz"

        x_values = f / scale_factor
        xlabel = f"Frequency [{unit}]"

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Add epsilon to avoid log(0) warnings
    ax.plot(x_values, 10 * np.log10(Pxx + 1e-20), **kwargs)
    ax.set_xlabel(xlabel)
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
    start_symbol: int = 0,
    num_symbols: Optional[int] = None,
    sps: Optional[float] = None,
    ax: Optional[Any] = None,
    title: Optional[str] = "Waveform",
    show: bool = False,
    **kwargs: Any,
) -> Optional[Tuple[Any, Any]]:
    """
    Plots the time-domain representation of the signal.

    For complex signals, both In-Phase (I) and Quadrature (Q) components
    are plotted. Handles SI scaling (s, ms, us, etc.) for the time axis.

    Parameters
    ----------
    samples : array_like or Signal
        Input signal samples. Shape: (..., N_samples).
    sampling_rate : float, default 1.0
        Sampling rate in Hz.
    start_symbol : int, default 0
        The starting symbol to plot.
    num_symbols : int, optional
        Limit plot to a specific number of symbol periods. Requires `sps`.
    sps : float, optional
        Samples per symbol (required if `num_symbols` is used).
    ax : matplotlib.axes.Axes, optional
        Existing axis to plot on.
    title : str, optional
        Plot title. Defaults to "Waveform".
    show : bool, default False
        If True, calls `plt.show()`.
    **kwargs : Any
        Additional keyword arguments passed to `ax.plot`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes or ndarray
        The axis or array of axes used for the plot.
    """
    logger.debug("Generating time-domain plot.")

    samples, xp, _ = dispatch(samples)

    # Handle Multichannel
    # Convention: (Channels, Time)
    if samples.ndim > 1:
        num_channels = samples.shape[0]

        if ax is None:
            nrows, ncols = _create_subplot_grid(num_channels)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False
            )
        else:
            if not isinstance(ax, (list, tuple, np.ndarray)):
                logger.warning(
                    "Multiple channels detected but single axis provided. Overlaying plots."
                )
                axes = np.array([[ax] * num_channels])
                fig = ax.figure
            else:
                axes = np.atleast_2d(ax)
                fig = axes.flat[0].figure

        for i in range(num_channels):
            channel_samples = samples[i]

            # Determine target axis using 2D indexing
            row, col = divmod(i, axes.shape[1])
            target_ax = axes[row, col] if row < axes.shape[0] else axes.flat[-1]

            ch_title = f"{title} (Ch {i})" if title else f"Channel {i}"

            time_domain(
                channel_samples,
                sampling_rate=sampling_rate,
                start_symbol=start_symbol,
                num_symbols=num_symbols,
                sps=sps,
                ax=target_ax,
                title=ch_title,
                show=False,
                **kwargs,
            )

        if show:
            plt.show()
            return None
        return fig, axes

    # --- 1D Logic ---

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    samples = to_device(samples, "cpu")

    start_idx = int(start_symbol * sps) if sps is not None else int(start_symbol)

    if num_symbols is not None and sps is not None:
        limit = start_idx + int(num_symbols * sps)
        if limit > len(samples):
            limit = len(samples)
            logger.warning(
                "Limit exceeds number of symbols. Plotting up to last symbol."
            )
        plot_samples = samples[start_idx:limit]
    else:
        plot_samples = samples[start_idx:]

    time_axis = np.arange(len(plot_samples)) / sampling_rate

    # Auto-scale time axis
    max_time = time_axis[-1] if len(time_axis) > 0 else 0
    if max_time < 1e-9:
        scale_factor = 1e12
        unit = "ps"
    elif max_time < 1e-6:
        scale_factor = 1e9
        unit = "ns"
    elif max_time < 1e-3:
        scale_factor = 1e6
        unit = "µs"
    elif max_time < 1:
        scale_factor = 1e3
        unit = "ms"
    else:
        scale_factor = 1.0
        unit = "s"

    time_axis = time_axis * scale_factor
    xlabel = f"Time [{unit}]"

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
    ax.set_xlabel(xlabel)
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
    **kwargs: Any,
) -> None:
    """
    Internal helper to plot eye diagram traces for a single signal component.

    Parameters
    ----------
    samples : array_like
        The component samples (e.g., Real or Imaginary part).
    sps : float
        Samples per symbol.
    num_symbols : int
        Number of symbol periods per window.
    ax : matplotlib.axes.Axes
        The axis to plot on.
    type : {"line", "hist"}
        Plotting strategy.
    title : str, optional
        Title for the subplot.
    **kwargs : Any
        Additional plotting parameters.
    """
    samples, xp, sp = dispatch(samples)

    # Normalize to max amplitude 1.0
    from .helpers import normalize

    samples = normalize(samples, mode="peak")

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
            indices = xp.arange(0, num_traces, skip)[:max_traces]
        else:
            indices = xp.arange(num_traces)

        # Vectorized trace extraction
        # indices shape: (num_traces,)
        # offsets shape: (trace_len,)
        # We want to extract samples at [indices[i] * sps + offset[j]]
        start_indices = (indices * int(sps)).astype(int)
        offsets = xp.arange(trace_len, dtype=int)

        # Matrix of indices: (num_traces, 1) + (1, trace_len) -> (num_traces, trace_len)
        gather_indices = start_indices[:, None] + offsets[None, :]

        # Gather samples
        traces = samples[gather_indices]  # (num_traces, trace_len)

        # Transpose for plotting
        traces = traces.T  # (trace_len, num_traces)

        # Move to cpu for plotting
        traces = to_device(traces, "cpu")

        # Time axis in symbols
        t = np.linspace(0, num_symbols, trace_len, endpoint=True)

        line_kwargs = {"alpha": 0.2, "linewidth": 1}
        line_kwargs.update(kwargs)

        ax.plot(t, traces, color="C0", **line_kwargs)

    elif type == "hist":
        max_traces_hist = 20000
        if num_traces > max_traces_hist:
            skip = num_traces // max_traces_hist
            indices = xp.arange(0, num_traces, skip)[:max_traces_hist]
        else:
            indices = xp.arange(num_traces)

        start_indices = (indices * int(sps)).astype(int)
        offsets = xp.arange(trace_len, dtype=int)
        gather_indices = start_indices[:, None] + offsets[None, :]
        traces = samples[gather_indices]  # (num_traces, trace_len)

        # Interpolate traces
        target_width = 500
        if trace_len < target_width:
            x_old = xp.arange(trace_len, dtype=float)
            x_new = xp.linspace(0, trace_len - 1, target_width, dtype=float)

            # Interpolate along the last axis (time)
            # interp1d expects (x, x_p, f_p, axis), result is interpolated samples
            # f_p is traces
            from .helpers import interp1d

            traces = interp1d(x_new, x_old, traces, axis=1)
            trace_len = target_width

        # Create time matrix
        # Use xp.linspace
        t = xp.linspace(0, num_symbols, trace_len, endpoint=True)
        # Use xp.tile
        t_matrix = xp.tile(t, (traces.shape[0], 1))  # shape: (num_traces, trace_len)

        # Flatten
        t_flat = t_matrix.flatten()
        y_flat = traces.flatten()

        # Compute 2D histogram
        # Bins: Time (x) and Amplitude (y)
        bins_x = trace_len
        bins_y = 500

        # Add padding to Y range
        y_min, y_max = xp.min(y_flat), xp.max(y_flat)
        y_range = y_max - y_min
        if y_range == 0:
            y_range = 1.0
        y_pad = y_range * 0.1
        range_y = [float(y_min - y_pad), float(y_max + y_pad)]

        # Min/max of t_flat
        t_min, t_max = xp.min(t_flat), xp.max(t_flat)

        h, xedges, yedges = xp.histogram2d(
            t_flat,
            y_flat,
            bins=[bins_x, bins_y],
            range=[[float(t_min), float(t_max)], range_y],
        )

        h = h.T
        h = sp.ndimage.gaussian_filter(h, sigma=1)

        # Normalize
        h_max = xp.max(h)
        if h_max > 0:
            h = h / h_max

        # Move to cpu for plotting
        h = to_device(h, "cpu")
        xedges = to_device(xedges, "cpu")
        yedges = to_device(yedges, "cpu")

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
    ax: Optional[Union[Any, Tuple[Any, Any]]] = None,
    num_symbols: int = 2,
    type: str = "hist",
    title: Optional[str] = "Eye Diagram",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show: bool = False,
    **kwargs: Any,
) -> Optional[Tuple[Any, Any]]:
    """
    Plots the eye diagram of the signal.

    Visualizes signal quality by overlapping segments of the signal
    synchronized to the symbol clock. Supports both fast trace-based
    plotting and high-definition density histograms.

    Parameters
    ----------
    samples : array_like or Signal
        Input signal samples. Usually matched-filtered.
    sps : float
        Samples per symbol (must be an integer for windowing).
    ax : matplotlib.axes.Axes or array_like, optional
        Target axis or list of axes. For complex signals, two axes are
        required per channel (I and Q).
    num_symbols : int, default 2
        Number of symbol periods TO display in each eye window.
    type : {"hist", "line"}, default "hist"
        Visualization mode:
        - "hist": 2D density histogram (recommended for noisy signals).
        - "line": Vectorized overlapping traces (classic look).
    title : str, optional
        Base title for the plot.
    vmin, vmax : float, optional
         Color scaling limits for "hist" mode.
    show : bool, default False
        If True, calls `plt.show()`.
    **kwargs : Any
        Additional keyword arguments passed to the plotting backend.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes or ndarray
        The axis or array of axes used.

    Notes
    -----
    The signal should typically be synchronized (no CFO or timing offset)
    and matched-filtered before plotting to produce a clear "eye".
    """
    logger.debug(f"Generating eye diagram ({type} mode).")

    if sps % 1 != 0:
        raise ValueError("sps must be an integer")

    # Dispatch to check backend
    samples, xp, _ = dispatch(samples)

    # Convention: (Channels, Time)
    if samples.ndim > 1:
        num_channels = samples.shape[0]

        # Complex eye uses 2 axes (I/Q). So for N channels we need 2*N axes.
        is_complex = xp.iscomplexobj(samples)
        axes_per_channel = 2 if is_complex else 1

        if ax is None:
            # Grid: Rows = Channels, Cols = Components
            fig, axes = plt.subplots(
                num_channels,
                axes_per_channel,
                figsize=(5 * axes_per_channel, 3.5 * num_channels),
                squeeze=False,  # Ensure 2D array
            )
        else:
            # User provided axes. Must be flat list or correct shape
            # We assume user knows what they are doing or we do best effort
            if isinstance(ax, (np.ndarray, list, tuple)):
                # Flatten
                axes_flat = np.array(ax).flatten()
                if len(axes_flat) < num_channels * axes_per_channel:
                    raise ValueError(
                        f"Not enough axes provided. Need {num_channels * axes_per_channel}."
                    )
                # Reshape to (Channels, Components)
                axes = axes_flat[: num_channels * axes_per_channel].reshape(
                    num_channels, axes_per_channel
                )
                fig = axes[0, 0].figure
            else:
                raise ValueError(
                    "For multichannel eye diagram, you must provide a list of axes."
                )

        for i in range(num_channels):
            channel_samples = samples[i]
            ch_axes = axes[i]

            ch_title = f"{title} (Ch {i})" if title else f"Channel {i}"

            # Recursive call with 1D sample
            eye_diagram(
                channel_samples,
                sps=sps,
                ax=ch_axes,
                num_symbols=num_symbols,
                type=type,
                title=ch_title,
                vmin=vmin,
                vmax=vmax,
                show=False,
                **kwargs,
            )

        if show:
            plt.show()
            return None
        return fig, axes

    # --- 1D Logic ---

    is_complex = xp.iscomplexobj(samples)

    if ax is None:
        if is_complex:
            fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
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

    Provides a 3-panel analysis showing the filter taps in the time domain,
    the magnitude response in dB, and the unwrapped phase response.

    Parameters
    ----------
    taps : array_like
        Filter taps (impulse response).
    sps : float, default 1.0
        Samples per symbol for time-axis normalization.
    ax : array_like, optional
        A list or tuple of 3 axes to plot on.
    show : bool, default False
        If True, calls `plt.show()`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    axes : tuple of matplotlib.axes.Axes
        The (impulse, magnitude, phase) axes.
    """

    import matplotlib.ticker as ticker

    # Dispatch
    taps, xp, sp = dispatch(taps)

    if ax is None:
        logger.debug("Generating filter response plot.")
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 7))
        fig.subplots_adjust(hspace=0.4)
    elif isinstance(ax, (list, tuple, np.ndarray)) and len(ax) == 3:
        fig = ax[0].figure
        ax1, ax2, ax3 = ax
    else:
        logger.warning("filter_response requires 3 axes. Creating new figure.")
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))

    # 1. Impulse Response
    num_taps = len(taps)

    t = (xp.arange(num_taps) - (num_taps - 1) / 2) / sps

    # Move to cpu for plotting
    t_cpu = to_device(t, "cpu")
    taps_cpu = to_device(taps, "cpu")

    if xp.iscomplexobj(taps):
        ax1.plot(t_cpu, taps_cpu.real, label="Real", color="C0")
        ax1.plot(t_cpu, taps_cpu.imag, label="Imag", color="C1")
        ax1.legend()
    else:
        ax1.plot(t_cpu, taps_cpu, color="C0")

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
    # Compute frequency response using backend sp
    w, h = sp.signal.freqz(taps, worN=2048)

    # Normalize to Nyquist (0 to 1)
    # w is in radians/sample (0 to pi)
    freqs = w / (2 * xp.pi)

    # Avoid log(0)
    mag = 20 * xp.log10(xp.abs(h) + 1e-12)
    angles = xp.unwrap(xp.angle(h))

    # Move to cpu
    freqs = to_device(freqs, "cpu")
    mag = to_device(mag, "cpu")
    angles = to_device(angles, "cpu")

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


def ideal_constellation(
    modulation: str,
    order: int,
    ax: Optional[Any] = None,
    title: Optional[str] = None,
    size: float = 5,
    show: bool = False,
    unipolar: Optional[bool] = None,
) -> Optional[Tuple[Any, Any]]:
    """
    Plots the ideal constellation diagram for a modulation format.

    Draws theoretical symbol points with their associated Gray-coded bit
    sequences. Includes concentric rings and center axes for reference.

    Parameters
    ----------
    modulation : {"psk", "qam", "ask", "pam"}
        Modulation scheme identifier.
    order : int
        Modulation order (e.g., 4, 16, 64).
    ax : matplotlib.axes.Axes, optional
        Target axis.
    title : str, optional
        Plot title.
    size : float, default 5
        Figure size (square).
    show : bool, default False
        If True, calls `plt.show()`.
    unipolar : bool, default False
        If True, use unipolar constellation (ASK/PAM).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The plotting axis.
    """
    logger.debug(f"Generating ideal constellation for {modulation} ({order}-level).")
    from .mapping import gray_constellation

    if ax is None:
        fig, ax = plt.subplots(figsize=(size, size))
    else:
        fig = ax.figure

    try:
        # Generate constellation on backend (returns NumPy)
        const = gray_constellation(modulation, order, unipolar=unipolar)
    except ValueError as e:
        logger.error(f"Error generating constellation: {e}")
        return None

    # Move to cpu for plotting (already NumPy but good practice)
    const = to_device(const, "cpu")

    real = const.real
    imag = const.imag

    # Plot points
    ax.scatter(real, imag, zorder=10)

    # Annotate points
    n_bits = int(np.log2(order))
    for i, point in enumerate(const):
        x, y = point.real, point.imag

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
    ax.axhline(0, color="black", zorder=0)
    ax.axvline(0, color="black", zorder=0)

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
            alpha=0.5,
            zorder=-5,
        )
        ax.add_artist(circle)

    if show:
        plt.show()
        return None
    return fig, ax


def constellation(
    samples: Any,
    bins: int = 100,
    cmap: str = "inferno",
    ax: Optional[Any] = None,
    overlay_ideal: bool = False,
    modulation: Optional[str] = None,
    order: Optional[int] = None,
    unipolar: Optional[bool] = None,
    title: Optional[str] = "Constellation",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show: bool = False,
    **kwargs: Any,
) -> Optional[Tuple[Any, Any]]:
    """
    Plots a constellation density diagram from received samples.

    Uses high-definition 2D histograms with Gaussian smoothing to
    visualize noisy or impaired signals. This is significantly more
    informative than scatter plots for large sample sets.

    Parameters
    ----------
    samples : array_like or Signal
        Received complex samples. Shape: (..., N_symbols).
    bins : int, default 100
        Density resolution (bins per axis).
    cmap : str, default "inferno"
        Colormap for the density field.
    ax : matplotlib.axes.Axes, optional
        Target axis.
    overlay_ideal : bool, default False
        If True, overlays theoretical points and scales them to signal power.
    modulation : str, optional
        Required parameter if `overlay_ideal` is enabled.
    order : int, optional
        Required parameter if `overlay_ideal` is enabled.
    unipolar : bool, optional
        Required parameter if `overlay_ideal` is enabled.
    title : str, optional
        Plot title.
    vmin, vmax : float, optional
        Color scaling limits. Defaults to auto-range [0, 1].
    show : bool, default False
        If True, calls `plt.show()`.
    **kwargs : Any
        Additional theoretical arguments passed to `ax.imshow`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes or ndarray
        The axis or array of axes used.
    """
    logger.debug("Generating constellation density plot.")

    samples, xp, sp = dispatch(samples)

    # Handle Multichannel (e.g. Dual-Pol)
    # Convention: (Channels, Time)
    if samples.ndim > 1:
        num_channels = samples.shape[0]

        if ax is None:
            nrows, ncols = _create_subplot_grid(num_channels)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False
            )
        else:
            if not isinstance(ax, (list, tuple, np.ndarray)):
                logger.warning(
                    "Multiple channels detected but single axis provided. Overlaying plots."
                )
                axes = np.array([[ax] * num_channels])
                fig = ax.figure
            else:
                axes = np.atleast_2d(ax)
                fig = axes.flat[0].figure

        for i in range(num_channels):
            channel_samples = samples[i]

            # Determine target axis using 2D indexing
            row, col = divmod(i, axes.shape[1])
            target_ax = axes[row, col] if row < axes.shape[0] else axes.flat[-1]

            ch_title = f"{title} (Ch {i})" if title else f"Channel {i}"

            constellation(
                channel_samples,
                bins=bins,
                cmap=cmap,
                ax=target_ax,
                overlay_ideal=overlay_ideal,
                modulation=modulation,
                order=order,
                title=ch_title,
                vmin=vmin,
                vmax=vmax,
                show=False,
                **kwargs,
            )

        if show:
            plt.show()
            return None
        return fig, axes

    # --- 1D Logic ---

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    # Ensure samples are complex
    if not xp.iscomplexobj(samples):
        logger.warning("Constellation plot expects complex samples. Converting.")
        samples = samples.astype(xp.complex64)

    # Extract I and Q
    i_data = samples.real.flatten()
    q_data = samples.imag.flatten()

    # Move to CPU for plotting
    i_data = to_device(i_data, "cpu")
    q_data = to_device(q_data, "cpu")

    # Compute 2D histogram
    # Determine range based on RMS (robust to noise outliers)
    # Using np.sqrt(np.mean(|I|² + |Q|²)) is equivalent to rms(complex_signal)
    signal_rms = helpers.rms(i_data + 1j * q_data)
    # Use ~3x RMS as limit (covers most constellation points + noise spread)
    limit = signal_rms * 2.0
    if limit == 0:
        limit = 1.0  # Default view range for zero signal

    h, xedges, yedges = np.histogram2d(
        i_data, q_data, bins=bins, range=[[-limit, limit], [-limit, limit]]
    )

    # Transpose for imshow (rows=y, cols=x)
    h = h.T

    # Apply Gaussian smoothing for nicer visuals
    from scipy.ndimage import gaussian_filter

    h = gaussian_filter(h, sigma=1)

    # Normalize histogram to [0, 1] for consistent colormap scaling
    h_max = np.max(h)
    if h_max > 0:
        h = h / h_max

    # Plot using imshow
    imshow_kwargs = {
        "origin": "lower",
        "extent": [-limit, limit, -limit, limit],
        "aspect": "equal",
        "cmap": cmap,
        "interpolation": "bilinear",
    }
    if vmin is not None:
        imshow_kwargs["vmin"] = vmin
    if vmax is not None:
        imshow_kwargs["vmax"] = vmax
    imshow_kwargs.update(kwargs)

    ax.imshow(h, **imshow_kwargs)

    # Overlay ideal constellation if requested
    if overlay_ideal:
        if modulation is None or order is None:
            logger.warning(
                "Modulation and order must be provided to overlay ideal constellation."
            )
        else:
            from .mapping import gray_constellation

            try:
                const = gray_constellation(modulation, order, unipolar=unipolar)
                const = to_device(const, "cpu")

                # Scale constellation to match signal amplitude
                # Use RMS-based scaling (robust to noise outliers)
                const_rms = helpers.rms(const)
                if const_rms > 0:
                    scale_factor = signal_rms / const_rms
                    const = const * scale_factor

                ax.scatter(
                    const.real,
                    const.imag,
                    c="white",
                    s=10,
                    edgecolors="black",
                    zorder=10,
                    marker="o",
                )
            except ValueError as e:
                logger.warning(f"Could not overlay ideal constellation: {e}")

    # Add center lines
    ax.axhline(0, color="white", alpha=0.5, zorder=5)
    ax.axvline(0, color="white", alpha=0.5, zorder=5)

    ax.set_xlabel("In-Phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    if title is not None:
        ax.set_title(title)

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.grid(False)

    if show:
        plt.show()
        return None
    return fig, ax


# ============================================================================
# EQUALIZER DIAGNOSTICS
# ============================================================================


def equalizer_result(
    result,
    smoothing: int = 50,
    ax=None,
    show: bool = False,
) -> Optional[Tuple[Any, Any]]:
    """
    Plots adaptive equalizer diagnostics: convergence curve and tap weights.

    Parameters
    ----------
    result : EqualizerResult
        Equalizer output containing ``error`` and ``weights`` fields.
    smoothing : int, default 50
        Moving-average window length for the MSE convergence curve.
    ax : list of 2 Axes, optional
        Pre-existing axes ``[ax_convergence, ax_taps]``. If None, a new
        figure with 2 subplots is created.
    show : bool, default False
        If True, calls ``plt.show()`` and returns None.

    Returns
    -------
    (fig, axes) or None
        Figure and axes array when ``show=False``, None otherwise.
    """
    error = np.asarray(to_device(result.error, "cpu"))
    weights = np.asarray(to_device(result.weights, "cpu"))

    is_mimo = error.ndim == 2

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    else:
        axes = np.asarray(ax).flatten()[:2]
        fig = axes[0].figure

    # --- Panel 1: MSE convergence ---
    ax_conv = axes[0]

    if is_mimo:
        num_ch = error.shape[0]
        for ch in range(num_ch):
            mse = np.abs(error[ch]) ** 2
            if smoothing > 1 and len(mse) > smoothing:
                kernel = np.ones(smoothing) / smoothing
                mse_smooth = np.convolve(mse, kernel, mode="valid")
            else:
                mse_smooth = mse
            mse_db = 10 * np.log10(mse_smooth + 1e-30)
            ax_conv.plot(mse_db, label=f"ch {ch}")
        ax_conv.legend(fontsize=8)
    else:
        mse = np.abs(error) ** 2
        if smoothing > 1 and len(mse) > smoothing:
            kernel = np.ones(smoothing) / smoothing
            mse_smooth = np.convolve(mse, kernel, mode="valid")
        else:
            mse_smooth = mse
        mse_db = 10 * np.log10(mse_smooth + 1e-30)
        ax_conv.plot(mse_db)

    ax_conv.set_xlabel("Symbol Index")
    ax_conv.set_ylabel("MSE (dB)")
    ax_conv.set_title("Convergence")

    # --- Panel 2: Final tap weights ---
    ax_taps = axes[1]

    if is_mimo:
        # Butterfly weights: (C, C, num_taps) — plot magnitude of each row
        num_ch = weights.shape[0]
        num_taps = weights.shape[2]
        tap_idx = np.arange(num_taps) - (num_taps // 2)
        for i in range(num_ch):
            for j in range(num_ch):
                label = f"w[{i},{j}]"
                ax_taps.plot(
                    tap_idx,
                    np.abs(weights[i, j]),
                    marker="o",
                    markersize=3,
                    label=label,
                )
        ax_taps.legend(fontsize=7, ncol=2)
    else:
        # SISO: (num_taps,) — stem plot
        num_taps = weights.shape[0]
        tap_idx = np.arange(num_taps) - (num_taps // 2)
        markerline, stemlines, _ = ax_taps.stem(tap_idx, np.abs(weights))
        plt.setp(stemlines)
        plt.setp(markerline, markersize=4)

    ax_taps.set_xlabel("Tap Index")
    ax_taps.set_ylabel("|w|")
    ax_taps.set_title("Tap Weights")

    if show:
        plt.show()
        return None
    return fig, axes
