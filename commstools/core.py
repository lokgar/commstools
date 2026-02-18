"""
Core signal processing abstractions and data containers.

This module defines the primary data structures used throughout the library.
It provides high-level abstractions for handling raw IQ samples, physical
layer metadata, and complex frame structures.

All core classes are built on Pydantic for robust validation and support
transparent backend switching between CPU (NumPy) and GPU (CuPy).

Classes
-------
Signal :
    The primary container for IQ samples and signal-centric metadata.
    Includes methods for filtering, resampling, and visualization.
Preamble :
    A structured container for frame synchronization sequences.
SingleCarrierFrame :
    A complex frame container supporting pilot patterns, guard intervals,
    and spatial multiplexing (MIMO).
SignalInfo :
    Metadata structure describing the physical bounds of a signal/frame.
"""

import types
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False


from . import helpers
from .backend import (
    ArrayType,
    from_jax,
    get_array_module,
    get_scipy_module,
    is_cupy_available,
    to_device,
    to_jax,
)
from .logger import logger


class SignalInfo(BaseModel):
    """
    Metadata describing the structural components of a `Signal`.

    This class encapsulates timing and structure parameters for various signal
    types (frames, preambles, continuous streams). It allows downstream
    processing to distinguish between segments like preambles, pilots, and payloads.

    Attributes
    ----------
    signal_type : {"single_carrier_frame", "ofdm_frame", "preamble", "generic"}
        The type of the signal structure.
    preamble_seq_len : int
        Number of symbols in the preamble/training sequence.
    preamble_type : Literal["barker", "zc"], optional
        The type of sequence (e.g., 'barker', 'zc').
    preamble_mode : Literal["same", "time_orthogonal"], optional
        The mode of preamble transmission (e.g., 'same', 'time_orthogonal').
    preamble_kwargs : dict
        Parameters used for preamble generation (e.g., 'root' for ZC).
    payload_len : int
        Number of symbols in the data payload.
    payload_mod_scheme : str, optional
        Modulation scheme used for the payload (e.g., 'QAM').
    payload_mod_order : int, optional
        Modulation order for the payload.
    payload_mod_unipolar : bool
        Whether the payload modulation is unipolar (for PAM/ASK).
    payload_mod_rz : bool
        Whether the payload modulation is RZ (for PAM/ASK).
    pilot_count : int
        Total number of pilot/reference symbols embedded in the frame.
    pilot_pattern : {"none", "block", "comb"}
        The pattern used for pilot insertion.
    pilot_period : int
        The repetition period for pilot insertion.
    pilot_block_len : int
        Length of pilot blocks if pattern is "block".
    pilot_mod_scheme : str, optional
        Modulation scheme used for pilot symbols (e.g., 'PSK').
    pilot_mod_order : int, default 0
        Modulation order for pilot symbols.
    pilot_mod_unipolar : bool
        Whether the pilot modulation is unipolar (for PAM/ASK).
    pilot_mod_rz : bool
        Whether the pilot modulation is RZ (for PAM/ASK).
    pilot_gain_db : float, default 0.0
        Gain of pilot symbols in dB.
    guard_len : int
        Length of the guard interval (e.g., cyclic prefix) in symbols.
    guard_type : {"zero", "cp"}
        Type of guard interval: "zero" for zero-padding, "cp" for cyclic prefix.
    num_streams : int, default 1
        Number of streams for MIMO.

    Notes
    -----
    The structure typically refers to: [Guard] + [Preamble] + [Body].
    """

    signal_type: Literal[
        "single_carrier_frame", "ofdm_frame", "preamble", "generic"
    ] = "generic"

    preamble_seq_len: Optional[int] = Field(default=None, ge=0)
    preamble_type: Optional[Literal["barker", "zc"]] = None
    preamble_mode: Optional[Literal["same", "time_orthogonal"]] = None
    preamble_kwargs: Optional[Dict[str, Any]] = None

    payload_len: Optional[int] = Field(default=None, ge=0)
    payload_mod_scheme: Optional[str] = None
    payload_mod_order: Optional[int] = Field(default=None, ge=2)
    payload_mod_unipolar: Optional[bool] = None

    pilot_count: Optional[int] = Field(default=None, ge=0)
    pilot_pattern: Optional[Literal["none", "block", "comb"]] = None
    pilot_period: Optional[int] = Field(default=None, ge=0)
    pilot_block_len: Optional[int] = Field(default=None, ge=0)
    pilot_mod_scheme: Optional[str] = None
    pilot_mod_order: Optional[int] = Field(default=None, ge=2)
    pilot_mod_unipolar: Optional[bool] = None
    pilot_gain_db: Optional[float] = None

    guard_len: Optional[int] = Field(default=None, ge=0)
    guard_type: Optional[Literal["zero", "cp"]] = None

    num_streams: Optional[int] = Field(default=None, ge=1)


class Signal(BaseModel):
    """
    Primary container for digital baseband or RF signals.

    The `Signal` class encapsulates complex-valued IQ samples along with the
    physical layer metadata (sampling rate, modulation, etc.) required for
    comprehensive Digital Signal Processing (DSP) pipelines. It supports
    seamless switching between CPU (NumPy) and GPU (CuPy) backends.

    Attributes
    ----------
    samples : array_like
        The complex IQ samples.
        Shape: (N_samples,) for SISO or (N_channels, N_samples) for MIMO.
        The last dimension is always assumed to be Time.
    sampling_rate : float
        Sampling frequency in Hertz (Hz). Must be > 0.
    symbol_rate : float
        Symbol frequency (Baud rate) in Hertz (Hz). Must be > 0.
    mod_scheme : str, optional
        Identifier for the modulation format (e.g., 'QPSK', '16QAM').
    mod_order : int, optional
        The size of the symbol constellation (e.g., 4, 16).
    mod_unipolar : bool, default False
        If True, uses a unipolar constellation (e.g., 0 to M-1).
    mod_rz : bool, default False
        If True, uses Return-to-Zero (RZ) signaling.
    source_bits : array_like, optional
        The original binary data that generated the signal.
    source_symbols : array_like, optional
        The mapped constellation symbols before pulse shaping.
    pulse_shape : str, optional
        Name of the pulse shaping filter (e.g., 'rrc', 'rect', 'gaussian').
    filter_span : int
        Span of the pulse-shaping filter in symbols.
    rrc_rolloff : float
        Roll-off factor for the Root-Raised Cosine (RRC) filter.
    rc_rolloff : float
        Roll-off factor for the Raised Cosine (RC) filter.
    gaussian_bt : float
        Bandwidth-Time (BT) product for Gaussian pulse shaping.
    smoothrect_bt : float
        BT product for SmoothRect shaping filters.
    spectral_domain : {"BASEBAND", "PASSBAND", "INTERMEDIATE"}
        The signal's current placement in the frequency spectrum.
    physical_domain : {"DIG", "RF", "OPT"}
        The physical transmission domain: 'DIG' (Digital), 'RF' (Radio), 'OPT' (Optical).
    center_frequency : float
        The carrier or center frequency in Hz.
    digital_frequency_offset : float
        Cumulative digital frequency shift applied to the signal in Hz.
    signal_info : SignalInfo, optional
        Metadata describing the frame structure if this signal is part of a frame.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )

    samples: Any
    sampling_rate: float = Field(..., gt=0)
    symbol_rate: float = Field(..., gt=0)

    mod_scheme: Optional[str] = None
    mod_order: Optional[int] = None
    mod_unipolar: Optional[bool] = None
    mod_rz: Optional[bool] = None

    source_bits: Optional[Any] = None
    source_symbols: Optional[Any] = None

    pulse_shape: Optional[str] = None
    filter_span: int = Field(default=10, ge=1)
    rrc_rolloff: float = Field(default=0.35, ge=0, le=1)
    rc_rolloff: float = Field(default=0.35, ge=0, le=1)
    gaussian_bt: float = Field(default=0.3, gt=0)
    smoothrect_bt: float = Field(default=1.0, gt=0)

    spectral_domain: Literal["BASEBAND", "PASSBAND", "INTERMEDIATE"] = "BASEBAND"
    physical_domain: Literal["DIG", "RF", "OPT"] = "DIG"

    center_frequency: float = Field(default=0, ge=0)
    digital_frequency_offset: float = Field(default=0)

    # Signal structure info (populated when Signal is generated from Frame/Preamble)
    signal_info: Optional[SignalInfo] = None

    # Resolved data from processing
    resolved_symbols: Optional[Any] = Field(default=None, repr=False)
    resolved_bits: Optional[Any] = Field(default=None, repr=False)

    # Private: cached equalizer result for post-hoc inspection
    _equalizer_result: Any = PrivateAttr(default=None)

    # =========================================================================
    # Validators and Post-Initialization Hooks
    # =========================================================================

    @field_validator("samples", mode="before")
    @classmethod
    def validate_samples(cls, v: Any) -> Any:
        """
        Validates and coerces the samples input into a backend-compatible array.

        This validator ensures that the input is converted to a NumPy or CuPy array
        and enforces a (Channels, Time) shape convention for multidimensional inputs.

        Parameters
        ----------
        v : array_like
            Input samples (list, tuple, NumPy array, CuPy array, or JAX array).

        Returns
        -------
        array_like
            The validated samples as a NumPy or CuPy array.

        Raises
        ------
        ValueError
            If the input cannot be converted to a supported array type or has
            unsupported dimensions (> 2).

        Notes
        -----
        The library enforces a **Time-Last** convention: (N_channels, N_samples)
        or simply (N_samples,) for 1D signals. This aligns with C-contiguous
        memory layout which is generally more performant for time-axis operations.
        """
        arr = helpers.validate_array(v, name="samples")

        # Check shape conventions
        # We enforce Time-Last convention: (Channels, Time) or (Time,) for 1D.
        # This aligns better with C-contiguous operations on the time axis (last axis)
        # which is critical for CuPy performance/stability.

        if arr.ndim > 2:
            raise ValueError(
                f"Samples array has {arr.ndim} dimensions. "
                "Only 1D (SISO) or 2D (MIMO/Dual-Pol) arrays are supported."
            )

        if arr.ndim == 2:
            # Check dimensions to guess orientation
            s0, s1 = arr.shape
            # If dim0 (rows) > dim1 (cols) and dim0 >> 10, it's likely (Time, Channels)
            # We want (Channels, Time).
            if s0 > s1 and s0 > 32:  # Heuristic: Time dim usually > 32
                logger.warning(
                    f"Samples shape is {arr.shape}. Converting to Time-Last convention (N_channels={s1}, N_samples={s0}). "
                    "Please provide input as (N_channels, N_samples) for MIMO signals."
                )
                arr = arr.T  # Transpose to (Channels, Time)

            # If shape is (2, 2), ambiguous but assumes (Channels, Time)
            # If s1 > s0, likely already correct.

        return arr

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization hook to handle metadata derivation and device placement.

        This method automatically derives `source_symbols` from `source_bits` if
        modulation parameters are present, and moves the signal samples to
        the GPU if a compatible device is available.
        """
        # Bit-first: derive symbols from bits if not provided
        if self.source_bits is not None and self.source_symbols is None:
            if self.mod_scheme and self.mod_order:
                from . import mapping

                self.source_symbols = mapping.map_bits(
                    self.source_bits,
                    self.mod_scheme,
                    self.mod_order,
                    self.mod_unipolar,
                )

        # Ensure source_symbols are normalized to unit average power for consistent metrics
        # Ensure source_symbols are normalized to unit average power for consistent metrics
        # For MIMO (multichannel), we normalize per-stream (axis=-1) to ensure each stream
        # independently adheres to E_s=1, facilitating per-stream metric calculation.
        if self.source_symbols is not None:
            self.source_symbols = helpers.normalize(
                self.source_symbols, mode="average_power", axis=-1
            )

        # Default to GPU if available and supported
        if is_cupy_available():
            self.to("gpu")

    # =========================================================================
    # Utilities
    # =========================================================================

    def print_info(self) -> None:
        """
        Prints a formatted summary of the signal's physical and digital properties.

        In Jupyter/IPython environments, this renders as an HTML table. In standard
        shells, it outputs a clean logarithmic log message.
        """
        import pandas as pd
        from IPython import get_ipython
        from IPython.display import display

        data = {
            "Property": [
                "Spectral Domain",
                "Physical Domain",
                "Modulation (Scheme/Order)",
                "Symbol Rate",
                "Bit Rate",
                "Sampling Rate",
                "Samples Per Symbol",
                "Pulse Shape",
                "Duration",
                "Center Frequency",
                "Digital Freq. Offset",
                "Backend",
                "Configuration",
                "Samples Shape",
            ],
            "Value": [
                self.spectral_domain,
                self.physical_domain,
                f"{self.mod_scheme or 'None'} / {self.mod_order or 'None'}{' (UNIPOL)' if self.mod_unipolar else ''}{' (RZ)' if self.mod_rz else ''}",
                helpers.format_si(self.symbol_rate, "Baud"),
                helpers.format_si(self.symbol_rate * np.log2(self.mod_order), "bps")
                if self.mod_order
                else "None",
                helpers.format_si(self.sampling_rate, "Hz"),
                f"{self.sps:.2f}",
                self.pulse_shape.upper() if self.pulse_shape else "None",
                helpers.format_si(self.duration, "s"),
                helpers.format_si(self.center_frequency, "Hz"),
                helpers.format_si(self.digital_frequency_offset, "Hz"),
                self.backend.upper(),
                "SISO" if self.num_streams == 1 else f"MIMO ({self.num_streams}x)",
                str(self.samples.shape),
            ],
        }
        df = pd.DataFrame(data)

        if get_ipython() is not None and "IPKernelApp" in get_ipython().config:
            display(df)
        else:
            logger.info("\n" + str(df))

    def copy(self) -> "Signal":
        """
        Creates a deep copy of the `Signal` instance.

        Returns
        -------
        Signal
            A new signal object with identical data and metadata.
        """
        return self.model_copy(deep=True)

    def to(self, device: str) -> "Signal":
        """
        Transfers signal data to the target device (CPU or GPU).

        Parameters
        ----------
        device : {"CPU", "GPU"}
            The target device. Case-insensitive.

        Returns
        -------
        Signal
            Returns self for method chaining.

        Raises
        ------
        ImportError
            If GPU is requested but CuPy is not installed/functional.
        """
        self.samples = to_device(self.samples, device)
        return self

    def export_samples_to_jax(self, device: Optional[str] = None) -> Any:
        """
        Exports the signal samples to a JAX array.

        Ensures zero-copy transfer to JAX when possible, preserving the
        device affinity of the underlying samples unless otherwise specified.

        Parameters
        ----------
        device : {"CPU", "GPU", "TPU"}, optional
            Target JAX device. If None, it targets the device matching the
            signal's current backend (CPU or GPU).

        Returns
        -------
        jax.Array
            JAX array containing signal samples.
            Shape: (N_channels, N_samples) or (N_samples,).
        """
        # If device is not explicitly requested, use the signal's backend
        target_device = device if device is not None else self.backend
        return to_jax(self.samples, device=target_device)

    def update_samples_from_jax(self, jax_array: Any) -> "Signal":
        """
        Updates signal samples from a JAX array.

        Converts the JAX array back to the signal's original backend (NumPy
        or CuPy) to maintain consistent state.

        Parameters
        ----------
        jax_array : jax.Array
            Input JAX array. Shape must match signal's expected shape.

        Returns
        -------
        Signal
            Returns self for method chaining.
        """
        original_backend = self.backend
        # Convert JAX array to backend-compatible array
        # from_jax will return NumPy (for CPU/TPU) or CuPy (for GPU)
        new_samples = from_jax(jax_array)

        # Ensure we move the data back to the original backend if it differs
        # (e.g., if signal was GPU but jax_array was on CPU/TPU)
        self.samples = to_device(new_samples, original_backend)

        return self

    def time_axis(self) -> ArrayType:
        """
        Generates the time vector associated with signal samples.

        Returns
        -------
        array_like
            Time axis in seconds, starting at 0.
            Shape: (N_samples,).
        """
        n_samples = self.samples.shape[-1]
        return self.xp.arange(0, n_samples) / self.sampling_rate

    def welch_psd(
        self,
        nperseg: int = 256,
        detrend: Optional[Union[str, bool]] = False,
        average: Optional[str] = "mean",
    ) -> Tuple[ArrayType, ArrayType]:
        """
        Compute the Power Spectral Density (PSD) using Welch's method.

        Parameters
        ----------
        nperseg : int, default 256
            Length of each segment for FFT.
        detrend : str or bool, default False
            Specifies how to detrend each segment (e.g., 'constant', 'linear').
        average : {"mean", "median"}, default "mean"
            Method to use for averaging overlapping segments.

        Returns
        -------
        f : array_like
            Array of sample frequencies. Shape: (N_freqs,).
        Pxx : array_like
            Power spectral density of the signal.
            Shape: (N_channels, N_freqs) or (N_freqs,).
        """
        from . import spectral

        return spectral.welch_psd(
            self.samples,
            sampling_rate=self.sampling_rate,
            nperseg=nperseg,
            detrend=detrend,
            average=average,
            axis=-1,  # Explicitly specify time axis
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def xp(self) -> types.ModuleType:
        """
        Access the active array backend (NumPy or CuPy).

        This property allows for backend-agnostic code by returning the
        appropriate module based on where the samples currently reside.

        Returns
        -------
        module
            `numpy` if data is on CPU, `cupy` if on GPU.
        """
        return get_array_module(self.samples)

    @property
    def sp(self) -> types.ModuleType:
        """
        Access the signal processing module (`scipy` or `cupyx.scipy`).

        Returns
        -------
        module
            Appropriate signal processing library for the current backend.
        """
        return get_scipy_module(self.xp)

    @property
    def backend(self) -> str:
        """
        Returns the current computational backend name.

        Returns
        -------
        {"CPU", "GPU"}
            A string indicating the device location of samples.
        """
        return "GPU" if self.xp == cp else "CPU"

    @property
    def num_streams(self) -> int:
        """
        Returns the number of spatial or polarization streams.

        Returns
        -------
        int
            1 for SISO signals, N for MIMO/Dual-Pol signals.
        """
        if self.samples.ndim == 1:
            return 1
        return self.samples.shape[0]

    @property
    def duration(self) -> float:
        """
        Returns the total duration of the signal.

        Returns
        -------
        float
            Duration in seconds.
        """
        if self.samples.ndim == 1:
            return self.samples.shape[0] / self.sampling_rate
        return self.samples.shape[-1] / self.sampling_rate

    @property
    def sps(self) -> float:
        """
        Samples per symbol.

        Returns
        -------
        float
            Ratio of sampling rate to symbol rate.
        """
        return self.sampling_rate / self.symbol_rate

    @property
    def bits_per_symbol(self) -> Optional[int]:
        """
        Bits per symbol for the active modulation scheme.

        Returns
        -------
        int or None
            Calculated as $\log_2(\text{modulation\_order})$.
        """
        if self.mod_order:
            return int(np.log2(self.mod_order))
        return None

    # =========================================================================
    # Plotting and Visualization
    # =========================================================================

    def plot_psd(
        self,
        nperseg: int = 128,
        detrend: Optional[Union[str, bool]] = False,
        average: Optional[str] = "mean",
        ax: Optional[Any] = None,
        title: Optional[str] = "Spectrum",
        x_axis: Optional[str] = "frequency",
        show: bool = False,
        **kwargs: Any,
    ) -> Optional[Tuple[Any, Any]]:
        """
        Plots the Power Spectral Density (PSD) of the signal.

        Parameters
        ----------
        nperseg : int, default 128
            Length of each segment for Welch's method.
        detrend : str or bool, default False
            Specifies how to detrend each segment.
        average : {"mean", "median"}, default "mean"
            Method to use for averaging segments.
        ax : matplotlib.axes.Axes, optional
            Pre-existing axis for plotting.
        title : str, default "Spectrum"
            Plot title.
        x_axis : {"frequency", "wavelength"}, default "frequency"
            The metric for the x-axis.
        show : bool, default False
            If True, calls `plt.show()` immediately.
        **kwargs : Any
            Additional arguments passed to `plotting.psd`.

        Returns
        -------
        figure, axis : tuple or None
            The matplotlib figure and axis objects, or None if `show=True`.
        """
        from . import plotting

        return plotting.psd(
            self.samples,
            sampling_rate=self.sampling_rate,
            nperseg=nperseg,
            detrend=detrend,
            average=average,
            center_frequency=self.center_frequency,
            domain=self.physical_domain if self.physical_domain else "RF",
            x_axis=x_axis,
            ax=ax,
            title=title,
            show=show,
            **kwargs,
        )

    def plot_symbols(
        self,
        start_symbol: int = 0,
        num_symbols: int = None,
        ax: Optional[Any] = None,
        title: Optional[str] = "Waveform",
        show: bool = False,
        **kwargs: Any,
    ) -> Optional[Tuple[Any, Any]]:
        """
        Plots the time-domain waveform of the signal symbols.

        Parameters
        ----------
        start_symbol : int, default 0
            The starting symbol to plot.
        num_symbols : int, optional
            Number of symbols to display. If None, plots the entire signal.
        ax : matplotlib.axes.Axes, optional
            Pre-existing axis for plotting.
        title : str, default "Waveform"
            Plot title.
        show : bool, default False
            If True, calls `plt.show()` immediately.
        **kwargs : Any
            Additional arguments passed to `plotting.time_domain`.

        Returns
        -------
        figure, axis : tuple or None
            The matplotlib figure and axis objects, or None if `show=True`.
        """
        from . import plotting

        return plotting.time_domain(
            self.samples,
            sampling_rate=self.sampling_rate,
            start_symbol=start_symbol,
            num_symbols=num_symbols,
            sps=self.sps,
            ax=ax,
            title=title,
            show=show,
            **kwargs,
        )

    def plot_eye(
        self,
        ax: Optional[Any] = None,
        type: str = "hist",
        title: Optional[str] = "Eye Diagram",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Optional[Tuple[Any, Any]]:
        """
        Plots the eye diagram of the signal.

        The eye diagram is a critical tool for assessing ISI and timing
        jitter. This method supports both high-density histograms and
        classic trace-based line plots.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or list, optional
            Plotting axis. For complex signals, a list of two axes can be
            provided (for I and Q eyes).
        type : {"hist", "line"}, default "hist"
            "hist": 2D histogram density plot (recommended for noisy signals).
            "line": Overlaid individual signal traces.
        title : str, default "Eye Diagram"
            Plot title.
        vmin, vmax : float, optional
            Density scale limits for "hist" mode.
        show : bool, default False
            If True, calls `plt.show()` immediately.
        **kwargs : Any
            Additional parameters passed to the underlying plotting functions.

        Returns
        -------
        figure, axis : tuple or None
            The matplotlib figure and axis objects, or None if `show=True`.
        """
        from . import plotting

        return plotting.eye_diagram(
            self.samples,
            ax=ax,
            sps=self.sps,
            type=type,
            title=title,
            vmin=vmin,
            vmax=vmax,
            show=show,
            **kwargs,
        )

    def plot_constellation(
        self,
        data: Literal["samples", "resolved"] = "samples",
        bins: int = 100,
        cmap: str = "inferno",
        overlay_ideal: bool = False,
        overlay_source: bool = False,
        ax: Optional[Any] = None,
        title: Optional[str] = "Constellation",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Optional[Tuple[Any, Any]]:
        """
        Plots the constellation diagram of the signal.

        Uses 2D histogram density mapping to visualize the distribution of
        received symbols, providing better insight into noise and
        impairments than traditional scatter plots.

        Parameters
        ----------
        data : {"samples", "resolved"}, default "samples"
            Which data to plot as the density field:
            - ``"samples"``: Raw oversampled IQ samples.
            - ``"resolved"``: Symbols at 1 sps from ``resolved_symbols``
              (call ``resolve_symbols()`` first).
        bins : int, default 100
            Number of bins per axis for the 2D density histogram.
        cmap : str, default "inferno"
            Matplotlib colormap for the density visualization.
        overlay_ideal : bool, default False
            If True, overlays the ideal constellation points on the plot.
        overlay_source : bool, default False
            If True, overlays ``source_symbols`` as white scatter markers
            on top of the density plot.
        ax : matplotlib.axes.Axes, optional
            Pre-existing axis for plotting.
        title : str, default "Constellation"
            Plot title.
        vmin, vmax : float, optional
            Density scale limits.
        show : bool, default False
            If True, calls `plt.show()` immediately.
        **kwargs : Any
            Additional arguments passed to `plotting.constellation`.

        Returns
        -------
        figure, axis : tuple or None
            The matplotlib figure and axis objects, or None if `show=True`.
        """
        from . import plotting

        if data == "resolved":
            if self.resolved_symbols is None:
                raise ValueError(
                    "No resolved_symbols available. Call resolve_symbols() first."
                )
            plot_data = self.resolved_symbols
        else:
            plot_data = self.samples

        result = plotting.constellation(
            plot_data,
            bins=bins,
            cmap=cmap,
            ax=ax,
            overlay_ideal=overlay_ideal,
            modulation=self.mod_scheme,
            order=self.mod_order,
            unipolar=self.mod_unipolar,
            title=title,
            vmin=vmin,
            vmax=vmax,
            show=False,
            **kwargs,
        )

        if overlay_source and self.source_symbols is not None and result is not None:
            fig, axes = result
            src = self.source_symbols
            src = to_device(src, "cpu")

            def _scatter_source(ax, symbols):
                ax.scatter(
                    symbols.real,
                    symbols.imag,
                    c="cyan",
                    edgecolors="black",
                    s=10,
                    linewidths=1,
                    zorder=10,
                    marker="o",
                )

            if src.ndim > 1:
                ax_list = list(np.asarray(axes).flat)
                for ch in range(min(src.shape[0], len(ax_list))):
                    _scatter_source(ax_list[ch], src[ch])
            else:
                # SISO: axes is a single Axes object
                _scatter_source(axes, src)

        if show:
            import matplotlib.pyplot as plt

            plt.show()
            return None
        return result

    def plot_equalizer(
        self,
        smoothing: int = 50,
        ax=None,
        title: Optional[str] = "Equalizer Diagnostics",
        show: bool = False,
    ) -> Optional[Tuple[Any, Any]]:
        """
        Plots equalizer diagnostics: convergence curve and tap weights.

        Requires ``equalize()`` to have been called first.

        Parameters
        ----------
        smoothing : int, default 50
            Moving-average window for the MSE convergence curve.
        ax : list of 2 Axes, optional
            Pre-existing axes ``[ax_convergence, ax_taps]``.
        title : str, optional
            Suptitle for the figure.
        show : bool, default False
            If True, calls ``plt.show()`` and returns None.

        Returns
        -------
        (fig, axes) or None
            Figure and axes array when ``show=False``, None otherwise.

        Raises
        ------
        ValueError
            If ``equalize()`` has not been called on this signal.
        """
        if self._equalizer_result is None:
            raise ValueError("No equalizer result available. Call equalize() first.")
        from . import plotting

        return plotting.equalizer_result(
            self._equalizer_result,
            smoothing=smoothing,
            ax=ax,
            title=title,
            show=show,
        )

    # =========================================================================
    # Signal Processing Methods
    # =========================================================================

    def upsample(self, factor: int) -> "Signal":
        """
        Upsamples the signal by an integer factor using polyphase filtering.

        This method increases the sampling rate by inserting zeros and
        applying an anti-imaging filter.

        Parameters
        ----------
        factor : int
            Upsampling factor (interpolation).

        Returns
        -------
        Signal
            self (modified in-place).
        """
        from . import multirate

        self.samples = multirate.upsample(self.samples, factor, axis=-1)
        self.sampling_rate = self.sampling_rate * factor
        return self

    def decimate(
        self, factor: int, filter_type: str = "fir", **kwargs: Any
    ) -> "Signal":
        """
        Decimates the signal by an integer factor using anti-aliasing filtering.

        Parameters
        ----------
        factor : int
            Decimation factor (e.g., 2 to reduce sample rate by half).
        filter_type : {"fir", "polyphase"}, default "fir"
            The type of decimation filter to use.
        **kwargs : Any
            Additional parameters passed to the decimation algorithm.

        Returns
        -------
        Signal
            self (modified in-place).

        Notes
        -----
        This method applies an anti-aliasing filter before downsampling. If
        the signal has already been matched-filtered, you should likely use
        `decimate_to_symbol_rate` instead to avoid double-filtering.
        """
        from . import multirate

        self.samples = multirate.decimate(
            self.samples, factor, filter_type=filter_type, axis=-1, **kwargs
        )
        self.sampling_rate = self.sampling_rate / factor
        return self

    def decimate_to_symbol_rate(self, offset: int = 0) -> "Signal":
        """
        Extracts symbols from an oversampled signal via direct slicing.

        This method is the canonical way to recover symbols at $1 \text{ sps}$
        after matched filtering. It does not apply any additional filtering,
        ensuring that the matched filter remains the optimal receiver.

        Parameters
        ----------
        offset : int, default 0
            The sampling phase offset in samples. Adjust this to sample
            at the peak of the impulse response or the maximum eye opening.

        Returns
        -------
        Signal
            self (modified in-place, sampling rate updated to symbol rate).

        Notes
        -----
        Use `resample` or `decimate` for general rate changes where
        anti-aliasing is required. Use this method ONLY when the signal
        is already filtered and aligned.
        """
        from . import multirate

        sps = int(self.sps)
        if sps <= 1:
            logger.info("Signal already at 1 sps, no downsampling needed.")
            return self
        self.samples = multirate.decimate_to_symbol_rate(
            self.samples, sps=sps, offset=offset, axis=-1
        )
        self.sampling_rate = self.symbol_rate
        return self

    def resample(
        self,
        up: Optional[int] = None,
        down: Optional[int] = None,
        sps_out: Optional[float] = None,
    ) -> "Signal":
        """
        Resamples the signal by a rational factor using polyphase filtering.

        This method is the canonical way to change the sampling rate of a
        signal while maintaining spectral integrity through anti-aliasing
        and anti-imaging filters.

        Parameters
        ----------
        up : int, optional
            Upsampling factor (interpolation).
        down : int, optional
            Downsampling factor (decimation).
        sps_out : float, optional
            Desired samples per symbol in the output signal. If provided,
            the required `up/down` factors are calculated automatically
            relative to the current `sps`.

        Returns
        -------
        Signal
            self (modified in-place, sampling rate updated).

        Notes
        -----
        .. warning::
            Do NOT use this method on a signal that has already been
            matched-filtered if the goal is to extract symbols at $1\text{ sps}$.
            The polyphase filter's transition band will distort the
            optimally filtered pulse shape. Use `decimate_to_symbol_rate` instead.
        """
        from . import multirate

        # If sps_out is provided, we use the current signal's sps as input sps
        sps_in = self.sps if sps_out is not None else None

        self.samples = multirate.resample(
            self.samples, up=up, down=down, sps_in=sps_in, sps_out=sps_out, axis=-1
        )

        # Update sampling rate
        if sps_out is not None:
            self.sampling_rate = sps_out * self.symbol_rate
        elif up is not None and down is not None:
            self.sampling_rate = self.sampling_rate * up / down

        return self

    def shift_frequency(self, offset: float) -> "Signal":
        """
        Applies a frequency offset (mixing) to the signal.

        This method shifts the signal's spectrum in the frequency domain.
        Metadata is updated to track total digital frequency offset.

        Parameters
        ----------
        offset : float
            Desired frequency shift in Hz.

        Returns
        -------
        Signal
            self (modified in-place).
        """
        from . import spectral

        self.samples, actual_offset = spectral.shift_frequency(
            self.samples, offset, self.sampling_rate
        )
        self.digital_frequency_offset += actual_offset
        return self

    def fir_filter(self, taps: ArrayType) -> "Signal":
        """
        Applies a Finite Impulse Response (FIR) filter to the signal.

        Parameters
        ----------
        taps : array_like
            FIR filter coefficients. Shape: (N_taps,).

        Returns
        -------
        Signal
            self (modified in-place).
        """
        from . import filtering

        self.samples = filtering.fir_filter(self.samples, taps, axis=-1)
        return self

    def shaping_filter_taps(self) -> ArrayType:
        """
        Computes shaping filter taps based on signal metadata.

        Returns
        -------
        array_like
            Generated filter taps normalized according to pulse type.
            Resides on the same device as the signal samples.

        Raises
        -------
        ValueError
            If `pulse_shape` is missing or unsupported.
        """
        if not self.pulse_shape or self.pulse_shape == "none":
            raise ValueError("No pulse shape defined for this signal.")
        logger.info(f"Generating shaping filter taps (shape: {self.pulse_shape}).")
        from . import filtering

        # Determine pulse width based on modulation if RZ
        p_width = 0.5 if self.mod_rz else 1.0

        # Generate taps using filtering module (returns default numpy usually)
        if self.pulse_shape == "rect":
            taps = np.ones(int(self.sps * p_width))
        elif self.pulse_shape == "smoothrect":
            taps = filtering.smoothrect_taps(
                sps=self.sps,
                span=self.filter_span,
                bt=self.smoothrect_bt,
                pulse_width=p_width,
            )
        elif self.pulse_shape == "gaussian":
            taps = filtering.gaussian_taps(
                sps=self.sps,
                span=self.filter_span,
                bt=self.gaussian_bt,
            )
        elif self.pulse_shape == "rrc":
            taps = filtering.rrc_taps(
                sps=self.sps,
                span=self.filter_span,
                rolloff=self.rrc_rolloff,
            )
        elif self.pulse_shape == "rc":
            taps = filtering.rc_taps(
                sps=self.sps,
                span=self.filter_span,
                rolloff=self.rc_rolloff,
            )
        else:
            raise ValueError(f"Unknown pulse shape: {self.pulse_shape}")

        # Ensure taps are on the correct device
        return to_device(taps, self.backend)

    def matched_filter(
        self,
        taps_normalization: str = "unit_energy",
        normalize_output: bool = False,
    ) -> "Signal":
        """
        Applies a matched filter to the signal samples.

        The filter taps are automatically derived from the `pulse_shape`
        and related parameters stored in the signal metadata.

        Parameters
        ----------
        taps_normalization : {"unit_energy", "unity_gain"}, default "unit_energy"
            Normalization strategy for the matched filter taps.
        normalize_output : bool, default False
            If True, normalizes the filtered samples to a peak amplitude of 1.0.

        Returns
        -------
        Signal
            self (modified in-place).
        """
        from . import filtering

        try:
            taps = self.shaping_filter_taps()
        except ValueError as e:
            logger.error(f"Cannot apply matched filter: {e}")
            return self
        self.samples = filtering.matched_filter(
            self.samples,
            taps,
            taps_normalization=taps_normalization,
            normalize_output=normalize_output,
            axis=-1,
        )
        return self

    def equalize(
        self,
        method: str = "lms",
        training_symbols: Optional[Any] = None,
        **kwargs,
    ) -> "Signal":
        """
        Applies adaptive or block equalization to the signal samples.

        Signal metadata is used automatically:

        - ``sps`` is read from the signal (``sampling_rate / symbol_rate``).
          Adaptive equalizers require 2 SPS (T/2-spaced input).
        - ``reference_constellation`` is built from ``mod_scheme`` /
          ``mod_order`` for LMS and RLS (DD slicing).
        - ``modulation`` / ``order`` are passed to CMA for R2 computation.

        Parameters
        ----------
        method : {"lms", "rls", "cma", "zf"}, default "lms"
            Equalization algorithm.
        training_symbols : array_like, optional
            Known symbols for data-aided equalization (at symbol rate, 1 SPS).
        **kwargs
            Algorithm-specific parameters forwarded to the equalizer
            function (e.g., ``num_taps``, ``step_size``, ``normalize``,
            ``forgetting_factor``, ``store_weights``).

        Returns
        -------
        Signal
            self (modified in-place). Equalizer result is accessible via
            ``signal._equalizer_result``.

        Raises
        ------
        ValueError
            If the signal is not at 2 SPS for adaptive methods.
        """
        from . import equalizers
        from .mapping import gray_constellation

        sps = int(self.sps) if self.sps else 2

        if method != "zf" and sps != 2:
            raise ValueError(
                f"Signal is at {sps} SPS. Adaptive equalizers require 2 SPS "
                f"(T/2-spaced input) — resample first."
            )

        constellation = None
        if self.mod_scheme and self.mod_order:
            constellation = gray_constellation(self.mod_scheme, self.mod_order)

        if method == "lms":
            kwargs.setdefault("reference_constellation", constellation)
            result = equalizers.lms(self.samples, training_symbols, sps=sps, **kwargs)
        elif method == "rls":
            kwargs.setdefault("reference_constellation", constellation)
            result = equalizers.rls(self.samples, training_symbols, sps=sps, **kwargs)
        elif method == "cma":
            kwargs.setdefault("modulation", self.mod_scheme)
            kwargs.setdefault("order", self.mod_order)
            result = equalizers.cma(self.samples, sps=sps, **kwargs)
        elif method == "zf":
            channel_estimate = kwargs.pop("channel_estimate")
            noise_variance = kwargs.pop("noise_variance", 0.0)
            self.samples = equalizers.zf_equalizer(
                self.samples, channel_estimate, noise_variance
            )
            return self
        else:
            raise ValueError(f"Unknown equalization method: {method}")

        self.samples = result.y_hat
        self._equalizer_result = result
        return self

    # =========================================================================
    # Generation Factory Methods
    # =========================================================================

    @classmethod
    def generate(
        cls,
        num_symbols: int,
        sps: float,
        symbol_rate: float,
        modulation: str,
        order: int,
        unipolar: bool = False,
        rz: bool = False,
        pulse_shape: str = "none",
        num_streams: int = 1,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> "Signal":
        """
        Generates a generic baseband waveform with specified modulation.

        This is the primary factory method for creating synthetic signals.
        It follows a bit-first architecture: random bits are generated,
        mapped to symbols, upsampled, and pulse-shaped.

        Parameters
        ----------
        num_symbols : int
            Number of symbols to generate per stream.
        sps : float
            Samples per symbol.
        symbol_rate : float
            Symbol rate in symbols per second (Baud).
        modulation : {"psk", "qam", "ask"}
            The modulation scheme identifier.
        order : int
            Modulation order (e.g., 4, 16, 64).
        unipolar : bool, default False
            If True, uses a unipolar constellation.
        rz : bool, default False
            If True, uses Return-to-Zero signaling.
        pulse_shape : str, default "none"
            Pulse shaping filter type (e.g., 'rrc', 'rect').
        num_streams : int, default 1
            Number of independent streams (MIMO).
        seed : int, optional
            Seed for reproducible random generation.
        **kwargs : Any
            Additional filter parameters (e.g., `filter_span`, `rrc_rolloff`).

        Returns
        -------
        Signal
            A new `Signal` instance.

        Notes
        -----
        Symbols are ``complex64`` for PSK/QAM and ``float32`` for ASK/PAM.
        The generated samples are automatically normalized to **peak amplitude (1.0)**
        via `filtering.shape_pulse`.
        However, the underlying bit mapping uses **unit average power (E_s=1)**
        by default to maintain mathematical consistency. Calling `resolve_symbols()`
        will restore the symbols to unit average power for consistent metric
        calculation and demapping.
        """
        from . import filtering, mapping

        # Bit-first architecture: generate bits → map to symbols
        k = int(np.log2(order))  # bits per symbol
        total_symbols = num_symbols * num_streams
        total_bits = total_symbols * k

        # Generate source bits
        bits = helpers.random_bits(total_bits, seed=seed)

        # Map bits to symbols
        symbols_flat = mapping.map_bits(bits, modulation, order, unipolar)

        if num_streams > 1:
            # Shape: (Channels, Time)
            symbols = symbols_flat.reshape(num_streams, num_symbols)
            bits = bits.reshape(num_streams, num_symbols * k)
        else:
            symbols = symbols_flat

        if is_cupy_available():
            symbols = to_device(symbols, "gpu")
            bits = to_device(bits, "gpu")

        # Apply pulse shaping
        # shape_pulse defaults to axis=-1 (Time) which is correct for (C, T)
        samples = filtering.shape_pulse(
            symbols=symbols, sps=sps, pulse_shape=pulse_shape, rz=rz, **kwargs
        )

        return cls(
            samples=samples,
            sampling_rate=symbol_rate * sps,
            symbol_rate=symbol_rate,
            mod_scheme=modulation.upper(),
            mod_order=order,
            mod_unipolar=unipolar,
            mod_rz=rz,
            source_bits=bits,
            source_symbols=symbols,
            pulse_shape=pulse_shape,
            **kwargs,
        )

    @classmethod
    def pam(
        cls,
        num_symbols: int,
        sps: int,
        symbol_rate: float,
        order: int,
        unipolar: bool = False,
        rz: bool = False,
        pulse_shape: Literal["rect", "smoothrect"] = "rect",
        num_streams: int = 1,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> "Signal":
        """
        Generates a Pulse Amplitude Modulation (PAM) baseband waveform.

        Supports both NRZ (Non-Return-to-Zero) and RZ (Return-to-Zero)
        signaling, with configurable pulse shaping and bipolar/unipolar
        constellations.

        Parameters
        ----------
        num_symbols : int
            Total number of symbols to generate per stream.
        sps : int
            Samples per symbol. For RZ mode, this must be an even integer.
        symbol_rate : float
            Symbol rate in symbols per second (Baud).
        order : int
            Modulation order (e.g., 2, 4, 8).
        unipolar : bool, default False
            If True, uses a unipolar constellation starting from 0 (e.g., 0, 1).
            If False, uses a symmetric bipolar constellation (e.g., -1, +1).
        rz : bool, default False
            If True, uses Return-to-Zero signaling.
        pulse_shape : {"rect", "smoothrect"}, default "rect"
            Pulse shaping filter type. Default is "rect" for PAM.
        num_streams : int, default 1
            Number of independent streams (channels) to generate.
        seed : int, optional
            Random seed for reproducible bit and symbol generation.
        **kwargs : Any
            Additional parameters passed to the pulse shaping filter.

        Returns
        -------
        Signal
            A `Signal` object containing the generated PAM waveform.

        Notes
        -----
        The generated samples are automatically normalized to **peak amplitude (1.0)**
        via `filtering.shape_pulse`.
        However, the underlying bit mapping uses **unit average power (E_s=1)**
        by default to maintain mathematical consistency. Calling `resolve_symbols()`
        will restore the symbols to unit average power for consistent metric
        calculation and demapping.
        """
        if rz:
            if sps % 2 != 0:
                raise ValueError("For correct RZ duty cycle, `sps` must be even")

            allowed_rz_pulses = ["rect", "smoothrect"]
            if pulse_shape not in allowed_rz_pulses:
                raise ValueError(
                    f"Pulse shape '{pulse_shape}' is not allowed for RZ PAM. "
                    f"Allowed: {allowed_rz_pulses}"
                )

        return cls.generate(
            num_symbols=num_symbols,
            sps=sps,
            symbol_rate=symbol_rate,
            modulation="PAM",
            order=order,
            unipolar=unipolar,
            rz=rz,
            pulse_shape=pulse_shape,
            num_streams=num_streams,
            seed=seed,
            **kwargs,
        )

    @classmethod
    def psk(
        cls,
        num_symbols: int,
        sps: float,
        symbol_rate: float,
        order: int,
        unipolar: bool = False,
        rz: bool = False,
        pulse_shape: str = "rrc",
        num_streams: int = 1,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> "Signal":
        """
        Generates a Phase Shift Keying (PSK) baseband waveform.

        Parameters
        ----------
        num_symbols : int
            Total number of symbols to generate per stream.
        sps : float
            Samples per symbol.
        symbol_rate : float
            Symbol rate in symbols per second (Baud).
        order : int
            Modulation order (e.g., 2 for BPSK, 4 for QPSK, 8 for 8-PSK).
        unipolar : bool, default False
            If True, uses a unipolar constellation.
        rz : bool, default False
            If True, uses Return-to-Zero signaling.
        pulse_shape : str, default "rrc"
            Pulse shaping filter type.
        num_streams : int, default 1
            Number of independent streams (channels) to generate.
        seed : int, optional
            Random seed for bit and symbol generation.
        **kwargs : Any
            Additional parameters passed to `filtering.shape_pulse`.

        Returns
        -------
        Signal
            A `Signal` object containing the PSK waveform.

        Notes
        -----
        The generated samples are automatically normalized to **peak amplitude (1.0)**
        via `filtering.shape_pulse`.
        However, the underlying bit mapping uses **unit average power (E_s=1)**
        by default to maintain mathematical consistency. Calling `resolve_symbols()`
        will restore the symbols to unit average power for consistent metric
        calculation and demapping.
        """
        return cls.generate(
            modulation="psk",
            order=order,
            num_symbols=num_symbols,
            sps=sps,
            symbol_rate=symbol_rate,
            pulse_shape=pulse_shape,
            num_streams=num_streams,
            seed=seed,
            unipolar=unipolar,
            rz=rz,
            **kwargs,
        )

    @classmethod
    def qam(
        cls,
        num_symbols: int,
        sps: float,
        symbol_rate: float,
        order: int,
        unipolar: bool = False,
        rz: bool = False,
        pulse_shape: str = "rrc",
        num_streams: int = 1,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> "Signal":
        """
        Generates a Quadrature Amplitude Modulation (QAM) baseband waveform.

        Parameters
        ----------
        num_symbols : int
            Number of symbols to generate per stream.
        sps : float
            Samples per symbol.
        symbol_rate : float
            Symbol rate in symbols per second (Baud).
        order : int
            Modulation order (e.g., 16, 64, 256).
        unipolar : bool, default False
            If True, uses a unipolar constellation.
        rz : bool, default False
            If True, uses Return-to-Zero signaling.
        pulse_shape : str, default "rrc"
            Pulse shaping filter type.
        num_streams : int, default 1
            Number of MIMO streams.
        seed : int, optional
            Seed for random generation.
        **kwargs : Any
            Additional filter parameters.

        Returns
        -------
        Signal
            A `Signal` object containing the QAM waveform.

        Notes
        -----
        The generated samples are automatically normalized to **peak amplitude (1.0)**
        via `filtering.shape_pulse`.
        However, the underlying bit mapping uses **unit average power (E_s=1)**
        by default to maintain mathematical consistency. Calling `resolve_symbols()`
        will restore the symbols to unit average power for consistent metric
        calculation and demapping.
        """
        return cls.generate(
            modulation="qam",
            order=order,
            num_symbols=num_symbols,
            sps=sps,
            symbol_rate=symbol_rate,
            pulse_shape=pulse_shape,
            num_streams=num_streams,
            seed=seed,
            unipolar=unipolar,
            rz=rz,
            **kwargs,
        )

    # =========================================================================
    # Resolving and Demapping Methods
    # =========================================================================

    def resolve_symbols(self, offset: int = 0) -> ArrayType:
        """
        Retrieves samples decimated to the symbol rate ($1\\text{ sps}$) and
        caches them in `self.resolved_symbols`.

        WARNING: Only integer offsets are supported. Use after timing correction.

        Parameters
        ----------
        offset : int, default 0
            Timing offset in samples to apply before decimation. This allows
            for choosing the optimal sampling point within the symbol period.

        Returns
        -------
        array_like
            Decimated samples at $1\\text{ sps}$.

        Notes
        -----
        This method automatically normalizes the decimated samples to **unit
        average power ($E_s=1$)**. This is critical because physical waveforms
        are often peak-normalized for transmission, which would skew Euclidean
        distance-based demapping and metrics (like EVM) relative to the ideal
        reference constellations.
        """
        sps = self.sps
        if sps is None:
            raise ValueError("Symbol rate or sampling rate missing.")

        if sps < 1:
            raise ValueError("Symbol rate must be >= 1.")

        if sps % 1 != 0:
            raise ValueError("Symbol rate must be an integer.")

        if sps == 1:
            logger.info("Signal already at 1 sps, no downsampling needed.")
            res = self.samples
        else:
            from . import multirate

            logger.info(
                f"SpS is not 1. Decimating to symbol rate with sps={sps} and offset={offset}."
            )
            res = multirate.decimate_to_symbol_rate(
                self.samples, sps=int(sps), offset=int(offset), axis=-1
            )

        # Normalize symbols to unit average power (E_s = 1) for consistent
        # demapping and metric calculation.
        # For MIMO, we normalize per-stream (axis=-1). Each resolved stream
        # is independently scaled to unit power.
        self.resolved_symbols = helpers.normalize(res, "average_power", axis=-1)
        return self.resolved_symbols

    def demap_symbols_hard(
        self,
        **kwargs: Any,
    ) -> ArrayType:
        """
        Maps resolved symbols to bits via minimum Euclidean distance (hard decision).

        Parameters
        ----------
        **kwargs : Any
            Additional arguments passed to `mapping.demap_symbols_hard`
            (e.g., ``unipolar``).

        Returns
        -------
        ndarray
            Array of recovered bits (int8). Shape: (..., N_symbols * log2(order)).
            Result is also stored in ``self.resolved_bits``.

        Raises
        ------
        ValueError
            If modulation metadata is missing or ``resolve_symbols()`` has not
            been called yet.
        """
        from .mapping import demap_symbols_hard

        if self.mod_scheme is None or self.mod_order is None:
            raise ValueError("Modulation scheme and order required for demapping.")

        if self.resolved_symbols is None:
            raise ValueError(
                "No resolved symbols available. Please call `resolve_symbols()` "
                "first to decimate the signal to symbol rate."
            )

        bits = demap_symbols_hard(
            symbols=self.resolved_symbols,
            modulation=self.mod_scheme,
            order=self.mod_order,
            unipolar=self.mod_unipolar,
            **kwargs,
        )
        self.resolved_bits = bits
        return bits

    # =========================================================================
    # Metrics Methods
    # =========================================================================

    def evm(
        self,
        reference_symbols: Optional[ArrayType] = None,
    ) -> Tuple[float, float]:
        """
        Computes the Error Vector Magnitude (EVM).

        EVM is a measure of the difference between the received symbols and
        the ideal reference symbols.

        Parameters
        ----------
        reference_symbols : array_like, optional
            Known transmitted symbols. If None, defaults to `source_symbols`
            stored in the signal metadata.

        Returns
        -------
        evm_percent : float
            EVM expressed as a percentage of the average symbol power.
        evm_db : float
            EVM expressed in decibels (dB).

        Raises
        -------
        ValueError
            If no reference symbols are available.
        """
        from . import metrics

        ref = (
            reference_symbols if reference_symbols is not None else self.source_symbols
        )
        if ref is None:
            raise ValueError(
                "No reference available. Either set source_symbols or provide "
                "reference_symbols argument."
            )

        if self.resolved_symbols is None:
            raise ValueError(
                "No resolved symbols available. Please call `resolve_symbols()` "
                "first to decimate the signal to symbol rate."
            )

        return metrics.evm(self.resolved_symbols, ref)

    def snr(
        self,
        reference_symbols: Optional[ArrayType] = None,
    ) -> float:
        """
        Estimates the Signal-to-Noise Ratio (SNR) using a Data-Aided method.

        This method calculates the ratio of reference signal power over
        the variance of the error vector.

        Parameters
        ----------
        reference_symbols : array_like, optional
            The known transmitted symbols. If None, defaults to the
            `source_symbols` stored in the signal metadata.

        Returns
        -------
        float
            Estimated SNR in decibels (dB).

        Raises
        ------
        ValueError
            If no reference symbols are available (neither provided
            nor stored in metadata).
        """
        from . import metrics

        ref = (
            reference_symbols if reference_symbols is not None else self.source_symbols
        )
        if ref is None:
            raise ValueError(
                "No reference available. Either set source_symbols or provide "
                "reference_symbols argument."
            )

        if self.resolved_symbols is None:
            raise ValueError(
                "No resolved symbols available. Please call `resolve_symbols()` "
                "first to decimate the signal to symbol rate."
            )
        return metrics.snr(self.resolved_symbols, ref)

    def ber(
        self,
        reference_bits: Optional[ArrayType] = None,
    ) -> Union[float, ArrayType]:
        """
        Computes the Bit Error Rate (BER).

        Compares `resolved_bits` against the reference bit sequence. Requires
        that `demap_symbols_hard()` (and `resolve_symbols()`) have been called previously.

        For MIMO signals, BER is computed independently per stream.

        Parameters
        ----------
        reference_bits : array_like, optional
            The original transmitted bits. If None, defaults to
            `source_bits` stored in metadata.

        Returns
        -------
        float or ndarray
            BER as a ratio in [0, 1]. Scalar for SISO, array for MIMO.

        Raises
        ------
        ValueError
            If no reference bits are available or if `resolved_bits` is not
            available.
        """
        from . import metrics

        ref = reference_bits if reference_bits is not None else self.source_bits
        if ref is None:
            raise ValueError(
                "No reference bits available. Either set source_bits or provide "
                "reference_bits argument."
            )

        if self.resolved_bits is None:
            raise ValueError(
                "No resolved bits available. Please call `demap_symbols_hard()` first."
            )

        return metrics.ber(self.resolved_bits, ref)


class Preamble(BaseModel):
    """
    Structured container for frame synchronization sequences (preambles).

    Preambles are automatically generated based on the specified sequence type
    and length. Manual overrides for bits or symbols are not supported to
    ensure consistency within the processing pipeline.

    Attributes
    ----------
    sequence_type : {"barker", "zc"}, default "barker"
        The synchronization sequence algorithm.
    length : int
        Total length of the preamble in symbols.
        For "barker": length must be from the set {2, 3, 4, 5, 7, 11, 13}.
        For "zc": length must be a prime number.
    kwargs : dict
        Additional parameters for sequence generation (e.g., 'root' for ZC).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    sequence_type: Literal["barker", "zc"] = "barker"
    length: int
    kwargs: Dict[str, Any] = Field(default_factory=dict)

    # Internal state managed during post-init
    _symbols: Any = PrivateAttr(default=None)

    # =========================================================================
    # Validators and Post-Initialization Hooks
    # =========================================================================

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization hook to automate symbol generation and device placement.

        This ensures that standard sequences are generated correctly according
        to the requested sequence properties.
        """
        from . import sync

        stype = self.sequence_type.lower()

        if stype == "barker":
            # Barker symbols (-1, +1)
            self._symbols = sync.barker_sequence(self.length)

        elif stype in ("zc", "zadoff_chu"):
            # ZC complex symbols
            root = self.kwargs.get("root", 1)
            self._symbols = sync.zadoff_chu_sequence(self.length, root=root)

        # Move to GPU if available
        if is_cupy_available():
            if self._symbols is not None:
                self._symbols = to_device(self._symbols, "gpu")

            # Ensure consistent internal dtype (complex64)
            self._symbols = self._symbols.astype("complex64")

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def symbols(self) -> Any:
        """The IQ symbols of the preamble."""
        return self._symbols

    @property
    def num_symbols(self) -> int:
        """Total number of symbols in the preamble."""
        return self.length

    # =========================================================================
    # Signal Generation
    # =========================================================================

    def to_signal(
        self,
        sps: int,
        symbol_rate: float,
        pulse_shape: str = "rrc",
        filter_span: int = 10,
        rrc_rolloff: float = 0.35,
        rc_rolloff: float = 0.35,
        smoothrect_bt: float = 1.0,
        gaussian_bt: float = 0.3,
        **kwargs: Any,
    ) -> Signal:
        """
        Generates a shaped waveform from the preamble sequence.

        Parameters
        ----------
        sps : int
            Samples per symbol.
        symbol_rate : float
            Symbol rate in Hz.
        pulse_shape : str, default "rrc"
            The pulse shaping type to apply.
        filter_span : int, default 10
            Filter span in symbols.
        rrc_rolloff : float, default 0.35
            Roll-off factor for RRC filter.
        rc_rolloff : float, default 0.35
            Roll-off factor for RC filter.
        smoothrect_bt : float, default 1.0
            Bandwidth-Time product for Smooth Rect filter.
        gaussian_bt : float, default 0.3
            Bandwidth-Time product for Gaussian filter.
        **kwargs : Any
            Additional arguments.

        Returns
        -------
        Signal
            A `Signal` object with the shaped preamble.

        Notes
        -----
        Like all waveform generation methods in `commstools`, this output is
        normalized to **peak amplitude (1.0)**.
        """
        from .filtering import shape_pulse

        samples = shape_pulse(
            self.symbols,
            sps=sps,
            pulse_shape=pulse_shape,
            filter_span=filter_span,
            rrc_rolloff=rrc_rolloff,
            rc_rolloff=rc_rolloff,
            smoothrect_bt=smoothrect_bt,
            gaussian_bt=gaussian_bt,
            **kwargs,
        )

        # Create minimal SignalInfo for the Preamble
        signal_info = SignalInfo(
            signal_type="preamble",
            preamble_seq_len=self.length,
            preamble_type=self.sequence_type,
            preamble_kwargs=self.kwargs,
        )

        return Signal(
            samples=samples,
            sampling_rate=symbol_rate * sps,
            symbol_rate=symbol_rate,
            mod_scheme=None,  # No single modulation for "preamble" frame
            mod_order=None,
            source_symbols=None,  # Avoid redundancy per user request
            pulse_shape=pulse_shape,
            signal_info=signal_info,
            **kwargs,
        )


class SingleCarrierFrame(BaseModel):
    """
    Represents a structured single-carrier frame with Preamble, Pilots, Payload, and Guard Interval.

    This class provides a high-level abstraction for constructing frames
    used in digital communication systems ($1/10/100\text{GbE}$, $5\text{G}$, etc.).
    It supports various pilot patterns for channel estimation and guard
    intervals for multi-path mitigation.

    Attributes
    ----------
    payload_len : int, default 1000
        Number of data symbols per spatial stream.
    payload_mod_scheme : str, default "PSK"
        Modulation for payload data (e.g., 'QAM').
    payload_mod_order : int, default 4
        Modulation order for payload (e.g., 16 for 16-QAM).
    payload_seed : int, default 42
        Seed for reproducible payload data generation.
    preamble : Preamble, optional
        Structured preamble for synchronization.
    preamble_mode : {"same", "time_orthogonal"}, default "same"
    pilot_pattern : {"none", "block", "comb"}, default "none"
        "none": No pilots.
        "block": A block of symbols at the start of the frame body.
        "comb": Single pilot symbols interleaved every `pilot_period`.
    pilot_period : int, default 0
        The period of pilot insertion in symbols.
    pilot_block_len : int, default 0
        Length of the pilot block (mode="block") in symbols.
    pilot_seed : int, default 1337
        Seed for pilot symbol generation.
    pilot_mod_scheme : str, default "PSK"
        Modulation for pilots.
    pilot_mod_order : int, default 4
        Modulation order for pilots.
    pilot_gain_db : float, default 0.0
        Pilot boosting in dB relative to the payload power.
    guard_type : {"zero", "cp"}, default "zero"
        "zero": Zero-padding at the end of the frame.
        "cp": Cyclic Prefix prepended to the frame.
    guard_len : int, default 0
        Length of the guard interval in symbols.
    num_streams : int, default 1
        Number of independent spatial streams (MIMO).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    payload_len: int = Field(default=1000, gt=0)
    payload_seed: int = 42
    payload_mod_scheme: str = "PSK"
    payload_mod_order: int = Field(default=4, ge=1)
    payload_mod_unipolar: bool = False
    preamble: Optional[Preamble] = None
    preamble_mode: Literal["same", "time_orthogonal"] = "same"

    pilot_pattern: Literal["none", "block", "comb"] = "none"
    pilot_period: int = Field(default=0, ge=0)
    pilot_block_len: int = Field(default=0, ge=0)
    pilot_seed: int = 1337
    pilot_mod_scheme: str = "PSK"
    pilot_mod_order: int = Field(default=4, ge=1)
    pilot_mod_unipolar: bool = False
    pilot_gain_db: float = 0.0

    guard_type: Literal["zero", "cp"] = "zero"
    guard_len: int = Field(default=0, ge=0)

    num_streams: int = Field(default=1, ge=1)

    # Internal cache
    _payload_bits: Optional[Any] = PrivateAttr(default=None)
    _payload_symbols: Optional[Any] = PrivateAttr(default=None)
    _pilot_bits: Optional[Any] = PrivateAttr(default=None)
    _pilot_symbols: Optional[Any] = PrivateAttr(default=None)

    # =========================================================================
    # Validators and Post-Initialization Hooks
    # =========================================================================

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization hook.
        """
        pass

    # =========================================================================
    # Mask Generation and Internal Data Preparation Methods
    # =========================================================================

    def _generate_pilot_mask(self) -> Tuple[ArrayType, int]:
        """
        Calculates the pilot placement mask and total frame length.

        Returns
        -------
        mask : array_like (bool)
            Boolean mask where True indicates a pilot symbol location.
        body_length : int
            Total number of symbols in the frame body (payload + pilots).
        """
        xp = cp if is_cupy_available() else np

        # No pilots: simple payload mapping
        if self.pilot_pattern == "none":
            body_length = self.payload_len
            mask = xp.zeros(body_length, dtype=bool)
            return mask, body_length

        # Comb pattern: single pilot every N symbols
        if self.pilot_pattern == "comb":
            if self.pilot_period <= 1:
                raise ValueError("pilot_period must be > 1 for 'comb' pattern.")
            data_per_period = self.pilot_period - 1
            num_full_periods = self.payload_len // data_per_period
            remainder = self.payload_len % data_per_period

            total_length = num_full_periods * self.pilot_period + remainder
            # If we have a remainder, we need one more pilot at the start of the partial period
            if remainder > 0:
                total_length += 1

            mask = xp.zeros(total_length, dtype=bool)
            mask[:: self.pilot_period] = True
            return mask, total_length

        # Block pattern: block of pilots followed by block of data
        if self.pilot_pattern == "block":
            if self.pilot_period <= self.pilot_block_len:
                raise ValueError(
                    "pilot_period must be > pilot_block_len for 'block' pattern."
                )
            data_per_block = self.pilot_period - self.pilot_block_len
            num_blocks = int(xp.ceil(self.payload_len / data_per_block))

            # Create a single block pattern [P P ... P D D ... D]
            block_pattern = xp.zeros(self.pilot_period, dtype=bool)
            block_pattern[: self.pilot_block_len] = True

            # Repeat the pattern for all blocks
            mask = xp.tile(block_pattern, num_blocks)

            # Truncation: Find the exact index where the required payload ends
            false_indices = xp.where(~mask)[0]
            last_idx = false_indices[self.payload_len - 1]
            mask = mask[: last_idx + 1]
            return mask, len(mask)

        return xp.zeros(self.payload_len, dtype=bool), self.payload_len

    def _ensure_payload_generated(self) -> None:
        """
        Generates and caches payload bits and symbols.

        This internal method ensures that the bit-first representations are available
        and correctly mapped to the target modulation scheme.
        """
        if self._payload_bits is not None:
            return

        from . import mapping

        k = int(np.log2(self.payload_mod_order))
        total_symbols = self.payload_len * self.num_streams
        bits = helpers.random_bits(total_symbols * k, seed=self.payload_seed)
        symbols = mapping.map_bits(
            bits=bits,
            modulation=self.payload_mod_scheme,
            order=self.payload_mod_order,
            unipolar=self.payload_mod_unipolar,
        )

        if self.num_streams > 1:
            bits = bits.reshape(self.num_streams, self.payload_len * k)
            symbols = symbols.reshape(self.num_streams, self.payload_len)

        self._payload_bits = bits
        self._payload_symbols = symbols

    def _ensure_pilot_generated(self) -> None:
        """
        Generates and caches pilot bits and symbols.

        This internal method ensuring pilots are generated with the correct
        seed and modulation before frame assembly.
        """
        if self._pilot_bits is not None or self.pilot_pattern == "none":
            return

        from . import mapping

        xp = cp if is_cupy_available() else np
        mask, _ = self._generate_pilot_mask()
        pilot_count = int(xp.sum(mask))
        if pilot_count == 0:
            return

        k = int(np.log2(self.pilot_mod_order))
        total_pilots = pilot_count * self.num_streams
        bits = helpers.random_bits(total_pilots * k, seed=self.pilot_seed)
        symbols = mapping.map_bits(
            bits=bits,
            modulation=self.pilot_mod_scheme,
            order=self.pilot_mod_order,
            unipolar=self.pilot_mod_unipolar,
        )

        if self.num_streams > 1:
            bits = bits.reshape(self.num_streams, pilot_count * k)
            symbols = symbols.reshape(self.num_streams, pilot_count)

        self._pilot_bits = bits
        self._pilot_symbols = symbols

    # =========================================================================
    # Properties for Accessing Payload and Pilot Data
    # =========================================================================

    @property
    def payload_bits(self) -> ArrayType:
        """
        Returns the raw payload bits.

        Returns
        -------
        array_like
            Binary bits (0s and 1s).
        """
        self._ensure_payload_generated()
        return self._payload_bits

    @property
    def payload_symbols(self) -> ArrayType:
        """
        Returns the modulated payload symbols.

        Returns
        -------
        array_like
            IQ symbols.
        """
        self._ensure_payload_generated()
        return self._payload_symbols

    @property
    def pilot_bits(self) -> Optional[ArrayType]:
        """
        Returns the raw pilot bits, if pilots are enabled.

        Returns
        -------
        array_like or None
            Binary bits if `pilot_pattern` is not "none".
        """
        if self.pilot_pattern == "none":
            return None
        self._ensure_pilot_generated()
        return self._pilot_bits

    @property
    def pilot_symbols(self) -> Optional[ArrayType]:
        """
        Returns the modulated pilot symbols.

        Returns
        -------
        array_like or None
            IQ symbols if `pilot_pattern` is not "none".
        """
        if self.pilot_pattern == "none":
            return None
        self._ensure_pilot_generated()
        return self._pilot_symbols

    @property
    def body_symbols(self) -> ArrayType:
        """
        Returns the interleaved payload and pilot symbols.

        WARNING: Pilot gain is applied if `pilot_gain_db` is not zero,
        so these are not "clean" symbols but scaled relatively.

        Returns
        -------
        array_like
            Determined by `pilot_pattern` and `pilot_period`.
        """
        xp = cp if is_cupy_available() else np
        mask, body_length = self._generate_pilot_mask()

        if self.num_streams > 1:
            # Shape: (Channels, Time)
            body = xp.zeros((self.num_streams, body_length), dtype="complex64")

            if self.pilot_pattern != "none":
                pilot_symbols = self.pilot_symbols
                # Apply pilot boosting/gain (dB to linear)
                if self.pilot_gain_db != 0.0:
                    pilot_symbols = pilot_symbols * (10 ** (self.pilot_gain_db / 20))

                body[:, mask] = pilot_symbols

            body[:, ~mask] = self.payload_symbols
        else:
            body = xp.zeros(body_length, dtype="complex64")
            if self.pilot_pattern != "none":
                pilot_symbols = self.pilot_symbols
                # Apply pilot boosting/gain (dB to linear)
                if self.pilot_gain_db != 0.0:
                    pilot_symbols = pilot_symbols * (10 ** (self.pilot_gain_db / 20))

                body[mask] = pilot_symbols
            body[~mask] = self.payload_symbols

        return body

    # =========================================================================
    # Frame Structure Mapping
    # =========================================================================

    def get_structure_map(
        self,
        unit: Literal["symbols", "samples"] = "symbols",
        sps: int = 1,
        include_preamble: bool = True,
    ) -> Dict[str, ArrayType]:
        """
        Generates boolean masks identifying the segments of the frame.

        Parameters
        ----------
        unit : {"symbols", "samples"}, default "symbols"
            The scale of the returned masks.
        sps : int, default 1
            Samples per symbol (required if unit="samples").
        include_preamble : bool, default True
            If True, returns masks for the full frame including preamble and
            guard intervals. If False, returns masks only for the segments
            after the preamble (and after CP removal if guard_type='cp').

        Returns
        -------
        dict
            Dictionary containing boolean masks for:
            - 'preamble' (only if include_preamble=True)
            - 'pilots'
            - 'payload'
            - 'guard' (only if include_preamble=True OR guard_type='zero')
        """
        xp = cp if is_cupy_available() else np
        mask, body_length = self._generate_pilot_mask()

        preamble_len = 0
        if self.preamble is not None:
            preamble_len = self.preamble.num_symbols
            if self.preamble_mode == "time_orthogonal" and self.num_streams > 1:
                preamble_len *= self.num_streams

        if include_preamble:
            total_len = preamble_len + body_length + self.guard_len

            preamble_bool = xp.zeros(total_len, dtype=bool)
            pilot_bool = xp.zeros(total_len, dtype=bool)
            payload_bool = xp.zeros(total_len, dtype=bool)
            guard_bool = xp.zeros(total_len, dtype=bool)

            if self.guard_type == "cp":
                g_slice = slice(0, self.guard_len)
                p_slice = slice(self.guard_len, self.guard_len + preamble_len)
                b_slice = slice(self.guard_len + preamble_len, total_len)
            else:
                p_slice = slice(0, preamble_len)
                b_slice = slice(preamble_len, preamble_len + body_length)
                g_slice = slice(preamble_len + body_length, total_len)

            if preamble_len > 0:
                preamble_bool[p_slice] = True

            pilot_bool[b_slice] = mask
            payload_bool[b_slice] = ~mask

            if self.guard_len > 0:
                guard_bool[g_slice] = True

            res = {
                "preamble": preamble_bool,
                "pilots": pilot_bool,
                "payload": payload_bool,
                "guard": guard_bool,
            }
        else:
            # Preamble removed.
            # If CP, guard is at the start and is typically removed with preamble.
            # If ZERO, guard is at the end and remains part of the signal.
            if self.guard_type == "cp":
                total_len = body_length
                pilot_bool = mask
                payload_bool = ~mask
                res = {
                    "pilots": pilot_bool,
                    "payload": payload_bool,
                }
            else:
                total_len = body_length + self.guard_len
                pilot_bool = xp.zeros(total_len, dtype=bool)
                payload_bool = xp.zeros(total_len, dtype=bool)
                guard_bool = xp.zeros(total_len, dtype=bool)

                b_slice = slice(0, body_length)
                g_slice = slice(body_length, total_len)

                pilot_bool[b_slice] = mask
                payload_bool[b_slice] = ~mask
                guard_bool[g_slice] = True

                res = {
                    "pilots": pilot_bool,
                    "payload": payload_bool,
                    "guard": guard_bool,
                }

        if unit == "samples":
            for k in res:
                res[k] = xp.repeat(res[k], int(sps))

        return res

    # =========================================================================
    # Signal Generation
    # =========================================================================

    def to_signal(
        self,
        sps: int = 4,
        symbol_rate: float = 1e6,
        pulse_shape: str = "rrc",
        filter_span: int = 10,
        rrc_rolloff: float = 0.35,
        rc_rolloff: float = 0.35,
        smoothrect_bt: float = 1.0,
        gaussian_bt: float = 0.3,
        **kwargs: Any,
    ) -> Signal:
        """
        Generates a shaped, oversampled waveform from the frame description.

        This is the primary method for moving from a logical frame to
        physical IQ samples. It handles upsampling, pulse shaping,
        guard interval insertion, and metadata population.

        Parameters
        ----------
        sps : int, default 4
            Samples per symbol (oversampling factor).
        pulse_shape : str, default "rrc"
            Pulse shaping filter type.
        filter_span : int, default 10
            Filter span in symbols.
        rrc_rolloff : float, default 0.35
            Roll-off factor for RRC filter.
        rc_rolloff : float, default 0.35
            Roll-off factor for RC filter.
        smoothrect_bt : float, default 1.0
            Bandwidth-Time product for Smooth Rect filter.
        gaussian_bt : float, default 0.3
            Bandwidth-Time product for Gaussian filter.
        **kwargs : Any
            Additional parameters passed to the shaping filter.

        Returns
        -------
        Signal
            A `Signal` object containing the IQ samples and `SignalInfo` metadata.

        Notes
        -----
        The resulting waveform samples are normalized to **peak amplitude (1.0)**.
        """
        xp = cp if is_cupy_available() else np
        from .filtering import shape_pulse

        # 1. Shape Body (Payload + Pilots)
        body_symbols = self.body_symbols
        body_samples = shape_pulse(
            symbols=body_symbols,
            sps=sps,
            pulse_shape=pulse_shape,
            filter_span=filter_span,
            rrc_rolloff=rrc_rolloff,
            rc_rolloff=rc_rolloff,
            smoothrect_bt=smoothrect_bt,
            gaussian_bt=gaussian_bt,
            **kwargs,
        )

        # Normalize Body to Peak 1.0
        body_samples = helpers.normalize(body_samples, mode="peak")

        # 2. Shape Preamble (if present)
        if self.preamble is not None:
            # Use Preamble's to_signal for shaping to reuse logic,
            # but we only need the samples.
            # CRITICAL: Must use EXACT same shaping parameters as body.
            preamble_signal = self.preamble.to_signal(
                sps=sps,
                symbol_rate=symbol_rate,
                pulse_shape=pulse_shape,
                filter_span=filter_span,
                rrc_rolloff=rrc_rolloff,
                rc_rolloff=rc_rolloff,
                smoothrect_bt=smoothrect_bt,
                gaussian_bt=gaussian_bt,
                **kwargs,
            )
            preamble_samples = preamble_signal.samples

            # Normalize Preamble to Peak 1.0
            preamble_samples = helpers.normalize(preamble_samples, mode="peak")

            # Handle MIMO preamble structure
            preamble_samples = helpers.expand_preamble_mimo(
                preamble_samples, self.num_streams, self.preamble_mode
            )

            # Concatenate Preamble + Body
            samples = xp.concatenate([preamble_samples, body_samples], axis=-1)
        else:
            samples = body_samples

        # 3. Apply Guard Interval at sample level
        if self.guard_len > 0:
            guard_len_samples = int(self.guard_len * sps)
            if self.guard_type == "zero":
                if self.num_streams > 1:
                    zeros = xp.zeros(
                        (self.num_streams, guard_len_samples), dtype="complex64"
                    )
                else:
                    zeros = xp.zeros(guard_len_samples, dtype="complex64")
                samples = xp.concatenate([samples, zeros], axis=-1)
            elif self.guard_type == "cp":
                cp_slice = samples[..., -guard_len_samples:]
                samples = xp.concatenate([cp_slice, samples], axis=-1)

        # 4. Build SignalInfo metadata
        mask, _ = self._generate_pilot_mask()
        pilot_count = int(xp.sum(mask)) if self.pilot_pattern != "none" else 0

        if self.preamble:
            preamble_base_len = self.preamble.num_symbols
        else:
            preamble_base_len = None

        signal_info = SignalInfo(
            signal_type="single_carrier_frame",
            payload_mod_scheme=self.payload_mod_scheme,
            payload_mod_order=self.payload_mod_order,
            payload_mod_unipolar=self.payload_mod_unipolar,
            preamble_seq_len=preamble_base_len,
            preamble_type=self.preamble.sequence_type if self.preamble else None,
            preamble_mode=self.preamble_mode if self.preamble else None,
            preamble_kwargs=self.preamble.kwargs if self.preamble else None,
            payload_len=self.payload_len,
            pilot_count=pilot_count,
            pilot_pattern=self.pilot_pattern,
            pilot_period=self.pilot_period,
            pilot_block_len=self.pilot_block_len,
            pilot_mod_scheme=self.pilot_mod_scheme,
            pilot_mod_order=self.pilot_mod_order,
            pilot_mod_unipolar=self.pilot_mod_unipolar,
            pilot_gain_db=self.pilot_gain_db,
            guard_len=self.guard_len,
            guard_type=self.guard_type,
            num_streams=self.num_streams,
        )

        return Signal(
            samples=samples,
            sampling_rate=symbol_rate * sps,
            symbol_rate=symbol_rate,
            mod_scheme=None,  # Moved to SignalInfo to avoid misleading metadata
            mod_order=None,  # Moved to SignalInfo
            mod_unipolar=None,
            mod_rz=None,
            source_bits=None,  # Avoid redundancy per user request; access via Frame
            source_symbols=None,  # Avoid redundancy; access via Frame
            pulse_shape=pulse_shape,
            filter_span=filter_span,
            rrc_rolloff=rrc_rolloff,
            rc_rolloff=rc_rolloff,
            smoothrect_bt=smoothrect_bt,
            gaussian_bt=gaussian_bt,
            signal_info=signal_info,
            **kwargs,
        )
