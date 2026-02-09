"""
Core signal processing abstractions.

This module defines the primary data structures for the library:
- `Signal`: Encapsulates complex IQ samples, physical layer metadata (sampling rate, symbol rate,
  modulation scheme/order), and methods for signal analysis and transformation.
- `SingleCarrierFrame`: A container for single carrier frames with preambles, pilots, and payloads.

All core classes are built on Pydantic for validation and support both CPU (NumPy) and GPU (CuPy) backends.
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


from . import utils
from .backend import (
    ArrayType,
    dispatch,
    from_jax,
    get_array_module,
    get_scipy_module,
    is_cupy_available,
    to_device,
    to_jax,
)
from .logger import logger


class FrameInfo(BaseModel):
    """
    Describes frame structure within a Signal.

    This class holds metadata about the frame structure when a Signal
    is generated from a SingleCarrierFrame, enabling distinguishing
    frame-generated Signals from standalone Signals.

    Attributes:
        preamble_len: Number of preamble symbols.
        payload_len: Number of payload symbols.
        pilot_count: Total number of pilot symbols.
        guard_len: Guard interval length in symbols.
        guard_type: Type of guard ('zero' or 'cp').
        preamble_mod_scheme: Modulation scheme used for preamble.
        preamble_mod_order: Modulation order used for preamble.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    preamble_len: int = 0
    payload_len: int = 0
    pilot_count: int = 0
    guard_len: int = 0
    guard_type: Literal["zero", "cp"] = "zero"
    preamble_mod_scheme: Optional[str] = None
    preamble_mod_order: Optional[int] = None


class Signal(BaseModel):
    """
    Represents a digital signal with associated physical layer metadata.

    This class serves as the primary data container for the library, encapsulating
    both the raw IQ samples and the contextual information required for digital
    signal processing (DSP) and analysis.

    Attributes:
        samples: The complex IQ samples of the signal. Can be NumPy or CuPy arrays.
                 Shape should be (N_samples,) for SISO or (N_channels, N_samples) for MIMO.
        sampling_rate: The sampling rate of the signal in Hz. Must be positive.
        symbol_rate: The symbol rate (baud rate) in Hz. Used for SPS calculation.
        modulation_scheme: Identifier for the modulation format (e.g., 'QPSK', '16QAM').
        modulation_order: The number of symbols in the constellation (e.g., 4, 16).
        source_symbols: Optional array of original symbols that generated this signal.
        pulse_shape: Name of the pulse shaping filter applied (e.g., 'rrc', 'rect').
        spectral_domain: The signal's placement in the spectrum ('BASEBAND', 'PASSBAND', 'INTERMEDIATE').
        physical_domain: The physical layer domain ('DIG' for digital, 'RF' for radio frequency, 'OPT' for optical).
        center_frequency: The carrier or center frequency in Hz.
        digital_frequency_offset: Any applied digital frequency shift in Hz.
        filter_span: Filter span in symbols for pulse shaping.
        rrc_rolloff: Roll-off factor for RRC filter.
        rc_rolloff: Roll-off factor for RC filter.
        smoothrect_bt: BT product for SmoothRect filter.
        gaussian_bt: BT product for Gaussian filter.

    Raises:
        ValidationError: If the input fields do not satisfy the Pydantic constraints (e.g., non-positive sampling rate).
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )

    samples: Any
    sampling_rate: float = Field(..., gt=0)
    symbol_rate: float = Field(..., gt=0)
    modulation_scheme: Optional[str] = None
    modulation_order: Optional[int] = None
    source_bits: Optional[Any] = None
    source_symbols: Optional[Any] = None
    pulse_shape: Optional[str] = None
    spectral_domain: Literal["BASEBAND", "PASSBAND", "INTERMEDIATE"] = "BASEBAND"
    physical_domain: Literal["DIG", "RF", "OPT"] = "DIG"
    center_frequency: float = Field(default=0, ge=0)
    digital_frequency_offset: float = Field(default=0)

    # Pulse shaping parameters
    filter_span: int = 10
    rrc_rolloff: float = 0.35
    rc_rolloff: float = 0.35
    gaussian_bt: float = 0.3
    smoothrect_bt: float = 1.0

    # Frame structure info (populated when Signal is generated from Frame)
    frame_info: Optional[FrameInfo] = None

    @field_validator("samples", mode="before")
    @classmethod
    def validate_samples(cls, v: Any) -> Any:
        """
        Validates and coerces the samples input into a backend-compatible array.
        Enforces (N_samples, N_channels) shape convention for multidimensional inputs.

        Args:
            v: The input samples (array-like, list, or tuple).

        Returns:
            The validated samples as a NumPy or CuPy array.

        Raises:
            ValueError: If the input cannot be converted to a supported array type.
        """
        arr = utils.validate_array(v, name="samples")

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
        Post-initialization hook.
        - Auto-derives source_symbols from source_bits if bits provided but symbols not.
        - Moves the signal to GPU if a compatible device is discovered.
        """
        # Bit-first: derive symbols from bits if not provided
        if self.source_bits is not None and self.source_symbols is None:
            if self.modulation_scheme and self.modulation_order:
                from . import mapping

                mod = self.modulation_scheme.lower()
                if "-" in mod:
                    mod = mod.split("-")[-1]
                self.source_symbols = mapping.map_bits(
                    self.source_bits, mod, self.modulation_order
                )

        # Default to GPU if available and supported
        if is_cupy_available():
            self.to("gpu")

    @property
    def xp(self) -> types.ModuleType:
        """
        Returns the array module (numpy or cupy) for the signal's data.

        This property allows for backend-agnostic array operations by returning
        the appropriate library based on whether the data is on CPU or GPU.
        """
        return get_array_module(self.samples)

    @property
    def sp(self) -> types.ModuleType:
        """
        Returns the scipy module (scipy or cupyx.scipy) for the signal's data.

        Similar to `xp`, this returns the appropriate signal processing library
        compatible with the current data backend.
        """
        return get_scipy_module(self.xp)

    @property
    def backend(self) -> str:
        """
        Returns the name of the backend ('CPU' or 'GPU').
        """
        return "GPU" if self.xp == cp else "CPU"

    @property
    def num_streams(self) -> int:
        """
        Returns the number of spatial/polarization streams.
        1 for SISO, N > 1 for MIMO/Dual-Pol.
        """
        if self.samples.ndim == 1:
            return 1
        return self.samples.shape[0]

    @property
    def duration(self) -> float:
        """
        Duration of the signal in seconds.

        Calculated as the number of samples divided by the sampling rate.
        """
        if self.samples.ndim == 1:
            return self.samples.shape[0] / self.sampling_rate
        return self.samples.shape[-1] / self.sampling_rate

    @property
    def sps(self) -> float:
        """
        Samples per symbol.

        Calculated as the sampling rate divided by the symbol rate.
        """
        return self.sampling_rate / self.symbol_rate

    def to(self, device: str) -> "Signal":
        """
        Moves the signal data to the specified device.

        Args:
            device: The target device ('CPU' or 'GPU').

        Returns:
            self
        """
        self.samples = to_device(self.samples, device)
        return self

    def export_samples_to_jax(self, device: Optional[str] = None) -> Any:
        """
        Exports the signal samples to a JAX array, ensuring consistency with the signal backend.

        Args:
            device: Optional target device ('CPU', 'GPU', or 'TPU').
                    If None, the signal's current backend is used.

        Returns:
            A JAX array containing the signal samples.
        """
        # If device is not explicitly requested, use the signal's backend
        target_device = device if device is not None else self.backend
        return to_jax(self.samples, device=target_device)

    def update_samples_from_jax(self, jax_array: Any) -> "Signal":
        """
        Updates the signal samples from a JAX array, preserving the original signal backend.

        Args:
            jax_array: Input JAX array.

        Returns:
            self
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
        Returns the time vector associated with the signal samples.

        The time vector starts at 0 and increments by 1/sampling_rate for each sample.

        Returns:
            An array representing the time axis in seconds.
        """
        n_samples = self.samples.shape[-1]
        return self.xp.arange(0, n_samples) / self.sampling_rate

    @property
    def num_bits(self) -> Optional[int]:
        """
        Total number of source bits.

        Returns:
            Number of bits if source_bits is available, else None.
        """
        if self.source_bits is not None:
            return int(self.source_bits.size)
        return None

    @property
    def bits_per_symbol(self) -> Optional[int]:
        """
        Bits per symbol for current modulation.

        Returns:
            Bits per symbol if modulation_order is defined, else None.
        """
        if self.modulation_order:
            return int(np.log2(self.modulation_order))
        return None

    def welch_psd(
        self,
        nperseg: int = 256,
        detrend: Optional[Union[str, bool]] = False,
        average: Optional[str] = "mean",
    ) -> Tuple[ArrayType, ArrayType]:
        """
        Compute the Power Spectral Density (PSD) using Welch's method.

        Args:
            nperseg: Length of each segment.
            detrend: Detrend method.
            average: Averaging method.

        Returns:
            Tuple of (frequency_axis, psd_values).
            If signal is MIMO, psd_values will be (N_channels, N_freqs) or similar depending on implementation.
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
        Plot the Power Spectral Density (PSD) of the signal.

        Args:
            nperseg: Length of each segment.
            detrend: Detrend method.
            average: Averaging method.
            ax: Optional matplotlib axis to plot on.
            title: Title of the plot.
            x_axis: X-axis type ('frequency' or 'wavelength').
            show: Whether to call plt.show().
            **kwargs: Additional plotting arguments.

        Returns:
            Tuple of (figure, axis) if show is False, else None.
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

    def print_info(self) -> None:
        """
        Prints a summary of the signal properties.
        """
        import pandas as pd
        from IPython import get_ipython
        from IPython.display import display

        data = {
            "Property": [
                "Spectral Domain",
                "Physical Domain",
                "Modulation Scheme",
                "Modulation Order",
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
                self.modulation_scheme,
                str(self.modulation_order) if self.modulation_order else "None",
                utils.format_si(self.symbol_rate, "Baud"),
                utils.format_si(
                    self.symbol_rate * np.log2(self.modulation_order), "bps"
                )
                if self.modulation_order
                else "None",
                utils.format_si(self.sampling_rate, "Hz"),
                f"{self.sps:.2f}",
                self.pulse_shape.upper() if self.pulse_shape else "None",
                utils.format_si(self.duration, "s"),
                utils.format_si(self.center_frequency, "Hz"),
                utils.format_si(self.digital_frequency_offset, "Hz"),
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

    def plot_symbols(
        self,
        num_symbols: int = None,
        ax: Optional[Any] = None,
        title: Optional[str] = "Waveform",
        show: bool = False,
        **kwargs: Any,
    ) -> Optional[Tuple[Any, Any]]:
        """
        Plot the time-domain waveform of the signal.

        Args:
            num_symbols: Number of symbols to plot.
            ax: Optional matplotlib axis to plot on.
            title: Title of the plot.
            show: Whether to call plt.show().
            **kwargs: Additional plotting arguments.

        Returns:
            Tuple of (figure, axis) if show is False, else None.
        """
        from . import plotting

        return plotting.time_domain(
            self.samples,
            sampling_rate=self.sampling_rate,
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
        Plot the eye diagram of the signal.

        Args:
            ax: Optional matplotlib axis (or list of axes for complex signals) to plot on.
            type: Type of plot ('hist' or 'line').
            title: Title of the plot.
            vmin: Minimum density value for colormap (hist mode only). If None, auto-scaled.
            vmax: Maximum density value for colormap (hist mode only). If None, auto-scaled.
                  Histogram is normalized to [0, 1], so vmax=1 shows full range.
            show: Whether to call plt.show().
            **kwargs: Additional arguments passed to plot (line mode) or imshow (hist mode).

        Returns:
            Tuple of (figure, axis) if show is False, else None.
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
        bins: int = 100,
        cmap: str = "inferno",
        overlay_ideal: bool = False,
        ax: Optional[Any] = None,
        title: Optional[str] = "Constellation",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        show: bool = False,
        **kwargs: Any,
    ) -> Optional[Tuple[Any, Any]]:
        """
        Plot the constellation density diagram of the signal.

        Args:
            bins: Number of histogram bins per axis.
            cmap: Colormap for density plot (default: 'inferno').
            overlay_ideal: If True, overlay ideal constellation points.
            ax: Optional matplotlib axis to plot on.
            title: Title of the plot.
            vmin: Minimum density value for colormap. If None, auto-scaled.
            vmax: Maximum density value for colormap. If None, auto-scaled.
                  Histogram is normalized to [0, 1], so vmax=1 shows full range.
            show: Whether to call plt.show().
            **kwargs: Additional arguments passed to imshow.

        Returns:
            Tuple of (figure, axis) if show is False, else None.
        """
        from . import plotting

        return plotting.constellation(
            self.samples,
            bins=bins,
            cmap=cmap,
            ax=ax,
            overlay_ideal=overlay_ideal,
            modulation=self.modulation_scheme.lower()
            if self.modulation_scheme
            else None,
            order=self.modulation_order,
            title=title,
            vmin=vmin,
            vmax=vmax,
            show=show,
            **kwargs,
        )

    def upsample(self, factor: int) -> "Signal":
        """
        Upsample the signal.

        Args:
            factor: Upsampling factor.

        Returns:
            self
        """
        from . import multirate

        self.samples = multirate.upsample(self.samples, factor, axis=-1)
        self.sampling_rate = self.sampling_rate * factor
        return self

    def decimate(
        self, factor: int, filter_type: str = "fir", **kwargs: Any
    ) -> "Signal":
        """
        Decimate the signal.

        Args:
            factor: Decimation factor.
            filter_type: Filter type.
            **kwargs: Additional arguments for decimate.

        Returns:
            self
        """
        from . import multirate

        self.samples = multirate.decimate(
            self.samples, factor, filter_type=filter_type, axis=-1, **kwargs
        )
        self.sampling_rate = self.sampling_rate / factor
        return self

    def resample(
        self,
        up: Optional[int] = None,
        down: Optional[int] = None,
        sps_out: Optional[float] = None,
    ) -> "Signal":
        """
        Resample the signal by a rational factor using polyphase filtering.

        This method applies anti-aliasing/anti-imaging filtering during resampling.
        Use this when changing sample rate BEFORE matched filtering or for general
        rate conversion.

        NOTE: Do NOT use this after matched filtering to go to 1 sps!
        The polyphase filter will degrade the already-filtered signal.
        Use `downsample_to_symbols()` instead for post-matched-filter decimation.

        Args:
            up: Upsampling factor.
            down: Downsampling factor.
            sps_out: Target samples per symbol (calculates up/down automatically).

        Returns:
            self
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

    def downsample_to_symbols(self, offset: int = 0) -> "Signal":
        """
        Extract symbols by simple decimation (no additional filtering).

        Use this AFTER matched filtering to go from oversampled to 1 sps.
        Unlike `resample()`, this does NOT apply additional filtering, which is
        correct when matched filter has already removed out-of-band noise.

        When to use:
        - downsample_to_symbols: After matched filter, for clean symbol extraction
        - resample: For general rate changes, BEFORE matched filtering

        Args:
            offset: Timing offset in samples (0 to sps-1). Adjusts the sampling
                    phase to find optimal eye opening. Default 0 assumes the
                    matched filter peak is at the first sample of each symbol.

        Returns:
            self (modified in-place)

        Example:
            >>> sig.matched_filter()
            >>> sig.downsample_to_symbols()  # Clean symbols at 1 sps
        """
        from . import multirate

        sps = int(self.sps)
        if sps <= 1:
            logger.warning("Signal already at 1 sps, no downsampling needed.")
            return self

        self.samples = multirate.downsample_to_symbols(
            self.samples, sps=sps, offset=offset, axis=-1
        )
        self.sampling_rate = self.symbol_rate

        return self

    def shift_frequency(self, offset: float) -> "Signal":
        """
        Apply a frequency offset to the signal.

        Args:
            offset: Desired frequency offset in Hz.

        Returns:
            self
        """
        from . import spectral

        (
            self.samples,
            actual_offset,
        ) = spectral.shift_frequency(self.samples, offset, self.sampling_rate)

        # Update metadata
        self.digital_frequency_offset += actual_offset

        return self

    def fir_filter(self, taps: ArrayType) -> "Signal":
        """
        Apply FIR filter to the signal.

        Args:
            taps: Filter taps.

        Returns:
            self
        """
        from . import filtering

        # Axis -1 is time
        self.samples = filtering.fir_filter(self.samples, taps, axis=-1)
        return self

    def shaping_filter_taps(self) -> ArrayType:
        """
        Calculates and returns the shaping filter taps based on the signal's
        pulse shape parameters.

        Returns:
            Filter taps array.

        Raises:
            ValueError: If pulse shape is not defined.
        """
        if not self.pulse_shape or self.pulse_shape == "none":
            raise ValueError("No pulse shape defined for this signal.")

        logger.debug(f"Generating shaping filter taps (shape: {self.pulse_shape}).")
        from . import filtering

        # Determine pulse width based on modulation if RZ
        p_width = 1.0
        if self.modulation_scheme and "RZ" in self.modulation_scheme.upper():
            p_width = 0.5

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
        Apply matched filter to the signal.
        The filter taps are calculated automatically based on the signal properties.

        Args:
            taps_normalization: Normalization to apply to the matched filter taps.
                                Options: 'unity_gain', 'unit_energy'. Default is 'unit_energy'.
            normalize_output: If True, normalizes the output samples to have a maximum
                              absolute value of 1.0. Default is False.

        Returns:
            self
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

    def copy(self) -> "Signal":
        """
        Creates a deep copy of the Signal object.

        Returns:
            A new Signal object with copied data.
        """
        return self.model_copy(deep=True)

    @classmethod
    def generate(
        cls,
        modulation: str,
        order: int,
        num_symbols: int,
        sps: float,
        symbol_rate: float,
        pulse_shape: str = "none",
        num_streams: int = 1,
        seed: Optional[int] = None,
        dtype: Optional[Any] = np.complex64,
        **kwargs: Any,
    ) -> "Signal":
        """
        Generate a baseband Signal with specified parameters.

        Args:
            modulation: Modulation scheme ('psk', 'qam', 'ask').
            order: Modulation order.
            num_symbols: Number of symbols to generate per stream.
            sps: Samples per symbol.
            symbol_rate: Symbol rate in Hz.
            pulse_shape: Pulse shaping type ('none', 'rect', 'smoothrect', 'gaussian', 'rrc', 'rc', 'sinc'). Default: 'none'.
            num_streams: Number of independent streams (1 for SISO, 2 for Dual-Pol, etc.).
            seed: Random seed for bit generation.
            dtype: Output dtype for precision control (e.g., np.complex64, np.complex128).
                   Default: complex64 for 2x memory savings and faster GPU.
            **kwargs: Pulse shaping parameters:
                filter_span (int): Filter span in symbols (default: 10).
                rrc_rolloff (float): Roll-off factor for RRC filter (default: 0.35).
                rc_rolloff (float): Roll-off factor for RC filter (default: 0.35).
                smoothrect_bt (float): BT product for SmoothRect filter (default: 1.0).
                gaussian_bt (float): BT product for Gaussian filter (default: 0.3).

        Returns:
            A `Signal` instance containing the generated waveform.
        """
        from . import filtering, mapping, utils

        # Bit-first architecture: generate bits → map to symbols
        k = int(np.log2(order))  # bits per symbol
        total_symbols = num_symbols * num_streams
        total_bits = total_symbols * k

        # Generate source bits
        bits = utils.random_bits(total_bits, seed=seed)

        # Map bits to symbols
        symbols_flat = mapping.map_bits(bits, modulation, order, dtype=dtype)

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
            symbols,
            sps=sps,
            pulse_shape=pulse_shape,
            **kwargs,
        )

        return cls(
            samples=samples,
            sampling_rate=symbol_rate * sps,
            symbol_rate=symbol_rate,
            modulation_scheme=modulation.upper(),
            modulation_order=order,
            source_bits=bits,
            source_symbols=symbols,
            pulse_shape=pulse_shape,
            **kwargs,
        )

    @classmethod
    def pam(
        cls,
        order: int,
        num_symbols: int,
        sps: int,
        symbol_rate: float,
        mode: Literal["rz", "nrz"] = "nrz",
        bipolar: bool = True,
        pulse_shape: Optional[str] = "rect",
        num_streams: int = 1,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> "Signal":
        """
        Generate a PAM baseband waveform (NRZ or RZ).

        Args:
            order: Modulation order (2, 4, 8, etc.).
            num_symbols: Number of symbols to generate.
            sps: Samples per symbol (integer).
            symbol_rate: Symbol rate in Hz.
            mode: Signaling mode ('nrz' or 'rz').
            bipolar: Whether to use bipolar (True) or unipolar (False) PAM.
            pulse_shape: Pulse shaping type ('none', 'rect', 'smoothrect', 'gaussian', 'rrc', 'rc', 'sinc'). Default: 'rect'.
            num_streams: Number of independent streams.
            seed: Random seed for bit generation.
            **kwargs: Pulse shaping parameters:
                filter_span (int): Filter span in symbols (default: 10).
                rrc_rolloff (float): Roll-off factor for RRC filter (default: 0.35).
                rc_rolloff (float): Roll-off factor for RC filter (default: 0.35).
                smoothrect_bt (float): BT product for SmoothRect filter (default: 1.0).
                gaussian_bt (float): BT product for Gaussian filter (default: 0.3).

        Returns:
            A `Signal` instance with PAM samples and metadata.
        """
        from . import filtering, utils

        if mode == "rz":
            if sps % 2 != 0:
                raise ValueError("For correct RZ duty cycle, `sps` must be even")

            p_shape = pulse_shape or "rect"
            allowed_rz_pulses = ["rect", "smoothrect"]
            if p_shape not in allowed_rz_pulses:
                raise ValueError(
                    f"Pulse shape '{p_shape}' is not allowed for RZ PAM. "
                    f"Allowed: {allowed_rz_pulses}"
                )

            # Generate symbols directly
            # Bit-first architecture: generate bits → map to symbols
            total_symbols = num_symbols * num_streams
            k = int(np.log2(order))  # bits per symbol
            bits = utils.random_bits(total_symbols * k, seed=seed)

            # Import mapping here to avoid circular imports
            from . import mapping

            symbols_flat = mapping.map_bits(bits, "ask", order, dtype=np.complex64)

            if num_streams > 1:
                symbols = symbols_flat.reshape(num_streams, num_symbols)
                bits = bits.reshape(num_streams, num_symbols * k)
            else:
                symbols = symbols_flat

            if not bipolar:
                symbols = symbols - np.min(symbols)

            # Apply RZ Pulse Shaping
            if p_shape == "rect":
                h = np.ones(int(sps / 2))
            elif p_shape == "smoothrect":
                h = filtering.smoothrect_taps(
                    sps=sps,
                    span=kwargs.get("filter_span", 10),
                    bt=kwargs.get("smoothrect_bt", 1.0),
                    pulse_width=0.5,
                )

            if is_cupy_available():
                symbols = to_device(symbols, "gpu")
                bits = to_device(bits, "gpu")
                h = to_device(h, "gpu")

            _, xp, sp = dispatch(symbols)

            # RZ hardcoded to 0.5 pulse width
            from . import multirate

            samples = utils.normalize(
                multirate.polyphase_resample(symbols, int(sps), 1, window=h, axis=-1),
                "max_amplitude",
            )

            return cls(
                samples=samples,
                sampling_rate=symbol_rate * sps,
                symbol_rate=symbol_rate,
                modulation_scheme=f"RZ-PAM{'-BIPOL' if bipolar else '-UNIPOL'}",
                modulation_order=order,
                source_bits=bits,
                source_symbols=symbols,
                pulse_shape=p_shape,
                **kwargs,
            )
        else:  # nrz
            p_shape = pulse_shape or "rect"
            sig = cls.generate(
                modulation="ask",
                order=order,
                num_symbols=num_symbols,
                sps=sps,
                symbol_rate=symbol_rate,
                pulse_shape=p_shape,
                num_streams=num_streams,
                seed=seed,
                **kwargs,
            )
            if not bipolar:
                xp = sig.xp
                sig.samples = sig.samples - xp.min(sig.samples)
                sig.samples = utils.normalize(sig.samples, "max_amplitude")

            sig.modulation_scheme = f"PAM{'-BIPOL' if bipolar else '-UNIPOL'}"
            return sig

    @classmethod
    def psk(
        cls,
        order: int,
        num_symbols: int,
        sps: float,
        symbol_rate: float,
        pulse_shape: str = "rrc",
        num_streams: int = 1,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> "Signal":
        """
        Generate a PSK baseband waveform.

        Args:
            order: Modulation order.
            num_symbols: Number of symbols to generate.
            sps: Samples per symbol.
            symbol_rate: Symbol rate in Hz.
            pulse_shape: Pulse shaping type ('none', 'rect', 'smoothrect', 'gaussian', 'rrc', 'rc', 'sinc'). Default: 'rrc'.
            num_streams: Number of independent streams.
            seed: Random seed for bit generation.
            **kwargs: Pulse shaping parameters:
                filter_span (int): Filter span in symbols (default: 10).
                rrc_rolloff (float): Roll-off factor for RRC filter (default: 0.35).
                rc_rolloff (float): Roll-off factor for RC filter (default: 0.35).
                smoothrect_bt (float): BT product for SmoothRect filter (default: 1.0).
                gaussian_bt (float): BT product for Gaussian filter (default: 0.3).

        Returns:
            A `Signal` instance with the PSK waveform.
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
            **kwargs,
        )

    @classmethod
    def qam(
        cls,
        order: int,
        num_symbols: int,
        sps: float,
        symbol_rate: float,
        pulse_shape: str = "rrc",
        num_streams: int = 1,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> "Signal":
        """
        Generate a QAM baseband waveform.

        Args:
            order: Modulation order.
            num_symbols: Number of symbols to generate.
            sps: Samples per symbol.
            symbol_rate: Symbol rate in Hz.
            pulse_shape: Pulse shaping type ('none', 'rect', 'smoothrect', 'gaussian', 'rrc', 'rc', 'sinc'). Default: 'rrc'.
            num_streams: Number of independent streams.
            seed: Random seed for bit generation.
            **kwargs: Pulse shaping parameters:
                filter_span (int): Filter span in symbols (default: 10).
                rrc_rolloff (float): Roll-off factor for RRC filter (default: 0.35).
                rc_rolloff (float): Roll-off factor for RC filter (default: 0.35).
                smoothrect_bt (float): BT product for SmoothRect filter (default: 1.0).
                gaussian_bt (float): BT product for Gaussian filter (default: 0.3).

        Returns:
            A `Signal` instance with the QAM waveform.
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
            **kwargs,
        )

    # =========================================================================
    # Metrics Methods - Signal-centric interface for quality assessment
    # =========================================================================

    def evm(
        self,
        reference_symbols: Optional[ArrayType] = None,
    ) -> Tuple[float, float]:
        """
        Compute Error Vector Magnitude comparing current symbols to reference.

        Uses source_symbols as reference by default, enabling simple workflow:
            sig = Signal.qam(16, 1000, ...)  # Has source_symbols
            noisy = add_awgn(sig, esn0_db=15)
            evm_pct, evm_db = noisy.evm()  # Uses source_symbols as ref

        Args:
            reference_symbols: Optional explicit reference. If None, uses
                source_symbols attribute.

        Returns:
            (evm_percent, evm_db): EVM as percentage and in decibels.

        Raises:
            ValueError: If no reference available (no source_symbols and no
                explicit reference provided).
        """
        from . import metrics

        ref = reference_symbols
        if ref is None:
            if self.source_symbols is None:
                raise ValueError(
                    "No reference available. Either set source_symbols or provide "
                    "reference_symbols argument."
                )
            ref = self.source_symbols

        # Get symbols at symbol rate (downsample if needed)
        rx_symbols = self._get_symbol_rate_samples()

        return metrics.evm(rx_symbols, ref)

    def snr_estimate(
        self,
        reference_symbols: Optional[ArrayType] = None,
    ) -> float:
        """
        Estimate SNR using data-aided method.

        Uses source_symbols as reference by default.

        Args:
            reference_symbols: Optional explicit reference. If None, uses
                source_symbols attribute.

        Returns:
            Estimated SNR in dB.

        Raises:
            ValueError: If no reference available.
        """
        from . import metrics

        ref = reference_symbols
        if ref is None:
            if self.source_symbols is None:
                raise ValueError(
                    "No reference available. Either set source_symbols or provide "
                    "reference_symbols argument."
                )
            ref = self.source_symbols

        rx_symbols = self._get_symbol_rate_samples()

        return metrics.snr_estimate(rx_symbols, ref)

    def ber(
        self,
        reference_bits: Optional[ArrayType] = None,
        noise_var: Optional[float] = None,
    ) -> float:
        """
        Compute Bit Error Rate.

        If samples need demapping, performs hard decision demapping first.
        Uses source_bits as reference by default.

        Args:
            reference_bits: Optional explicit reference bits. If None, uses
                source_bits attribute.
            noise_var: Noise variance for soft demapping. If None, uses
                hard demapping.

        Returns:
            Bit error rate (0 to 1).

        Raises:
            ValueError: If no reference bits available or modulation info missing.
        """
        from . import metrics

        ref = reference_bits
        if ref is None:
            if self.source_bits is None:
                raise ValueError(
                    "No reference bits available. Either set source_bits or provide "
                    "reference_bits argument."
                )
            ref = self.source_bits

        # Demap to get recovered bits
        rx_bits = self.demap(hard=True)

        return metrics.ber(rx_bits, ref)

    def demap(
        self,
        hard: bool = True,
        noise_var: Optional[float] = None,
        method: str = "maxlog",
    ) -> ArrayType:
        """
        Demap symbols to bits (hard or soft decision).

        Args:
            hard: If True, returns hard bit decisions (0/1).
                  If False, returns LLRs (soft decision).
            noise_var: Noise variance for soft demapping. Required if hard=False.
            method: LLR method for soft demapping ('maxlog' or 'exact').

        Returns:
            Hard demapping: Bit array (0s and 1s).
            Soft demapping: LLR array (positive = bit 0 likely).

        Raises:
            ValueError: If modulation info is missing or noise_var not provided
                for soft demapping.
        """
        from .mapping import demap_symbols, demap_symbols_soft

        if self.modulation_scheme is None or self.modulation_order is None:
            raise ValueError("Modulation scheme and order required for demapping.")

        rx_symbols = self._get_symbol_rate_samples()

        if hard:
            return demap_symbols(
                rx_symbols, self.modulation_scheme, self.modulation_order
            )
        else:
            if noise_var is None:
                raise ValueError("noise_var required for soft demapping.")
            return demap_symbols_soft(
                rx_symbols,
                self.modulation_scheme,
                self.modulation_order,
                noise_var,
                method=method,
            )

    def _get_symbol_rate_samples(self) -> ArrayType:
        """Get samples at symbol rate (1 sps), downsampling if needed."""
        sps = self.sps
        if sps is None or sps <= 1:
            # Already at symbol rate
            return self.samples

        # Downsample by taking every sps-th sample
        sps_int = int(round(sps))
        if self.samples.ndim == 2:
            # MIMO: downsample each stream, shape (num_streams, num_symbols)
            return self.samples[:, ::sps_int]
        return self.samples[::sps_int]


class Preamble(BaseModel):
    """
    Structured preamble container for frame synchronization sequences.

    Uses bit-first architecture: bits are the primary representation,
    symbols are derived automatically via mapping.

    Attributes:
        bits: Source bits (primary representation).
        symbols: Mapped symbols (derived from bits).
        modulation_scheme: Modulation type used ('PSK', 'QAM', 'ASK').
        modulation_order: Modulation order (2, 4, 16, etc.).
        sequence_type: Type of preamble sequence ('custom', 'barker', 'zc').
        dtype: Data type for symbols (default: np.complex64).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    bits: Any
    symbols: Optional[Any] = None
    modulation_scheme: str = "PSK"
    modulation_order: int = 2
    sequence_type: str = "custom"
    dtype: Any = np.complex64

    @field_validator("bits", mode="before")
    @classmethod
    def validate_bits(cls, v: Any) -> Any:
        """Validates and coerces the bits input into a backend-compatible array."""
        return utils.validate_array(v, name="preamble_bits")

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization hook.
        Maps bits to symbols if symbols not provided.
        Moves data to GPU if available.
        """
        from . import mapping

        # Map bits to symbols if not provided
        if self.symbols is None and self.bits is not None:
            self.symbols = mapping.map_bits(
                self.bits,
                self.modulation_scheme.lower(),
                self.modulation_order,
                dtype=self.dtype,
            )

        # Move to GPU if available
        if is_cupy_available():
            if self.bits is not None:
                self.bits = to_device(self.bits, "gpu")
            if self.symbols is not None:
                self.symbols = to_device(self.symbols, "gpu")

    @property
    def num_symbols(self) -> int:
        """Number of symbols in the preamble."""
        if self.symbols is None:
            return 0
        return int(self.symbols.size)

    @property
    def num_bits(self) -> int:
        """Number of bits in the preamble."""
        if self.bits is None:
            return 0
        return int(self.bits.size)

    def to_waveform(
        self,
        sps: int,
        symbol_rate: float,
        pulse_shape: str = "rrc",
        **kwargs: Any,
    ) -> Signal:
        """
        Generate a preamble waveform as a Signal with clear metadata.

        Args:
            sps: Samples per symbol.
            symbol_rate: Symbol rate in Hz.
            pulse_shape: Pulse shaping type ('rect', 'rrc', etc.).
            **kwargs: Additional pulse shaping parameters.

        Returns:
            A Signal object with modulation_scheme marked as 'PREAMBLE-{scheme}'.
        """
        from .filtering import shape_pulse

        samples = shape_pulse(
            self.symbols,
            sps=sps,
            pulse_shape=pulse_shape,
            **kwargs,
        )

        return Signal(
            samples=samples,
            sampling_rate=symbol_rate * sps,
            symbol_rate=symbol_rate,
            modulation_scheme=f"PREAMBLE-{self.modulation_scheme}",
            modulation_order=self.modulation_order,
            source_bits=self.bits,
            source_symbols=self.symbols,
            pulse_shape=pulse_shape,
            **kwargs,
        )


class SingleCarrierFrame(BaseModel):
    """
    Represents a structured single-carrier frame with preamble, pilots, and payload.

    This class provides a high-level abstraction for constructing frames commonly
    used in digital communication systems, supporting various pilot patterns
    (none, block, comb) and guard intervals (zero-padding or cyclic prefix).
    Also supports MIMO frames (spatial multiplexing).

    Attributes:
        payload_len: Number of data SYMBOLS in the payload per stream.
        payload_mod_scheme: Modulation scheme for the payload (e.g., "PSK", "QAM").
        payload_mod_order: Modulation order for the payload (e.g., 2, 4, 16).
        payload_seed: Random seed for payload bit generation.
        preamble: Structured Preamble object containing sync sequence.
        pilot_pattern: The arrangement of pilot symbols ("none", "block", "comb").
        pilot_period: Periodicity of pilots (for "comb" and "block").
        pilot_block_len: Length of the pilot block (for "block" pattern) in SYMBOLS.
        pilot_seed: Random seed for pilot symbol generation.
        pilot_mod_scheme: Modulation scheme for pilots.
        pilot_mod_order: Modulation order for pilots.
        guard_type: Type of guard interval ("zero" for padding, "cp" for cyclic prefix).
        guard_len: Length of the guard interval in SYMBOLS.
        symbol_rate: The symbol rate of the frame in Hz.
        num_streams: Number of independent spatial streams (default: 1).
        dtype: Data type for symbol arrays (default: np.complex64).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    payload_len: int = 1000
    payload_seed: int = 42
    payload_mod_scheme: str = "PSK"
    payload_mod_order: int = 4

    preamble: Optional[Preamble] = None

    pilot_pattern: Literal["none", "block", "comb"] = "none"
    pilot_period: int = 0
    pilot_block_len: int = 0
    pilot_seed: int = 1337
    pilot_mod_scheme: str = "PSK"
    pilot_mod_order: int = 4

    guard_type: Literal["zero", "cp"] = "zero"
    guard_len: int = 0

    symbol_rate: float = Field(..., gt=0)
    num_streams: int = Field(default=1, ge=1)
    dtype: Any = Field(default=np.complex64)

    # Cached bit-first data (not serialized)
    _payload_bits: Optional[Any] = PrivateAttr(default=None)
    _payload_symbols: Optional[Any] = PrivateAttr(default=None)
    _pilot_bits: Optional[Any] = PrivateAttr(default=None)
    _pilot_symbols: Optional[Any] = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization hook.
        Preamble handles its own GPU transfer.
        """
        pass  # Preamble handles its own device placement

    def _generate_pilot_mask(self) -> Tuple[ArrayType, int]:
        """
        Pre-calculates the pilot placement mask and the total frame body length.

        Returns:
            A tuple containing:
                - pilot_mask: A boolean array where True indicates a pilot symbol.
                - body_length: The total number of symbols in the frame body.
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
        """Generate and cache payload bits → symbols (bit-first architecture)."""
        if self._payload_bits is not None:
            return

        from . import mapping

        k = int(np.log2(self.payload_mod_order))
        total_symbols = self.payload_len * self.num_streams
        bits = utils.random_bits(total_symbols * k, seed=self.payload_seed)
        symbols = mapping.map_bits(
            bits, self.payload_mod_scheme, self.payload_mod_order, dtype=self.dtype
        )

        if self.num_streams > 1:
            bits = bits.reshape(self.num_streams, self.payload_len * k)
            symbols = symbols.reshape(self.num_streams, self.payload_len)

        if is_cupy_available():
            bits = to_device(bits, "gpu")
            symbols = to_device(symbols, "gpu")

        self._payload_bits = bits
        self._payload_symbols = symbols

    def _ensure_pilot_generated(self) -> None:
        """Generate and cache pilot bits → symbols (bit-first architecture)."""
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
        bits = utils.random_bits(total_pilots * k, seed=self.pilot_seed)
        symbols = mapping.map_bits(
            bits, self.pilot_mod_scheme, self.pilot_mod_order, dtype=self.dtype
        )

        if self.num_streams > 1:
            bits = bits.reshape(self.num_streams, pilot_count * k)
            symbols = symbols.reshape(self.num_streams, pilot_count)

        if is_cupy_available():
            bits = to_device(bits, "gpu")
            symbols = to_device(symbols, "gpu")

        self._pilot_bits = bits
        self._pilot_symbols = symbols

    @property
    def payload_bits(self) -> ArrayType:
        """Returns the payload bits (bit-first architecture)."""
        self._ensure_payload_generated()
        return self._payload_bits

    @property
    def payload_symbols(self) -> ArrayType:
        """Returns the mapped symbols of the payload."""
        self._ensure_payload_generated()
        return self._payload_symbols

    @property
    def pilot_bits(self) -> Optional[ArrayType]:
        """Returns the pilot bits (bit-first architecture), if any."""
        if self.pilot_pattern == "none":
            return None
        self._ensure_pilot_generated()
        return self._pilot_bits

    @property
    def pilot_symbols(self) -> Optional[ArrayType]:
        """Returns the mapped symbols used for pilots, if any."""
        if self.pilot_pattern == "none":
            return None
        self._ensure_pilot_generated()
        return self._pilot_symbols

    @property
    def body_symbols(self) -> ArrayType:
        """Returns the interleaved pilot and payload symbols (frame body)."""
        xp = cp if is_cupy_available() else np
        mask, body_length = self._generate_pilot_mask()
        if self.num_streams > 1:
            # Shape: (Channels, Time)
            body = xp.zeros((self.num_streams, body_length), dtype=self.dtype)

            if self.pilot_pattern != "none":
                body[:, mask] = self.pilot_symbols

            body[:, ~mask] = self.payload_symbols
        else:
            body = xp.zeros(body_length, dtype=self.dtype)
            if self.pilot_pattern != "none":
                body[mask] = self.pilot_symbols
            body[~mask] = self.payload_symbols

        return body

    def _assemble_symbols(self) -> ArrayType:
        """
        Assembles the symbol sequence (sps=1) without guard interval.
        Combines preamble and frame body.

        Returns:
            A concatenated array of symbols.
        """
        xp = cp if is_cupy_available() else np
        body = self.body_symbols

        # Assemble Core (Preamble + Body)
        if self.preamble is not None:
            preamble_symbols = to_device(
                self.preamble.symbols, "gpu" if is_cupy_available() else "cpu"
            )
            # Broadcast preamble if needed
            if self.num_streams > 1 and preamble_symbols.ndim == 1:
                # Need (Channels, Time)
                # (L,) -> (1, L) -> (C, L)
                preamble_symbols = xp.tile(
                    preamble_symbols[None, :], (self.num_streams, 1)
                )

            # Concatenate along time axis (-1)
            # If 1D: (L1) + (L2) -> (L1+L2)
            # If 2D: (C, L1) + (C, L2) -> (C, L1+L2)
            # axis=-1 typically works for 1D too (last axis)
            return xp.concatenate([preamble_symbols, body], axis=-1)
        else:
            return body

    def generate_waveform(
        self, sps: int = 4, pulse_shape: str = "rrc", **kwargs: Any
    ) -> Signal:
        """
        Generates a proper waveform (upsampled and shaped) for the frame.

        Args:
            sps: Samples per symbol.
            pulse_shape: Pulse shaping type ('rect', 'smoothrect', 'gaussian', 'rrc', 'rc', 'sinc').
            **kwargs: Additional pulse shaping parameters.

        Returns:
            A `Signal` object representing the shaped waveform with FrameInfo metadata.
        """
        xp = cp if is_cupy_available() else np
        from .filtering import shape_pulse

        # 1. Assemble Symbols (sps=1)
        symbols = self._assemble_symbols()

        # Determine logical/source symbols at sps=1 including guards
        source_symbols = symbols
        if self.guard_len > 0:
            if self.guard_type == "zero":
                if self.num_streams > 1:
                    zeros = xp.zeros(
                        (self.num_streams, self.guard_len), dtype=self.dtype
                    )
                else:
                    zeros = xp.zeros(self.guard_len, dtype=self.dtype)

                source_symbols = xp.concatenate([source_symbols, zeros], axis=-1)
            elif self.guard_type == "cp":
                cp_slice = source_symbols[..., -self.guard_len :]
                source_symbols = xp.concatenate([cp_slice, source_symbols], axis=-1)

        # 2. Apply pulse shaping
        samples = shape_pulse(
            symbols,
            sps=sps,
            pulse_shape=pulse_shape,
            **kwargs,
        )

        # 3. Apply Guard Interval at sample level
        if self.guard_len > 0:
            guard_len_samples = int(self.guard_len * sps)
            if self.guard_type == "zero":
                if self.num_streams > 1:
                    zeros = xp.zeros(
                        (self.num_streams, guard_len_samples), dtype=self.dtype
                    )
                else:
                    zeros = xp.zeros(guard_len_samples, dtype=self.dtype)
                samples = xp.concatenate([samples, zeros], axis=-1)
            elif self.guard_type == "cp":
                cp_slice = samples[..., -guard_len_samples:]
                samples = xp.concatenate([cp_slice, samples], axis=-1)

        # 4. Build FrameInfo metadata
        mask, _ = self._generate_pilot_mask()
        pilot_count = int(xp.sum(mask)) if self.pilot_pattern != "none" else 0

        frame_info = FrameInfo(
            preamble_len=self.preamble.num_symbols if self.preamble else 0,
            payload_len=self.payload_len,
            pilot_count=pilot_count,
            guard_len=self.guard_len,
            guard_type=self.guard_type,
            preamble_mod_scheme=self.preamble.modulation_scheme
            if self.preamble
            else None,
            preamble_mod_order=self.preamble.modulation_order
            if self.preamble
            else None,
        )

        return Signal(
            samples=samples,
            sampling_rate=self.symbol_rate * sps,
            symbol_rate=self.symbol_rate,
            modulation_scheme=self.payload_mod_scheme,
            modulation_order=self.payload_mod_order,
            source_bits=self.payload_bits,
            source_symbols=source_symbols,
            pulse_shape=pulse_shape,
            frame_info=frame_info,
            **kwargs,
        )

    def get_structure_map(
        self, unit: Literal["symbols", "samples"] = "symbols", sps: int = 1
    ) -> Dict[str, ArrayType]:
        """
        Returns a dictionary of boolean masks for each part of the frame.

        Args:
            unit: The unit of the mask length ('symbols' or 'samples').
            sps: Samples per symbol (only used if unit='samples').

        Returns:
            A dictionary with keys 'preamble', 'pilot', 'payload', 'guard'.
        """
        xp = cp if is_cupy_available() else np
        mask, body_length = self._generate_pilot_mask()
        preamble_len = self.preamble.num_symbols if self.preamble is not None else 0

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
            "pilot": pilot_bool,
            "payload": payload_bool,
            "guard": guard_bool,
        }

        if unit == "samples":
            for k in res:
                res[k] = xp.repeat(res[k], int(sps))

        return res
