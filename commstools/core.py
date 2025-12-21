"""
Core signal processing abstractions.

This module defines the primary data structures for the library:
- `Signal`: Encapsulates complex IQ samples, physical layer metadata (sampling rate, symbol rate,
  modulation scheme/order), and methods for signal analysis and transformation.
- `Frame`: A container for structured signals, allowing the assembly of signals with
  preambles, pilots, and payloads.

All core classes are built on Pydantic for validation and support both CPU (NumPy) and GPU (CuPy) backends.
"""

import types
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False


from . import utils
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


class Signal(BaseModel):
    """
    Represents a digital signal with associated physical layer metadata.

    This class serves as the primary data container for the library, encapsulating
    both the raw IQ samples and the contextual information required for digital
    signal processing (DSP) and analysis.

    Attributes:
        samples: The complex IQ samples of the signal. Can be NumPy or CuPy arrays.
        sampling_rate: The sampling rate of the signal in Hz. Must be positive.
        symbol_rate: The symbol rate (baud rate) in Hz. Used for SPS calculation.
        modulation_scheme: Identifier for the modulation format (e.g., 'QPSK', '16QAM').
        modulation_order: The number of symbols in the constellation (e.g., 4, 16).
        source_bits: Optional array of original bits that generated this signal.
        pulse_shape: Name of the pulse shaping filter applied (e.g., 'rrc', 'rect').
        pulse_params: Dictionary of parameters for the pulse shape (e.g., 'rolloff').
        spectral_domain: The signal's placement in the spectrum ('BASEBAND', 'PASSBAND', 'INTERMEDIATE').
        physical_domain: The physical layer domain ('DIG' for digital, 'RF' for radio frequency, 'OPT' for optical).
        center_frequency: The carrier or center frequency in Hz.
        digital_frequency_offset: Any applied digital frequency shift in Hz.
        metadata: Generic dictionary for additional custom attributes (e.g., frame structure info).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    samples: Any
    sampling_rate: float = Field(..., gt=0)
    symbol_rate: float = Field(..., gt=0)
    modulation_scheme: Optional[str] = None
    modulation_order: Optional[int] = None
    source_bits: Optional[Any] = None
    pulse_shape: Optional[str] = None
    pulse_params: Optional[Dict[str, Any]] = None
    spectral_domain: Literal["BASEBAND", "PASSBAND", "INTERMEDIATE"] = "BASEBAND"
    physical_domain: Literal["DIG", "RF", "OPT"] = "DIG"
    center_frequency: float = Field(default=0, ge=0)
    digital_frequency_offset: float = Field(default=0, ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("samples", mode="before")
    @classmethod
    def validate_samples(cls, v: Any) -> Any:
        # Coerce lists/tuples to numpy arrays
        if isinstance(v, (list, tuple)):
            try:
                return np.asarray(v)
            except Exception:
                raise ValueError(
                    f"Could not convert input of type {type(v)} to numpy array."
                )

        # Ensure it's something we can work with
        if not isinstance(v, (np.ndarray, getattr(cp, "ndarray", type(None)))):
            # Try to convert to numpy array to see if it's a valid array-like
            try:
                return np.asarray(v)
            except Exception:
                raise ValueError(
                    f"Unsupported samples type: {type(v)}. Must be array-like."
                )
        return v

    def model_post_init(self, __context: Any) -> None:
        # Default to GPU if available and supported
        if is_cupy_available():
            self.to("gpu")

    @property
    def xp(self) -> types.ModuleType:
        """Returns the array module (numpy or cupy) for the signal's data."""
        return get_array_module(self.samples)

    @property
    def sp(self) -> types.ModuleType:
        """Returns the scipy module (scipy or cupyx.scipy) for the signal's data."""
        return get_scipy_module(self.xp)

    @property
    def backend(self) -> str:
        """
        Returns the name of the backend ('CPU' or 'GPU').
        """
        return "GPU" if self.xp == cp else "CPU"

    @property
    def duration(self) -> float:
        """Duration of the signal in seconds."""
        return self.samples.shape[0] / self.sampling_rate

    @property
    def sps(self) -> float:
        """Samples per symbol."""
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
        """Returns the time vector associated with the signal samples."""
        return self.xp.arange(0, self.samples.shape[0]) / self.sampling_rate

    def source_symbols(self) -> ArrayType:
        """
        Remaps source_bits to symbols using the current modulation scheme and order.
        This provides a memory-efficient way to access symbols if only bits are stored.

        Returns:
            The remapped symbols as an array.
        """
        logger.debug(f"Remapping bits to symbols ({self.modulation_scheme}).")
        if self.source_bits is None:
            raise ValueError("No source bits available for remapping.")
        if self.modulation_order is None:
            raise ValueError("Modulation order must be defined for remapping.")

        from . import mapping

        # Ensure we use bits on the right backend for the mapping
        # mapping.map_bits currently uses numpy internally but we might want to handle dispatch
        return mapping.map_bits(
            self.source_bits,
            modulation=self.modulation_scheme.split("-")[-1].lower(),
            order=self.modulation_order,
        )

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
        """
        if self.xp.iscomplexobj(self.samples):
            f, Pxx = self.sp.signal.welch(
                self.samples,
                fs=self.sampling_rate,
                nperseg=nperseg,
                detrend=detrend,
                average=average,
                return_onesided=False,
            )
            f = self.xp.fft.fftshift(f)
            Pxx = self.xp.fft.fftshift(Pxx)
            return f, Pxx
        else:
            return self.sp.signal.welch(
                self.samples,
                fs=self.sampling_rate,
                nperseg=nperseg,
                detrend=detrend,
                average=average,
                return_onesided=True,
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

    @staticmethod
    def _format_si(value: Optional[float], unit: str = "Hz") -> str:
        return utils.format_si(value, unit)

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
                "Samples Shape",
            ],
            "Value": [
                self.spectral_domain,
                self.physical_domain,
                self.modulation_scheme,
                str(self.modulation_order) if self.modulation_order else "None",
                self._format_si(self.symbol_rate, "Baud"),
                self._format_si(
                    self.symbol_rate * np.log2(self.modulation_order), "bps"
                )
                if self.modulation_order
                else "None",
                self._format_si(self.sampling_rate, "Hz"),
                f"{self.sps:.2f}",
                self.pulse_shape.upper() if self.pulse_shape else "None",
                self._format_si(self.duration, "s"),
                self._format_si(self.center_frequency, "Hz"),
                self._format_si(self.digital_frequency_offset, "Hz"),
                self.backend.upper(),
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
        show: bool = False,
        **kwargs: Any,
    ) -> Optional[Tuple[Any, Any]]:
        """
        Plot the eye diagram of the signal.

        Args:
            ax: Optional matplotlib axis (or list of axes for complex signals) to plot on.
            type: Type of plot ('hist' or 'line').
            title: Title of the plot.
            show: Whether to call plt.show().
            **kwargs: Additional plotting arguments.

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

        self.samples = multirate.upsample(self.samples, factor)
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
            self.samples, factor, filter_type=filter_type, **kwargs
        )
        self.sampling_rate = self.sampling_rate / factor
        return self

    def resample(self, up: int, down: int) -> "Signal":
        """
        Resample the signal by a rational factor.

        Args:
            up: Upsampling factor.
            down: Downsampling factor.

        Returns:
            self
        """
        from . import multirate

        self.samples = multirate.resample(self.samples, up, down)
        self.sampling_rate = self.sampling_rate * up / down
        return self

    def fir_filter(self, taps: ArrayType) -> "Signal":
        """
        Apply FIR filter to the signal.

        Args:
            taps: Filter taps.
            mode: Convolution mode.

        Returns:
            self
        """
        from . import filtering

        self.samples = filtering.fir_filter(self.samples, taps)
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

        params = self.pulse_params if self.pulse_params else {}

        # Default span if not present
        span = params.get("filter_span", 10)
        pulse_width = params.get("pulse_width", 1.0)

        # Generate taps using filtering module (returns default numpy usually)
        if self.pulse_shape == "rect":
            taps = np.ones(int(self.sps * pulse_width))
        elif self.pulse_shape == "smoothrect":
            taps = filtering.smoothrect_taps(
                sps=self.sps,
                span=span,
                bt=params.get("smoothrect_bt", 1.0),
                pulse_width=pulse_width,
            )
        elif self.pulse_shape == "gaussian":
            taps = filtering.gaussian_taps(
                sps=self.sps,
                span=span,
                bt=params.get("gaussian_bt", 0.3),
            )
        elif self.pulse_shape == "rrc":
            taps = filtering.rrc_taps(
                sps=self.sps,
                span=span,
                rolloff=params.get("rrc_rolloff", 0.35),
            )
        elif self.pulse_shape == "rc":
            taps = filtering.rc_taps(
                sps=self.sps,
                span=span,
                rolloff=params.get("rc_rolloff", 0.35),
            )
        else:
            raise ValueError(f"Unknown pulse shape: {self.pulse_shape}")

        # Ensure taps are on the correct device
        return to_device(taps, self.backend)

    def matched_filter(
        self,
        taps_normalization: str = "unity_gain",
        normalize_output: bool = False,
    ) -> "Signal":
        """
        Apply matched filter to the signal.
        The filter taps are calculated automatically based on the signal properties.

        Args:
            taps_normalization: Normalization to apply to the matched filter taps.
                                Options: 'unity_gain', 'unit_energy'.
            normalize_output: If True, normalizes the output samples to have a maximum
                              absolute value of 1.0.

        Returns:
            self
        """
        from . import filtering

        try:
            pulse_taps = self.shaping_filter_taps()
        except ValueError as e:
            logger.error(f"Cannot apply matched filter: {e}")
            return self

        self.samples = filtering.matched_filter(
            self.samples,
            pulse_taps=pulse_taps,
            taps_normalization=taps_normalization,
            normalize_output=normalize_output,
        )
        return self

    def copy(self) -> "Signal":
        """
        Creates a deep copy of the Signal object.

        Returns:
            A new Signal object with copied data.
        """
        return self.model_copy(deep=True)


class Frame(BaseModel):
    """
    Structured signal container for managing frame-based communication.

    The `Frame` class facilitates the construction of structured waveforms by
    grouping a preamble, pilots, and payload. It provides a mechanism to
    assemble these components into a single, continuous `Signal` object.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    preamble: Optional[Signal] = None
    pilots: Optional[Signal] = None
    payload: Optional[Signal] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def assemble(self) -> Signal:
        """
        Assembles the parts into a single continuous Signal.
        Currently implements simple time-multiplexed concatenation: Preamble -> Payload (with Pilots if applicable).
        Note: Complex pilot insertion patterns should be handled by specific methods.

        Returns:
            A new Signal object containing the combined samples and identifying metadata.
        """
        logger.info("Assembling frame components.")
        parts = []
        if self.preamble:
            parts.append(self.preamble.samples)
        if self.payload:
            parts.append(self.payload.samples)

        if not parts:
            raise ValueError("Frame is empty; cannot assemble.")

        # Assume all parts share same properties (sampling_rate, etc.) from payload or first part
        ref = self.payload or self.preamble
        xp = ref.xp

        combined_samples = xp.concatenate(parts, axis=0)

        frame_metadata = {
            "is_frame": True,
            "has_preamble": self.preamble is not None,
            "has_pilots": self.pilots is not None,
            **self.metadata,
        }

        return Signal(
            samples=combined_samples,
            sampling_rate=ref.sampling_rate,
            symbol_rate=ref.symbol_rate,
            modulation_scheme=ref.modulation_scheme,
            modulation_order=ref.modulation_order,
            pulse_shape=ref.pulse_shape,
            pulse_params=ref.pulse_params,
            spectral_domain=ref.spectral_domain,
            physical_domain=ref.physical_domain,
            center_frequency=ref.center_frequency,
            metadata=frame_metadata,
        )
