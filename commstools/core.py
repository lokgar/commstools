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
        spectral_domain: The signal's placement in the spectrum ('BASEBAND', 'PASSBAND', 'INTERMEDIATE').
        physical_domain: The physical layer domain ('DIG' for digital, 'RF' for radio frequency, 'OPT' for optical).
        center_frequency: The carrier or center frequency in Hz.
        digital_frequency_offset: Any applied digital frequency shift in Hz.
        filter_span: Filter span in symbols for pulse shaping.
        rrc_rolloff: Roll-off factor for RRC filter.
        rc_rolloff: Roll-off factor for RC filter.
        smoothrect_bt: BT product for SmoothRect filter.
        gaussian_bt: BT product for Gaussian filter.
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
    pulse_shape: Optional[str] = None
    spectral_domain: Literal["BASEBAND", "PASSBAND", "INTERMEDIATE"] = "BASEBAND"
    physical_domain: Literal["DIG", "RF", "OPT"] = "DIG"
    center_frequency: float = Field(default=0, ge=0)
    digital_frequency_offset: float = Field(default=0, ge=0)

    # Pulse shaping parameters
    filter_span: int = 10
    rrc_rolloff: float = 0.35
    rc_rolloff: float = 0.35
    gaussian_bt: float = 0.3
    smoothrect_bt: float = 1.0

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

    @classmethod
    def create(
        cls,
        modulation: str,
        order: int,
        num_symbols: int,
        sps: float,
        symbol_rate: float,
        pulse_shape: str = "rrc",
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> "Signal":
        """
        Create a baseband Signal with specified parameters.

        Args:
            modulation: Modulation scheme ('psk', 'qam', 'ask').
            order: Modulation order.
            num_symbols: Number of symbols to generate.
            sps: Samples per symbol.
            symbol_rate: Symbol rate in Hz.
            pulse_shape: Pulse shaping type ('none', 'rect', 'smoothrect', 'gaussian', 'rrc', 'rc', 'sinc').
            seed: Random seed for bit generation.
            **kwargs: Pulse shaping parameters:
                filter_span (int): Filter span in symbols (default: 10).
                rrc_rolloff (float): Roll-off factor for RRC filter (default: 0.35).
                rc_rolloff (float): Roll-off factor for RC filter (default: 0.35).
                smoothrect_bt (float): BT product for SmoothRect filter (default: 1.0).
                gaussian_bt (float): BT product for Gaussian filter (default: 0.3).

        Returns:
            A `Signal` instance containing the generated waveform.
        """
        from . import filtering, mapping, sequences

        # calculate number of bits
        k = int(np.log2(order))
        num_bits = num_symbols * k

        # Generate random bits (NumPy)
        bits = sequences.random_bits(num_bits, seed=seed)

        # Map to symbols (NumPy)
        symbols = mapping.map_bits(bits, modulation=modulation, order=order)

        # Apply pulse shaping
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
        pulse_shape: Optional[str] = None,
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
            pulse_shape: Pulse shaping type ('none', 'rect', 'smoothrect', 'gaussian', 'rrc', 'rc', 'sinc').
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
        from . import filtering, mapping, sequences
        import scipy.signal

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

            # Generate symbols
            k = int(np.log2(order))
            if 2**k != order:
                raise ValueError(f"PAM order must be power of 2, got {order}")
            num_bits = num_symbols * k
            bits = sequences.random_bits(num_bits, seed=seed)
            symbols = mapping.map_bits(bits, modulation="ask", order=order)

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

            # RZ hardcoded to 0.5 pulse width
            samples = utils.normalize(
                scipy.signal.resample_poly(symbols, int(sps), 1, window=h),
                "max_amplitude",
            )

            return cls(
                samples=samples,
                sampling_rate=symbol_rate * sps,
                symbol_rate=symbol_rate,
                modulation_scheme=f"RZ-PAM{'-BIPOL' if bipolar else '-UNIPOL'}",
                modulation_order=order,
                source_bits=bits,
                pulse_shape=p_shape,
                **kwargs,
            )
        else:  # nrz
            p_shape = pulse_shape or "rect"
            sig = cls.create(
                modulation="ask",
                order=order,
                num_symbols=num_symbols,
                sps=sps,
                symbol_rate=symbol_rate,
                pulse_shape=p_shape,
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
            pulse_shape: Pulse shaping type ('none', 'rect', 'smoothrect', 'gaussian', 'rrc', 'rc', 'sinc').
            seed: Random seed for bit generation.
            **kwargs: Pulse shaping parameters:
                filter_span (int): Filter span in symbols (default: 10).
                rrc_rolloff (float): Roll-off factor for RRC filter (default: 0.35).
                rc_rolloff (float): Roll-off factor for RC filter (default: 0.35).
                smoothrect_bt (float): BT product for SmoothRect filter (default: 1.0).
                gaussian_bt (float): BT product for Gaussian filter (default: 0.3).
        """
        return cls.create(
            modulation="psk",
            order=order,
            num_symbols=num_symbols,
            sps=sps,
            symbol_rate=symbol_rate,
            pulse_shape=pulse_shape,
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
            pulse_shape: Pulse shaping type ('none', 'rect', 'smoothrect', 'gaussian', 'rrc', 'rc', 'sinc').
            seed: Random seed for bit generation.
            **kwargs: Pulse shaping parameters:
                filter_span (int): Filter span in symbols (default: 10).
                rrc_rolloff (float): Roll-off factor for RRC filter (default: 0.35).
                rc_rolloff (float): Roll-off factor for RC filter (default: 0.35).
                smoothrect_bt (float): BT product for SmoothRect filter (default: 1.0).
                gaussian_bt (float): BT product for Gaussian filter (default: 0.3).
        """
        return cls.create(
            modulation="qam",
            order=order,
            num_symbols=num_symbols,
            sps=sps,
            symbol_rate=symbol_rate,
            pulse_shape=pulse_shape,
            seed=seed,
            **kwargs,
        )


class Frame(BaseModel):
    """
    Structured signal container for managing frame-based communication.

    The `Frame` class facilitates the construction of structured waveforms by
    grouping a preamble, pilots, and payload. It provides a mechanism to
    assemble these components into a single, continuous `Signal` object.

    Attributes:
        preamble: Optional array of complex symbols for the preamble.
        payload_seed: Random seed for generating payload bits.
        payload_len_symbols: Number of payload symbols to generate.
        payload_modulation: Modulation scheme for the payload (e.g., 'QPSK').
        pilot_pattern: Pilot insertion pattern ('none', 'block', 'comb').
        pilot_period_symbols: Period of pilot insertion (for 'comb' or 'block').
        pilot_len_symbols: Number of pilots per insertion event (for 'block').
        pilot_seed: Random seed for generating pilot symbols.
        pilot_modulation: Modulation scheme for the pilots.
        guard_period: Number of zero samples to append at the end of the frame.
        symbol_rate: Symbol rate of the constructed signal.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    preamble: Optional[Any] = None
    payload_seed: int = 42
    payload_len_symbols: int = 1000
    payload_modulation: str = "QPSK"
    payload_mod_order: Optional[int] = None
    pilot_pattern: Literal["none", "block", "comb"] = "none"
    pilot_period_symbols: int = 0
    pilot_len_symbols: int = 0
    pilot_seed: int = 1337
    pilot_modulation: str = "QPSK"
    pilot_mod_order: Optional[int] = None
    guard_period: int = 0
    symbol_rate: float = Field(..., gt=0)

    @field_validator("preamble", mode="before")
    @classmethod
    def validate_arrays(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, (list, tuple)):
            return np.asarray(v)
        return v

    def compose(self, device: Optional[str] = None) -> Signal:
        """
        Assembles the parts into a single continuous Signal (sps=1).

        Args:
            device: Target device ('CPU' or 'GPU'). If None, tries to infer from preamble
                    or defaults to available backend.

        Returns:
            A new Signal object containing the combined symbol sequence.
        """
        logger.info("Composing frame symbols.")
        from . import mapping, sequences  # Import locally to avoid circular deps
        from .backend import get_array_module, to_device, is_cupy_available

        # Determine target device logic
        # 1. Explicit argument
        # 2. Preamble location (if available)
        # 3. Default (GPU if available)
        target_device = "cpu"
        if device:
            target_device = device.lower()
        elif self.preamble is not None:
            # Check if preamble is cupy array
            xp = get_array_module(self.preamble)
            if xp.__name__ == "cupy":
                target_device = "gpu"
        elif is_cupy_available():
            target_device = "gpu"

        # Helper to analyze modulation
        def analyze_mod(scheme: str, explicit_order: Optional[int]):
            s = scheme.lower()
            if "psk" in s or "bpsk" in s or "qpsk" in s:
                mod_type = "psk"
            elif "ask" in s:
                mod_type = "ask"
            else:
                mod_type = "qam"

            if explicit_order:
                return mod_type, explicit_order

            # Simple order guessing
            if "bpsk" in s:
                order = 2
            elif "qpsk" in s or "4qam" in s:
                order = 4
            elif "8qam" in s:
                order = 8
            elif "16qam" in s:
                order = 16
            elif "64qam" in s:
                order = 64
            elif "256qam" in s:
                order = 256
            else:
                logger.warning(
                    f"Could not determine order for {scheme}, defaulting to QPSK (4)."
                )
                order = 4
            return mod_type, order

        # 1. Generate Payload
        pay_type, pay_order = analyze_mod(
            self.payload_modulation, self.payload_mod_order
        )
        bits_per_sym_pay = int(np.log2(pay_order))
        num_pay_bits = self.payload_len_symbols * bits_per_sym_pay

        # Generation typically on CPU for reproducibility/sequences module
        pay_bits = sequences.random_bits(num_pay_bits, seed=self.payload_seed)
        # Move bits to target device for mapping (mapping can happen on GPU)
        pay_bits = to_device(pay_bits, target_device)

        pay_syms = mapping.map_bits(pay_bits, pay_type, pay_order)

        # 2. Generate Pilots
        max_pilots = 0
        if self.pilot_pattern == "comb" and self.pilot_period_symbols > 0:
            max_pilots = self.payload_len_symbols // 1 + 10
        elif self.pilot_pattern == "block" and self.pilot_period_symbols > 0:
            num_blocks = self.payload_len_symbols // self.pilot_period_symbols + 2
            max_pilots = num_blocks * self.pilot_len_symbols

        plt_type, plt_order = analyze_mod(self.pilot_modulation, self.pilot_mod_order)

        if max_pilots > 0:
            bits_per_sym_plt = int(np.log2(plt_order))
            num_plt_bits = max_pilots * bits_per_sym_plt
            plt_bits = sequences.random_bits(num_plt_bits, seed=self.pilot_seed)
            plt_bits = to_device(plt_bits, target_device)
            all_pilot_syms = mapping.map_bits(plt_bits, plt_type, plt_order)
        else:
            # Empty array on correct device
            xp = get_array_module(pay_syms)  # Should check pay_syms backend
            all_pilot_syms = xp.array([], dtype=np.complex64)

        # 3. Construct Body
        # We need to use valid array op references (xp)
        xp = get_array_module(pay_syms)

        # Optimized Loop
        body_chunks = []
        current_len = 0
        pay_idx = 0
        plt_idx = 0

        while pay_idx < len(pay_syms):
            # Check pilot condition
            insert_pilot = False
            pilots_n = 0

            if self.pilot_pattern == "comb":
                if self.pilot_period_symbols > 0 and (
                    current_len % self.pilot_period_symbols == 0
                ):
                    insert_pilot = True
                    pilots_n = 1
            elif self.pilot_pattern == "block":
                if self.pilot_period_symbols > 0 and (
                    current_len % self.pilot_period_symbols == 0
                ):
                    insert_pilot = True
                    pilots_n = self.pilot_len_symbols

            if insert_pilot and plt_idx + pilots_n <= len(all_pilot_syms):
                chunk = all_pilot_syms[plt_idx : plt_idx + pilots_n]
                body_chunks.append(chunk)
                plt_idx += pilots_n
                current_len += pilots_n
            else:
                # Add payload
                # Optimization: can we add a chunk of payload?
                # If comb with period P=10, we insert pilot, then 9 payload symbols?
                # "Pilot every 10 symbols" usually means P D D D D D D D D D
                # So at 0 (len=0): Pilot.
                # then 1..9: Payload.
                # then 10: Pilot.

                # If we just inserted a pilot, we are at len = pilots_n.
                # Next pilot at len + period? No, at K * period.

                # If we didn't insert pilot, we insert 1 payload symbol.
                # To avoid symbol-by-symbol list append (slow), ideally we slice.
                # But simplicity first.
                chunk = pay_syms[pay_idx : pay_idx + 1]
                body_chunks.append(chunk)
                pay_idx += 1
                current_len += 1

        if body_chunks:
            body = xp.concatenate(body_chunks)
        else:
            body = xp.array([], dtype=np.complex64)

        # 4. Assemble Full Frame
        parts = []
        if self.preamble is not None:
            # Ensure preamble is on target device
            pre_dev = to_device(self.preamble, target_device)
            parts.append(pre_dev)

        parts.append(body)

        if self.guard_period > 0:
            parts.append(xp.zeros(self.guard_period, dtype=np.complex64))

        combined_samples = xp.concatenate(parts)

        # 5. Create Signal
        return Signal(
            samples=combined_samples,
            sampling_rate=self.symbol_rate,  # sps=1
            symbol_rate=self.symbol_rate,
            modulation_scheme=self.payload_modulation,
            modulation_order=pay_order,
            pulse_shape=None,
            spectral_domain="BASEBAND",
            physical_domain="DIG",
        )

    def get_waveform(
        self,
        pulse_shape: str = "rrc",
        sps: int = 4,
        **kwargs: Any,
    ) -> Signal:
        """
        Generates the time-domain waveform for the frame.

        This method compiles the frame symbols and applies pulse shaping
        using the `filtering` module.

        Args:
            pulse_shape: Pulse shape type (e.g., 'rrc', 'root_raised_cosine').
            sps: Samples per symbol (upsampling factor).
            **kwargs: Pulse shaping parameters:
                filter_span (int): Filter span in symbols (default: 10).
                rrc_rolloff (float): Roll-off factor for RRC filter (default: 0.35).
                rc_rolloff (float): Roll-off factor for RC filter (default: 0.35).
                smoothrect_bt (float): BT product for SmoothRect filter (default: 1.0).
                gaussian_bt (float): BT product for Gaussian filter (default: 0.3).

        Returns:
            A Signal object representing the waveform (sps > 1).
        """
        from . import filtering

        # 1. Compose symbols
        sig = self.compose()

        # 2. Update Signal properties
        sig.pulse_shape = pulse_shape
        for k, v in kwargs.items():
            setattr(sig, k, v)

        # 3. Apply Pulse Shaping
        shaped_samples = filtering.shape_pulse(
            sig.samples,
            sps=sps,
            pulse_shape=pulse_shape,
            **kwargs,
        )

        # Update signal samples and rate
        sig.samples = shaped_samples
        sig.sampling_rate = sig.symbol_rate * sps

        return sig

    def get_map(self) -> Dict[str, Any]:
        """
        Returns a map of the frame structure indices.

        Returns:
             Dictionary containing indices/slices for preamble, pilots, and payload.
        """
        # We need to simulate the composition to get indices
        # This is a bit inefficient to duplicate logic, but safer than estimating.
        # TODO: Refactor compose to produce map as artifact?

        preamble_len = len(self.preamble) if self.preamble is not None else 0

        # Simulate Body Construction
        # We only need lengths

        pilot_indices = []
        payload_indices = []

        curr_idx = preamble_len
        pay_done = 0

        # We need to know loop limits. The compose loop runs until payload is exhausted.
        # Let's just run a dry loop.

        sim_body_len = 0
        while pay_done < self.payload_len_symbols:
            insert_pilot = False
            pilots_to_insert = 0

            if self.pilot_pattern == "comb":
                if self.pilot_period_symbols > 0 and (
                    sim_body_len % self.pilot_period_symbols == 0
                ):
                    insert_pilot = True
                    pilots_to_insert = 1

            elif self.pilot_pattern == "block":
                if self.pilot_period_symbols > 0 and (
                    sim_body_len % self.pilot_period_symbols == 0
                ):
                    insert_pilot = True
                    pilots_to_insert = self.pilot_len_symbols

            if insert_pilot:
                # Add pilot indices
                for _ in range(pilots_to_insert):
                    pilot_indices.append(curr_idx)
                    curr_idx += 1
                    sim_body_len += 1
                # If we run out of pilots in real compose, it behaves differently,
                # but map should reflect ideal structure or we need to know pilot count.
                # Assuming infinite pilots for map, or we check against a max limit if needed.
            else:
                # Add payload index
                payload_indices.append(curr_idx)
                curr_idx += 1
                sim_body_len += 1
                pay_done += 1

        return {
            "preamble": slice(0, preamble_len),
            "pilots": pilot_indices,
            "payload": payload_indices,
            "total_length": curr_idx + self.guard_period,
        }
