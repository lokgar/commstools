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
    dispatch,
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
    source_symbols: Optional[Any] = None
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
        """
        Validates and coerces the samples input into a backend-compatible array.

        Args:
            v: The input samples (array-like, list, or tuple).

        Returns:
            The validated samples as a NumPy or CuPy array.

        Raises:
            ValueError: If the input cannot be converted to a supported array type.
        """
        return utils.validate_array(v, name="samples")

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization hook to set the default device.
        Automatically moves the signal to GPU if a compatible device is discovered.
        """
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
    def duration(self) -> float:
        """
        Duration of the signal in seconds.

        Calculated as the number of samples divided by the sampling rate.
        """
        return self.samples.shape[0] / self.sampling_rate

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
        return self.xp.arange(0, self.samples.shape[0]) / self.sampling_rate

    def demap_source_symbols(self) -> Optional[ArrayType]:
        """
        Demaps the source_symbols to bits using the current modulation.

        Returns:
            The demapped bits as an array, or None if source_symbols are not available.
        """
        if self.source_symbols is None:
            return None

        from . import mapping

        mod = self.modulation_scheme.lower() if self.modulation_scheme else "qam"
        if "-" in mod:
            mod = mod.split("-")[-1]

        return mapping.demap_symbols(
            self.source_symbols, modulation=mod, order=self.modulation_order
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
        from . import spectral

        return spectral.welch_psd(
            self.samples,
            sampling_rate=self.sampling_rate,
            nperseg=nperseg,
            detrend=detrend,
            average=average,
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

    def resample(
        self,
        up: Optional[int] = None,
        down: Optional[int] = None,
        sps_out: Optional[float] = None,
    ) -> "Signal":
        """
        Resample the signal by a rational factor.

        Args:
            up: Upsampling factor.
            down: Downsampling factor.
            sps_out: Target samples per symbol (if provided, up and down will be calculated automatically and don't have to be provided).

        Returns:
            self
        """
        from . import multirate

        # If sps_out is provided, we use the current signal's sps as input sps
        sps_in = self.sps if sps_out is not None else None

        self.samples = multirate.resample(
            self.samples, up=up, down=down, sps_in=sps_in, sps_out=sps_out
        )

        # Update sampling rate
        if sps_out is not None:
            self.sampling_rate = sps_out * self.symbol_rate
        elif up is not None and down is not None:
            self.sampling_rate = self.sampling_rate * up / down

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
    def generate(
        cls,
        modulation: str,
        order: int,
        num_symbols: int,
        sps: float,
        symbol_rate: float,
        pulse_shape: str = "none",
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> "Signal":
        """
        Generate a baseband Signal with specified parameters.

        Args:
            modulation: Modulation scheme ('psk', 'qam', 'ask').
            order: Modulation order.
            num_symbols: Number of symbols to generate.
            sps: Samples per symbol.
            symbol_rate: Symbol rate in Hz.
            pulse_shape: Pulse shaping type ('none', 'rect', 'smoothrect', 'gaussian', 'rrc', 'rc', 'sinc'). Default: 'none'.
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
        from . import filtering, utils

        # Generate symbols directly
        symbols = utils.random_symbols(
            num_symbols, modulation=modulation, order=order, seed=seed
        )

        if is_cupy_available():
            symbols = to_device(symbols, "gpu")

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
            symbols = utils.random_symbols(
                num_symbols, modulation="ask", order=order, seed=seed
            )

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
                h = to_device(h, "gpu")

            _, xp, sp = dispatch(symbols)

            # RZ hardcoded to 0.5 pulse width
            samples = utils.normalize(
                sp.signal.resample_poly(symbols, int(sps), 1, window=h),
                "max_amplitude",
            )

            return cls(
                samples=samples,
                sampling_rate=symbol_rate * sps,
                symbol_rate=symbol_rate,
                modulation_scheme=f"RZ-PAM{'-BIPOL' if bipolar else '-UNIPOL'}",
                modulation_order=order,
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
            pulse_shape: Pulse shaping type ('none', 'rect', 'smoothrect', 'gaussian', 'rrc', 'rc', 'sinc'). Default: 'rrc'.
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
            pulse_shape: Pulse shaping type ('none', 'rect', 'smoothrect', 'gaussian', 'rrc', 'rc', 'sinc'). Default: 'rrc'.
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
            seed=seed,
            **kwargs,
        )


class SingleCarrierFrame(BaseModel):
    """
    Represents a structured single-carrier frame with preamble, pilots, and payload.

    This class provides a high-level abstraction for constructing frames commonly
    used in digital communication systems, supporting various pilot patterns
    (none, block, comb) and guard intervals (zero-padding or cyclic prefix).

    Attributes:
        payload_len: Number of data SYMBOLS in the payload.
        payload_mod_scheme: Modulation scheme for the payload (e.g., "PSK", "QAM").
        payload_mod_order: Modulation order for the payload (e.g., 2, 4, 16).
        payload_seed: Random seed for payload bit generation.
        preamble: Optional leading signal (as an array-like) prepended to the frame.
        pilot_pattern: The arrangement of pilot symbols ("none", "block", "comb").
        pilot_period: Periodicity of pilots (for "comb" and "block").
        pilot_block_len: Length of the pilot block (for "block" pattern) in SYMBOLS.
        pilot_seed: Random seed for pilot symbol generation.
        pilot_mod_scheme: Modulation scheme for pilots.
        pilot_mod_order: Modulation order for pilots.
        guard_type: Type of guard interval ("zero" for padding, "cp" for cyclic prefix).
        guard_len: Length of the guard interval in SYMBOLS.
        symbol_rate: The symbol rate of the frame in Hz.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    payload_len: int = 1000
    payload_seed: int = 42
    payload_mod_scheme: str = "PSK"
    payload_mod_order: int = 4

    preamble: Optional[Any] = None

    pilot_pattern: Literal["none", "block", "comb"] = "none"
    pilot_period: int = 0
    pilot_block_len: int = 0
    pilot_seed: int = 1337
    pilot_mod_scheme: str = "PSK"
    pilot_mod_order: int = 4

    guard_type: Literal["zero", "cp"] = "zero"
    guard_len: int = 0

    symbol_rate: float = Field(..., gt=0)

    @field_validator("preamble", mode="before")
    @classmethod
    def validate_preamble(cls, v: Any) -> Any:
        """
        Coerces the preamble input into a backend-compatible array.

        Args:
            v: The input preamble (array-like, list, or tuple).

        Returns:
            The validated preamble as a NumPy or CuPy array.

        Raises:
            ValueError: If the input cannot be converted to a supported array type.
        """
        return utils.validate_array(v, name="preamble")

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization hook.
        Ensures the preamble is on the correct device (GPU if available) after initialization.
        """
        if is_cupy_available() and self.preamble is not None:
            self.preamble = to_device(self.preamble, "gpu")

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

    @property
    def payload_symbols(self) -> ArrayType:
        """Returns the mapped symbols of the payload."""
        from .utils import random_symbols

        return random_symbols(
            self.payload_len,
            modulation=self.payload_mod_scheme,
            order=self.payload_mod_order,
            seed=self.payload_seed,
        )

    @property
    def pilot_symbols(self) -> Optional[ArrayType]:
        """Returns the mapped symbols used for pilots, if any."""
        if self.pilot_pattern == "none":
            return None

        xp = cp if is_cupy_available() else np
        mask, _ = self._generate_pilot_mask()
        pilot_count = int(xp.sum(mask))
        if pilot_count == 0:
            return None

        from .utils import random_symbols

        return random_symbols(
            pilot_count,
            modulation=self.pilot_mod_scheme,
            order=self.pilot_mod_order,
            seed=self.pilot_seed,
        )

    @property
    def body_symbols(self) -> ArrayType:
        """Returns the interleaved pilot and payload symbols (frame body)."""
        xp = cp if is_cupy_available() else np
        mask, body_length = self._generate_pilot_mask()
        body = xp.zeros(body_length, dtype=xp.complex128)

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
            preamble = to_device(self.preamble, "gpu" if is_cupy_available() else "cpu")
            return xp.concatenate([preamble, body])
        else:
            return body

    def generate_sequence(self) -> Signal:
        """
        Constructs the complete frame sequence including preamble, pilots, payload, and guard interval.
        sps=1.

        Returns:
            A `Signal` object representing the generated frame.
        """
        xp = cp if is_cupy_available() else np

        # 1. Assemble Symbols
        signal_samples = self._assemble_symbols()

        # 2. Apply Guard Interval
        if self.guard_len > 0:
            if self.guard_type == "zero":
                # End-of-frame zero padding
                zeros = xp.zeros(self.guard_len, dtype=xp.complex128)
                signal_samples = xp.concatenate([signal_samples, zeros])
            elif self.guard_type == "cp":
                # Front-of-frame cyclic prefix
                cp_slice = signal_samples[-self.guard_len :]
                signal_samples = xp.concatenate([cp_slice, signal_samples])

        # 3. Create and return Signal metadata
        return Signal(
            samples=signal_samples,
            sampling_rate=self.symbol_rate,
            symbol_rate=self.symbol_rate,
            modulation_scheme=self.payload_mod_scheme,
            modulation_order=self.payload_mod_order,
            source_symbols=signal_samples,
        )

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
            A `Signal` object representing the shaped waveform.
        """
        xp = cp if is_cupy_available() else np
        from .filtering import shape_pulse

        # 1. Assemble Symbols (sps=1)
        symbols = self._assemble_symbols()

        # Determine logical/source symbols at sps=1 including guards
        xp = cp if is_cupy_available() else np
        source_symbols = symbols
        if self.guard_len > 0:
            if self.guard_type == "zero":
                source_symbols = xp.concatenate(
                    [source_symbols, xp.zeros(self.guard_len, dtype=xp.complex128)]
                )
            elif self.guard_type == "cp":
                cp_slice = source_symbols[-self.guard_len :]
                source_symbols = xp.concatenate([cp_slice, source_symbols])

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
                zeros = xp.zeros(guard_len_samples, dtype=xp.complex128)
                samples = xp.concatenate([samples, zeros])
            elif self.guard_type == "cp":
                cp_slice = samples[-guard_len_samples:]
                samples = xp.concatenate([cp_slice, samples])

        return Signal(
            samples=samples,
            sampling_rate=self.symbol_rate * sps,
            symbol_rate=self.symbol_rate,
            modulation_scheme=self.payload_mod_scheme,
            modulation_order=self.payload_mod_order,
            pulse_shape=pulse_shape,
            source_symbols=source_symbols,
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
        preamble_len = len(self.preamble) if self.preamble is not None else 0

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
