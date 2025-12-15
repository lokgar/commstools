from typing import Any, Optional, Tuple, Union, Literal, Dict
from pydantic import BaseModel, ConfigDict, Field, field_validator

import numpy as np

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False


from .backend import ArrayType, Backend, CupyBackend, NumpyBackend, get_backend


class Signal(BaseModel):
    """
    Represents a digital signal with associated physical layer metadata.

    Attributes:
        samples: The complex IQ samples of the signal.
        sampling_rate: The sampling rate in Hz.
        symbol_rate: The symbol rate in Hz.
        modulation_scheme: Description of the modulation format (e.g., 'QPSK', '16QAM').
        spectral_domain: The spectral domain of the signal ('BASEBAND', 'PASSBAND', 'INTERMEDIATE').
        physical_domain: The physical domain of the signal ('DIG', 'RF', 'OPT').
        center_frequency: The center frequency in Hz.
        digital_frequency_offset: The digital frequency offset in Hz.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    samples: Any
    sampling_rate: float = Field(..., gt=0)
    symbol_rate: float = Field(..., gt=0)
    modulation_scheme: str = "None"
    pulse_shape: Optional[str] = None
    pulse_params: Optional[Dict[str, Any]] = None
    spectral_domain: Literal["BASEBAND", "PASSBAND", "INTERMEDIATE"] = "BASEBAND"
    physical_domain: Literal["DIG", "RF", "OPT"] = "DIG"
    center_frequency: float = Field(default=0, ge=0)
    digital_frequency_offset: float = Field(default=0, ge=0)

    @field_validator("samples", mode="before")
    @classmethod
    def validate_samples(cls, v: Any) -> Any:
        # Ensure it's something we can work with
        if not isinstance(v, (np.ndarray, list, tuple)):
            if _CUPY_AVAILABLE and isinstance(v, cp.ndarray):
                return v
            else:
                # Try to convert to numpy array to see if it's a valid array-like
                try:
                    return np.asarray(v)
                except Exception:
                    raise ValueError(
                        f"Unsupported samples type: {type(v)}. Must be array-like."
                    )

        if isinstance(v, (list, tuple)):
            return np.asarray(v)

        return v

    def model_post_init(self, __context: Any) -> None:
        current_backend = get_backend()
        if current_backend.name == "gpu":
            self.to("gpu")
        else:
            self.to("cpu")

    @property
    def backend(self) -> Backend:
        """Returns the backend associated with the signal's data."""
        if isinstance(self.samples, np.ndarray):
            return NumpyBackend()
        if _CUPY_AVAILABLE and isinstance(self.samples, cp.ndarray):
            return CupyBackend()

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
            device: The target device ('cpu' or 'gpu').

        Returns:
            self
        """
        device = device.lower()
        if device == "cpu":
            if isinstance(self.samples, np.ndarray):
                return self
            if _CUPY_AVAILABLE and isinstance(self.samples, cp.ndarray):
                self.samples = cp.asnumpy(self.samples)
                return self
            # Fallback if unknown array type but requested CPU
            # Try to convert to numpy array
            self.samples = np.asarray(self.samples)
            return self

        elif device == "gpu":
            if not _CUPY_AVAILABLE:
                raise ImportError("CuPy is not installed or not available.")
            if isinstance(self.samples, cp.ndarray):
                return self
            self.samples = cp.asarray(self.samples)
            return self

        else:
            raise ValueError(f"Unknown device: {device}. Use 'cpu' or 'gpu'.")

    def to_jax(self) -> Any:
        """
        Exports the signal samples to a JAX array.

        Returns:
            A JAX array containing the signal samples.
        """
        from .backend import to_jax

        return to_jax(self.samples)

    def from_jax(self, jax_array: Any) -> "Signal":
        """
        Updates the signal samples from a JAX array.

        Args:
            jax_array: Input JAX array.

        Returns:
            self
        """
        from .backend import from_jax

        self.samples = from_jax(jax_array)
        return self

    def time_axis(self) -> ArrayType:
        """Returns the time vector associated with the signal samples."""
        return self.backend.xp.arange(0, self.samples.shape[0]) / self.sampling_rate

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
        if self.backend.xp.iscomplexobj(self.samples):
            f, Pxx = self.backend.sp.signal.welch(
                self.samples,
                fs=self.sampling_rate,
                nperseg=nperseg,
                detrend=detrend,
                average=average,
                return_onesided=False,
            )
            f = self.backend.xp.fft.fftshift(f)
            Pxx = self.backend.xp.fft.fftshift(Pxx)
            return f, Pxx
        else:
            return self.backend.sp.signal.welch(
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
        if value is None:
            return "None"

        if abs(value) == 0:
            return f"0.00 {unit}"

        # Standard SI prefixes
        si_units = {
            -5: "f",
            -4: "p",
            -3: "n",
            -2: "µ",
            -1: "m",
            0: "",
            1: "k",
            2: "M",
            3: "G",
            4: "T",
            5: "P",
        }

        rank = int(np.floor(np.log10(abs(value)) / 3))
        # clamp to supported range
        rank = max(min(si_units.keys()), min(rank, max(si_units.keys())))

        scaled = value / (1000.0**rank)
        return f"{scaled:.2f} {si_units[rank]}{unit}"

    def print_summary(self) -> None:
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
                "Sampling Rate",
                "Symbol Rate",
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
                self._format_si(self.sampling_rate, "Hz"),
                self._format_si(self.symbol_rate, "Baud"),
                f"{self.sps:.2f}",
                self.pulse_shape.upper() if self.pulse_shape else "None",
                self._format_si(self.duration, "s"),
                self._format_si(self.center_frequency, "Hz"),
                self._format_si(self.digital_frequency_offset, "Hz"),
                self.backend.name.upper(),
                str(self.samples.shape),
            ],
        }
        df = pd.DataFrame(data)

        if get_ipython() is not None and "IPKernelApp" in get_ipython().config:
            display(df)
        else:
            print(df)

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

        from . import filtering

        params = self.pulse_params if self.pulse_params else {}

        # Default span if not present
        span = params.get("filter_span", 10)
        pulse_width = params.get("pulse_width", 1.0)

        if self.pulse_shape == "rect":
            return self.backend.xp.ones(int(self.sps * pulse_width))
        elif self.pulse_shape == "smoothrect":
            return filtering.smoothrect_taps(
                sps=self.sps,
                span=span,
                bt=params.get("smoothrect_bt", 1.0),
                pulse_width=pulse_width,
            )
        elif self.pulse_shape == "gaussian":
            return filtering.gaussian_taps(
                sps=self.sps,
                span=span,
                bt=params.get("gaussian_bt", 0.3),
            )
        elif self.pulse_shape == "rrc":
            return filtering.rrc_taps(
                sps=self.sps,
                span=span,
                rolloff=params.get("rrc_rolloff", 0.35),
            )
        elif self.pulse_shape == "rc":
            return filtering.rc_taps(
                sps=self.sps,
                span=span,
                rolloff=params.get("rc_rolloff", 0.35),
            )
        else:
            raise ValueError(f"Unknown pulse shape: {self.pulse_shape}")

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
            print(f"Cannot apply matched filter: {e}")
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
