from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False


from .backend import ArrayType, Backend, CupyBackend, NumpyBackend, get_backend


@dataclass
class Signal:
    """
    Represents a digital signal with associated physical layer metadata.

    Attributes:
        samples: The complex IQ samples of the signal.
        sampling_rate: The sampling rate in Hz.
        modulation_format: Description of the modulation format (e.g., 'QPSK', '16QAM').
    """

    samples: ArrayType
    sampling_rate: Optional[float] = None
    symbol_rate: Optional[float] = None
    modulation_format: Optional[str] = None

    def __post_init__(self) -> None:
        # Validate required fields
        if self.sampling_rate is None:
            raise ValueError("sampling_rate must be provided explicitly")

        if self.symbol_rate is None:
            raise ValueError("symbol_rate must be provided explicitly")

        if self.modulation_format is None:
            self.modulation_format = "None"

        # Ensure samples are on the current backend upon initialization
        # First, ensure it's something we can work with
        if not isinstance(self.samples, (np.ndarray, list, tuple)):
            if _CUPY_AVAILABLE and isinstance(self.samples, cp.ndarray):
                pass
            else:
                # Try to convert to numpy array to see if it's a valid array-like
                try:
                    self.samples = np.asarray(self.samples)
                except Exception:
                    raise ValueError(
                        f"Unsupported samples type: {type(self.samples)}. Must be array-like."
                    )

        current_backend = get_backend()
        if current_backend.name == "cupy":
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
        return self.backend.arange(0, self.samples.shape[0]) / self.sampling_rate

    def welch_psd(
        self,
        nperseg: int = 256,
        detrend: Optional[Union[str, bool]] = False,
        average: Optional[str] = "mean",
    ) -> Tuple[ArrayType, ArrayType]:
        if self.backend.iscomplexobj(self.samples):
            f, Pxx = self.backend.welch(
                self.samples,
                fs=self.sampling_rate,
                nperseg=nperseg,
                detrend=detrend,
                average=average,
                return_onesided=False,
            )
            f = self.backend.fftshift(f)
            Pxx = self.backend.fftshift(Pxx)
            return f, Pxx
        else:
            return self.backend.welch(
                self.samples,
                fs=self.sampling_rate,
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
        show: bool = False,
        **kwargs: Any,
    ) -> Optional[Tuple[Any, Any]]:
        from .. import plotting

        return plotting.psd(
            self.samples,
            sampling_rate=self.sampling_rate,
            nperseg=nperseg,
            detrend=detrend,
            average=average,
            ax=ax,
            title=title,
            show=show,
            **kwargs,
        )

    def plot_signal(
        self,
        num_symbols: int = None,
        ax: Optional[Any] = None,
        title: Optional[str] = "Waveform",
        show: bool = False,
        **kwargs: Any,
    ) -> Optional[Tuple[Any, Any]]:
        from .. import plotting

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
        plot_type: str = "line",
        title: Optional[str] = "Eye Diagram",
        show: bool = False,
        **kwargs: Any,
    ) -> Optional[Tuple[Any, Any]]:
        from .. import plotting

        return plotting.eye_diagram(
            self.samples,
            ax=ax,
            sps=self.sampling_rate / self.symbol_rate,
            plot_type=plot_type,
            title=title,
            show=show,
            **kwargs,
        )

    def fir_filter(self, taps: ArrayType, mode: str = "same") -> "Signal":
        """
        Apply FIR filter to the signal.

        Args:
            taps: Filter taps.
            mode: Convolution mode.

        Returns:
            self
        """
        from ..dsp import filtering

        self.samples = filtering.fir_filter(self.samples, taps, mode=mode)
        return self

    def upsample(self, factor: int) -> "Signal":
        """
        Upsample the signal.

        Args:
            factor: Upsampling factor.

        Returns:
            self
        """
        from ..dsp import multirate

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
        from ..dsp import multirate

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
        from ..dsp import multirate

        self.samples = multirate.resample(self.samples, up, down)
        self.sampling_rate = self.sampling_rate * up / down
        return self

    def matched_filter(
        self,
        pulse_taps: ArrayType,
        taps_normalization: str = "unity_gain",
        mode: str = "same",
        normalize_output: bool = False,
    ) -> "Signal":
        """
        Apply matched filter to the signal.

        Args:
            pulse_taps: Pulse shape filter taps.
            taps_normalization: Normalization to apply to the matched filter taps.
                                Options: 'unity_gain', 'unit_energy'.
            mode: Convolution mode ('same', 'full', 'valid').
            normalize_output: If True, normalizes the output samples to have a maximum
                              absolute value of 1.0.

        Returns:
            self
        """
        from ..dsp import filtering

        self.samples = filtering.matched_filter(
            self.samples,
            pulse_taps=pulse_taps,
            taps_normalization=taps_normalization,
            mode=mode,
            normalize_output=normalize_output,
        )
        return self
