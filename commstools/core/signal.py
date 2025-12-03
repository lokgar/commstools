import dataclasses
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union


from .backend import ArrayType, Backend, get_backend


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
            self.modulation_format = "unknown"

        # Ensure samples are on the current backend upon initialization
        # unless they are already a valid array type (Numpy or JAX)
        import numpy as np

        is_known_array = isinstance(self.samples, np.ndarray)

        if not is_known_array:
            try:
                import jax.numpy as jnp

                if isinstance(self.samples, type(jnp.array([]))) or hasattr(
                    self.samples, "device_buffer"
                ):
                    is_known_array = True
            except ImportError:
                pass

        if not is_known_array:
            current_backend = get_backend()
            self.samples = current_backend.asarray(self.samples)

    @property
    def backend(self) -> Backend:
        """Returns the backend associated with the signal's data."""
        return self._get_backend()

    @property
    def duration(self) -> float:
        """Duration of the signal in seconds."""
        return self.samples.shape[0] / self.sampling_rate

    @property
    def sps(self) -> float:
        """Samples per symbol."""
        return self.sampling_rate / self.symbol_rate

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
            return self.backend.welch(
                self.samples,
                fs=self.sampling_rate,
                nperseg=nperseg,
                detrend=detrend,
                average=average,
                return_onesided=False,
            )
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
            New Signal with filtered samples.
        """
        from ..dsp import filtering

        new_samples = filtering.fir_filter(self.samples, taps, mode=mode)
        return self.update(samples=new_samples)

    def upsample(self, factor: int) -> "Signal":
        """
        Upsample the signal.

        Args:
            factor: Upsampling factor.

        Returns:
            New Signal with upsampled samples and updated sampling rate.
        """
        from ..dsp import multirate

        new_samples = multirate.upsample(self.samples, factor)
        new_rate = self.sampling_rate * factor
        return self.update(samples=new_samples, sampling_rate=new_rate)

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
            New Signal with decimated samples and updated sampling rate.
        """
        from ..dsp import multirate

        new_samples = multirate.decimate(
            self.samples, factor, filter_type=filter_type, **kwargs
        )
        new_rate = self.sampling_rate / factor
        return self.update(samples=new_samples, sampling_rate=new_rate)

    def resample(self, up: int, down: int) -> "Signal":
        """
        Resample the signal by a rational factor.

        Args:
            up: Upsampling factor.
            down: Downsampling factor.

        Returns:
            New Signal with resampled samples and updated sampling rate.
        """
        from ..dsp import multirate

        new_samples = multirate.resample(self.samples, up, down)
        new_rate = self.sampling_rate * up / down
        return self.update(samples=new_samples, sampling_rate=new_rate)

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
            New Signal with matched filtered samples.
        """
        from ..dsp import filtering

        new_samples = filtering.matched_filter(
            self.samples,
            pulse_taps=pulse_taps,
            taps_normalization=taps_normalization,
            mode=mode,
            normalize_output=normalize_output,
        )
        return self.update(samples=new_samples)

    def update(
        self,
        samples: Optional[ArrayType] = None,
        sampling_rate: Optional[float] = None,
        modulation_format: Optional[str] = None,
    ) -> "Signal":
        """
        Creates a new Signal with updated fields, defaulting to the original values.

        Args:
            samples: The new samples array. Defaults to the current samples.
            sampling_rate: The new sampling rate in Hz. Defaults to the current sampling rate.
            modulation_format: The new modulation format. Defaults to the current modulation format.

        Returns:
            A new Signal instance with the specified updates.
        """
        changes: dict[str, Any] = {}
        if samples is not None:
            changes["samples"] = samples
        if sampling_rate is not None:
            changes["sampling_rate"] = sampling_rate
        if modulation_format is not None:
            changes["modulation_format"] = modulation_format

        return dataclasses.replace(self, **changes)

    def _get_backend(self) -> Backend:
        """
        Infers the backend based on the type of 'samples'.
        Fallback to global backend if uncertain.
        """
        # This is a bit heuristic.
        # Ideally we check the type of self.samples against known backend array types.

        # Check for Numpy array first to avoid importing JAX if not needed
        import numpy as np

        if isinstance(self.samples, np.ndarray):
            from .backend import NumpyBackend

            return NumpyBackend()

        # Check for JAX array
        try:
            import jax.numpy as jnp

            if isinstance(self.samples, type(jnp.array([]))) or (
                hasattr(self.samples, "device_buffer")
            ):
                from .backend import JaxBackend

                return JaxBackend()
        except ImportError:
            pass

        # Fallback to global default
        return get_backend()

    def to(self, backend: str) -> "Signal":
        """
        Moves the signal data to the specified backend.

        Args:
            backend: The name of the target backend ('numpy' or 'jax').

        Returns:
            A new Signal instance with data on the requested backend, or self if already on that backend.
        """
        if self.backend.name == backend:
            return self

        from .backend import JaxBackend, NumpyBackend

        target_backend: Backend
        if backend.lower() == "numpy":
            target_backend = NumpyBackend()
        elif backend.lower() == "jax":
            target_backend = JaxBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}")

        # Convert samples
        new_samples = target_backend.array(self.samples)

        return dataclasses.replace(self, samples=new_samples)


# Register Signal as a JAX Pytree node if JAX is available
try:
    import jax

    def _signal_tree_flatten(signal: Signal) -> tuple:
        children = (signal.samples,)  # Arrays/tensors to be traced
        aux_data = (
            signal.sampling_rate,
            signal.symbol_rate,
            signal.modulation_format,
        )  # Static metadata
        return children, aux_data

    def _signal_tree_unflatten(aux_data: tuple, children: tuple) -> Signal:
        return Signal(
            samples=children[0],
            sampling_rate=aux_data[0],
            symbol_rate=aux_data[1],
            modulation_format=aux_data[2],
        )

    jax.tree_util.register_pytree_node(
        Signal, _signal_tree_flatten, _signal_tree_unflatten
    )
except ImportError:
    pass
