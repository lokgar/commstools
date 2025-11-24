from typing import Optional, Tuple, Any
from dataclasses import dataclass
from .backend import get_backend, Backend, ArrayType


@dataclass
class Signal:
    """
    Represents a digital signal with associated physical layer metadata.

    Attributes:
        samples: The complex IQ samples of the signal.
        sample_rate: The sampling rate in Hz.
        center_freq: The center frequency in Hz.
        modulation_format: Description of the modulation format (e.g., 'QPSK', '16QAM').
    """

    samples: ArrayType
    sample_rate: float
    center_freq: float = 0.0
    modulation_format: str = "unknown"

    def __post_init__(self):
        # Ensure samples are on the current backend upon initialization if they aren't already
        # This might be too aggressive if we want to allow mixed backend usage,
        # but for now it ensures consistency.
        # However, to avoid implicit copying, we might just trust the user or
        # provide a method to ensure backend.
        pass

    @property
    def duration(self) -> float:
        """Duration of the signal in seconds."""
        backend = self._get_backend()
        return self.samples.shape[0] / self.sample_rate

    def time_axis(self) -> ArrayType:
        """Returns the time vector associated with the signal samples."""
        backend = self._get_backend()
        return backend.arange(0, self.samples.shape[0]) / self.sample_rate

    def spectrum(self, nfft: Optional[int] = None) -> Tuple[ArrayType, ArrayType]:
        """
        Computes the power spectral density of the signal.

        Args:
            nfft: Number of FFT points. Defaults to the length of the signal.

        Returns:
            A tuple containing (frequency_axis, psd_magnitude).
        """
        backend = self._get_backend()
        if nfft is None:
            nfft = self.samples.shape[0]

        # Compute FFT
        # Use backend.fft
        spectrum = backend.fftshift(backend.fft(self.samples, n=nfft))

        # Compute PSD (magnitude squared)
        psd = backend.abs(spectrum) ** 2

        # Frequency axis
        freqs = (
            backend.fftshift(backend.fftfreq(nfft, d=1 / self.sample_rate))
            + self.center_freq
        )

        return freqs, psd

    def to(self, backend_name: str) -> "Signal":
        """
        Moves the signal data to the specified backend.

        Args:
            backend_name: The name of the target backend ('numpy' or 'jax').

        Returns:
            A new Signal instance with data on the requested backend.
        """
        from .backend import set_backend, get_backend, NumpyBackend, JaxBackend

        target_backend: Backend
        if backend_name.lower() == "numpy":
            target_backend = NumpyBackend()
        elif backend_name.lower() == "jax":
            target_backend = JaxBackend()
        else:
            raise ValueError(f"Unknown backend: {backend_name}")

        # Convert samples
        new_samples = target_backend.array(self.samples)

        return Signal(
            samples=new_samples,
            sample_rate=self.sample_rate,
            center_freq=self.center_freq,
            modulation_format=self.modulation_format,
        )

    def _get_backend(self) -> Backend:
        """
        Infers the backend based on the type of 'samples'.
        Fallback to global backend if uncertain.
        """
        # This is a bit heuristic.
        # Ideally we check the type of self.samples against known backend array types.

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

        # Check for Numpy array
        import numpy as np

        if isinstance(self.samples, np.ndarray):
            from .backend import NumpyBackend

            return NumpyBackend()

        # Fallback to global default
        # Fallback to global default
        return get_backend()


# Register Signal as a JAX Pytree node if JAX is available
try:
    import jax

    def _signal_tree_flatten(signal):
        children = (signal.samples,)  # Arrays/tensors to be traced
        aux_data = (
            signal.sample_rate,
            signal.center_freq,
            signal.modulation_format,
        )  # Static metadata
        return children, aux_data

    def _signal_tree_unflatten(aux_data, children):
        return Signal(children[0], *aux_data)

    jax.tree_util.register_pytree_node(
        Signal, _signal_tree_flatten, _signal_tree_unflatten
    )
except ImportError:
    pass
