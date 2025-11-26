import dataclasses
from dataclasses import dataclass
from typing import Any, Optional, Tuple


from .backend import ArrayType, Backend, get_backend
from .config import get_config


@dataclass
class Signal:
    """
    Represents a digital signal with associated physical layer metadata.

    Attributes:
        samples: The complex IQ samples of the signal.
        sampling_rate: The sampling rate in Hz.
        center_freq: The center frequency in Hz.
        modulation_format: Description of the modulation format (e.g., 'QPSK', '16QAM').
    """

    samples: ArrayType
    sampling_rate: Optional[float] = None
    center_freq: Optional[float] = None
    modulation_format: Optional[str] = None
    use_config: dataclasses.InitVar[bool] = False

    def __post_init__(self, use_config: bool):
        # Merge with global config if requested
        if use_config:
            from .config import require_config

            config = require_config()
            if self.sampling_rate is None:
                self.sampling_rate = config.sampling_rate
            if self.center_freq is None:
                self.center_freq = config.center_freq
            if self.modulation_format is None:
                self.modulation_format = config.modulation_format

        # Validate required fields
        if self.sampling_rate is None:
            raise ValueError(
                "sampling_rate must be provided either explicitly or via global config (use_config=True)"
            )

        # Set defaults if still None
        if self.center_freq is None:
            self.center_freq = 0.0
        if self.modulation_format is None:
            self.modulation_format = "unknown"

        # Ensure samples are on the current backend upon initialization
        # This aligns the signal with the globally configured backend
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

    def time_axis(self) -> ArrayType:
        """Returns the time vector associated with the signal samples."""
        return self.backend.arange(0, self.samples.shape[0]) / self.sampling_rate

    def plot_psd(self, NFFT: int = 256, ax: Optional[Any] = None) -> Tuple[Any, Any]:
        from .. import plotting

        return plotting.psd(self, NFFT=NFFT, ax=ax)

    def plot_signal(self, ax: Optional[Any] = None) -> Tuple[Any, Any]:
        from .. import plotting

        return plotting.signal(self, ax=ax)

    def plot_eye(self, ax: Optional[Any] = None, plot_type="line") -> Tuple[Any, Any]:
        from .. import plotting

        return plotting.eye_diagram(self, ax=ax, plot_type=plot_type)

    def ensure_backend(self, backend_name: str = None) -> "Signal":
        """
        Ensures the signal is on the specified backend (or global default).

        Args:
            backend_name: Target backend name. If None, uses global default.

        Returns:
            Signal on the target backend (self if already there).
        """
        target = backend_name or get_backend().name
        if self.backend.name != target:
            return self._to_backend(target)
        return self

    def update(
        self,
        samples: Optional[ArrayType] = None,
        sampling_rate: Optional[float] = None,
        center_freq: Optional[float] = None,
        modulation_format: Optional[str] = None,
    ) -> "Signal":
        """
        Creates a new Signal with updated fields, defaulting to the original values.

        Args:
            samples: The new samples array. Defaults to the current samples.
            sampling_rate: The new sampling rate in Hz. Defaults to the current sampling rate.
            center_freq: The new center frequency in Hz. Defaults to the current center frequency.
            modulation_format: The new modulation format. Defaults to the current modulation format.

        Returns:
            A new Signal instance with the specified updates.
        """
        changes = {}
        if samples is not None:
            changes["samples"] = samples
        if sampling_rate is not None:
            changes["sampling_rate"] = sampling_rate
        if center_freq is not None:
            changes["center_freq"] = center_freq
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

    def _to_backend(self, backend_name: str) -> "Signal":
        """
        Moves the signal data to the specified backend.

        Args:
            backend_name: The name of the target backend ('numpy' or 'jax').

        Returns:
            A new Signal instance with data on the requested backend.
        """
        from .backend import JaxBackend, NumpyBackend

        target_backend: Backend
        if backend_name.lower() == "numpy":
            target_backend = NumpyBackend()
        elif backend_name.lower() == "jax":
            target_backend = JaxBackend()
        else:
            raise ValueError(f"Unknown backend: {backend_name}")

        # Convert samples
        new_samples = target_backend.array(self.samples)

        return dataclasses.replace(self, samples=new_samples)


# Register Signal as a JAX Pytree node if JAX is available
try:
    import jax

    def _signal_tree_flatten(signal):
        children = (signal.samples,)  # Arrays/tensors to be traced
        aux_data = (
            signal.sampling_rate,
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
