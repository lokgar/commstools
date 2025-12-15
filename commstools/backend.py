"""
Computational backend management.

This module abstracts the underlying computational library (NumPy or CuPy) to support
transparent execution on CPU and GPU. It defines the `Backend` protocol and helper
functions to:
- Switch between CPU (NumPy) and GPU (CuPy) backends.
- Ensure data is on the active backend.
- Interoperate with JAX.
"""

import types
from typing import Any, Protocol, Union

import numpy as np

# Try to import CuPy, but don't fail if it's not available
try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False
    cp = None

ArrayType = Union[
    np.ndarray, Any
]  # Any for CuPy array to avoid hard dependency in type hint if not installed


class Backend(Protocol):
    """Protocol defining the interface for computational backends."""

    @property
    def name(self) -> str: ...

    @property
    def xp(self) -> types.ModuleType: ...

    @property
    def sp(self) -> types.ModuleType: ...


class NumpyBackend:
    """NumPy implementation of the Backend protocol."""

    @property
    def name(self) -> str:
        return "cpu"

    @property
    def xp(self) -> types.ModuleType:
        return np

    @property
    def sp(self) -> types.ModuleType:
        import scipy
        import scipy.ndimage
        import scipy.signal
        import scipy.special

        return scipy


class CupyBackend:
    """CuPy implementation of the Backend protocol."""

    def __init__(self) -> None:
        if not _CUPY_AVAILABLE:
            raise ImportError(
                "CuPy is not available. Please install it to use CupyBackend."
            )

    @property
    def name(self) -> str:
        return "gpu"

    @property
    def xp(self) -> types.ModuleType:
        return cp

    @property
    def sp(self) -> types.ModuleType:
        import cupyx.scipy
        import cupyx.scipy.ndimage
        import cupyx.scipy.signal
        import cupyx.scipy.special

        return cupyx.scipy


# Global state for current backend
_CURRENT_BACKEND: Backend = NumpyBackend()


def get_backend() -> Backend:
    """Get the currently active global backend."""
    return _CURRENT_BACKEND


def get_xp() -> types.ModuleType:
    """Get the array library (numpy or cupy) for the current backend."""
    return _CURRENT_BACKEND.xp


def get_sp() -> types.ModuleType:
    """Get the signal library (scipy.signal or cupyx.scipy.signal) for the current backend."""
    return _CURRENT_BACKEND.sp


def set_backend(backend_name: str) -> None:
    """
    Set the active backend globally.

    Args:
        backend_name (str): The name of the backend to set: "cpu" or "gpu".
    """
    global _CURRENT_BACKEND
    if backend_name.lower() == "cpu":
        _CURRENT_BACKEND = NumpyBackend()
    elif backend_name.lower() == "gpu":
        _CURRENT_BACKEND = CupyBackend()
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


def ensure_on_backend(data: Any) -> ArrayType:
    """
    Ensures the data is on the currently active global backend.

    Args:
        data: Input data (list, tuple, np.ndarray, cp.ndarray).

    Returns:
        Array on the active backend.
    """
    backend = get_backend()

    # Optimization: Check if already on correct backend
    if backend.name == "cpu":
        if isinstance(data, np.ndarray):
            return data
        if _CUPY_AVAILABLE and isinstance(data, cp.ndarray):
            return cp.asnumpy(data)
        return np.asarray(data)

    elif backend.name == "gpu":
        if _CUPY_AVAILABLE and isinstance(data, cp.ndarray):
            return data
        # If CuPy is active, we expect CuPy to be available
        return cp.asarray(data)

    return backend.xp.asarray(data)


def to_host(data: Any) -> np.ndarray:
    """
    Moves data to the host (CPU/NumPy) for plotting or I/O.

    Args:
        data: Input data.

    Returns:
        NumPy array.
    """
    if _CUPY_AVAILABLE and isinstance(data, cp.ndarray):
        return data.get()

    if isinstance(data, np.ndarray):
        return data

    return np.asarray(data)


def to_jax(data: Any) -> Any:
    """
    Converts data to a JAX array, preserving the device if possible.

    Args:
        data: Input data (NumPy array, CuPy array, list, etc.).

    Returns:
        JAX array on the corresponding device (CPU or GPU).

    Raises:
        ImportError: If JAX is not installed.
    """
    try:
        import jax.numpy as jnp
        from jax import dlpack as jax_dlpack
    except ImportError:
        raise ImportError("JAX is not installed.")

    if _CUPY_AVAILABLE and isinstance(data, cp.ndarray):
        # Zero-copy transfer from CuPy (GPU) to JAX (GPU) via DLPack
        return jax_dlpack.from_dlpack(data.toDlpack())

    if isinstance(data, np.ndarray):
        # Zero-copy transfer from NumPy (CPU) to JAX (CPU)
        return jnp.asarray(data)

    # Fallback for lists, etc.
    return jnp.array(data)


def from_jax(data: Any) -> ArrayType:
    """
    Converts a JAX array to a backend-compatible array (NumPy or CuPy).

    Args:
        data: Input JAX array.

    Returns:
        NumPy array (if on CPU) or CuPy array (if on GPU and CuPy avaliable).
    """
    # Check device - try to handle JAX arrays or compatible objects
    try:
        if hasattr(data, "device"):
            is_cpu = data.device.platform == "cpu"
        elif hasattr(data, "devices"):
            # For sharded arrays, assume first device
            is_cpu = list(data.devices())[0].platform == "cpu"
        else:
            is_cpu = True
    except Exception:
        # Graceful fallback
        is_cpu = True

    if not is_cpu and _CUPY_AVAILABLE:
        # Try zero-copy via DLPack to CuPy
        try:
            from jax import dlpack as jax_dlpack

            return cp.from_dlpack(jax_dlpack.to_dlpack(data))
        except Exception:
            # Fallback to implicit conversion if DLPack fails
            pass

    # Convert to numpy (will copy from GPU if needed via JAX implicit conversion)
    return np.asarray(data)
