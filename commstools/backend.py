"""
Computational backend management.

This module abstracts the underlying computational library (NumPy or CuPy) to support
transparent execution on CPU and GPU using a stateless, data-driven approach.
It defines helper functions to:
- Infer the active backend module (NumPy or CuPy) from data.
- Manage data transfer between devices.
- Interoperate with JAX.
"""

import types
from functools import lru_cache
from typing import Any, Tuple, Union

import numpy as np

# Try to import CuPy and verify functionality
try:
    import cupy as cp

    # Aggressive check: try to allocate and run a simple operation.
    # This catches cases where CuPy is installed but shared libraries (nvrtc, cublas) are missing.
    try:
        cp.arange(1)
        _CUPY_AVAILABLE = True
        print("CuPy is available and functional, defaulting Signals to GPU.")
    except Exception:
        # Fallback if functional check fails
        _CUPY_AVAILABLE = False
        cp = None
        print("CuPy has problems with shared libraries, falling back to NumPy.")

except ImportError:
    _CUPY_AVAILABLE = False
    cp = None
    print("CuPy is not available, falling back to NumPy.")

ArrayType = Union[
    np.ndarray, Any
]  # Any for CuPy array to avoid hard dependency in type hint if not installed


_FORCE_CPU = False


def use_cpu_only(force: bool = True) -> None:
    """
    Forces the library to use CPU only, pretending CuPy is not available.

    Args:
        force: If True, blocks CuPy availability.
    """
    global _FORCE_CPU
    _FORCE_CPU = force


def is_cupy_available() -> bool:
    """Returns True if CuPy is available and functional, and not forced off."""
    if _FORCE_CPU:
        return False
    return _CUPY_AVAILABLE


def get_array_module(data: Any) -> types.ModuleType:
    """
    Returns the array module (numpy or cupy) for the given data.

    Args:
        data: Input data (array or list).

    Returns:
        The numpy module if data is on CPU (or a list), or cupy if on GPU.
    """
    if is_cupy_available():
        return cp.get_array_module(data)
    return np


@lru_cache(maxsize=None)
def get_scipy_module(xp: types.ModuleType) -> types.ModuleType:
    """
    Returns the signal library (scipy or cupyx.scipy) corresponding to the array module.

    Args:
        xp: The array module (numpy or cupy).

    Returns:
        The corresponding scipy-compatible module.
    """
    if is_cupy_available() and xp == cp:
        import cupyx.scipy
        import cupyx.scipy.ndimage
        import cupyx.scipy.signal
        import cupyx.scipy.special

        return cupyx.scipy

    import scipy
    import scipy.ndimage
    import scipy.signal
    import scipy.special

    return scipy


def to_device(data: Any, device: str) -> ArrayType:
    """
    Moves data to the specified device.

    Args:
        data: Input data.
        device: 'cpu' or 'gpu'.

    Returns:
        Array on the target device.
    """
    device = device.lower()
    if device == "cpu":
        # Verify both availability and force flag to behave safely
        # But for conversion from GPU to CPU, we can be lenient if logic allows
        # However, to avoid touching cp if forced off:
        if is_cupy_available() and isinstance(data, cp.ndarray):
            return data.get()
        if isinstance(data, np.ndarray):
            return data
        return np.asarray(data)

    elif device == "gpu":
        if not is_cupy_available():
            raise ImportError("CuPy is not available.")
        if isinstance(data, cp.ndarray):
            return data
        return cp.asarray(data)

    else:
        raise ValueError(f"Unknown device: {device}")


def dispatch(
    data: Any,
) -> Tuple[ArrayType, types.ModuleType, types.ModuleType]:
    """
    Prepare data and return appropriate backend modules (xp, sp).
    Infers backend from the input data.

    Args:
        data: Input data.

    Returns:
        Tuple of (data_array, xp_module, sp_module).
    """
    xp = get_array_module(data)
    sp = get_scipy_module(xp)

    # Ensure data is an array of the inferred capability
    # If forced CPU, cp.ndarray might not be resolvable if we didn't import it,
    # but here cp is imported if installed.
    # checking getattr(cp, ...) is safe.
    if not isinstance(data, (np.ndarray, getattr(cp, "ndarray", type(None)))):
        data = xp.asarray(data)

    return data, xp, sp


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

    if is_cupy_available() and isinstance(data, cp.ndarray):
        # Zero-copy transfer from CuPy (GPU) to JAX (GPU) via DLPack
        try:
            return jax_dlpack.from_dlpack(data)
        except Exception:
            # Fallback if dlpack not supported or fails
            pass

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

    if not is_cpu and is_cupy_available():
        # Try zero-copy via DLPack to CuPy
        try:
            return cp.from_dlpack(data)
        except Exception:
            # Fallback to implicit conversion if DLPack fails
            pass

    # Convert to numpy (will copy from GPU if needed via JAX implicit conversion)
    return np.asarray(data)
