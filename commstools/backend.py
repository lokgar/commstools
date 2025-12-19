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
from typing import Any, Optional, Tuple, Union

import numpy as np

from .logger import logger

# Try to import CuPy and verify functionality
try:
    import cupy as cp

    # Aggressive check: try to allocate and run a simple operation.
    # This catches cases where CuPy is installed but shared libraries (nvrtc, cublas) are missing.
    try:
        cp.arange(1)
        _CUPY_AVAILABLE = True
        logger.info("CuPy is available and functional, defaulting Signals to GPU.")
    except Exception:
        # Fallback if functional check fails
        _CUPY_AVAILABLE = False
        cp = None
        logger.warning(
            "CuPy has problems with shared libraries, falling back to NumPy."
        )

except ImportError:
    _CUPY_AVAILABLE = False
    cp = None
    logger.debug("CuPy is not available, falling back to NumPy.")

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
        device: 'CPU' or 'GPU'.

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
        raise ValueError(f"Unknown device: {device.upper()}")


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


def to_jax(data: Any, device: Optional[str] = None) -> Any:
    """
    Converts data to a JAX array, preserving the device if possible or targeting a specific device.

    Args:
        data: Input data (NumPy array, CuPy array, list, etc.).
        device: Optional target device ('CPU', 'GPU', or 'TPU'). If None, it attempts
                to preserve the original data's device.

    Returns:
        JAX array on the corresponding device.

    Raises:
        ImportError: If JAX is not installed.
        ValueError: If the requested device is not available.
    """
    try:
        import jax
        import jax.numpy as jnp
        from jax import dlpack as jax_dlpack
    except ImportError:
        raise ImportError("JAX is not installed.")

    target_jax_device = None
    if device is not None:
        device = device.lower()
        # Map our device names to JAX platform names
        platform_map = {"cpu": "cpu", "gpu": "cuda", "tpu": "tpu"}
        jax_platform = platform_map.get(device, device)

        try:
            target_jax_device = jax.devices(jax_platform)[0]
        except (RuntimeError, IndexError):
            raise ValueError(
                f"Requested JAX device '{device}' (platform '{jax_platform}') is not available."
            )

    # 1. Handle CuPy -> JAX (GPU)
    if is_cupy_available() and isinstance(data, cp.ndarray):
        try:
            jax_arr = jax_dlpack.from_dlpack(data)
            # If a specific device was requested, move it if necessary
            if target_jax_device and jax_arr.device != target_jax_device:
                jax_arr = jax.device_put(jax_arr, target_jax_device)
            return jax_arr
        except Exception as e:
            logger.debug(
                f"DLPack transfer from CuPy to JAX failed: {e}. Falling back to explicit conversion."
            )

    # 2. General case
    jax_arr = jnp.asarray(data)

    # 3. Explicit device placement if requested or if we need to ensure consistency
    if target_jax_device:
        if jax_arr.device != target_jax_device:
            jax_arr = jax.device_put(jax_arr, target_jax_device)
    elif isinstance(data, np.ndarray):
        # Default for NumPy is CPU; ensure it stays there unless JAX defaults otherwise
        try:
            cpu_device = jax.devices("cpu")[0]
            if jax_arr.device != cpu_device:
                jax_arr = jax.device_put(jax_arr, cpu_device)
        except Exception:
            pass

    return jax_arr


def from_jax(data: Any) -> ArrayType:
    """
    Converts a JAX array to a backend-compatible array (NumPy or CuPy).
    Standardizes on NumPy for CPU/TPU and CuPy for GPU.

    Args:
        data: Input JAX array.

    Returns:
        NumPy array (if on CPU or TPU) or CuPy array (if on GPU and CuPy available).
    """
    # Check device
    device_platform = "cpu"
    try:
        if hasattr(data, "device"):
            device_platform = data.device.platform
        elif hasattr(data, "devices"):
            device_platform = list(data.devices())[0].platform
    except Exception:
        pass

    if device_platform == "cuda" and is_cupy_available():
        # Try zero-copy via DLPack to CuPy
        try:
            return cp.from_dlpack(data)
        except Exception as e:
            logger.debug(
                f"DLPack transfer from JAX to CuPy failed: {e}. Falling back to NumPy conversion."
            )

    if device_platform == "cuda" and not is_cupy_available():
        logger.warning(
            "JAX array is on GPU, but CuPy is not available. Falling back to NumPy (CPU)."
        )

    # Convert to numpy (will copy from GPU/TPU if needed)
    return np.asarray(data)
