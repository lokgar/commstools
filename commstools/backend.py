"""
Computational backend management and device orchestration.

This module provides the infrastructure for backend-agnostic execution across
CPU (NumPy), GPU (CuPy), and JAX. It implements a data-driven dispatch mechanism
that allows the library to automatically adjust its internal logic based on where
the input data resides.

The backend system is designed to be stateless and transparent, requiring
minimal explicit device management from the user.

Functions
---------
get_array_module :
    Infers the correct array module (NumPy/CuPy) from data.
get_scipy_module :
    Retrieves the compatible signal processing library.
to_device :
    Moves data between CPU and GPU devices.
dispatch :
    Returns the resolved array and signal processing modules for given data.
to_jax / from_jax :
    Utilities for high-performance JAX interoperability.
is_cupy_available :
    Checks if NVIDIA GPU acceleration is functional via CuPy.
use_cpu_only :
    Explicitly disables GPU discovery.
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

# JAX lazy loading cache
_JAX_CACHE = {}


def _get_jax() -> Tuple[
    Optional[types.ModuleType], Optional[types.ModuleType], Optional[Any]
]:
    """
    Lazy loader for JAX modules and its DLPack interface.

    Returns
    -------
    jax : module or None
        The base `jax` module if installed, else None.
    jnp : module or None
        The `jax.numpy` namespace if installed, else None.
    dlpack : module or None
        The `jax.dlpack` interface for zero-copy transfers, else None.
    """
    if "jax" not in _JAX_CACHE:
        try:
            import jax
            import jax.numpy as jnp
            from jax import dlpack

            _JAX_CACHE["jax"] = jax
            _JAX_CACHE["jnp"] = jnp
            _JAX_CACHE["dlpack"] = dlpack
        except ImportError:
            _JAX_CACHE["jax"] = None

    return _JAX_CACHE.get("jax"), _JAX_CACHE.get("jnp"), _JAX_CACHE.get("dlpack")


@lru_cache(maxsize=8)
def _get_jax_device(platform: str) -> Optional[Any]:
    """
    Retrieves a specific JAX device by platform name.

    Parameters
    ----------
    platform : {"cpu", "gpu", "tpu"}
        The target hardware platform identifier.

    Returns
    -------
    device : Device or None
        The first discovered device for the specified platform, or None
        if JAX is missing or the platform is unsupported.
    """
    jax, _, _ = _get_jax()
    if jax is None:
        return None
    try:
        # Map our common names to JAX platform names
        platform_map = {"cpu": "cpu", "gpu": "cuda", "tpu": "tpu"}
        jax_platform = platform_map.get(platform, platform)
        return jax.devices(jax_platform)[0]
    except (RuntimeError, IndexError):
        return None


_FORCE_CPU = False


def use_cpu_only(force: bool = True) -> None:
    """
    Enforces a CPU-only execution path, disabling GPU discovery.

    This function effectively hides CuPy from the library, even if a
    functional NVIDIA GPU and CuPy installation are present.

    Parameters
    ----------
    force : bool, default True
        If True, blocks all CUDA-accelerated operations.
    """
    global _FORCE_CPU
    _FORCE_CPU = force


def is_cupy_available() -> bool:
    """
    Checks if NVIDIA GPU acceleration is functional via CuPy.

    Returns
    -------
    bool
        True if CuPy is installed, functional, and not explicitly disabled
        via `use_cpu_only`.
    """
    if _FORCE_CPU:
        return False
    return _CUPY_AVAILABLE


def get_array_module(data: Any) -> types.ModuleType:
    """
    Infers the array module (NumPy or CuPy) for the given data.

    Parameters
    ----------
    data : array_like or list
        The input data to inspect.

    Returns
    -------
    module
        `numpy` if the data is on CPU or is a basic sequence,
        `cupy` if the data resides on a CUDA device.
    """
    if is_cupy_available():
        return cp.get_array_module(data)
    return np


@lru_cache(maxsize=None)
def get_scipy_module(xp: types.ModuleType) -> types.ModuleType:
    """
    Returns the signal processing library compatible with the given array module.

    Parameters
    ----------
    xp : module
        The array module (typically `numpy` or `cupy`).

    Returns
    -------
    sp : module
        The corresponding signal processing module (`scipy` or `cupyx.scipy`).
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
    Moves data between CPU and GPU devices.

    Parameters
    ----------
    data : array_like
        The data to move.
    device : {"CPU", "GPU"}
        Target device name (case-insensitive).

    Returns
    -------
    array_like
        The data residing on the target device.

    Raises
    ------
    ImportError
        If "GPU" is requested but CuPy is not available.
    ValueError
        If an unsupported device name is provided.

    Notes
    -----
    If the data is already on the target device, this operation
    typically returns a view or the original array to avoid
    unnecessary copies.
    """
    logger.debug(f"Moving data to {device.upper()}.")
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
    Inspects data and returns appropriate backend modules.

    This helper is used throughout the library to implement backend-agnostic
    functional logic.

    Parameters
    ----------
    data : array_like
        The input data to analyze.

    Returns
    -------
    data_array : array_like
        The input data forced to an array on its current device.
    xp : module
        The array module (`numpy` or `cupy`).
    sp : module
        The signal processing module (`scipy` or `cupyx.scipy`).
    """
    xp = get_array_module(data)
    sp = get_scipy_module(xp)

    if not isinstance(data, (np.ndarray, getattr(cp, "ndarray", type(None)))):
        data = xp.asarray(data)

    return data, xp, sp


def to_jax(data: Any, device: Optional[str] = None) -> Any:
    """
    Converts data to a JAX array with optimized device placement.

    This function supports zero-copy transfers from CuPy using DLPack
    when moving data between CUDA-managed memories.

    Parameters
    ----------
    data : array_like
        Input data (NumPy array, CuPy array, list, or scalar).
    device : {"CPU", "GPU", "TPU"}, optional
        Target JAX device platform. If None, the function attempts to
        preserve the device of the original data.

    Returns
    -------
    jax_array : jax.Array
        A JAX array residing on the specified or inferred device.

    Raises
    ------
    ImportError
        If the `jax` library is not installed.
    ValueError
        If the requested `device` platform is not available in the
        local JAX environment.
    """
    jax, jnp, jax_dlpack = _get_jax()
    if jax is None:
        raise ImportError("JAX is not installed.")

    target_device = None
    if device is not None:
        target_device = _get_jax_device(device.lower())
        if target_device is None:
            raise ValueError(f"Requested JAX device '{device}' is not available.")

    # 1. Handle CuPy -> JAX (GPU)
    if is_cupy_available() and isinstance(data, cp.ndarray):
        try:
            # DLPack is the fastest way for zero-copy GPU transfer
            jax_arr = jax_dlpack.from_dlpack(data)
            if target_device and jax_arr.device != target_device:
                return jax.device_put(jax_arr, target_device)
            return jax_arr
        except Exception as e:
            logger.debug(
                f"DLPack transfer from CuPy to JAX failed: {e}. Falling back to explicit conversion."
            )

    # 2. Optimized Placement
    # If a target device is specified, use device_put directly.
    # This is more efficient than jnp.asarray(data) + device_put because it avoids
    # an intermediate placement on the JAX default device.
    if target_device:
        return jax.device_put(data, target_device)

    # 3. Preservation Logic (No target device specified)
    if isinstance(data, np.ndarray):
        # Default for NumPy is CPU; ensure it stays there to preserve device origin.
        # JAX might otherwise default to placing it on GPU if available.
        cpu_dev = _get_jax_device("cpu")
        if cpu_dev:
            return jax.device_put(data, cpu_dev)

    # 4. General case (lists, scalars, or existing JAX arrays)
    return jnp.asarray(data)


def from_jax(data: Any) -> ArrayType:
    """
    Converts a JAX array to a backend-compatible array (NumPy or CuPy).

    Standardizes on NumPy for CPU/TPU arrays and CuPy for GPU arrays
    to maintain compatibility with the rest of the library. Uses zero-copy
    DLPack transfers for GPU arrays when available.

    Parameters
    ----------
    data : jax.Array
        Input JAX array to convert.

    Returns
    -------
    array : array_like
        A NumPy array (if on CPU/TPU) or a CuPy array (if on GPU).
    """
    # Detect platform
    platform = "cpu"
    try:
        # Standard JAX 0.4.x+ device inspection
        if hasattr(data, "device"):
            platform = data.device.platform
        elif hasattr(data, "devices"):
            platform = list(data.devices())[0].platform
    except Exception:
        pass

    is_gpu = platform in ("cuda", "gpu")

    if is_gpu and is_cupy_available():
        # Try zero-copy via DLPack to CuPy
        try:
            return cp.from_dlpack(data)
        except Exception as e:
            logger.debug(
                f"DLPack transfer from JAX to CuPy failed: {e}. Falling back to NumPy conversion."
            )

    if is_gpu and not is_cupy_available():
        logger.warning(
            "JAX array is on GPU, but CuPy is not available. Falling back to NumPy (CPU)."
        )

    # Convert to numpy (will copy from GPU/TPU if needed)
    return np.asarray(data)


def is_jax_array(data: Any) -> bool:
    """
    Checks if the given data is a JAX array without eagerly importing JAX.

    Parameters
    ----------
    data : any
        The object to check.

    Returns
    -------
    bool
        True if `data` is a `jax.Array` instance.
    """
    jax, _, _ = _get_jax()
    if jax is None:
        return False
    return isinstance(data, jax.Array)
