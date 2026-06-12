"""
Hand-written CUDA kernel infrastructure for the CuPy backend.

This subpackage hosts raw CUDA C++ kernels (under ``src/``) together with the
lazy compilation machinery built on :class:`cupy.RawModule`. Its public
surface is two functions:

is_available :
    True when CuPy is functional and a CUDA device with compute
    capability >= 7.0 is present.
get_kernel :
    Returns a launchable kernel wrapper, or ``None`` — in which case the
    caller **must** fall back to its existing ``xp`` implementation.

Importing this package (or ``commstools``) never touches NVRTC: kernel
sources are read and compiled on the first ``get_kernel()`` call for a
given specialization, and cached in-process afterwards. Cross-process
reuse comes from CuPy's on-disk kernel cache (``~/.cupy/kernel_cache``).
"""

from functools import lru_cache
from typing import Any, Callable, Optional

from ..backend import is_cupy_available
from ..logger import logger

# Volta and newer. Older parts lack the independent-thread-scheduling and
# shared-memory sizes the kernels in src/ are written against.
_MIN_COMPUTE_CAPABILITY = 70

# Kernel names that have already produced a compile/load warning; the
# fallback contract promises at most one warning per process per kernel.
_warned_kernels: set[str] = set()


@lru_cache(maxsize=1)
def _device_supported() -> bool:
    """Hardware probe, cached for the process lifetime.

    Checks that at least one CUDA device is present and that the current
    device has compute capability >= 7.0. Only called when CuPy itself is
    importable and functional.
    """
    import cupy as cp

    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            return False
        cc = int(cp.cuda.Device().compute_capability)
    except Exception:
        return False
    return cc >= _MIN_COMPUTE_CAPABILITY


def is_available() -> bool:
    """Checks whether custom CUDA kernels can be compiled and launched.

    Returns
    -------
    bool
        True if CuPy is installed and functional (and not disabled via
        ``backend.use_cpu_only``), at least one CUDA device is present,
        and the current device has compute capability >= 7.0.
    """
    # The CuPy check is evaluated fresh on every call so that
    # backend.use_cpu_only() is honored; only the hardware probe is cached.
    if not is_cupy_available():
        return False
    return _device_supported()


def get_kernel(name: str, **spec: Any) -> Optional[Callable]:
    """Returns a launchable kernel wrapper, or ``None`` if unavailable.

    ``None`` means the caller **must** fall back to the existing ``xp``
    implementation. Any compile/load failure logs one warning per kernel
    name per process and returns ``None``.

    Parameters
    ----------
    name : str
        Registered kernel name (see ``_KERNEL_FACTORIES``).
    **spec
        Kernel-specific specialization options (e.g. mode or dtype),
        forwarded to the kernel's wrapper factory.

    Returns
    -------
    callable or None
        A wrapper that handles grid/block computation and input
        validation. Call sites pass CuPy arrays and plain Python scalars
        only. ``None`` when no usable GPU is present or compilation fails.

    Raises
    ------
    KeyError
        If `name` is not a registered kernel. Unknown names are
        programming errors, not runtime fallback conditions.
    """
    factory = _KERNEL_FACTORIES.get(name)
    if factory is None:
        raise KeyError(
            f"Unknown CUDA kernel {name!r}; registered: {sorted(_KERNEL_FACTORIES)}"
        )
    if not is_available():
        return None
    try:
        return factory(**spec)
    except Exception as exc:
        if name not in _warned_kernels:
            _warned_kernels.add(name)
            logger.warning(
                "CUDA kernel %r failed to compile/load (%s); "
                "falling back to the array-module implementation.",
                name,
                exc,
            )
        return None


def _selftest_scale_factory(dtype: str = "float32") -> Callable:
    """Wrapper factory for the infrastructure self-test kernel.

    Validates the full compile -> specialize -> launch path; not used by
    any DSP code.
    """
    import cupy as cp

    from . import compiler

    ctype = {"float32": "float", "float64": "double"}[dtype]
    kern = compiler.get_raw_kernel("selftest", f"selftest_scale<{ctype}>")
    np_dtype = cp.dtype(dtype)
    block = 256

    def launch(x: Any, alpha: float) -> Any:
        x = cp.ascontiguousarray(x)
        if x.dtype != np_dtype:
            raise TypeError(f"expected {np_dtype} input, got {x.dtype}")
        y = cp.empty_like(x)
        n = x.size
        grid = (n + block - 1) // block
        kern((grid,), (block,), (x, y, np_dtype.type(alpha), cp.int64(n)))
        return y

    return launch


# name -> wrapper factory. Factories may raise; get_kernel translates any
# failure into the warn-once-and-return-None fallback contract.
_KERNEL_FACTORIES: dict[str, Callable[..., Callable]] = {
    "selftest_scale": _selftest_scale_factory,
}
