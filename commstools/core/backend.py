from typing import Any, Protocol, Union, Optional, Callable, TypeVar
import numpy as np
import contextlib

# Try to import JAX, but don't fail if it's not available
try:
    import jax
    import jax.numpy as jnp

    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False
    jax = None
    jnp = None

ArrayType = Union[
    np.ndarray, Any
]  # Any for JAX array to avoid hard dependency in type hint if not installed


class Backend(Protocol):
    """Protocol defining the interface for computational backends."""

    @property
    def name(self) -> str: ...

    def array(self, data: Any, dtype: Any = None) -> ArrayType: ...
    def asarray(self, data: Any, dtype: Any = None) -> ArrayType: ...
    def zeros(self, shape: Any, dtype: Any = None) -> ArrayType: ...
    def ones(self, shape: Any, dtype: Any = None) -> ArrayType: ...
    def arange(
        self, start: Any, stop: Any = None, step: Any = None, dtype: Any = None
    ) -> ArrayType: ...
    def linspace(
        self, start: Any, stop: Any, num: int, endpoint: bool = True, dtype: Any = None
    ) -> ArrayType: ...

    def exp(self, x: ArrayType) -> ArrayType: ...
    def log(self, x: ArrayType) -> ArrayType: ...
    def log10(self, x: ArrayType) -> ArrayType: ...
    def sqrt(self, x: ArrayType) -> ArrayType: ...
    def abs(self, x: ArrayType) -> ArrayType: ...
    def angle(self, x: ArrayType) -> ArrayType: ...
    def conj(self, x: ArrayType) -> ArrayType: ...
    def real(self, x: ArrayType) -> ArrayType: ...
    def imag(self, x: ArrayType) -> ArrayType: ...

    def sum(
        self, x: ArrayType, axis: Any = None, keepdims: bool = False
    ) -> ArrayType: ...
    def mean(
        self, x: ArrayType, axis: Any = None, keepdims: bool = False
    ) -> ArrayType: ...
    def max(
        self, x: ArrayType, axis: Any = None, keepdims: bool = False
    ) -> ArrayType: ...
    def min(
        self, x: ArrayType, axis: Any = None, keepdims: bool = False
    ) -> ArrayType: ...

    def fft(
        self, x: ArrayType, n: Optional[int] = None, axis: int = -1
    ) -> ArrayType: ...
    def ifft(
        self, x: ArrayType, n: Optional[int] = None, axis: int = -1
    ) -> ArrayType: ...
    def fftshift(self, x: ArrayType, axes: Any = None) -> ArrayType: ...
    def ifftshift(self, x: ArrayType, axes: Any = None) -> ArrayType: ...
    def fftfreq(self, n: int, d: float = 1.0) -> ArrayType: ...

    def jit(
        self, fun: Callable, static_argnums: Optional[Union[int, tuple]] = None
    ) -> Callable: ...


class NumpyBackend:
    """NumPy implementation of the Backend protocol."""

    @property
    def name(self) -> str:
        return "numpy"

    def array(self, data: Any, dtype: Any = None) -> ArrayType:
        return np.array(data, dtype=dtype)

    def asarray(self, data: Any, dtype: Any = None) -> ArrayType:
        return np.asarray(data, dtype=dtype)

    def zeros(self, shape: Any, dtype: Any = None) -> ArrayType:
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape: Any, dtype: Any = None) -> ArrayType:
        return np.ones(shape, dtype=dtype)

    def arange(
        self, start: Any, stop: Any = None, step: Any = None, dtype: Any = None
    ) -> ArrayType:
        return np.arange(start, stop, step, dtype=dtype)

    def linspace(
        self, start: Any, stop: Any, num: int, endpoint: bool = True, dtype: Any = None
    ) -> ArrayType:
        return np.linspace(start, stop, num, endpoint=endpoint, dtype=dtype)

    def exp(self, x: ArrayType) -> ArrayType:
        return np.exp(x)

    def log(self, x: ArrayType) -> ArrayType:
        return np.log(x)

    def log10(self, x: ArrayType) -> ArrayType:
        return np.log10(x)

    def sqrt(self, x: ArrayType) -> ArrayType:
        return np.sqrt(x)

    def abs(self, x: ArrayType) -> ArrayType:
        return np.abs(x)

    def angle(self, x: ArrayType) -> ArrayType:
        return np.angle(x)

    def conj(self, x: ArrayType) -> ArrayType:
        return np.conj(x)

    def real(self, x: ArrayType) -> ArrayType:
        return np.real(x)

    def imag(self, x: ArrayType) -> ArrayType:
        return np.imag(x)

    def sum(self, x: ArrayType, axis: Any = None, keepdims: bool = False) -> ArrayType:
        return np.sum(x, axis=axis, keepdims=keepdims)

    def mean(self, x: ArrayType, axis: Any = None, keepdims: bool = False) -> ArrayType:
        return np.mean(x, axis=axis, keepdims=keepdims)

    def max(self, x: ArrayType, axis: Any = None, keepdims: bool = False) -> ArrayType:
        return np.max(x, axis=axis, keepdims=keepdims)

    def min(self, x: ArrayType, axis: Any = None, keepdims: bool = False) -> ArrayType:
        return np.min(x, axis=axis, keepdims=keepdims)

    def fft(self, x: ArrayType, n: Optional[int] = None, axis: int = -1) -> ArrayType:
        return np.fft.fft(x, n=n, axis=axis)

    def ifft(self, x: ArrayType, n: Optional[int] = None, axis: int = -1) -> ArrayType:
        return np.fft.ifft(x, n=n, axis=axis)

    def fftshift(self, x: ArrayType, axes: Any = None) -> ArrayType:
        return np.fft.fftshift(x, axes=axes)

    def ifftshift(self, x: ArrayType, axes: Any = None) -> ArrayType:
        return np.fft.ifftshift(x, axes=axes)

    def fftfreq(self, n: int, d: float = 1.0) -> ArrayType:
        return np.fft.fftfreq(n, d=d)

    def jit(
        self, fun: Callable, static_argnums: Optional[Union[int, tuple]] = None
    ) -> Callable:
        # Numpy backend does not support JIT, return function as is
        return fun


class JaxBackend:
    """JAX implementation of the Backend protocol."""

    def __init__(self):
        if not _JAX_AVAILABLE:
            raise ImportError(
                "JAX is not available. Please install it to use JaxBackend."
            )

    @property
    def name(self) -> str:
        return "jax"

    def array(self, data: Any, dtype: Any = None) -> ArrayType:
        return jnp.array(data, dtype=dtype)

    def asarray(self, data: Any, dtype: Any = None) -> ArrayType:
        return jnp.asarray(data, dtype=dtype)

    def zeros(self, shape: Any, dtype: Any = None) -> ArrayType:
        return jnp.zeros(shape, dtype=dtype)

    def ones(self, shape: Any, dtype: Any = None) -> ArrayType:
        return jnp.ones(shape, dtype=dtype)

    def arange(
        self, start: Any, stop: Any = None, step: Any = None, dtype: Any = None
    ) -> ArrayType:
        return jnp.arange(start, stop, step, dtype=dtype)

    def linspace(
        self, start: Any, stop: Any, num: int, endpoint: bool = True, dtype: Any = None
    ) -> ArrayType:
        return jnp.linspace(start, stop, num, endpoint=endpoint, dtype=dtype)

    def exp(self, x: ArrayType) -> ArrayType:
        return jnp.exp(x)

    def log(self, x: ArrayType) -> ArrayType:
        return jnp.log(x)

    def log10(self, x: ArrayType) -> ArrayType:
        return jnp.log10(x)

    def sqrt(self, x: ArrayType) -> ArrayType:
        return jnp.sqrt(x)

    def abs(self, x: ArrayType) -> ArrayType:
        return jnp.abs(x)

    def angle(self, x: ArrayType) -> ArrayType:
        return jnp.angle(x)

    def conj(self, x: ArrayType) -> ArrayType:
        return jnp.conj(x)

    def real(self, x: ArrayType) -> ArrayType:
        return jnp.real(x)

    def imag(self, x: ArrayType) -> ArrayType:
        return jnp.imag(x)

    def sum(self, x: ArrayType, axis: Any = None, keepdims: bool = False) -> ArrayType:
        return jnp.sum(x, axis=axis, keepdims=keepdims)

    def mean(self, x: ArrayType, axis: Any = None, keepdims: bool = False) -> ArrayType:
        return jnp.mean(x, axis=axis, keepdims=keepdims)

    def max(self, x: ArrayType, axis: Any = None, keepdims: bool = False) -> ArrayType:
        return jnp.max(x, axis=axis, keepdims=keepdims)

    def min(self, x: ArrayType, axis: Any = None, keepdims: bool = False) -> ArrayType:
        return jnp.min(x, axis=axis, keepdims=keepdims)

    def fft(self, x: ArrayType, n: Optional[int] = None, axis: int = -1) -> ArrayType:
        return jnp.fft.fft(x, n=n, axis=axis)

    def ifft(self, x: ArrayType, n: Optional[int] = None, axis: int = -1) -> ArrayType:
        return jnp.fft.ifft(x, n=n, axis=axis)

    def fftshift(self, x: ArrayType, axes: Any = None) -> ArrayType:
        return jnp.fft.fftshift(x, axes=axes)

    def ifftshift(self, x: ArrayType, axes: Any = None) -> ArrayType:
        return jnp.fft.ifftshift(x, axes=axes)

    def fftfreq(self, n: int, d: float = 1.0) -> ArrayType:
        return jnp.fft.fftfreq(n, d=d)

    def jit(
        self, fun: Callable, static_argnums: Optional[Union[int, tuple]] = None
    ) -> Callable:
        import jax

        return jax.jit(fun, static_argnums=static_argnums)


# Global state for current backend
_CURRENT_BACKEND: Backend = NumpyBackend()


def get_backend() -> Backend:
    """Get the currently active backend."""
    return _CURRENT_BACKEND


def set_backend(backend_name: str) -> None:
    """Set the active backend globally."""
    global _CURRENT_BACKEND
    if backend_name.lower() == "numpy":
        _CURRENT_BACKEND = NumpyBackend()
    elif backend_name.lower() == "jax":
        _CURRENT_BACKEND = JaxBackend()
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


@contextlib.contextmanager
def using_backend(backend_name: str):
    """Context manager to temporarily switch backend."""
    global _CURRENT_BACKEND
    prev_backend = _CURRENT_BACKEND
    try:
        set_backend(backend_name)
        yield
    finally:
        _CURRENT_BACKEND = prev_backend
