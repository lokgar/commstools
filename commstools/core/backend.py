from typing import Any, Optional, Protocol, Union, Tuple

import numpy as np

# Try to import CuPy, but don't fail if it's not available
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cpx_ndimage
    import cupyx.scipy.signal as cpx_signal

    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False
    cp = None
    cpx_signal = None
    cpx_ndimage = None

ArrayType = Union[
    np.ndarray, Any
]  # Any for CuPy array to avoid hard dependency in type hint if not installed


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
    def sin(self, x: ArrayType) -> ArrayType: ...
    def cos(self, x: ArrayType) -> ArrayType: ...
    def sinc(self, x: ArrayType) -> ArrayType: ...
    def isclose(
        self, a: ArrayType, b: Any, rtol: float = 1e-05, atol: float = 1e-08
    ) -> ArrayType: ...
    def full(self, shape: Any, fill_value: Any, dtype: Any = None) -> ArrayType: ...
    def clip(self, a: ArrayType, a_min: Any, a_max: Any) -> ArrayType: ...
    def zeros_like(self, a: ArrayType, dtype: Any = None) -> ArrayType: ...
    def ones_like(self, a: ArrayType, dtype: Any = None) -> ArrayType: ...

    @property
    def pi(self) -> float: ...

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
    def where(self, condition: ArrayType, x: ArrayType, y: ArrayType) -> ArrayType: ...

    def fft(
        self, x: ArrayType, n: Optional[int] = None, axis: int = -1
    ) -> ArrayType: ...
    def ifft(
        self, x: ArrayType, n: Optional[int] = None, axis: int = -1
    ) -> ArrayType: ...
    def fftshift(self, x: ArrayType, axes: Any = None) -> ArrayType: ...
    def ifftshift(self, x: ArrayType, axes: Any = None) -> ArrayType: ...
    def fftfreq(self, n: int, d: float = 1.0) -> ArrayType: ...
    def convolve(
        self, in1: ArrayType, in2: ArrayType, mode: str = "full", method: str = "auto"
    ) -> ArrayType: ...
    def expand(self, x: ArrayType, factor: int) -> ArrayType: ...
    def decimate(
        self, x: ArrayType, factor: int, ftype: str = "fir", zero_phase: bool = True
    ) -> ArrayType: ...
    def resample_poly(self, x: ArrayType, up: int, down: int) -> ArrayType: ...
    def blackman(self, M: int) -> ArrayType: ...
    def hamming(self, M: int) -> ArrayType: ...
    def firwin(
        self,
        numtaps: int,
        cutoff: Any,
        window: str = "hamming",
        pass_zero: bool = True,
        scale: bool = True,
        fs: Optional[float] = None,
    ) -> ArrayType: ...
    def freqz(
        self,
        b: ArrayType,
        a: Any = 1,
        worN: Optional[int] = None,
        whole: bool = False,
        fs: Optional[float] = None,
    ) -> Tuple[ArrayType, ArrayType]: ...
    def gaussian_filter(
        self, input: ArrayType, sigma: float, order: int = 0, mode: str = "reflect"
    ) -> ArrayType: ...
    def welch(
        self,
        x: ArrayType,
        fs: Optional[float] = 1.0,
        window: Optional[str] = "hann",
        nperseg: Optional[int] = None,
        noverlap: Optional[int] = None,
        nfft: Optional[int] = None,
        detrend: Optional[Union[str, bool]] = "constant",
        return_onesided: Optional[bool] = True,
        scaling: Optional[str] = "density",
        axis: Optional[int] = -1,
        average: Optional[str] = "mean",
    ) -> ArrayType: ...
    def iscomplexobj(self, x: ArrayType) -> bool: ...


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

    def sin(self, x: ArrayType) -> ArrayType:
        return np.sin(x)

    def cos(self, x: ArrayType) -> ArrayType:
        return np.cos(x)

    def sinc(self, x: ArrayType) -> ArrayType:
        return np.sinc(x)

    def isclose(
        self, a: ArrayType, b: Any, rtol: float = 1e-05, atol: float = 1e-08
    ) -> ArrayType:
        return np.isclose(a, b, rtol=rtol, atol=atol)

    def full(self, shape: Any, fill_value: Any, dtype: Any = None) -> ArrayType:
        return np.full(shape, fill_value, dtype=dtype)

    def clip(self, a: ArrayType, a_min: Any, a_max: Any) -> ArrayType:
        return np.clip(a, a_min, a_max)

    def zeros_like(self, a: ArrayType, dtype: Any = None) -> ArrayType:
        return np.zeros_like(a, dtype=dtype)

    def ones_like(self, a: ArrayType, dtype: Any = None) -> ArrayType:
        return np.ones_like(a, dtype=dtype)

    @property
    def pi(self) -> float:
        return np.pi

    def sum(self, x: ArrayType, axis: Any = None, keepdims: bool = False) -> ArrayType:
        return np.sum(x, axis=axis, keepdims=keepdims)

    def mean(self, x: ArrayType, axis: Any = None, keepdims: bool = False) -> ArrayType:
        return np.mean(x, axis=axis, keepdims=keepdims)

    def max(self, x: ArrayType, axis: Any = None, keepdims: bool = False) -> ArrayType:
        return np.max(x, axis=axis, keepdims=keepdims)

    def min(self, x: ArrayType, axis: Any = None, keepdims: bool = False) -> ArrayType:
        return np.min(x, axis=axis, keepdims=keepdims)

    def where(self, condition: ArrayType, x: ArrayType, y: ArrayType) -> ArrayType:
        return np.where(condition, x, y)

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

    def convolve(
        self, in1: ArrayType, in2: ArrayType, mode: str = "full", method: str = "auto"
    ) -> ArrayType:
        import scipy.signal

        return scipy.signal.convolve(in1, in2, mode=mode, method=method)

    def expand(self, x: ArrayType, factor: int) -> ArrayType:
        """Zero-insertion: Insert (factor-1) zeros between samples."""
        n_in = x.shape[0]
        n_out = n_in * factor
        out = np.zeros(n_out, dtype=x.dtype)
        out[::factor] = x
        return out

    def decimate(
        self, x: ArrayType, factor: int, ftype: str = "fir", zero_phase: bool = True
    ) -> ArrayType:
        """Decimate signal: Anti-aliasing filter + downsample."""
        import scipy.signal

        return scipy.signal.decimate(x, factor, ftype=ftype, zero_phase=zero_phase)

    def resample_poly(self, x: ArrayType, up: int, down: int) -> ArrayType:
        """Resample signal using polyphase filtering."""
        import scipy.signal

        return scipy.signal.resample_poly(x, up, down)

    def blackman(self, M: int) -> ArrayType:
        """Return a Blackman window of length M."""
        return np.blackman(M)

    def hamming(self, M: int) -> ArrayType:
        """Return a Hamming window of length M."""
        return np.hamming(M)

    def firwin(
        self,
        numtaps: int,
        cutoff: Any,
        window: str = "hamming",
        pass_zero: bool = True,
        scale: bool = True,
        fs: Optional[float] = None,
    ) -> ArrayType:
        """Design FIR filter using window method."""
        import scipy.signal

        return scipy.signal.firwin(
            numtaps, cutoff, window=window, pass_zero=pass_zero, scale=scale, fs=fs
        )

    def freqz(
        self,
        b: ArrayType,
        a: Any = 1,
        worN: Optional[int] = None,
        whole: bool = False,
        fs: Optional[float] = None,
    ) -> tuple:
        """Compute frequency response of a digital filter."""
        import scipy.signal

        # scipy.signal.freqz uses fs=2*pi as default (angular frequency)
        # Don't pass fs=None explicitly to avoid issues
        if fs is None:
            return scipy.signal.freqz(b, a, worN=worN, whole=whole)
        return scipy.signal.freqz(b, a, worN=worN, whole=whole, fs=fs)

    def gaussian_filter(
        self, input: ArrayType, sigma: float, order: int = 0, mode: str = "reflect"
    ) -> ArrayType:
        """Apply Gaussian filter."""
        import scipy.ndimage

        return scipy.ndimage.gaussian_filter(input, sigma=sigma, order=order, mode=mode)

    def welch(
        self,
        x: ArrayType,
        fs: Optional[float] = 1.0,
        window: Optional[str] = "hann",
        nperseg: Optional[int] = None,
        noverlap: Optional[int] = None,
        nfft: Optional[int] = None,
        detrend: Optional[Union[str, bool]] = "constant",
        return_onesided: Optional[bool] = True,
        scaling: Optional[str] = "density",
        axis: Optional[int] = -1,
        average: Optional[str] = "mean",
    ) -> ArrayType:
        import scipy.signal

        return scipy.signal.welch(
            x,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            detrend=detrend,
            return_onesided=return_onesided,
            scaling=scaling,
            axis=axis,
            average=average,
        )

    def iscomplexobj(self, x: ArrayType) -> bool:
        return np.iscomplexobj(x)


class CupyBackend:
    """CuPy implementation of the Backend protocol."""

    def __init__(self) -> None:
        if not _CUPY_AVAILABLE:
            raise ImportError(
                "CuPy is not available. Please install it to use CupyBackend."
            )

    @property
    def name(self) -> str:
        return "cupy"

    def array(self, data: Any, dtype: Any = None) -> ArrayType:
        return cp.array(data, dtype=dtype)

    def asarray(self, data: Any, dtype: Any = None) -> ArrayType:
        return cp.asarray(data, dtype=dtype)

    def zeros(self, shape: Any, dtype: Any = None) -> ArrayType:
        return cp.zeros(shape, dtype=dtype)

    def ones(self, shape: Any, dtype: Any = None) -> ArrayType:
        return cp.ones(shape, dtype=dtype)

    def arange(
        self, start: Any, stop: Any = None, step: Any = None, dtype: Any = None
    ) -> ArrayType:
        return cp.arange(start, stop, step, dtype=dtype)

    def linspace(
        self, start: Any, stop: Any, num: int, endpoint: bool = True, dtype: Any = None
    ) -> ArrayType:
        return cp.linspace(start, stop, num, endpoint=endpoint, dtype=dtype)

    def exp(self, x: ArrayType) -> ArrayType:
        return cp.exp(x)

    def log(self, x: ArrayType) -> ArrayType:
        return cp.log(x)

    def log10(self, x: ArrayType) -> ArrayType:
        return cp.log10(x)

    def sqrt(self, x: ArrayType) -> ArrayType:
        return cp.sqrt(x)

    def abs(self, x: ArrayType) -> ArrayType:
        return cp.abs(x)

    def angle(self, x: ArrayType) -> ArrayType:
        return cp.angle(x)

    def conj(self, x: ArrayType) -> ArrayType:
        return cp.conj(x)

    def real(self, x: ArrayType) -> ArrayType:
        return cp.real(x)

    def imag(self, x: ArrayType) -> ArrayType:
        return cp.imag(x)

    def sin(self, x: ArrayType) -> ArrayType:
        return cp.sin(x)

    def cos(self, x: ArrayType) -> ArrayType:
        return cp.cos(x)

    def sinc(self, x: ArrayType) -> ArrayType:
        return cp.sinc(x)

    def isclose(
        self, a: ArrayType, b: Any, rtol: float = 1e-05, atol: float = 1e-08
    ) -> ArrayType:
        return cp.isclose(a, b, rtol=rtol, atol=atol)

    def full(self, shape: Any, fill_value: Any, dtype: Any = None) -> ArrayType:
        return cp.full(shape, fill_value, dtype=dtype)

    def clip(self, a: ArrayType, a_min: Any, a_max: Any) -> ArrayType:
        return cp.clip(a, a_min, a_max)

    def zeros_like(self, a: ArrayType, dtype: Any = None) -> ArrayType:
        return cp.zeros_like(a, dtype=dtype)

    def ones_like(self, a: ArrayType, dtype: Any = None) -> ArrayType:
        return cp.ones_like(a, dtype=dtype)

    @property
    def pi(self) -> float:
        return cp.pi

    def sum(self, x: ArrayType, axis: Any = None, keepdims: bool = False) -> ArrayType:
        return cp.sum(x, axis=axis, keepdims=keepdims)

    def mean(self, x: ArrayType, axis: Any = None, keepdims: bool = False) -> ArrayType:
        return cp.mean(x, axis=axis, keepdims=keepdims)

    def max(self, x: ArrayType, axis: Any = None, keepdims: bool = False) -> ArrayType:
        return cp.max(x, axis=axis, keepdims=keepdims)

    def min(self, x: ArrayType, axis: Any = None, keepdims: bool = False) -> ArrayType:
        return cp.min(x, axis=axis, keepdims=keepdims)

    def where(self, condition: ArrayType, x: ArrayType, y: ArrayType) -> ArrayType:
        return cp.where(condition, x, y)

    def fft(self, x: ArrayType, n: Optional[int] = None, axis: int = -1) -> ArrayType:
        return cp.fft.fft(x, n=n, axis=axis)

    def ifft(self, x: ArrayType, n: Optional[int] = None, axis: int = -1) -> ArrayType:
        return cp.fft.ifft(x, n=n, axis=axis)

    def fftshift(self, x: ArrayType, axes: Any = None) -> ArrayType:
        return cp.fft.fftshift(x, axes=axes)

    def ifftshift(self, x: ArrayType, axes: Any = None) -> ArrayType:
        return cp.fft.ifftshift(x, axes=axes)

    def fftfreq(self, n: int, d: float = 1.0) -> ArrayType:
        return cp.fft.fftfreq(n, d=d)

    def convolve(
        self, in1: ArrayType, in2: ArrayType, mode: str = "full", method: str = "auto"
    ) -> ArrayType:
        return cpx_signal.convolve(in1, in2, mode=mode, method=method)

    def expand(self, x: ArrayType, factor: int) -> ArrayType:
        """Zero-insertion: Insert (factor-1) zeros between samples."""
        n_in = x.shape[0]
        n_out = n_in * factor
        out = cp.zeros(n_out, dtype=x.dtype)
        out[::factor] = x
        return out

    def decimate(
        self, x: ArrayType, factor: int, ftype: str = "fir", zero_phase: bool = True
    ) -> ArrayType:
        """Decimate signal: Anti-aliasing filter + downsample."""
        return cpx_signal.decimate(x, factor, ftype=ftype, zero_phase=zero_phase)

    def resample_poly(self, x: ArrayType, up: int, down: int) -> ArrayType:
        """Resample signal using polyphase filtering."""
        return cpx_signal.resample_poly(x, up, down)

    def blackman(self, M: int) -> ArrayType:
        """Return a Blackman window of length M."""
        return cp.blackman(M)

    def hamming(self, M: int) -> ArrayType:
        """Return a Hamming window of length M."""
        return cp.hamming(M)

    def firwin(
        self,
        numtaps: int,
        cutoff: Any,
        window: str = "hamming",
        pass_zero: bool = True,
        scale: bool = True,
        fs: Optional[float] = None,
    ) -> ArrayType:
        """Design FIR filter using window method."""
        return cpx_signal.firwin(
            numtaps, cutoff, window=window, pass_zero=pass_zero, scale=scale, fs=fs
        )

    def freqz(
        self,
        b: ArrayType,
        a: Any = 1,
        worN: Optional[int] = None,
        whole: bool = False,
        fs: Optional[float] = None,
    ) -> tuple:
        """Compute frequency response of a digital filter."""
        # cupyx.scipy.signal.freqz has issues with fs=None
        # When fs is None, don't pass it (uses default angular frequency)
        if fs is None:
            return cpx_signal.freqz(b, a, worN=worN, whole=whole)
        return cpx_signal.freqz(b, a, worN=worN, whole=whole, fs=fs)

    def gaussian_filter(
        self, input: ArrayType, sigma: float, order: int = 0, mode: str = "reflect"
    ) -> ArrayType:
        """Apply Gaussian filter."""
        return cpx_ndimage.gaussian_filter(input, sigma=sigma, order=order, mode=mode)

    def welch(
        self,
        x: ArrayType,
        fs: Optional[float] = 1.0,
        window: Optional[str] = "hann",
        nperseg: Optional[int] = None,
        noverlap: Optional[int] = None,
        nfft: Optional[int] = None,
        detrend: Optional[Union[str, bool]] = "constant",
        return_onesided: Optional[bool] = True,
        scaling: Optional[str] = "density",
        axis: Optional[int] = -1,
        average: Optional[str] = "mean",
    ) -> ArrayType:
        return cpx_signal.welch(
            x,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            detrend=detrend,
            return_onesided=return_onesided,
            scaling=scaling,
            axis=axis,
            average=average,
        )

    def iscomplexobj(self, x: ArrayType) -> bool:
        return cp.iscomplexobj(x)


# Global state for current backend
_CURRENT_BACKEND: Backend = NumpyBackend()


def get_backend() -> Backend:
    """Get the currently active global backend."""
    return _CURRENT_BACKEND


def set_backend(backend_name: str) -> None:
    """
    Set the active backend globally.

    Args:
        backend_name (str): The name of the backend to set: "cpu" or "gpu".
    """
    global _CURRENT_BACKEND
    if backend_name.lower() in ("numpy", "cpu"):
        _CURRENT_BACKEND = NumpyBackend()
    elif backend_name.lower() in ("cupy", "gpu", "cuda"):
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
    if backend.name == "numpy":
        if isinstance(data, np.ndarray):
            return data
        if _CUPY_AVAILABLE and isinstance(data, cp.ndarray):
            return cp.asnumpy(data)
        return np.asarray(data)

    elif backend.name == "cupy":
        if _CUPY_AVAILABLE and isinstance(data, cp.ndarray):
            return data
        # If CuPy is active, we expect CuPy to be available
        return cp.asarray(data)

    return backend.asarray(data)


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
