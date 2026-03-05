"""
Signal I/O and file management utilities.

Functions
---------
save_npz(signal, path, *, compressed=True, include_cache=False)
    Save a Signal to a compressed NumPy archive (.npz).

load_npz(path, *, device="cpu") -> Signal
    Load a Signal from a .npz archive written by save_npz.

File layout
-----------
The .npz file contains the following named entries:

  ``samples``          – IQ sample array  (always present)
  ``source_bits``      – source bit array  (omitted if None)
  ``source_symbols``   – source symbol array  (omitted if None)
  ``resolved_symbols`` – cached symbol array  (only with include_cache=True)
  ``resolved_bits``    – cached bit array     (only with include_cache=True)
  ``__metadata__``     – zero-d object array holding a YAML string with all
                         scalar fields and the serialised SignalInfo dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import yaml

from . import backend as _backend

if TYPE_CHECKING:
    from .core import Signal

# -----------------------------------------------------------------------------
# Internal constants
# -----------------------------------------------------------------------------

# Scalar / primitive metadata fields to round-trip through YAML
_META_FIELDS: tuple[str, ...] = (
    "sampling_rate",
    "symbol_rate",
    "mod_scheme",
    "mod_order",
    "mod_unipolar",
    "mod_rz",
    "pulse_shape",
    "filter_span",
    "rrc_rolloff",
    "rc_rolloff",
    "gaussian_bt",
    "smoothrect_bt",
    "spectral_domain",
    "physical_domain",
    "center_frequency",
    "digital_frequency_offset",
)

# Optional array fields (not always present)
_OPTIONAL_ARRAY_FIELDS: tuple[str, ...] = ("source_bits", "source_symbols")

# Derived / cached array fields (only written when include_cache=True)
_CACHE_FIELDS: tuple[str, ...] = ("resolved_symbols", "resolved_bits")


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def save_npz(
    signal: Signal,
    path: Union[str, Path],
    *,
    compressed: bool = True,
    include_cache: bool = False,
) -> None:
    """
    Save a :class:`~commstools.Signal` to a NumPy archive (.npz).

    Parameters
    ----------
    signal : Signal
        The signal to persist.
    path : str or Path
        Destination path.  A ``.npz`` extension is appended automatically
        if absent.
    compressed : bool, default True
        Use :func:`numpy.savez_compressed` (zlib).  Set to ``False`` to use
        the uncompressed :func:`numpy.savez` (faster write, larger file).
    include_cache : bool, default False
        Also save ``resolved_symbols`` and ``resolved_bits`` if present.
        These can be recomputed from the signal, so they are omitted by
        default to keep file sizes small.

    Examples
    --------
    >>> save_npz(sig, "capture.npz")
    >>> save_npz(sig, "capture", compressed=False, include_cache=True)
    """
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")

    # -------------------------------------------------------------------------
    # Collect arrays
    # -------------------------------------------------------------------------
    arrays: dict[str, np.ndarray] = {"samples": _backend.to_device(signal.samples, "CPU")}

    for field in _OPTIONAL_ARRAY_FIELDS:
        arr = getattr(signal, field, None)
        if arr is not None:
            arrays[field] = _backend.to_device(arr, "CPU")

    if include_cache:
        for field in _CACHE_FIELDS:
            arr = getattr(signal, field, None)
            if arr is not None:
                arrays[field] = _backend.to_device(arr, "CPU")

    # -------------------------------------------------------------------------
    # Build metadata dict and serialise to YAML
    # -------------------------------------------------------------------------
    meta: dict = {f: getattr(signal, f) for f in _META_FIELDS}

    if signal.signal_info is not None:
        meta["signal_info"] = signal.signal_info.model_dump()
    else:
        meta["signal_info"] = None

    yaml_str = yaml.dump(meta, default_flow_style=False, allow_unicode=True)
    # Store as a zero-d object array so np.savez treats it as a single entry
    arrays["__metadata__"] = np.array(yaml_str, dtype=object)

    # -------------------------------------------------------------------------
    # Write
    # -------------------------------------------------------------------------
    save_fn = np.savez_compressed if compressed else np.savez
    save_fn(path, **arrays)


def load_npz(
    path: Union[str, Path],
    *,
    device: str = "auto",
) -> Signal:
    """
    Load a :class:`~commstools.Signal` from a .npz archive.

    Parameters
    ----------
    path : str or Path
        Path to the ``.npz`` file.  A ``.npz`` extension is appended
        automatically if absent.
    device : {"auto", "cpu", "gpu"}, default "auto"
        Target device after loading.  ``"auto"`` moves to GPU when CuPy is
        available, otherwise stays on CPU.

    Returns
    -------
    Signal

    Examples
    --------
    >>> sig = load_npz("capture.npz")           # auto: GPU if available
    >>> sig_cpu = load_npz("capture.npz", device="cpu")
    >>> sig_gpu = load_npz("capture.npz", device="gpu")
    """
    from .core import Signal, SignalInfo

    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")

    # allow_pickle=True is required to read the zero-d object array that
    # holds the YAML string; no arbitrary Python objects are loaded.
    data = np.load(path, allow_pickle=True)

    # -------------------------------------------------------------------------
    # Parse YAML metadata
    # -------------------------------------------------------------------------
    yaml_str = str(data["__metadata__"])
    meta: dict = yaml.safe_load(yaml_str)

    signal_info = None
    if meta.get("signal_info") is not None:
        signal_info = SignalInfo(**meta["signal_info"])

    # -------------------------------------------------------------------------
    # Build Signal constructor kwargs
    # -------------------------------------------------------------------------
    kwargs: dict = {f: meta.get(f) for f in _META_FIELDS}
    kwargs["samples"] = data["samples"]
    kwargs["signal_info"] = signal_info

    for field in _OPTIONAL_ARRAY_FIELDS:
        if field in data:
            kwargs[field] = data[field]

    sig = Signal(**kwargs)

    # -------------------------------------------------------------------------
    # Restore cached arrays (bypass re-computation if present in file)
    # -------------------------------------------------------------------------
    for field in _CACHE_FIELDS:
        if field in data:
            setattr(sig, field, data[field])

    target = device.lower()
    if target == "auto":
        target = "gpu" if _backend.is_cupy_available() else "cpu"
    sig = sig.to(target)
    return sig
