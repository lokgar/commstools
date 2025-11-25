from ..core.backend import get_backend, ArrayType


def ook_map(bits: ArrayType) -> ArrayType:
    """
    Maps bits to OOK symbols.
    0 -> 0
    1 -> 1

    Args:
        bits: Array of bits (0s and 1s).

    Returns:
        Array of symbols (complex or float).
    """
    backend = get_backend()
    # For OOK, the mapping is identity, but we ensure type is float/complex for signal processing
    return backend.asarray(bits, dtype=float)
