from ..core.backend import ArrayType, get_backend, ensure_on_backend


def normalize(x: ArrayType, mode: str = "unity_gain") -> ArrayType:
    """
    Normalize array based on the specified mode.

    Args:
        x: Input array.
        mode: Normalization mode.
            'unity_gain': Sum of elements is 1.
            'unit_energy': Sum of squared magnitudes is 1.
            'max_amplitude': Maximum absolute value is 1.
            'average_power': Mean of squared magnitudes is 1.

    Returns:
        Normalized array.

    Raises:
        ValueError: If the normalization factor is zero (Numpy only).
    """
    x = ensure_on_backend(x)
    backend = get_backend()

    if mode == "unity_gain":
        norm_factor = backend.sum(x)
    elif mode == "unit_energy":
        norm_factor = backend.sqrt(backend.sum(backend.abs(x) ** 2))
    elif mode == "max_amplitude":
        norm_factor = backend.max(backend.abs(x))
    elif mode == "average_power":
        norm_factor = backend.sqrt(backend.mean(backend.abs(x) ** 2))
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    # Handle division by zero safely for both NumPy and CuPy
    # Avoid control flow based on data values to prevent host-device synchronization.
    safe_norm = backend.where(norm_factor == 0, 1.0, norm_factor)
    result = x / safe_norm

    # Zero out result if norm was 0 (since 0/0 or x/0 should be handled)
    # If norm_factor is 0, it means the signal is all zeros (for energy/power/max/sum)
    # So the normalized result should also be all zeros.
    return backend.where(
        norm_factor == 0, backend.zeros(x.shape, dtype=x.dtype), result
    )
