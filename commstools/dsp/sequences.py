import numpy as np
from typing import Optional
from ..core.backend import get_backend, ArrayType


def prbs(length: int, seed: int = 0x1, order: int = 7) -> ArrayType:
    """
    Generates a Pseudo-Random Binary Sequence (PRBS).

    Args:
        length: Length of the sequence to generate.
        seed: Initial state of the shift register.
        order: Order of the PRBS (e.g., 7 for PRBS7).

    Returns:
        Array of bits (0s and 1s).
    """
    backend = get_backend()

    # Generate using numpy for simplicity as bitwise ops are standard
    # Then convert to backend array

    # Feedback taps for common PRBS orders
    taps = {7: (6, 5), 9: (8, 4), 11: (10, 8), 15: (14, 13), 23: (22, 17), 31: (30, 27)}

    if order not in taps:
        raise ValueError(
            f"Unsupported PRBS order: {order}. Supported: {list(taps.keys())}"
        )

    tap1, tap2 = taps[order]

    # Simple LFSR implementation
    # Note: For very long sequences, a more optimized approach or pre-computed table might be better.
    # But for typical comms simulation lengths, this is fine.

    seq = np.zeros(length, dtype=int)
    state = seed

    for i in range(length):
        # Output is the LSB (or MSB, convention varies, using LSB here)
        out = state & 1
        seq[i] = out

        # Feedback
        fb = ((state >> tap1) ^ (state >> tap2)) & 1

        # Shift
        state = (state >> 1) | (fb << (order - 1))

    return backend.array(seq)
