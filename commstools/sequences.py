"""
Random sequence generation.

This module provides functions to generate random and pseudo-random bit sequences:
- Random bits (uniform distribution).
- Pseudo-Random Binary Sequences (PRBS) using LFSRs.
"""
# For reproducibility we generate the random bits on the CPU,
# as sequences differ between backends with given seed.

import numpy as np

from typing import Optional

from .backend import ArrayType, ensure_on_backend


def random_bits(length: int, seed: Optional[int] = None) -> ArrayType:
    """
    Generates a sequence of random bits (0s and 1s).

    Args:
        length: Length of the sequence to generate.
        seed: Random seed for reproducibility.

    Returns:
        Array of bits (0s and 1s) on the active backend.
    """
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=length)
    return ensure_on_backend(bits)


def prbs(length: int, seed: int = 0x7F, order: int = 7) -> ArrayType:
    """
    Generates a Pseudo-Random Binary Sequence (PRBS).

    Note: This function currently uses CPU-only numpy implementation for LFSR bitwise operations.
    The result is then transferred to the active backend.

    Args:
        length: Length of the sequence to generate.
        seed: Initial state of the shift register.
        order: Order of the PRBS (e.g., 7 for PRBS7).

    Returns:
        Array of bits (0s and 1s) on the active backend.
    """
    # Feedback taps for common PRBS orders
    taps = {7: (6, 5), 9: (8, 4), 11: (10, 8), 15: (14, 13), 23: (22, 17), 31: (30, 27)}

    if order not in taps:
        raise ValueError(
            f"Unsupported PRBS order: {order}. Supported: {list(taps.keys())}"
        )

    tap1, tap2 = taps[order]

    # Simple LFSR implementation using numpy
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

    # Transfer to active backend
    return ensure_on_backend(seq)
