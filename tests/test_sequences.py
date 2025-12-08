import pytest
from commstools import sequences, backend


def test_random_bits(backend_device, xp):
    backend.set_backend(backend_device)

    bits = sequences.random_bits(100, seed=42)
    assert isinstance(bits, xp.ndarray)
    assert len(bits) == 100
    # Check values are 0 or 1
    # Use simple boolean logic compatible with both
    assert xp.all((bits == 0) | (bits == 1))
