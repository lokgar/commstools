"""Tests for random bit and symbol sequence generation."""

from commstools import helpers


def test_random_bits(backend_device, xp):
    """Verify that random bit generation produces expected length and values."""
    bits = helpers.random_bits(100, seed=42)
    assert len(bits) == 100
    assert xp.all((bits == 0) | (bits == 1))
    assert isinstance(bits, xp.ndarray)
