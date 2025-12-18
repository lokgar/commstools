import pytest
import numpy as np
from commstools import mapping, backend


def test_qam_mapping(backend_device, xp):
    bits = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Ensure bits are on device
    bits = backend.to_device(bits, backend_device)

    syms = mapping.map_bits(bits, modulation="qam", order=16)

    assert isinstance(syms, xp.ndarray)
    assert len(syms) == 2


def test_psk_mapping(backend_device, xp):
    bits = np.array([0, 1])
    bits = backend.to_device(bits, backend_device)

    syms = mapping.map_bits(bits, modulation="psk", order=2)

    assert isinstance(syms, xp.ndarray)
    assert len(syms) == 2
