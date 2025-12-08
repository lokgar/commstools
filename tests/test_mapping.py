import pytest
import numpy as np
from commstools import mapping


def test_qam_mapping(backend_device, xp):
    bits = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # 2 symbols for QAM16 (4 bits each)

    if backend_device == "gpu":
        from commstools.backend import set_backend

        set_backend("gpu")
        # mapping usually takes inputs and returns backend arrays?
        # or takes inputs on backend.
        # Let's pass numpy bits, see if it handles it or we need backend bits.
        bits = xp.asarray(bits)

    syms = mapping.map_bits(bits, modulation="qam", order=16)
    assert isinstance(syms, xp.ndarray)
    assert len(syms) == 2

    if backend_device == "gpu":
        from commstools.backend import set_backend

        set_backend("cpu")


def test_psk_mapping(backend_device, xp):
    bits = np.array([0, 1])  # 2 symbols for BPSK

    if backend_device == "gpu":
        from commstools.backend import set_backend

        set_backend("gpu")
        bits = xp.asarray(bits)

    syms = mapping.map_bits(bits, modulation="psk", order=2)
    assert len(syms) == 2

    if backend_device == "gpu":
        from commstools.backend import set_backend

        set_backend("cpu")
