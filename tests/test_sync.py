"""Tests for synchronization utilities module."""

import numpy as np
from commstools import sync
from commstools.core import Preamble


def test_barker_sequences(backend_device, xp):
    """Verify all standard Barker sequence lengths."""
    valid_lengths = [2, 3, 4, 5, 7, 11, 13]

    for length in valid_lengths:
        seq = sync.barker_sequence(length)
        assert len(seq) == length
        # All values should be +1 or -1
        assert xp.all((seq == 1) | (seq == -1))


def test_barker_invalid_length(backend_device, xp):
    """Invalid Barker length should raise error."""
    import pytest

    with pytest.raises(ValueError):
        sync.barker_sequence(6)  # No Barker-6 exists


def test_barker_autocorrelation(backend_device, xp):
    """Barker sequences have optimal autocorrelation."""
    seq = sync.barker_sequence(13)

    # Auto-correlation via correlate
    acorr = sync.correlate(seq, seq, mode="full", normalize=False)

    # Peak should be at center
    peak_idx = len(acorr) // 2
    peak_val = float(xp.abs(acorr[peak_idx]))

    # Sidelobes should be at most 1
    sidelobes = xp.abs(acorr)
    sidelobes_max = float(
        xp.max(xp.concatenate([sidelobes[:peak_idx], sidelobes[peak_idx + 1 :]]))
    )

    assert peak_val == 13.0  # Sum of squared elements
    assert sidelobes_max <= 1.0 + 1e-5  # Barker property (with float tolerance)


def test_zadoff_chu_cazac(backend_device, xp):
    """ZC sequences have constant amplitude (CAZAC)."""
    zc = sync.zadoff_chu_sequence(63, root=25)

    # All magnitudes should be 1
    magnitudes = xp.abs(zc)
    assert xp.allclose(magnitudes, 1.0, atol=1e-5)


def test_zadoff_chu_length(backend_device, xp):
    """ZC sequence has correct length."""
    for length in [31, 63, 127]:
        zc = sync.zadoff_chu_sequence(length, root=1)
        assert len(zc) == length


def test_correlate_delta(backend_device, xp):
    """Correlation peak should be at correct location."""
    # Delta-like signal
    signal = xp.zeros(100, dtype=xp.float32)
    signal[50] = 1.0

    template = xp.array([1.0], dtype=xp.float32)

    corr = sync.correlate(signal, template, mode="same")

    # Peak should be at position 50
    peak_idx = int(xp.argmax(xp.abs(corr)))
    assert peak_idx == 50


def test_correlate_shift_detection(backend_device, xp):
    """Correlation should detect template shift."""
    # Template
    template = xp.array([1.0, 1.0, 1.0, -1.0, -1.0], dtype=xp.float32)

    # Signal with template at different position
    signal = xp.zeros(50, dtype=xp.float32)
    signal[20:25] = template

    corr = sync.correlate(signal, template, mode="same")

    # Peak should be near position 22 (center of template)
    peak_idx = int(xp.argmax(xp.abs(corr)))
    assert abs(peak_idx - 22) <= 1


def test_correlate_mimo(backend_device, xp):
    """Correlation should work on MIMO signals."""
    # 2 channels
    signal = xp.zeros((2, 50), dtype=xp.float32)
    signal[0, 20] = 1.0
    signal[1, 30] = 1.0

    template = xp.array([1.0], dtype=xp.float32)

    corr = sync.correlate(signal, template, mode="same")

    assert corr.shape == (2, 50)

    # Peaks at different locations per channel
    peak_0 = int(xp.argmax(xp.abs(corr[0])))
    peak_1 = int(xp.argmax(xp.abs(corr[1])))

    assert peak_0 == 20
    assert peak_1 == 30


def test_detect_frame_known_position(backend_device, xp):
    """Frame detection at known preamble position."""
    # Create preamble
    preamble_symbols = sync.barker_sequence(13)

    # Embed in longer signal at known position
    signal = xp.zeros(200, dtype=xp.complex64)
    start_pos = 50
    signal[start_pos : start_pos + 13] = preamble_symbols

    # Detect
    detected_pos = sync.detect_frame(signal, preamble_symbols, threshold=0.3)

    # Should be within 1 sample of true position
    assert abs(detected_pos - start_pos) <= 1


def test_detect_frame_with_preamble_object(backend_device, xp):
    """Frame detection using Preamble class."""
    # Create Preamble from Barker bits
    barker_bits = ((sync.barker_sequence(13) + 1) / 2).astype(int)
    # Convert to numpy for Preamble if on GPU
    if hasattr(barker_bits, "get"):
        barker_bits = barker_bits.get()

    preamble = Preamble(
        bits=np.array(barker_bits),
        modulation_scheme="PSK",
        modulation_order=2,
    )

    # Create signal with preamble embedded
    signal = xp.zeros(200, dtype=xp.complex64)
    start_pos = 75

    # Embed preamble symbols
    preamble_syms = xp.asarray(
        preamble.symbols
        if not hasattr(preamble.symbols, "get")
        else preamble.symbols.get()
    )
    signal[start_pos : start_pos + 13] = preamble_syms

    # Detect using Preamble object
    detected_pos = sync.detect_frame(signal, preamble, threshold=0.3)

    assert abs(detected_pos - start_pos) <= 1


def test_detect_frame_returns_metric(backend_device, xp):
    """Frame detection should return correlation metric when requested."""
    preamble = sync.barker_sequence(7)

    signal = xp.zeros(100, dtype=xp.complex64)
    signal[30:37] = preamble

    pos, metric = sync.detect_frame(signal, preamble, threshold=0.1, return_metric=True)

    assert isinstance(pos, int)
    assert isinstance(metric, float)
    assert 0 <= metric <= 1


def test_generate_preamble_bits_barker(backend_device, xp):
    """Generate Barker preamble bits."""
    bits = sync.generate_preamble_bits("barker", 13)

    assert len(bits) == 13
    # All bits should be 0 or 1
    assert xp.all((bits == 0) | (bits == 1))


def test_generate_preamble_bits_random(backend_device, xp):
    """Generate random preamble bits with seed."""
    bits1 = sync.generate_preamble_bits("random", 32, seed=42)
    bits2 = sync.generate_preamble_bits("random", 32, seed=42)

    assert len(bits1) == 32
    assert xp.array_equal(bits1, bits2)  # Same seed = same bits
