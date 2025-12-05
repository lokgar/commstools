import numpy as np
import pytest
from commstools.dsp import filtering


def test_boxcar_taps():
    sps = 4
    taps = filtering.boxcar_taps(sps)
    assert len(taps) == sps
    assert np.allclose(taps, 0.25)  # Unity gain normalized: 1s / 4


def test_gaussian_taps():
    sps = 4
    taps = filtering.gaussian_taps(sps, bt=0.3, span=2)
    # Span=2, sps=4 -> 8 taps. If odd required, it adds 1 -> 9.
    # The code says if num_taps % 2 == 0: num_taps += 1. So 9.
    assert len(taps) == 9
    assert np.sum(taps) > 0


def test_rrc_taps():
    sps = 4
    taps = filtering.rrc_taps(sps, span=4)
    # 4 * 4 = 16 (even) -> +1 = 17
    assert len(taps) == 17
    # Peak should be at center
    assert np.argmax(taps) == len(taps) // 2


def test_rc_taps():
    sps = 4
    taps = filtering.rc_taps(sps, span=4)
    assert len(taps) == 17
    assert np.argmax(taps) == len(taps) // 2


def test_fir_filter():
    samples = np.array([1.0, 0.0, 0.0, 0.0])
    taps = np.array([0.5, 0.5])
    # Convolution: [0.5, 0.5, 0, 0] ...
    filtered = filtering.fir_filter(samples, taps, mode="full")
    assert np.allclose(filtered, [0.5, 0.5, 0, 0, 0])


def test_shape_pulse_none():
    symbols = np.array([1.0, -1.0])
    sps = 2
    shaped = filtering.shape_pulse(symbols, sps, pulse_shape="none")
    # sps=2 -> [1, 1, -1, -1] ? No, expand just inserts zeros?
    # Current implementation of 'none' uses expand:
    # backend.expand(symbols, int(sps)) -> [1, 0, -1, 0] ??
    # Let's check source code behaviour if needed.
    # Actually wait, `expand` in this codebase likely means zero insertion (upsampling without filter).
    # BUT `expand` usually means repeat in some libraries. In `multirate.py` it says "Zero-insertion".
    # So expected: [1, 0, -1, 0] normalized to max amplitude.
    # And `normalize` with "max_amplitude" divides by max.
    expected = np.array([1.0, 0.0, -1.0, 0.0])
    assert len(shaped) == len(symbols) * sps
    assert np.allclose(shaped, expected)


def test_shape_pulse_boxcar():
    symbols = np.array([1.0, -1.0])
    sps = 2
    shaped = filtering.shape_pulse(symbols, sps, pulse_shape="boxcar")
    # Boxcar: [0.5, 0.5] convolved with [1, 0, -1, 0] -> [0.5, 0.5, -0.5, -0.5]
    # Then normalized to max amplitude 1.0 -> [1, 1, -1, -1]
    expected = np.array([1.0, 1.0, -1.0, -1.0])
    # It might be slightly different due to mode='same' vs 'full' in polyphase resample?
    # polyphase resample usually gives length approx len(symbols) * sps.
    assert len(shaped) == 4
    assert np.allclose(shaped, expected)


def test_matched_filter():
    # Simple case: pulse = [1], signal = [1]
    signal = np.array([1.0])
    pulse = np.array([0.5])  # normalized boxcar tap
    matched = filtering.matched_filter(signal, pulse)
    # Matched filter matches the pulse.
    assert len(matched) == 1
