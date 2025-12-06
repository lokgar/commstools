import numpy as np
from commstools import filtering


def test_boxcar_taps():
    sps = 4
    taps = filtering.boxcar_taps(sps)
    assert len(taps) == sps
    assert np.allclose(taps, 0.25)  # Unity gain normalized: 1s / 4


def test_gaussian_taps():
    sps = 4
    taps = filtering.gaussian_taps(sps, bt=0.3, span=2)
    assert len(taps) == 9
    assert np.sum(taps) > 0


def test_rrc_taps():
    sps = 4
    taps = filtering.rrc_taps(sps, span=4)
    assert len(taps) == 17
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
    expected = np.array([1.0, 0.0, -1.0, 0.0])
    assert len(shaped) == len(symbols) * sps
    assert np.allclose(shaped, expected)


def test_shape_pulse_boxcar():
    symbols = np.array([1.0, -1.0])
    sps = 2
    shaped = filtering.shape_pulse(symbols, sps, pulse_shape="boxcar")
    expected = np.array([1.0, 1.0, -1.0, -1.0])
    assert len(shaped) == 4
    assert np.allclose(shaped, expected)


def test_matched_filter():
    signal = np.array([1.0])
    pulse = np.array([0.5])  # normalized boxcar tap
    matched = filtering.matched_filter(signal, pulse)
    assert len(matched) == 1
