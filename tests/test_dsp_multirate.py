import numpy as np
import pytest
from commstools.dsp import multirate


def test_expand():
    samples = np.array([1.0, 2.0])
    expanded = multirate.expand(samples, factor=2)
    assert np.allclose(expanded, [1.0, 0.0, 2.0, 0.0])


def test_upsample():
    samples = np.array([1.0])
    # Upsample inserts zeros then lowpass filters.
    # Factor 2.
    upsampled = multirate.upsample(samples, factor=2)
    assert len(upsampled) == 2


def test_decimate():
    samples = np.array([1.0, 1.0, 1.0, 1.0])
    decimated = multirate.decimate(samples, factor=2)
    assert len(decimated) == 2


def test_resample():
    samples = np.array([1.0, 1.0])
    # Up 3, Down 2 -> 1.5x length
    resampled = multirate.resample(samples, up=3, down=2)
    assert len(resampled) == 3
