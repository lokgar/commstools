"""Tests for the linewidth estimators (increment-slope, β-separation).

Each estimator is checked against a synthetic Wiener phase of known linewidth
with calibrated AWGN.  Inputs are built on the active backend via the ``xp``
fixture.
"""

import numpy as np
import pytest

from commstools import analysis

R = 32e9  # symbol rate (Baud)
T = 1.0 / R


def _wiener_phase(linewidth, n, seed=0):
    """Discrete Wiener phase walk at the symbol rate (NumPy, float64)."""
    rng = np.random.default_rng(seed)
    incr = rng.normal(0.0, np.sqrt(2.0 * np.pi * linewidth * T), n)
    return np.cumsum(incr)


def test_linewidth_increment_slope_awgn_free(xp):
    """Lag-slope linewidth must reject AWGN (intercept) and recover Δν."""
    n = 1 << 18
    dnu = 2e6
    phi = _wiener_phase(dnu, n, seed=7)
    rng = np.random.default_rng(8)
    sigma2 = 10 ** (-25 / 10)  # heavy AWGN angle noise
    awgn = rng.normal(0, np.sqrt(sigma2 / 2), n)  # small-angle phase error
    res = analysis.linewidth_increment(xp.asarray(phi + awgn), R, method="slope")
    assert res["linewidth"] == pytest.approx(dnu, rel=0.10)
    # The fitted intercept ≈ 2σ_φ² ≈ σ_n² for unit-power QPSK.
    assert res["awgn_var"] == pytest.approx(sigma2, rel=0.30)


def test_linewidth_increment_subtract_matches_slope(xp):
    n = 1 << 18
    dnu = 1.5e6
    phi = _wiener_phase(dnu, n, seed=11)
    res = analysis.linewidth_increment(
        xp.asarray(phi), R, method="subtract", noise_var=0.0
    )
    assert res["linewidth"] == pytest.approx(dnu, rel=0.10)


def test_linewidth_beta_separation_floor(xp):
    n = 1 << 19
    dnu = 1.5e6
    phi = _wiener_phase(dnu, n, seed=13)
    out = analysis.linewidth_beta_separation(
        xp.asarray(phi), R, nperseg=1 << 13, f_min=5e6, f_max=2e8
    )
    # White-FM floor Δν = π·S_f is the robust estimator at high baud.
    assert out["linewidth_floor"] == pytest.approx(dnu, rel=0.10)
