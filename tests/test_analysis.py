"""Tests for the carrier-phase / laser-characterization module.

Each estimator is checked against an *analytic* ground truth: a synthetic
Wiener phase of known linewidth, a deterministic sinusoidal frequency wander,
and calibrated AWGN.  Inputs are built on the active backend via the ``xp``
fixture so CPU and GPU paths are both exercised.
"""

import numpy as np
import pytest

from commstools import analysis
from commstools.backend import to_device

R = 32e9  # symbol rate (Baud)
T = 1.0 / R


def _wiener_phase(linewidth, n, seed=0):
    """Discrete Wiener phase walk at the symbol rate (NumPy, float64)."""
    rng = np.random.default_rng(seed)
    incr = rng.normal(0.0, np.sqrt(2.0 * np.pi * linewidth * T), n)
    return np.cumsum(incr)


def _qpsk(n, seed=1):
    rng = np.random.default_rng(seed)
    return rng.choice([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], n) / np.sqrt(2.0)


# -----------------------------------------------------------------------------
def test_carrier_phase_trajectory_recovers_phase(xp, xpt):
    n = 1 << 14
    phi_true = _wiener_phase(1e6, n)
    d = _qpsk(n)
    y = d * np.exp(1j * phi_true)  # noise-free
    phi = analysis.carrier_phase_trajectory(xp.asarray(y), xp.asarray(d))
    # Only the time-variation is meaningful (constant offset is irrelevant).
    dphi = xp.diff(phi)
    xpt.assert_allclose(dphi, xp.asarray(np.diff(phi_true)), atol=1e-6)


def test_carrier_phase_trajectory_auto_pairing(xp):
    n = 1 << 13
    d0, d1 = _qpsk(n, 1), _qpsk(n, 2)
    p0, p1 = _wiener_phase(5e5, n, 3), _wiener_phase(5e5, n, 4)
    y = np.stack([d0 * np.exp(1j * p0), d1 * np.exp(1j * p1)])
    d_swapped = np.stack([d1, d0])  # equalizer mapped pol 0↔1
    phi = analysis.carrier_phase_trajectory(
        xp.asarray(y), xp.asarray(d_swapped), channel_pairing="auto"
    )
    # With the right pairing restored, the phase-error increment variance is
    # tiny (only the phase walk); a wrong pairing would be ~uniform on [-π, π].
    var = float(xp.var(xp.diff(phi, axis=-1)))
    assert var < 0.1


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


def test_frequency_drift_metrics_sinusoid(xp):
    n = 1 << 16
    amp, periods = 4e6, 10.0
    t = np.arange(n) * T
    fm = periods / (n * T)
    # drift phase whose derivative is amp·sin(2π fm t)
    drift_phase = 2.0 * np.pi * np.cumsum(amp * np.sin(2 * np.pi * fm * t)) * T
    m = analysis.frequency_drift_metrics(xp.asarray(drift_phase), R)
    assert m["std"] == pytest.approx(amp / np.sqrt(2.0), rel=0.05)
    assert m["pp"] == pytest.approx(2.0 * amp, rel=0.10)


def test_separate_drift_phase_noise_splits(xp):
    n = 1 << 16
    t = np.arange(n) * T
    # slow drift (100 kHz tone in phase) + fast jitter
    slow = 0.5 * np.sin(2 * np.pi * 1e5 * t)
    rng = np.random.default_rng(21)
    fast = rng.normal(0, 0.05, n)
    phi = slow + fast
    drift, pn = analysis.separate_drift_phase_noise(xp.asarray(phi), R, cutoff=1e6)
    assert drift.shape == pn.shape == (n,)
    # Drift tracks the slow tone; pn carries the fast jitter.
    edge = 200
    dr = np.asarray(to_device(drift, "cpu"))[edge:-edge]
    sl = slow[edge:-edge]
    assert float(np.std(dr - sl)) < 0.05


def test_allan_deviation_white_fm_slope(xp):
    """White-FM frequency noise ⇒ Allan deviation slope ≈ -1/2 in log-log."""
    n = 1 << 16
    rng = np.random.default_rng(31)
    df = rng.normal(0, 1e5, n)  # white frequency noise
    out = analysis.allan_deviation(xp.asarray(df), R, n_taus=20)
    tau, adev = out["tau_s"], out["adev"]
    good = np.isfinite(adev) & (adev > 0)
    slope = np.polyfit(np.log(tau[good]), np.log(adev[good]), 1)[0]
    assert slope == pytest.approx(-0.5, abs=0.15)


def test_characterize_carrier_phase_report(xp):
    n = 1 << 17
    dnu = 1.5e6
    phi_pn = _wiener_phase(dnu, n, seed=41)
    d = _qpsk(n, 42)
    y = d * np.exp(1j * phi_pn)
    rep = analysis.characterize_carrier_phase(
        xp.asarray(y),
        xp.asarray(d),
        R,
        drift_cutoff=3e6,
        nperseg=1 << 12,
        f_min=5e6,
        f_max=2e8,
    )
    for key in (
        "phi",
        "drift",
        "pn",
        "drift_metrics",
        "linewidth_increment",
        "linewidth_beta",
        "allan",
    ):
        assert key in rep
    assert rep["linewidth_increment"]["linewidth"] == pytest.approx(dnu, rel=0.15)
