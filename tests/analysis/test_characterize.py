"""Tests for the end-to-end carrier-phase characterization orchestrator.

The full report is built from a synthetic Wiener phase of known linewidth and
checked for completeness and linewidth recovery.  Inputs are built on the
active backend via the ``xp`` fixture.
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


def _qpsk(n, seed=1):
    rng = np.random.default_rng(seed)
    return rng.choice([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], n) / np.sqrt(2.0)


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
