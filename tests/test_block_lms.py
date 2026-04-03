"""Tests for block_lms — frequency-domain block LMS equalizer.

Coverage:
  1. Output shapes — SISO and MIMO, with/without CPR
  2. Gradient descent — MSE decreases over blocks (identity channel, noise)
  3. Identity channel parity — with sufficient training, output ≈ input symbols
  4. w_init passthrough — warm-start weights are used
  5. store_weights shape — weights_history has expected layout
  6. num_train_symbols boundary — DA/DD switch is respected
  7. Last-block edge — n_sym not a multiple of block_size
  8. cpr_type validation — pll raises ValueError
  9. BPS + CPR — phase_trajectory shape and MSE better than no CPR under phase noise
 10. BPS block_size vs cpr_bps_block_size independence — different values accepted
 11. MIMO butterfly convergence — 2x2, training on both channels
"""

import numpy as np
import pytest

from commstools.backend import use_cpu_only
from commstools.equalization import block_lms
from commstools.mapping import gray_constellation
from commstools.helpers import normalize


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

use_cpu_only(True)


def _qam16(n_sym=4096, snr_db=25.0, sps=2, rng=None):
    """Return (samples_sps, symbols) for 16-QAM with AWGN, repeat upsampling."""
    if rng is None:
        rng = np.random.default_rng(42)
    const = gray_constellation("qam", 16).astype(np.complex64)
    const = normalize(const, "average_power").astype(np.complex64)
    syms = const[rng.integers(0, 16, n_sym)]
    samples = np.repeat(syms, sps).astype(np.complex64)
    noise_std = np.sqrt(10 ** (-snr_db / 10) / 2)
    samples += noise_std * (
        rng.standard_normal(len(samples)) + 1j * rng.standard_normal(len(samples))
    ).astype(np.complex64)
    return samples, syms


def _qpsk(n_sym=4096, snr_db=20.0, sps=2, rng=None):
    """Return (samples_sps, symbols) for QPSK with AWGN, repeat upsampling."""
    if rng is None:
        rng = np.random.default_rng(7)
    const = gray_constellation("psk", 4).astype(np.complex64)
    syms = const[rng.integers(0, 4, n_sym)]
    samples = np.repeat(syms, sps).astype(np.complex64)
    noise_std = np.sqrt(10 ** (-snr_db / 10) / 2)
    samples += noise_std * (
        rng.standard_normal(len(samples)) + 1j * rng.standard_normal(len(samples))
    ).astype(np.complex64)
    return samples, syms


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Output shapes — SISO
# ─────────────────────────────────────────────────────────────────────────────


def test_output_shape_siso():
    samples, syms = _qam16(n_sym=1024, sps=2)
    r = block_lms(
        samples, syms, num_taps=11, sps=2, modulation="qam", order=16, block_size=128
    )
    n_sym = len(syms)
    assert r.y_hat.shape == (n_sym,), f"y_hat shape {r.y_hat.shape}"
    assert r.weights.shape == (11,), f"weights shape {r.weights.shape}"
    assert r.error.shape == (n_sym,)
    assert r.phase_trajectory is None


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Output shapes — MIMO (C=2)
# ─────────────────────────────────────────────────────────────────────────────


def test_output_shape_mimo():
    rng = np.random.default_rng(1)
    s1, t1 = _qam16(n_sym=1024, sps=2, rng=rng)
    s2, t2 = _qam16(n_sym=1024, sps=2, rng=rng)
    samples = np.stack([s1, s2])
    training = np.stack([t1, t2])
    r = block_lms(
        samples,
        training,
        num_taps=11,
        sps=2,
        modulation="qam",
        order=16,
        block_size=128,
    )
    assert r.y_hat.shape == (2, 1024)
    assert r.weights.shape == (2, 2, 11)
    assert r.error.shape == (2, 1024)


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Output shapes — BPS CPR
# ─────────────────────────────────────────────────────────────────────────────


def test_output_shape_bps():
    samples, syms = _qam16(n_sym=1024, sps=2)
    r = block_lms(
        samples,
        syms,
        num_taps=11,
        sps=2,
        modulation="qam",
        order=16,
        block_size=128,
        cpr_type="bps",
        cpr_bps_test_phases=16,
        cpr_bps_block_size=32,
    )
    assert r.y_hat.shape == (1024,)
    assert r.phase_trajectory is not None
    assert r.phase_trajectory.shape == (1024,), (
        f"phase_trajectory shape {r.phase_trajectory.shape}"
    )


def test_output_shape_bps_mimo():
    rng = np.random.default_rng(2)
    s1, t1 = _qam16(n_sym=1024, sps=2, rng=rng)
    s2, t2 = _qam16(n_sym=1024, sps=2, rng=rng)
    r = block_lms(
        np.stack([s1, s2]),
        np.stack([t1, t2]),
        num_taps=11,
        sps=2,
        modulation="qam",
        order=16,
        block_size=128,
        cpr_type="bps",
        cpr_bps_test_phases=16,
    )
    assert r.phase_trajectory.shape == (2, 1024)


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Gradient descent — MSE decreases
# ─────────────────────────────────────────────────────────────────────────────


def test_mse_decreases():
    """MSE in the last quarter of the signal must be less than in the first quarter."""
    samples, syms = _qam16(n_sym=8192, snr_db=25.0, sps=2)
    r = block_lms(
        samples,
        syms[:512],
        num_taps=11,
        sps=2,
        step_size=0.05,
        modulation="qam",
        order=16,
        block_size=128,
    )
    err = np.abs(r.error) ** 2
    n = len(err)
    assert err[: n // 4].mean() > err[3 * n // 4 :].mean(), (
        "MSE did not decrease from first to last quarter"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Identity channel — output close to training symbols
# ─────────────────────────────────────────────────────────────────────────────


def test_identity_channel_convergence():
    """On a near-identity channel, symbols after training should match reference."""
    samples, syms = _qam16(n_sym=4096, snr_db=30.0, sps=2)
    n_train = 1024
    r = block_lms(
        samples,
        syms,
        num_taps=11,
        sps=2,
        step_size=0.05,
        modulation="qam",
        order=16,
        block_size=128,
    )
    # EVM in the last half (well past training)
    y_eval = r.y_hat[n_train:]
    s_eval = syms[n_train:]
    evm = np.sqrt(np.mean(np.abs(y_eval - s_eval) ** 2) / np.mean(np.abs(s_eval) ** 2))
    assert evm < 0.15, f"EVM {evm:.3f} too high — equalizer did not converge"


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: w_init passthrough
# ─────────────────────────────────────────────────────────────────────────────


def test_w_init_used():
    """Warm-starting from converged weights should give lower initial MSE."""
    samples, syms = _qam16(n_sym=4096, snr_db=25.0, sps=2)
    # First pass: converge weights
    r1 = block_lms(
        samples,
        syms,
        num_taps=11,
        sps=2,
        step_size=0.05,
        modulation="qam",
        order=16,
        block_size=128,
    )
    # Second pass: warm-start; MSE at start should be low
    r2 = block_lms(
        samples,
        syms,
        num_taps=11,
        sps=2,
        step_size=0.05,
        modulation="qam",
        order=16,
        block_size=128,
        w_init=r1.weights,
    )
    # First block MSE with warm-start should be better than cold-start
    mse_cold_start = float(np.mean(np.abs(r1.error[:128]) ** 2))
    mse_warm_start = float(np.mean(np.abs(r2.error[:128]) ** 2))
    assert mse_warm_start < mse_cold_start, (
        f"warm-start MSE ({mse_warm_start:.4f}) not better than cold ({mse_cold_start:.4f})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: store_weights shape
# ─────────────────────────────────────────────────────────────────────────────


def test_store_weights_shape_siso():
    samples, syms = _qam16(n_sym=512, sps=2)
    r = block_lms(
        samples,
        syms,
        num_taps=11,
        sps=2,
        modulation="qam",
        order=16,
        block_size=64,
        store_weights=True,
    )
    assert r.weights_history is not None
    assert r.weights_history.shape == (512, 11), (
        f"weights_history shape {r.weights_history.shape}"
    )


def test_store_weights_shape_mimo():
    rng = np.random.default_rng(3)
    s1, t1 = _qam16(n_sym=512, sps=2, rng=rng)
    s2, t2 = _qam16(n_sym=512, sps=2, rng=rng)
    r = block_lms(
        np.stack([s1, s2]),
        np.stack([t1, t2]),
        num_taps=11,
        sps=2,
        modulation="qam",
        order=16,
        block_size=64,
        store_weights=True,
    )
    assert r.weights_history.shape == (512, 2, 2, 11)


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: num_train_symbols — DA/DD boundary
# ─────────────────────────────────────────────────────────────────────────────


def test_num_train_symbols_respected():
    """Training length is determined by the length of training_symbols passed in."""
    samples, syms = _qam16(n_sym=2048, snr_db=30.0, sps=2)
    # Pure DA (all training)
    r_da = block_lms(
        samples,
        syms,
        num_taps=11,
        sps=2,
        step_size=0.05,
        modulation="qam",
        order=16,
        block_size=128,
    )
    assert r_da.num_train_symbols == 2048

    # Pre-sliced training to 256
    r_clip = block_lms(
        samples,
        syms[..., :256],
        num_taps=11,
        sps=2,
        step_size=0.05,
        modulation="qam",
        order=16,
        block_size=128,
    )
    assert r_clip.num_train_symbols == 256


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: Last-block edge — n_sym not a multiple of block_size
# ─────────────────────────────────────────────────────────────────────────────


def test_non_multiple_block_size():
    samples, syms = _qam16(n_sym=1000, sps=2)  # 1000 is not a multiple of 128
    r = block_lms(
        samples, syms, num_taps=11, sps=2, modulation="qam", order=16, block_size=128
    )
    assert r.y_hat.shape == (1000,)
    assert r.error.shape == (1000,)


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: cpr_type='pll' raises ValueError
# ─────────────────────────────────────────────────────────────────────────────


def test_pll_raises():
    samples, syms = _qam16(n_sym=512, sps=2)
    with pytest.raises(ValueError, match="pll"):
        block_lms(
            samples,
            syms,
            num_taps=11,
            sps=2,
            modulation="qam",
            order=16,
            cpr_type="pll",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 11: BPS block_size independent of cpr_bps_block_size
# ─────────────────────────────────────────────────────────────────────────────


def test_bps_block_size_independent():
    """block_size=256 with cpr_bps_block_size=16 must produce per-symbol phi."""
    samples, syms = _qam16(n_sym=1024, sps=2)
    r = block_lms(
        samples,
        syms,
        num_taps=11,
        sps=2,
        modulation="qam",
        order=16,
        block_size=256,
        cpr_type="bps",
        cpr_bps_test_phases=16,
        cpr_bps_block_size=16,
    )
    assert r.phase_trajectory.shape == (1024,)
    # Phase estimates should not all be identical (per-symbol, not per-block)
    phi = r.phase_trajectory
    assert not np.all(phi == phi[0]), (
        "All phi identical — expected per-symbol variation with cpr_bps_block_size=16"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 12: BPS under phase noise — CPR reduces MSE vs no CPR
# ─────────────────────────────────────────────────────────────────────────────


def test_bps_reduces_mse_under_phase_noise():
    """With strong phase noise, BPS should produce lower steady-state MSE."""
    rng = np.random.default_rng(99)
    n_sym = 4096
    sps = 2
    const = gray_constellation("psk", 4).astype(np.complex64)
    syms = const[rng.integers(0, 4, n_sym)]
    samples = np.repeat(syms, sps).astype(np.complex64)

    # Add AWGN
    samples += 0.1 * (
        rng.standard_normal(len(samples)) + 1j * rng.standard_normal(len(samples))
    ).astype(np.complex64)
    # Add random-walk phase noise
    phase_noise = np.cumsum(0.03 * rng.standard_normal(n_sym)).astype(np.float32)
    samples_pn = np.repeat(syms * np.exp(1j * phase_noise), sps) + 0.1 * (
        rng.standard_normal(2 * n_sym) + 1j * rng.standard_normal(2 * n_sym)
    ).astype(np.complex64)

    n_eval = n_sym // 2  # evaluate on second half
    r_no_cpr = block_lms(
        samples_pn,
        syms[:512],
        num_taps=7,
        sps=sps,
        step_size=0.05,
        block_size=128,
        modulation="psk",
        order=4,
    )
    r_bps = block_lms(
        samples_pn,
        syms[:512],
        num_taps=7,
        sps=sps,
        step_size=0.05,
        block_size=128,
        modulation="psk",
        order=4,
        cpr_type="bps",
        cpr_bps_test_phases=32,
        cpr_bps_block_size=32,
    )

    mse_no_cpr = float(np.mean(np.abs(r_no_cpr.error[n_eval:]) ** 2))
    mse_bps = float(np.mean(np.abs(r_bps.error[n_eval:]) ** 2))
    assert mse_bps < mse_no_cpr, (
        f"BPS MSE ({mse_bps:.4f}) not better than no-CPR ({mse_no_cpr:.4f}) "
        "under phase noise"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 13: MIMO butterfly convergence (2x2)
# ─────────────────────────────────────────────────────────────────────────────


def test_mimo_convergence():
    """2x2 MIMO: both channels should converge to low EVM."""
    rng = np.random.default_rng(5)
    n_sym = 4096
    sps = 1
    const = gray_constellation("qam", 16).astype(np.complex64)
    const = normalize(const, "average_power").astype(np.complex64)
    t1 = const[rng.integers(0, 16, n_sym)]
    t2 = const[rng.integers(0, 16, n_sym)]
    noise = 0.05 * (
        rng.standard_normal((2, n_sym)) + 1j * rng.standard_normal((2, n_sym))
    ).astype(np.complex64)
    samples = np.stack([t1, t2]) + noise

    r = block_lms(
        samples,
        np.stack([t1, t2]),
        num_taps=5,
        sps=sps,
        step_size=0.1,
        modulation="qam",
        order=16,
        block_size=64,
    )
    n_eval = n_sym // 2
    for ch in range(2):
        evm = float(
            np.sqrt(
                np.mean(
                    np.abs(r.y_hat[ch, n_eval:] - np.stack([t1, t2])[ch, n_eval:]) ** 2
                )
                / np.mean(np.abs(np.stack([t1, t2])[ch, n_eval:]) ** 2)
            )
        )
        assert evm < 0.15, f"MIMO ch{ch} EVM {evm:.3f} too high"
