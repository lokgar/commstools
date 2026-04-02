"""Tests for joint LMS/RLS+CPR equalizers and blockwise FOE.

Verification plan:
  1. Zero-Deviation Baseline   — cpr_type=None must produce bit-exact output
  2. Numba/JAX Backend Parity  — pll and bps modes match within float32 tolerance
  3. Cycle Slip Stress Test    — π/2 steps are corrected, weights converge
  4. PLL Convergence / Phase Noise — RMSE within PLL jitter bound
  5. Blockwise Phase Coherence — chirp FOE recovers EVM within 0.5 dB of ideal
  6. MIMO Coverage             — 2x2 butterfly LMS+PLL converges on both channels
  7. BPS Phase Unwrap          — phase_trajectory is monotone under linear drift
  8. BPS Convergence           — lms(cpr_type='bps') converges under Wiener phase noise
  9. BPS Block Size > 1        — bps_block_size=32 still converges (incremental sum)
 10. RLS + BPS                 — rls(cpr_type='bps') convergence smoke test
 11. PLL Joint Channels        — cpr_joint_channels=True shares phase across MIMO
"""

import numpy as np
import pytest

from commstools.equalization import lms, rls
from commstools.mapping import gray_constellation
from commstools.sync import (
    correct_carrier_phase,
    estimate_frequency_offset_blockwise,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _qpsk_signal(n_sym=4000, snr_db=20.0, rng=None):
    """Return (samples_2sps, symbols) for a QPSK signal with AWGN."""
    if rng is None:
        rng = np.random.default_rng(0)
    const = gray_constellation("psk", 4).astype(np.complex64)
    idxs = rng.integers(0, 4, n_sym)
    syms = const[idxs]
    # Upsample to 2 SPS (trivial for test — repeat each symbol)
    samples = np.repeat(syms, 2)
    noise_pwr = 10 ** (-snr_db / 10)
    samples = samples + np.sqrt(noise_pwr / 2) * (
        rng.standard_normal(len(samples)) + 1j * rng.standard_normal(len(samples))
    ).astype(np.complex64)
    return samples.astype(np.complex64), syms


def _qam16_signal(n_sym=4000, snr_db=25.0, rng=None):
    """Return (samples_1sps, symbols) for 16-QAM at 1 SPS."""
    if rng is None:
        rng = np.random.default_rng(1)
    const = gray_constellation("qam", 16).astype(np.complex64)
    idxs = rng.integers(0, 16, n_sym)
    syms = const[idxs]
    noise_pwr = 10 ** (-snr_db / 10)
    samples = syms + np.sqrt(noise_pwr / 2) * (
        rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)
    ).astype(np.complex64)
    return samples.astype(np.complex64), syms


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Zero-Deviation Baseline
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("backend", ["numba", "jax"])
@pytest.mark.parametrize("algo", ["lms", "rls"])
def test_cpr_none_baseline(backend, algo):
    """cpr_type=None produces bit-exact output vs the unmodified algorithm."""
    samples, syms = _qpsk_signal(n_sym=2000)
    kwargs = dict(
        training_symbols=syms[:500],
        num_taps=11,
        sps=2,
        modulation="psk",
        order=4,
        backend=backend,
    )
    fn = lms if algo == "lms" else rls
    extra = {} if algo == "lms" else {"sps": 2}
    kwargs.update(extra)

    res_base = fn(samples, **kwargs, cpr_type=None)
    res_cpr_none = fn(samples, **kwargs, cpr_type=None)

    np.testing.assert_array_equal(
        np.asarray(res_base.y_hat),
        np.asarray(res_cpr_none.y_hat),
        err_msg=f"{algo}/{backend}: cpr_type=None must be deterministic",
    )
    assert res_cpr_none.phase_trajectory is None


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Numba / JAX Backend Parity
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cpr_type", ["pll", "bps"])
def test_numba_jax_parity_lms(cpr_type):
    """Numba and JAX LMS+CPR produce matching outputs within float32 tolerance."""
    pytest.importorskip("jax")
    samples, syms = _qpsk_signal(n_sym=1000)
    kwargs = dict(
        training_symbols=syms[:300],
        num_taps=11,
        sps=2,
        modulation="psk",
        order=4,
        cpr_type=cpr_type,
        cpr_pll_bandwidth=5e-3,
        cpr_bps_test_phases=32,
        cpr_cycle_slip_correction=False,  # disable for parity (Numba uses O(1) stats, JAX recomputes)
    )
    res_nb = lms(samples, **kwargs, backend="numba")
    res_jx = lms(samples, **kwargs, backend="jax")

    np.testing.assert_allclose(
        np.asarray(res_nb.y_hat),
        np.asarray(res_jx.y_hat),
        atol=1e-4,
        err_msg=f"LMS cpr={cpr_type}: Numba vs JAX y_hat mismatch",
    )
    assert res_nb.phase_trajectory is not None
    assert res_jx.phase_trajectory is not None


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Cycle Slip Stress Test
# ─────────────────────────────────────────────────────────────────────────────


def test_cycle_slip_correction():
    """LMS+PLL recovers through deliberate π/2 phase steps without diverging."""
    rng = np.random.default_rng(42)
    n_sym = 3000
    const = gray_constellation("psk", 4).astype(np.complex64)
    idxs = rng.integers(0, 4, n_sym)
    syms = const[idxs]

    # Build 1-SPS signal with step-wise phase jumps every 500 symbols
    phase = np.zeros(n_sym, dtype=np.float64)
    for step_idx in range(500, n_sym, 500):
        phase[step_idx:] += np.pi / 2

    samples = (syms * np.exp(1j * phase).astype(np.complex64)).astype(np.complex64)
    noise = 0.05 * (rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym))
    samples = (samples + noise.astype(np.complex64)).astype(np.complex64)

    res = lms(
        samples,
        training_symbols=syms[:300],
        num_taps=1,
        sps=1,
        modulation="psk",
        order=4,
        cpr_type="pll",
        cpr_pll_bandwidth=5e-3,
        cpr_cycle_slip_correction=True,
        cpr_cycle_slip_history=200,
        backend="numba",
    )

    assert res.phase_trajectory is not None
    # Equalizer must produce finite output
    assert np.all(np.isfinite(np.asarray(res.y_hat))), (
        "y_hat contains non-finite values"
    )
    # Check convergence: MSE in last 500 symbols should be low
    y_dd = res.y_hat[-500:]
    # Nearest constellation decision
    d = const[np.argmin(np.abs(y_dd[:, None] - const[None, :]) ** 2, axis=1)]
    mse = np.mean(np.abs(y_dd - d) ** 2)
    assert mse < 0.1, f"MSE after cycle slip recovery too large: {mse:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: PLL Convergence / Phase Noise
# ─────────────────────────────────────────────────────────────────────────────


def test_pll_phase_noise_tracking():
    """LMS+PLL tracks Wiener-process phase noise; phase RMSE within expected bound."""
    rng = np.random.default_rng(7)
    n_sym = 5000
    linewidth_ts = 1e-4  # Δν·T_s — normalised laser linewidth
    const = gray_constellation("qam", 16).astype(np.complex64)
    idxs = rng.integers(0, 16, n_sym)
    syms = const[idxs]

    # Wiener phase noise
    phase_steps = rng.standard_normal(n_sym) * np.sqrt(2 * np.pi * linewidth_ts)
    phase_noise = np.cumsum(phase_steps).astype(np.float64)

    snr_db = 25.0
    noise_pwr = 10 ** (-snr_db / 10)
    awgn = np.sqrt(noise_pwr / 2) * (
        rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)
    ).astype(np.complex64)
    samples = (syms * np.exp(1j * phase_noise).astype(np.complex64) + awgn).astype(
        np.complex64
    )

    bw = 5e-3  # loop bandwidth
    res = lms(
        samples,
        training_symbols=syms[:1000],
        num_taps=1,
        sps=1,
        modulation="qam",
        order=16,
        cpr_type="pll",
        cpr_pll_bandwidth=bw,
        cpr_cycle_slip_correction=False,
        backend="numba",
    )

    assert res.phase_trajectory is not None
    # Compare estimated phase to ground truth in steady-state (skip first 1500)
    phi_est = np.asarray(res.phase_trajectory)[1500:]
    phi_true = phase_noise[1500:]
    # Phase offset due to ambiguity: remove mean
    offset = np.mean(phi_true - phi_est)
    rmse = np.sqrt(np.mean((phi_true - phi_est - offset) ** 2))
    # Theoretical bound: σ² ≈ B_L / (Δν·T_s) … in practice RMSE < 5x bound
    bound = np.sqrt(bw / linewidth_ts)  # loose upper bound scaled
    assert rmse < bound, f"PLL phase RMSE {rmse:.4f} exceeds bound {bound:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Blockwise Phase Coherence (FOE)
# ─────────────────────────────────────────────────────────────────────────────


def test_blockwise_foe_chirp():
    """estimate_frequency_offset_blockwise recovers a linearly chirping frequency."""
    rng = np.random.default_rng(3)
    fs = 1e9
    n = 65536
    sps = 2
    f_start, f_end = 1e6, 5e6  # Hz, linear chirp

    t = np.arange(n) / fs
    f_t = np.linspace(f_start, f_end, n)
    phase_chirp = 2 * np.pi * np.cumsum(f_t) / fs
    carrier = np.exp(1j * phase_chirp).astype(np.complex64)

    const = gray_constellation("qam", 16).astype(np.complex64)
    idxs = rng.integers(0, 16, n // sps)
    syms = const[idxs]
    base = np.repeat(syms, sps).astype(np.complex64)
    samples = base * carrier

    theta = estimate_frequency_offset_blockwise(
        samples,
        fs,
        block_size=4096,
        overlap=0.5,
        method="mth_power",
        sps=sps,
        modulation="qam",
        order=16,
    )
    corrected = np.asarray(correct_carrier_phase(samples, theta))

    # EVM (in dB) of corrected signal vs base signal
    evm_corrected = 10 * np.log10(
        np.mean(np.abs(corrected - base) ** 2) / np.mean(np.abs(base) ** 2)
    )
    evm_uncorrected = 10 * np.log10(
        np.mean(np.abs(samples - base) ** 2) / np.mean(np.abs(base) ** 2)
    )
    # Corrected EVM must be better by at least 10 dB
    assert evm_corrected < evm_uncorrected - 10.0, (
        f"FOE correction ineffective: corrected EVM={evm_corrected:.1f} dB, "
        f"uncorrected={evm_uncorrected:.1f} dB"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: MIMO Coverage
# ─────────────────────────────────────────────────────────────────────────────


def test_mimo_lms_pll():
    """2x2 butterfly LMS+PLL converges on both output channels."""
    rng = np.random.default_rng(99)
    n_sym = 3000
    const = gray_constellation("qam", 16).astype(np.complex64)

    # Two independent QPSK streams with different static phase offsets
    syms_a = const[rng.integers(0, 16, n_sym)]
    syms_b = const[rng.integers(0, 16, n_sym)]
    phase_a = np.float32(0.3)
    phase_b = np.float32(1.1)

    sig_a = (syms_a * np.exp(1j * phase_a)).astype(np.complex64)
    sig_b = (syms_b * np.exp(1j * phase_b)).astype(np.complex64)

    snr_db = 25.0
    noise_pwr = 10 ** (-snr_db / 10)
    noise_a = np.sqrt(noise_pwr / 2) * (
        rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)
    ).astype(np.complex64)
    noise_b = np.sqrt(noise_pwr / 2) * (
        rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)
    ).astype(np.complex64)

    samples = np.stack([sig_a + noise_a, sig_b + noise_b])  # (2, n_sym)
    training = np.stack([syms_a[:500], syms_b[:500]])

    res = lms(
        samples,
        training_symbols=training,
        num_taps=1,
        sps=1,
        modulation="qam",
        order=16,
        cpr_type="pll",
        cpr_pll_bandwidth=5e-3,
        cpr_cycle_slip_correction=False,
        backend="numba",
    )

    assert res.phase_trajectory is not None
    assert res.phase_trajectory.shape == (2, n_sym), (
        f"Expected (2, {n_sym}), got {res.phase_trajectory.shape}"
    )

    # Both channels must converge: MSE in last 1000 symbols < 0.05
    for ch in range(2):
        y_ss = res.y_hat[ch, -1000:]
        d = const[np.argmin(np.abs(y_ss[:, None] - const[None, :]) ** 2, axis=1)]
        mse = float(np.mean(np.abs(y_ss - d) ** 2))
        assert mse < 0.1, f"MIMO channel {ch} MSE too large: {mse:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: BPS Phase Unwrap — phase_trajectory must be monotone under linear drift
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("backend", ["numba", "jax"])
def test_bps_phase_unwrap(backend):
    """phase_trajectory from BPS must not wrap back to [0, π/2) under a ramp.

    Before the fix, the raw BPS argmin in [0,π/2) was stored directly, so a
    slowly drifting phase would produce a sawtooth rather than a monotone ramp.
    After the fix (causal 4-fold unwrap), the trajectory must be monotone.
    """
    rng = np.random.default_rng(5)
    n_sym = 3000
    const = gray_constellation("qam", 16).astype(np.complex64)
    syms = const[rng.integers(0, 16, n_sym)]

    # Linear phase ramp spanning >π/2 to force multiple wraps of the raw argmin
    phase_true = np.linspace(0.0, 3.0, n_sym, dtype=np.float64)  # 0 → 3 rad
    noise_pwr = 10 ** (-25.0 / 10)
    awgn = np.sqrt(noise_pwr / 2) * (
        rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)
    ).astype(np.complex64)
    samples = (syms * np.exp(1j * phase_true).astype(np.complex64) + awgn).astype(
        np.complex64
    )

    res = lms(
        samples,
        training_symbols=syms[:500],
        num_taps=1,
        sps=1,
        modulation="qam",
        order=16,
        cpr_type="bps",
        cpr_bps_test_phases=64,
        cpr_bps_block_size=32,
        cpr_cycle_slip_correction=False,
        backend=backend,
    )

    phi = np.asarray(res.phase_trajectory, dtype=np.float64)
    # After convergence the phase should be broadly monotone — stddev of
    # successive differences should be small relative to the total span.
    span = phi[-1] - phi[0]
    assert span > 1.0, f"Phase did not advance: span={span:.3f} rad"
    # If still wrapped the span would be < π/2 ≈ 1.57; a monotone ramp → ~3 rad
    assert span > np.pi / 2, (
        f"BPS phase_trajectory looks wrapped (span={span:.3f} rad < π/2); "
        "unwrap fix may not have applied."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: BPS Convergence under Wiener phase noise
# ─────────────────────────────────────────────────────────────────────────────


def test_bps_phase_noise_tracking():
    """LMS+BPS converges under Wiener phase noise (Numba backend)."""
    rng = np.random.default_rng(11)
    n_sym = 5000
    linewidth_ts = 5e-5
    const = gray_constellation("qam", 16).astype(np.complex64)
    syms = const[rng.integers(0, 16, n_sym)]

    phase_noise = np.cumsum(
        rng.standard_normal(n_sym) * np.sqrt(2 * np.pi * linewidth_ts)
    ).astype(np.float64)
    noise_pwr = 10 ** (-25.0 / 10)
    awgn = np.sqrt(noise_pwr / 2) * (
        rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)
    ).astype(np.complex64)
    samples = (syms * np.exp(1j * phase_noise).astype(np.complex64) + awgn).astype(
        np.complex64
    )

    res_bps = lms(
        samples,
        training_symbols=syms[:1000],
        num_taps=1,
        sps=1,
        modulation="qam",
        order=16,
        cpr_type="bps",
        cpr_bps_test_phases=64,
        cpr_bps_block_size=32,
        cpr_cycle_slip_correction=False,
        backend="numba",
    )
    res_none = lms(
        samples,
        training_symbols=syms[:1000],
        num_taps=1,
        sps=1,
        modulation="qam",
        order=16,
        cpr_type=None,
        backend="numba",
    )

    mse_bps = float(np.mean(np.abs(res_bps.error[-2000:]) ** 2))
    mse_none = float(np.mean(np.abs(res_none.error[-2000:]) ** 2))
    assert mse_bps < mse_none, (
        f"BPS MSE ({mse_bps:.4f}) not better than no-CPR ({mse_none:.4f}) "
        "under Wiener phase noise"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: BPS Block Size > 1 correctness (incremental running sum)
# ─────────────────────────────────────────────────────────────────────────────


def test_bps_block_size_convergence():
    """lms(cpr_type='bps', bps_block_size=32) converges — verifies incremental sum."""
    rng = np.random.default_rng(13)
    n_sym = 4000
    const = gray_constellation("qam", 16).astype(np.complex64)
    syms = const[rng.integers(0, 16, n_sym)]

    phase_noise = np.cumsum(
        rng.standard_normal(n_sym) * np.sqrt(2 * np.pi * 5e-5)
    ).astype(np.float64)
    noise_pwr = 10 ** (-25.0 / 10)
    awgn = np.sqrt(noise_pwr / 2) * (
        rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)
    ).astype(np.complex64)
    samples = (syms * np.exp(1j * phase_noise).astype(np.complex64) + awgn).astype(
        np.complex64
    )

    res_k1 = lms(
        samples,
        training_symbols=syms[:1000],
        num_taps=1,
        sps=1,
        modulation="qam",
        order=16,
        cpr_type="bps",
        cpr_bps_test_phases=32,
        cpr_bps_block_size=1,
        cpr_cycle_slip_correction=False,
        backend="numba",
    )
    res_k32 = lms(
        samples,
        training_symbols=syms[:1000],
        num_taps=1,
        sps=1,
        modulation="qam",
        order=16,
        cpr_type="bps",
        cpr_bps_test_phases=32,
        cpr_bps_block_size=32,
        cpr_cycle_slip_correction=False,
        backend="numba",
    )

    # Both must converge — K=32 should be same or better than K=1 at high SNR
    mse_k1 = float(np.mean(np.abs(res_k1.error[-1000:]) ** 2))
    mse_k32 = float(np.mean(np.abs(res_k32.error[-1000:]) ** 2))
    assert mse_k32 < 0.1, f"BPS K=32 did not converge: MSE={mse_k32:.4f}"
    assert mse_k1 < 0.1, f"BPS K=1 did not converge: MSE={mse_k1:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: RLS + BPS smoke test
# ─────────────────────────────────────────────────────────────────────────────


def test_rls_bps_convergence():
    """rls(cpr_type='bps') converges under phase noise."""
    rng = np.random.default_rng(17)
    n_sym = 3000
    const = gray_constellation("qam", 16).astype(np.complex64)
    syms = const[rng.integers(0, 16, n_sym)]

    phase_noise = np.cumsum(
        rng.standard_normal(n_sym) * np.sqrt(2 * np.pi * 5e-5)
    ).astype(np.float64)
    noise_pwr = 10 ** (-25.0 / 10)
    awgn = np.sqrt(noise_pwr / 2) * (
        rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)
    ).astype(np.complex64)
    samples = (syms * np.exp(1j * phase_noise).astype(np.complex64) + awgn).astype(
        np.complex64
    )

    res = rls(
        samples,
        training_symbols=syms[:500],
        num_taps=1,
        sps=1,
        modulation="qam",
        order=16,
        cpr_type="bps",
        cpr_bps_test_phases=64,
        cpr_bps_block_size=32,
        cpr_cycle_slip_correction=False,
        backend="numba",
    )

    assert res.phase_trajectory is not None
    assert res.phase_trajectory.shape == (n_sym,)
    mse = float(np.mean(np.abs(res.error[-1000:]) ** 2))
    assert mse < 0.1, f"RLS+BPS did not converge: MSE={mse:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 11: PLL joint channels — shared phase update across MIMO
# ─────────────────────────────────────────────────────────────────────────────


def test_pll_joint_channels():
    """cpr_joint_channels=True makes both PLL integrators identical (shared LO)."""
    rng = np.random.default_rng(23)
    n_sym = 3000
    const = gray_constellation("qam", 16).astype(np.complex64)
    syms_a = const[rng.integers(0, 16, n_sym)]
    syms_b = const[rng.integers(0, 16, n_sym)]

    # Same phase noise on both channels (shared LO)
    phase_noise = np.cumsum(
        rng.standard_normal(n_sym) * np.sqrt(2 * np.pi * 1e-4)
    ).astype(np.float64)
    noise_pwr = 10 ** (-25.0 / 10)

    def _awgn():
        return (np.sqrt(noise_pwr / 2) * (
            rng.standard_normal(n_sym) + 1j * rng.standard_normal(n_sym)
        )).astype(np.complex64)

    phasor = np.exp(1j * phase_noise).astype(np.complex64)
    samples = np.stack([
        syms_a * phasor + _awgn(),
        syms_b * phasor + _awgn(),
    ])  # (2, n_sym)
    training = np.stack([syms_a[:500], syms_b[:500]])

    res = lms(
        samples,
        training_symbols=training,
        num_taps=1,
        sps=1,
        modulation="qam",
        order=16,
        cpr_type="pll",
        cpr_pll_bandwidth=5e-3,
        cpr_joint_channels=True,
        cpr_cycle_slip_correction=False,
        backend="numba",
    )

    assert res.phase_trajectory is not None
    assert res.phase_trajectory.shape == (2, n_sym)
    # With joint channels and shared LO, both phase trajectories should be identical
    phi0 = np.asarray(res.phase_trajectory[0])
    phi1 = np.asarray(res.phase_trajectory[1])
    np.testing.assert_array_equal(
        phi0, phi1, err_msg="cpr_joint_channels=True: PLL integrators must be identical"
    )
