"""
Tests for Carrier Phase Recovery: Frequency Offset Estimation (FOE)
and Carrier Phase Recovery (CPR) algorithms in commstools.sync.
"""

import numpy as np
import pytest

from commstools import spectral, sync
from commstools.core import Signal
from commstools.impairments import apply_awgn

# ─────────────────────────────────────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────────────────────────────────────

FS = 1e6  # 1 MHz sampling rate, common to all tests
SNR_DB = 30  # generous SNR so numerical algorithms converge reliably


def _qam_signal(xp, order, n_symbols, fo_hz=0.0, snr_db=SNR_DB, fs=FS, seed=42):
    """Generate a 1-SPS QAM signal with optional frequency offset and AWGN."""
    sig = Signal.qam(order=order, num_symbols=n_symbols, sps=1, symbol_rate=fs, seed=seed)
    sig.samples = apply_awgn(sig.samples, esn0_db=snr_db, sps=1, seed=seed)
    if fo_hz != 0.0:
        sig.samples, _ = spectral.shift_frequency(sig.samples, fo_hz, fs)
    return sig


def _psk_signal(xp, order, n_symbols, fo_hz=0.0, snr_db=SNR_DB, fs=FS, seed=42):
    """Generate a 1-SPS PSK signal with optional frequency offset and AWGN."""
    sig = Signal.psk(order=order, num_symbols=n_symbols, sps=1, symbol_rate=fs, seed=seed)
    sig.samples = apply_awgn(sig.samples, esn0_db=snr_db, sps=1, seed=seed)
    if fo_hz != 0.0:
        sig.samples, _ = spectral.shift_frequency(sig.samples, fo_hz, fs)
    return sig


def _apply_phase_ramp(xp, samples, phase_per_sample):
    """Apply a linear phase ramp: samples[n] *= exp(j * n * phase_per_sample)."""
    n = xp.arange(samples.shape[-1], dtype=xp.float64)
    ramp = xp.exp(1j * n * phase_per_sample).astype(samples.dtype)
    return samples * ramp


def _rms_phase_error(xp, phase_est, phase_true):
    """RMS of the phase error, after removing any constant offset (M-fold ambiguity)."""
    err = phase_est - phase_true
    # Remove the mean bias (accounts for the irreducible global ambiguity)
    err = err - float(xp.mean(err))
    return float(xp.sqrt(xp.mean(err**2)))


# ─────────────────────────────────────────────────────────────────────────────
# FOE — M-th Power
# ─────────────────────────────────────────────────────────────────────────────


class TestFoeMthPower:
    @pytest.mark.parametrize("order", [4, 16, 64])
    @pytest.mark.parametrize("fo_hz", [5_000.0, -8_000.0, 15_000.0])
    def test_accuracy_qam(self, backend_device, xp, order, fo_hz):
        """Estimated offset within 5% of true value for QAM at SNR=30 dB."""
        sig = _qam_signal(xp, order, 4096, fo_hz=fo_hz)
        est = sync.estimate_frequency_offset_mth_power(
            sig.samples, sampling_rate=FS, modulation="qam", order=order
        )
        assert abs(est - fo_hz) / abs(fo_hz) < 0.05

    @pytest.mark.parametrize("order", [4, 8])
    @pytest.mark.parametrize("fo_hz", [3_000.0, -6_000.0])
    def test_accuracy_psk(self, backend_device, xp, order, fo_hz):
        """Estimated offset within 5% of true value for PSK at SNR=30 dB."""
        sig = _psk_signal(xp, order, 4096, fo_hz=fo_hz)
        est = sync.estimate_frequency_offset_mth_power(
            sig.samples, sampling_rate=FS, modulation="psk", order=order
        )
        assert abs(est - fo_hz) / abs(fo_hz) < 0.05

    def test_zero_offset_within_lock_range(self, backend_device, xp):
        """With no frequency offset, estimate stays within the lock range [-fs/2M, fs/2M]."""
        sig = _qam_signal(xp, 16, 4096, fo_hz=0.0)
        est = sync.estimate_frequency_offset_mth_power(
            sig.samples, sampling_rate=FS, modulation="qam", order=16
        )
        # Lock range for QAM with M=4: [-fs/8, fs/8] = ±125 kHz at 1 MHz
        assert abs(est) < FS / (2 * 4)

    def test_search_range_rejects_out_of_range(self, backend_device, xp):
        """Peak outside search_range returns an estimate outside the true value."""
        # True offset is 20 kHz; search range covers only [-5 kHz, 5 kHz]
        sig = _qam_signal(xp, 4, 8192, fo_hz=20_000.0)
        est = sync.estimate_frequency_offset_mth_power(
            sig.samples,
            sampling_rate=FS,
            modulation="qam",
            order=4,
            search_range=(-5_000.0, 5_000.0),
        )
        # The true offset must not be found — estimate stays within the window
        assert abs(est) <= 5_000.0 + 200.0  # small tolerance for bin quantization

    def test_search_range_empty_raises(self, backend_device, xp):
        """Empty search_range raises ValueError."""
        sig = _qam_signal(xp, 4, 1024, fo_hz=0.0)
        # For QPSK (M=4) at 1 MHz, the tone window for range=[fs/2, fs] would be empty
        # Use a range that maps to negative frequencies only to force empty positive window
        with pytest.raises(ValueError, match="empty search window"):
            sync.estimate_frequency_offset_mth_power(
                sig.samples,
                sampling_rate=FS,
                modulation="qam",
                order=4,
                search_range=(400_000.0, 500_000.0),
            )

    def test_mimo_returns_per_channel(self, backend_device, xp):
        """MIMO input (C, N) returns ndarray(C,) by default."""
        sig_a = _qam_signal(xp, 4, 2048, fo_hz=5_000.0)
        sig_b = _qam_signal(xp, 4, 2048, fo_hz=5_000.0)
        mimo = xp.stack([sig_a.samples, sig_b.samples], axis=0)  # (2, N)
        est = sync.estimate_frequency_offset_mth_power(
            mimo, sampling_rate=FS, modulation="qam", order=4
        )
        assert isinstance(est, np.ndarray)
        assert est.shape == (2,)
        assert all(abs(e - 5_000.0) / 5_000.0 < 0.05 for e in est)

    def test_mimo_combine_channels_returns_scalar(self, backend_device, xp):
        """MIMO + combine_channels=True returns a single Python float."""
        sig_a = _qam_signal(xp, 4, 2048, fo_hz=5_000.0)
        sig_b = _qam_signal(xp, 4, 2048, fo_hz=5_000.0)
        mimo = xp.stack([sig_a.samples, sig_b.samples], axis=0)
        est = sync.estimate_frequency_offset_mth_power(
            mimo, sampling_rate=FS, modulation="qam", order=4, combine_channels=True
        )
        assert isinstance(est, float)
        assert abs(est - 5_000.0) / 5_000.0 < 0.05


# ─────────────────────────────────────────────────────────────────────────────
# CPR — Viterbi-Viterbi
# ─────────────────────────────────────────────────────────────────────────────


class TestCprViterbiViterbi:
    @pytest.mark.parametrize(
        "order,modulation,block_size",
        [
            # PSK: constant envelope → reliable even with small blocks
            (4, "psk", 16), (4, "psk", 32), (4, "psk", 64),
            # QAM-16: moderate amplitude variation → works with block_size ≥ 16
            (16, "qam", 16), (16, "qam", 32), (16, "qam", 64),
            # QAM-64: high amplitude variation in s^4 → requires block_size ≥ 32
            # for the coherent sum to be stable near the ±π/M unwrap boundary
            (64, "qam", 32), (64, "qam", 64),
        ],
    )
    def test_phase_residual(self, backend_device, xp, order, modulation, block_size):
        """VV CPR: mean phase estimate within 0.1 rad of true carrier phase (mod M-fold)."""
        sig = _qam_signal(xp, order, 2048) if modulation == "qam" else _psk_signal(xp, order, 2048)
        phi_true = 0.3  # radians
        sig.samples = sig.samples * xp.exp(1j * phi_true)

        phase_est = sync.recover_carrier_phase_viterbi_viterbi(
            sig.samples, modulation=modulation, order=order, block_size=block_size
        )

        # Check that the mean estimate is within 0.1 rad of phi_true, modulo the
        # irreducible M-fold (2π/M) ambiguity.  Wrapping the error to [-π/M, π/M)
        # removes the ambiguity and leaves only the estimation error.
        M = 4 if modulation == "qam" else order
        step = 2 * np.pi / M
        err = float(xp.mean(phase_est)) - phi_true
        err = err - step * round(err / step)  # wrap to [-step/2, step/2)
        assert abs(err) < 0.1

    def test_output_shape_siso(self, backend_device, xp):
        """VV CPR: 1D input → 1D phase output of same length."""
        sig = _qam_signal(xp, 16, 512)
        phase = sync.recover_carrier_phase_viterbi_viterbi(
            sig.samples, modulation="qam", order=16
        )
        assert phase.shape == sig.samples.shape

    def test_output_shape_mimo(self, backend_device, xp):
        """VV CPR: 2D input (C, N) → 2D phase output (C, N)."""
        sig_a = _qam_signal(xp, 4, 512)
        sig_b = _qam_signal(xp, 4, 512)
        mimo = xp.stack([sig_a.samples, sig_b.samples])  # (2, 512)
        phase = sync.recover_carrier_phase_viterbi_viterbi(
            mimo, modulation="qam", order=4
        )
        assert phase.shape == mimo.shape

    def test_too_short_raises(self, backend_device, xp):
        """VV CPR: signal shorter than block_size raises ValueError."""
        sig = _qam_signal(xp, 4, 20)
        with pytest.raises(ValueError, match="shorter than block_size"):
            sync.recover_carrier_phase_viterbi_viterbi(
                sig.samples[:10], modulation="qam", order=4, block_size=32
            )


# ─────────────────────────────────────────────────────────────────────────────
# CPR — Blind Phase Search (BPS)
# ─────────────────────────────────────────────────────────────────────────────


class TestCprBps:
    @pytest.mark.parametrize("order", [16, 64])
    def test_phase_residual(self, backend_device, xp, order):
        """BPS CPR: RMS phase residual < 0.05 rad for constant phase offset."""
        sig = _qam_signal(xp, order, 1024)
        phi_true = 0.2  # radians
        sig.samples = sig.samples * xp.exp(1j * phi_true)

        phase_est = sync.recover_carrier_phase_bps(
            sig.samples, modulation="qam", order=order
        )
        corrected = sync.correct_carrier_phase(sig.samples, phase_est)

        phase_resid = sync.recover_carrier_phase_bps(
            corrected, modulation="qam", order=order
        )
        assert float(xp.sqrt(xp.mean(phase_resid**2))) < 0.05

    def test_output_shape_siso(self, backend_device, xp):
        """BPS CPR: 1D input → 1D phase output of same length."""
        sig = _qam_signal(xp, 16, 512)
        phase = sync.recover_carrier_phase_bps(
            sig.samples, modulation="qam", order=16
        )
        assert phase.shape == sig.samples.shape

    def test_output_shape_mimo(self, backend_device, xp):
        """BPS CPR: 2D input (C, N) → 2D phase output (C, N)."""
        sig_a = _qam_signal(xp, 16, 512)
        sig_b = _qam_signal(xp, 16, 512)
        mimo = xp.stack([sig_a.samples, sig_b.samples])
        phase = sync.recover_carrier_phase_bps(mimo, modulation="qam", order=16)
        assert phase.shape == mimo.shape

    def test_too_short_raises(self, backend_device, xp):
        """BPS CPR: signal shorter than block_size raises ValueError."""
        sig = _qam_signal(xp, 16, 20)
        with pytest.raises(ValueError, match="shorter than block_size"):
            sync.recover_carrier_phase_bps(
                sig.samples[:10], modulation="qam", order=16, block_size=32
            )


# ─────────────────────────────────────────────────────────────────────────────
# CPR — Pilots
# ─────────────────────────────────────────────────────────────────────────────


class TestCprPilots:
    def _pilot_setup(self, xp, n_symbols=512, pilot_period=16, phase_per_sym=0.001):
        """Return (noisy+rotated samples, pilot_indices, pilot_values, true_phase).

        pilot_values are the ideal (noiseless, unrotated) symbols at pilot positions,
        derived from the signal's source_symbols which are set by Signal.qam().
        """
        sig = Signal.qam(order=16, num_symbols=n_symbols, sps=1, symbol_rate=FS)
        # Save ideal symbols before adding noise
        ideal_symbols = xp.asarray(sig.samples.copy())
        sig.samples = apply_awgn(sig.samples, esn0_db=SNR_DB, sps=1)
        # Apply a slow linear phase ramp on top of noise
        sig.samples = _apply_phase_ramp(xp, sig.samples, phase_per_sym)

        pilot_indices = np.arange(0, n_symbols, pilot_period)
        # Known pilot values = noiseless, unrotated symbols at pilot positions
        pilot_values = ideal_symbols[pilot_indices]
        true_phase = phase_per_sym * xp.arange(n_symbols, dtype=xp.float64)

        return sig.samples, pilot_indices, pilot_values, true_phase

    @pytest.mark.parametrize("pilot_period", [8, 16, 32])
    def test_phase_residual(self, backend_device, xp, pilot_period):
        """Pilot CPR: RMS residual < 0.05 rad for a linear phase ramp."""
        samples, pilot_indices, pilot_values, true_phase = self._pilot_setup(
            xp, n_symbols=512, pilot_period=pilot_period, phase_per_sym=0.001
        )
        phase_est = sync.recover_carrier_phase_pilots(
            samples, pilot_indices=pilot_indices, pilot_values=pilot_values
        )
        err = _rms_phase_error(xp, phase_est, true_phase)
        assert err < 0.05

    def test_output_shape_siso(self, backend_device, xp):
        """Pilot CPR: 1D input → 1D phase output."""
        samples, pilot_indices, pilot_values, _ = self._pilot_setup(xp)
        phase = sync.recover_carrier_phase_pilots(
            samples, pilot_indices=pilot_indices, pilot_values=pilot_values
        )
        assert phase.shape == samples.shape

    def test_output_shape_mimo(self, backend_device, xp):
        """Pilot CPR: 2D input (C, N) → 2D phase output (C, N)."""
        samples_a, pilot_indices, pilot_values, _ = self._pilot_setup(xp, n_symbols=256)
        samples_b, _, _, _ = self._pilot_setup(xp, n_symbols=256)
        mimo = xp.stack([samples_a, samples_b])  # (2, 256)
        phase = sync.recover_carrier_phase_pilots(
            mimo, pilot_indices=pilot_indices, pilot_values=pilot_values
        )
        assert phase.shape == mimo.shape

    def test_large_phase_unwrap(self, backend_device, xp):
        """Pilot CPR: unwrapping handles cumulative phase > 2π correctly."""
        n_symbols = 512
        # Phase ramp of 4π total (spans two full cycles)
        phase_per_sym = 4 * np.pi / n_symbols
        samples, pilot_indices, pilot_values, true_phase = self._pilot_setup(
            xp, n_symbols=n_symbols, pilot_period=8, phase_per_sym=phase_per_sym
        )
        phase_est = sync.recover_carrier_phase_pilots(
            samples, pilot_indices=pilot_indices, pilot_values=pilot_values
        )
        err = _rms_phase_error(xp, phase_est, true_phase)
        assert err < 0.1  # relaxed tolerance for large phase

    def test_cubic_interpolation(self, backend_device, xp):
        """Cubic interpolation works on both CPU and GPU."""
        samples, pilot_indices, pilot_values, _ = self._pilot_setup(xp)
        phase = sync.recover_carrier_phase_pilots(
            samples,
            pilot_indices=pilot_indices,
            pilot_values=pilot_values,
            interpolation="cubic",
        )
        assert phase.shape == samples.shape

    def test_invalid_interpolation_raises(self, backend_device, xp):
        """Unknown interpolation method raises ValueError."""
        samples, pilot_indices, pilot_values, _ = self._pilot_setup(xp)
        with pytest.raises(ValueError, match="Unknown interpolation method"):
            sync.recover_carrier_phase_pilots(
                samples,
                pilot_indices=pilot_indices,
                pilot_values=pilot_values,
                interpolation="spline_42",
            )


# ─────────────────────────────────────────────────────────────────────────────
# CPR — Tikhonov-RTS
# ─────────────────────────────────────────────────────────────────────────────


class TestCprTikhonov:
    @pytest.mark.parametrize(
        "order,modulation,block_size",
        [
            (4, "psk", 16),
            (4, "psk", 32),
            (16, "qam", 16),
            (16, "qam", 32),
            (64, "qam", 32),
        ],
    )
    def test_phase_residual(self, backend_device, xp, order, modulation, block_size):
        """Tikhonov CPR: mean estimate within 0.1 rad of true carrier phase (mod M-fold)."""
        sig = _qam_signal(xp, order, 2048) if modulation == "qam" else _psk_signal(xp, order, 2048)
        phi_true = 0.3
        sig.samples = sig.samples * xp.exp(1j * phi_true)

        phase_est = sync.recover_carrier_phase_tikhonov(
            sig.samples,
            modulation=modulation,
            order=order,
            linewidth_symbol_periods=1e-4,
            block_size=block_size,
            snr_db=SNR_DB,
        )

        M = 4 if modulation == "qam" else order
        step = 2 * np.pi / M
        err = float(xp.mean(phase_est)) - phi_true
        err = err - step * round(err / step)
        assert abs(err) < 0.1

    def test_output_shape_siso(self, backend_device, xp):
        """Tikhonov CPR: 1D input → 1D output of same length."""
        sig = _qam_signal(xp, 16, 512)
        phase = sync.recover_carrier_phase_tikhonov(
            sig.samples, modulation="qam", order=16, linewidth_symbol_periods=1e-4, snr_db=SNR_DB
        )
        assert phase.shape == sig.samples.shape

    def test_output_shape_mimo(self, backend_device, xp):
        """Tikhonov CPR: 2D input (C, N) → 2D output (C, N)."""
        sig_a = _qam_signal(xp, 16, 512)
        sig_b = _qam_signal(xp, 16, 512)
        mimo = xp.stack([sig_a.samples, sig_b.samples])
        phase = sync.recover_carrier_phase_tikhonov(
            mimo, modulation="qam", order=16, linewidth_symbol_periods=1e-4, snr_db=SNR_DB
        )
        assert phase.shape == mimo.shape

    def test_too_short_raises(self, backend_device, xp):
        """Tikhonov CPR: signal shorter than block_size raises ValueError."""
        sig = _qam_signal(xp, 4, 20)
        with pytest.raises(ValueError, match="shorter than block_size"):
            sync.recover_carrier_phase_tikhonov(
                sig.samples[:10], modulation="qam", order=4,
                linewidth_symbol_periods=1e-4, block_size=32,
            )

    def test_invalid_method_raises(self, backend_device, xp):
        """Tikhonov CPR: unknown method raises ValueError."""
        sig = _qam_signal(xp, 16, 512)
        with pytest.raises(ValueError, match="Unknown method"):
            sync.recover_carrier_phase_tikhonov(
                sig.samples, modulation="qam", order=16,
                linewidth_symbol_periods=1e-4, method="bad",
            )

    @pytest.mark.parametrize("order,modulation", [(4, "psk"), (16, "qam")])
    def test_sskf_phase_residual(self, backend_device, xp, order, modulation):
        """Tikhonov SSKF: mean estimate within 0.1 rad of true offset (mod M-fold)."""
        sig = _qam_signal(xp, order, 2048) if modulation == "qam" else _psk_signal(xp, order, 2048)
        phi_true = 0.3
        sig.samples = sig.samples * xp.exp(1j * phi_true)

        phase_est = sync.recover_carrier_phase_tikhonov(
            sig.samples, modulation=modulation, order=order,
            linewidth_symbol_periods=1e-4, snr_db=SNR_DB, method="sskf",
        )

        M = 4 if modulation == "qam" else order
        step = 2 * np.pi / M
        err = float(xp.mean(phase_est)) - phi_true
        err = err - step * round(err / step)
        assert abs(err) < 0.1

    def test_sskf_exact_close(self, backend_device, xp):
        """SSKF and exact RTS produce similar phase estimates (within 0.05 rad RMS)."""
        sig = _qam_signal(xp, 16, 2048)
        sig.samples = sig.samples * xp.exp(1j * 0.2)

        phi_exact = sync.recover_carrier_phase_tikhonov(
            sig.samples, modulation="qam", order=16,
            linewidth_symbol_periods=1e-4, snr_db=SNR_DB, method="exact",
        )
        phi_sskf = sync.recover_carrier_phase_tikhonov(
            sig.samples, modulation="qam", order=16,
            linewidth_symbol_periods=1e-4, snr_db=SNR_DB, method="sskf",
        )
        rms_diff = float(xp.sqrt(xp.mean((phi_exact - phi_sskf) ** 2)))
        assert rms_diff < 0.05

    def test_smoother_reduces_noise_vs_vv(self, backend_device, xp):
        """Tikhonov produces smoother phase trajectory than VV when σ_p² < σ_v².

        Regime: QPSK at snr_db=15, linewidth_symbol_periods=1e-7, block_size=32.
          σ_p² = 2π · 1e-7 · 32 ≈ 2e-5 rad²/block  (slow phase noise)
          σ_v² = 1/(4² · 31.6 · 32) ≈ 6e-5 rad²/block  (noisy VV at 15 dB)
        K_∞ ≈ 0.4  →  substantial smoothing: Tikhonov std < VV std.
        """
        linewidth_symbol_periods = 1e-7
        snr_test = 15
        sig = _psk_signal(xp, 4, 2048, snr_db=snr_test, seed=123)
        sig.samples = sig.samples * xp.exp(1j * 0.3)

        phi_vv = sync.recover_carrier_phase_viterbi_viterbi(
            sig.samples, modulation="psk", order=4, block_size=32
        )
        phi_tik = sync.recover_carrier_phase_tikhonov(
            sig.samples, modulation="psk", order=4,
            linewidth_symbol_periods=linewidth_symbol_periods, block_size=32, snr_db=snr_test,
        )

        # With constant true phase, VV block estimates fluctuate around the
        # true value with std ≈ sqrt(σ_v²).  The Tikhonov smoother suppresses
        # this noise: std(phi_tik) < std(phi_vv).
        assert float(xp.std(phi_tik)) < float(xp.std(phi_vv))


# ─────────────────────────────────────────────────────────────────────────────
# Correction functions
# ─────────────────────────────────────────────────────────────────────────────


class TestCorrectionFunctions:
    def test_correct_frequency_offset_dtype_preserved(self, backend_device, xp):
        """correct_frequency_offset: complex64 input → complex64 output."""
        sig = _qam_signal(xp, 4, 1024)
        assert sig.samples.dtype == xp.complex64
        corrected = sync.correct_frequency_offset(sig.samples, offset=5_000.0, sampling_rate=FS)
        assert corrected.dtype == xp.complex64

    def test_correct_frequency_offset_roundtrip(self, backend_device, xp):
        """Applying +Δf then correcting with -Δf restores the signal."""
        sig = _qam_signal(xp, 4, 1024)
        original = sig.samples.copy()
        # shift_frequency quantizes to the nearest bin; capture actual offset so
        # the correction can cancel it exactly (no residual due to quantization)
        shifted, actual_fo = spectral.shift_frequency(sig.samples, 10_000.0, FS)
        restored = sync.correct_frequency_offset(shifted, offset=actual_fo, sampling_rate=FS)
        assert float(xp.max(xp.abs(restored - original))) < 1e-4

    def test_correct_carrier_phase_dtype_preserved(self, backend_device, xp):
        """correct_carrier_phase: complex64 input → complex64 output."""
        sig = _qam_signal(xp, 4, 512)
        phase = xp.zeros(512, dtype=xp.float64)
        corrected = sync.correct_carrier_phase(sig.samples, phase)
        assert corrected.dtype == xp.complex64

    def test_correct_carrier_phase_zero_phase_identity(self, backend_device, xp):
        """Applying zero phase correction leaves samples unchanged."""
        sig = _qam_signal(xp, 4, 512)
        phase = xp.zeros(512, dtype=xp.float64)
        corrected = sync.correct_carrier_phase(sig.samples, phase)
        assert float(xp.max(xp.abs(corrected - sig.samples))) < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# FOE — Scattered Pilots (phase slope)
# ─────────────────────────────────────────────────────────────────────────────


class TestFoePilots:
    """Tests for estimate_frequency_offset_pilots (least-squares phase slope)."""

    N_SAMPLES = 4096
    PILOT_PERIOD = 16  # one pilot every 16 samples → lock range ±31.25 kHz

    @staticmethod
    def _setup(xp, fo_hz, n_samples=4096, pilot_period=16, snr_db=SNR_DB, fs=FS):
        """
        Build a QPSK-like signal with BPSK (+1) pilots inserted at a regular
        grid, apply an exact frequency offset, and return everything needed
        by the estimator.
        """
        rng = np.random.default_rng(seed=7)
        # Random QPSK-like data (4-point alphabet, unit power)
        symbols = (
            rng.choice([-1, 1], size=n_samples)
            + 1j * rng.choice([-1, 1], size=n_samples)
        ).astype(np.complex64) / np.sqrt(2)

        pilot_indices = np.arange(0, n_samples, pilot_period, dtype=np.intp)
        pilot_values = np.ones(len(pilot_indices), dtype=np.complex64)  # BPSK pilots
        symbols[pilot_indices] = pilot_values  # overwrite data with known pilots

        if snr_db < 100:
            noise_power = 10 ** (-snr_db / 10)
            noise = (
                rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
            ).astype(np.complex64) * np.sqrt(noise_power / 2)
            symbols = (symbols + noise).astype(np.complex64)

        samples = xp.asarray(symbols)
        # Apply exact frequency offset (no bin quantization)
        t = xp.arange(n_samples, dtype=xp.float64) / fs
        samples = (samples * xp.exp(1j * 2 * np.pi * fo_hz * t).astype(xp.complex64))
        return samples, pilot_indices, pilot_values

    @pytest.mark.parametrize("fo_hz", [1_000.0, 5_000.0, -3_000.0])
    def test_accuracy(self, backend_device, xp, fo_hz):
        """Estimated offset within 1 % of true offset at 30 dB SNR."""
        samples, pilot_indices, pilot_values = self._setup(xp, fo_hz)
        est = sync.estimate_frequency_offset_pilots(
            samples, pilot_indices=pilot_indices, pilot_values=pilot_values, sampling_rate=FS
        )
        assert abs(est - fo_hz) < 0.01 * abs(fo_hz) + 1.0

    def test_zero_offset(self, backend_device, xp):
        """Zero frequency offset: estimate is within ±20 Hz."""
        samples, pilot_indices, pilot_values = self._setup(xp, fo_hz=0.0)
        est = sync.estimate_frequency_offset_pilots(
            samples, pilot_indices=pilot_indices, pilot_values=pilot_values, sampling_rate=FS
        )
        assert abs(est) < 20.0

    def test_mimo_returns_per_channel(self, backend_device, xp):
        """MIMO input returns ndarray(C,) by default."""
        fo_hz = 2_000.0
        s0, pilot_indices, pilot_values = self._setup(xp, fo_hz)
        s1, _, _ = self._setup(xp, fo_hz)
        samples_mimo = xp.stack([s0, s1], axis=0)  # (2, N)
        est = sync.estimate_frequency_offset_pilots(
            samples_mimo,
            pilot_indices=pilot_indices,
            pilot_values=pilot_values,
            sampling_rate=FS,
        )
        assert isinstance(est, np.ndarray)
        assert est.shape == (2,)
        assert all(abs(e - fo_hz) < 0.01 * fo_hz + 1.0 for e in est)

    def test_mimo_combine_channels_returns_scalar(self, backend_device, xp):
        """MIMO + combine_channels=True returns a single Python float."""
        fo_hz = 2_000.0
        s0, pilot_indices, pilot_values = self._setup(xp, fo_hz)
        s1, _, _ = self._setup(xp, fo_hz)
        samples_mimo = xp.stack([s0, s1], axis=0)
        est = sync.estimate_frequency_offset_pilots(
            samples_mimo,
            pilot_indices=pilot_indices,
            pilot_values=pilot_values,
            sampling_rate=FS,
            combine_channels=True,
        )
        assert isinstance(est, float)
        assert abs(est - fo_hz) < 0.01 * fo_hz + 1.0


# ─────────────────────────────────────────────────────────────────────────────
# FOE — Mengali-Morelli multi-lag
# ─────────────────────────────────────────────────────────────────────────────


class TestFoeMengaliMorelli:
    """Tests for estimate_frequency_offset_mengali_morelli."""

    @pytest.mark.parametrize("fo_hz", [5_000.0, -12_000.0, 30_000.0])
    @pytest.mark.parametrize("order", [4, 16])
    def test_blind_qam_accuracy(self, backend_device, xp, fo_hz, order):
        """Blind QAM mode: estimate within 2 % of true offset at 30 dB SNR.

        Uses exact complex mixing (no bin quantization) so the true offset is
        known precisely.
        """
        sig = _qam_signal(xp, order, 4096, fo_hz=0.0)  # generate without offset
        # Apply exact frequency offset via direct complex mixing
        n = xp.arange(sig.samples.shape[-1], dtype=xp.float64)
        sig.samples = (sig.samples * xp.exp(1j * 2 * np.pi * fo_hz / FS * n)).astype(
            sig.samples.dtype
        )
        est = sync.estimate_frequency_offset_mengali_morelli(
            sig.samples, sampling_rate=FS, modulation="qam", order=order
        )
        assert abs(est - fo_hz) / abs(fo_hz) < 0.02

    @pytest.mark.parametrize("fo_hz", [4_000.0, -8_000.0])
    def test_data_aided_accuracy(self, backend_device, xp, fo_hz):
        """Data-aided mode: estimate within 1 % using known reference (exact mixing)."""
        sig = Signal.psk(order=4, num_symbols=4096, sps=1, symbol_rate=FS)
        ideal = sig.samples.copy()
        sig.samples = apply_awgn(sig.samples, esn0_db=25, sps=1)
        n = xp.arange(sig.samples.shape[-1], dtype=xp.float64)
        sig.samples = (sig.samples * xp.exp(1j * 2 * np.pi * fo_hz / FS * n)).astype(
            sig.samples.dtype
        )
        est = sync.estimate_frequency_offset_mengali_morelli(
            sig.samples, sampling_rate=FS, ref_signal=ideal
        )
        assert abs(est - fo_hz) / abs(fo_hz) < 0.01

    def test_large_offset_near_nyquist(self, backend_device, xp):
        """M&M lock range [-fs/2, fs/2]: succeeds at 40 % of Nyquist where Kay wraps."""
        N = 4096
        fo_hz = 0.40 * FS  # 40 % of sampling rate — well beyond Kay lock range for QPSK
        n = xp.arange(N, dtype=xp.float64)
        tone = xp.exp(1j * 2 * np.pi * fo_hz / FS * n).astype(xp.complex64)
        # Generic blind mode (no modulation — pure tone)
        est = sync.estimate_frequency_offset_mengali_morelli(tone, sampling_rate=FS)
        assert abs(est - fo_hz) < 0.02 * fo_hz

    def test_generic_blind_pure_tone(self, backend_device, xp):
        """Generic mode (no modulation): pure tone estimated accurately."""
        N = 2048
        fo_hz = 7_500.0
        n = xp.arange(N, dtype=xp.float64)
        tone = xp.exp(1j * 2 * np.pi * fo_hz / FS * n).astype(xp.complex64)
        est = sync.estimate_frequency_offset_mengali_morelli(tone, sampling_rate=FS)
        assert abs(est - fo_hz) < 500.0

    def test_mimo_returns_per_channel(self, backend_device, xp):
        """MIMO (C, N) input returns ndarray(C,) by default."""
        fo_hz = 6_000.0
        sig_a = _qam_signal(xp, 4, 2048, fo_hz=0.0)
        sig_b = _qam_signal(xp, 4, 2048, fo_hz=0.0)
        n = xp.arange(2048, dtype=xp.float64)
        mixer = xp.exp(1j * 2 * np.pi * fo_hz / FS * n).astype(xp.complex64)
        mimo = xp.stack([sig_a.samples * mixer, sig_b.samples * mixer], axis=0)
        est = sync.estimate_frequency_offset_mengali_morelli(
            mimo, sampling_rate=FS, modulation="qam", order=4
        )
        assert isinstance(est, np.ndarray)
        assert est.shape == (2,)
        assert all(abs(e - fo_hz) / fo_hz < 0.02 for e in est)

    def test_mimo_combine_channels_returns_scalar(self, backend_device, xp):
        """MIMO + combine_channels=True returns a single Python float."""
        fo_hz = 6_000.0
        sig_a = _qam_signal(xp, 4, 2048, fo_hz=0.0)
        sig_b = _qam_signal(xp, 4, 2048, fo_hz=0.0)
        n = xp.arange(2048, dtype=xp.float64)
        mixer = xp.exp(1j * 2 * np.pi * fo_hz / FS * n).astype(xp.complex64)
        mimo = xp.stack([sig_a.samples * mixer, sig_b.samples * mixer], axis=0)
        est = sync.estimate_frequency_offset_mengali_morelli(
            mimo, sampling_rate=FS, modulation="qam", order=4, combine_channels=True
        )
        assert isinstance(est, float)
        assert abs(est - fo_hz) / fo_hz < 0.02

    def test_custom_max_lag(self, backend_device, xp):
        """Custom max_lag parameter: still converges to correct estimate (exact mixing)."""
        fo_hz = 3_000.0
        sig = _qam_signal(xp, 4, 2048, fo_hz=0.0)
        n = xp.arange(sig.samples.shape[-1], dtype=xp.float64)
        sig.samples = (sig.samples * xp.exp(1j * 2 * np.pi * fo_hz / FS * n)).astype(
            sig.samples.dtype
        )
        est = sync.estimate_frequency_offset_mengali_morelli(
            sig.samples, sampling_rate=FS, modulation="qam", order=4, max_lag=16
        )
        assert abs(est - fo_hz) / fo_hz < 0.05


# ─────────────────────────────────────────────────────────────────────────────
# FOE — Regression: Jacobsen interpolation, WLSQ pilots
# ─────────────────────────────────────────────────────────────────────────────


class TestFoeRegression:

    def test_jacobsen_vs_parabolic_accuracy(self, backend_device, xp):
        """Jacobsen interpolation accuracy is at least as good as parabolic for N=256."""
        fo_hz = 7_777.0  # non-round number to stress sub-bin interpolation
        sig = _qam_signal(xp, 4, 256, fo_hz=fo_hz)
        est_j = sync.estimate_frequency_offset_mth_power(
            sig.samples, sampling_rate=FS, modulation="qam", order=4, interpolation="jacobsen"
        )
        est_p = sync.estimate_frequency_offset_mth_power(
            sig.samples, sampling_rate=FS, modulation="qam", order=4, interpolation="parabolic"
        )
        # Jacobsen error must be ≤ parabolic error (with generous 20 % slack for noise)
        assert abs(est_j - fo_hz) <= abs(est_p - fo_hz) * 1.2 + 100.0

    def test_mth_power_short_signal_raises(self, backend_device, xp):
        """M-th power FOE raises ValueError for signals shorter than 8 samples."""
        short = xp.ones(5, dtype=xp.complex64)
        with pytest.raises(ValueError, match="too short"):
            sync.estimate_frequency_offset_mth_power(
                short, sampling_rate=FS, modulation="qam", order=4
            )

    def test_pilot_wlsq_vs_ols_at_snr(self, backend_device, xp):
        """WLSQ pilot FOE returns a valid estimate; result within 2% of true offset."""
        fo_hz = 5_000.0
        n_samples = 2048
        pilot_period = 8
        rng = np.random.default_rng(42)
        symbols = (
            rng.choice([-1, 1], size=n_samples)
            + 1j * rng.choice([-1, 1], size=n_samples)
        ).astype(np.complex64) / np.sqrt(2)
        pilot_indices = np.arange(0, n_samples, pilot_period, dtype=np.intp)
        pilot_values = np.ones(len(pilot_indices), dtype=np.complex64)
        symbols[pilot_indices] = pilot_values
        noise = (
            rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
        ).astype(np.complex64) * np.sqrt(10 ** (-SNR_DB / 10) / 2)
        symbols = (symbols + noise).astype(np.complex64)
        t = np.arange(n_samples, dtype=np.float64) / FS
        samples = xp.asarray(
            symbols * np.exp(1j * 2 * np.pi * fo_hz * t).astype(np.complex64)
        )
        est = sync.estimate_frequency_offset_pilots(
            samples,
            pilot_indices=pilot_indices,
            pilot_values=pilot_values,
            sampling_rate=FS,
            snr_weighted=True,
        )
        assert abs(est - fo_hz) / fo_hz < 0.02


# ─────────────────────────────────────────────────────────────────────────────
# Joint channels — BPS, VV, Tikhonov
# ─────────────────────────────────────────────────────────────────────────────


class TestJointChannels:
    """joint_channels=True produces identical rows and zero inter-channel spread."""

    N = 4096
    PHASE = 0.3  # rad constant carrier phase offset

    def _make_mimo(self, xp, order=16, phase=PHASE, snr_db=SNR_DB):
        sig_a = _qam_signal(xp, order, self.N, snr_db=snr_db, seed=1)
        sig_b = _qam_signal(xp, order, self.N, snr_db=snr_db, seed=2)
        mimo = xp.stack([sig_a.samples, sig_b.samples], axis=0)
        mimo = mimo * xp.exp(1j * phase).astype(mimo.dtype)
        return mimo

    def test_bps_joint_rows_identical(self, backend_device, xp):
        """joint_channels=True: both phi_full rows are bitwise identical."""
        mimo = self._make_mimo(xp)
        phi = sync.recover_carrier_phase_bps(
            mimo, "qam", 16, joint_channels=True, cycle_slip_correction=False
        )
        assert phi.shape == (2, self.N)
        phi_np = phi if xp is np else phi.get()
        np.testing.assert_array_equal(phi_np[0], phi_np[1])

    def test_vv_joint_rows_identical(self, backend_device, xp):
        """VV joint_channels=True: both phi_full rows are bitwise identical."""
        mimo = self._make_mimo(xp)
        phi = sync.recover_carrier_phase_viterbi_viterbi(
            mimo, "qam", 16, joint_channels=True, cycle_slip_correction=False
        )
        assert phi.shape == (2, self.N)
        phi_np = phi if xp is np else phi.get()
        np.testing.assert_array_equal(phi_np[0], phi_np[1])

    def test_tikhonov_joint_rows_identical(self, backend_device, xp):
        """Tikhonov joint_channels=True: both phi_full rows are bitwise identical."""
        mimo = self._make_mimo(xp)
        phi = sync.recover_carrier_phase_tikhonov(
            mimo, "qam", 16,
            linewidth_symbol_periods=1e-4,
            snr_db=SNR_DB,
            joint_channels=True,
            cycle_slip_correction=False,
        )
        assert phi.shape == (2, self.N)
        phi_np = phi if xp is np else phi.get()
        np.testing.assert_array_equal(phi_np[0], phi_np[1])

    def test_bps_joint_zero_spread(self, backend_device, xp):
        """Joint BPS: inter-channel spread is exactly zero."""
        mimo = self._make_mimo(xp, snr_db=20)
        phi_joint = sync.recover_carrier_phase_bps(
            mimo, "qam", 16, joint_channels=True, cycle_slip_correction=False
        )
        phi_np = phi_joint if xp is np else phi_joint.get()
        assert float(np.std(phi_np[0] - phi_np[1])) == 0.0

    def test_siso_joint_noop(self, backend_device, xp):
        """joint_channels=True on SISO returns identical result to False."""
        sig = _qam_signal(xp, 16, self.N, seed=7)
        phi_a = sync.recover_carrier_phase_bps(
            sig.samples, "qam", 16, joint_channels=False, cycle_slip_correction=False
        )
        phi_b = sync.recover_carrier_phase_bps(
            sig.samples, "qam", 16, joint_channels=True, cycle_slip_correction=False
        )
        phi_a_np = phi_a if xp is np else phi_a.get()
        phi_b_np = phi_b if xp is np else phi_b.get()
        np.testing.assert_allclose(phi_a_np, phi_b_np, atol=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# Cycle-slip correction
# ─────────────────────────────────────────────────────────────────────────────


class TestCycleSlipCorrection:
    """correct_cycle_slips() detects and corrects injected slips."""

    def test_standalone_no_slip(self, backend_device, xp):
        """Smooth linear ramp with no slips is returned unchanged."""
        B = 200
        phi_u = np.linspace(0.0, 2.0, B)
        phi_out = sync.correct_cycle_slips(phi_u.copy(), symmetry=4, history_length=50)
        np.testing.assert_allclose(phi_out, phi_u, atol=1e-10)

    def test_standalone_single_slip(self, backend_device, xp):
        """A single injected pi/2 slip is corrected back to the original ramp."""
        B = 300
        phi_u = np.linspace(0.0, 1.0, B)
        phi_slipped = phi_u.copy()
        phi_slipped[150:] += np.pi / 2
        phi_out = sync.correct_cycle_slips(phi_slipped, symmetry=4, history_length=100)
        np.testing.assert_allclose(phi_out, phi_u, atol=0.05)

    def test_standalone_multiple_slips(self, backend_device, xp):
        """Multiple +/-pi/2 slips are all corrected."""
        B = 500
        phi_u = np.linspace(0.0, 1.5, B)
        phi_slipped = phi_u.copy()
        phi_slipped[100:] += np.pi / 2
        phi_slipped[300:] -= np.pi / 2
        phi_out = sync.correct_cycle_slips(phi_slipped, symmetry=4, history_length=80)
        np.testing.assert_allclose(phi_out, phi_u, atol=0.05)

    def test_bps_correction_bounded_output(self, backend_device, xp):
        """BPS cycle_slip_correction=True returns phase within reasonable bounds."""
        sig = _qam_signal(xp, 16, 2048, snr_db=SNR_DB)
        phi = sync.recover_carrier_phase_bps(sig.samples, "qam", 16, cycle_slip_correction=True)
        assert phi.shape == sig.samples.shape
        phi_np = phi if xp is np else phi.get()
        assert np.max(np.abs(phi_np)) < 10 * np.pi

    def test_vv_correction_shape(self, backend_device, xp):
        """VV cycle_slip_correction=True returns correct shape."""
        sig = _qam_signal(xp, 16, 2048, snr_db=SNR_DB)
        phi = sync.recover_carrier_phase_viterbi_viterbi(
            sig.samples, "qam", 16, cycle_slip_correction=True
        )
        assert phi.shape == sig.samples.shape

    def test_tikhonov_correction_shape(self, backend_device, xp):
        """Tikhonov cycle_slip_correction=True returns correct shape."""
        sig = _qam_signal(xp, 16, 2048, snr_db=SNR_DB)
        phi = sync.recover_carrier_phase_tikhonov(
            sig.samples, "qam", 16,
            linewidth_symbol_periods=1e-4,
            snr_db=SNR_DB,
            cycle_slip_correction=True,
        )
        assert phi.shape == sig.samples.shape


# ─────────────────────────────────────────────────────────────────────────────
# Phase ambiguity resolution
# ─────────────────────────────────────────────────────────────────────────────


class TestResolvePhaseAmbiguity:
    """resolve_phase_ambiguity selects the rotation with lowest SER."""

    N = 2048

    def test_best_rotation_is_zero(self, backend_device, xp):
        """Already-aligned symbols: k=0 chosen and SER is minimal."""
        from commstools.helpers import normalize
        from commstools.metrics import ser
        sig = _qam_signal(xp, 16, self.N, snr_db=30, seed=5)
        sym = normalize(sig.samples, "average_power")
        ref = normalize(xp.asarray(sig.source_symbols), "average_power")
        resolved = sync.resolve_phase_ambiguity(sym, ref, "qam", 16)
        s0 = float(ser(resolved, ref, "qam", 16))
        for k in range(1, 4):
            sk = float(ser(resolved * xp.exp(1j * k * np.pi / 2).astype(sym.dtype), ref, "qam", 16))
            assert s0 <= sk + 1e-6

    def test_corrects_pi_half_rotation(self, backend_device, xp):
        """Symbols rotated by pi/2 are corrected; post-resolution SER is low."""
        from commstools.helpers import normalize
        from commstools.metrics import ser
        sig = _qam_signal(xp, 16, self.N, snr_db=30, seed=5)
        sym = normalize(sig.samples, "average_power")
        ref = normalize(xp.asarray(sig.source_symbols), "average_power")
        rotated = sym * xp.exp(1j * np.pi / 2).astype(sym.dtype)
        resolved = sync.resolve_phase_ambiguity(rotated, ref, "qam", 16)
        assert float(ser(resolved, ref, "qam", 16)) < 0.05

    def test_mimo_independent_per_channel(self, backend_device, xp):
        """MIMO: channels with different rotations are each independently corrected."""
        from commstools.helpers import normalize
        from commstools.metrics import ser
        sig_a = _qam_signal(xp, 16, self.N, seed=1)
        sig_b = _qam_signal(xp, 16, self.N, seed=2)
        sym_a = normalize(sig_a.samples, "average_power")
        sym_b = normalize(sig_b.samples, "average_power")
        ref_a = normalize(xp.asarray(sig_a.source_symbols), "average_power")
        ref_b = normalize(xp.asarray(sig_b.source_symbols), "average_power")
        mimo = xp.stack([sym_a * xp.exp(1j * np.pi / 2).astype(sym_a.dtype),
                         sym_b * xp.exp(1j * np.pi).astype(sym_b.dtype)], axis=0)
        ref_mimo = xp.stack([ref_a, ref_b], axis=0)
        resolved = sync.resolve_phase_ambiguity(mimo, ref_mimo, "qam", 16)
        assert resolved.shape == (2, self.N)
        s = ser(resolved, ref_mimo, "qam", 16)
        s_np = s if xp is np else s.get()
        assert float(s_np[0]) < 0.05
        assert float(s_np[1]) < 0.05

    def test_signal_method_in_place(self, backend_device, xp):
        """Signal.resolve_phase_ambiguity() updates resolved_symbols in place."""
        from commstools.helpers import normalize
        from commstools.metrics import ser
        sig = Signal.qam(order=16, num_symbols=self.N, sps=1, symbol_rate=1e6, seed=9)
        sig.samples = apply_awgn(sig.samples, esn0_db=30, sps=1, seed=9)
        sym = normalize(sig.samples, "average_power")
        sig.resolved_symbols = sym * xp.exp(1j * np.pi / 2).astype(sym.dtype)
        sig.resolve_phase_ambiguity()
        assert sig.resolved_symbols is not None
        ref = normalize(xp.asarray(sig.source_symbols), "average_power")
        assert float(ser(sig.resolved_symbols, ref, "qam", 16)) < 0.1

    def test_signal_method_raises_without_resolved(self, backend_device, xp):
        """Raises ValueError when resolved_symbols is None."""
        sig = Signal.qam(order=16, num_symbols=256, sps=1, symbol_rate=1e6, seed=0)
        with pytest.raises(ValueError, match="resolved_symbols"):
            sig.resolve_phase_ambiguity()

    def test_signal_method_raises_without_source(self, backend_device, xp):
        """Raises ValueError when source_symbols is None."""
        sig = Signal.qam(order=16, num_symbols=256, sps=1, symbol_rate=1e6, seed=0)
        sig.resolved_symbols = sig.samples
        sig.source_symbols = None
        with pytest.raises(ValueError, match="source_symbols"):
            sig.resolve_phase_ambiguity()
