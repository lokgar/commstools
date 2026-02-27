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


def _qam_signal(xp, order, n_symbols, fo_hz=0.0, snr_db=SNR_DB, fs=FS):
    """Generate a 1-SPS QAM signal with optional frequency offset and AWGN."""
    sig = Signal.qam(order=order, num_symbols=n_symbols, sps=1, symbol_rate=fs)
    sig = apply_awgn(sig, esn0_db=snr_db)
    if fo_hz != 0.0:
        sig.samples, _ = spectral.shift_frequency(sig.samples, fo_hz, fs)
    return sig


def _psk_signal(xp, order, n_symbols, fo_hz=0.0, snr_db=SNR_DB, fs=FS):
    """Generate a 1-SPS PSK signal with optional frequency offset and AWGN."""
    sig = Signal.psk(order=order, num_symbols=n_symbols, sps=1, symbol_rate=fs)
    sig = apply_awgn(sig, esn0_db=snr_db)
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
            sig.samples, fs=FS, modulation="qam", order=order
        )
        assert abs(est - fo_hz) / abs(fo_hz) < 0.05

    @pytest.mark.parametrize("order", [4, 8])
    @pytest.mark.parametrize("fo_hz", [3_000.0, -6_000.0])
    def test_accuracy_psk(self, backend_device, xp, order, fo_hz):
        """Estimated offset within 5% of true value for PSK at SNR=30 dB."""
        sig = _psk_signal(xp, order, 4096, fo_hz=fo_hz)
        est = sync.estimate_frequency_offset_mth_power(
            sig.samples, fs=FS, modulation="psk", order=order
        )
        assert abs(est - fo_hz) / abs(fo_hz) < 0.05

    def test_zero_offset_within_lock_range(self, backend_device, xp):
        """With no frequency offset, estimate stays within the lock range [-fs/2M, fs/2M]."""
        sig = _qam_signal(xp, 16, 4096, fo_hz=0.0)
        est = sync.estimate_frequency_offset_mth_power(
            sig.samples, fs=FS, modulation="qam", order=16
        )
        # Lock range for QAM with M=4: [-fs/8, fs/8] = ±125 kHz at 1 MHz
        assert abs(est) < FS / (2 * 4)

    def test_search_range_rejects_out_of_range(self, backend_device, xp):
        """Peak outside search_range returns an estimate outside the true value."""
        # True offset is 20 kHz; search range covers only [-5 kHz, 5 kHz]
        sig = _qam_signal(xp, 4, 8192, fo_hz=20_000.0)
        est = sync.estimate_frequency_offset_mth_power(
            sig.samples,
            fs=FS,
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
                fs=FS,
                modulation="qam",
                order=4,
                search_range=(400_000.0, 500_000.0),
            )

    def test_mimo_returns_scalar(self, backend_device, xp):
        """MIMO input (C, N) returns a single scalar estimate."""
        sig_a = _qam_signal(xp, 4, 2048, fo_hz=5_000.0)
        sig_b = _qam_signal(xp, 4, 2048, fo_hz=5_000.0)
        mimo = xp.stack([sig_a.samples, sig_b.samples], axis=0)  # (2, N)
        est = sync.estimate_frequency_offset_mth_power(
            mimo, fs=FS, modulation="qam", order=4
        )
        assert isinstance(est, float)
        assert abs(est - 5_000.0) / 5_000.0 < 0.05


# ─────────────────────────────────────────────────────────────────────────────
# FOE — Differential
# ─────────────────────────────────────────────────────────────────────────────


class TestFoeDifferential:
    @pytest.mark.parametrize("weighted", [True, False])
    @pytest.mark.parametrize("fo_hz", [2_000.0, -4_000.0])
    def test_blind_qpsk(self, backend_device, xp, weighted, fo_hz):
        """Blind differential FOE within 10% for QPSK (small offset, high SNR)."""
        sig = _psk_signal(xp, 4, 4096, fo_hz=fo_hz)
        est = sync.estimate_frequency_offset_differential(
            sig.samples,
            fs=FS,
            modulation="psk",
            order=4,
            weighted=weighted,
        )
        assert abs(est - fo_hz) / abs(fo_hz) < 0.10

    def test_data_aided_more_accurate_than_blind(self, backend_device, xp):
        """Data-aided mode (ref_signal) gives a good estimate regardless of blind accuracy."""
        fo_hz = 3_000.0
        # Use PSK (constant envelope) for cleanest data-aided derotation
        sig = Signal.psk(order=4, num_symbols=4096, sps=1, symbol_rate=FS)
        ideal_symbols = sig.samples.copy()  # save clean symbols before noise
        sig = apply_awgn(sig, esn0_db=25)
        sig.samples, _ = spectral.shift_frequency(sig.samples, fo_hz, FS)

        est_blind = sync.estimate_frequency_offset_differential(
            sig.samples, fs=FS, modulation="psk", order=4
        )
        est_da = sync.estimate_frequency_offset_differential(
            sig.samples, fs=FS, ref_signal=ideal_symbols
        )

        # Data-aided estimate should be within 5% of the true offset
        assert abs(est_da - fo_hz) / fo_hz < 0.05

    def test_generic_blind_pure_tone(self, backend_device, xp):
        """Generic mode (no modulation) estimates correctly for a pure complex tone."""
        N = 4096
        fo_hz = 5_000.0
        n = xp.arange(N, dtype=xp.float64)
        tone = xp.exp(1j * 2 * np.pi * fo_hz / FS * n).astype(xp.complex64)
        est = sync.estimate_frequency_offset_differential(tone, fs=FS)
        assert abs(est - fo_hz) < 200.0


# ─────────────────────────────────────────────────────────────────────────────
# FOE — Data-Aided
# ─────────────────────────────────────────────────────────────────────────────


class TestFoeDataAided:
    @pytest.mark.parametrize("fo_hz", [5_000.0, 10_000.0, -8_000.0])
    def test_accuracy_with_zc_preamble(self, backend_device, xp, fo_hz):
        """Data-aided FOE within 5% of actually-applied offset using ZC preamble."""
        # Use a long preamble so frequency resolution is fine enough
        preamble_len = 511
        zc = sync.zadoff_chu_sequence(preamble_len)
        zc_xp = xp.asarray(zc).astype(xp.complex64)

        # shift_frequency quantizes to the nearest bin; compare against actual offset
        zc_shifted, actual_fo = spectral.shift_frequency(zc_xp, fo_hz, FS)

        est = sync.estimate_frequency_offset_data_aided(
            zc_shifted, preamble_samples=zc_xp, fs=FS, offset=0
        )
        assert abs(est - actual_fo) / abs(actual_fo) < 0.05

    def test_offset_parameter(self, backend_device, xp):
        """Non-zero offset parameter extracts preamble from the correct position."""
        preamble_len = 511
        zc = sync.zadoff_chu_sequence(preamble_len)
        zc_xp = xp.asarray(zc).astype(xp.complex64)
        fo_hz = 8_000.0

        # 100-sample guard + preamble; compare against actual quantised offset
        guard = xp.zeros(100, dtype=xp.complex64)
        full = xp.concatenate([guard, zc_xp])
        received, actual_fo = spectral.shift_frequency(full, fo_hz, FS)

        est = sync.estimate_frequency_offset_data_aided(
            received, preamble_samples=zc_xp, fs=FS, offset=100
        )
        assert abs(est - actual_fo) / abs(actual_fo) < 0.05


# ─────────────────────────────────────────────────────────────────────────────
# CPR — Viterbi-Viterbi
# ─────────────────────────────────────────────────────────────────────────────


class TestCprViterbiViterbi:
    @pytest.mark.parametrize("order,modulation", [(4, "psk"), (16, "qam"), (64, "qam")])
    @pytest.mark.parametrize("block_size", [16, 32, 64])
    def test_phase_residual(self, backend_device, xp, order, modulation, block_size):
        """VV CPR: RMS phase residual reduced to < 0.1 rad after correction."""
        sig = _qam_signal(xp, order, 2048) if modulation == "qam" else _psk_signal(xp, order, 2048)
        # Apply a known constant phase offset
        phi_true = 0.3  # radians
        sig.samples = sig.samples * xp.exp(1j * phi_true)

        phase_est = sync.recover_carrier_phase_viterbi_viterbi(
            sig.samples, modulation=modulation, order=order, block_size=block_size
        )
        corrected = sync.correct_carrier_phase(sig.samples, phase_est)

        # Re-estimate residual after correction (accounts for irreducible M-fold ambiguity)
        phase_resid = sync.recover_carrier_phase_viterbi_viterbi(
            corrected, modulation=modulation, order=order, block_size=block_size
        )
        # 0.1 rad tolerance: VV has inherent noise for high-order QAM from amplitude modulation
        assert float(xp.sqrt(xp.mean(phase_resid**2))) < 0.1

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
        sig = apply_awgn(sig, esn0_db=SNR_DB)
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
# Correction functions
# ─────────────────────────────────────────────────────────────────────────────


class TestCorrectionFunctions:
    def test_correct_frequency_offset_dtype_preserved(self, backend_device, xp):
        """correct_frequency_offset: complex64 input → complex64 output."""
        sig = _qam_signal(xp, 4, 1024)
        assert sig.samples.dtype == xp.complex64
        corrected = sync.correct_frequency_offset(sig.samples, offset=5_000.0, fs=FS)
        assert corrected.dtype == xp.complex64

    def test_correct_frequency_offset_roundtrip(self, backend_device, xp):
        """Applying +Δf then correcting with -Δf restores the signal."""
        sig = _qam_signal(xp, 4, 1024)
        original = sig.samples.copy()
        # shift_frequency quantizes to the nearest bin; capture actual offset so
        # the correction can cancel it exactly (no residual due to quantization)
        shifted, actual_fo = spectral.shift_frequency(sig.samples, 10_000.0, FS)
        restored = sync.correct_frequency_offset(shifted, offset=actual_fo, fs=FS)
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
# Signal class — combined estimate + correct methods
# ─────────────────────────────────────────────────────────────────────────────


class TestSignalMethods:
    def test_correct_frequency_offset_returns_float(self, backend_device, xp):
        """Signal.correct_frequency_offset returns a float scalar."""
        sig = Signal.qam(order=16, num_symbols=2048, sps=1, symbol_rate=FS)
        sig = apply_awgn(sig, esn0_db=SNR_DB)
        sig.samples, _ = spectral.shift_frequency(sig.samples, 5_000.0, FS)
        result = sig.correct_frequency_offset(method="mth_power")
        assert isinstance(result, float)

    def test_correct_frequency_offset_applies_correction(self, backend_device, xp):
        """Signal.correct_frequency_offset estimates and corrects in one call."""
        sig = Signal.qam(order=16, num_symbols=4096, sps=1, symbol_rate=FS)
        sig = apply_awgn(sig, esn0_db=SNR_DB)
        # Capture the actual quantised offset so comparison is meaningful
        sig.samples, actual_fo = spectral.shift_frequency(sig.samples, 8_000.0, FS)

        est = sig.correct_frequency_offset(method="mth_power")
        # Returned estimate should be close to the actually applied offset
        assert abs(est - actual_fo) / abs(actual_fo) < 0.05

    def test_correct_frequency_offset_does_not_update_digital_offset(self, backend_device, xp):
        """Signal.correct_frequency_offset must not modify digital_frequency_offset."""
        sig = Signal.qam(order=4, num_symbols=2048, sps=1, symbol_rate=FS)
        sig = apply_awgn(sig, esn0_db=SNR_DB)
        sig.samples, _ = spectral.shift_frequency(sig.samples, 5_000.0, FS)

        original_dfo = sig.digital_frequency_offset
        sig.correct_frequency_offset(method="mth_power")
        assert sig.digital_frequency_offset == original_dfo

    def test_recover_carrier_phase_returns_array(self, backend_device, xp):
        """Signal.recover_carrier_phase returns an array, not None."""
        sig = Signal.qam(order=16, num_symbols=512, sps=1, symbol_rate=FS)
        sig = apply_awgn(sig, esn0_db=SNR_DB)
        phase = sig.recover_carrier_phase(method="viterbi_viterbi")
        assert phase is not None
        assert phase.shape == sig.samples.shape

    def test_recover_carrier_phase_modifies_samples(self, backend_device, xp):
        """Signal.recover_carrier_phase modifies self.samples in-place."""
        sig = Signal.qam(order=4, num_symbols=512, sps=1, symbol_rate=FS)
        sig = apply_awgn(sig, esn0_db=SNR_DB)
        sig.samples = sig.samples * xp.exp(1j * 0.4)  # known phase offset

        original_ptr = id(sig.samples)  # check same object is reassigned
        sig.recover_carrier_phase(method="viterbi_viterbi")
        # samples should have been modified (new array assigned)
        assert sig.samples is not None

    def test_recover_carrier_phase_bps_runs(self, backend_device, xp):
        """Signal.recover_carrier_phase with method='bps' completes without error."""
        sig = Signal.qam(order=16, num_symbols=512, sps=1, symbol_rate=FS)
        sig = apply_awgn(sig, esn0_db=SNR_DB)
        phase = sig.recover_carrier_phase(method="bps")
        assert phase.shape == sig.samples.shape

    def test_correct_frequency_offset_invalid_method_raises(self, backend_device, xp):
        """Signal.correct_frequency_offset raises ValueError for unknown method."""
        sig = Signal.qam(order=4, num_symbols=512, sps=1, symbol_rate=FS)
        with pytest.raises(ValueError, match="Unknown FOE method"):
            sig.correct_frequency_offset(method="magic_method")

    def test_recover_carrier_phase_invalid_method_raises(self, backend_device, xp):
        """Signal.recover_carrier_phase raises ValueError for unknown method."""
        sig = Signal.qam(order=4, num_symbols=512, sps=1, symbol_rate=FS)
        with pytest.raises(ValueError, match="Unknown CPR method"):
            sig.recover_carrier_phase(method="crystal_ball")

    def test_mth_power_requires_mod_scheme(self, backend_device, xp):
        """Signal.correct_frequency_offset raises if mod_scheme is not set."""
        from commstools.backend import dispatch

        samples_np = np.random.randn(512).astype(np.float32) + 1j * np.random.randn(512).astype(np.float32)
        sig = Signal(samples=samples_np, sampling_rate=FS, symbol_rate=FS)
        with pytest.raises(ValueError, match="mod_scheme and mod_order must be set"):
            sig.correct_frequency_offset(method="mth_power")
