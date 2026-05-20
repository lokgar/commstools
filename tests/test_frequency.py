"""Tests for Frequency Offset Estimation (FOE) algorithms in commstools.frequency."""

import numpy as np
import pytest

from commstools import frequency, spectral
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


# ─────────────────────────────────────────────────────────────────────────────
# FOE — M-th Power
# ─────────────────────────────────────────────────────────────────────────────


class TestFoeMthPower:
    @pytest.mark.parametrize("order", [4, 16, 64])
    @pytest.mark.parametrize("fo_hz", [5_000.0, -8_000.0, 15_000.0])
    def test_accuracy_qam(self, backend_device, xp, order, fo_hz):
        """Estimated offset within 5% of true value for QAM at SNR=30 dB."""
        sig = _qam_signal(xp, order, 4096, fo_hz=fo_hz)
        est = frequency.estimate_frequency_offset_mth_power(
            sig.samples, sampling_rate=FS, modulation="qam", order=order
        )
        assert abs(est - fo_hz) / abs(fo_hz) < 0.05

    @pytest.mark.parametrize("order", [4, 8])
    @pytest.mark.parametrize("fo_hz", [3_000.0, -6_000.0])
    def test_accuracy_psk(self, backend_device, xp, order, fo_hz):
        """Estimated offset within 5% of true value for PSK at SNR=30 dB."""
        sig = _psk_signal(xp, order, 4096, fo_hz=fo_hz)
        est = frequency.estimate_frequency_offset_mth_power(
            sig.samples, sampling_rate=FS, modulation="psk", order=order
        )
        assert abs(est - fo_hz) / abs(fo_hz) < 0.05

    def test_zero_offset_within_lock_range(self, backend_device, xp):
        """With no frequency offset, estimate stays within the lock range [-fs/2M, fs/2M]."""
        sig = _qam_signal(xp, 16, 4096, fo_hz=0.0)
        est = frequency.estimate_frequency_offset_mth_power(
            sig.samples, sampling_rate=FS, modulation="qam", order=16
        )
        # Lock range for QAM with M=4: [-fs/8, fs/8] = ±125 kHz at 1 MHz
        assert abs(est) < FS / (2 * 4)

    def test_search_range_rejects_out_of_range(self, backend_device, xp):
        """Peak outside search_range returns an estimate outside the true value."""
        # True offset is 20 kHz; search range covers only [-5 kHz, 5 kHz]
        sig = _qam_signal(xp, 4, 8192, fo_hz=20_000.0)
        est = frequency.estimate_frequency_offset_mth_power(
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
        with pytest.raises(ValueError, match="empty search window"):
            frequency.estimate_frequency_offset_mth_power(
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
        est = frequency.estimate_frequency_offset_mth_power(
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
        est = frequency.estimate_frequency_offset_mth_power(
            mimo, sampling_rate=FS, modulation="qam", order=4, combine_channels=True
        )
        assert isinstance(est, float)
        assert abs(est - 5_000.0) / 5_000.0 < 0.05


# ─────────────────────────────────────────────────────────────────────────────
# FOE — Correction functions (frequency half)
# ─────────────────────────────────────────────────────────────────────────────


class TestCorrectionFunctions:
    def test_correct_frequency_offset_dtype_preserved(self, backend_device, xp):
        """correct_frequency_offset: complex64 input → complex64 output."""
        sig = _qam_signal(xp, 4, 1024)
        assert sig.samples.dtype == xp.complex64
        corrected = frequency.correct_frequency_offset(sig.samples, offset=5_000.0, sampling_rate=FS)
        assert corrected.dtype == xp.complex64

    def test_correct_frequency_offset_roundtrip(self, backend_device, xp):
        """Applying +Δf then correcting with -Δf restores the signal."""
        sig = _qam_signal(xp, 4, 1024)
        original = sig.samples.copy()
        # shift_frequency quantizes to the nearest bin; capture actual offset so
        # the correction can cancel it exactly (no residual due to quantization)
        shifted, actual_fo = spectral.shift_frequency(sig.samples, 10_000.0, FS)
        restored = frequency.correct_frequency_offset(shifted, offset=actual_fo, sampling_rate=FS)
        assert float(xp.max(xp.abs(restored - original))) < 1e-4


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
        est = frequency.estimate_frequency_offset_pilots(
            samples, pilot_indices=pilot_indices, pilot_values=pilot_values, sampling_rate=FS
        )
        assert abs(est - fo_hz) < 0.01 * abs(fo_hz) + 1.0

    def test_zero_offset(self, backend_device, xp):
        """Zero frequency offset: estimate is within ±20 Hz."""
        samples, pilot_indices, pilot_values = self._setup(xp, fo_hz=0.0)
        est = frequency.estimate_frequency_offset_pilots(
            samples, pilot_indices=pilot_indices, pilot_values=pilot_values, sampling_rate=FS
        )
        assert abs(est) < 20.0

    def test_mimo_returns_per_channel(self, backend_device, xp):
        """MIMO input returns ndarray(C,) by default."""
        fo_hz = 2_000.0
        s0, pilot_indices, pilot_values = self._setup(xp, fo_hz)
        s1, _, _ = self._setup(xp, fo_hz)
        samples_mimo = xp.stack([s0, s1], axis=0)  # (2, N)
        est = frequency.estimate_frequency_offset_pilots(
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
        est = frequency.estimate_frequency_offset_pilots(
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
        """Blind QAM mode: estimate within 2 % of true offset at 30 dB SNR."""
        sig = _qam_signal(xp, order, 4096, fo_hz=0.0)  # generate without offset
        # Apply exact frequency offset via direct complex mixing
        n = xp.arange(sig.samples.shape[-1], dtype=xp.float64)
        sig.samples = (sig.samples * xp.exp(1j * 2 * np.pi * fo_hz / FS * n)).astype(
            sig.samples.dtype
        )
        est = frequency.estimate_frequency_offset_mengali_morelli(
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
        est = frequency.estimate_frequency_offset_mengali_morelli(
            sig.samples, sampling_rate=FS, ref_signal=ideal
        )
        assert abs(est - fo_hz) / abs(fo_hz) < 0.01

    def test_large_offset_near_nyquist(self, backend_device, xp):
        """M&M lock range [-fs/2, fs/2]: succeeds at 40 % of Nyquist where Kay wraps."""
        N = 4096
        fo_hz = 0.40 * FS  # 40 % of sampling rate
        n = xp.arange(N, dtype=xp.float64)
        tone = xp.exp(1j * 2 * np.pi * fo_hz / FS * n).astype(xp.complex64)
        # Generic blind mode (no modulation — pure tone)
        est = frequency.estimate_frequency_offset_mengali_morelli(tone, sampling_rate=FS)
        assert abs(est - fo_hz) < 0.02 * fo_hz

    def test_generic_blind_pure_tone(self, backend_device, xp):
        """Generic mode (no modulation): pure tone estimated accurately."""
        N = 2048
        fo_hz = 7_500.0
        n = xp.arange(N, dtype=xp.float64)
        tone = xp.exp(1j * 2 * np.pi * fo_hz / FS * n).astype(xp.complex64)
        est = frequency.estimate_frequency_offset_mengali_morelli(tone, sampling_rate=FS)
        assert abs(est - fo_hz) < 500.0

    def test_mimo_returns_per_channel(self, backend_device, xp):
        """MIMO (C, N) input returns ndarray(C,) by default."""
        fo_hz = 6_000.0
        sig_a = _qam_signal(xp, 4, 2048, fo_hz=0.0)
        sig_b = _qam_signal(xp, 4, 2048, fo_hz=0.0)
        n = xp.arange(2048, dtype=xp.float64)
        mixer = xp.exp(1j * 2 * np.pi * fo_hz / FS * n).astype(xp.complex64)
        mimo = xp.stack([sig_a.samples * mixer, sig_b.samples * mixer], axis=0)
        est = frequency.estimate_frequency_offset_mengali_morelli(
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
        est = frequency.estimate_frequency_offset_mengali_morelli(
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
        est = frequency.estimate_frequency_offset_mengali_morelli(
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
        est_j = frequency.estimate_frequency_offset_mth_power(
            sig.samples, sampling_rate=FS, modulation="qam", order=4, interpolation="jacobsen"
        )
        est_p = frequency.estimate_frequency_offset_mth_power(
            sig.samples, sampling_rate=FS, modulation="qam", order=4, interpolation="parabolic"
        )
        # Jacobsen error must be ≤ parabolic error (with generous 20 % slack for noise)
        assert abs(est_j - fo_hz) <= abs(est_p - fo_hz) * 1.2 + 100.0

    def test_mth_power_short_signal_raises(self, backend_device, xp):
        """M-th power FOE raises ValueError for signals shorter than 8 samples."""
        short = xp.ones(5, dtype=xp.complex64)
        with pytest.raises(ValueError, match="too short"):
            frequency.estimate_frequency_offset_mth_power(
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
        est = frequency.estimate_frequency_offset_pilots(
            samples,
            pilot_indices=pilot_indices,
            pilot_values=pilot_values,
            sampling_rate=FS,
            snr_weighted=True,
        )
        assert abs(est - fo_hz) / fo_hz < 0.02


# =============================================================================
# CORRECT FREQUENCY OFFSET — REAL AND MIMO BRANCHES
# =============================================================================


class TestCorrectFrequencyOffsetBranches:
    """Branches for real-valued and MIMO input in correct_frequency_offset."""

    def test_real_float32_input(self, backend_device, xp):
        """Real float32 input is cast to complex64 and frequency-corrected."""

        N = 256
        t = xp.arange(N, dtype=xp.float32) / 1e6
        # Simple real cosine as stand-in for a real-baseband signal
        sig = xp.cos(t)
        sig = sig.astype(xp.float32)
        out = frequency.correct_frequency_offset(sig, offset=5000.0, sampling_rate=1e6)
        assert out.dtype == xp.complex64
        assert out.shape == sig.shape

    def test_mimo_input_broadcasts_mixer(self, backend_device, xp):
        """MIMO (C, N) input: mixer is broadcast over channels without error."""
        import numpy as np

        C, N = 2, 256
        rng = np.random.default_rng(99)
        sig = xp.asarray(
            (rng.standard_normal((C, N)) + 1j * rng.standard_normal((C, N))).astype(
                np.complex64
            )
        )
        out = frequency.correct_frequency_offset(sig, offset=3000.0, sampling_rate=1e6)
        assert out.shape == (C, N)
        assert out.dtype == xp.complex64
