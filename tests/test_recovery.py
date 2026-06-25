"""Tests for Carrier Phase Recovery (CPR) algorithms in commstools.recovery."""

import numpy as np
import pytest

from commstools import psk, qam, recovery, spectral
from commstools.backend import to_device
from commstools.impairments import apply_awgn
from commstools.mapping import gray_constellation

# ─────────────────────────────────────────────────────────────────────────────
# Test helpers
# ─────────────────────────────────────────────────────────────────────────────

FS = 1e6  # 1 MHz sampling rate, common to all tests
SNR_DB = 30  # generous SNR so numerical algorithms converge reliably


def _qam_signal(xp, order, n_symbols, fo_hz=0.0, snr_db=SNR_DB, fs=FS, seed=42):
    """Generate a 1-SPS QAM signal with optional frequency offset and AWGN."""
    sig = qam(order=order, num_symbols=n_symbols, sps=1, symbol_rate=fs, seed=seed)
    sig.samples = apply_awgn(sig.samples, esn0_db=snr_db, sps=1, seed=seed)
    if fo_hz != 0.0:
        sig.samples, _ = spectral.shift_frequency(sig.samples, fo_hz, fs)
    return sig


def _psk_signal(xp, order, n_symbols, fo_hz=0.0, snr_db=SNR_DB, fs=FS, seed=42):
    """Generate a 1-SPS PSK signal with optional frequency offset and AWGN."""
    sig = psk(order=order, num_symbols=n_symbols, sps=1, symbol_rate=fs, seed=seed)
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
# CPR — Viterbi-Viterbi (from test_carrier_sync.py)
# ─────────────────────────────────────────────────────────────────────────────


class TestCprViterbiViterbi:
    @pytest.mark.parametrize(
        "order,modulation,block_size",
        [
            (4, "psk", 16),
            (4, "psk", 32),
            (4, "psk", 64),
            (16, "qam", 16),
            (16, "qam", 32),
            (16, "qam", 64),
            (64, "qam", 32),
            (64, "qam", 64),
        ],
    )
    def test_phase_residual(self, backend_device, xp, order, modulation, block_size):
        """VV CPR: mean phase estimate within 0.1 rad of true carrier phase (mod M-fold)."""
        sig = (
            _qam_signal(xp, order, 2048)
            if modulation == "qam"
            else _psk_signal(xp, order, 2048)
        )
        phi_true = 0.3  # radians
        sig.samples = sig.samples * xp.exp(1j * phi_true)

        phase_est = recovery.recover_carrier_phase_viterbi_viterbi(
            sig.samples, modulation=modulation, order=order, block_size=block_size
        )

        M = 4 if modulation == "qam" else order
        step = 2 * np.pi / M
        err = float(xp.mean(phase_est)) - phi_true
        err = err - step * round(err / step)  # wrap to [-step/2, step/2)
        assert abs(err) < 0.1

    def test_output_shape_siso(self, backend_device, xp):
        """VV CPR: 1D input → 1D phase output of same length."""
        sig = _qam_signal(xp, 16, 512)
        phase = recovery.recover_carrier_phase_viterbi_viterbi(
            sig.samples, modulation="qam", order=16
        )
        assert phase.shape == sig.samples.shape

    def test_output_shape_mimo(self, backend_device, xp):
        """VV CPR: 2D input (C, N) → 2D phase output (C, N)."""
        sig_a = _qam_signal(xp, 4, 512)
        sig_b = _qam_signal(xp, 4, 512)
        mimo = xp.stack([sig_a.samples, sig_b.samples])  # (2, 512)
        phase = recovery.recover_carrier_phase_viterbi_viterbi(
            mimo, modulation="qam", order=4
        )
        assert phase.shape == mimo.shape

    def test_too_short_raises(self, backend_device, xp):
        """VV CPR: signal shorter than block_size raises ValueError."""
        sig = _qam_signal(xp, 4, 20)
        with pytest.raises(ValueError, match="shorter than block_size"):
            recovery.recover_carrier_phase_viterbi_viterbi(
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

        phase_est = recovery.recover_carrier_phase_bps(
            sig.samples, modulation="qam", order=order
        )
        corrected = recovery.correct_carrier_phase(sig.samples, phase_est)

        phase_resid = recovery.recover_carrier_phase_bps(
            corrected, modulation="qam", order=order
        )
        assert float(xp.sqrt(xp.mean(phase_resid**2))) < 0.05

    def test_output_shape_siso(self, backend_device, xp):
        """BPS CPR: 1D input → 1D phase output of same length."""
        sig = _qam_signal(xp, 16, 512)
        phase = recovery.recover_carrier_phase_bps(
            sig.samples, modulation="qam", order=16
        )
        assert phase.shape == sig.samples.shape

    def test_output_shape_mimo(self, backend_device, xp):
        """BPS CPR: 2D input (C, N) → 2D phase output (C, N)."""
        sig_a = _qam_signal(xp, 16, 512)
        sig_b = _qam_signal(xp, 16, 512)
        mimo = xp.stack([sig_a.samples, sig_b.samples])
        phase = recovery.recover_carrier_phase_bps(mimo, modulation="qam", order=16)
        assert phase.shape == mimo.shape

    def test_too_short_raises(self, backend_device, xp):
        """BPS CPR: signal shorter than block_size raises ValueError."""
        sig = _qam_signal(xp, 16, 20)
        with pytest.raises(ValueError, match="shorter than block_size"):
            recovery.recover_carrier_phase_bps(
                sig.samples[:10], modulation="qam", order=16, block_size=32
            )


# ─────────────────────────────────────────────────────────────────────────────
# CPR — Pilots
# ─────────────────────────────────────────────────────────────────────────────


class TestCprPilots:
    def _pilot_setup(self, xp, n_symbols=512, pilot_period=16, phase_per_sym=0.001):
        """Return (noisy+rotated samples, pilot_indices, pilot_values, true_phase)."""
        sig = qam(order=16, num_symbols=n_symbols, sps=1, symbol_rate=FS)
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
        phase_est = recovery.recover_carrier_phase_pilots(
            samples, pilot_indices=pilot_indices, pilot_values=pilot_values
        )
        err = _rms_phase_error(xp, phase_est, true_phase)
        assert err < 0.05

    def test_output_shape_siso(self, backend_device, xp):
        """Pilot CPR: 1D input → 1D phase output."""
        samples, pilot_indices, pilot_values, _ = self._pilot_setup(xp)
        phase = recovery.recover_carrier_phase_pilots(
            samples, pilot_indices=pilot_indices, pilot_values=pilot_values
        )
        assert phase.shape == samples.shape

    def test_output_shape_mimo(self, backend_device, xp):
        """Pilot CPR: 2D input (C, N) → 2D phase output (C, N)."""
        samples_a, pilot_indices, pilot_values, _ = self._pilot_setup(xp, n_symbols=256)
        samples_b, _, _, _ = self._pilot_setup(xp, n_symbols=256)
        mimo = xp.stack([samples_a, samples_b])  # (2, 256)
        phase = recovery.recover_carrier_phase_pilots(
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
        phase_est = recovery.recover_carrier_phase_pilots(
            samples, pilot_indices=pilot_indices, pilot_values=pilot_values
        )
        err = _rms_phase_error(xp, phase_est, true_phase)
        assert err < 0.1  # relaxed tolerance for large phase

    def test_cubic_interpolation(self, backend_device, xp):
        """Cubic interpolation works on both CPU and GPU."""
        samples, pilot_indices, pilot_values, _ = self._pilot_setup(xp)
        phase = recovery.recover_carrier_phase_pilots(
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
            recovery.recover_carrier_phase_pilots(
                samples,
                pilot_indices=pilot_indices,
                pilot_values=pilot_values,
                interpolation="spline_42",
            )


# ─────────────────────────────────────────────────────────────────────────────
# CPR — Pilot Tone
# ─────────────────────────────────────────────────────────────────────────────


class TestCprPilotTone:
    """Pilot-tone CPR on an oversampled RRC waveform with a guard-band tone."""

    SPS = 8
    BETA = 0.1  # RRC roll-off → data edge at (1+β)/2 · Rs
    F_TONE = 2.0e6  # guard band: > (1.1/2)·Rs (=0.55 MHz) + B, and < fs/2 (=4 MHz)
    BW = 0.3e6  # extraction half-width B
    EDGE = 600  # trim FFT-window edge ringing before comparing

    def _setup(
        self,
        xp,
        n_symbols=2000,
        df=0.0,
        linewidth=0.0,
        psr_db=-12.0,
        num_streams=1,
        seed=7,
    ):
        """Return (samples, fs, true_common_phase) for a tone-bearing waveform.

        The common phase = frequency-offset ramp + Wiener phase noise is applied
        *after* the tone, so the tone carries exactly the phase to be recovered.
        """
        fs = self.SPS * FS  # symbol rate = FS
        sig = qam(
            order=16,
            num_symbols=n_symbols,
            sps=self.SPS,
            symbol_rate=FS,
            rrc_rolloff=self.BETA,
            num_streams=num_streams,
            seed=seed,
        )
        samples = xp.asarray(sig.samples)
        samples, _ = spectral.add_pilot_tone(
            samples, fs, self.F_TONE, power_ratio_db=psr_db
        )

        N = samples.shape[-1]
        n = xp.arange(N, dtype=xp.float64)
        if linewidth > 0.0:
            rng = xp.random.RandomState(seed)
            incr = rng.normal(0.0, float(np.sqrt(2 * np.pi * linewidth / fs)), N)
            pn = xp.cumsum(incr)
        else:
            pn = xp.zeros(N, dtype=xp.float64)
        common = 2 * np.pi * df * n / fs + pn  # (N,) float64
        samples = samples * xp.exp(1j * common).astype(samples.dtype)
        return samples, fs, common

    def _interior_rms(self, xp, phase_est, common):
        g = slice(self.EDGE, -self.EDGE)
        return _rms_phase_error(xp, phase_est[..., g], common[..., g])

    @pytest.mark.parametrize("df", [0.0, 0.05e6, 0.1e6])
    def test_phase_residual_foe(self, backend_device, xp, df):
        """Recovers a frequency-offset ramp (< B) to < 0.1 rad RMS."""
        samples, fs, common = self._setup(xp, df=df)
        theta = recovery.recover_carrier_phase_pilot_tone(
            samples, fs, self.F_TONE, bandwidth=self.BW
        )
        assert self._interior_rms(xp, theta, common) < 0.1

    def test_phase_residual_foe_and_phase_noise(self, backend_device, xp):
        """Recovers joint frequency offset + Wiener phase noise."""
        samples, fs, common = self._setup(xp, df=0.05e6, linewidth=5e3)
        theta = recovery.recover_carrier_phase_pilot_tone(
            samples, fs, self.F_TONE, bandwidth=self.BW
        )
        assert self._interior_rms(xp, theta, common) < 0.15

    @pytest.mark.parametrize(
        "window", ["tukey", ("tukey", 0.3), "boxcar", ("gaussian", 250), "hann"]
    )
    def test_window_options(self, backend_device, xp, window):
        """Any scipy.get_window spec tracks the common phase to < 0.12 rad RMS."""
        samples, fs, common = self._setup(xp, df=0.05e6)
        theta = recovery.recover_carrier_phase_pilot_tone(
            samples, fs, self.F_TONE, bandwidth=self.BW, window=window
        )
        assert self._interior_rms(xp, theta, common) < 0.12

    def test_output_shape_siso(self, backend_device, xp):
        """1D input → 1D phase output of matching length."""
        samples, fs, _ = self._setup(xp)
        theta = recovery.recover_carrier_phase_pilot_tone(
            samples, fs, self.F_TONE, bandwidth=self.BW
        )
        assert theta.shape == samples.shape

    def test_output_shape_mimo(self, backend_device, xp):
        """2D input (C, N) → 2D phase output (C, N)."""
        samples, fs, _ = self._setup(xp, num_streams=2)
        theta = recovery.recover_carrier_phase_pilot_tone(
            samples, fs, self.F_TONE, bandwidth=self.BW
        )
        assert theta.shape == samples.shape

    def test_joint_rows_identical(self, backend_device, xp):
        """joint_channels broadcasts a single trajectory to all rows."""
        samples, fs, _ = self._setup(xp, num_streams=2, df=0.05e6)
        theta = recovery.recover_carrier_phase_pilot_tone(
            samples, fs, self.F_TONE, bandwidth=self.BW, joint_channels=True
        )
        assert bool(xp.allclose(theta[0], theta[1]))

    def test_remove_frequency_offset_false_leaves_pure_pn(self, backend_device, xp):
        """With FOE removal off, the estimate matches the detrended common phase."""
        samples, fs, common = self._setup(xp, df=0.08e6, linewidth=5e3)
        theta = recovery.recover_carrier_phase_pilot_tone(
            samples,
            fs,
            self.F_TONE,
            bandwidth=self.BW,
            remove_frequency_offset=False,
        )
        N = samples.shape[-1]
        n = np.arange(N, dtype=np.float64)
        common_np = np.asarray(to_device(common, "cpu"))
        common_detrended = common_np - np.polyval(np.polyfit(n, common_np, 1), n)
        theta_np = np.asarray(to_device(theta, "cpu"))
        g = slice(self.EDGE, -self.EDGE)
        err = theta_np[g] - common_detrended[g]
        err -= err.mean()
        assert float(np.sqrt(np.mean(err**2))) < 0.15

    def test_refine_relocates_large_offset(self, backend_device, xp):
        """An offset > B is tracked when the search band covers the shifted tone."""
        # df = 0.6 MHz exceeds B = 0.3 MHz: a window centred at nominal would miss
        # the tone, but refine + a wide search band relocates it.
        samples, fs, common = self._setup(xp, df=0.6e6)
        theta = recovery.recover_carrier_phase_pilot_tone(
            samples, fs, self.F_TONE, bandwidth=self.BW, search_band=1.0e6
        )
        assert self._interior_rms(xp, theta, common) < 0.1

    def test_refine_off_fails_when_tone_leaves_window(self, backend_device, xp):
        """Without refinement, an offset > B drags the tone out of the window."""
        samples, fs, common = self._setup(xp, df=0.6e6)
        theta = recovery.recover_carrier_phase_pilot_tone(
            samples, fs, self.F_TONE, bandwidth=self.BW, refine_tone=False
        )
        # The tone is outside the nominal window → estimate is garbage, not tracking.
        assert self._interior_rms(xp, theta, common) > 1.0

    def test_invalid_window_raises(self, backend_device, xp):
        samples, fs, _ = self._setup(xp, n_symbols=256)
        with pytest.raises(ValueError, match="Invalid window"):
            recovery.recover_carrier_phase_pilot_tone(
                samples, fs, self.F_TONE, bandwidth=self.BW, window="brick"
            )

    def test_invalid_bandwidth_raises(self, backend_device, xp):
        samples, fs, _ = self._setup(xp, n_symbols=256)
        with pytest.raises(ValueError, match="bandwidth must be > 0"):
            recovery.recover_carrier_phase_pilot_tone(
                samples, fs, self.F_TONE, bandwidth=0.0
            )

    def test_tone_frequency_out_of_range_raises(self, backend_device, xp):
        samples, fs, _ = self._setup(xp, n_symbols=256)
        with pytest.raises(ValueError, match=r"must lie in \(-fs/2, fs/2\)"):
            recovery.recover_carrier_phase_pilot_tone(
                samples, fs, fs, bandwidth=self.BW
            )


# ─────────────────────────────────────────────────────────────────────────────
# CPR — Two-pilot MRC (recover_carrier_phase_pilot_tones)
# ─────────────────────────────────────────────────────────────────────────────


class TestCprPilotTones:
    """Multi-pilot common-phase CPR: SNR-weighted MRC + slow inter-tone tracking."""

    SPS = 8
    BETA = 0.1
    F0 = 1.5e6  # tone 0 — guard band, below fs/2 = 4 MHz
    F1 = 2.0e6  # tone 1 — distinct frequency, 0.5 MHz away
    BW = 0.3e6  # per-tone extraction half-width
    DIFF_BW = 5e3  # slow-δ low-pass cut-off — narrow, so δ averages out the noise
    EDGE = 600

    def _setup(
        self,
        xp,
        df=0.0,
        linewidth=0.0,
        snr_db=SNR_DB,
        n_symbols=2000,
        psr_db=(-12.0, -12.0),
        delta=0.4,
        seed=7,
    ):
        """Two isolated tones (one per channel) sharing a common carrier phase.

        Models the post-demux case: channel 0 carries tone ``F0``, channel 1
        carries tone ``F1``; both ride the *same* common phase (FOE ramp + Wiener
        phase noise).  ``delta`` is a static inter-tone offset applied to tone 1
        (the differential the slow-δ tracker must absorb).  Returns
        ``(samples, fs, common)``.
        """
        fs = self.SPS * FS
        sig = qam(
            order=16,
            num_symbols=n_symbols,
            sps=self.SPS,
            symbol_rate=FS,
            rrc_rolloff=self.BETA,
            num_streams=2,
            seed=seed,
        )
        samples = apply_awgn(
            xp.asarray(sig.samples), esn0_db=snr_db, sps=self.SPS, seed=seed
        )
        # One tone per channel (channel c gets frequency[c]); tone 1 gets a static
        # phase offset δ so the inter-tone differential is non-trivial.
        samples, _ = spectral.add_pilot_tone(
            samples,
            fs,
            [self.F0, self.F1],
            power_ratio_db=list(psr_db),
            phase_init=0.0,
        )
        n = xp.arange(samples.shape[-1], dtype=xp.float64)
        samples[1] = samples[1] * xp.exp(1j * float(delta)).astype(samples.dtype)

        if linewidth > 0.0:
            rng = xp.random.RandomState(seed)
            incr = rng.normal(
                0.0, float(np.sqrt(2 * np.pi * linewidth / fs)), samples.shape[-1]
            )
            pn = xp.cumsum(incr)
        else:
            pn = xp.zeros(samples.shape[-1], dtype=xp.float64)
        common = 2 * np.pi * df * n / fs + pn
        samples = samples * xp.exp(1j * common).astype(samples.dtype)
        return samples, fs, common

    def _interior_rms(self, xp, phase_est, common):
        g = slice(self.EDGE, -self.EDGE)
        return _rms_phase_error(xp, phase_est[..., g], common[..., g])

    @pytest.mark.parametrize("df", [0.0, 0.05e6])
    def test_recovers_common_phase(self, backend_device, xp, df):
        """Two-pilot MRC tracks the shared FOE + phase noise to < 0.15 rad RMS."""
        samples, fs, common = self._setup(xp, df=df, linewidth=5e3)
        phi = recovery.recover_carrier_phase_pilot_tones(
            samples,
            fs,
            [self.F0, self.F1],
            bandwidth=self.BW,
            differential_bandwidth=self.DIFF_BW,
            per_tone_channel=[0, 1],
        )
        assert self._interior_rms(xp, phi[0], common) < 0.15

    def test_mrc_beats_single_tone(self, backend_device, xp):
        """Combining two equal-SNR pilots lowers the residual vs a single tone.

        Run in a noise-limited regime (low tone SNR) where the √2 of the combine
        is visible; a *narrow* δ low-pass is essential — a wide one tracks noise
        into δ and the combine can lose to single-tone.
        """
        samples, fs, common = self._setup(
            xp,
            df=0.0,
            linewidth=0.0,
            snr_db=6,
            psr_db=(-10.0, -10.0),
        )
        single = recovery.recover_carrier_phase_pilot_tone(
            samples,
            fs,
            self.F0,
            bandwidth=self.BW,
            joint_channels=False,
        )
        both, diag = recovery.recover_carrier_phase_pilot_tones(
            samples,
            fs,
            [self.F0, self.F1],
            bandwidth=self.BW,
            differential_bandwidth=self.DIFF_BW,
            per_tone_channel=[0, 1],
            return_diagnostics=True,
        )
        assert diag["used"] == [0, 1]  # both tones genuinely combined
        # single-tone reads tone 0 on channel 0; compare like-for-like on row 0.
        rms_single = self._interior_rms(xp, single[0], common)
        rms_both = self._interior_rms(xp, both[0], common)
        assert rms_both < rms_single

    def test_output_shape_and_common_broadcast(self, backend_device, xp):
        """(C, N) input → (C, N) output; the common track is identical on all rows."""
        samples, fs, _ = self._setup(xp, df=0.05e6)
        phi = recovery.recover_carrier_phase_pilot_tones(
            samples,
            fs,
            [self.F0, self.F1],
            bandwidth=self.BW,
            differential_bandwidth=self.DIFF_BW,
            per_tone_channel=[0, 1],
        )
        assert phi.shape == samples.shape
        assert bool(xp.allclose(phi[0], phi[1]))

    def test_gating_drops_faded_tone(self, backend_device, xp):
        """A weak pilot (low tone-to-noise) is gated out → single-tone fallback.

        Fade tone 1 by lowering its PSR (not by scaling the whole channel, which
        is SNR-invariant): a 48 dB weaker tone falls below the SNR gate.
        """
        samples, fs, _ = self._setup(xp, psr_db=(-12.0, -60.0))
        _, diag = recovery.recover_carrier_phase_pilot_tones(
            samples,
            fs,
            [self.F0, self.F1],
            bandwidth=self.BW,
            differential_bandwidth=self.DIFF_BW,
            per_tone_channel=[0, 1],
            snr_gate_db=3.0,
            return_diagnostics=True,
        )
        assert diag["used"] == [diag["ref"]]
        assert diag["ref"] == 0  # the strong tone

    def test_single_tone_K1_tracks(self, backend_device, xp):
        """K=1 reduces to a single-tone tracker and still recovers the phase."""
        samples, fs, common = self._setup(xp, df=0.05e6)
        phi = recovery.recover_carrier_phase_pilot_tones(
            samples,
            fs,
            [self.F0],
            bandwidth=self.BW,
            differential_bandwidth=self.DIFF_BW,
            per_tone_channel=[0],
        )
        assert self._interior_rms(xp, phi[0], common) < 0.15

    def test_joint_pre_demux_combine(self, backend_device, xp):
        """per_tone_channel=None coherently sums each tone across channels."""
        samples, fs, common = self._setup(xp, df=0.05e6)
        phi = recovery.recover_carrier_phase_pilot_tones(
            samples,
            fs,
            [self.F0, self.F1],
            bandwidth=self.BW,
            differential_bandwidth=self.DIFF_BW,
            per_tone_channel=None,
        )
        assert self._interior_rms(xp, phi[0], common) < 0.15

    def test_diagnostics_keys(self, backend_device, xp):
        samples, fs, _ = self._setup(xp)
        _, diag = recovery.recover_carrier_phase_pilot_tones(
            samples,
            fs,
            [self.F0, self.F1],
            bandwidth=self.BW,
            differential_bandwidth=self.DIFF_BW,
            per_tone_channel=[0, 1],
            return_diagnostics=True,
        )
        assert set(diag) == {"delta", "snr_db", "ref", "used", "f_centers"}
        assert len(diag["snr_db"]) == 2

    def test_per_tone_channel_length_mismatch_raises(self, backend_device, xp):
        samples, fs, _ = self._setup(xp, n_symbols=256)
        with pytest.raises(ValueError, match="per_tone_channel must have one entry"):
            recovery.recover_carrier_phase_pilot_tones(
                samples,
                fs,
                [self.F0, self.F1],
                bandwidth=self.BW,
                per_tone_channel=[0],
            )

    def test_invalid_bandwidth_raises(self, backend_device, xp):
        samples, fs, _ = self._setup(xp, n_symbols=256)
        with pytest.raises(ValueError, match="bandwidth must be > 0"):
            recovery.recover_carrier_phase_pilot_tones(
                samples,
                fs,
                [self.F0, self.F1],
                bandwidth=0.0,
            )

    def test_empty_tone_list_raises(self, backend_device, xp):
        samples, fs, _ = self._setup(xp, n_symbols=256)
        with pytest.raises(ValueError, match="at least one frequency"):
            recovery.recover_carrier_phase_pilot_tones(
                samples,
                fs,
                [],
                bandwidth=self.BW,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Wiener phase smoother (smooth_phase_wiener)
# ─────────────────────────────────────────────────────────────────────────────


class TestWienerPhaseSmoother:
    """Zero-phase Wiener smoother for a random-walk carrier phase."""

    def _random_walk(self, xp, N=20000, q=4e-4, seed=3):
        rng = xp.random.RandomState(seed)
        return xp.cumsum(rng.normal(0.0, float(np.sqrt(q)), N)), q

    def test_reduces_random_walk_noise(self, backend_device, xp):
        """Smoothing a noisy random-walk phase lowers the RMS error vs truth."""
        truth, q = self._random_walk(xp)
        rng = xp.random.RandomState(11)
        r = 0.05
        noisy = truth + rng.normal(0.0, float(np.sqrt(r)), truth.shape[-1])
        smoothed = recovery.smooth_phase_wiener(
            noisy, process_variance=q, measurement_variance=r
        )
        g = slice(200, -200)  # trim FFT edge transients
        rms_noisy = _rms_phase_error(xp, noisy[g], truth[g])
        rms_smooth = _rms_phase_error(xp, smoothed[g], truth[g])
        assert rms_smooth < rms_noisy

    def test_preserves_linear_trend(self, backend_device, xp):
        """A pure FOE ramp (+ tiny noise) survives the detrend/add-back path."""
        N = 8000
        n = xp.arange(N, dtype=xp.float64)
        slope = 1e-3
        ramp = slope * n
        rng = xp.random.RandomState(5)
        noisy = ramp + rng.normal(0.0, 0.01, N)
        smoothed = recovery.smooth_phase_wiener(
            noisy, process_variance=1e-6, measurement_variance=1e-2
        )
        g = slice(200, -200)
        # The recovered slope (via endpoints) matches the true ramp slope.
        est_slope = float((smoothed[g][-1] - smoothed[g][0]) / (n[g][-1] - n[g][0]))
        assert abs(est_slope - slope) < 0.05 * slope + 1e-5

    def test_shape_siso_and_mimo(self, backend_device, xp):
        truth, q = self._random_walk(xp, N=4000)
        phi1d = recovery.smooth_phase_wiener(
            truth, process_variance=q, measurement_variance=0.05
        )
        assert phi1d.shape == truth.shape
        phi2d_in = xp.stack([truth, truth + 0.3])
        phi2d = recovery.smooth_phase_wiener(
            phi2d_in, process_variance=q, measurement_variance=0.05
        )
        assert phi2d.shape == phi2d_in.shape

    def test_derive_variances_from_physical_params(self, backend_device, xp):
        """linewidth + sampling_rate + tone_snr derive q and r internally."""
        truth, _ = self._random_walk(xp, N=4000)
        smoothed = recovery.smooth_phase_wiener(
            truth, linewidth=1e5, sampling_rate=4e9, tone_snr=100.0
        )
        assert smoothed.shape == truth.shape
        assert bool(xp.all(xp.isfinite(smoothed)))

    def test_missing_process_params_raise(self, backend_device, xp):
        truth, _ = self._random_walk(xp, N=512)
        with pytest.raises(ValueError, match="process_variance"):
            recovery.smooth_phase_wiener(truth, measurement_variance=0.05)

    def test_missing_measurement_params_raise(self, backend_device, xp):
        truth, q = self._random_walk(xp, N=512)
        with pytest.raises(ValueError, match="measurement_variance, or tone_snr"):
            recovery.smooth_phase_wiener(truth, process_variance=q)

    def test_invalid_variance_raises(self, backend_device, xp):
        truth, _ = self._random_walk(xp, N=512)
        with pytest.raises(ValueError, match="must be > 0"):
            recovery.smooth_phase_wiener(
                truth, process_variance=0.0, measurement_variance=0.05
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
        sig = (
            _qam_signal(xp, order, 2048)
            if modulation == "qam"
            else _psk_signal(xp, order, 2048)
        )
        phi_true = 0.3
        sig.samples = sig.samples * xp.exp(1j * phi_true)

        phase_est = recovery.recover_carrier_phase_tikhonov(
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
        phase = recovery.recover_carrier_phase_tikhonov(
            sig.samples,
            modulation="qam",
            order=16,
            linewidth_symbol_periods=1e-4,
            snr_db=SNR_DB,
        )
        assert phase.shape == sig.samples.shape

    def test_output_shape_mimo(self, backend_device, xp):
        """Tikhonov CPR: 2D input (C, N) → 2D output (C, N)."""
        sig_a = _qam_signal(xp, 16, 512)
        sig_b = _qam_signal(xp, 16, 512)
        mimo = xp.stack([sig_a.samples, sig_b.samples])
        phase = recovery.recover_carrier_phase_tikhonov(
            mimo,
            modulation="qam",
            order=16,
            linewidth_symbol_periods=1e-4,
            snr_db=SNR_DB,
        )
        assert phase.shape == mimo.shape

    def test_too_short_raises(self, backend_device, xp):
        """Tikhonov CPR: signal shorter than block_size raises ValueError."""
        sig = _qam_signal(xp, 4, 20)
        with pytest.raises(ValueError, match="shorter than block_size"):
            recovery.recover_carrier_phase_tikhonov(
                sig.samples[:10],
                modulation="qam",
                order=4,
                linewidth_symbol_periods=1e-4,
                block_size=32,
            )

    def test_invalid_method_raises(self, backend_device, xp):
        """Tikhonov CPR: unknown method raises ValueError."""
        sig = _qam_signal(xp, 16, 512)
        with pytest.raises(ValueError, match="Unknown method"):
            recovery.recover_carrier_phase_tikhonov(
                sig.samples,
                modulation="qam",
                order=16,
                linewidth_symbol_periods=1e-4,
                method="bad",
            )

    @pytest.mark.parametrize("order,modulation", [(4, "psk"), (16, "qam")])
    def test_sskf_phase_residual(self, backend_device, xp, order, modulation):
        """Tikhonov SSKF: mean estimate within 0.1 rad of true offset (mod M-fold)."""
        sig = (
            _qam_signal(xp, order, 2048)
            if modulation == "qam"
            else _psk_signal(xp, order, 2048)
        )
        phi_true = 0.3
        sig.samples = sig.samples * xp.exp(1j * phi_true)

        phase_est = recovery.recover_carrier_phase_tikhonov(
            sig.samples,
            modulation=modulation,
            order=order,
            linewidth_symbol_periods=1e-4,
            snr_db=SNR_DB,
            method="sskf",
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

        phi_exact = recovery.recover_carrier_phase_tikhonov(
            sig.samples,
            modulation="qam",
            order=16,
            linewidth_symbol_periods=1e-4,
            snr_db=SNR_DB,
            method="exact",
        )
        phi_sskf = recovery.recover_carrier_phase_tikhonov(
            sig.samples,
            modulation="qam",
            order=16,
            linewidth_symbol_periods=1e-4,
            snr_db=SNR_DB,
            method="sskf",
        )
        rms_diff = float(xp.sqrt(xp.mean((phi_exact - phi_sskf) ** 2)))
        assert rms_diff < 0.05

    def test_smoother_reduces_noise_vs_vv(self, backend_device, xp):
        """Tikhonov produces smoother phase trajectory than VV when σ_p² < σ_v²."""
        linewidth_symbol_periods = 1e-7
        snr_test = 15
        sig = _psk_signal(xp, 4, 2048, snr_db=snr_test, seed=123)
        sig.samples = sig.samples * xp.exp(1j * 0.3)

        phi_vv = recovery.recover_carrier_phase_viterbi_viterbi(
            sig.samples, modulation="psk", order=4, block_size=32
        )
        phi_tik = recovery.recover_carrier_phase_tikhonov(
            sig.samples,
            modulation="psk",
            order=4,
            linewidth_symbol_periods=linewidth_symbol_periods,
            block_size=32,
            snr_db=snr_test,
        )

        assert float(xp.std(phi_tik)) < float(xp.std(phi_vv))


# ─────────────────────────────────────────────────────────────────────────────
# Correction functions (phase half)
# ─────────────────────────────────────────────────────────────────────────────


class TestCorrectionFunctions:
    def test_correct_carrier_phase_dtype_preserved(self, backend_device, xp):
        """correct_carrier_phase: complex64 input → complex64 output."""
        sig = _qam_signal(xp, 4, 512)
        phase = xp.zeros(512, dtype=xp.float64)
        corrected = recovery.correct_carrier_phase(sig.samples, phase)
        assert corrected.dtype == xp.complex64

    def test_correct_carrier_phase_zero_phase_identity(self, backend_device, xp):
        """Applying zero phase correction leaves samples unchanged."""
        sig = _qam_signal(xp, 4, 512)
        phase = xp.zeros(512, dtype=xp.float64)
        corrected = recovery.correct_carrier_phase(sig.samples, phase)
        assert float(xp.max(xp.abs(corrected - sig.samples))) < 1e-5


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
        phi = recovery.recover_carrier_phase_bps(
            mimo, "qam", 16, joint_channels=True, cycle_slip_correction=False
        )
        assert phi.shape == (2, self.N)
        phi_np = phi if xp is np else phi.get()
        np.testing.assert_array_equal(phi_np[0], phi_np[1])

    def test_vv_joint_rows_identical(self, backend_device, xp):
        """VV joint_channels=True: both phi_full rows are bitwise identical."""
        mimo = self._make_mimo(xp)
        phi = recovery.recover_carrier_phase_viterbi_viterbi(
            mimo, "qam", 16, joint_channels=True, cycle_slip_correction=False
        )
        assert phi.shape == (2, self.N)
        phi_np = phi if xp is np else phi.get()
        np.testing.assert_array_equal(phi_np[0], phi_np[1])

    def test_tikhonov_joint_rows_identical(self, backend_device, xp):
        """Tikhonov joint_channels=True: both phi_full rows are bitwise identical."""
        mimo = self._make_mimo(xp)
        phi = recovery.recover_carrier_phase_tikhonov(
            mimo,
            "qam",
            16,
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
        phi_joint = recovery.recover_carrier_phase_bps(
            mimo, "qam", 16, joint_channels=True, cycle_slip_correction=False
        )
        phi_np = phi_joint if xp is np else phi_joint.get()
        assert float(np.std(phi_np[0] - phi_np[1])) == 0.0

    def test_siso_joint_noop(self, backend_device, xp):
        """joint_channels=True on SISO returns identical result to False."""
        sig = _qam_signal(xp, 16, self.N, seed=7)
        phi_a = recovery.recover_carrier_phase_bps(
            sig.samples, "qam", 16, joint_channels=False, cycle_slip_correction=False
        )
        phi_b = recovery.recover_carrier_phase_bps(
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
        phi_out = recovery.correct_cycle_slips(
            phi_u.copy(), symmetry=4, history_length=50
        )
        np.testing.assert_allclose(phi_out, phi_u, atol=1e-10)

    def test_standalone_single_slip(self, backend_device, xp):
        """A single injected pi/2 slip is corrected back to the original ramp."""
        B = 300
        phi_u = np.linspace(0.0, 1.0, B)
        phi_slipped = phi_u.copy()
        phi_slipped[150:] += np.pi / 2
        phi_out = recovery.correct_cycle_slips(
            phi_slipped, symmetry=4, history_length=100
        )
        np.testing.assert_allclose(phi_out, phi_u, atol=0.05)

    def test_standalone_multiple_slips(self, backend_device, xp):
        """Multiple +/-pi/2 slips are all corrected."""
        B = 500
        phi_u = np.linspace(0.0, 1.5, B)
        phi_slipped = phi_u.copy()
        phi_slipped[100:] += np.pi / 2
        phi_slipped[300:] -= np.pi / 2
        phi_out = recovery.correct_cycle_slips(
            phi_slipped, symmetry=4, history_length=80
        )
        np.testing.assert_allclose(phi_out, phi_u, atol=0.05)

    def test_bps_correction_bounded_output(self, backend_device, xp):
        """BPS cycle_slip_correction=True returns phase within reasonable bounds."""
        sig = _qam_signal(xp, 16, 2048, snr_db=SNR_DB)
        phi = recovery.recover_carrier_phase_bps(
            sig.samples, "qam", 16, cycle_slip_correction=True
        )
        assert phi.shape == sig.samples.shape
        phi_np = phi if xp is np else phi.get()
        assert np.max(np.abs(phi_np)) < 10 * np.pi

    def test_vv_correction_shape(self, backend_device, xp):
        """VV cycle_slip_correction=True returns correct shape."""
        sig = _qam_signal(xp, 16, 2048, snr_db=SNR_DB)
        phi = recovery.recover_carrier_phase_viterbi_viterbi(
            sig.samples, "qam", 16, cycle_slip_correction=True
        )
        assert phi.shape == sig.samples.shape

    def test_tikhonov_correction_shape(self, backend_device, xp):
        """Tikhonov cycle_slip_correction=True returns correct shape."""
        sig = _qam_signal(xp, 16, 2048, snr_db=SNR_DB)
        phi = recovery.recover_carrier_phase_tikhonov(
            sig.samples,
            "qam",
            16,
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
        resolved = recovery.resolve_phase_ambiguity(sym, ref, "qam", 16)
        s0 = float(ser(resolved, ref, "qam", 16))
        for k in range(1, 4):
            sk = float(
                ser(
                    resolved * xp.exp(1j * k * np.pi / 2).astype(sym.dtype),
                    ref,
                    "qam",
                    16,
                )
            )
            assert s0 <= sk + 1e-6

    def test_corrects_pi_half_rotation(self, backend_device, xp):
        """Symbols rotated by pi/2 are corrected; post-resolution SER is low."""
        from commstools.helpers import normalize
        from commstools.metrics import ser

        sig = _qam_signal(xp, 16, self.N, snr_db=30, seed=5)
        sym = normalize(sig.samples, "average_power")
        ref = normalize(xp.asarray(sig.source_symbols), "average_power")
        rotated = sym * xp.exp(1j * np.pi / 2).astype(sym.dtype)
        resolved = recovery.resolve_phase_ambiguity(rotated, ref, "qam", 16)
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
        mimo = xp.stack(
            [
                sym_a * xp.exp(1j * np.pi / 2).astype(sym_a.dtype),
                sym_b * xp.exp(1j * np.pi).astype(sym_b.dtype),
            ],
            axis=0,
        )
        ref_mimo = xp.stack([ref_a, ref_b], axis=0)
        resolved = recovery.resolve_phase_ambiguity(mimo, ref_mimo, "qam", 16)
        assert resolved.shape == (2, self.N)
        s = ser(resolved, ref_mimo, "qam", 16)
        s_np = s if xp is np else s.get()
        assert float(s_np[0]) < 0.05
        assert float(s_np[1]) < 0.05

    def test_signal_method_in_place(self, backend_device, xp):
        """Signal.resolve_phase_ambiguity() updates resolved_symbols in place."""
        from commstools.helpers import normalize
        from commstools.metrics import ser

        sig = qam(order=16, num_symbols=self.N, sps=1, symbol_rate=1e6, seed=9)
        sig.samples = apply_awgn(sig.samples, esn0_db=30, sps=1, seed=9)
        sym = normalize(sig.samples, "average_power")
        sig.resolved_symbols = sym * xp.exp(1j * np.pi / 2).astype(sym.dtype)
        sig = recovery.resolve_phase_ambiguity(sig)
        assert sig.resolved_symbols is not None
        ref = normalize(xp.asarray(sig.source_symbols), "average_power")
        assert float(ser(sig.resolved_symbols, ref, "qam", 16)) < 0.1

    def test_signal_method_raises_without_resolved(self, backend_device, xp):
        """Raises ValueError when resolved_symbols is None."""
        sig = qam(order=16, num_symbols=256, sps=1, symbol_rate=1e6, seed=0)
        with pytest.raises(ValueError, match="resolved_symbols"):
            sig = recovery.resolve_phase_ambiguity(sig)

    def test_signal_method_raises_without_source(self, backend_device, xp):
        """Raises ValueError when source_symbols is None."""
        sig = qam(order=16, num_symbols=256, sps=1, symbol_rate=1e6, seed=0)
        sig.resolved_symbols = sig.samples
        sig.source_symbols = None
        with pytest.raises(ValueError, match="source_symbols"):
            sig = recovery.resolve_phase_ambiguity(sig)


# ─────────────────────────────────────────────────────────────────────────────
# DD-PLL — joint_channels and cycle_slip_correction
# ─────────────────────────────────────────────────────────────────────────────


class TestDDPLLEnhancements:
    """DD-PLL joint_channels and cycle_slip_correction parameters."""

    N = 4096
    PHASE = 0.4  # rad constant phase offset

    def _make_mimo(self, xp, order=16, phase=PHASE, snr_db=SNR_DB):
        sig_a = _qam_signal(xp, order, self.N, snr_db=snr_db, seed=10)
        sig_b = _qam_signal(xp, order, self.N, snr_db=snr_db, seed=11)
        mimo = xp.stack([sig_a.samples, sig_b.samples], axis=0)
        return mimo * xp.exp(1j * phase).astype(mimo.dtype)

    def test_joint_rows_identical(self, backend_device, xp):
        """PI loop + joint_channels=True: both phi_full rows are bitwise identical."""
        mimo = self._make_mimo(xp)
        phi = recovery.recover_carrier_phase_pll(
            mimo, "qam", 16, joint_channels=True, cycle_slip_correction=False
        )
        assert phi.shape == (2, self.N)
        phi_np = phi if xp is np else phi.get()
        np.testing.assert_array_equal(phi_np[0], phi_np[1])

    def test_siso_joint_noop(self, backend_device, xp):
        """joint_channels=True on SISO returns identical result to False."""
        sig = _qam_signal(xp, 16, self.N, seed=12)
        phi_a = recovery.recover_carrier_phase_pll(
            sig.samples, "qam", 16, joint_channels=False, cycle_slip_correction=False
        )
        phi_b = recovery.recover_carrier_phase_pll(
            sig.samples, "qam", 16, joint_channels=True, cycle_slip_correction=False
        )
        phi_a_np = phi_a if xp is np else phi_a.get()
        phi_b_np = phi_b if xp is np else phi_b.get()
        np.testing.assert_allclose(phi_a_np, phi_b_np, atol=1e-10)

    def test_cycle_slip_shape(self, backend_device, xp):
        """cycle_slip_correction=True (PI loop) returns correct shape."""
        sig = _qam_signal(xp, 16, self.N, snr_db=SNR_DB)
        phi = recovery.recover_carrier_phase_pll(
            sig.samples, "qam", 16, cycle_slip_correction=True
        )
        assert phi.shape == sig.samples.shape

    def test_joint_cycle_slip_mimo_rows_identical(self, backend_device, xp):
        """joint_channels=True + cycle_slip_correction=True: rows remain identical."""
        mimo = self._make_mimo(xp)
        phi = recovery.recover_carrier_phase_pll(
            mimo,
            "qam",
            16,
            joint_channels=True,
            cycle_slip_correction=True,
            cycle_slip_history=1000,
        )
        assert phi.shape == (2, self.N)
        phi_np = phi if xp is np else phi.get()
        np.testing.assert_array_equal(phi_np[0], phi_np[1])


# ─────────────────────────────────────────────────────────────────────────────
# Pilots CPR — joint_channels and cycle_slip_correction
# ─────────────────────────────────────────────────────────────────────────────


class TestPilotsCPREnhancements:
    """Pilots CPR joint_channels and cycle_slip_correction parameters."""

    def _pilot_setup(
        self, xp, n_symbols=512, pilot_period=8, phase_per_sym=0.001, seed=42
    ):
        """Return (samples, pilot_indices, pilot_values) for a 16-QAM signal."""
        sig = qam(order=16, num_symbols=n_symbols, sps=1, symbol_rate=FS, seed=seed)
        ideal = xp.asarray(sig.samples.copy())
        sig.samples = apply_awgn(sig.samples, esn0_db=SNR_DB, sps=1, seed=seed)
        sig.samples = _apply_phase_ramp(xp, sig.samples, phase_per_sym)
        pilot_indices = np.arange(0, n_symbols, pilot_period)
        pilot_values = ideal[pilot_indices]
        return sig.samples, pilot_indices, pilot_values

    def test_joint_rows_identical(self, backend_device, xp):
        """joint_channels=True: both phi_full rows are bitwise identical."""
        samples_a, pilot_indices, pilot_values = self._pilot_setup(xp, seed=1)
        samples_b, _, _ = self._pilot_setup(xp, seed=2)
        mimo = xp.stack([samples_a, samples_b], axis=0)  # (2, N)
        phi = recovery.recover_carrier_phase_pilots(
            mimo,
            pilot_indices=pilot_indices,
            pilot_values=pilot_values,
            joint_channels=True,
            cycle_slip_correction=False,
        )
        assert phi.shape == (2, mimo.shape[-1])
        phi_np = phi if xp is np else phi.get()
        np.testing.assert_array_equal(phi_np[0], phi_np[1])

    def test_joint_siso_noop(self, backend_device, xp):
        """joint_channels=True on SISO returns identical result to False."""
        samples, pilot_indices, pilot_values = self._pilot_setup(xp, seed=3)
        phi_a = recovery.recover_carrier_phase_pilots(
            samples,
            pilot_indices=pilot_indices,
            pilot_values=pilot_values,
            joint_channels=False,
            cycle_slip_correction=False,
        )
        phi_b = recovery.recover_carrier_phase_pilots(
            samples,
            pilot_indices=pilot_indices,
            pilot_values=pilot_values,
            joint_channels=True,
            cycle_slip_correction=False,
        )
        phi_a_np = phi_a if xp is np else phi_a.get()
        phi_b_np = phi_b if xp is np else phi_b.get()
        np.testing.assert_allclose(phi_a_np, phi_b_np, atol=1e-10)

    def test_cycle_slip_shape(self, backend_device, xp):
        """cycle_slip_correction=True returns correct shape."""
        samples, pilot_indices, pilot_values = self._pilot_setup(xp)
        phi = recovery.recover_carrier_phase_pilots(
            samples,
            pilot_indices=pilot_indices,
            pilot_values=pilot_values,
            cycle_slip_correction=True,
        )
        assert phi.shape == samples.shape

    def test_cycle_slip_standalone_symmetry_1(self, backend_device, xp):
        """correct_cycle_slips with symmetry=1 corrects an injected 2π slip."""
        B = 200
        phi_u = np.linspace(0.0, 3.0, B)
        phi_slipped = phi_u.copy()
        phi_slipped[100:] += 2.0 * np.pi
        phi_out = recovery.correct_cycle_slips(
            phi_slipped, symmetry=1, history_length=50
        )
        np.testing.assert_allclose(phi_out, phi_u, atol=0.1)

    def test_joint_cycle_slip_mimo_rows_identical(self, backend_device, xp):
        """joint_channels=True + cycle_slip_correction=True: rows remain identical."""
        samples_a, pilot_indices, pilot_values = self._pilot_setup(xp, seed=4)
        samples_b, _, _ = self._pilot_setup(xp, seed=5)
        mimo = xp.stack([samples_a, samples_b], axis=0)
        phi = recovery.recover_carrier_phase_pilots(
            mimo,
            pilot_indices=pilot_indices,
            pilot_values=pilot_values,
            joint_channels=True,
            cycle_slip_correction=True,
        )
        assert phi.shape == (2, mimo.shape[-1])
        phi_np = phi if xp is np else phi.get()
        np.testing.assert_array_equal(phi_np[0], phi_np[1])


# =============================================================================
# CARRIER PHASE RECOVERY — VITERBI-VITERBI (from test_sync.py)
# =============================================================================


class TestViterbiViterbi:
    """Tests for recover_carrier_phase_viterbi_viterbi."""

    def _qpsk_symbols(self, xp, N=512, seed=0):
        import numpy as np

        rng = np.random.default_rng(seed)
        bits = rng.integers(0, 4, N)
        angles = (2 * np.pi / 4) * bits + np.pi / 4
        return xp.asarray(np.exp(1j * angles).astype(np.complex64))

    def _qam16_symbols(self, xp, N=512, seed=1):
        import numpy as np

        from commstools.mapping import gray_constellation

        rng = np.random.default_rng(seed)
        const = gray_constellation("qam", 16)
        idx = rng.integers(0, 16, N)
        return xp.asarray(const[idx].astype(np.complex64))

    def test_siso_qpsk_output_shape(self, backend_device, xp):
        """SISO QPSK: output is (N,) float64."""
        syms = self._qpsk_symbols(xp)
        phi_est = recovery.recover_carrier_phase_viterbi_viterbi(
            syms, "psk", 4, block_size=32
        )
        assert phi_est.shape == syms.shape
        assert phi_est.dtype == xp.float64

    def test_siso_qpsk_recovers_static_phase(self, backend_device, xp):
        """VV tracks applied phase: difference between rotated and unrotated estimate equals phi_true mod π/2."""
        import numpy as np

        phi_true = 0.3  # radians
        syms = self._qpsk_symbols(xp, N=512)
        # Baseline (no rotation)
        phi_base = float(
            xp.mean(
                recovery.recover_carrier_phase_viterbi_viterbi(
                    syms, "psk", 4, block_size=32
                )
            )
        )
        # Rotated by phi_true
        rotated = syms * xp.asarray(np.complex64(np.exp(1j * phi_true)))
        phi_rot = float(
            xp.mean(
                recovery.recover_carrier_phase_viterbi_viterbi(
                    rotated, "psk", 4, block_size=32
                )
            )
        )
        # The shift should equal phi_true modulo π/2
        delta = phi_rot - phi_base
        residual = (delta - phi_true + np.pi / 4) % (np.pi / 2) - np.pi / 4
        assert abs(residual) < 0.15, (
            f"Phase tracking error too large: {residual:.3f} rad"
        )

    def test_siso_qam16_output_shape(self, backend_device, xp):
        """SISO QAM16: output shape matches input."""
        syms = self._qam16_symbols(xp)
        phi_est = recovery.recover_carrier_phase_viterbi_viterbi(
            syms, "qam", 16, block_size=32
        )
        assert phi_est.shape == syms.shape

    def test_mimo_output_shape(self, backend_device, xp):
        """MIMO input (C, N): output shape is (C, N)."""
        import numpy as np

        C, N = 2, 256
        syms = xp.asarray(
            np.random.default_rng(5).standard_normal((C, N)).astype(np.float32)
            + 1j * np.random.default_rng(6).standard_normal((C, N)).astype(np.float32)
        )
        phi_est = recovery.recover_carrier_phase_viterbi_viterbi(
            syms, "qam", 16, block_size=32
        )
        assert phi_est.shape == (C, N)

    def test_block_size_too_large_raises(self, backend_device, xp):
        """block_size > N should raise ValueError."""
        syms = self._qpsk_symbols(xp, N=16)
        with pytest.raises(ValueError, match="block_size"):
            recovery.recover_carrier_phase_viterbi_viterbi(
                syms, "psk", 4, block_size=64
            )


# =============================================================================
# CARRIER PHASE RECOVERY — BLIND PHASE SEARCH (from test_sync.py)
# =============================================================================


class TestBPS:
    """Tests for recover_carrier_phase_bps."""

    def _qam16_symbols(self, xp, N=512, seed=2):
        import numpy as np

        from commstools.mapping import gray_constellation

        rng = np.random.default_rng(seed)
        const = gray_constellation("qam", 16)
        idx = rng.integers(0, 16, N)
        return xp.asarray(const[idx].astype(np.complex64))

    def _qpsk_symbols(self, xp, N=512, seed=3):
        import numpy as np

        from commstools.mapping import gray_constellation

        rng = np.random.default_rng(seed)
        const = gray_constellation("qpsk", 4)
        idx = rng.integers(0, 4, N)
        return xp.asarray(const[idx].astype(np.complex64))

    def test_siso_qam16_output_shape(self, backend_device, xp):
        """SISO QAM16 (square QAM fast path): output is (N,) float64."""
        syms = self._qam16_symbols(xp)
        phi_est = recovery.recover_carrier_phase_bps(
            syms, "qam", 16, num_test_phases=32, block_size=32
        )
        assert phi_est.shape == syms.shape
        assert phi_est.dtype == xp.float64

    def test_siso_qam16_recovers_static_phase(self, backend_device, xp):
        """BPS should estimate a static QAM16 phase offset to within π/8 tolerance."""
        import numpy as np

        phi_true = 0.25
        syms = self._qam16_symbols(xp, N=512)
        rotated = syms * xp.asarray(np.complex64(np.exp(1j * phi_true)))
        phi_est = recovery.recover_carrier_phase_bps(
            rotated, "qam", 16, num_test_phases=64, block_size=32
        )
        phi_mean = float(xp.mean(phi_est))
        # 4-fold ambiguity: allow ±π/8 residual
        residual = (phi_mean - phi_true + np.pi / 4) % (np.pi / 2) - np.pi / 4
        assert abs(residual) < 0.15, (
            f"Residual phase error too large: {residual:.3f} rad"
        )

    def test_siso_qpsk_general_path(self, backend_device, xp):
        """SISO QPSK (non-square: triggers general distance path): output shape correct."""
        syms = self._qpsk_symbols(xp, N=256)
        phi_est = recovery.recover_carrier_phase_bps(
            syms, "psk", 4, num_test_phases=16, block_size=32
        )
        assert phi_est.shape == syms.shape

    def test_mimo_output_shape(self, backend_device, xp):
        """MIMO input (C, N): output shape is (C, N)."""
        import numpy as np

        C, N = 2, 256
        rng = np.random.default_rng(7)
        syms = xp.asarray(
            (rng.standard_normal((C, N)) + 1j * rng.standard_normal((C, N))).astype(
                np.complex64
            )
        )
        phi_est = recovery.recover_carrier_phase_bps(
            syms, "qam", 16, num_test_phases=16, block_size=32
        )
        assert phi_est.shape == (C, N)

    def test_block_size_too_large_raises(self, backend_device, xp):
        """block_size > N should raise ValueError."""
        syms = self._qam16_symbols(xp, N=16)
        with pytest.raises(ValueError, match="block_size"):
            recovery.recover_carrier_phase_bps(syms, "qam", 16, block_size=64)


# =============================================================================
# CARRIER PHASE RECOVERY — DECISION-DIRECTED PLL (from test_sync.py)
# =============================================================================


class TestDDPLL:
    """Tests for recover_carrier_phase_pll."""

    def _qpsk_symbols(self, xp, N=512, seed=10):
        import numpy as np

        from commstools.mapping import gray_constellation

        rng = np.random.default_rng(seed)
        const = gray_constellation("qpsk", 4)
        return xp.asarray(const[rng.integers(0, 4, N)].astype(np.complex64))

    def test_siso_output_shape(self, backend_device, xp):
        """SISO: output is (N,) float64."""
        syms = self._qpsk_symbols(xp)
        phi = recovery.recover_carrier_phase_pll(syms, "psk", 4)
        assert phi.shape == syms.shape
        assert phi.dtype == xp.float64

    def test_mimo_output_shape(self, backend_device, xp):
        """MIMO (C, N): output shape is (C, N)."""
        import numpy as np

        C, N = 2, 256
        rng = np.random.default_rng(11)
        from commstools.mapping import gray_constellation

        const = gray_constellation("qpsk", 4)
        syms = xp.asarray(
            const[rng.integers(0, 4, C * N)].reshape(C, N).astype(np.complex64)
        )
        phi = recovery.recover_carrier_phase_pll(syms, "psk", 4)
        assert phi.shape == (C, N)

    def test_second_order_loop(self, backend_device, xp):
        """beta > 0 engages 2nd-order loop without raising."""
        syms = self._qpsk_symbols(xp, N=256)
        phi = recovery.recover_carrier_phase_pll(syms, "psk", 4, mu=0.02, beta=1e-4)
        assert phi.shape == syms.shape

    def test_phase_init_applied(self, backend_device, xp):
        """phase_init shifts the starting phase estimate."""
        syms = self._qpsk_symbols(xp, N=256)
        phi_init = 0.5
        phi = recovery.recover_carrier_phase_pll(syms, "psk", 4, phase_init=phi_init)
        # First estimate should be close to phase_init before any loop correction
        assert abs(float(phi[0]) - phi_init) < 0.5

    def test_bandwidth_shortcut_shape(self, backend_device, xp):
        """mu=None opts into the loop_bandwidth_normalized shortcut; shape/dtype hold."""
        syms = self._qpsk_symbols(xp, N=512)
        phi = recovery.recover_carrier_phase_pll(
            syms, "psk", 4, mu=None, loop_bandwidth_normalized=1e-3
        )
        assert phi.shape == syms.shape
        assert phi.dtype == xp.float64

    def test_bandwidth_shortcut_equals_raw_gains(self, backend_device, xp):
        """mu=None, bandwidth=B is identical to raw mu=4B, beta=4B² (the resolver mapping)."""
        import numpy as np

        from commstools.mapping import gray_constellation

        N = 512
        const = gray_constellation("qpsk", 4)
        rng = np.random.default_rng(7)
        syms = xp.asarray(const[rng.integers(0, 4, N)].astype(np.complex64))
        syms = syms * np.exp(1j * 0.2)

        B = 1e-3
        phi_bw = recovery.recover_carrier_phase_pll(
            syms, "psk", 4, mu=None, loop_bandwidth_normalized=B
        )
        phi_raw = recovery.recover_carrier_phase_pll(
            syms, "psk", 4, mu=4.0 * B, beta=4.0 * B**2
        )
        phi_bw_np = phi_bw if xp is np else phi_bw.get()
        phi_raw_np = phi_raw if xp is np else phi_raw.get()
        np.testing.assert_allclose(phi_bw_np, phi_raw_np, rtol=1e-6, atol=1e-9)

    def test_first_vs_second_order_under_frequency_offset(self, backend_device, xp):
        """Under a frequency offset, a 1st-order loop (beta=0) settles to a constant
        phase lag; a 2nd-order loop (beta>0) nulls it to ~zero steady-state error."""
        import numpy as np

        from commstools.mapping import gray_constellation

        N = 4000
        const = gray_constellation("qpsk", 4)
        rng = np.random.default_rng(5)
        clean = const[rng.integers(0, 4, N)].astype(np.complex64)
        # Small constant frequency offset (rad/sample) → phase ramp the loop tracks.
        df = 1e-3
        ramp = np.exp(1j * df * np.arange(N)).astype(np.complex64)
        syms = xp.asarray(clean * ramp)

        true_phase = df * np.arange(N)
        tail = slice(N // 2, N)

        phi1 = recovery.recover_carrier_phase_pll(syms, "psk", 4, mu=0.02, beta=0.0)
        phi2 = recovery.recover_carrier_phase_pll(syms, "psk", 4, mu=0.02, beta=2e-4)
        phi1_np = phi1 if xp is np else phi1.get()
        phi2_np = phi2 if xp is np else phi2.get()

        # The steady-state error is a constant lag (its std ≈ 0), so measure the mean.
        lag1 = abs(float(np.mean(np.unwrap(phi1_np[tail]) - true_phase[tail])))
        lag2 = abs(float(np.mean(np.unwrap(phi2_np[tail]) - true_phase[tail])))
        assert lag1 > 1e-3, (
            "1st-order loop should exhibit a finite lag under freq offset"
        )
        assert lag2 < lag1 / 100.0, (
            "2nd-order loop should null the frequency-induced lag"
        )

    def test_bandwidth_shortcut_invalid_bandwidth_raises(self, backend_device, xp):
        """On the bandwidth path (mu=None), bandwidth outside (0, 0.5) raises ValueError."""
        syms = self._qpsk_symbols(xp, N=64)
        with pytest.raises(ValueError, match="loop_bandwidth_normalized"):
            recovery.recover_carrier_phase_pll(
                syms, "psk", 4, mu=None, loop_bandwidth_normalized=0.6
            )

    def test_beta_without_mu_raises(self, backend_device, xp):
        """Passing beta with mu=None is ambiguous and must raise ValueError."""
        syms = self._qpsk_symbols(xp, N=64)
        with pytest.raises(ValueError, match="beta requires mu"):
            recovery.recover_carrier_phase_pll(syms, "psk", 4, mu=None, beta=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# resolve_phase_ambiguity — num_skip_symbols (from test_sync.py)
# ─────────────────────────────────────────────────────────────────────────────


def _make_ambiguous_qam16(n_sym=2000, corrupt_head=500, seed=0):
    """Return (symbols, ref) where the first corrupt_head symbols are rotated by π/2."""
    rng = np.random.default_rng(seed)
    const = gray_constellation("qam", 16).astype(np.complex64)
    const /= np.sqrt(np.mean(np.abs(const) ** 2))
    ref = const[rng.integers(0, 16, n_sym)]
    # True ambiguity k=1: rotate entire stream by π/2
    rot1 = np.exp(1j * np.pi / 2).astype(np.complex64)
    symbols = ref * rot1
    # Corrupt only the first corrupt_head symbols with an additional π/2 (total π)
    symbols[:corrupt_head] = ref[:corrupt_head] * np.exp(1j * np.pi).astype(
        np.complex64
    )
    return symbols, ref


def test_resolve_phase_ambiguity_skip(backend_device, xp, xpt):
    """num_skip_symbols bypasses the corrupt head and picks the correct rotation."""
    n_sym, corrupt_head = 2000, 500
    symbols_np, ref_np = _make_ambiguous_qam16(n_sym=n_sym, corrupt_head=corrupt_head)
    symbols, ref = xp.asarray(symbols_np), xp.asarray(ref_np)

    out_no_skip = recovery.resolve_phase_ambiguity(
        symbols, ref, "qam", 16, num_skip_symbols=0
    )
    out_skip = recovery.resolve_phase_ambiguity(
        symbols, ref, "qam", 16, num_skip_symbols=corrupt_head
    )

    from commstools.metrics import ser as _ser_fn

    def _ser(y, r):
        return float(xp.mean(xp.asarray(_ser_fn(y, r, "qam", 16))))

    ser_skip_tail = _ser(out_skip[corrupt_head:], ref[corrupt_head:])
    ser_no_skip_tail = _ser(out_no_skip[corrupt_head:], ref[corrupt_head:])
    assert ser_skip_tail <= ser_no_skip_tail, (
        f"Skip should improve tail SER: {ser_skip_tail:.4f} vs {ser_no_skip_tail:.4f}"
    )


def test_resolve_phase_ambiguity_skip_zero_is_baseline(backend_device, xp, xpt):
    """num_skip_symbols=0 must produce identical output to the default call."""
    symbols_np, ref_np = _make_ambiguous_qam16(n_sym=1000, corrupt_head=0)
    symbols, ref = xp.asarray(symbols_np), xp.asarray(ref_np)

    out_default = recovery.resolve_phase_ambiguity(symbols, ref, "qam", 16)
    out_skip0 = recovery.resolve_phase_ambiguity(
        symbols, ref, "qam", 16, num_skip_symbols=0
    )

    assert bool(xp.all(out_default == out_skip0))


def test_resolve_phase_ambiguity_skip_ge_n_raises(backend_device, xp):
    """num_skip_symbols >= N must raise ValueError."""
    symbols_np, ref_np = _make_ambiguous_qam16(n_sym=100, corrupt_head=0)
    symbols, ref = xp.asarray(symbols_np), xp.asarray(ref_np)

    with pytest.raises(ValueError, match="num_skip_symbols"):
        recovery.resolve_phase_ambiguity(symbols, ref, "qam", 16, num_skip_symbols=100)

    with pytest.raises(ValueError, match="num_skip_symbols"):
        recovery.resolve_phase_ambiguity(symbols, ref, "qam", 16, num_skip_symbols=200)


# ─────────────────────────────────────────────────────────────────────────────
# correct_phase_rotation
# ─────────────────────────────────────────────────────────────────────────────


def _clean_qam16(xp, n, seed=0):
    """Noiseless unit-power 16-QAM symbols."""
    rng = np.random.default_rng(seed)
    const = gray_constellation("qam", 16).astype(np.complex64)
    const /= np.sqrt(np.mean(np.abs(const) ** 2))
    return xp.asarray(const[rng.integers(0, 16, n)])


class TestCorrectPhaseRotation:
    """correct_phase_rotation corrects arbitrary constant per-channel rotation."""

    N = 2048

    def test_arbitrary_rotation_corrected_siso(self, backend_device, xp, xpt):
        """Arbitrary non-grid rotation is removed; residual angle is near zero."""
        ref = _clean_qam16(xp, self.N, seed=0)
        theta_true = 0.7  # ~40°, not a π/2 multiple
        rotated = ref * xp.array(np.exp(1j * theta_true), dtype=ref.dtype)
        out = recovery.correct_phase_rotation(rotated, ref)
        residual = float(xp.abs(xp.angle(xp.mean(out * xp.conj(ref)))))
        assert residual < 0.02

    def test_short_ref_applies_to_full_sequence(self, backend_device, xp):
        """Estimation from first N_pre symbols; correction spans the full N sequence."""
        N, N_pre = self.N, 256
        ref_full = _clean_qam16(xp, N, seed=1)
        rotated = ref_full * xp.array(np.exp(1j * 1.2), dtype=ref_full.dtype)
        out = recovery.correct_phase_rotation(rotated, ref_full[:N_pre])
        assert out.shape == rotated.shape
        residual = float(xp.abs(xp.angle(xp.mean(out * xp.conj(ref_full)))))
        assert residual < 0.02

    def test_mimo_independent_channels(self, backend_device, xp):
        """Each MIMO channel gets its own rotation corrected independently."""
        ref_a = _clean_qam16(xp, self.N, seed=2)
        ref_b = _clean_qam16(xp, self.N, seed=3)
        ref = xp.stack([ref_a, ref_b])
        rotated = xp.stack(
            [
                ref_a * xp.array(np.exp(1j * 0.4), dtype=ref_a.dtype),
                ref_b * xp.array(np.exp(1j * -1.1), dtype=ref_b.dtype),
            ]
        )
        out = recovery.correct_phase_rotation(rotated, ref)
        assert out.shape == (2, self.N)
        for ch in range(2):
            residual = float(xp.abs(xp.angle(xp.mean(out[ch] * xp.conj(ref[ch])))))
            assert residual < 0.02

    def test_num_skip_symbols_excludes_transient(self, backend_device, xp):
        """Corrupted head is excluded; tail correction uses the clean portion only."""
        N, skip = self.N, 200
        ref = _clean_qam16(xp, N, seed=4)
        rotated = ref * xp.array(np.exp(1j * 0.9), dtype=ref.dtype)
        corrupted = xp.array(rotated)
        corrupted[:skip] = ref[:skip] * xp.array(np.exp(1j * 2.5), dtype=ref.dtype)
        out = recovery.correct_phase_rotation(corrupted, ref, num_skip_symbols=skip)
        residual = float(xp.abs(xp.angle(xp.mean(out[skip:] * xp.conj(ref[skip:])))))
        assert residual < 0.02

    def test_num_skip_ge_nref_raises(self, backend_device, xp):
        """num_skip_symbols >= N_ref must raise ValueError."""
        ref = _clean_qam16(xp, 100, seed=0)
        symbols = _clean_qam16(xp, 500, seed=1)
        with pytest.raises(ValueError, match="num_skip_symbols"):
            recovery.correct_phase_rotation(symbols, ref, num_skip_symbols=100)
        with pytest.raises(ValueError, match="num_skip_symbols"):
            recovery.correct_phase_rotation(symbols, ref, num_skip_symbols=200)

    def test_dtype_preserved(self, backend_device, xp):
        """complex64 input → complex64 output."""
        ref = _clean_qam16(xp, 256, seed=0)
        out = recovery.correct_phase_rotation(
            ref * xp.array(np.exp(1j * 0.5), dtype=ref.dtype), ref
        )
        assert out.dtype == ref.dtype

    def test_1d_input_returns_1d(self, backend_device, xp):
        """1-D input returns 1-D output."""
        ref = _clean_qam16(xp, 256, seed=0)
        out = recovery.correct_phase_rotation(
            ref * xp.array(np.exp(1j * 0.3), dtype=ref.dtype), ref
        )
        assert out.ndim == 1
