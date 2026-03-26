"""Tests for channel impairment models."""

import numpy as np
import pytest

from commstools.impairments import (
    apply_awgn,
    apply_chromatic_dispersion,
    apply_iq_imbalance,
    apply_phase_noise,
    apply_pmd,
    apply_polarization_mixing,
)


class TestApplyPMD:
    """Tests for the apply_pmd function."""

    def test_identity_no_dgd_no_rotation(self, backend_device, xp, xpt):
        """PMD with dgd=0, theta=0 should return the input unchanged."""
        N = 256
        rng = xp.random.RandomState(42)
        samples = (rng.randn(2, N) + 1j * rng.randn(2, N)).astype(xp.complex64)
        fs = 56e9

        out = apply_pmd(samples, dgd=0.0, theta=0.0, sampling_rate=fs)

        xpt.assert_allclose(out, samples, atol=1e-5)

    def test_energy_conservation(self, backend_device, xp, xpt):
        """PMD is unitary — output power should equal input power."""
        N = 1024
        rng = xp.random.RandomState(7)
        samples = (rng.randn(2, N) + 1j * rng.randn(2, N)).astype(xp.complex64)
        fs = 56e9

        out = apply_pmd(samples, dgd=5e-12, theta=np.pi / 5, sampling_rate=fs)

        power_in = float(xp.sum(xp.abs(samples) ** 2))
        power_out = float(xp.sum(xp.abs(out) ** 2))

        xpt.assert_allclose(power_out, power_in, rtol=1e-4)

    def test_pure_rotation(self, backend_device, xp, xpt):
        """With dgd=0, H(f)=R(+θ)·I·R(-θ)=I — output equals input regardless of theta."""
        N = 128
        theta = np.pi / 4
        fs = 56e9

        # Constant signal: X-pol = 1+0j, Y-pol = 0+0j
        samples = xp.zeros((2, N), dtype=xp.complex64)
        samples[0, :] = 1.0 + 0j

        out = apply_pmd(samples, dgd=0.0, theta=theta, sampling_rate=fs)

        # With correct Jones matrix H=R(+θ)·D(f)·R(-θ), DGD=0 ⟹ D(f)=I ⟹ H=I.
        # Output must equal input for any theta.
        xpt.assert_allclose(out, samples, atol=1e-5)

    def test_pure_rotation_with_dgd_zero_coupling(self, backend_device, xp, xpt):
        """With theta=0, DGD applies a phase shift but no polarization coupling."""
        N = 256
        rng = xp.random.RandomState(99)
        samples = (rng.randn(2, N) + 1j * rng.randn(2, N)).astype(xp.complex64)
        fs = 56e9
        dgd = 5e-12

        out = apply_pmd(samples, dgd=dgd, theta=0.0, sampling_rate=fs)

        # With theta=0, R=I so H=diag(D0, D1) — only phase shift per pol
        # Output power should still be preserved
        power_in = float(xp.sum(xp.abs(samples) ** 2))
        power_out = float(xp.sum(xp.abs(out) ** 2))

        xpt.assert_allclose(power_out, power_in, rtol=1e-4)

    def test_output_shape(self, backend_device, xp):
        """Output shape should match input shape."""
        samples = xp.ones((2, 512), dtype=xp.complex64)
        out = apply_pmd(samples, dgd=1e-12, theta=0.3, sampling_rate=56e9)

        assert out.shape == (2, 512)

    def test_output_dtype(self, backend_device, xp):
        """Output dtype should match input dtype."""
        samples = xp.ones((2, 128), dtype=xp.complex64)
        out = apply_pmd(samples, dgd=1e-12, theta=0.3, sampling_rate=56e9)

        assert out.dtype == xp.complex64

    def test_rejects_siso(self, backend_device, xp):
        """Should raise ValueError for 1D (SISO) input."""
        samples = xp.ones(100, dtype=xp.complex64)
        with pytest.raises(ValueError, match="dual-pol"):
            apply_pmd(samples, dgd=1e-12, sampling_rate=56e9)

    def test_rejects_wrong_channels(self, backend_device, xp):
        """Should raise ValueError for non-2 first dimension."""
        samples = xp.ones((3, 100), dtype=xp.complex64)
        with pytest.raises(ValueError, match="dual-pol"):
            apply_pmd(samples, dgd=1e-12, sampling_rate=56e9)

    def test_nonzero_dgd_causes_distortion(self, backend_device, xp):
        """Non-zero DGD with non-zero theta should change the signal."""
        N = 512
        rng = xp.random.RandomState(42)
        samples = (rng.randn(2, N) + 1j * rng.randn(2, N)).astype(xp.complex64)
        fs = 56e9

        out = apply_pmd(samples, dgd=10e-12, theta=np.pi / 4, sampling_rate=fs)

        # Signal should be different from input
        diff = float(xp.max(xp.abs(out - samples)))
        assert diff > 0.01, "PMD with DGD should modify the signal"

    def test_round_trip_nonzero_theta(self, backend_device, xp, xpt):
        """Apply PMD then invert it: applying with -theta and negated DGD phase reverses CD.

        Simpler correctness check: apply PMD forward then backward (theta, -theta with
        complex conjugate of H) to verify the Jones matrix fix gives a round-trip near identity.
        """
        N = 512
        rng = xp.random.RandomState(13)
        samples = (rng.randn(2, N) + 1j * rng.randn(2, N)).astype(xp.complex64)
        fs = 56e9
        dgd = 5e-12
        theta = np.pi / 6

        # Forward: apply PMD (adds DGD at angle theta)
        distorted = apply_pmd(samples, dgd=dgd, theta=theta, sampling_rate=fs)

        # Inverse: apply PMD with same theta but negated DGD (conjugate filter)
        # H_inv(f) = H*(f)  →  apply_pmd with dgd=-dgd
        recovered = apply_pmd(distorted, dgd=-dgd, theta=theta, sampling_rate=fs)

        xpt.assert_allclose(recovered, samples, atol=1e-4)


class TestAddAWGN:
    """Tests for apply_awgn."""

    def test_awgn_adds_noise(self, backend_device, xp):
        """Output should differ from input."""
        samples = xp.ones(1000, dtype=xp.complex64)
        noisy = apply_awgn(samples, esn0_db=10, sps=1)

        diff = float(xp.max(xp.abs(noisy - samples)))
        assert diff > 0.001

    def test_awgn_preserves_shape(self, backend_device, xp):
        """Output shape should match input."""
        samples = xp.ones((2, 500), dtype=xp.complex64)
        noisy = apply_awgn(samples, esn0_db=20, sps=2)

        assert noisy.shape == samples.shape

    def test_awgn_noise_power(self, backend_device, xp):
        """Noise power should match 10^(-SNR/10) for a unit-power signal."""
        data = xp.ones(1000, dtype=complex)
        noisy = apply_awgn(data, esn0_db=10.0, sps=1)
        noise = noisy - data
        measured = float(xp.mean(xp.abs(noise) ** 2))
        # Signal power = 1, SNR = 10 dB → noise power = 10^(-1) = 0.1
        assert 0.08 < measured < 0.12, f"Noise power {measured:.4f} outside expected range"

    def test_awgn_real_data(self, backend_device, xp):
        """apply_awgn should handle real-valued input and return a real output."""
        data = xp.ones(1000, dtype="float32")
        noisy = apply_awgn(data, esn0_db=10, sps=1)

        assert xp.isrealobj(noisy)
        assert not xp.allclose(noisy, data)

    def test_awgn_low_snr(self, backend_device, xp):
        """Extremely low SNR should produce very high noise power."""
        data = xp.ones(100, dtype=complex)
        noisy = apply_awgn(data, esn0_db=-300, sps=1)
        measured_power = float(xp.mean(xp.abs(noisy) ** 2))
        assert measured_power > 1e15


class TestApplyIQImbalance:
    """Tests for apply_iq_imbalance."""

    def test_identity_zero_imbalance(self, backend_device, xp, xpt):
        """Zero imbalance (0 dB, 0 deg) should leave the signal unchanged."""
        N = 1024
        rng = xp.random.RandomState(42)
        samples = (rng.randn(N) + 1j * rng.randn(N)).astype(xp.complex64)

        out = apply_iq_imbalance(samples, amplitude_imbalance_db=0.0, phase_imbalance_deg=0.0)

        xpt.assert_allclose(out, samples, atol=1e-5)

    def test_causes_impropriety(self, backend_device, xp):
        """Imbalance should make a circular signal improper (κ > 0)."""
        N = 4096
        rng = xp.random.RandomState(7)
        samples = (rng.randn(N) + 1j * rng.randn(N)).astype(xp.complex64)

        # κ = |E[r²]| / E[|r|²]: zero for a circular signal
        def kappa(x):
            return float(xp.abs(xp.mean(x ** 2))) / float(xp.mean(xp.abs(x) ** 2))

        kappa_before = kappa(samples)
        out = apply_iq_imbalance(samples, amplitude_imbalance_db=2.0, phase_imbalance_deg=5.0)
        kappa_after = kappa(out)

        assert kappa_after > kappa_before + 0.05

    def test_output_shape_siso(self, backend_device, xp):
        """Output shape should match SISO input."""
        samples = xp.ones(512, dtype=xp.complex64)
        out = apply_iq_imbalance(samples, amplitude_imbalance_db=1.0, phase_imbalance_deg=3.0)
        assert out.shape == (512,)

    def test_output_shape_mimo(self, backend_device, xp):
        """Output shape should match MIMO input."""
        samples = xp.ones((4, 512), dtype=xp.complex64)
        out = apply_iq_imbalance(samples, amplitude_imbalance_db=1.0, phase_imbalance_deg=3.0)
        assert out.shape == (4, 512)

    def test_output_dtype_preserved(self, backend_device, xp):
        """Output dtype should match input dtype."""
        samples = xp.ones(256, dtype=xp.complex64)
        out = apply_iq_imbalance(samples, amplitude_imbalance_db=1.0, phase_imbalance_deg=2.0)
        assert out.dtype == xp.complex64


class TestApplyPhaseNoise:
    """Tests for apply_phase_noise."""

    def test_siso_output_shape(self, backend_device, xp):
        """SISO: output shape matches input."""
        samples = xp.ones(1024, dtype=xp.complex64)
        out = apply_phase_noise(samples, linewidth_hz=100e3, sampling_rate=64e9, seed=1)
        assert out.shape == (1024,)

    def test_mimo_output_shape(self, backend_device, xp):
        """MIMO (C, N): output shape matches input."""
        samples = xp.ones((4, 512), dtype=xp.complex64)
        out = apply_phase_noise(samples, linewidth_hz=100e3, sampling_rate=64e9, seed=2)
        assert out.shape == (4, 512)

    def test_modifies_signal(self, backend_device, xp):
        """Phase noise should change the signal."""
        samples = xp.ones(1024, dtype=xp.complex64)
        out = apply_phase_noise(samples, linewidth_hz=1e6, sampling_rate=64e9, seed=3)
        diff = float(xp.max(xp.abs(out - samples)))
        assert diff > 1e-4

    def test_preserves_amplitude(self, backend_device, xp):
        """Phase rotation must not change sample amplitude."""
        rng = xp.random.RandomState(42)
        samples = (rng.randn(2048) + 1j * rng.randn(2048)).astype(xp.complex64)
        out = apply_phase_noise(samples, linewidth_hz=100e3, sampling_rate=64e9, seed=4)
        amp_in = xp.abs(samples)
        amp_out = xp.abs(out)
        assert float(xp.max(xp.abs(amp_out - amp_in))) < 1e-4

    def test_phase_variance_matches_linewidth(self, backend_device, xp):
        """Incremental phase variance per sample should equal 2π·Δν/fs.

        Extracts phase increments from a long single-channel trajectory and
        measures their variance.  A flat-amplitude input of ones means
        angle(out[n]) = cumulative phase, so diff(angle(out)) = increments.
        """
        import math
        N = 200000
        linewidth = 100e3
        fs = 64e9
        expected_variance = 2.0 * math.pi * linewidth / fs

        # Unit-amplitude input so abs(out)=1 and angle(out) = cumulative phase
        samples = xp.ones(N, dtype=xp.complex128)
        out = apply_phase_noise(samples, linewidth_hz=linewidth, sampling_rate=fs, seed=42)

        cumphase = xp.angle(out)  # (N,) wrapped, but increments are tiny
        increments = xp.diff(cumphase)  # (N-1,)

        # Small increments so wrapping is not an issue
        measured_var = float(xp.var(increments))
        assert abs(measured_var - expected_variance) / expected_variance < 0.05

    def test_shared_lo_all_channels_equal(self, backend_device, xp):
        """shared_lo=True: all channels receive identical phase trajectory."""
        C, N = 4, 512
        samples = xp.ones((C, N), dtype=xp.complex128)
        out = apply_phase_noise(samples, linewidth_hz=1e6, sampling_rate=64e9,
                                seed=7, shared_lo=True)
        # All channels should have identical output since same phase is applied
        for c in range(1, C):
            assert float(xp.max(xp.abs(out[c] - out[0]))) < 1e-10

    def test_independent_lo_channels_differ(self, backend_device, xp):
        """shared_lo=False (default): channels have independent phase trajectories."""
        C, N = 2, 512
        samples = xp.ones((C, N), dtype=xp.complex128)
        out = apply_phase_noise(samples, linewidth_hz=10e6, sampling_rate=64e9,
                                seed=5, shared_lo=False)
        # Independent trajectories should differ
        diff = float(xp.max(xp.abs(out[0] - out[1])))
        assert diff > 1e-4


class TestApplyPolarizationMixing:
    """Tests for apply_polarization_mixing."""

    def test_static_identity(self, backend_device, xp, xpt):
        """theta=0 should return the input unchanged."""
        N = 256
        rng = xp.random.RandomState(1)
        samples = (rng.randn(2, N) + 1j * rng.randn(2, N)).astype(xp.complex64)
        out = apply_polarization_mixing(samples, theta=0.0)
        xpt.assert_allclose(out, samples, atol=1e-5)

    def test_static_90deg_rotation(self, backend_device, xp, xpt):
        """θ=π/2 swaps X↔Y with a sign flip: Ex' = -Ey, Ey' = Ex."""
        N = 128
        samples = xp.zeros((2, N), dtype=xp.complex64)
        samples[0, :] = 1.0 + 0j  # only X-pol
        out = apply_polarization_mixing(samples, theta=np.pi / 2)
        # cos(π/2)=0, sin(π/2)=1 → Ex'=0-1*Ey=0, Ey'=1*Ex+0*Ey=1
        xpt.assert_allclose(out[0], xp.zeros(N, dtype=xp.complex64), atol=1e-5)
        xpt.assert_allclose(out[1], xp.ones(N, dtype=xp.complex64), atol=1e-5)

    def test_static_energy_preserved(self, backend_device, xp, xpt):
        """Static rotation is unitary — total power preserved."""
        N = 512
        rng = xp.random.RandomState(7)
        samples = (rng.randn(2, N) + 1j * rng.randn(2, N)).astype(xp.complex64)
        out = apply_polarization_mixing(samples, theta=np.pi / 3)
        power_in = float(xp.sum(xp.abs(samples) ** 2))
        power_out = float(xp.sum(xp.abs(out) ** 2))
        xpt.assert_allclose(power_out, power_in, rtol=1e-4)

    def test_variable_theta_energy_preserved(self, backend_device, xp, xpt):
        """Time-varying rotation must preserve energy at every sample."""
        N = 256
        rng = xp.random.RandomState(3)
        samples = (rng.randn(2, N) + 1j * rng.randn(2, N)).astype(xp.complex64)
        thetas = xp.linspace(0, 2 * np.pi, N)
        out = apply_polarization_mixing(samples, theta=thetas)
        # Per-sample energy is conserved: |Ex'[n]|² + |Ey'[n]|² = |Ex[n]|² + |Ey[n]|²
        power_in_per_sample = xp.abs(samples[0]) ** 2 + xp.abs(samples[1]) ** 2
        power_out_per_sample = xp.abs(out[0]) ** 2 + xp.abs(out[1]) ** 2
        xpt.assert_allclose(power_out_per_sample, power_in_per_sample, rtol=1e-4)

    def test_drift_rate_produces_varying_output(self, backend_device, xp):
        """Non-zero drift_rate should produce time-varying output."""
        N = 256
        samples = xp.zeros((2, N), dtype=xp.complex64)
        samples[0, :] = 1.0 + 0j
        out = apply_polarization_mixing(samples, theta=0.0, drift_rate_rad_per_sym=1e-2)
        # First sample: θ=0 → Ex'=1, Ey'=0; later samples should differ
        assert float(xp.abs(out[0, -1] - out[0, 0])) > 1e-3

    def test_rejects_siso(self, backend_device, xp):
        """Should raise ValueError for 1D input."""
        with pytest.raises(ValueError, match="dual-pol"):
            apply_polarization_mixing(xp.ones(100, dtype=xp.complex64), theta=0.1)

    def test_rejects_wrong_channels(self, backend_device, xp):
        """Should raise ValueError when first axis != 2."""
        with pytest.raises(ValueError, match="dual-pol"):
            apply_polarization_mixing(xp.ones((3, 100), dtype=xp.complex64), theta=0.1)

    def test_output_shape(self, backend_device, xp):
        """Output shape should match input shape."""
        samples = xp.ones((2, 512), dtype=xp.complex64)
        out = apply_polarization_mixing(samples, theta=np.pi / 4)
        assert out.shape == (2, 512)


class TestApplyChomaticDispersion:
    """Tests for apply_chromatic_dispersion and round-trip with compensate_chromatic_dispersion."""

    def test_round_trip_siso(self, backend_device, xp, xpt):
        """Apply CD then compensate: output should be ~equal to input."""
        from commstools.filtering import compensate_chromatic_dispersion

        N = 1024
        rng = xp.random.RandomState(42)
        samples = (rng.randn(N) + 1j * rng.randn(N)).astype(xp.complex64)
        fs = 64e9
        D, L, lam = 17.0, 80.0, 1550.0

        distorted = apply_chromatic_dispersion(samples, fs, D, L, lam)
        recovered = compensate_chromatic_dispersion(distorted, fs, D, L, lam)

        xpt.assert_allclose(recovered, samples, atol=1e-3)

    def test_round_trip_mimo(self, backend_device, xp, xpt):
        """Round-trip (MIMO): each channel should recover to input."""
        from commstools.filtering import compensate_chromatic_dispersion

        C, N = 2, 512
        rng = xp.random.RandomState(7)
        samples = (rng.randn(C, N) + 1j * rng.randn(C, N)).astype(xp.complex64)
        fs = 64e9
        D, L, lam = 17.0, 80.0, 1550.0

        distorted = apply_chromatic_dispersion(samples, D, L, lam, fs)
        recovered = compensate_chromatic_dispersion(distorted, D, L, lam, fs)

        xpt.assert_allclose(recovered, samples, atol=1e-3)

    def test_output_shape_siso(self, backend_device, xp):
        """SISO output shape matches input."""
        samples = xp.ones(512, dtype=xp.complex64)
        out = apply_chromatic_dispersion(samples, 17.0, 80.0, 1550.0, 64e9)
        assert out.shape == (512,)

    def test_output_shape_mimo(self, backend_device, xp):
        """MIMO output shape matches input."""
        samples = xp.ones((2, 512), dtype=xp.complex64)
        out = apply_chromatic_dispersion(samples, 17.0, 80.0, 1550.0, 64e9)
        assert out.shape == (2, 512)

    def test_modifies_signal(self, backend_device, xp):
        """CD should change the signal (non-trivial dispersion)."""
        rng = xp.random.RandomState(99)
        samples = (rng.randn(512) + 1j * rng.randn(512)).astype(xp.complex64)
        out = apply_chromatic_dispersion(samples, 17.0, 80.0, 1550.0, 64e9)
        diff = float(xp.max(xp.abs(out - samples)))
        assert diff > 1e-3

    def test_energy_preserved(self, backend_device, xp, xpt):
        """CD is an all-pass filter: energy must be preserved."""
        rng = xp.random.RandomState(11)
        samples = (rng.randn(1024) + 1j * rng.randn(1024)).astype(xp.complex64)
        out = apply_chromatic_dispersion(samples, 17.0, 80.0, 1550.0, 64e9)
        power_in = float(xp.sum(xp.abs(samples) ** 2))
        power_out = float(xp.sum(xp.abs(out) ** 2))
        xpt.assert_allclose(power_out, power_in, rtol=1e-4)
