"""Tests for channel impairment models."""

import numpy as np
import pytest

from commstools.impairments import apply_awgn, apply_pmd


class TestApplyPMD:
    """Tests for the apply_pmd function."""

    def test_identity_no_dgd_no_rotation(self, backend_device, xp):
        """PMD with dgd=0, theta=0 should return the input unchanged."""
        N = 256
        rng = xp.random.RandomState(42)
        samples = (rng.randn(2, N) + 1j * rng.randn(2, N)).astype(xp.complex64)
        fs = 56e9

        out = apply_pmd(samples, dgd=0.0, theta=0.0, sampling_rate=fs)

        assert xp.allclose(out, samples, atol=1e-5)

    def test_energy_conservation(self, backend_device, xp):
        """PMD is unitary — output power should equal input power."""
        N = 1024
        rng = xp.random.RandomState(7)
        samples = (rng.randn(2, N) + 1j * rng.randn(2, N)).astype(xp.complex64)
        fs = 56e9

        out = apply_pmd(samples, dgd=5e-12, theta=np.pi / 5, sampling_rate=fs)

        power_in = float(xp.sum(xp.abs(samples) ** 2))
        power_out = float(xp.sum(xp.abs(out) ** 2))

        np.testing.assert_allclose(power_out, power_in, rtol=1e-4)

    def test_pure_rotation(self, backend_device, xp):
        """With dgd=0, PMD reduces to a polarization rotation by theta."""
        N = 128
        theta = np.pi / 4
        fs = 56e9

        # Constant signal: X-pol = 1+0j, Y-pol = 0+0j
        samples = xp.zeros((2, N), dtype=xp.complex64)
        samples[0, :] = 1.0 + 0j

        out = apply_pmd(samples, dgd=0.0, theta=theta, sampling_rate=fs)

        # After rotation with dgd=0, the channel should simply be R(theta).
        # x_out = cos(theta) * x_in - sin(theta) * y_in
        # y_out = sin(theta) * x_in + cos(theta) * y_in
        expected = xp.zeros((2, N), dtype=xp.complex64)
        expected[0, :] = np.cos(theta)
        expected[1, :] = np.sin(theta)

        assert xp.allclose(out, expected, atol=1e-5)

    def test_pure_rotation_with_dgd_zero_coupling(self, backend_device, xp):
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

        np.testing.assert_allclose(power_out, power_in, rtol=1e-4)

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

    def test_requires_sampling_rate_for_array(self, backend_device, xp):
        """Should raise ValueError if sampling_rate not provided for raw arrays."""
        samples = xp.ones((2, 100), dtype=xp.complex64)
        with pytest.raises(ValueError, match="sampling_rate"):
            apply_pmd(samples, dgd=1e-12)

    def test_signal_object_integration(self, backend_device, xp):
        """Should work with Signal objects, extracting sampling_rate."""
        from commstools.core import Signal
        from commstools.helpers import random_bits
        from commstools.mapping import map_bits

        n_sym = 256
        bits0 = xp.asarray(random_bits(n_sym * 4, seed=10))
        bits1 = xp.asarray(random_bits(n_sym * 4, seed=20))
        tx0 = map_bits(bits0, "qam", 16).astype(xp.complex64)
        tx1 = map_bits(bits1, "qam", 16).astype(xp.complex64)

        # 2-pol at 2 SPS
        samples = xp.zeros((2, n_sym * 2), dtype=xp.complex64)
        samples[0, ::2] = tx0
        samples[1, ::2] = tx1

        sig = Signal(
            samples=samples,
            sampling_rate=56e9,
            symbol_rate=28e9,
        )

        result = apply_pmd(sig, dgd=5e-12, theta=np.pi / 6)

        assert isinstance(result, Signal)
        assert result.samples.shape == sig.samples.shape
        assert result.sampling_rate == sig.sampling_rate

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

    def test_awgn_signal_object(self, backend_device, xp):
        """apply_awgn should accept Signal objects and return a Signal."""
        from commstools.core import Signal

        sig = Signal.pam(order=2, num_symbols=100, sps=4, symbol_rate=1e6)
        noisy_sig = apply_awgn(sig, esn0_db=10)

        assert isinstance(noisy_sig, Signal)
        assert noisy_sig.samples.shape == sig.samples.shape
        assert noisy_sig.sps == 4

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

    def test_awgn_array_without_sps(self, backend_device, xp):
        """Array input without sps argument should raise ValueError."""
        data = xp.ones(100, dtype=complex)
        with pytest.raises(ValueError, match="sps must be provided"):
            apply_awgn(data, esn0_db=10)
