"""Tests for source impairments (laser linewidth phase noise)."""

from commstools.impairments import apply_phase_noise


class TestApplyPhaseNoise:
    """Tests for apply_phase_noise."""

    def test_siso_output_shape(self, backend_device, xp):
        """SISO: output shape matches input."""
        samples = xp.ones(1024, dtype=xp.complex64)
        out = apply_phase_noise(samples, linewidth=100e3, sampling_rate=64e9, seed=1)
        assert out.shape == (1024,)

    def test_mimo_output_shape(self, backend_device, xp):
        """MIMO (C, N): output shape matches input."""
        samples = xp.ones((4, 512), dtype=xp.complex64)
        out = apply_phase_noise(samples, linewidth=100e3, sampling_rate=64e9, seed=2)
        assert out.shape == (4, 512)

    def test_modifies_signal(self, backend_device, xp):
        """Phase noise should change the signal."""
        samples = xp.ones(1024, dtype=xp.complex64)
        out = apply_phase_noise(samples, linewidth=1e6, sampling_rate=64e9, seed=3)
        diff = float(xp.max(xp.abs(out - samples)))
        assert diff > 1e-4

    def test_preserves_amplitude(self, backend_device, xp):
        """Phase rotation must not change sample amplitude."""
        rng = xp.random.RandomState(42)
        samples = (rng.randn(2048) + 1j * rng.randn(2048)).astype(xp.complex64)
        out = apply_phase_noise(samples, linewidth=100e3, sampling_rate=64e9, seed=4)
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
        out = apply_phase_noise(samples, linewidth=linewidth, sampling_rate=fs, seed=42)

        cumphase = xp.angle(out)  # (N,) wrapped, but increments are tiny
        increments = xp.diff(cumphase)  # (N-1,)

        # Small increments so wrapping is not an issue
        measured_var = float(xp.var(increments))
        assert abs(measured_var - expected_variance) / expected_variance < 0.05

    def test_shared_lo_all_channels_equal(self, backend_device, xp):
        """shared_lo=True: all channels receive identical phase trajectory."""
        C, N = 4, 512
        samples = xp.ones((C, N), dtype=xp.complex128)
        out = apply_phase_noise(
            samples, linewidth=1e6, sampling_rate=64e9, seed=7, shared_lo=True
        )
        # All channels should have identical output since same phase is applied
        for c in range(1, C):
            assert float(xp.max(xp.abs(out[c] - out[0]))) < 1e-10

    def test_independent_lo_channels_differ(self, backend_device, xp):
        """shared_lo=False (default): channels have independent phase trajectories."""
        C, N = 2, 512
        samples = xp.ones((C, N), dtype=xp.complex128)
        out = apply_phase_noise(
            samples, linewidth=10e6, sampling_rate=64e9, seed=5, shared_lo=False
        )
        # Independent trajectories should differ
        diff = float(xp.max(xp.abs(out[0] - out[1])))
        assert diff > 1e-4
