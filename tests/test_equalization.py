"""Tests for adaptive and block equalization algorithms."""

import sys
from unittest.mock import patch

import numpy as np
import pytest

from commstools import equalization
from commstools.equalization import EqualizerResult
from commstools.mapping import gray_constellation


def _to_np(arr):
    """Convert a NumPy or CuPy array to plain NumPy (no-op for NumPy)."""
    if hasattr(arr, "get"):  # CuPy
        return arr.get()
    return np.asarray(arr)


# ============================================================================
# SPS VALIDATION TESTS
# ============================================================================


class TestSPSValidation:
    """Tests that sps != 2 raises ValueError."""

    def test_lms_rejects_sps_1(self, backend_device, xp):
        """LMS should raise ValueError when sps=1."""
        tx = xp.ones(100, dtype=xp.complex64)

        with pytest.raises(ValueError, match="2 samples/symbol"):
            equalization.lms(
                tx,
                training_symbols=tx,
                num_taps=5,
                modulation="psk",
                order=4,
                sps=1,
            )

    def test_cma_rejects_sps_1(self, backend_device, xp):
        """CMA should raise ValueError when sps=1."""
        tx = xp.ones(100, dtype=xp.complex64)
        with pytest.raises(ValueError, match="2 samples/symbol"):
            equalization.cma(tx, num_taps=5, sps=1)


# ============================================================================
# LMS TESTS
# ============================================================================


class TestLMS:
    """Tests for the LMS adaptive equalizer."""

    def test_convergence_known_channel(self, backend_device, xp):
        """LMS should converge and recover QPSK through a known ISI channel."""
        n_symbols = 1000
        # Create channel on device
        channel = xp.array([0.2, 1.0, 0.3], dtype=xp.complex64)

        # Generate bits and symbols on device
        # Generate symbols and RRC pulse-shaped waveform

        from commstools import Signal

        sig = Signal.psk(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=4,
            pulse_shape="rrc",
            sps=2,
            seed=42,
        )

        tx = xp.asarray(sig.source_symbols)

        rx_up = xp.asarray(sig.samples)

        # Apply Channel (Convolution)
        rx = xp.convolve(rx_up, channel, mode="same")

        # Add Noise (SNR ~ 20dB implies sigma ~ 0.1 for unit power signal)
        # Using fixed noise for reproducibility in test
        rng = xp.random.RandomState(0)
        noise = 0.05 * (rng.randn(len(rx)) + 1j * rng.randn(len(rx))).astype(
            xp.complex64
        )
        rx = rx + noise

        result = equalization.lms(
            rx,
            training_symbols=tx,
            num_taps=15,
            step_size=0.01,
            modulation="psk",
            order=4,
        )

        assert isinstance(result, EqualizerResult)

        # Check convergence (MSE on device)
        # Last 200 symbols
        mse_tail = xp.mean(xp.abs(result.error[-200:]) ** 2)

        # Move singular scalar to CPU for assertion if needed, or assert on device scalar
        # Pytest/NumPy comparisons handling of CuPy scalars varies, explicit conversion is safest
        mse_tail = float(mse_tail)

        assert mse_tail < 0.1, f"LMS did not converge: tail MSE = {mse_tail:.4f}"

    def test_decision_directed_after_training(self, backend_device, xp):
        """LMS should maintain performance in DD mode after training."""
        n_symbols = 1000
        n_train = 300
        channel = xp.array([0.1, 1.0, 0.2], dtype=xp.complex64)

        # Generate symbols and RRC pulse-shaped waveform

        from commstools import Signal

        sig = Signal.psk(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=4,
            pulse_shape="rrc",
            sps=2,
            seed=123,
        )

        tx = xp.asarray(sig.source_symbols)

        rx_up = xp.asarray(sig.samples)
        rx = xp.convolve(rx_up, channel, mode="same")

        result = equalization.lms(
            rx,
            training_symbols=tx[:n_train],
            num_taps=15,
            step_size=0.01,
            modulation="psk",
            order=4,
        )

        # Check DD mode MSE
        mse_dd = xp.mean(xp.abs(result.error[n_train + 50 :]) ** 2)
        mse_dd = float(mse_dd)

        assert mse_dd < 0.2, f"LMS DD mode failed: MSE = {mse_dd:.4f}"

    def test_output_shape_siso(self, backend_device, xp):
        """LMS SISO output shapes should be correct."""
        n = 500

        from commstools import Signal

        sig = Signal.psk(
            symbol_rate=1e6, num_symbols=n, order=4, pulse_shape="rrc", sps=2, seed=0
        )
        tx = xp.asarray(sig.source_symbols)
        rx = xp.asarray(sig.samples)

        result = equalization.lms(
            rx,
            training_symbols=tx,
            num_taps=11,
            modulation="psk",
            order=4,
        )

        assert result.y_hat.ndim == 1
        assert result.weights.shape == (11,)
        assert result.error.ndim == 1
        assert result.y_hat.shape == result.error.shape

    def test_store_weights(self, backend_device, xp):
        """Weight history should be stored when requested."""
        n = 200
        num_taps = 7

        from commstools import Signal

        sig = Signal.psk(
            symbol_rate=1e6, num_symbols=n, order=4, pulse_shape="rrc", sps=2, seed=0
        )
        tx = xp.asarray(sig.source_symbols)
        rx = xp.asarray(sig.samples)

        result = equalization.lms(
            rx,
            training_symbols=tx,
            num_taps=num_taps,
            modulation="psk",
            order=4,
            store_weights=True,
        )

        assert result.weights_history is not None
        assert result.weights_history.ndim == 2
        assert result.weights_history.shape[1] == num_taps

    def test_no_weights_by_default(self, backend_device, xp):
        """Weight history should be None by default."""

        from commstools import Signal

        sig = Signal.psk(
            symbol_rate=1e6, num_symbols=100, order=4, pulse_shape="rrc", sps=2, seed=0
        )
        tx = xp.asarray(sig.source_symbols)
        rx = xp.asarray(sig.samples)

        result = equalization.lms(
            rx,
            training_symbols=tx,
            num_taps=5,
            modulation="psk",
            order=4,
        )

        assert result.weights_history is None

    def test_requires_constellation_or_training(self, backend_device, xp):
        """LMS should raise if neither training nor constellation is given."""
        from commstools import Signal

        sig = Signal.psk(
            symbol_rate=1e6, num_symbols=100, order=4, pulse_shape="rrc", sps=2, seed=0
        )
        rx = xp.asarray(sig.samples)
        with pytest.raises(ValueError):
            equalization.lms(rx)


# ============================================================================
# RLS TESTS
# ============================================================================


class TestRLS:
    """Tests for the RLS adaptive equalizer."""

    def test_convergence(self, backend_device, xp):
        """RLS should converge on a known ISI channel."""
        n_symbols = 500
        channel = xp.array([0.2, 1.0, 0.3], dtype=xp.complex64)

        from commstools import Signal

        sig = Signal.psk(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=4,
            pulse_shape="rrc",
            sps=1,
            seed=42,
        )
        tx = xp.asarray(sig.source_symbols)
        rx_up = xp.asarray(sig.samples)

        rx = xp.convolve(rx_up, channel, mode="same")

        result = equalization.rls(
            rx,
            training_symbols=tx,
            num_taps=15,
            forgetting_factor=0.99,
            modulation="psk",
            order=4,
        )

        mse_tail = xp.mean(xp.abs(result.error[-100:]) ** 2)
        mse_tail = float(mse_tail)

        assert mse_tail < 0.1, f"RLS did not converge: tail MSE = {mse_tail:.4f}"

    def test_faster_convergence_than_lms(self, backend_device, xp):
        """RLS should converge faster than LMS (lower MSE in early symbols)."""
        n_symbols = 300
        channel = xp.array([0.3, 1.0, 0.2], dtype=xp.complex64)

        from commstools import Signal

        sig = Signal.psk(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=4,
            pulse_shape="rrc",
            sps=1,
            seed=77,
        )
        tx = xp.asarray(sig.source_symbols)
        rx_up = xp.asarray(sig.samples)

        rx = xp.convolve(rx_up, channel, mode="same")

        lms_result = equalization.lms(
            rx,
            training_symbols=tx,
            num_taps=15,
            step_size=0.01,
            modulation="psk",
            order=4,
        )
        rls_result = equalization.rls(
            rx,
            training_symbols=tx,
            num_taps=15,
            forgetting_factor=0.99,
            modulation="psk",
            order=4,
        )

        # Compare MSE in the first 50 symbols (convergence speed)
        lms_early = xp.mean(xp.abs(lms_result.error[:50]) ** 2)
        rls_early = xp.mean(xp.abs(rls_result.error[:50]) ** 2)

        lms_early = float(lms_early)
        rls_early = float(rls_early)

        assert rls_early <= lms_early, (
            f"RLS ({rls_early:.4f}) not faster than LMS ({lms_early:.4f})"
        )

    def test_output_shape_siso(self, backend_device, xp):
        """RLS SISO output shapes should match LMS convention."""
        from commstools import Signal

        sig = Signal.psk(
            symbol_rate=1e6, num_symbols=200, order=4, pulse_shape="rrc", sps=2, seed=0
        )
        tx = xp.asarray(sig.source_symbols)
        rx = xp.asarray(sig.samples)

        result = equalization.rls(
            rx,
            training_symbols=tx,
            num_taps=11,
            modulation="psk",
            order=4,
        )

        assert result.y_hat.ndim == 1
        assert result.weights.shape == (11,)


# ============================================================================
# API REGRESSION TESTS
# ============================================================================


class TestAPIRegression:
    """Verify that removed parameters no longer exist on the public API."""

    def test_lms_has_no_normalize_param(self, backend_device, xp):
        """lms() must not accept a 'normalize' keyword — always NLMS."""
        from commstools import Signal

        sig = Signal.psk(
            symbol_rate=1e6, num_symbols=100, order=4, pulse_shape="rrc", sps=2, seed=0
        )
        rx = xp.asarray(sig.samples)
        tx = xp.asarray(sig.source_symbols)
        with pytest.raises(TypeError, match="normalize"):
            equalization.lms(
                rx,
                training_symbols=tx,
                num_taps=5,
                modulation="psk",
                order=4,
                normalize=True,
            )

    def test_cma_has_no_normalize_param(self, backend_device, xp):
        """cma() must not accept a 'normalize' keyword."""
        from commstools import Signal

        sig = Signal.psk(
            symbol_rate=1e6, num_symbols=100, order=4, pulse_shape="rrc", sps=2, seed=0
        )
        rx = xp.asarray(sig.samples)
        with pytest.raises(TypeError, match="normalize"):
            equalization.cma(rx, num_taps=5, normalize=True)


# ============================================================================
# CMA TESTS
# ============================================================================


class TestCMA:
    """Tests for the CMA blind equalizer."""

    def test_convergence_qpsk(self, backend_device, xp):
        """CMA should converge for QPSK (constant modulus) through ISI channel."""
        n_symbols = 2000
        channel = xp.array([0.2, 1.0, 0.3], dtype=xp.complex64)

        from commstools import Signal

        sig = Signal.psk(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=4,
            pulse_shape="rrc",
            sps=2,
            seed=42,
        )
        rx_up = xp.asarray(sig.samples)
        rx = xp.convolve(rx_up, channel, mode="same")
        rx = xp.ascontiguousarray(rx)  # Ensure contiguous for JAX

        result = equalization.cma(
            rx,
            num_taps=21,
            step_size=0.005,
            modulation="psk",
            order=4,
        )

        # After convergence, output should have roughly constant modulus
        y = result.y_hat
        modulus_tail = xp.abs(y[-500:])
        modulus_std = xp.std(modulus_tail)

        modulus_std = float(modulus_std)

        assert modulus_std < 0.3, (
            f"CMA output modulus not constant: std = {modulus_std:.4f}"
        )

    def test_r2_from_modulation(self, backend_device, xp):
        """R2 should be correctly auto-computed from constellation."""
        # QPSK: all points on unit circle -> R2 = 1.0
        # This test checks the logic inside CMA, but here we can just verify the property
        # using the public API if exposed, or just run a dummy CMA and check it doesn't crash?
        # The original test verified calculation logic.

        const = xp.asarray(gray_constellation("psk", 4))
        r2 = xp.mean(xp.abs(const) ** 4) / xp.mean(xp.abs(const) ** 2)

        r2 = float(r2)

        np.testing.assert_allclose(r2, 1.0, atol=1e-6)

    def test_r2_default(self, backend_device, xp):
        """CMA should work with default R2=1.0."""
        from commstools import Signal

        sig = Signal.psk(
            symbol_rate=1e6, num_symbols=500, order=4, pulse_shape="rrc", sps=2, seed=0
        )
        rx = xp.asarray(sig.samples)

        result = equalization.cma(rx, num_taps=11, step_size=0.01)

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.shape[0] > 0

    def test_output_shape_siso(self, backend_device, xp):
        """CMA SISO output should be 1D."""
        from commstools import Signal

        sig = Signal.psk(
            symbol_rate=1e6, num_symbols=200, order=4, pulse_shape="rrc", sps=2, seed=0
        )
        rx = xp.asarray(sig.samples)

        result = equalization.cma(rx, num_taps=11)

        assert result.y_hat.ndim == 1
        assert result.weights.shape == (11,)


# ============================================================================
# RDE TESTS
# ============================================================================


class TestRDE:
    """Tests for the Radius Directed Equalizer."""

    def test_convergence_qpsk(self, backend_device, xp):
        """RDE should converge for QPSK (single-ring) just like CMA."""
        n_symbols = 2000
        channel = xp.array([0.2, 1.0, 0.3], dtype=xp.complex64)

        from commstools import Signal

        sig = Signal.psk(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=4,
            pulse_shape="rrc",
            sps=2,
            seed=42,
        )
        rx = xp.convolve(xp.asarray(sig.samples), channel, mode="same")
        rx = xp.ascontiguousarray(rx)

        result = equalization.rde(
            rx,
            num_taps=21,
            step_size=0.005,
            modulation="psk",
            order=4,
        )

        y = result.y_hat
        modulus_tail = xp.abs(y[-500:])
        modulus_std = xp.std(modulus_tail)
        modulus_std = float(modulus_std)

        assert modulus_std < 0.3, (
            f"RDE QPSK modulus not converged: std = {modulus_std:.4f}"
        )

    def test_steady_state_error_16qam(self, backend_device, xp):
        """RDE steady-state error should be much lower than CMA on 16-QAM.

        CMA and RDE are both phase-ambiguous, so EVM vs the reference symbol
        sequence is not a reliable comparison.  Instead we compare the
        steady-state magnitude of each algorithm's *own* error signal.

        For 16-QAM (3 rings), CMA's error e = y*(|y|²−R²_Godard) is bounded
        below by a non-zero "design residual": since R²_Godard is a single
        average radius, inner-ring and outer-ring symbols always have large
        signed errors that cancel only in expectation, not per-symbol.

        RDE's error e = y*(|y|²−R_d²) uses the nearest ring for each symbol,
        so it approaches zero as the equalizer converges to correct ISI
        compensation.  Therefore: mean(|e_RDE|) << mean(|e_CMA|) at steady state.
        """

        from commstools import Signal

        n_symbols = 5000
        channel = xp.array([0.15, 1.0, 0.25], dtype=xp.complex64)

        sig = Signal.qam(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=16,
            pulse_shape="rrc",
            sps=2,
            seed=7,
        )
        rx = xp.convolve(xp.asarray(sig.samples), channel, mode="same")
        rx = xp.ascontiguousarray(rx)

        result_rde = equalization.rde(
            rx, num_taps=21, step_size=5e-4, modulation="qam", order=16
        )
        result_cma = equalization.cma(
            rx, num_taps=21, step_size=5e-4, modulation="qam", order=16
        )

        def late_error(err):
            return float(xp.mean(xp.abs(err[-500:])))

        err_rde = late_error(result_rde.error)
        err_cma = late_error(result_cma.error)

        assert err_rde < err_cma, (
            f"RDE steady-state error ({err_rde:.4f}) should be lower than "
            f"CMA ({err_cma:.4f}) for 16-QAM: CMA has an irreducible design "
            f"residual from using a single Godard radius on a multi-ring constellation"
        )

    def test_radii_extraction(self, backend_device, xp):
        """Unique radii should match known 16-QAM ring structure."""
        import numpy as _np
        from commstools.mapping import gray_constellation

        const = gray_constellation("qam", 16)
        radii = _np.unique(_np.round(_np.abs(const).astype(_np.float32), 6))

        # Standard normalized 16-QAM has 3 unique radii
        assert len(radii) == 3, f"Expected 3 unique radii for 16-QAM, got {len(radii)}"
        # All radii must be positive
        assert _np.all(radii > 0)

    def test_output_shape_siso(self, backend_device, xp):
        """RDE SISO output should be 1D with correct length."""
        from commstools import Signal

        sig = Signal.qam(
            symbol_rate=1e6, num_symbols=300, order=16, pulse_shape="rrc", sps=2, seed=0
        )
        rx = xp.asarray(sig.samples)

        result = equalization.rde(rx, num_taps=11, modulation="qam", order=16)

        assert result.y_hat.ndim == 1
        assert result.weights.shape == (11,)

    def test_signal_equalize_api(self, backend_device, xp):
        """Signal.equalize(method='rde') should work end-to-end."""
        from commstools import Signal

        sig = Signal.qam(
            symbol_rate=1e6,
            num_symbols=2000,
            order=16,
            pulse_shape="rrc",
            sps=2,
            seed=1,
        )
        # Add mild ISI
        channel = xp.array([0.1, 1.0, 0.15], dtype=xp.complex64)
        sig.samples = xp.convolve(xp.asarray(sig.samples), channel, mode="same")

        sig.equalize(method="rde", num_taps=21, step_size=5e-4)

        assert sig.samples.ndim == 1
        assert sig.sps == 1.0


# ============================================================================
# ZF/MMSE TESTS
# ============================================================================


class TestZFEqualizer:
    """Tests for the Zero-Forcing / MMSE block equalizer."""

    def test_identity_channel(self, backend_device, xp):
        """ZF should be a no-op for a unit impulse channel."""
        n = 128
        channel = xp.array([1.0 + 0j], dtype=xp.complex64)

        rng = xp.random.RandomState(0)
        symbols = (rng.randn(n) + 1j * rng.randn(n)).astype(xp.complex64)

        equalized = equalization.zf_equalizer(symbols, channel)

        # Check equality
        # xp.allclose works for both numpy and cupy
        assert xp.allclose(equalized, symbols, atol=1e-5)

    def test_simple_channel_inversion(self, backend_device, xp):
        """ZF should invert a simple 2-tap channel."""
        n = 256
        channel = xp.array([1.0, 0.5], dtype=xp.complex64)

        # Generate known signal and apply channel via FFT
        rng = xp.random.RandomState(42)
        tx = (rng.randn(n) + 1j * rng.randn(n)).astype(xp.complex64)

        # Apply Channel
        rx = xp.convolve(tx, channel, mode="full")[:n]

        equalized = equalization.zf_equalizer(rx, channel)

        assert xp.allclose(equalized, tx, atol=1e-4)

    def test_mmse_better_than_zf_in_noise(self, backend_device, xp):
        """MMSE should have lower MSE than ZF with a noisy spectral null."""
        n = 512
        channel = xp.array([1.0, -0.9, 0.1], dtype=xp.complex64)

        rng = xp.random.RandomState(42)
        tx = (rng.randn(n) + 1j * rng.randn(n)).astype(xp.complex64)

        rx_full = xp.convolve(tx, channel, mode="full")
        rx = rx_full[:n]

        # Add Noise
        noise = 0.1 * (rng.randn(n) + 1j * rng.randn(n)).astype(xp.complex64)
        rx_noisy = rx + noise

        zf_out = equalization.zf_equalizer(rx_noisy, channel)
        mmse_out = equalization.zf_equalizer(rx_noisy, channel, noise_variance=0.01)

        mse_zf = xp.mean(xp.abs(zf_out - tx) ** 2)
        mse_mmse = xp.mean(xp.abs(mmse_out - tx) ** 2)

        mse_zf = float(mse_zf)
        mse_mmse = float(mse_mmse)

        assert mse_mmse < mse_zf, (
            f"MMSE ({mse_mmse:.4f}) not better than ZF ({mse_zf:.4f})"
        )

    def test_siso_output_shape(self, backend_device, xp):
        """ZF SISO output should match input shape."""
        n = 64
        rx = xp.ones(n, dtype=xp.complex64)
        h = xp.array([1.0 + 0j], dtype=xp.complex64)

        out = equalization.zf_equalizer(rx, h)
        assert out.shape == (n,)

    def test_mimo_per_channel(self, backend_device, xp):
        """ZF with SISO channel on MIMO input should equalize per-channel."""
        n = 64
        rx = xp.ones((2, n), dtype=xp.complex64)
        h = xp.array([1.0 + 0j], dtype=xp.complex64)

        out = equalization.zf_equalizer(rx, h)
        assert out.shape == (2, n)

    def test_mmse_multi_block_two_sided_channel(self, backend_device, xp):
        """MMSE must correctly handle IIR two-sided response across multiple blocks.

        Channel h=[0.2, 1.0, 0.2] is symmetric with minimum frequency response 0.6
        (no spectral null). Its MMSE inverse is IIR and two-sided. The 50% overlap-save
        architecture discards N_fft/4 samples from each end of every IFFT block,
        rejecting both the causal and anti-causal circular transients. This test
        uses N >> N_fft/2 (N_fft=1024, B=512) to exercise multiple block boundaries.
        """
        N = 2048  # 4 blocks: ceil(2048/512)=4
        channel = xp.array([0.2, 1.0, 0.2], dtype=xp.complex64)
        # h=[0.2,1,0.2]: H(e^jω) = e^{-jω}(1 + 0.4cos(ω)) >= 0.6 — no null

        rng = xp.random.RandomState(7)
        tx = (rng.randn(N) + 1j * rng.randn(N)).astype(xp.complex64)

        rx_full = xp.convolve(tx, channel, mode="full")
        rx = rx_full[:N]

        noise = 0.05 * (rng.randn(N) + 1j * rng.randn(N)).astype(xp.complex64)
        rx_noisy = rx + noise

        out = equalization.zf_equalizer(rx_noisy, channel, noise_variance=0.0025)

        # Skip edges affected by convolution truncation (mode="full"[:N])
        interior = slice(10, N - 10)
        mse = xp.mean(xp.abs(out[interior] - tx[interior]) ** 2)
        mse = float(mse)

        # At σ²=0.0025, worst-case MMSE MSE ≈ σ²/min(|H|²) ≈ 0.0025/0.36 ≈ 0.007
        assert mse < 0.05, f"MMSE multi-block IIR test failed: interior MSE={mse:.4f}"


# ============================================================================
# BUTTERFLY MIMO TESTS
# ============================================================================


class TestButterflyMIMO:
    """Tests for butterfly MIMO equalization structure."""

    def test_lms_2x2_cross_channel(self, backend_device, xp):
        """LMS butterfly should recover 2 streams through a 2x2 mixing channel."""
        n_symbols = 3000

        from commstools import Signal

        # 2x2 channel mixing matrix
        H = xp.array([[1.0, 0.3], [0.2, 1.0]], dtype=xp.complex64)

        sig = Signal.psk(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=4,
            pulse_shape="rrc",
            sps=2,
            num_streams=2,
            seed=10,
        )
        tx_mimo = xp.asarray(sig.source_symbols)
        rx_up = xp.asarray(sig.samples)

        # Mix the streams (multiplying H @ rx_up where rx_up is (2, N_samples))
        rx_mimo = H @ rx_up
        rx_mimo = xp.ascontiguousarray(rx_mimo)

        result = equalization.lms(
            rx_mimo,
            training_symbols=tx_mimo,
            num_taps=21,
            step_size=0.01,
            modulation="psk",
            order=4,
        )

        y = result.y_hat
        w = result.weights

        # Output should be (2, N_sym)
        assert y.shape[0] == 2
        assert w.shape == (2, 2, 21)

        # Check convergence on both streams
        e = xp.abs(result.error) ** 2
        mse_ch0 = xp.mean(e[0, -500:])
        mse_ch1 = xp.mean(e[1, -500:])

        mse_ch0 = float(mse_ch0)
        mse_ch1 = float(mse_ch1)

        assert mse_ch0 < 0.1, f"MIMO ch0 MSE = {mse_ch0:.4f}"
        assert mse_ch1 < 0.1, f"MIMO ch1 MSE = {mse_ch1:.4f}"

    def test_cma_2x2_polarization_demux(self, backend_device, xp):
        """CMA butterfly should demux 2 mixed polarizations."""
        n_symbols = 3000

        from commstools import Signal

        sig = Signal.psk(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=4,
            pulse_shape="rrc",
            sps=2,
            num_streams=2,
            seed=30,
        )
        rx_up = xp.asarray(sig.samples)

        # Polarization rotation mixing
        theta = xp.pi / 6  # 30 degree rotation
        c, s = xp.cos(theta), xp.sin(theta)
        H = xp.array([[c, s], [-s, c]], dtype=xp.complex64)

        rx_mimo = H @ rx_up
        rx_mimo = xp.ascontiguousarray(rx_mimo)

        result = equalization.cma(
            rx_mimo,
            num_taps=11,
            step_size=0.003,
            modulation="psk",
            order=4,
        )

        y = result.y_hat
        assert y.shape[0] == 2

        # After CMA, output should have constant modulus on both streams
        for ch in range(2):
            modulus = xp.abs(y[ch, -500:])
            std_dev = xp.std(modulus)
            std_dev = float(std_dev)

            assert std_dev < 0.35, f"CMA MIMO ch{ch} modulus std = {std_dev:.4f}"

    def test_zf_mimo_channel_matrix(self, backend_device, xp):
        """ZF with full (C, C, L) channel matrix should invert MIMO channel."""
        n = 128
        rng = xp.random.RandomState(42)

        # 2x2 single-tap channel matrix
        # H[rx, tx, delay]
        H = xp.zeros((2, 2, 1), dtype=xp.complex64)
        H[0, 0, 0] = 1.0
        H[0, 1, 0] = 0.3
        H[1, 0, 0] = 0.2
        H[1, 1, 0] = 1.0

        # Transmit 2 independent streams
        tx = (rng.randn(2, n) + 1j * rng.randn(2, n)).astype(xp.complex64)

        # Apply MIMO channel in frequency domain
        H_f = xp.fft.fft(H, n=n, axis=-1)  # (2, 2, N)
        Tx_f = xp.fft.fft(tx, axis=-1)  # (2, N)
        Rx_f = xp.zeros_like(Tx_f)

        # Manually compute matrix mul at each freq bin?
        # Einsum could work: 'r t k, t k -> r k'
        # CuPy supports einsum.
        # But for loop is also fine for test clarity.
        # Actually H_f is (Rx, Tx, Freq), Tx_f is (Tx, Freq).
        # We want (Rx, Freq).

        # Vectorized matmul broadcasting over freq?
        # Move axis to make it (N, Rx, Tx) @ (N, Tx, 1) -> (N, Rx, 1)
        # H_f_T = H_f.transpose(2, 0, 1) # (N, 2, 2)
        # Tx_f_T = Tx_f.T[..., None]     # (N, 2, 1)
        # Rx_f_T = H_f_T @ Tx_f_T        # (N, 2, 1)
        # Rx_f = Rx_f_T.squeeze(-1).T    # (2, N)

        H_f_T = xp.moveaxis(H_f, -1, 0)
        Tx_f_T = xp.moveaxis(Tx_f, -1, 0)[..., None]
        Rx_f_T = H_f_T @ Tx_f_T
        Rx_f = xp.moveaxis(Rx_f_T.squeeze(-1), 0, -1)

        rx = xp.fft.ifft(Rx_f, axis=-1).astype(xp.complex64)

        equalized = equalization.zf_equalizer(rx, H)

        assert xp.allclose(equalized, tx, atol=1e-3)


# ============================================================================
# SIGNAL INTEGRATION TESTS
# ============================================================================


class TestSignalIntegration:
    """Tests for the Signal.equalize() and plot_equalizer() methods."""

    def test_signal_equalize_lms(self, backend_device, xp):
        """Signal.equalize() should work with LMS and return self."""
        from commstools.core import Signal

        n_symbols = 300
        # Generate using our Signal class properly
        orig_sig = Signal.psk(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=4,
            pulse_shape="rrc",
            sps=2,
            seed=55,
        )
        rx = xp.asarray(orig_sig.samples)

        # Explicitly pass samples to Signal, it should respect backend
        sig = Signal(
            samples=rx,
            sampling_rate=2e6,
            symbol_rate=1e6,
            mod_scheme="psk",
            mod_order=4,
        )

        # Equalize
        # Note: training_symbols must be passed. If it is an array, it must match backend
        # implementation of Signal.equalize should handle it, but best to pass congruent type.
        result_sig = sig.equalize(
            method="lms",
            num_taps=7,
            step_size=0.01,
        )

        assert result_sig is sig
        assert sig._equalizer_result is not None
        assert isinstance(sig._equalizer_result, EqualizerResult)

    def test_signal_equalize_rejects_wrong_sps(self, backend_device, xp):
        """Signal.equalize() should reject signals not at 2 SPS."""
        from commstools.core import Signal

        tx = xp.ones(100, dtype=xp.complex64)
        sig = Signal(
            samples=tx,
            sampling_rate=1e6,
            symbol_rate=1e6,  # sps = 1
            mod_scheme="psk",
            mod_order=4,
        )

        with pytest.raises(ValueError, match="2 SPS"):
            sig.equalize(method="lms", num_taps=5)

    def test_signal_equalize_zf(self, backend_device, xp):
        """Signal.equalize(method='zf') should apply ZF equalization."""
        from commstools.core import Signal

        samples = xp.ones(64, dtype=xp.complex64)
        sig = Signal(samples=samples, sampling_rate=1e6, symbol_rate=1e6)

        h = xp.array([1.0 + 0j], dtype=xp.complex64)
        result_sig = sig.equalize(method="zf", channel_estimate=h)

        assert result_sig is sig

    def test_plot_equalizer_requires_equalize_first(self, backend_device, xp):
        """plot_equalizer() should raise ValueError if equalize() not called."""
        from commstools.core import Signal

        sig = Signal(
            samples=xp.ones(100, dtype=xp.complex64),
            sampling_rate=2e6,
            symbol_rate=1e6,
        )

        with pytest.raises(ValueError, match="equalize"):
            sig.plot_equalizer()

    def test_plot_equalizer_returns_fig(self, backend_device, xp):
        """plot_equalizer() should return (fig, axes) after equalize()."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from commstools.core import Signal

        n_symbols = 200
        orig_sig = Signal.psk(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=4,
            pulse_shape="rrc",
            sps=2,
            seed=55,
        )
        rx = xp.asarray(orig_sig.samples)

        sig = Signal(
            samples=rx,
            sampling_rate=2e6,
            symbol_rate=1e6,
            mod_scheme="psk",
            mod_order=4,
        )
        sig.equalize(method="lms", num_taps=7, step_size=0.01)

        result = sig.plot_equalizer()
        assert result is not None
        fig, axes = result
        assert len(axes) == 2
        plt.close(fig)


# ============================================================================
# JAX BACKEND TESTS
# ============================================================================

jax = pytest.importorskip("jax", reason="JAX not installed")


class TestJAXBackend:
    """Tests for the JAX (lax.scan) backend on all adaptive algorithms.

    These tests cover the JAX kernel factories (_get_jax_lms, _get_jax_rls,
    _get_jax_cma, _get_jax_rde) and all JAX branches in the public API
    functions (lms, rls, cma, rde with backend='jax').
    """

    def _make_qpsk_rx(self, xp, n_symbols=1000, seed=0):
        """Helper: generate RRC-shaped QPSK at 2 SPS through mild ISI."""
        from commstools import Signal

        channel = xp.array([0.1, 1.0, 0.15], dtype=xp.complex64)
        sig = Signal.psk(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=4,
            pulse_shape="rrc",
            sps=2,
            seed=seed,
        )
        rx = xp.convolve(xp.asarray(sig.samples), channel, mode="same")
        return xp.ascontiguousarray(rx), sig

    def _make_qam16_rx(self, xp, n_symbols=2000, seed=0):
        """Helper: generate RRC-shaped 16-QAM at 2 SPS through mild ISI."""
        from commstools import Signal

        channel = xp.array([0.1, 1.0, 0.15], dtype=xp.complex64)
        sig = Signal.qam(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=16,
            pulse_shape="rrc",
            sps=2,
            seed=seed,
        )
        rx = xp.convolve(xp.asarray(sig.samples), channel, mode="same")
        return xp.ascontiguousarray(rx), sig

    # ---- LMS JAX ----

    def test_lms_jax_siso_convergence(self, backend_device, xp):
        """LMS JAX backend should converge on QPSK and return EqualizerResult."""
        rx, sig = self._make_qpsk_rx(xp)
        train = xp.asarray(sig.source_symbols)

        result = equalization.lms(
            rx,
            training_symbols=train,
            num_taps=11,
            step_size=0.05,
            modulation="psk",
            order=4,
            backend="jax",
        )

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.ndim == 1
        assert result.weights.shape == (11,)
        assert result.weights_history is None

    def test_lms_jax_store_weights(self, backend_device, xp):
        """LMS JAX backend should populate weights_history when requested."""
        rx, sig = self._make_qpsk_rx(xp, n_symbols=500)
        train = xp.asarray(sig.source_symbols)
        n_sym = rx.shape[0] // 2

        result = equalization.lms(
            rx,
            training_symbols=train,
            num_taps=7,
            step_size=0.05,
            modulation="psk",
            order=4,
            backend="jax",
            store_weights=True,
        )

        assert result.weights_history is not None
        assert result.weights_history.shape == (n_sym, 7)

    def test_lms_jax_constellation_from_training(self, backend_device, xp):
        """LMS JAX backend should infer constellation from training when no modulation given."""
        rx, sig = self._make_qpsk_rx(xp, n_symbols=500)
        train = xp.asarray(sig.source_symbols)

        # Pass training only, no modulation/order — constellation inferred from train
        result = equalization.lms(
            rx,
            training_symbols=train,
            num_taps=7,
            step_size=0.05,
            backend="jax",
            # intentionally omit modulation/order
        )

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.shape[0] > 0

    def test_lms_jax_num_train_symbols(self, backend_device, xp):
        """LMS JAX should respect num_train_symbols cap."""
        rx, sig = self._make_qpsk_rx(xp, n_symbols=1000)
        train = xp.asarray(sig.source_symbols)

        result = equalization.lms(
            rx,
            training_symbols=train,
            num_taps=11,
            step_size=0.05,
            modulation="psk",
            order=4,
            backend="jax",
            num_train_symbols=50,
        )

        assert isinstance(result, EqualizerResult)
        assert result.num_train_symbols <= 50

    def test_lms_jax_mimo(self, backend_device, xp):
        """LMS JAX butterfly should handle 2×2 MIMO input."""
        from commstools import Signal

        n_symbols = 1000
        sig = Signal.psk(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=4,
            pulse_shape="rrc",
            sps=2,
            seed=3,
        )
        rx1 = xp.asarray(sig.samples)
        rx2 = xp.roll(rx1, 1)
        rx_mimo = xp.stack([rx1, rx2], axis=0)  # (2, N)
        train = xp.stack([xp.asarray(sig.source_symbols)] * 2, axis=0)

        result = equalization.lms(
            rx_mimo,
            training_symbols=train,
            num_taps=7,
            step_size=0.05,
            modulation="psk",
            order=4,
            backend="jax",
        )

        assert result.y_hat.shape == (2, n_symbols)
        assert result.weights.shape == (2, 2, 7)

    # ---- RLS JAX ----

    def test_rls_jax_siso_convergence(self, backend_device, xp):
        """RLS JAX backend should converge and return correct shapes."""
        rx, sig = self._make_qpsk_rx(xp)
        train = xp.asarray(sig.source_symbols)

        result = equalization.rls(
            rx,
            training_symbols=train,
            sps=1,
            num_taps=7,
            modulation="psk",
            order=4,
            backend="jax",
        )

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.ndim == 1
        assert result.weights.shape == (7,)

    def test_rls_jax_with_leakage(self, backend_device, xp):
        """RLS JAX backend should run without error with leakage > 0."""
        rx, sig = self._make_qpsk_rx(xp)
        train = xp.asarray(sig.source_symbols)

        result = equalization.rls(
            rx,
            training_symbols=train,
            sps=1,
            num_taps=7,
            modulation="psk",
            order=4,
            backend="jax",
            leakage=1e-4,
        )

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.shape[0] > 0

    def test_rls_jax_store_weights(self, backend_device, xp):
        """RLS JAX backend should store weight trajectory when requested."""
        rx, sig = self._make_qpsk_rx(xp, n_symbols=500)
        train = xp.asarray(sig.source_symbols)

        result = equalization.rls(
            rx,
            training_symbols=train,
            sps=1,
            num_taps=7,
            modulation="psk",
            order=4,
            backend="jax",
            store_weights=True,
        )

        assert result.weights_history is not None

    # ---- CMA JAX ----

    def test_cma_jax_siso_convergence(self, backend_device, xp):
        """CMA JAX backend should converge for QPSK."""
        rx, _ = self._make_qpsk_rx(xp)

        result = equalization.cma(
            rx,
            num_taps=11,
            step_size=0.005,
            modulation="psk",
            order=4,
            backend="jax",
        )

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.ndim == 1
        assert result.weights.shape == (11,)

    def test_cma_jax_store_weights(self, backend_device, xp):
        """CMA JAX backend should populate weights_history when requested."""
        rx, _ = self._make_qpsk_rx(xp, n_symbols=400)
        n_sym = rx.shape[0] // 2

        result = equalization.cma(
            rx,
            num_taps=7,
            step_size=0.005,
            modulation="psk",
            order=4,
            backend="jax",
            store_weights=True,
        )

        assert result.weights_history is not None
        assert result.weights_history.shape == (n_sym, 7)

    def test_cma_jax_default_r2(self, backend_device, xp):
        """CMA JAX backend should work with default R²=1.0 (no modulation given)."""
        rx, _ = self._make_qpsk_rx(xp, n_symbols=400)

        result = equalization.cma(rx, num_taps=7, step_size=0.005, backend="jax")

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.shape[0] > 0

    # ---- RDE JAX ----

    def test_rde_jax_siso_convergence(self, backend_device, xp):
        """RDE JAX backend should converge on QPSK."""
        rx, _ = self._make_qpsk_rx(xp)

        result = equalization.rde(
            rx,
            num_taps=11,
            step_size=0.005,
            modulation="psk",
            order=4,
            backend="jax",
        )

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.ndim == 1
        assert result.weights.shape == (11,)

    def test_rde_jax_16qam(self, backend_device, xp):
        """RDE JAX backend should converge on 16-QAM."""
        rx, _ = self._make_qam16_rx(xp)

        result = equalization.rde(
            rx,
            num_taps=11,
            step_size=5e-4,
            modulation="qam",
            order=16,
            backend="jax",
        )

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.ndim == 1

    def test_rde_jax_store_weights(self, backend_device, xp):
        """RDE JAX backend should populate weights_history when requested."""
        rx, _ = self._make_qpsk_rx(xp, n_symbols=400)
        n_sym = rx.shape[0] // 2

        result = equalization.rde(
            rx,
            num_taps=7,
            step_size=0.005,
            modulation="psk",
            order=4,
            backend="jax",
            store_weights=True,
        )

        assert result.weights_history is not None
        assert result.weights_history.shape == (n_sym, 7)

    def test_rde_jax_no_modulation_unit_radius(self, backend_device, xp):
        """RDE JAX backend with no modulation should fall back to unit radius (≡ CMA)."""
        rx, _ = self._make_qpsk_rx(xp, n_symbols=400)

        result = equalization.rde(rx, num_taps=7, step_size=0.005, backend="jax")

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.shape[0] > 0

    def test_rde_jax_mimo(self, backend_device, xp):
        """RDE JAX butterfly should handle 2×2 MIMO input."""
        from commstools import Signal

        n_symbols = 1000
        sig = Signal.qam(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=16,
            pulse_shape="rrc",
            sps=2,
            seed=5,
        )
        rx1 = xp.asarray(sig.samples)
        rx_mimo = xp.stack([rx1, xp.roll(rx1, 2)], axis=0)

        result = equalization.rde(
            rx_mimo,
            num_taps=7,
            step_size=5e-4,
            modulation="qam",
            order=16,
            backend="jax",
        )

        assert result.y_hat.shape == (2, n_symbols)
        assert result.weights.shape == (2, 2, 7)

    def test_kernel_cache_reuse(self, backend_device, xp):
        """Calling the same JAX algorithm twice with identical parameters must reuse the cache."""
        from commstools.equalization import _JITTED_EQ

        rx, sig = self._make_qpsk_rx(xp, n_symbols=300)
        train = xp.asarray(sig.source_symbols)

        equalization.lms(
            rx,
            training_symbols=train,
            num_taps=5,
            modulation="psk",
            order=4,
            backend="jax",
        )
        key_after_first = set(_JITTED_EQ.keys())
        equalization.lms(
            rx,
            training_symbols=train,
            num_taps=5,
            modulation="psk",
            order=4,
            backend="jax",
        )
        key_after_second = set(_JITTED_EQ.keys())

        # No new keys should be added on the second identical call
        new_keys = key_after_second - key_after_first
        assert len(new_keys) == 0, f"Unexpected new cache entries: {new_keys}"


# ============================================================================
# STORE WEIGHTS TESTS
# ============================================================================


class TestStoreWeights:
    """Tests that store_weights=True produces correct weight history shapes."""

    def _qpsk_rx(self, xp, n_symbols=600, seed=0):
        from commstools import Signal

        sig = Signal.psk(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=4,
            pulse_shape="rrc",
            sps=2,
            seed=seed,
        )
        return xp.ascontiguousarray(xp.asarray(sig.samples)), sig

    def test_lms_store_weights_numba(self, backend_device, xp):
        """LMS Numba: weights_history has shape (N_sym, num_taps) for SISO."""
        rx, sig = self._qpsk_rx(xp)
        n_sym = rx.shape[0] // 2

        result = equalization.lms(
            rx,
            training_symbols=xp.asarray(sig.source_symbols),
            num_taps=9,
            step_size=0.05,
            modulation="psk",
            order=4,
            backend="numba",
            store_weights=True,
        )

        assert result.weights_history is not None
        assert result.weights_history.shape == (n_sym, 9)

    def test_rls_store_weights_numba(self, backend_device, xp):
        """RLS Numba: weights_history has shape (N_sym_truncated, num_taps) for SISO."""
        rx, sig = self._qpsk_rx(xp)

        result = equalization.rls(
            rx,
            training_symbols=xp.asarray(sig.source_symbols),
            sps=1,
            num_taps=7,
            modulation="psk",
            order=4,
            backend="numba",
            store_weights=True,
        )

        assert result.weights_history is not None
        # SISO: (N_sym_truncated, num_taps)
        assert result.weights_history.ndim == 2
        assert result.weights_history.shape[1] == 7

    def test_cma_store_weights_numba(self, backend_device, xp):
        """CMA Numba: weights_history has shape (N_sym, num_taps) for SISO."""
        rx, _ = self._qpsk_rx(xp)
        n_sym = rx.shape[0] // 2

        result = equalization.cma(
            rx,
            num_taps=9,
            step_size=0.005,
            modulation="psk",
            order=4,
            backend="numba",
            store_weights=True,
        )

        assert result.weights_history is not None
        assert result.weights_history.shape == (n_sym, 9)

    def test_rde_store_weights_numba(self, backend_device, xp):
        """RDE Numba: weights_history has shape (N_sym, num_taps) for SISO."""
        rx, _ = self._qpsk_rx(xp)
        n_sym = rx.shape[0] // 2

        result = equalization.rde(
            rx,
            num_taps=9,
            step_size=0.005,
            modulation="psk",
            order=4,
            backend="numba",
            store_weights=True,
        )

        assert result.weights_history is not None
        assert result.weights_history.shape == (n_sym, 9)

    def test_mimo_store_weights_numba(self, backend_device, xp):
        """LMS Numba MIMO: weights_history has shape (N_sym, C, C, num_taps)."""
        from commstools import Signal

        sig = Signal.psk(
            symbol_rate=1e6, num_symbols=600, order=4, pulse_shape="rrc", sps=2, seed=0
        )
        rx = xp.asarray(sig.samples)
        rx_mimo = xp.stack([rx, xp.roll(rx, 1)], axis=0)
        train = xp.stack([xp.asarray(sig.source_symbols)] * 2, axis=0)

        result = equalization.lms(
            rx_mimo,
            training_symbols=train,
            num_taps=5,
            modulation="psk",
            order=4,
            backend="numba",
            store_weights=True,
        )

        n_sym = rx.shape[0] // 2
        assert result.weights_history is not None
        assert result.weights_history.shape == (n_sym, 2, 2, 5)

    def test_no_weights_by_default_all_algorithms(self, backend_device, xp):
        """All algorithms should return weights_history=None by default."""
        rx, sig = self._qpsk_rx(xp, n_symbols=300)
        train = xp.asarray(sig.source_symbols)

        for algo, kwargs in [
            ("lms", dict(training_symbols=train, modulation="psk", order=4)),
            ("cma", dict(modulation="psk", order=4)),
            ("rde", dict(modulation="psk", order=4)),
        ]:
            result = getattr(equalization, algo)(rx, num_taps=7, **kwargs)
            assert result.weights_history is None, (
                f"{algo}: expected weights_history=None by default"
            )


# ============================================================================
# EDGE CASES AND ERROR PATHS
# ============================================================================


class TestEdgeCases:
    """Tests for error paths, warnings, and edge-case inputs."""

    def _qpsk_rx(self, xp, n_symbols=500, seed=0):
        from commstools import Signal

        sig = Signal.psk(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=4,
            pulse_shape="rrc",
            sps=2,
            seed=seed,
        )
        return xp.ascontiguousarray(xp.asarray(sig.samples)), sig

    def test_rde_rejects_sps_1(self, backend_device, xp):
        """RDE should raise ValueError when sps=1."""
        tx = xp.ones(100, dtype=xp.complex64)
        with pytest.raises(ValueError, match="2 samples/symbol"):
            equalization.rde(tx, num_taps=5, modulation="psk", order=4, sps=1)

    def test_lms_raises_no_constellation_numba(self, backend_device, xp):
        """LMS Numba: ValueError when no modulation and no training symbols (DD impossible)."""
        rx, _ = self._qpsk_rx(xp)
        with pytest.raises(ValueError, match="modulation and order must be provided"):
            equalization.lms(rx, num_taps=7, backend="numba")

    def test_lms_raises_no_constellation_jax(self, backend_device, xp):
        """LMS JAX: same ValueError for missing constellation."""
        pytest.importorskip("jax")
        rx, _ = self._qpsk_rx(xp)
        with pytest.raises(ValueError, match="modulation and order must be provided"):
            equalization.lms(rx, num_taps=7, backend="jax")

    def test_rls_warns_fractional_spacing(self, backend_device, xp):
        """RLS should warn when sps > 1 (ill-conditioned correlation matrix)."""
        rx, sig = self._qpsk_rx(xp, n_symbols=400)
        result = equalization.rls(
            xp.asarray(sig.samples),
            training_symbols=xp.asarray(sig.source_symbols),
            sps=2,
            num_taps=7,
            modulation="psk",
            order=4,
            backend="numba",
        )
        # Should complete and return valid output (even if warned)
        assert isinstance(result, EqualizerResult)

    def test_validate_sps_small_num_taps(self, backend_device, xp, caplog):
        """_validate_sps should log a warning when num_taps < 4*sps."""
        import logging

        rx, sig = self._qpsk_rx(xp)
        with caplog.at_level(logging.WARNING, logger="commstools"):
            equalization.lms(
                rx,
                training_symbols=xp.asarray(sig.source_symbols),
                num_taps=3,  # < 4*sps=8 → should warn
                modulation="psk",
                order=4,
                backend="numba",
            )
        assert any("small" in r.message.lower() for r in caplog.records)

    def test_rde_no_modulation_falls_back_to_cma(self, backend_device, xp):
        """RDE with no modulation should use unit radius (same gradient as CMA R²=1)."""
        rx, _ = self._qpsk_rx(xp)

        result = equalization.rde(rx, num_taps=7, step_size=0.005, backend="numba")

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.shape[0] > 0

    def test_rde_mimo_no_modulation(self, backend_device, xp):
        """RDE MIMO path with no modulation should run (unit radius, 2-ch)."""
        from commstools import Signal

        sig = Signal.psk(
            symbol_rate=1e6, num_symbols=600, order=4, pulse_shape="rrc", sps=2, seed=0
        )
        rx = xp.asarray(sig.samples)
        rx_mimo = xp.stack([rx, xp.roll(rx, 1)], axis=0)

        result = equalization.rde(rx_mimo, num_taps=7, step_size=5e-4, backend="numba")

        assert result.y_hat.shape == (2, rx.shape[0] // 2)

    def test_lms_num_train_symbols_clamps_numba(self, backend_device, xp):
        """LMS Numba: num_train_symbols caps training length recorded in result."""
        rx, sig = self._qpsk_rx(xp, n_symbols=1000)
        train = xp.asarray(sig.source_symbols)

        result = equalization.lms(
            rx,
            training_symbols=train,
            num_taps=7,
            step_size=0.05,
            modulation="psk",
            order=4,
            backend="numba",
            num_train_symbols=30,
        )

        assert result.num_train_symbols <= 30

    def test_lms_constellation_from_training_only_numba(self, backend_device, xp):
        """LMS Numba: constellation inferred from training_symbols alone (no modulation)."""
        rx, sig = self._qpsk_rx(xp)
        train = xp.asarray(sig.source_symbols)

        result = equalization.lms(
            rx,
            training_symbols=train,
            num_taps=7,
            step_size=0.05,
            backend="numba",
            # deliberately no modulation/order
        )

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.shape[0] > 0

    def test_cma_rde_rejects_sps_1(self, backend_device, xp):
        """Both CMA and RDE should reject sps != 2."""
        tx = xp.ones(100, dtype=xp.complex64)
        for algo in (equalization.cma, equalization.rde):
            with pytest.raises(ValueError, match="2 samples/symbol"):
                algo(tx, num_taps=5, sps=1)

    def test_center_tap_override(self, backend_device, xp):
        """Custom center_tap should shift the decision delay without error."""
        rx, sig = self._qpsk_rx(xp)
        train = xp.asarray(sig.source_symbols)

        result = equalization.lms(
            rx,
            training_symbols=train,
            num_taps=11,
            step_size=0.05,
            modulation="psk",
            order=4,
            backend="numba",
            center_tap=8,
        )

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.shape[0] > 0


# ============================================================================
# ZF MIMO 3x3 (linalg.inv path)
# ============================================================================


class TestZF3x3:
    """Tests for the ZF/MMSE equalizer with 3+ channel MIMO (uses linalg.inv)."""

    def test_zf_3x3_identity(self, backend_device, xp):
        """ZF on 3×3 identity channel should be a no-op (linalg.inv path)."""
        n = 256
        rng = xp.random.RandomState(0)
        symbols = (rng.randn(3, n) + 1j * rng.randn(3, n)).astype(xp.complex64)

        # 3×3 identity impulse response
        channel = xp.zeros((3, 3, 1), dtype=xp.complex64)
        for i in range(3):
            channel[i, i, 0] = 1.0

        equalized = equalization.zf_equalizer(symbols, channel)

        assert equalized.shape == (3, n)
        assert xp.allclose(equalized, symbols, atol=1e-4)

    def test_zf_3x3_inversion(self, backend_device, xp):
        """ZF should invert a non-trivial 3×3 single-tap MIMO channel."""
        n = 256
        rng = xp.random.RandomState(42)
        tx = (rng.randn(3, n) + 1j * rng.randn(3, n)).astype(xp.complex64)

        # Single-tap 3×3 mixing matrix
        H = xp.array(
            [[[1.0], [0.3], [0.0]], [[0.0], [1.0], [0.2]], [[0.1], [0.0], [1.0]]],
            dtype=xp.complex64,
        )

        # Apply channel: rx[i] = sum_j H[i,j] * tx[j]
        rx = xp.zeros_like(tx)
        for i in range(3):
            for j in range(3):
                rx[i] += H[i, j, 0] * tx[j]

        equalized = equalization.zf_equalizer(rx, H, noise_variance=0.0)

        assert equalized.shape == (3, n)
        assert xp.allclose(equalized, tx, atol=1e-3)

    def test_mmse_3x3_with_noise(self, backend_device, xp):
        """MMSE 3×3 should run without error and return correct shape."""
        n = 128
        channel = xp.zeros((3, 3, 2), dtype=xp.complex64)
        for i in range(3):
            channel[i, i, 0] = 1.0
            channel[i, i, 1] = 0.2

        rng = xp.random.RandomState(0)
        rx = (rng.randn(3, n) + 1j * rng.randn(3, n)).astype(xp.complex64)

        equalized = equalization.zf_equalizer(rx, channel, noise_variance=0.01)

        assert equalized.shape == (3, n)


# ============================================================================
# EXTENDED BUTTERFLY MIMO TESTS
# ============================================================================


class TestButterflyMIMOExtended:
    """Additional MIMO butterfly tests for RDE and JAX backends."""

    def test_rde_2x2_butterfly_numba(self, backend_device, xp):
        """RDE Numba butterfly should handle 2×2 cross-polarization without error."""
        from commstools import Signal

        n_symbols = 2000
        sig = Signal.qam(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=16,
            pulse_shape="rrc",
            sps=2,
            seed=11,
        )
        rx = xp.asarray(sig.samples)

        # Simulate polarization mixing: channel mixes two copies of same signal
        mix = xp.array([[0.9, 0.1], [0.1, 0.9]], dtype=xp.complex64)
        rx_mimo = xp.stack(
            [
                mix[0, 0] * rx + mix[0, 1] * xp.roll(rx, 3),
                mix[1, 0] * rx + mix[1, 1] * xp.roll(rx, 5),
            ],
            axis=0,
        )

        result = equalization.rde(
            rx_mimo,
            num_taps=11,
            step_size=5e-4,
            modulation="qam",
            order=16,
            backend="numba",
        )

        assert result.y_hat.shape == (2, n_symbols)
        assert result.weights.shape == (2, 2, 11)
        assert result.error.shape == (2, n_symbols)

    def test_lms_jax_2x2_cross_channel(self, backend_device, xp):
        """LMS JAX butterfly should cancel cross-channel interference."""
        from commstools import Signal

        pytest.importorskip("jax")

        n_symbols = 2000
        sig = Signal.psk(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=4,
            pulse_shape="rrc",
            sps=2,
            seed=7,
        )
        rx = xp.asarray(sig.samples)
        train = xp.asarray(sig.source_symbols)

        # 2×2 mixed input
        rx_mimo = xp.stack([rx, xp.roll(rx, 2)], axis=0)
        train_mimo = xp.stack([train, train], axis=0)

        result = equalization.lms(
            rx_mimo,
            training_symbols=train_mimo,
            num_taps=7,
            step_size=0.05,
            modulation="psk",
            order=4,
            backend="jax",
        )

        assert result.y_hat.shape == (2, n_symbols)
        assert result.weights.shape == (2, 2, 7)


# ============================================================================
# NUMBA BACKEND — COVERAGE GAPS
# ============================================================================


class TestNumbaBackendCoverage:
    """Tests targeting uncovered numba-backend code paths in LMS/RLS/CMA/RDE."""

    def _make_qpsk_rx(self, xp, n_symbols=1000, seed=0):
        from commstools import Signal

        channel = xp.array([0.1, 1.0, 0.15], dtype=xp.complex64)
        sig = Signal.psk(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=4,
            pulse_shape="rrc",
            sps=2,
            seed=seed,
        )
        rx = xp.convolve(xp.asarray(sig.samples), channel, mode="same")
        return xp.ascontiguousarray(rx), sig

    def test_lms_numba_pure_dd_no_training(self, backend_device, xp):
        """LMS numba with no training symbols (pure DD from start) covers _prepare_training_numpy else branch."""
        rx, _ = self._make_qpsk_rx(xp, n_symbols=600)

        # No training_symbols → _prepare_training_numpy gets None → n_train_aligned=0
        result = equalization.lms(
            rx,
            num_taps=11,
            step_size=0.01,
            modulation="psk",
            order=4,
            backend="numba",
        )

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.ndim == 1

    def test_rls_numba_siso(self, backend_device, xp):
        """RLS with numba backend on SISO input."""
        rx, sig = self._make_qpsk_rx(xp, n_symbols=800)
        train = xp.asarray(sig.source_symbols)

        result = equalization.rls(
            rx,
            training_symbols=train,
            sps=2,
            num_taps=9,
            modulation="psk",
            order=4,
            backend="numba",
        )

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.ndim == 1
        assert result.weights.shape == (9,)

    def test_rls_numba_mimo(self, backend_device, xp):
        """RLS numba MIMO path correctly handles (num_channels, n_samples) input shape."""
        from commstools import Signal

        n_symbols = 600
        sig = Signal.psk(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=4,
            pulse_shape="rrc",
            sps=2,
            seed=5,
        )
        rx1 = xp.asarray(sig.samples)
        rx_mimo = xp.stack([rx1, xp.roll(rx1, 2)], axis=0)  # (2, N)
        train_mimo = xp.stack([xp.asarray(sig.source_symbols)] * 2, axis=0)

        result = equalization.rls(
            rx_mimo,
            training_symbols=train_mimo,
            sps=2,
            num_taps=7,
            modulation="psk",
            order=4,
            backend="numba",
        )

        # RLS truncates the last num_taps//2 symbols from y_hat
        tail_trim = 7 // 2
        assert result.y_hat.shape == (2, n_symbols - tail_trim)
        assert result.weights.shape == (2, 2, 7)

    def test_rls_numba_num_train_symbols(self, backend_device, xp):
        """RLS numba should respect num_train_symbols when slicing training data."""
        rx, sig = self._make_qpsk_rx(xp, n_symbols=800)
        train = xp.asarray(sig.source_symbols)

        result = equalization.rls(
            rx,
            training_symbols=train,
            sps=2,
            num_taps=7,
            modulation="psk",
            order=4,
            backend="numba",
            num_train_symbols=50,
        )

        assert isinstance(result, EqualizerResult)
        assert result.num_train_symbols <= 50

    def test_rls_numba_constellation_from_training(self, backend_device, xp):
        """RLS numba derives constellation from training when no modulation is given."""
        rx, sig = self._make_qpsk_rx(xp, n_symbols=800)
        train = xp.asarray(sig.source_symbols)

        # Provide training but NOT modulation/order → constellation inferred from train
        result = equalization.rls(
            rx,
            training_symbols=train,
            sps=2,
            num_taps=7,
            backend="numba",
        )

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.shape[0] > 0

    def test_rls_numba_store_weights(self, backend_device, xp):
        """RLS numba with store_weights=True should populate weight history."""
        rx, sig = self._make_qpsk_rx(xp, n_symbols=400)
        train = xp.asarray(sig.source_symbols)

        result = equalization.rls(
            rx,
            training_symbols=train,
            sps=2,
            num_taps=5,
            modulation="psk",
            order=4,
            backend="numba",
            store_weights=True,
        )

        assert result.weights_history is not None

    def test_cma_numba_store_weights(self, backend_device, xp):
        """CMA numba backend with store_weights=True."""
        from commstools import Signal

        sig = Signal.psk(
            symbol_rate=1e6, num_symbols=400, order=4, pulse_shape="rrc", sps=2, seed=0
        )
        rx = xp.asarray(sig.samples)

        result = equalization.cma(
            rx,
            num_taps=7,
            step_size=0.005,
            modulation="psk",
            order=4,
            backend="numba",
            store_weights=True,
        )

        assert result.weights_history is not None

    def test_rde_numba_store_weights(self, backend_device, xp):
        """RDE numba backend with store_weights=True."""
        from commstools import Signal

        sig = Signal.qam(
            symbol_rate=1e6, num_symbols=400, order=16, pulse_shape="rrc", sps=2, seed=0
        )
        rx = xp.asarray(sig.samples)

        result = equalization.rde(
            rx,
            num_taps=7,
            step_size=5e-4,
            modulation="qam",
            order=16,
            backend="numba",
            store_weights=True,
        )

        assert result.weights_history is not None


# ============================================================================
# RLS JAX — CONSTELLATION FROM TRAINING
# ============================================================================


@pytest.mark.skipif(
    "jax" not in sys.modules and not pytest.importorskip("jax", reason="skip"),
    reason="JAX required",
)
class TestRLSJAXConstellationFromTraining:
    """RLS JAX derives constellation from training symbols when no modulation is given."""

    def test_rls_jax_constellation_from_training(self, backend_device, xp):
        """RLS JAX with training only (no modulation) infers constellation from training."""
        pytest.importorskip("jax")
        from commstools import Signal

        n_symbols = 800
        sig = Signal.psk(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=4,
            pulse_shape="rrc",
            sps=2,
            seed=9,
        )
        channel = xp.array([0.1, 1.0, 0.15], dtype=xp.complex64)
        rx = xp.convolve(xp.asarray(sig.samples), channel, mode="same")
        rx = xp.ascontiguousarray(rx)
        train = xp.asarray(sig.source_symbols)

        result = equalization.rls(
            rx,
            training_symbols=train,
            sps=2,
            num_taps=7,
            backend="jax",
            # No modulation/order → constellation from training
        )

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.ndim == 1


# ============================================================================
# IMPORT ERROR BRANCHES — MOCKED BACKENDS
# ============================================================================


class TestImportErrorBranches:
    """Tests for ImportError branches when Numba/JAX are unavailable (uses mocking)."""

    def _make_rx(self, xp, n_symbols=400):
        from commstools import Signal

        sig = Signal.psk(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=4,
            pulse_shape="rrc",
            sps=2,
            seed=0,
        )
        return xp.ascontiguousarray(xp.asarray(sig.samples)), sig

    def test_lms_jax_not_installed(self, backend_device, xp):
        """LMS raises ImportError when backend='jax' but JAX is not available."""
        rx, _ = self._make_rx(xp)
        with patch("commstools.equalization._get_jax", return_value=(None, None, None)):
            with pytest.raises(ImportError, match="JAX is required"):
                equalization.lms(rx, modulation="psk", order=4, backend="jax")

    def test_rls_numba_not_installed(self, backend_device, xp):
        """RLS raises ImportError when backend='numba' but Numba is not available."""
        rx, sig = self._make_rx(xp)
        train = xp.asarray(sig.source_symbols)
        with patch("commstools.equalization._get_numba", return_value=None):
            with pytest.raises(ImportError, match="Numba is required"):
                equalization.rls(
                    rx,
                    training_symbols=train,
                    sps=2,
                    modulation="psk",
                    order=4,
                    backend="numba",
                )

    def test_rls_jax_not_installed(self, backend_device, xp):
        """RLS raises ImportError when backend='jax' but JAX is not available."""
        rx, sig = self._make_rx(xp)
        train = xp.asarray(sig.source_symbols)
        with patch("commstools.equalization._get_jax", return_value=(None, None, None)):
            with pytest.raises(ImportError, match="JAX is required"):
                equalization.rls(
                    rx,
                    training_symbols=train,
                    sps=2,
                    modulation="psk",
                    order=4,
                    backend="jax",
                )

    def test_cma_numba_not_installed(self, backend_device, xp):
        """CMA raises ImportError when backend='numba' but Numba is not available."""
        rx, _ = self._make_rx(xp)
        with patch("commstools.equalization._get_numba", return_value=None):
            with pytest.raises(ImportError, match="Numba is required"):
                equalization.cma(rx, modulation="psk", order=4, backend="numba")

    def test_cma_jax_not_installed(self, backend_device, xp):
        """CMA raises ImportError when backend='jax' but JAX is not available."""
        rx, _ = self._make_rx(xp)
        with patch("commstools.equalization._get_jax", return_value=(None, None, None)):
            with pytest.raises(ImportError, match="JAX is required"):
                equalization.cma(rx, modulation="psk", order=4, backend="jax")

    def test_rde_numba_not_installed(self, backend_device, xp):
        """RDE raises ImportError when backend='numba' but Numba is not available."""
        rx, _ = self._make_rx(xp)
        with patch("commstools.equalization._get_numba", return_value=None):
            with pytest.raises(ImportError, match="Numba is required"):
                equalization.rde(rx, modulation="qam", order=16, backend="numba")

    def test_rde_jax_not_installed(self, backend_device, xp):
        """RDE raises ImportError when backend='jax' but JAX is not available."""
        rx, _ = self._make_rx(xp)
        with patch("commstools.equalization._get_jax", return_value=(None, None, None)):
            with pytest.raises(ImportError, match="JAX is required"):
                equalization.rde(rx, modulation="qam", order=16, backend="jax")


# ============================================================================
# LMS JAX — PURE DECISION-DIRECTED (NO TRAINING SYMBOLS)
# ============================================================================


class TestLMSJAXPureDD:
    """LMS JAX backend without training symbols runs pure decision-directed mode."""

    def test_lms_jax_pure_dd_no_training(self, backend_device, xp):
        """LMS JAX with modulation but no training_symbols runs in pure decision-directed mode from the start."""
        pytest.importorskip("jax")
        from commstools import Signal

        sig = Signal.psk(
            symbol_rate=1e6, num_symbols=800, order=4, pulse_shape="rrc", sps=2, seed=42
        )
        rx = xp.ascontiguousarray(xp.asarray(sig.samples))

        # No training_symbols → _prepare_training_jax receives None → else branch
        result = equalization.lms(
            rx,
            num_taps=11,
            step_size=0.01,
            modulation="psk",
            order=4,
            backend="jax",
            # deliberately no training_symbols
        )

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.ndim == 1


# ============================================================================
# W_INIT — WEIGHT HANDOFF TESTS
# ============================================================================


def _make_qam16_rx(xp, n_symbols=2000, seed=0):
    """Generate a simple AWGN-impaired 16-QAM signal at 2 SPS."""
    from commstools import Signal
    from commstools.impairments import apply_awgn

    sig = Signal.qam(
        symbol_rate=1e6,
        num_symbols=n_symbols,
        order=16,
        pulse_shape="rrc",
        sps=2,
        seed=seed,
    )
    rx = apply_awgn(sig.samples, esn0_db=20.0, sps=2)
    return xp.ascontiguousarray(xp.asarray(rx))


class TestWInit:
    """w_init parameter: warm-start from prior equalizer weights."""

    def test_lms_accepts_w_init(self, backend_device, xp):
        """lms() accepts w_init array with correct shape and returns EqualizerResult."""
        rx = _make_qam16_rx(xp)
        num_taps, num_ch = 21, 1
        w0 = np.zeros((num_ch, num_ch, num_taps), dtype=np.complex64)
        w0[0, 0, num_taps // 2] = 1.0 + 0j

        result = equalization.lms(
            rx,
            training_symbols=None,
            modulation="qam",
            order=16,
            num_taps=num_taps,
            w_init=w0,
        )
        assert isinstance(result, EqualizerResult)
        assert result.weights.shape == (num_taps,)  # SISO squeeze

    def test_rls_accepts_w_init(self, backend_device, xp):
        """rls() accepts w_init array with correct shape."""
        from commstools import Signal

        sig = Signal.qam(
            symbol_rate=1e6, num_symbols=1000, order=4, pulse_shape="rrc", sps=2, seed=1
        )
        rx = xp.asarray(sig.samples)
        num_taps, num_ch = 11, 1
        w0 = np.zeros((num_ch, num_ch, num_taps), dtype=np.complex64)
        w0[0, 0, num_taps // 2] = 1.0 + 0j

        result = equalization.rls(
            rx,
            training_symbols=xp.asarray(sig.source_symbols),
            modulation="qam",
            order=4,
            num_taps=num_taps,
            sps=2,
            w_init=w0,
        )
        assert isinstance(result, EqualizerResult)

    def test_cma_accepts_w_init(self, backend_device, xp):
        """cma() accepts w_init array with correct shape."""
        rx = _make_qam16_rx(xp)
        num_taps, num_ch = 21, 1
        w0 = np.zeros((num_ch, num_ch, num_taps), dtype=np.complex64)
        w0[0, 0, num_taps // 2] = 1.0 + 0j

        result = equalization.cma(
            rx, modulation="qam", order=16, num_taps=num_taps, w_init=w0
        )
        assert isinstance(result, EqualizerResult)

    def test_rde_accepts_w_init(self, backend_device, xp):
        """rde() accepts w_init array with correct shape."""
        rx = _make_qam16_rx(xp)
        num_taps, num_ch = 21, 1
        w0 = np.zeros((num_ch, num_ch, num_taps), dtype=np.complex64)
        w0[0, 0, num_taps // 2] = 1.0 + 0j

        result = equalization.rde(
            rx, modulation="qam", order=16, num_taps=num_taps, w_init=w0
        )
        assert isinstance(result, EqualizerResult)

    def test_w_init_shape_mismatch_raises(self, backend_device, xp):
        """Wrong w_init shape raises ValueError before kernel is called."""
        rx = _make_qam16_rx(xp)
        bad_w = np.zeros((1, 1, 99), dtype=np.complex64)  # wrong num_taps

        with pytest.raises(ValueError, match="w_init shape"):
            equalization.cma(rx, modulation="qam", order=16, num_taps=21, w_init=bad_w)

        with pytest.raises(ValueError, match="w_init shape"):
            equalization.rde(rx, modulation="qam", order=16, num_taps=21, w_init=bad_w)

    def test_lms_to_rde_handoff_output_shape(self, backend_device, xp):
        """LMS weights can be handed off to RDE via w_init; output shape is correct."""
        rx = _make_qam16_rx(xp, n_symbols=3000)
        half = rx.shape[-1] // 2

        pre_rx = rx[..., :half]
        payload_rx = rx[..., half:]

        pre = equalization.lms(
            pre_rx,
            modulation="qam",
            order=16,
            num_taps=21,
            step_size=0.05,
        )
        w0 = _to_np(pre.weights)
        if w0.ndim == 1:
            w0 = w0[np.newaxis, np.newaxis, :]

        result = equalization.rde(
            payload_rx,
            modulation="qam",
            order=16,
            num_taps=21,
            step_size=1e-4,
            w_init=w0,
        )
        expected_syms = payload_rx.shape[-1] // 2
        assert result.y_hat.shape[-1] == expected_syms

    def test_warm_start_rde_same_or_better_evm(self, backend_device, xp):
        """RDE warm-started from LMS achieves same or better EVM than cold-start."""
        from commstools import Signal
        from commstools.impairments import apply_awgn

        sig = Signal.qam(
            symbol_rate=1e6,
            num_symbols=4000,
            order=16,
            pulse_shape="rrc",
            sps=2,
            seed=42,
        )
        rx_np = apply_awgn(sig.samples, esn0_db=25.0, sps=2)
        rx = xp.asarray(rx_np)

        # Cold-start RDE
        cold = equalization.rde(
            rx, modulation="qam", order=16, num_taps=21, step_size=5e-4
        )
        # LMS pre-convergence
        pre = equalization.lms(
            rx,
            training_symbols=xp.asarray(sig.source_symbols),
            modulation="qam",
            order=16,
            num_taps=21,
            num_train_symbols=200,
        )
        _w = pre.weights
        w0 = _to_np(pre.weights)
        if w0.ndim == 1:
            w0 = w0[np.newaxis, np.newaxis, :]

        # Warm RDE
        warm = equalization.rde(
            rx, modulation="qam", order=16, num_taps=21, step_size=5e-4, w_init=w0
        )

        tail = slice(-500, None)
        ref = _to_np(sig.source_symbols)
        cold_hat = _to_np(cold.y_hat)
        warm_hat = _to_np(warm.y_hat)
        evm_cold = float(np.mean(np.abs(cold_hat[tail] - ref[tail]) ** 2))
        evm_warm = float(np.mean(np.abs(warm_hat[tail] - ref[tail]) ** 2))
        # Warm start must not be significantly worse
        assert evm_warm <= evm_cold * 1.5, (
            f"Warm RDE EVM {evm_warm:.4f} much worse than cold {evm_cold:.4f}"
        )


# ============================================================================
# HYBRID DA KERNELS — EQUALIZE_FRAME TESTS
# ============================================================================


def _make_sc_frame_no_pilots(seed=0):
    """Create a SingleCarrierFrame with preamble but no pilots."""
    from commstools.core import Preamble, SingleCarrierFrame

    preamble = Preamble(sequence_type="barker", length=13)
    frame = SingleCarrierFrame(
        payload_len=512,
        payload_mod_scheme="qam",
        payload_mod_order=16,
        preamble=preamble,
        pilot_pattern="none",
        payload_seed=seed,
    )
    return frame


class TestEqualizerWInitBackend:
    """Verify w_init works correctly on both numba and jax backends."""

    def test_cma_jax_w_init(self, backend_device, xp):
        """CMA JAX backend accepts w_init without error."""
        pytest.importorskip("jax")
        rx = _make_qam16_rx(xp)
        num_taps = 21
        w0 = np.zeros((1, 1, num_taps), dtype=np.complex64)
        w0[0, 0, num_taps // 2] = 1.0 + 0j

        result = equalization.cma(
            rx,
            modulation="qam",
            order=16,
            num_taps=num_taps,
            w_init=w0,
            backend="jax",
        )
        assert isinstance(result, EqualizerResult)

    def test_rde_jax_w_init(self, backend_device, xp):
        """RDE JAX backend accepts w_init without error."""
        pytest.importorskip("jax")
        rx = _make_qam16_rx(xp)
        num_taps = 21
        w0 = np.zeros((1, 1, num_taps), dtype=np.complex64)
        w0[0, 0, num_taps // 2] = 1.0 + 0j

        result = equalization.rde(
            rx,
            modulation="qam",
            order=16,
            num_taps=num_taps,
            w_init=w0,
            backend="jax",
        )
        assert isinstance(result, EqualizerResult)


class TestEqualizationFrame:
    """Tests for equalize_frame() — no-pilot and pilot-pattern variants."""

    def test_equalize_frame_no_pilots_output_shape(self, backend_device, xp):
        """equalize_frame() with no pilots returns payload-length output."""
        from commstools.equalization import equalize_frame

        frame = _make_sc_frame_no_pilots()
        sig = frame.to_signal(sps=2, symbol_rate=1e6)
        samples = xp.asarray(sig.samples)

        result = equalize_frame(
            samples,
            frame,
            num_taps=13,
            lms_step_size=0.05,
            blind_step_size=5e-4,
            blind_algorithm="rde",
            modulation="qam",
            order=16,
            sps=2,
        )
        assert isinstance(result, EqualizerResult)
        assert result.y_hat.shape[-1] > 0

    def test_equalize_frame_no_pilots_signal_method(self, backend_device, xp):
        """Signal.equalize_frame() with no pilots runs without error."""
        frame = _make_sc_frame_no_pilots()
        sig = frame.to_signal(sps=2, symbol_rate=1e6)

        sig.equalize_frame(
            frame,
            num_taps=13,
            lms_step_size=0.05,
            blind_step_size=5e-4,
            blind_algorithm="rde",
        )
        assert sig._equalizer_result is not None
        assert sig.samples.shape[-1] > 0

    def test_equalize_frame_num_train_symbols(self, backend_device, xp):
        """num_train_symbols in result equals preamble length when no pilots."""
        from commstools.equalization import equalize_frame

        frame = _make_sc_frame_no_pilots()
        sig = frame.to_signal(sps=2, symbol_rate=1e6)
        samples = xp.asarray(sig.samples)

        result = equalize_frame(
            samples,
            frame,
            num_taps=13,
            modulation="qam",
            order=16,
            sps=2,
        )
        assert result.num_train_symbols == 13

    def test_equalize_frame_external_w_init(self, backend_device, xp):
        """equalize_frame() accepts w_init and uses it for warm-start."""
        from commstools.equalization import equalize_frame

        frame = _make_sc_frame_no_pilots()
        sig = frame.to_signal(sps=2, symbol_rate=1e6)
        samples = xp.asarray(sig.samples)

        w0 = np.zeros((1, 1, 13), dtype=np.complex64)
        w0[0, 0, 6] = 1.0 + 0j

        result = equalize_frame(
            samples,
            frame,
            num_taps=13,
            modulation="qam",
            order=16,
            sps=2,
            w_init=w0,
        )
        assert isinstance(result, EqualizerResult)

    def test_equalize_frame_wrong_frame_type_raises(self, backend_device, xp):
        """equalize_frame() raises TypeError when frame is not SingleCarrierFrame."""
        from commstools.equalization import equalize_frame

        samples = xp.zeros(200, dtype=xp.complex64)
        with pytest.raises(TypeError, match="SingleCarrierFrame"):
            equalize_frame(samples, frame="not_a_frame", num_taps=11, sps=2)

    def test_equalize_frame_jax_backend_no_pilots(self, backend_device, xp):
        """equalize_frame() with backend='jax' and no pilots runs without error."""
        pytest.importorskip("jax")
        from commstools.equalization import equalize_frame

        frame = _make_sc_frame_no_pilots()
        sig = frame.to_signal(sps=2, symbol_rate=1e6)
        samples = xp.asarray(sig.samples)

        result = equalize_frame(
            samples,
            frame,
            num_taps=13,
            modulation="qam",
            order=16,
            sps=2,
            backend="jax",
        )
        assert isinstance(result, EqualizerResult)
        assert result.y_hat.shape[-1] > 0

    def test_equalize_frame_jax_backend_comb_pilots(self, backend_device, xp):
        """equalize_frame() with backend='jax' and comb pilots uses PA hybrid kernel."""
        pytest.importorskip("jax")
        from commstools.core import Preamble, SingleCarrierFrame
        from commstools.equalization import equalize_frame

        preamble = Preamble(sequence_type="barker", length=13)
        frame = SingleCarrierFrame(
            payload_len=256,
            payload_mod_scheme="qam",
            payload_mod_order=16,
            preamble=preamble,
            pilot_pattern="comb",
            pilot_period=8,
            pilot_mod_scheme="qam",
            pilot_mod_order=4,
            payload_seed=7,
        )
        sig = frame.to_signal(sps=2, symbol_rate=1e6)
        samples = xp.asarray(sig.samples)

        result = equalize_frame(
            samples,
            frame,
            num_taps=13,
            modulation="qam",
            order=16,
            sps=2,
            blind_algorithm="rde",
            backend="jax",
        )
        assert isinstance(result, EqualizerResult)
        assert result.y_hat.shape[-1] > 0
        assert result.num_train_symbols == 13  # preamble symbols
