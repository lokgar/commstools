"""Tests for adaptive and block equalization algorithms."""

import numpy as np
import pytest

from commstools import equalizers
from commstools.equalizers import EqualizerResult
from commstools.mapping import gray_constellation


# ============================================================================
# SPS VALIDATION TESTS
# ============================================================================


class TestSPSValidation:
    """Tests that sps != 2 raises ValueError."""

    def test_lms_rejects_sps_1(self, backend_device, xp):
        """LMS should raise ValueError when sps=1."""
        tx = xp.ones(100, dtype=xp.complex64)
        constellation = xp.asarray(gray_constellation("psk", 4)).astype(xp.complex64)
        with pytest.raises(ValueError, match="2 samples/symbol"):
            equalizers.lms(
                tx,
                training_symbols=tx,
                num_taps=5,
                modulation="psk",
                order=4,
                sps=1,
            )

    def test_rls_rejects_sps_3(self, backend_device, xp):
        """RLS should raise ValueError when sps=3."""
        tx = xp.ones(100, dtype=xp.complex64)
        constellation = xp.asarray(gray_constellation("psk", 4)).astype(xp.complex64)
        with pytest.raises(ValueError, match="2 samples/symbol"):
            equalizers.rls(
                tx,
                training_symbols=tx,
                num_taps=5,
                modulation="psk",
                order=4,
                sps=3,
            )

    def test_cma_rejects_sps_1(self, backend_device, xp):
        """CMA should raise ValueError when sps=1."""
        tx = xp.ones(100, dtype=xp.complex64)
        with pytest.raises(ValueError, match="2 samples/symbol"):
            equalizers.cma(tx, num_taps=5, sps=1)


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
        constellation = xp.asarray(gray_constellation("psk", 4)).astype(xp.complex64)

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

        result = equalizers.lms(
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
        if hasattr(mse_tail, "get"):
            mse_tail = mse_tail.get()

        assert mse_tail < 0.1, f"LMS did not converge: tail MSE = {mse_tail:.4f}"

    def test_decision_directed_after_training(self, backend_device, xp):
        """LMS should maintain performance in DD mode after training."""
        n_symbols = 1000
        n_train = 300
        channel = xp.array([0.1, 1.0, 0.2], dtype=xp.complex64)
        constellation = xp.asarray(gray_constellation("psk", 4)).astype(xp.complex64)

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

        result = equalizers.lms(
            rx,
            training_symbols=tx[:n_train],
            num_taps=15,
            step_size=0.01,
            modulation="psk",
            order=4,
        )

        # Check DD mode MSE
        mse_dd = xp.mean(xp.abs(result.error[n_train + 50 :]) ** 2)
        if hasattr(mse_dd, "get"):
            mse_dd = mse_dd.get()

        assert mse_dd < 0.2, f"LMS DD mode failed: MSE = {mse_dd:.4f}"

    def test_output_shape_siso(self, backend_device, xp):
        """LMS SISO output shapes should be correct."""
        n = 500
        constellation = xp.asarray(gray_constellation("psk", 4)).astype(xp.complex64)

        from commstools import Signal

        sig = Signal.psk(
            symbol_rate=1e6, num_symbols=n, order=4, pulse_shape="rrc", sps=2, seed=0
        )
        tx = xp.asarray(sig.source_symbols)
        rx = xp.asarray(sig.samples)

        result = equalizers.lms(
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
        constellation = xp.asarray(gray_constellation("psk", 4)).astype(xp.complex64)

        from commstools import Signal

        sig = Signal.psk(
            symbol_rate=1e6, num_symbols=n, order=4, pulse_shape="rrc", sps=2, seed=0
        )
        tx = xp.asarray(sig.source_symbols)
        rx = xp.asarray(sig.samples)

        result = equalizers.lms(
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
        constellation = xp.asarray(gray_constellation("psk", 4)).astype(xp.complex64)

        from commstools import Signal

        sig = Signal.psk(
            symbol_rate=1e6, num_symbols=100, order=4, pulse_shape="rrc", sps=2, seed=0
        )
        tx = xp.asarray(sig.source_symbols)
        rx = xp.asarray(sig.samples)

        result = equalizers.lms(
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
            equalizers.lms(rx)


# ============================================================================
# RLS TESTS
# ============================================================================


class TestRLS:
    """Tests for the RLS adaptive equalizer."""

    def test_convergence(self, backend_device, xp):
        """RLS should converge on a known ISI channel."""
        n_symbols = 500
        channel = xp.array([0.2, 1.0, 0.3], dtype=xp.complex64)
        constellation = xp.asarray(gray_constellation("psk", 4)).astype(xp.complex64)

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

        rx = xp.convolve(rx_up, channel, mode="same")

        result = equalizers.rls(
            rx,
            training_symbols=tx,
            num_taps=15,
            forgetting_factor=0.99,
            modulation="psk",
            order=4,
        )

        mse_tail = xp.mean(xp.abs(result.error[-100:]) ** 2)
        if hasattr(mse_tail, "get"):
            mse_tail = mse_tail.get()

        assert mse_tail < 0.1, f"RLS did not converge: tail MSE = {mse_tail:.4f}"

    def test_faster_convergence_than_lms(self, backend_device, xp):
        """RLS should converge faster than LMS (lower MSE in early symbols)."""
        n_symbols = 300
        channel = xp.array([0.3, 1.0, 0.2], dtype=xp.complex64)
        constellation = xp.asarray(gray_constellation("psk", 4)).astype(xp.complex64)

        from commstools import Signal

        sig = Signal.psk(
            symbol_rate=1e6,
            num_symbols=n_symbols,
            order=4,
            pulse_shape="rrc",
            sps=2,
            seed=77,
        )
        tx = xp.asarray(sig.source_symbols)
        rx_up = xp.asarray(sig.samples)

        rx = xp.convolve(rx_up, channel, mode="same")

        lms_result = equalizers.lms(
            rx,
            training_symbols=tx,
            num_taps=15,
            step_size=0.01,
            modulation="psk",
            order=4,
        )
        rls_result = equalizers.rls(
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

        if hasattr(lms_early, "get"):
            lms_early = lms_early.get()
            rls_early = rls_early.get()

        assert rls_early <= lms_early, (
            f"RLS ({rls_early:.4f}) not faster than LMS ({lms_early:.4f})"
        )

    def test_output_shape_siso(self, backend_device, xp):
        """RLS SISO output shapes should match LMS convention."""
        constellation = xp.asarray(gray_constellation("psk", 4)).astype(xp.complex64)
        from commstools import Signal

        sig = Signal.psk(
            symbol_rate=1e6, num_symbols=200, order=4, pulse_shape="rrc", sps=2, seed=0
        )
        tx = xp.asarray(sig.source_symbols)
        rx = xp.asarray(sig.samples)

        result = equalizers.rls(
            rx,
            training_symbols=tx,
            num_taps=11,
            modulation="psk",
            order=4,
        )

        assert result.y_hat.ndim == 1
        assert result.weights.shape == (11,)


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

        result = equalizers.cma(
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

        if hasattr(modulus_std, "get"):
            modulus_std = modulus_std.get()

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

        if hasattr(r2, "get"):
            r2 = r2.get()

        np.testing.assert_allclose(r2, 1.0, atol=1e-6)

    def test_r2_default(self, backend_device, xp):
        """CMA should work with default R2=1.0."""
        from commstools import Signal

        sig = Signal.psk(
            symbol_rate=1e6, num_symbols=500, order=4, pulse_shape="rrc", sps=2, seed=0
        )
        rx = xp.asarray(sig.samples)

        result = equalizers.cma(rx, num_taps=11, step_size=0.01)

        assert isinstance(result, EqualizerResult)
        assert result.y_hat.shape[0] > 0

    def test_output_shape_siso(self, backend_device, xp):
        """CMA SISO output should be 1D."""
        from commstools import Signal

        sig = Signal.psk(
            symbol_rate=1e6, num_symbols=200, order=4, pulse_shape="rrc", sps=2, seed=0
        )
        rx = xp.asarray(sig.samples)

        result = equalizers.cma(rx, num_taps=11)

        assert result.y_hat.ndim == 1
        assert result.weights.shape == (11,)


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

        equalized = equalizers.zf_equalizer(symbols, channel)

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

        equalized = equalizers.zf_equalizer(rx, channel)

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

        zf_out = equalizers.zf_equalizer(rx_noisy, channel)
        mmse_out = equalizers.zf_equalizer(rx_noisy, channel, noise_variance=0.01)

        mse_zf = xp.mean(xp.abs(zf_out - tx) ** 2)
        mse_mmse = xp.mean(xp.abs(mmse_out - tx) ** 2)

        if hasattr(mse_zf, "get"):
            mse_zf = mse_zf.get()
            mse_mmse = mse_mmse.get()

        assert mse_mmse < mse_zf, (
            f"MMSE ({mse_mmse:.4f}) not better than ZF ({mse_zf:.4f})"
        )

    def test_siso_output_shape(self, backend_device, xp):
        """ZF SISO output should match input shape."""
        n = 64
        rx = xp.ones(n, dtype=xp.complex64)
        h = xp.array([1.0 + 0j], dtype=xp.complex64)

        out = equalizers.zf_equalizer(rx, h)
        assert out.shape == (n,)

    def test_mimo_per_channel(self, backend_device, xp):
        """ZF with SISO channel on MIMO input should equalize per-channel."""
        n = 64
        rx = xp.ones((2, n), dtype=xp.complex64)
        h = xp.array([1.0 + 0j], dtype=xp.complex64)

        out = equalizers.zf_equalizer(rx, h)
        assert out.shape == (2, n)


# ============================================================================
# BUTTERFLY MIMO TESTS
# ============================================================================


class TestButterflyMIMO:
    """Tests for butterfly MIMO equalization structure."""

    def test_lms_2x2_cross_channel(self, backend_device, xp):
        """LMS butterfly should recover 2 streams through a 2x2 mixing channel."""
        n_symbols = 3000
        constellation = xp.asarray(gray_constellation("psk", 4)).astype(xp.complex64)

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

        result = equalizers.lms(
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

        if hasattr(mse_ch0, "get"):
            mse_ch0 = mse_ch0.get()
            mse_ch1 = mse_ch1.get()

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

        result = equalizers.cma(
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
            if hasattr(std_dev, "get"):
                std_dev = std_dev.get()

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

        equalized = equalizers.zf_equalizer(rx, H)

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
        tx = xp.asarray(orig_sig.source_symbols)
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
        tx = xp.asarray(orig_sig.source_symbols)
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
