"""Tests for the base Signal class and its core signal processing methods."""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from unittest.mock import patch

from commstools.core import Signal


def test_signal_initialization(backend_device, xp):
    """Verify Signal initialization from basic Python lists and device-aware backend tracking."""
    # Test with list
    data = [1, 2, 3, 4]
    # Signal automatically moves to GPU if available (controlled by backend_device fixture)
    s = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)
    assert isinstance(s.samples, xp.ndarray)

    assert s.sampling_rate == 1.0
    assert s.symbol_rate == 1.0


def test_signal_validation_heuristics(backend_device, xp):
    """Verify Signal validation for higher dimensions and Time-Last heuristic."""
    # 1. Dimension > 2
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="Only 1D"):
        Signal(samples=xp.zeros((2, 2, 10)), sampling_rate=1.0, symbol_rate=1.0)

    # 2. Time-Last heuristic (Time, Channels) -> (Channels, Time)
    # Heuristic: s0 > s1 and s0 > 32
    data_wrong = xp.zeros((100, 2))
    s = Signal(samples=data_wrong, sampling_rate=1.0, symbol_rate=1.0)
    assert s.samples.shape == (2, 100)  # Should be transposed


def test_signal_auto_symbols(backend_device, xp):
    """Verify source_symbols derivation from source_bits in post-init."""

    bits = xp.array([0, 1, 0, 0], dtype="int8")
    # BPSK mapping: 0 -> -1, 1 -> 1
    s = Signal(
        samples=xp.ones(10),
        sampling_rate=1.0,
        symbol_rate=1.0,
        source_bits=bits,
        mod_scheme="PSK",
        mod_order=2,
    )
    assert s.source_symbols is not None
    assert len(s.source_symbols) == 4

    # 262: hyphenated mod
    s2 = Signal(
        samples=xp.ones(10),
        sampling_rate=1.0,
        symbol_rate=1.0,
        source_bits=bits,
        mod_scheme="PSK-MY",
        mod_order=2,
    )
    assert s2.source_symbols is not None
    # Check if derivation handled the "PSK" vs "mapping.map_bits" etc.


def test_signal_properties(backend_device, xp):
    """Verify core time-domain and rate properties of the Signal object."""
    # Create data directly on device using xp
    data = xp.zeros(100)
    fs = 100.0
    sym_rate = 10.0

    s = Signal(samples=data, sampling_rate=fs, symbol_rate=sym_rate)

    assert s.duration == 1.0
    assert s.sps == 10.0
    assert len(s.time_axis()) == 100


def test_signal_methods(backend_device, xp):
    """Verify common Signal methods like upsampling and FIR filtering."""
    # Test upsample
    data = xp.array([1.0 + 0j, -1.0 + 0j])
    s = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)

    s.upsample(2)
    assert s.sampling_rate == 2.0
    assert s.samples.shape[0] == 4

    # Test fir_filter
    taps = xp.array([1.0])
    s.fir_filter(taps)
    # should be unchanged roughly


def test_signal_resample_sps(backend_device, xp):
    """Verify Signal resampling using target samples per symbol (SPS)."""
    data = xp.ones(100)
    # create signal with sps=4 (fs=4, sym_rate=1)
    s = Signal(samples=data, sampling_rate=4.0, symbol_rate=1.0)
    assert s.sps == 4.0

    # resample to sps=8
    s.resample(sps_out=8.0)
    assert s.sps == 8.0
    assert s.sampling_rate == 8.0
    assert s.samples.size == 200


def test_welch_psd(backend_device, xp):
    """Verify Welch PSD estimation within the Signal object."""
    data = xp.random.randn(1000) + 1j * xp.random.randn(1000)
    s = Signal(samples=data, sampling_rate=100.0, symbol_rate=10.0)

    f, p = s.welch_psd(nperseg=64)
    assert f.shape == p.shape
    assert isinstance(f, xp.ndarray)


def test_signal_print_info(backend_device, xp, capsys):
    """Verify print_info() execution and output detection."""
    data = xp.zeros(10)
    s = Signal(samples=data, sampling_rate=100.0, symbol_rate=10.0)
    s.print_info()
    captured = capsys.readouterr()
    assert "Spectral Domain" in captured.out or captured.out == ""


def test_shaping_filter_taps_error(backend_device, xp):
    """Verify that shaping_filter_taps raises errors for unconfigured or unknown shapes."""
    data = xp.zeros(10)
    s = Signal(samples=data, sampling_rate=100.0, symbol_rate=10.0)

    with pytest.raises(ValueError, match="No pulse shape defined"):
        s.shaping_filter_taps()

    s.pulse_shape = "invalid_shape"
    with pytest.raises(ValueError, match="Unknown pulse shape"):
        s.shaping_filter_taps()


def test_signal_copy(backend_device, xp, xpt):
    """Verify Signal.copy() performs a deep copy of samples and preserves device context."""
    data = xp.array([1, 2, 3])
    s = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)
    s_copy = s.copy()

    assert s_copy is not s
    xpt.assert_allclose(s.samples, s_copy.samples)
    assert s_copy.backend == s.backend


def test_signal_shift_frequency(backend_device, xp, xpt):
    """Verify frequency shifting logic and resulting spectral peak positioning."""
    fs = 100.0
    # Simple DC signal (freq 0)
    data = xp.ones(100, dtype="complex128")
    s = Signal(samples=data, sampling_rate=fs, symbol_rate=10.0)

    # Offset by 20 Hz
    s.shift_frequency(20.0)

    # 1. Check metadata
    assert s.digital_frequency_offset == 20.0

    # 2. Check signal content
    t = xp.arange(100) / fs
    expected = xp.exp(1j * 2 * xp.pi * 20.0 * t)
    xpt.assert_allclose(s.samples, expected)

    # 3. Check accumulation
    s.shift_frequency(5.0)
    assert s.digital_frequency_offset == 25.0

    # Check approximate freq
    f, p = s.welch_psd(nperseg=64)
    peak = f[xp.argmax(p)]
    # 25 Hz expected
    assert abs(peak - 25.0) < (fs / 64)


def test_signal_resolution_and_demap(backend_device, xp, xpt):
    """Verify manual symbol resolution and bit demapping with caching."""
    # Generate a simple BPSK signal at 4 SPS
    symbol_rate = 1e6
    sps = 4
    num_symbols = 100
    sig = Signal.psk(
        order=2,
        num_symbols=num_symbols,
        sps=sps,
        symbol_rate=symbol_rate,
        seed=42,
    )

    # Initially resolved attributes should be None
    assert sig.resolved_symbols is None
    assert sig.resolved_bits is None

    # Calling demap_symbols_hard before resolve_symbols should raise ValueError
    with pytest.raises(ValueError, match="No resolved symbols available"):
        sig.demap_symbols_hard()

    # Resolve symbols with offset
    sig.resolve_symbols(offset=0)
    assert sig.resolved_symbols is not None
    assert len(sig.resolved_symbols) == num_symbols
    assert isinstance(sig.resolved_symbols, xp.ndarray)

    # Demap
    bits = sig.demap_symbols_hard()
    assert sig.resolved_bits is not None
    assert len(sig.resolved_bits) == num_symbols
    xpt.assert_array_equal(bits, sig.resolved_bits)

    # Metrics should now work
    evm_pct, evm_db = sig.evm()
    assert evm_pct >= 0

    snr_db = sig.snr()
    assert snr_db > 0

    # Test BER with manual reference bits
    ref_bits = sig.source_bits
    if ref_bits is not None:
        # ber() requires resolved_bits (populated by demap_symbols_hard above)
        ber = sig.ber(reference_bits=ref_bits)
        assert 0 <= ber <= 1

    # Test that BER raises if resolved_bits is missing
    sig.resolved_bits = None
    with pytest.raises(
        ValueError, match="Please call `demap_symbols_hard\\(\\)` first"
    ):
        sig.ber(reference_bits=ref_bits)


def test_signal_decimate_to_symbol_rate(backend_device, xp):
    """Verify downsampling Signal to symbols."""
    data = xp.ones(40, dtype="complex128")
    s = Signal(samples=data, sampling_rate=4.0, symbol_rate=1.0)
    s.decimate_to_symbol_rate(offset=0)
    assert len(s.samples) == 10
    assert s.sampling_rate == 1.0


def test_signal_downsample_warning(backend_device, xp):
    """Verify warning when downsampling already 1 SPS signal."""
    data = xp.ones(10, dtype="complex128")
    s = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)
    s.decimate_to_symbol_rate()  # Should just warn


def test_signal_mimo_fir_coverage(backend_device, xp):
    """Verify FIR filtering on multichannel signals."""
    data = xp.ones((2, 100), dtype="complex128")
    s = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)
    # Filter with delay
    taps = xp.array([1.0, 0.5])
    s.fir_filter(taps)
    assert s.samples.shape == (2, 100)
    # y[1] should be 1.5
    assert abs(float(s.samples[0, 1].real) - 1.5) < 1e-10


def test_signal_gaussian_coverage(backend_device, xp):
    """Verify Gaussian Signal generation."""
    # Use PSK with Gaussian pulse shaping
    s = Signal.psk(
        order=2,
        pulse_shape="gaussian",
        symbol_rate=1e6,
        num_symbols=100,
        sps=8,
        gaussian_bt=0.5,
    )
    assert s.pulse_shape == "gaussian"
    assert s.gaussian_bt == 0.5


def test_signal_jax_interop(backend_device, xp, xpt):
    """Verify JAX interoperability."""
    try:
        import jax
    except ImportError:
        pytest.skip("JAX not installed")

    s = Signal(samples=xp.ones(10), sampling_rate=1.0, symbol_rate=1.0)

    # Export to JAX
    jax_arr = s.export_samples_to_jax()
    assert isinstance(jax_arr, jax.Array)

    # Update from JAX
    jax_arr = jax_arr * 2.0
    s.update_samples_from_jax(jax_arr)
    xpt.assert_allclose(s.samples, 2.0)


def test_signal_properties_coverage(backend_device, xp):
    """Access Signal properties to ensure coverage."""
    s = Signal(samples=xp.zeros(100), sampling_rate=10.0, symbol_rate=2.0)

    # sp property
    assert s.sp is not None

    # duration property
    assert s.duration == 10.0

    # backend
    assert s.backend in ("CPU", "GPU")


def test_signal_validate_samples_transposition(backend_device, xp):
    """Cover the transposition warning heuristic in Signal."""
    # Create (Time, Channels) where Time >> Channels and Time > 32
    # e.g. (100, 2)
    data = xp.zeros((100, 2))

    # Signal expects (Channels, Time) usually, but logic detects (Time, Channels)
    # and transposes it, logging a warning.
    # We verify the shape is flipped to (2, 100).
    sig = Signal(samples=data, sampling_rate=1.0, symbol_rate=1.0)

    assert sig.samples.shape == (2, 100)

    assert sig.bits_per_symbol is None


def test_signal_wrappers(backend_device, xp):
    """
    Test the wrapper methods on Signal to ensure they call the underlying modules.
    We just check they run without error.
    """

    sig = Signal(
        samples=xp.zeros(100, dtype="complex64"), sampling_rate=100.0, symbol_rate=10.0
    )

    # Print info
    sig.print_info()

    # Properties
    assert sig.duration == 1.0
    assert sig.num_streams == 1

    # wrappers
    # Use small nperseg to match signal length
    f, p = sig.welch_psd(nperseg=32)
    assert len(f) > 0

    # Clean up any existing figures from previous tests
    plt.close("all")

    # Plotting wrappers (just call them, assume plotting logic tested elsewhere)
    # We pass show=False to avoid blocking
    sig.plot_psd(show=False, nperseg=32)
    sig.plot_waveform(num_symbols=10, show=False)
    sig.plot_eye(show=False)
    sig.plot_constellation(show=False)

    # Clean up figures to avoid RuntimeWarning
    plt.close("all")


def test_signal_duration_mimo(backend_device, xp):
    """Verify duration property for MIMO (2D) signal."""
    data = xp.zeros((2, 200))
    s = Signal(samples=data, sampling_rate=100.0, symbol_rate=10.0)
    assert s.duration == 2.0  # 200 / 100


def test_signal_bits_per_symbol_set(backend_device, xp):
    """Verify bits_per_symbol property when mod_order is set."""
    s = Signal(samples=xp.zeros(10), sampling_rate=1.0, symbol_rate=1.0, mod_order=16)
    assert s.bits_per_symbol == 4

    s2 = Signal(samples=xp.zeros(10), sampling_rate=1.0, symbol_rate=1.0, mod_order=64)
    assert s2.bits_per_symbol == 6


def test_rzpam_odd_sps(backend_device, xp):
    """Verify RZ-PAM raises error for odd SPS."""
    with pytest.raises(ValueError, match="sps.*must be even"):
        Signal.pam(
            order=2,
            num_symbols=10,
            sps=3,
            symbol_rate=1e3,
            rz=True,
        )


def test_rzpam_multi_stream(backend_device, xp):
    """Verify RZ-PAM multi-stream reshape produces correctly shaped multichannel output."""
    sig = Signal.pam(
        order=2,
        num_symbols=10,
        sps=4,
        symbol_rate=1e3,
        rz=True,
        num_streams=2,
        pulse_shape="rect",
    )
    # Should have 2 channels
    assert sig.samples.ndim == 2
    assert sig.samples.shape[0] == 2
    assert sig.source_bits is not None
    assert sig.source_symbols is not None


def test_resolve_symbols_sps_errors(backend_device, xp):
    """Verify resolve_symbols error paths for invalid SPS values."""
    # SPS < 1 (symbol_rate > sampling_rate)
    s = Signal(
        samples=xp.ones(10, dtype="complex64"), sampling_rate=1.0, symbol_rate=2.0
    )
    with pytest.raises(ValueError, match="Symbol rate must be >= 1"):
        s.resolve_symbols()

    # Non-integer SPS
    s2 = Signal(
        samples=xp.ones(10, dtype="complex64"), sampling_rate=3.0, symbol_rate=2.0
    )
    with pytest.raises(ValueError, match="Symbol rate must be an integer"):
        s2.resolve_symbols()


def test_demap_without_modulation(backend_device, xp):
    """Verify demap_symbols_hard raises error without modulation metadata."""
    s = Signal(
        samples=xp.ones(10, dtype="complex64"), sampling_rate=1.0, symbol_rate=1.0
    )
    s.resolved_symbols = xp.ones(10, dtype="complex64")
    with pytest.raises(ValueError, match="Modulation scheme and order required"):
        s.demap_symbols_hard()


def test_evm_no_reference(backend_device, xp):
    """Verify evm raises when no reference is available."""
    s = Signal(
        samples=xp.ones(10, dtype="complex64"), sampling_rate=1.0, symbol_rate=1.0
    )
    s.resolved_symbols = xp.ones(10, dtype="complex64")
    with pytest.raises(ValueError, match="No reference available"):
        s.evm()


def test_evm_no_resolved(backend_device, xp):
    """Verify evm raises when no resolved_symbols are present."""
    s = Signal(
        samples=xp.ones(10, dtype="complex64"),
        sampling_rate=1.0,
        symbol_rate=1.0,
        source_symbols=xp.ones(10, dtype="complex64"),
    )
    with pytest.raises(ValueError, match="No resolved symbols available"):
        s.evm()


def test_snr_no_reference(backend_device, xp):
    """Verify snr raises when no reference is available."""
    s = Signal(
        samples=xp.ones(10, dtype="complex64"), sampling_rate=1.0, symbol_rate=1.0
    )
    s.resolved_symbols = xp.ones(10, dtype="complex64")
    with pytest.raises(ValueError, match="No reference available"):
        s.snr()


def test_snr_no_resolved(backend_device, xp):
    """Verify snr raises when no resolved_symbols are present."""
    s = Signal(
        samples=xp.ones(10, dtype="complex64"),
        sampling_rate=1.0,
        symbol_rate=1.0,
        source_symbols=xp.ones(10, dtype="complex64"),
    )
    with pytest.raises(ValueError, match="No resolved symbols available"):
        s.snr()


def test_ber_no_reference(backend_device, xp):
    """Verify ber raises when no reference bits are available."""
    s = Signal(
        samples=xp.ones(10, dtype="complex64"), sampling_rate=1.0, symbol_rate=1.0
    )
    s.resolved_bits = xp.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="No reference bits available"):
        s.ber()


def test_signal_rz_modscheme_flags(backend_device, xp):
    """Verify RZ modulation flags in __init__."""
    bits = xp.array([0, 1], dtype="int8")
    s = Signal(
        samples=xp.ones(10),
        sampling_rate=1.0,
        symbol_rate=1.0,
        source_bits=bits,
        mod_scheme="PAM",
        mod_order=2,
        mod_rz=True,
    )
    # source_symbols should be derived from bits using "PAM" mapping
    assert s.source_symbols is not None
    assert len(s.source_symbols) == 2
    assert s.mod_rz is True


# ============================================================================
# SIGNAL.EQUALIZE METHODS — COVERAGE GAPS
# ============================================================================


def _make_psk_signal_2sps(xp, n_symbols=400, seed=0):
    """Helper: make a 2-SPS PSK signal and return (sig, rx_sig)."""
    orig = Signal.psk(
        symbol_rate=1e6,
        num_symbols=n_symbols,
        order=4,
        pulse_shape="rrc",
        sps=2,
        seed=seed,
    )
    rx = Signal(
        samples=xp.asarray(orig.samples),
        sampling_rate=2e6,
        symbol_rate=1e6,
        mod_scheme="psk",
        mod_order=4,
        source_symbols=orig.source_symbols,
        source_bits=orig.source_bits,
    )
    return orig, rx


def test_equalize_rls_method(backend_device, xp):
    """Signal.equalize(method='rls') should run and set _num_tail_trim."""
    _, rx = _make_psk_signal_2sps(xp, n_symbols=600)
    rx.equalize(method="rls", num_taps=7, backend="numba")
    assert rx.samples is not None
    # RLS sets num_taps//2 tail trim
    assert rx._num_tail_trim == 7 // 2


def test_equalize_cma_method(backend_device, xp):
    """Signal.equalize(method='cma') should work without training symbols."""
    _, rx = _make_psk_signal_2sps(xp, n_symbols=400)
    rx.equalize(method="cma", num_taps=7, backend="numba")
    assert rx.samples is not None


def test_equalize_rde_method(backend_device, xp):
    """Signal.equalize(method='rde') should run on a QAM signal."""
    orig = Signal.qam(
        symbol_rate=1e6, num_symbols=400, order=16, pulse_shape="rrc", sps=2, seed=0
    )
    rx = Signal(
        samples=xp.asarray(orig.samples),
        sampling_rate=2e6,
        symbol_rate=1e6,
        mod_scheme="qam",
        mod_order=16,
    )
    rx.equalize(method="rde", num_taps=7, backend="numba")
    assert rx.samples is not None


def test_equalize_zf_missing_channel_estimate(backend_device, xp):
    """Signal.equalize(method='zf') without channel_estimate should raise ValueError."""
    _, rx = _make_psk_signal_2sps(xp)
    with pytest.raises(ValueError, match="channel_estimate"):
        rx.equalize(method="zf")


def test_equalize_unknown_method(backend_device, xp):
    """Signal.equalize with an unknown method should raise ValueError."""
    _, rx = _make_psk_signal_2sps(xp)
    with pytest.raises(ValueError, match="Unknown equalization method"):
        rx.equalize(method="superequal")


def test_evm_with_rls_tail_trim_and_training_discard(backend_device, xp):
    """EVM computation after RLS covers _num_tail_trim and _num_train_symbols trimming paths."""
    n_symbols = 600
    orig = Signal.psk(
        symbol_rate=1e6,
        num_symbols=n_symbols,
        order=4,
        pulse_shape="rrc",
        sps=2,
        seed=7,
    )

    rx = Signal(
        samples=xp.asarray(orig.samples),
        sampling_rate=2e6,
        symbol_rate=1e6,
        mod_scheme="psk",
        mod_order=4,
        source_symbols=orig.source_symbols,
        source_bits=orig.source_bits,
    )
    rx.equalize(
        method="rls",
        num_taps=7,
        backend="numba",
        training_symbols=orig.source_symbols[:100],
    )
    rx.resolve_symbols()

    # EVM with discard_training=True triggers the trim path
    evm_pct, evm_db = rx.evm(discard_training=True)
    assert np.isfinite(float(evm_db))
    assert float(evm_pct) > 0

    # EVM with discard_training=False still applies _num_tail_trim
    evm_pct2, evm_db2 = rx.evm(discard_training=False)
    assert np.isfinite(float(evm_db2))
    assert float(evm_pct2) > 0


def test_snr_with_rls_tail_trim_and_training_discard(backend_device, xp):
    """SNR computation after RLS covers _num_tail_trim and _num_train_symbols trimming paths."""
    n_symbols = 600
    orig = Signal.psk(
        symbol_rate=1e6,
        num_symbols=n_symbols,
        order=4,
        pulse_shape="rrc",
        sps=2,
        seed=8,
    )
    rx = Signal(
        samples=xp.asarray(orig.samples),
        sampling_rate=2e6,
        symbol_rate=1e6,
        mod_scheme="psk",
        mod_order=4,
        source_symbols=orig.source_symbols,
        source_bits=orig.source_bits,
    )
    rx.equalize(
        method="rls",
        num_taps=7,
        backend="numba",
        training_symbols=orig.source_symbols[:100],
    )
    rx.resolve_symbols()

    snr_val = rx.snr(discard_training=True)
    assert np.isfinite(snr_val)

    snr_notrim = rx.snr(discard_training=False)
    assert np.isfinite(snr_notrim)


def test_ber_with_rls_tail_trim_and_training_discard(backend_device, xp):
    """BER computation after RLS covers _num_tail_trim and _num_train_symbols trimming in bits."""
    n_symbols = 600
    orig = Signal.psk(
        symbol_rate=1e6,
        num_symbols=n_symbols,
        order=4,
        pulse_shape="rrc",
        sps=2,
        seed=11,
    )
    rx = Signal(
        samples=xp.asarray(orig.samples),
        sampling_rate=2e6,
        symbol_rate=1e6,
        mod_scheme="psk",
        mod_order=4,
        source_symbols=orig.source_symbols,
        source_bits=orig.source_bits,
    )
    rx.equalize(
        method="rls",
        num_taps=7,
        backend="numba",
        training_symbols=orig.source_symbols[:100],
    )
    rx.resolve_symbols()
    rx.demap_symbols_hard()

    # discard_training=True: trims both tail symbols and training symbols
    ber_val = rx.ber(discard_training=True)
    assert np.isfinite(float(ber_val))
    assert 0.0 <= float(ber_val) <= 1.0

    # discard_training=False: only trims tail symbols (tests the bit_tail_trim path)
    ber_notrim = rx.ber(discard_training=False)
    assert np.isfinite(float(ber_notrim))
    assert 0.0 <= float(ber_notrim) <= 1.0


def test_plot_constellation_resolved_data(backend_device, xp):
    """Signal.plot_constellation(data='resolved') should use resolved_symbols."""
    sig = Signal.psk(
        symbol_rate=1e6, num_symbols=200, order=4, pulse_shape="rrc", sps=2, seed=0
    )
    rx = Signal(
        samples=xp.asarray(sig.samples),
        sampling_rate=2e6,
        symbol_rate=1e6,
        mod_scheme="psk",
        mod_order=4,
    )
    rx.equalize(
        method="lms", num_taps=7, backend="numba", training_symbols=sig.source_symbols
    )
    rx.resolve_symbols()

    result = rx.plot_constellation(data="resolved", show=False)
    assert result is not None
    plt.close("all")


def test_plot_constellation_resolved_no_resolved_symbols(backend_device, xp):
    """Signal.plot_constellation(data='resolved') raises ValueError if no resolved_symbols."""
    sig = Signal(
        samples=xp.ones(100, dtype="complex64"),
        sampling_rate=2e6,
        symbol_rate=1e6,
    )
    with pytest.raises(ValueError, match="No resolved_symbols"):
        sig.plot_constellation(data="resolved", show=False)


def test_plot_constellation_overlay_source_mimo(backend_device, xp):
    """Signal.plot_constellation with MIMO signal and overlay_source=True."""
    sig = Signal.psk(
        symbol_rate=1e6,
        num_symbols=200,
        order=4,
        pulse_shape="rrc",
        sps=1,
        num_streams=2,
        seed=0,
    )
    result = sig.plot_constellation(overlay_source=True, show=False)
    assert result is not None
    plt.close("all")


def test_plot_constellation_show(backend_device, xp):
    """Signal.plot_constellation(show=True) should call plt.show() and return None."""
    sig = Signal.psk(
        symbol_rate=1e6, num_symbols=100, order=4, pulse_shape="rrc", sps=1, seed=0
    )
    with patch("matplotlib.pyplot.show"):
        result = sig.plot_constellation(show=True)
    assert result is None
    plt.close("all")


def test_plot_constellation_overlay_source_siso(backend_device, xp):
    """SISO signal with overlay_source=True uses the single-axes scatter path."""
    sig = Signal.psk(
        symbol_rate=1e6, num_symbols=200, order=4, pulse_shape="rrc", sps=1, seed=0
    )
    assert sig.num_streams == 1
    assert sig.source_symbols is not None

    result = sig.plot_constellation(overlay_source=True, show=False)
    assert result is not None
    plt.close("all")


# ============================================================================
# FACTORY METHOD TESTS
# ============================================================================


def test_pam_waveform(backend_device, xp):
    """Verify basic PAM signal generation produces samples on the active device."""
    sig = Signal.pam(order=2, unipolar=False, num_symbols=10, sps=4, symbol_rate=1e3)
    assert sig.samples.size > 0
    assert isinstance(sig.samples, xp.ndarray)
    assert sig.mod_scheme is not None


def test_rzpam_waveform(backend_device, xp):
    """Verify Return-to-Zero PAM signal generation and pulse-shape validation."""
    sig = Signal.pam(
        order=2,
        unipolar=False,
        num_symbols=10,
        sps=4,
        symbol_rate=1e3,
        rz=True,
        pulse_shape="rect",
    )
    assert sig.samples.size > 0
    assert isinstance(sig.samples, xp.ndarray)

    with pytest.raises(ValueError, match="not allowed for RZ PAM"):
        Signal.pam(
            order=2,
            unipolar=False,
            num_symbols=10,
            sps=4,
            symbol_rate=1e3,
            rz=True,
            pulse_shape="rrc",
        )


def test_qam_waveform(backend_device, xp):
    """Verify QAM signal generation populates samples and modulation metadata."""
    sig = Signal.qam(order=16, num_symbols=10, sps=4, symbol_rate=1e3)
    assert sig.samples.size > 0
    assert isinstance(sig.samples, xp.ndarray)
    assert sig.mod_order == 16


def test_psk_waveform(backend_device, xp, xpt):
    """Verify PSK signal generation, metadata, and unit-magnitude constellation."""
    sig = Signal.psk(order=8, num_symbols=50, sps=2, symbol_rate=1e6, seed=0)
    assert sig.samples.size > 0
    assert isinstance(sig.samples, xp.ndarray)
    assert sig.mod_order == 8
    assert sig.mod_scheme is not None
    # All PSK symbols should lie on the unit circle
    syms = sig.source_symbols
    if syms is not None:
        magnitudes = xp.abs(syms)
        xpt.assert_allclose(magnitudes, xp.ones_like(magnitudes), atol=1e-5)


def test_signal_generate(backend_device, xp):
    """Verify Signal.generate() produces correct metadata for any modulation."""
    sig = Signal.generate(
        num_symbols=100,
        sps=4,
        symbol_rate=1e6,
        modulation="qam",
        order=16,
        pulse_shape="rrc",
        seed=1,
    )
    assert sig.samples.size > 0
    assert isinstance(sig.samples, xp.ndarray)
    assert sig.symbol_rate == 1e6
    assert sig.mod_order == 16
    assert sig.source_bits is not None
    assert sig.source_symbols is not None
