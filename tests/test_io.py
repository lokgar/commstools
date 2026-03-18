"""
Tests for commstools.io — save_npz / load_npz round-trips.
"""

import numpy as np
import numpy.testing as npt
import pytest

import commstools
from commstools import Signal, SingleCarrierFrame, Preamble
from commstools.io import load_npz, save_npz

try:
    import cupy as cp

    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False


def _np(arr) -> np.ndarray:
    """Move any array to NumPy for comparison."""
    if _CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _siso_signal() -> Signal:
    return Signal.qam(num_symbols=256, sps=4, symbol_rate=1e9, order=16, seed=0)


def _mimo_signal() -> Signal:
    return Signal.qam(
        num_symbols=128, sps=4, symbol_rate=1e9, order=4, num_streams=2, seed=1
    )


def _frame_signal() -> Signal:
    frame = SingleCarrierFrame(
        payload_len=200,
        payload_mod_scheme="QAM",
        payload_mod_order=16,
        preamble=Preamble(sequence_type="barker", length=13),
        pilot_pattern="comb",
        pilot_period=8,
        pilot_mod_scheme="PSK",
        pilot_mod_order=4,
        guard_type="zero",
        guard_len=4,
    )
    return frame.to_signal(sps=4, symbol_rate=1e9)


# -----------------------------------------------------------------------------
# Basic round-trip
# -----------------------------------------------------------------------------


def test_roundtrip_samples(tmp_path):
    sig = _siso_signal()
    p = tmp_path / "sig.npz"
    save_npz(sig, p)
    sig2 = load_npz(p)
    npt.assert_array_equal(_np(sig.samples), _np(sig2.samples))


def test_roundtrip_scalar_metadata(tmp_path):
    sig = _siso_signal()
    p = tmp_path / "sig.npz"
    save_npz(sig, p)
    sig2 = load_npz(p)

    assert sig2.sampling_rate == sig.sampling_rate
    assert sig2.symbol_rate == sig.symbol_rate
    assert sig2.mod_scheme == sig.mod_scheme
    assert sig2.mod_order == sig.mod_order
    assert sig2.mod_unipolar == sig.mod_unipolar
    assert sig2.pulse_shape == sig.pulse_shape
    assert sig2.rrc_rolloff == sig.rrc_rolloff
    assert sig2.filter_span == sig.filter_span
    assert sig2.spectral_domain == sig.spectral_domain
    assert sig2.physical_domain == sig.physical_domain
    assert sig2.center_frequency == sig.center_frequency
    assert sig2.digital_frequency_offset == sig.digital_frequency_offset


def test_roundtrip_source_bits(tmp_path):
    sig = _siso_signal()
    assert sig.source_bits is not None
    p = tmp_path / "sig.npz"
    save_npz(sig, p)
    sig2 = load_npz(p)
    npt.assert_array_equal(_np(sig.source_bits), _np(sig2.source_bits))


def test_roundtrip_source_symbols(tmp_path):
    sig = _siso_signal()
    assert sig.source_symbols is not None
    p = tmp_path / "sig.npz"
    save_npz(sig, p)
    sig2 = load_npz(p)
    npt.assert_array_almost_equal(_np(sig.source_symbols), _np(sig2.source_symbols))


# -----------------------------------------------------------------------------
# Extension handling
# -----------------------------------------------------------------------------


def test_extension_appended_automatically(tmp_path):
    sig = _siso_signal()
    # Pass path without .npz extension
    p_no_ext = tmp_path / "capture"
    save_npz(sig, p_no_ext)
    assert (tmp_path / "capture.npz").exists()

    sig2 = load_npz(p_no_ext)  # load also accepts without extension
    npt.assert_array_equal(_np(sig.samples), _np(sig2.samples))


# -----------------------------------------------------------------------------
# MIMO
# -----------------------------------------------------------------------------


def test_roundtrip_mimo(tmp_path):
    sig = _mimo_signal()
    assert sig.samples.ndim == 2
    p = tmp_path / "mimo.npz"
    save_npz(sig, p)
    sig2 = load_npz(p)
    assert sig2.samples.shape == sig.samples.shape
    npt.assert_array_equal(_np(sig.samples), _np(sig2.samples))


# -----------------------------------------------------------------------------
# Optional arrays absent
# -----------------------------------------------------------------------------


def test_no_source_arrays_when_none(tmp_path):
    """Signal with no source_bits/source_symbols should load without them."""
    sig = Signal(
        samples=np.random.randn(512) + 1j * np.random.randn(512),
        sampling_rate=1e9,
        symbol_rate=250e6,
    )
    assert sig.source_bits is None
    assert sig.source_symbols is None

    p = tmp_path / "raw.npz"
    save_npz(sig, p)
    sig2 = load_npz(p)
    assert sig2.source_bits is None
    assert sig2.source_symbols is None


# -----------------------------------------------------------------------------
# Cache fields
# -----------------------------------------------------------------------------


def test_include_cache_false_by_default(tmp_path):
    sig = _siso_signal()
    sig.resolve_symbols()  # populate cache
    assert sig.resolved_symbols is not None

    p = tmp_path / "sig.npz"
    save_npz(sig, p)  # include_cache=False by default

    data = np.load(p, allow_pickle=True)
    assert "resolved_symbols" not in data.files
    assert "resolved_bits" not in data.files


def test_include_cache_roundtrip(tmp_path):
    sig = _siso_signal()
    sig.resolve_symbols()
    assert sig.resolved_symbols is not None

    p = tmp_path / "sig_cache.npz"
    save_npz(sig, p, include_cache=True)
    sig2 = load_npz(p)

    npt.assert_array_almost_equal(_np(sig.resolved_symbols), _np(sig2.resolved_symbols))


# -----------------------------------------------------------------------------
# Compressed vs uncompressed
# -----------------------------------------------------------------------------


def test_uncompressed_roundtrip(tmp_path):
    sig = _siso_signal()
    p = tmp_path / "uncompressed.npz"
    save_npz(sig, p, compressed=False)
    sig2 = load_npz(p)
    npt.assert_array_equal(_np(sig.samples), _np(sig2.samples))


def test_compressed_smaller_than_uncompressed(tmp_path):
    sig = _siso_signal()
    p_c = tmp_path / "c.npz"
    p_u = tmp_path / "u.npz"
    save_npz(sig, p_c, compressed=True)
    save_npz(sig, p_u, compressed=False)
    assert p_c.stat().st_size <= p_u.stat().st_size


# -----------------------------------------------------------------------------
# Frame signal with SignalInfo
# -----------------------------------------------------------------------------


def test_roundtrip_frame_metadata(tmp_path):
    sig = _frame_signal()
    assert sig.signal_type == "Single-Carrier Frame"
    assert sig.frame is not None

    p = tmp_path / "frame.npz"
    save_npz(sig, p)
    sig2 = load_npz(p)

    assert sig2.signal_type == "Single-Carrier Frame"
    assert sig2.frame is not None
    assert sig2.frame.payload_len == sig.frame.payload_len
    assert sig2.frame.payload_mod_scheme == sig.frame.payload_mod_scheme
    assert sig2.frame.payload_mod_order == sig.frame.payload_mod_order
    assert sig2.frame.preamble.length == sig.frame.preamble.length
    assert sig2.frame.preamble.sequence_type == sig.frame.preamble.sequence_type
    assert sig2.frame.pilot_pattern == sig.frame.pilot_pattern
    assert sig2.frame.pilot_period == sig.frame.pilot_period
    assert sig2.frame.guard_type == sig.frame.guard_type
    assert sig2.frame.guard_len == sig.frame.guard_len
    assert sig2.frame.num_streams == sig.frame.num_streams


def test_roundtrip_signal_type_none(tmp_path):
    """Signal without signal_type should load with signal_type=None."""
    sig = _siso_signal()
    assert sig.signal_type is None
    p = tmp_path / "plain.npz"
    save_npz(sig, p)
    sig2 = load_npz(p)
    assert sig2.signal_type is None


# -----------------------------------------------------------------------------
# ZC preamble kwargs round-trip
# -----------------------------------------------------------------------------


def test_roundtrip_zc_preamble_kwargs(tmp_path):
    frame = SingleCarrierFrame(
        payload_len=100,
        preamble=Preamble(sequence_type="zc", length=31, root=7),
    )
    sig = frame.to_signal(sps=4, symbol_rate=1e9)
    assert sig.frame.preamble.root == 7

    p = tmp_path / "zc.npz"
    save_npz(sig, p)
    sig2 = load_npz(p)
    assert sig2.frame.preamble.root == 7


# -----------------------------------------------------------------------------
# Public API export
# -----------------------------------------------------------------------------


def test_exported_from_package():
    assert hasattr(commstools, "save_npz")
    assert hasattr(commstools, "load_npz")


# -----------------------------------------------------------------------------
# GPU round-trip (skipped if CuPy unavailable)
# -----------------------------------------------------------------------------


def test_roundtrip_device_gpu(tmp_path):
    pytest.importorskip("cupy")
    sig = _siso_signal()
    p = tmp_path / "sig_gpu.npz"
    save_npz(sig, p)
    sig_gpu = load_npz(p, device="gpu")
    assert sig_gpu.backend == "GPU"
    npt.assert_array_equal(_np(sig.samples), _np(sig_gpu.samples))


def test_auto_device_uses_gpu_when_available(tmp_path):
    pytest.importorskip("cupy")
    sig = _siso_signal()
    p = tmp_path / "auto.npz"
    save_npz(sig, p)
    sig2 = load_npz(p)  # device="auto"
    assert sig2.backend == "GPU"
