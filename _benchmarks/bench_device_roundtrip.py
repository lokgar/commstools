#!/usr/bin/env python3
"""
Benchmark: Device roundtrip correctness for all JAX-based library functions.

Verifies that data stays on the correct device throughout the full pipeline:
  - CPU path:  NumPy → JAX[cpu] → NumPy
  - GPU path:  CuPy  → JAX[gpu] → CuPy

Tests every JAX-consuming function in the library:
  - compute_llr (mapping.py)     — returns JAX array
  - cma / lms / rls (equalizers) — returns NumPy/CuPy via from_jax()
  - to_jax / from_jax (backend)  — explicit conversions

Usage
-----
    uv run python _benchmarks/bench_device_roundtrip.py
"""

import sys
import traceback

import numpy as np

from commstools import Signal
from commstools.backend import (
    dispatch,
    from_jax,
    get_array_module,
    is_cupy_available,
    is_jax_array,
    to_jax,
    use_cpu_only,
)
from commstools.equalizers import cma, lms, rls
from commstools.impairments import add_awgn, apply_pmd
from commstools.mapping import compute_llr, gray_constellation

# ── Configuration ──────────────────────────────────────────────────────────
NUM_SYMBOLS = 2**14  # small — correctness, not speed
SPS = 2
SYMBOL_RATE = 28e9
ORDER = 16
ESN0_DB = 20
NUM_TAPS = 15

# ── Helpers ────────────────────────────────────────────────────────────────

_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"


def _arr_info(arr):
    """Return (backend_name, device_str) for any array."""
    if is_jax_array(arr):
        platform = arr.device.platform  # "cpu", "cuda", "gpu"
        return "jax", platform
    mod = get_array_module(arr)
    if mod.__name__ == "cupy":
        return "cupy", "gpu"
    return "numpy", "cpu"


def _check(name, arr, expect_backend, expect_device, results):
    """Assert array backend and device match expectations."""
    backend, device = _arr_info(arr)

    # Normalize device strings for comparison
    device_norm = "gpu" if device in ("cuda", "gpu") else "cpu"
    expect_norm = "gpu" if expect_device in ("cuda", "gpu") else "cpu"

    ok = backend == expect_backend and device_norm == expect_norm
    status = _PASS if ok else _FAIL
    actual = f"{backend}[{device}]"
    expected = f"{expect_backend}[{expect_device}]"
    msg = f"  {status}  {name:40s}  got {actual:15s}  expect {expected}"
    print(msg)
    results.append((name, ok))
    return ok


def _make_signal(device="cpu"):
    """Generate dual-pol 16-QAM with PMD + AWGN on the target device."""
    sig = Signal.qam(
        num_symbols=NUM_SYMBOLS,
        sps=SPS,
        symbol_rate=SYMBOL_RATE,
        order=ORDER,
        pulse_shape="none",
        num_streams=2,
        seed=42,
    )
    samples = sig.samples
    training_syms = sig.source_symbols

    # Apply PMD
    dgd_sec = 0.3 / SYMBOL_RATE
    distorted = apply_pmd(
        samples, dgd=dgd_sec, theta=np.pi / 5, sampling_rate=sig.sampling_rate
    )

    # Apply AWGN
    noisy = add_awgn(distorted, esn0_db=ESN0_DB, sps=SPS)

    ref_const = gray_constellation("qam", ORDER).astype(np.complex64)

    if device == "gpu":
        from commstools.backend import to_device

        noisy = to_device(noisy, "gpu")
        training_syms = to_device(training_syms, "gpu")
        ref_const = to_device(ref_const, "gpu")

    return noisy, training_syms, ref_const


# ── Test functions ─────────────────────────────────────────────────────────


def test_to_jax_from_jax(xp, device, results):
    """Test explicit to_jax / from_jax roundtrip."""
    arr = xp.ones((2, 64), dtype=xp.complex64)

    jax_arr = to_jax(arr, dtype=np.complex64)
    _check("to_jax: input type", arr, "cupy" if device == "gpu" else "numpy", device, results)
    _check("to_jax: output is JAX", jax_arr, "jax", device, results)

    back = from_jax(jax_arr)
    expect_back = "cupy" if device == "gpu" else "numpy"
    _check("from_jax: roundtrip type", back, expect_back, device, results)

    # Verify values preserved
    if device == "gpu":
        vals_ok = xp.allclose(back, arr)
    else:
        vals_ok = np.allclose(back, arr)
    status = _PASS if vals_ok else _FAIL
    print(f"  {status}  {'from_jax: values preserved':40s}")
    results.append(("from_jax: values preserved", bool(vals_ok)))


def test_to_jax_dtype_guard(xp, device, results):
    """Test that to_jax enforces requested dtype."""
    arr64 = xp.ones(32, dtype=xp.complex128)
    jax_arr = to_jax(arr64, dtype=np.complex64)

    dtype_ok = str(jax_arr.dtype) == "complex64"
    status = _PASS if dtype_ok else _FAIL
    print(f"  {status}  {'to_jax: dtype guard c128→c64':40s}  got {jax_arr.dtype}")
    results.append(("to_jax: dtype guard c128→c64", dtype_ok))

    arr_f64 = xp.ones(32, dtype=xp.float64)
    jax_f32 = to_jax(arr_f64, dtype="float32")
    dtype_ok2 = str(jax_f32.dtype) == "float32"
    status = _PASS if dtype_ok2 else _FAIL
    print(f"  {status}  {'to_jax: dtype guard f64→f32':40s}  got {jax_f32.dtype}")
    results.append(("to_jax: dtype guard f64→f32", dtype_ok2))


def test_compute_llr(xp, device, results):
    """Test compute_llr returns JAX array on correct device."""
    n_sym = 256
    rng = np.random.RandomState(42)
    syms_np = (rng.randn(n_sym) + 1j * rng.randn(n_sym)).astype(np.complex64)
    syms = xp.asarray(syms_np) if device == "gpu" else syms_np

    noise_var = 10 ** (-ESN0_DB / 10)
    llrs = compute_llr(syms, "qam", ORDER, noise_var, method="maxlog")

    # compute_llr always returns JAX array
    _check("compute_llr: output is JAX", llrs, "jax", device, results)

    # Verify shape: n_sym * log2(ORDER) = 256 * 4 = 1024
    expected_len = n_sym * int(np.log2(ORDER))
    shape_ok = llrs.shape[-1] == expected_len
    status = _PASS if shape_ok else _FAIL
    print(f"  {status}  {'compute_llr: output shape':40s}  {llrs.shape}")
    results.append(("compute_llr: output shape", shape_ok))

    # Roundtrip: from_jax should give back correct backend
    llrs_back = from_jax(llrs)
    expect_back = "cupy" if device == "gpu" else "numpy"
    _check("compute_llr→from_jax: roundtrip", llrs_back, expect_back, device, results)


def test_cma(noisy, ref_const, xp, device, results):
    """Test CMA equalizer returns arrays on correct device."""
    result = cma(
        samples=noisy,
        num_taps=NUM_TAPS,
        step_size=1e-3,
        modulation="qam",
        order=ORDER,
        sps=SPS,
    )

    expect = "cupy" if device == "gpu" else "numpy"
    _check("cma: y_hat type", result.y_hat, expect, device, results)
    _check("cma: error type", result.error, expect, device, results)
    _check("cma: weights type", result.weights, expect, device, results)

    # Sanity: output is finite
    y = result.y_hat
    if hasattr(y, "get"):
        y = y.get()
    finite_ok = np.all(np.isfinite(y))
    status = _PASS if finite_ok else _FAIL
    print(f"  {status}  {'cma: output is finite':40s}")
    results.append(("cma: output is finite", bool(finite_ok)))


def test_lms(noisy, training_syms, ref_const, xp, device, results):
    """Test LMS equalizer returns arrays on correct device."""
    result = lms(
        samples=noisy,
        training_symbols=training_syms,
        num_taps=NUM_TAPS,
        step_size=1e-3,
        reference_constellation=ref_const,
        sps=SPS,
        normalize=True,
    )

    expect = "cupy" if device == "gpu" else "numpy"
    _check("lms: y_hat type", result.y_hat, expect, device, results)
    _check("lms: error type", result.error, expect, device, results)
    _check("lms: weights type", result.weights, expect, device, results)

    y = result.y_hat
    if hasattr(y, "get"):
        y = y.get()
    finite_ok = np.all(np.isfinite(y))
    status = _PASS if finite_ok else _FAIL
    print(f"  {status}  {'lms: output is finite':40s}")
    results.append(("lms: output is finite", bool(finite_ok)))


def test_rls(noisy, training_syms, ref_const, xp, device, results):
    """Test RLS equalizer returns arrays on correct device."""
    result = rls(
        samples=noisy,
        training_symbols=training_syms,
        num_taps=NUM_TAPS,
        forgetting_factor=0.99,
        reference_constellation=ref_const,
        sps=SPS,
    )

    expect = "cupy" if device == "gpu" else "numpy"
    _check("rls: y_hat type", result.y_hat, expect, device, results)
    _check("rls: error type", result.error, expect, device, results)
    _check("rls: weights type", result.weights, expect, device, results)

    y = result.y_hat
    if hasattr(y, "get"):
        y = y.get()
    finite_ok = np.all(np.isfinite(y))
    status = _PASS if finite_ok else _FAIL
    print(f"  {status}  {'rls: output is finite':40s}")
    results.append(("rls: output is finite", bool(finite_ok)))


def test_impairments(xp, device, results):
    """Test that impairments preserve array backend."""
    N = 512
    if device == "gpu":
        rng = xp.random.RandomState(42)
        samples = (rng.randn(2, N) + 1j * rng.randn(2, N)).astype(xp.complex64)
    else:
        rng = np.random.RandomState(42)
        samples = (rng.randn(2, N) + 1j * rng.randn(2, N)).astype(np.complex64)

    expect = "cupy" if device == "gpu" else "numpy"
    _check("impairments: input type", samples, expect, device, results)

    pmd_out = apply_pmd(samples, dgd=5e-12, theta=np.pi / 5, sampling_rate=56e9)
    _check("apply_pmd: output type", pmd_out, expect, device, results)

    awgn_out = add_awgn(pmd_out, esn0_db=20, sps=2)
    _check("add_awgn: output type", awgn_out, expect, device, results)


# ── Main ───────────────────────────────────────────────────────────────────


def run_device_tests(device):
    """Run all roundtrip tests on a single device."""
    results = []

    if device == "gpu":
        import cupy as cp
        xp = cp
        use_cpu_only(False)
    else:
        xp = np
        use_cpu_only(True)

    print(f"\n{'=' * 70}")
    print(f"  Device Roundtrip Tests — {device.upper()}")
    print(f"  Backend: {'CuPy → JAX[gpu] → CuPy' if device == 'gpu' else 'NumPy → JAX[cpu] → NumPy'}")
    print(f"{'=' * 70}")

    # -- Basic backend roundtrip --
    print(f"\n  --- to_jax / from_jax roundtrip ---")
    try:
        test_to_jax_from_jax(xp, device, results)
    except Exception as e:
        print(f"  {_FAIL}  to_jax/from_jax roundtrip: {e}")
        traceback.print_exc()
        results.append(("to_jax/from_jax roundtrip", False))

    # -- Dtype guard --
    print(f"\n  --- to_jax dtype guard ---")
    try:
        test_to_jax_dtype_guard(xp, device, results)
    except Exception as e:
        print(f"  {_FAIL}  to_jax dtype guard: {e}")
        traceback.print_exc()
        results.append(("to_jax dtype guard", False))

    # -- Impairments --
    print(f"\n  --- Impairments (PMD, AWGN) ---")
    try:
        test_impairments(xp, device, results)
    except Exception as e:
        print(f"  {_FAIL}  impairments: {e}")
        traceback.print_exc()
        results.append(("impairments", False))

    # -- compute_llr --
    print(f"\n  --- compute_llr ---")
    try:
        test_compute_llr(xp, device, results)
    except Exception as e:
        print(f"  {_FAIL}  compute_llr: {e}")
        traceback.print_exc()
        results.append(("compute_llr", False))

    # -- Generate signal for equalizer tests --
    print(f"\n  Generating {device.upper()} signal for equalizer tests...")
    use_cpu_only(True)  # generate on CPU first
    noisy, training_syms, ref_const = _make_signal(device="cpu")
    use_cpu_only(False if device == "gpu" else True)

    if device == "gpu":
        from commstools.backend import to_device
        noisy = to_device(noisy, "gpu")
        training_syms = to_device(training_syms, "gpu")
        ref_const = to_device(ref_const, "gpu")

    # -- CMA --
    print(f"\n  --- CMA equalizer ---")
    try:
        test_cma(noisy, ref_const, xp, device, results)
    except Exception as e:
        print(f"  {_FAIL}  CMA: {e}")
        traceback.print_exc()
        results.append(("CMA", False))

    # -- LMS --
    print(f"\n  --- LMS equalizer ---")
    try:
        test_lms(noisy, training_syms, ref_const, xp, device, results)
    except Exception as e:
        print(f"  {_FAIL}  LMS: {e}")
        traceback.print_exc()
        results.append(("LMS", False))

    # -- RLS --
    print(f"\n  --- RLS equalizer ---")
    try:
        test_rls(noisy, training_syms, ref_const, xp, device, results)
    except Exception as e:
        print(f"  {_FAIL}  RLS: {e}")
        traceback.print_exc()
        results.append(("RLS", False))

    # Restore default
    use_cpu_only(False)

    return results


def main():
    print("=" * 70)
    print("  Device Roundtrip Correctness Benchmark")
    print("  Verifies: array backend + device residency through full pipeline")
    print("=" * 70)

    all_results = {}

    # Always run CPU
    all_results["CPU"] = run_device_tests("cpu")

    # Run GPU if available
    if is_cupy_available():
        all_results["GPU"] = run_device_tests("gpu")
    else:
        print("\n  CuPy not available — skipping GPU roundtrip tests.")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")

    total_pass = 0
    total_fail = 0

    for device_label, results in all_results.items():
        n_pass = sum(1 for _, ok in results if ok)
        n_fail = sum(1 for _, ok in results if not ok)
        total_pass += n_pass
        total_fail += n_fail
        status = _PASS if n_fail == 0 else _FAIL
        print(f"  {status}  {device_label}: {n_pass}/{n_pass + n_fail} checks passed")

        if n_fail > 0:
            for name, ok in results:
                if not ok:
                    print(f"         FAILED: {name}")

    print(f"\n  Total: {total_pass}/{total_pass + total_fail} checks passed")
    print()

    sys.exit(1 if total_fail > 0 else 0)


if __name__ == "__main__":
    main()
