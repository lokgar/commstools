#!/usr/bin/env python3
"""
Benchmark: Adaptive equalizers on PMD-impaired dual-pol QAM signals.

Generates a dual-polarization 16-QAM signal, applies static PMD and AWGN,
then runs CMA, LMS, and RLS equalizers with timing and quality metrics.

Timing notes
------------
JAX operations are dispatched asynchronously. The adaptive equalizers in
commstools convert JAX outputs back to NumPy via ``from_jax()`` →
``np.asarray()``, which blocks until the computation completes. This means
wall-clock timing around the public API is already correct without explicit
``block_until_ready()`` calls.

Usage
-----
    uv run python _benchmarks/bench_equalizers.py
"""

import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from commstools import Signal
from commstools.backend import is_cupy_available, to_device, use_cpu_only
from commstools.equalizers import cma, lms, rls
from commstools.impairments import add_awgn, apply_pmd
from commstools.mapping import gray_constellation
from commstools.metrics import evm
from commstools.plotting import apply_default_theme, constellation, equalizer_result

# ── Configuration ──────────────────────────────────────────────────────────
NUM_SYMBOLS = 2**12
SPS = 2  # T/2-spaced (required by adaptive EQs)
SYMBOL_RATE = 2e9  # 28 GBaud (coherent optical)
ORDER = 4  # 16-QAM
ESN0_DB = 20  # Es/N0 in dB
NUM_TAPS = 11  # equalizer filter length
DGD_SYMBOLS = 0.0  # DGD as fraction of symbol period
THETA = np.pi / 5  # polarization rotation angle

N_WARMUP = 1  # JIT warmup iterations
N_BENCH = 5  # timed iterations


def _make_impaired_signal(device="cpu"):
    """Generate dual-pol 16-QAM, apply PMD + AWGN.

    Parameters
    ----------
    device : {"cpu", "gpu"}
        Target device for the output arrays.

    Returns
    -------
    noisy : ndarray (2, N_samples)
    training_syms : ndarray (2, N_sym)
    ref_const : ndarray (M,) complex64
    """
    # Force CPU mode during generation so arrays start on CPU
    use_cpu_only(True)

    sig = Signal.qam(
        num_symbols=NUM_SYMBOLS,
        sps=SPS,
        symbol_rate=SYMBOL_RATE,
        order=ORDER,
        pulse_shape="none",  # no pulse shaping — direct 2 SPS upsample
        num_streams=2,
        seed=42,
    )

    clean = sig.samples  # (2, N_samples)
    training_syms = sig.source_symbols  # (2, N_sym)

    # Apply PMD
    dgd_sec = DGD_SYMBOLS / SYMBOL_RATE  # convert to seconds
    distorted = apply_pmd(
        clean, dgd=dgd_sec, theta=THETA, sampling_rate=sig.sampling_rate
    )

    # Apply AWGN
    noisy = add_awgn(distorted, esn0_db=ESN0_DB, sps=SPS)

    ref_const = gray_constellation("qam", ORDER).astype(np.complex64)

    # Restore default and move to GPU if requested
    use_cpu_only(False)
    if device == "gpu":
        noisy = to_device(noisy, "gpu")
        training_syms = to_device(training_syms, "gpu")
        ref_const = to_device(ref_const, "gpu")

    return noisy, training_syms, ref_const


def _run_equalizers(noisy, training_syms, ref_const):
    """Run all three equalizers, return dict of {name: (result, times)}."""
    configs = {
        "CMA (blind)": {
            "fn": cma,
            "kwargs": dict(
                samples=noisy,
                num_taps=NUM_TAPS,
                step_size=1e-4,
                modulation="qam",
                order=ORDER,
                sps=SPS,
                normalize=False,
            ),
        },
        "LMS (DA)": {
            "fn": lms,
            "kwargs": dict(
                samples=noisy,
                training_symbols=training_syms,
                num_taps=NUM_TAPS,
                step_size=1e-4,
                reference_constellation=ref_const,
                sps=SPS,
                normalize=False,
            ),
        },
        "RLS (DA)": {
            "fn": rls,
            "kwargs": dict(
                samples=noisy,
                training_symbols=training_syms,
                num_taps=NUM_TAPS,
                forgetting_factor=0.99,
                reference_constellation=ref_const,
                sps=SPS,
            ),
        },
    }

    results = {}
    for name, cfg in configs.items():
        fn, kw = cfg["fn"], cfg["kwargs"]

        # Warmup (JIT compilation happens here)
        for _ in range(N_WARMUP):
            fn(**kw)

        # Timed runs
        times = []
        for _ in range(N_BENCH):
            t0 = time.perf_counter()
            result = fn(**kw)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        results[name] = (result, times)

    return results


def _to_numpy(arr):
    """Safely convert any array (NumPy/CuPy) to NumPy."""
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def _compute_metrics(results, ref_const):
    """Compute EVM for each equalizer output using nearest-constellation mapping."""
    from commstools.mapping import demap_symbols_hard, map_bits

    metrics = {}
    ref_np = _to_numpy(ref_const).flatten()
    for name, (result, _) in results.items():
        y = _to_numpy(result.y_hat)
        # Demap → remap to get ideal reference symbols with matching shape
        bits = demap_symbols_hard(y, "qam", ORDER)
        tx_ideal = map_bits(bits.flatten(), "qam", ORDER)
        if y.ndim == 2:
            tx_ideal = tx_ideal.reshape(y.shape[0], -1)
        # Trim to same length (equalizer may output fewer symbols)
        n = min(y.shape[-1], tx_ideal.shape[-1])
        y_trim = y[..., :n]
        tx_trim = tx_ideal[..., :n]
        evm_pct, evm_db = evm(y_trim, tx_trim)
        if hasattr(evm_pct, "__len__"):
            evm_pct = np.mean(evm_pct)
            evm_db = np.mean(evm_db)
        metrics[name] = {"evm_pct": float(evm_pct), "evm_db": float(evm_db)}
    return metrics


def _print_summary(label, results, metrics):
    """Print timing + quality table."""
    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"{'=' * 65}")
    print(
        f"  {'Equalizer':15s} {'Mean (ms)':>10s} {'Std (ms)':>10s} "
        f"{'Min (ms)':>10s} {'EVM (dB)':>10s}"
    )
    print(f"  {'-' * 60}")
    for name, (_, times) in results.items():
        t_arr = np.array(times) * 1000
        m = metrics[name]
        print(
            f"  {name:15s} {t_arr.mean():10.2f} {t_arr.std():10.2f} "
            f"{t_arr.min():10.2f} {m['evm_db']:10.1f}"
        )
    print()


def _plot_results(cpu_results, gpu_results, cpu_metrics, gpu_metrics, noisy_cpu):
    """Generate comparison plot."""
    # matplotlib.use("Agg")
    apply_default_theme()

    n_eq = len(cpu_results)
    fig, axes = plt.subplots(3, n_eq + 1, figsize=(5 * (n_eq + 1), 12))

    # Row 0: Constellations — input + each equalizer output
    constellation(_to_numpy(noisy_cpu[0]), ax=axes[0, 0], title="Input (X-pol)")

    for i, (name, (result, _)) in enumerate(cpu_results.items()):
        y = _to_numpy(result.y_hat)
        ch_data = y[0] if y.ndim == 2 else y
        constellation(ch_data, ax=axes[0, i + 1], title=f"{name} (X-pol)")

    # Row 1: Convergence curves
    axes[1, 0].set_visible(False)
    for i, (name, (result, _)) in enumerate(cpu_results.items()):
        err = _to_numpy(result.error)
        if err.ndim == 2:
            for ch in range(err.shape[0]):
                mse = np.abs(err[ch]) ** 2
                kernel = np.ones(50) / 50
                mse_s = np.convolve(mse, kernel, mode="valid")
                mse_db = 10 * np.log10(mse_s + 1e-30)
                axes[1, i + 1].plot(mse_db, linewidth=0.8, label=f"ch {ch}")
            axes[1, i + 1].legend(fontsize=7)
        else:
            mse = np.abs(err) ** 2
            kernel = np.ones(50) / 50
            mse_s = np.convolve(mse, kernel, mode="valid")
            mse_db = 10 * np.log10(mse_s + 1e-30)
            axes[1, i + 1].plot(mse_db, linewidth=0.8)
        axes[1, i + 1].set_title(f"{name} Convergence")
        axes[1, i + 1].set_xlabel("Symbol Index")
        axes[1, i + 1].set_ylabel("MSE (dB)")

    # Row 2: Timing comparison bar chart
    for a in axes[2, 1:]:
        a.set_visible(False)
    ax_bar = axes[2, 0]

    names = list(cpu_results.keys())
    cpu_means = [np.mean(cpu_results[n][1]) * 1000 for n in names]

    x = np.arange(len(names))
    width = 0.35

    if gpu_results:
        gpu_means = [np.mean(gpu_results[n][1]) * 1000 for n in names]
        ax_bar.barh(x + width / 2, cpu_means, width, label="CPU", color="#4c72b0")
        ax_bar.barh(x - width / 2, gpu_means, width, label="GPU", color="#dd8452")
        ax_bar.legend()
    else:
        ax_bar.barh(x, cpu_means, width, label="CPU", color="#4c72b0")

    ax_bar.set_yticks(x)
    ax_bar.set_yticklabels(names)
    ax_bar.set_xlabel("Time (ms)")
    ax_bar.set_title("Equalizer Execution Time")
    ax_bar.invert_yaxis()

    fig.suptitle(
        f"Equalizer Benchmark — {ORDER}-QAM, {NUM_SYMBOLS} sym, 2-pol MIMO\n"
        f"PMD: DGD={DGD_SYMBOLS} Tsym, θ={THETA:.2f} rad | "
        f"SNR: {ESN0_DB} dB | Taps: {NUM_TAPS}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()
    # plt.savefig("_benchmarks/equalizer_benchmark.png", dpi=150)
    # print("Plot saved to _benchmarks/equalizer_benchmark.png")


def main():
    print("=" * 65)
    print(f"  Equalizer Benchmark")
    print(f"  {ORDER}-QAM | {NUM_SYMBOLS} symbols | 2-pol MIMO | {NUM_TAPS} taps")
    print(f"  PMD: DGD={DGD_SYMBOLS} Tsym, theta={THETA:.2f} rad")
    print(f"  SNR: {ESN0_DB} dB | Warmup: {N_WARMUP} | Bench: {N_BENCH}")
    print("=" * 65)

    # ── CPU benchmark ──────────────────────────────────────────────
    print("\nGenerating CPU signal...")
    noisy_cpu, train_cpu, const_cpu = _make_impaired_signal(device="cpu")

    print("Running CPU equalizers...")
    cpu_results = _run_equalizers(noisy_cpu, train_cpu, const_cpu)
    cpu_metrics = _compute_metrics(cpu_results, const_cpu)
    _print_summary("CPU Results (NumPy → JAX CPU)", cpu_results, cpu_metrics)

    # ── GPU benchmark (if available) ───────────────────────────────
    gpu_results = None
    gpu_metrics = None

    if is_cupy_available():
        print("\nCuPy available — running GPU benchmark...")
        noisy_gpu, train_gpu, const_gpu = _make_impaired_signal(device="gpu")
        gpu_results = _run_equalizers(noisy_gpu, train_gpu, const_gpu)
        gpu_metrics = _compute_metrics(gpu_results, const_gpu)
        _print_summary("GPU Results (CuPy → JAX GPU)", gpu_results, gpu_metrics)
    else:
        print("\nCuPy not available — skipping GPU benchmark.")

    # ── Plot ───────────────────────────────────────────────────────
    _plot_results(cpu_results, gpu_results, cpu_metrics, gpu_metrics, noisy_cpu)


if __name__ == "__main__":
    main()
