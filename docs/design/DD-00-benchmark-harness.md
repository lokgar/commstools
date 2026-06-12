# DD-00 — Profiling & Benchmark Harness

**Phase**: 0 (prerequisite for all other DDs)
**Depends on**: nothing
**Estimated effort**: ~2-3 days
**Code touched**: new `benchmarks/` directory, `pyproject.toml` (dev extra), no library changes

---

## 1. Motivation

Every other design doc (DD-01…DD-05) makes a quantitative promise (e.g. "≥10× wall time on 128-cross BPS", "zero D2H in the block loop"). The original analysis contained at least two wrong quantitative claims (see review report §1, P1/P3), which is exactly what happens without a harness. This doc creates the measurement infrastructure so that **every subsequent PR merges against checked-in baseline numbers**, and reviewers can re-run the comparison mechanically.

Nothing like this exists today: there is no `benchmarks/` directory and no `pytest-benchmark` dependency.

## 2. Design

### 2.1 Layout

```text
benchmarks/
├── conftest.py          # device fixtures (reuse tests/conftest.py patterns), workload factory
├── workloads.py         # canonical signal generators (fixed seeds)
├── bench_bps.py         # recover_carrier_phase_bps (DD-01 gate)
├── bench_block_lms.py   # block_lms with BPS + cycle-slip (DD-02 gate)
├── bench_equalizers.py  # lms/cma/rde/rls, numba vs jax vs (later) cuda (DD-03/DD-04 gates)
├── bench_sync_misc.py   # timing/frequency/recovery host-sync functions (DD-05 gate)
└── baselines/           # checked-in pytest-benchmark JSON per reference machine
    └── <gpu-name>/...
```

- `benchmarks/` lives at repo root, **outside `tests/`**, excluded from the default `pytest` collection (configure `testpaths = ["tests"]` in `pyproject.toml` if not already set; run benchmarks explicitly via `uv run pytest benchmarks/ --benchmark-only`).
- Dependency: `uv add --dev pytest-benchmark`.

### 2.2 Timing methodology

- **Wall time**: `pytest-benchmark` with its default calibration. This is what acceptance criteria quote.
- **Device time** (diagnostic): pairs of `cp.cuda.Event` with `record()`/`synchronize()`/`get_elapsed_time()`, wrapped in a small helper in `benchmarks/conftest.py`. Mandatory **warmup call before timing** so NVRTC/JIT/Numba compilation and CuPy memory-pool growth are excluded.
- **JAX**: call `.block_until_ready()` on outputs inside the timed callable; run one untimed warmup to exclude XLA compilation.
- **NVTX annotation**: helper context manager using `cp.cuda.nvtx.RangePush/RangePop` around DSP stages, so `nsys profile uv run pytest benchmarks/bench_block_lms.py --benchmark-only` produces a readable timeline. DD-02 and DD-05 acceptance criteria use nsys to count D2H operations — annotate accordingly.

### 2.3 Workload matrix

All generators in `workloads.py`, **fixed seeds**, parameters as ID strings so baselines are comparable across runs:

| Axis | Values |
| --- | --- |
| Modulation | QPSK, 16-QAM, 64-QAM (square); **32-cross, 128-cross**; 256-pt geometric (arbitrary) — the non-square ones exercise the TABLE paths |
| N symbols | 1e5, 1e6 |
| Channels C | 1, 2 |
| sps (equalizer benches) | 2 |
| Impairments | AWGN (Es/N0 per workload), phase noise for CPR benches, PMD-mixed channel for MIMO equalizer benches |

Each DD names **one canonical workload ID** in its acceptance criteria (e.g. DD-01: `bps/128cross/N1e6/C2`). Signal generation reuses the library's own mapping/impairments functions with `seed=` set.

### 2.4 Baselines & comparison

- Capture: `uv run pytest benchmarks/ --benchmark-only --benchmark-save=<label>`; commit the JSON under `benchmarks/baselines/<gpu-name>/`.
- Compare: `--benchmark-compare=<label> --benchmark-compare-fail=median:10%` in review.
- Baselines are **per reference GPU** (file under a directory named after `cp.cuda.runtime.getDeviceProperties(0)["name"]`). CPU baselines analogous under `cpu/`.

### 2.5 Correctness companions & tolerance policy

Every benchmark has a **paired correctness test** (may live in `tests/`) comparing the new path against the existing `xp` path on the same fixed-seed workload. Policy (binding for DD-01/02/03/04):

- Fused kernels change summation order ⇒ **never assert bit-exactness**.
- Continuous metrics: `xpt.assert_allclose(rtol=1e-5)` (float32 paths) / `rtol=1e-6` (float64 phase paths, e.g. DD-02).
- Discrete decisions (argmin phase index, slicer decision): **≥ 99.9 % agreement**, ties allowed to differ.
- Equalizer convergence (DD-03/04): statistical — steady-state EVM/MSE within stated dB bounds, not sample-exact trajectories.
- Use the `xpt` helper assertions and `backend_device`/`xp` fixtures per the project's testing conventions.

## 3. Acceptance criteria

1. `uv run pytest benchmarks/ --benchmark-only --device=cpu` and `--device=gpu` both run green on a CUDA machine; CPU-only machines skip GPU benches cleanly (same skip mechanism as `tests/conftest.py`).
2. Baseline JSONs captured on current `main` for every workload referenced by DD-01…DD-05 acceptance criteria, committed under `benchmarks/baselines/`.
3. The NVTX helper demonstrably produces named ranges in an `nsys` capture of `bench_block_lms.py`.
4. Default `uv run pytest` (CI) does **not** collect `benchmarks/`.

## 4. Out of scope

- No changes to library code.
- No CI-enforced performance gates (benchmarks are run/compared manually or in a dedicated job; flaky-perf CI is explicitly avoided).
