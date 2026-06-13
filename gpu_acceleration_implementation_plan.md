# GPU Acceleration — Implementation Plan

**Basis**: [gpu_acceleration_review.md](gpu_acceleration_review.md) and the design docs in [docs/design/](docs/design/).
**Process**: points are implemented strictly one at a time; after each point lands (code + tests + benchmark numbers), work **stops for user approval** before the next point starts.
**Reference machine**: RTX 3070 (CC 8.6, 8 GB, GeForce — FP64 1:64), CuPy 14.1.1, JAX 0.10.1 (CUDA), Python via `uv`.

## Point sequence

| # | Point | Design doc | Scope | Verification gate |
|---|---|---|---|---|
| 1 | Benchmark harness | [DD-00](docs/design/DD-00-benchmark-harness.md) | `benchmarks/` dir, `pytest-benchmark` dev dep, workload generators, bench files for BPS / block_lms / equalizers / sync-misc, CUDA-event + NVTX helpers, baseline capture on this machine (CPU + RTX 3070) | Harness runs green on `--device=cpu` and `--device=gpu`; baselines saved; default `uv run pytest` unaffected |
| 2 | Host-sync hygiene | [DD-05](docs/design/DD-05-host-sync-hygiene.md) | Items 1-12, one commit each (vectorized gathers, batched D2H, logging gates, on-device padding, docstring note) | Full test suite green on both devices; D2H count per function independent of C; bench before/after recorded |
| 3 | `_cuda` infrastructure | [DD-01 Part A](docs/design/DD-01-bps-fused-kernel.md) | `commstools/_cuda/` subpackage: availability probe, lazy `RawModule` compiler with cache, `get_kernel()` fallback contract, package-data wiring, CPU-leg fallback test | `get_kernel` compiles a trivial test kernel on GPU; returns `None` cleanly with CuPy unavailable; wheel ships `.cu` sources |
| 4 | Fused BPS kernel | [DD-01 Part B](docs/design/DD-01-bps-fused-kernel.md) | `bps_min_d2.cu` (TABLE first, then GRID + RETURN_ARGMIN), call-site integration in `recovery.py` + `block_lms` inline BPS + DD slicer | rtol ≤ 1e-5 metric match, ≥ 99.9 % argmin agreement, ≥ 10× on `bps/128cross/N1e6/C2` (GPU), fallback tests green |
| 5 | `block_lms` device unwrap carry | [DD-02 step 1](docs/design/DD-02-block-lms-gpu-cpr-state.md) | `bps_prev4`/`bps_offset4` device-resident in the loop; CPRState CPU contract preserved at entry/exit | `phase_trajectory`/`y_hat` rtol 1e-6 vs main; warm-start round-trip green; per-block D2H/H2D for the carry gone |
| 6 | GPU cycle-slip kernel | [DD-02 step 2](docs/design/DD-02-block-lms-gpu-cpr-state.md) | `cs_block.cu` (one block, C threads, faithful OLS-regression port), persistent device state buffers, Numba fallback kept | Zero D2H inside block loop (nsys / spy); rtol 1e-6; ≥ 2× wall on `block_lms/bps+cs` GPU workload |
| 7 | CUDA Graph capture *(optional)* | [DD-02 step 3](docs/design/DD-02-block-lms-gpu-cpr-state.md) | Workspace pre-allocation refactor of the block-loop body, capture/replay, eager final partial block | ≥ 5× cumulative on the DD-02 workload; decide go/no-go at this checkpoint based on Point 6 profiling |
| 8 | JAX inline-BPS incremental sum | [DD-04 §3.4](docs/design/DD-04-block-update-equalizers.md) | Scan-carry per-slot distance buffer + running metric in `_get_jax_lms_cpr`/`_get_jax_rls_cpr`; warm-start reconstruction | Output matches current JAX impl (float64 reduction tolerance) and Numba reference within existing cross-backend tolerances |
| 9 | Block-update equalizers | [DD-04 main](docs/design/DD-04-block-update-equalizers.md) | `update_mode='block'`/`block_len` for `lms`/`cma` (JAX chunked scan + CuPy einsum), pilot-aided masking, `error_fn='cma'\|'rde'` for `block_lms` | EVM floor within 0.2 dB of per-symbol reference; ≥ 10× vs JAX per-symbol scan at D=16; blind blockwise CMA converges |
| 10 | CUDA sequential equalizers *(deferred decision)* | [DD-03](docs/design/DD-03-cuda-sequential-equalizers.md) | Warp-cooperative LMS/CMA/RDE (+ pilots), FP64 RLS kernel, optional DD-PLL kernel, `backend='cuda'` | Decide after Points 1-9: on this GeForce box the RLS-kernel case is weak (FP64 1:64); LMS/CMA case rests on the pipeline-residency benchmark |

## Standing rules (apply to every point)

- All commands through `uv run`; deps via `uv add`.
- Tests: `uv run pytest --device=all` must be green before a point is declared done; new tests use `backend_device`/`xp`/`xpt` fixtures.
- Benchmarks: capture before/after with `--benchmark-save` into `benchmarks/baselines/`; quote the numbers in the point report.
- Dtype/precision constraints from CLAUDE.md are binding (complex128 RLS P, float64 accumulators/unwrap, `Precision.HIGHEST`).
- No version bumps or pushes unless explicitly requested; one commit per logical unit (DD-05: one per item).
- Each point ends with a short report: what changed, test results, benchmark deltas, anything that deviated from the design doc — then **wait for approval**.

## Status

- [x] Point 1 — DD-00 benchmark harness
- [x] Point 2 — DD-05 hygiene
- [x] Point 3 — DD-01 Part A infra
- [x] Point 4 — DD-01 Part B BPS kernel
- [x] Point 5 — DD-02 step 1
- [x] Point 6 — DD-02 step 2 (cumulative steps 1-2 speedup on the N1e5 gate workload: 1.48× — see point report; the ≥2× target is expected to be met by Point 7 graph capture)
- [x] Point 7 — DD-02 step 3 (CUDA-graph capture; **GO**: graph==eager bit-exact, step-3 speedup 12-15× on `bps`, 4.6× on `bps+cs` (cs kernel is sequential-compute-bound, not launch-bound, so the graph can't collapse it); GPU now beats CPU at block_size=256 in DD mode. Deviation: capture forbids cuBLAS, so the butterfly einsum became a complex128-accumulated broadcast (per CLAUDE.md dot-product rule) — block_lms is no longer bit-identical to pre-Point-7 `main` (±1 float32 ulp), which flips BPS argmin at the ambiguity boundary; EVM/MSE unchanged, all quality tests pass.)
- [x] Point 8 — DD-04 §3.4 (JAX inline-BPS incremental running sum: O(B·KB·M)→O(B·M)/symbol; **cpu-jax 606→102 ms = 5.9×** at KB=B=64. On gpu-jax the per-symbol scan is launch-bound so the compute cut is invisible (950→965 ms, noise) — GPU throughput is Point 9's job. Also fixed a latent pre-existing bug: JAX-CPR result unpacking used `np.asarray(from_jax(...))`, which raised on `device='gpu'` (from_jax returns CuPy); coerced to host NumPy via `to_device`, so GPU-JAX CPR now runs end-to-end and matches CPU-JAX to 3.4e-7. Tests: 32-QAM Numba/JAX parity + GPU-device regression test; baseline 0010.)
- [x] Point 9 — DD-04 main (time-domain block-update for `lms`/`cma`/`rde` via `update_mode='block'`/`block_len`, `backend='jax'` chunked scan + `backend='xp'` array-native loop, pilots; and blind frequency-domain `block_cma`/`block_rde` over a shared overlap-save FDAF engine. **JAX block LMS 47.8× the per-symbol scan at D=16** (2104→44 ms; `unroll=4` was needed — D=16 was 9.3× without it), 17×/30× at D=32/64; **block_cma 14.6×**. Floor parity with the per-symbol reference confirmed; all-pilots `block_cma`≡`block_lms` to 3.8e-7. Deviations from DD-04: (1) **µ convention** follows `block_lms` (same µ ⇒ same floor, stability ceiling `D`× lower) not §2's `µ/D` — verified empirically; (2) **siblings** `block_cma`/`block_rde` over extracted FDAF primitives instead of `error_fn=` on `block_lms` (user decision — the engine is entangled with CUDA-graph/CPR code); (3) **`rde` included** in `update_mode='block'`; (4) `backend='numba'`+block and `cpr_type`/`store_weights`+block raise. The `backend='xp'` path is launch-bound on GPU at small `D` (use `backend='jax'` for GPU block). Bonus: generic `_unpack_result_jax` device-coercion fix (separate commit). Baseline 0011; 21 new tests, green on CPU+GPU.)
- [ ] Point 10 — DD-03 (go/no-go decision)
