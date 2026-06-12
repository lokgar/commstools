# DD-01 — Fused BPS Distance Kernel + `commstools/_cuda` Infrastructure

**Phase**: 1b
**Depends on**: DD-00 (benchmark gate)
**Estimated effort**: ~1-1.5 weeks
**Code touched**: new `commstools/_cuda/` subpackage; call-site edits in [recovery.py](../../commstools/recovery.py) (`recover_carrier_phase_bps`) and [equalization.py](../../commstools/equalization.py) (`block_lms` inline BPS, DD slicer); `pyproject.toml` (package data)

---

## 1. Motivation

For **non-square constellations** (32/128-cross-QAM, APSK, geometrically-shaped sets), both BPS implementations materialize the full candidate-distance tensor:

- Standalone BPS, [recovery.py:986-989](../../commstools/recovery.py#L986-L989):
  `d_sq = xp.abs(x_rot[:, :, None] - const_xp[None, None, :]) ** 2` → `(CHUNK_N≈1024, B, M)` per chunk per channel.
- `block_lms` inline BPS, [equalization.py:6052-6055](../../commstools/equalization.py#L6052-L6055):
  `(P, C, B, M)` per block.
- `block_lms` DD slicer (related, same family), [equalization.py:6263-6266](../../commstools/equalization.py#L6263-L6266): `(C, B, M)` per block.

Because CuPy does not fuse elementwise chains, each chunk produces ~270 MB of intermediates written + re-read (~0.55 GB traffic per 1024 symbols per channel at B=64, M=256) — **~0.55 TB of DRAM traffic per 10⁶ symbols**, purely memory-bound (see review report §3.1). A fused kernel reads the symbols once and writes only `min_d2`: **~2000× traffic reduction**; the stage becomes compute-bound at ~7-10 ms per 10⁶ symbols on an A100-class part.

**Square QAM already has an O(1) fast path** ([recovery.py:969-985](../../commstools/recovery.py#L969-L985), [equalization.py:6033-6050](../../commstools/equalization.py#L6033-L6050)) and is *not* the motivation — but a `GRID` kernel mode still collapses its ~8 separate elementwise launches into one (secondary win).

This doc also defines the **`commstools/_cuda` infrastructure contract** that DD-02 and DD-03 reuse — implement Part A first.

---

## 2. Part A — `commstools/_cuda` infrastructure (contract for DD-02/DD-03)

### 2.1 Layout

```text
commstools/_cuda/
├── __init__.py      # availability probe + public get_kernel()
├── compiler.py      # lazy RawModule compilation + in-process cache
└── src/
    ├── bps_min_d2.cu        (this DD)
    ├── cs_block.cu          (DD-02)
    └── seq_equalizers.cu    (DD-03)
```

- `.cu` files are **real files** (syntax highlighting, reviewable diffs), loaded via `importlib.resources.files("commstools._cuda").joinpath("src/<name>.cu").read_text()`. Declare `commstools/_cuda/src/*.cu` as package data in `pyproject.toml` (and `MANIFEST.in` if used) — wheels must ship them.
- **No kernel source strings inside `recovery.py`/`equalization.py`** — all CUDA C++ lives in `_cuda/src/`.

### 2.2 Availability probe (`__init__.py`)

`is_available() -> bool`, cached at first call:

1. CuPy importable (reuse `backend.is_cupy_available()`),
2. at least one CUDA device present,
3. compute capability ≥ 7.0.

### 2.3 Compilation (`compiler.py`)

- `cp.RawModule(code=source, options=("-std=c++17", "--use_fast_math"), name_expressions=[...])` with **C++ templates** for mode/dtype specialization (e.g. `bps_min_d2<TABLE>`, `bps_min_d2<GRID>`); `name_expressions` gives mangled-name resolution.
- In-process cache: module-level dict keyed by `(source_name, options, name_expression)`. Cross-process reuse comes free from CuPy's on-disk NVRTC cache (`~/.cupy/kernel_cache`).
- Compilation is **lazy** — first `get_kernel()` call compiles; import of `commstools` must not touch NVRTC.

### 2.4 Public entry point & fallback contract

```python
def get_kernel(name: str, **spec) -> Optional[Callable]:
    """Return a launchable kernel wrapper, or None if unavailable.

    None ⇒ caller MUST fall back to the existing xp path.
    Any compile/load failure logs ONE warning per process and returns None.
    """
```

- Every call site keeps the current `xp` implementation as the fallback branch. **CPU/NumPy and JAX behavior is untouched by construction** — `get_kernel` is only consulted when `xp is cp`.
- Returned wrapper handles grid/block computation and dtype/contiguity validation (`cp.ascontiguousarray` where needed); call sites pass CuPy arrays and plain Python scalars only.

### 2.5 CI

- Kernel tests run only under `--device=gpu` (existing fixture machinery).
- One CPU-leg test asserts: with CuPy absent/unavailable, `get_kernel(...)` returns `None` and the public BPS functions still produce correct results via the fallback.

---

## 3. Part B — `bps_min_d2` kernel

### 3.1 Specification

Computes, in a single pass:

```text
min_d2[p, c, n] = min_m | x[c, n] * phasor[p] − const[m] |²
```

- Inputs: `x` (C, N) `complex64` (contiguous, time last); `phasor` (P,) `complex64`; constellation `const` (M,) `complex64` (TABLE mode) **or** grid params `lev_min, d_grid, side` (GRID mode).
- Output: `min_d2` (P, C, N) `float32`.
- P ≤ 128 (B candidate phases in recovery.py naming), M ≤ 1024.

The same kernel serves both call sites: recovery.py calls it with its (B,) phasor table over each chunk (its `(CHUNK, B)` layout is a transpose of `(P, n)` — see §3.4); `block_lms` calls it per block with `(P,)` phases over `(C, B)` symbols, output directly in its native `(P, C, B)` layout.

### 3.2 Thread mapping & memory plan

- Block: `(128, 1, 1)` threads. Grid: `(ceil(N/128), P, C)`.
- `phasor[p]`: stored in `__constant__` memory (P ≤ 128 ⇒ ≤ 1 KB). Prepared on the host in **float64** (`exp(-1j·φ)` in double, then cast to `complex64`) to match the existing phasor-precision behavior ([recovery.py:936](../../commstools/recovery.py#L936)).
- Constellation (TABLE mode): cooperatively loaded into **shared memory** at block start (M·8 B ≤ 8 KB). The per-thread `for m` loop is warp-uniform ⇒ shared-memory **broadcast**, no bank conflicts.
- Per thread: load `x[c, n]` once into registers (coalesced in `n`), rotate by `phasor[blockIdx.y]`, loop over M tracking a running `float32` minimum in a register, write `min_d2[p, c, n]` (coalesced in `n`).
- Arithmetic: complex sub + norm² in FP32 (`__fmaf` friendly); matches the precision of the current `float32` metric path.

### 3.3 Template modes

- **`TABLE`** (primary): general constellation search as above. Variant flag `RETURN_ARGMIN` additionally writes `int32` indices — reused by the `block_lms` DD slicer ([equalization.py:6263-6266](../../commstools/equalization.py#L6263-L6266)) and available to DD-05 item 7 (LLR/metrics distance matrices).
- **`GRID`** (secondary, implement after TABLE): square-QAM rounding — `clip(round((re−lev_min)/d_grid), 0, side−1)` per component, distance to the snapped point. Replaces the ~8 elementwise launches of the existing fast path with one. Functionally identical to [equalization.py:6033-6050](../../commstools/equalization.py#L6033-L6050).

### 3.4 What is deliberately NOT fused

**Stop at `min_d2`. Do not fuse the windowed sum or the argmin.** The two consumers have *different* reductions over a tensor that is already small (~P·C·B·4 B ≈ 128 KB per `block_lms` block):

- recovery.py: **non-overlapping** block sums — `reshape(n_b, block_size, B).sum(axis=1)` ([recovery.py:993-995](../../commstools/recovery.py#L993-L995));
- `block_lms`: **causal sliding window** with a cross-block history prefix (`bps_d2_hist`) via cumsum ([equalization.py:6057-6071](../../commstools/equalization.py#L6057-L6071)).

The distance stage carries ~99 % of the traffic; fusing it alone converts the stage to compute-bound. A fused sum+argmin kernel is a stretch goal **only if** DD-00 profiling later shows the cumsum/argmin chain matters (review report §7).

### 3.5 Integration points

| Call site | Change |
| --- | --- |
| [recovery.py:986-989](../../commstools/recovery.py#L986-L989) (non-square branch) | `k = get_kernel("bps_min_d2", mode="TABLE")`; if not None, one kernel call per chunk covering **all C channels** (hoist the kernel call above the `for ch` loop if convenient, or call per channel with C=1 — implementer's choice; document which). Output reshaped/transposed to the existing `(CHUNK, B)` consumption. `else`: existing tensor path unchanged. |
| [equalization.py:6052-6055](../../commstools/equalization.py#L6052-L6055) (inline BPS, non-square) | Kernel emits `(P, C, B)` `min_d2` directly, replacing `d2_all` + `xp.min`. |
| [equalization.py:6033-6050](../../commstools/equalization.py#L6033-L6050) and [recovery.py:969-985](../../commstools/recovery.py#L969-L985) (square paths) | Optional `GRID` mode swap-in; keep xp path as fallback. |
| [equalization.py:6263-6266](../../commstools/equalization.py#L6263-L6266) (DD slicer) | `TABLE + RETURN_ARGMIN` variant replaces the (C, B, M) tensor + argmin. |

Dtypes: in `complex64`, out `float32` — identical to today. The `complex128` input path (rare) keeps the xp fallback; do not template FP64 in v1.

## 4. Test & acceptance criteria

Workload (DD-00): `bps/128cross/N1e6/C2`, plus a 256-pt geometric constellation case and one square case for GRID.

1. Correctness: metric `xpt.assert_allclose(rtol≤1e-5)` vs. the existing xp path; **argmin phase-index agreement ≥ 99.9 %** (ties allowed); final `phi_full` trajectories within `rtol=1e-6` after unwrap (float64 domain).
2. Performance: **≥ 10× wall-time** on the canonical workload vs. the DD-00 baseline (expectation is considerably higher; 10× is the merge gate).
3. CPU (`--device=cpu`) and JAX paths byte-identical to `main` (no code path touched).
4. Fallback: with CuPy unavailable (CPU CI leg), public BPS functions still pass their existing tests; exactly one warning logged when GPU present but compile fails (simulate by monkeypatching).
5. Package data: `uv build` wheel contains `_cuda/src/*.cu`; fresh-venv install can compile the kernel.

## 5. Out of scope

- Fused windowed-sum/argmin (stretch, see §3.4).
- FP64/complex128 kernel variants.
- Pallas/Triton equivalents (review report §7).
