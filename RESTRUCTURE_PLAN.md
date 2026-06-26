# CommsTools Restructuring Plan

Step-by-step plan to fix module size, the `Signal` god-object / circular-import
problem, naming conventions, tooling gaps, and test layout.

**Guiding rule for every step:** the public import surface
(`from commstools.X import Y`) and module-level function names stay stable
wherever possible. Internals move; user imports don't break. Run the full test
suite (`uv run pytest`) after each step — the existing ~1,200 tests are the
safety net.

**Signal decision (locked in):** move to a **thin `Signal` container + free
functions**. Fluent method-chaining is intentionally dropped in favor of a
simpler, flatter structure. `Signal` will own only data, validation, device
placement, rate/shape properties, and `copy`. All DSP, metrics, plotting, and
recovery behavior becomes module-level functions that take a `Signal` (or raw
array) as their first argument.

---

## Phase 0 — Repo hygiene (zero behavior change)

Low-risk, do first. No code logic touched.

**Already in good shape (verified, no action needed):** stray `main.py` is
deleted; `tmp/` is gitignored and untracked; `.gitignore` already covers `dist/`,
`*.egg-info`, `__pycache__/`, `.venv`, `.mypy_cache/`, `.ruff_cache`,
`.benchmarks`, and `*.pdf`/binaries. (The old `examples/` directory — notebooks,
scripts, and rendered images — was removed during Phase 3; new examples on the
free-function API will be added later.)

1. **Add `py.typed` marker** *(the one real action in this phase)*
   - Create empty `commstools/py.typed`.
   - Add to wheel artifacts in `pyproject.toml`
     (`[tool.hatch.build.targets.wheel]`), alongside the existing `.cu` artifact line.
   - Commit as "chore: add py.typed"; verify `uv build` and `uv run pytest` pass.
2. **(Optional) Pin notebook stripping** — `nbstripout` is already a dev dep and
   notebooks are clean today; wiring it as a pre-commit hook just guarantees they
   stay that way. Skip if a manual workflow is preferred.

---

## Phase 1 — Tooling & packaging (still no logic change)

1. **Pin Ruff config** — add `[tool.ruff]` + `[tool.ruff.lint]` to `pyproject.toml`:
   - `line-length = 100` (or current de-facto width).
   - `select = ["E", "F", "I", "UP", "B", "SIM", "RUF"]`; add per-file ignores
     for kernel modules if needed.
   - Run `uv run ruff check . --fix` and `uv run ruff format .`; commit the churn
     separately from the config so review is clean.
2. **Tighten mypy** — keep `ignore_missing_imports = true`, but add the scaffolding
   to ratchet strictness module-by-module (`[[tool.mypy.overrides]]` blocks).
   Start with the leaf modules that are already well-typed.
3. **Move non-core deps to extras**
   - `jupyter` is no longer needed now that the notebooks are gone — drop it (or
     move it to a future `docs` extra if examples return). It must not be a
     runtime dep.
   - Audit whether `pandas` is used in library code at all; if not, drop it too.
4. **Add CI** — a GitHub Actions workflow running, on push/PR:
   `uv sync`, `uv run ruff check .`, `uv run mypy commstools/`,
   `uv run pytest --device=cpu --cov=commstools --cov-fail-under=<current>`.
   GPU legs can stay manual/self-hosted.

---

## Phase 2 — Split `equalization.py` (9,677 LOC → `equalization/` package)

Highest readability payoff, mechanical, fully guarded by existing tests.
**No public symbol changes** — `equalization/__init__.py` re-exports everything
that is importable today.

1. **Create the package skeleton** `commstools/equalization/`:
   ```
   __init__.py          # re-export public API (see step 3)
   result.py            # EqualizerResult, CPRState, _log_equalizer_exit
   _kernels_numba.py    # _get_numba_lms/rls/cma/rde, *_cpr variants, cs_block
   _kernels_jax.py      # all _get_jax_* (sequential, block, pilot-aided)
   _block.py            # _block_eq_xp, _run_block_equalizer, FDAF, block_lms
   sequential.py        # public lms(), rls(), cma(), rde()
   blind.py             # block_cma, block_rde, godard/rde radii helpers
   linear.py            # zf_equalizer, apply_taps
   polarization.py      # demultiplex_polarization_tones_static/dynamic
   _common.py           # _normalize_inputs, _build_padded_samples,
                        #   weight init/validation, _cpr_symmetry, _validate_sps
   ```
2. **Move code in dependency order** (leaves first): `_common` → `result` →
   `_kernels_*` → `_block` → `sequential`/`blind`/`linear`/`polarization`.
   Keep each move a separate commit and run tests between moves.
3. **`__init__.py` re-exports** the current public names so
   `from commstools.equalization import lms` is unchanged:
   `lms, rls, cma, rde, block_lms, block_cma, block_rde, zf_equalizer,
   apply_taps, build_pilot_ref, EqualizerResult` (+ polarization demux).
4. **Verify** with `uv run pytest tests/test_equalization.py tests/test_block_lms.py
   tests/test_block_*.py tests/test_cpr_equalizer.py tests/test_bps_kernel.py
   tests/test_cs_kernel.py` then the full suite. Confirm benchmarks still import
   (`uv run pytest benchmarks/ --benchmark-only --device=cpu -k bench_block_lms`).

---

## Phase 3 — Thin `Signal` core + free functions (the architectural fix)

This is the deliberate, higher-touch change. Do it after Phase 2 so the biggest
module is already tamed. **Plan the call-site migration before moving code.**

### 3a. Inventory current `Signal` responsibilities
`Signal` today bundles five concerns (see `core.py`):
- **Container/meta:** validation, `to`/`xp`/`sp`/`backend`, `sps`/`duration`/
  `num_streams`/`bits_per_symbol`, `copy`, jax export/import. → **stays.**
- **Generation:** `generate`, `pam`, `psk`, `qam`, `psqam`. → move to `generation.py`.
- **DSP:** `fir_filter`, `matched_filter`, `resample`, `decimate`,
  `decimate_to_symbol_rate`, `upsample`, `shift_frequency`, `add_pilot_tone`,
  `welch_psd`, `spectrogram`, `shaping_filter_taps`. → move to their owning
  modules (`filtering`, `multirate`, `spectral`).
- **Metrics:** `evm`, `snr`, `ber`, `ser`, `mi`, `gmi`. → already thin delegators
  to `metrics`; convert fully to free functions.
- **Recovery/symbols:** `resolve_symbols`, `demap_symbols_hard`,
  `resolve_phase_ambiguity`, `resolve_channel_permutation`. → `recovery`/`mapping`.
- **Plotting:** `plot_psd`, `plot_eye`, `plot_constellation`, `plot_waveform`,
  `plot_spectrogram`. → `plotting`.

### 3b. Target structure
```
commstools/core/
  __init__.py     # re-export Signal, Preamble, SingleCarrierFrame
  signal.py       # thin Signal: data, validation, device, rate props, copy
  generation.py   # qam()/psk()/pam()/psqam()/generate() as free functions
  frame.py        # Preamble, SingleCarrierFrame
```
- Free functions live in their domain modules and take `sig: Signal` as the
  first arg, e.g. `metrics.evm(sig, ref)`, `plotting.constellation(sig)`,
  `recovery.recover_carrier_phase_viterbi_viterbi(sig, ...)`,
  `filtering.matched_filter(sig)`.
- Because `signal.py` no longer imports leaf modules, the leaves can import
  `Signal` **at module top level** — the circular dependency is gone and the
  ~30 function-local `from . import …` calls are deleted.

### 3c. Migration order (one concern per commit, tests between each)
1. **Generation** → move bodies to `core/generation.py`. Keep
   `Signal.qam(...)` etc. as one-line classmethod delegates for one release, OR
   expose top-level `commstools.qam(...)`. Update tests to the chosen form.
2. **Metrics** → make `metrics.evm/snr/ber/ser/mi/gmi(sig, ...)` the real API;
   drop the `Signal` methods (they're already thin wrappers).
3. **Plotting** → `plotting.constellation(sig, ...)` etc.; drop `Signal.plot_*`.
4. **DSP** → fold the DSP methods into `filtering`/`multirate`/`spectral` free
   functions; drop the methods.
5. **Recovery/symbols** → consolidate (see Phase 4 dedup) and drop the methods.
6. **Delete all function-local `from . import` in `core/`.** This is the
   completion check for the whole phase — grep must return zero hits:
   `grep -rn "    from \. import" commstools/core/` → empty.
7. Update `tests/` to the free-function API. (The old `examples/` were removed
   rather than migrated; new examples will be written against the new API later.)
   Update README snippets to match.

### 3d. Acceptance criteria for Phase 3
- No lazy intra-package imports anywhere (`grep -rn "    from \. import" commstools/`).
- Leaf modules import `Signal` at top level with no `ImportError`.
- `Signal` class body fits in well under ~600 lines.
- Full suite green on `--device=all` (or `cpu` if no GPU available).

---

## Phase 4 — Naming / API consistency

1. **Resolve same-name collisions** between compute and plot layers:
   - `analysis.carrier_phase_trajectory` (compute) vs
     `plotting.carrier_phase_trajectory` (plot) — rename the plot side to
     `plotting.plot_carrier_phase_trajectory` (or move under a plotting submodule).
   - Same for `allan_deviation` (analysis vs plotting) and `spectrogram`
     (compute vs plot).
   - Convention to document: **compute keeps the noun; plot functions take a
     `plot_` prefix.**
2. **Document the verb convention** already followed (`estimate_*` / `correct_*`
   / `recover_*`) in `CLAUDE.md` so contributors keep it.
3. **De-duplicate recovery** — `resolve_phase_ambiguity` /
   `resolve_channel_permutation` exist as both `Signal` methods and `recovery`
   functions; after Phase 3 they exist only as `recovery` functions.

---

## Phase 5 — Split remaining mega-modules

Same mechanical pattern as Phase 2 (subpackage + stable `__init__` re-exports).

1. **`plotting.py` (3,342 LOC) → `plotting/`**:
   `theme.py` (`apply_default_theme`, `_create_subplot_grid`, `_decimate_minmax`),
   `constellation.py`, `eye.py`, `spectral.py` (psd/spectrogram),
   `equalizer.py`, `sync.py` (timing/frequency/phase-trajectory plots),
   `analysis.py` (drift/allan/linewidth plots).
2. **`recovery.py` (2,777 LOC) → `recovery/`**:
   `pll.py`, `viterbi_viterbi.py`, `bps.py`, `tikhonov.py` (RTS/SSKF smoothers),
   `pilots.py`, `corrections.py` (`correct_carrier_phase`, `correct_cycle_slips`,
   `resolve_phase_ambiguity`, `resolve_channel_permutation`).
3. Keep `__init__.py` re-exports identical to today's flat module API.

---

## Phase 6 — Test reorganization ✅ DONE

Implemented: the two giants were split with an AST-based splitter (preserving
parametrize/fixture decorators and pruning unreachable helpers per file), the
already-focused equalization files were relocated, and the core tests moved
under `tests/core/`. Test count is unchanged (944 collected / 919 passed / 25
skipped on `--device=cpu`). Final layout:

```text
tests/equalization/   test_sequential.py  test_sequential_jax.py  test_mimo.py
                      test_winit.py  test_linear.py  test_polarization.py
                      test_blind.py  test_block_update.py  test_block.py
                      test_cpr.py  test_bps_kernel.py  test_cs_kernel.py
tests/recovery/       test_viterbi_viterbi.py  test_bps.py  test_pilots.py
                      test_tikhonov.py  test_pll.py  test_corrections.py
                      test_joint.py
tests/core/           test_signal.py  test_signal_mimo.py  test_frame.py
                      test_psqam.py
```

The largest file dropped from 2,817 LOC (`test_equalization.py`) to 1,166
(`tests/equalization/test_sequential.py`). The layout rule is documented in
`CLAUDE.md` §5. Step 4 (CI `--cov-fail-under`) remains tied to Phase 1's CI
workflow and is tracked there.

Original plan:

1. **Mirror the new source layout** — one test file per source module:
   ```
   tests/equalization/test_sequential.py   # lms/rls/cma/rde
   tests/equalization/test_block.py        # block_lms / FDAF
   tests/equalization/test_blind.py        # block_cma / block_rde
   tests/equalization/test_kernels.py      # bps / cs kernels
   tests/equalization/test_cpr.py
   tests/recovery/...
   tests/core/...
   ```
   This replaces the current fragmented set (`test_equalization`, `test_block_lms`,
   `test_block_update_equalizers`, `test_block_blind_equalizers`,
   `test_cpr_equalizer`, `test_bps_kernel`, `test_cs_kernel`).
2. **Split the giants** — `test_equalization.py` (2,854 LOC / 115 tests) and
   `test_recovery.py` (1,748 LOC / 108 tests) along the same boundaries.
3. **Rule to document:** a test file maps 1:1 to a source module.
4. **Enforce coverage** in CI with `--cov-fail-under` set to the current measured
   value so the refactor cannot silently drop coverage.

---

## Sequencing summary

| Phase | Risk | Value | Behavior change |
|---|---|---|---|
| 0 — Hygiene | none | medium | none |
| 1 — Tooling/packaging | low | medium | none (deps move to extras) |
| 2 — Split equalization | low | **high** | none (re-exports) |
| 3 — Thin Signal | medium | **high** | **yes — API: methods → free functions** |
| 4 — Naming | low | medium | minor renames (plot fns) |
| 5 — Split plotting/recovery | low | medium | none (re-exports) |
| 6 — Test reorg | low | medium | none |

Phases 0–2 and 5–6 are non-breaking. Phase 3 is the one deliberate breaking
change; consider shipping it as a single major version bump with the method→
function migration documented in the changelog.
