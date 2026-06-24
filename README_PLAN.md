# README Enhancement Plan

Step-by-step plan to replace the code-heavy "Visual Pipeline Tour" with a
result-first **plot gallery**, surface the strongest existing renders, and make
the GitHub landing page convert visitors into users.

**Core idea:** lead with *what CommsTools produces* (publication-quality, themed,
one-call plots), not with matplotlib plumbing. Show images and one-liners; push
full scripts into `examples/`.

---

## Phase 0 — Asset audit & prep

1. **Inventory existing images** in `examples/images/`:
   - Currently shown: `psd.png`, `eye_diagram.png`, `constellation.png`,
     `laser_phase_characterization.png`.
   - **Unused but strong** (surface these): `siso_transmission_diagnostics.png`
     (1.2 MB), `multi_stage_equalization_diagnostics.png` (1.6 MB),
     `psqam_intradyne_diagnostics.png` (800 KB).
2. **Down-res for web** — inline 1–1.6 MB PNGs make the page load slowly.
   Generate web-sized copies (e.g. max ~1200 px wide, < ~300 KB) into
   `examples/images/web/` (or `docs/assets/`). Keep full-res originals for the
   example scripts.
3. **Confirm each image is reproducible** from a script in `examples/` so every
   gallery item can link to the code that made it. Note any gaps to fill in
   Phase 2.
4. **Pick the hero image** — the SISO or intradyne diagnostics dashboard is the
   best "wow" candidate for above-the-fold.

---

## Phase 1 — Restructure the top of the README

Reorder so the first screen sells, in this order:

1. **Title + tagline** (keep).
2. **Badges** — keep, but add a project-status badge (alpha/beta) so the
   "not yet on PyPI" install instructions match expectations.
3. **Hero image** — one wide, full-width diagnostics dashboard with a caption:
   *"End-to-end coherent pipeline diagnostics — one `Signal`, one theme,
   GPU-transparent."* This is the new above-the-fold hook.
4. **One-paragraph positioning** — keep the existing "hardware as a first-class
   concern" pitch but add **one sentence on why this over raw NumPy/SciPy or
   other comms libraries** (metadata-carrying `Signal` + backend-transparent
   dispatch + publication-grade plots).
5. **The 60-second snippet** (see Phase 3) — a single tight end-to-end example.
6. **Gallery** (Phase 2).
7. Feature table → keep but trim; everything reference-y moves down/out (Phase 4).

---

## Phase 2 — Build the plot gallery (replaces "Visual Pipeline Tour")

Replace README sections 1–4 ("Visual Pipeline Tour") with a gallery grid.

1. **Layout** — a 2- or 3-column HTML/markdown table of thumbnails. Each cell:
   - thumbnail (web-sized, linked to full-res),
   - **bold title that promotes the capability**,
   - the **single API call** that produced it (no `plt.subplots`/`legend`/`show`),
   - a link → the `examples/…py` script for the full version.
2. **Suggested gallery items + selling titles:**

   | Image | Title | One-liner shown |
   |---|---|---|
   | `constellation.png` | Density constellation with ideal-grid overlay | `sig.plot_constellation(overlay_ideal=True)`* |
   | `eye_diagram.png` | Vectorized 2-D histogram eye diagram | `sig.plot_eye(type="hist")`* |
   | `psd.png` | Welch PSD — clean vs. noisy channel | `sig.plot_psd(label=...)`* |
   | `laser_phase_characterization.png` | Laser linewidth & Allan-deviation characterization | `analysis.characterize_carrier_phase(...)` |
   | `multi_stage_equalization_diagnostics.png` | Cascaded CMA → LMS equalizer convergence | `plotting.equalizer_result(result)` |
   | `psqam_intradyne_diagnostics.png` | Probabilistically-shaped QAM intradyne receiver | see `examples/psqam_intradyne_pipeline.py` |

   *\*If the Signal-method API is migrated to free functions per the
   restructuring plan (Phase 3), update these one-liners to the free-function
   form (e.g. `plotting.constellation(sig, overlay_ideal=True)`). Keep README
   and code in sync — do this step after the API decision lands.*

3. **Strip all matplotlib scaffolding** from README code blocks — show the
   library call, not figure/axis/legend/show ceremony. The verbose versions
   live in the example scripts.

---

## Phase 3 — Tighten the runnable example

1. Keep exactly **one** end-to-end snippet near the top: generate → impair →
   recover → measure, ~15 lines, no plotting boilerplate. It proves the API is
   ergonomic.
2. Everything else points to `examples/`. Ensure the referenced scripts
   (`single_carrier_transmission.py`, `multi_stage_equalization.py`,
   `psqam_intradyne_pipeline.py`, `laser_phase_characterization.py`,
   `generate_plots.py`) actually run end-to-end and produce the gallery images.
3. Keep the existing `Signal` "Core Concept" block — it's good — but trim it to
   the essential `sps`/`backend`/`duration` demo.

---

## Phase 4 — Trim and relocate reference material

1. Move the long **Module Reference**, **Requirements**, and **Running Tests**
   tables out of the landing flow — either to the bottom under a collapsed
   `<details>` block or into `docs/` / `CONTRIBUTING.md`.
2. Keep a **short** feature table on the landing page (the most compelling rows).
3. Keep **Installation** concise and near the top; the GPU/CUDA detail can go in
   a `<details>` block.

---

## Phase 5 — Polish & verify

1. **Page-weight check** — confirm total inline image weight is reasonable
   (web-sized assets, not the 1 MB+ originals).
2. **Render check** — preview on GitHub (not just locally); confirm relative
   image paths resolve and tables render.
3. **Link check** — every gallery item links to a real, runnable script.
4. **Consistency pass** — README API calls match the actual current API
   (re-check after restructuring Phase 3 if both efforts run together).
5. **Alt text** — give every `<img>` meaningful alt text for accessibility.

---

## Sequencing summary

| Phase | Output |
|---|---|
| 0 — Asset audit | web-sized images, hero pick, reproducibility confirmed |
| 1 — Top restructure | hero + positioning + status badge above the fold |
| 2 — Gallery | "Visual Pipeline Tour" replaced by titled thumbnail grid |
| 3 — Tighten example | one clean end-to-end snippet, rest in `examples/` |
| 4 — Trim reference | long tables collapsed/moved to docs |
| 5 — Polish | page weight, render, links, alt text verified |

**Dependency note:** the gallery one-liners (Phase 2) and the runnable snippet
(Phase 3) should reflect the final `Signal` API. If the restructuring plan's
Phase 3 (methods → free functions) is going ahead, do this README work *after*
that lands, or write both in the free-function form from the start to avoid a
second rewrite.
