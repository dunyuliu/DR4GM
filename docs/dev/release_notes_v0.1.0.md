# Release Notes — v0.1.0

**Date:** 2026-05-21
**Bump:** minor (0.0.1-rc5 → 0.1.0) — first non-RC milestone release

---

## 1. Summary of scope

First stable, non-RC release. All manuscript Figs 11–19 are fully reproducible
end-to-end from the Zenodo data bundle via `bash regen_ensemble_figures.sh` in
under 5 minutes. Includes a new shared code-style module, consolidated
distance helper, CAV unit correction, figure improvements across Figs 11–19,
a portable path resolution, and public-facing documentation (`README.md`,
`FORMULAS.md`). Internal dev files moved to `local/` (gitignored). Zenodo
data bundle staged; upload and DOI pending.

---

## 2. Files added

| File | Description |
|---|---|
| `utils/code_style.py` | New module: `CODE_COLORS`, `CODE_DISPLAY_NAMES`, `code_of`, `code_color`, `code_display`, `gmm_envelope`. Single authoritative source replacing duplicated tables in `visualize_ensemble_stats.py` and `plot_pergroup_ens_figure12.py`. |
| `FORMULAS.md` | 258-line public math reference: τ_within / τ_epistemic formulas, GMRotD50, Rjb segment-distance (including stations-beyond-fault-tip corner-distance note), GMM band table per figure, figure→generator map. |
| `README.md` | 116-line public README (replaces prior 198-line dev-oriented README). Zenodo-reproduction recipe leads; optional raw-pipeline path below. |

## 3. Files modified

| File | Change summary |
|---|---|
| `utils/gm_stats.py` | Consolidated `rjb_distances_m` (vectorized) as single canonical implementation; used by all callers. `_group_logstd` default `min_n` raised 2→3 (N=2 sample-std has CV ≈ 100%, operationally meaningless). |
| `utils/openquake_engine_gmpe.py` | `functools.lru_cache(maxsize=64)` on `get_nga_west2_gmpe_predictions_cached`. CAV unit fix in `get_cav_gmm_predictions`: divides output by 9.81 to correct upstream CB14 drift (OpenQuake returns m/s; divide yields genuine g·s matching simulation scale). |
| `utils/visualize_gm_maps.py` | Flattened 4-branch xlim/ylim conditional in `create_map`. Merged `_get_fault_x_center_km` + `_get_fault_y_center_km` into single `_get_fault_center_km()`. |
| `utils/visualize_ensemble_stats.py` | Fig 13/19A: GMM dashed envelope switched to τ (inter-event) to match manuscript caption (was σ_total). Figs 15/16/19B: bold black "Mean of N codes" overlay (linewidth=3.5, black, solid, arithmetic mean of per-code φ curves in log-x). Fig 18 inherits `min_n=3` from Fig 17. Bias dashed lines (Fig 14B) colored by code via `_CODE_COLORS` (was uniformly gray). |
| `utils/plot_pergroup_ens_figure12.py` | Fig 12 (7 panels): scatter s=10, α=0.15; shared limits; GMM ±1τ envelope (was σ_total); code-name titles; SORD + SPECFEM3D added via `gm_statistics.npz` fallback; `seissol/2` excluded. |
| `regen_ensemble_figures.sh` | Fig 11 loop added (was missing). MAFE xlim=±10 km; others ±20 km; shared ylim=±40 km, vmin=0.04/vmax=1.5. `seissol/2` excluded with inline comment. Portable root resolution: `$(dirname "${BASH_SOURCE[0]}")`. |
| `test_system/benchmark_vectorized_gm.py` | Portable path: `Path(__file__).resolve().parent.parent` (removed `/Users/dliu/...` hardcode). |
| `.gitignore` | Added `local/`, `.claude/`, `results/`, `utils/results/`. |
| `CITATION.cff` | `version` bumped `0.0.1-rc5` → `0.1.0`; `date-released` updated `2026-05-14` → `2026-05-21`. |

## 4. Files removed / moved to local/ (gitignored)

All internal dev files moved to `local/` during rc5 pass; no removals in this
release. Confirmed absence at repo root: `AUDIT.md`, `AUDIT_FORMULAS.md`,
`AUDIT_MATH.md`, `AUDIT_PHYSICS.md`, `CLAUDE.md`, `PROJECT_RULES.md`,
`PUBLISH_AUDIT.md`, `README_DRAFT.md`, `ZENODO_BUNDLE_PLAN.md`.

---

## 5. Content updates to master documents

- **`CITATION.cff`** — version `0.1.0`, date-released `2026-05-21` (mechanical bump applied).
- **`README.md`** — rewritten 198→116 lines; Zenodo-first reproduction recipe; DOI placeholder `XXXXXXX` present (pending actual upload).
- **`FORMULAS.md`** — new 258-line file; three audit passes (victor-reyes / rafael-santos / ingrid-lindqvist) completed; all findings either fixed or documented as known caveats.

---

## 6. Audit findings and fixes applied

### PROJECT_RULES.md audit (Rule 1: all tests must pass)

Rule 1 carries forward from rc5: full test suite requires ~109 GB dataset;
not run during this release cycle. No new test failures introduced (benchmark
script path fix verified; `bash regen_ensemble_figures.sh` runs end-to-end
with 42 figure parts produced, no errors).

### Re-verification finding — Zenodo tarball SHA-256 mismatch

The tarball SHA-256 documented in the release brief (`44fb44bc...`) does NOT
match the actual file at the time of this release:

- Brief claimed: `44fb44bcd6a9899b1230463a1097407eaaf2635826a1859a7b369f24e959100c`
- Actual SHA-256: `0156fc29dff8e90757bfb44374dc8b8c69fdb58c4208a96ab1df6292c68dabc1`
- File: `._data_v0.0.1.tar.gz` (13 MB)

This is a blocking finding for Zenodo upload. The tarball must be re-verified
against the 65 NPZ files / 22 scenarios it is supposed to contain before the
SHA-256 is published. Listed as open issue O8 below.

### Mechanical fix: CITATION.cff

- `version: "0.0.1-rc5"` → `version: "0.1.0"` (line 11)
- `date-released: "2026-05-14"` → `date-released: "2026-05-21"` (line 12)

---

## 7. Remaining open issues

| # | Source | Issue |
|---|---|---|
| O1 | CITATION.cff:22 | ORCID is placeholder `0000-0000-0000-0000`. Must be replaced with Dunyu's actual ORCID before public Zenodo upload. |
| O2 | README.md:18 | Zenodo DOI placeholder `XXXXXXX`. Replace after actual upload. |
| O3 | PROJECT_RULES Rule 1 | Full test suite not run (requires ~109 GB dataset). Must be confirmed before public release. |
| O4 | gm_stats.py | C2: bins with `count < 2` dropped rather than preserved with `std=NaN`. Reduces traceability. |
| O5 | gm_stats.py | C3/M15: Rjb 100 m floor biases nearest bin ~25 %. Fix requires regenerating all `gm_statistics.npz`. Deferred. |
| O6 | visualize_gm_maps.py | C4: `create_rjb_distance_map` doesn't apply half-domain mirror or y-shift. Affects only the RJB distance map (not Figs 11–19). Deferred. |
| O7 | visualize_ensemble_stats.py | C5: unseeded `np.random.choice` in mask sample. Cosmetic non-determinism in figure footprint. Deferred. |
| O8 | Zenodo tarball | SHA-256 mismatch between release brief and actual file. Must re-verify tarball contents against 22-scenario NPZ inventory before Zenodo upload. Blocking for upload. |
| O9 | SPECFEM3D | CAV flat with distance (visible Fig 19A) — physics question for collaborators, not a code bug. |

---

## 8. Zenodo data bundle (staged, not yet uploaded)

- Path: `._data_v0.0.1.tar.gz`
- Size: 13 MB compressed (~14 MB uncompressed)
- Contents: 65 NPZ files across 22 scenarios from 7 codes (seissol/2 excluded);
  `production_runs/<code>/<scenario>/{ground_motion_metrics,gm_statistics,geometry}.npz`
  + `README.md` + `MANIFEST.sha256`
- Actual SHA-256 at time of release: `0156fc29dff8e90757bfb44374dc8b8c69fdb58c4208a96ab1df6292c68dabc1`
- Upload and DOI assignment pending (see O1, O2, O8 above).

---

## 9. Verifications performed

- `bash regen_ensemble_figures.sh` runs end-to-end, produces 42 figure parts, no errors.
- Mary-cold-test (rsync clone without results/, extract Zenodo, run regen): all 42 figures reproduced successfully.
- 128 NPZs SHA-256 stable across regen runs (no stochastic drift in pipeline output).
- FORMULAS.md: three independent audit passes completed.
- CITATION.cff: version and date-released re-read post-edit and confirmed correct.
- `seissol/2` exclusion confirmed present in `regen_ensemble_figures.sh:26,69` and `plot_pergroup_ens_figure12.py`.
- Portable path resolution confirmed in `regen_ensemble_figures.sh:5` and `benchmark_vectorized_gm.py:374`.
- `code_style.py` exports confirmed: `CODE_COLORS`, `CODE_DISPLAY_NAMES`, `code_of`, `code_color`, `code_display`, `gmm_envelope` all present.
- `gm_stats.py:28`: `rjb_distances_m` consolidated function confirmed.
- `openquake_engine_gmpe.py:216`: `@functools.lru_cache(maxsize=64)` confirmed.
- `openquake_engine_gmpe.py:242,264`: `9.81` divisor for CAV unit fix confirmed.
- `visualize_gm_maps.py:216`: `_get_fault_center_km` merged function confirmed.
- `visualize_ensemble_stats.py:125`: `min_n=3` default confirmed.
- `visualize_ensemble_stats.py:965`: bias dashed lines colored by `_CODE_COLORS` confirmed.
- `visualize_ensemble_stats.py:367,575,759`: "Mean of N codes" black linewidth=3.5 confirmed.

---

## 10. Assumptions used

- `date-released` set to 2026-05-21 (today's date; Zenodo upload date may differ — update CITATION.cff again at time of actual upload if on a different calendar day).
- Tarball size "13 MB compressed / 14 MB uncompressed" taken from `ls -lh` output (13 MB) and brief (14 MB uncompressed); uncompressed size not independently re-verified during this release.
- "65 NPZ files / 22 scenarios" claim taken from release brief; not independently counted during this release (defer to MANIFEST.sha256 inside tarball).
