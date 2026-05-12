# DR4GM v0.0.1-rc2

**Date:** 2026-04-27

## Summary of scope

Second release candidate of DR4GM (Data Repository for Ground Motion).
Replaces rc1, which was prepared locally but never pushed: rc2 keeps the
`web/` Streamlit explorer (pulled from `origin/main`'s last tracked
state) instead of dropping `web/` entirely, and force-pushes over the
prior `origin/main` history.
Establishes the full processing pipeline (raw simulation → standardized
NPZ → ground-motion metrics → distance-binned statistics → NGA-West2
GMPE comparison) for seven simulation codes, plus a regression test, a
frozen reference baseline, batch figure collection, and the GUI / web /
demo interfaces.

## Files added / removed / renamed / cleaned up

### Added

- `CLAUDE.md` — developer-facing notes (gmpe-smtk attribution, coordinate
  rotation rules, release workflow).
- `PROJECT_RULES.md` — single rule: all tests must pass before release.
- `fetch_figures_for_publication.sh` — pure-cp collector that pulls
  per-scenario T=1 s SA maps, per-scenario SA-vs-distance plots, and
  ensemble plots into `<results>/figs_to_publish/` named per the
  DRV / SRL manuscript. Code-letter scheme:
  A=eqdyna, B=fd3d, C=mafe, D=seissol, E=waveqlab3d, F=specfem3d, G=sord;
  realization number suffix (e.g. Fig 11D3 = seissol scenario 3).
  Collects Fig 11A1-E3, Fig 12A1-E3, Fig 13A/B/C, Fig 14, Fig 15A/B/C,
  Fig 16, Fig 19A/B (50 figures for the 20-scenario production set).
- `test_system/` — regression runner (`run_tests.sh`), float32-aware diff
  (`diff_gm_metrics.py`), and bundled 1 km reference set
  (`reference_results/`, ~3 MB, five canonical scenarios).
- `gmpe-smtk/` — vendored in-tree (AGPLv3, GEM Foundation; inner `.git/`
  removed). Provides `get_peak_measures`, `get_cav`, `gmrotdpp_withPG`,
  `NigamJennings`. Local NumPy 2.x / SciPy ≥ 1.14 patches recorded in
  `gmpe-smtk/LOCAL_MODIFICATIONS.md`.
- `utils/` — full converter / processor / visualization suite:
  `eqdyna_converter_api.py`, `fd3d_converter_api.py`,
  `seissol_converter_api.py`, `waveqlab3d_converter_api.py`,
  `specfem3d_converter_api.py`, `mafe_converter_api.py`,
  `sord_plot_converter_api.py`, `npz_gm_processor.py`,
  `station_subset_selector.py`, `gm_stats.py`, `visualize_gm_maps.py`,
  `visualize_gm_stats.py`, `visualize_ensemble_stats.py`,
  `vectorized_gmrotd50.py`, `openquake_engine_gmpe.py`, `run_all.sh`, etc.
- `gui/` — Tkinter desktop GUI (`tkGUI_dr4gm_new.py`).
- `web/` — Streamlit interactive explorer
  (`dr4gm_interactive_explorer.py`, `requirements.txt`,
  `sample_stations_list.csv`); local-only Streamlit state
  (`.streamlit/`, `usage_analytics.log`, `__pycache__/`) is gitignored.
- `demo/` — capability / business diagram generators.
- `docs/` — slides, flyer, capability diagrams, FORMAT.md, WORKFLOW.md.

### Removed (legacy / pre-rename)

- `installDepreciatedGMPE-SMTK.sh`, `make.scripts.executable.sh`,
  `test.smtk.py`, `misc/driver/` — pre-rename helpers superseded by
  `install.sh` and `gmpe-smtk/` vendoring.
- `utils/{archiveGMData,genMaps,gmGetSimuAndGMPEScaling,gmProcessor,
  metadataWriter,tkGUI.dr4gm}` and the matching helper modules
  (`genMapFuncLib.py`, `gmFuncLib.py`, `metadataDict.py`,
  `ComputeGroundMotionParametersFromSurfaceOutput_Hybrid_Lite.py`) —
  legacy entry points replaced by the `*_converter_api.py` /
  `npz_gm_processor.py` / `visualize_*` chain.
- `utils/rename_figures.sh` — outdated duplicate of `make_figures.sh`
  (hard-coded the pre-rename `results/<code>.<scenario>/` flat layout).

### Renamed

- `install.dr4gm.sh` → `install.sh`.

### Cleaned up

- `web/.streamlit/`, `web/usage_analytics.log`, `web/__pycache__/`,
  `web/.DS_Store` — gitignored. Streamlit secrets and runtime telemetry
  must never be committed; only the explorer source, requirements, and
  sample station list are tracked.
- `.gitignore` expanded to cover `__pycache__/`, `reference/`, `datasets`,
  `results/`, `*.webm`, editor noise, and the new `web/` local-state
  rules.

## Content updates to master documents

- `README.md` — full rewrite. Install via `source install.sh`, pipeline
  diagram, supported-code table, regression-test instructions with
  wall-clock table (Apple M3 / 8 cores / 24 GB), reference-baseline
  description, GUI / web / demo pointers, dependencies, citation, and
  AGPLv3 license / attribution.
- `LICENSE` — repository LICENSE updated.
- `CLAUDE.md` — gmpe-smtk attribution and AGPLv3 file-protection list,
  reference-baseline tree (`reference/datasets/`,
  `reference/results_original_resolution/`), test-system reference path,
  coordinate-rotation table for all seven simulation codes, release
  workflow.

## Audit findings and fixes

| # | Finding | Fix |
|---|---|---|
| 1 | `utils/rename_figures.sh` duplicates `make_figures.sh` and uses pre-rename `results/<code>.<scenario>/` paths | Deleted. |
| 2 | `web/.streamlit/secrets.toml` had been tracked previously and held a live Google Apps Script webhook URL | Pulled `web/` content from `origin/main` (3 files: explorer, requirements, sample list — no secrets). `web/.streamlit/`, `usage_analytics.log`, `__pycache__/`, `.DS_Store` gitignored. Webhook should still be rotated since it remains in pre-release history on GitHub. |
| 3 | PROJECT_RULES.md rule 1: "All tests must pass" | Ran `bash test_system/run_tests.sh` — 5 PASS / 0 NOREF / 0 FAIL in 15m47s. |
| 4 | Fig 14 / Fig 16 (SA / std vs period at Rjb = 10 km) referenced by old `rename_figures.sh` were never produced — `--plot-spectra-vs-periods` in `visualize_ensemble_stats.py` was opt-in | Made it default to 10 km. The two plots now appear in every ensemble run. |

## Remaining open issues

- SPECFEM3D and SORD scenarios are not in the 1 km regression set. Their
  converter APIs ship, but no reference `ground_motion_metrics.npz` is
  bundled (they need a full end-to-end run before they can be validated
  bit-exact).
- `reference/` (~200 GB raw + processed baseline) is `chmod -R a-w` and
  excluded from the repo via `.gitignore`. Collaborators reproduce by
  rerunning the pipeline; figshare links for raw datasets are in
  `reference/datasets/README.md`.

## Totals

- Five canonical regression scenarios, ~14 min wall-clock on Apple M3
  (8 cores / 24 GB RAM).
- All twenty scenarios (`--all`), ~59 min on the same machine.
- Bundled 1 km reference set ~3 MB.

## Assumptions

- Apple M3 / 8 cores / 24 GB RAM is the wall-clock reference machine.
- Float32-aware diff tolerance: `1e-6` rel for float32 input velocities,
  `1e-12` rel otherwise (`test_system/diff_gm_metrics.py`).
- `Vs30 = 760 m/s` and `M = 7.0` for NGA-West2 GMPE overlays in
  `visualize_ensemble_stats.py`.
