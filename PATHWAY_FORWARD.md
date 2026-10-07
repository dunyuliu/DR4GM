# DR4GM board

Status board for /autopilot (named by `PROJECT_RULES.md` rule 3).
Status: TODO · DOING · DONE · BLOCKED(owner). Every row has an evidence command.
Repo is public (github.com/dunyuliu/DR4GM). Dev cycle per `/autopilot`: branch → PR → CI → merge;
tag gated on green CI for that SHA. CI does not exist yet, so row 9 comes first.

## Gate tiers
| Tier | Command | Needs | Time |
|---|---|---|---|
| smoke | `bash test_system/check_layout.sh && python3 -m pytest -q test_system/unit` | repo only | < 1 min |
| fast | smoke + `bash test_system/run_e2e_bundle.sh ../dr4gm_data_v0.1.1.tar.gz` | 14 MB bundle | ~5–10 min |
| full | fast + `bash test_system/run_tests.sh` | 199 GB reference/ | ~13 min |

CI runs smoke (fast needs the bundle; add once the bundle is downloadable). Merge needs CI green + fast run locally; release needs full green.

## Owner decisions
- 2026-10-07 `/autopilot reorg. And enhcnace tests, particularly e2e, design light CI, and refactor` (verbatim). Row 10 unblocked. Rows 11, 12 remain BLOCKED(owner).
- 2026-10-07 `Why? Board at root` (verbatim). Board (this file) moves to tracked repo root, not `docs/dev/`.
- 2026-10-07 `Follow Zofia template` (verbatim). Row 10's reorg follows zofia-kaminska's full root-layout template, superseding the partial mapping first drafted here. Zofia owns producing the old→new path map (delivered — see row 10); execution routes through surface owners (kai-fischer for code/dir moves, etc.) once the map is applied.
- 2026-10-07 `Whole workflow in e2e?` (verbatim), answered: current e2e (row 2/3) covers stats→figures only; `run_tests.sh` covers raw→convert→subset→metrics but needs 199 GB `reference/`, so nothing CI-runnable touches converters or GM computation today. New scope under "enhance tests, particularly e2e": a whole-workflow e2e on tiny raw fixtures (see row 13).
- 2026-10-07 `Those can be anchor to develop the new lighter e2e?` (verbatim), answered yes — design requirement for row 13, binding: (1) per-station GM metrics on the cropped raw fixture must equal `test_system/reference_results` (the `run_tests.sh` 1 km reference) for the same station IDs, within the existing float32 tolerance; (2) expected stats/figure inputs = `gm_stats` run on the full reference's metrics restricted to the cropped station set; (3) a local-only check (needs `reference/`) proves light-reference == full-oracle subset, then CI uses the frozen light reference — never the reverse derivation; (4) when the full reference is re-blessed, the light reference is re-derived from it. Also closes a provenance seam: a local-only check that `make_zenodo_bundle.sh` output from a fresh `run_tests.sh --all` matches the shipped bundle NPZs.

## Rows
| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Layout gate (strict, run first by `run_tests.sh`) | DONE | PR #1, `ff418c2` on main; `bash test_system/check_layout.sh` exits 0; stray root file → exit 1 |
| 2 | E2E from the Zenodo bundle: bundle → `regen_ensemble_figures.sh` → 41 figure parts + ensemble statistics vs stored reference | DONE | PR #3, `f95d049` on main; `bash test_system/run_e2e_bundle.sh ../dr4gm_data_v0.1.1.tar.gz` -> exit 0 (conductor fresh re-run, not just the builder's report); openquake-missing skip is explicit (named banner), not silent — any other figure going missing still fails |
| 3 | E2E hardening: run from a clean `git archive` export (no untracked files), assert per-figure image dims, check seissol/2 stays excluded, check Fig 11 panel set (no SORD/SPECFEM3D panels), test against BOTH bundle versions or pin one | TODO | e2e run log; deliberately re-adding seissol/2 must fail |
| 4 | Unit tier (`test_system/unit/`, pytest, no raw data): `gm_stats.rjb_distances_m` vs hand cases (incl. y-offset WaveQLab3D/FD3D geometry), binning/`count<2` behaviour pinned as-is, `vectorized_gmrotd50` vs vendored gmpe-smtk on a synthetic 3-component record, one converter per code on a tiny fixture (≤ 200 kB each), `code_style` registry consistency | TODO | `python3 -m pytest -q test_system/unit` green; each test fails when its target is mutated |
| 5 | Integration tier: `utils/run_all.sh` on one small real scenario subset (fixture ≤ 2 MB) → `ground_motion_metrics.npz` matches reference | TODO | `pytest -q test_system/unit -k run_all` |
| 6 | Gate script header must not reference internal `local/` files (repo is public) | DONE | PR #1, `ff418c2`; `grep -n local/ test_system/*.sh` empty (only path-check line, no doc reference) |
| 7 | `web/usage_analytics.log` is tracked runtime output → untrack + gitignore | DONE (already true pre-session) | `git ls-files web/usage_analytics.log` empty; `.gitignore` already lists it |
| 8 | README names bundle v0.0.1, current is v0.1.1 → align | DONE | PR #1, `ff418c2`; `grep -n dr4gm_data_v README.md` shows v0.1.1 |
| 9 | CI (GitHub Actions): smoke tier on push/PR — **first**, the merge gate depends on it | DOING — layout-gate job live and green (PR #1, run 37702616835, SHA `ff418c2`); `pytest test_system/unit` step is a TODO in the workflow, blocked on row 4 (no unit dir exists yet) | `.github/workflows/ci.yml`; `gh run list --branch main` |
| 10 | Root layout per zofia's full template — old → new path map (delivered 2026-10-07): `README.md`, `LICENSE`, `.gitignore`, `.github/`, `CITATION.cff`, `requirements.txt` stay at root; `PATHWAY_FORWARD.md`, `PROJECT_RULES.md` promoted to tracked root (DONE, PR #2); `FORMULAS.md`→`docs/user/`, entry-point scripts (`install.sh`,`run_pipeline.sh`,`regen_ensemble_figures.sh`,`fetch_figures_for_publication.sh`,`make_zenodo_bundle.sh`)→`scripts/` (DONE, PR #5, `4d74440`); still TODO: `local/CLAUDE.md`→root `CLAUDE.md` (needs its own scrub pass — absolute path + email); current `release_notes_v0.1.1.md`→root, older ones→`docs/dev/`; `AUDIT*.md`, `PUBLISH_AUDIT.md`, `ZENODO_BUNDLE_PLAN.md`→`docs/dev/`; `test_system/`→one `tests/` (defer until row 13 lands, same directory); `utils/`,`gui/`,`web/`→`src/{utils,gui,web}/`; `gmpe-smtk/`→`src/gmpe-smtk/` (attribution files move as a unit, import paths need a code fix — route to lars-eriksson); `reference/` stays in place, linked under `data/reference` (never copied); `results/`→git-ignored `runs/<date>_<slug>/` convention; `datasets` legacy root symlink → flagged for retirement/repoint; `local/` dissolves once all of the above lands | DOING — 2 of ~9 slices landed | `bash test_system/check_layout.sh` with the new whitelist |
| 11 | Vendored `gmpe-smtk` test data > 5 MB (38.6 MB CSV, 19.6 MB HDF5): untrack or LFS | BLOCKED(owner) | `check_layout.sh` without SIZE_EXEMPT |
| 12 | `gm_stats.py` Rjb 100 m floor (C3/M15) and `count<2` drop (C2) | BLOCKED(owner) — changes published figures; needs `gm_statistics.npz` regen + paper check | `local/AUDIT.md` C2/C3 |
| 13 | Whole-workflow e2e on tiny raw fixtures (a few hundred stations cropped from one real scenario per code, a few MB total, 5 MB/file cap, under `test_system/` for now — relocates with row 10): raw → convert → subset → GM metrics → stats → ≥1 figure, diffed against a light reference. Light reference must be DERIVED from the full local oracle, never self-blessed — see the 2026-10-07 "anchor" owner decision above for the exact 4-point derivation chain and the `make_zenodo_bundle.sh`-vs-shipped-NPZ provenance check. Fixtures read-only from `reference/`, never written into it | TODO | a local-only (needs `reference/`) check proving light-reference == full-oracle subset, then a CI-runnable run against the frozen light reference; fast enough for light CI |
