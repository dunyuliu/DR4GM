# Release Notes — v0.1.2

**Date:** 2026-10-08
**Bump:** patch (0.1.1 → 0.1.2) — two landed fixes (default distance-bin size,
removal of an autonomous git-commit path) plus the test/CI/layout
infrastructure built since v0.1.1 to gate this and future releases.

---

## 1. Summary of scope

Since v0.1.1 (tag `v0.1.1`, `2870e8f`'s ancestor), the project went from "no CI,
no layout rule, no test infra beyond the 199 GB full regression" to a gated
three-tier test pyramid (smoke/fast/full), a GitHub Actions smoke CI, a
repo-layout whitelist gate, and a public root-layout reorg (`utils/`→
`src/utils/`, `gmpe-smtk/`→`src/gmpe-smtk/`, `gui/`+`web/`→`src/`,
`FORMULAS.md`→`docs/user/FORMULAS.md`, entry-point scripts → `scripts/`,
board/rules promoted to tracked root). That infrastructure work is what let
this release be audited, gated, and verified at all; it ships as part of
v0.1.2 because it has not been in a tagged release before.

Two behavior-affecting fixes ship in this release:

- **Board row 14 / PR #16 (`fdd5e20`):** `src/utils/gm_stats.py`'s
  `--distance_bin_size` default changed 500 m → 2000 m, so running
  `gm_stats.py` with no explicit bin-size flag reproduces the v0.1.1 bundle's
  published `gm_statistics.npz` exactly (owner-chosen option (b); verified by
  conductor re-run: `distance_bin_edges`/`PGA_count`/min/max exact match,
  mean/std matching to float64 machine epsilon, 4.6e-13 worst relative
  deviation across all 103 arrays).
- **Board row 15 / PR #18 (`5e86feb`):** removed (not relocated)
  `_commit_to_git()` from the Streamlit usage-analytics tracker
  (`src/web/dr4gm_interactive_explorer.py`), which had the running app
  autonomously `git add`/`commit`/`push` its own log into whatever repo it
  was checked out in, on a timer, wrapped in a bare `except Exception: pass`
  — a violation of `PROJECT_RULES.md` rule 5 (added in this same campaign).
  `grep -rn "subprocess.run(\['git'" src/` now returns zero hits, and
  `test_system/check_layout.sh` gates this going forward.

## 2. Files added / removed / renamed / cleaned up

This release accumulates the full repo reorg landed since v0.1.1 (see board
row 10, `PARTIAL — 4 of ~9 slices landed, DOING`). Renamed/moved, not
deleted: `utils/`→`src/utils/`, `gmpe-smtk/`→`src/gmpe-smtk/` (attribution
files moved as a unit), `gui/`+`web/`→`src/{gui,web}/`, `FORMULAS.md`→
`docs/user/FORMULAS.md`, `install.sh`/`run_pipeline.sh`/
`regen_ensemble_figures.sh`/`fetch_figures_for_publication.sh`/
`make_zenodo_bundle.sh`→`scripts/`, older `release_notes_v*.md` + `AUDIT*.md`
+ `PUBLISH_AUDIT.md` + `ZENODO_BUNDLE_PLAN.md`→`docs/dev/`, `local/CLAUDE.md`
→ tracked root `CLAUDE.md` (scrubbed), `PATHWAY_FORWARD.md`/`PROJECT_RULES.md`
promoted to tracked root. Added: `.github/workflows/ci.yml` (smoke CI),
`test_system/check_layout.sh` (layout gate), `test_system/unit/` (39 tests),
`test_system/run_fixture_e2e.sh` + three whole-workflow fixtures (SeisSol,
EQdyna, FD3D), `data/` (4 vendored Streamlit demo NPZs + `MANIFEST.md`),
session logs and audit docs under `docs/dev/`. This commit additionally
archives `release_notes_v0.1.1.md` to `docs/dev/` (never deleted) and adds
this file at tracked root.

Still outstanding on row 10 (not part of this release's scope, not a
blocker): `test_system/`→`tests/` consolidation, `results/`→git-ignored
`runs/<date>_<slug>/`, vendored `gmpe-smtk/` test files still over the 5 MB
cap (row 11, BLOCKED(owner)).

## 3. Version and master-document sync

- `CITATION.cff`: `version` 0.1.1 → 0.1.2, `date-released` 2026-08-03 →
  2026-10-08.
- `CLAUDE.md`: "Current version" header synced to 0.1.2.
- No other tracked file referenced the old version string except the Zenodo
  bundle filename `dr4gm_data_v0.1.1.tar.gz` in `README.md`, which is a data
  artifact version (unchanged by this code release — no new NPZ data
  shipped) and is left as-is.

## 4. Audit findings and fixes

**Audit performed:** zofia-kaminska (project-rules pass, clean except one
MAJOR, since fixed and merged — PR #18/`5e86feb`, board row 15) and
victor-reyes/lars-eriksson (code-correctness pass, full regression tier
`test_system/run_tests.sh` 5/5 PASS, three further MAJOR findings logged as
board rows 24–26 plus one unscoped finding as row 27 — none release-blocking,
all pre-existing in code that predates this release). Both passes are logged
in `PATHWAY_FORWARD.md`.

**Gate checks run fresh for this release (conductor, this session):**
- `bash test_system/check_layout.sh` → PASS (exit 0).
- `python3 -m pytest -q test_system/unit` → 39 passed (smoke tier; full
  13-min tier already confirmed 5/5 PASS by victor-reyes this session, not
  re-run per the task's standing instruction to trust that result).
- `grep -rn "subprocess.run(\['git'" src/` → zero hits (row 15 still holds).

**Fixes applied this release:** none beyond the version-bump/master-doc sync
above (§3) and archiving the old release note — all substantive fixes (rows
14, 15) landed in prior PRs #16/#18 and are described in §1, not re-applied
here.

## 5. Remaining open issues (deferred, not fixed in this release)

Carried on the board, not release-blockers for this GitHub tag:

- **Row 24 (MAJOR, attempted-and-reverted).** `src/utils/run_all.sh:163`
  still hardcodes `--distance_bin_size 500`, overriding row 14's new 2000 m
  default — the documented single-scenario pipeline still produces 500 m
  bins, not the bundle/paper's 2000 m. A fix was attempted in a worktree
  (dropping the explicit flag) but reverted: it broke
  `test_system/unit/test_run_all_pointer.py::test_run_all_seissol_fixture_matches_light_reference`
  because that test's frozen `light_reference/gm_statistics.npz` fixture is
  independently pinned at 500 m for row 13's fixture-vs-oracle chain, not for
  this reason. Regenerating that fixture under release time pressure risked
  breaking an already-proven chain, so the revert stands and the fix is
  deferred to either (a) passing `--distance_bin_size 2000` explicitly in
  `run_all.sh` (decouples from the row 5/13 fixture entirely) or (b)
  regenerating and re-blessing the fixture at 2000 m in its own commit per
  the stale-oracle lesson in `CLAUDE.md`. Pre-existing, not introduced this
  release. Out of scope for this release per explicit instruction — not
  attempted again here.
- **Row 25 (MAJOR).** `src/utils/seissol_converter_api.py:345-347` logs a
  failed `.h5` file and continues, silently dropping those stations from the
  scenario while still exiting 0; only an all-files-failed case stops it.
  Pre-existing.
- **Row 26 (MAJOR).** `src/utils/visualize_ensemble_stats.py:232,1018,1105`
  catches any `gm_statistics.npz` load failure with `except Exception:
  continue` and no message — Figs 13–19 can silently drop an ensemble member.
  Pre-existing.
- **Row 27.** The Streamlit explorer's email-webhook POST
  (`src/web/dr4gm_interactive_explorer.py` ~470-492) swallows all errors via
  a bare `except Exception: pass`, adjacent to but outside row 15's scope
  (row 15 covered only the git subprocess calls). Needs explicit inclusion in
  row 20 or its own row.
- **Row 10 (DOING).** Root-layout reorg is 4 of ~9 slices landed; the
  remaining slices (`test_system/`→`tests/`, `results/`→`runs/<date>_<slug>/`)
  are intentionally not rushed into this release.
- **Row 11 (BLOCKED(owner)).** Vendored `src/gmpe-smtk` test data exceeds the
  5 MB root cap (38.6 MB CSV, 19.6 MB HDF5) — untrack or LFS, owner decision
  pending. Not touched this release.
- **Row 12 (BLOCKED(owner)).** `gm_stats.py` Rjb 100 m floor (biases nearest
  bin ~25%) and `count<2` bin-drop behavior — changes published figures,
  needs a `gm_statistics.npz` regen plus a paper check. Not touched this
  release.
- **Row 23 (BLOCKED, unconfirmed cause).**
  `test_system/unit/test_streamlit_app.py::test_data_source_and_dataset_widgets_present_and_settable`
  failed reproducibly on a heavily loaded shared host (load average 33-40)
  with an AppTest 60 s timeout; GitHub's hosted runner stayed green on the
  same test at the commits in question. Not reproduced in this session's
  `test_system/unit` runs (39/39 passed, see §4).
- `CITATION.cff` ORCID is still the placeholder `0000-0000-0000-0000` —
  blocks Zenodo upload, not this GitHub release.
- README's Zenodo DOI is still the placeholder `XXXXXXX` — fill in after a
  Zenodo upload; blocks the "reproduce Figs 11-19 from Zenodo" README path
  for a stranger clone today (see §6).

## 6. Stranger-clone gate

Tested after the tag was created — see the release report for the exact
commands run, pass/fail per step, and SHA. Summary: `scripts/install.sh`'s
unconditional `pip3 install -r requirements.txt` (including
`openquake.engine`) is not what CI runs (CI installs
`requirements.txt` minus `openquake`, per `PROJECT_RULES.md` rule 4, because
`openquake.engine` needs system GDAL headers CI's runner doesn't have); a
stranger clone following `README.md`'s literal `source scripts/install.sh`
on a host without `libgdal-dev`/`gdal-config` hits the same failure CI was
fixed to avoid, and the README does not document the workaround. See the
release report for whether this reproduced on the gate host.

## 7. Assumptions used

- "The full regression tier" (`test_system/run_tests.sh`, needs 199 GB
  `reference/`) was not re-run in this session; victor-reyes's 5/5 PASS from
  this same v0.1.2 audit cycle (same tree hash range) is treated as current
  per the task's explicit instruction, consistent with rule 15a's "science
  sweep carries forward by tree hash" provision.
- The Zenodo data bundle (`dr4gm_data_v0.1.1.tar.gz`) is unchanged by this
  release (no NPZ data touched) and keeps its v0.1.1 name; this is a data
  artifact version, independent of the code's semver.
