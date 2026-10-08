# DR4GM Project Rules

Authoritative checklist used by the release workflow's audit step
(see `CLAUDE.md` → "Release Workflow" → step 4).

## Rules

1. **All tests must pass.** A release cannot ship while any test is
   failing. The audit step must run the project's tests and treat any
   failure as a blocking finding.

2. **Tracked root is a whitelist; no committed file over 5 MB.** The
   repo root may hold only: `README.md`, `CLAUDE.md`, `PATHWAY_FORWARD.md`,
   `PROJECT_RULES.md`, `LICENSE`, `CITATION.cff`, `requirements.txt`,
   `.gitignore`, `.github/` (CI workflows), and
   `release_notes_v<X.Y.Z>.md` (current release only — older ones archive
   to `docs/dev/`). Everything else moves under a directory:
   `FORMULAS.md` → `docs/user/FORMULAS.md`; audits and planning docs →
   `docs/dev/`; `utils/` → `src/utils/`; `gui/` → `src/gui/`; `web/` →
   `src/web/`; the vendored `gmpe-smtk/` (attribution files protected,
   see `CLAUDE.md`) → `src/gmpe-smtk/`; `test_system/` and any other
   test directory consolidated to one `tests/` (done, board row 10); the entry-point scripts
   (`install.sh`, `run_pipeline.sh`, `regen_ensemble_figures.sh`,
   `fetch_figures_for_publication.sh`, `make_zenodo_bundle.sh`) →
   `scripts/`; `reference/` stays in place on disk and is linked under
   `data/reference` (never copied); `data/` also holds small vendored
   assets (each < 5 MB, with an md5 + source URL recorded in
   `data/MANIFEST.md`) that production code loads directly, such as the
   Streamlit explorer's demo NPZs; `results/` output moves to
   git-ignored `runs/<YYYYMMDD>_<slug>/`. `local/` is being dissolved —
   nothing ships from it (it's gitignored and untracked), but the
   directory itself still has stale content on disk pending a manual
   owner clear (agents can't see or `rm` the maintainer's own working
   dir; see `PATHWAY_FORWARD.md` row 10). No other new file or
   directory lands at the root — see `CLAUDE.md`: "never create a new
   file at the repo root unless it ships publicly." No file tracked by
   git exceeds 5 MB.

   **Rationale**: a root that grows ad hoc is a root nobody can audit
   by eye before a public release, and a large binary committed by
   accident bloats every future clone.

   **How to apply**: run `tests/check_layout.sh` before a release
   (also wired into the release-workflow audit, step 4); it exits non-zero on any violation and is run first by
   `tests/run_tests.sh`; it never moves or deletes anything. Any new root entry needing to ship publicly is proposed
   here as a rule update, not added silently.

   **2a. The enforcement script, not this prose, is the gate.** As of
   this version, `tests/check_layout.sh` enforces the
   board/rules-at-root part of the set above (`PATHWAY_FORWARD.md`,
   `PROJECT_RULES.md` added to its whitelist in the same PR as this
   rule text), and now also the `tests/` part of the
   `tests/`/`src/`/`docs/`/`scripts/` consolidation (board row 10):
   its `ALLOWED` whitelist regex lists `tests`, not `test_system`, so a
   stray `test_system/` reappearing at root would fail the gate. The
   `src/`/`docs/`/`scripts/` parts of the same consolidation remain a
   stated target only, not yet asserted by `check_layout.sh`.

3. **The board is `PATHWAY_FORWARD.md` (tracked repo root).** Every open
   work item is a row with a status and an evidence command; /autopilot
   works only from it. Gate tiers (smoke / fast / full) are defined there.
   Dev cycle: feature branch → PR → CI green → merge to `main`; release
   tag only on green CI for that exact SHA plus a full-tier run and a
   stranger clone.

4. **CI installs `requirements.txt` minus `openquake`.** `openquake.engine`
   pulls GDAL as a transitive build dependency; the hosted CI runner has no
   system `libgdal-dev`/`gdal-config`, so `pip install -r requirements.txt`
   fails outright before any test runs. This is consistent with the rest of
   the codebase already treating `openquake` as optional and guarded
   (`PLOT_GMPE_AVAILABLE` in `src/utils/visualize_ensemble_stats.py`; the
   e2e suite's own documented Figure14B skip banner) — nothing in
   `tests/unit` or the smoke tier needs it. CI's install step must
   stay `grep -v '^openquake' requirements.txt | pip install -r /dev/stdin
   pytest` (or equivalent) — adding `pip install -r requirements.txt`
   verbatim to CI broke the smoke tier for 3 consecutive pushes on
   2026-10-08 (runs `37712597109`, `37712697030`, `37713635530`) before
   this was caught and fixed in `11191ab`. A full-tier/release run that
   genuinely needs GMPE comparison installs `openquake` separately in an
   environment with GDAL available (see `docs/dev/` release workflow),
   never via CI's smoke-tier step.

5. **Production code never commits to its own running repo.** No code
   path under `src/` may invoke `git add`, `git commit`, or `git push`
   against the working tree it executes inside, on a timer, a callback,
   or any other autonomous trigger. A human- or CI-triggered commit
   (the release workflow, a developer running it by hand) is fine; a
   long-running app committing its own output while serving users is
   not — removal, not relocation, is the fix when found.

   **Rationale**: an app that commits to whatever repo it happens to be
   checked out in writes to a target it does not control — a
   contributor's fork, a CI checkout, a clone with uncommitted local
   work — silently, outside any review or CI gate. A trailing `git
   push` compounds it by publishing without a human in the loop.

   **Incident (2026-10-08)**: `_commit_to_git()` in
   `src/web/dr4gm_interactive_explorer.py:495-530` runs `git add` +
   `git commit` + `git push` on the usage-analytics log on a timer,
   wrapped in a bare `except Exception: pass` — found during a
   board/rules scoping pass (board row 15), not from an incident report.

   **How to apply**: `grep -rn "subprocess.run(\['git'" src/` — any hit
   outside the release/CI tooling that is meant to commit (none exist
   under `src/` today; release commits are run by a human or CI, not by
   `src/` code) is a violation. Tier: mechanical via that grep; the
   check is now wired into `tests/check_layout.sh` (board row 15, DONE)
   so a reintroduced violation fails the gate directly, not just this grep.
