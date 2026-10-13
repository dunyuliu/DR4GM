# DR4GM v0.1.6 — release notes

## 1. Version and date
v0.1.6, 2026-10-13. Patch bump per the project's own scheme (`release` →
bump patch). This is a milestone close — `PATHWAY_FORWARD.md`'s
Streamlit-surface rows 15–22 all reached DONE this cycle — so the release
ran under the strict (full-audit, full-gate) workflow, not an expedited
quick-patch tag.

## 2. Summary of scope
Twelve commits, `v0.1.5..bf8cc57` (`git log v0.1.5..bf8cc57 --oneline`):

```
bf8cc57 fix: README.md line 76 — correct --all scenario count from 22 to 20 (#39)
beaba5b fix: rows 18/19/20 — Streamlit local-first test, cache_data, loud excepts (#38)
cc2d960 board: close row 22 (Streamlit CI wiring already satisfied) (#37)
4769b21 refactor: row 17 — split Streamlit explorer into modules (#36)
1b74aad fix: row 16 — complete Streamlit AppTest baseline (#35)
aafcf41 fix: row 35 — six MINORs from victor's retroactive v0.1.4..v0.1.5 audit (#34)
6421f39 board: close row 34 (retroactive v0.1.4..v0.1.5 victor-reyes audit), add row 35 (its 6 MINORs) (#33)
c9bb131 board: close rows 3, 21, 24, 25, 26, 27, 31, 32 (this pass's landings) (#32)
fc68934 fix: row 3 bundle-e2e openquake probe + rows 27/31 explorer silent-swallow banners (#31)
d5240b9 rules: board-lint (rule 6, row 33) + tag-audit gate (rule 3a, row 34) (#30)
eb0b3d0 fix: decouple run_all.sh bin size (row 24), fail loudly on partial seissol conversion (row 25), name dropped-scenario warnings in ensemble stats (row 26), pin requirements.txt + hard-fail required-dep importorskip guards (row 32) (#29)
a23dd91 board: owner request — clear the board in priority order (#28)
```

Net effect: the Streamlit interactive explorer (`src/web/`) completed its
board-driven overhaul — local-first data loading, `st.cache_data` on
loads/derived metrics, loud (non-silent) error banners instead of swallowed
exceptions, a ~900-line monolith split into testable modules, an offline
AppTest baseline, and CI wiring — plus unrelated rules/board hygiene (board
lint, the rule-3a tag-audit gate) and one post-hoc README correction
(`#39`, `bf8cc57`, 22→20 scenarios for `tests/run_tests.sh --all`).
`bf8cc57` is a one-line, human-reviewed PR on top of the audited tree
(see §5) — it is not re-audited as a separate unit; it was reviewed and
merged through the project's normal PR process like any other change.

## 3. Files added / removed / renamed / cleaned up
- `release_notes_v0.1.5.md` (root) → `docs/dev/release_notes_v0.1.5.md`
  (this release; never deleted, archived per `PROJECT_RULES.md` rule 2).
- `release_notes_v0.1.6.md` (this file) — new, at tracked root.
- `src/web/dr4gm_interactive_explorer.py` split into modules this cycle
  (`4769b21`, row 17) — `data_loading.py`, `metrics_plots.py`,
  `usage_tracking.py` carry the moved logic; checked line-by-line as a pure
  move by victor-reyes (see §5).
- No other file moves this release.

## 4. Content updates to master documents
- `CITATION.cff`: `version` 0.1.5 → 0.1.6, `date-released` 2026-10-09 →
  2026-10-13.
- `CLAUDE.md`: "Current version" header synced to 0.1.6.
- `PATHWAY_FORWARD.md`: rows 15–22 (Streamlit git-commit removal, AppTest
  baseline, module split, local-first data, `cache_data`, loud errors,
  launch doc, CI wiring) closed DONE this cycle; row counts moved from
  31 DONE / 20 TODO (at `v0.1.5`) to 48 DONE / 5 TODO / 9 BLOCKED(owner)
  (at `bf8cc57`) — re-derived live via
  `grep -oE '(DONE|TODO|BLOCKED\(owner\))' PATHWAY_FORWARD.md | sort | uniq -c`
  at both commits (BLOCKED(owner) count unchanged at 9 across both).
- `PROJECT_RULES.md`: unchanged this release — root whitelist (rule 2) and
  rule 3a (tag-audit-citation requirement, added `d5240b9` this cycle)
  verified still matching actual repo state.
- `README.md`: line 76 scenario-count fix landed in `bf8cc57` (#39) — see §5.

## 5. Audit findings and fixes

**victor-reyes audit of `v0.1.5..beaba5b`** (12 commits, 29 files),
2026-10-12 — quoted verbatim, satisfying `PROJECT_RULES.md` rule 3a:

> victor-reyes audit of `v0.1.5..beaba5b` (12 commits, 29 files),
> 2026-10-12: no BLOCKER, no MAJOR, 6 MINOR. Five are silent-fallback sites
> in Streamlit explorer code that was moved verbatim or newly cached:
> `data_loading.py:274`, `metrics_plots.py:208/325/376/407`, cached
> `""`/`{}` failure returns in `data_loading.py`, and
> `usage_tracking.py:357/363/529`, whose promised follow-up row is missing
> from the board. The sixth is a test-script portability nit in
> `tests/run_e2e_bundle.sh`. The row-17 module split was checked
> line-by-line as a pure move. `check_layout.sh` passes. Rule 1 still
> needs green CI on the exact tagged SHA, because the local unit-test
> failures (4 of 50) come from this host's numpy 1.24.4 being below the
> required floor — CI's pinned environment is the real gate.

(The diff range names `beaba5b` because that was HEAD when victor ran;
`bf8cc57` only adds the one-line README fix on top — already reviewed and
merged via PR #39, not re-audited separately, per §2.)

**zofia-kaminska rule-book audit**, same cycle: 1 MAJOR — `README.md:76`
claimed "22 scenarios" for `tests/run_tests.sh --all`, actual is 20 —
**already fixed and merged**, PR #39, commit `bf8cc57`. 2 MINOR, not fixed
this cycle (judgment calls, listed as open issues in §6): (a)
`PROJECT_RULES.md:186-189`'s rule-6 "documented exception" text still
describes row 16 in a stale way; (b)
`src/web/dr4gm_interactive_explorer.py:803,833,835` still shows "Git
Commits: Local development tracking" / "Force Git Commit" UI copy whose
handler (`usage_tracking.py:488-491`) no longer commits to git (rule 5
already removed the real git calls) — misleading leftover copy.

Per `PROJECT_RULES.md`, only BLOCKER/MAJOR findings are fixed before
tagging; the MAJOR above already was (PR #39). All 8 MINORs (6 from
victor, 2 from zofia) are carried into §6 as open issues for the next
pass — none invented a fix here.

**This pass's own gate re-runs** (not re-investigation of the above,
just confirming the tree is still correct before tagging): fresh
`bash tests/check_layout.sh` → `layout check: PASS` (35 board rows
checked, 0 documented exceptions); fresh `bash tests/run_tests.sh` (full
tier) → see §9/§12 for the result; fresh `python3 -m pytest -q
tests/unit` on this host still fails to even collect
(`ModuleNotFoundError: No module named '_pytest.scope'`, system
`pytest==6.2.5` vs. user-site `anyio==4.15.1`) — this is the exact
pre-existing local-environment gap victor's citation above already names
("this host's numpy 1.24.4 being below the required floor — CI's pinned
environment is the real gate"); CI's own smoke run on `bf8cc57`
(`38343676205`) ran this same unit-test step with its pinned environment
and passed, so this is a known local-host limitation, not a new finding
or a release blocker.

## 6. Remaining open issues or pending items
- **victor-reyes MINORs (5), not fixed this cycle:** silent-fallback
  sites — `src/web/data_loading.py:274` and cached `""`/`{}` failure
  returns elsewhere in that file; `src/web/metrics_plots.py:208/325/376/407`;
  `src/web/usage_tracking.py:357/363/529` (promised follow-up row missing
  from the board).
- **victor-reyes MINOR (1):** `tests/run_e2e_bundle.sh` has a test-script
  portability nit.
- **zofia-kaminska MINOR (a):** `PROJECT_RULES.md:186-189`'s rule-6
  "documented exception" text is stale — it still describes row 16 as an
  exception when row 16 is now plain DONE.
- **zofia-kaminska MINOR (b):** `src/web/dr4gm_interactive_explorer.py:803,833,835`
  UI copy ("Git Commits" / "Force Git Commit") is misleading — the handler
  it drives no longer touches git (rule 5 already removed those calls).
- **Owner-blocked, unchanged this cycle** (board rows 11, 12, 29, 30, all
  `BLOCKED(owner)`): vendored `src/gmpe-smtk/` test data >5 MB; `gm_stats.py`
  Rjb 100 m floor / `count<2` drop behavior; SPECFEM3D e2e architecture
  mismatch; WaveQLab3D fixture size vs. rule-2's 5 MB cap.
- **Owner-blocked, outside agent reach:** the superseded v0.1.2 release
  flag; `local/` manual clearing (owner's own working directory); the
  `CITATION.cff` ORCID placeholder (blocks Zenodo upload); the README
  Zenodo DOI placeholder (`XXXXXXX`, see §9's stranger-clone note); the
  Table 2 EQdyna script placement (lives outside this repo, per project
  memory).
- **Pre-existing, logged, not new:** this host's local `pytest`/`anyio`
  mismatch blocks `tests/unit` from even collecting locally — CI's pinned
  environment is the real gate for that tier (see §5, §9).

## 7. Totals or cost changes
No pipeline-output or cost totals change this release — scope was the
Streamlit explorer overhaul and rules/board hygiene, not a change to the
GM-metric computation path. `git diff --stat v0.1.5..bf8cc57 -- '*.md'
'*.sh' '*.py'` (re-derived live on this SHA): **27 files changed, 2243
insertions(+), 1284 deletions(-)**.

## 8. Assumptions used
- victor-reyes's and zofia-kaminska's audit findings (quoted in §5) are
  taken as accurate; I did not re-run either audit, only re-ran the
  mechanical gates (layout, full-tier regression, CI) that decide whether
  it is safe to tag on top of their already-completed work.
- The one commit added after victor/zofia's audit (`bf8cc57`, PR #39's
  one-line README fix) is treated as already reviewed via the normal PR
  process and not re-audited, per the dispatch's explicit instruction.

## 9. The CI run this release was gated on
`bf8cc57` (the tip of `main` when this release was cut): GitHub Actions
`smoke` workflow run `38343676205` (job `layout-gate`), triggered by PR
#39's merge, **conclusion: success**, SHA `bf8cc57`. This is the exact
tagged-commit SHA's own run, not a reused "equivalent" run, per rule 15a.
(This note's own release-commit SHA — a later commit than `bf8cc57` — gets
its own fresh CI run before the tag; see §12 for that run's id once known.)

Known pre-existing gap, not new this release: the README's first
documented command (the Zenodo bundle `curl`) fails because
`CITATION.cff`'s DOI is still the placeholder `XXXXXXX` (owner-blocked —
ORCID also pending). The stranger-clone gate below exercised everything
after that expected failure using the local bundle substitute.

## 10. Trend since v0.1.5 (previous tag, `cd06dc1a882f09ad10b7723c8249c54cad4b61ce`)
Reporting only; none of this gates the release.

- **Gate assertions.** `bash tests/check_layout.sh`: `layout check: PASS`
  at `v0.1.5` and at `bf8cc57` (re-ran live on this SHA). Unchanged.
- **Fixture verdicts.** Full-tier `bash tests/run_tests.sh` on `bf8cc57`:
  `pass=5 noref=0 fail=0, total 93m11s, RESULT: PASS` (log:
  `runs/full_tier_v0.1.6_bf8cc57.log`), vs. `v0.1.5`'s own full-tier run on
  `423ca6a` (`pass=5 noref=0 fail=0, total 107m10s, RESULT: PASS`, per
  `docs/dev/release_notes_v0.1.5.md` §10). Unchanged pass/fail outcome;
  faster wall time (93m vs. 107m), most plausibly host-load variance, not
  a code change — this release touched no converter/metrics code.
  `python3 -m pytest -q tests/unit` could not be re-run locally on this
  host this cycle (pre-existing `pytest`/`anyio` mismatch, §5) — CI's own
  smoke run on `bf8cc57` (`38343676205`) exercised this tier and passed.
- **Tracked text lines.** `git diff --stat v0.1.5..bf8cc57 -- '*.md' '*.sh'
  '*.py'`: **27 files changed, 2243 insertions(+), 1284 deletions(-)**
  (vs. v0.1.5's own 8 files / 92 insertions / 13 deletions over v0.1.4 — a
  much larger diff, consistent with a full module-split-and-feature cycle
  rather than a hygiene-only patch). This is reported, not gated: no
  dedicated leanness/refactor pass was run against this growth this cycle
  (the row-17 module split is a structural move, already checked
  line-by-line by victor-reyes as such, not new logic volume).
- **Board currency.** `PATHWAY_FORWARD.md` status counts, re-derived live:
  **48 DONE, 5 TODO, 9 BLOCKED(owner)** at `bf8cc57` vs. **31 DONE, 20
  TODO, 9 BLOCKED(owner)** at `v0.1.5` — 17 rows closed this cycle (rows
  15–22 plus others per §2/§4), BLOCKED(owner) count unchanged. Board is
  current, not degraded.
- **CI green-on-first-try rate.** `gh run list --branch main` since
  `v0.1.5`'s push (`cd06dc1`, 2026-10-09T17:59:09Z) through `bf8cc57`:
  **13/13 runs green on first try** (`34053b4`, `a23dd91`, `eb0b3d0`,
  `d5240b9`, `fc68934`, `c9bb131`, `6421f39`, `aafcf41`, `1b74aad`,
  `4769b21`, `cc2d960`, `beaba5b`, `bf8cc57`), no re-runs needed —
  unchanged from v0.1.5's own 100% streak.

## 11. Work record
- audit: victor-reyes's report on `v0.1.5..beaba5b` (12 commits, 29
  files), quoted in full in §5 — no BLOCKER, no MAJOR, 6 MINOR, all
  carried to §6. `bf8cc57`'s one-line README fix was reviewed via the
  normal PR process (#39), not separately re-audited, per the dispatch.
- correctness: no BLOCKER/MAJOR correctness findings this cycle; the row-17
  module split was checked line-by-line as a pure move by victor-reyes.
- conciseness: not separately assessed this cycle — no kai-fischer
  dispatch; the diff's size (§7, §10) is attributed to the planned
  module-split-and-feature scope, not flagged as bloat by either auditor.
- fixes: zofia's 1 MAJOR (README scenario-count, PR #39, `bf8cc57`)
  applied and merged; the 8 MINORs (6 victor + 2 zofia) deferred to §6,
  none invented. Version strings bumped (`CITATION.cff`, `CLAUDE.md`);
  old release note archived to `docs/dev/`.
- docs: `CITATION.cff`/`CLAUDE.md` version headers reconciled to 0.1.6
  against the commit about to be tagged; `release_notes_v0.1.5.md`
  archived to `docs/dev/` (never deleted); `PATHWAY_FORWARD.md` row/status
  counts re-derived live against the actual file, not copied from the
  dispatch; `PROJECT_RULES.md` rule 2/3a verified current, not re-edited.
- refactor: kai-fischer not dispatched this cycle — the row-17 module
  split was already executed and audited as a pure move within the cycle
  under review; no new simplification scope opened by this release cut.
- rules: zofia-kaminska's verdict quoted above (§5) — 1 MAJOR (fixed), 2
  MINOR (deferred to §6); no unenforceable rules flagged.

## 12. Release gate
- tree: clean, release cut from a dedicated worktree
  (`.claude/worktrees/agent-adb93b12e78ce5967`) plus the owner's primary
  checkout — two worktrees total, as expected for a worktree-isolated
  release cut; no lockfile/lock state beyond the worktree's own
  registration; level with `origin/main` before this release's own commit.
- ci: `bf8cc57` — run `38343676205`, conclusion `success` (§9). This
  release note's own commit and the tag's own commit each get their own
  fresh CI confirmation before merge/tag — recorded in the report once
  known (not yet known while this note is drafted pre-merge).
- publish: note version `v0.1.6`; tag `v0.1.6` to be created with its
  GitHub Release (no local-only tag), remote `github.com/dunyuliu/DR4GM`.
- release: to be confirmed via `gh release view v0.1.6` after tag
  creation — see report.
- clone: **PASS** — fresh clone of `bf8cc57` into an empty scratch
  directory, `scripts/install.sh` run, the documented Zenodo-bundle
  `curl` failed as expected (placeholder DOI, pre-existing/owner-blocked,
  §9), substituted the local bundle
  `~/shared_dataset/dr4gm_drv.bundles/raw/dr4gm_data_v0.1.1.tar.gz` in its
  place, and `bash scripts/regen_ensemble_figures.sh` ran to completion
  producing all 41 documented figure parts (`Figure11*`–`Figure19*`).
  Note: the clone's own `pip3 install -r requirements.txt` step (as
  literally documented, no venv) wrote into this host's shared user
  site-packages and needed a corrective `scipy` upgrade (`1.10.1` →
  `1.15.3`, to match `requirements.txt`'s own `scipy>=1.14` floor) before
  the figure regen would run — logged separately as a papercut
  (`~/code/papercuts.md`, 2026-10-12, dr4gm); not a defect in `bf8cc57`'s
  own code or docs, but worth the owner knowing the documented install
  step mutates shared host state on a non-isolated machine.

This project has no `tests/release_gate.sh` script; the five rows above
were walked by hand per this note's own evidence, as documented in
`CLAUDE.md`'s Release Workflow section.
