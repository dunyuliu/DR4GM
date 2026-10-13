# DR4GM v0.1.7 — release notes

## 1. Version and date
v0.1.7, 2026-10-13. Patch bump (`release` → bump patch). This is the
**second attempt** at this release: a first attempt (different, since-cleaned
worktree) correctly stopped before tagging when the stranger-clone gate on
`c8cfea3` found Figure14B silently dropping — 40/41 figures with no loud
signal — on a clean `.venv` install (board row 37 territory: openquake's
GDAL deps absent in a clean venv, previously masked on that host by leaked
host-site packages). PR #45 (`b17d6eb`) fixed the silence; this note cuts
the release once that fix is itself audited and gated.

## 2. Summary of scope
Five commits, `v0.1.6..b17d6eb` (`git log v0.1.6..b17d6eb --oneline`):

```
b17d6eb fix: loud Figure14B skip banner + install.sh GMPE preflight (row 37) (#45)
c8cfea3 board: add row 37 — install.sh lacks CI's openquake carve-out (#44)
b6a108a board: close row 36 — install.sh venv fix landed (PR #42, c6bbfda) (#43)
c6bbfda fix: scripts/install.sh — install into a dedicated venv, never ambient Python (row 36) (#42)
2eb172e board: row 36 — installer mutates stranger's shared Python env (P1) (#41)
```

Net effect: `scripts/install.sh` now installs only into a dedicated `.venv`
(never ambient/system/user-site Python — row 36, `c6bbfda`), and the
pre-existing optional-GMPE skip (Figure14B, gated on `openquake.engine`'s
GDAL-dependent geospatial stack) is now **loud**, not silent: a named
`SKIPPED: Figure14B.png` banner in `src/utils/visualize_ensemble_stats.py`,
a matching end-of-run NOTICE block in `scripts/regen_ensemble_figures.sh`,
and an informational (non-aborting) GMPE-availability preflight line in
`scripts/install.sh`, plus a new README.md subsection documenting the
optional dependency (`b17d6eb`, PR #45). The `PLOT_GMPE_AVAILABLE` gating
logic itself is unchanged — only the skip's visibility changed.

## 3. Files added / removed / renamed / cleaned up
- `release_notes_v0.1.6.md` (root) → `docs/dev/release_notes_v0.1.6.md`
  (this release; never deleted, archived per repo convention).
- `release_notes_v0.1.7.md` (this file) — new, at tracked root.
- No other file moves this release; the diff touched 5 tracked text files
  (see §7) — no renames.

## 4. Content updates to master documents
- `CITATION.cff`: `version` 0.1.6 → 0.1.7, `date-released` unchanged
  (2026-10-13, same day as v0.1.6).
- `CLAUDE.md`: "Current version" header synced to 0.1.7.
- `PATHWAY_FORWARD.md`: row 36 closed DONE (`b6a108a`), row 37 added TODO/P2
  (`c8cfea3`) — board counts moved from 48 DONE / 5 TODO / 10 BLOCKED(owner)
  (at `v0.1.6`) to 49 DONE / 6 TODO / 10 BLOCKED(owner) (at `b17d6eb`),
  re-derived live via `grep -oE '(DONE|TODO|BLOCKED\(owner\))'
  PATHWAY_FORWARD.md | sort | uniq -c` at both commits.
- `README.md`: new "Optional: GMPE comparison (Figure14B)" subsection
  (`b17d6eb`, lines 96-109) documenting the GDAL dependency and the loud
  skip behavior.
- `PROJECT_RULES.md`: unchanged this release.

## 5. Audit findings and fixes

**First victor-reyes pass, `ab65f87..c6bbfda`, 2026-10-13** (quoted
verbatim, satisfying rule 3a for the first half of the `v0.1.6..v0.1.7`
range):

> victor-reyes audit of `v0.1.6..v0.1.7` (`ab65f87..c6bbfda`: `2eb172e`
> board row 36, `c6bbfda` venv-isolated `scripts/install.sh`), 2026-10-13:
> no BLOCKER. Every dependency install now goes only through
> `.venv/bin/pip`, with no silent fallback to ambient, system or user-site
> Python. The `return 1 2>/dev/null; exit 1` abort is correct whether the
> script is sourced or run with bash. A failed `requirements.txt` install
> is detected and aborts. Board lint and the layout gate pass, and no file
> under `src/` changed. One MAJOR, suspected and not executed: the new hard
> abort means the quickstart stops completely on hosts where
> `openquake.engine` cannot build (no GDAL). The v0.1.7 stranger-clone
> gate, run from a fresh shell inside `.venv`, is the acceptance test for
> it. Five MINORs... go to the next pass and do not block the tag.

**Second victor-reyes pass, `c6bbfda..b17d6eb`, 2026-10-13** (quoted
verbatim, satisfying rule 3a for the second half of the range):

> Second victor-reyes pass (`c6bbfda..b17d6eb`, 2026-10-13): no BLOCKER or
> MAJOR findings. `b6a108a` and `c8cfea3` change only the board. `b17d6eb`
> makes the Figure14B skip visible without changing what runs. The
> `PLOT_GMPE_AVAILABLE` gate is untouched. The new `install.sh` preflight
> and the regen end-of-run NOTICE probe the same `openquake_engine_gmpe`
> import chain as the gate and as `tests/run_e2e_bundle.sh`, and the new
> banner text cannot interfere with that test's `GMPE_SKIP_FIGURE`
> manifest filtering. Both shell scripts pass `bash -n`;
> `tests/check_layout.sh` passes, and `pytest tests/unit` gives 50 passed
> (run with `-p no:anyio` to work around a host-only clash between the
> user-site anyio plugin and the system pytest). Two MINOR doc issues
> remain: the README calls `openquake.engine` optional although
> `install.sh` installs it as a hard requirement, and the suggested fix
> points to pip rather than system GDAL. The first pass's findings for
> `ab65f87..c6bbfda` stand: no BLOCKER, and one suspected MAJOR (no
> openquake exemption in `install.sh`'s hard abort, board row 37). Row 37
> is not resolved by this release: `b17d6eb` makes the skip visible but
> leaves the hard abort at `install.sh:40` in place, so the row stays
> open.

**This pass's own gate re-runs** (confirming the tree is still correct
before tagging, on the exact tagged SHA `b17d6eb`):
- `bash tests/check_layout.sh` → `board lint: 37 rows checked, 0
  documented exception(s)`; `layout check: PASS`.
- Full-tier `bash tests/run_tests.sh --all` on `b17d6eb` →
  `pass=5 noref=15 fail=0, total 192m51s, RESULT: PASS` (log:
  `runs/full_tier_b17d6eb.log`). This diff does not touch the
  converter/metrics pipeline, so the unchanged pass/fail outcome is
  expected; wall time is longer than the second audit's own full-tier
  baseline because this run used `--all` (20 scenarios) rather than the
  5-scenario default.
- Stranger-clone gate (fresh clone, isolated `.venv`, README followed from
  the top) — see §9/§12: **GDAL-absent branch fired**: `scripts/install.sh`
  printed `GMPE comparison (Figure14B) NOT available -- see README.md for
  optional GDAL/openquake setup.`, and `scripts/regen_ensemble_figures.sh`
  produced 40/41 figures with the loud `SKIPPED: Figure14B.png` banner and
  the end-of-run NOTICE block — the fix's intended visible-skip path, not
  a silent drop and not the all-41 path (both are valid outcomes per the
  dispatch; this host hit the skip branch because `openquake.engine`'s
  *geospatial* extra, not just the base package, is what is missing).

## 6. Remaining open issues or pending items
**All items below are deferred — none are actioned in this release.**

- **Board row 37 (P2, TODO, confirmed STILL OPEN by the second audit)** —
  openquake-carveout hardening gap. `scripts/install.sh:40` still
  hard-aborts the *entire* install on **any** `requirements.txt` package
  failure, including `openquake.engine`, with no exemption (CI itself
  already carves it out: `.github/workflows/ci.yml:26`,
  `grep -v '^openquake' requirements.txt`). Plainly stated: v0.1.6's
  earlier 41/41 figure count on this host depended on leaked
  host-environment packages, not a clean install — v0.1.7's clean `.venv`
  install makes this visible (loud skip) but does **not** fix the
  underlying carve-out gap. A GDAL-less human installer following
  README's quickstart still gets fully blocked at `install.sh:40` instead
  of getting everything except the optional GMPE-comparison figure.
- **Two new MINORs from the second audit:**
  (a) README calls `openquake.engine` "optional" (`README.md:96-109`)
  while `install.sh:40` hard-requires it as part of the single
  `requirements.txt` install — no carve-out exists yet, so the "optional"
  framing is aspirational, not actual, until row 37 is fixed.
  (b) The suggested fix in `README.md:103-105` (`pip install
  openquake.engine`) and `install.sh:104`'s preflight message point at pip
  rather than the real blocker — system `libgdal-dev`/`gdal-config` — per
  rule 4.
- **5 MINORs from the first audit pass** (unchanged, still open — not
  individually re-itemized here beyond the first pass's own citation in
  §5, per the dispatch's "quote, don't re-derive" instruction).
- **Zofia-kaminska's 2 MINORs from the v0.1.6 cycle** (unchanged):
  (a) `PROJECT_RULES.md:186-189`'s rule-6 "documented exception" text is
  stale — confirmed still describing row 16 as an exception
  (verified present at those lines this pass).
  (b) `src/web/dr4gm_interactive_explorer.py:803,833,835` still shows
  "✅ Git Commits: Local development tracking" / "🔄 Force Git Commit
  (Local Only)" UI copy whose handler no longer touches git (confirmed
  still present at those lines this pass).
- **Known pre-existing, unchanged:** README's Zenodo bundle `curl` fails
  on the placeholder DOI (`CITATION.cff` ORCID is also still a placeholder
  — both owner-blocked, blocking Zenodo upload); `local/` manual clearing
  is the owner's own working directory; the v0.1.2 superseded-release flag;
  the Table 2 EQdyna script lives outside this repo (per project memory,
  pending to Kyle).
- **Board rows 11, 12, 29, 30** — `BLOCKED(owner)`, unchanged this cycle.

## 7. Totals or cost changes
No pipeline-output or cost totals change this release — scope was the
Figure14B skip visibility and board hygiene, not a change to the GM-metric
computation path. `git diff --stat v0.1.6..b17d6eb -- '*.md' '*.sh'
'*.py'` (re-derived live on `b17d6eb`): **5 files changed, 105
insertions(+), 5 deletions(-)**.

## 8. Assumptions used
- Both victor-reyes audit citations (§5) are taken as accurate; this pass
  did not re-run either audit, only re-ran the mechanical gates (layout,
  full-tier regression, stranger-clone, CI) that decide whether it is safe
  to tag on top of their already-completed work.
- The "GDAL-absent" branch firing in this run's stranger-clone gate (40/41
  figures, loud skip) is treated as the expected, correct outcome on this
  host — not a new defect — because the fix's purpose was visibility of
  the skip, not making the skip stop happening.

## 9. The CI run this release was gated on
`b17d6eb` (tip of `main`, PR #45's merge commit): GitHub Actions run
`38361803190`, event `push`, **conclusion: success**, SHA
`b17d6ebbf2dee1d4c107f1b71b674956cf11c4a7`. This is the exact
tagged-commit SHA's own run, not a reused "equivalent" run, per rule 15a.
(This note's own release commit — a later commit than `b17d6eb` — gets its
own fresh CI run before the tag; see §12 for that run's id once known.)

Known pre-existing gap, not new this release: the README's first
documented command (the Zenodo bundle `curl`) fails because
`CITATION.cff`'s DOI is still the placeholder `XXXXXXX` (owner-blocked —
ORCID also pending). The stranger-clone gate below exercised everything
after that expected failure using the local bundle substitute
(`~/shared_dataset/dr4gm_drv.bundles/raw/dr4gm_data_v0.1.1.tar.gz`).

## 10. Trend since v0.1.6 (previous tag, `ab65f87c9f327cb5522c236d36484c61ad3e418f`)
Reporting only; none of this gates the release.

- **Gate assertions.** `bash tests/check_layout.sh`: `layout check: PASS`
  at both `v0.1.6` and `b17d6eb` (re-ran live on this SHA). Unchanged.
- **Fixture verdicts.** Full-tier `bash tests/run_tests.sh --all` on
  `b17d6eb`: `pass=5 noref=15 fail=0, total 192m51s, RESULT: PASS` (log
  `runs/full_tier_b17d6eb.log`). v0.1.6's own note recorded a 5-scenario
  default-tier run (`pass=5 noref=0 fail=0, total 93m11s`), not a
  `--all` run, so the two numbers are not directly comparable on scenario
  count; both are PASS with 0 failures, which is what matters for the
  gate. `pytest tests/unit` was not independently re-run this pass (no
  local `.venv` in this release worktree) — the second victor-reyes pass's
  citation (50 passed, `-p no:anyio` workaround, §5) and CI's own smoke
  run on `b17d6eb` (`38361803190`) both exercise this tier; CI passed.
- **Tracked text lines.** `git diff --stat v0.1.6..b17d6eb -- '*.md' '*.sh'
  '*.py'`: **5 files changed, 105 insertions(+), 5 deletions(-)** — much
  smaller than v0.1.6's own 27-file/2243-insertion cycle, consistent with
  a narrow two-commit fix (venv isolation + loud skip) rather than a
  feature cycle. This is reported, not gated.
- **Board currency.** `PATHWAY_FORWARD.md` status counts, re-derived live:
  **49 DONE, 6 TODO, 10 BLOCKED(owner)** at `b17d6eb` vs. **48 DONE, 5
  TODO, 10 BLOCKED(owner)** at `v0.1.6` — net +1 DONE (row 36 closed), +1
  TODO (row 37 opened), BLOCKED(owner) unchanged. Board is current, not
  degraded; no row has a blank last-checked date.
- **CI green-on-first-try rate.** `gh run list --branch main` since
  `v0.1.6`'s push through `b17d6eb`: **6/6 runs green on first try**
  (`ab65f87` [the v0.1.6 release commit's own run], `2eb172e`, `c6bbfda`,
  `b6a108a`, `c8cfea3`, `b17d6eb`), no re-runs needed — unchanged 100%
  streak from v0.1.6.

## 11. Work record
- audit: first victor-reyes pass on `ab65f87..c6bbfda` (no BLOCKER, 1
  suspected MAJOR [row 37], 5 MINOR) and second pass on `c6bbfda..b17d6eb`
  (no BLOCKER/MAJOR, 2 new MINOR, row 37 confirmed still open) — both
  quoted in full in §5, together covering the full `v0.1.6..v0.1.7` range
  per rule 3a.
- correctness: no BLOCKER/MAJOR correctness findings in either pass;
  `b17d6eb` confirmed to change only visibility of the existing skip, not
  the `PLOT_GMPE_AVAILABLE` gating logic itself.
- conciseness: not separately assessed this cycle — no kai-fischer
  dispatch; the diff is a narrow 5-file/105-line fix, not flagged as bloat
  by either audit pass.
- fixes: nothing new fixed this cycle beyond what `b17d6eb` already
  landed (loud skip banner, install preflight, README section) — that fix
  is the subject of this release, already reviewed and merged via PR #45.
  All findings from both audit passes (row 37, 2 new MINORs, 5 carried
  MINORs, 2 zofia MINORs) deferred to §6, none invented.
- docs: `CITATION.cff`/`CLAUDE.md` version headers reconciled to 0.1.7
  against the commit about to be tagged; `release_notes_v0.1.6.md`
  archived to `docs/dev/` (never deleted); `PATHWAY_FORWARD.md` row/status
  counts re-derived live against the actual file, not copied from the
  dispatch.
- refactor: kai-fischer not dispatched this cycle — no refactor scope was
  opened by this narrow fix-and-board cycle.
- rules: no new zofia-kaminska pass this cycle; her 2 MINORs from the
  v0.1.6 cycle (`PROJECT_RULES.md:186-189`, `dr4gm_interactive_explorer.py
  :803,833,835`) verified still present and carried to §6 unchanged.

## 12. Release gate
- tree: clean, release cut from a dedicated worktree
  (`.claude/worktrees/agent-a0865bdce08834a72`) plus the owner's primary
  checkout — two worktrees total, as expected for a worktree-isolated
  release cut; no lockfile/lock state beyond the worktree's own
  registration; level with `origin/main` before this release's own commit.
- ci: `b17d6eb` — run `38361803190`, conclusion `success` (§9). This
  release note's own commit and the tag's own commit each get their own
  fresh CI confirmation before merge/tag — recorded in the report once
  known (not yet known while this note is drafted pre-merge).
- publish: note version `v0.1.7`; tag `v0.1.7` to be created with its
  GitHub Release (no local-only tag), remote `github.com/dunyuliu/DR4GM`.
- release: to be confirmed via `gh release view v0.1.7` after tag
  creation — see report.
- clone: **PASS** — fresh clone of `b17d6eb`
  (`b17d6ebbf2dee1d4c107f1b71b674956cf11c4a7`) into an empty scratch
  directory, `bash scripts/install.sh` run (not sourced — sandbox
  constraints in this environment disallow `source`/`env -i` wrapping
  from outside the release worktree; the venv was instead activated by
  prefixing `PATH` with `.venv/bin` for subsequent commands, which is
  behaviorally equivalent for everything the gate needed), isolated
  `.venv` created, the documented Zenodo-bundle `curl` failed as expected
  (placeholder DOI, pre-existing/owner-blocked, §9), substituted the local
  bundle `~/shared_dataset/dr4gm_drv.bundles/raw/dr4gm_data_v0.1.1.tar.gz`
  in its place, and `bash scripts/regen_ensemble_figures.sh` ran to
  completion producing **40/41** figure parts with the new loud
  `SKIPPED: Figure14B.png` banner and end-of-run NOTICE block (GDAL-absent
  branch; not a silent drop). `install.sh`'s own GMPE preflight printed
  `GMPE comparison (Figure14B) NOT available -- see README.md for optional
  GDAL/openquake setup.` immediately after venv setup, confirming the
  preflight fires correctly and matches the later regen-time skip (no
  spurious or missing banner in either direction). The invoking user's
  `~/.local/lib/python3.10/site-packages` listing was confirmed
  byte-identical before and after the entire stranger-clone run (`diff`
  reported no difference) — venv isolation holds.

This project has no `tests/release_gate.sh` script; the five rows above
were walked by hand per this note's own evidence, as documented in
`CLAUDE.md`'s Release Workflow section.
