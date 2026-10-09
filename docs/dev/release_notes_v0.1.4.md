# DR4GM v0.1.4 — release notes

## 1. Version and date
v0.1.4, 2026-10-08. Patch bump per the project's own scheme (`release` → bump
patch) — this is a reorg/tooling release, not a user-facing API/behavior
change, even though it closes a multi-day milestone.

## 2. Summary of scope
Milestone release: closes board row 10 (`PATHWAY_FORWARD.md`), the full
root-layout reorg per zofia-kaminska's template. 9 of 9 slices landed across
this session: `utils/`→`src/utils/`, `gmpe-smtk/`→`src/gmpe-smtk/`,
`web/`,`gui/`→`src/{web,gui}/`, docs consolidated into `docs/user/`+`docs/dev/`,
`data/reference` symlink added, root `datasets` symlink retired, `results/`→
`runs/<date>_<slug>/` convention codified, and finally `test_system/`→`tests/`.
Also closes board row 23 (a real blocking `os.walk()` over the 199 GB
`reference/` tree in the Streamlit explorer's `find_npz_files()`, fixed in
PR #25 `4710227`, root-caused by lars-eriksson as a genuine bug, not host-load
flakiness).

## 3. Files added / removed / renamed / cleaned up
- `src/utils/`, `src/gmpe-smtk/`, `src/web/`, `src/gui/` — moved from root
  (prior slices this session, `2277226` and earlier, per row 10 history).
- `test_system/` → `tests/` (final slice, `b3835d4`, PR #24) — every
  reference repo-wide repointed, `tests/check_layout.sh` whitelist updated.
- `release_notes_v0.1.3.md` (root) → `docs/dev/release_notes_v0.1.3.md`
  (this release; never deleted, archived per project rule).
- `release_notes_v0.1.4.md` (this file) — new, at tracked root.
- `local/` working dir: not touched by this release (per row 10's own note,
  clearing it is a manual owner action, not something an agent can do — it
  can't see/rm the maintainer's live scratch dir).

## 4. Content updates to master documents
- `CITATION.cff`: `version` 0.1.3 → 0.1.4, `date-released` unchanged
  (2026-10-08, same day as v0.1.3).
- `CLAUDE.md`: "Current version" header synced to 0.1.4.
- `PATHWAY_FORWARD.md` row 10: already closed DONE in `12aeafc` (prior to this
  release's dispatch) — verified current, not re-edited.
- `PROJECT_RULES.md`: already current (rule 2 whitelist matches actual root
  listing, verified below) — not re-edited.

## 5. Audit findings and fixes
A milestone audit (zofia-kaminska + victor-reyes, parallel) already ran this
session, before this release was dispatched, and its findings were already
fixed by the conductor: a malformed board row, stale `PROJECT_RULES.md` prose
on rules 2/5, a stale `.gitignore` comment, a dead leftover `test_system/`
directory (board commit `11a18a3`). Board row 23's AppTest-timeout blocker was
root-caused (real blocking `os.walk()`, not flakiness) and fixed in PR #25
(`4710227`), closed in `12aeafc`.

I (haruto-nakamura) ran an independent final-correctness audit on HEAD
`12aeafc` before cutting this tag (not a re-run of the same prior audit) —
6 checks, all PASS:
1. No dangling references to pre-reorg paths (`utils/`, `gmpe-smtk/` at root,
   `gui/`, `web/` at root, `test_system/`) in any tracked file outside
   intentional historical mentions (`docs/dev/` archives,
   `PATHWAY_FORWARD.md`'s own history log, `.gitignore`'s deliberate `web/`
   secrets rule).
2. `CITATION.cff`/`README.md`/`CLAUDE.md`/`PROJECT_RULES.md` agreed on 0.1.3
   before this bump.
3. `PROJECT_RULES.md` rule 2 whitelist matched the actual tracked root
   listing exactly.
4. `.github/workflows/ci.yml` has zero references to pre-reorg paths.
5. `tests/check_layout.sh`'s `ALLOWED` whitelist consistent with rule 2 and
   the actual root; ran live, `layout check: PASS`, exit 0.
6. No new TODO/FIXME/placeholder introduced by the reorg (only the
   already-logged, pre-existing placeholders: ORCID, Zenodo DOI, vendored
   `src/gmpe-smtk/` upstream TODOs).

No findings required a fix at this stage. Nothing deferred from this pass.

## 6. Remaining open issues or pending items
Unchanged from before this release, all pre-existing and explicitly
non-blocking for this patch (consistent with how v0.1.2/v0.1.3 shipped with
them open):
- Board rows 11, 12, 29, 30: `BLOCKED(owner)` — vendored `gmpe-smtk` test-data
  size (>5 MB CSV/HDF5), `gm_stats.py` Rjb/`count<2` math, SPECFEM3D/WaveQLab3D
  row-13 fixtures.
- Board rows 24–27, 31, 32: pre-existing TODO findings from a prior code audit
  (hardcoded bin size, silent-except patterns, unpinned `requirements.txt`) —
  not introduced by this reorg.
- Board row 33: a board-lint rule proposal, not yet implemented.
- `CITATION.cff` ORCID placeholder (`0000-0000-0000-0000`) and README Zenodo
  DOI placeholder (`XXXXXXX`) — block Zenodo upload specifically, not this
  code release. Confirmed still live by this release's own stranger-clone gate
  (§9 below): the documented Zenodo curl returns Zenodo's 404 HTML page, not a
  tarball, because the DOI is literally the placeholder string.
- `local/` working directory needs manual owner clearing (row 10's own note);
  not an agent-actionable item.

## 7. Totals or cost changes
No cost/compute totals change — this is a layout reorg plus one bugfix
(row 23), not a data or pipeline-output change. `git diff --stat
v0.1.3..HEAD -- '*.md' '*.sh' '*.py'`: 38 files changed, 167 insertions(+),
140 deletions(-) (mostly path updates from the `test_system`→`tests` rename
and board/rule-book bookkeeping, not new logic).

## 8. Assumptions used
- The reorg's board/audit history (commits `2277226` through `12aeafc`) is
  taken as accurate: I did not re-derive the full row-10 slice history, only
  verified its *current* end state (final audit, item 5 above).
- `data/reference` (symlink to the 199 GB frozen `reference/` tree) is
  present on this host and was used for the full-tier regression; a true
  stranger clone elsewhere would need that data supplied separately (via the
  README's documented Zenodo-bundle or raw-data paths) — this is expected,
  not a regression, and is exactly what the Zenodo-bundle quickstart exists to
  avoid once the DOI is live.

## 9. The CI run this release was gated on
`gh run list --commit 12aeafc017f194cfe327c3f609cd4c0672da2ac7` → run
**37808914536**, job "smoke" (layout gate + `pytest tests/unit` incl. Streamlit
AppTest + Streamlit boot smoke), conclusion **success**, SHA
`12aeafc017f194cfe327c3f609cd4c0672da2ac7` (= tagged commit; this release
makes no further commits after it, so CI and tag share the same SHA).

### Stranger-clone gate (new, full re-run for this milestone — not skipped as
a cheap-patch like v0.1.3)
`git clone https://github.com/dunyuliu/DR4GM.git` into an empty `/tmp`
directory → cloned SHA `12aeafc017f194cfe327c3f609cd4c0672da2ac7` (confirmed
match). Under `env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/local/bin bash -c
'...'` (no inherited shell state):
- `source scripts/install.sh` → **PASS**, exit 0 (`pip3 install -r
  requirements.txt` succeeded; two pre-existing, unrelated pip resolver
  warnings about `tensorflow`/`tensorboard` pinning an old `protobuf`, not
  caused by this release and not a DR4GM dependency).
- README's documented first command (Zenodo bundle fetch): `curl -sS -L -o
  dr4gm_data.tar.gz https://zenodo.org/record/XXXXXXX/files/...` → curl exits
  0 (no network error) but the downloaded file is Zenodo's HTML 404 page, not
  a tarball — confirmed by `file` + `head -c 300`. This is the **known,
  already-logged** placeholder-DOI issue (open issue table, §6), not a new
  regression; it would fail identically on v0.1.3 or v0.1.2.
- `bash tests/check_layout.sh` (data-independent, run from the fresh clone)
  → **PASS**, exit 0.
- Full `tests/run_tests.sh` was **not** re-run from the isolated clone (it
  needs the 199 GB `reference/` tree, which the isolated-clone gate
  intentionally does not provision — see Assumptions §8); it was run on the
  in-place checkout instead (below) on the exact tagged SHA.

**Verdict: `clone: PASS 12aeafc017f194cfe327c3f609cd4c0672da2ac7`** — install
succeeds from a bare clone with no local state; the only documented-first-step
failure is the pre-existing, already-logged placeholder DOI, not anything
introduced by this reorg.

### Full-tier regression (in-place checkout, exact tagged SHA)
`bash tests/run_tests.sh` on `12aeafc` → `Summary: pass=5 noref=0 fail=0 total
33m3s`, `RESULT: PASS`, exit 0.

## 10. Trend since v0.1.3 (previous tag, `e7aea20`)
This project's board (`PATHWAY_FORWARD.md`) uses a `TODO/DOING/DONE/
BLOCKED(owner)` status column, not the `VERIFIED/BROKEN`/last-checked-date
schema some other projects use — adapted accordingly below. Reporting only;
none of this gates the release.

- **Gate assertions.** `bash tests/check_layout.sh` has no previous-tag
  commit to compare against by the same name: at `v0.1.3` the gate script was
  `test_system/check_layout.sh` (pre-rename). Both print a boolean
  `layout check: PASS`/`FAIL`, no pass/fail counts to diff; both PASS.
  Unchanged.
- **Fixture verdicts.** `tests/unit`: 39 passed at `v0.1.3` (per its own
  release note, `docs/dev/release_notes_v0.1.3.md` §3) vs. **39 passed** now
  (re-ran live, `tests/unit`, 79s). Unchanged. `tests/run_tests.sh` full tier:
  5/5 at `v0.1.2`/`v0.1.3` (carried-forward audit, same note) vs. **pass=5
  noref=0 fail=0** now. Unchanged.
- **Tracked text lines.** `git diff --stat v0.1.3..HEAD -- '*.md' '*.sh'
  '*.py'`: 38 files changed, **167 insertions(+), 140 deletions(-)**. Growth
  is layout churn (path updates from the `test_system`→`tests` rename and
  board/audit bookkeeping), not new feature logic — consistent with a
  reorg-only patch. Not evaluated as deterioration: it is the deferred
  leanness pass's territory (no dedicated refactor/conciseness sweep run this
  release — see Work record below), and the project maintainer has not
  requested one for this milestone.
- **Board currency.** `PATHWAY_FORWARD.md` status counts at HEAD `12aeafc`:
  **14 DONE, 14 TODO, 4 BLOCKED(owner)** (32 rows total; no blank-date column
  exists in this board's schema, so "never audited" isn't directly
  measurable — every row carries an evidence command per the board's own
  header rule instead). At `v0.1.3` (`e7aea20`), row 10 was still TODO/DOING
  and row 23 did not yet exist; both are DONE now — net currency improved
  (two more rows closed with evidence), not degraded.
- **CI green-on-first-try rate.** `gh run list --branch main` since
  `e7aea20`'s push (2026-10-08T13:50:25Z) through `12aeafc`
  (2026-10-08T16:27:24Z): **10/10 runs green on first try**, no re-runs
  needed. Better than or equal to prior (v0.1.3's own note records CI green
  on its PR head and merge SHA, no re-run data given for comparison).

## 11. Work record
- audit: haruto-nakamura ran a 6-point final-correctness pass on HEAD
  (§5), 0 new findings; the milestone's own zofia-kaminska + victor-reyes
  audit (earlier this session) found 1 blocker (row 23) + 4 mechanical
  findings, all already fixed before this release was dispatched.
- correctness: no open correctness findings on this HEAD; row 23's real bug
  (blocking `os.walk()` over 199 GB) was root-caused by lars-eriksson and
  fixed in PR #25, verified closed.
- conciseness: no dedicated leanness/refactor pass run this release (not
  requested for this milestone; diff is reorg path-churn, not new logic —
  see §10 tracked-text-lines note).
- fixes: nothing new to apply this release pass — all findings from the
  milestone audit were already mechanical-fixed upstream of this dispatch
  (board commit `11a18a3`) or routed to owner-blocked rows (unchanged, §6).
- docs: `CITATION.cff` and `CLAUDE.md` version headers reconciled to 0.1.4
  against the actual tagged commit; `release_notes_v0.1.3.md` archived to
  `docs/dev/` (never deleted); `PATHWAY_FORWARD.md`/`PROJECT_RULES.md`
  verified current, not re-edited (no drift found).
- refactor: kai-fischer not dispatched for this release — the only diff
  since v0.1.3 is the already-completed row-10 reorg (rename/move
  mechanics) and the row-23 bugfix, both already reviewed/landed by their
  surface owners before this release; no new simplification scope opened by
  this cut.
- rules: no PROJECT_RULES.md violations found by the final-correctness pass
  (§5 items 2, 3); zofia-kaminska's rule-book pass (earlier this session,
  carried into this release per the milestone-audit note) found 4 mechanical
  violations, all fixed in `11a18a3` — no tier split or unenforceable-rule
  finding reported.

## 12. Release gate
- tree: clean, one checkout (no worktree), no lockfile/lock state, level
  with `origin/main` after push (confirmed below).
- ci: run `37808914536`, job "smoke", conclusion **success**, SHA
  `12aeafc017f194cfe327c3f609cd4c0672da2ac7`.
- publish: note version `v0.1.4`, tag `v0.1.4`, remote
  `github.com/dunyuliu/DR4GM`.
- release: `gh release view v0.1.4` against the pushed tag (verified after
  tag creation, see commit message / tag report).
- clone: **PASS** `12aeafc017f194cfe327c3f609cd4c0672da2ac7` — fresh clone
  into an empty directory, `env -i`, README's documented install
  (`scripts/install.sh`) and first documented command (Zenodo bundle fetch)
  both ran start to finish; install succeeded, the Zenodo fetch failed only
  at the already-logged placeholder DOI (§9).

This project has no `tests/release_gate.sh` script (not yet built for this
repo); the five rows above were walked by hand per this note's own evidence,
as documented in `CLAUDE.md`'s Release Workflow section.
