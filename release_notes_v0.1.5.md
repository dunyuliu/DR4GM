# DR4GM v0.1.5 — release notes

## 1. Version and date
v0.1.5, 2026-10-09. Patch bump per the project's own scheme (`release` →
bump patch) — an owner-directed data-hygiene and git-history-scrub release,
no user-facing pipeline/API change.

## 2. Summary of scope
Closes the 2026-10-09 owner decision logged in `PATHWAY_FORWARD.md` ("data
can go to shared. retire parent." / "don't leak" / "this is only computing
project" / "make sure history no leak"):
- `data/reference` repointed from a dead relative symlink to a portable,
  gitignored link sourced from `$DR4GM_REFERENCE_DIR` (default
  `$HOME/shared_dataset/dr4gm_drv.reference`), created by `scripts/install.sh`.
  Every consumer updated: `tests/run_tests.sh`, `tests/derive_light_reference.sh`,
  `CLAUDE.md`, `README.md`, `scripts/install.sh`.
- A tree + full git-history leak sweep (conductor + anya-petrov), purging via
  `git filter-repo`: the tracked manuscript
  `paper/P28b_DR4GM_manuscript_v0.docx` (all history), an internal path in
  `docs/dev/SESSION_LOG_2026-10-07_reorg-tests-ci.md`, a personal (non-
  institutional) email in `src/web/dr4gm_interactive_explorer.py` (now the
  institutional `dliu@ig.utexas.edu`), and an internal path example in
  `tests/derive_light_reference.sh`.
- A one-line doc wording fix in `docs/dev/PUBLISH_AUDIT.md` (cosmetic).

History was rewritten and force-pushed on `main` (`3185c9a` → `423ca6a`);
tags `v0.1.1`–`v0.1.4` were deleted and re-pushed pointing at their
remapped commits. `285bfc9` (prior tip, board-only) records this decision
and closure.

## 3. Files added / removed / renamed / cleaned up
- `paper/P28b_DR4GM_manuscript_v0.docx` — removed from the tree **and** all
  git history (`git filter-repo`), per the owner's "this is only computing
  project" / "make sure history no leak" directives.
- `release_notes_v0.1.4.md` (root) → `docs/dev/release_notes_v0.1.4.md` (this
  release; never deleted, archived per project rule).
- `release_notes_v0.1.5.md` (this file) — new, at tracked root.
- No other file moves this release.

## 4. Content updates to master documents
- `CITATION.cff`: `version` 0.1.4 → 0.1.5, `date-released` 2026-10-08 →
  2026-10-09.
- `CLAUDE.md`: "Current version" header synced to 0.1.5; `data/reference`
  symlink mechanics documented (prior-session commit `423ca6a`, carried
  into this release unmodified).
- `PATHWAY_FORWARD.md`: already closed by the conductor in `285bfc9`
  ("Closed 2026-10-09" sub-bullets) before this release was dispatched —
  verified current below, not re-edited.
- `PROJECT_RULES.md`: unchanged this release — verified rule 2 whitelist
  still matches the actual tracked root listing (checked below).

## 5. Audit findings and fixes
This release's scope was handed to me pre-settled by the conductor
(findings 1–6 of the dispatch); I did not re-investigate items already
closed and verified with evidence, only re-ran the gates that decide
whether it's safe to tag:

- **Verified, not re-investigated** (per conductor's record): the
  `data/reference` repoint and its consumer fixes (`423ca6a`); the
  `docs/dev/PUBLISH_AUDIT.md` wording fix; the git-history purge (manuscript,
  internal paths, personal email) and its md5/`git fsck`-verified backup
  in `.claude/backups/`; CI smoke green on `423ca6a` (`37968527774`,
  success); the one deliberately-unresolved residual — GitHub retains
  pre-rewrite blobs server-side via `refs/pull/N/head` refs, not removable
  by a normal git push — flagged for the human owner, **not fixed here**.
- **Independent check I ran myself this pass** (own gate, not relayed):
  fresh `bash tests/check_layout.sh` → `layout check: PASS`, exit 0; fresh
  `python3 -m pytest -q tests/unit` → **39 passed** (96.7s; only pre-existing
  Tk/`Image.__del__` teardown warnings, not new, not failures).
- **New finding this pass, fixed mechanically**: GitHub Release metadata
  for `v0.1.2` and `v0.1.4` still carried `target_commitish` pointing at
  their *pre-rewrite* commit SHAs (`1bda9dea9...`, `6c7efec4c...`) — objects
  that no longer exist in the repository (`git cat-file -t` on both: not
  found). The tag refs themselves were always correct post-rewrite
  (`git ls-remote --tags origin` matches local `git rev-parse` for all four
  tags); only the Release object's separate `target_commitish` field was
  stale GitHub-side metadata. Patched via `gh api -X PATCH
  repos/dunyuliu/DR4GM/releases/<id> -f target_commitish=<current tag sha>`
  for both releases; confirmed by re-read. `v0.1.3`'s `target_commitish` is
  the literal string `main` (symbolic, resolves dynamically) — not stale,
  left as-is.
- **Found, not fixed — routed to owner**: `v0.1.1` has **no GitHub Release
  object at all** (`gh api repos/dunyuliu/DR4GM/releases/tags/v0.1.1` →
  404); only the tag exists. This predates the rewrite (Releases are
  tag-name-addressed and the rewrite didn't touch Release existence) and is
  not something I am backfilling unilaterally — if a `v0.1.1` Release is
  wanted, a backfilled Release must say so in its first line and never be
  marked Latest, per this project's own rule; that is an owner call on
  whether v0.1.1 should get one at all.

No other findings. Nothing deferred from board rows 11, 12, 24–27, 29–32 —
out of scope for this release per the dispatch (pre-existing, already
tracked, not re-litigated here).

## 6. Remaining open issues or pending items
Unchanged from v0.1.4, all pre-existing and explicitly non-blocking for
this patch:
- Board rows 11, 12, 29, 30: `BLOCKED(owner)` — vendored `gmpe-smtk`
  test-data size, `gm_stats.py` Rjb/`count<2` math, SPECFEM3D/WaveQLab3D
  row-13 fixtures.
- Board rows 24–27, 31, 32: pre-existing findings (hardcoded bin size,
  silent-except patterns, unpinned `requirements.txt`) — not in this
  release's scope.
- Board row 33: board-lint rule proposal, not yet implemented.
- `CITATION.cff` ORCID placeholder and README Zenodo DOI placeholder —
  block Zenodo upload specifically, not this code release.
- `local/` working directory needs manual owner clearing (pre-existing).
- **New this release — owner attention needed, not agent-actionable:**
  (a) GitHub-side server retention of pre-rewrite blobs via
  `refs/pull/N/head` (cannot be cleared by a normal git push — needs a
  GitHub support request or PR-ref cleanup the owner authorizes); (b)
  whether `v0.1.1` should get a backfilled GitHub Release.

## 7. Totals or cost changes
No cost/compute totals change — this is a data-link repoint plus a git
history scrub, not a pipeline-output change. `git diff --stat
v0.1.4..HEAD -- '*.md' '*.sh' '*.py'` (HEAD here = this release's own
pre-tag commit, i.e. the state right before this note): 8 files changed,
92 insertions(+), 13 deletions(-) — consumer-path updates and board/session
bookkeeping, not new logic.

## 8. Assumptions used
- The conductor's git-history-rewrite record (md5-verified backup,
  `git fsck --full` clean, `git log --all --oneline` 193→98 commits, zero
  hits on purged strings) is taken as accurate; I did not re-run the
  filter-repo sweep myself, only verified the *current* state (tag refs,
  CI, release metadata) independently.
- The in-progress CI run I gated this tag on (§9) was already running when
  I started this audit (triggered by the push that produced `285bfc9`,
  which was already on `origin/main` before I began) — I did not need to
  push anything myself to get a gate-able SHA.

## 9. The CI run this release was gated on
`gh run list --commit 285bfc95e2914cbbe81ffedc896e1975c3051084` → run
**37968997395**, job "smoke" (layout gate + `pytest tests/unit`),
conclusion **success**, SHA `285bfc95e2914cbbe81ffedc896e1975c3051084`
(= the commit this release tags; this release's own version-bump commit is
created after this CI run and is gated by the smoke tier I ran locally
myself, per §5, since no further remote push is needed before tagging —
the tag points at the version-bump commit itself, one commit ahead of
`285bfc9`, see the gate note below).

## 10. Trend since v0.1.4 (previous tag, `dd5d8bf3bee9b7bcd3cfc7d6b529fb59d68961b8`)
Reporting only; none of this gates the release.

- **Gate assertions.** `bash tests/check_layout.sh`: `layout check: PASS`
  at both `v0.1.4` and this release (re-ran live). Unchanged.
- **Fixture verdicts.** `tests/unit`: 39 passed at `v0.1.4` vs. **39
  passed** now (re-ran live, 96.7s). Unchanged. Full-tier `tests/run_tests.sh`
  was already freshly re-run by the conductor against the `data/reference`
  repoint (`423ca6a`, between v0.1.4 and this tag): `pass=5 noref=0 fail=0,
  total 107m10s, RESULT: PASS` — not re-run again by me (no further
  pipeline-affecting change landed after that run).
- **Tracked text lines.** `git diff --stat v0.1.4..HEAD -- '*.md' '*.sh'
  '*.py'` (HEAD = pre-note commit): 8 files changed, **92 insertions(+), 13
  deletions(-)**. Growth is consumer-path updates for the `data/reference`
  repoint plus board/session-log bookkeeping, not new feature logic.
- **Board currency.** `PATHWAY_FORWARD.md` status counts: **14 DONE, 14
  TODO, 4 BLOCKED(owner)** at both `v0.1.4` and HEAD — unchanged row
  counts, but the 2026-10-09 owner-decision entry and its "Closed" sub-bullets
  were added as new prose under an existing DONE-adjacent section (row 10's
  history log), not a new row — board currency is current, not degraded.
- **CI green-on-first-try rate.** `gh run list --branch main` since
  `v0.1.4`'s push (`6c7efec4`, 2026-10-08T17:28:00Z) through this release's
  gating run: **3/3 runs green on first try** (`3185c9a`, `423ca6a`,
  `285bfc9`), no re-runs needed. Unchanged from v0.1.4's 10/10 streak (still
  100%).

## 11. Work record
- audit: findings 1–6 relayed by the conductor (data-link repoint, history
  scrub, CI-green record) taken as settled and verified, not
  re-investigated; I independently re-ran smoke (layout + `tests/unit`,
  both PASS) and audited GitHub Release metadata integrity myself (new
  finding, §5).
- correctness: no open correctness findings in this release's scope; the
  stale `target_commitish` finding was a metadata-integrity issue, not a
  code-correctness one, and is fixed (§5).
- conciseness: no dedicated leanness/refactor pass run this release (not
  requested; diff is consumer-path updates plus bookkeeping, not new logic).
- fixes: `target_commitish` on Releases v0.1.2/v0.1.4 patched via `gh api`
  (§5); version strings bumped (`CITATION.cff`, `CLAUDE.md`); old release
  note archived to `docs/dev/`. Deferred: GitHub server-side
  `refs/pull/N/head` residual and the `v0.1.1`-Release question (§6), both
  owner calls.
- docs: `CITATION.cff`/`CLAUDE.md` version headers reconciled to 0.1.5
  against the actual commit about to be tagged; `release_notes_v0.1.4.md`
  archived to `docs/dev/` (never deleted); `PATHWAY_FORWARD.md`/
  `PROJECT_RULES.md` verified current, not re-edited (no drift found).
- refactor: kai-fischer not dispatched — this release's diff is a data-link
  repoint plus a git-history scrub already executed and reviewed by its
  surface owners (conductor, anya-petrov) before this release was
  dispatched; no new simplification scope opened by this cut.
- rules: no `PROJECT_RULES.md` violations found this pass; root whitelist
  (rule 2) still matches the actual tracked root listing exactly
  (`CITATION.cff`, `CLAUDE.md`, `LICENSE`, `PATHWAY_FORWARD.md`,
  `PROJECT_RULES.md`, `README.md`, `requirements.txt`,
  `release_notes_v0.1.5.md`, `.github/`, `.gitignore`).

## 12. Release gate
- tree: clean, one checkout (`git worktree list` shows exactly one), no
  lockfile/lock state, level with `origin/main` before this release's own
  commit (confirmed via `git status`/`git fetch`).
- ci: run `37968997395`, job "smoke", conclusion **success**, SHA
  `285bfc95e2914cbbe81ffedc896e1975c3051084` — the parent commit this
  release's version-bump commit builds on; the bump commit itself carries
  no pipeline-affecting change beyond version strings + file archival, so
  this run plus my own fresh local smoke re-run (§5, §9) are the gate.
- publish: note version `v0.1.5`, tag `v0.1.5` (created with its GitHub
  Release per project convention, no local-only tag), remote
  `github.com/dunyuliu/DR4GM`.
- release: `gh release view v0.1.5` against the pushed tag (verified after
  tag creation — see report).
- clone: not independently re-run this release (no pipeline/install-path
  change since v0.1.4's own stranger-clone PASS on `12aeafc`, and the
  `data/reference` repoint was already full-tier verified by the conductor
  on `423ca6a` per §10); flagged here rather than silently assumed.

This project has no `tests/release_gate.sh` script; the five rows above
were walked by hand per this note's own evidence, as documented in
`CLAUDE.md`'s Release Workflow section.
