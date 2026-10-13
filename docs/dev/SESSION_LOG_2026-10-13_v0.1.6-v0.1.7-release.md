# Session log — 2026-10-13 — v0.1.6 and v0.1.7 milestone releases (conductor: wei-lin)

Seeded from `PATHWAY_FORWARD.md`, `PROJECT_RULES.md`, `CLAUDE.md`, and
`docs/dev/SESSION_LOG_2026-10-12_board-clear.md` (predecessor stopped with the
board exhausted except owner rows). Streamlit surface (rows 15-22) had reached
DONE and 10 PRs (#29-#38) had landed since v0.1.5 with no tag cut — that is a
milestone (a surface reaching target state), so the strict release cycle
(audit -> fix -> refactor -> release) ran, per `/autopilot`'s instruction,
ending in two patch tags.

## v0.1.6 (first milestone cycle)

1. **Audit** (parallel, isolated worktrees): victor-reyes on `v0.1.5..beaba5b`
   (no BLOCKER/MAJOR, 6 MINOR, deferred) + zofia-kaminska rule-book pass (1
   MAJOR: README.md:76 claimed 22 scenarios for `tests/run_tests.sh --all`,
   actual 20; 2 MINOR, deferred).
2. **Fix**: sophia-okafor fixed the MAJOR doc drift (PR #39, `bf8cc57`),
   conductor-verified diff (single line, no leaks) before merge.
3. **Refactor**: none scoped — the only MAJOR was a one-line doc fix, not a
   structural finding.
4. **Release**: haruto-nakamura — CI green on `bf8cc57`; full-tier gate
   `pass=5 noref=0 fail=0, total 93m11s`; stranger-clone gate PASS (README's
   Zenodo-DOI curl fails as a known pre-existing item, continued via the
   local bundle). Tagged **v0.1.6** at `ab65f87` (PR #40), CI run
   `38351945086` success, release marked latest.

**Finding during v0.1.6's stranger-clone gate (not a hypothesis — conductor
independently re-verified on this host):** `scripts/install.sh:7` ran bare
`pip3 install -r requirements.txt` with no venv, silently mutating the
invoking user's SHARED host Python environment (numpy 1.24.4 -> 2.2.6, scipy
left ABI-broken until manually re-pinned to 1.15.3). `pip3 check` showed no
other numpy/scipy-dependent breakage on this host; the only flagged conflict
(protobuf vs tensorflow) was pre-existing and unrelated.

- Opened board row 36 (P1) via zofia-kaminska (PR #41, merged).
- kai-fischer's first fix attempt (`python3 -m venv`) was independently
  re-verified by the conductor on the ACTUAL release host and found broken
  there: `ensurepip`/`python3-venv` is missing, so the fix as written would
  have broken `install.sh` outright on the very host releases are gated from
  — this was caught by re-running the fix myself, not by trusting the
  subagent's sandbox-limited report.
- Sent back to kai-fischer for a `virtualenv`-fallback fix; independently
  re-verified end-to-end this time (`pyvenv.cfg` showed `virtualenv =
  20.29.3`, numpy/scipy landed inside `.venv` only, user site-packages
  byte-identical before/after). Merged PR #42 (`c6bbfda`). Row 36 closed
  (PR #43, `b6a108a`).

## v0.1.7 (second milestone cycle, same session)

Victor-reyes's own audit of `ab65f87..c6bbfda` flagged a **suspected MAJOR**
(not executed): the new hard-abort-on-pip-failure has no carve-out for
`openquake.engine`, which PROJECT_RULES.md rule 4 already documents as
failing without system GDAL — unlike CI's own install step. Opened as board
row 37 (P2, TODO) via zofia-kaminska (PR #44, `c8cfea3`).

Dispatched haruto-nakamura for the v0.1.7 release on `c8cfea3`. **Haruto
correctly stopped before tagging**: the stranger-clone gate reproduced row
37's concern for real — not the openquake-abort itself (this host has GDAL),
but a related silent-failure defect it surfaced: the figure-regeneration
path (`scripts/regen_ensemble_figures.sh` -> `visualize_ensemble_stats.py`)
silently drops Figure14B (40/41 figures, no loud signal) whenever
openquake's deeper GDAL-dependent import fails, previously masked because
v0.1.6's leaked host-site packages papered over it; the new clean `.venv`
isolation (row 36's own fix) made the gap visible for the first time.
Conductor independently re-confirmed this by reading the production code
path (`visualize_ensemble_stats.py` prints only an easily-missed stdout
warning, unlike `tests/run_e2e_bundle.sh`'s already-correct named-skip
handling).

- Routed the fix to kai-fischer first; **kai correctly declined** — this is
  new behaviour (loud banners, a preflight check, new docs), not
  behaviour-preserving refactor, and said so rather than quietly stretching
  scope. Re-routed to a general-purpose engineer with the exact file:line
  targets and a 40-call budget. Fix landed clean: loud named skip banner in
  `visualize_ensemble_stats.py`, end-of-run NOTICE in
  `regen_ensemble_figures.sh`, informational (non-aborting) GMPE-availability
  preflight in `install.sh`, README documentation. `PLOT_GMPE_AVAILABLE`
  gating logic itself untouched. Conductor diffed and privacy-swept before
  merge (PR #45, `b17d6eb`).
- Second victor-reyes audit (`c6bbfda..b17d6eb`): no BLOCKER/MAJOR; 2 new
  MINOR (README calls openquake "optional" while install.sh hard-requires
  it; suggested fix points at pip rather than system GDAL). Confirmed row 37
  stays open — the fix makes the skip visible but does not add the
  openquake carve-out to the hard-abort itself.
- Re-dispatched haruto on `b17d6eb`: CI green; full-tier `--all` gate
  `pass=5 noref=15 fail=0, total 192m51s`; stranger-clone gate re-run twice
  (`b17d6eb` and the eventual tag `e8ab37e`) — venv isolation holds (user
  site-packages byte-identical), the GDAL-absent path now prints the named
  Figure14B skip with a loud banner. Tagged **v0.1.7** at `e8ab37e` (PR #46),
  CI run `38385425240` success, release marked latest. Conductor
  independently re-verified every claim (tag existence, CI conclusion, PR
  merge state, release note presence) before accepting the relay.

## Open items carried forward (not actioned this session, per the rule book)
- Board row 37 (P2, TODO) — `scripts/install.sh:40` still hard-aborts on
  ANY `requirements.txt` package failure, no openquake carve-out matching
  CI's own rule-4 precedent. Latent, not reproduced as a true install
  failure on this host (only the Figure14B visibility symptom reproduced).
- 2 new MINORs (v0.1.7 second audit): README.md:96-109 calls openquake
  "optional" while `install.sh:40` hard-requires it; README.md:103-105 /
  install.sh:104 point the fix at `pip install openquake.engine` rather
  than the real blocker (system `libgdal-dev`/`gdal-config`).
- 5 MINORs from the first v0.1.6..v0.1.7 audit pass (unchanged).
- 2 MINORs from the v0.1.6 cycle's zofia audit (stale rule-6 exception text,
  `PROJECT_RULES.md:186-189`; misleading "Force Git Commit" UI copy,
  `src/web/dr4gm_interactive_explorer.py:803,833,835`).
- Owner-blocked, unchanged, not re-raised: board rows 11, 12, 29, 30; the
  v0.1.2 superseded flag; `local/` manual clearing; CITATION.cff ORCID
  placeholder (blocks Zenodo upload, also the cause of README's documented
  bundle-curl step failing for any stranger); Table 2 EQdyna script
  placement (lives outside this repo).

## Housekeeping note (self-flagged slip)
While bulk-deleting stale local branch refs at session close, I deleted the
local pointer for `worktree-agent-a690be0f423c3ad98` without first checking
it was merged — it is NOT (no PR, not an ancestor of `main`), though its
content (an early draft of rule 3a) was already independently landed via
`d5240b9`/PR #30 well before this session. Only the LOCAL ref was removed;
the remote branch `worktree-agent-a690be0f423c3ad98` is untouched and still
on origin — left alone, not deleted, pending the owner's own call on an
orphan branch that pre-dates this entire campaign.

## Worktrees / branches
All worktrees spawned this session were reaped after their PRs merged
(`agent-ae66f8ebcd66177fd`, `agent-ace046f6f626bc6e4`, `agent-a7dc768464fd0005e`,
`agent-adb93b12e78ce5967`, `agent-a7cf2980f64621e9a`, `agent-a6413ecc8d6500918`,
`agent-ab1a9ddc2d701df81`, `agent-a211173e579d6f0da`, `agent-a942516685e3ccacb`,
`agent-a9c5320d70fa51074`, `agent-ac6d516c6765fdd3f` [no-op, auto-cleaned],
`agent-ac98bcdc160c32b23`, `agent-a0865bdce08834a72`, `agent-aff87a312dec4be88`).
Remote branches whose `--delete-branch` failed at merge time (worktree still
locked) were swept after confirming each PR's MERGED state via `gh pr list
--head <branch>`: `board/close-row-36`, `board/row36-install-venv-leak`,
`board/row37-install-sh-openquake-carveout`, `worktree-agent-a0865bdce08834a72`,
`worktree-agent-ac98bcdc160c32b23`, `rows-18-19-20-streamlit-cleanup`,
`release/v0.1.6`. `git worktree list` and `git branch` are clean at session
end (only `main`, no stray worktrees).

Main is at `e8ab37e` (tag `v0.1.7`), fast-forwarded and verified against
`origin/main`.
