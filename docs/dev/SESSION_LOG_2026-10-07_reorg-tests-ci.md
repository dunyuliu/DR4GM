# DR4GM session log — 2026-10-07 reorg / tests / CI campaign

Conductor: Wei Lin. Mission: `/autopilot reorg. And enhance tests, particularly
e2e, design light CI, and refactor` (owner, verbatim, 2026-10-07). Budget: 6h.

No priority column exists on the board (`PATHWAY_FORWARD.md`) — noted here
rather than invented; working state order within no-priority: BROKEN (none) >
OPEN (TODO/DOING) > overdue VERIFIED (none), with leverage ties (a row that
unblocks others) broken first.

## Checkpoint inherited (prior conductor instance)

main at `8e3d876` after 11 merged PRs (#1–#11), ~1h22m of budget spent.
Board rows 1,2,6,7,8 DONE; row 9 DOING (CI smoke, blocked on row 4); row 10
DOING (4/9 reorg slices landed); row 13 DONE for SeisSol only; rows 11,12,14
BLOCKED(owner) — relay only, no fix without owner answer.

## This instance — turn 1

- **Security fix (immediate, not a board row):** `web/.streamlit/secrets.toml`
  (pre-move `web/` dir, 970 B, untracked since Aug 2025) was not covered by
  `.gitignore` — `git check-ignore` returned no match, so a plain `git add -A`
  would have published a credential to the public repo. Added
  `**/.streamlit/secrets.toml` and `web/` to `.gitignore`. File itself left
  in place, untouched, not read. Commit `27890f8`, pushed to origin/main
  directly (docs/board-only mechanical change, no code/physics touched, fast
  lane per merge-gate rules).
- No `SESSION_LOG_*` file existed anywhere in the tree (checked root, `docs/dev/`,
  `local/`) despite the brief citing one "at the repo root" — code/filesystem
  wins per loop rule 2; creating this file now at `docs/dev/` (internal-only,
  gitignored? no — `docs/dev/` is tracked per `PROJECT_RULES.md` rule 2 for
  audits/planning docs; session logs are the same class, so tracked here, not
  at root, since root is a strict whitelist that does not include session logs).
- Dispatched two specialists in parallel (2-cap, disjoint files, verified via
  `git worktree list` showing only the main checkout before dispatch — no
  invisible peer agents found):
  - **kai-fischer** — row 10 slice: `gmpe-smtk/` → `src/gmpe-smtk/` (git mv +
    8 import/path fixups + check_layout.sh whitelist/SIZE_EXEMPT update).
    agentId `a416f82abe0a7e37d`.
  - **iris-vermeulen** — row 3: harden `run_e2e_bundle.sh` (clean git-archive
    run, per-figure dim assertions, seissol/2 negative control, Fig 11
    SORD/SPECFEM3D absence check, dual bundle-version coverage or explicit pin).
    agentId `ae64b2e66863c9036`.
- Both briefed with: worktree isolation, no touching `src/` production code
  (iris), no touching gmpe-smtk attribution files beyond the move (kai), fresh
  self-verification required, `Agent: <name>` commit trailer, no PR opened by
  them (conductor opens PRs per rule).

## Roster (live children)

| Agent | Mission | Worktree | Status |
|---|---|---|---|
| kai-fischer (`a416f82abe0a7e37d`) | gmpe-smtk relocation | TBD (reported on completion) | dispatched |
| iris-vermeulen (`ae64b2e66863c9036`) | e2e bundle hardening | TBD (reported on completion) | dispatched |

## Pending owner items (relay only, unchanged this session)

- Row 14: `gm_stats.py --distance_bin_size` 2000 vs 500 default — options
  (a)/(b)/(c) already with owner, (b) recommended, awaiting answer.
- Rows 11, 12: BLOCKED(owner), unchanged.
