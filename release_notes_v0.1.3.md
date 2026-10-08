# DR4GM v0.1.3 — release notes

Patch release. Cheap, single-fix cadence per `CLAUDE.md`'s versioning scheme
(`release` → bump patch) — not a milestone release, so this does not re-run
the full zofia/victor audit cycle; it carries forward v0.1.2's audit
(`docs/dev/release_notes_v0.1.2.md`) with one correction (below) plus one
fix.

## 1. Summary of scope

v0.1.2's own stranger-clone gate (run by haruto-nakamura after that tag was
cut, following the README literally on a clean clone) found a real,
reproducible bug that the tag had already shipped: `src/utils/run_all.sh`
invoked every converter/processing step via bare `python`, not `python3`.
On any host where `/usr/bin/python` is Python 2 — a common default on
shared/older Linux hosts, and the exact situation on the gate host here —
the first converter call crashes:

```
SyntaxError: invalid syntax
    def __init__(self, input_dir: str, output_dir: str):
```

Python 2 cannot parse Python 3's type-hinted function signatures. This bug
predates v0.1.2 (confirmed via `git show v0.1.1:utils/run_all.sh` — same
bare `python` calls) but was never caught because every prior
test/audit/CI run happened where `python` already aliased `python3`
(venv/conda `PATH`). The stranger-clone gate is the first check in this
project's history to run with a bare, unaliased `PATH`.

The same bug existed in two other scripts (`test_system/run_tests.sh`,
`scripts/regen_ensemble_figures.sh`) — fixed uniformly, all three scripts,
every call site (15 total: 8 in `run_all.sh`, 4 in `run_tests.sh`, 3 in
`regen_ensemble_figures.sh`).

## 2. What landed

- PR #20, `409ea7d` — `python` → `python3` at all 15 call sites across the
  three scripts. No other code changed.
- This release note + `docs/dev/release_notes_v0.1.2.md` §6 correction (the
  original stranger-clone write-up speculated a GDAL/openquake install
  failure that did not reproduce on the actual gate host — the real,
  confirmed failure was the `python`/`python3` bug above).
- `CITATION.cff` version 0.1.2 → 0.1.3, date-released unchanged (same day).
- `CLAUDE.md` version header synced to 0.1.3.

## 3. Verification

- Reproduced the bug directly before fixing: `env -i PATH=/usr/bin:/bin
  python src/utils/eqdyna_converter_api.py --help` → `SyntaxError`; same
  command with `python3` → prints usage correctly. (This host's
  `/usr/bin/python` is in fact Python 2.7.18, confirming the gate's finding
  was not host-specific speculation.)
- `bash test_system/check_layout.sh` → PASS.
- `python3 -m pytest -q test_system/unit` → 39 passed.
- CI green on the PR head SHA and on the squash-merge SHA (`409ea7d`) before
  this tag.

## 4. Audit status (carried forward from v0.1.2, not re-run)

zofia-kaminska's rule-book pass and victor-reyes's code-correctness pass
(including the real `test_system/run_tests.sh` full tier, 5/5) were run
fresh for v0.1.2 and are unaffected by this patch's 3-file, no-logic-change
diff. Their non-blocking findings (board rows 24-27) are unchanged and still
open — see `docs/dev/release_notes_v0.1.2.md` §5 and `PATHWAY_FORWARD.md`.

## 5. Remaining open issues (unchanged from v0.1.2)

See `docs/dev/release_notes_v0.1.2.md` §5 for the full list (board rows
24-27, plus the pre-existing `CLAUDE.md` "Known open issues" table: Zenodo
ORCID/DOI placeholders, Rjb 100 m floor, `count<2` bin drop,
`create_rjb_distance_map` mirror/shift gap). None are introduced or resolved
by this patch.

## 6. Stranger-clone gate for this release

Not independently re-run for this patch (cheap-patch cadence); the fix was
verified directly against the exact failure mode the v0.1.2 gate found (see
§3). The next milestone release (row-10 reorg completion) re-runs the full
stranger-clone gate from scratch.
