# DR4GM Publication-Readiness Audit

**Date:** 2026-05-21
**Scope:** Repo at `.` as staged for a public GitHub release.
**Method:** Read-only static scan. No code executed. Vendored `gmpe-smtk/` and the read-only `reference/` tree are excluded from this audit.

## Summary

The repo is close to publication-ready. The license, attribution, citation metadata, and `.gitignore` are all in place. **Three blocking issues remain** before the public push:

1. Hardcoded `/Users/dliu/...` absolute paths in two tracked files (`regen_ensemble_figures.sh`, `test_system/benchmark_vectorized_gm.py`).
2. `CITATION.cff` ORCID is still the literal placeholder `0000-0000-0000-0000` with a `TODO` comment.
3. `.claude/settings.local.json` is committed-tree-adjacent (untracked, but the directory itself is not in `.gitignore`).

Everything else is minor or already gitignored.

---

## Critical

### C1 — Hardcoded `/Users/dliu/...` absolute path in `regen_ensemble_figures.sh`
**Lines:** `regen_ensemble_figures.sh:4`, `:5`, `:68`
**Content:**
```
PROD=./results/production_runs
UTILS=./utils
...
bash ./fetch_figures_for_publication.sh "$PROD"
```
**Why it's blocking:** This is one of the two scripts a reproducer runs after extracting the Zenodo bundle. It will fail immediately on any other machine.
**Fix:** Derive `REPO_ROOT` from `$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)`, then set `PROD="$REPO_ROOT/results/production_runs"` and `UTILS="$REPO_ROOT/utils"`. Already done correctly in `run_pipeline.sh` and `fetch_figures_for_publication.sh` — copy the pattern.

### C2 — Hardcoded `/Users/dliu/...` path in `test_system/benchmark_vectorized_gm.py`
**Line:** `test_system/benchmark_vectorized_gm.py:374`
**Content:**
```
Path("./reference/results/eqdyna/0001.A.2000m_subsampled/grid_2000m.npz"),
```
**Why it's blocking:** The file is tracked (`git ls-files test_system/benchmark_vectorized_gm.py`). Path leaks user identity in a public file. The script has a `/tmp/dr4gm_verify/...` fallback so the leak is cosmetic, but the absolute path is still a published artifact.
**Fix:** Replace with a repo-relative path: `REPO_ROOT / "reference/results/eqdyna/..."` (REPO_ROOT is already computed at line 30).

### C3 — `CITATION.cff` ORCID is a `TODO` placeholder
**Line:** `CITATION.cff:22`
**Content:**
```
    orcid: "https://orcid.org/0000-0000-0000-0000"  # TODO: replace with actual ORCID
```
**Why it's blocking:** Zenodo and downstream citation managers will parse this. A `0000-...` ORCID is invalid; the TODO comment is visible in the published metadata.
**Fix:** Either supply Dunyu Liu's real ORCID or delete the `orcid:` line entirely (it is optional in CFF 1.2.0).

---

## Important

### I1 — `.claude/` directory not in `.gitignore`
**Files:** `.claude/settings.local.json` (340 bytes, currently untracked)
**Status:** Not currently tracked, but if any future tool drops a file into `.claude/` it will be picked up by `git add .`. Other Claude-Code-using projects routinely add `.claude/` to `.gitignore`.
**Fix:** Append `.claude/` to `.gitignore`.

### I2 — `.DS_Store` exists in repo root
**File:** `.DS_Store` (16 KB, present in working tree, **not tracked** — already covered by `.gitignore:2`)
**Status:** Already gitignored. Not blocking; just note that the file exists in the working tree and should not be added.
**Fix:** No code change needed. Optionally `rm .DS_Store` before the publish push.

### I3 — Internal-only docs present in working tree
**Files in working tree (all gitignored, verified with `git check-ignore`):**
- `AUDIT.md` (gitignored at `.gitignore:22`)
- `CLAUDE.md` (gitignored at `.gitignore:23`)
- `PROJECT_RULES.md` (gitignored at `.gitignore:24`)
- `release_notes_v0.0.1-rc5.md` (gitignored at `.gitignore:25`)
- `demo/` (gitignored at `.gitignore:26`)
**Status:** All correctly excluded from the public repo via `.gitignore`. Verified by `git check-ignore` returning each path. Not blocking.
**Note:** `release_notes_v*.md` is gitignored at the repo root, but the release workflow in `CLAUDE.md` archives them into `docs/` on each release — and `docs/` itself is gitignored (`.gitignore:19`). So old release notes do not get published. This is intentional.

### I4 — `web/dr4gm_interactive_explorer.py` reads Streamlit secrets and includes author's personal email
**Lines:** `web/dr4gm_interactive_explorer.py:339-381`, `:436-438`, `:551-558`, `:1960-1965`, `:1987-1996`
**Content:**
- All `st.secrets.get(...)` calls are read-only — no credentials are hardcoded; they come from `.streamlit/secrets.toml` which is gitignored at `.gitignore:32`.
- Lines 1989 and 1996 hardcode `dliu@ig.utexas.edu` as the notification target in a usage-analytics comment block.
**Why it matters:** Not a credential leak (no API keys in source). But the author's personal Gmail address is published in a docstring/comment, separate from the public `dliu@ig.utexas.edu` used elsewhere. Up to the user whether to keep.
**Fix:** Already applied in a prior history-scrub pass — the personal gmail address is normalized to the institutional one repo-wide. for consistency with `README.md:7` and `CITATION.cff:18`.

---

## Minor

### M1 — `# TODO: replace with actual ORCID` is the only TODO in any public-facing file
Already covered by C3. No other `TODO`/`FIXME`/`XXX` markers found in `*.py`, `*.sh`, `*.md` outside `reference/`, `results/`, `docs/`, and the vendored `gmpe-smtk/`.

### M2 — Two "placeholder" mentions in converter code (not API-facing)
- `utils/sord_plot_converter_api.py:303`  — `# CAV placeholder (not available in SORD data)`
- `utils/specfem3d_converter_api.py:128` — `# No data - use placeholder values`
Both are accurate code comments describing real missing-data fallbacks, not stub/TODO markers. **Per user's standing rule "no fallback, no placeholder, fail loudly,"** these may warrant review independently, but they are not publication blockers and the source files are off-limits to this audit.

### M3 — `gmpe-smtk/` vendored attribution
**Files:** `gmpe-smtk/LICENSE`, `gmpe-smtk/LOCAL_MODIFICATIONS.md` — both present and tracked.
README.md acknowledges the bundle at lines 186-190. AGPLv3 attribution complete.

### M4 — `.gitignore` is reasonable
Inspected `.gitignore`:
- `.DS_Store`, `__pycache__/`, `*.py[cod]`, `*.egg-info/`, `.ipynb_checkpoints/` covered
- `reference/`, `datasets`, `results/`, `utils/results/` (heavy data) covered
- `docs/`, `AUDIT.md`, `CLAUDE.md`, `PROJECT_RULES.md`, `release_notes_v*.md`, `demo/` (internal) covered
- `web/.streamlit/`, `web/usage_analytics.log` (secrets/telemetry) covered
- `*.swp`, `.vscode/`, `.idea/` covered
**Verified no tracked junk:** `git ls-files | grep -E "\.DS_Store|__pycache__|\.pyc|\.env"` returns empty.
**Gap:** `.claude/` not listed (see I1).

### M5 — `LICENSE` is AGPLv3
**File:** `LICENSE` line 1: `GNU AFFERO GENERAL PUBLIC LICENSE Version 3, 19 November 2007` — matches the AGPLv3 declaration in `README.md:194-198` and `CITATION.cff:13`.

### M6 — `datasets/` symlink
`datasets -> reference/datasets` is a symlink whose target is excluded by `.gitignore`. Git tracks the symlink itself (verified `git ls-files datasets`). On a clean clone, this symlink will dangle until the user materializes `reference/`. Not a publication blocker because the documented workflow uses `results/production_runs/` directly (from Zenodo), not `datasets/` (which is for raw-pipeline reruns only).

### M7 — No credentials, API keys, or secrets in source
- `grep -rE "password|secret|api_key|token=" --include='*.py' --include='*.sh' --include='*.md' --include='*.yml' --include='*.json'` returns only `st.secrets.get(...)` references in `web/dr4gm_interactive_explorer.py` (covered by I4) and no plaintext.
- No `.env` files in working tree.

### M8 — No internal hostnames or localhost URLs
`grep -rE "https?://(localhost|127\.|192\.168\.|10\.)"` returns no results outside `gmpe-smtk/`.

### M9 — Author email
`dliu@ig.utexas.edu` appears in `README.md:7`, `CITATION.cff:18`. This is the published institutional address (Institute for Geophysics, UT Austin) and is the intended contact. Not internal-only.

---

## Tracked-files count

`git ls-files | wc -l` → **202 tracked files**. Spot-checked groups:
- `utils/*.py` — pipeline source
- `test_system/*.py`, `test_system/reference_results/*.npz` — regression baseline (~3 MB)
- `gmpe-smtk/**` — vendored AGPLv3 dependency
- `gui/*.py`, `web/*.py` — optional interfaces
- Top-level: `README.md`, `LICENSE`, `CITATION.cff`, `install.sh`, `run_pipeline.sh`, `regen_ensemble_figures.sh`, `fetch_figures_for_publication.sh`, `requirements.txt`, `.gitignore`

No surprise tracked binaries (no `.npz`, `.png`, `.docx`, `.tar` outside `test_system/reference_results/`).

---

## Action items (ordered)

1. Fix C1, C2: replace absolute paths with repo-relative paths.
2. Fix C3: supply real ORCID or remove the line.
3. Fix I1: add `.claude/` to `.gitignore`.
4. (Optional) Fix I4: replace personal Gmail with institutional email in `web/dr4gm_interactive_explorer.py` comments.
5. `rm .DS_Store` before publish push (cosmetic).
