# Release Notes — v0.0.1-rc5

**Date:** 2026-05-14
**Bump:** patch (rc4 → rc5)

---

## 1. Summary of scope

Publication-readiness pass: git history squashed to a clean linear history,
all internal dev files (AUDIT.md, CLAUDE.md, PROJECT_RULES.md, release notes,
demo/) removed from git tracking, `docs/` fully untracked. The tracked
file set now contains only files necessary for public software release.

No source code changes since rc4.

---

## 2. Files added

None.

## 3. Files removed from tracking / cleaned up

| File / Directory | Action |
|---|---|
| `AUDIT.md` | Removed from git index (internal audit report) |
| `CLAUDE.md` | Removed from git index (Claude Code dev instructions) |
| `PROJECT_RULES.md` | Removed from git index (internal release rules) |
| `release_notes_v*.md` | Removed from git index; gitignored |
| `demo/` | Removed from git index |
| `docs/` | Fully gitignored (paper, slides, release notes managed separately) |

## 4. Git history

7 local commits since rc3 were squashed into 2 clean commits:
- `bac568d release: v0.0.1-rc3`
- `a2e676b release: v0.0.1-rc4 — audit fixes, SRL manuscript, docs cleanup`

Force-pushed to `origin/main` via `--force-with-lease`.

---

## 5. Content updates to master documents

- **`CITATION.cff`** — version bumped to `0.0.1-rc5`
- **`.gitignore`** — added `docs/`, `AUDIT.md`, `CLAUDE.md`, `PROJECT_RULES.md`,
  `release_notes_v*.md`, `demo/`

---

## 6. Remaining open issues

| Issue | Notes |
|---|---|
| **Tests not run** (PROJECT_RULES Rule 1) | Full suite requires ~109 GB dataset; must be confirmed before public release |
| **GUI screenshot** (fig:gui) | Placeholder in manuscript |
| **Dataset repository** | `[repository TBD]` in Data and Resources section |
| **Acknowledgments** | Funding placeholder in manuscript |
| **ORCID** (D.L.) | Placeholder `0000-0000-0000-0000` in `CITATION.cff` |
| **references.bib TODOs** | 7 entries missing DOI/volume/pages: Gong2025, LiuBecker2025, Tainpakdipat2025, Premus2020, Ramos2021, WangDay2020, Withers2023 |

---

## 7. Tracked file inventory (post-release)

```
.gitignore          LICENSE             README.md
CITATION.cff        requirements.txt    install.sh
run_pipeline.sh     regen_ensemble_figures.sh
fetch_figures_for_publication.sh
utils/              web/                gui/
gmpe-smtk/          test_system/
```
