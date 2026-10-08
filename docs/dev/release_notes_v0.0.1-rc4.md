# Release Notes — v0.0.1-rc4

**Date:** 2026-05-14
**Bump:** patch (rc3 → rc4)

---

## 1. Summary of scope

Pre-release candidate 4 consolidates three categories of work completed since rc3:

1. **SRL manuscript** — paper moved to `docs/paper/`, TikZ design figure, all floats
   embedded in text, figures selected and copied to `docs/paper/figures/`.
2. **Repository hygiene** — `docs/` cleaned to `paper/` + `slides/`; new top-level
   files `CITATION.cff`, `requirements.txt`, `run_pipeline.sh`; `install.sh` updated
   to use `requirements.txt`.
3. **Bug fixes from pre-release audit** — nine correctness and reliability fixes
   across `gm_stats.py`, `eqdyna_converter_api.py`, `seissol_converter_api.py`,
   and `run_pipeline.sh`.

---

## 2. Files added

| File | Purpose |
|---|---|
| `AUDIT.md` | Full static audit report (victor-reyes, 2026-05-14) |
| `CITATION.cff` | Standard software citation metadata (CFF v1.2.0) |
| `requirements.txt` | Pinned Python dependency list |
| `run_pipeline.sh` | Single entry point to reproduce all manuscript results |
| `docs/paper/main.tex` | SRL manuscript source (tectonic-compiled) |
| `docs/paper/fig1_design.tex` | TikZ design figure (replaces matplotlib PNG) |
| `docs/paper/references.bib` | BibTeX database (13 entries) |
| `docs/paper/main.pdf` | Compiled manuscript PDF (v0.0.1-rc4) |
| `docs/paper/figures/fig2a_eqdyna_map.png` | RSA map — EQdyna |
| `docs/paper/figures/fig2b_seissol_map.png` | RSA map — SeisSol |
| `docs/paper/figures/fig3_percode_eqdyna.png` | Per-code ensemble panel |
| `docs/paper/figures/fig4a_rsa_vs_distance.png` | Ensemble RSA vs Rjb |
| `docs/paper/figures/fig4b_rsa_vs_period.png` | Ensemble RSA vs period |
| `docs/paper/figures/fig5a_bias.png` | Spectral bias |
| `docs/paper/figures/fig5b_cav.png` | Median CAV vs Rjb |
| `docs/paper/figures/DR4GM_Capabilities_Diagram.pdf` | Web portal screenshot (fig:web) |
| `docs/paper/P28b_DR4GM_manuscript_v0.docx` | Original Word draft (archived) |
| `docs/slides/DR4GM_comprehensive_presentation.html` | Presentation (kept) |
| `docs/slides/DR4GM_Flyer.html` | Flyer (kept) |
| `docs/slides/DR4GM_comprehensive_diagram.pdf` | Diagram PDF (kept) |

## 3. Files removed / cleaned up

| File | Reason |
|---|---|
| `docs/FORMAT.md` | Superseded by paper Tables 2–5 |
| `docs/WORKFLOW.md` | Superseded by paper Section 2 + README |
| `docs/DR4GM_Capabilities_Diagram.md` | Superseded by TikZ `fig1_design.tex` |
| `docs/DR4GM_{Beautiful,Presentation,comprehensive}_Presentation.html` | Redundant drafts |
| `docs/business_demo.html` | Redundant draft |
| `docs/{Benefits,Title,Workflow}_Slide.png` | Generated slide PNGs, no source value |
| `docs/DR4GM_{business,technical,comprehensive}_diagram.png` | Superseded by TikZ |
| `docs/DR4GM_comprehensive_diagram.pdf` | Moved to `docs/slides/` |
| `paper/P28b_DR4GM_manuscript_v0.docx` | Moved to `docs/paper/` |
| `release_notes_v0.0.1-rc2.md` (root) | Deleted from root (already in `docs/`) |
| `release_notes_v0.0.1-rc3.md` (root) | Archived to `docs/` this release |

---

## 4. Content updates to master documents

- **`install.sh`** — `pip3 install numpy scipy` replaced by `pip3 install -r requirements.txt`
- **`CLAUDE.md`** — EQdyna fault_strike corrected from 90° to 0° (post-rotation value matches code)
- **`CITATION.cff`** — version bumped to `0.0.1-rc4`

---

## 5. Audit findings and fixes (from AUDIT.md, 2026-05-14)

All nine top-priority findings were fixed:

| # | Finding | Fix applied |
|---|---|---|
| 1 | `gm_stats.py` hard-coded 50,000-station bin buffer (overflow risk on 367k-station EQdyna) | Replaced with `np.digitize` vectorized binning — no buffer |
| 2 | `gm_stats.py` numerically-unstable sample-std formula (sqrt-of-negative NaN risk) | Replaced with `np.std(log_vals, ddof=1)` |
| 3 | `gm_stats.py` CAV unit label `cm·s` (wrong) | Fixed to `cm/s` |
| 4 | `gm_stats.py` default `distance_bin_size=1000` disagreed with paper and `run_all.sh` | Changed to 500 (class default + CLI default) |
| 5 | `eqdyna_converter_api.py` bare `except:` silently returned default dt=0.05 | Raises `FileNotFoundError` / `RuntimeError` on missing or unreadable params |
| 6 | `eqdyna_converter_api.py` binary size mismatch zero-padded/truncated with only a warning | Raises `ValueError` |
| 7 | `eqdyna_converter_api.py` reshape fallback produced zeroed stations silently | Fallback removed; reshape failure raises |
| 8 | `eqdyna_converter_api.py` chunk failure swallowed, returned empty arrays | `except/continue` removed; failure propagates |
| 9 | `seissol_converter_api.py` default dt=0.01 on single-timestep file | Raises `ValueError` |
| 10 | `run_pipeline.sh` tee'd to `$RESULTS/` before directory existed | `mkdir -p "$RESULTS"` added before first tee |

Additional docs fixes:
- Paper **Table 4** (`ground_motion_metrics.npz`): removed wrong `fault_trace_start/end` rows; added `station_ids` and `RSA_T_*` rows; added note that fault trace is in `geometry.npz`
- Paper **Table 5** caption: corrected "default bin size is 500 m" to "run_all.sh uses 500 m bins by default"

---

## 6. Remaining open issues

| Issue | Notes |
|---|---|
| **Tests not run** (PROJECT_RULES Rule 1) | Full test suite requires ~109 GB reference dataset; cannot be confirmed in this session. Must be verified before public release. |
| **GUI screenshot** (fig:gui) | Still a placeholder `\fbox`. Screenshot to be provided. |
| **Dataset repository** | `[repository TBD]` in Data and Resources section of manuscript. |
| **Acknowledgments** | `[Additional funding acknowledgments...]` placeholder. |
| **ORCID** (D.L.) | Placeholder `0000-0000-0000-0000` in `CITATION.cff`. |
| **references.bib** TODOs | 7 entries need journal/volume/pages/DOI: Gong2025, LiuBecker2025, Tainpakdipat2025, Premus2020, Ramos2021, WangDay2020, Withers2023. |
| **Magic constant 981** | Used in 7 places; should be `980.665`. Low priority. |
| **NaN guard on `np.log`** | `gm_stats.py:240` has no guard for zero/negative values before log. |
| **`geometry.npz` path heuristic** | `gm_stats.py` looks for `geometry.npz` as sibling of `ground_motion_metrics.npz`; fails if files are separated. |

---

## 7. Totals

- Source files modified this release: 5 (`gm_stats.py`, `eqdyna_converter_api.py`, `seissol_converter_api.py`, `install.sh`, `run_pipeline.sh`)
- Docs files added: 20 (paper + slides)
- Docs files removed: 13 (superseded presentations, diagrams, format/workflow docs)
- Tests passing: unconfirmed (see open issues)
