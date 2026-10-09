# DR4GM Development Notes

Internal-facing development notes, now tracked at the repo root (promoted
from the formerly gitignored `local/CLAUDE.md` — see `PATHWAY_FORWARD.md` row 10).
For the science (formulas, units, τ vs φ, figure→script map) read **`docs/user/FORMULAS.md`**,
which is the accurate reference and ships with the repo.

**Author:** Dunyu Liu — <dliu@ig.utexas.edu>, Institute for Geophysics, UT Austin.
**Current version:** 0.1.5 (`CITATION.cff`).

---

## Repository layout rule

The board (`PATHWAY_FORWARD.md`) and rule book (`PROJECT_RULES.md`) ship
publicly at the tracked repo root, along with this file, `README.md`, and
`docs/user/FORMULAS.md`. Older release notes and the development audits live
in tracked `docs/dev/`; the current release notes file stays at tracked root.

**Never create a new file at the repo root unless it is meant to ship publicly.**
Draft/scratch artifacts not yet promoted into a tracked doc go in `local/`,
which stays gitignored (`.gitignore` also covers `.claude/`, `results/`,
`reference/`, `demo/`).

---

## Core pipeline

```
raw sim output → <code>_converter_api.py → npz_gm_processor.py → gm_stats.py → figures
                 (stations/velocities/    (per-station PGA/PGV/  (binned mean/
                  geometry .npz)           PGD/RSA/CAV)          std vs Rjb)
```

`src/utils/run_all.sh <raw_dir> <code> <out_dir>` chains this for one scenario.

**GM metric computation** goes through `src/utils/vectorized_gmrotd50.py`
(station-vectorized GMRotD50). The vendored `src/gmpe-smtk/` is the *reference*
implementation used for validation and unit tests, **not** the production path.

**Shared style/registry:** `src/utils/code_style.py` holds `CODE_COLORS`,
`CODE_DISPLAY_NAMES`, `code_of/code_color/code_display`, and `gmm_envelope`.
Both `visualize_ensemble_stats.py` and `plot_pergroup_ens_figure12.py` import
from it — do not re-declare these tables locally (they drifted twice before).

**Rjb:** one vectorized implementation, `gm_stats.py:rjb_distances_m`. All three
call sites use it. The 100 m floor is applied only in `gm_stats.py`.

---

## Manuscript figures

```bash
bash scripts/regen_ensemble_figures.sh     # Figs 11–19 → results/production_runs/figs_to_publish/
```

That script does everything: per-scenario RSA maps (Fig 11), per-code ensemble
panels (Fig 12), ensemble stats (Figs 13–19), then calls
`fetch_figures_for_publication.sh` to rename into `Figure<NN><letter>.png`.
Currently produces **41 parts**.

Conventions baked into the script (don't silently change):
- **Fig 11**: shared caxis `--vmin 0.04 --vmax 1.5`, shared `--ylim -40 40`;
  MAFE gets `--xlim -10 10` (its fault-normal extent is bounded at 10 km),
  all others `--xlim -20 20`. Fault auto-centered to (0,0) from `geometry.npz`.
- **Fig 12**: `--xlim 0.5 20` (azimuthally uniform coverage ends ~20 km).
- **`seissol/2` is excluded everywhere** — that simulation did not reach Mw 7
  (median SA(T=1 s) ≈ 5× below the other four).
- GMM bands: ±τ on Figs 12/13/19A, ±σ on 14A, φ range on 15/16/19B, τ range on
  17/18. See `docs/user/FORMULAS.md` §7.1 — that table is authoritative.

**Fig 11 gaps (data, not code):** SORD and SPECFEM3D have no per-station NPZ,
so they get no Fig 11 panel; they appear in Figs 12–19 via the
`gm_statistics.npz` fallback. No hypocenter data exists for any code.

---

## Fault geometry (verified from `geometry.npz`, not from memory)

All converters write `fault_strike = 0.0` and an N–S fault along y.
Two codes are **not** centered at the origin:

| Code | fault x (km) | fault y (km) | Note |
|---|---|---|---|
| EQdyna, SeisSol, MAFE, SPECFEM3D | 0 | −20 … +20 | centered |
| FD3D_TSN | 0 | 25.1 … 65.1 | y-offset |
| WaveQLab3D | **20** | 20 … 60 | x *and* y offset |

`visualize_gm_maps.py` handles this: when `--xlim`/`--ylim` are given it shifts
the display so the fault lands at (0,0), using `_get_fault_center_km()`.
MAFE and FD3D are half-domain and get mirrored across x=0 (`_MIRROR_CODES`);
WaveQLab3D is full-domain and must **not** be mirrored.

EQdyna and SeisSol converters apply a 90° CCW rotation `(x,y) → (−y,x)` at
conversion time so raw E–W faults become N–S. Visualization does no rotation.

---

## Data

```
data/reference/                     local-only link (gitignored), not copied into worktrees
├── datasets/                       ~109 GB raw sim data, 7 codes
└── results_original_resolution/    ~92 GB per-station native-resolution outputs
results/                            mutable; fresh runs go here
```

`data/reference` is created by `scripts/install.sh`: it links to
`$DR4GM_REFERENCE_DIR` (default `$HOME/shared_dataset/dr4gm_drv.reference`),
the machine's shared, read-only copy of the frozen 199 GB dataset. Never a
tracked symlink (the repo is public; the target path is machine-specific) and
never modified in place — direct reruns to `results/` and diff.

**Zenodo bundle:** `../dr4gm_data_v0.0.1.tar.gz` (13 MB) — the 65 NPZ files
(22 scenarios × 3) that `regen_ensemble_figures.sh` needs, so anyone can
reproduce Figs 11–19 without the 109 GB raw data. README documents the recipe.

---

## Tests

```bash
bash tests/run_tests.sh            # 5 canonical scenarios, ~13 min
bash tests/run_tests.sh --all      # all 20, ~60 min
```

Diffs fresh `ground_motion_metrics.npz` against `tests/reference_results/`
(3.5 MB in-tree, **15 RSA periods**). Pass = float32-aware bit equivalence
(1e-6 rel for float32 inputs, 1e-12 otherwise).

⚠️ **Piping through `tee` masks the exit code** (no `pipefail`). For CI use
`bash tests/run_tests.sh > run.log 2>&1` or set `pipefail` first.

⚠️ **Stale-oracle lesson (2026-08):** the baselines sat at 13 periods while the
code produced 15 for ~3 months; every run failed on shape mismatch and nobody
noticed because the suite wasn't being run. If you change the period list, or
anything in the GM computation, **regenerate and re-bless the baselines in the
same commit**, and verify the overlapping columns stay bit-exact first.

---

## Release Workflow

When I say **release** (patch), **release minor**, or **release major**, execute
this end to end.

- `release` → bump C · `release minor` → bump B, reset C · `release major` → bump A, reset B and C

1. **Inspect changes.** `git status` and `git diff HEAD`.
2. **Find current version.** Highest semver among root `release_notes_v*.md`
   and `docs/dev/release_notes_v*.md` (parse A.B.C — not mtime, not filename
   sort). Cross-check `CITATION.cff`.
3. **Keep history.** Never delete a release note; superseded ones move to
   `docs/dev/`.
4. **Gate:** `bash tests/check_layout.sh` must exit 0 (no release otherwise).
   **Audit against `PROJECT_RULES.md`.** If missing, stop and ask rather
   than improvising. Check: unprocessed files, naming violations, duplicates,
   cross-file consistency, master docs needing updates.
5. **Apply fixes.** Mechanical ones (rename, move, sync a date) directly.
   Anything needing judgment → "remaining open issues", do not invent a fix.
6. **Write `release_notes_v<new>.md`** at tracked root, describing the
   post-audit filesystem state — not the pre-audit state, not the raw git diff.
   Draft it first in `local/` (gitignored) if you want a scratch pass before
   it goes tracked.
7. **Re-verify.** Re-read the note; spot-check every claim against the filesystem.
8. **Commit** `release: v<A.B.C> — <summary>`, including the new release notes file.

**Hard rules:** never skip the audit · never write the note from git diff alone ·
never delete old notes · never invent fixes for judgment calls · never put a new
file at the repo root unless it ships publicly.

---

## Known open issues

| Issue | Status |
|---|---|
| `CITATION.cff` ORCID is placeholder `0000-0000-0000-0000` | **blocks Zenodo upload** |
| README Zenodo DOI is `XXXXXXX` | fill in after upload |
| `gm_stats.py` Rjb 100 m floor biases nearest bin ~25 % (C3/M15) | deferred — needs `gm_statistics.npz` regen |
| `gm_stats.py` drops bins with `count < 2` instead of `std=NaN` (C2) | deferred — same |
| `create_rjb_distance_map` skips mirror/y-shift (C4) | deferred — affects only the Rjb map, not Figs 11–19 |
| SPECFEM3D CAV nearly flat with distance (Fig 19A) | physics question for the modelers, not a bug |
| Fig 11 excludes `seissol/2` partly by absence of its map PNG | a manual `run_all.sh seissol/2` would resurrect it; consider a hard exclude in `fetch_figures_for_publication.sh` |
| 2 vendored `src/gmpe-smtk/` test files >5 MB (38.6 MB CSV, 19.6 MB HDF5) | pending owner — rest of root layout per zofia template DONE, incl. `utils/`→`src/utils/`, `gmpe-smtk/`→`src/gmpe-smtk/`, and `test_system/`→`tests/` (row 10 slice) — evidence: `bash tests/check_layout.sh` (gate, also run first by `run_tests.sh`) |

Full detail in `docs/dev/AUDIT.md`, `docs/dev/AUDIT_MATH.md`, `docs/dev/AUDIT_PHYSICS.md`,
`docs/dev/AUDIT_FORMULAS.md`.

---

## Bundled dependency: gmpe-smtk

Vendored in-tree under `src/gmpe-smtk/` (AGPLv3, © GEM Foundation,
<https://github.com/GEMScienceTools/gmpe-smtk>). **Not** a git submodule — the
inner `.git/` was removed and the source committed as part of DR4GM.

Local NumPy 2.x / SciPy ≥ 1.14 compatibility edits are recorded in
`src/gmpe-smtk/LOCAL_MODIFICATIONS.md`. Credit appears in `README.md`.
**Do not delete or rewrite** `src/gmpe-smtk/LICENSE`, `src/gmpe-smtk/README.md`, or
`src/gmpe-smtk/LOCAL_MODIFICATIONS.md` — required for AGPLv3 attribution.
