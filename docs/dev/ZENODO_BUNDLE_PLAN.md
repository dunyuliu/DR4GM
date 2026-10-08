# DR4GM Zenodo Data Bundle — Staging Plan

**Date:** 2026-05-21
**Status:** Plan only. No tarball created yet.

## Goal

Ship the per-scenario NPZ files that drive manuscript Figs 11–19 so that a user who clones the public DR4GM repo and downloads this bundle can reproduce all manuscript figures with one command (`bash regen_ensemble_figures.sh results/production_runs` followed by `bash fetch_figures_for_publication.sh results/production_runs`).

## Scenarios in scope (22)

From `regen_ensemble_figures.sh` `ALL_SCENARIOS` array, excluding `seissol/2` (excluded from manuscript because it did not reach Mw 7 — median SA(T=1s) ~5× lower than the others). Listed in figure-script order:

| # | scenario path                  | code        |
|---|--------------------------------|-------------|
| 1 | `eqdyna/0001.A.100m`           | eqdyna      |
| 2 | `eqdyna/0001.B.100m`           | eqdyna      |
| 3 | `eqdyna/0001.C.100m`           | eqdyna      |
| 4 | `fd3d/ncent.sd4`               | fd3d        |
| 5 | `fd3d/ncent.sd8`               | fd3d        |
| 6 | `fd3d/nleft.sd4`               | fd3d        |
| 7 | `fd3d/nleft.sd8`               | fd3d        |
| 8 | `fd3d/nright.sd4`              | fd3d        |
| 9 | `fd3d/nright.sd8`              | fd3d        |
|10 | `mafe/1`                       | mafe        |
|11 | `mafe/2`                       | mafe        |
|12 | `mafe/3`                       | mafe        |
|13 | `seissol/1`                    | seissol     |
|14 | `seissol/3`                    | seissol     |
|15 | `seissol/4`                    | seissol     |
|16 | `seissol/5`                    | seissol     |
|17 | `waveqlab3d/a24`               | waveqlab3d  |
|18 | `waveqlab3d/c24`               | waveqlab3d  |
|19 | `waveqlab3d/d24`               | waveqlab3d  |
|20 | `sord/1/sord_scenario`         | sord        |
|21 | `specfem3d/1`                  | specfem3d   |
|22 | `specfem3d/2`                  | specfem3d   |
|23 | `specfem3d/3`                  | specfem3d   |

(That's 23 scenario directories; `sord/1/sord_scenario` is a single scenario nested one deeper. Total = 22 manuscript scenarios across 7 codes, matching `FIG12_SCENARIOS`. The duplicated row count is just how `sord` paths nest.)

## What each NPZ is used for

| NPZ file                      | size (typ.) | feeds Fig                 | required for bundle? |
|-------------------------------|-------------|---------------------------|----------------------|
| `ground_motion_metrics.npz`   | ~0.6–1.0 MB | Fig 11 (maps), Fig 12 (per-code scatter+median panels) | **yes** — for codes that have per-station data |
| `gm_statistics.npz`           | ~30 KB      | Fig 12 fallback line, Fig 13/14/15/16/17/18/19 | **yes** — all 22 scenarios |
| `geometry.npz`                | ~2–3 KB     | Fault trace + strike used by Fig 11 maps and Fig 12 panels | **yes** — all 22 scenarios |

Notes:
- `sord` and `specfem3d` ship `gm_statistics.npz` directly from their converters (no per-station file). For those four scenarios (sord/1, specfem3d/{1,2,3}), `ground_motion_metrics.npz` does not exist on disk and is not needed — Fig 12 falls back to the binned-median line from `gm_statistics.npz` (verified in `utils/plot_pergroup_ens_figure12.py:120-175`).
- `processed_stations.npz`, `stations.npz`, `velocities.npz` are intermediate pipeline artifacts. **Not needed** for figure regeneration. Excluded from the bundle.

## Bundle size

Measured `stat -f %z` on each file, summed:

| variant                            | bytes      | MB     |
|------------------------------------|------------|--------|
| **Full**  (3 NPZ types, 22 scen.)  | 14,333,783 | ~14 MB |
| Lite      (gm_statistics+geometry) |    696,769 | ~0.7 MB|

Per-scenario file sizes (bytes):

```
scenario                        gmm.npz      gm_stat      geometry
eqdyna/0001.A.100m               991595        31160         2015
eqdyna/0001.B.100m               990635        31151         2015
eqdyna/0001.C.100m               991384        31148         2015
fd3d/ncent.sd4                   624068        31138         1978
fd3d/ncent.sd8                   624202        31135         1978
fd3d/nleft.sd4                   626898        31135         1977
fd3d/nleft.sd8                   626627        31136         1977
fd3d/nright.sd4                  626636        31130         1979
fd3d/nright.sd8                  626017        31135         1979
mafe/1                           321121        30958         2749
mafe/2                           321108        30948         2749
mafe/3                           321245        30953         2750
seissol/1                        922876        31165         2415
seissol/3                        924576        31187         2415
seissol/4                        924114        31214         2415
seissol/5                        920836        31205         2415
waveqlab3d/a24                   750877        30112         2564
waveqlab3d/c24                   751445        30107         2567
waveqlab3d/d24                   750754        30108         2565
sord/1/sord_scenario                  0        10116         2719
specfem3d/1                           0        15272         2755
specfem3d/2                           0        14947         2756
specfem3d/3                           0        13707         2755
```

(`0` means the file does not exist for that scenario; that is expected for sord and specfem3d.)

## Recommendation: ship the **full** bundle

**Why include `ground_motion_metrics.npz`:**
1. Cost is trivial — full bundle is ~14 MB, well under any sensible Zenodo or download limit. Compressed (NPZ is already deflate-compressed inside) the .tar.gz will be ~14 MB.
2. Lite bundle (0.7 MB) regenerates Figs 13–19 only. Figs 11 (per-scenario SA maps) and Fig 12 (per-code scatter+median) require per-station data — without `ground_motion_metrics.npz`:
   - Fig 11 cannot be reproduced at all (no per-station SA values to map).
   - Fig 12 falls back to the binned-median line only — loses the scatter cloud and the per-simulation dashed traces that are the visual point of the figure.
3. The bundle is the single anchor for reproducibility of Figs 11–19. Splitting it forces users to download two artifacts to get the full figure set, which defeats the "one Zenodo DOI → all figures" promise.

**Tradeoff considered and rejected:** A lite bundle (gm_statistics + geometry only, ~0.7 MB) reproduces Figs 13–19 but cannot regenerate Figs 11 and 12. Bandwidth savings (~13 MB) are not worth losing two of nine manuscript figure groups.

## Tarball layout

```
dr4gm_data_bundle_v0.0.1-rc5.tar.gz
└── production_runs/
    ├── README.md                          (this file, abridged — describes contents + checksums)
    ├── MANIFEST.txt                       (list of every file with sha256 + bytes)
    ├── eqdyna/
    │   ├── 0001.A.100m/
    │   │   ├── ground_motion_metrics.npz
    │   │   ├── gm_statistics.npz
    │   │   └── geometry.npz
    │   ├── 0001.B.100m/  (same three files)
    │   └── 0001.C.100m/  (same three files)
    ├── fd3d/
    │   ├── ncent.sd4/  (same three files)
    │   ├── ncent.sd8/  ...
    │   ├── nleft.sd4/  ...
    │   ├── nleft.sd8/  ...
    │   ├── nright.sd4/ ...
    │   └── nright.sd8/ ...
    ├── mafe/
    │   ├── 1/   (same three files)
    │   ├── 2/   ...
    │   └── 3/   ...
    ├── seissol/
    │   ├── 1/   (same three files)
    │   ├── 3/   ...
    │   ├── 4/   ...
    │   └── 5/   ...
    ├── waveqlab3d/
    │   ├── a24/  (same three files)
    │   ├── c24/  ...
    │   └── d24/  ...
    ├── sord/
    │   └── 1/
    │       └── sord_scenario/
    │           ├── gm_statistics.npz
    │           └── geometry.npz       (no ground_motion_metrics.npz — pre-binned by converter)
    └── specfem3d/
        ├── 1/   (gm_statistics.npz + geometry.npz only)
        ├── 2/   ...
        └── 3/   ...
```

After extraction, the user moves `production_runs/` into the repo's `results/` directory:

```bash
tar xzf dr4gm_data_bundle_v0.0.1-rc5.tar.gz
mkdir -p results
mv production_runs results/
```

…then `results/production_runs/...` matches the layout that `regen_ensemble_figures.sh` and `fetch_figures_for_publication.sh` expect.

## Expected tarball size

- Uncompressed sum: 14,333,783 bytes ≈ **13.7 MB**
- NPZ files are already DEFLATE-compressed internally → `tar.gz` adds little. Realistic tarball: **~14 MB**.

## Versioning / metadata

- Tarball name: `dr4gm_data_bundle_v0.0.1-rc5.tar.gz` (track to repo tag `v0.0.1-rc5` per `CITATION.cff:11` and `release_notes_v0.0.1-rc5.md`).
- Bump the suffix in lock-step with future repo releases when the per-scenario NPZ contents change.
- Zenodo `description` should reference: (a) the DR4GM GitHub release tag, (b) the manuscript DOI when assigned, (c) AGPLv3 license, (d) license/attribution for the upstream simulation codes whose outputs were processed.

## Not in the bundle (intentionally)

- Raw simulation data (`reference/datasets/`, ~109 GB) — too large; provenance with each contributing modeling group.
- Full per-station baseline at native resolution (`reference/results_original_resolution/`, ~92 GB).
- Intermediate pipeline NPZ (`processed_stations.npz`, `stations.npz`, `velocities.npz`) — derivable from the raw data via `utils/run_all.sh`.
- Pre-rendered figures (`results/production_runs/figs_to_publish/*.png`) — the point is to regenerate them from the data.
- Logs, `.DS_Store`, etc.

## Manifest generation (deferred — do not run yet)

When the user approves this plan, generate the bundle with:

```bash
# from repo root
SCEN=(eqdyna/0001.A.100m eqdyna/0001.B.100m eqdyna/0001.C.100m
      fd3d/ncent.sd4 fd3d/ncent.sd8 fd3d/nleft.sd4 fd3d/nleft.sd8
      fd3d/nright.sd4 fd3d/nright.sd8
      mafe/1 mafe/2 mafe/3
      seissol/1 seissol/3 seissol/4 seissol/5
      waveqlab3d/a24 waveqlab3d/c24 waveqlab3d/d24
      sord/1/sord_scenario
      specfem3d/1 specfem3d/2 specfem3d/3)
STAGE=$(mktemp -d)/production_runs
for s in "${SCEN[@]}"; do
  mkdir -p "$STAGE/$s"
  for f in ground_motion_metrics.npz gm_statistics.npz geometry.npz; do
    cp "results/production_runs/$s/$f" "$STAGE/$s/" 2>/dev/null || true
  done
done
# Generate MANIFEST.txt with sha256 + byte sizes
( cd "$STAGE/.." && find production_runs -type f | sort | \
  xargs -I{} sh -c 'printf "%s  %s  %s\n" "$(shasum -a 256 "{}" | cut -d" " -f1)" "$(stat -f %z "{}")" "{}"' \
  > production_runs/MANIFEST.txt )
tar czf dr4gm_data_bundle_v0.0.1-rc5.tar.gz -C "$STAGE/.." production_runs
```

**Do not execute until the user has approved this plan.**
