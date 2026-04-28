# DR4GM — Data Repository for Ground Motion

Process high-resolution physics-based earthquake simulations into ground-motion
metrics (PGA, PGV, PGD, CAV, RSA) and distance-binned statistics, then compare
against NGA-West2 GMPEs.

**Author:** Dunyu Liu (<dliu@ig.utexas.edu>), Institute for Geophysics,
The University of Texas at Austin.

## Install

```bash
git clone https://github.com/dunyuliu/dr4gm.git
cd dr4gm
source install.sh        # or: docker pull dunyuliu/dr4gm
```

`install.sh` installs Python dependencies and exports `PATH` /
`PYTHONPATH` for `gmpe-smtk/` and `utils/` in the current shell. Source
it (don't `bash` it) so the env vars stick.

`gmpe-smtk/` is vendored in-tree (AGPLv3, © GEM Foundation). No separate clone.

## Run one case

```bash
cd utils
./run_all.sh ../datasets/eqdyna/eqdyna.0001.A.100m eqdyna ./results/eqdyna_A_100m
```

Outputs land in `./results/eqdyna_A_100m/`:

| file | from | what it is |
|---|---|---|
| `stations.npz`, `velocities.npz`, `geometry.npz` | converter | standardized inputs |
| `processed_stations.npz` | `station_subset_selector.py` | grid-subset (or all) |
| `ground_motion_metrics.npz` | `npz_gm_processor.py` | per-station PGA/PGV/PGD/CAV/RSA |
| `gm_statistics.npz` | `gm_stats.py` | distance-binned mean/std/min/max |
| `*.png` | `visualize_gm_maps.py`, `visualize_gm_stats.py` | maps + attenuation plots |

For a single command per step, see `utils/run_all.sh`.

## Supported simulation codes

| code | converter |
|---|---|
| EQDyna | `eqdyna_converter_api.py` |
| FD3D | `fd3d_converter_api.py` |
| SeisSol | `seissol_converter_api.py` |
| WaveQLab3D | `waveqlab3d_converter_api.py` |
| SPECFEM3D | `specfem3d_converter_api.py` |
| MAFE (GMPE comparison) | `mafe_converter_api.py` |
| SORD (pre-computed SA) | `sord_plot_converter_api.py` |

MAFE and SORD bypass `npz_gm_processor.py` because they ship pre-computed
statistics; their converters write `gm_statistics.npz` directly.

## Pipeline

```
raw → converter → npz_gm_processor → gm_stats → visualize_gm_{maps,stats}
```

`run_all.sh` chains these. Use it for one scenario; for cross-scenario plots
(GMPE overlays, response-spectra ensembles) use:

```bash
python utils/visualize_ensemble_stats.py \
    --input-dir results/regression --output-dir results/regression/ensemble --add-gmpe \
    eqdyna/0001.A.100m fd3d/ncent.sd4 mafe/1 seissol/1 waveqlab3d/a24
```

## Reference baseline (`reference/`)

Time-frozen, read-only inputs and outputs of the per-station `gmpe-smtk`
pipeline.

```
reference/
├── datasets/                       (~109 GB) raw simulation data — 7 codes
└── results_original_resolution/    ( ~92 GB) per-station outputs at native
                                              resolution (21 scenarios, 5 codes)
```

Frozen 2026-04-27 (`chmod -R a-w`). The repo root keeps a symlink
`datasets/ → reference/datasets/`. Use as ground-truth when validating
refactors of `utils/npz_gm_processor.py`. Direct fresh runs to a path
under `results/`; never modify `reference/` in place. See
`reference/README.md`.

The lightweight regression baseline used by the test suite (1 km grid,
five canonical scenarios) lives separately at
`test_system/reference_results/` and is shipped with the repo.

## Regression test

```bash
bash test_system/run_tests.sh           # 5 canonical scenarios → results/regression
bash test_system/run_tests.sh --all     # all 20 scenarios   → results/production_runs
```

**The test verifies the pipeline at a uniform 1 km station grid.** Each
scenario runs:

1. raw → standardized NPZ (per-code converter)
2. `station_subset_selector.py --grid_resolution 1000` (downsample to 1 km grid)
3. `npz_gm_processor.py` (per-station PGA/PGV/PGD/CAV/RSA)
4. diff fresh `ground_motion_metrics.npz` against the bundled 1 km
   baseline at `test_system/reference_results/<code>/<scenario>/`

The reference set is shipped with the repo (~3 MB total — five
`ground_motion_metrics.npz` files, one per canonical scenario). Pass =
float32-aware bit equivalence (`1e-6` rel for float32 input velocities,
`1e-12` rel otherwise; see `test_system/diff_gm_metrics.py`). Scenarios
without a reference (e.g. the extra 15 in `--all`) are reported NOREF.

**Wall-clock on Apple M3 / 8 cores / 24 GB RAM:**

| mode | scenarios | total | per-scenario range |
|---|---|---|---|
| default | 5  | ~14 min | 1.5–5 min |
| `--all` | 20 | ~59 min | 1.5–5 min |

## Manuscript figures

After per-scenario figures and `ensemble/` plots have been produced
(`utils/run_all.sh` per scenario plus `utils/visualize_ensemble_stats.py`
for the ensemble), collect manuscript-numbered parts with:

```bash
bash fetch_figures_for_publication.sh results/production_runs
# → results/production_runs/figs_to_publish/Figure{11,12}<L><n>.png
#   results/production_runs/figs_to_publish/Figure{13A,13B,13C,14,15A,15B,15C,16,19A,19B}.png
```

Code letters: `A=eqdyna`, `B=fd3d`, `C=mafe`, `D=seissol`, `E=waveqlab3d`,
`F=specfem3d`, `G=sord` (realization-number suffix per scenario).
Pure cp; assumes the upstream figures already exist.

## Other interfaces

| dir | what | entry point |
|---|---|---|
| `gui/` | Tkinter desktop GUI for the 6-phase processing workflow | `python gui/tkGUI_dr4gm_new.py` |
| `web/` | Streamlit interactive explorer (cloud-deployable) | `streamlit run web/dr4gm_interactive_explorer.py` |
| `demo/` | One-shot scripts that generate the capability/business diagrams in `docs/` | `python demo/generate_capability_diagram.py` |

The core pipeline (converters + `npz_gm_processor` + ensemble stats)
runs entirely from `utils/`; the dirs above are optional.

## Dependencies

`numpy`, `scipy`, `matplotlib`, `pandas`, `netCDF4`, `h5py`, `openquake.engine`
(for GMPE overlays). `streamlit`, `pillow` for the optional GUI / web app.

## Citation

```
Liu, D. DR4GM: Data Repository for Ground Motion — a platform for processing
physics-based earthquake simulation data. Institute for Geophysics, The
University of Texas at Austin.
```

## Acknowledgments

Bundles the **GMPE Strong Motion Modeller's Toolkit** (`gmpe-smtk/`) by the
**GEM Foundation** (© 2014–2018, AGPLv3,
<https://github.com/GEMScienceTools/gmpe-smtk>) — provides the response-spectrum
and intensity-measure routines. See `gmpe-smtk/LICENSE` and
`gmpe-smtk/LOCAL_MODIFICATIONS.md`.

Built with assistance from **[Claude Code](https://github.com/anthropics/claude-code)**, Anthropic's agentic coding CLI.

## License

AGPLv3, © 2024–2026 Dunyu Liu, Institute for Geophysics, The University of
Texas at Austin. See [`LICENSE`](LICENSE). AGPLv3 §13 applies to network
service deployment.
