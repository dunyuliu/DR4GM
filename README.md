# DR4GM — Data Repository for Ground Motion

Process high-resolution physics-based earthquake simulations into ground-motion
metrics (PGA, PGV, PGD, CAV, RSA) and distance-binned statistics, and compare
against NGA-West2 GMPEs.

**Author**: Dunyu Liu (<dliu@ig.utexas.edu>), Institute for Geophysics, UT Austin.

## Reproduce manuscript Figs 11–19 (from Zenodo data, ~5 min)

```bash
git clone https://github.com/dunyuliu/dr4gm.git
cd dr4gm
source install.sh                                                 # pip + PATH/PYTHONPATH

# Download ~14 MB Zenodo bundle of post-processed NPZs (22 scenarios × 3 NPZ each)
curl -L -o dr4gm_data.tar.gz \
  https://zenodo.org/record/XXXXXXX/files/dr4gm_data_v0.0.1.tar.gz   # TODO: real DOI
mkdir -p results && tar xzf dr4gm_data.tar.gz -C results/
# After extract: results/production_runs/<code>/<scenario>/{ground_motion_metrics,gm_statistics,geometry}.npz

bash regen_ensemble_figures.sh                                    # Figs 11–19 → figs_to_publish/
```

That's it — every manuscript figure part lands in
`results/production_runs/figs_to_publish/Figure<NN><letter>.png`.

Code letters in filenames: A=WaveQLab3D, B=SeisSol, C=SORD, D=EQdyna,
E=MAFE, F=SPECFEM3D, G=FD3D_TSN.

For the formulas behind each figure (τ vs φ vs σ, GMRotD50, Rjb segment
distance, etc.), see [`FORMULAS.md`](FORMULAS.md).

## Reproduce from raw simulation data (~200 GB, optional)

If you have access to the raw simulation outputs:

```bash
ln -s /path/to/raw_simulation_archive reference
bash run_pipeline.sh                                              # raw → NPZ → figures
```

Raw inputs are available on request from each modeling group; contact
authors of the respective simulation codes (EQdyna, SeisSol, SORD, MAFE,
SPECFEM3D, WaveQLab3D, FD3D_TSN).

## One-scenario pipeline

```bash
cd utils
./run_all.sh ../datasets/eqdyna/eqdyna.0001.A.100m eqdyna ./results/eqdyna_A_100m
```

Produces per-scenario `ground_motion_metrics.npz` + `gm_statistics.npz` +
per-metric maps + attenuation plots in the chosen output directory.

Supported simulation codes & converters (in `utils/`):

| Code | Converter |
|---|---|
| EQdyna | `eqdyna_converter_api.py` |
| FD3D_TSN | `fd3d_converter_api.py` |
| SeisSol | `seissol_converter_api.py` |
| WaveQLab3D | `waveqlab3d_converter_api.py` |
| SPECFEM3D | `specfem3d_converter_api.py` |
| MAFE | `mafe_converter_api.py` |
| SORD | `sord_plot_converter_api.py` |

MAFE/SORD bypass `npz_gm_processor.py` (they ship pre-computed statistics;
converters write `gm_statistics.npz` directly).

## Regression test

```bash
bash test_system/run_tests.sh           # 5 canonical scenarios (~14 min on M3)
bash test_system/run_tests.sh --all     # all 22 scenarios     (~59 min)
```

Diffs fresh `ground_motion_metrics.npz` against the bundled 1 km baseline
(`test_system/reference_results/`, ~3 MB). Pass = float32-aware bit
equivalence (1e-6 rel for float32 inputs, 1e-12 otherwise).

## Optional interfaces

| Dir | What | Entry point |
|---|---|---|
| `gui/` | Tkinter desktop GUI for the 6-phase workflow | `python gui/tkGUI_dr4gm_new.py` |
| `web/` | Streamlit interactive explorer | `streamlit run web/dr4gm_interactive_explorer.py` |

## Dependencies

`numpy`, `scipy`, `matplotlib`, `pandas`, `netCDF4`, `h5py`,
`openquake.engine` (GMPE overlays). `streamlit`, `pillow` for the optional
GUI/web app.

## Citation

```
Liu, D. DR4GM: Data Repository for Ground Motion — a platform for processing
physics-based earthquake simulation data. Institute for Geophysics, The
University of Texas at Austin. (Manuscript in prep.)
```

## Acknowledgments

Bundles the **GMPE Strong Motion Modeller's Toolkit** (`gmpe-smtk/`) by the
**GEM Foundation** (© 2014–2018, AGPLv3,
<https://github.com/GEMScienceTools/gmpe-smtk>). See `gmpe-smtk/LICENSE`
and `gmpe-smtk/LOCAL_MODIFICATIONS.md`.

Built with assistance from **[Claude Code](https://github.com/anthropics/claude-code)**.

## License

AGPLv3, © 2024–2026 Dunyu Liu, Institute for Geophysics, The University of
Texas at Austin. See [`LICENSE`](LICENSE).
