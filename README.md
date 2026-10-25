# DR4GM — Dynamic Rupture for Ground Motion

DR4GM processes high-resolution, physics-based earthquake dynamic-rupture
simulation output into ground-motion intensity measures (PGA, PGV, PGD, CAV,
RSA) and distance-binned statistics, and compares them against NGA-West2
GMPEs. It supports seven simulation codes: EQdyna, SeisSol, WaveQLab3D,
FD3D_TSN, SPECFEM3D, MAFE, SORD.

## Install

```bash
git clone https://github.com/dunyuliu/DR4GM.git
cd DR4GM
bash scripts/install.sh
```

This creates `.venv/` and installs the pinned Python dependencies into it.
Activate it in each new shell:

```bash
source .venv/bin/activate
```

One optional dependency, `openquake.engine`'s GDAL-based geospatial stack,
is not always installable in a clean environment. If it is missing, every
figure still generates except **Figure14B** (SA bias vs. period against
NGA-West2 GMPEs), and `regen_ensemble_figures.sh` prints a named
`SKIPPED: Figure14B.png` notice rather than failing. `scripts/install.sh`
reports whether GMPE comparison is available right after setup.

## Run one example

A small (~560 KB) EQdyna fixture ships in the repo so you can run the full
per-scenario pipeline without downloading anything:

```bash
bash src/utils/run_all.sh \
  tests/fixture_reference/eqdyna_0001A_fixture/raw \
  eqdyna \
  results/readme_example
```

This converts the raw output, subsets stations to a 1 km grid, computes GM
metrics and distance-binned statistics, and renders maps/plots. Expect it to
finish in well under a minute and to populate `results/readme_example/`
with `ground_motion_metrics.npz`, `gm_statistics.npz`, and PNGs such as
`gmPGAStatsVsR.png` and `ground_motion_summary.png`.

To run the same pipeline on any other simulation's raw output:

```bash
bash src/utils/run_all.sh <input_dir> <code> <output_dir>
# <code> is one of: eqdyna, fd3d, waveqlab3d, seissol, sord, mafe, specfem3d
```

## Reproduce the manuscript figures (Figs 11–19)

The post-processed NPZs behind the manuscript figures (22 scenarios × 3 NPZ
each, ~14 MB) are archived on Zenodo, concept DOI pending — the record is
not public yet, so the usual `curl`/Zenodo recipe does not work for outside
readers today. Once the record is public this will reproduce Figs 11–19:

```bash
curl -L -o dr4gm_data.tar.gz "<Zenodo record URL, once minted>"
mkdir -p results && tar xzf dr4gm_data.tar.gz -C results/
bash scripts/regen_ensemble_figures.sh
```

Check the repository's
[GitHub Releases](https://github.com/dunyuliu/dr4gm/releases) for the
current status; if a bundle is attached to a release there, download it from
there instead of Zenodo and skip straight to the `tar xzf` step. Figures land
in `results/production_runs/figs_to_publish/Figure<NN><letter>.png` (code
letters: A=WaveQLab3D, B=SeisSol, C=SORD, D=EQdyna, E=MAFE, F=SPECFEM3D,
G=FD3D_TSN). For the formulas behind each figure (τ vs φ vs σ, GMRotD50, Rjb
segment distance, etc.), see [`docs/user/FORMULAS.md`](docs/user/FORMULAS.md).

## Where outputs land

| Command | Output |
|---|---|
| `src/utils/run_all.sh <in> <code> <out>` | `<out>/` — one scenario's NPZs and plots |
| `tests/run_tests.sh` | `results/regression/` |
| `tests/run_tests.sh --all` | `results/production_runs/` |
| `scripts/regen_ensemble_figures.sh` | `results/production_runs/figs_to_publish/` |

## Tests

```bash
bash tests/check_layout.sh         # repo layout gate (no network, no data needed)
```

The full regression suite (`tests/run_tests.sh`) and the bundle end-to-end
test (`tests/run_e2e_bundle.sh`) need the ~199 GB raw reference dataset or
the Zenodo bundle respectively, neither of which is public yet — see
`tests/run_tests.sh` and `tests/run_e2e_bundle.sh` headers for usage once you
have one of those inputs.

## Optional interfaces

| Dir | What | Entry point |
|---|---|---|
| `src/gui/` | Tkinter desktop GUI for the 6-phase workflow | `python src/gui/tkGUI_dr4gm_new.py` |
| `web/` | Streamlit interactive explorer ([live demo](https://dr4gm-web-explorer.streamlit.app)) | `streamlit run web/dr4gm_interactive_explorer.py` |

## Dependencies

Pinned in [`requirements.txt`](requirements.txt): `numpy`, `scipy`,
`matplotlib`, `pandas`, `netCDF4`, `h5py`, `openquake.engine` (GMPE
overlays), `streamlit`/`pillow` (web app). `scripts/install.sh` installs all
of these into `.venv/`. No GPU or special hardware is required; the examples
above run on a laptop.

## Citing DR4GM

See [`CITATION.cff`](CITATION.cff) for the full author list, ORCID, and
version metadata (GitHub's "Cite this repository" button reads it directly).

## Getting help

Open an [issue](https://github.com/dunyuliu/dr4gm/issues) on this
repository.

## Acknowledgments

Bundles the **GMPE Strong Motion Modeller's Toolkit** (`src/gmpe-smtk/`) by
the **GEM Foundation** (© 2014–2018, AGPLv3,
<https://github.com/GEMScienceTools/gmpe-smtk>). See
`src/gmpe-smtk/LICENSE` and `src/gmpe-smtk/LOCAL_MODIFICATIONS.md`.

Built with assistance from **[Claude Code](https://github.com/anthropics/claude-code)**.

## License

AGPLv3, © 2024–2026 Dunyu Liu, Institute for Geophysics, The University of
Texas at Austin. See [`LICENSE`](LICENSE).
