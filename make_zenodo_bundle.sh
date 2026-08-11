#!/bin/bash
#
# make_zenodo_bundle.sh — build the Zenodo data bundle for the DR4GM paper pair.
#
# The bundle holds only what is needed to regenerate Figures 11-19 of both
# papers: three NPZ archives per scenario. Raw simulation output (~109 GB) and
# the rupture-summary images contributed by each group are NOT included.
#
# Usage:
#   bash make_zenodo_bundle.sh [output_dir]     # default: one level above repo
#
# Output: dr4gm_data_v<VERSION>.tar.gz + printed SHA-256.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROD="$REPO_ROOT/results/production_runs"
OUTDIR="${1:-$(dirname "$REPO_ROOT")}"
VERSION="$(grep '^version:' "$REPO_ROOT/CITATION.cff" | tr -d '"' | awk '{print $2}')"
NAME="dr4gm_data_v${VERSION}"
STAGE="$(mktemp -d)/$NAME"

# Scenarios in the published ensemble. seissol/2 is excluded: it did not reach
# the target Mw 7 (median SA(T=1s) ~5x below the other SeisSol realizations).
SCENARIOS=(
    eqdyna/0001.A.100m eqdyna/0001.B.100m eqdyna/0001.C.100m
    fd3d/ncent.sd4 fd3d/ncent.sd8 fd3d/nleft.sd4
    fd3d/nleft.sd8 fd3d/nright.sd4 fd3d/nright.sd8
    mafe/1 mafe/2 mafe/3
    seissol/1 seissol/3 seissol/4 seissol/5
    waveqlab3d/a24 waveqlab3d/c24 waveqlab3d/d24
    sord/1/sord_scenario
    specfem3d/1 specfem3d/2 specfem3d/3
)

echo "Staging ${#SCENARIOS[@]} scenarios from $PROD"
mkdir -p "$STAGE/production_runs"
n_files=0
for s in "${SCENARIOS[@]}"; do
    mkdir -p "$STAGE/production_runs/$s"
    for f in ground_motion_metrics.npz gm_statistics.npz geometry.npz; do
        if [ -f "$PROD/$s/$f" ]; then
            cp "$PROD/$s/$f" "$STAGE/production_runs/$s/$f"
            n_files=$((n_files + 1))
        fi
    done
done
echo "  $n_files NPZ files"
# SORD and SPECFEM3D ship binned statistics only, so they contribute 2 files
# each rather than 3; that is expected, not a missing-file error.

cat > "$STAGE/README.md" <<EOF
# DR4GM data bundle v${VERSION}

Post-processed ground motion archives for:

- Withers et al., *Physics-Based Simulation of Near-Source Ground Motions*
  (science paper)
- Liu et al., *DR4GM: A Data Standard and Verified Reference Implementation*
  (software companion)

## Scope

Reproduces **Figures 11-19** of both papers. Does **not** include raw simulation
output (~109 GB) or the rupture-summary images of the science paper's Figure 10,
which were contributed directly by each modeling group.

## Contents

\`\`\`
production_runs/<code>/<scenario>/
  ground_motion_metrics.npz   per-station PGA/PGV/PGD/RSA/CAV
  gm_statistics.npz           Rjb-binned geometric means and log std devs
  geometry.npz                fault trace, strike, dip
\`\`\`

${#SCENARIOS[@]} scenarios from 7 codes. SORD and SPECFEM3D contributed binned
statistics only and therefore have no \`ground_motion_metrics.npz\`.

## Reproducing the figures

\`\`\`bash
git clone https://github.com/dunyuliu/DR4GM.git && cd DR4GM
source install.sh
mkdir -p results && tar xzf ${NAME}.tar.gz -C results/
bash regen_ensemble_figures.sh
\`\`\`

Output lands in \`results/production_runs/figs_to_publish/\`.

## Integrity

\`MANIFEST.sha256\` lists a SHA-256 for every archive:

\`\`\`bash
shasum -a 256 -c MANIFEST.sha256
\`\`\`

## License

CC-BY-4.0. The DR4GM software is separately licensed AGPLv3
(https://github.com/dunyuliu/DR4GM).
EOF

( cd "$STAGE" && find production_runs -name '*.npz' | sort | xargs shasum -a 256 > MANIFEST.sha256 )

TARBALL="$OUTDIR/${NAME}.tar.gz"
( cd "$STAGE" && tar czf "$TARBALL" production_runs MANIFEST.sha256 README.md )
rm -rf "$(dirname "$STAGE")"

echo
echo "Bundle:   $TARBALL"
echo "Size:     $(du -h "$TARBALL" | cut -f1)"
echo "SHA-256:  $(shasum -a 256 "$TARBALL" | cut -d' ' -f1)"
