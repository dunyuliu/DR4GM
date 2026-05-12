#!/bin/bash
set -euo pipefail

PROD=/Users/dliu/scratch/dr4gm.dev/dr4gm/results/production_runs
UTILS=/Users/dliu/scratch/dr4gm.dev/dr4gm/utils
ENS="$PROD/ensemble"

cd "$UTILS"

ALL_SCENARIOS=(
    eqdyna/0001.A.100m
    eqdyna/0001.B.100m
    eqdyna/0001.C.100m
    fd3d/ncent.sd4
    fd3d/ncent.sd8
    fd3d/nleft.sd4
    fd3d/nleft.sd8
    fd3d/nright.sd4
    fd3d/nright.sd8
    mafe/1
    mafe/2
    mafe/3
    seissol/1
    seissol/2
    seissol/3
    seissol/4
    seissol/5
    waveqlab3d/a24
    waveqlab3d/c24
    waveqlab3d/d24
    sord/1/sord_scenario
    specfem3d/1
    specfem3d/2
    specfem3d/3
)

echo "=== Regenerating ensemble figures (24 scenarios) ==="
PYTHONPATH="$UTILS" python visualize_ensemble_stats.py \
    --input-dir "$PROD" \
    --output-dir "$ENS" \
    --add-gmpe \
    "${ALL_SCENARIOS[@]}"

echo ""
echo "=== Regenerating Fig 12 per-group panels ==="
VEL_SCENARIOS=(
    eqdyna/0001.A.100m eqdyna/0001.B.100m eqdyna/0001.C.100m
    fd3d/ncent.sd4 fd3d/ncent.sd8 fd3d/nleft.sd4 fd3d/nleft.sd8 fd3d/nright.sd4 fd3d/nright.sd8
    mafe/1 mafe/2 mafe/3
    seissol/1 seissol/2 seissol/3 seissol/4 seissol/5
    waveqlab3d/a24 waveqlab3d/c24 waveqlab3d/d24
)
PYTHONPATH="$UTILS" python plot_pergroup_ens_figure12.py \
    --input-dir "$PROD" \
    --output-dir "$ENS" \
    --period 1.0 --magnitude 7.0 --vs30 760 \
    "${VEL_SCENARIOS[@]}"

echo ""
echo "=== Collecting manuscript figures ==="
bash /Users/dliu/scratch/dr4gm.dev/dr4gm/fetch_figures_for_publication.sh "$PROD"
