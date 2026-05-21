#!/bin/bash
set -euo pipefail

# Resolve repo root from this script's location so the pipeline is portable.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROD="$REPO_ROOT/results/production_runs"
UTILS="$REPO_ROOT/utils"
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
    # seissol/2 — excluded: did not reach Mw 7 (median SA(T=1s) ≈ 5× lower than others)
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

echo "=== Regenerating Fig 11 per-scenario RSA(T=1s) maps ==="
# MAFE uses xlim=±10 km (perpendicular fault-normal extent is bounded at 10 km);
# all other codes use xlim=±20 km. ylim=±40 km is shared across all panels so
# 10 km corresponds to the same physical length on every panel.
for s in "${ALL_SCENARIOS[@]}"; do
    NPZ="$PROD/$s/ground_motion_metrics.npz"
    OUT="$PROD/$s"
    if [ ! -f "$NPZ" ]; then
        continue  # SORD has stats only (no per-station NPZ) → no Fig 11 panel
    fi
    if [[ "$s" == mafe/* ]]; then
        XLIM=(--xlim -10 10)
    else
        XLIM=(--xlim -20 20)
    fi
    PYTHONPATH="$UTILS" python visualize_gm_maps.py \
        --gm_npz "$NPZ" --output_dir "$OUT" --metric RSA_T_1.000 \
        --vmin 0.04 --vmax 1.5 "${XLIM[@]}" --ylim -40 40 2>&1 | tail -1
done

echo ""
echo "=== Regenerating ensemble figures (Figs 13-19) ==="
PYTHONPATH="$UTILS" python visualize_ensemble_stats.py \
    --input-dir "$PROD" \
    --output-dir "$ENS" \
    --add-gmpe \
    "${ALL_SCENARIOS[@]}"

echo ""
echo "=== Regenerating Fig 12 per-group panels ==="
# seissol/2 dropped — that simulation did not reach Mw 7 (median SA(T=1s) ≈ 0.027 g
# vs ≈ 0.12 g for the other 4 seissol scenarios).
# sord and specfem3d are included via gm_statistics.npz fallback (no per-station
# scatter, just the binned median line).
FIG12_SCENARIOS=(
    eqdyna/0001.A.100m eqdyna/0001.B.100m eqdyna/0001.C.100m
    fd3d/ncent.sd4 fd3d/ncent.sd8 fd3d/nleft.sd4 fd3d/nleft.sd8 fd3d/nright.sd4 fd3d/nright.sd8
    mafe/1 mafe/2 mafe/3
    seissol/1 seissol/3 seissol/4 seissol/5
    waveqlab3d/a24 waveqlab3d/c24 waveqlab3d/d24
    sord/1/sord_scenario
    specfem3d/1 specfem3d/2 specfem3d/3
)
PYTHONPATH="$UTILS" python plot_pergroup_ens_figure12.py \
    --input-dir "$PROD" \
    --output-dir "$ENS" \
    --period 1.0 --magnitude 7.0 --vs30 760 \
    --ylim 0.003 2.0 --xlim 0.5 40 \
    "${FIG12_SCENARIOS[@]}"

echo ""
echo "=== Collecting manuscript figures ==="
bash "$REPO_ROOT/fetch_figures_for_publication.sh" "$PROD"
