#!/bin/bash
#
# run_tests.sh — DR4GM regression / production runner, sequential.
#
# Default: runs one canonical scenario per simulation code (eqdyna, fd3d,
# mafe, seissol, waveqlab3d). Outputs to results/regression/.
#
# --all: runs every scenario across all codes (20 total). Outputs to
# results/production_runs/. Use this to regenerate the full reference set.
#
# Each scenario runs Step 1 (convert) + Step 2 (subset to a uniform 1 km
# grid via station_subset_selector.py --grid_resolution 1000) + Step 3
# (GM metrics), then diffs ground_motion_metrics.npz against the bundled
# 1 km reference at test_system/reference_results/<scenario>/. Pass =
# float32-aware bit equivalence (1e-6 rel for float32 inputs, 1e-12 rel
# otherwise). Scenarios with no reference are reported NOREF.
# All output streams to stdout so progress is visible live.
#
# To capture: bash test_system/run_tests.sh 2>&1 | tee run.log
#
# Usage:
#   bash test_system/run_tests.sh           # 5 canonical scenarios
#   bash test_system/run_tests.sh --all     # all 20 scenarios

set -u

MODE="regression"
if [ "${1:-}" = "--all" ]; then
    MODE="all"
elif [ -n "${1:-}" ]; then
    echo "Unknown argument: $1"
    echo "Usage: bash test_system/run_tests.sh [--all]"
    exit 2
fi

cd "$(dirname "$0")/.."
REPO="$(pwd)"
TEST_DIR="$REPO/test_system"
UTILS="$REPO/utils"
DATASETS="$REPO/datasets"
REF="$REPO/test_system/reference_results"

if [ "$MODE" = "all" ]; then
    OUT="$REPO/results/production_runs"
    # code  raw_subdir_under_datasets  reference_subdir_under_results
    scenarios=(
        "eqdyna       eqdyna/eqdyna.0001.A.100m   eqdyna/0001.A.100m"
        "eqdyna       eqdyna/eqdyna.0001.B.100m   eqdyna/0001.B.100m"
        "eqdyna       eqdyna/eqdyna.0001.C.100m   eqdyna/0001.C.100m"
        "fd3d         fd3d/SD4/Ncenter            fd3d/ncent.sd4"
        "fd3d         fd3d/SD4/Nleft              fd3d/nleft.sd4"
        "fd3d         fd3d/SD4/Nright             fd3d/nright.sd4"
        "fd3d         fd3d/SD8/Ncenter            fd3d/ncent.sd8"
        "fd3d         fd3d/SD8/Nleft              fd3d/nleft.sd8"
        "fd3d         fd3d/SD8/Nright             fd3d/nright.sd8"
        "mafe         mafe/1                      mafe/1"
        "mafe         mafe/2                      mafe/2"
        "mafe         mafe/3                      mafe/3"
        "seissol      seissol/sim1_big_0123       seissol/1"
        "seissol      seissol/sim2_big_0123       seissol/2"
        "seissol      seissol/sim3_big_0123       seissol/3"
        "seissol      seissol/sim4_big_0123       seissol/4"
        "seissol      seissol/sim5_big_0123       seissol/5"
        "waveqlab3d   waveqlab3d/a24              waveqlab3d/a24"
        "waveqlab3d   waveqlab3d/c24              waveqlab3d/c24"
        "waveqlab3d   waveqlab3d/d24              waveqlab3d/d24"
    )
else
    OUT="$REPO/results/regression"
    scenarios=(
        "eqdyna       eqdyna/eqdyna.0001.A.100m   eqdyna/0001.A.100m"
        "fd3d         fd3d/SD4/Ncenter            fd3d/ncent.sd4"
        "mafe         mafe/1                      mafe/1"
        "seissol      seissol/sim1_big_0123       seissol/1"
        "waveqlab3d   waveqlab3d/a24              waveqlab3d/a24"
    )
fi

REPORT="$OUT/REPORT.txt"
mkdir -p "$OUT"
: > "$REPORT"

N=${#scenarios[@]}

ts() { date '+%H:%M:%S'; }

elapsed_str() {
    local s=$1 m r
    m=$(( s / 60 ))
    r=$(( s - m * 60 ))
    if [ "$m" -gt 0 ]; then printf '%dm%ds' "$m" "$r"; else printf '%ds' "$r"; fi
}

run_one() {
    local code="$1" raw_sub="$2" ref_sub="$3"
    local raw="$DATASETS/$raw_sub"
    local ref_metrics="$REF/$ref_sub/ground_motion_metrics.npz"
    local dst="$OUT/$ref_sub"

    mkdir -p "$dst"

    echo "[$(ts)] === $code  $ref_sub ==="
    echo "raw=$raw"
    echo "ref=$ref_metrics"
    echo "dst=$dst"

    if [ ! -d "$raw" ]; then
        echo "MISSING raw input dir — SKIP"
        echo FAIL > "$dst/STATUS"
        return 1
    fi
    local have_ref=1
    if [ ! -f "$ref_metrics" ]; then
        echo "NOTE: reference $ref_metrics not found — will generate outputs and skip diff"
        have_ref=0
    fi

    local t0

    echo "[$(ts)] Step 1/3 convert ..."
    t0=$SECONDS
    if ! python "$UTILS/${code}_converter_api.py" \
            --input_dir "$raw" --output_dir "$dst"; then
        echo "Step 1 FAILED"
        echo FAIL > "$dst/STATUS"
        return 1
    fi
    echo "[$(ts)] Step 1 done in $(elapsed_str $((SECONDS - t0)))"

    if [ ! -f "$dst/velocities.npz" ]; then
        echo "Step 1 produced no velocities.npz — FAIL"
        echo FAIL > "$dst/STATUS"
        return 1
    fi

    echo "[$(ts)] Step 2/3 subset to 1 km grid ..."
    t0=$SECONDS
    if ! python "$UTILS/station_subset_selector.py" \
            --input_npz "$dst/velocities.npz" \
            --output_npz "$dst/processed_stations.npz" \
            --grid_resolution 1000; then
        echo "Step 2 FAILED"
        echo FAIL > "$dst/STATUS"
        return 1
    fi
    echo "[$(ts)] Step 2 done in $(elapsed_str $((SECONDS - t0)))"

    echo "[$(ts)] Step 3/3 vectorized GM metrics ..."
    t0=$SECONDS
    if ! python "$UTILS/npz_gm_processor.py" \
            --velocity_npz "$dst/processed_stations.npz" \
            --output_dir "$dst"; then
        echo "Step 3 FAILED"
        echo FAIL > "$dst/STATUS"
        return 1
    fi
    echo "[$(ts)] Step 3 done in $(elapsed_str $((SECONDS - t0)))"

    if [ "$have_ref" -eq 0 ]; then
        echo "[$(ts)] Diff skipped (no reference) — outputs ready at $dst"
        echo NOREF > "$dst/STATUS"
        return 0
    fi

    echo "[$(ts)] Diff vs reference ..."
    if python "$TEST_DIR/diff_gm_metrics.py" "$ref_metrics" \
            "$dst/ground_motion_metrics.npz" \
            --input "$dst/processed_stations.npz"; then
        echo "[$(ts)] DIFF PASS"
        echo PASS > "$dst/STATUS"
        return 0
    else
        echo "[$(ts)] DIFF FAIL"
        echo FAIL > "$dst/STATUS"
        return 1
    fi
}

total_t0=$SECONDS
overall_rc=0
n_pass=0
n_noref=0
n_fail=0
echo "[$(ts)] Running $N scenarios sequentially (mode=$MODE, out=$OUT)" | tee -a "$REPORT"
echo                                              | tee -a "$REPORT"

for entry in "${scenarios[@]}"; do
    set -- $entry
    code="$1"; raw_sub="$2"; ref_sub="$3"
    if run_one "$code" "$raw_sub" "$ref_sub"; then
        verdict=$(cat "$OUT/$ref_sub/STATUS" 2>/dev/null || echo "?")
        echo "[$(ts)] ✓ $code/$ref_sub $verdict" | tee -a "$REPORT"
        case "$verdict" in
            PASS)  n_pass=$((n_pass + 1)) ;;
            NOREF) n_noref=$((n_noref + 1)) ;;
            *)     n_fail=$((n_fail + 1)); overall_rc=1 ;;
        esac
    else
        echo "[$(ts)] ✗ $code/$ref_sub FAIL" | tee -a "$REPORT"
        n_fail=$((n_fail + 1))
        overall_rc=1
    fi
    echo "" | tee -a "$REPORT"
done

total_str=$(elapsed_str $((SECONDS - total_t0)))

echo "================================================================" | tee -a "$REPORT"
echo "Summary: pass=$n_pass  noref=$n_noref  fail=$n_fail  total $total_str" | tee -a "$REPORT"
if [ $overall_rc -eq 0 ]; then
    echo "================================================================" | tee -a "$REPORT"
    echo "  RESULT: PASS"                                                    | tee -a "$REPORT"
    echo "================================================================" | tee -a "$REPORT"
else
    echo "================================================================" | tee -a "$REPORT"
    echo "  RESULT: FAIL — see $REPORT"                                      | tee -a "$REPORT"
    echo "================================================================" | tee -a "$REPORT"
fi

exit $overall_rc
