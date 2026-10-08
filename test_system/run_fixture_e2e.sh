#!/bin/bash
#
# run_fixture_e2e.sh — CI-facing whole-workflow e2e on a tiny raw fixture.
#
# Runs the real pipeline (convert -> subset -> GM metrics -> stats) on a
# cropped raw fixture and diffs the output against a FROZEN "light reference"
# committed alongside it. Uses the same float32-aware tolerance convention as
# test_system/diff_gm_metrics.py / run_tests.sh (1e-6 rel for float32 input,
# 1e-12 otherwise).
#
# This script is CI-runnable: it reads ONLY committed repo paths (the fixture
# + the frozen light reference) and its own fresh temp dir. It NEVER reads
# reference/ (the 199 GB raw dataset tree) and never writes into the
# fixture_reference/ tree.
#
# The frozen light reference must be RE-DERIVED from the full reference/ tree
# (test_system/derive_light_reference.sh) whenever the full oracle in
# test_system/reference_results/ is re-blessed. Never the reverse: this
# script must never be used to regenerate the light reference.
#
# Usage: bash test_system/run_fixture_e2e.sh [CODE]
#   CODE defaults to "seissol" (170-station seissol_sim1_fixture, unchanged
#   behavior). Other supported CODE values add their own fixture dir +
#   converter below; see the FIXTURE_DIR/CONVERTER case statement.

set -u
set -o pipefail

cd "$(dirname "$0")/.."
REPO="$(pwd)"
UTILS="$REPO/src/utils"
TEST_DIR="$REPO/test_system"

CODE="${1:-seissol}"
case "$CODE" in
    seissol)
        FIXTURE_DIR="seissol_sim1_fixture"
        CONVERTER="seissol_converter_api.py"
        ;;
    eqdyna)
        FIXTURE_DIR="eqdyna_0001A_fixture"
        CONVERTER="eqdyna_converter_api.py"
        ;;
    *)
        echo "FAIL: unknown CODE '$CODE' (supported: seissol, eqdyna)"
        exit 2
        ;;
esac

FIXTURE="$TEST_DIR/fixture_reference/$FIXTURE_DIR"
RAW="$FIXTURE/raw"
LIGHT_REF="$FIXTURE/light_reference"

if [ ! -f "$RAW/fault_geometry.json" ]; then
    echo "FAIL: fixture raw dir not found at $RAW"
    exit 1
fi
if [ ! -f "$LIGHT_REF/ground_motion_metrics.npz" ] || [ ! -f "$LIGHT_REF/gm_statistics.npz" ]; then
    echo "FAIL: frozen light reference not found under $LIGHT_REF"
    exit 1
fi

WORK="$(mktemp -d -t dr4gm_fixture_e2e.XXXXXX)"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

echo "=== Fixture e2e ($CODE): raw -> convert -> subset -> GM metrics -> stats ==="
echo "fixture raw: $RAW"
echo "work dir:    $WORK"

echo "--- Step 1/4 convert ($CODE) ---"
python3 "$UTILS/$CONVERTER" --input_dir "$RAW" --output_dir "$WORK" || { echo "FAIL: convert step"; exit 1; }

echo "--- Step 2/4 subset to 1 km grid ---"
python3 "$UTILS/station_subset_selector.py" \
    --input_npz "$WORK/velocities.npz" \
    --output_npz "$WORK/processed_stations.npz" \
    --grid_resolution 1000 || { echo "FAIL: subset step"; exit 1; }

echo "--- Step 3/4 GM metrics ---"
python3 "$UTILS/npz_gm_processor.py" \
    --velocity_npz "$WORK/processed_stations.npz" \
    --output_dir "$WORK" || { echo "FAIL: GM metrics step"; exit 1; }

echo "--- Step 4/4 GM statistics vs distance ---"
python3 "$UTILS/gm_stats.py" \
    --gm_data "$WORK/ground_motion_metrics.npz" \
    --output_dir "$WORK" \
    --distance_range 0 30000 \
    --distance_bin_size 500 || { echo "FAIL: GM stats step"; exit 1; }

# Sort the freshly computed per-station metrics by station_id so the
# comparison does not depend on any incidental row ordering produced by the
# grid selector (the frozen light reference is stored sorted by station_id).
python3 - "$WORK/ground_motion_metrics.npz" "$WORK/metrics_sorted.npz" << 'PYEOF'
import sys
import numpy as np
src_path, dst_path = sys.argv[1], sys.argv[2]
src = np.load(src_path)
ids = src["station_ids"]
order = np.argsort(ids)
out = {}
for k in src.files:
    arr = src[k]
    if hasattr(arr, "shape") and arr.ndim >= 1 and arr.shape[0] == len(ids):
        out[k] = arr[order]
    else:
        out[k] = arr
np.savez(dst_path, **out)
PYEOF

echo "--- Diff: per-station GM metrics vs frozen light reference ---"
rc_metrics=0
python3 "$TEST_DIR/diff_gm_metrics.py" \
    "$LIGHT_REF/ground_motion_metrics.npz" \
    "$WORK/metrics_sorted.npz" \
    --input "$WORK/processed_stations.npz" || rc_metrics=1

echo "--- Diff: GM statistics vs frozen light reference ---"
rc_stats=0
python3 "$TEST_DIR/diff_gm_metrics.py" \
    "$LIGHT_REF/gm_statistics.npz" \
    "$WORK/gm_statistics.npz" \
    --input "$WORK/processed_stations.npz" || rc_stats=1

echo "================================================================"
if [ $rc_metrics -eq 0 ] && [ $rc_stats -eq 0 ]; then
    echo "RESULT: PASS"
    exit 0
else
    echo "RESULT: FAIL (metrics_rc=$rc_metrics stats_rc=$rc_stats)"
    exit 1
fi
