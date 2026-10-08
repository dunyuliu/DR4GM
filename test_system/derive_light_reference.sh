#!/bin/bash
#
# derive_light_reference.sh — LOCAL-ONLY. Re-derive and prove the frozen
# "light reference" for test_system/run_fixture_e2e.sh.
#
# Needs the full ~199 GB reference/ tree (chmod -R a-w). NOT CI-runnable.
# Takes ~30-40 min for the seissol scenario (GM metrics on ~3300 stations).
#
# What it proves (binding design, PATHWAY_FORWARD.md row 13):
#   light-reference (frozen, fixture-derived numbers) == full-oracle subset
#   (gm_stats run on the FULL reference's metrics, restricted to the cropped
#   fixture's station IDs — sliced from the full oracle, never regenerated
#   from the fixture alone).
#
# Procedure:
#   1. Convert the committed tiny raw fixture -> its own GM metrics + stats
#      ("fixture-computed").
#   2. Run the FULL seissol converter + 1km subset + GM metrics fresh from
#      reference/ ("full-oracle"), restricted to the SAME raw dataset the
#      fixture was cropped from (reference/datasets/seissol/sim1_big_0123).
#   3. Slice the full-oracle metrics to the fixture's station IDs, sort by
#      station_id, run gm_stats.py on the slice ("full-oracle subset").
#   4. Diff fixture-computed vs full-oracle subset with the SAME
#      diff_gm_metrics.py / tolerance convention run_tests.sh uses.
#   5. If (and only if) the diff passes, overwrite the committed
#      light_reference/*.npz with the full-oracle-subset-derived files (the
#      light reference is always derived FROM the full reference, never from
#      the fixture's own run, even though step 1 and step 3 agree numerically).
#
# Re-run this (and re-commit the result) whenever
# test_system/reference_results/seissol/1/ is re-blessed. Never derive the
# light reference the other way around (never hand-edit it to match a
# fixture run).
#
# Usage:
#   bash test_system/derive_light_reference.sh [REFERENCE_DIR]
#   REFERENCE_DIR defaults to <repo_root>/reference (a normal checkout, not
#   this worktree, since reference/ is gitignored and not copied into
#   worktrees).

set -u
set -o pipefail

cd "$(dirname "$0")/.."
REPO="$(pwd)"
UTILS="$REPO/utils"
TEST_DIR="$REPO/test_system"
FIXTURE="$TEST_DIR/fixture_reference/seissol_sim1_fixture"
RAW_FIXTURE="$FIXTURE/raw"
LIGHT_REF="$FIXTURE/light_reference"
REFERENCE_DIR="${1:-$REPO/reference}"
RAW_FULL="$REFERENCE_DIR/datasets/seissol/sim1_big_0123"

if [ ! -d "$RAW_FULL" ]; then
    echo "FAIL: full reference raw dir not found at $RAW_FULL"
    echo "Pass the path to a checkout with reference/ as \$1, e.g.:"
    echo "  bash test_system/derive_light_reference.sh /home/utig5/dliu/dr4gm/dr4gm/reference"
    exit 1
fi

WORK="$(mktemp -d -t dr4gm_derive_light_ref.XXXXXX)"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT
mkdir -p "$WORK/fixture_run" "$WORK/full_run"

echo "work dir: $WORK"

echo "=== Step 1: fixture-computed (raw fixture -> metrics -> stats) ==="
python3 "$UTILS/seissol_converter_api.py" --input_dir "$RAW_FIXTURE" --output_dir "$WORK/fixture_run" || exit 1
python3 "$UTILS/station_subset_selector.py" \
    --input_npz "$WORK/fixture_run/velocities.npz" \
    --output_npz "$WORK/fixture_run/processed_stations.npz" \
    --grid_resolution 1000 || exit 1
python3 "$UTILS/npz_gm_processor.py" \
    --velocity_npz "$WORK/fixture_run/processed_stations.npz" \
    --output_dir "$WORK/fixture_run" || exit 1
python3 "$UTILS/gm_stats.py" \
    --gm_data "$WORK/fixture_run/ground_motion_metrics.npz" \
    --output_dir "$WORK/fixture_run" \
    --distance_range 0 30000 --distance_bin_size 500 || exit 1

FIXTURE_N=$(python3 -c "import numpy as np; print(len(np.load('$WORK/fixture_run/ground_motion_metrics.npz')['station_ids']))")
echo "fixture station count: $FIXTURE_N"

echo "=== Step 2: full-oracle (full reference/ raw -> metrics), ~30-40 min ==="
python3 "$UTILS/seissol_converter_api.py" --input_dir "$RAW_FULL" --output_dir "$WORK/full_run" || exit 1
python3 "$UTILS/station_subset_selector.py" \
    --input_npz "$WORK/full_run/velocities.npz" \
    --output_npz "$WORK/full_run/processed_stations.npz" \
    --grid_resolution 1000 || exit 1
python3 "$UTILS/npz_gm_processor.py" \
    --velocity_npz "$WORK/full_run/processed_stations.npz" \
    --output_dir "$WORK/full_run" || exit 1

echo "=== Step 3: slice full-oracle metrics to fixture's station IDs, sort, run gm_stats ==="
python3 - "$WORK/full_run/ground_motion_metrics.npz" "$FIXTURE_N" "$WORK/full_oracle_subset.npz" << 'PYEOF'
import sys
import numpy as np
full_path, n_str, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(n_str)
full = np.load(full_path)
ids = full["station_ids"]
mask = np.isin(ids, np.arange(n))
order = np.argsort(ids[mask])
out = {}
for k in full.files:
    arr = full[k]
    if hasattr(arr, "shape") and arr.ndim >= 1 and arr.shape[0] == len(ids):
        out[k] = arr[mask][order]
    else:
        out[k] = arr
if out["station_ids"].shape[0] != n:
    raise SystemExit(
        f"FAIL: expected {n} stations in full-oracle slice, got {out['station_ids'].shape[0]}"
    )
np.savez(out_path, **out)
PYEOF
[ $? -eq 0 ] || exit 1

mkdir -p "$WORK/full_oracle_subset_dir"
cp "$WORK/full_run/geometry.npz" "$WORK/full_oracle_subset_dir/geometry.npz"
mv "$WORK/full_oracle_subset.npz" "$WORK/full_oracle_subset_dir/ground_motion_metrics.npz"
python3 "$UTILS/gm_stats.py" \
    --gm_data "$WORK/full_oracle_subset_dir/ground_motion_metrics.npz" \
    --output_dir "$WORK/full_oracle_subset_dir" \
    --distance_range 0 30000 --distance_bin_size 500 || exit 1

echo "=== Step 4: PROOF — fixture-computed vs full-oracle subset ==="
python3 - "$WORK/fixture_run/ground_motion_metrics.npz" "$WORK/fixture_run/metrics_sorted.npz" << 'PYEOF'
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

rc=0
echo "--- metrics ---"
python3 "$TEST_DIR/diff_gm_metrics.py" \
    "$WORK/full_oracle_subset_dir/ground_motion_metrics.npz" \
    "$WORK/fixture_run/metrics_sorted.npz" \
    --input "$WORK/fixture_run/processed_stations.npz" || rc=1
echo "--- stats ---"
python3 "$TEST_DIR/diff_gm_metrics.py" \
    "$WORK/full_oracle_subset_dir/gm_statistics.npz" \
    "$WORK/fixture_run/gm_statistics.npz" \
    --input "$WORK/fixture_run/processed_stations.npz" || rc=1

if [ $rc -ne 0 ]; then
    echo "PROOF FAILED — NOT freezing light reference. Investigate before re-committing."
    exit 1
fi

echo "=== PROOF PASSED — freezing light reference from the full-oracle subset ==="
cp "$WORK/full_oracle_subset_dir/ground_motion_metrics.npz" "$LIGHT_REF/ground_motion_metrics.npz"
cp "$WORK/full_oracle_subset_dir/gm_statistics.npz" "$LIGHT_REF/gm_statistics.npz"
echo "Wrote $LIGHT_REF/ground_motion_metrics.npz and gm_statistics.npz — review and commit."
exit 0
