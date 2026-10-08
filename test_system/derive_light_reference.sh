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
# Re-run this (and re-commit the result) whenever the corresponding
# test_system/reference_results/<code>/... is re-blessed. Never derive the
# light reference the other way around (never hand-edit it to match a
# fixture run).
#
# Usage:
#   bash test_system/derive_light_reference.sh [CODE] [REFERENCE_DIR]
#   CODE defaults to "seissol" (unchanged behavior/output from before this
#   script took a CODE parameter). Other supported CODE values add their own
#   fixture dir + converter + full-oracle raw subdir below; see the
#   FIXTURE_DIR/CONVERTER/RAW_FULL case statement.
#   REFERENCE_DIR defaults to <repo_root>/reference (a normal checkout, not
#   this worktree, since reference/ is gitignored and not copied into
#   worktrees).

set -u
set -o pipefail

cd "$(dirname "$0")/.."
REPO="$(pwd)"
UTILS="$REPO/src/utils"
TEST_DIR="$REPO/test_system"

CODE="${1:-seissol}"
REFERENCE_DIR="${2:-$REPO/reference}"

case "$CODE" in
    seissol)
        FIXTURE_DIR="seissol_sim1_fixture"
        CONVERTER="seissol_converter_api.py"
        RAW_FULL_SUB="seissol/sim1_big_0123"
        ;;
    eqdyna)
        FIXTURE_DIR="eqdyna_0001A_fixture"
        CONVERTER="eqdyna_converter_api.py"
        RAW_FULL_SUB="eqdyna/eqdyna.0001.A.100m"
        ;;
    *)
        echo "FAIL: unknown CODE '$CODE' (supported: seissol, eqdyna)"
        exit 2
        ;;
esac

FIXTURE="$TEST_DIR/fixture_reference/$FIXTURE_DIR"
RAW_FIXTURE="$FIXTURE/raw"
LIGHT_REF="$FIXTURE/light_reference"
RAW_FULL="$REFERENCE_DIR/datasets/$RAW_FULL_SUB"

if [ ! -d "$RAW_FULL" ]; then
    echo "FAIL: full reference raw dir not found at $RAW_FULL"
    echo "Pass the path to a checkout with reference/ as \$2, e.g.:"
    echo "  bash test_system/derive_light_reference.sh $CODE REDACTED_PATH/reference"
    exit 1
fi

WORK="$(mktemp -d -t dr4gm_derive_light_ref.XXXXXX)"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT
mkdir -p "$WORK/fixture_run" "$WORK/full_run"

echo "work dir: $WORK"

echo "=== Step 1: fixture-computed (raw fixture -> metrics -> stats) ==="
python3 "$UTILS/$CONVERTER" --input_dir "$RAW_FIXTURE" --output_dir "$WORK/fixture_run" || exit 1
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
python3 "$UTILS/$CONVERTER" --input_dir "$RAW_FULL" --output_dir "$WORK/full_run" || exit 1
python3 "$UTILS/station_subset_selector.py" \
    --input_npz "$WORK/full_run/velocities.npz" \
    --output_npz "$WORK/full_run/processed_stations.npz" \
    --grid_resolution 1000 || exit 1
python3 "$UTILS/npz_gm_processor.py" \
    --velocity_npz "$WORK/full_run/processed_stations.npz" \
    --output_dir "$WORK/full_run" || exit 1

echo "=== Step 3: slice full-oracle metrics to fixture's ACTUAL (post-subset) stations, sort, run gm_stats ==="
# Matches by PHYSICAL LOCATION (exact (x,y,z) equality), not by station_id.
# station_subset_selector's grid-cell "closest to grid center" selection is
# anchored to the bounding box of whichever candidate set it is given, so a
# raw fixture containing a different (smaller) candidate pool than the full
# dataset can legitimately pick a DIFFERENT representative station inside
# the same 1 km cell even though both point sets cover the same ground --
# station_id intersection is not a reliable key across different-sized
# candidate pools. Location equality is exact here because both runs derive
# `locations` from the same raw floats via the same fixed rotation, with no
# resampling in between. Output is ordered by the fixture run's OWN
# station_id ascending order (0..n-1 by construction below), matching
# run_fixture_e2e.sh's "sort fresh run by station_id" convention.
python3 - "$WORK/full_run/ground_motion_metrics.npz" "$WORK/fixture_run/ground_motion_metrics.npz" "$WORK/full_oracle_subset.npz" << 'PYEOF'
import sys
import numpy as np
full_path, fixture_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
full = np.load(full_path)
fixture = np.load(fixture_path)
fixture_ids = fixture["station_ids"]
fixture_locs = fixture["locations"]
n = len(fixture_ids)
id_order = np.argsort(fixture_ids)  # ascending fixture station_id order

full_locs = full["locations"]
# Map each full-run row's location to its row index (exact float key).
loc_to_idx = {tuple(row): i for i, row in enumerate(full_locs)}

match_idx = []
for k in id_order:
    key = tuple(fixture_locs[k])
    if key not in loc_to_idx:
        raise SystemExit(f"FAIL: fixture station_id={fixture_ids[k]} location {key} not found in full-oracle run")
    match_idx.append(loc_to_idx[key])
match_idx = np.array(match_idx)

out = {}
for k in full.files:
    arr = full[k]
    if hasattr(arr, "shape") and arr.ndim >= 1 and arr.shape[0] == full_locs.shape[0]:
        out[k] = arr[match_idx]
    else:
        out[k] = arr
# Relabel with the fixture's OWN station_ids (not the full-oracle's raw
# numbering scheme, e.g. per-chunk ids): run_fixture_e2e.sh always diffs a
# freshly-converted fixture run (ids assigned by the committed fixture's own
# chunk layout) against this frozen reference, so the ids must match that
# scheme, not the full dataset's. Physical identity was already verified by
# the location-equality match above.
out["station_ids"] = fixture_ids[id_order]
if out["station_ids"].shape[0] != n:
    raise SystemExit(
        f"FAIL: expected {n} stations in full-oracle slice, got {out['station_ids'].shape[0]}"
    )
if not np.array_equal(out["locations"], fixture_locs[id_order]):
    raise SystemExit("FAIL: full-oracle slice locations do not match fixture locations exactly")
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
