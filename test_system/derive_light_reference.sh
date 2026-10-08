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
#
#   fd3d note (two raw-format issues, discovered while building this
#   fixture):
#
#   (1) block-size quirk: the real SD4/Ncenter seisout{U,V}.surface.gnuplot.dat
#   files are NOT flat 900-line-per-timestep blocks as fd3d_converter_api.py's
#   default nxt=900 assumes -- each timestep is actually a TRUE 902-line
#   block: 900 real fault-parallel stations (i=1..900) followed by exactly
#   2 BLANK lines (i=901,902; verified: every one of the 1600 blocks has
#   blanks at block-relative lines 901 and 902, 3200 blanks total,
#   1600*902=1,443,200 = the file's real line count). numpy.loadtxt
#   silently skips blank lines; for the FULL file this coincidentally
#   self-heals (3200 blanks removed from 1,443,200 lines leaves exactly
#   1,440,000 = 1600*900, matching fd3d_converter_api.py's
#   expected_total_lines with no mismatch-pad branch taken, so the full
#   run reshapes correctly) -- but a NAIVE partial crop that doesn't know
#   about the true 902-line block boundary can clip into the blank pair
#   and trigger the mismatch/pad branch, which pads with zero rows at the
#   END rather than at the point of loss, silently misaligning every
#   station after it. The fixture crop reads the source using the TRUE
#   902-line block size to avoid this.
#
#   (2) sparse-candidate-pool / grid-cell-boundary issue, worse than
#   eqdyna's: fd3d's (x,y) coordinates are a RIGID function of raw line
#   position (sta_north=dh*i has no decoupling parameter, unlike
#   sta_east=-dh*(nyt-j+1) which can be re-centered via nyt), so unlike
#   eqdyna's per-station chunk files, a fd3d fixture cannot pick arbitrary
#   station identities -- a cropped fixture's station_subset_selector pick
#   for a given grid cell matches the full dataset's pick ONLY when the
#   fixture happens to contain the EXACT cell-center coordinate itself
#   (distance 0 to the cell center, unbeatable by any candidate outside
#   the crop, so the match holds regardless of what else surrounds it).
#   Grid centers occur every grid_resolution=1000m starting at the
#   dataset's true (x_min, y_min) corner (100, 100) -- since 1000m is a
#   multiple of the native 100m spacing, every center coordinate is itself
#   a native grid point, so a fixture built to include exactly those
#   points is guaranteed to match. The fixture here uses a single
#   fault-normal column at x=100m (the domain's true edge) times 91
#   fault-parallel points y=100..9100m (--nxt 91), which lands exactly on
#   10 grid centers (y=100,1100,...,9100) under station_subset_selector's
#   grid_resolution=1000 -- verified directly: `station_subset_selector.py
#   --grid_resolution 1000` on the fixture's own velocities.npz selects
#   exactly those 10 stations, nothing else. A SECOND dummy all-zero
#   column is included (--nyt 2, not --nyt 1) purely to keep
#   fd3d_converter_api.py's vectorized (2D) ASCII reader on its happy
#   path: a true single-column (--nyt 1) file makes np.loadtxt return a 1D
#   array, which raises on `.shape[1]` and silently falls back to the
#   loop-based reader -- that reader has no explicit dtype and produces
#   float64 instead of the production float32, which then fools
#   diff_gm_metrics.py's float32-vs-float64 tolerance auto-detection (it
#   inspects the --input processed_stations.npz's vel_strike dtype) into
#   using the strict 1e-12 tolerance instead of the correct 1e-6 float32
#   tolerance, even though the underlying noise is genuine float32-grade
#   (~1e-7). The dummy column (j=1 -> x=200m) is always beaten by the
#   exact j=2 -> x=100m match for every target cell (distance 0 in x
#   beats distance 100), so it never gets selected and is otherwise inert.
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

FIXTURE_CONVERTER_ARGS=()
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
    fd3d)
        FIXTURE_DIR="fd3d_ncent_sd4_fixture"
        CONVERTER="fd3d_converter_api.py"
        RAW_FULL_SUB="fd3d/SD4/Ncenter"
        # Fixture raw is a single fault-normal column (x=100m, the domain's
        # true edge) x 91 fault-parallel points (y=100..9100m) — see fd3d
        # note above for why only EXACT-grid-center stations are usable at
        # all for fd3d (unlike eqdyna/seissol's per-station files, fd3d's
        # y-coordinate is a rigid function of raw line position with no
        # decoupling parameter, so "closest-to-cell-center" is only
        # guaranteed to match between a cropped fixture and the full
        # dataset at points where the fixture happens to contain the EXACT
        # cell-center coordinate itself, distance 0, unbeatable by anything
        # outside the crop). This single column at x=100 hits 10 exact grid
        # centers (y=100,1100,...,9100) when run through
        # station_subset_selector's grid_resolution=1000 selection. Only
        # the FIXTURE conversion needs non-default --nxt/--nyt; the
        # full-oracle run below uses the converter's defaults
        # (900/250/1600), matching what the production pipeline
        # (run_all.sh) actually passes.
        FIXTURE_CONVERTER_ARGS=(--nxt 91 --nyt 2)
        ;;
    *)
        echo "FAIL: unknown CODE '$CODE' (supported: seissol, eqdyna, fd3d)"
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
    echo "  bash test_system/derive_light_reference.sh $CODE /home/utig5/dliu/dr4gm/dr4gm/reference"
    exit 1
fi

WORK="$(mktemp -d -t dr4gm_derive_light_ref.XXXXXX)"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT
mkdir -p "$WORK/fixture_run" "$WORK/full_run"

echo "work dir: $WORK"

echo "=== Step 1: fixture-computed (raw fixture -> metrics -> stats) ==="
python3 "$UTILS/$CONVERTER" --input_dir "$RAW_FIXTURE" --output_dir "$WORK/fixture_run" "${FIXTURE_CONVERTER_ARGS[@]}" || exit 1
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
