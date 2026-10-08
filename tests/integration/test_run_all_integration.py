"""Integration test for src/utils/run_all.sh — PATHWAY_FORWARD.md row 5.

This is the shell-script chaining point itself (raw fixture -> converter ->
station_subset_selector -> npz_gm_processor -> gm_stats -> visualize_*),
NOT the higher-level whole-workflow/figure-generation paths already covered
by tests/run_fixture_e2e.sh and tests/run_e2e_bundle.sh.

Lives under tests/integration/, not tests/unit/, because it reads
committed raw fixture data (an HDF5 file) and takes minutes to run (it also
exercises run_all.sh's visualization steps, which the lower-level
run_fixture_e2e.sh comparison does not invoke) -- row 4's unit tier is
documented "no raw data, <50ms". A thin pointer test in tests/unit/
re-exposes this same test function so the board's evidence command
(`pytest -q tests/unit -k run_all`) still finds and runs it.

Fixture size note (flagged, not silently resolved): PATHWAY_FORWARD.md row 5
states a <=2 MB fixture budget; the existing seissol_sim1_fixture this test
reuses (per row 13, which already proved its light_reference matches the
full reference/ oracle exactly, worst_rel=0.0) is ~4.3 MB total (raw + frozen
light_reference), consistent with row 13's own stated "4.2 MB HDF5, 5 MB cap"
-- a different, larger cap than row 5's. We reuse the existing, already-
validated fixture rather than building a second smaller one to chase row 5's
number; the discrepancy itself belongs back on the board.

Oracle: this test does NOT re-derive a reference. It reuses the exact same
frozen light_reference/*.npz that tests/run_fixture_e2e.sh already
diffs against (tests/fixture_reference/seissol_sim1_fixture/
light_reference/), which tests/derive_light_reference.sh proved once
(and documents how to re-derive) equals a slice of the full 199 GB reference/
oracle. Comparison uses the same tests/diff_gm_metrics.py tool and
tolerance convention (1e-6 rel for float32 input, 1e-12 otherwise) as
run_tests.sh / run_fixture_e2e.sh.

Runtime: ~3-4 min (run_all.sh's visualize_gm_maps.py/visualize_gm_stats.py
steps dominate -- many per-period PNGs). Acceptable for an integration/e2e
tier per the test-pyramid guidance ("seconds to minutes"); too slow for the
unit tier's <50ms budget, which is the other reason this lives in
integration/ rather than unit/.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
UTILS_DIR = REPO_ROOT / "src" / "utils"
TEST_DIR = REPO_ROOT / "tests"
RUN_ALL = UTILS_DIR / "run_all.sh"
FIXTURE = TEST_DIR / "fixture_reference" / "seissol_sim1_fixture"
RAW = FIXTURE / "raw"
LIGHT_REF = FIXTURE / "light_reference"
DIFF_TOOL = TEST_DIR / "diff_gm_metrics.py"


def _sort_by_station_id(src_path: Path, dst_path: Path) -> None:
    """Re-order a ground_motion_metrics-shaped npz by station_ids.

    run_all.sh's grid subsetting does not guarantee the same row order the
    frozen light_reference was stored in (sorted by station_id) -- same
    normalization step tests/run_fixture_e2e.sh applies before diffing.
    """
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


def _diff_passes(ref_npz: Path, new_npz: Path, input_npz: Path) -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, str(DIFF_TOOL), str(ref_npz), str(new_npz), "--input", str(input_npz)],
        capture_output=True, text=True,
    )
    return result.returncode == 0, result.stdout + result.stderr


def test_run_all_seissol_fixture_matches_light_reference(tmp_path):
    """run_all.sh raw->...->ground_motion_metrics.npz/gm_statistics.npz
    on the committed SeisSol fixture must reproduce the frozen light
    reference exactly (same oracle run_fixture_e2e.sh already validated
    against the full 199 GB reference/ tree, per row 13)."""
    pytest.importorskip("h5py", reason="h5py not installed in this environment")
    pytest.importorskip("matplotlib", reason="matplotlib not installed in this environment")

    assert (RAW / "fault_geometry.json").exists(), f"fixture raw dir not found at {RAW}"
    assert (LIGHT_REF / "ground_motion_metrics.npz").exists(), \
        f"frozen light reference not found under {LIGHT_REF}"

    out_dir = tmp_path / "run_all_out"

    result = subprocess.run(
        ["bash", str(RUN_ALL), str(RAW), "seissol", str(out_dir)],
        capture_output=True, text=True, cwd=str(UTILS_DIR),
    )
    assert result.returncode == 0, (
        f"run_all.sh exited {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )

    metrics_npz = out_dir / "ground_motion_metrics.npz"
    stats_npz = out_dir / "gm_statistics.npz"
    subset_npz = out_dir / "grid_1000m.npz"
    assert metrics_npz.exists(), "run_all.sh did not produce ground_motion_metrics.npz"
    assert stats_npz.exists(), "run_all.sh did not produce gm_statistics.npz"
    assert subset_npz.exists(), "run_all.sh did not produce the expected grid_1000m.npz subset"

    sorted_metrics = out_dir / "metrics_sorted.npz"
    _sort_by_station_id(metrics_npz, sorted_metrics)

    ok, diff_output = _diff_passes(LIGHT_REF / "ground_motion_metrics.npz", sorted_metrics, subset_npz)
    assert ok, f"ground_motion_metrics.npz diverged from frozen light reference:\n{diff_output}"

    ok, diff_output = _diff_passes(LIGHT_REF / "gm_statistics.npz", stats_npz, subset_npz)
    assert ok, f"gm_statistics.npz diverged from frozen light reference:\n{diff_output}"
