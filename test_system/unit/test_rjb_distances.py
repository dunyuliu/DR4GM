"""Unit tests for src/utils/gm_stats.py:rjb_distances_m.

Every expected distance below is computed independently by hand (simple
Pythagorean / point-to-segment geometry), not by calling the function under
test with different inputs. Two cases use the exact offset fault geometries
documented in CLAUDE.md's "Fault geometry" table: FD3D_TSN (fault y in
25.1-65.1 km) and WaveQLab3D (fault x=20 km, y in 20-60 km).
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "utils"))

from gm_stats import rjb_distances_m  # noqa: E402


def test_station_beyond_segment_end_uses_endpoint_distance():
    """Station off the end of a horizontal fault: Rjb = straight-line distance
    to the nearest endpoint (Pythagoras: 3-4-5 triangle), not perpendicular
    distance to the infinite line."""
    locations = np.array([[13.0, 4.0]])  # 3 m east of fault end, 4 m north
    fault_start = [0.0, 0.0]
    fault_end = [10.0, 0.0]
    dist = rjb_distances_m(locations, fault_start, fault_end)
    np.testing.assert_allclose(dist, [5.0])


def test_station_abreast_of_segment_uses_perpendicular_distance():
    """Station whose projection lands inside the segment: Rjb is the
    perpendicular distance to the trace."""
    locations = np.array([[5.0, 7.0]])  # directly above the midpoint
    fault_start = [0.0, 0.0]
    fault_end = [10.0, 0.0]
    dist = rjb_distances_m(locations, fault_start, fault_end)
    np.testing.assert_allclose(dist, [7.0])


def test_station_on_trace_is_zero():
    locations = np.array([[5.0, 0.0]])
    dist = rjb_distances_m(locations, [0.0, 0.0], [10.0, 0.0])
    np.testing.assert_allclose(dist, [0.0], atol=1e-12)


def test_zero_length_segment_falls_back_to_point_distance():
    """fault_start == fault_end: distance is plain Euclidean to that point."""
    locations = np.array([[3.0, 4.0]])
    dist = rjb_distances_m(locations, [0.0, 0.0], [0.0, 0.0])
    np.testing.assert_allclose(dist, [5.0])


def test_fd3d_tsn_offset_fault_geometry():
    """FD3D_TSN fault trace (CLAUDE.md): x=0, y from 25.1 to 65.1 km, i.e. a
    vertical (N-S) segment offset 25.1-65.1 km from the origin in y.
    A station at (3000 m, 45100 m) [3 km east, abreast of the trace] is
    perpendicular-distance 3000 m from the fault -- computed by hand."""
    fault_start_m = [0.0, 25100.0]
    fault_end_m = [0.0, 65100.0]
    locations_m = np.array([
        [3000.0, 45100.0],   # abreast -> perpendicular distance 3000 m
        [0.0, 65100.0 + 500.0],  # 500 m past the north end -> endpoint distance
    ])
    dist = rjb_distances_m(locations_m, fault_start_m, fault_end_m)
    np.testing.assert_allclose(dist, [3000.0, 500.0])


def test_waveqlab3d_offset_fault_geometry():
    """WaveQLab3D fault trace (CLAUDE.md): x=20 km, y from 20 to 60 km.
    A station at (24000 m, 40000 m) is perpendicular-distance 4000 m from the
    trace (abreast); a station south of the segment at (20000 m, 19000 m) is
    1000 m from the nearest endpoint -- both computed by hand."""
    fault_start_m = [20000.0, 20000.0]
    fault_end_m = [20000.0, 60000.0]
    locations_m = np.array([
        [24000.0, 40000.0],
        [20000.0, 19000.0],
    ])
    dist = rjb_distances_m(locations_m, fault_start_m, fault_end_m)
    np.testing.assert_allclose(dist, [4000.0, 1000.0])


def test_vectorized_over_multiple_stations_matches_per_station_hand_calc():
    """Non-axis-aligned fault (45 degrees) to confirm the projection logic
    handles arbitrary orientation, not just the axis-aligned special cases
    above. Expected distances computed via independent point-to-segment
    geometry (perpendicular distance = |cross product| / segment length)."""
    fault_start = [0.0, 0.0]
    fault_end = [10.0, 10.0]  # 45-degree trace, length sqrt(200)
    # Station at (10, 0): perpendicular distance to the line y=x is
    # |10 - 0| / sqrt(2) = 10/sqrt(2); its projection (t=0.5) lies inside
    # the segment, so this is the perpendicular distance.
    locations = np.array([[10.0, 0.0]])
    dist = rjb_distances_m(locations, fault_start, fault_end)
    np.testing.assert_allclose(dist, [10.0 / np.sqrt(2.0)])


def test_extra_location_columns_beyond_xy_are_ignored():
    """Docstring contract: 'only [:, :2] used' -- a 3rd (z) column must not
    change the result versus the pure-2D case computed by hand."""
    locations_3d = np.array([[13.0, 4.0, 999.0]])
    locations_2d = np.array([[13.0, 4.0]])
    dist_3d = rjb_distances_m(locations_3d, [0.0, 0.0], [10.0, 0.0])
    dist_2d = rjb_distances_m(locations_2d, [0.0, 0.0], [10.0, 0.0])
    np.testing.assert_allclose(dist_3d, dist_2d)
    np.testing.assert_allclose(dist_3d, [5.0])


def test_mutation_sensitivity_endpoint_clip(tmp_path, monkeypatch):
    """Mutation probe: if the t-clipping to [0, 1] is removed (so the
    projection is allowed to run past the segment ends), the endpoint-distance
    case above must give a different, smaller number. Verifies the test
    actually exercises the clipping behaviour rather than passing for free."""
    src = (REPO_ROOT / "src" / "utils" / "gm_stats.py").read_text()
    needle = "t = np.clip(rel @ seg / seg_len_sq, 0.0, 1.0)"
    assert needle in src, "gm_stats.py source changed shape; update this mutation probe"
    mutated_src = src.replace(needle, "t = rel @ seg / seg_len_sq", 1)

    mutated_file = tmp_path / "gm_stats_mutated.py"
    mutated_file.write_text(mutated_src)

    import importlib.util
    spec = importlib.util.spec_from_file_location("gm_stats_mutated", mutated_file)
    mutated_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mutated_module)

    locations = np.array([[13.0, 4.0]])
    correct = rjb_distances_m(locations, [0.0, 0.0], [10.0, 0.0])
    mutated = mutated_module.rjb_distances_m(locations, [0.0, 0.0], [10.0, 0.0])
    assert not np.allclose(mutated, correct), (
        "Removing the t-clip did not change the endpoint-distance result -- "
        "this test would not catch a broken clip"
    )
    np.testing.assert_allclose(correct, [5.0])
