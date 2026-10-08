"""One test per simulation-code converter (src/utils/*_converter_api.py),
each run against a tiny (<=200 kB) fixture this file constructs in a
pytest tmp_path -- never against the 109 GB frozen reference/ data.

Fixture provenance (all synthetic, built here; none copied from real data):

  eqdyna      synthetic ASCII station coords + float64 binary velocity file,
              matching EQDynaConverter's documented surface_coor.txt / gm
              binary schema (2 stations, 4 time steps).
  fd3d        synthetic ASCII seisoutU/V.surface.gnuplot.dat grid files,
              matching FD3DConverter's nxt x nyt x np_time schema, shrunk via
              the converter's own constructor keyword args (nxt=2, nyt=2,
              np_time=3) -- not a production-code change.
  seissol     synthetic HDF5 file with time/xyz/v1/v2/v3 datasets, matching
              SeisSolConverter's documented .h5 schema (3 stations, 5 steps).
  mafe        synthetic MATLAB .mat file (scipy.io.savemat) with vx/vy/vz/Mw,
              matching MafeConverter's documented grid-cube schema
              (2x2x6 grid).
  specfem3d   synthetic tab-delimited CSV with the exact required_cols list
              from SPECFEM3DConverter.load_csv_data (4 synthetic stations).
  waveqlab3d  synthetic float32 binary Hslice files. WaveQLab3dConverter
              hard-codes its station grid (lsl=401, rsl=401, stl=1601) in
              __init__ rather than taking it as a constructor argument, so a
              schema-faithful fixture at full grid size would be multiple MB
              per file -- over the 200 kB budget. This test instead
              overrides the instance attributes self.lsl/self.rsl/self.stl
              to a tiny grid (2x2x2) *after* construction (a normal Python
              attribute set, not a src/ edit) and writes binary data sized to
              match -- the read/reshape logic under test is exercised
              identically, just at a size that fits the budget.

Every fault_geometry.json fixture carries the full superset of fields that
any one converter's create_fault_geometry_npz might require (eqdyna/fd3d/
seissol need a subset; mafe/specfem3d/waveqlab3d also read fault_width,
top_depth, bottom_depth) so the same helper can be reused everywhere.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
UTILS_DIR = REPO_ROOT / "src" / "utils"
sys.path.insert(0, str(UTILS_DIR))

from npz_format_standard import DR4GM_NPZ_Standard  # noqa: E402


def _fault_geometry_json(path: Path) -> None:
    payload = {
        "fault_type": "strike-slip",
        "fault_trace_start": [0.0, -1000.0, 0.0],
        "fault_trace_end": [0.0, 1000.0, 0.0],
        "fault_dip": 90.0,
        "fault_strike": 0.0,
        "fault_length": 2000.0,
        "fault_width": 15000.0,
        "top_depth": 0.0,
        "bottom_depth": 15000.0,
        "coordinate_units": "m",
        "description": "Synthetic test fixture fault, not from real data",
        "rotation": "no",
    }
    path.write_text(json.dumps(payload))


def _assert_dir_under_budget(directory: Path, max_bytes: int = 200_000) -> None:
    total = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
    assert total <= max_bytes, f"fixture directory {directory} is {total} bytes (> {max_bytes})"


# ---------------------------------------------------------------------------
# eqdyna
# ---------------------------------------------------------------------------

def test_eqdyna_converter_on_tiny_synthetic_fixture(tmp_path):
    from eqdyna_converter_api import EQDynaConverter

    input_dir = tmp_path / "eqdyna_in"
    output_dir = tmp_path / "eqdyna_out"
    input_dir.mkdir()

    n_stations, n_steps, dt = 2, 4, 0.1
    coords = np.array([[100.0, 50.0, 0.0], [200.0, -30.0, 0.0]])
    np.savetxt(input_dir / "surface_coor.txt", coords)

    # Data layout: [t0_s0_c0, t0_s0_c1, t0_s0_c2, t0_s1_c0, ...], float64.
    rng = np.random.default_rng(0)
    gm_data = rng.normal(size=(n_steps, n_stations, 3)).astype(np.float64)
    gm_data.tofile(input_dir / "gm")

    _fault_geometry_json(input_dir / "fault_geometry.json")
    _assert_dir_under_budget(input_dir)

    converter = EQDynaConverter(str(input_dir), str(output_dir), dt=dt)
    results = converter.convert_dataset()

    assert results["total_stations"] == n_stations
    assert results["chunks_processed"] == 1

    stations_npz = np.load(output_dir / "stations.npz")
    assert len(stations_npz["station_ids"]) == n_stations
    assert stations_npz["locations"].shape == (n_stations, 3)
    assert DR4GM_NPZ_Standard.validate_npz(str(output_dir / "stations.npz"), "layer2_stations")

    velocities_npz = np.load(output_dir / "velocities.npz")
    assert velocities_npz["vel_strike"].shape == (n_stations, n_steps)
    assert velocities_npz["units"].item() == "m/s"
    assert DR4GM_NPZ_Standard.validate_npz(str(output_dir / "velocities.npz"), "layer3_velocities")

    # Known rotation applied by the converter: (x, y) -> (-y, x) -- computed
    # by hand from the fixture's original (x, y).
    np.testing.assert_allclose(stations_npz["locations"][:, 0], -coords[:, 1])
    np.testing.assert_allclose(stations_npz["locations"][:, 1], coords[:, 0])


# ---------------------------------------------------------------------------
# fd3d
# ---------------------------------------------------------------------------

def test_fd3d_converter_on_tiny_synthetic_fixture(tmp_path):
    from fd3d_converter_api import FD3DConverter

    input_dir = tmp_path / "fd3d_in"
    output_dir = tmp_path / "fd3d_out"
    input_dir.mkdir()

    nxt, nyt, np_time, dh, dt = 2, 2, 3, 0.1, 0.025
    rng = np.random.default_rng(1)

    u_data = rng.normal(size=(np_time * nxt, nyt)).astype(np.float32)
    v_data = rng.normal(size=(np_time * nxt, nyt)).astype(np.float32)
    np.savetxt(input_dir / "seisoutU.surface.gnuplot.dat", u_data)
    np.savetxt(input_dir / "seisoutV.surface.gnuplot.dat", v_data)

    _fault_geometry_json(input_dir / "fault_geometry.json")
    _assert_dir_under_budget(input_dir)

    converter = FD3DConverter(str(input_dir), str(output_dir),
                              nxt=nxt, nyt=nyt, dh=dh, np_time=np_time, dt=dt)
    results = converter.convert_all_datasets()

    total_stations = nxt * nyt
    assert results["total_stations"] == total_stations

    stations_npz = np.load(output_dir / "stations.npz")
    assert len(stations_npz["station_ids"]) == total_stations
    assert DR4GM_NPZ_Standard.validate_npz(str(output_dir / "stations.npz"), "layer2_stations")

    velocities_npz = np.load(output_dir / "velocities.npz")
    assert velocities_npz["vel_strike"].shape == (total_stations, np_time)
    assert velocities_npz["vel_normal"].shape == (total_stations, np_time)
    assert DR4GM_NPZ_Standard.validate_npz(str(output_dir / "velocities.npz"), "layer3_velocities")


# ---------------------------------------------------------------------------
# seissol
# ---------------------------------------------------------------------------

def test_seissol_converter_on_tiny_synthetic_fixture(tmp_path):
    h5py = pytest.importorskip("h5py", reason="h5py not installed in this environment")
    from seissol_converter_api import SeisSolConverter

    input_dir = tmp_path / "seissol_in"
    output_dir = tmp_path / "seissol_out"
    input_dir.mkdir()

    n_stations, n_steps, dt = 3, 5, 0.02
    time_array = np.arange(n_steps) * dt
    locations = np.column_stack([
        np.array([100.0, 200.0, 300.0]),
        np.zeros(n_stations),
        np.zeros(n_stations),
    ])
    rng = np.random.default_rng(2)
    v1 = rng.normal(size=(n_stations, n_steps))
    v2 = rng.normal(size=(n_stations, n_steps))
    v3 = rng.normal(size=(n_stations, n_steps))

    with h5py.File(input_dir / "synthetic.h5", "w") as f:
        f.create_dataset("time", data=time_array)
        f.create_dataset("xyz", data=locations)
        f.create_dataset("v1", data=v1)
        f.create_dataset("v2", data=v2)
        f.create_dataset("v3", data=v3)

    _fault_geometry_json(input_dir / "fault_geometry.json")
    _assert_dir_under_budget(input_dir)

    converter = SeisSolConverter(str(input_dir), str(output_dir))
    results = converter.convert_all_datasets()

    assert results["total_stations"] == n_stations
    assert results["files_processed"] == 1

    velocities_npz = np.load(output_dir / "velocities.npz")
    assert velocities_npz["vel_strike"].shape == (n_stations, n_steps)
    np.testing.assert_allclose(velocities_npz["dt_values"][0], dt)
    assert DR4GM_NPZ_Standard.validate_npz(str(output_dir / "velocities.npz"), "layer3_velocities")


# ---------------------------------------------------------------------------
# mafe
# ---------------------------------------------------------------------------

def test_mafe_converter_on_tiny_synthetic_fixture(tmp_path):
    scipy_io = pytest.importorskip("scipy.io", reason="scipy not installed in this environment")
    from mafe_converter_api import MafeConverter

    input_dir = tmp_path / "mafe_in"
    output_dir = tmp_path / "mafe_out"
    input_dir.mkdir()

    nx, ny, nt = 2, 2, 6
    rng = np.random.default_rng(3)
    mat_payload = {
        "vx": rng.normal(size=(nx, ny, nt)).astype(np.float32),
        "vy": rng.normal(size=(nx, ny, nt)).astype(np.float32),
        "vz": rng.normal(size=(nx, ny, nt)).astype(np.float32),
        "Mw": np.array([[7.0]]),
    }
    scipy_io.savemat(input_dir / "realization_001.mat", mat_payload)

    _fault_geometry_json(input_dir / "fault_geometry.json")
    _assert_dir_under_budget(input_dir)

    converter = MafeConverter(input_dir, output_dir, dt=0.007)
    results = converter.convert()

    stations_npz = np.load(output_dir / "stations.npz")
    assert len(stations_npz["station_ids"]) == nx * ny
    assert DR4GM_NPZ_Standard.validate_npz(str(output_dir / "stations.npz"), "layer2_stations")

    velocities_npz = np.load(output_dir / "velocities.npz")
    assert velocities_npz["vel_strike"].shape == (nx * ny, nt)
    assert DR4GM_NPZ_Standard.validate_npz(str(output_dir / "velocities.npz"), "layer3_velocities")

    geometry_npz = np.load(output_dir / "geometry.npz")
    np.testing.assert_allclose(float(geometry_npz["moment_magnitude"]), 7.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# specfem3d
# ---------------------------------------------------------------------------

def test_specfem3d_converter_on_tiny_synthetic_fixture(tmp_path):
    pd = pytest.importorskip("pandas", reason="pandas not installed in this environment")
    from specfem3d_converter_api import SPECFEM3DConverter

    input_dir = tmp_path / "specfem3d_in"
    output_dir = tmp_path / "specfem3d_out"
    input_dir.mkdir()

    df = pd.DataFrame({
        "x": [0.0, 0.0, 0.0, 0.0],
        "y": [1000.0, 5000.0, 10000.0, 20000.0],
        "z": [0.0, 0.0, 0.0, 0.0],
        "R_rup": [1.0, 5.0, 10.0, 20.0],  # km
        "gm50_T0p5": [0.5, 0.3, 0.15, 0.05],
        "gm50_T1": [0.3, 0.2, 0.1, 0.03],
        "gm50_T3": [0.1, 0.07, 0.04, 0.01],
        "CAV_geometric_mean": [2.0, 1.5, 1.0, 0.5],
        "Mw": [7.0, 7.0, 7.0, 7.0],
    })
    csv_path = input_dir / "synthetic_GMROT_with_CAV.csv"
    df.to_csv(csv_path, sep="\t", index=False)

    _fault_geometry_json(input_dir / "fault_geometry.json")
    _assert_dir_under_budget(input_dir)

    converter = SPECFEM3DConverter(csv_path, output_dir,
                                   distance_range=(0, 30000), distance_bin_size=5000)
    results = converter.convert()

    stats_npz = np.load(output_dir / "gm_statistics.npz")
    assert "PGA_mean" in stats_npz
    assert int(stats_npz["total_stations"]) == 4
    np.testing.assert_allclose(float(stats_npz["moment_magnitude"]), 7.0)


# ---------------------------------------------------------------------------
# waveqlab3d
# ---------------------------------------------------------------------------

def test_waveqlab3d_converter_on_tiny_synthetic_fixture(tmp_path):
    from waveqlab3d_converter_api import Waveqlab3dConverter

    input_dir = tmp_path / "waveqlab3d_in"
    output_dir = tmp_path / "waveqlab3d_out"
    input_dir.mkdir()

    converter = Waveqlab3dConverter(str(input_dir), str(output_dir), dt=0.05)
    # Hard-coded grid (lsl=401, rsl=401, stl=1601) would need multi-MB fixture
    # files; override the instance attributes (plain attribute set, not a
    # src/ edit) to a tiny grid so a <=200 kB fixture can exercise the same
    # read/reshape code path.
    # stl/lsl/rsl=3 (not 2) so each axis has an interior point that survives
    # the converter's 2 km domain-boundary trim -- with only endpoints, every
    # station would sit on the boundary and get dropped, leaving an empty array.
    converter.lsl = 3
    converter.rsl = 3
    converter.stl = 3
    n_time = 3

    left_total = converter.lsl * converter.stl   # 9
    right_total = converter.rsl * converter.stl  # 9
    rng = np.random.default_rng(4)

    base = "synthetic"
    # Layout: [st0_t0, st1_t0, ..., stN_t0, st0_t1, ...], float32.
    (rng.normal(size=n_time * left_total).astype(np.float32)
        .tofile(input_dir / f"{base}.Hslice1seisx"))
    (rng.normal(size=n_time * left_total).astype(np.float32)
        .tofile(input_dir / f"{base}.Hslice1seisy"))
    (rng.normal(size=n_time * right_total).astype(np.float32)
        .tofile(input_dir / f"{base}.Hslice2seisx"))
    (rng.normal(size=n_time * right_total).astype(np.float32)
        .tofile(input_dir / f"{base}.Hslice2seisy"))

    _fault_geometry_json(input_dir / "fault_geometry.json")
    _assert_dir_under_budget(input_dir)

    file_sets = converter.discover_hslice_files()
    assert base in file_sets
    conversion_data = converter.convert_dataset(base, file_sets[base])

    expected_total = left_total + right_total
    assert 0 < conversion_data["vel_strike"].shape[0] <= expected_total  # boundary trim drops edge stations
    assert conversion_data["time_steps"] == n_time

    station_file = converter.create_station_list_npz(conversion_data)
    velocity_file = converter.create_velocity_database_npz(conversion_data)

    stations_npz = np.load(station_file)
    velocities_npz = np.load(velocity_file)
    assert len(stations_npz["station_ids"]) == conversion_data["vel_strike"].shape[0]
    assert velocities_npz["vel_strike"].shape[1] == n_time
    assert DR4GM_NPZ_Standard.validate_npz(station_file, "layer2_stations")
    assert DR4GM_NPZ_Standard.validate_npz(velocity_file, "layer3_velocities")

