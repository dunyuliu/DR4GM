"""Single source of truth for the DR4GM Interactive Explorer AppTest baseline.

Computes expected display values directly from the vendored NPZ assets under
`data/` using plain numpy -- independent of Streamlit entirely. Used by:

  - tests/unit/test_streamlit_app.py    (asserts the live AppTest run
    against values freshly computed here, every CI run)
  - tests/capture_streamlit_baseline.py (regenerates the frozen JSON files
    in tests/e2e_reference/)

Keeping one function as the source of truth per baseline means the AppTest
assertions and the frozen baseline can never silently drift apart. A later
module-split refactor re-runs the same functions and diffs against the
frozen JSON to prove the displayed numbers are unchanged.

Two baselines are maintained:
  - compute_eqdyna_a_expected() / streamlit_apptest_baseline.json
    (original single-dataset baseline: default dataset, PGA + PGV only;
    kept as-is so the original 6 tests in test_streamlit_app.py do not
    need to change shape).
  - compute_dataset_expected()/compute_all_datasets_expected() /
    streamlit_apptest_multi_baseline.json (row 16 extension: all four
    vendored "DR4GM Data Archive"-mode datasets, covering every basic GM
    metric (PGA/PGV/PGD/CAV) and every spectral-acceleration period the
    app renders in the Ground Motion Metrics table for the
    default-selected station, plus the "Available Metrics" count).
"""

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
EQDYNA_A_NPZ = DATA_DIR / "eqdyna.0001.A.coarse.npz"
BASELINE_PATH = REPO_ROOT / "tests" / "e2e_reference" / "streamlit_apptest_baseline.json"
MULTI_BASELINE_PATH = REPO_ROOT / "tests" / "e2e_reference" / "streamlit_apptest_multi_baseline.json"

# Mirrors GroundMotionExplorer._LOCAL_ASSETS in
# web/dr4gm_interactive_explorer.py -- the "DR4GM Data Archive" source's
# local-first vendored demo files (data/MANIFEST.md). Kept as a literal
# dict here (not imported from the app) so this helper stays importable
# without pulling in streamlit/plotly at collection time.
LOCAL_ASSETS = {
    "EQDyna A Coarse Simulation": "eqdyna.0001.A.coarse.npz",
    "EQDyna B Coarse Simulation": "eqdyna.0001.B.coarse.npz",
    "FD3D A Coarse Simulation": "fd3d.0001.A.coarse.npz",
    "Waveqlab3D A Coarse Simulation": "waveqlab3d.0001.A.coarse.npz",
}

BASIC_METRICS = ["PGA", "PGV", "PGD", "CAV"]
BASIC_UNITS = {"PGA": "cm/s²", "PGV": "cm/s", "PGD": "cm", "CAV": "cm/s"}


def _default_station_idx(num_stations: int) -> int:
    """Mirrors main()'s default station pick in
    web/dr4gm_interactive_explorer.py:
        st.session_state.selected_station_idx = min(num_stations // 2, num_stations - 1)
    """
    return min(num_stations // 2, num_stations - 1)


def compute_eqdyna_a_expected() -> dict:
    """Independently compute the values the app should display for the
    default ("EQDyna A Coarse Simulation") dataset on first run, with no
    user interaction, by loading the NPZ directly."""
    if not EQDYNA_A_NPZ.is_file():
        raise FileNotFoundError(
            f"Expected vendored demo asset not found: {EQDYNA_A_NPZ}. "
            "It ships in data/ per PATHWAY_FORWARD.md commit 187ebd9; "
            "do not substitute a different dataset."
        )

    with np.load(EQDYNA_A_NPZ, allow_pickle=True) as data:
        station_ids = data["station_ids"]
        n_stations = int(len(station_ids))
        idx = _default_station_idx(n_stations)

        station_id = int(station_ids[idx])
        pga_raw = float(data["PGA"][idx])
        pgv_raw = float(data["PGV"][idx])
        location = data["locations"][idx]

        # App's coordinate-unit detection (filename contains "eqdyna" -> meters,
        # displayed converted to km).
        x_km = float(location[0]) / 1000.0
        y_km = float(location[1]) / 1000.0

    return {
        "dataset_label": "EQDyna A Coarse Simulation",
        "npz_path": str(EQDYNA_A_NPZ.relative_to(REPO_ROOT)),
        "station_count": n_stations,
        "selected_station_idx": idx,
        "selected_station_id": station_id,
        "pga_raw": pga_raw,
        "pga_formatted": f"{pga_raw:.3e}",
        "pgv_raw": pgv_raw,
        "pgv_formatted": f"{pgv_raw:.3e}",
        "location_x_km": round(x_km, 2),
        "location_y_km": round(y_km, 2),
    }


def compute_dataset_expected(dataset_label: str, *, carried_station_idx: int = None) -> dict:
    """Independently compute every value the app's "Ground Motion Metrics"
    table, "Total Stations" metric and "Available Metrics" metric should
    display for `dataset_label` (one of LOCAL_ASSETS) when loaded from the
    "DR4GM Data Archive" source (the default radio option, index 0), by
    loading the NPZ directly. Mirrors GroundMotionExplorer.
    _extract_data_from_npz and create_all_metrics_tables in
    web/dr4gm_interactive_explorer.py.

    `carried_station_idx`: the app only initializes
    `st.session_state.selected_station_idx` the first time
    (`if 'selected_station_idx' not in st.session_state`) -- switching the
    "Choose Dataset" selectbox to a different dataset in the SAME session
    does NOT recompute a fresh per-dataset default, it keeps whatever index
    was already selected (clamped to 0 if it is now out of range). Pass the
    previously-active index here to reproduce that; leave as None only for
    the dataset that is loaded first in a fresh session (today, always
    "EQDyna A Coarse Simulation", the selectbox's first/default option),
    which gets the one true fresh-session default
    `min(n_stations // 2, n_stations - 1)`.
    """
    if dataset_label not in LOCAL_ASSETS:
        raise KeyError(f"Unknown dataset label: {dataset_label!r}; expected one of {list(LOCAL_ASSETS)}")

    fname = LOCAL_ASSETS[dataset_label]
    npz_path = DATA_DIR / fname
    if not npz_path.is_file():
        raise FileNotFoundError(
            f"Expected vendored demo asset not found: {npz_path}. "
            "It ships in data/ per data/MANIFEST.md; do not substitute a different dataset."
        )

    with np.load(npz_path, allow_pickle=True) as data:
        station_ids = data["station_ids"]
        n_stations = int(len(station_ids))
        if carried_station_idx is None:
            idx = _default_station_idx(n_stations)
        else:
            idx = carried_station_idx if carried_station_idx < n_stations else 0

        station_id = int(station_ids[idx])
        location = data["locations"][idx]

        # App's coordinate-unit detection override by filename: "eqdyna" ->
        # meters (displayed converted to km); "fd3d"/"waveqlab3d" -> km as-is.
        coord_unit = "m" if "eqdyna" in fname.lower() else "km"
        if coord_unit == "m":
            x_km = float(location[0]) / 1000.0
            y_km = float(location[1]) / 1000.0
        else:
            x_km = float(location[0])
            y_km = float(location[1])

        basic_metrics = {}
        for metric in BASIC_METRICS:
            if metric in data:
                raw = float(data[metric][idx])
                basic_metrics[metric] = {
                    "raw": raw,
                    "formatted": f"{raw:.3e}",
                    "unit": BASIC_UNITS[metric],
                }

        sa_keys = sorted(key for key in data.keys() if key.startswith("RSA_T_"))
        sa_metrics = {}
        for key in sa_keys:
            period = key.replace("RSA_T_", "").replace("_", ".")
            raw = float(data[key][idx])
            sa_metrics[period] = {"raw": raw, "formatted": f"{raw:.3e}"}

        available_metrics_count = len(basic_metrics) + len(sa_keys)

    return {
        "dataset_label": dataset_label,
        "npz_path": str(npz_path.relative_to(REPO_ROOT)),
        "station_count": n_stations,
        "selected_station_idx": idx,
        "selected_station_id": station_id,
        "location_x_km": round(x_km, 2),
        "location_y_km": round(y_km, 2),
        "available_metrics_count": available_metrics_count,
        "basic_metrics": basic_metrics,
        "sa_metrics": sa_metrics,
    }


def compute_all_datasets_expected() -> dict:
    """compute_dataset_expected() for every vendored "DR4GM Data Archive"
    dataset, keyed by dataset label -- reproducing the real single-session
    AppTest interaction sequence (load default EQDyna A, then switch the
    "Choose Dataset" selectbox to each of the others in turn) so the
    selected-station index matches what the app actually displays, including
    its session-state carryover behavior documented on
    compute_dataset_expected."""
    default_label = "EQDyna A Coarse Simulation"
    default_expected = compute_dataset_expected(default_label)
    carried_idx = default_expected["selected_station_idx"]

    all_expected = {default_label: default_expected}
    for label in LOCAL_ASSETS:
        if label == default_label:
            continue
        all_expected[label] = compute_dataset_expected(label, carried_station_idx=carried_idx)
    return all_expected
