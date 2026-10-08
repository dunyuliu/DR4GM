"""Single source of truth for the DR4GM Interactive Explorer AppTest baseline.

Computes expected display values directly from the vendored NPZ asset
(`data/eqdyna.0001.A.coarse.npz`) using plain numpy -- independent of
Streamlit entirely. Used by:

  - tests/unit/test_streamlit_app.py    (asserts the live AppTest run
    against values freshly computed here, every CI run)
  - tests/capture_streamlit_baseline.py (regenerates the frozen JSON
    in tests/e2e_reference/streamlit_apptest_baseline.json)

Keeping one function as the source of truth means the AppTest assertions
and the frozen baseline can never silently drift apart: both call
`compute_eqdyna_a_expected()`. A later module-split refactor re-runs the
same function and diffs against the frozen JSON to prove the displayed
numbers are unchanged.
"""

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EQDYNA_A_NPZ = REPO_ROOT / "data" / "eqdyna.0001.A.coarse.npz"
BASELINE_PATH = REPO_ROOT / "tests" / "e2e_reference" / "streamlit_apptest_baseline.json"


def _default_station_idx(num_stations: int) -> int:
    """Mirrors main()'s default station pick in
    src/web/dr4gm_interactive_explorer.py:
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
