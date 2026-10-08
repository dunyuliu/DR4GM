"""Headless Streamlit AppTest coverage for src/web/dr4gm_interactive_explorer.py.

Runs fully offline:
  - st.secrets is replaced with a dummy dict via AppTest's own `.secrets`
    attribute (never the real, gitignored src/web/.streamlit/secrets.toml,
    which does not exist in this checkout anyway).
  - The default dataset ("EQDyna A Coarse Simulation") resolves to the
    vendored local copy in data/eqdyna.0001.A.coarse.npz (commit 187ebd9),
    so no network access occurs on `at.run()`.

Displayed-value assertions are checked against values computed independently
from the NPZ by test_system/streamlit_baseline_helper.py (plain numpy, no
Streamlit), and cross-checked against the frozen baseline JSON in
test_system/e2e_reference/ -- see that module's docstring for why both.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = REPO_ROOT / "src" / "web" / "dr4gm_interactive_explorer.py"

sys.path.insert(0, str(REPO_ROOT / "test_system"))
from streamlit_baseline_helper import BASELINE_PATH, compute_eqdyna_a_expected  # noqa: E402

pytest.importorskip("streamlit", reason="streamlit not installed in this environment")
from streamlit.testing.v1 import AppTest  # noqa: E402

# Dummy secrets only -- all empty so every outbound-network branch in
# UsageTracker (Supabase / Google Sheets / email webhook) short-circuits via
# its own `if not <secret>: return` guard. Never read real secrets.toml.
DUMMY_SECRETS = {
    "supabase_url": "",
    "supabase_key": "",
    "google_sheets_webhook": "",
    "email_webhook": "",
    "notification_email": "",
}


def _fresh_app_test() -> AppTest:
    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.secrets = dict(DUMMY_SECRETS)
    return at


def test_app_boots_with_no_exception():
    at = _fresh_app_test()
    at.run()
    assert not at.exception, f"App raised on boot: {[str(e) for e in at.exception]}"


def test_data_source_and_dataset_widgets_present_and_settable():
    at = _fresh_app_test()
    at.run()
    assert not at.exception

    data_source_radio = at.sidebar.radio[0]
    assert data_source_radio.label == "Data Source"
    assert "DR4GM Data Archive" in data_source_radio.options
    assert "Local Files" in data_source_radio.options

    dataset_selectbox = at.sidebar.selectbox[0]
    assert dataset_selectbox.label == "Choose Dataset"
    assert dataset_selectbox.value == "EQDyna A Coarse Simulation"

    # Settability: changing the radio must not raise and must take effect.
    data_source_radio.set_value("Local Files")
    at.run()
    assert not at.exception, f"App raised after switching Data Source: {[str(e) for e in at.exception]}"
    assert at.sidebar.radio[0].value == "Local Files"


def test_displayed_station_count_matches_npz():
    """'Total Stations' st.metric must equal the station count independently
    computed from data/eqdyna.0001.A.coarse.npz -- not a hardcoded number."""
    expected = compute_eqdyna_a_expected()

    at = _fresh_app_test()
    at.run()
    assert not at.exception

    total_stations_metric = next(m for m in at.metric if m.label == "Total Stations")
    # st.metric values round-trip as strings through the AppTest proto layer.
    assert int(total_stations_metric.value) == expected["station_count"]


def test_displayed_pga_value_matches_npz():
    """The per-station Ground Motion Metrics table's PGA entry (for the
    default-selected station) must equal the NPZ value, formatted the same
    way the app formats it (`.3e`), computed independently in the test."""
    expected = compute_eqdyna_a_expected()

    at = _fresh_app_test()
    at.run()
    assert not at.exception

    station_id_blocks = [m.value for m in at.markdown if "Station ID:" in m.value]
    assert len(station_id_blocks) == 1
    assert f"<strong>Station ID:</strong> {expected['selected_station_id']}" in station_id_blocks[0]

    metrics_table_blocks = [m.value for m in at.markdown if "<table" in m.value]
    assert len(metrics_table_blocks) == 1
    table_html = metrics_table_blocks[0]
    assert f">PGA</td><td style='padding: 2px 4px; border: 1px solid #ddd; text-align: right; font-family: monospace;'>{expected['pga_formatted']}</td>" in table_html


def test_baseline_json_matches_fresh_computation():
    """Guards against the frozen baseline silently drifting from the NPZ it
    claims to describe. If this fails after an intentional data change,
    regenerate via `python3 test_system/capture_streamlit_baseline.py`."""
    assert BASELINE_PATH.is_file(), f"Baseline missing: {BASELINE_PATH}"
    with open(BASELINE_PATH) as f:
        frozen = json.load(f)
    fresh = compute_eqdyna_a_expected()
    assert frozen == fresh


def test_mutation_sensitivity_station_count(tmp_path):
    """Mutation-style self-check (not part of the production assertions):
    confirms the station-count assertion actually fails when the displayed
    value is wrong, by running AppTest against a scratch copy of the app
    with one line deliberately mutated. Never mutates src/ in place."""
    mutated = tmp_path / "mutated_app.py"
    original = APP_PATH.read_text()
    needle = 'st.metric("Total Stations", len(data.get(\'station_ids\', [])))'
    assert needle in original, "App source changed shape; update this mutation probe"
    mutated_needle = needle.replace(
        "len(data.get('station_ids', []))", "len(data.get('station_ids', [])) + 1"
    )
    mutated.write_text(original.replace(needle, mutated_needle, 1))

    at = AppTest.from_file(str(mutated), default_timeout=60)
    at.secrets = dict(DUMMY_SECRETS)
    at.run()
    assert not at.exception

    expected = compute_eqdyna_a_expected()
    total_stations_metric = next(m for m in at.metric if m.label == "Total Stations")
    mutated_value = int(total_stations_metric.value)
    assert mutated_value != expected["station_count"], (
        "Mutation probe did not change the displayed value -- "
        "the station-count assertion would not catch this class of bug"
    )
    assert mutated_value == expected["station_count"] + 1
