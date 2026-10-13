"""Headless Streamlit AppTest coverage for src/web/dr4gm_interactive_explorer.py.

Runs fully offline:
  - st.secrets is replaced with a dummy dict via AppTest's own `.secrets`
    attribute (never the real, gitignored src/web/.streamlit/secrets.toml,
    which does not exist in this checkout anyway).
  - The default dataset ("EQDyna A Coarse Simulation") resolves to the
    vendored local copy in data/eqdyna.0001.A.coarse.npz (commit 187ebd9),
    so no network access occurs on `at.run()`.

Displayed-value assertions are checked against values computed independently
from the NPZ by tests/streamlit_baseline_helper.py (plain numpy, no
Streamlit), and cross-checked against the frozen baseline JSON in
tests/e2e_reference/ -- see that module's docstring for why both.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
APP_PATH = REPO_ROOT / "src" / "web" / "dr4gm_interactive_explorer.py"

sys.path.insert(0, str(REPO_ROOT / "tests"))
from streamlit_baseline_helper import (  # noqa: E402
    BASELINE_PATH,
    LOCAL_ASSETS,
    MULTI_BASELINE_PATH,
    compute_all_datasets_expected,
    compute_eqdyna_a_expected,
)

# streamlit is a hard requirement (requirements.txt), not optional like
# openquake -- a missing/broken install must hard-fail collection, not
# silently skip this AppTest suite (PATHWAY_FORWARD.md row 32).
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


def _assert_metric_row(table_html: str, metric_name: str, formatted_value: str) -> None:
    """A single Ground Motion Metrics table row, as rendered by
    GroundMotionExplorer.create_all_metrics_tables's HTML string builder in
    src/web/dr4gm_interactive_explorer.py (see the `html_table +=` f-string
    building `<td>{Metric}</td><td ...>{Value}</td>`)."""
    needle = (
        f">{metric_name}</td>"
        f"<td style='padding: 2px 4px; border: 1px solid #ddd; text-align: right; "
        f"font-family: monospace;'>{formatted_value}</td>"
    )
    assert needle in table_html, f"Row for {metric_name!r} (expected value {formatted_value!r}) not found in table"


@pytest.mark.parametrize("dataset_label", sorted(LOCAL_ASSETS.keys()))
def test_full_metrics_surface_matches_npz_for_every_local_dataset(dataset_label):
    """Row 16 extension: for every vendored "DR4GM Data Archive" dataset
    (data/MANIFEST.md), pin Total Stations, Available Metrics, the selected
    station's id, and every row of its Ground Motion Metrics table (all four
    basic metrics PGA/PGV/PGD/CAV plus every spectral-acceleration period) --
    not just the single EQDyna-A/PGA cell the original row-16 tests covered.
    Expected values are computed independently from the NPZ by
    streamlit_baseline_helper.compute_all_datasets_expected (which reproduces
    the same "load default, then switch dataset" session sequence this test
    drives through AppTest, including session-state index carryover -- see
    that function's docstring), never read back from the app itself.
    """
    expected = compute_all_datasets_expected()[dataset_label]

    at = _fresh_app_test()
    at.run()
    assert not at.exception

    # "DR4GM Data Archive" is the default Data Source (index 0); switch only
    # the dataset selectbox to the one under test.
    dataset_selectbox = at.sidebar.selectbox[0]
    assert dataset_selectbox.label == "Choose Dataset"
    dataset_selectbox.set_value(dataset_label)
    at.run()
    assert not at.exception, f"App raised after selecting {dataset_label!r}: {[str(e) for e in at.exception]}"

    total_stations_metric = next(m for m in at.metric if m.label == "Total Stations")
    assert int(total_stations_metric.value) == expected["station_count"]

    available_metrics_metric = next(m for m in at.metric if m.label == "Available Metrics")
    assert int(available_metrics_metric.value) == expected["available_metrics_count"]

    station_id_blocks = [m.value for m in at.markdown if "Station ID:" in m.value]
    assert len(station_id_blocks) == 1
    assert f"<strong>Station ID:</strong> {expected['selected_station_id']}" in station_id_blocks[0]

    metrics_table_blocks = [m.value for m in at.markdown if "<table" in m.value]
    assert len(metrics_table_blocks) == 1
    table_html = metrics_table_blocks[0]

    for metric_name, values in expected["basic_metrics"].items():
        _assert_metric_row(table_html, metric_name, values["formatted"])

    for period, values in expected["sa_metrics"].items():
        _assert_metric_row(table_html, f"SA(T={period}s)", values["formatted"])


def test_multi_baseline_json_matches_fresh_computation():
    """Guards the frozen multi-dataset baseline against silently drifting
    from the NPZs it claims to describe. If this fails after an intentional
    data change, regenerate via `python3 tests/capture_streamlit_baseline.py`."""
    assert MULTI_BASELINE_PATH.is_file(), f"Baseline missing: {MULTI_BASELINE_PATH}"
    with open(MULTI_BASELINE_PATH) as f:
        frozen = json.load(f)
    fresh = compute_all_datasets_expected()
    assert frozen == fresh


def test_baseline_json_matches_fresh_computation():
    """Guards against the frozen baseline silently drifting from the NPZ it
    claims to describe. If this fails after an intentional data change,
    regenerate via `python3 tests/capture_streamlit_baseline.py`."""
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
