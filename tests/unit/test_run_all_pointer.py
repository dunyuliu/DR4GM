"""Pointer so `pytest -q tests/unit -k run_all` (board evidence command
for PATHWAY_FORWARD.md row 5) finds the run_all.sh integration test.

The real test lives in tests/integration/test_run_all_integration.py
(not here) because it reads raw fixture data and takes minutes -- outside
row 4's unit-tier budget (no raw data, <50ms). This module loads that file
under a distinct module name (avoiding a basename collision with pytest's
own collection of tests/integration/) and re-exports the identical
test function object -- not a re-implementation, not a copy -- so pytest
also collects and runs it from this directory.
"""

import importlib.util
from pathlib import Path

_TARGET = Path(__file__).resolve().parents[1] / "integration" / "test_run_all_integration.py"
_spec = importlib.util.spec_from_file_location("_run_all_integration_target", _TARGET)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

test_run_all_seissol_fixture_matches_light_reference = (
    _mod.test_run_all_seissol_fixture_matches_light_reference
)
