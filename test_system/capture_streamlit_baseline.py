#!/usr/bin/env python3
"""Regenerate test_system/e2e_reference/streamlit_apptest_baseline.json.

Run this deliberately, after confirming the displayed numbers changed for a
legitimate reason (e.g. the vendored data/eqdyna.0001.A.coarse.npz asset was
intentionally replaced). A later module-split refactor of
src/web/dr4gm_interactive_explorer.py is explicitly gated on this baseline
NOT changing -- do not regenerate it to make a refactor's diff disappear.

Usage:
    python3 test_system/capture_streamlit_baseline.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from streamlit_baseline_helper import BASELINE_PATH, compute_eqdyna_a_expected  # noqa: E402


def main() -> None:
    expected = compute_eqdyna_a_expected()
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(expected, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {BASELINE_PATH}")
    print(json.dumps(expected, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
