#!/usr/bin/env python3
"""Regenerate tests/e2e_reference/streamlit_apptest_baseline.json and
tests/e2e_reference/streamlit_apptest_multi_baseline.json.

Run this deliberately, after confirming the displayed numbers changed for a
legitimate reason (e.g. one of the vendored data/*.npz assets was
intentionally replaced). A later module-split refactor of
src/web/dr4gm_interactive_explorer.py is explicitly gated on these baselines
NOT changing -- do not regenerate them to make a refactor's diff disappear.

Usage:
    python3 tests/capture_streamlit_baseline.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from streamlit_baseline_helper import (  # noqa: E402
    BASELINE_PATH,
    MULTI_BASELINE_PATH,
    compute_all_datasets_expected,
    compute_eqdyna_a_expected,
)


def main() -> None:
    expected = compute_eqdyna_a_expected()
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(expected, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {BASELINE_PATH}")
    print(json.dumps(expected, indent=2, sort_keys=True))

    all_expected = compute_all_datasets_expected()
    MULTI_BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MULTI_BASELINE_PATH, "w") as f:
        json.dump(all_expected, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Wrote {MULTI_BASELINE_PATH}")
    print(json.dumps(all_expected, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
