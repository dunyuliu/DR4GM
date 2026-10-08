#!/bin/bash
#
# run_e2e_bundle_clean.sh -- same test as run_e2e_bundle.sh, but exported
# from a clean `git archive HEAD` instead of the live working tree, so no
# untracked file and no uncommitted edit in this checkout can contaminate the
# run. This is the CI-facing / reproducibility-grade invocation: it proves
# the committed HEAD alone reproduces Figs 11-19 byte-for-byte-equivalent
# (manifest + PNG dims) and numerically.
#
# Usage:
#   bash tests/run_e2e_bundle_clean.sh [--bless] <bundle.tar.gz>
#   DR4GM_BUNDLE=/path/to/bundle.tar.gz bash tests/run_e2e_bundle_clean.sh
#
# Thin wrapper: all assertions, --bless behavior, and the INJECT_SEISSOL2
# negative-control hook live in run_e2e_bundle.sh (EXPORT_MODE=clean).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPORT_MODE=clean exec bash "$SCRIPT_DIR/run_e2e_bundle.sh" "$@"
