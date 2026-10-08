#!/bin/bash
#
# run_e2e_bundle_negative_control.sh -- proves the seissol/2 exclusion in
# scripts/regen_ensemble_figures.sh is load-bearing, not decorative.
#
# Mutation-style check (per local/CLAUDE.md testing discipline): re-adds the
# deliberately-excluded seissol/2 scenario (median SA(T=1s) ~5x below the
# other 4 seissol runs -- see CLAUDE.md "seissol/2 is excluded everywhere")
# to a SCRATCH COPY of the repo only (via run_e2e_bundle.sh's
# INJECT_SEISSOL2=1 hook; the real checkout is never touched) and asserts
# that run_e2e_bundle_clean.sh then FAILS against the real committed
# references. If it does not fail, the exclusion check is not actually
# catching this regression on the real data -- that is reported as a gap,
# not silently fixed by weakening this script.
#
# Usage:
#   bash tests/run_e2e_bundle_negative_control.sh <bundle.tar.gz>
#   DR4GM_BUNDLE=/path/to/bundle.tar.gz bash tests/run_e2e_bundle_negative_control.sh
#
# Exit 0  = negative control PASSED (the mutated run correctly failed).
# Exit 1  = GAP: the mutated run did NOT fail -- the exclusion is not
#           actually enforced by this test suite on this data. Fix the test,
#           don't suppress this script.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUNDLE="${1:-${DR4GM_BUNDLE:-}}"
if [ -z "$BUNDLE" ]; then
    echo "Usage: bash tests/run_e2e_bundle_negative_control.sh <bundle.tar.gz>" >&2
    exit 2
fi

LOG="$(mktemp -t dr4gm-negctl-XXXXXX.log)"
cleanup() { rm -f "$LOG"; }
trap cleanup EXIT

echo "=== Negative control: re-adding excluded seissol/2 in a scratch copy, expecting FAILURE ==="
EXPORT_MODE=clean INJECT_SEISSOL2=1 bash "$SCRIPT_DIR/run_e2e_bundle.sh" "$BUNDLE" > "$LOG" 2>&1
MUTATED_RC=$?

echo "--- mutated run (seissol/2 re-added) log tail ---"
tail -n 40 "$LOG"
echo "--- mutated run exit code: $MUTATED_RC ---"

if [ "$MUTATED_RC" -eq 0 ]; then
    echo "" >&2
    echo "################################################################################" >&2
    echo "# GAP: re-adding seissol/2 did NOT make run_e2e_bundle.sh fail." >&2
    echo "# The exclusion is documented in scripts/regen_ensemble_figures.sh and" >&2
    echo "# CLAUDE.md but this test suite's manifest/dims/numeric checks do not" >&2
    echo "# actually catch its regression on this data. Report this as a real finding" >&2
    echo "# -- do not loosen or delete this negative control to make it pass." >&2
    echo "################################################################################" >&2
    exit 1
fi

echo ""
echo "Negative control PASSED: re-adding seissol/2 correctly made run_e2e_bundle.sh fail (rc=$MUTATED_RC)."
echo "(No repo file was modified -- the mutation existed only inside run_e2e_bundle.sh's own scratch copy.)"
exit 0
