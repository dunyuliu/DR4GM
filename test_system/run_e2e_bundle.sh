#!/bin/bash
#
# run_e2e_bundle.sh -- end-to-end test of the PUBLIC reproduction path:
#   Zenodo data bundle -> regen_ensemble_figures.sh -> figures + statistics.
#
# Exercises exactly the README "Reproduce manuscript Figs 11-19" recipe
# (mkdir -p results && tar xzf <bundle> -C results/ && bash
# regen_ensemble_figures.sh), but inside a scratch copy of the repo so the
# real checkout's results/ is never touched. Needs only the ~14 MB Zenodo
# bundle -- no 199 GB raw data, no reference/ access.
#
# Usage:
#   bash test_system/run_e2e_bundle.sh <path/to/dr4gm_data_vX.Y.Z.tar.gz>
#   DR4GM_BUNDLE=/path/to/bundle.tar.gz bash test_system/run_e2e_bundle.sh
#
# Asserts:
#   (a) regen_ensemble_figures.sh exits 0
#   (b) figs_to_publish/ contains exactly the committed FULL manifest
#       (test_system/e2e_reference/figure_manifest_full.txt, 41 parts) of
#       Figure<NN><letter>.png parts, each a non-empty, valid PNG --
#       UNLESS `openquake` is not importable in this Python, in which case
#       Figure14B.png (SA bias vs period; needs the NGA-West2 GMPE from
#       openquake.hazardlib via src/utils/openquake_engine_gmpe.py) is known to
#       be produced by the pipeline's own optional-dependency guard
#       (visualize_ensemble_stats.py: PLOT_GMPE_AVAILABLE). In that case the
#       test prints a loud, unmissable SKIP banner naming the missing
#       dependency and the blocked figure, and asserts the manifest matches
#       the full list MINUS exactly Figure14B.png -- any other figure going
#       missing still fails the test.
#   (c) a small numeric summary of the Figs 13/17 binned curves (per-code
#       group-mean PGA/CAV/RSA_T_1.000 vs distance, epistemic tau at T=1s)
#       matches test_system/e2e_reference/ensemble_summary_reference.npz
#       within float32-aware tolerance (rel 1e-6), via
#       test_system/extract_ensemble_summary.py
#
# Regenerating the reference (after an intentional, explained pipeline
# change):
#   bash test_system/run_e2e_bundle.sh --bless <bundle>
# writes a fresh manifest + reference npz. Requires `openquake` to be
# importable (bless always blesses against the FULL 41-figure run -- never
# bless a degraded manifest). Only do this when the change is understood and
# explained in the commit message, per the golden-file policy in
# local/CLAUDE.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_SRC="$(cd "$SCRIPT_DIR/.." && pwd)"
REF_DIR="$SCRIPT_DIR/e2e_reference"
FULL_MANIFEST="$REF_DIR/figure_manifest_full.txt"
SUMMARY_REF="$REF_DIR/ensemble_summary_reference.npz"
GMPE_SKIP_FIGURE="Figure14B.png"

BLESS=0
if [ "${1:-}" = "--bless" ]; then
    BLESS=1
    shift
fi

BUNDLE="${1:-${DR4GM_BUNDLE:-}}"
if [ -z "$BUNDLE" ]; then
    echo "Usage: bash test_system/run_e2e_bundle.sh [--bless] <bundle.tar.gz>" >&2
    echo "   or: DR4GM_BUNDLE=<bundle.tar.gz> bash test_system/run_e2e_bundle.sh" >&2
    exit 2
fi
if [ ! -f "$BUNDLE" ]; then
    echo "FAIL: bundle not found: $BUNDLE" >&2
    exit 1
fi
BUNDLE="$(cd "$(dirname "$BUNDLE")" && pwd)/$(basename "$BUNDLE")"

TMPDIR="$(mktemp -d -t dr4gm-e2e-XXXXXX)"
cleanup() { rm -rf "$TMPDIR"; }
trap cleanup EXIT

echo "=== Exporting repo (tracked files + working changes) to scratch dir ==="
WORK="$TMPDIR/repo"
mkdir -p "$WORK"
REPO_FILELIST="$TMPDIR/filelist"
( cd "$REPO_SRC" && command git ls-files -z ) > "$REPO_FILELIST"
if [ ! -s "$REPO_FILELIST" ]; then
    echo "FAIL: repo file listing is empty -- is $REPO_SRC a git checkout?" >&2
    exit 1
fi
rsync -a --files-from="$REPO_FILELIST" --from0 "$REPO_SRC/" "$WORK/"

echo "=== Unpacking bundle per README recipe: $BUNDLE ==="
mkdir -p "$WORK/results"
tar xzf "$BUNDLE" -C "$WORK/results/"
if [ ! -d "$WORK/results/production_runs" ]; then
    echo "FAIL: bundle did not produce results/production_runs/ (README recipe mismatch)" >&2
    exit 1
fi

HAVE_OPENQUAKE=0
if python3 -c "import openquake" >/dev/null 2>&1; then
    HAVE_OPENQUAKE=1
fi

if [ "$HAVE_OPENQUAKE" -eq 0 ]; then
    cat >&2 <<'BANNER'
################################################################################
# SKIP WARNING: `openquake` is NOT importable in this Python environment.
# The NGA-West2 GMPE comparison (src/utils/openquake_engine_gmpe.py) is disabled
# by the pipeline's own optional-dependency guard (PLOT_GMPE_AVAILABLE in
# src/utils/visualize_ensemble_stats.py), so Figure14B.png (SA bias vs period)
# will NOT be produced this run.
#
# This test will assert the full manifest MINUS Figure14B.png only. It will
# still FAIL if any other figure is missing. Install `openquake.engine`
# (pip install openquake.engine) to exercise the complete 41-figure path.
################################################################################
BANNER
fi

echo "=== Running regen_ensemble_figures.sh ==="
set +e
( cd "$WORK" && bash scripts/regen_ensemble_figures.sh ) > "$TMPDIR/regen.log" 2>&1
REGEN_RC=$?
set -e
if [ "$REGEN_RC" -ne 0 ]; then
    echo "FAIL: regen_ensemble_figures.sh exited $REGEN_RC" >&2
    tail -n 60 "$TMPDIR/regen.log" >&2
    exit 1
fi
echo "regen_ensemble_figures.sh: exit 0"

PF="$WORK/results/production_runs/figs_to_publish"
if [ ! -d "$PF" ]; then
    echo "FAIL: $PF does not exist" >&2
    exit 1
fi

echo "=== Checking figure manifest ==="
( cd "$PF" && find . -maxdepth 1 -name 'Figure*.png' -printf '%f\n' | sort ) > "$TMPDIR/got_manifest.txt"
GOT_COUNT="$(wc -l < "$TMPDIR/got_manifest.txt")"
echo "Found $GOT_COUNT Figure*.png parts"

while IFS= read -r f; do
    path="$PF/$f"
    if [ ! -s "$path" ]; then
        echo "FAIL: $f is empty" >&2
        exit 1
    fi
    magic="$(head -c 8 "$path" | od -An -tx1 | tr -d ' \n')"
    if [ "$magic" != "89504e470d0a1a0a" ]; then
        echo "FAIL: $f is not a valid PNG (bad magic bytes)" >&2
        exit 1
    fi
done < "$TMPDIR/got_manifest.txt"
echo "All $GOT_COUNT parts are non-empty, valid PNGs"

if [ "$BLESS" -eq 1 ]; then
    if [ "$HAVE_OPENQUAKE" -eq 0 ]; then
        echo "FAIL: --bless requires \`openquake\` importable (bless always blesses the FULL 41-figure run, never a degraded manifest)." >&2
        exit 1
    fi
    mkdir -p "$REF_DIR"
    cp "$TMPDIR/got_manifest.txt" "$FULL_MANIFEST"
    echo "Blessed full manifest -> $FULL_MANIFEST ($GOT_COUNT parts)"
else
    if [ ! -f "$FULL_MANIFEST" ]; then
        echo "FAIL: no committed manifest at $FULL_MANIFEST (run with --bless first, with openquake installed)" >&2
        exit 1
    fi
    if [ "$HAVE_OPENQUAKE" -eq 1 ]; then
        EXPECTED="$FULL_MANIFEST"
        EXPECTED_DESC="full manifest ($FULL_MANIFEST)"
    else
        EXPECTED="$TMPDIR/expected_degraded_manifest.txt"
        grep -vFx "$GMPE_SKIP_FIGURE" "$FULL_MANIFEST" > "$EXPECTED"
        EXPECTED_DESC="full manifest minus $GMPE_SKIP_FIGURE (openquake unavailable)"
    fi
    if ! diff -u "$EXPECTED" "$TMPDIR/got_manifest.txt"; then
        echo "FAIL: figure manifest differs from $EXPECTED_DESC" >&2
        echo "      (a diff here that is NOT exactly $GMPE_SKIP_FIGURE means something" >&2
        echo "       besides the known openquake-gated figure broke -- investigate it,"  >&2
        echo "       do not add it to the skip list)" >&2
        exit 1
    fi
    echo "Figure manifest matches $EXPECTED_DESC exactly ($GOT_COUNT parts)"
fi

echo "=== Extracting ensemble numeric summary ==="
SUMMARY="$TMPDIR/ensemble_summary.npz"
PYTHONPATH="$WORK/src/utils" python3 "$SCRIPT_DIR/extract_ensemble_summary.py" \
    "$WORK/results/production_runs" "$SUMMARY"

if [ "$BLESS" -eq 1 ]; then
    mkdir -p "$REF_DIR"
    cp "$SUMMARY" "$SUMMARY_REF"
    echo "Blessed numeric reference -> $SUMMARY_REF"
    echo "=== BLESS complete. Review and commit $REF_DIR. ==="
    exit 0
fi

if [ ! -f "$SUMMARY_REF" ]; then
    echo "FAIL: no committed numeric reference at $SUMMARY_REF (run with --bless first)" >&2
    exit 1
fi

echo "=== Comparing numeric summary to reference (rel tol 1e-6, float32-aware) ==="
set +e
python3 "$SCRIPT_DIR/compare_e2e_summary.py" "$SUMMARY_REF" "$SUMMARY"
PYRC=$?
set -e
if [ "$PYRC" -ne 0 ]; then
    echo "FAIL: numeric summary diverged from reference" >&2
    exit 1
fi

echo "=== E2E bundle test PASSED ==="
