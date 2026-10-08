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
#   bash tests/run_e2e_bundle.sh <path/to/dr4gm_data_vX.Y.Z.tar.gz>
#   DR4GM_BUNDLE=/path/to/bundle.tar.gz bash tests/run_e2e_bundle.sh
#
# Asserts:
#   (a) regen_ensemble_figures.sh exits 0
#   (b) figs_to_publish/ contains exactly the committed FULL manifest
#       (tests/e2e_reference/figure_manifest_full.txt, 41 parts) of
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
#       matches tests/e2e_reference/ensemble_summary_reference.npz
#       within float32-aware tolerance (rel 1e-6), via
#       tests/extract_ensemble_summary.py
#
# Regenerating the reference (after an intentional, explained pipeline
# change):
#   bash tests/run_e2e_bundle.sh --bless <bundle>
# writes a fresh manifest + reference npz + per-figure PNG dimension
# manifest. Requires `openquake` to be importable (bless always blesses
# against the FULL 41-figure run -- never bless a degraded manifest). Only do
# this when the change is understood and explained in the commit message,
# per the golden-file policy in local/CLAUDE.md.
#
# (d) per-figure PNG pixel dimensions match the committed manifest
#     tests/e2e_reference/figure_dims_full.txt (WIDTHxHEIGHT per part,
#     read via Pillow -- a figure that exists but silently changed aspect
#     ratio / dpi / layout is a regression the existence+magic-byte checks
#     above cannot see). Figure14B.png is exempted from the dims check under
#     the same openquake-missing condition as (b) (it never gets blessed
#     dims on a machine without openquake, so there is nothing to compare).
#
# (e) the Fig 11 panel set contains NO Figure11C*.png (sord) or
#     Figure11F*.png (specfem3d) parts -- those two codes have no
#     per-station NPZ by design (see CLAUDE.md "Fig 11 gaps"), so a panel
#     appearing for either is itself a regression, independent of the
#     41-part count.
#
# Export modes (EXPORT_MODE env var, default "worktree"):
#   worktree (default) -- exports tracked files AS THEY STAND IN THE WORKING
#     TREE (git ls-files + rsync), so local uncommitted edits are exercised.
#   clean -- exports strictly the HEAD commit via `git archive HEAD`, so no
#     untracked/uncommitted file can contaminate the run. Use via the
#     run_e2e_bundle_clean.sh wrapper for the CI-facing, reproducibility-grade
#     invocation.
#
# INJECT_SEISSOL2=1 env var (test-of-the-test / negative control only, see
# run_e2e_bundle_negative_control.sh): after exporting, patches the scratch
# copy's scripts/regen_ensemble_figures.sh to re-include the excluded
# seissol/2 scenario. Never touches the real repo. This is expected to make
# the test FAIL (manifest and/or numeric-summary mismatch) -- that failure is
# the proof the exclusion is actually load-bearing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_SRC="$(cd "$SCRIPT_DIR/.." && pwd)"
REF_DIR="$SCRIPT_DIR/e2e_reference"
FULL_MANIFEST="$REF_DIR/figure_manifest_full.txt"
DIMS_MANIFEST="$REF_DIR/figure_dims_full.txt"
SUMMARY_REF="$REF_DIR/ensemble_summary_reference.npz"
GMPE_SKIP_FIGURE="Figure14B.png"
EXPORT_MODE="${EXPORT_MODE:-worktree}"
INJECT_SEISSOL2="${INJECT_SEISSOL2:-0}"

BLESS=0
if [ "${1:-}" = "--bless" ]; then
    BLESS=1
    shift
fi

BUNDLE="${1:-${DR4GM_BUNDLE:-}}"
if [ -z "$BUNDLE" ]; then
    echo "Usage: bash tests/run_e2e_bundle.sh [--bless] <bundle.tar.gz>" >&2
    echo "   or: DR4GM_BUNDLE=<bundle.tar.gz> bash tests/run_e2e_bundle.sh" >&2
    exit 2
fi
if [ ! -f "$BUNDLE" ]; then
    echo "FAIL: bundle not found: $BUNDLE" >&2
    exit 1
fi
BUNDLE="$(cd "$(dirname "$BUNDLE")" && pwd)/$(basename "$BUNDLE")"

TMPDIR="$(mktemp -d -t dr4gm-e2e-XXXXXX)"
# DR4GM_DEBUG_KEEP_DIMS=<path> (maintainer debug hook, not part of any
# documented interface): copies out the freshly-computed got_dims.txt before
# the scratch dir is removed, so a maintainer blessing figure_dims_full.txt
# on a machine without openquake can still harvest the 40 non-gated dims
# without hand-editing this script.
cleanup() { [ -n "${DR4GM_DEBUG_KEEP_DIMS:-}" ] && cp "$TMPDIR/got_dims.txt" "$DR4GM_DEBUG_KEEP_DIMS" 2>/dev/null; rm -rf "$TMPDIR"; }
trap cleanup EXIT

WORK="$TMPDIR/repo"
mkdir -p "$WORK"
if [ "$EXPORT_MODE" = "clean" ]; then
    echo "=== Exporting repo (clean: git archive HEAD, no working-tree contamination) ==="
    ARCHIVE="$TMPDIR/head.tar"
    ( cd "$REPO_SRC" && command git archive HEAD -o "$ARCHIVE" )
    if [ ! -s "$ARCHIVE" ]; then
        echo "FAIL: git archive HEAD produced an empty archive -- is $REPO_SRC a git checkout?" >&2
        exit 1
    fi
    tar -xf "$ARCHIVE" -C "$WORK"
elif [ "$EXPORT_MODE" = "worktree" ]; then
    echo "=== Exporting repo (tracked files + working changes) to scratch dir ==="
    REPO_FILELIST="$TMPDIR/filelist"
    ( cd "$REPO_SRC" && command git ls-files -z ) > "$REPO_FILELIST"
    if [ ! -s "$REPO_FILELIST" ]; then
        echo "FAIL: repo file listing is empty -- is $REPO_SRC a git checkout?" >&2
        exit 1
    fi
    rsync -a --files-from="$REPO_FILELIST" --from0 "$REPO_SRC/" "$WORK/"
else
    echo "FAIL: unknown EXPORT_MODE='$EXPORT_MODE' (expected 'worktree' or 'clean')" >&2
    exit 2
fi

if [ "$INJECT_SEISSOL2" = "1" ]; then
    echo "=== INJECT_SEISSOL2=1: patching scratch copy to re-include excluded seissol/2 ===" >&2
    REGEN="$WORK/scripts/regen_ensemble_figures.sh"
    if ! grep -q '# seissol/2 — excluded' "$REGEN"; then
        echo "FAIL: INJECT_SEISSOL2 expected a commented-out 'seissol/2' line in $REGEN and did not find one (exclusion mechanism changed -- update the injector)" >&2
        exit 1
    fi
    sed -i 's/^    # seissol\/2 — excluded.*/    seissol\/2/' "$REGEN"
    sed -i 's/^    seissol\/1 seissol\/3 seissol\/4 seissol\/5$/    seissol\/1 seissol\/2 seissol\/3 seissol\/4 seissol\/5/' "$REGEN"
    if ! grep -qx '    seissol/2' "$REGEN"; then
        echo "FAIL: INJECT_SEISSOL2 patch did not take (ALL_SCENARIOS still excludes seissol/2)" >&2
        exit 1
    fi
    if ! grep -q 'seissol/2 seissol/3' "$REGEN"; then
        echo "FAIL: INJECT_SEISSOL2 patch did not take (FIG12_SCENARIOS still excludes seissol/2)" >&2
        exit 1
    fi
fi

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

echo "=== Checking Fig 11 panel set excludes SORD (C) / SPECFEM3D (F) ==="
# Per CLAUDE.md "Fig 11 gaps": SORD and SPECFEM3D have no per-station NPZ, so
# fetch_figures_for_publication.sh's Fig-11 loop (which only fires on
# RSA_T_1.000_map.png existing) can never emit Figure11C*/Figure11F* -- this
# is a structural invariant, not a golden value, so it is checked
# unconditionally (not gated on --bless or openquake availability).
if grep -q '^Figure11C' "$TMPDIR/got_manifest.txt"; then
    echo "FAIL: found a Figure11C*.png (sord) panel -- sord has no per-station NPZ by design; this should be impossible" >&2
    grep '^Figure11C' "$TMPDIR/got_manifest.txt" >&2
    exit 1
fi
if grep -q '^Figure11F' "$TMPDIR/got_manifest.txt"; then
    echo "FAIL: found a Figure11F*.png (specfem3d) panel -- specfem3d has no per-station NPZ by design; this should be impossible" >&2
    grep '^Figure11F' "$TMPDIR/got_manifest.txt" >&2
    exit 1
fi
echo "Confirmed: no Figure11C*/Figure11F* (sord/specfem3d) panels present"

echo "=== Checking per-figure PNG pixel dimensions ==="
: > "$TMPDIR/got_dims.txt"
while IFS= read -r f; do
    path="$PF/$f"
    dims="$(python3 -c "from PIL import Image; im = Image.open('$path'); print(f'{im.size[0]}x{im.size[1]}')")"
    echo "$f $dims" >> "$TMPDIR/got_dims.txt"
done < "$TMPDIR/got_manifest.txt"
sort -o "$TMPDIR/got_dims.txt" "$TMPDIR/got_dims.txt"

if [ "$BLESS" -eq 1 ]; then
    if [ "$HAVE_OPENQUAKE" -eq 0 ]; then
        echo "FAIL: --bless requires \`openquake\` importable (bless always blesses the FULL 41-figure run, never a degraded manifest)." >&2
        exit 1
    fi
    mkdir -p "$REF_DIR"
    cp "$TMPDIR/got_manifest.txt" "$FULL_MANIFEST"
    cp "$TMPDIR/got_dims.txt" "$DIMS_MANIFEST"
    echo "Blessed full manifest -> $FULL_MANIFEST ($GOT_COUNT parts)"
    echo "Blessed dims manifest -> $DIMS_MANIFEST ($GOT_COUNT parts)"
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

    if [ ! -f "$DIMS_MANIFEST" ]; then
        echo "FAIL: no committed dims manifest at $DIMS_MANIFEST (run with --bless first, with openquake installed)" >&2
        exit 1
    fi
    if [ "$HAVE_OPENQUAKE" -eq 1 ]; then
        EXPECTED_DIMS="$DIMS_MANIFEST"
    else
        EXPECTED_DIMS="$TMPDIR/expected_degraded_dims.txt"
        grep -v "^$GMPE_SKIP_FIGURE " "$DIMS_MANIFEST" > "$EXPECTED_DIMS"
    fi
    if ! diff -u "$EXPECTED_DIMS" "$TMPDIR/got_dims.txt"; then
        echo "FAIL: figure pixel dimensions differ from $DIMS_MANIFEST (aspect ratio / dpi / layout regression)" >&2
        exit 1
    fi
    echo "All figure pixel dimensions match $DIMS_MANIFEST exactly"
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
