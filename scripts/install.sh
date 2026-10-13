#! /bin/bash
# DR4GM installer.
# gmpe-smtk is vendored in-tree under ./src/gmpe-smtk (AGPLv3, (C) GEM Foundation).
# No separate clone or post-install patching is required.

# Install into a dedicated virtual environment, never into the invoking
# user's ambient/system/user-site Python. A bare `pip3 install` here would
# silently mutate whatever environment the caller happens to have active
# (observed: it upgraded a shared host's numpy and left scipy ABI-broken).
DR4GM=$(pwd)
VENV="$DR4GM/.venv"

if [ ! -d "$VENV" ]; then
    echo "Creating virtual environment at $VENV..."
    # Note: if this script is `source`d (as README.md instructs), `return`
    # stops just the script without killing the caller's shell; if it was
    # run via `bash scripts/install.sh`, `return` errors here and falls
    # through to `exit 1`.
    #
    # Fast path: stdlib venv. This fails on hosts where ensurepip/python3-venv
    # is missing or broken (observed on the release host itself), even though
    # pip3 and virtualenv both work fine there. Fall back to the third-party
    # `virtualenv` package, which bundles its own pip bootstrap and does not
    # depend on stdlib ensurepip. Only if both paths fail do we abort — never
    # silently fall through to installing into ambient/system/user-site Python.
    if ! python3 -m venv "$VENV" 2>/tmp/dr4gm_venv_err.$$; then
        cat /tmp/dr4gm_venv_err.$$ >&2
        rm -f /tmp/dr4gm_venv_err.$$
        echo "WARNING: 'python3 -m venv' failed (see above, e.g. missing python3-venv/ensurepip). Falling back to the 'virtualenv' package." >&2
        if ! command -v virtualenv >/dev/null 2>&1; then
            echo "Installing 'virtualenv' into the invoking user's own site (pip3 install --user)..." >&2
            pip3 install --user virtualenv || { echo "ERROR: 'pip3 install --user virtualenv' failed. Aborting — not falling back to the ambient/system/user-site Python." >&2; return 1 2>/dev/null; exit 1; }
        fi
        virtualenv "$VENV" || { echo "ERROR: both 'python3 -m venv' and 'virtualenv' failed to create $VENV. Aborting — not falling back to the ambient/system/user-site Python." >&2; return 1 2>/dev/null; exit 1; }
    fi
    rm -f /tmp/dr4gm_venv_err.$$
fi

echo "Installing DR4GM Python dependencies into $VENV..."
"$VENV/bin/pip" install -r requirements.txt || { echo "ERROR: dependency install into $VENV failed. Aborting." >&2; return 1 2>/dev/null; exit 1; }

# Activate for the remainder of this script. If this script was `source`d
# (as README.md instructs), the activation also persists into the caller's
# shell; if it was run via `bash scripts/install.sh`, it does not — see the
# reminder printed at the end.
# shellcheck disable=SC1091
source "$VENV/bin/activate"

chmod -R 755 src/utils

# Set up environment variables for the current shell
SMTK=$DR4GM/src/gmpe-smtk
UTILS=$DR4GM/src/utils
echo "DR4GM=$DR4GM"
echo "SMTK=$SMTK"
echo "UTILS=$UTILS"

additionalPath="$SMTK:$UTILS"
export PATH=$PATH:$additionalPath
export PYTHONPATH=$PYTHONPATH:$additionalPath
echo "PATH appended with: $additionalPath"
echo "PYTHONPATH appended with: $additionalPath"

# data/reference: local-only link to the ~199 GB full reference dataset, used
# by tests/run_tests.sh's full tier and tests/derive_light_reference.sh. Never
# tracked (see .gitignore). Override the source with DR4GM_REFERENCE_DIR; the
# default matches this machine's shared dataset store layout. No-op if the
# target doesn't exist (e.g. a stranger clone with no full-tier access) or a
# link/file is already there.
REFDIR="${DR4GM_REFERENCE_DIR:-$HOME/shared_dataset/dr4gm_drv.reference}"
if [ -L "$DR4GM/data/reference" ] && [ ! -e "$DR4GM/data/reference" ]; then
    # Pre-existing dangling symlink (e.g. stale target from a prior host/run):
    # `ln -s` refuses to overwrite it since the path itself still exists, so
    # relink explicitly instead of silently leaving the dangling link in place.
    echo "data/reference: replacing dangling symlink (was -> $(readlink "$DR4GM/data/reference"))"
    rm "$DR4GM/data/reference"
fi
if [ -e "$DR4GM/data/reference" ]; then
    : # already a valid link/file, leave it alone
elif [ -e "$REFDIR" ]; then
    ln -s "$REFDIR" "$DR4GM/data/reference"
    echo "data/reference -> $REFDIR"
else
    echo "data/reference: NOT created — $REFDIR does not exist on this host." \
         "Full-tier tests (tests/run_tests.sh) and tests/derive_light_reference.sh" \
         "will fail until this is set (override with DR4GM_REFERENCE_DIR)."
fi

echo ""
# GMPE comparison (Figure14B) preflight. Informational only -- GMPE comparison
# is an optional, documented-as-optional dependency (PROJECT_RULES.md rule 4),
# so this never fails/aborts the install. Checking bare `import openquake`
# is not enough: that top-level import can succeed while the deeper
# openquake_engine_gmpe import still fails (missing GEM geospatial/GDAL
# wheels raise from inside openquake's hazardlib on first real use), so probe
# the actual module the pipeline imports, matching PLOT_GMPE_AVAILABLE in
# src/utils/visualize_ensemble_stats.py.
if "$VENV/bin/python" -c "
import sys
sys.path.insert(0, '$UTILS')
from openquake_engine_gmpe import get_nga_west2_gmpe_predictions_cached, get_cav_gmm_predictions
" >/dev/null 2>&1; then
    echo "GMPE comparison (Figure14B) available."
else
    echo "GMPE comparison (Figure14B) NOT available -- see README.md for optional GDAL/openquake setup."
fi

echo ""
echo "Virtual environment ready: $VENV"
echo "If this script was run (e.g. 'bash scripts/install.sh') rather than"
echo "sourced, the venv activation above did not persist into your shell."
echo "Activate it yourself with:"
echo "    source $VENV/bin/activate"
