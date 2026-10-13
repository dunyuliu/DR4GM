#! /bin/bash
# DR4GM installer.
# gmpe-smtk is vendored in-tree under ./src/gmpe-smtk (AGPLv3, (C) GEM Foundation).
# No separate clone or post-install patching is required.

echo "Installing DR4GM Python dependencies..."
pip3 install -r requirements.txt

chmod -R 755 src/utils

# Set up environment variables for the current shell
DR4GM=$(pwd)
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
