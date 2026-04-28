#! /bin/bash
# DR4GM installer.
# gmpe-smtk is vendored in-tree under ./gmpe-smtk (AGPLv3, (C) GEM Foundation).
# No separate clone or post-install patching is required.

echo "Installing DR4GM Python dependencies (numpy, scipy)..."
pip3 install numpy scipy

chmod -R 755 utils

# Set up environment variables for the current shell
DR4GM=$(pwd)
SMTK=$DR4GM/gmpe-smtk
UTILS=$DR4GM/utils
echo "DR4GM=$DR4GM"
echo "SMTK=$SMTK"
echo "UTILS=$UTILS"

additionalPath="$SMTK:$UTILS"
export PATH=$PATH:$additionalPath
export PYTHONPATH=$PYTHONPATH:$additionalPath
echo "PATH appended with: $additionalPath"
echo "PYTHONPATH appended with: $additionalPath"
