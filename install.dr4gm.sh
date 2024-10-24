#! /bin/bash

echo "To install dr4gm, Python libaries numpy and scipy is required."
#apt-get update
#apt-get install vim git python-is-python3 python3-pip
pip3 install numpy scipy

chmod -R 755 utils
bash installDepreciatedGMPE-SMTK.sh

# Set up the environment variables
# Add paths to the environment variables
DR4GM=$(pwd)
SMTK=$DR4GM/gmpe-smtk
UTILS=$DR4GM/utils
echo $DR4GM
echo $SMTK
echo $UTILS

additionalPath="$SMTK:$UTILS"
echo $additionalPath
export PATH=$PATH:$additionalPath
export PYTHONPATH=$PYTHONPATH:$additionalPath
echo $PATH
echo $PYTHONPATH
