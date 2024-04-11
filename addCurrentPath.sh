#!/bin/bash

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
