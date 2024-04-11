#! /bin/bash

apt-get update
apt-get install vim git python-is-python3 python3-pip
pip install numpy scipy

chmod -R 755 utils
bash installDepreciatedGMPE-SMTK.sh

