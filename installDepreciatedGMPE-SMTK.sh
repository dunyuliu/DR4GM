#! bin/bash 
git clone https://github.com/GEMScienceTools/gmpe-smtk
cd gmpe-smtk
git checkout 4f008173e89f6e4ba4450fb43e95ffdf51a7c2ba^
cd smtk
sed -i '' 's/cumtrapz/cumulative_trapezoid/g' *.py # cumtrapz is depreciated from scipy as of 20241023.
cd ..
cd ..
#now add gmpe-smtk to your python path
#export PYTHONPATH=$PYTHONPATH:$(pwd)
##you could also consider adding it to your ~.bashrc
#echo 'export PYTHONPATH=$PYTHONPATH:$(pwd)' >> ~/.bashrc
