#NOAH project software repository

# In order to run the NOAH RTC Tool create a conda environment as described below: 

# set up a conda environment that ensures that the packages are available. 

# Type in all lines one by one in the anaconda prompt to create this environment.
conda create -n NOAH python=3.7 numpy pandas matplotlib scipy 
# conda install -n NOAH (probably not necessary)
conda activate NOAH
conda install spyder
# Then install with pip in the same environment
pip install pyswmm
pip install tkintertable
pip install swmmtoolbox

# For older versions from Gitlab see project https://gitlab.gbar.dtu.dk/jowi/NOAH 