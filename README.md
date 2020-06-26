# NOAH project software repository

# There are two ways to install the NOAH RTC Tool.

# Run the "install" batch file with python set in the system environment variables. 
To execute the program run the "run" batchfile. 

# If you are familiar with anaconda, create a conda environment as described below: 
# set up a conda environment that ensures that the packages are available. 
# Type in all lines one by one in the anaconda prompt to create this environment.
conda create -n NOAH python=3.7 numpy pandas matplotlib scipy 
conda activate NOAH
conda install spyder
# Then install with pip in the same environment
pip install pyswmm==0.5.2
pip install tkintertable==1.3.2
pip install swmmtoolbox==2.7.12.10
pip install swmmio==0.3.7
# To run the NOAH Tool, run pyswmm_GUI.py from within this conda environment. 
# For older versions from Gitlab see project https://gitlab.gbar.dtu.dk/jowi/NOAH 