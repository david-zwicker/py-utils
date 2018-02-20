# py-utils
Utility functions for python

The python functions are organized in packages and modules in the subdirectory
`utils`. Unittests reside in subdirectories `tests` in the respective packages.
The top-level folder `tests` provides shell scripts to run all unittests and
determine the test coverage. Finally, the top-level folder `scripts` contains
python code that can be executed as a command directly from the shell. 


## Necessary python packages

The following external packages are required for many of the functions 

Package       | Usage                                      
--------------|-------------------------------------------
numpy         | Array library used for manipulating data
scipy         | Miscellaneous scientific functions
six           | Compatibility layer between python 2 and 3


## Optional python packages

The following packages are only necessary for some packages and need thus not
all be installed. These packages can be installed through `pip`, `macports`,
or similar repositories:

Package       | Usage                                      
--------------|-------------------------------------------
cython        | Compiling code to accelerate some calculations
h5py          | Handling hdf5 files
pandas        | Managing data
portalocker   | Locking files on different platforms
matplotlib    | Plotting graphs
networkx      | Handling graph theory
numba         | Just-in time compiling of python code
tqdm          | Displaying progress bars 
yaml          | Reading and writing structured data (package name: pyyaml)