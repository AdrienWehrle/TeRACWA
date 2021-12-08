# Terrestrial Radar Assessment of Calving Wave Activity (TeRACWA)

TeRACWA is a novel automated method for the detection and the quantification of ocean waves generated by glacier calving using Terrestrial Radar Interferometer (TRI) acquisitions of backscatter intensity. 

## Environment setup

It is recommended to use TeRACWA in a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) environment that can be created as follows.
``` bash
conda env create -f TeRACWA_environment.yml
```
To complete the environment setup, [gpritools](https://git.sr.ht/~scinu/gpritools) needs to be cloned and made available for import by e.g. updating ```PYTHONPATH```. It can also be modified using standard list operations as follows.

``` python
import sys
sys.path.append('/path/to/gpritools')
``` 

## Usage

The easiest and most direct way to run TeRACWA is to import it as a module and run the wrapper method with the default arguments.

``` python
import TeRACWA

# create an instance of TeRACWA with default run parameters
my_teracwa = TeRACWA(slc_directory='/path/to/slc')

# run TeRAWA
my_teracwa.run()
``` 

In this case, the different steps of TeRACWA are run with the parameters described in [Wehrlé et al, 2021](https://tc.copernicus.org/preprints/tc-2021-33/). The results were saved locally in the class attribute ```my_teracwa.results``` which is a dictionnary containing the outputs of the different steps, and the whole instance (including run parameters) was saved in a pickle file in the current directory.

The different steps can also be be run individually and some results affected to a local variable, see example below.

``` python
import TeRACWA

# create a TeRACWA instance
my_teracwa = TeRACWA(slc_directory='/path/to/slc')

# list the files to process
my_teracwa.list_slc_files()

# get general acquisition characteristics
my_teracwa.get_data_specs()

# run the core method of TeRACWA
raw_power_maxima = my_teracwa.run_spectral_analysis()
``` 
