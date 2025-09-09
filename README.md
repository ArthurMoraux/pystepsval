# Project Overview

This project provides an automatic computation of scores for Pysteps files, and store them in a dataset. This project is built upon the pystepsval library, which provides a more general framework for validating and analyzing precipitation data using `xarray` and `dask`. It includes tools to process RMI precipitation data, calculate various statistical scores, and handle different types of input data. The main functionalities involve processing datasets, performing spatial alignment, and calculating various metrics such as RMSE, Brier score, and Fraction Skill Score.

## Requirements for DataFrames (`df_model` and `df_observation`)

### General Requirements:
1. **Data Access**:
    - **Pysteps data**: Access to Pysteps .nc files to validate
    - **RADQPE data**: Access to the radar data used as input for Pysteps, (RADQPE2 with mfb correction at RMI).
## Functionality Overview

### 1. `importer.py`
- **Description**: Provides functionality for reading and validating datasets, particularly those related to RMI precipitation data.
- **Main Functions**:
  - `test_xarray_dataset`: Verifies that a dataset has the necessary structure and variables (including `precipitation_rate`).
  - `read_netcdf`: Reads a specific netCDF file from a tarball and adds metadata.
  - `read_radqpe_h5`: Reads RADQPE HDF5 file data and metadata, returned as numpy ndarray and dictionnary.
  - `read_radar`: Reads RMI radar-QPE files based on provided dates and returns an xarray Dataset.
  - `read_in_zarr_file`: Reads a Zarr file and ensures the dataset is correctly formatted.

### 2. `processor.py`
- **Description**: This file contains functions to validate input data, align datasets, and process data for individual or multiple initialization times.
- **Main Functions**:
  - `validate_inputs`: Ensures all input parameters are of the expected type and shape.
  - `create_cached_projector`: Creates and caches projectors for spatial alignment.
  - `spatial_alignment`: Reprojects model data onto the observation grid.
  - `process_data_single_init_time`: Calculates scores for a single initialization time.

### 2. `compute_scores.py`
- **Description**: Automatically compute scores for each Pysteps files available in the `pysteps_dir`. The scores are stored on the disk in a Zarr dataset. The script run continuously, checking if new Pysteps files are available to process.

## Setting up the Environment

An `environment.yml` file is included in the repository to make it easy to set up the required dependencies. The file defines the necessary Python packages and their versions, including:

- `xesmf`
- `pyproj`
- `pysteps`
- `xskillscore`
- `zarr`
- `esmpy`
- `setuptools`

To create and activate the environment:

```bash
# Create the environment from the environment.yml file
conda env create -f environment.yml

# Activate the newly created environment
conda activate xrValidation
```

Ensure all required packages are installed correctly by running the `setup.sh` script:

```bash
./setup.sh
```

### What the `setup.sh` Script Does:
1. **Verify** that all required packages are installed.
2. **Check** that `esmf.mk` is correctly located and set.
3. **Ensure** that `esmf` and `ESMPy` versions match.
4. **Clone or update** the FSSprob repository and compile the Fortran module.
5. **Install pFSS for Python**: This script ensures that the `pFSS` module (a key Fortran-based module for computing the fraction skill score) is installed and configured to work seamlessly with Python.

