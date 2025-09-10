# To debug segmentation fault
import faulthandler; faulthandler.enable()

import os
import shutil
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from numcodecs import Blosc
import time
import traceback
import logging
import sys

pystepsval_path = "C:/Users/u0168535/.conda/envs/pysteps_dev/pysteps_master/merge"
sys.path.append(pystepsval_path)
from importer import read_netcdf, read_radar
from processor import process_data_single_init_time


# Configure logging
log_dir = "C:/Users/u0168535/.conda/envs/pysteps_dev/pystepsval/refactoring_arthur/logs"
os.makedirs(log_dir, exist_ok=True)
log_fp = os.path.join(log_dir, "pysteps_scoring.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_fp),
        logging.StreamHandler(sys.stdout),  
    ],
)
logger = logging.getLogger(__name__)

traceback_log_dir = os.path.join(log_dir, "errors")
os.makedirs(traceback_log_dir, exist_ok=True)

# Paths definition
pysteps_dir = "C:/Users/u0168535/.conda/envs/pysteps_dev/pystepsval/refactoring_arthur/test_data/pysteps"
local_data_dir = "C:/Users/u0168535/.conda/envs/pysteps_dev/pystepsval/refactoring_arthur/test_data/local"
scores_dir = "C:/Users/u0168535/.conda/envs/pysteps_dev/pystepsval/refactoring_arthur/test_data/scores"
zarr_fp = os.path.join(scores_dir, "scores.zarr")

# Create directories if they don't exist
os.makedirs(local_data_dir, exist_ok=True)
os.makedirs(scores_dir, exist_ok=True)

# Scoring parameters
thresholds = [0.1,0.5,1.0,5.0]
bin_edges = np.linspace(-0.000001, 1.000001, 11)
bins_x = np.arange(0.05, 1, 0.1)
window_sizes = [1, 5, 11, 21]
conditioning = 'single'
extent_mask = None

def load_from_afd_pysteps(pysteps_fn, pysteps_dir, local_data_dir):
    """Copy a PySTEPS NetCDF file from a remote directory to a temporary local directory, 
    load it as an xarray Dataset, delete the temporary file, and return the xarray Dataset.
    
    Args:
        pysteps_fn (str): Filename of the PySTEPS NetCDF file to process.
        pysteps_dir (str): Path to the remote directory containing PySTEPS files.
        local_data_dir (str): Local directory for temporary copy of the file.
    
    Returns:
        xr.Dataset: Loaded dataset from the NetCDF file.
    """
    # copy/load/delete last pysteps nc file from afd
    pysteps_fp = os.path.join(pysteps_dir, pysteps_fn)
    local_nowcast_fp = os.path.join(local_data_dir, pysteps_fn)
    shutil.copyfile(pysteps_fp, local_nowcast_fp)
    
    pysteps_ds = read_netcdf(local_nowcast_fp)
    # os.remove(local_nowcast_fp)
    
    return pysteps_ds

def transform_dims(ds):
    """Transform dimensions and coordinates names of an xarray Dataset for consistency.
    Adjusts the dataset to use 'forecast_reference_time' as a dimension for the pysteps start time of 
    the run and renames 'valid_time' to 'leadtime'. Drops unnecessary coordinates like 'samples'.
    
    Args:
        ds (xr.Dataset): Input dataset with original dimensions.
    
    Returns:
        xr.Dataset: Transformed dataset with updated dimensions and coordinates.
    """
    # Add forecast_reference_time as new dim for all vars
    ds = ds.expand_dims({"time":ds.forecast_reference_time.values})
    ds = ds.drop_vars("forecast_reference_time")
    ds = ds.rename({"time":"forecast_reference_time"})
    
    # Drop the samples coordinates (from the reliability score)
    ds = ds.drop_vars("samples")
    
    # Replace valid_time dimension by leadtime for all vars and drop valid_time
    ds = ds.assign_coords(valid_time=ds.leadtime.values)
    ds = ds.drop_vars("leadtime")
    ds = ds.rename_dims({"valid_time":"leadtime"})
    ds = ds.rename_vars({"valid_time":"leadtime"})
    return ds
            
def create_zarr_dataset(zarr_fp, scores):
    """Create a new Zarr dataset of Pysteps scores on the disk, with optimized encoding and chunking.
    
    Args:
        zarr_fp (str): Path to the Zarr dataset to create.
        scores (xr.Dataset): Dataset containing score variables to store.
    """
    # Define encoding for Zarr file
    compressor = Blosc(cname="zstd", clevel=5)
    
    encoding = {}
    for data_var in scores.data_vars:
        encoding[data_var] = {"compressor": compressor}
        encoding[data_var]["chunks"] = (1, *(-1 for i in scores[data_var].shape[1:]))
        
        if "float" in str(scores[data_var].dtype):
            encoding[data_var]["dtype"] = "float32"
        if "int" in str(scores[data_var].dtype):
            encoding[data_var]["dtype"] = "int32"
        
    encoding["forecast_reference_time"] = {"dtype":"float64"}
    
    # Rechunk scores variables
    for data_var in scores.data_vars:
        var_chunking = {
            scores[data_var].dims[i]:encoding[data_var]["chunks"][i]
            for i in range(len(scores[data_var].dims))
        }
        scores[data_var] = scores[data_var].chunk(var_chunking)
    
    # Create Zarr file
    scores.to_zarr(zarr_fp, mode="w", encoding=encoding, consolidated=True)
    
def initialize_zarr_dataset(zarr_fp, pysteps_dir, local_data_dir):
    """Initialize the Zarr dataset on the disk by iterating through PySTEPS files until it successfully 
    computes scores for one, then creates the Zarr dataset.
    
    Args:
        zarr_fp (str): Path to the Zarr file.
        pysteps_dir (str): Directory containing PySTEPS files.
        local_data_dir (str): Local temporary directory.
    
    Raises:
        Exception: If no valid file were successfully processed to initialize the dataset.
    """
    pysteps_fns = sorted(os.listdir(pysteps_dir))
    
    for pysteps_fn in pysteps_fns:
        # pysteps_fn = pysteps_fns[0]
        try:
            model = load_from_afd_pysteps(pysteps_fn, pysteps_dir, local_data_dir)

            # Load radqpe files
            obs = read_radar(model.time)
            
            # Computing scores
            scores = process_data_single_init_time(
                model,#.load(),#.isel(valid_time=slice(0,10), ens_number=slice(0,4)), 
                obs,#.load(), 
                thresholds=thresholds, 
                bin_edges=bin_edges, 
                bins_x=bins_x, 
                window_sizes=window_sizes, 
                timing=True)
            
            scores = scores.compute()
            
            scores.to_netcdf('C:/Users/u0168535/.conda/envs/pysteps_dev/pystepsval/refactoring_arthur/test_data/scores/scores.nc')
            create_zarr_dataset(zarr_fp, scores)
            return
        
        except Exception as e:
            logging.error(f"Unexpected error while processing {pysteps_fn}: {e}")
            # Log the traceback in a separate file
            traceback_log_fp = os.path.join(traceback_log_dir, pysteps_fn.split(".")[0] + "_traceback.txt")
            with open(traceback_log_fp, 'w') as f:
                f.write(str(e))
                f.write(traceback.format_exc())
                continue
    # If here, failed to initialize the Zarr dataset
    e_str = "Failed to compute scores for a pysteps nowcast: Initialization of the Zarr dataset failed."
    logger.error(e_str)
    raise Exception(e_str)
        

def get_encoding_from_zarr(zarr_fp):
    """Retrieve the encoding configuration from an existing Zarr dataset.
    
    Args:
        zarr_fp (str): Path to the Zarr dataset.
    
    Returns:
        dict: Encoding settings for each data variable in the Zarr dataset.
    """
    scores_ds = xr.open_zarr(zarr_fp)
    encoding = {}
    for data_var in scores_ds.data_vars:
        encoding[data_var] = scores_ds[data_var].encoding
    return encoding

def start_date_from_pysteps_fn(pysteps_fn):
    """Extract the start date from a PySTEPS filename.

    Args:
        pysteps_fn (str): Filename (e.g., 'pysteps_202501011230.nc').
    
    Returns:
        str: Extracted date string (e.g., '202501011230').
    """
    return pysteps_fn.split("_")[-1].split(".")[0]

def process_pysteps_file(pysteps_fn, pysteps_dir, local_data_dir):
    """Compute the scores of a single PySTEPS file.
    
    Loads the model data and corresponding radar observations, computes scores, and returns them as a dataset.
    Handles cases where radar data is missing by returning an empty dataset with NaNs.
    
    Args:
        pysteps_fn (str): PySTEPS filename to process.
        pysteps_dir (str): Directory containing PySTEPS files.
        local_data_dir (str): Local temporary directory.
    
    Returns:
        xr.Dataset: Dataset containing computed scores, or an empty dataset if radar data is missing.
    
    Note:
        Logs warnings if radar data is not available.
    """
    start_date = start_date_from_pysteps_fn(pysteps_fn)
    print(f"Processing {start_date}")
    
    t0 = time.time()
    
    model = load_from_afd_pysteps(pysteps_fn, pysteps_dir, local_data_dir)
    
    try:
        obs = read_radar(model.valid_time)
    
    except:
        logging.warning(f"RADQPE file not available for {start_date}: Can not compute scores")
        # return empty scores with nans
        scores_ds = xr.open_zarr(zarr_fp)
        scores = scores_ds.isel(forecast_reference_time=[0]).where(False)
        frt = np.array([pd.to_datetime(start_date)], dtype='datetime64[ns]')
        scores = scores.assign_coords(forecast_reference_time=frt)
        return scores
    
    t1 = time.time()
    print(f"Elapsed time for data loading: {round(t1 - t0, 3)} s")

    # Compute the scores
    scores = process_data_single_init_time(
        model,#.load(),#.isel(valid_time=slice(0,10), ens_number=slice(0,4)), 
        obs,#.load(), 
        thresholds=thresholds, 
        bin_edges=bin_edges, 
        bins_x=bins_x,
        window_sizes=window_sizes, 
        conditioning=conditioning,
        timing=False)
    
    scores = transform_dims(scores)
    scores = scores.compute()
    
    t2 = time.time()
    logger.info(f"Total elapsed time for scores computation of {pysteps_fn}: {round(t2 - t1, 2)} s")
    
    return scores

def save_scores_to_zarr(scores, zarr_fp):
    """Append scores to an existing Zarr dataset along the forecast_reference_time dimension.
    Uses encoding from the existing Zarr file to ensure consistency. Rechunks the data before appending.
    
    Args:
        scores (xr.Dataset): Dataset to append.
        zarr_fp (str): Path to the Zarr file.
    """
    # Load encoding from the existing zarr archive
    encoding = get_encoding_from_zarr(zarr_fp)
    
    # Rechunk scores variables
    for data_var in scores.data_vars:
        var_chunking = {
            scores[data_var].dims[i]:encoding[data_var]["chunks"][i] 
            for i in range(len(scores[data_var].dims))
        }
        scores[data_var] = scores[data_var].chunk(var_chunking)
    
    scores.to_zarr(zarr_fp, mode="a", append_dim="forecast_reference_time", consolidated=True)
    print("Scores saved in zarr archive")

def check_scores_exist(pysteps_fn, zarr_fp):
    """Check if scores for a given PySTEPS file already exist in the Zarr dataset.
    
    Args:
        pysteps_fn (str): PySTEPS filename.
        zarr_fp (str): Path to the Zarr file.
    
    Returns:
        bool: True if scores exist, False otherwise.
    """
    scores_ds = xr.open_zarr(zarr_fp)
    start_date = start_date_from_pysteps_fn(pysteps_fn)
    
    if pd.to_datetime(start_date) in scores_ds.forecast_reference_time:
        return True
    else:
        return False
    
def check_pysteps_obs_time_diff(pysteps_fn, time_diff="7hours"):
    """Check if a PySTEPS run is too recent to have corresponding radar data available to compute the scores.

    Args:
        pysteps_fn (str): PySTEPS filename.
        time_diff (str, optional): Time delta string (e.g., '7hours'). Defaults to "7hours".
    
    Returns:
        bool: True if the run is older than time_diff, False otherwise.
    """
    start_date = start_date_from_pysteps_fn(pysteps_fn)
    diff_time = pd.Timestamp.utcnow().tz_localize(None) - pd.to_datetime(start_date)
    if diff_time < pd.Timedelta(time_diff):
        return True
    else:
        return False
    
# The filtering could be vectorized for optimization
def get_pysteps_fns(pysteps_dir, zarr_fp, min_age_hours=7):
    """Get a list of PySTEPS files that are unprocessed and sufficiently old.
    Filters out files that have already been processed or are too recent for radar data availability.
    
    Args:
        pysteps_dir (str): Directory containing PySTEPS files.
        zarr_fp (str): Path to the Zarr file to check for existing scores.
        min_age_hours (int, optional): Minimum age in hours for a file to be processed. Defaults to 7.
    
    Returns:
        filtered_fns (list): Sorted list of filenames that needs to be processed.
    """
    pysteps_fns = os.listdir(pysteps_dir)
    pysteps_fns = sorted(pysteps_fns)
    
    filtered_fns = []
   
    for pysteps_fn in pysteps_fns:
        # Check if scores already exist
        if not check_scores_exist(pysteps_fn, zarr_fp):
            # Check if the file is not too recent
            if not check_pysteps_obs_time_diff(pysteps_fn):
                filtered_fns.append(pysteps_fn)
                
    return filtered_fns


# Ininitialize the Zarr dataset if not existing
if not os.path.exists(zarr_fp):
    logger.info("Zarr dataset not existing: computing scores before initializing the Zarr dataset")
    initialize_zarr_dataset(zarr_fp, pysteps_dir, local_data_dir)
    logger.info("Zarr dataset initialized successfully")
else:
    logger.info("Zarr dataset already exists. Skipping initialization for the Zarr dataset.")
    
# Process continuously pysteps files, compute the scores and append them in the Zarr dataset
try:
    # while True:
    
    pysteps_fns = get_pysteps_fns(pysteps_dir, zarr_fp)
    
    # Compute and save scores for all unprocessed pysteps files
    for pysteps_fn in pysteps_fns:
        
        try:
            # Compute scores and store them in zarr file
            scores = process_pysteps_file(pysteps_fn, pysteps_dir, local_data_dir)
            save_scores_to_zarr(scores, zarr_fp)
        
        except Exception as e:
            
            # Log the exception
            logging.error(f"Unexpected error while processing {pysteps_fn}: {e}")
            
            # Print traceback
            traceback.print_exc()
            
            # Log the traceback in a separate file
            traceback_log_fp = os.path.join(traceback_log_dir, pysteps_fn.split(".")[0] + "_traceback.txt")
            with open(traceback_log_fp, 'w') as f:
                f.write(str(e))
                f.write(traceback.format_exc())
            
            continue
            
    # time.sleep(1)

except KeyboardInterrupt:
    logging.error("Script stopped by user")