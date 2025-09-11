import xarray as xr
import numpy as np
import xskillscore as xs
import xesmf as xe
from typing import List
import sys
import os
import time
from dask import delayed
import importer

# Add the ./FSSprob folder to the Python path
fss_path = os.path.join("/home/armoraux/Python/pysteps_eval/github/pystepsval", 'FSSprob')
sys.path.append(fss_path)
from fss90 import mod_fss

def validate_inputs(
        df_observation: xr.Dataset,
        df_model: xr.Dataset,
        thresholds: List[float],
        bin_edges: np.ndarray,
        bins_x: np.ndarray,
        fss_window_sizes: List[int]
):
    """
    Validate the types and dimensions of the inputs for the model and observation data.

    Parameters:
    - df_model (xarray.Dataset): The model dataset containing precipitation forecasts.
    - df_observation (xarray.Dataset): The observation dataset containing observed precipitation data.
    - thresholds (List[float]): List of thresholds to be used for binary events.
    - bin_edges (np.ndarray): Array of bin edges used for histograms and reliability diagrams.
    - bins_x (np.ndarray): Array of bin centers for probability bins.
    - fss_window_sizes (List[int]): List of window sizes for calculating the Fraction Skill Score (FSS).

    Raises:
    - TypeError: If any of the input parameters do not match the expected type or dimensions.
    """
    # Check if df_model and df_observation are xarray Datasets
    if not isinstance(df_model, xr.Dataset):
        raise TypeError("df_model must be an xarray Dataset")
    if not isinstance(df_observation, xr.Dataset):
        raise TypeError("df_observation must be an xarray Dataset")

    # Check thresholds is a list of floats
    if not isinstance(thresholds, list) or not all(isinstance(th, float) for th in thresholds):
        raise TypeError("thresholds must be a list of floats")

    # Check bin_edges and bins_x are numpy arrays
    if not isinstance(bin_edges, np.ndarray):
        raise TypeError("bin_edges must be a numpy array")
    if not isinstance(bins_x, np.ndarray):
        raise TypeError("bins_x must be a numpy array")

    # Check fss_window_sizes is a list of integers >= 1
    if not isinstance(fss_window_sizes, list) or not all(
            isinstance(w, int) and w >= 1 for w in fss_window_sizes):
        raise TypeError("fss_window_sizes must be a list of integers, each >= 1")


def create_cached_projector(type_of_data, source_data, projection_source, interpolation_method):
    """
    Create and cache a projector for spatial alignment between model and observation data.

    Parameters:
    - type_of_data (str): Type of data to be regridded (e.g., 'model').
    - source_data (xarray.Dataset): The source dataset to be regridded.
    - projection_source (xarray.Dataset): The target projection grid.
    - interpolation_method (str): The method to use for interpolation (e.g., 'nearest_s2d').

    Returns:
    - xesmf.Regridder: A cached projector object for spatial alignment.
    """
    # Initialize the projector cache if not already done
    if not hasattr(create_cached_projector, 'projector_cache'):
        create_cached_projector.projector_cache = {}

    # Check if the projector for the given type already exists in the cache
    if type_of_data in create_cached_projector.projector_cache:
        return create_cached_projector.projector_cache[type_of_data]
    else:
        projector = xe.Regridder(source_data, projection_source, interpolation_method)
        create_cached_projector.projector_cache[type_of_data] = projector
        return projector


def spatial_alignment(df_model, df_observation, interpolation_method):
    """
    Align the model data to the observation grid using a cached projector.

    Parameters:
    - df_model (xarray.Dataset): The model dataset to be aligned.
    - df_observation (xarray.Dataset): The observation dataset providing the target grid.

    Returns:
    - xarray.Dataset: The reprojected model data aligned to the observation grid.
    - xarray.Dataset: The original observation data.
    """
    df_source = df_model
    df_target = df_observation
    # Project from model to observation grid
    projector = create_cached_projector('model', df_source, df_target, interpolation_method)
    reprojected_data = projector(df_source, keep_attrs=True)
    reprojected_data = reprojected_data.chunk(dict(x=-1, y=-1))
    return reprojected_data, df_observation


def histogram_only(data, bins):
    """
    Compute a histogram of the provided data using the specified bins.

    Parameters:
    - data (np.ndarray): Input data array for which the histogram is to be calculated.
    - bins (np.ndarray): Array of bin edges used to compute the histogram.

    Returns:
    - np.ndarray: Histogram counts for each bin.
    """
    # Ensure data is finite and within the range of bins
    data = data[np.isfinite(data)]
    data = data[(data >= bins.min()) & (data <= bins.max())]
    return np.histogram(data, bins=bins)[0]


# Function to calculate scores for each lead time
def print_timing_and_reset(prefix, label, start_time):
    current_time = time.time()
    time_diff = current_time - start_time
    print('\033[1m' + prefix + f"Elapsed time for {label}: {time_diff:.4f} seconds" + '\033[0m')
    return time.time()


def process_data_single_init_time(
        df_model: xr.Dataset,
        df_observation: xr.Dataset,
        thresholds: List[float],
        bin_edges: np.ndarray,
        bins_x: np.ndarray,
        fss_window_sizes: List[int],
        interpolation_method='nearest_s2d',
        timing=False
):
    """Compute the scores of a Pysteps file based on its input radar precipitation data.
    Return the scores as xarray Dataset; some scores are not computed inside the function, 
    only the dask graph is defined and .compute() must be called on the returned Dataset to compute all scores.
    
    Args:
        df_model (xr.Dataset): Dataset with the Pysteps run.
        df_observation (xr.Dataset): Dataset with the input radar data used for the Pysteps run.
        thresholds (List[float]): Precipitation rate threshold for computing the scores.
        bin_edges (np.ndarray): 
        bins_x (np.ndarray):
        fss_window_sizes (List[int]):
        interpolation_method (str):
        timing (Bool):
        
    Returns:
        ds_scores (xr.Dataset): Dataset with the scores of the Pysteps run.
    """
    # Start the timer
    start_time = time.time()

    # Validate input types and dimensions
    validate_inputs(df_model, df_observation, thresholds, bin_edges, bins_x,
                    fss_window_sizes)

    # Some checks that need to be done before the code can be run
    try:
        importer.test_xarray_dataset(df_observation)
        importer.test_xarray_dataset(df_model)
    except Exception as e:
        raise ValueError("The inputted data for single_init_time "
                         "does not match the expected naming conventions. Please read the README."
                          + str(e))

    print("\tAll checks are done for runtime " + str(
        df_model["forecast_reference_time"].values) + ", provided data conforms to expectations")

    # Define validation parameters
    bins_x_len = len(bins_x)
    init_time = df_model["forecast_reference_time"].values

    # Initialize scores dictionary
    scores = {}

    # TODO: This set to nans is placed in the correct version
    # Replace NaN values with zero before calculation
    df_model_subselection_no_nan = df_model.fillna(0)
    df_observation_subselection_no_nan = df_observation.fillna(0)

    # Align the observations and model spatially. This calls a projector and re-gridder
    df_model_aligned, df_observation_aligned = spatial_alignment(df_model_subselection_no_nan,
                                                                 df_observation_subselection_no_nan,
                                                                 interpolation_method)

    # If the model data has an ensemble dimension, we need to chunk it to make it parallelizable
    df_model_chunked = df_model_aligned.chunk({'y': -1, 'x': -1, 'time': 1, 'ens_number': -1})

    # Computing the mask for the data
    # TODO: mask should be computed before the setting to nans
    data_mask = ((np.isfinite(df_observation_aligned['precip_intensity'])) & (
        np.isfinite(df_model_chunked['precip_intensity'].mean('ens_number')))).chunk({"time": 1})

    # Mask non-finite values
    df_obs_masked = df_observation_aligned['precip_intensity'].where(data_mask).chunk({"time": 1})
    df_model_masked = df_model_chunked['precip_intensity'].where(data_mask).chunk({"time": 1, "ens_number": -1})

    # Threshold assignment for observations and forecasts
    df_obs_threshold_bool = [df_obs_masked >= threshold for threshold in thresholds]
    df_obs_threshold_bool = [df_obs_threshold_bool[i].assign_coords({"threshold": ((), thresholds[i])}) for i in
                             range(len(thresholds))]
    df_obs_threshold_bool = xr.concat(df_obs_threshold_bool, "threshold").chunk({"threshold": -1})

    df_model_threshold_bool = [df_model_masked >= threshold for threshold in thresholds]
    df_model_threshold_bool = [df_model_threshold_bool[i].assign_coords({"threshold": ((), thresholds[i])}) for i in
                               range(len(thresholds))]
    df_model_threshold_bool = xr.concat(df_model_threshold_bool, "threshold").chunk({"threshold": -1})

    # Rechunk data to a single chunk along 'y' and 'x', enabling dask to parallelize the computation
    df_obs_masked = df_obs_masked.chunk({'y': -1, 'x': -1, 'time': 1})
    df_model_masked = df_model_masked.chunk({'y': -1, 'x': -1, 'time': 1, 'ens_number': 1})

    if timing:
        start_time = print_timing_and_reset('\t', 'prepocessing', start_time)
    print("\tDone with all preprocessing for runtime: " + str(init_time))

#     # Deterministic scores
#     # Note: The deterministic scores are calculated on the ensemble mean
#     df_model_masked_mean = df_model_masked.mean(dim='ens_number')

#     scores["RMSE_deterministic"] = xs.rmse(df_obs_masked, df_model_masked_mean, ["y", "x"], skipna=True)
#     scores["ME_deterministic"] = xs.me(df_obs_masked, df_model_masked_mean, ["y", "x"], skipna=True)
#     scores["MAPE_deterministic"] = xs.mape(df_obs_masked, df_model_masked_mean, ["y", "x"], skipna=True)

#     # Probabilistic scores
#     if df_model_masked.sizes['ens_number'] > 1:
#         scores["RMSE_probabilistic"] = xs.rmse(df_obs_masked, df_model_masked, ["y", "x"], skipna=True)
#         scores["ME_probabilistic"] = xs.me(df_obs_masked, df_model_masked, ["y", "x"], skipna=True)
#         scores["MAPE_probabilistic"] = xs.mape(df_obs_masked, df_model_masked, ["y", "x"], skipna=True)
#         scores["Brier"] = xs.threshold_brier_score(df_obs_masked, df_model_masked.chunk({"ens_number":-1}), 
#                                          thresholds, member_dim='ens_number',
#                                          dim=['y', 'x'])
#         scores["CRPS"] = xs.crps_ensemble(df_obs_masked, df_model_masked.chunk({"ens_number":-1}), 
#                                                                member_dim='ens_number', dim=['y', 'x'])
#         scores["RankHist"] = xs.rank_histogram(df_obs_masked.chunk({"time":1}), 
#                                       df_model_masked.chunk({"ens_number":-1, "time":1}), 
#                                       dim=['y', 'x'], 
#                                       member_dim='ens_number')
#         scores["Reliability"] = xs.reliability(df_obs_threshold_bool, df_model_threshold_bool.mean(dim='ens_number'), 
#                                      probability_bin_edges=bin_edges, dim=['y', 'x'])

#     else:
#         # If no ensemble dimension, set probabilistic scores to None
#         scores["RMSE_probabilistic"] = None
#         scores["ME_probabilistic"] = None
#         scores["MAPE_probabilistic"] = None
#         scores["Brier"] = None
#         scores["CRPS"] = None
#         scores["RankHist"] = None
#         scores["Reliability"] = None

    # Probabilistic FSS
    fractional_skill_scores_prob = []
    threshold_np = np.array(thresholds)
    kernel_np = np.array(fss_window_sizes)

    df_obs_masked = df_obs_masked.chunk({"time":1})
    df_model_masked = df_model_masked.chunk({"ens_number":-1, "time":1})

    for valtime in df_model_masked.time.values:

        fss = delayed(mod_fss.fss_prob)(fcst=df_model_masked.sel(time=valtime), 
                                        obs=df_obs_masked.sel(time=valtime), thrsh=threshold_np, kernel=kernel_np)
        fractional_skill_scores_prob.append(fss)

    results = delayed(np.stack)(fractional_skill_scores_prob)

    # Convert list of FSS into a DataArray with time and thresholds as coordinates
    scores["FSS_probabilistic"] = delayed(xr.DataArray)(
        results,
        dims=["time", "threshold", "window_size"],
        coords={
            "time": df_model_masked.time.values,
            "threshold": thresholds,
            "window_size": fss_window_sizes
        }
    )

    # FSS per member
    # Determine if we have a single member or multiple ones
    if df_model_masked.sizes['ens_number'] > 1:

        @delayed
        def fss_det_dasked(df_model, df_obs, thrsh=threshold_np, kernel=kernel_np):
            scores = mod_fss.fss_det(
                fcst=df_model.transpose("y", "x").values,
                obs=df_obs.transpose("y", "x").values, 
                thrsh=thrsh,
                kernel=kernel)
            return scores

        df_obs_masked = df_obs_masked.chunk({"time":1})
        df_model_masked = df_model_masked.chunk({"ens_number":1, "time":1})

        tasks = []
        for i in range(len(df_model_masked.ens_number)):
            for j in range(len(df_model_masked.time)):
                tasks.append(fss_det_dasked(df_model_masked.isel(ens_number=i, time=j), df_obs_masked.isel(time=j)))

        results = delayed(np.array)(tasks)
        results = delayed(np.reshape)(results, (len(df_model_masked.ens_number), 
                                                len(df_model_masked.time), 
                                                len(threshold_np), len(kernel_np)))

        scores["FSS_per_member"] = delayed(xr.DataArray)(
            results,
            dims=["ens_number", "time", "threshold", "window_size"],
            coords={
                "ens_number": df_model_masked.ens_number.values,
                "time": df_model_masked.time.values,
                "threshold": thresholds,
                "window_size": fss_window_sizes
            }
        )

    else:
        scores["FSS_per_member"] = None

    # Histogram
#     df_model_masked_ens_mean = df_model_threshold_bool.mean('ens_number')
#     df_model_masked_ens_mean = df_model_masked_ens_mean.chunk({'y': -1, 'x': -1})

#     histogram = xr.apply_ufunc(histogram_only,
#                                df_model_masked_ens_mean,
#                                kwargs={'bins': bin_edges},
#                                input_core_dims=[['x', 'y']],
#                                output_core_dims=[['bins_x']],
#                                dask_gufunc_kwargs={'output_sizes': {'bins_x': bins_x_len}},
#                                dask='parallelized',
#                                vectorize=True)

#     scores["Histogram"] = histogram.assign_coords(bins_x=bins_x)

#     # Contingency-based metrics for each step
#     contingency = xs.Contingency(
#         df_obs_threshold_bool,
#         df_model_threshold_bool,
#         observation_category_edges=np.array([0, 0.5, 1]),
#         forecast_category_edges=np.array([0, 0.5, 1]),
#         dim=['y', 'x']
#     )

#     scores["POD"] = contingency.hit_rate()
#     scores["FAR"] = contingency.false_alarm_ratio()
#     scores["ETS"] = contingency.equit_threat_score()
#     scores["CSI"] = contingency.threat_score()

    # Build the Xarray dataset for all scores
    ds_scores = delayed(xr.Dataset)(
        scores,
        coords={'forecast_reference_time': ('forecast_reference_time', [init_time]),
                'leadtime': ('leadtime', df_model_masked.leadtime.values)}
    )
    
    return ds_scores