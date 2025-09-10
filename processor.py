import xarray as xr
import numpy as np
import xskillscore as xs
from typing import List
import sys
import time

pystepsval_path = "C:/Users/u0168535/.conda/envs/pysteps_dev/pysteps_master/merge"
sys.path.append(pystepsval_path)
import importer

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


def compute_fss(NP_o, NP_f, fss_version='fss', dim=['y', 'x'], ens_dim='ens_number',
                window_size=11, noise_0=1e-10):
    """
    Compute the Fraction Skill Score (FSS) and its ensemble variants for spatial verification.

    Parameters
    ----------
    BP_o : xarray.DataArray
        Observed binary field (1 if event occurs, 0 otherwise). Dimensions should include `dim`.
    BP_f : xarray.DataArray
        Forecast binary field. For ensemble forecasts, includes dimension `ens_dim`.
    fss_version : str, optional
        Type of FSS to compute:
        - 'fss'    : Deterministic FSS (single forecast)
        - 'pfss'   : Probabilistic FSS (uses ensemble mean)
        - 'agfss'  : Member-aggregated FSS (pools all members)
        - 'avfss'  : Member-averaged FSS (mean of member FSS)
    dim : list, optional
        Spatial dimensions to aggregate over (default: ['y','x']).
    ens_dim : str, optional
        Ensemble member dimension name (default: 'ens_number').
    window_size : int, optional
        Size of the rolling window for neighborhood probability (default: 11).
    noise_0 : float, optional
        Small constant to avoid division by zero (default: 1e-10).

    Returns
    -------
    fss : xarray.DataArray
        Fraction Skill Score with dimensions:
        - For 'fss'/'pfss' : Same as input but without `dim`.
        - For 'agfss'       : Same as 'fss' but without `ens_dim`.
        - For 'avfss'       : Same as 'fss' but averaged over `ens_dim`.

    Notes
    -----
    - **FSS (Deterministic)**: Standard FSS comparing a single forecast to observations.
    - **pFSS**: Uses ensemble mean forecast to compute neighborhood probabilities.
    - **agFSS**: Aggregates FBS/WFBS across all members before computing FSS.
    - **avFSS**: Averages FSS values computed for each member individually.
    """
    if fss_version.lower() not in ['fss', 'pfss', 'agfss', 'avfss']:
        return "fss version should be 'fss', 'pfss', 'agfss' or 'avfss'"
    else:
        fss_version = fss_version.lower()
    
    # # Smooth observed and forecast fields
    # NP_o = BP_o.rolling(y=window_size, x=window_size, center=True).mean()
    # NP_f = BP_f.rolling(y=window_size, x=window_size, center=True).mean()

    # Probabilistic FSS: Use ensemble mean forecast
    if fss_version == 'pfss':
        NP_f = NP_f.mean(dim=ens_dim, keepdims=True)

    # Compute FBS and WFBS
    FBS = (NP_f - NP_o) ** 2
    WFBS = NP_f ** 2 + NP_o ** 2
    FBS_sum = FBS.sum(dim)
    WFBS_sum = WFBS.sum(dim)

    # Member-aggregated FSS: Pool FBS/WFBS across members
    if fss_version == 'agfss':
        FBS_sum = FBS_sum.sum(dim=ens_dim, keepdims=True)
        WFBS_sum = WFBS_sum.sum(dim=ens_dim, keepdims=True)

    # Compute FSS
    fss = 1 - (FBS_sum / (WFBS_sum + noise_0))

    # Member-averaged FSS: Average FSS over members
    if fss_version == 'avfss':
        fss = fss.mean(dim=ens_dim, keepdims=True)

    return fss


def apply_precip_masks(
        df_observation,
        df_model,
        thresholds=[0.1],
        window_sizes=[1],
        reduce_ensemble=None,
        conditioning='single',
        interpolation_method='nearest_s2d',
        extent_mask=None):
    
    # Computing the mask for the data
    if extent_mask is not None:
        df_observation_ext_masked = df_observation.where(extent_mask,0)
        df_model_ext_masked = df_model.where(extent_mask,0)
    else:
        df_observation_ext_masked = df_observation
        df_model_ext_masked = df_model
    
    # If the model data has an ensemble dimension, we need to chunk it to make it parallelizable
    df_obs_chunked = df_observation_ext_masked.chunk({'y': -1, 'x': -1, 'time': 1})
    df_model_chunked = df_model_ext_masked.chunk({'y': -1, 'x': -1, 'time': 1, 'ens_number': -1})
    
    # Threshold assignment for observations and forecasts
    df_obs_threshold_bool = [df_obs_chunked >= threshold for threshold in thresholds]
    df_obs_threshold_bool = [df_obs_threshold_bool[i].assign_coords({"threshold": ((), thresholds[i])}) for i in
                             range(len(thresholds))]
    df_obs_threshold_bool = xr.concat(df_obs_threshold_bool, "threshold").chunk({"threshold": -1})
    
    df_model_threshold_bool = [df_model_chunked >= threshold for threshold in thresholds]
    df_model_threshold_bool = [df_model_threshold_bool[i].assign_coords({"threshold": ((), thresholds[i])}) for i in
                               range(len(thresholds))]
    df_model_threshold_bool = xr.concat(df_model_threshold_bool, "threshold").chunk({"threshold": -1})
    df_model_threshold_bool_mean = df_model_threshold_bool.mean('ens_number')
    
    df_obs_threshold_bool_winsize = [
        df_obs_threshold_bool.rolling(
            y=window_sizes[i],
            x=window_sizes[i],
            center=True
            ).mean(
                ).assign_coords(
            {"window_size":((),window_sizes[i])}
            ) for i in range(len(window_sizes))
        ]
    df_obs_threshold_bool_winsize = xr.concat(df_obs_threshold_bool_winsize,"window_size").chunk({"window_size":-1})
    df_obs_threshold_bool_winsize = df_obs_threshold_bool_winsize.transpose('window_size', 'threshold', 'time', 'y', 'x')
    df_obs_threshold_bool_winsize = df_obs_threshold_bool_winsize > 0 #weakest constraint, if any is higher than 0 then 1
    
    df_model_threshold_bool_winsize = [
        df_model_threshold_bool.rolling(
            y=window_sizes[i],
            x=window_sizes[i],
            center=True
            ).mean(
                ).assign_coords(
            {"window_size":((),window_sizes[i])}
            ) for i in range(len(window_sizes))
        ]
    df_model_threshold_bool_winsize = xr.concat(df_model_threshold_bool_winsize,"window_size").chunk({"window_size":-1})
    df_model_threshold_bool_winsize = df_model_threshold_bool_winsize.transpose('window_size', 'threshold', 'ens_number', 'time', 'y', 'x')
    df_model_threshold_bool_winsize = df_model_threshold_bool_winsize > 0
    
    #Data when compared with thresholds
    df_obs_threshold = df_observation.where(df_obs_threshold_bool)
    df_obs_threshold = df_obs_threshold.transpose('threshold', 'time', 'y', 'x')
    df_obs_threshold_nan = np.isfinite(df_obs_threshold)
    
    df_model_threshold = df_model.where(df_model_threshold_bool)
    df_model_threshold = df_model_threshold.transpose('threshold', 'ens_number', 'time', 'y', 'x')
    df_model_threshold_nan = np.isfinite(df_model_threshold)
    
    # Mask depending on the conditioning for None, "single", "double"
    if conditioning is None:
        mask_threshold = df_obs_threshold_nan
    elif conditioning == "single":
        mask_threshold = df_obs_threshold_nan | df_model_threshold_nan
    elif conditioning == "double":
        mask_threshold = df_obs_threshold_nan & df_model_threshold_nan
    mask_threshold = mask_threshold.transpose('threshold', 'ens_number', 'time', 'y', 'x')
    
    #Make nan values 0
    df_obs_fill_zero = df_obs_threshold.fillna(0)
    df_model_fill_zero = df_model_threshold.fillna(0)
    
    #Get final values after threshold conditioning and mask values
    df_obs_masked = df_obs_fill_zero.where(mask_threshold, np.nan).chunk({"threshold":-1}).chunk({"time":-1})
    df_obs_masked = df_obs_masked.transpose('threshold', 'ens_number', 'time', 'y', 'x')
    df_model_masked = df_model_fill_zero.where(mask_threshold, np.nan).chunk({"threshold":-1}).chunk({"ens_number":-1}).chunk({"time":-1})
    df_model_masked = df_model_masked.transpose('threshold', 'ens_number', 'time', 'y', 'x')
    
    # Rechunk data to a single chunk along 'y' and 'x', enabling dask to parallelize the computation
    df_obs_masked = df_obs_masked.chunk({'y': -1, 'x': -1, 'time': 1, 'ens_number': 1})
    df_model_masked = df_model_masked.chunk({'y': -1, 'x': -1, 'time': 1, 'ens_number': 1})
    
    return df_obs_masked, df_model_masked, df_obs_threshold_bool_winsize, df_model_threshold_bool_winsize, df_model_threshold_bool_mean
    

def process_data_single_init_time(
        df_model: xr.Dataset,
        df_observation: xr.Dataset,
        thresholds: List[float],
        window_sizes: List[int],
        bin_edges: np.ndarray,
        bins_x: np.ndarray,
        reduce_ensemble=None,
        conditioning='single',
        interpolation_method='nearest_s2d',
        extent_mask=None,
        timing=False
):
    """Compute the scores of a Pysteps file based on its input radar precipitation data.
    Return the scores as xarray Dataset; some scores are not computed inside the function, 
    only the dask graph is defined and .compute() must be called on the returned Dataset to compute all scores.
    
    Args:
        df_model (xr.Dataset): Dataset with the Pysteps run.
        df_observation (xr.Dataset): Dataset with the input radar data used for the Pysteps run.
        thresholds (List[float]): Precipitation rate threshold for computing the scores.
        window_sizes (List[int]):
        bin_edges (np.ndarray): 
        bins_x (np.ndarray):
        interpolation_method (str):
        timing (Bool):
        
    Returns:
        ds_scores (xr.Dataset): Dataset with the scores of the Pysteps run.
    """
    # Start the timer
    start_time = time.time()

    # Validate input types and dimensions
    validate_inputs(df_model, df_observation, thresholds, bin_edges, bins_x,
                    window_sizes)
    
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
    
    [df_obs_masked, df_model_masked,
     df_obs_threshold_bool_winsize,
     df_model_threshold_bool_winsize,
     df_model_threshold_bool_mean] = apply_precip_masks(df_observation,
                                                        df_model,
                                                        thresholds=thresholds,
                                                        window_sizes=window_sizes,
                                                        reduce_ensemble=reduce_ensemble,
                                                        conditioning=conditioning,
                                                        interpolation_method=interpolation_method,
                                                        extent_mask=extent_mask)
    
    if timing:
        start_time = print_timing_and_reset('\t', 'prepocessing', start_time)
    print("\tDone with all preprocessing for runtime: " + str(init_time))
    
    # Deterministic scores
    # Note: The deterministic scores are calculated on the ensemble mean
    df_obs_masked_mean = df_obs_masked.mean(dim='ens_number')
    df_model_masked_mean = df_model_masked.mean(dim='ens_number')
    
    scores["RMSE_deterministic"] = xs.rmse(df_obs_masked_mean, df_model_masked_mean, ["y", "x"], skipna=True)
    scores["ME_deterministic"] = xs.me(df_obs_masked_mean, df_model_masked_mean, ["y", "x"], skipna=True)
    scores["MAPE_deterministic"] = xs.mape(df_obs_masked_mean, df_model_masked_mean, ["y", "x"], skipna=True)

    # Probabilistic scores
    if df_model_masked.sizes['ens_number'] > 1:
        scores["RMSE_probabilistic"] = xs.rmse(df_obs_masked, df_model_masked, ["y", "x"], skipna=True)
        scores["ME_probabilistic"] = xs.me(df_obs_masked, df_model_masked, ["y", "x"], skipna=True)
        scores["MAPE_probabilistic"] = xs.mape(df_obs_masked, df_model_masked, ["y", "x"], skipna=True)
        scores["Brier"] = xs.threshold_brier_score(df_obs_masked, df_model_masked.chunk({"ens_number":-1}), 
                                         thresholds, member_dim='ens_number',
                                         dim=['y', 'x'])
        scores["CRPS"] = xs.crps_ensemble(df_obs_masked, df_model_masked.chunk({"ens_number":-1}), 
                                                               member_dim='ens_number', dim=['y', 'x'])
        scores["RankHist"] = xs.rank_histogram(df_obs_masked.chunk({"time":1}), 
                                      df_model_masked.chunk({"ens_number":-1, "time":1}), 
                                      dim=['y', 'x'], 
                                      member_dim='ens_number')
        scores["Reliability"] = xs.reliability(df_obs_threshold_bool_winsize, df_model_threshold_bool_mean, 
                                     probability_bin_edges=bin_edges, dim=['y', 'x'])
        
    else:
        # If no ensemble dimension, set probabilistic scores to None
        scores["RMSE_probabilistic"] = None
        scores["ME_probabilistic"] = None
        scores["MAPE_probabilistic"] = None
        scores["Brier"] = None
        scores["CRPS"] = None
        scores["RankHist"] = None
        scores["Reliability"] = None

    # Probabilistic FSS
    scores["FSS_probabilistic"] = [
        compute_fss(
            df_obs_threshold_bool_winsize,
            df_model_threshold_bool_winsize,
            fss_version='pfss',
            dim=['y', 'x'],
            ens_dim='ens_number',
            window_size=winsize
            ).assign_coords(
                {"window_size":((), winsize)}
                ) for winsize in window_sizes
        ]
    
    # FSS per member
    # Determine if we have a single member or multiple ones
    if df_model_masked.sizes['ens_number'] > 1:
        
        scores["FSS_deterministic"] = [
            compute_fss(
                df_obs_threshold_bool_winsize,
                df_model_threshold_bool_winsize,
                fss_version='fss',
                dim=['y', 'x'],
                ens_dim='ens_number',
                window_size=winsize
                ).assign_coords(
                    {"window_size":((), winsize)}
                    ) for winsize in window_sizes
            ]
    else:
        scores["FSS_per_member"] = None

    # Histogram
    histogram = xr.apply_ufunc(histogram_only,
                               df_model_threshold_bool_mean,
                               kwargs={'bins': bin_edges},
                               input_core_dims=[['x', 'y']],
                               output_core_dims=[['bins_x']],
                               dask_gufunc_kwargs={'output_sizes': {'bins_x': bins_x_len}},
                               dask='parallelized',
                               vectorize=True)
    scores["Histogram"] = histogram.assign_coords(bins_x=bins_x)

    # Contingency-based metrics for each step
    contingency = xs.Contingency(
        df_obs_threshold_bool_winsize,
        df_obs_threshold_bool_winsize,
        observation_category_edges=np.array([0, 0.5, 1]),
        forecast_category_edges=np.array([0, 0.5, 1]),
        dim=['y', 'x']
    )
    scores["POD"] = contingency.hit_rate()
    scores["FAR"] = contingency.false_alarm_ratio()
    scores["ETS"] = contingency.equit_threat_score()
    scores["CSI"] = contingency.threat_score()

    # Build the Xarray dataset for all scores
    ds_scores = xr.Dataset(
        scores,
        coords={'forecast_reference_time': ('forecast_reference_time', [init_time]),
                'leadtime': ('leadtime', df_model_masked.leadtime.values)}
    )
    
    return ds_scores