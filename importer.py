import xarray as xr
import pyproj
import numpy as np
import pandas as pd
import time
import pysteps
from pysteps.io.importers import import_odim_hdf5 as importer
from pysteps.io.archive import find_by_date
from pysteps.io.readers import read_timeseries
import datetime as dt
import h5py

def test_xarray_dataset(ds):
    """
    Function to test the validity of an xarray dataset for specific requirements.

    Parameters:
    ds (xarray.Dataset): The dataset to be tested.

    Returns:
    bool: True if the dataset passes all checks, otherwise raises appropriate errors or warnings.
    """
    # Check if the object is an xarray Dataset
    if not isinstance(ds, xr.Dataset):
        raise TypeError("The provided object is not an xarray Dataset.")
    else:
        print("Dataset is a valid xarray Dataset.")

    required_dims = {'time', 'y', 'x'}

    # Check for required dimensions
    missing_dims = required_dims - set(ds.dims)
    if missing_dims:
        raise ValueError(f"Missing required dimensions: {missing_dims}")
    else:
        print(f"All required dimensions {required_dims} are present.")

    # Check for 'precipitation_rate' variable
    if 'precip_intensity' not in ds.data_vars:
        raise ValueError("The dataset does not contain the required data variable 'precipitation'.")
    else:
        print("'precipitation' variable is present in the dataset.")

    return True

# Read a particular (pySTEPS) netCDF file from a tarball
# and add some additional information to the Dataset
def read_netcdf(filename):
    """
    Reads a netCDF file.

    Parameters:
    filename (str): The name or path of the netCDF file.
    tarball (tarfile.TarFile, optional): The tarball object containing the file. If None, file is read directly from disk.

    Returns:
    xarray.Dataset: The processed dataset.
    """
    print("Reading netCDF-file...", end="", flush=True)
    t0 = time.time()

    # Get start date from filename
    start_date = filename.split("_")[-1].split(".")[0]

    # Open Dataset
    ds = xr.open_dataset(filename, engine='h5netcdf')

    # move lon and lat from data variables to coordinates and add start date
    ds = ds.assign_coords(
        lon=(("y", "x"), ds.lon.data),
        lat=(("y", "x"), ds.lat.data),
        forecast_reference_time=((), pd.to_datetime(start_date))
    )
    ds = ds.drop_vars("lcc")
    
    # add lead times to coordinates
    step = ds.time.data[1] - ds.time.data[0]
    steps = np.array([step + i * step for i in range(ds.sizes["time"])])
    ds = ds.assign_coords(leadtime=(("time"), steps))
    
#     # chunk the data for use with dask
#     ds = ds.chunk(dict(valid_time=1, ens_number=-1, y=-1, x=-1))
    print(f"done ({(time.time() - t0) / 60:.2f}m)")
    
    if test_xarray_dataset(ds):
        return ds
    else:
        raise ValueError("The dataset does not pass the required checks.")
   
def read_radqpe_h5(file_path):
    """Read and load a RADQPE HDF5 file, and returns the precipitation data and metadata.
    
    Args:
        file_path (str): Path to the RADQPE HDF5 file.
    
    Returns:
        data (np.ndarray): 2D array containing the RADQPE precipitation data.
        metadata (dict): Associated metadata of the RADQPE file.
    """
    metadata = {}
    
    with h5py.File(file_path, 'r') as hf:
        
        # Read location metadata
        g = hf["/dataset1/where"]

        metadata["x_step"] = g.attrs["xscale"]
        metadata["x_size"] = g.attrs["xsize"]
        metadata["x_1"] = g.attrs["UL_x"]
        metadata["x_2"] = metadata["x_1"] + metadata["x_size"] * metadata["x_step"]

        metadata["y_step"] = g.attrs["yscale"]
        metadata["y_size"] = g.attrs["ysize"]
        metadata["y_2"] = g.attrs["UL_y"]
        metadata["y_1"] = metadata["y_2"] - metadata["y_size"] * metadata["y_step"]

        metadata["proj4_str"] = hf["/where"].attrs["projdef"].decode("utf-8")
        
        # Read data
        data = hf["/dataset1/data1/data"][:].astype(np.float32)
        
    return data, metadata

# Read the RMI radar-QPE files given a DataArray of validdates
# The location and filename format of the radar files is taken from the pysteps.rcparams specifications
# validdates = pysteps.validtime
def read_radar(datetimes):
    """Read RADQPE precipitation rate data for the specified dates, and return them as xr.Dataset 
    with the associated metadata.
    
    Args:
        datetimes (xr.DataArray): Array of datetimes for the RADQPE data to be read.
    
    Returns:
        ds (xr.Dataset): Dataset of the RADQPE precipitation rate data, with the associated metadata.
    """
    t0 = time.time()
    print("Reading RADAR data...", end="", flush=True)
    # get the radar paths and filename formats
    rmi = pysteps.rcparams.data_sources.rmi
    # root_path = rmi.root_path
    root_path = "C:/Users/u0168535/.conda/envs/pysteps_dev/pystepsval/refactoring_arthur/test_data/radar"
    path_fmt = "" #rmi.path_fmt
    fn_pattern = "%Y%m%d%H%M00.rad.best.comp.rate.qpe_edk" #rmi.fn_pattern
    fn_ext = rmi.fn_ext
    importer_kwargs = rmi.importer_kwargs
    startdate = datetimes.isel(time=0).data
    enddate = datetimes.isel(time=len(datetimes) - 1).data
    # find the number of previous timesteps
    timestep = (datetimes.isel(time=1).data - startdate) / np.timedelta64(1, "m")
    nprev = int((enddate - startdate) / np.timedelta64(1, "m") / timestep)
    # build the filenames
    print(enddate)
    fns = find_by_date(
        dt.datetime.utcfromtimestamp(int(enddate) / 1e9),
        root_path,
        path_fmt,
        fn_pattern,
        fn_ext,
        timestep,
        num_prev_files=nprev
    )

    # read the data
    r, _, meta = read_timeseries(fns, importer, **importer_kwargs)
    
    # Load rates from hdf5 files
    vs = []
    for fn in fns[0]:
        try:
            v, _ = read_radqpe_h5(fn)
        except:
            v = np.full(v.shape, np.nan)
        vs.append(v)
    vs = np.array(vs)

    # convert to xarray
    x = np.linspace(meta['x1'], meta['x2'], r.shape[2] + 1)[:-1]
    x += 0.5 * (x[1] - x[0])
    y = np.linspace(meta["y2"], meta["y1"], r.shape[1] + 1)[:-1]
    y += 0.5 * (y[1] - y[0])
    x_2d, y_2d = np.meshgrid(x, y)
    pr = pyproj.Proj(meta['projection'])
    lon, lat = pr(x_2d.flatten(), y_2d.flatten(), inverse=True)
    lon = lon.reshape(r.shape[1], r.shape[2])
    lat = lat.reshape(r.shape[1], r.shape[2])

    short_name = 'precip_intensity'
    if meta['unit'] == 'mm/h':
        long_name = 'instantaneous precipitation rate'
        units = 'mm h-1'
    ds = xr.Dataset(
        # data variables
        data_vars={
            short_name: (
                ['time', 'y', 'x'],
                vs,
                {'long_name': long_name, 'units': units}
            )
        },
        # coordinates
        coords={
            'time': (
                ['time'],
                pd.to_datetime(meta['timestamps']),
            ),
            'x': (
                ['x'],
                x.round(),
                {
                    'axis': 'X',
                    'standard_name': 'projection_x_coordinate',
                    'long_name': 'x-coordinate in Cartesian system',
                    'units': 'm'
                }
            ),
            'y': (
                ['y'],
                y.round(),
                {
                    'axis': 'Y',
                    'standard_name': 'projection_y_coordinate',
                    'long_name': 'y-coordinate in Cartesian system',
                    'units': 'm'
                }
            ),
            'lon': (
                ['y', 'x'],
                lon,
                {
                    'standard_name': 'longitude',
                    'long_name': 'longitude coordinate',
                    'units': 'degrees_east'
                }
            ),
            'lat': (
                ['y', 'x'],
                lat,
                {
                    'standard_name': 'latitude',
                    'long_name': 'latitude coordinate',
                    'units': 'degrees_north'

                }
            )
        }
    )
#     # chunk for use with dask
#     ds = ds.chunk(dict(valid_time=1, y=-1, x=-1))
    print(f"done ({(time.time() - t0) / 60:.2f}m)")

    if test_xarray_dataset(ds):
        return ds
    else:
        raise ValueError("The dataset does not pass the required checks.")