#!/usr/bin/env python
"""Use a processed IFS output file and generate one ecRad input file for each time step

**Required User Input:**

* step: at which intervals should the IFS data be interpolated on the aircraft data (default: 2s from :ref:`processing:ecrad_read_ifs.py`)

**Output:**

* well documented ecRad input file in netCDF format for each time step

*author*: Johannes Röttenbacher
"""
# %% module import
import pylim.helpers as h
import pylim.solar_position as sp
from pylim.ecrad import apply_ice_effective_radius, apply_liquid_effective_radius
import numpy as np
import xarray as xr
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import time
from distutils.util import strtobool

start = time.time()


# %% read in command line arguments
args = h.read_command_line_args()
# set interpolate flag
t_interp = strtobool(args["t_interp"]) if "t_interp" in args else False  # interpolate between timesteps?
date = args["date"] if "date" in args else "20210629"
init_time = args["init"] if "init" in args else "00"
flight = args["flight"] if "flight" in args else "Flight_20210629a"
aircraft = args["aircraft"] if "aircraft" in args else "halo"
campaign = args["campaign"] if "campaign" in args else "cirrus-hl"
dt_day = datetime.strptime(date, '%Y%m%d')  # convert date to date time for further use
flight_key = flight[-4:] if campaign == "halo-ac3" else flight
# setup logging
try:
    file = __file__
except NameError:
    file = None
log = h.setup_logging("./logs", file, flight_key)
# print options to user
log.info(f"Options set: \ncampaign: {campaign}\naircraft: {aircraft}\nflight: {flight}\ndate: {date}"
         f"\ninit time: {init_time}Z\nt_interp: {t_interp}")

# %% set paths
path_ifs_output = os.path.join(h.get_path(campaign, "ifs"), date)
path_ecrad = os.path.join(h.get_path(campaign, "ecrad"), date, "ecrad_input")
# create output path
h.make_dir(path_ecrad)

# %% read in intermediate files from read_ifs
if init_time == "yesterday":
    ifs_date = int(date) - 1
    init_time = 12
else:
    ifs_date = date

nav_data_ip = pd.read_csv(f"{path_ifs_output}/nav_data_ip_{date}.csv", index_col="time", parse_dates=True)
data_ml = xr.open_dataset(f"{path_ifs_output}/ifs_{ifs_date}_{init_time}_ml_processed.nc")

# %% subsample 2s nav_data by step
step = 1
nav_data_ip = nav_data_ip[::step]

# %% loop through time steps and write one file per time step
idx = len(nav_data_ip)
dt_nav_data = nav_data_ip.index.to_pydatetime()

for i in tqdm(range(0, idx), desc="Time loop"):
    # select a 3 x 11 lat/lon grid around closest grid point
    lat_id = h.arg_nearest(data_ml.lat, nav_data_ip.lat.iat[i])
    lon_id = h.arg_nearest(data_ml.lon, nav_data_ip.lon.iat[i])
    lat_circle = np.arange(lat_id-1, lat_id+2)
    lon_circle = np.arange(lon_id-5, lon_id+6)
    ds_sel = data_ml.isel(lat=lat_circle, lon=lon_circle)

    if t_interp:
        dsi_ml_out = ds_sel.interp(time=dt_nav_data[i])  # interpolate to time step
    else:
        dsi_ml_out = ds_sel.sel(time=dt_nav_data[i], method="nearest")  # select closest time step

    # add cos_sza for each grid point using only model data
    cos_sza = np.empty((len(lat_circle), len(lon_circle)))
    sza = np.empty((len(lat_circle), len(lon_circle)))
    sod = nav_data_ip.seconds.iloc[i]
    for lat_idx in range(cos_sza.shape[0]):
        for lon_idx in range(cos_sza.shape[1]):
            p_surf_nearest = dsi_ml_out.pressure_hl.isel(lat=lat_idx, lon=lon_idx,
                                                         half_level=137).values / 100  # hPa
            t_surf_nearest = dsi_ml_out.temperature_hl.isel(lat=lat_idx, lon=lon_idx,
                                                            half_level=137).values - 273.15  # degree Celsius
            ypos = dsi_ml_out.lat.isel(lat=lat_idx).values
            xpos = dsi_ml_out.lon.isel(lon=lon_idx).values
            sza[lat_idx, lon_idx] = sp.get_sza(sod / 3600, ypos, xpos, dt_day.year, dt_day.month, dt_day.day,
                                               p_surf_nearest, t_surf_nearest)
            cos_sza[lat_idx, lon_idx] = np.cos(sza[lat_idx, lon_idx] / 180. * np.pi)

    dsi_ml_out["cos_solar_zenith_angle"] = xr.DataArray(cos_sza,
                                                        dims=["lat", "lon"],
                                                        attrs=dict(unit="1",
                                                                   long_name="Cosine of the solar zenith angle"))

    # calculate effective radius for all levels
    dsi_ml_out = apply_ice_effective_radius(dsi_ml_out)
    dsi_ml_out = apply_liquid_effective_radius(dsi_ml_out)
    # drop dimension column by selecting it, then stack lat, lon to a multi index named column, reset the index,
    # turn lat, lon, time into variables for cleaner output and to avoid later problems when merging data
    # this turns the two dimensions lat lon into one new dimension column with which ecrad can work
    dsi_ml_out = dsi_ml_out.sel(column=0).stack(column=("lat", "lon")).reset_index("column").reset_coords(
        ["lat", "lon", "time"])
    # some variables now need to have the dimension column as well
    variables = ["overlap_param", "fractional_std", "inv_cloud_effective_size"]
    n_column = dsi_ml_out.dims["column"]
    for var in variables:
        arr = dsi_ml_out[var].values
        # add new dimension to 1D variable and repeat the array times len(column)
        # then concatenate all array along the new dimension
        # check the shape of the input array and adjust the axis argument accordingly
        axis = 0 if arr.ndim == 0 else 1
        # define the dims accordingly
        dims = ["column"] if axis == 0 else [dsi_ml_out[var].dims[0], "column"]
        arr2D = np.concatenate([arr[..., np.newaxis]] * n_column, axis=axis)
        dsi_ml_out[var] = xr.DataArray(arr2D, dims=dims, attrs=dsi_ml_out[var].attrs)
    dsi_ml_out = dsi_ml_out.transpose("column", ...)
    dsi_ml_out = dsi_ml_out.astype(np.float32)  # change type from double to float32

    # ds_ml_out.to_netcdf(path=f"{path_ecrad}/ecrad_input_standard_{nav_data_ip.time.iloc[i]:7.1f}_sod_inp.nc4",
    #                     format='NETCDF4')
    dsi_ml_out.to_netcdf(path=f"{path_ecrad}/ecrad_input_standard_{nav_data_ip.seconds.iloc[i]:7.1f}_sod_inp.nc",
                         format='NETCDF4_CLASSIC')
    # dsi_ml_out.to_netcdf(path=f"{path_ecrad}/{date}/ecrad_input_standard_{nav_data_ip.time.iloc[i]:7.1f}_sod_inp.nc",
    #                      format='NETCDF3_CLASSIC')

log.info(f"Done with date {date}: {h.seconds_to_fstring(time.time() - start)} [hr:min:sec]")
