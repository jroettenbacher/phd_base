#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 10.10.2023

Select closest points of CAMS global reanalysis and global greenhouse gas reanalysis data to flight track.
Read in CAMS from different sources (ADS, MARS online, MARS ECMWF, Copernicus Knowledge Base).

Required User Input:

- source (ADS, 47r1, MARS, ECMWF)
- year (2019, 2020)

Input:

- monthly mean CAMS data (either monthly mean or monthly mean by hour of day)

Output:

- trace gas and aerosol monthly climatology interpolated to flight day along flight track

"""

# %% import modules
import pylim.helpers as h
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.neighbors import BallTree
from datetime import datetime

# %% set source and paths
campaign = "halo-ac3"
source = "47r1"
year = "2020"
date = "20220411"

cams_path = h.get_path("cams_raw", campaign=campaign)
cams_output_path = h.get_path("cams", campaign=campaign)
# IFS path for flight track file
ifs_path = h.get_path("ifs", campaign=campaign)
if source == "ADS":
    aerosol_file = f"cams_eac4_global_reanalysis_mm_{year}_pl.nc"
    trace_gas_file = f"cams_global_ghg_reanalysis_mm_{year}_pl.nc"
elif source == "47r1":
    aerosol_file = "aerosol_cams_3d_climatology_47r1.nc"
    trace_gas_file = "greenhouse_gas_climatology_46r1.nc"

# %% read in data
nav_data_ip = pd.read_csv(f"{ifs_path}/{date}/nav_data_ip_{date}.csv", index_col="time", parse_dates=True)
aerosol = xr.open_dataset(f"{cams_path}/{aerosol_file}")
trace_gas = xr.open_dataset(f"{cams_path}/{trace_gas_file}")

# %% calculate pressure at full model level for aerosol file
# add half the difference between the pressure at the base and top of the layer to the pressure at the base of the layer
aerosol["full_level_pressure"] = (aerosol.half_level_pressure
                                  + 0.5 * aerosol.half_level_delta_pressure)

# %% linearly interpolate in time and rename dimensions
new_time_axis = pd.date_range(f"{date[0:4]}-01-15", f"{date[0:4]}-12-15", freq=pd.offsets.SemiMonthBegin(2))
date_dt = pd.to_datetime(date)
if source == "ADS":
    aerosol = (aerosol
               .assign_coords(time=new_time_axis)
               .interp(time=date_dt)
               .rename({"latitude": "lat", "longitude": "lon"}))
    trace_gas = (trace_gas
                 .assign_coords(time=new_time_axis)
                 .interp(time=date_dt)
                 .rename({"latitude": "lat", "longitude": "lon"}))
elif source == "47r1":
    aerosol = (aerosol
               .assign_coords(month=new_time_axis)
               .interp(month=date_dt)
               .rename(month="time"))
    trace_gas = (trace_gas
                 .assign_coords(month=new_time_axis)
                 .interp(month=date_dt)
                 .rename(latitude="lat", month="time"))

# %% create array of aircraft locations
points = np.deg2rad(
    np.column_stack(
        (nav_data_ip.lat.to_numpy(), nav_data_ip.lon.to_numpy())))

# %% select closest points along flight track from aerosol data
aerosol_latlon = aerosol.stack(latlon=["lat", "lon"])  # combine lat and lon into one dimension
# make an array with all lat lon combinations
aerosol_lat_lon = np.array([np.array(element) for element in aerosol_latlon["latlon"].to_numpy()])
# build the look up tree
aerosol_tree = BallTree(np.deg2rad(aerosol_lat_lon), metric="haversine")
# query the tree for the closest CAMS grid points to the flight track
dist, idx = aerosol_tree.query(points, k=1)
# select only the closest grid points along the flight track
aerosol_sel = aerosol_latlon.isel(latlon=idx.flatten())
# reset index and make lat, lon and time (month) a variable
aerosol_sel = (aerosol_sel
               .reset_index(["latlon", "lat", "lon"])
               .reset_coords(["lat", "lon", "time"])
               .drop_vars(["time", "lat", "lon"])
               .rename(latlon="time", lev="level")
               .assign(time=nav_data_ip.index.to_numpy()))  # replace latlon with time as a dimension/coordinate

# %% select zonal mean closest to flight track from greenhouse gas data
trace_gas_sel = (trace_gas
                 .sel(lat=nav_data_ip.lat.to_numpy(), method="nearest")
                 .drop_vars("time")
                 .rename(lat="time")
                 .assign(time=nav_data_ip.index.to_numpy())
                 .assign(level=trace_gas.pressure))

# %% save files to netcdf
history_str = (f"\n{datetime.today().strftime('%c')}: "
               f"formatted file to serve as input to ecRad"
               f" using ecrad_cams_preprocessing.py")
try:
    aerosol_sel.attrs["history"] = aerosol_sel.attrs["history"] + history_str
    trace_gas_sel.attrs["history"] = trace_gas_sel.attrs["history"] + history_str
except KeyError:
    aerosol_sel.attrs["history"] = history_str
    trace_gas_sel.attrs["history"] = history_str

aerosol_sel.to_netcdf(f"{cams_output_path}/aerosol_mm_climatology_{year}_{source}_{date}.nc",
                      format='NETCDF4_CLASSIC')
trace_gas_sel.to_netcdf(f"{cams_output_path}/trace_gas_mm_climatology_{year}_{source}_{date}.nc",
                        format='NETCDF4_CLASSIC')
