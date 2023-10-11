#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 10.10.2023

Select closest points of CAMS global reanalysis and global greenhouse gas reanalysis data to flight track.
Read in CAMS from different sources (ADS, MARS online, MARS ECMWF, Copernicus Knowledge Base).

Required User Input:

- source (ADS, MARS, ECMWF, CKB)
- year (2019, 2020)

Input:

- monthly mean CAMS data (either monthly mean or monthly mean by hour of day)

Output:

- merged CAMS monthly mean data along flight track

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
source = "ADS"
year = "2020"
date = "20220411"

cams_path = h.get_path("cams_raw", campaign=campaign)
cams_output_path = h.get_path("cams", campaign=campaign)
eac4_file = f"cams_eac4_global_reanalysis_mm_{year}_pl.nc"
gghg_file = f"cams_global_ghg_reanalysis_mm_{year}_pl.nc"
# IFS path for flight track file
ifs_path = h.get_path("ifs", campaign=campaign)

# %% read in data
eac4 = xr.open_dataset(f"{cams_path}/{eac4_file}")
gghg = xr.open_dataset(f"{cams_path}/{gghg_file}")
cams_ml = xr.merge([eac4, gghg]).sel(time=f"2020-{date[4:6]}-01")
nav_data_ip = pd.read_csv(f"{ifs_path}/{date}/nav_data_ip_{date}.csv", index_col="time", parse_dates=True)

# %% create array of aircraft locations
points = np.deg2rad(
    np.column_stack(
        (nav_data_ip.lat.to_numpy(), nav_data_ip.lon.to_numpy())))

# %% select closest points along flight track from cams data
cams_latlon = cams_ml.stack(latlon=["latitude", "longitude"])  # combine lat and lon into one dimension
# make an array with all lat lon combinations
cams_lat_lon = np.array([np.array(element) for element in cams_latlon["latlon"].to_numpy()])
# build the look up tree
cams_tree = BallTree(np.deg2rad(cams_lat_lon), metric="haversine")
# query the tree for the closest CAMS grid points to the flight track
dist, idx = cams_tree.query(points, k=1)
# select only the closest grid points along the flight track
cams_sel = cams_latlon.isel(latlon=idx.flatten())
# reset index and make lat, lon and time (month) a variable
cams_sel = (cams_sel
            .reset_index(["latlon", "latitude", "longitude"])
            .reset_coords(["latitude", "longitude", "time"])
            .drop_vars(["time", "latitude", "longitude"])
            .rename(latlon="time")
            .assign(time=nav_data_ip.index.to_numpy()))  # replace latlon with time as a dimension/coordinate

# %% save file to netcdf
cams_sel.attrs["history"] = (cams_sel.attrs["history"]
                             + f" {datetime.today().strftime('%c')}: "
                               f"formatted file to serve as input to ecRad"
                               f" using ecrad_cams_preprocessing.py")
cams_sel.to_netcdf(f"{cams_output_path}/cams_mm_climatology_{year}_{source}_{date}.nc",
                   format='NETCDF4_CLASSIC')