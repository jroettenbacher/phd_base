#!/usr/bin/env python
"""Analysis script for statistics about the whole campaign
author: Johannes Röttenbacher
"""

# %% module import
import xarray as xr
from smart import get_path

# %% define functions


def preprocess_bahamas(ds: xr.Dataset) -> xr.Dataset:
    # swap dimensions
    ds = ds.swap_dims({"tid": "TIME"})
    return ds


# %% set up paths
bahamas = "BAHAMAS"
path = get_path('all', instrument=bahamas)
plot_path = get_path('plot')

# %% read in bahamas data
ds = xr.open_mfdataset(f"{path}/*.nc", preprocess=preprocess_bahamas)

# %% find max and min lat and lon
lat_min = ds.IRS_LAT.min().compute().values
lat_max = ds.IRS_LAT.max().compute().values
lon_min = ds.IRS_LON.min().compute().values
lon_max = ds.IRS_LON.max().compute().values
alt_max = ds.IRS_ALT.max().compute().values
print(f"Minumum Latitude: {lat_min}°N\nMaximum Latitude: {lat_max}°N\nMinumum Longitude: {lon_min}°E"
      f"\nMaximum Longitude: {lon_max}°E\nMaximum Altitutde: {alt_max}m")
