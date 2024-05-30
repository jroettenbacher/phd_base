#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 28.06.2023

Use CAMS data and calculate a monthly mean climatology for relevant input parameters for ecRad.

"""

# %% import modules
import pylim.helpers as h
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %% set paths and options
campaign = "halo-ac3"
cams_path = h.get_path("cams", campaign=campaign)
cams_file = "cams_ml_halo_ac3.nc"
ystart, yend = 2003, 2020  # which years to use
lon_bounds = [-60, 30]

# %% read cams data
ds = xr.open_dataset(f"{cams_path}/{cams_file}")
ds = ds.sel(time=slice(pd.to_datetime(f"{ystart}-01-01"), pd.to_datetime(f"{yend}-12-31")))

# %% get latitude and longitude values
lat_values, lon_values = h.longitude_values_for_gaussian_grid(ds.lat.to_numpy(), ds.reduced_points.to_numpy(), lon_bounds)

# %% assign coordinates to rgrid
ds = ds.drop_dims("lat")
ds = ds.assign_coords(lat=("rgrid", lat_values), lon=("rgrid", lon_values))
ds = ds.set_index(rgrid=["lat", "lon"])

# %% calculate monthly mean climatology
ds_clim = ds.groupby("time.month").mean().sel(month=[3, 4])

# %% contour plot
x, y, z = ds.lon, ds.lat, ds_clim.o3.isel(month=0, lev=50)
_, ax = plt.subplots()
ax.tricontour(x,y,z, levels=14, linewidths=0.5, colors='k')
cntr2 = ax.tricontourf(x, y, z, levels=14, cmap="viridis")

_.colorbar(cntr2, ax=ax)
ax.plot(ds.lon, ds.lat, 'k.', ms=3)

plt.show()
plt.close()
# %% convert MultiIndex back to normal index and make lat and lon variables to save to netCDF file
ds_out = ds_clim.reset_index(["rgrid", "lat", "lon"])
ds_out["rgrid"]= np.arange(0, ds_clim.lat.shape[0])
ds_out = ds_out.reset_coords(["lat", "lon"])
for var in ["lat", "lon"]:
    ds_out[var] = xr.DataArray(ds_out[var].to_numpy(), dims="rgrid")
ds_out.to_netcdf(f"{cams_path}/cams_gg_monthly_climatology_{ystart}-{yend}.nc")

