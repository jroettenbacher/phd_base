#!/usr/bin/env python
"""Quicklook of infrared camera KT19

- read in data and check for time axis errors -> Panoply cannot plot data due to duplicate values
- RF02:
    - need to drop first 20 values

*author*: Johannes Röttenbacher
"""

# %% module import
import pylim.helpers as h
import pylim.halo_ac3 as campaign_meta
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% set up paths and options
campaign = "halo-ac3"
date = "20220313"
key = "RF03"
flight = f"HALO-AC3_{date}_HALO_{key}"
kt19_dir = h.get_path("kt19", flight, campaign)
kt19_file = f"HALO-AC3_HALO_KT19_BrightnessTemperature_{date}_{key}_new.nc"
start, end = campaign_meta.take_offs_landings[key]

# %% read in file
kt19 = xr.open_dataset(f"{kt19_dir}/{kt19_file}")
kt19 = kt19.sel(time=slice(start.strftime("%Y-%m-%d %H:%M:%S"),
                           end.strftime("%Y-%m-%d %H:%M:%S")))
time_diff = kt19.time.diff(dim="time")

# %% select only timestamps where time difference is greater 0
idx_neg = np.where(time_diff > pd.Timedelta(seconds=0))[0]
kt19 = kt19.isel(time=idx_neg)

# %% plot time
kt19.time.plot()
plt.show()
plt.close()

# %% plot time differences
time_diff.plot(label="time diff")
plt.ylabel("Time Difference (ns)")
plt.show()
plt.close()

# %% check time differences
m = time_diff.median()
min = time_diff.min()
max = time_diff.max()
np.unique(time_diff)
idx = time_diff.where(time_diff == min, drop=True)
idx.plot()
plt.show()

k19_res = kt19.resample(time="50ms").first()
k19_res = kt19.resample(time="50ms").mean()
ds = kt19

df_h = ds.to_dataframe().resample("50ms").mean()  # what we want (quickly), but in Pandas form
vals = [xr.DataArray(data=df_h[c], dims=['time'], coords={'time': df_h.index}, attrs=ds[c].attrs) for c in df_h.columns]
ds_h = xr.Dataset(dict(zip(df_h.columns, vals)), attrs=ds.attrs)

ds_h.to_netcdf(f"{kt19_dir}/tmp.nc")