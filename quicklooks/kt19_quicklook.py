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
kt19_file = f"HALO-AC3_HALO_KT19_BrightnessTemperature_{date}_{key}.nc"
start, end = campaign_meta.take_offs_landings[key]

# %% read in file
kt19 = xr.open_dataset(f"{kt19_dir}/{kt19_file}")
kt19 = kt19.sel()
kt19 = kt19.sel(time=slice(start.strftime("%Y-%m-%d %H:%M:%S"),
                           end.strftime("%Y-%m-%d %H:%M:%S")))
time_diff = kt19.time.diff(dim="time")

# %% check where time difference is negative
idx_neg = np.where(time_diff < pd.Timedelta(seconds=0))

# %% plot time
kt19.time.plot()
plt.show()
plt.close()

# %% plot time differences
time_diff.plot(label="time diff")
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



