#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 13.07.2023

Compare different resampling strategies when creating 1 minutely input files for ecRad.

Options:
    - asfreq() (same as resample().first())
    - mean()

For the 1 minutely ecRad input files created by using BAHAMAS data, the BAHAMAS data has been resampled using the mean function.
This is important when finding the closest IFS grid point as the mean latitude/longitude is used instead of the first value.

To problem is caused by the conversion to a pandas DataFrame in ecrad_read_ifs.py.
For xarray DataSets resample(time="1Min").mean() and resample(time="1Min").asfreq() returns the same.
However, for a pandas DataFrame this is not true.
"""
import os

# %% import modules
import pylim.helpers as h
import pylim.halo_ac3 as meta
from pylim import reader
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

h.set_cb_friendly_colors()

# %% set paths
campaign = "halo-ac3"
key = "RF17"
flight = meta.flight_names[key]
date = flight[9:17]

plot_path = f"{h.get_path('plot', flight, campaign)}/varcloud_resampling"
h.make_dir(plot_path)
ecrad_path = f"{h.get_path('ecrad', flight, campaign)}/{date}"
ecrad_mean_file = f"ecrad_merged_inout_{date}_v17.nc"
ecrad_asfreq_file = f"ecrad_merged_inout_{date}_v17_old.nc"
bahamas_path = h.get_path("bahamas", flight, campaign)
bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{key}_v1.nc"
nav_data_ip_path = f"{h.get_path('ifs', flight, campaign)}/{date}"
varcloud_path = h.get_path("varcloud", flight, campaign)
varcloud_file = "VAR2LAGR_L1D_V1_AC3_HALO_RF17_A20220411_090100_152000_TS1_AS100_P20230306190254.nc"

# %% read in data
ecrad_mean = xr.open_dataset(f"{ecrad_path}/{ecrad_mean_file}")
ecrad_asfreq = xr.open_dataset(f"{ecrad_path}/{ecrad_asfreq_file}")
bahamas_ds = reader.read_bahamas(f"{bahamas_path}/{bahamas_file}")
nav_data_ip = pd.read_csv(f"{nav_data_ip_path}/nav_data_ip_{date}.csv", index_col="time", parse_dates=True)
varcloud_ds = xr.open_dataset(f"{varcloud_path}/{varcloud_file}").swap_dims(time="Time", height="Height").rename(
    Time="time")

# %% plot difference in lat and lon
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(ecrad_asfreq.lon, ecrad_asfreq.lat, label="Resampling: asfreq()")
ax.plot(ecrad_mean.lon, ecrad_mean.lat, label="Resampling: mean()")
ax.legend()
plt.tight_layout()
plt.show()
plt.close()

# %% plot bahamas against varcloud
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(ecrad_mean.lat, bahamas_ds.IRS_LAT)
plt.tight_layout()
plt.show()
plt.close()

# %% plot nav data ip against bahamas
bahamas_plot = bahamas_ds.resample(time="1Min").asfreq()
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(nav_data_ip.lat - bahamas_plot.IRS_LAT)
plt.tight_layout()
plt.show()
plt.close()

# %% plot og varcloud against nav ip
varcloud_plot = varcloud_ds.resample(time="1Min").mean()
nav_data_plot = nav_data_ip.loc[varcloud_plot.time]
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(nav_data_plot.lat - varcloud_plot.Latitude)
plt.tight_layout()
plt.show()
plt.close()

# %% plot varcloud against bahamas
varcloud_plot = varcloud_ds
bahamas_plot = bahamas_ds.resample(time="1s").first()
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(bahamas_plot.IRS_LAT - varcloud_plot.Latitude)
plt.tight_layout()
plt.show()
plt.close()
