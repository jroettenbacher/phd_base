#!/usr/bin/env python
"""Prepare overview plots for each flight
author: Johannes Röttenbacher
"""

# %% module import
import numpy as np
from smart import get_path
import logging
from bahamas import plot_props, read_bahamas, plot_bahamas_flight_track
from libradtran import read_libradtran
from cirrus_hl import stop_over_locations, coordinates, flight_hours
import os
from geopy.distance import geodesic
import smart
from helpers import make_dir, set_cb_friendly_colors, set_xticks_and_xlabels
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Patch
import cartopy
import cartopy.crs as ccrs
import xarray as xr
import pandas as pd
import rasterio
from rasterio.plot import show

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)
# %% set paths
flight = "Flight_20210625a"
bahamas_dir = get_path("bahamas", flight)
bahamas_dir_all = get_path("all", instrument="BAHAMAS")
bacardi_dir = get_path("bacardi", flight)
smart_dir = get_path("calibrated", flight)
sat_image = "/projekt_agmwend/data/Cirrus_HL/01_Flights/Flight_20210629a/satellite/snapshot-2021-06-29T00_00_00Z.tiff"
outpath = f"C:/Users/Johannes/Documents/Doktor/campaigns/CIRRUS-HL/overview"
make_dir(outpath)
# read in BAHAMAS and BACARDI data
file = [f for f in os.listdir(bahamas_dir) if f.endswith(".nc")][0]
bahamas = read_bahamas(flight)
bacardi_file = [f for f in os.listdir(bacardi_dir) if f.endswith("R0.nc")][0]
bacardi = xr.open_dataset(f"{bacardi_dir}/{bacardi_file}")
# %% plot bahamas flight track
plot_bahamas_flight_track(flight, outpath=outpath)

# %% calculate total flight hours
total_flight_hours = np.sum(flight_hours).total_seconds() / 60 / 60

# %% plot bahamas heading and solar azimuth angle
set_cb_friendly_colors()
fig, ax = plt.subplots()
bahamas.IRS_HDG.plot(x="TIME", label="BAHAMAS Heading", ax=ax)
bacardi.saa.plot(x="time", label="Solar Azimuth Angle", ax=ax)
ax.grid()
ax.set_ylabel("Angle (°)")
ax.set_xlabel("Time (UTC)")
ax.legend()
plt.show()
plt.close()

# %% read in all BAHAMAS data


def preprocess_bahamas(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.swap_dims({"tid": "TIME"})
    ds = ds.rename({"TIME": "time"})
    return ds


bahamas_all_files = [os.path.join(bahamas_dir_all, f) for f in os.listdir(bahamas_dir_all)]
bahamas_all = xr.open_mfdataset(bahamas_all_files, preprocess=preprocess_bahamas)

# %% check if measurement is inside 50km radius to Leipzig and remove everything else
lim_lat_lon = (51.334, 12.388)
radius = 50  # radius in km
# only calculate every 1000th measurement (every 100s) to reduce computing time
distances = [geodesic((lat, lon), lim_lat_lon).km for lat, lon in
             zip(bahamas_all.IRS_LAT.values[::1000], bahamas_all.IRS_LON.values[::1000])]
# repeat each distance 1000 times and cut array to length of bahamas data to be able to select only the relevant times
distances = np.array(distances).repeat(1000)[:len(bahamas_all.IRS_LAT)]
bahamas_all_masked = bahamas_all.where(distances <= radius)  # set all values outside of radius to nan
# check for nan values -> returns array with True/False, diff returns True when consecutive elements differ
start_stop_sel = np.diff(np.isnan(bahamas_all_masked.IRS_LON), append=True)  # append one value to keep shape of bahamas
# select only start and stop times of being inside the circle around Leipzig
start_stop_times = bahamas_all.time[start_stop_sel].values
# create data frame with two columns
out_df = pd.DataFrame({"start": start_stop_times[0:-1:2], "end": start_stop_times[1::2]})
csv_filepath = "C:/Users/Johannes/Documents/Doktor/campaigns/CIRRUS-HL/overview/cirrus-hl_leipzig_overpasses.csv"
out_df.to_csv(csv_filepath, sep="\t", index=False)
log.info(f"Saved {csv_filepath}")
