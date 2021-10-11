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
import smart
from functions_jr import make_dir, set_cb_friendly_colors, set_xticks_and_xlabels
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
flight = "Flight_20210712b"
bahamas_dir = get_path("bahamas", flight)
bacardi_dir = get_path("bacardi", flight)
smart_dir = get_path("calibrated", flight)
sat_image = "/projekt_agmwend/data/Cirrus_HL/01_Flights/Flight_20210629a/satellite/snapshot-2021-06-29T00_00_00Z.tiff"
outpath = f"C:/Users/Johannes/Documents/Doktor/campaigns/CIRRUS-HL/overview"
make_dir(outpath)
# read in BAHAMAS and BACARDI data
file = [f for f in os.listdir(bahamas_dir) if f.endswith(".nc")][0]
bahamas = read_bahamas(f"{bahamas_dir}/{file}")
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
