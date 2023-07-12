#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 07.06.2023

Figures for the manuscript Röttenbacher et al. 2023

- comparison of lidar-radar cloudmask with IFS predicted cloud fraction
- comparison of temperature and humidity profile between IFS and dropsonde
- ice effective radius parameterization
- plot of re_ice against temperature
- plot of vertical resolution for the IFS model

"""

# %% import modules
import pylim.helpers as h
import pylim.halo_ac3 as meta
from pylim import ecrad, reader
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cmasher as cm
from matplotlib import colors
from metpy import constants as mc

h.set_cb_friendly_colors()

# %% set paths
campaign = "halo-ac3"
key = "RF17"
flight = meta.flight_names[key]
date = flight[9:17]
plot_path = "C:/Users/Johannes/Documents/Doktor/manuscripts/2023_arctic_cirrus/figures"
bahamas_path = h.get_path("bahamas", flight, campaign)
ecrad_path = f"{h.get_path('ecrad', flight, campaign)}/{date}"

# filenames
bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{key}_v1.nc"
ecrad_input = f"ecrad_merged_input_{date}_v1.nc"
ecrad_file = f"ecrad_merged_inout_{date}_v1_mean.nc"

# %% read in data
bahamas_ds = reader.read_bahamas(f"{bahamas_path}/{bahamas_file}")
ecrad_ds = xr.open_dataset(f"{ecrad_path}/{ecrad_file}")
ecrad_input_ds = xr.open_dataset(f"{ecrad_path}/{ecrad_input}")

# %% plot minimum ice effective radius from Sun2001 parameterization
latitudes = np.arange(0, 91)
de2re = 0.64952  # from suecrad.f90
min_ice = 60
min_diameter_um = 20 + (min_ice - 20) * np.cos((np.deg2rad(latitudes)))
min_radius_um = de2re * min_diameter_um

_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(latitudes, min_radius_um, ".")
ax.set(xlabel="Latitude (°N)", ylabel=r"Minimum ice effective radius ($\mu m$)")
ax.grid()
plt.show()
plt.close()

# %% calculate ice effective radius for different IWC and T combinations covering the Arctic to the Tropics
lats = [0, 18, 36, 54, 72, 90]
iwc_kgkg = np.logspace(-5.5, -2.5, base=10, num=100)
t = np.arange(182, 273)
empty_array = np.empty((len(t), len(iwc_kgkg), len(lats)))
for i, temperature in enumerate(t):
    for j, iwc in enumerate(iwc_kgkg):
        for k, lat in enumerate(lats):
            empty_array[i,j,k] = ecrad.ice_effective_radius(25000, temperature, 1, iwc, 0, lat)

da = xr.DataArray(empty_array*1e6, coords=[("temperature", t), ("iwc_kgkg", iwc_kgkg*1e3), ("Latitude", lats)], name="re_ice")

# %% plot re_ice iwc t combinations
g = da.plot(col="Latitude", col_wrap=3, cbar_kwargs=dict(label="Ice effective radius ($\mu$m)"), cmap=cm.chroma)
for i, ax in enumerate(g.axes.flat[::3]):
    ax.set_ylabel("Temperature (K)")
for i, ax in enumerate(g.axes.flat[3:]):
    ax.set_xlabel("IWC (g/kg)")
for i, ax in enumerate(g.axes.flat):
    ax.set_xscale("log")
figname = f"{plot_path}/re_ice_parameterization_T-IWC-Lat_log.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% minimum distance between two grid points
a = 6371229  # radius of sphere assumed by IFS
distances_between_longitudes = np.pi / 180 * a * np.cos(np.deg2rad(ecrad_input_ds.lat)) / 1000
distances_between_longitudes.min()
bahamas_ds.IRS_LAT.max()
longitude_diff = ecrad_input_ds.lon.diff(dim="time")
distance_between_gridcells = longitude_diff * distances_between_longitudes


_, ax = plt.subplots(figsize=h.figsize_wide)
distance_between_gridcells.plot(x="time", y="column",
                                cbar_kwargs={"label": "East-West distance between grid cells (km)"})
ax.grid()
ax.set(xlabel="Time (UTC)", ylabel="Column")
h.set_xticks_and_xlabels(ax, pd.to_timedelta(bahamas_ds.time[-1].to_numpy() - bahamas_ds.time[0].to_numpy()))
plt.tight_layout()
plt.show()
plt.close()
