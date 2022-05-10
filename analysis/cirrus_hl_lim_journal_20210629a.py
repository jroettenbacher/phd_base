#!/usr/bin/env python
"""Analysis for LIM Journal contribution 2021

- average spectra over height during staircase pattern

*author*: Johannes Röttenbacher
"""

# %% module import
import pylim.helpers as h
from pylim import reader
from pylim import smart
from pylim.cirrus_hl import stop_over_locations, coordinates
from pylim.bahamas import plot_props
import numpy as np
import os
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
from tqdm import tqdm
import logging

log = logging.getLogger("pylim")
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

# %% set up paths and meta data
flight = "Flight_20210629a"
campaign = "cirrus-hl"
smart_dir = h.get_path("calibrated", flight, campaign)
bahamas_dir = h.get_path("bahamas", flight, campaign)
plot_path = f"{h.get_path('plot')}/{flight}"
start_dt = pd.Timestamp(2021, 6, 29, 9, 42)
end_dt = pd.Timestamp(2021, 6, 29, 12, 10)

# %% read in data
fdw_vnir = xr.open_dataset(f"{smart_dir}/cirrus-hl_SMART_Fdw_VNIR_2021_06_29.nc")
fdw_swir = xr.open_dataset(f"{smart_dir}/cirrus-hl_SMART_Fdw_SWIR_2021_06_29.nc")
# filter VNIR data
fdw_vnir = fdw_vnir.sel(wavelength=slice(420, 950))
# merge VNIR and SWIR channel
smart_fdw = xr.merge([fdw_vnir.Fdw, fdw_swir.Fdw])

bahamas = reader.read_bahamas(f"{bahamas_dir}/CIRRUSHL_F05_20210629a_ADLR_BAHAMAS_v1.nc")

# %% select only relevant times
smart_fdw = smart_fdw.sel(time=slice(start_dt, end_dt))
bahamas = bahamas.sel(time=slice(start_dt, end_dt))

# %% define staircase sections according to flight levels
altitude = np.round(bahamas.H / 100, 0) * 100  # rounded to 10m to remove fluctuations
idx = np.where(np.diff(altitude) != 0)
altitude[idx].plot(marker="x")
plt.show()
plt.close()
times = bahamas.time[idx]
manual_idx = [6,  7, 12, 13, 18, 19, 25, 26, 31, 32, 37, 38, 47, 48]
times_sel = times[manual_idx]
start_dts = times_sel[::2]
end_dts = times_sel[1::2]

# TODO: Add section labels
altitude.sel(time=times).plot()
altitude.sel(time=start_dts).plot(ls="", marker="x", label="Start times")
altitude.sel(time=end_dts).plot(ls="", marker="x", label="End times")
plt.xlabel("Time (UTC)")
plt.ylabel("Altitude (m)")
plt.grid()
plt.legend()
plt.show()
plt.close()

altitude.sel(time=start_dts).values - altitude.sel(time=end_dts).values  # differences between start and end of section

# %% select SMART data according to altitude and take averages over height
sections = dict()
for i, (st, et) in enumerate(zip(start_dts, end_dts)):
    sections[f"mean_spectra_{i}"] = smart_fdw.sel(time=slice(st, et)).mean(dim="time")

h.set_cb_friendly_colors()
labels = ["Section 1 (8.3$\,$km)", "Section 2 (8.7$\,$km)", "Section 3 (9.3$\,$km)", "Section 4 (10$\,$km)",
          "Section 5 (10.6$\,$km)", "Section 6 (11.2$\,$km)", "Section 7 (12.2$\,$km)"]
fig, ax = plt.subplots()
for section, label in zip(sections, labels):
    fdw = sections[section].Fdw
    fdw.plot(ax=ax, label=label)
    # for x_, y_, label in zip(fdw.wavelength.values, fdw.values, range(len(fdw))):
    #     if label > 700 and label < 790:
    #         plt.annotate(label, (x_, y_))

ax.legend()
ax.set_ylim(0, 1.35)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Downward Irradiance (W$\\,$m$^{-2}\\,$nm$^{-1}$)")
ax.grid()
# plt.show()
plt.savefig(f"{plot_path}/{campaign.swapcase()}_SMART_staircase_spectra_{flight}.png", dpi=300)
plt.close()

# %%
fdw_vnir.Fdw.mean(dim="time").plot()
plt.show()
plt.close()
