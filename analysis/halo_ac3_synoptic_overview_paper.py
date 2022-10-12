#!/usr/bin/env python
"""Figures for the synoptic overview paper by Walbröl et al.

- first plot: 2 cols, 2 rows with trajectories on top and one axis with a radar plot

*author*: Johannes Röttenbacher
"""

# %% import modules
import pylim.helpers as h
import pylim.halo_ac3 as meta
import ac3airborne
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import patheffects
import cmasher as cmr
import cartopy
import cartopy.crs as ccrs
from tqdm import tqdm
cm = 1/2.54  # inch to centimeter conversion

# %% set paths and filenames
campaign = "halo-ac3"
halo_key = "RF17"
flight = meta.flight_names[halo_key]
date = flight[9:17]
plot_dir = f"{h.get_path('plot', campaign=campaign)}/../manuscripts/2022_HALO-AC3_synoptic_overview"
trajectory_dir = f"{h.get_path('trajectories', campaign=campaign)}/{date}"
era5_dir = "/projekt_agmwend/home_rad/BenjaminK/HALO-AC3/ERA5/ERA5_ml_noOMEGAscale"
radar_dir = h.get_path("hamp_mira", flight, campaign)

# set options and credentials for HALO-AC3 cloud and intake catalog
kwds = {'simplecache': dict(cache_storage='E:/HALO-AC3/cloud', same_names=True)}
credentials = {"user": os.environ.get("AC3_CLOUD_USER"), "password": os.environ.get("AC3_CLOUD_PASSWORD")}
cat = ac3airborne.get_intake_catalog()

# filenames
era5_files = [os.path.join(era5_dir, f) for f in os.listdir(era5_dir) if f"P{date}" in f]
era5_files.sort()
radar_file = "radar_20220411_v0.6.nc"
# meta data
start_dt, end_dt = pd.to_datetime("2022-04-11 09:30"), pd.to_datetime("2022-04-11 14:30")

# %% read in data
# radar data from halo-ac3 cloud
# radar_ds = cat["HALO-AC3"]["HALO"]["HAMP_MIRA"][f"HALO-AC3_HALO_{halo_key}"](storage_options=kwds, **credentials).to_dask()
ins = cat["HALO-AC3"]["HALO"]["GPS_INS"][f"HALO-AC3_HALO_{halo_key}"](storage_options=kwds, **credentials).to_dask()
# radar_ds = xr.open_dataset(f"{radar_dir}/{radar_file}")
era5_ds = xr.open_mfdataset(era5_files)

# %% preprocess radar data
# radar_ds = radar_ds.sel(time=slice(start_dt, end_dt))
# radar_ds["dBZg"] = radar_ds["dBZg"].where(radar_ds["dBZg"] > -90, np.nan)
# radar_ds["dBZg"] = radar_ds["dBZg"].where(radar_ds["dBZg"] < 50, np.nan)

# %% preprocess ERA5 data
q_air = 1.292  # dry air density at 0°C in kg/m3
g_geo = 9.80665  # gravitational acceleration [m s^-2]
era5_ds = era5_ds.sel(time=slice(start_dt, end_dt), lon=slice(-60, 30), lat=slice(65, 90)).compute()
# calculate pressure on all model levels
pressure = era5_ds.hyam + era5_ds.hybm * era5_ds.PS * 100
# select only relevant model levels and swap dimension names
era5_ds["pressure"] = pressure.sel(nhym=slice(39, 137)).swap_dims({"nhym": "lev"})
# calculate pressure height
p_hl = era5_ds["pressure"]
p_sur = era5_ds.PS * 100
era5_ds["press_height"] = -p_sur * np.log(p_hl / p_sur) / (q_air * g_geo)

# select grid cells closest to aircraft track
ins_tmp = ins.sel(time=era5_ds.time, method="nearest")
# convert lat and lon to precision of ERA5 data
lats = (np.round(ins_tmp.lat/0.25, 0)*0.25)
lons = (np.round(ins_tmp.lon/0.25, 0)*0.25)
era5_sel = era5_ds.sel(lat=lats, lon=lons).reset_coords(["lat", "lon"])

# calculate model altitude to aircraft altitude
aircraft_height_level = list()
for i in range(len(era5_sel.time)):
    ins_altitude = ins_tmp.alt.isel(time=i).values
    p_height = era5_sel.press_height.isel(time=i, drop=True).values
    aircraft_height_level.append(int(h.arg_nearest(p_height, ins_altitude)))

height_level_da = xr.DataArray(aircraft_height_level, dims=["time"], coords={"time": era5_sel.time})+40
aircraft_height_da = era5_sel.press_height.sel(lev=height_level_da)

# # %% plot radar cross-section
# h.set_cb_friendly_colors()
# plt.rc("font", family="serif", size=12)
# fig, ax = plt.subplots(figsize=(18*cm, 10.125*cm))
# ins.alt.plot(x="time", label="HALO altitude", color="#888888", lw=3)
# radar_ds["dBZg"].plot(x="time", robust=True, cmap="viridis", cbar_kwargs=dict(label="Radar Reflectivity (dBZ)"))
# yticks = ax.get_yticks()
# ax.set_yticklabels(yticks/1000)
# h.set_xticks_and_xlabels(ax, time_extend=end_dt-start_dt)
# ax.set_xlabel("Time (UTC)")
# ax.set_ylabel("Altitude (km)")
# ax.legend(loc=1)
# plt.tight_layout()
# figname = f"{plot_dir}/{flight}_MIRA_radar-reflectivity.png"
# plt.savefig(figname, dpi=100)
# plt.show()
# plt.close()

# %% plot ERA5 cloud cover/IWC/water vapour etc. (2D variables) along flight track
variable = "Q"
units = dict(CC="", CLWC="g$\,$kg$^{-1}$", CIWC="g$\,$kg$^{-1}$", CSWC="g$\,$kg$^{-1}$", CRWC="g$\,$kg$^{-1}$", T="K",
             Q="g$\,$kg$^{-1}$")
scale_factor = dict(CC=1, CLWC=1000, CIWC=1000, CSWC=1000, CRWC=1000, T=1, Q=1000)
colorbarlabel = dict(CC="Cloud Cover", CLWC="Cloud Liquid Water Content", CIWC="Cloud Ice Water Content",
                     CSWC="Cloud Snow Water Content", CRWC="Cloud Rain Water Content", T="Temperature",
                     Q="Specific Humidity")
robust = dict(CC=False)
robust = robust[variable] if variable in robust else True
cmap = dict(T="bwr")
cmap = cmap[variable] if variable in cmap else "YlGnBu"
cmap = plt.get_cmap(cmap).copy()
cmap.set_bad(color="white")
plot_ds = era5_sel[variable] * scale_factor[variable]
plot_ds = plot_ds.where(plot_ds > 0, np.nan)  # set 0 values to nan to mask them in the plot
clabel = f"{colorbarlabel[variable]} ({units[variable]})"

h.set_cb_friendly_colors()
plt.rc("font", family="serif", size=10)
fig, ax = plt.subplots(figsize=(18*cm, 10.125*cm))
height_level_da.plot(x="time", label="HALO altitude", ax=ax, color="#888888", lw=3)
cmap = cmr.get_sub_cmap("cmr.freeze", .25, 0.85) if variable == "CIWC" else cmap
plot_ds.plot(x="time", robust=robust, cmap=cmap, cbar_kwargs=dict(pad=0.12, label=clabel))

# set first y axis
ax.set_ylim(60, 138)
ax.yaxis.set_major_locator(plt.FixedLocator(range(60, 138, 20)))
ax.invert_yaxis()
# set labels
h.set_xticks_and_xlabels(ax, time_extend=end_dt-start_dt)
ax.set_xlabel("Time 11 April 2022 (UTC)")
ax.set_ylabel("Model Level")
ax.legend(loc=1)

# add axis with pressure height
ax2 = ax.twinx()
ax2.set_ylim(60, 138)
ax2.yaxis.set_major_locator(plt.FixedLocator(range(60, 138, 20)))
ax2.invert_yaxis()
yticks = ax.get_yticks()
ylabels = np.round(era5_sel["press_height"].isel(time=2).sel(lev=yticks).values / 1000, 1)
ax2.set_yticklabels(ylabels)
ax2.set_ylabel("Pressure Altitude (km)")

plt.tight_layout()
figname = f"{plot_dir}/{flight}_ERA5_{variable}.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot ERA5 CIWC and CLWC along flight track
variable = "CIWC"
units = dict(CC="", CLWC="g$\,$kg$^{-1}$", CIWC="g$\,$kg$^{-1}$", CSWC="g$\,$kg$^{-1}$", CRWC="g$\,$kg$^{-1}$", T="K")
scale_factor = dict(CC=1, CLWC=1000, CIWC=1000, CSWC=1000, CRWC=1000, T=1)
colorbarlabel = dict(CC="Cloud Cover", CLWC="Cloud Liquid Water Content", CIWC="Cloud Ice Water Content",
                     CSWC="Cloud Snow Water Content", CRWC="Cloud Rain Water Content", T="Temperature")
robust = dict(CC=False)
robust = robust[variable] if variable in robust else True
plot_ds = era5_sel[variable] * scale_factor[variable]
plot_ds = plot_ds.where(plot_ds > 0, np.nan)  # set 0 values to nan to mask them in the plot
clabel = f"{colorbarlabel[variable]} ({units[variable]})"

h.set_cb_friendly_colors()
plt.rc("font", family="serif", size=10)
fig, ax = plt.subplots(figsize=(18*cm, 10.125*cm))
height_level_da.plot(x="time", label="HALO altitude", ax=ax, color="#888888", lw=3)
cmap = cmr.get_sub_cmap("cmr.freeze", .25, 0.85)
plot_ds.plot(x="time", robust=robust, cmap=cmap, cbar_kwargs=dict(pad=0.01, label=clabel))

# plot CLWC
variable = "CLWC"
plot_ds = era5_sel[variable] * scale_factor[variable]
plot_ds = plot_ds.where(plot_ds > 0, np.nan)  # set 0 values to nan to mask them in the plot
clabel = f"{colorbarlabel[variable]} ({units[variable]})"
cmap = cmr.get_sub_cmap("cmr.flamingo", .25, 0.9)
plot_ds.plot(x="time", robust=robust, cmap=cmap, cbar_kwargs=dict(pad=0.14, label=clabel))

# set first y axis
ax.set_ylim(60, 138)
ax.yaxis.set_major_locator(plt.FixedLocator(range(60, 138, 20)))
ax.invert_yaxis()
# set labels
h.set_xticks_and_xlabels(ax, time_extend=end_dt-start_dt)
ax.set_xlabel("Time 11 April 2022 (UTC)")
ax.set_ylabel("Model Level")
ax.legend(loc=1)

# add axis with pressure height
ax2 = ax.twinx()
ax2.set_ylim(60, 138)
ax2.yaxis.set_major_locator(plt.FixedLocator(range(60, 138, 20)))
ax2.invert_yaxis()
yticks = ax.get_yticks()
ylabels = np.round(era5_sel["press_height"].isel(time=2).sel(lev=yticks).values / 1000, 1)
ax2.set_yticklabels(ylabels)
ax2.set_ylabel("Pressure Altitude (km)")

plt.tight_layout()
figname = f"{plot_dir}/{flight}_ERA5_CIWC_CLWC.pdf"
plt.savefig(figname, dpi=100)
plt.show()
plt.close()

# %% prepare ERA5 data for map plots
variable = "CIWC"
units = dict(CC="", CLWC="g$\,$kg$^{-1}$", CIWC="g$\,$kg$^{-1}$", CSWC="g$\,$kg$^{-1}$", CRWC="g$\,$kg$^{-1}$", T="K")
scale_factor = dict(CC=1, CLWC=1000, CIWC=1000, CSWC=1000, CRWC=1000, T=1)
colorbarlabel = dict(CC="Cloud Cover", CLWC="Cloud Liquid Water Content", CIWC="Cloud Ice Water Content",
                     CSWC="Cloud Snow Water Content", CRWC="Cloud Rain Water Content", T="Temperature")
pressure_level = 30000  # Pa
plot_ds = era5_ds.sel(lat=slice(68, 90), lon=slice(-45, 30))  # select area to plot
plot_ds = plot_ds.where(plot_ds["pressure"] < pressure_level, drop=True)
# select variable and time
plot_da = plot_ds[variable].isel(time=3, drop=True)
# sum over level
plot_da = plot_da.sum(dim=["lev"], skipna=True)
# set 0 to nan for clearer plotting
plot_da = plot_da.where(plot_da > 0, np.nan)
# scale by a 1000 to convert kg/kg to g/kg
plot_da = plot_da * scale_factor[variable]
# plotting options
extent = [-15, 30, 68, 90]
data_crs = ccrs.PlateCarree()
cmap = cmr.get_sub_cmap("cmr.freeze", .25, 0.85)
cmap.set_bad(color="white")

# %% plot ERA5 maps of integrated IWC over certain pressure level
cbar_label = f"Integrated {colorbarlabel[variable]} ({units[variable]})"
plt.rc("font", size=10)
fig, ax = plt.subplots(figsize=(6.1, 6), subplot_kw={"projection": ccrs.NorthPolarStereo()})
ax.coastlines()
ax.add_feature(cartopy.feature.BORDERS)
ax.set_extent(extent, crs=data_crs)
gl = ax.gridlines(crs=data_crs, draw_labels=True, x_inline=False, y_inline=False)
gl.bottom_labels = False
gl.left_labels = False

# plot ERA5 data
plot_da.plot(transform=data_crs, robust=True, ax=ax, cmap=cmap, cbar_kwargs={"pad": 0.08, "label": cbar_label})
# plot flight track
points = ax.scatter(ins.lon, ins.lat, s=1, c="orange", transform=data_crs, label="Flight Track")
# plot airports Kiruna
x_kiruna, y_kiruna = meta.coordinates["Kiruna"]
ax.plot(x_kiruna, y_kiruna, '.', color="#117733", markersize=8, transform=data_crs)
ax.text(x_kiruna + 0.1, y_kiruna + 0.1, "Kiruna", fontsize=11, transform=data_crs)
# Longyearbyen
x_longyear, y_longyear = meta.coordinates["Longyearbyen"]
ax.plot(x_longyear, y_longyear, '.', color="#117733", markersize=8, transform=data_crs)
ax.text(x_longyear + 0.1, y_longyear + 0.1, "Longyearbyen", fontsize=11, transform=data_crs,
        path_effects=[patheffects.withStroke(linewidth=0.5, foreground="white")])
ax.legend(handles=[plt.plot([], ls="-", color="orange")[0]], labels=[points.get_label()], loc=1)
plt.tight_layout()
figname = f"{plot_dir}/{flight}_ERA5_{variable}_over_{pressure_level/100:.0f}hPa_map.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()
