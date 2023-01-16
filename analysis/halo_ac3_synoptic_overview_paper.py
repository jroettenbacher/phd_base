#!/usr/bin/env python
"""Figures for the synoptic overview paper by Walbröl et al.

*author*: Johannes Röttenbacher, Benjamin Kirbus
"""

# %% import modules
import pylim.helpers as h
import pylim.halo_ac3 as meta
import pylim.meteorological_formulas as met
import ac3airborne
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib import patheffects
from matplotlib.collections import LineCollection
import cartopy.crs as ccrs

cm = 1 / 2.54  # inch to centimeter conversion

# %% set paths and filenames
campaign = "halo-ac3"
halo_key = "RF17"
flight = meta.flight_names[halo_key]
date = flight[9:17]
plot_path = f"{h.get_path('plot', campaign=campaign)}/../manuscripts/2022_HALO-AC3_synoptic_overview"
trajectory_path = f"{h.get_path('trajectories', campaign=campaign)}/selection_CC_and_altitude"
era5_path = "/projekt_agmwend/home_rad/BenjaminK/HALO-AC3/ERA5/ERA5_ml_noOMEGAscale"

# set options and credentials for HALO-AC3 cloud and intake catalog
kwds = {'simplecache': dict(cache_storage='E:/HALO-AC3/cloud', same_names=True)}
credentials = {"user": os.environ.get("AC3_CLOUD_USER"), "password": os.environ.get("AC3_CLOUD_PASSWORD")}
cat = ac3airborne.get_intake_catalog()

# %% plot two maps of trajectories with surface pressure, flight track, dropsonde locations and high cloud cover and humidity profiles from radiosonde
plt_sett = {
    'TIME': {
        'label': 'Time Relative to Release (h)',
        'norm': plt.Normalize(-120, 0),
        'ylim': [-120, 0],
        'cmap_sel': 'tab20b_r',
    }
}
var_name = "TIME"
data_crs = ccrs.PlateCarree()
h.set_cb_friendly_colors()

plt.rc("font", size=6)
fig = plt.figure(figsize=(18 * cm, 15 * cm))
gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1])

# plot trajectory map 11 April in first row and first column
ax = fig.add_subplot(gs[0, 0], projection=ccrs.NorthPolarStereo())
ax.coastlines(alpha=0.5)
ax.set_xlim((-2000000, 2000000))
ax.set_ylim((-3000000, 500000))
gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=1, color='gray', alpha=0.5,
                  linestyle=':', x_inline=False, y_inline=False, rotate_labels=False)
gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 20))
gl.ylocator = mticker.FixedLocator(np.arange(60, 90, 5))
gl.top_labels = False
gl.right_labels = False

# read in ERA5 data - 11 April
era5_files = [os.path.join(era5_path, f) for f in os.listdir(era5_path) if f"P20220411" in f]
era5_files.sort()
era5_ds = xr.open_mfdataset(era5_files).sel(time=f"2022-04-11T12:00")
# calculate pressure on all model levels
pressure = era5_ds.hyam + era5_ds.hybm * era5_ds.PS * 100
# select only relevant model levels and swap dimension names
era5_ds["pressure"] = pressure.sel(nhym=slice(39, 137)).swap_dims({"nhym": "lev"})

# Plot the surface pressure - 11 April
pressure_levels = np.arange(900, 1125, 5)
E5_press = era5_ds.MSL / 100  # conversion to hPa
cp = ax.contour(E5_press.lon, E5_press.lat, E5_press, levels=pressure_levels, colors='k', linewidths=0.7,
                linestyles='solid', alpha=1, transform=data_crs)
cp.clabel(fontsize=4, inline=1, inline_spacing=4, fmt='%i', rightside_up=True, use_clabeltext=True)

# add seaice edge
ci_levels = [0.8]
E5_ci = era5_ds.CI
cci = ax.contour(E5_ci.lon, E5_ci.lat, E5_ci, ci_levels, transform=data_crs, linestyles="--",
                 colors="#332288")

# add high cloud cover
E5_cc = era5_ds.CC.where(era5_ds.pressure < 55000, drop=True).sum(dim="lev")
ax.contourf(E5_cc.lon, E5_cc.lat, E5_cc, levels=20, transform=data_crs, cmap="Greys", alpha=1)

# plot trajectories - 11 April
header_line = [2]  # header-line of .1 files is always line #2 (counting from 0)
date_h = f"20220411_07"
# get filenames
fname_traj = "traj_CIRR_HALO_" + date_h + ".1"
trajs = np.loadtxt(f"{trajectory_path}/{fname_traj}", dtype="f", skiprows=5)
times = trajs[:, 0]
# generate object to only load specific header line
gen = h.generate_specific_rows(f"{trajectory_path}/{fname_traj}", userows=header_line)
header = np.loadtxt(gen, dtype="str", unpack=True)
header = header.tolist()  # convert to list
# convert to upper char
for j in range(len(header)):
    header[j] = header[j].upper()

print("\tTraj_select.1 could be opened, processing...")

# get the time step of the trajectories # here: manually set
dt = 0.01
traj_single_len = 4320  # int(tmax/dt)
traj_overall_len = int(len(times))
traj_num = int(traj_overall_len / (traj_single_len + 1))  # +1 for the empty line after
# each traj
var_index = header.index(var_name.upper())

for k in range(traj_single_len + 1):
    # reduce to hourly? --> [::60]
    lon = trajs[k * (traj_single_len + 1):(k + 1) * (traj_single_len + 1), 1][::60]
    lat = trajs[k * (traj_single_len + 1):(k + 1) * (traj_single_len + 1), 2][::60]
    var = trajs[k * (traj_single_len + 1):(k + 1) * (traj_single_len + 1), var_index][::60]
    x, y = lon, lat
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=plt_sett[var_name]['cmap_sel'], norm=plt_sett[var_name]['norm'],
                        alpha=1, transform=data_crs)
    # Set the values used for colormapping
    lc.set_array(var)
    if int(traj_num) == 1:
        lc.set_linewidth(5)
    elif int(traj_num) >= 2:
        lc.set_linewidth(1)
    line = ax.add_collection(lc)

plt.colorbar(line, ax=ax, pad=0.01,
             ticks=np.arange(-120, 0.1, 12)).set_label(label=plt_sett[var_name]['label'], size=6)

# plot flight track - 11 April
ins = cat["HALO-AC3"]["HALO"]["GPS_INS"][f"HALO-AC3_HALO_RF17"](storage_options=kwds, **credentials).to_dask()
track_lons, track_lats = ins["lon"], ins["lat"]
ax.scatter(track_lons[::10], track_lats[::10], c="k", alpha=1, marker=".", s=1, zorder=400,
           label='HALO flight track', transform=data_crs, linestyle="solid")

# plot dropsonde locations - 11 April
dropsondes_ds = cat["HALO-AC3"]["HALO"]["DROPSONDES_GRIDDED"][f"HALO-AC3_HALO_RF17"](storage_options=kwds,
                                                                                     **credentials).to_dask()
dropsondes_ds["alt"] = dropsondes_ds.alt / 1000  # convert altitude to km
for i in range(dropsondes_ds.lon.shape[0]):
    launch_time = pd.to_datetime(dropsondes_ds.launch_time[i].values)
    x, y = dropsondes_ds.lon[i].mean().values, dropsondes_ds.lat[i].mean().values
    cross = ax.plot(x, y, "x", color="orangered", markersize=3, label="Dropsonde", transform=data_crs,
                    zorder=450)
    ax.text(x, y, f"{launch_time:%H:%M}", c="k", fontsize=4, transform=data_crs, zorder=500,
            path_effects=[patheffects.withStroke(linewidth=0.25, foreground="white")])

# make legend for flight track and dropsondes - 11 April
handles = [plt.plot([], ls="-", color="k")[0],  # flight track
           cross[0],  # dropsondes
           plt.plot([], ls="--", color="#332288")[0],  # sea ice edge
           plt.plot([], ls="solid", lw=0.7, color="k")[0],  # isobars
           Patch(facecolor="grey")]  # cloud cover
labels = ["HALO flight track", "Dropsonde", "Sea ice edge", "Mean sea level\npressure (hPa)",
          "High cloud cover\nat 12:00 UTC"]
ax.legend(handles=handles, labels=labels, framealpha=1, loc=2)

title = f"11 April 2022"
ax.set_title(title, fontsize=7)
ax.text(0.05, 0.6, "(a)", size=7, transform=ax.transAxes, ha="center", va="center",
        bbox=dict(boxstyle="round", ec="grey", fc="white"))

# plot dropsonde profiles in row 1 and column 2
rh_ice = met.relative_humidity_water_to_relative_humidity_ice(dropsondes_ds.rh, dropsondes_ds.T - 273.15)
labels = [f"{lt.replace('2022-04-11T', '')} UTC" for lt in np.datetime_as_string(rh_ice.launch_time.values, unit="m")]
ax = fig.add_subplot(gs[0, 1])
rh_ice.plot.line(y="alt", alpha=0.5, label=labels, lw=1, ax=ax)
rh_ice.mean(dim="launch_time").plot(y="alt", lw=2, label="Mean", c="k", ax=ax)

# plot vertical line at 100%
ax.axvline(x=100, color="#661100", lw=2)
ax.set_xlabel("Relative Humidity over Ice (%)")
ax.set_ylabel("Altitude (km)")
ax.grid()
ax.legend(bbox_to_anchor=(1, 1.01), loc="upper left")
ax.text(0.1, 0.95, "(b)", size=7, transform=ax.transAxes, ha="center", va="center",
        bbox=dict(boxstyle="round", ec="grey", fc="white"))

# plot trajectories 12 April in second row first column
ax = fig.add_subplot(gs[1, 0], projection=ccrs.NorthPolarStereo())
ax.coastlines(alpha=0.5)
ax.set_xlim((-2000000, 2000000))
ax.set_ylim((-3000000, 500000))
gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=1, color='gray', alpha=0.5,
                  linestyle=':', x_inline=False, y_inline=False, rotate_labels=False)
gl.top_labels = False
gl.right_labels = False
gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 20))
gl.ylocator = mticker.FixedLocator(np.arange(60, 90, 5))

# read in ERA5 data - 12 April
era5_files = [os.path.join(era5_path, f) for f in os.listdir(era5_path) if f"P20220412" in f]
era5_files.sort()
era5_ds = xr.open_mfdataset(era5_files).sel(time=f"2022-04-12T12:00")
# calculate pressure on all model levels
pressure = era5_ds.hyam + era5_ds.hybm * era5_ds.PS * 100
# select only relevant model levels and swap dimension names
era5_ds["pressure"] = pressure.sel(nhym=slice(39, 137)).swap_dims({"nhym": "lev"})

# Plot the surface pressure - 12 April
pressure_levels = np.arange(900, 1125, 5)
E5_press = era5_ds.MSL / 100
cp = ax.contour(E5_press.lon, E5_press.lat, E5_press, levels=pressure_levels, colors='k', linewidths=0.7,
                linestyles='solid', alpha=1, transform=ccrs.PlateCarree())
cp.clabel(fontsize=4, inline=1, inline_spacing=4, fmt='%i', rightside_up=True, use_clabeltext=True)

# add seaice edge - 12 April
ci_levels = [0.8]
E5_ci = era5_ds.CI
cci = ax.contour(E5_ci.lon, E5_ci.lat, E5_ci, ci_levels, transform=ccrs.PlateCarree(), linestyles="--",
                 colors="#332288")

# add high cloud cover - 12 April
E5_cc = era5_ds.CC.where(era5_ds.pressure < 55000, drop=True).sum(dim="lev")
ax.contourf(E5_cc.lon, E5_cc.lat, E5_cc, levels=20, transform=ccrs.PlateCarree(), cmap="Greys")

# plot trajectories - 12 April
header_line = [2]  # header-line of .1 files is always line #2 (counting from 0)
date_h = f"20220412_07"
# get filenames
fname_traj = "traj_CIRR_HALO_" + date_h + ".1"
trajs = np.loadtxt(f"{trajectory_path}/{fname_traj}", dtype="f", skiprows=5)
times = trajs[:, 0]
# generate object to only load specific header line
gen = h.generate_specific_rows(f"{trajectory_path}/{fname_traj}", userows=header_line)
header = np.loadtxt(gen, dtype="str", unpack=True)
header = header.tolist()  # convert to list
# convert to lower char.
for j in range(len(header)):
    header[j] = header[j].upper()  # convert to lower
print("\tTraj_select.1 could be opened, processing...")

# get the time step of the trajectories # here: manually set
dt = 0.01
traj_single_len = 4320  # int(tmax/dt)
traj_overall_len = int(len(times))
traj_num = int(traj_overall_len / (traj_single_len + 1))  # +1 for the empty line after
# each traj

for k in range(traj_single_len + 1):
    # reduce to hourly? --> [::60]
    lon = trajs[k * (traj_single_len + 1):(k + 1) * (traj_single_len + 1), 1][::60]
    lat = trajs[k * (traj_single_len + 1):(k + 1) * (traj_single_len + 1), 2][::60]
    var = trajs[k * (traj_single_len + 1):(k + 1) * (traj_single_len + 1), var_index][::60]
    x, y = lon, lat
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=plt_sett[var_name]['cmap_sel'], norm=plt_sett[var_name]['norm'],
                        alpha=1, transform=ccrs.PlateCarree())
    # Set the values used for colormapping
    lc.set_array(var)
    if int(traj_num) == 1:
        lc.set_linewidth(5)
    elif int(traj_num) >= 2:
        lc.set_linewidth(1)
    line = ax.add_collection(lc)

plt.colorbar(line, ax=ax, pad=0.01,
             ticks=np.arange(-120, 0.1, 12)).set_label(label=plt_sett[var_name]['label'], size=6)

# plot flight track - 12 April
ins = cat["HALO-AC3"]["HALO"]["GPS_INS"][f"HALO-AC3_HALO_RF18"](storage_options=kwds, **credentials).to_dask()
track_lons, track_lats = ins["lon"], ins["lat"]
ax.scatter(track_lons[::10], track_lats[::10], c="k", alpha=1, marker=".", s=1, zorder=400,
           label='HALO flight track', transform=ccrs.PlateCarree(), linestyle="solid")

# plot dropsonde locations - 12 April
dropsondes_ds = cat["HALO-AC3"]["HALO"]["DROPSONDES_GRIDDED"][f"HALO-AC3_HALO_RF18"](storage_options=kwds,
                                                                                     **credentials).to_dask()
dropsondes_ds["alt"] = dropsondes_ds.alt / 1000  # convert altitude to km
for i in range(dropsondes_ds.lon.shape[0]):
    x, y = dropsondes_ds.lon[i].mean().values, dropsondes_ds.lat[i].mean().values
    cross = ax.plot(x, y, "x", color="orangered", markersize=3, label="Dropsonde", transform=data_crs,
                    zorder=450)
# add time to only a selected range of dropsondes
for i in [0, 3, 8, 10, 11, 12, 13]:
    launch_time = pd.to_datetime(dropsondes_ds.launch_time[i].values)
    x, y = dropsondes_ds.lon[i].mean().values, dropsondes_ds.lat[i].mean().values
    ax.text(x, y, f"{launch_time:%H:%M}", color="k", fontsize=4, transform=data_crs, zorder=500,
            path_effects=[patheffects.withStroke(linewidth=0.25, foreground="white")])

# make legend for flight track and dropsondes - 12 April
handles = [plt.plot([], ls="-", color="k")[0],  # flight track
           cross[0],  # dropsondes
           plt.plot([], ls="--", color="#332288")[0],  # sea ice edge
           plt.plot([], ls="solid", lw=0.7, color="k")[0],  # isobars
           Patch(facecolor="grey")]  # cloud cover
labels = ["HALO flight track", "Dropsonde", "Sea ice edge", "Mean sea level\npressure (hPa)",
          "High cloud cover\nat 12:00 UTC"]
ax.legend(handles=handles, labels=labels, framealpha=1, loc=2)

title = f"12 April 2022"
ax.set_title(title, fontsize=7)
ax.text(0.05, 0.6, "(c)", size=7, transform=ax.transAxes, ha="center", va="center",
        bbox=dict(boxstyle="round", ec="grey", fc="white"))

# plot dropsonde profiles in row 2 and column 2 - 12 April
rh_ice = met.relative_humidity_water_to_relative_humidity_ice(dropsondes_ds.rh, dropsondes_ds.T - 273.15)
labels = [f"{lt.replace('2022-04-12T', '')} UTC" for lt in np.datetime_as_string(rh_ice.launch_time.values, unit="m")]
ax = fig.add_subplot(gs[1, 1])
ax.set_prop_cycle('color', [plt.cm.tab20(i) for i in np.linspace(0, 1, 14)])
rh_ice.plot.line(y="alt", alpha=0.5, label=labels, lw=1, ax=ax)
rh_ice.mean(dim="launch_time").plot(y="alt", lw=2, label="Mean", c="k", ax=ax)

# plot vertical line at 100%
ax.axvline(x=100, color="#661100", lw=2)
ax.set_xlabel("Relative Humidity over Ice (%)")
ax.set_ylabel("Altitude (km)")
ax.grid()
ax.legend(bbox_to_anchor=(1, 1.01), loc="upper left")
ax.text(0.1, 0.95, "(d)", size=7, transform=ax.transAxes, ha="center", va="center",
        bbox=dict(boxstyle="round", ec="grey", fc="white"))

plt.tight_layout()
figname = f"{plot_path}/HALO-AC3_trajectories_dropsonde_plot_overview.pdf"
plt.savefig(figname, dpi=300, bbox_inches='tight')
print(f"saved as: {figname}")
plt.close()
