#!/usr/bin/env python
"""Case Study for RF07 2022-03-20

Joint flight of HALO, P5 and P6

*author*: Johannes Röttenbacher
"""

# %% import modules
import pylim.helpers as h
from pylim.halo_ac3 import coordinates
import ac3airborne
from ac3airborne.tools.get_amsr2_seaice import get_amsr2_seaice
import os
import xarray as xr
import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
import holoviews as hv
from holoviews import opts
import cartopy
import cartopy.crs as ccrs

hv.extension("bokeh")

# %% set paths
campaign = "halo-ac3"
date = "20220320"
halo_key = "RF07"
P5_key = "RF01"
halo_flight = f"HALO-AC3_{date}_HALO_{halo_key}"
P5_flight = f"HALO-AC3_{date}_P5_{P5_key}"

plot_path = f"{h.get_path('plot', halo_flight, campaign)}/{halo_flight}"
halo_smart_path = h.get_path("calibrated", halo_flight, campaign)
P5_smart_path = h.get_path("P5_smart", P5_flight, campaign)
swir_file = f"HALO-AC3_HALO_SMART_Fdw_SWIR_{date}_{halo_key}.nc"
vnir_file = f"HALO-AC3_HALO_SMART_Fdw_VNIR_{date}_{halo_key}.nc"

# flight tracks from halo-ac3 cloud
kwds = {'simplecache': dict(cache_storage='E:/HALO-AC3/cloud', same_names=True)}
credentials = {"user": os.environ.get("AC3_CLOUD_USER"), "password": os.environ.get("AC3_CLOUD_PASSWORD")}
cat = ac3airborne.get_intake_catalog()
P5_track = cat["HALO-AC3"]["P5"]["GPS_INS"][f"HALO-AC3_P5_{P5_key}"](storage_options=kwds, **credentials).to_dask()
HALO_track = cat["HALO-AC3"]["HALO"]["GPS_INS"]["HALO-AC3_HALO_RF11"](storage_options=kwds, **credentials).to_dask()

# %% read in HALO smart data
halo_swir = xr.open_dataset(f"{halo_smart_path}/{swir_file}")
halo_vnir = xr.open_dataset(f"{halo_smart_path}/{vnir_file}")
# rename variables to merge them
halo_vnir = halo_vnir.rename(dict(Fdw_VNIR="Fdw"))
halo_swir = halo_swir.rename(dict(Fdw_SWIR="Fdw"))
# merge datasets
halo_all = xr.merge([halo_swir, halo_vnir])
halo_all["Fdw_bb"] = halo_all["Fdw_VNIR_bb"] + halo_all["Fdw_SWIR_bb"]

# %% read in P5 SMART data
P5_fdw_vnir = sio.readsav(f"{P5_smart_path}/fdw_vis_all.sav")
P5_fdw_swir = sio.readsav(f"{P5_smart_path}/fdw_nir_all.sav")
P5_fup_vnir = sio.readsav(f"{P5_smart_path}/fup_vis_all.sav")
P5_fup_swir = sio.readsav(f"{P5_smart_path}/fup_nir_all.sav")

# %% convert P5 data to xarray Dataset
# vnir and swir have different time axis
vnir_time = pd.to_datetime(P5_fdw_vnir["time_fdw_vis_day"], unit="h", origin=date)
swir_time = pd.to_datetime(P5_fdw_swir["time_fdw_nir_day"], unit="h", origin=date)
# fdw and fup have the same wavelengths
vnir_wl = P5_fdw_vnir["wl_vis_int"]
swir_wl = P5_fdw_swir["wl_nir_int"]
P5_vnir = xr.Dataset(data_vars={"Fdw": (["wavelength", "time"], P5_fdw_vnir["fdw_vis_day"]),
                                "Fup": (["wavelength", "time"], P5_fup_vnir["fup_vis_day"])},
                     coords={"wavelength": (["wavelength"], vnir_wl),
                             "time": vnir_time})
P5_swir = xr.Dataset(data_vars={"Fdw": (["wavelength", "time"], P5_fdw_swir["fdw_nir_day"]),
                                "Fup": (["wavelength", "time"], P5_fup_swir["fup_nir_day"])},
                     coords={"wavelength": (["wavelength"], swir_wl),
                             "time": swir_time})
P5_all = xr.merge([P5_vnir, P5_swir])

# %% filter high roll angles
P5_track_inp = P5_track.interp(time=P5_vnir.time)
HALO_track_inp = HALO_track.interp(time=halo_all.time)
P5_vnir_filtered = P5_vnir.where(np.abs(P5_track_inp["roll"]) < 2)
halo_all_filtered = halo_all.where(np.abs(HALO_track_inp["roll"]) < 2)

# %% reduce size of track files
P5_track = P5_track.resample(time="1Min").asfreq()
HALO_track = HALO_track.resample(time="1Min").asfreq()

# %% find overlapping flight paths
fig, ax = plt.subplots()
P5_track.plot.scatter(x="lon", y="lat", ax=ax, label="P5", c=P5_track.time)
HALO_track.plot.scatter(x="lon", y="lat", ax=ax, label="HALO", c=HALO_track.time)
ax.legend()
ax.set_ylim(75, 82)
ax.set_xlim(-2, 15)
plt.show()
plt.close()
# idea: pre filter HALO flight track by sight -> only allow track points in a certain box around the P5 flight track
# # %% try with holoviews now
# P5_df = hv.Dataset(P5_track)
# HALO_df = hv.Dataset(HALO_track)
# # annotate data
# layout = hv.Scatter(P5_df, vdims=["lat", "lon"], label="P5").opts(color="green") \
#          * hv.Scatter(HALO_df, vdims=["lat", "lon"], label="HALO").opts(color="red")
# layout.opts(
#     opts.Scatter(responsive=True, height=350, show_grid=True, tools=["hover"],
#                fontsize={'title': 16, 'labels': 14, 'xticks': 12, 'yticks': 12, 'legend': 12}),
#     opts.Overlay(legend_position="right", legend_offset=(0, 100))
# )
# layout.opts(title=f"{date} P5 and HALO track")
# figname = f"{plot_path}/HALO-AC3_HALO_P5_tracks_{date}_{halo_key}.html"
# hv.save(layout, figname)

# %% plot SMART measurements together - no cirrus
x_sel_p5 = (pd.Timestamp(2022, 3, 20, 12, 25), pd.Timestamp(2022, 3, 20, 14, 5))
x_sel_halo = (pd.Timestamp(2022, 3, 20, 11, 40), pd.Timestamp(2022, 3, 20, 12, 5))
h.set_cb_friendly_colors()
plt.rc('font', size=20)
plt.rc('lines', linewidth=3)
fig, axs = plt.subplots(nrows=2, figsize=(13, 9))

# first row -> P5
ax = axs[0]
ax.plot(P5_vnir_filtered["time"], P5_vnir_filtered["Fdw"].sel(wavelength=600), label=r"P5 SMART F$_\downarrow$ 600nm")
ax.set_xlim(x_sel_p5[0], x_sel_p5[1])
ax.set_ylim(0, 0.4)
h.set_xticks_and_xlabels(ax, x_sel_p5[1] - x_sel_p5[0])
ax.set_title(f"Comparison of collocated SMART measurements P5 and HALO\n {date} HALO {halo_key}, P5 {P5_key}")

# second row -> HALO
ax = axs[1]
ax.plot(halo_all_filtered["time"], halo_all_filtered["Fdw"].sel(wavelength=600, method="nearest"),
        label=r"HALO SMART F$_\downarrow$ 600nm")
ax.set_xlim(x_sel_halo[0], x_sel_halo[1])
ax.set_ylim(0, 0.4)
h.set_xticks_and_xlabels(ax, x_sel_halo[1] - x_sel_halo[0])
ax.set_xlabel("Time (UTC)")

# both axis
for ax in axs:
    ax.legend()
    ax.set_ylabel("Spectral Solar Downward \nIrradiance (W$\,$m$^{-2}\,$nm$^{-1}$)")
    ax.grid()

# plt.show()
figname = f"{plot_path}/HALO-AC3_HALO_P5_SMART_Fdw_comparison_nocirrus_{date}_{halo_key}.png"
plt.savefig(figname, dpi=300)
plt.close()

# %% plot SMART measurements together - with cirrus
x_sel_p5 = (pd.Timestamp(2022, 3, 20, 14, 0), pd.Timestamp(2022, 3, 20, 15, 5))
x_sel_halo = (pd.Timestamp(2022, 3, 20, 13, 55), pd.Timestamp(2022, 3, 20, 14, 15))
h.set_cb_friendly_colors()
plt.rc('font', size=20)
plt.rc('lines', linewidth=3)
fig, axs = plt.subplots(nrows=2, figsize=(13, 9))

# first row -> P5
ax = axs[0]
ax.plot(P5_vnir_filtered["time"], P5_vnir_filtered["Fdw"].sel(wavelength=600), label=r"P5 SMART F$_\downarrow$ 600nm")
ax.set_xlim(x_sel_p5[0], x_sel_p5[1])
ax.set_ylim(0, 0.4)
h.set_xticks_and_xlabels(ax, x_sel_p5[1] - x_sel_p5[0])
ax.set_title(f"Comparison of collocated SMART measurements P5 and HALO\n {date} HALO {halo_key}, P5 {P5_key}")

# second row -> HALO
ax = axs[1]
ax.plot(halo_all_filtered["time"], halo_all_filtered["Fdw"].sel(wavelength=600, method="nearest"),
        label=r"HALO SMART F$_\downarrow$ 600nm")
ax.set_xlim(x_sel_halo[0], x_sel_halo[1])
ax.set_ylim(0, 0.4)
h.set_xticks_and_xlabels(ax, x_sel_halo[1] - x_sel_halo[0])
ax.set_xlabel("Time (UTC)")

# both axis
for ax in axs:
    ax.legend()
    ax.set_ylabel("Spectral Solar Downward \nIrradiance (W$\,$m$^{-2}\,$nm$^{-1}$)")
    ax.grid()

# plt.show()
figname = f"{plot_path}/HALO-AC3_HALO_P5_SMART_Fdw_comparison_cloudy_{date}_{halo_key}.png"
plt.savefig(figname, dpi=300)
plt.close()

# %% plot map of both flight tracks
orig_map = plt.cm.get_cmap('Blues')  # getting the original colormap using cm.get_cmap() function
reversed_map = orig_map.reversed()  # reversing the original colormap using reversed() function
data_crs = ccrs.PlateCarree()
plt.rc('font', size=20)

fig, ax = plt.subplots(figsize=(6.5, 9), subplot_kw={"projection": ccrs.NorthPolarStereo()})
ax.coastlines()
ax.add_feature(cartopy.feature.BORDERS)
ax.set_extent([-5, 15, 75, 82], crs=data_crs)
gl = ax.gridlines(crs=data_crs, draw_labels=True, x_inline=False, y_inline=False)
gl.bottom_labels = False
gl.left_labels = False

# add sea ice concentration
seaice = get_amsr2_seaice(f"{(pd.to_datetime(date) - pd.Timedelta(days=0)):%Y%m%d}")
seaice = seaice.seaice
ax.pcolormesh(seaice.lon, seaice.lat, seaice, transform=data_crs, cmap=reversed_map)

# Longyearbyen
x_longyear, y_longyear = coordinates["Longyearbyen"]
ax.plot(x_longyear, y_longyear, '.', color="#117733", markersize=12, transform=data_crs)
ax.text(x_longyear + 0.1, y_longyear + 0.1, "Longyearbyen", fontsize=16, transform=data_crs,
        path_effects=[patheffects.withStroke(linewidth=0.5, foreground="white")])

# plot flight track P5
points_P5 = ax.scatter(P5_track["lon"], P5_track["lat"], s=10, c="#117733", transform=data_crs, label="P5 Track")
# plot flight track HALO
points_HALO = ax.scatter(HALO_track["lon"], HALO_track["lat"], s=10, c="red", transform=data_crs, label="HALO track")
ax.legend(loc=2)

# add a waypoint every 15 minutes and add time stamp
for i in range(1, len(P5_track.lon), 30):
    # P5 points
    lon, lat = P5_track.lon[i], P5_track.lat[i]
    time_str = str(P5_track.time[i].dt.strftime("%H:%M").values)
    ax.plot(lon, lat, 'x', transform=data_crs, c="red")
    ax.text(lon, lat, time_str, fontsize=14, transform=data_crs, c="#117733",
            path_effects=[patheffects.withStroke(linewidth=1, foreground="white")])
for i in range(1, len(HALO_track.lon), 15):
    # HALO points
    lon, lat = HALO_track.lon[i], HALO_track.lat[i]
    if (lon > -5) and (lat > 75):
        time_str = str(HALO_track.time[i].dt.strftime("%H:%M").values)
        ax.plot(lon, lat, 'x', transform=data_crs, c="red")
        ax.text(lon, lat, time_str, fontsize=14, transform=data_crs, c="red",
                path_effects=[patheffects.withStroke(linewidth=1, foreground="white")])

plt.tight_layout(pad=0.1)
# plt.show()
figname = f"{plot_path}/HALO-AC3_HALO_P5_track_{date}_{halo_key}.png"
plt.savefig(figname, dpi=300)
plt.close()

