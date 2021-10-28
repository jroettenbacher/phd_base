#!/usr/bin/env python
"""Case studies for Cirrus-HL
* 29.06.2021: cirrus over Atlantic west and north of Iceland
author: Johannes Röttenbacher
"""

# %% module import
import numpy as np
from smart import get_path
import logging
from bahamas import plot_props, read_bahamas
from bacardi import read_bacardi_raw, fdw_attitude_correction
from libradtran import read_libradtran
from cirrus_hl import stop_over_locations, coordinates
import os
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
from scipy.interpolate import interp1d

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

# %% 20210629
print(20210629)

# %% set paths
flight = "Flight_20210629a"
bahamas_dir = get_path("bahamas", flight)
bacardi_dir = get_path("bacardi", flight)
smart_dir = get_path("calibrated", flight)
sat_image = "/projekt_agmwend/data/Cirrus_HL/01_Flights/Flight_20210629a/satellite/snapshot-2021-06-29T00_00_00Z.tiff"
if os.getcwd().startswith("C:"):
    outpath = f"C:/Users/Johannes/Documents/Doktor/campaigns/CIRRUS-HL/case_studies/{flight}"
else:
    outpath = f"/projekt_agmwend/home_rad/jroettenbacher/case_studies/{flight}"
make_dir(outpath)
start_dt = pd.Timestamp(2021, 6, 29, 10, 10)
end_dt = pd.Timestamp(2021, 6, 29, 11, 54)
below_cloud = (start_dt, pd.Timestamp(2021, 6, 29, 10, 15))
in_cloud = (pd.Timestamp(2021, 6, 29, 10, 15), pd.Timestamp(2021, 6, 29, 11, 54))
above_cloud = (pd.Timestamp(2021, 6, 29, 11, 54), pd.Timestamp(2021, 6, 29, 12, 5))

# %% find bahamas file and read in bahamas data
file = [f for f in os.listdir(bahamas_dir) if f.endswith(".nc")][0]
bahamas = read_bahamas(flight)

# %% read in satellite picture
sat_ds = rasterio.open(sat_image)

# %% select flight sections for plotting
bahamas_belowcloud = (below_cloud[0] < bahamas.TIME) & (bahamas.TIME < below_cloud[1])
bahamas_abovecloud = (above_cloud[0] < bahamas.TIME) & (bahamas.TIME < above_cloud[1])
bahamas_incloud = (in_cloud[0] < bahamas.TIME) & (bahamas.TIME < in_cloud[1])

# %% select further points to plot
x_edmo, y_edmo = coordinates["EDMO"]
airport = stop_over_locations[flight] if flight in stop_over_locations else None
x2, y2 = coordinates[airport]
torshavn_x, torshavn_y = coordinates["Torshavn"]

# %% select position and time data and set extent
lon, lat, altitude, times = bahamas["IRS_LON"], bahamas["IRS_LAT"], bahamas["IRS_ALT"], bahamas["TIME"]
pad = 2
llcrnlat = lat.min(skipna=True) - pad
llcrnlon = lon.min(skipna=True) - pad
urcrnlat = lat.max(skipna=True) + pad
urcrnlon = lon.max(skipna=True) + pad
extent = [llcrnlon, urcrnlon, llcrnlat, urcrnlat]
font = {'weight': 'bold', 'size': 26}
matplotlib.rc('font', **font)
# get plot properties
props = plot_props[flight]
set_cb_friendly_colors()

# %% plot bahamas map with highlighted below and above cloud sections and sat image
fig, ax = plt.subplots(figsize=(11, 9), subplot_kw={"projection": ccrs.PlateCarree()})
# ax.stock_img()
show(sat_ds, ax=ax)
ax.coastlines(linewidth=3)
ax.add_feature(cartopy.feature.BORDERS, linewidth=3)
ax.set_extent(extent)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
gl.bottom_labels = False
gl.left_labels = False

# plot flight track
points = ax.plot(lon, lat, c="orange", linewidth=6)
# plot in and below cloud points for case study
lon_incloud = bahamas["IRS_LON"].where(bahamas_incloud, drop=True)
lat_incloud = bahamas["IRS_LAT"].where(bahamas_incloud, drop=True)
ax.plot(lon_incloud, lat_incloud, c="cornflowerblue", linewidth=6, label="inside cloud")
lon_below = bahamas["IRS_LON"].where(bahamas_belowcloud, drop=True)
lat_below = bahamas["IRS_LAT"].where(bahamas_belowcloud, drop=True)
ax.plot(lon_below, lat_below, c="green", linewidth=6, label="below cloud")
lon_above = bahamas["IRS_LON"].where(bahamas_abovecloud, drop=True)
lat_above = bahamas["IRS_LAT"].where(bahamas_abovecloud, drop=True)
ax.plot(lon_above, lat_above, c="red", linewidth=6, label="above cloud")

# plot a way point every 15 minutes = 9000 seconds with a time stamp next to it
for long, lati, time_stamp in zip(lon[9000::9000], lat[9000::9000], times[9000::9000]):
    ax.annotate(time_stamp.dt.strftime("%H:%M").values, (long, lati), fontsize=16,
                path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])
    ax.plot(long, lati, '.k', markersize=10)

# plot points with labels and white line around text
ax.plot(x_edmo, y_edmo, 'ok')
ax.text(x_edmo + 0.1, y_edmo + 0.1, "EDMO", fontsize=22,
        path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])
ax.plot(x2, y2, 'ok')
ax.text(x2 + 0.1, y2 + 0.1, airport, fontsize=22,
        path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])
ax.plot(torshavn_x, torshavn_y, 'ok')
ax.text(torshavn_x + 0.1, torshavn_y + 0.1, "Torshavn", fontsize=22,
        path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])

ax.legend(loc=3, fontsize=18, markerscale=6)
plt.tight_layout(pad=0.1)
fig_name = f"{outpath}/{flight}_bahamas_track.png"
# plt.show()
plt.savefig(fig_name, dpi=100)
log.info(f"Saved {fig_name}")
plt.close()

# %% plot bahamas data to check for clouds
ylabels = ["Static Air\nTemperature (K)", "Relative \nHumidity (%)", "Static \nPressure (hPa)"]
fig, axs = plt.subplots(nrows=3)
bahamas.TS.plot(ax=axs[0])
axs[0].axhline(y=235, color="r", linestyle="--", label="$235\,$K")
bahamas.RELHUM.plot(ax=axs[1])
bahamas.PS.plot(ax=axs[2])
axs[2].invert_yaxis()
timedelta = pd.to_datetime(bahamas.time[-1].values) - pd.to_datetime(bahamas.time[0].values)

for ax, ylabel in zip(axs, ylabels):
    ax.set_ylabel(ylabel)
    ax.grid()
    set_xticks_and_xlabels(ax, timedelta)
    ax.fill_between(bahamas.time, 0, 1, where=((below_cloud[0] < bahamas.time) & (bahamas.time < below_cloud[1])),
                    transform=ax.get_xaxis_transform(), label="below cloud", color="green", alpha=0.5)
    ax.fill_between(bahamas.time, 0, 1, where=((in_cloud[0] < bahamas.time) & (bahamas.time < in_cloud[1])),
                    transform=ax.get_xaxis_transform(), label="inside cloud", color="grey", alpha=0.5)
    ax.fill_between(bahamas.time, 0, 1, where=((above_cloud[0] < bahamas.time) & (bahamas.time < above_cloud[1])),
                    transform=ax.get_xaxis_transform(), label="above cloud", color="red", alpha=0.5)

axs[2].set_xlabel("Time (UTC)")
for ax in axs[0:2]:
    ax.set_xlabel("")
    ax.set_xticklabels("")

axs[0].legend()
axs[0].set_ylim((150, 300))
# axs[2].legend(bbox_to_anchor=(0.05, 0), loc="lower left", bbox_transform=fig.transFigure, ncol=4)
# plt.subplots_adjust(bottom=0.2)
plt.tight_layout()
# plt.show()
plt.savefig(f"{outpath}/{flight}_bahamas_overview.png", dpi=100)
plt.close()

# %% read in libradtran and bacardi files
libradtran_file = "BBR_Fdn_clear_sky_Flight_20210629a_R0_ds_high.dat"
libradtran_file_ter = "BBR_Fdn_clear_sky_Flight_20210629a_R0_ds_high_ter.dat"
bacardi_file = "CIRRUS_HL_F05_20210629a_ADLR_BACARDI_BroadbandFluxes_R0.nc"
bbr_sim = read_libradtran(flight, libradtran_file)
bbr_sim_ter = read_libradtran(flight, libradtran_file_ter)
bacardi_ds = xr.open_dataset(f"{bacardi_dir}/{bacardi_file}")

# %% select flight sections for libRadtran simulations and BACARDI measurements
bbr_belowcloud = ((below_cloud[0] < bbr_sim.index) & (bbr_sim.index < below_cloud[1]))
bbr_ter_belowcloud = ((below_cloud[0] < bbr_sim_ter.index) & (bbr_sim_ter.index < below_cloud[1]))
bacardi_belowcloud = ((below_cloud[0] < bacardi_ds.time) & (bacardi_ds.time < below_cloud[1]))
bbr_abovecloud = ((above_cloud[0] < bbr_sim.index) & (bbr_sim.index < above_cloud[1]))
bbr_ter_abovecloud = ((above_cloud[0] < bbr_sim_ter.index) & (bbr_sim_ter.index < above_cloud[1]))
bacardi_abovecloud = ((above_cloud[0] < bacardi_ds.time) & (bacardi_ds.time < above_cloud[1]))

# %% get mean values for flight sections
bbr_sim[bbr_belowcloud].mean()
bbr_sim_ter[bbr_ter_belowcloud].mean()
bacardi_ds.sel(time=bacardi_belowcloud).mean()
bbr_sim[bbr_abovecloud].mean()
bbr_sim_ter[bbr_ter_abovecloud].mean()
bacardi_ds.sel(time=bacardi_abovecloud).mean()
# %% plot libradtran simulations together with BACARDI measurements (solar + terrestrial)
plt.rcdefaults()
set_cb_friendly_colors()
plt.rc('font', size=20)
plt.rc('lines', linewidth=3)

x_sel = (pd.Timestamp(2021, 6, 29, 10), pd.Timestamp(2021, 6, 29, 12, 15))
fig, ax = plt.subplots(figsize=(13, 9))
# solar radiation
bacardi_ds.F_up_solar.plot(x="time", label="F_up BACARDI", ax=ax, c="#6699CC", ls="-")
bacardi_ds.F_down_solar.plot(x="time", label="F_down BACARDI", ax=ax, c="#117733", ls="-")
bbr_sim.plot(y="F_up", ax=ax, label="F_up libRadtran", c="#6699CC", ls="--",
             path_effects=[patheffects.withStroke(linewidth=6, foreground="k")])
bbr_sim.plot(y="F_dw", ax=ax, ylabel="Broadband irradiance (W$\,$m$^{-2}$)", label="F_down libRadtran",
             c="#117733", ls="--", path_effects=[patheffects.withStroke(linewidth=6, foreground="k")])
# terrestrial radiation
bacardi_ds.F_up_terrestrial.plot(x="time", label="F_up BACARDI", ax=ax, c="#CC6677", ls="-")
bacardi_ds.F_down_terrestrial.plot(x="time", label="F_down BACARDI", ax=ax, c="#f89c20", ls="-")
bbr_sim_ter.plot(y="F_up", ax=ax, label="F_up libRadtran", c="#CC6677", ls="--",
                 path_effects=[patheffects.withStroke(linewidth=6, foreground="k")])
bbr_sim_ter.plot(y="F_dw", ax=ax, ylabel="Broadband irradiance (W$\,$m$^{-2}$)", label="F_down libRadtran",
                 c="#f89c20", ls="--", path_effects=[patheffects.withStroke(linewidth=6, foreground="k")])
ax.set_xlabel("Time (UTC)")
ax.set_xlim(x_sel)
set_xticks_and_xlabels(ax, x_sel[1] - x_sel[0])
ax.grid()
# ax.fill_between(bbr_sim.index, 0, 1, where=((start_dt < bbr_sim.index) & (bbr_sim.index < end_dt)),
#                 transform=ax.get_xaxis_transform(), label="Case Study", color="grey")
ax.fill_between(bbr_sim.index, 0, 1, where=bbr_belowcloud,
                transform=ax.get_xaxis_transform(), label="below cloud", color="green", alpha=0.5)
ax.fill_between(bbr_sim.index, 0, 1, where=((in_cloud[0] < bbr_sim.index) & (bbr_sim.index < in_cloud[1])),
                transform=ax.get_xaxis_transform(), label="inside cloud", color="grey", alpha=0.5)
ax.fill_between(bbr_sim.index, 0, 1, where=bbr_abovecloud,
                transform=ax.get_xaxis_transform(), label="above cloud", color="red", alpha=0.5)
handles, labels = ax.get_legend_handles_labels()
legend_column_headers = ["Solar", "Terrestrial"]
handles.insert(0, Patch(color='none', label=legend_column_headers[0]))
handles.insert(5, Patch(color='none', label=legend_column_headers[1]))
# add dummy legend entries to get the right amount of rows per column
handles.append(Patch(color='none', label=""))
handles.append(Patch(color='none', label=""))
ax.legend(handles=handles, bbox_to_anchor=(0.1, 0), loc="lower left", bbox_transform=fig.transFigure, ncol=3)
plt.subplots_adjust(bottom=0.4)
plt.tight_layout()
# plt.show()
plt.savefig(f"{outpath}/{flight}_bacardi_libradtran_broadband_irradiance.png", dpi=100)
plt.close()

# %% plot libradtran simulations together with BACARDI measurements (terrestrial)
x_sel = (pd.Timestamp(2021, 6, 29, 9), pd.Timestamp(2021, 6, 29, 13))
fig, ax = plt.subplots()
bacardi_ds.F_up_terrestrial.plot(x="time", label="F_up BACARDI", ax=ax)
bacardi_ds.F_down_terrestrial.plot(x="time", label="F_dw BACARDI", ax=ax)
bbr_sim_ter.plot(y="F_dw", ax=ax, ylabel="Broadband irradiance (W$\,$m$^{-2}$)", label="F_dw libRadtran")
bbr_sim_ter.plot(y="F_up", ax=ax, label="F_up libRadtran")
ax.set_xlabel("Time (UTC)")
ax.set_xlim(x_sel)
set_xticks_and_xlabels(ax, x_sel[1] - x_sel[0])
ax.grid()
# ax.fill_between(bbr_sim.index, 0, 1, where=((start_dt < bbr_sim.index) & (bbr_sim.index < end_dt)),
#                 transform=ax.get_xaxis_transform(), label="Case Study", color="grey")
ax.fill_between(bbr_sim.index, 0, 1, where=bbr_belowcloud,
                transform=ax.get_xaxis_transform(), label="below cloud", color="green", alpha=0.5)
ax.fill_between(bbr_sim.index, 0, 1, where=((in_cloud[0] < bbr_sim.index) & (bbr_sim.index < in_cloud[1])),
                transform=ax.get_xaxis_transform(), label="inside cloud", color="grey", alpha=0.5)
ax.fill_between(bbr_sim.index, 0, 1, where=bbr_abovecloud,
                transform=ax.get_xaxis_transform(), label="above cloud", color="red", alpha=0.5)
ax.legend(bbox_to_anchor=(0.1, 0), loc="lower left", bbox_transform=fig.transFigure, ncol=3)
plt.subplots_adjust(bottom=0.3)
plt.tight_layout()
# plt.show()
plt.savefig(f"{outpath}/{flight}_bacardi_libradtran_broadband_irradiance_terrestrial.png", dpi=100)
plt.close()

# %% read in SMART data
smart_files = [f for f in os.listdir(smart_dir)]
smart_files.sort()
fdw_swir = smart.read_smart_cor(smart_dir, smart_files[2])
fdw_vnir = smart.read_smart_cor(smart_dir, smart_files[3])
fup_swir = smart.read_smart_cor(smart_dir, smart_files[4])
fup_vnir = smart.read_smart_cor(smart_dir, smart_files[5])

# %% average smart spectra over different flight sections
below_cloud_mean_fdw_vnir = fdw_vnir[below_cloud[0]:below_cloud[1]].mean()
below_cloud_mean_fup_vnir = fup_vnir[below_cloud[0]:below_cloud[1]].mean()
below_cloud_mean_fdw_swir = fdw_swir[below_cloud[0]:below_cloud[1]].mean()
below_cloud_mean_fup_swir = fup_swir[below_cloud[0]:below_cloud[1]].mean()

above_cloud_mean_fdw_vnir = fdw_vnir[above_cloud[0]:above_cloud[1]].mean()
above_cloud_mean_fup_vnir = fup_vnir[above_cloud[0]:above_cloud[1]].mean()
above_cloud_mean_fdw_swir = fdw_swir[above_cloud[0]:above_cloud[1]].mean()
above_cloud_mean_fup_swir = fup_swir[above_cloud[0]:above_cloud[1]].mean()

# %% get pixel to wavelength mapping for each spectrometer
pixel_wl_dict = dict()
for filename in smart_files[2:]:
    date_str, channel, direction = smart.get_info_from_filename(filename)
    name = f"{direction}_{channel}"
    spectrometer = smart.lookup[name]
    pixel_wl_dict[name.casefold()] = smart.read_pixel_to_wavelength(get_path("pixel_wl"), spectrometer)

# %% prepare data frame for plotting VNIR
plot_fup_vnir = pixel_wl_dict["fup_vnir"]
plot_fup_vnir["fup_below_cloud"] = below_cloud_mean_fup_vnir.reset_index(drop=True)
plot_fup_vnir["fup_above_cloud"] = above_cloud_mean_fup_vnir.reset_index(drop=True)
plot_fdw_vnir = pixel_wl_dict["fdw_vnir"]
plot_fdw_vnir["fdw_below_cloud"] = below_cloud_mean_fdw_vnir.reset_index(drop=True)
plot_fdw_vnir["fdw_above_cloud"] = above_cloud_mean_fdw_vnir.reset_index(drop=True)

# filter wrong calibrated wavelengths
min_wl, max_wl = 385, 900
plot_fup_vnir = plot_fup_vnir[plot_fup_vnir["wavelength"].between(min_wl, max_wl)]
plot_fdw_vnir = plot_fdw_vnir[plot_fdw_vnir["wavelength"].between(min_wl, max_wl)]

# %% prepare data frame for plotting SWIR
plot_fup_swir = pixel_wl_dict["fup_swir"]
plot_fup_swir["fup_below_cloud"] = below_cloud_mean_fup_swir.reset_index(drop=True)
plot_fup_swir["fup_above_cloud"] = above_cloud_mean_fup_swir.reset_index(drop=True)
plot_fdw_swir = pixel_wl_dict["fdw_swir"]
plot_fdw_swir["fdw_below_cloud"] = below_cloud_mean_fdw_swir.reset_index(drop=True)
plot_fdw_swir["fdw_above_cloud"] = above_cloud_mean_fdw_swir.reset_index(drop=True)

# %% merge VNIR and SWIR data
plot_fup = pd.concat([plot_fup_vnir, plot_fup_swir], ignore_index=True)
plot_fdw = pd.concat([plot_fdw_vnir, plot_fdw_swir], ignore_index=True)

# %% sort dataframes by wavelength
plot_fup.sort_values(by="wavelength", inplace=True)
plot_fdw.sort_values(by="wavelength", inplace=True)

# %% remove 800 - 950 nm from fup -> calibration problem
plot_fup.iloc[:, 2:] = plot_fup.iloc[:, 2:].where(~plot_fup["wavelength"].between(850, 950), np.nan)

# %% calculate albedo below and above cloud
albedo = plot_fup.loc[:, ("pixel", "wavelength")].copy()
albedo["albedo_below_cloud"] = np.abs(plot_fup["fup_below_cloud"] / plot_fdw["fdw_below_cloud"])
albedo["albedo_above_cloud"] = np.abs(plot_fup["fup_above_cloud"] / plot_fdw["fdw_above_cloud"])
albedo = albedo.rename(columns={"fup_below_cloud": "albedo_below_cloud", "fup_above_cloud": "albedo_above_cloud"})
albedo = albedo[albedo["wavelength"] < 2180]

# %% plot averaged spectra F_up and F_dw
plt.rcParams.update({'font.size': 14})
set_cb_friendly_colors()
fig, axs = plt.subplots(figsize=(10, 8), nrows=3)
plot_fup.plot(x='wavelength', y='fup_below_cloud', ax=axs[0], label="F_up below cloud", linewidth=2)
plot_fup.plot(x='wavelength', y='fup_above_cloud', ax=axs[0], label="F_up above cloud", linewidth=2)
# axs[0].fill_between(plot_fup.wavelength, 0.1, 0.6, where=(plot_fup.wavelength.between(800, 1000)),
#                     label="Calibration offset", color="grey")
axs[0].set_ylabel("Irradiance (W$\\,$m$^{-2}\\,$nm$^{-1}$)")
axs[0].set_xlabel("")
axs[0].grid()
axs[0].legend()

# plot f_dw
plot_fdw.plot(x='wavelength', y='fdw_below_cloud', ax=axs[1], label="F_down below cloud", linewidth=2)
plot_fdw.plot(x='wavelength', y='fdw_above_cloud', ax=axs[1], label="F_down above cloud", linewidth=2)
axs[1].set_ylabel("Irradiance (W$\\,$m$^{-2}\\,$nm$^{-1}$)")
axs[1].set_xlabel("")
axs[1].grid()
axs[1].legend()

# plot albedo
albedo.plot(x='wavelength', y='albedo_below_cloud', ax=axs[2], label="Albedo below cloud", linewidth=2)
albedo.plot(x='wavelength', y='albedo_above_cloud', ax=axs[2], label="Albedo above cloud", linewidth=2)
axs[2].set_ylabel("Albedo")
axs[2].set_xlabel("")
axs[2].grid()
axs[2].legend()
axs[2].set_ylim((0, 1))

# fig.supylabel("Irradiance (W$\\,$m$^{-2}\\,$nm$^{-1}$)")
fig.supxlabel("Wavelength (nm)")
plt.tight_layout(pad=0.5)
# plt.show()
plt.savefig(f"{outpath}/{flight}_SMART_average_spectra_albedo.png", dpi=100)
plt.close()

# %% 20210625 - Radiation Square
print("20210625 - Radiation Square")
# 90° = W, 180° = S, usw.
rs_start = pd.Timestamp(2021, 6, 25, 11, 45)
rs_end = pd.Timestamp(2021, 6, 25, 12, 27)

# %% set paths
flight = "Flight_20210625a"
bahamas_dir = get_path("bahamas", flight)
bacardi_dir = get_path("bacardi", flight)
smart_dir = get_path("calibrated", flight)
sat_dir = get_path("satellite", flight)
if os.getcwd().startswith("C:"):
    outpath = f"C:/Users/Johannes/Documents/Doktor/campaigns/CIRRUS-HL/case_studies/{flight}"
else:
    outpath = f"/projekt_agmwend/home_rad/jroettenbacher/case_studies/{flight}"
make_dir(outpath)

# %% find bahamas file and read in bahamas data and satellite picture
file = [f for f in os.listdir(bahamas_dir) if f.endswith(".nc")][0]
bahamas = read_bahamas(f"{bahamas_dir}/{file}")
sat_image = [f for f in os.listdir(sat_dir) if "MODIS" in f][0]
sat_ds = rasterio.open(f"{sat_dir}/{sat_image}")
bahamas_subset = bahamas.sel(TIME=slice(rs_start, rs_end))  # select subset of radiation square
bahamas_rs = bahamas_subset.where(np.abs(bahamas_subset["IRS_PHI"]) < 1)  # select only sections with roll < 1°

# %% BAHAMAS: select position and time data and set extent
x_edmo, y_edmo = coordinates["EDMO"]
lon, lat, altitude, times = bahamas["IRS_LON"], bahamas["IRS_LAT"], bahamas["IRS_ALT"], bahamas["TIME"]
pad = 2
llcrnlat = lat.min(skipna=True) - pad
llcrnlon = lon.min(skipna=True) - pad
urcrnlat = lat.max(skipna=True) + pad
urcrnlon = lon.max(skipna=True) + pad
extent = [llcrnlon, urcrnlon, llcrnlat, urcrnlat]
font = {'weight': 'bold', 'size': 26}
matplotlib.rc('font', **font)
# get plot properties
props = plot_props[flight]
set_cb_friendly_colors()

# %% plot bahamas map with sat image
fig, ax = plt.subplots(figsize=(11, 9), subplot_kw={"projection": ccrs.PlateCarree()})
# ax.stock_img()
show(sat_ds, ax=ax)
ax.coastlines(linewidth=3)
ax.add_feature(cartopy.feature.BORDERS, linewidth=3)
ax.set_extent(extent)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
gl.bottom_labels = False
gl.left_labels = False

# plot flight track
points = ax.scatter(lon, lat, c=bahamas["IRS_HDG"], linewidth=6)
# add the corresponding colorbar and decide whether to plot it horizontally or vertically
plt.colorbar(points, ax=ax, pad=0.01, location=props["cb_loc"], label="Heading (°)", shrink=props["shrink"])

# plot a way point every 15 minutes = 9000 seconds with a time stamp next to it
for long, lati, time_stamp in zip(lon[9000::9000], lat[9000::9000], times[9000::9000]):
    ax.annotate(time_stamp.dt.strftime("%H:%M").values, (long, lati), fontsize=16,
                path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])
    ax.plot(long, lati, '.k', markersize=10)

# plot points with labels and white line around text
ax.plot(x_edmo, y_edmo, 'ok')
ax.text(x_edmo + 0.1, y_edmo + 0.1, "EDMO", fontsize=22,
        path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])
plt.tight_layout(pad=0.1)
fig_name = f"{outpath}/{flight}_bahamas_track_with_sat.png"
# plt.show()
plt.savefig(fig_name, dpi=100)
log.info(f"Saved {fig_name}")
plt.close()
# %% plot BAHAMAS data for Radiation Square
matplotlib.rcdefaults()
fig, axs = plt.subplots(nrows=1)
axs.plot(bahamas_subset["TIME"], bahamas_subset["IRS_HDG"], label="Heading")
axs.set_ylabel("Heading (°) 0=N")
axs.set_xlabel("Time (UTC)")
axs.set_ylim((0, 360))
axs.grid()
axs2 = axs.twinx()
axs2.plot(bahamas_subset["TIME"], bahamas_subset["IRS_PHI"], color="red", label="Roll")
axs2.set_ylabel("Roll Angle (°)")
axs2.set_ylim((-1.5, 1.5))
axs.set_title(f"BAHAMAS Aircraft Data - {flight}")
fig.legend(loc=2)
fig.autofmt_xdate()
# plt.show()
fig_name = f"{outpath}/{flight}_roll_heading_rad_square.png"
plt.savefig(fig_name)
log.info(f"Saved {fig_name}")
plt.close()

# %% select only the relevant flight sections, removing the turns and plot it
matplotlib.rcdefaults()
fig, axs = plt.subplots(nrows=1)
axs.plot(bahamas_rs["TIME"], bahamas_rs["IRS_HDG"], label="Heading")
axs.set_ylabel("Heading (°) 0=N")
axs.set_xlabel("Time (UTC)")
axs.set_ylim((0, 360))
axs.grid()
axs2 = axs.twinx()
axs2.plot(bahamas_rs["TIME"], bahamas_rs["IRS_PHI"], color="red", label="Roll")
axs2.set_ylabel("Roll Angle (°)")
axs2.set_ylim((-1.5, 1.5))
axs.set_title(f"BAHAMAS Aircraft Data - {flight}")
fig.legend(loc=2)
fig.autofmt_xdate()
fig_name = f"{outpath}/{flight}_roll_heading_rad_square_filtered.png"
# plt.show()
plt.savefig(fig_name)
log.info(f"Saved {fig_name}")
plt.close()

# %% read in uncorrected and corrected BACARDI data and check for offsets
bacardi_ds = xr.open_dataset(f"{bacardi_dir}/CIRRUS_HL_F02_20210625a_ADLR_BACARDI_BroadbandFluxes_R0.nc")
bacardi_rs = bacardi_ds.sel(time=slice(rs_start, rs_end))  # select only radiation square data
bacardi_raw = read_bacardi_raw("QL-CIRRUS-HL_F02_20210625a_ADLR_BACARDI_v1.nc", bacardi_dir)
bacardi_raw_rs = bacardi_raw.sel(time=slice(rs_start, rs_end))
bacardi_nooffset = xr.open_dataset(f"{bacardi_dir}/CIRRUS_HL_F02_20210625a_ADLR_BACARDI_BroadbandFluxes_R0_0offset.nc")
bacardi_nooffset_rs = bacardi_nooffset.sel(time=slice(rs_start, rs_end))
bacardi_uncor = xr.open_dataset(f"{bacardi_dir}/CIRRUS_HL_F02_20210625a_ADLR_BACARDI_BroadbandFluxes_R0_noattcor.nc")
bacardi_uncor_rs = bacardi_uncor.sel(time=slice(rs_start, rs_end))
ylims = (1120, 1200)

# %% plot all radiation square BACARDI data
plt.rc('font', size=14)
plt.rc('lines', linewidth=3)
plt.rc('font', family="serif")
fig, ax = plt.subplots(figsize=(9, 5.5))
# solar radiation
bacardi_rs.F_up_solar.plot(x="time", label=r"$F_{\uparrow}$ BACARDI", ax=ax, c="#6699CC", ls="-")
bacardi_rs.F_down_solar.plot(x="time", label=r"$F_{\downarrow}$ BACARDI", ax=ax, c="#117733", ls="-")
# terrestrial radiation
bacardi_rs.F_up_terrestrial.plot(x="time", label=r"$F_{\uparrow}$ BACARDI", ax=ax, c="#CC6677", ls="-")
bacardi_rs.F_down_terrestrial.plot(x="time", label=r"$F_{\downarrow}$ BACARDI", ax=ax, c="#f89c20", ls="-")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Irradiance (W$\,$m$^{-2}$)")
set_xticks_and_xlabels(ax, pd.to_timedelta((bacardi_rs.time[-1] - bacardi_rs.time[0]).values))
ax.grid()
handles, labels = ax.get_legend_handles_labels()
legend_column_headers = ["Solar", "Terrestrial"]
handles.insert(0, Patch(color='none', label=legend_column_headers[0]))
handles.insert(3, Patch(color='none', label=legend_column_headers[1]))
ax.legend(handles=handles, bbox_to_anchor=(0.5, 0), loc="lower center", ncol=2, bbox_transform=fig.transFigure)
plt.subplots_adjust(bottom=0.31)
plt.tight_layout()
fig_name = f"{outpath}/CIRRUS_HL_{flight}_bacardi_broadband_irradiance.png"
# plt.show()
plt.savefig(fig_name, dpi=100)
log.info(f"Saved {fig_name}")
plt.close()

# %% plot F_dw radiation square BACARDI data old EUREC4A offsets
plt.rc('font', size=14)
plt.rc('lines', linewidth=3)
plt.rc('font', family="serif")
fig, ax = plt.subplots(figsize=(9, 5.5))
# solar radiation
bacardi_rs.F_down_solar.plot(x="time", label=r"$F_{\downarrow}$ BACARDI (0.3, 2.55)", ax=ax, c="#117733", ls="-")
bacardi_rs.F_down_solar_sim.plot(x="time", label=r"$F_{\downarrow}$ libRadtran", ax=ax, c="#332288")
ax2 = ax.twinx()
heading = ax2.plot(bahamas_rs["TIME"], bahamas_rs["IRS_HDG"], label="Heading")
saa = bacardi_rs.saa.plot(x="time", label="Solar Azimuth Angle")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Irradiance (W$\,$m$^{-2}$)")
ax.set_ylim(ylims)  # fix y limits for better comparison
ax2.set_ylabel("Heading (°) 0=N, Solar Azimuth Angle")
set_xticks_and_xlabels(ax, pd.to_timedelta((bacardi_rs.time[-1] - bacardi_rs.time[0]).values))
ax.grid(axis='x')
ax2.grid()
handles, labels = ax.get_legend_handles_labels()
legend_column_headers = ["Solar (Roll offset, Pitch offset)", "Terrestrial"]
handles.insert(0, Patch(color='none', label=legend_column_headers[0]))
handles.insert(3, heading[0])
handles.insert(4, saa[0])
ax.legend(handles=handles, bbox_to_anchor=(0.5, 0), loc="lower center", ncol=2, bbox_transform=fig.transFigure)
plt.subplots_adjust(bottom=0.31)
plt.tight_layout()
fig_name = f"{outpath}/CIRRUS_HL_{flight}_bacardi_fdw_saa_heading.png"
# plt.show()
plt.savefig(fig_name, dpi=100)
log.info(f"Saved {fig_name}")
plt.close()

# %% plot BACARDI data which has been attitude corrected with no offset
plt.rc('font', size=14)
plt.rc('lines', linewidth=3)
plt.rc('font', family="serif")
fig, ax = plt.subplots(figsize=(9, 5.5))
# solar radiation
bacardi_nooffset_rs.F_down_solar.plot(x="time", label=r"$F_{\downarrow}$ BACARDI (0, 0)", ax=ax, c="#117733", ls="-")
bacardi_nooffset_rs.F_down_solar_sim.plot(x="time", label=r"$F_{\downarrow}$ libRadtran", ax=ax, c="#332288")
ax2 = ax.twinx()
heading = ax2.plot(bahamas_rs["TIME"], bahamas_rs["IRS_HDG"], label="Heading")
saa = bacardi_nooffset_rs.saa.plot(x="time", label="Solar Azimuth Angle")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Irradiance (W$\,$m$^{-2}$)")
ax.set_ylim(ylims)  # fix y limits for better comparison
ax2.set_ylabel("Heading (°) 0=N, Solar Azimuth Angle")
set_xticks_and_xlabels(ax, pd.to_timedelta((bacardi_nooffset_rs.time[-1] - bacardi_nooffset_rs.time[0]).values))
ax.grid(axis='x')
ax2.grid()
handles, labels = ax.get_legend_handles_labels()
legend_column_headers = ["Solar (Roll offset, Pitch offset)", "Terrestrial"]
handles.insert(0, Patch(color='none', label=legend_column_headers[0]))
handles.insert(3, heading[0])
handles.insert(4, saa[0])
ax.legend(handles=handles, bbox_to_anchor=(0.5, 0), loc="lower center", ncol=2, bbox_transform=fig.transFigure)
plt.subplots_adjust(bottom=0.31)
plt.tight_layout()
fig_name = f"{outpath}/CIRRUS_HL_{flight}_bacardi_fdw_saa_heading_no_offset.png"
# plt.show()
plt.savefig(fig_name, dpi=100)
log.info(f"Saved {fig_name}")
plt.close()

# %% plot BACARDI data with no attitude correction
plt.rc('font', size=14)
plt.rc('lines', linewidth=3)
plt.rc('font', family="serif")
fig, ax = plt.subplots(figsize=(9, 5.5))
# solar radiation
bacardi_uncor_rs.F_down_solar.plot(x="time", label=r"$F_{\downarrow}$ BACARDI (None, None)", ax=ax, c="#117733", ls="-")
bacardi_uncor_rs.F_down_solar_sim.plot(x="time", label=r"$F_{\downarrow}$ libRadtran", ax=ax, c="#332288")
ax2 = ax.twinx()
heading = ax2.plot(bahamas_rs["TIME"], bahamas_rs["IRS_HDG"], label="Heading")
saa = bacardi_uncor_rs.saa.plot(x="time", label="Solar Azimuth Angle")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Irradiance (W$\,$m$^{-2}$)")
ax.set_ylim(ylims)  # fix y limits for better comparison
ax2.set_ylabel("Heading (°) 0=N, Solar Azimuth Angle")
set_xticks_and_xlabels(ax, pd.to_timedelta((bacardi_uncor_rs.time[-1] - bacardi_uncor_rs.time[0]).values))
ax.grid(axis='x')
ax2.grid()
handles, labels = ax.get_legend_handles_labels()
legend_column_headers = ["Solar (Roll offset, Pitch offset)", "Terrestrial"]
handles.insert(0, Patch(color='none', label=legend_column_headers[0]))
handles.insert(3, heading[0])
handles.insert(4, saa[0])
ax.legend(handles=handles, bbox_to_anchor=(0.5, 0), loc="lower center", ncol=2, bbox_transform=fig.transFigure)
plt.subplots_adjust(bottom=0.31)
plt.tight_layout()
fig_name = f"{outpath}/CIRRUS_HL_{flight}_bacardi_fdw_saa_heading_no_att_cor.png"
# plt.show()
plt.savefig(fig_name, dpi=100)
log.info(f"Saved {fig_name}")
plt.close()

# %% vary roll and pitch offsets and correct F_dw
roll_offset = -0.15
pitch_offset = 2.85
dirdiff = read_libradtran(flight, "BBR_DirectFraction_Flight_20210625a_R0_ds_high.dat")
dirdiff_rs = dirdiff.loc[rs_start:rs_end]
# interpolate f_dir on bacardi time
f_dir_func = interp1d(dirdiff_rs.index.values.astype(float), dirdiff_rs.f_dir, fill_value="extrapolate")
f_dir_inp = f_dir_func(bacardi_raw_rs.time.values.astype(float))
F_down_solar_att, factor = fdw_attitude_correction(bacardi_uncor_rs.F_down_solar.values,
                                                   roll=bacardi_raw_rs.IRS_PHI.values,
                                                   pitch=-bacardi_raw_rs.IRS_THE.values,
                                                   yaw=bacardi_raw_rs.IRS_HDG.values, sza=bacardi_uncor_rs.sza.values,
                                                   saa=bacardi_uncor_rs.saa.values, fdir=f_dir_inp,
                                                   r_off=roll_offset, p_off=pitch_offset)

# %% plot new attitude corrected downward solar irradiance
plt.rc('font', size=14)
plt.rc('lines', linewidth=3)
plt.rc('font', family="serif")
fig, ax = plt.subplots(figsize=(9, 5.5))
# solar radiation
bacardi_uncor_rs.F_down_solar.plot(x="time", label=r"$F_{\downarrow}$ BACARDI (None, None)", ax=ax, c="#117733", ls="-")
bacardi_uncor_rs.F_down_solar_sim.plot(x="time", label=r"$F_{\downarrow}$ simulated clear sky", ax=ax, ls="-", c="#332288")
ax.plot(bacardi_uncor_rs.time, F_down_solar_att, label=f"F_dw BACARDI ({roll_offset}, {pitch_offset})", ls="-",
        c="#CC6677")
ax2 = ax.twinx()
heading = ax2.plot(bahamas_rs["TIME"], bahamas_rs["IRS_HDG"], label="Heading")
saa = bacardi_uncor_rs.saa.plot(x="time", label="Solar Azimuth Angle")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Irradiance (W$\,$m$^{-2}$)")
ax.set_ylim(ylims)  # fix y limits for better comparison (1140, 1170)
ax2.set_ylabel("Heading (°) 0=N, Solar Azimuth Angle")
set_xticks_and_xlabels(ax, pd.to_timedelta((bacardi_uncor_rs.time[-1] - bacardi_uncor_rs.time[0]).values))
ax.grid(axis='x')
ax2.grid()
handles, labels = ax.get_legend_handles_labels()
legend_column_headers = ["Solar (Roll offset, Pitch offset)", "Terrestrial"]
handles.insert(0, Patch(color='none', label=legend_column_headers[0]))
handles.insert(4, Patch(color='none'))
handles.insert(5, heading[0])
handles.insert(6, saa[0])
ax.legend(handles=handles, bbox_to_anchor=(0.5, 0), loc="lower center", ncol=2, bbox_transform=fig.transFigure)
plt.subplots_adjust(bottom=0.37)
plt.tight_layout()
fig_name = f"{outpath}/CIRRUS_HL_{flight}_bacardi_fdw_saa_heading_new_att_corr_{roll_offset}_{pitch_offset}.png"
# plt.show()
plt.savefig(fig_name, dpi=100)
log.info(f"Saved {fig_name}")
plt.close()

# %% 20210712 First day with SMART fixed
print("20210712 SMART fixed")
rs_start = pd.Timestamp(2021, 7, 12, 13, 15)
rs_end = pd.Timestamp(2021, 7, 12, 15, 15)
flight = "Flight_20210712b"
bahamas_dir = get_path("bahamas", flight)
bacardi_dir = get_path("bacardi", flight)
smart_dir = get_path("calibrated", flight)
sat_dir = get_path("satellite", flight)
libradtran_dir = get_path("libradtran", flight)
if os.getcwd().startswith("C:"):
    outpath = f"C:/Users/Johannes/Documents/Doktor/campaigns/CIRRUS-HL/case_studies/{flight}"
else:
    outpath = f"/projekt_agmwend/home_rad/jroettenbacher/case_studies/{flight}"
make_dir(outpath)

# %% find bahamas file and read in bahamas data and satellite picture
file = [f for f in os.listdir(bahamas_dir) if f.endswith(".nc")][0]
bahamas = read_bahamas(flight)
sat_image = [f for f in os.listdir(sat_dir) if "MODIS" in f][0]
sat_ds = rasterio.open(f"{sat_dir}/{sat_image}")
# bahamas_rs = bahamas_subset.where(np.abs(bahamas_subset["IRS_PHI"]) < 1)  # select only sections with roll < 1°
bahamas_rs = bahamas.sel(time=slice(rs_start, rs_end))  # select subset of radiation square

# %% BAHAMAS: select position and time data and set extent
x_edmo, y_edmo = coordinates["EDMO"]
lon, lat, altitude, times = bahamas_rs["IRS_LON"], bahamas_rs["IRS_LAT"], bahamas_rs["IRS_ALT"], bahamas_rs["TIME"]
pad = 2
llcrnlat = lat.min(skipna=True) - pad
llcrnlon = lon.min(skipna=True) - pad
urcrnlat = lat.max(skipna=True) + pad
urcrnlon = lon.max(skipna=True) + pad
extent = [llcrnlon, urcrnlon, llcrnlat, urcrnlat]
font = {'weight': 'bold', 'size': 26}
matplotlib.rc('font', **font)
# get plot properties
props = plot_props[flight]
set_cb_friendly_colors()

# %% plot bahamas map with sat image
fig, ax = plt.subplots(figsize=(11, 9), subplot_kw={"projection": ccrs.PlateCarree()})
# ax.stock_img()
show(sat_ds, ax=ax)
ax.coastlines(linewidth=3)
ax.add_feature(cartopy.feature.BORDERS, linewidth=3)
ax.set_extent(extent)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
gl.bottom_labels = False
gl.left_labels = False

# plot flight track
points = ax.scatter(lon, lat, c=bahamas_rs["IRS_ALT"] / 1000, linewidth=6)
# add the corresponding colorbar and decide whether to plot it horizontally or vertically
plt.colorbar(points, ax=ax, pad=0.01, location=props["cb_loc"], label="Height (km)", shrink=props["shrink"])

# plot a way point every 15 minutes = 9000 seconds with a time stamp next to it
for long, lati, time_stamp in zip(lon[9000::9000], lat[9000::9000], times[9000::9000]):
    ax.annotate(time_stamp.dt.strftime("%H:%M").values, (long, lati), fontsize=16,
                path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])
    ax.plot(long, lati, '.k', markersize=10)

# plot points with labels and white line around text
# ax.plot(x_edmo, y_edmo, 'ok')
# ax.text(x_edmo + 0.1, y_edmo + 0.1, "EDMO", fontsize=22,
#         path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])
plt.tight_layout(pad=0.1)
fig_name = f"{outpath}/{flight}_bahamas_track_with_sat.png"
# plt.show()
plt.savefig(fig_name, dpi=100)
log.info(f"Saved {fig_name}")
plt.close()

# %% 20210715 Spain flight for contrail outbreak and Radiation Square candidate
print("20210715 SMART fixed - contrail outbreak over Spain - Radiation Square")
from cirrus_hl import lookup
rs_start = pd.Timestamp(2021, 7, 15, 8, 0)
rs_end = pd.Timestamp(2021, 7, 15, 10, 0)
flight = "Flight_20210715a"
bacardi_dir = get_path("bacardi", flight)
smart_dir = get_path("calibrated", flight)
pixel_wl_dir = get_path("pixel_wl")
libradtran_dir = get_path("libradtran", flight)
sat_dir = get_path("satellite", flight)
if os.getcwd().startswith("C:"):
    outpath = f"C:/Users/Johannes/Documents/Doktor/campaigns/CIRRUS-HL/case_studies/{flight}"
else:
    outpath = f"/projekt_agmwend/home_rad/jroettenbacher/case_studies/{flight}"
make_dir(outpath)
ylims = (600, 1000)  # set ylims for all irradiance plots

# %% find files and read them in
bahamas = read_bahamas(flight)
bahamas_rs = bahamas.sel(time=slice(rs_start, rs_end))  # select subset of radiation square
bacardi_file = [f for f in os.listdir(bacardi_dir) if f.endswith("R0.nc")][0]
bacardi = xr.open_dataset(f"{bacardi_dir}/{bacardi_file}")
bacardi_rs = bacardi.sel(time=slice(rs_start, rs_end))
sat_image = [f for f in os.listdir(sat_dir) if "MODIS" in f][0]
sat_ds = rasterio.open(f"{sat_dir}/{sat_image}")
smart_fdw_vnir_file = [f for f in os.listdir(smart_dir) if "Fdw_VNIR" in f][0]
smart_fdw_vnir = smart.read_smart_cor(smart_dir, smart_fdw_vnir_file)
smart_fdw_swir_file = [f for f in os.listdir(smart_dir) if "Fdw_SWIR" in f][0]
smart_fdw_swir = smart.read_smart_cor(smart_dir, smart_fdw_swir_file)
fdw_sim_file = [f for f in os.listdir(libradtran_dir) if "smart_bb" in f][0]
fdw_sim = xr.open_dataset(f"{libradtran_dir}/{fdw_sim_file}")
fdw_sim_rs = fdw_sim.sel(time=slice(rs_start, rs_end))
fdw_800_sim_file = [f for f in os.listdir(libradtran_dir) if "800nm" in f][0]
fdw_800_sim = xr.open_dataset(f"{libradtran_dir}/{fdw_800_sim_file}")
fdw_800_sim_rs = fdw_800_sim.sel(time=slice(rs_start, rs_end))

# %% integrate SMART measurements
smart_fdw_vnir_int = smart_fdw_vnir.sum(axis=1)
smart_fdw_swir_int = smart_fdw_swir.sum(axis=1, skipna=False)
smart_fdw = smart_fdw_vnir_int + smart_fdw_swir_int
smart_fdw_rs = smart_fdw.loc[slice(rs_start, rs_end)]

# %% select one wavelength with low diffuse part for correction
pixel_wl = smart.read_pixel_to_wavelength(pixel_wl_dir, lookup["Fdw_VNIR"])
pixel, wl = smart.find_pixel(pixel_wl, 800)
smart_800 = smart_fdw_vnir.iloc[:, pixel]
smart_800_rs = smart_800.loc[slice(rs_start, rs_end)]

# %% filter high roll angles
roll_filter = np.abs(bahamas_rs["IRS_PHI"]) < 1
bahamas_rs_filtered = bahamas_rs.where(roll_filter)  # select only sections with roll < 1°
bahamas_inp = bahamas_rs.interp(time=smart_fdw_rs.index)  # interpolate bahamas on SMART time
roll_filter_smart = np.abs(bahamas_inp["IRS_PHI"]) < 1  # create filter for smart data
smart_fdw_rs_filtered = smart_fdw_rs.where(roll_filter_smart)
smart_800_rs_filtered = smart_800_rs.where(roll_filter_smart)

# %% calculate angle towards sun
angle_towards_sun = np.abs(bahamas_rs.IRS_HDG - fdw_sim_rs.saa)

# %% BAHAMAS: select position and time data and set extent
x_santiago, y_santiago = coordinates["Santiago"]
lon, lat, altitude, times = bahamas_rs["IRS_LON"], bahamas_rs["IRS_LAT"], bahamas_rs["IRS_ALT"], bahamas_rs["time"]
pad = 2
llcrnlat = lat.min(skipna=True) - pad
llcrnlon = lon.min(skipna=True) - pad
urcrnlat = lat.max(skipna=True) + pad
urcrnlon = lon.max(skipna=True) + pad
extent = [llcrnlon, urcrnlon, llcrnlat, urcrnlat]
font = {'weight': 'bold', 'size': 26}
matplotlib.rc('font', **font)
# get plot properties
props = plot_props[flight]
set_cb_friendly_colors()

# %% plot bahamas map with sat image
fig, ax = plt.subplots(figsize=(11, 9), subplot_kw={"projection": ccrs.PlateCarree()})
# ax.stock_img()
show(sat_ds, ax=ax)
ax.coastlines(linewidth=3)
ax.add_feature(cartopy.feature.BORDERS, linewidth=3)
ax.set_extent(extent)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
gl.bottom_labels = False
gl.left_labels = False

# plot flight track
points = ax.scatter(lon, lat, c=bahamas_rs["IRS_ALT"] / 1000, linewidth=6)
# add the corresponding colorbar and decide whether to plot it horizontally or vertically
plt.colorbar(points, ax=ax, pad=0.01, location=props["cb_loc"], label="Height (km)", shrink=props["shrink"])

# plot a way point every 15 minutes = 9000 seconds with a time stamp next to it
for long, lati, time_stamp in zip(lon[9000::9000], lat[9000::9000], times[9000::9000]):
    ax.annotate(time_stamp.dt.strftime("%H:%M").values, (long, lati), fontsize=16,
                path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])
    ax.plot(long, lati, '.k', markersize=10)

# plot points with labels and white line around text
ax.plot(x_santiago, y_santiago, 'ok')
ax.text(x_santiago + 0.1, y_santiago + 0.1, "Santiago", fontsize=22,
        path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])
plt.tight_layout(pad=0.1)
fig_name = f"{outpath}/{flight}_bahamas_track_with_sat.png"
# plt.show()
plt.savefig(fig_name, dpi=100)
log.info(f"Saved {fig_name}")
plt.close()

# %% plot BAHAMAS data for Radiation Square
matplotlib.rcdefaults()
set_cb_friendly_colors()
fig, axs = plt.subplots(nrows=1)
axs.plot(bahamas_rs["time"], bahamas_rs["IRS_HDG"], label="Heading")
axs.set_ylabel("Heading (°) 0=N")
axs.set_xlabel("Time (UTC)")
axs.set_ylim((0, 360))
axs.grid()
axs2 = axs.twinx()
axs2.plot(bahamas_rs["time"], bahamas_rs["IRS_PHI"], label="Roll", c="#117733")
axs2.set_ylabel("Roll Angle (°)")
axs2.set_ylim((-1.5, 1.5))
axs.set_title(f"BAHAMAS Aircraft Data - {flight}")
fig.legend(loc=2)
fig.autofmt_xdate()
fig_name = f"{outpath}/{flight}_roll_heading_rad_square.png"
plt.show()
# plt.savefig(fig_name)
# log.info(f"Saved {fig_name}")
plt.close()
# %% plot BAHAMAS filtered for high roll angles and solar azimuth angle
matplotlib.rcdefaults()
plt.rc('font', size=14)
plt.rc('lines', linewidth=3)
plt.rc('font', family="serif")
fig, ax = plt.subplots(figsize=(9, 5.5))
bahamas_rs_filtered.IRS_HDG.plot(x="time", label="Heading", ax=ax)
fdw_sim_rs.saa.plot(x="time", label="Solar Azimuth Angle", ax=ax)
ax.set_ylabel("Angle (°) 0=N")
ax.set_xlabel("Time (UTC)")
ax.set_ylim((0, 360))
ax.grid()
ax2 = ax.twinx()
roll = bahamas_rs_filtered.IRS_PHI.plot(x="time", color="#117733", label="Roll", ax=ax2)
ax2.set_ylabel("Roll Angle (°)")
ax2.set_ylim((-1.5, 1.5))
ax.set_title(f"BAHAMAS Aircraft Data and calculated SAA - {flight}")
handles, labels = ax.get_legend_handles_labels()
handles.insert(2, roll[0])
ax.legend(handles=handles, bbox_to_anchor=(0.5, 0), loc="lower center", ncol=3, bbox_transform=fig.transFigure)
set_xticks_and_xlabels(ax, pd.to_timedelta((bahamas_rs_filtered.time[-1] - bahamas_rs_filtered.time[0]).values))
plt.subplots_adjust(bottom=0.2)
plt.tight_layout()
fig_name = f"{outpath}/{flight}_roll_heading_saa_rad_square_filtered.png"
# plt.show()
plt.savefig(fig_name)
log.info(f"Saved {fig_name}")
plt.close()

# %% plot integrated SMART data and libRadtran simulation for Radiation Square
plt.rc('font', size=14)
plt.rc('lines', linewidth=3)
plt.rc('font', family="serif")
fig, ax = plt.subplots(figsize=(9, 5.5))
# solar radiation
smart_fdw_rs_filtered.plot(label=r"$F_{\downarrow}$ SMART integrated (None, None)", ax=ax, c="#117733", ls="-")
fdw_sim_rs.fdw.plot(x="time", label=r"$F_{\downarrow}$ simulated clear sky", ax=ax, ls="-", c="#332288")
ax2 = ax.twinx()
heading = ax2.plot(bahamas_rs["time"], bahamas_rs_filtered["IRS_HDG"], label="Heading")
saa = fdw_sim_rs.saa.plot(x="time", label="Solar Azimuth Angle", ax=ax2)
sza = fdw_sim_rs.sza.plot(x="time", label="Solar Zenith Angle", ax=ax2)
ats = angle_towards_sun.plot(x="time", label="Angle towards Sun", ax=ax2)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Irradiance (W$\,$m$^{-2}$)")
ax.set_ylim(ylims)  # fix y limits for better comparison (1140, 1170)
ax2.set_ylabel("Angle (°)")
set_xticks_and_xlabels(ax, pd.to_timedelta((bahamas_rs.time[-1] - bahamas_rs.time[0]).values))
ax.grid(axis='x')
ax2.grid()
handles, labels = ax.get_legend_handles_labels()
handles.insert(0, Patch(color='none', label="Solar (Roll offset, Pitch offset)"))
handles.insert(3, Patch(color='none'))
handles.insert(4, heading[0])
handles.insert(5, saa[0])
handles.insert(6, sza[0])
handles.insert(7, ats[0])
ax.legend(handles=handles, bbox_to_anchor=(0.5, 0), loc="lower center", ncol=2, bbox_transform=fig.transFigure)
plt.subplots_adjust(bottom=0.37)
plt.tight_layout()
fig_name = f"{outpath}/CIRRUS_HL_{flight}_smart_fdw_saa_sza_heading.png"
# plt.show()
plt.savefig(fig_name, dpi=100)
log.info(f"Saved {fig_name}")
plt.close()

# %% attitude correct f_dw with no offsets
smart_fdw_df = pd.DataFrame(smart_fdw).rename(columns={0: "F_down"})  # make dataframe from series
# interpolate libradtran simulation and bahamas on smart time
fdw_sim_inp = fdw_sim.interp(time=smart_fdw.index, kwargs=dict(fill_value="extrapolate"))
bahamas_inp = bahamas.interp(time=smart_fdw.index, kwargs=dict(fill_value="extrapolate"))
F_down_att_no_offset, factor = fdw_attitude_correction(smart_fdw_df["F_down"].values,
                                                       roll=bahamas_inp.IRS_PHI.values,
                                                       pitch=-bahamas_inp.IRS_THE.values,
                                                       yaw=bahamas_inp.IRS_HDG.values, sza=fdw_sim_inp.sza.values,
                                                       saa=fdw_sim_inp.saa.values,
                                                       fdir=fdw_sim_inp.direct_fraction.values,
                                                       r_off=0, p_off=0)
smart_fdw_df["F_down_att_no"] = F_down_att_no_offset

# %% vary roll and pitch offset and attitude correct f_dw
roll_offset = -1.4
pitch_offset = 2.9
F_down_att, factor = fdw_attitude_correction(smart_fdw_df["F_down"].values,
                                             roll=bahamas_inp.IRS_PHI.values, pitch=-bahamas_inp.IRS_THE.values,
                                             yaw=bahamas_inp.IRS_HDG.values, sza=fdw_sim_inp.sza.values,
                                             saa=fdw_sim_inp.saa.values, fdir=fdw_sim_inp.direct_fraction.values,
                                             r_off=roll_offset, p_off=pitch_offset)

smart_fdw_df["F_down_att"] = F_down_att
smart_fdw_df_rs = smart_fdw_df.loc[slice(rs_start, rs_end)]
smart_fdw_df_rs_filtered = smart_fdw_df_rs.loc[roll_filter_smart.values]

# %% plot attitude corrected SMART F_dw measurements
plt.rc('font', size=14)
plt.rc('lines', linewidth=3)
plt.rc('font', family="serif")
zoom = False
zoom_str = "_zoom" if zoom else ""
fig, ax = plt.subplots(figsize=(9, 5.5))
# solar radiation
smart_fdw_df_rs_filtered.F_down.plot(label=r"$F_{\downarrow}$ SMART (None, None)", ax=ax, c="#117733", ls="-")
# smart_fdw_df_rs_filtered.F_down_att_no.plot(label=r"$F_{\downarrow}$ SMART (0, 0)", ax=ax, c="#D55E00", ls="-")
smart_fdw_df_rs_filtered.F_down_att.plot(label=r"$F_{\downarrow}$" + f" SMART ({roll_offset}, {pitch_offset})", ls="-", c="#CC6677")
fdw_sim_rs.fdw.plot(x="time", label=r"$F_{\downarrow}$ simulated clear sky", ax=ax, ls="-", c="#332288")
ax2 = ax.twinx()
heading = bahamas_rs_filtered.IRS_HDG.plot(x="time", label="Heading", ax=ax2)
ats = angle_towards_sun.plot(x="time", label="Angle towards Sun", ax=ax2)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Irradiance (W$\,$m$^{-2}$)")
if zoom:
    ax.set_ylim((700, 900))
else:
    ax.set_ylim(ylims)  # fix y limits for better comparison (1140, 1170)
ax2.set_ylabel("Angle (°)")
set_xticks_and_xlabels(ax, pd.to_timedelta((bahamas_rs.time[-1] - bahamas_rs.time[0]).values))
ax.grid(axis='x')
ax2.grid()
handles, labels = ax.get_legend_handles_labels()
handles.insert(0, Patch(color='none', label="Solar (Roll offset, Pitch offset)"))
handles.insert(4, Patch(color='none'))
handles.insert(6, heading[0])
handles.insert(7, ats[0])
ax.legend(handles=handles, bbox_to_anchor=(0.5, 0), loc="lower center", ncol=2, bbox_transform=fig.transFigure)
plt.subplots_adjust(bottom=0.37)
plt.tight_layout()
fig_name = f"{outpath}/CIRRUS_HL_{flight}_smart_fdw_saa_heading_new_att_corr_{roll_offset}_{pitch_offset}{zoom_str}.png"
# plt.show()
plt.savefig(fig_name, dpi=100)
log.info(f"Saved {fig_name}")
plt.close()

# %% plot difference between simulation and measurement
fdw_diff = smart_fdw_df_rs.F_down_att - fdw_sim_inp["fdw"].sel(time=slice(rs_start, rs_end))
fdw_diff_pc = np.abs(fdw_diff / smart_fdw_df_rs.F_down_att) * 100
fdw_diff_pc_filtered = fdw_diff_pc.loc[roll_filter_smart.values]
plt.rc('font', size=14)
plt.rc('lines', linewidth=3)
plt.rc('font', family="serif")
fig, ax = plt.subplots(figsize=(9, 5.5))
fdw_diff_pc_filtered.plot(ax=ax)
ax.set_ylabel("Difference (%)")
ax.set_xlabel("Time (UTC)")
ax.set_ylim((0, 4))
ax.grid()
ax.set_title(f"Difference between corrected measurement and clear sky simulation\n"
             f"Roll offset: {roll_offset}; Pitch offset: {pitch_offset}")
fig_name = f"{outpath}/CIRRUS_HL_{flight}_difference_smart_cor_simulation_{roll_offset}_{pitch_offset}.png"
# plt.show()
plt.savefig(fig_name, dpi=100)
log.info(f"Saved {fig_name}")
plt.close()
# %% plot 800nm SMART data and libRadtran simulation for Radiation Square
plt.rc('font', size=14)
plt.rc('lines', linewidth=3)
plt.rc('font', family="serif")
fig, ax = plt.subplots(figsize=(9, 5.5))
# solar radiation
smart_800_rs_filtered.plot(label=r"$F_{\downarrow}$ SMART 800nm (None, None)", ax=ax, c="#117733", ls="-")
fdw_800_sim_rs.fdw.plot(x="time", label=r"$F_{\downarrow}$ simulated clear sky 800nm", ax=ax, ls="-", c="#332288")
ax2 = ax.twinx()
heading = ax2.plot(bahamas_rs["time"], bahamas_rs_filtered["IRS_HDG"], label="Heading")
saa = fdw_sim_rs.saa.plot(x="time", label="Solar Azimuth Angle", ax=ax2)
sza = fdw_sim_rs.sza.plot(x="time", label="Solar Zenith Angle", ax=ax2)
ats = angle_towards_sun.plot(x="time", label="Angle towards Sun", ax=ax2)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Spectral Irradiance (W$\,$m$^{-2}\,$nm$^{-1}$)")
ax.set_ylim((0.5, 1))  # fix y limits for better comparison (0.5, 1)
ax2.set_ylabel("Angle (°)")
set_xticks_and_xlabels(ax, pd.to_timedelta((bahamas_rs.time[-1] - bahamas_rs.time[0]).values))
ax.grid(axis='x')
ax2.grid()
handles, labels = ax.get_legend_handles_labels()
handles.insert(0, Patch(color='none', label="Solar (Roll offset, Pitch offset)"))
handles.insert(3, Patch(color='none'))
handles.insert(4, heading[0])
handles.insert(5, saa[0])
handles.insert(6, sza[0])
handles.insert(7, ats[0])
ax.legend(handles=handles, bbox_to_anchor=(0.5, 0), loc="lower center", ncol=2, bbox_transform=fig.transFigure)
plt.subplots_adjust(bottom=0.37)
plt.tight_layout()
fig_name = f"{outpath}/CIRRUS_HL_{flight}_smart_fdw_800nm_saa_sza_heading.png"
# plt.show()
plt.savefig(fig_name, dpi=100)
log.info(f"Saved {fig_name}")
plt.close()

# %% attitude correct f_dw with no offsets
smart_800_df = pd.DataFrame(smart_800).rename(columns={740: "F_down"})  # make dataframe from series
# interpolate libradtran simulation and bahamas on smart time
fdw_800_sim_inp = fdw_800_sim.interp(time=smart_800.index, kwargs=dict(fill_value="extrapolate"))
bahamas_inp = bahamas.interp(time=smart_800.index, kwargs=dict(fill_value="extrapolate"))
F_down_att_no_offset, factor = fdw_attitude_correction(smart_800_df["F_down"].values,
                                                       roll=bahamas_inp.IRS_PHI.values,
                                                       pitch=-bahamas_inp.IRS_THE.values,
                                                       yaw=bahamas_inp.IRS_HDG.values, sza=fdw_800_sim_inp.sza.values,
                                                       saa=fdw_800_sim_inp.saa.values,
                                                       fdir=fdw_800_sim_inp.direct_fraction.values,
                                                       r_off=0, p_off=0)
smart_800_df["F_down_att_no"] = F_down_att_no_offset

# %% vary roll and pitch offset and attitude correct f_dw
roll_offset = -1.4
pitch_offset = 2.9
F_down_att, factor = fdw_attitude_correction(smart_800_df["F_down"].values,
                                             roll=bahamas_inp.IRS_PHI.values, pitch=-bahamas_inp.IRS_THE.values,
                                             yaw=bahamas_inp.IRS_HDG.values, sza=fdw_800_sim_inp.sza.values,
                                             saa=fdw_800_sim_inp.saa.values, fdir=fdw_800_sim_inp.direct_fraction.values,
                                             r_off=roll_offset, p_off=pitch_offset)

smart_800_df["F_down_att"] = F_down_att
smart_800_df_rs = smart_800_df.loc[slice(rs_start, rs_end)]
smart_800_df_rs_filtered = smart_800_df_rs.loc[roll_filter_smart.values]

# %% plot attitude corrected SMART F_dw measurements
plt.rc('font', size=14)
plt.rc('lines', linewidth=3)
plt.rc('font', family="serif")
zoom = False
zoom_str = "_zoom" if zoom else ""
fig, ax = plt.subplots(figsize=(9, 5.5))
# solar radiation
smart_800_df_rs_filtered.F_down.plot(label=r"$F_{\downarrow}$ SMART 800nm (None, None)", ax=ax, c="#117733", ls="-")
# smart_fdw_df_rs_filtered.F_down_att_no.plot(label=r"$F_{\downarrow}$ SMART (0, 0)", ax=ax, c="#D55E00", ls="-")
smart_800_df_rs_filtered.F_down_att.plot(label=r"$F_{\downarrow}$" + f" SMART 800nm ({roll_offset}, {pitch_offset})",
                                         ls="-", c="#CC6677")
fdw_800_sim_rs.fdw.plot(x="time", label=r"$F_{\downarrow}$ simulated clear sky 800nm", ax=ax, ls="-", c="#332288")
ax2 = ax.twinx()
heading = bahamas_rs_filtered.IRS_HDG.plot(x="time", label="Heading", ax=ax2)
ats = angle_towards_sun.plot(x="time", label="Angle towards Sun", ax=ax2)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Spectral Irradiance (W$\,$m$^{-2}\,$nm$^{-1}$)")
if zoom:
    ax.set_ylim((700, 900))
else:
    ax.set_ylim((0.5, 1))  # fix y limits for better comparison (0.5, 1)
ax2.set_ylabel("Angle (°)")
set_xticks_and_xlabels(ax, pd.to_timedelta((bahamas_rs.time[-1] - bahamas_rs.time[0]).values))
ax.grid(axis='x')
ax2.grid()
handles, labels = ax.get_legend_handles_labels()
handles.insert(0, Patch(color='none', label="Solar (Roll offset, Pitch offset)"))
handles.insert(4, Patch(color='none'))
handles.insert(6, heading[0])
handles.insert(7, ats[0])
ax.legend(handles=handles, bbox_to_anchor=(0.5, 0), loc="lower center", ncol=2, bbox_transform=fig.transFigure)
plt.subplots_adjust(bottom=0.37)
plt.tight_layout()
fig_name = f"{outpath}/CIRRUS_HL_{flight}_smart_800nm_saa_heading_new_att_corr_{roll_offset}_{pitch_offset}{zoom_str}.png"
# plt.show()
plt.savefig(fig_name, dpi=100)
log.info(f"Saved {fig_name}")
plt.close()

# %% plot difference between simulation and measurement
fdw_diff = smart_800_df_rs.F_down_att - fdw_800_sim_inp["fdw"].sel(time=slice(rs_start, rs_end))
fdw_diff_pc = np.abs(fdw_diff / smart_800_df_rs.F_down_att) * 100
fdw_diff_pc_filtered = fdw_diff_pc.loc[roll_filter_smart.values]
plt.rc('font', size=14)
plt.rc('lines', linewidth=3)
plt.rc('font', family="serif")
fig, ax = plt.subplots(figsize=(9, 5.5))
fdw_diff_pc_filtered.plot(ax=ax)
ax.set_ylabel("Difference (%)")
ax.set_xlabel("Time (UTC)")
# ax.set_ylim((0, 4))
ax.grid()
ax.set_title(f"Difference between corrected measurement and clear sky simulation (800nm)\n"
             f"Roll offset: {roll_offset}; Pitch offset: {pitch_offset}")
fig_name = f"{outpath}/CIRRUS_HL_{flight}_difference_smart_800nm_cor_simulation_{roll_offset}_{pitch_offset}.png"
# plt.show()
plt.savefig(fig_name, dpi=100)
log.info(f"Saved {fig_name}")
plt.close()
