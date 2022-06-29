#!/usr/bin/env python
"""Analysis for IRS talk 2022

- define staircase pattern
- average spectra over height during staircase pattern
- band SMART data to ecRad bands
- calculate reflectivity
- plot map of all flight tracks from CIRRUS-HL
- plot map of all flight tracks from HALO-AC3
- plot SMART spectra for flight sections
- plot ecRad Input/IFS output

*author*: Johannes Röttenbacher
"""

# %% module import
import pylim.helpers as h
from pylim import reader, smart
from pylim.halo_ac3 import coordinates
import numpy as np
import os
from matplotlib import patheffects
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib import colors
from matplotlib.lines import Line2D
import cmasher as cmr
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
ecrad_dir = f"{h.get_path('ecrad', campaign)}/{flight[7:15]}"
libradtran_dir = h.get_path("libradtran", flight, campaign)
bahamas_dir = h.get_path("bahamas", flight, campaign)
sat_dir = h.get_path("satellite", flight, campaign)
sat_file = [f for f in os.listdir(sat_dir) if "MODIS" in f][0]
sat_image = os.path.join(sat_dir, sat_file)
plot_path = f"{h.get_path('plot')}/{flight}"
start_dt = pd.Timestamp(2021, 6, 29, 9, 42)
end_dt = pd.Timestamp(2021, 6, 29, 12, 10)
cm = 1 / 2.54

# %% read in data
fdw = xr.open_dataset(f"{smart_dir}/CIRRUS-HL_HALO_SMART_Fdw_20210629_Flight_20210629a_v1.0.nc")
fup = xr.open_dataset(f"{smart_dir}/CIRRUS-HL_HALO_SMART_Fup_20210629_Flight_20210629a_v1.0.nc")
libradtran = xr.open_dataset(
    f"{libradtran_dir}/CIRRUS-HL_HALO_libRadtran_clearsky_simulation_smart_spectral_20210629_{flight}.nc")
ecrad_fu = reader.read_ecrad_output(f"{ecrad_dir}/ecrad_merged_output_20210629_inp_v1.nc")
ecrad_baran = reader.read_ecrad_output(f"{ecrad_dir}/ecrad_merged_output_20210629_inp_v2.nc")
ecrad_baran16 = reader.read_ecrad_output(f"{ecrad_dir}/ecrad_merged_output_20210629_inp_v3.nc")
ecrad_yi = reader.read_ecrad_output(f"{ecrad_dir}/ecrad_merged_output_20210629_inp_v4.nc")

bahamas = reader.read_bahamas(f"{bahamas_dir}/CIRRUSHL_F05_20210629a_ADLR_BAHAMAS_v1.nc")

ecrad_input = xr.open_dataset(f"{ecrad_dir}/ecrad_merged_input_20210629_inp.nc")

sat_ds = rasterio.open(sat_image)

# %% select only relevant times and take mean of all ecRad columns
sel = slice(start_dt, end_dt)
fdw = fdw.sel(time=sel)
fup = fup.sel(time=sel)
libradtran = libradtran.sel(time=sel)
bahamas_sel = bahamas.sel(time=sel)
ecrad_input = ecrad_input.sel(time=sel).mean(dim="column")
mean_file_fu = f"{ecrad_dir}/ecrad_output_20210629_inp_v1_mean.nc"
mean_file_baran = f"{ecrad_dir}/ecrad_output_20210629_inp_v2_mean.nc"
mean_file_baran16 = f"{ecrad_dir}/ecrad_output_20210629_inp_v3_mean.nc"
mean_file_yi = f"{ecrad_dir}/ecrad_output_20210629_inp_v4_mean.nc"
if os.path.isfile(mean_file_fu):
    ecrad_fu = xr.open_dataset(mean_file_fu)
else:
    ecrad_fu = ecrad_fu.sel(time=sel)
    # take mean of column and calculate standard deviation
    if ecrad_fu.dims["column"] > 1:
        ecrad_fu_std = ecrad_fu.std(dim="column")
        ecrad_fu = ecrad_fu.mean(dim="column")
        # save intermediate file to save time
        ecrad_fu_std.to_netcdf(f"{ecrad_dir}/ecrad_output_20210629_inp_v1_stds.nc")
        ecrad_fu.to_netcdf(mean_file_fu)

if os.path.isfile(mean_file_baran):
    ecrad_baran = xr.open_dataset(mean_file_baran)
else:
    ecrad_baran = ecrad_baran.sel(time=sel)
    # take mean of column and calculate standard deviation
    if ecrad_baran.dims["column"] > 1:
        ecrad_baran_std = ecrad_baran.std(dim="column")
        ecrad_baran = ecrad_baran.mean(dim="column")
        # save intermediate file to save time
        ecrad_baran_std.to_netcdf(f"{ecrad_dir}/ecrad_output_20210629_inp_v2_stds.nc")
        ecrad_baran.to_netcdf(mean_file_baran)

if os.path.isfile(mean_file_baran16):
    ecrad_baran16 = xr.open_dataset(mean_file_baran16)
else:
    ecrad_baran16 = ecrad_baran16.sel(time=sel)
    # take mean of column and calculate standard deviation
    if ecrad_baran16.dims["column"] > 1:
        ecrad_baran16_std = ecrad_baran16.std(dim="column")
        ecrad_baran16 = ecrad_baran16.mean(dim="column")
        # save intermediate file to save time
        ecrad_baran16_std.to_netcdf(f"{ecrad_dir}/ecrad_output_20210629_inp_v3_stds.nc")
        ecrad_baran16.to_netcdf(mean_file_baran16)

if os.path.isfile(mean_file_yi):
    ecrad_yi = xr.open_dataset(mean_file_yi)
else:
    ecrad_yi = ecrad_yi.sel(time=sel)
    # take mean of column and calculate standard deviation
    if ecrad_yi.dims["column"] > 1:
        ecrad_yi_std = ecrad_yi.std(dim="column")
        ecrad_yi = ecrad_yi.mean(dim="column")
        # save intermediate file to save time
        ecrad_yi_std.to_netcdf(f"{ecrad_dir}/ecrad_output_20210629_inp_v4_stds.nc")
        ecrad_yi.to_netcdf(mean_file_yi)

# %% get height level of actual flight altitude in ecRad model, this determines only the index of the level
bahamas_tmp = bahamas.sel(time=ecrad_fu.time)
ecrad_timesteps = len(ecrad_fu.time)
aircraft_height_level = np.zeros(ecrad_timesteps)

for i in tqdm(range(ecrad_timesteps)):
    aircraft_height_level[i] = h.arg_nearest(ecrad_fu["press_height"][i, :].values, bahamas_tmp.IRS_ALT[i].values)

aircraft_height_level = aircraft_height_level.astype(int)
height_level_da = xr.DataArray(aircraft_height_level, dims=["time"], coords={"time": ecrad_fu.time})
aircraft_height = [ecrad_fu["press_height"].isel(half_level=i, time=100).values for i in aircraft_height_level]
aircraft_height_da = xr.DataArray(aircraft_height, dims=["time"], coords={"time": ecrad_fu.time})

# %% create new coordinate for ecRad bands to show spectrum on a continuous scale
ecrad_bands = [np.mean(h.ecRad_bands[band]) for band in h.ecRad_bands]

# %% band SMART and libRadtran data to ecRad bands
nr_bands = len(h.ecRad_bands)
fdw_banded = np.empty((nr_bands, fdw.time.shape[0]))
fup_banded = np.empty((nr_bands, fup.time.shape[0]))
fdw_lib_banded = np.empty((nr_bands, libradtran.time.shape[0]))
for i, band in enumerate(h.ecRad_bands):
    wl1 = h.ecRad_bands[band][0]
    wl2 = h.ecRad_bands[band][1]
    fdw_banded[i, :] = fdw.Fdw_cor.sel(wavelength=slice(wl1, wl2)).integrate(coord="wavelength")
    fup_banded[i, :] = fup.Fup_cor.sel(wavelength=slice(wl1, wl2)).integrate(coord="wavelength")
    fdw_lib_banded[i, :] = libradtran.fdw.sel(wavelength=slice(wl1, wl2)).integrate(coord="wavelength")

fdw_banded = xr.DataArray(fdw_banded, coords={"ecrad_band": range(1, 15), "time": fdw.time}, name="Fdw")
fup_banded = xr.DataArray(fup_banded, coords={"ecrad_band": range(1, 15), "time": fup.time}, name="Fup")
fdw_lib_banded = xr.DataArray(fdw_lib_banded, coords={"ecrad_band": range(1, 15), "time": libradtran.time}, name="Fdw")

# %% define staircase sections according to flight levels
altitude = np.round(bahamas_sel.H, 1)  # rounded to 1 decimal to avoid tiny differences
idx = np.where(np.diff(altitude) == 0)  # find times with constant altitude
# draw a quicklook
# altitude[idx].plot(marker="x")
# plt.show()
# plt.close()
# select times with constant altitude
times = bahamas_sel.time[idx]
# find indices where the difference to the next timestep is greater 1 second -> change of altitude
ids = np.argwhere((np.diff(times) / 10 ** 9).astype(float) > 1)
ids2 = ids + 1  # add 1 to get the indices at the start of the next section
ids = np.insert(ids, 0, 0)  # add the start of the first section
ids = np.append(ids, ids2)  # combine all indices
ids.sort()  # sort them
ids = np.delete(ids, [7, 8, 9, 10, -1])  # delete doubles
times_sel = times[ids]  # select only the start and end times
# get start and end times
start_dts = times_sel[::2]
end_dts = times_sel[1::2]
# export times for Anna
# start_dts.name = "start"
# end_dts.name = "end"
# start_dts.to_netcdf("start_dts.nc")
# end_dts.to_netcdf("end_dts.nc")
log.info("Defined Sections")

# %% calculate reflectivity from SMART measurements and ecRad simulations
reflectivity = fup.Fup_cor.interp_like(fdw.wavelength) / fdw.Fdw_cor
reflectivity_banded = fup_banded / fdw_banded
reflectivity_ecrad_fu = ecrad_fu.spectral_flux_up_sw / ecrad_fu.spectral_flux_dn_sw
reflectivity_ecrad_baran = ecrad_baran.spectral_flux_up_sw / ecrad_baran.spectral_flux_dn_sw
reflectivity_ecrad_baran16 = ecrad_baran16.spectral_flux_up_sw / ecrad_baran16.spectral_flux_dn_sw
reflectivity_ecrad_yi = ecrad_yi.spectral_flux_up_sw / ecrad_yi.spectral_flux_dn_sw

# %% set some variables for the coming plots
labels = ["Section 1 (FL260, 8.3$\,$km)", "Section 2 (FL280, 8.7$\,$km)", "Section 3 (FL300, 9.3$\,$km)",
          "Section 4 (FL320, 10$\,$km)", "Section 5 (FL340, 10.6$\,$km)", "Section 6 (FL360, 11.2$\,$km)",
          "Section 7 (FL390, 12.2$\,$km)"]
ecrad_xlabels = [str(l).replace(",", " -") for l in h.ecRad_bands.values()]

# %% set plotting options for map plot
lon, lat, times = bahamas["IRS_LON"], bahamas["IRS_LAT"], bahamas["time"]
pad = 2
llcrnlat = lat.min(skipna=True) - pad
llcrnlon = lon.min(skipna=True) - pad
urcrnlat = lat.max(skipna=True) + pad
urcrnlon = lon.max(skipna=True) + pad
extent = [llcrnlon, urcrnlon, llcrnlat, urcrnlat]
edmo = coordinates["EDMO"]
torshavn = coordinates["Torshavn"]
bergen = coordinates["Bergen"]

# %% plot bahamas map with highlighted below and above cloud sections and sat image
h.set_cb_friendly_colors()
plt.rc("font", family="serif")
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
show(sat_ds, ax=ax)
ax.coastlines(linewidth=1)
ax.add_feature(cartopy.feature.BORDERS, linewidth=1)
ax.set_extent(extent)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
gl.bottom_labels = False
gl.left_labels = False

# plot flight track
points = ax.plot(lon, lat, c="orange", linewidth=4)
# plot staircase section
for label, st, et in zip(labels, start_dts, end_dts):
    lon_sec = lon.sel(time=slice(st, et))
    lat_sec = lat.sel(time=slice(st, et))
    ax.plot(lon_sec, lat_sec, linewidth=4, label=label)
    ax.annotate(st.dt.strftime("%H:%M").values, (lon_sec[0], lat_sec[0]), fontsize=8,
                path_effects=[patheffects.withStroke(linewidth=1, foreground="w")])
    ax.plot(lon_sec[0], lat_sec[0], marker=8, c="#CC6677", markersize=8)
    ax.annotate(et.dt.strftime("%H:%M").values, (lon_sec[-1], lat_sec[-1]), xytext=(0, -9),
                textcoords='offset points',
                fontsize=8, path_effects=[patheffects.withStroke(linewidth=1, foreground="w")])
    ax.plot(lon_sec[-1], lat_sec[-1], marker=9, c="#DDCC77", markersize=8)

# plot points with labels and white line around text
ax.plot(edmo[0], edmo[1], 'ok')
ax.text(edmo[0] + 0.1, edmo[1] + 0.1, "EDMO", fontsize=10,
        path_effects=[patheffects.withStroke(linewidth=1, foreground="w")])
ax.plot(bergen[0], bergen[1], 'ok')
ax.text(bergen[0] + 0.1, bergen[1] + 0.1, "Bergen", fontsize=10,
        path_effects=[patheffects.withStroke(linewidth=1, foreground="w")])
ax.plot(torshavn[0], torshavn[1], 'ok')
ax.text(torshavn[0] + 0.1, torshavn[1] + 0.1, "Torshavn", fontsize=10,
        path_effects=[patheffects.withStroke(linewidth=1, foreground="w")])

# make a legend entry for the start and end times
handles, labels = ax.get_legend_handles_labels()
handles.insert(0, plt.plot([0], ls="", marker=8, markersize=3, color="#CC6677", label="Start Times (UTC)")[0])
handles.insert(1, plt.plot([0], ls="", marker=9, markersize=3, color="#DDCC77", label="End Times (UTC)")[0])
ax.legend(handles=handles, loc=3, fontsize=10, markerscale=4)
plt.tight_layout(pad=0.1)
fig_name = f"{plot_path}/{campaign.swapcase()}_BAHAMAS_staircase_track_{flight}.png"
# plt.show()
plt.savefig(fig_name, dpi=300)
log.info(f"Saved {fig_name}")
plt.close()

# %% plot all flight tracks of CIRRUS-HL with the natural earth as background
bahamas_all_dir = h.get_path("all", campaign=campaign, instrument="bahamas")
bahamas_all_files = [f"{bahamas_all_dir}/{file}" for file in os.listdir(bahamas_all_dir)]
extent = [-35, 30, 35, 80]
edmo = coordinates["EDMO"]
plt.rc("font", family="serif")

fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
ax.coastlines(linewidth=1)
ax.add_feature(cartopy.feature.BORDERS, linewidth=1)
ax.set_extent(extent)
ax.background_img(name='BM', resolution='high')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
gl.bottom_labels = False
gl.left_labels = False

# plot flight tracks
cmap = plt.get_cmap("hsv")
for i in range(len(bahamas_all_files)):
    bahamas_tmp = reader.read_bahamas(bahamas_all_files[i])
    lon, lat = bahamas_tmp.IRS_LON, bahamas_tmp.IRS_LAT
    ax.plot(lon, lat, color=cmap(i / len(bahamas_all_files)), linewidth=2)

# Add Oberpfaffenhofen
ax.plot(edmo[0], edmo[1], 'or')
ax.text(edmo[0] + 0.1, edmo[1] + 0.1, "EDMO", fontsize=10,
        path_effects=[patheffects.withStroke(linewidth=1, foreground="w")])

plt.tight_layout()
# plt.show()
plt.savefig(f"{plot_path}/CIRRUS-HL_BAHAMAS_all_tracks.png", dpi=300)
plt.close()

# %% plot all flight tracks if HALO-AC3 with the natural earth as background
bahamas_all_dir = h.get_path("all", campaign="halo-ac3", instrument="bahamas")
bahamas_all_files = [f"{bahamas_all_dir}/{file}" for file in os.listdir(bahamas_all_dir)]
extent = [-35, 30, 65, 90]
kiruna, longyear = coordinates["Kiruna"], coordinates["Longyearbyen"]
data_cs = ccrs.PlateCarree()
plt.rc("font", family="serif")

fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.NorthPolarStereo()))
ax.coastlines(linewidth=1)
ax.add_feature(cartopy.feature.BORDERS, linewidth=1)
ax.set_extent(extent)
ax.background_img(name='BM', resolution='high')
gl = ax.gridlines(crs=data_cs, draw_labels=True, y_inline=True, x_inline=False,
                  xlabel_style={'color': 'black', 'weight': 'bold'},
                  ylabel_style={'color': 'white', 'weight': 'bold'})
gl.xlocator = FixedLocator([-60, -45, -30, -15, 0, 15, 30, 45, 60])
gl.bottom_labels = True
gl.top_labels = False

# plot flight tracks
cmap = plt.get_cmap("hsv")
for i in range(len(bahamas_all_files)):
    bahamas_tmp = reader.read_bahamas(bahamas_all_files[i])
    lon, lat = bahamas_tmp.IRS_LON, bahamas_tmp.IRS_LAT
    ax.plot(lon, lat, color=cmap(i / len(bahamas_all_files)), linewidth=2, transform=data_cs)

# Add Kiruna and Longyearbyen
ax.plot(kiruna[0], kiruna[1], 'or', transform=data_cs)
ax.text(kiruna[0] + 0.1, kiruna[1] + 0.1, "Kiruna", fontsize=10,
        path_effects=[patheffects.withStroke(linewidth=1, foreground="w")], transform=data_cs)
ax.plot(longyear[0], longyear[1], 'or', transform=data_cs)
ax.text(longyear[0] + 0.1, longyear[1] + 0.1, "Longyearbyen", fontsize=10,
        path_effects=[patheffects.withStroke(linewidth=1, foreground="w")], transform=data_cs)

plt.tight_layout()
# plt.show()
plt.savefig(f"C:/Users/Johannes/Documents/Doktor/campaigns/HALO-AC3/case_studies/HALO-AC3_BAHAMAS_all_tracks.png",
            dpi=300)
plt.close()

# %% plot SMART spectra for each flight section Fdw
sections = dict()
for i, (st, et) in enumerate(zip(start_dts, end_dts)):
    sections[f"mean_spectra_{i}"] = fdw.Fdw_cor.sel(time=slice(st, et)).mean(dim="time")

h.set_cb_friendly_colors()
plt.rc("font", family="serif")
fig, ax = plt.subplots(figsize=(10, 6))
# SMART measurement
for section, label in zip(sections, labels):
    sections[section].plot(ax=ax, label=label, linewidth=4)

# aesthetics
ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=3)
ax.set_xlim((300, 2100))
ax.set_ylim((0, 1.65))
ax.set_title(f"Mean Downward Irradiance measured by SMART")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Spectral Irradiance $(\mathrm{W\,m}^{-2}\mathrm{\,nm}^{-1})$")
ax.grid()
plt.tight_layout()
plt.subplots_adjust(bottom=0.25)
plt.show()
# figname = f"{plot_path}/cirrus-hl_smart_fdw_spectra_sections_{flight}.png"
# plt.savefig(figname, dpi=100)
# log.info(f"Saved {figname}")
plt.close()

# %% plot SMART spectra for each flight section Fup
sections = dict()
for i, (st, et) in enumerate(zip(start_dts, end_dts)):
    sections[f"mean_spectra_{i}"] = fup.Fup_cor.sel(time=slice(st, et)).mean(dim="time")

h.set_cb_friendly_colors()
plt.rc("font", family="serif")
fig, ax = plt.subplots(figsize=(10, 6))
# banded SMART measurement
for section, label in zip(sections, labels):
    sections[section].plot(ax=ax, label=label, linewidth=4)

# aesthetics
ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=4)
ax.set_title(f"Mean Upward Irradiance measured by SMART")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Spectral Irradiance $(\mathrm{W\,m}^{-2}\mathrm{\,nm}^{-1})$")
ax.grid()
plt.subplots_adjust(bottom=0.2)
# plt.show()
figname = f"{plot_path}/cirrus-hl_smart_fup_spectra_sections_{flight}.png"
plt.savefig(figname, dpi=100)
log.info(f"Saved {figname}")
plt.close()

# %% plot SMART spectra for each flight section Fup and Fdw
sections, sections_fdw = dict(), dict()
for i, (st, et) in enumerate(zip(start_dts, end_dts)):
    sections[f"mean_spectra_{i}"] = fup.Fup_cor.sel(time=slice(st, et)).mean(dim="time")
    sections_fdw[f"mean_spectra_{i}"] = fdw.Fdw_cor.sel(time=slice(st, et)).mean(dim="time")

h.set_cb_friendly_colors()
plt.rc("font", family="serif", size=16)
fig, axs = plt.subplots(nrows=2, figsize=(24 * cm, 20 * cm))
# Fdw SMART measurement
# for section, label in zip(sections_fdw, labels):
#     sections_fdw[section].plot(ax=axs[0], label=label, linewidth=4)
# Fup SMART measurements
# for section, label in zip(sections, labels):
#     sections[section].plot(ax=axs[1])

# plot only section 1, 2 and 7
# Fdw
sections_fdw[f"mean_spectra_0"].plot(ax=axs[0], label=labels[0], linewidth=4)
sections_fdw[f"mean_spectra_1"].plot(ax=axs[0], label=labels[1], linewidth=4)
sections_fdw[f"mean_spectra_6"].plot(ax=axs[0], label=labels[6], color="#44AA99", linewidth=4)
# Fup
sections[f"mean_spectra_0"].plot(ax=axs[1], linewidth=4)
sections[f"mean_spectra_1"].plot(ax=axs[1], linewidth=4)
sections[f"mean_spectra_6"].plot(ax=axs[1], color="#44AA99", linewidth=4)

# aesthetics
# fig.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=2)  # legend below plot
axs[0].legend(loc=1)  # legend in upper right corner
axs[0].set_title(f"Mean Irradiance measured by SMART")
axs[0].set_xlabel("")
axs[0].set_ylabel("Spectral Downward \nIrradiance \n$(\mathrm{W\,m}^{-2}\mathrm{\,nm}^{-1})$")
axs[1].set_xlabel("Wavelength (nm)")
axs[1].set_ylabel("Spectral Upward \nIrradiance \n$(\mathrm{W\,m}^{-2}\mathrm{\,nm}^{-1})$")
for ax in axs:
    ax.grid()
    ax.set_ylim((0, 1.65))
plt.tight_layout()
# plt.subplots_adjust(bottom=0.3)
# plt.subplots_adjust(bottom=0.21)
# plt.show()
# figname = f"{plot_path}/cirrus-hl_smart_spectra_sections_{flight}.png"
figname = f"{plot_path}/cirrus-hl_smart_spectra_sections127_{flight}.png"
plt.savefig(figname, dpi=200)
log.info(f"Saved {figname}")
plt.close()

# %% plot banded SMART spectra for each flight section
sections = dict()
for i, (st, et) in enumerate(zip(start_dts, end_dts)):
    sections[f"mean_spectra_{i}"] = fdw_banded.sel(time=slice(st, et)).mean(dim="time")

h.set_cb_friendly_colors()
plt.rc("font", family="serif")
fig, ax = plt.subplots(figsize=(10, 6))
# banded SMART measurement
for section, label in zip(sections, labels):
    sections[section].plot(ax=ax, label=label, linewidth=4)

# aesthetics
ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=3)
ax.set_title(f"Mean Downward Irradiance measured by SMART integrated over ecRad Bands")
ax.set_xticks(np.arange(1, 15), labels=ecrad_xlabels, rotation=45)
ax.set_xlabel("ecRad band (nm)")
ax.set_xlim((4, 12))
ax.set_ylim((0, 300))
ax.invert_xaxis()
ax.set_ylabel("Spectral Irradiance $(\mathrm{W\,m}^{-2}\mathrm{\,nm}^{-1})$")
ax.grid()
plt.tight_layout()
plt.subplots_adjust(bottom=0.4)
plt.show()
# figname = f"{plot_path}/cirrus-hl_smart_fdw_banded_sections_{flight}.png"
# plt.savefig(figname, dpi=200)
# log.info(f"Saved {figname}")
plt.close()

# %% plot banded SMART spectra for each flight section Fup and Fdw
sections, sections_fdw = dict(), dict()
for i, (st, et) in enumerate(zip(start_dts, end_dts)):
    sections[f"mean_spectra_{i}"] = fup_banded.sel(time=slice(st, et)).mean(dim="time")
    sections_fdw[f"mean_spectra_{i}"] = fdw_banded.sel(time=slice(st, et)).mean(dim="time")

h.set_cb_friendly_colors()
plt.rc("font", family="serif", size=12)
fig, axs = plt.subplots(nrows=2, figsize=(10, 6))
# Fdw SMART measurement
# for section, label in zip(sections_fdw, labels):
#     sections_fdw[section].plot(ax=axs[0], label=label, linewidth=4)
# Fup SMART measurements
# for section, label in zip(sections, labels):
#     sections[section].plot(ax=axs[1])

# plot only section 1, 2 and 7
# Fdw
sections_fdw[f"mean_spectra_0"].plot(ax=axs[0], label=labels[0], linewidth=4)
sections_fdw[f"mean_spectra_1"].plot(ax=axs[0], label=labels[1], linewidth=4)
sections_fdw[f"mean_spectra_6"].plot(ax=axs[0], label=labels[6], color="#44AA99", linewidth=4)
# Fup
sections[f"mean_spectra_0"].plot(ax=axs[1], linewidth=4)
sections[f"mean_spectra_1"].plot(ax=axs[1], linewidth=4)
sections[f"mean_spectra_6"].plot(ax=axs[1], color="#44AA99", linewidth=4)

# aesthetics
axs[0].legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=3)
axs[0].set_title(f"Mean Irradiance measured by SMART integrated over ecRad Bands")
axs[0].set_xlabel("")
axs[0].set_ylabel("Spectral Downward \nIrradiance \n$(\mathrm{W\,m}^{-2}\mathrm{\,nm}^{-1})$")
axs[0].set_xticks(np.arange(1, 15), labels=np.repeat([""], 14))
# aesthetics second row Fup
axs[1].set_title("")
axs[1].set_xlabel("ecRad band (nm)")
axs[1].set_ylabel("Spectral Upward \nIrradiance \n$(\mathrm{W\,m}^{-2}\mathrm{\,nm}^{-1})$")
axs[1].set_xticks(np.arange(1, 15), labels=ecrad_xlabels, rotation=45)
for ax in axs:
    ax.grid()
    ax.set_xlim(4, 12)
    ax.set_ylim((0, 300))
    ax.invert_xaxis()
plt.subplots_adjust(bottom=0.28)
# plt.show()
# figname = f"{plot_path}/cirrus-hl_smart_banded_spectra_sections_{flight}.png"
figname = f"{plot_path}/cirrus-hl_smart_banded_spectra_sections127_{flight}.png"
plt.savefig(figname, dpi=200)
log.info(f"Saved {figname}")
plt.close()

# %% plot SMART reflectivity for each flight section
sections = dict()
for i, (st, et) in enumerate(zip(start_dts, end_dts)):
    sections[f"mean_spectra_{i}"] = reflectivity.sel(time=slice(st, et)).mean(dim="time")

h.set_cb_friendly_colors()
plt.rc("font", family="serif")
fig, ax = plt.subplots(figsize=(40 * cm, 22 * cm))
# SMART reflectivity
# for section, label in zip(sections, labels):
#     sections[section].plot(ax=ax, label=label, linewidth=4)

# plot only section 1, 2 and 7
# reflectivity
sections[f"mean_spectra_0"].plot(ax=ax, label=labels[0], linewidth=4)
sections[f"mean_spectra_1"].plot(ax=ax, label=labels[1], linewidth=4)
sections[f"mean_spectra_6"].plot(ax=ax, label=labels[6], color="#44AA99", linewidth=4)

# aesthetics
# ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=3, fontsize=18)
ax.legend(loc=1, ncol=2, fontsize=22)
ax.set_title(f"Mean Spectral Reflectivity derived from SMART Measurements", size=26)
ax.set_xlabel("Wavelength (nm)", size=24)
ax.set_ylabel("Reflectivity", size=24)
ax.tick_params(axis="x", labelsize=22)
ax.tick_params(axis="y", labelsize=22)
ax.set_ylim((0, 1))
ax.grid()
plt.tight_layout()
# plt.subplots_adjust(bottom=0.18)
# plt.show()
# figname = f"{plot_path}/cirrus-hl_smart_reflectivity_sections_{flight}.png"
figname = f"{plot_path}/cirrus-hl_smart_reflectivity_sections127_{flight}.png"
plt.savefig(figname, dpi=200)
log.info(f"Saved {figname}")
plt.close()

# %% plot banded SMART reflectivity for each flight section
sections = dict()
for i, (st, et) in enumerate(zip(start_dts, end_dts)):
    sections[f"mean_spectra_{i}"] = reflectivity_banded.sel(time=slice(st, et)).mean(dim="time")

h.set_cb_friendly_colors()
plt.rc("font", family="serif")
fig, ax = plt.subplots(figsize=(26 * cm, 16 * cm))
# banded SMART measurement
# for section, label in zip(sections, labels):
#     sections[section].plot(ax=ax, label=label, linewidth=4)

# plot only section 1, 2 and 7
# Fdw
sections[f"mean_spectra_0"].plot(ax=ax, label=labels[0], linewidth=4)
sections[f"mean_spectra_1"].plot(ax=ax, label=labels[1], linewidth=4)
sections[f"mean_spectra_6"].plot(ax=ax, label=labels[6], color="#44AA99", linewidth=4)

# aesthetics
# ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=3)
ax.set_title(f"Mean Reflectivity derived from SMART Measurements\n integrated over ecRad bands", size=24)
ax.set_xticks(np.arange(1, 15), labels=ecrad_xlabels, rotation=45, size=16)
ax.tick_params(axis="y", labelsize=16)
ax.set_xlabel("ecRad band (nm)", size=20)
ax.set_ylabel("Reflectivity", size=20)
ax.set_xlim((4, 12))
ax.set_ylim((0, 1))
ax.grid()
ax.invert_xaxis()
plt.tight_layout()
# plt.subplots_adjust(bottom=0.28)
# plt.show()
# figname = f"{plot_path}/cirrus-hl_smart_banded_reflectivity_sections_{flight}.png"
figname = f"{plot_path}/cirrus-hl_smart_banded_reflectivity_sections127_{flight}.png"
plt.savefig(figname, dpi=200)
log.info(f"Saved {figname}")
plt.close()

# %% plot ecRad bands for each section using Fu-IFS parameterization Fdw
sections = dict()
for i, (st, et) in enumerate(zip(start_dts, end_dts)):
    level = height_level_da.sel(time=slice(st, et)).median(dim="time")
    sections[f"mean_spectra_{i}"] = ecrad_fu["spectral_flux_dn_sw"].sel(time=slice(st, et),
                                                                        half_level=level).mean(dim="time")

h.set_cb_friendly_colors()
plt.rc("font", family="serif")
fig, ax = plt.subplots(figsize=(10, 6))
# ecRad simulation
for section, label in zip(sections, labels):
    sections[section].plot(ax=ax, label=label, linewidth=4)

# aesthetics
ax.set_ylim((0, 300))
ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=3)
ax.set_title(f"Mean Downward Irradiance for each Flight Section - ecRad Fu-IFS Parameterization")
ax.set_xticks(np.arange(1, 15), labels=ecrad_xlabels, rotation=45)
ax.set_xlabel("ecRad band (nm)")
ax.invert_xaxis()
ax.set_ylabel("Spectral Irradiance $(\mathrm{W\,m}^{-2}\mathrm{\,nm}^{-1})$")
ax.grid()
plt.tight_layout()
plt.subplots_adjust(bottom=0.4)
# plt.show()
figname = f"{plot_path}/cirrus-hl_ecrad_fdw_fu_sections_{flight}.png"
plt.savefig(figname, dpi=100)
log.info(f"Saved {figname}")
plt.close()

# %% plot ecRad bands for each section using Fu parameterization Fdw Fup
sections = dict()
sections_fup = dict()
for i, (st, et) in enumerate(zip(start_dts, end_dts)):
    level = height_level_da.sel(time=slice(st, et)).median(dim="time")
    sections[f"mean_spectra_{i}"] = ecrad_fu["spectral_flux_dn_sw"].sel(time=slice(st, et),
                                                                        half_level=level).mean(dim="time")
    sections_fup[f"mean_spectra_{i}"] = ecrad_fu["spectral_flux_up_sw"].sel(time=slice(st, et),
                                                                            half_level=level).mean(dim="time")

h.set_cb_friendly_colors()
plt.rc("font", family="serif", size=12)
fig, axs = plt.subplots(nrows=2, figsize=(10, 6))
# Fdw ecRad simulation
# for section, label in zip(sections, labels):
#     sections[section].plot(ax=axs[0], label=label, linewidth=4)
# Fup ecRad simulation
# for section, label in zip(sections_fup, labels):
#     sections_fup[section].plot(ax=axs[1])

# plot only section 1, 2 and 7
# Fdw
sections[f"mean_spectra_0"].plot(ax=axs[0], label=labels[0], linewidth=4)
sections[f"mean_spectra_1"].plot(ax=axs[0], label=labels[1], linewidth=4)
sections[f"mean_spectra_6"].plot(ax=axs[0], label=labels[6], color="#44AA99", linewidth=4)
# Fup
sections_fup[f"mean_spectra_0"].plot(ax=axs[1], linewidth=4)
sections_fup[f"mean_spectra_1"].plot(ax=axs[1], linewidth=4)
sections_fup[f"mean_spectra_6"].plot(ax=axs[1], color="#44AA99", linewidth=4)

# aesthetics first row Fdw
axs[0].legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=3)
axs[0].set_title(f"Mean Irradiance for Flight Section - ecRad Fu-IFS Parameterization")
axs[0].set_xlabel("")
axs[0].set_ylabel("Spectral Downward \nIrradiance \n$(\mathrm{W\,m}^{-2}\mathrm{\,nm}^{-1})$")
axs[0].set_xticks(np.arange(1, 15), labels=np.repeat([""], 14))
# aesthetics second row Fup
axs[1].set_title("")
axs[1].set_xlabel("ecRad band (nm)")
axs[1].set_ylabel("Spectral Upward \nIrradiance \n$(\mathrm{W\,m}^{-2}\mathrm{\,nm}^{-1})$")
axs[1].set_xticks(np.arange(1, 15), labels=ecrad_xlabels, rotation=45)
for ax in axs:
    ax.grid()
    ax.set_xlim(4, 12)
    ax.set_ylim((0, 300))
    ax.invert_xaxis()
# plt.tight_layout()
plt.subplots_adjust(bottom=0.28)
# plt.show()
# figname = f"{plot_path}/cirrus-hl_ecrad_spectra_fu_sections_{flight}.png"
figname = f"{plot_path}/cirrus-hl_ecrad_spectra_fu_sections127_{flight}.png"
plt.savefig(figname, dpi=100)
log.info(f"Saved {figname}")
plt.close()

# %% plot ecRad bands for each section using baran2017 parameterization
sections = dict()
for i, (st, et) in enumerate(zip(start_dts, end_dts)):
    level = height_level_da.sel(time=slice(st, et)).median(dim="time")
    sections[f"mean_spectra_{i}"] = ecrad_baran["spectral_flux_dn_sw"].sel(time=slice(st, et),
                                                                           half_level=level).mean(dim="time")

h.set_cb_friendly_colors()
plt.rc("font", family="serif")
fig, ax = plt.subplots(figsize=(10, 6))
# ecRad simulation
for section, label in zip(sections, labels):
    sections[section].plot(ax=ax, label=label, linewidth=4)

# aesthetics
ax.set_ylim((0, 300))
ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=3)
ax.set_title(f"Mean Downward Irradiance for each Flight Section - ecRad Baran2017 Parameterization")
ax.set_xticks(np.arange(1, 15), labels=ecrad_xlabels, rotation=45)
ax.set_xlabel("ecRad band (nm)")
ax.invert_xaxis()
ax.set_ylabel("Spectral Irradiance $(\mathrm{W\,m}^{-2}\mathrm{\,nm}^{-1})$")
ax.grid()
plt.tight_layout()
plt.subplots_adjust(bottom=0.4)
# plt.show()
figname = f"{plot_path}/cirrus-hl_ecrad_fdw_baran2017_sections_{flight}.png"
plt.savefig(figname, dpi=100)
log.info(f"Saved {figname}")
plt.close()

# %% plot ecRad bands for each section using Baran2016 parameterization
sections = dict()
for i, (st, et) in enumerate(zip(start_dts, end_dts)):
    level = height_level_da.sel(time=slice(st, et)).median(dim="time")
    sections[f"mean_spectra_{i}"] = ecrad_baran16["spectral_flux_dn_sw"].sel(time=slice(st, et),
                                                                             half_level=level).mean(dim="time")

h.set_cb_friendly_colors()
plt.rc("font", family="serif")
fig, ax = plt.subplots(figsize=(10, 6))
# ecRad simulation
for section, label in zip(sections, labels):
    sections[section].plot(ax=ax, label=label, linewidth=4)

# aesthetics
ax.set_ylim((0, 300))
ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=3)
ax.set_title(f"Mean Downward Irradiance for each Flight Section - ecRad Baran2016 Parameterization")
ax.set_xticks(np.arange(1, 15), labels=ecrad_xlabels, rotation=45)
ax.set_xlabel("ecRad band (nm)")
ax.invert_xaxis()
ax.set_ylabel("Spectral Irradiance $(\mathrm{W\,m}^{-2}\mathrm{\,nm}^{-1})$")
ax.grid()
plt.tight_layout()
plt.subplots_adjust(bottom=0.4)
# plt.show()
figname = f"{plot_path}/cirrus-hl_ecrad_fdw_baran2016_sections_{flight}.png"
plt.savefig(figname, dpi=100)
log.info(f"Saved {figname}")
plt.close()

# %% plot ecRad bands for each section using Yi parameterization
sections = dict()
for i, (st, et) in enumerate(zip(start_dts, end_dts)):
    level = height_level_da.sel(time=slice(st, et)).median(dim="time")
    sections[f"mean_spectra_{i}"] = ecrad_yi["spectral_flux_dn_sw"].sel(time=slice(st, et),
                                                                        half_level=level).mean(dim="time")

h.set_cb_friendly_colors()
plt.rc("font", family="serif")
fig, ax = plt.subplots(figsize=(10, 6))
# ecRad simulation
for section, label in zip(sections, labels):
    sections[section].plot(ax=ax, label=label, linewidth=4)

# aesthetics
ax.set_ylim((0, 300))
ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=3)
ax.set_title(f"Mean Downward Irradiance for each Flight Section - ecRad Yi Parameterization")
ax.set_xticks(np.arange(1, 15), labels=ecrad_xlabels, rotation=45)
ax.set_xlabel("ecRad band (nm)")
ax.invert_xaxis()
ax.set_ylabel("Spectral Irradiance $(\mathrm{W\,m}^{-2}\mathrm{\,nm}^{-1})$")
ax.grid()
plt.tight_layout()
plt.subplots_adjust(bottom=0.4)
# plt.show()
figname = f"{plot_path}/cirrus-hl_ecrad_fdw_yi_sections_{flight}.png"
plt.savefig(figname, dpi=100)
log.info(f"Saved {figname}")
plt.close()

# %% plot ecrad reflectivity for each flight section Fu
sections = dict()
for i, (st, et) in enumerate(zip(start_dts, end_dts)):
    level = height_level_da.sel(time=slice(st, et)).median(dim="time")
    sections[f"mean_spectra_{i}"] = reflectivity_ecrad_fu.sel(time=slice(st, et), half_level=level).mean(dim="time")

h.set_cb_friendly_colors()
plt.rc("font", family="serif")
fig, ax = plt.subplots(figsize=(26 * cm, 16 * cm))
# ecrad simulation
# for section, label in zip(sections, labels):
#     sections[section].plot(ax=ax, label=label, linewidth=4)

# plot only section 1, 2 and 7
# Fdw
sections[f"mean_spectra_0"].plot(ax=ax, label=labels[0], linewidth=4)
sections[f"mean_spectra_1"].plot(ax=ax, label=labels[1], linewidth=4)
sections[f"mean_spectra_6"].plot(ax=ax, label=labels[6], color="#44AA99", linewidth=4)

# aesthetics
# ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=3)
ax.set_title(f"Mean Reflectivity derived from ecRad Simulations \n Fu-IFS Parameterization", size=24)
# ax.set_title("")
ax.set_xticks(np.arange(1, 15), labels=ecrad_xlabels, rotation=45, size=16)
ax.tick_params(axis="y", labelsize=16)
ax.set_xlabel("ecRad band (nm)", size=20)
ax.set_ylabel("Reflectivity", size=20)
ax.set_xlim((4, 12))
ax.set_ylim((0, 1))
ax.grid()
ax.invert_xaxis()
plt.tight_layout()
# plt.subplots_adjust(bottom=0.31)
plt.show()
# figname = f"{plot_path}/cirrus-hl_ecrad_reflectivity_fu_sections_{flight}.png"
# figname = f"{plot_path}/cirrus-hl_ecrad_reflectivity_fu_sections127_{flight}.png"
# plt.savefig(figname, dpi=200)
# log.info(f"Saved {figname}")
plt.close()

# %% plot ecrad reflectivity for each flight section Baran2017
sections = dict()
for i, (st, et) in enumerate(zip(start_dts, end_dts)):
    level = height_level_da.sel(time=slice(st, et)).median(dim="time")
    sections[f"mean_spectra_{i}"] = reflectivity_ecrad_baran.sel(time=slice(st, et), half_level=level).mean(
        dim="time")

h.set_cb_friendly_colors()
plt.rc("font", family="serif")
fig, ax = plt.subplots(figsize=(10, 6))
# ecrad simulation
for section, label in zip(sections, labels):
    sections[section].plot(ax=ax, label=label, linewidth=4)

# aesthetics
ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=3)
ax.set_title(f"Mean Reflectivity derived from ecRad Simulations - Baran2017 Parameterization")
ax.set_xticks(np.arange(1, 15), labels=ecrad_xlabels, rotation=45)
ax.set_xlabel("ecRad band (nm)")
ax.set_ylabel("Reflectivity")
ax.set_ylim((0, 1))
ax.grid()
ax.invert_xaxis()
plt.tight_layout()
plt.subplots_adjust(bottom=0.4)
plt.show()
# figname = f"{plot_path}/cirrus-hl_ecrad_reflectivity_baran2017_sections_{flight}.png"
# plt.savefig(figname, dpi=200)
# log.info(f"Saved {figname}")
plt.close()

# %% plot ecrad reflectivity for each flight section Baran2016
sections = dict()
for i, (st, et) in enumerate(zip(start_dts, end_dts)):
    level = height_level_da.sel(time=slice(st, et)).median(dim="time")
    sections[f"mean_spectra_{i}"] = reflectivity_ecrad_baran16.sel(time=slice(st, et),
                                                                   half_level=level).mean(dim="time")

h.set_cb_friendly_colors()
plt.rc("font", family="serif")
fig, ax = plt.subplots(figsize=(10, 6))
# ecrad simulation
for section, label in zip(sections, labels):
    sections[section].plot(ax=ax, label=label, linewidth=4)

# aesthetics
ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=3)
ax.set_title(f"Mean Reflectivity derived from ecRad Simulations - Baran2016 Parameterization")
ax.set_xticks(np.arange(1, 15), labels=ecrad_xlabels, rotation=45)
ax.set_xlabel("ecRad band (nm)")
ax.set_ylabel("Reflectivity")
ax.set_ylim((0, 1))
ax.grid()
ax.invert_xaxis()
plt.tight_layout()
plt.subplots_adjust(bottom=0.4)
plt.show()
# figname = f"{plot_path}/cirrus-hl_ecrad_reflectivity_baran2016_sections_{flight}.png"
# plt.savefig(figname, dpi=200)
# log.info(f"Saved {figname}")
plt.close()

# %% plot ecrad reflectivity for each flight section Yi
sections = dict()
for i, (st, et) in enumerate(zip(start_dts, end_dts)):
    level = height_level_da.sel(time=slice(st, et)).median(dim="time")
    sections[f"mean_spectra_{i}"] = reflectivity_ecrad_yi.sel(time=slice(st, et), half_level=level).mean(dim="time")

h.set_cb_friendly_colors()
plt.rc("font", family="serif")
fig, ax = plt.subplots(figsize=(26 * cm, 16 * cm))
# ecrad simulation
# for section, label in zip(sections, labels):
#     sections[section].plot(ax=ax, label=label, linewidth=4)

# plot only section 1, 2 and 7
# Fdw
sections[f"mean_spectra_0"].plot(ax=ax, label=labels[0], linewidth=4)
sections[f"mean_spectra_1"].plot(ax=ax, label=labels[1], linewidth=4)
sections[f"mean_spectra_6"].plot(ax=ax, label=labels[6], color="#44AA99", linewidth=4)

# aesthetics
# ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=3)
ax.set_title(f"Mean Reflectivity derived from ecRad Simulations \n Yi Parameterization", size=24)
ax.set_xticks(np.arange(1, 15), labels=ecrad_xlabels, rotation=45, size=16)
ax.tick_params(axis="y", labelsize=16)
ax.set_xlabel("ecRad band (nm)", size=20)
ax.set_ylabel("Reflectivity", size=20)
ax.set_xlim((4, 12))
ax.set_ylim((0, 1))
ax.grid()
ax.invert_xaxis()
plt.tight_layout()
# plt.subplots_adjust(bottom=0.28)
plt.show()
# figname = f"{plot_path}/cirrus-hl_ecrad_reflectivity_yi_sections_{flight}.png"
# figname = f"{plot_path}/cirrus-hl_ecrad_reflectivity_yi_sections127_{flight}.png"
# plt.savefig(figname, dpi=200)
# log.info(f"Saved {figname}")
plt.close()

# %% plot libRadtran spectra for each flight section Fdw
sections = dict()
for i, (st, et) in enumerate(zip(start_dts, end_dts)):
    sections[f"mean_spectra_{i}"] = libradtran.fdw.sel(time=slice(st, et)).mean(dim="time")

h.set_cb_friendly_colors()
plt.rc("font", family="serif", size=12)
fig, ax = plt.subplots(figsize=(10, 6))
# Fdw libRadtran simulations
for section, label in zip(sections, labels):
    sections[section].plot(ax=ax, label=label, linewidth=4)

# aesthetics
ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=3)
ax.set_title(f"Mean Clear Sky Downward Irradiance simulated by libRadtran")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Spectral Irradiance $(\mathrm{W\,m}^{-2}\mathrm{\,nm}^{-1})$")
ax.grid()
ax.set_xlim((300, 2100))
ax.set_ylim((0, 1.65))
plt.tight_layout()
plt.subplots_adjust(bottom=0.25)
plt.show()
# figname = f"{plot_path}/cirrus-hl_libradtran_fdw_sections_{flight}.png"
# plt.savefig(figname, dpi=200)
# log.info(f"Saved {figname}")
plt.close()

# %% plot banded libRadtran spectra for each flight section Fdw
sections = dict()
for i, (st, et) in enumerate(zip(start_dts, end_dts)):
    sections[f"mean_spectra_{i}"] = fdw_lib_banded.sel(time=slice(st, et)).mean(dim="time")

h.set_cb_friendly_colors()
plt.rc("font", family="serif", size=12)
fig, ax = plt.subplots(figsize=(10, 6))
# Fdw libRadtran simulation
for section, label in zip(sections, labels):
    sections[section].plot(ax=ax, label=label, linewidth=4)

# aesthetics
ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=3)
ax.set_title(f"Mean Clear Sky Irradiance simulated by libRadtran integrated over ecRad Bands")
ax.set_ylabel("Spectral Irradiance $(\mathrm{W\,m}^{-2}\mathrm{\,nm}^{-1})$")
ax.set_xlabel("ecRad band (nm)")
ax.set_xticks(np.arange(1, 15), labels=ecrad_xlabels, rotation=45)
ax.grid()
ax.set_xlim(4, 12)
ax.set_ylim((0, 300))
ax.invert_xaxis()
plt.tight_layout()
plt.subplots_adjust(bottom=0.4)
plt.show()
# figname = f"{plot_path}/cirrus-hl_libradtran_banded_spectra_sections_{flight}.png"
# plt.savefig(figname, dpi=200)
# log.info(f"Saved {figname}")
plt.close()
# %% read in SMART measurement from HALO-AC3 case study
smart_dir = h.get_path("calibrated", "HALO-AC3_20220411_HALO_RF17", "halo-ac3")
fdw_vnir = xr.open_dataset(f"{smart_dir}/HALO-AC3_HALO_SMART_Fdw_VNIR_20220411_RF17.nc")
fdw_swir = xr.open_dataset(f"{smart_dir}/HALO-AC3_HALO_SMART_Fdw_SWIR_20220411_RF17.nc")

fdw_rf17 = smart.merge_vnir_swir_nc(fdw_vnir, fdw_swir)

# set start and stops of above cirrus and below cirrus section
starts = [pd.to_datetime("2022-04-11 10:50"), pd.to_datetime("2022-04-11 11:38")]
ends = [pd.to_datetime("2022-04-11 11:02"), pd.to_datetime("2022-04-11 11:58")]
wavelength_sel = (fdw_rf17.wavelength > 400) & (fdw_rf17.wavelength < 800) | (fdw_rf17.wavelength > 1050) & \
                 (fdw_rf17.wavelength < 2100)
fdw_rf17_plot = fdw_rf17.where(wavelength_sel)

# %% plot mean spectra above and below cirrus
sections = dict()
for i, (st, et) in enumerate(zip(starts, ends)):
    sections[f"mean_spectra_{i}"] = fdw_rf17_plot.Fdw.sel(time=slice(st, et), wavelength=slice(400, 2100)).mean(
        dim="time")

h.set_cb_friendly_colors()
plt.rc("font", family="serif")
fig, ax = plt.subplots(figsize=(32 * cm, 20 * cm))
# Fdw
sections[f"mean_spectra_0"].plot(ax=ax, linewidth=4, label="Above Cirrus", color="#44AA99")
sections[f"mean_spectra_1"].plot(ax=ax, linewidth=4, label="Below Cirrus")

# aesthetics
ax.legend(fontsize=20)
ax.set_title(f"Mean Spectral Downward Irradiance - SMART Measurement", size=26)
ax.set_xlabel("Wavelength (nm)", size=22)
ax.set_ylabel("Spectral Irradiance $(\mathrm{W\,m}^{-2}\mathrm{\,nm}^{-1})$", size=22)
ax.tick_params(axis="x", labelsize=20)
ax.tick_params(axis="y", labelsize=20)
ax.grid()
plt.tight_layout()
# plt.show()
figname = f"C:/Users/Johannes/Documents/Doktor/campaigns/HALO-AC3/case_studies/HALO-AC3_20220411_HALO_RF17/halo-ac3_smart_fdw_RF17.png"
plt.savefig(figname, dpi=200)
log.info(f"Saved {figname}")
plt.close()

# %% calculate pressure height for ecrad_input
q_air = 1.292
g_geo = 9.81
pressure_hl = ecrad_input["pressure_hl"]
ecrad_input["press_height"] = -(pressure_hl[:, 137]) * np.log(pressure_hl[:, :] / pressure_hl[:, 137]) / (
        q_air * g_geo)
# replace TOA height (calculated as infinity) with nan
ecrad_input["press_height"] = ecrad_input["press_height"].where(ecrad_input["press_height"] != np.inf, np.nan)

# %% plot aircraft track through ecrad input, combine clwc and ciwc
variable = "clwc"
units = dict(clwc="g/kg", ciwc="g/kg", q_ice="g/kg", cswc="g/kg", crwc="g/kg", t="K", re_ice="m x $10^{-6}$",
             re_liquid="m x $10^{-6}$")
scale_factor = dict(clwc=1000, ciwc=1000, q_ice=1000, cswc=1000, crwc=1000, t=1, re_ice=1e6, re_liquid=1e6)
colorbarlabel = dict(clwc="Cloud Liquid Water Content", ciwc="Cloud Ice Water Content", q_ice="Ice and Snow Content",
                     cswc="Cloud Snow Water Content", crwc="Cloud Rain Water Content", t="Temperature",
                     re_ice="Effective Ice Particle Radius", re_liquid="Effective Droplet Radius")
cmap = dict(t="bwr")
cmap = cmap[variable] if variable in cmap else cmr.get_sub_cmap("cmr.flamingo", .25, .9)
cmap = plt.get_cmap(cmap).copy().reversed()
cmap.set_under("white")
x_sel = (pd.Timestamp(2021, 6, 29, 10), pd.Timestamp(2021, 6, 29, 12, 10))
ecrad_plot = ecrad_input[variable] * scale_factor[variable]
ecrad_plot = ecrad_plot.assign_coords(level=ecrad_input["press_height"].isel(time=100, drop=True)[1:].values / 1000)
ecrad_plot = ecrad_plot.rename(level="height")
aircraft_height_plot = aircraft_height_da / 1000
norm = colors.TwoSlopeNorm(vmin=193, vcenter=273, vmax=293)

plt.rcdefaults()
h.set_cb_friendly_colors()
plt.rc('font', size=20)
plt.rc('lines', linewidth=3)
fig, ax = plt.subplots(figsize=(40*cm, 20*cm))
# ecrad input clwc
ecrad_plot = ecrad_plot.where(ecrad_plot > 0.001)  # filter very low values
ecrad_plot.plot(x="time", y="height", cmap=cmap, ax=ax, robust=True,
                cbar_kwargs={"pad": 0.01, "label": f"{colorbarlabel[variable]} ({units[variable]})"})

# ecrad input ciwc
variable = "ciwc"
cmap = cmr.get_sub_cmap("cmr.freeze", .25, 0.85)
cmap = plt.get_cmap(cmap).copy().reversed()
ecrad_plot = ecrad_input[variable] * scale_factor[variable]
ecrad_plot = ecrad_plot.assign_coords(level=ecrad_input["press_height"].isel(time=100, drop=True)[1:].values / 1000)
ecrad_plot = ecrad_plot.rename(level="height")
ecrad_plot = ecrad_plot.where(ecrad_plot > 0.01)  # filter very low values
ecrad_plot.plot(x="time", y="height", cmap=cmap, ax=ax, robust=True, alpha=0.6,
                cbar_kwargs={"pad": 0.01, "label": f"{colorbarlabel[variable]} ({units[variable]})"})

# aircraft altitude through model
aircraft_height_plot.plot(x="time", color="k", ax=ax, label="HALO altitude")

ax.legend(loc=2, fontsize=22)
ax.set_xlim(x_sel)
ax.set_ylim(0, 14)
h.set_xticks_and_xlabels(ax, x_sel[1] - x_sel[0])
ax.tick_params(axis="x", labelsize=18)
ax.tick_params(axis="y", labelsize=18)
ax.grid()
ax.set_title("IFS Output along HALO Flight Track 29. June 2021", size=24)
ax.set_ylabel("Pressure Height (km)", size=22)
ax.set_xlabel("Time (UTC)", size=22)
plt.tight_layout()
# plt.show()
figname = f"{plot_path}/cirrus-hl_IFS_clwc-ciwc_HALO_alt_{flight}.png"
plt.savefig(figname, dpi=300)
log.info(f"Saved {figname}")
plt.close()

# %% replace band coordinates in ecRad data
ecrad_fu_2 = reflectivity_ecrad_fu.assign_coords(band_sw=ecrad_bands).isel(band_sw=slice(3, 12))
ecrad_yi_2 = reflectivity_ecrad_yi.assign_coords(band_sw=ecrad_bands).isel(band_sw=slice(3, 12))
reflectivity_banded_2 = reflectivity_banded.assign_coords(ecrad_band=ecrad_bands).isel(ecrad_band=slice(3, 12))

# %% create list with ecRad band limits for plot
band_limits = [h.ecRad_bands[key] for key in h.ecRad_bands]
band_limits = [item for t in band_limits[3:12] for item in t]
band_limits.sort()
band_limits = band_limits[::2]
band_limits.append(2150)

# %% plot SMART reflectivity on a continuous scale with ecRad bands
sections_smart = dict()
for i, (st, et) in enumerate(zip(start_dts, end_dts)):
    level = height_level_da.sel(time=slice(st, et)).median(dim="time")
    sections_smart[f"mean_spectra_{i}"] = reflectivity.sel(time=slice(st, et)).mean(dim="time")

h.set_cb_friendly_colors()
plt.rc("font", family="serif")
fig, ax = plt.subplots(figsize=(10, 6))
ax.vlines(band_limits, ymin=0, ymax=1, colors=["grey"], linestyles="dashed")
band_names = [key.replace("and", "") for key in h.ecRad_bands][3:12]
for x, name in zip(ecrad_fu_2.band_sw, band_names):
    ax.text(x-35, 0.93, name, size=14)

# SMART data
sections_smart[f"mean_spectra_0"].plot(ax=ax, label=labels[0], linewidth=3, color="#88CCEE")
sections_smart[f"mean_spectra_1"].plot(ax=ax, label=labels[1], linewidth=3, color="#CC6677")
sections_smart[f"mean_spectra_6"].plot(ax=ax, label=labels[6], linewidth=3, color="#44AA99")

ax.set_title(f"Mean Spectral Reflectivity derived from SMART Measurements", size=20)
ax.legend(loc="upper right", bbox_to_anchor=(1, 0.9), fontsize=16)
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)
ax.set_xlabel("Wavelength (nm)", size=20)
ax.set_ylabel("Reflectivity", size=20)
ax.set_ylim((0, 1))
ax.grid(axis="y")
plt.tight_layout()
# plt.show()
figname = f"{plot_path}/cirrus-hl_smart_reflectivity_sections127_continuous_{flight}.png"
plt.savefig(figname, dpi=200)
log.info(f"Saved {figname}")
plt.close()

# %% plot SMART reflectivity and banded reflectivity on a continuous scale
sections_smart, sections_smart_banded = dict(), dict()
for i, (st, et) in enumerate(zip(start_dts, end_dts)):
    level = height_level_da.sel(time=slice(st, et)).median(dim="time")
    sections_smart[f"mean_spectra_{i}"] = reflectivity.sel(time=slice(st, et)).mean(dim="time")
    sections_smart_banded[f"mean_spectra_{i}"] = reflectivity_banded_2.sel(time=slice(st, et)).mean(dim="time")

h.set_cb_friendly_colors()
plt.rc("font", family="serif")
fig, ax = plt.subplots(figsize=(10, 6))

ax.vlines(band_limits, ymin=0, ymax=1, colors=["grey"], linestyles="dashed")
band_names = [key.replace("and", "") for key in h.ecRad_bands][3:12]
for x, name in zip(ecrad_fu_2.band_sw, band_names):
    ax.text(x-35, 0.93, name, size=14)

# SMART data
sections_smart[f"mean_spectra_0"].plot(ax=ax, label=labels[0], linewidth=1, color="#88CCEE")
sections_smart[f"mean_spectra_1"].plot(ax=ax, label=labels[1], linewidth=1, color="#CC6677")
sections_smart[f"mean_spectra_6"].plot(ax=ax, label=labels[6], linewidth=1, color="#44AA99")

# SMART data banded
sections_smart_banded[f"mean_spectra_0"].plot(ax=ax, label=labels[0], ls="", color="#88CCEE", marker="o", markersize=14)
sections_smart_banded[f"mean_spectra_1"].plot(ax=ax, label=labels[1], ls="", color="#CC6677", marker="o", markersize=14)
sections_smart_banded[f"mean_spectra_6"].plot(ax=ax, label=labels[6], color="#44AA99", ls="", marker="o", markersize=14)

# legend for smart + smart banded
handles, llabels = ax.get_legend_handles_labels()
ax.legend(handles=handles[3:], loc="upper right", bbox_to_anchor=(1, 0.9), fontsize=16)

ax.set_title(f"Mean Spectral Reflectivity derived from SMART Measurements", size=20)
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)
ax.set_xlabel("Wavelength (nm)", size=20)
ax.set_ylabel("Reflectivity", size=20)
ax.set_ylim((0, 1))
ax.grid(axis="y")
plt.tight_layout()
# plt.show()
figname = f"{plot_path}/cirrus-hl_smart_reflectivity_banded_sections127_continuous_{flight}.png"
plt.savefig(figname, dpi=200)
log.info(f"Saved {figname}")
plt.close()

# %% plot ecRad reflectivity on a continuous scale Fu-IFS
sections = dict()
for i, (st, et) in enumerate(zip(start_dts, end_dts)):
    level = height_level_da.sel(time=slice(st, et)).median(dim="time")
    sections[f"mean_spectra_{i}"] = ecrad_fu_2.sel(time=slice(st, et), half_level=level).mean(dim="time")

h.set_cb_friendly_colors()
plt.rc("font", family="serif")
fig, ax = plt.subplots(figsize=(10, 6))
ax.vlines(band_limits, ymin=0, ymax=1, colors=["grey"], linestyles="dashed")
band_names = [key.replace("and", "") for key in h.ecRad_bands][3:12]
for x, name in zip(ecrad_fu_2.band_sw, band_names):
    ax.text(x-35, 0.93, name, size=14)

# ecRad data Fu-IFS
sections[f"mean_spectra_0"].plot(ax=ax, label=labels[0], ls="", marker="d", markersize=14, color="#88CCEE")
sections[f"mean_spectra_1"].plot(ax=ax, label=labels[1], ls="", marker="d", markersize=14, color="#CC6677")
sections[f"mean_spectra_6"].plot(ax=ax, label=labels[6], color="#44AA99", ls="", marker="d", markersize=14)

ax.set_title(f"Mean Spectral Reflectivity derived from ecRad Simulation\n Fu-IFS", size=20)
ax.legend(loc="upper right", bbox_to_anchor=(1, 0.9), fontsize=16)
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)
ax.set_xlabel("Wavelength (nm)", size=20)
ax.set_ylabel("Reflectivity", size=20)
ax.set_ylim((0, 1))
ax.grid(axis="y")
plt.tight_layout()
# plt.show()
figname = f"{plot_path}/cirrus-hl_ecrad_reflectivity_fu_sections127_continuous_{flight}.png"
plt.savefig(figname, dpi=200)
log.info(f"Saved {figname}")
plt.close()

# %% plot ecRad reflectivity on a continuous scale Fu-IFS + Yi2013
sections, sections_yi = dict(), dict()
for i, (st, et) in enumerate(zip(start_dts, end_dts)):
    level = height_level_da.sel(time=slice(st, et)).median(dim="time")
    sections[f"mean_spectra_{i}"] = ecrad_fu_2.sel(time=slice(st, et), half_level=level).mean(dim="time")
    sections_yi[f"mean_spectra_{i}"] = ecrad_yi_2.sel(time=slice(st, et), half_level=level).mean(dim="time")

h.set_cb_friendly_colors()
plt.rc("font", family="serif")
fig, ax = plt.subplots(figsize=(10, 6))
ax.vlines(band_limits, ymin=0, ymax=1, colors=["grey"], linestyles="dashed")
band_names = [key.replace("and", "") for key in h.ecRad_bands][3:12]
for x, name in zip(ecrad_fu_2.band_sw, band_names):
    ax.text(x-35, 0.93, name, size=14)

# ecRad data Fu-IFS
sections[f"mean_spectra_0"].plot(ax=ax, label=labels[0], ls="", marker="d", markersize=14, color="#88CCEE")
sections[f"mean_spectra_1"].plot(ax=ax, label=labels[1], ls="", marker="d", markersize=14, color="#CC6677")
sections[f"mean_spectra_6"].plot(ax=ax, label=labels[6], color="#44AA99", ls="", marker="d", markersize=14)
# dummy for legend
fu_legend = Line2D([], [], ls="", color="k", marker="d", markersize=14, label="Fu-IFS")

# ecRad data Yi
sections_yi[f"mean_spectra_0"].plot(ax=ax, label=labels[0], ls="", marker="X", markersize=14, color="#88CCEE")
sections_yi[f"mean_spectra_1"].plot(ax=ax, label=labels[1], ls="", marker="X", markersize=14, color="#CC6677"   )
sections_yi[f"mean_spectra_6"].plot(ax=ax, label=labels[6], color="#44AA99", ls="", marker="X", markersize=14)
# dummy for legend
yi_legend = Line2D([], [], ls="", color="k", marker="X", markersize=14, label="Yi 2013")

# legend for ecRad Fu + Yi
handles = [fu_legend, yi_legend]
ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1, 0.9), fontsize=20)

ax.set_title(f"Mean Spectral Reflectivity derived from ecRad Simulation", size=20)
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)
ax.set_xlabel("Wavelength (nm)", size=20)
ax.set_ylabel("Reflectivity", size=20)
ax.set_ylim((0, 1))
ax.grid(axis="y")
plt.tight_layout()
# plt.show()
figname = f"{plot_path}/cirrus-hl_ecrad_reflectivity_fu-yi_sections127_continuous_{flight}.png"
plt.savefig(figname, dpi=200)
log.info(f"Saved {figname}")
plt.close()
