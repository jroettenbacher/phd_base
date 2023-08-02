#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 07.06.2023

Figures for the manuscript Röttenbacher et al. 2023

- incoming solar irradiance model vs. measurement
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
import ac3airborne
from ac3airborne.tools import flightphase
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch, Polygon
import cmasher as cm
import cartopy.crs as ccrs
from tqdm import tqdm
from matplotlib import colors
from metpy import constants as mc
from metpy.calc import relative_humidity_from_specific_humidity
from metpy.units import units as u

h.set_cb_friendly_colors()
cbc = h.get_cb_friendly_colors()

# %% set paths
campaign = "halo-ac3"
plot_path = "C:/Users/Johannes/Documents/Doktor/manuscripts/2023_arctic_cirrus/figures"
keys = ["RF17", "RF18"]

# %% read in data
(
    bahamas_ds,
    bacardi_ds,
    ecrad_dicts,
    varcloud_ds,
    above_clouds,
    below_clouds,
    slices,
    ecrad_orgs,
    ifs_ds,
    dropsonde_ds
) = (dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict())

for key in keys:
    flight = meta.flight_names[key]
    date = flight[9:17]
    bacardi_path = h.get_path("bacardi", flight, campaign)
    bahamas_path = h.get_path("bahamas", flight, campaign)
    libradtran_path = h.get_path("libradtran", flight, campaign)
    libradtran_exp_path = h.get_path("libradtran_exp", flight, campaign)
    ifs_path = f"{h.get_path('ifs', flight, campaign)}/{date}"
    ecrad_path = f"{h.get_path('ecrad', flight, campaign)}/{date}"
    varcloud_path = h.get_path("varcloud", flight, campaign)
    dropsonde_path = h.get_path("dropsondes", flight, campaign)

    # filenames
    bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{key}_v1.nc"
    bacardi_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1.nc"
    libradtran_bb_solar_si = f"HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_solar_si_{date}_{key}.nc"
    libradtran_bb_thermal_si = f"HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_thermal_si_{date}_{key}.nc"
    ifs_file = f"ifs_{date}_00_ml_O1280_processed.nc"
    varcloud_file = [f for f in os.listdir(varcloud_path) if f.endswith(".nc")][0]
    dropsonde_files = [f for f in os.listdir(dropsonde_path) if f.endswith("QC.nc")]

    # read in aircraft data
    bahamas_ds[key] = reader.read_bahamas(f"{bahamas_path}/{bahamas_file}")
    bacardi = xr.open_dataset(f"{bacardi_path}/{bacardi_file}")

    # read in dropsonde data
    dropsondes = dict()
    for file in dropsonde_files:
        k = file[-11:-5]
        dropsondes[k] = xr.open_dataset(f"{dropsonde_path}/{file}")
    dropsonde_ds[key] = dropsondes

    # read in ifs data
    ifs_ds[key] = xr.open_dataset(f"{ifs_path}/{ifs_file}")

    # read in varcloud data
    varcloud = xr.open_dataset(f"{varcloud_path}/{varcloud_file}")
    varcloud = varcloud.swap_dims(time="Time", height="Height").rename(Time="time", Height="height")
    varcloud = varcloud.rename(Varcloud_Cloud_Ice_Water_Content="iwc", Varcloud_Cloud_Ice_Effective_Radius="re_ice")
    varcloud_ds[key] = varcloud

    # read in libRadtran simulation
    bb_sim_solar_si = xr.open_dataset(f"{libradtran_exp_path}/{libradtran_bb_solar_si}")
    bb_sim_thermal_si = xr.open_dataset(f"{libradtran_exp_path}/{libradtran_bb_thermal_si}")
    # interpolate simualtion onto BACARDI time
    bb_sim_solar_si_inp = bb_sim_solar_si.interp(time=bacardi.time)
    bb_sim_thermal_si_inp = bb_sim_thermal_si.interp(time=bacardi.time)

    # calculate transmissivity BACARDI/libRadtran
    bacardi["F_down_solar_sim_si"] = bb_sim_solar_si_inp.fdw
    bacardi["F_down_terrestrial_sim_si"] = bb_sim_thermal_si_inp.edn
    bacardi["transmissivity_solar"] = bacardi["F_down_solar"] / bb_sim_solar_si_inp.fdw
    bacardi["transmissivity_terrestrial"] = bacardi["F_down_terrestrial"] / bb_sim_thermal_si_inp.edn

    # calculate radiative effect from BACARDI and libRadtran sea ice simulation
    bacardi["F_net_solar"] = bacardi["F_down_solar"] - bacardi["F_up_solar"]
    bacardi["F_net_terrestrial"] = bacardi["F_down_terrestrial"] - bacardi["F_up_terrestrial"]
    bacardi["F_net_solar_sim"] = bb_sim_solar_si_inp.fdw - bb_sim_solar_si_inp.eup
    bacardi["F_net_terrestrial_sim"] = bb_sim_thermal_si_inp.edn - bb_sim_thermal_si_inp.eup
    bacardi["CRE_solar"] = bacardi["F_net_solar"] - bacardi["F_net_solar_sim"]
    bacardi["CRE_terrestrial"] = bacardi["F_net_terrestrial"] - bacardi["F_net_terrestrial_sim"]
    bacardi["CRE_total"] = bacardi["CRE_solar"] + bacardi["CRE_terrestrial"]
    bacardi_ds[key] = bacardi

    # read in ecrad data
    ecrad_versions = ["v15", "v16", "v17", "v18", "v19", "v20", "v21"]
    ecrad_dict = dict()
    ecrad_orgs[key] = xr.open_dataset(f"{ecrad_path}/ecrad_merged_inout_{date}_v15.nc")

    for k in ecrad_versions:
        ds = xr.open_dataset(f"{ecrad_path}/ecrad_merged_inout_{date}_{k}.nc")
        if "column" in ds.dims:
            if k == "v1":
                ds = ds.sel(column=16,
                            drop=True)  # select center column which corresponds to grid cell closest to aircraft
            else:
                # other versions have their nearest points selected via kdTree, thus the first column should be the closest
                ds = ds.sel(column=0, drop=True)

        ds["tiwp"] = ds.iwp.where(ds.iwp != np.inf, np.nan).sum(dim="level")

        ecrad_dict[k] = ds.copy()

    ecrad_dicts[key] = ecrad_dict

    # get flight segmentation and select below and above cloud section
    fl_segments = ac3airborne.get_flight_segments()["HALO-AC3"]["HALO"][f"HALO-AC3_HALO_{key}"]
    segments = flightphase.FlightPhaseFile(fl_segments)
    above_cloud, below_cloud = dict(), dict()
    if key == "RF17":
        above_cloud["start"] = segments.select("name", "high level 7")[0]["start"]
        above_cloud["end"] = segments.select("name", "high level 8")[0]["end"]
        below_cloud["start"] = segments.select("name", "high level 9")[0]["start"]
        below_cloud["end"] = segments.select("name", "high level 10")[0]["end"]
        above_slice = slice(above_cloud["start"], above_cloud["end"])
        below_slice = slice(below_cloud["start"], below_cloud["end"])
        case_slice = slice(pd.to_datetime("2022-04-11 10:30"), pd.to_datetime("2022-04-11 12:29"))
    else:
        above_cloud["start"] = segments.select("name", "polygon pattern 1")[0]["start"]
        above_cloud["end"] = segments.select("name", "polygon pattern 1")[0]["parts"][-1]["start"]
        below_cloud["start"] = segments.select("name", "polygon pattern 2")[0]["start"]
        below_cloud["end"] = segments.select("name", "polygon pattern 2")[0]["end"]
        above_slice = slice(above_cloud["start"], above_cloud["end"])
        below_slice = slice(below_cloud["start"], below_cloud["end"])
        case_slice = slice(above_cloud["start"], below_cloud["end"])

    above_clouds[key] = above_cloud
    below_clouds[key] = below_cloud
    slices[key] = dict(case=case_slice, above=above_slice, below=below_slice)

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
            empty_array[i, j, k] = ecrad.ice_effective_radius(25000, temperature, 1, iwc, 0, lat)

da = xr.DataArray(empty_array * 1e6, coords=[("temperature", t), ("iwc_kgkg", iwc_kgkg * 1e3), ("Latitude", lats)],
                  name="re_ice")

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
ds = ecrad_orgs["RF17"].sel(column=0)
a = 6371229  # radius of sphere assumed by IFS
distances_between_longitudes = np.pi / 180 * a * np.cos(np.deg2rad(ds.lat)) / 1000
distances_between_longitudes.min()
bahamas_ds["RF17"].IRS_LAT.max()
longitude_diff = ds.lon.diff(dim="time")
distance_between_gridcells = longitude_diff * distances_between_longitudes

_, ax = plt.subplots(figsize=h.figsize_wide)
distance_between_gridcells.plot(x="time")  # ,# y="column",
# cbar_kwargs={"label": "East-West distance between grid cells (km)"})
ax.grid()
ax.set(xlabel="Time (UTC)", ylabel="Column")
h.set_xticks_and_xlabels(ax, pd.to_timedelta(
    bahamas_ds["RF17"].time[-1].to_numpy() - bahamas_ds["RF17"].time[0].to_numpy()))
plt.tight_layout()
plt.show()
plt.close()

# %% plot IWC before and after dividing by cloud fraction
plot_ds = ecrad_dicts["RF17"]["v15"]
plot_ds["iwc_new"] = plot_ds["iwc"].where(plot_ds["cloud_fraction"] == 0,
                                          plot_ds["iwc"] / plot_ds["cloud_fraction"]) * 1e6
plot_ds = plot_ds["iwc_new"].sel(time=slices["RF17"]["below"])
plot_ds = plot_ds.to_numpy().flatten()

plt.hist(plot_ds, bins=np.arange(0, 10, 0.25), density=True)
plt.show()
plt.close()

# %% plot flight tracks of RF17 and RF18
plt.rc("font", size=6)

data_crs = ccrs.PlateCarree()
_, axs = plt.subplots(1, 2, figsize=h.figsize_equal, subplot_kw=dict(projection=ccrs.NorthPolarStereo()))

for i, key in enumerate(keys):
    # map and flight track RF17
    ax = axs[i]
    ax.set(title=f"{key} - 1{1 + i} April 2022")
    ifs = ifs_ds[key].sel(time=f"2022-04-1{1 + i} 11:00")
    ax.coastlines(alpha=0.5)
    gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=1, color='gray', alpha=0.5,
                      linestyle=':', x_inline=False, y_inline=False, rotate_labels=False)
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 10))
    gl.ylocator = mticker.FixedLocator(np.arange(60, 90, 5))
    gl.top_labels = False
    gl.right_labels = False

    # Plot the surface pressure - 11 April
    pressure_levels = np.arange(900, 1125, 5)
    ifs_press = ifs.mean_sea_level_pressure / 100
    # cp = ax.tricontour(ifs.lon, ifs.lat, ifs_press, levels=pressure_levels, colors='k', linewidths=0.5,
    #                 linestyles='solid', alpha=1, transform=data_crs)
    # cp.clabel(fontsize=2, inline=1, inline_spacing=1, fmt='%i hPa', rightside_up=True, use_clabeltext=True)

    # add seaice edge
    ci_levels = [0.8]
    # cci = ax.tricontour(ifs.lon, ifs.lat, ifs.CI, ci_levels, transform=data_crs, linestyles="--", colors="#332288",
    #                  linewidths=2)

    # add high cloud cover
    # ifs_cc = ifs.cloud_fraction.where(ifs.pressure_full < 60000, drop=True).sum(dim="level")
    # ifs_cc = ifs_cc.where(ifs_cc > 1)
    # ax.tricontourf(ifs.lon, ifs.lat, ifs_cc, levels=24, transform=data_crs, cmap="Blues", alpha=1)

    # plot flight track - 11 April
    track_lons, track_lats = bahamas_ds[key].IRS_LON, bahamas_ds[key].IRS_LAT
    ax.scatter(track_lons[::10], track_lats[::10], c="k", alpha=1, marker=".", s=1, zorder=400,
               label='HALO flight track', transform=data_crs, linestyle="solid")

    # plot dropsonde locations - 11 April
    # for i in range(dropsonde_ds.lon.shape[0]):
    #     launch_time = pd.to_datetime(dropsonde_ds.launch_time[i].values)
    #     x, y = dropsonde_ds.lon[i].mean().values, dropsonde_ds.lat[i].mean().values
    #     cross = ax.plot(x, y, "x", color="orangered", markersize=12, label="Dropsonde", transform=data_crs,
    #                     zorder=450)
    #     ax.text(x, y, f"{launch_time:%H:%M}", c="k", fontsize=12, transform=data_crs, zorder=500,
    #             path_effects=[patheffects.withStroke(linewidth=0.25, foreground="white")])

    # make legend for flight track and dropsondes - 11 April
    handles = [plt.plot([], ls="-", color="#000000", lw=1)[0],  # flight track
               # cross[0],  # dropsondes
               plt.plot([], ls="--", color="#332288", lw=2)[0],  # sea ice edge
               Patch(facecolor="royalblue")]  # cloud cover
    labels = ["HALO flight track", "Sea Ice Edge", "High Cloud Cover\nat 11:00 UTC"]
    ax.legend(handles=handles, labels=labels, framealpha=1, loc=1)

axs[0].text(0.03, 0.95, "a)", transform=axs[0].transAxes, bbox=dict(boxstyle="Round", fc="white"))
axs[1].text(0.03, 0.95, "b)", transform=axs[1].transAxes, bbox=dict(boxstyle="Round", fc="white"))
# TODO: add red rectangles
lat_corners = np.array([88, 89.5, 87, 85])
lon_corners = np.array([-37, -37, 20, 18])

poly_corners = np.zeros((len(lat_corners), 2), np.float64)
poly_corners[:, 0] = lon_corners
poly_corners[:, 1] = lat_corners

poly = Polygon(poly_corners, closed=True, ec='r', fill=False, lw=1, transform=data_crs)
axs[0].add_patch(poly)

# plt.tight_layout()
# figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_BAHAMAS_IFS_flight_tracks.png"
# plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot BACARDI four panel view with above and below cloud measurements and transmissivity
plt.rc("font", size=6.5)
_, axs = plt.subplots(2, 2, figsize=(17 * h.cm, 10 * h.cm))

# upper left panel
ax = axs[0, 0]
plot_ds = bacardi_ds["RF17"].sel(time=slices["RF17"]["case"])
time_extend = pd.to_timedelta((plot_ds.time[-1] - plot_ds.time[0]).to_numpy())
# bacardi measurements
for var in ["F_down_solar", "F_up_solar"]:
    ax.plot(plot_ds.time, plot_ds[var], label=f"{h.bacardi_labels[var]}")
# libradtran clearsky simulation
ax.plot(plot_ds.time, plot_ds.F_down_solar_sim_si, label=f"simulated {h.bacardi_labels['F_down_solar']}")
ax.legend()
h.set_xticks_and_xlabels(ax, time_extend)
ax.grid()
ax.text(0.03, 0.88, "a)", transform=ax.transAxes, bbox=dict(boxstyle="Round", fc="white"))
ax.set(title="RF 17 - 11 April 2022", ylabel=f"{h.cbarlabels['flux_dn_sw']} ({h.plot_units['flux_dn_sw']})")

# lower left panel
ax = axs[1, 0]
ax.axhline(y=1, color="k")
ax.plot(plot_ds.time, plot_ds["transmissivity_solar"], label="Solar transmissivity", color=cbc[4])
h.set_xticks_and_xlabels(ax, time_extend)
ax.grid()
ax.text(0.03, 0.88, "c)", transform=ax.transAxes, bbox=dict(boxstyle="Round", fc="white"))
ax.set(ylabel="Solar Transmissivity", xlabel="Time (UTC)")

# upper right panel
ax = axs[0, 1]
plot_ds = bacardi_ds["RF18"].sel(time=slices["RF18"]["case"])
time_extend = pd.to_timedelta((plot_ds.time[-1] - plot_ds.time[0]).to_numpy())
# bacardi measurements
for var in ["F_down_solar", "F_up_solar"]:
    ax.plot(plot_ds.time, plot_ds[var], label=f"{h.bacardi_labels[var]}")
# libradtran clearsky simulation
ax.plot(plot_ds.time, plot_ds.F_down_solar_sim_si, label=f"simulated {h.bacardi_labels['F_down_solar']}")
ax.legend()
h.set_xticks_and_xlabels(ax, time_extend)
ax.grid()
ax.text(0.03, 0.88, "b)", transform=ax.transAxes, bbox=dict(boxstyle="Round", fc="white"))
ax.set(title="RF 18 - 12 April 2022")

# lower right panel
ax = axs[1, 1]
ax.axhline(y=1, color="k")
ax.plot(plot_ds.time, plot_ds["transmissivity_solar"], label="Solar transmissivity", color=cbc[4])
h.set_xticks_and_xlabels(ax, time_extend)
ax.grid()
ax.text(0.03, 0.88, "d)", transform=ax.transAxes, bbox=dict(boxstyle="Round", fc="white"))
ax.set(xlabel="Time (UTC)")

plt.tight_layout()
figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_BACARDI_case_studies.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot BACARDI/libRadtran cloud radiative effect
plt.rc("font", size=6.5)
_, axs = plt.subplots(2, 1, figsize=(16 * h.cm, 9 * h.cm))
# RF 17
for i, k in enumerate(keys):
    ax = axs[i]
    plot_ds = bacardi_ds[k].sel(time=slices[k]["case"])
    time_extend = pd.to_timedelta((plot_ds.time[-1] - plot_ds.time[0]).to_numpy())
    ax.plot(plot_ds.time, plot_ds["CRE_solar"], label=h.bacardi_labels["CRE_solar"])
    ax.plot(plot_ds.time, plot_ds["CRE_terrestrial"], label=h.bacardi_labels["CRE_terrestrial"])
    ax.plot(plot_ds.time, plot_ds["CRE_total"], label=h.bacardi_labels["CRE_total"])
    ax.axhline(y=0, color="k")
    h.set_xticks_and_xlabels(ax, time_extend)
    ax.grid()
    ax.set(ylabel=f"Cloud radiative effect ({h.plot_units['cre_sw']})")

axs[0].text(0.03, 0.88, "a)", transform=axs[0].transAxes, bbox=dict(boxstyle="Round", fc="white"))
axs[1].text(0.03, 0.88, "b)", transform=axs[1].transAxes, bbox=dict(boxstyle="Round", fc="white"))
axs[1].set_xlabel("Time (UTC)")
axs[0].legend(loc=3)

plt.tight_layout()
figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_BACARDI_libRadtran_CRE.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% prepare IFS data for plotting
plt.rc("font", size=6.5)
_, axs = plt.subplots(2, 1, figsize=(16 * h.cm, 9 * h.cm))
for i, key in enumerate(keys):
    ax = axs[i]
    ds = ecrad_dicts[key]["v15"].sel(time=slices[key]["above"])
    ifs_plot = ds["cloud_fraction"]
    # add new z axis mean pressure altitude
    if "half_level" in ifs_plot.dims:
        new_z = ds["press_height_hl"].mean(dim="time") / 1000
    else:
        new_z = ds["press_height_full"].mean(dim="time") / 1000

    ifs_plot_new_z = list()
    for t in tqdm(ifs_plot.time, desc="New Z-Axis"):
        tmp_plot = ifs_plot.sel(time=t)
        if "half_level" in tmp_plot.dims:
            tmp_plot = tmp_plot.assign_coords(
                half_level=ds["press_height_hl"].sel(time=t, drop=True).to_numpy() / 1000)
            tmp_plot = tmp_plot.rename(half_level="height")
        else:
            tmp_plot = tmp_plot.assign_coords(
                level=ds["press_height_full"].sel(time=t, drop=True).to_numpy() / 1000)
            tmp_plot = tmp_plot.rename(level="height")

        tmp_plot = tmp_plot.interp(height=new_z.to_numpy())
        ifs_plot_new_z.append(tmp_plot)

    ifs_plot = xr.concat(ifs_plot_new_z, dim="time").sortby("height").sel(height=slice(0, 12))
    ifs_plot = ifs_plot.where(ifs_plot > 0)
    halo_plot = varcloud_ds[key].sel(time=slices[key]["case"]).Varcloud_Input_Mask
    halo_plot = halo_plot.assign_coords(height=halo_plot.height / 1000).sortby("height")
    time_extend = pd.to_timedelta((ifs_plot.time[-1] - ifs_plot.time[0]).to_numpy())

    # plot IFS cloud cover prediction and Radar lidar mask
    ifs_plot.plot(x="time", cmap=cm.sapphire, cbar_kwargs=dict(label=f"IFS {h.cbarlabels['cloud_fraction']}"), ax=ax)
    halo_plot.plot.contour(x="time", levels=[0.9], colors=cbc[5], ax=ax)
    ax.plot([], color=cbc[5], label="Radar & Lidar Mask", lw=1)
    ax.legend()
    h.set_xticks_and_xlabels(ax, time_extend)
    ax.set(xlabel="Time (UTC)", ylabel="Height (km)")
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=0, ha="center")

axs[0].set_xlabel("")
axs[0].text(0.03, 0.88, "a)", transform=axs[0].transAxes, bbox=dict(boxstyle="Round", fc="white"))
axs[1].text(0.03, 0.88, "b)", transform=axs[1].transAxes, bbox=dict(boxstyle="Round", fc="white"))
plt.tight_layout()

figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_IFS_cloud_fraction_radar_lidar_mask.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot temperature and humidity profiles from IFS and from dropsonde
plt.rc("font", size=6.5)
_, axs = plt.subplots(1, 4, figsize=(18 * h.cm, 10 * h.cm))
for i, key in enumerate(keys):
    ax = axs[i*2]
    ifs_plot = ecrad_dicts[key]["v15"].sel(time=slices[key]["case"])

    # Air temperature
    for t in ifs_plot.time:
        ifs_p = ifs_plot.sel(time=t)
        ax.plot(ifs_p.temperature_hl, ifs_p.press_height_hl / 1000, color="grey", lw=0.5)
    ds_plot = dropsonde_ds[key]
    times = ["104205", "110137"] if key == "RF17" else ["110321", "110823"]
    for k in times:
        ds = ds_plot[k]
        ds = ds.where(~np.isnan(ds.tdry), drop=True)
        ax.plot(ds.tdry + 273.15, ds.alt / 1000, label=f"Dropsonde {k[:2]}:{k[2:4]} UTC", lw=2)
    ax.set(xlim=(215, 265), ylim=(0, 12), xlabel="Air temperature (K)", title=f"{key}")
    ax.plot([], color="grey", label="IFS profiles")
    ax.grid()

    # RH
    ax = axs[i*2+1]
    ifs_plot = ecrad_dicts[key]["v15"].sel(time=slices[key]["case"])
    for t in ifs_plot.time:
        ifs_p = ifs_plot.sel(time=t)
        rh = relative_humidity_from_specific_humidity(ifs_p.pressure_full * u.Pa, ifs_p.t * u.K, ifs_p.q * u("kg/kg"))
        ax.plot(rh * 100, ifs_p.press_height_full / 1000, color="grey", lw=0.5)
    ds_plot = dropsonde_ds[key]
    times = ["104205", "110137"] if key == "RF17" else ["110321", "110823"]
    for k in times:
        ds = ds_plot[k]
        ds = ds.where(~np.isnan(ds.rh), drop=True)
        ax.plot(ds.rh, ds.alt / 1000, label=f"Dropsonde {k[:2]}:{k[2:4]} UTC", lw=2)
    ax.set(xlim=(0, 100), ylim=(0, 12), xlabel="Relative humidity (%)", title=f"{key}")
    ax.plot([], color="grey", label="IFS profiles")
    ax.legend()
    ax.grid()

axs[0].set_ylabel("Altitude (km)")
axs[0].text(0.05, 0.95, "a)", transform=axs[0].transAxes, bbox=dict(boxstyle="Round", fc="white"))
axs[1].text(0.05, 0.95, "b)", transform=axs[1].transAxes, bbox=dict(boxstyle="Round", fc="white"))
axs[2].text(0.05, 0.95, "c)", transform=axs[2].transAxes, bbox=dict(boxstyle="Round", fc="white"))
axs[3].text(0.05, 0.95, "d)", transform=axs[3].transAxes, bbox=dict(boxstyle="Round", fc="white"))

plt.tight_layout()

figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_ifs_dropsonde_t_rh.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot BACARDI and libRadtran simulation
_, ax = plt.subplots(figsize=h.figsize_wide)
plot_ds = bacardi_ds["RF17"].sel(time=slices["RF17"]["case"])
plot_ds.F_down_solar_sim.plot(x="time")
plot_ds.F_down_solar_sim_si.plot(x="time")
plt.show()
plt.close()

