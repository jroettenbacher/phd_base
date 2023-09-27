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
import pylim.meteorological_formulas as met
from pylim import ecrad, reader
import ac3airborne
from ac3airborne.tools import flightphase
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import gridspec, patheffects
from matplotlib.patches import Patch, Polygon
from matplotlib.collections import LineCollection
import seaborn as sns
import cmasher as cm
import cartopy.crs as ccrs
from tqdm import tqdm
from metpy import constants as mc
from metpy.calc import relative_humidity_from_specific_humidity
from metpy.units import units as u

h.set_cb_friendly_colors()
cbc = h.get_cb_friendly_colors()

# %% set paths
campaign = "halo-ac3"
plot_path = "C:/Users/Johannes/Documents/Doktor/manuscripts/2023_arctic_cirrus/figures"
trajectory_path = f"{h.get_path('trajectories', campaign=campaign)}/selection_CC_and_altitude"
keys = ["RF17", "RF18"]
ecrad_versions = ["v15", "v16", "v17", "v18", "v19", "v20", "v21", "v28", "v29"]

# %% read in data
(
    bahamas_ds,
    bacardi_ds,
    bacardi_ds_res,
    ecrad_dicts,
    height_sels,
    varcloud_ds,
    above_clouds,
    below_clouds,
    slices,
    ecrad_orgs,
    ifs_ds,
    dropsonde_ds,
    albedo_dfs
) = (dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict())

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
    bacardi_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1_JR.nc"
    libradtran_bb_solar_si = f"HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_solar_si_{date}_{key}.nc"
    libradtran_bb_thermal_si = f"HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_thermal_si_{date}_{key}.nc"
    ifs_file = f"ifs_{date}_00_ml_O1280_processed.nc"
    varcloud_file = [f for f in os.listdir(varcloud_path) if f.endswith(".nc")][0]
    dropsonde_files = [f for f in os.listdir(dropsonde_path) if f.endswith("QC.nc")]

    # read in aircraft data
    bahamas_ds[key] = reader.read_bahamas(f"{bahamas_path}/{bahamas_file}")
    bacardi = xr.open_dataset(f"{bacardi_path}/{bacardi_file}")

    # read in results of albedo experiment
    albedo_dfs[key] = pd.read_csv(f"{plot_path}/{flight}_boxplot_data.csv")

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
    bacardi["F_up_terrestrial_sim_si"] = bb_sim_thermal_si_inp.eup
    bacardi["transmissivity_solar"] = bacardi["F_down_solar"] / bb_sim_solar_si_inp.fdw
    bacardi["transmissivity_terrestrial"] = bacardi["F_down_terrestrial"] / bb_sim_thermal_si_inp.edn
    # calculate reflectivity
    bacardi["reflectivity_solar"] = bacardi["F_up_solar"] / bacardi["F_down_solar"]
    bacardi["reflectivity_terrestrial"] = bacardi["F_up_terrestrial"] / bacardi["F_down_terrestrial"]

    # calculate radiative effect from BACARDI and libRadtran sea ice simulation
    bacardi["F_net_solar"] = bacardi["F_down_solar"] - bacardi["F_up_solar"]
    bacardi["F_net_terrestrial"] = bacardi["F_down_terrestrial"] - bacardi["F_up_terrestrial"]
    bacardi["F_net_solar_sim"] = bb_sim_solar_si_inp.fdw - bb_sim_solar_si_inp.eup
    bacardi["F_net_terrestrial_sim"] = bb_sim_thermal_si_inp.edn - bb_sim_thermal_si_inp.eup
    bacardi["CRE_solar"] = bacardi["F_net_solar"] - bacardi["F_net_solar_sim"]
    bacardi["CRE_terrestrial"] = bacardi["F_net_terrestrial"] - bacardi["F_net_terrestrial_sim"]
    bacardi["CRE_total"] = bacardi["CRE_solar"] + bacardi["CRE_terrestrial"]
    bacardi["F_down_solar_error"] = np.abs(bacardi["F_down_solar"] * 0.03)
    # normalize downward irradiance for cos SZA
    for var in ["F_down_solar", "F_down_solar_diff"]:
        bacardi[f"{var}_norm"] = bacardi[var] / np.cos(np.deg2rad(bacardi["sza"]))
    # filter data for motion flag
    bacardi_org = bacardi.copy()
    bacardi = bacardi.where(bacardi.motion_flag)
    # overwrite variables which do not need to be filtered with original values
    for var in ["alt", "lat", "lon", "sza", "saa", "attitude_flag", "segment_flag", "motion_flag"]:
        bacardi[var] = bacardi_org[var]
    bacardi_ds[key] = bacardi

    # read in resampled BACARDI data
    bacardi_res = xr.open_dataset(f"{bacardi_path}/{bacardi_file.replace('.nc', '_1Min.nc')}")
    # normalize downward irradiance for cos SZA
    for var in ["F_down_solar", "F_down_solar_diff"]:
        bacardi_res[f"{var}_norm"] = bacardi_res[var] / np.cos(np.deg2rad(bacardi_res["sza"]))
    bacardi_ds_res[key] = bacardi_res.copy()

    # read in ecrad data
    ecrad_dict, ecrad_org, height_sel = dict(), dict(), dict()

    for k in ecrad_versions:
        ds = xr.open_dataset(f"{ecrad_path}/ecrad_merged_inout_{date}_{k}.nc")

        if "column" in ds.dims:
            b_ds = bacardi_res.expand_dims(dict(column=np.arange(0, len(ds.column)))).copy()
            height_sel[k] = ecrad.get_model_level_of_altitude(b_ds.sel(column=0).alt, ds.sel(column=0), "half_level")
            ecrad_ds = ds.isel(half_level=height_sel[k], column=slice(0, 10))
            ds["flux_dn_sw_diff"] = ecrad_ds.flux_dn_sw - b_ds.F_down_solar
            ds["spread"] = xr.DataArray(
                np.array(
                    [
                        ds["flux_dn_sw_diff"].min(dim="column").to_numpy(),
                        ds["flux_dn_sw_diff"].max(dim="column").to_numpy(),
                    ]
                ),
                coords=dict(x=[0, 1], time=ecrad_ds.time),
            )
            ds["flux_dn_sw_std"] = ds["flux_dn_sw"].std(dim="column")

            ecrad_org[k] = ds.copy(deep=True)
            if k == "v1":
                ds = ds.sel(column=16,
                            drop=True)  # select center column which corresponds to grid cell closest to aircraft
            else:
                # other versions have their nearest points selected via
                # kdTree, thus the first column should be the closest
                ds = ds.sel(column=0, drop=True)
        else:
            height_sel[k] = ecrad.get_model_level_of_altitude(bacardi.sel(time=ds.time).alt, ds, "half_level")

        ds["tiwp"] = ds.iwp.where(ds.iwp != np.inf, np.nan).sum(dim="level")
        for var in ["flux_dn_sw", "flux_dn_direct_sw"]:
            ds[f"{var}_norm"] = ds[var] / ds["cos_solar_zenith_angle"]

        ecrad_dict[k] = ds.copy()

    ecrad_dicts[key] = ecrad_dict
    ecrad_orgs[key] = ecrad_org
    height_sels[key] = height_sel

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

# %% calculate statistics from BACARDI
bacardi_vars = ["reflectivity_solar", "F_up_solar", "F_down_solar", "F_down_solar_diff"]
bacardi_stats = list()
for key in keys:
    for var in bacardi_vars:
        for section in ["above", "below"]:
            time_sel = slices[key][section]
            ds = bacardi_ds[key][var].sel(time=time_sel)
            bacardi_min = np.min(ds).to_numpy()
            bacardi_max = np.max(ds).to_numpy()
            bacardi_spread = bacardi_max - bacardi_min
            bacardi_mean = ds.mean().to_numpy()
            bacardi_std = ds.std().to_numpy()
            bacardi_stats.append(
                ("BACARDI", key, var, section, bacardi_min, bacardi_max, bacardi_spread, bacardi_mean, bacardi_std))

# %% calculate statstics from ecRad
ecrad_stats = list()
ecrad_vars = ["reflectivity_sw", "flux_up_sw", "flux_dn_direct_sw", "flux_dn_sw"]
for version in ["v15", "v17", "v18", "v19", "v21", "v29"]:
    v_name = ecrad.version_names[version]
    for key in keys:
        bds = bacardi_ds[key]
        height_sel = height_sels[key][version]
        ecrad_ds = ecrad_dicts[key][version].isel(half_level=height_sel)
        for evar in ecrad_vars:
            for section in ["above", "below"]:
                time_sel = slices[key][section]
                eds = ecrad_ds[evar].sel(time=time_sel)
                ecrad_min = np.min(eds).to_numpy()
                ecrad_max = np.max(eds).to_numpy()
                ecrad_spread = ecrad_max - ecrad_min
                ecrad_mean = eds.mean().to_numpy()
                ecrad_std = eds.std().to_numpy()
                ecrad_stats.append(
                    (v_name, key, evar, section, ecrad_min, ecrad_max, ecrad_spread, ecrad_mean, ecrad_std))

# %% convert statistics to dataframe
columns = ["source", "key", "variable", "section", "min", "max", "spread", "mean", "std"]
ecrad_df = pd.DataFrame(ecrad_stats, columns=columns)
bacardi_df = pd.DataFrame(bacardi_stats, columns=columns)
df = pd.concat([ecrad_df, bacardi_df]).reset_index(drop=True)
df.to_csv(f"{plot_path}/statistics.csv", index=False)

# %% get maximum spread between ecRad versions (ice optic parameterizations) and for BACARDI
for key in keys:
    for var in ["reflectivity_sw", "reflectivity_solar"]:
        selection = (df["variable"] == var) & (df["key"] == key) & (df["section"] == "above")
        versions_min = df["min"][selection].min()
        versions_max = df["max"][selection].max()
        print(f"{key}: Maximum spread in {var} above cloud: {versions_min:.3f} - {versions_max:.3f}")

# %% print mean reflectivity for above cloud section
for key in keys:
    for var in ["reflectivity_sw", "reflectivity_solar"]:
        selection = (df["variable"] == var) & (df["key"] == key) & (df["section"] == "above")
        v_mean = df[["source", "mean"]][selection]
        print(f"{key}: Mean {var} above cloud:\n{v_mean}")

# %% print mean flux_dn_sw for below cloud section
for key in keys:
    for var in ["flux_dn_sw", "F_down_solar_diff"]:
        selection = (df["variable"] == var) & (df["key"] == key) & (df["section"] == "below")
        v_mean = df[["source", "mean"]][selection]
        print(f"{key}: Mean {var} below cloud:\n{v_mean}")

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
ds = ecrad_orgs["RF17"]["v15"].sel(column=0)
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

# %% plot BACARDI four panel view with above and below cloud measurements and transmissivity - solar
plt.rc("font", size=6.5)
ylim_transmissivity = (0.55, 1.1)
_, axs = plt.subplots(2, 2, figsize=(17 * h.cm, 10 * h.cm))

# upper left panel
ax = axs[0, 0]
plot_ds = bacardi_ds["RF17"].sel(time=slices["RF17"]["case"])
time_extend = pd.to_timedelta((plot_ds.time[-1] - plot_ds.time[0]).to_numpy())
# bacardi measurements
for var in ["F_down_solar", "F_up_solar"]:
    ax.plot(plot_ds.time, plot_ds[var], label=f"{h.bacardi_labels[var]}")
ax.axvline(above_clouds["RF17"]["end"], color="grey")
ax.axvline(below_clouds["RF17"]["start"], color="grey")
# libradtran clearsky simulation
# ax.plot(plot_ds.time, plot_ds.F_down_solar_sim_si, label=f"simulated {h.bacardi_labels['F_down_solar']}")
ax.legend()
h.set_xticks_and_xlabels(ax, time_extend)
ax.grid()
ax.text(0.03, 0.88, "a)", transform=ax.transAxes, bbox=dict(boxstyle="Round", fc="white"))
ax.set(title="RF 17 - 11 April 2022", ylabel=f"{h.cbarlabels['flux_dn_sw']} ({h.plot_units['flux_dn_sw']})")

# lower left panel
ax = axs[1, 0]
ax.axhline(y=1, color="k")
ax.plot(plot_ds.time, plot_ds["transmissivity_solar"], label="Solar transmissivity", color=cbc[4])
ax.axvline(above_clouds["RF17"]["end"], color="grey")
ax.axvline(below_clouds["RF17"]["start"], color="grey")
h.set_xticks_and_xlabels(ax, time_extend)
ax.grid()
ax.text(0.03, 0.88, "c)", transform=ax.transAxes, bbox=dict(boxstyle="Round", fc="white"))
ax.set(ylabel="Solar Transmissivity", xlabel="Time (UTC)", ylim=ylim_transmissivity)

# upper right panel
ax = axs[0, 1]
plot_ds = bacardi_ds["RF18"].sel(time=slices["RF18"]["case"])
time_extend = pd.to_timedelta((plot_ds.time[-1] - plot_ds.time[0]).to_numpy())
# bacardi measurements
for var in ["F_down_solar", "F_up_solar"]:
    ax.plot(plot_ds.time, plot_ds[var], label=f"{h.bacardi_labels[var]}")
ax.axvline(above_clouds["RF18"]["end"], color="grey")
ax.axvline(below_clouds["RF18"]["start"], color="grey")
# libradtran clearsky simulation
# ax.plot(plot_ds.time, plot_ds.F_down_solar_sim_si, label=f"simulated {h.bacardi_labels['F_down_solar']}")
ax.legend()
h.set_xticks_and_xlabels(ax, time_extend)
ax.grid()
ax.text(0.03, 0.88, "b)", transform=ax.transAxes, bbox=dict(boxstyle="Round", fc="white"))
ax.set(title="RF 18 - 12 April 2022")

# lower right panel
ax = axs[1, 1]
ax.axhline(y=1, color="k")
ax.plot(plot_ds.time, plot_ds["transmissivity_solar"], label="Solar transmissivity", color=cbc[4])
ax.axvline(above_clouds["RF18"]["end"], color="grey")
ax.axvline(below_clouds["RF18"]["start"], color="grey")
h.set_xticks_and_xlabels(ax, time_extend)
ax.grid()
ax.text(0.03, 0.88, "d)", transform=ax.transAxes, bbox=dict(boxstyle="Round", fc="white"))
ax.set(xlabel="Time (UTC)", ylim=ylim_transmissivity)

plt.tight_layout()
figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_BACARDI_case_studies.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot BACARDI four panel view with above and below cloud measurements and transmissivity - terrestrial
plt.rc("font", size=6.5)
_, axs = plt.subplots(2, 2, figsize=(17 * h.cm, 10 * h.cm))

# upper left panel
ax = axs[0, 0]
plot_ds = bacardi_ds["RF17"].sel(time=slices["RF17"]["case"])
time_extend = pd.to_timedelta((plot_ds.time[-1] - plot_ds.time[0]).to_numpy())
# bacardi measurements
for var in ["F_down_terrestrial", "F_up_terrestrial"]:
    ax.plot(plot_ds.time, plot_ds[var], label=f"{h.bacardi_labels[var]}")
# libradtran clearsky simulation
ax.plot(plot_ds.time, plot_ds.F_down_terrestrial_sim_si, label=f"simulated {h.bacardi_labels['F_down_terrestrial']}")
ax.plot(plot_ds.time, plot_ds.F_up_terrestrial_sim_si, label=f"simulated {h.bacardi_labels['F_up_terrestrial']}")
ax.legend()
h.set_xticks_and_xlabels(ax, time_extend)
ax.grid()
ax.text(0.03, 0.88, "a)", transform=ax.transAxes, bbox=dict(boxstyle="Round", fc="white"))
ax.set(title="RF 17 - 11 April 2022", ylabel=f"{h.cbarlabels['flux_dn_lw']} ({h.plot_units['flux_dn_lw']})")

# lower left panel
ax = axs[1, 0]
ax.axhline(y=1, color="k")
ax.plot(plot_ds.time, plot_ds["transmissivity_terrestrial"], label="Terrestrial transmissivity", color=cbc[4])
h.set_xticks_and_xlabels(ax, time_extend)
ax.grid()
ax.text(0.03, 0.88, "c)", transform=ax.transAxes, bbox=dict(boxstyle="Round", fc="white"))
ax.set(ylabel="Terrestrial Transmissivity", xlabel="Time (UTC)")

# upper right panel
ax = axs[0, 1]
plot_ds = bacardi_ds["RF18"].sel(time=slices["RF18"]["case"])
time_extend = pd.to_timedelta((plot_ds.time[-1] - plot_ds.time[0]).to_numpy())
# bacardi measurements
for var in ["F_down_terrestrial", "F_up_terrestrial"]:
    ax.plot(plot_ds.time, plot_ds[var], label=f"{h.bacardi_labels[var]}")
# libradtran clearsky simulation
ax.plot(plot_ds.time, plot_ds.F_down_terrestrial_sim_si, label=f"simulated {h.bacardi_labels['F_down_terrestrial']}")
ax.plot(plot_ds.time, plot_ds.F_up_terrestrial_sim_si, label=f"simulated {h.bacardi_labels['F_up_terrestrial']}")
ax.legend()
h.set_xticks_and_xlabels(ax, time_extend)
ax.grid()
ax.text(0.03, 0.88, "b)", transform=ax.transAxes, bbox=dict(boxstyle="Round", fc="white"))
ax.set(title="RF 18 - 12 April 2022")

# lower right panel
ax = axs[1, 1]
ax.axhline(y=1, color="k")
ax.plot(plot_ds.time, plot_ds["transmissivity_terrestrial"], label="Terrestrial transmissivity", color=cbc[4])
h.set_xticks_and_xlabels(ax, time_extend)
ax.grid()
ax.text(0.03, 0.88, "d)", transform=ax.transAxes, bbox=dict(boxstyle="Round", fc="white"))
ax.set(xlabel="Time (UTC)")

plt.tight_layout()
figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_BACARDI_terrestrial_case_studies.png"
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

# %% plot IFS cloud fraction lidar/mask comparison
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

# %% plot flight track together with trajectories
cmap = mpl.colormaps["tab20b_r"]([20, 20, 0, 3, 4, 7, 8, 11, 12, 15, 16, 19])
cmap[:2] = mpl.colormaps["tab20c"]([7, 4])
cmap = mpl.colors.ListedColormap(cmap)
plt_sett = {
        'label': 'Time relative to release (h)',
        'norm': plt.Normalize(-72, 0),
        'ylim': [-72, 0],
        'cmap_sel': cmap,
        'cmap_ticks': np.arange(-72, 0.1, 12),
        'shrink': 0.74
}
data_crs = ccrs.PlateCarree()

plt.rc("font", size=6)
fig = plt.figure(figsize=(18 * h.cm, 8 * h.cm))
gs = gridspec.GridSpec(1, 2)

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

# Plot the surface pressure - 11 April
ifs = ifs_ds["RF17"].sel(time="2022-04-11 12:00")
pressure_levels = np.arange(900, 1125, 5)
press = ifs.mean_sea_level_pressure / 100  # conversion to hPa
cp = ax.tricontour(ifs.lon, ifs.lat, press, levels=pressure_levels, colors='k', linewidths=0.5,
                   linestyles='solid', alpha=1, transform=data_crs)
# cp.clabel(fontsize=2, inline=1, inline_spacing=1, fmt='%i hPa', rightside_up=True, use_clabeltext=True)
cp.clabel(fontsize=4, inline=1, inline_spacing=4, fmt='%i', rightside_up=True, use_clabeltext=True)

# add seaice edge
ci_levels = [0.8]
cci = ax.tricontour(ifs.lon, ifs.lat, ifs.CI, ci_levels, transform=data_crs, linestyles="--", colors="#332288",
                    linewidths=1)

# add high cloud cover
ifs_cc = ifs.cloud_fraction.where(ifs.pressure_full < 60000, drop=True).sum(dim="level")
ax.tricontourf(ifs.lon, ifs.lat, ifs_cc, levels=24, transform=data_crs, cmap="Blues", alpha=1)

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

# get the time step of the trajectories # here: manually set
dt = 0.01
traj_single_len = 4320  # int(tmax/dt)
traj_overall_len = int(len(times))
traj_num = int(traj_overall_len / (traj_single_len + 1))  # +1 for the empty line after
# each traj
var_index = header.index("TIME")

for k in range(traj_single_len + 1):
    # reduce to hourly? --> [::60]
    lon = trajs[k * (traj_single_len + 1):(k + 1) * (traj_single_len + 1), 1][::60]
    lat = trajs[k * (traj_single_len + 1):(k + 1) * (traj_single_len + 1), 2][::60]
    var = trajs[k * (traj_single_len + 1):(k + 1) * (traj_single_len + 1), var_index][::60]
    x, y = lon, lat
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=plt_sett['cmap_sel'], norm=plt_sett['norm'],
                        alpha=1, transform=data_crs)
    # Set the values used for colormapping
    lc.set_array(var)
    if int(traj_num) == 1:
        lc.set_linewidth(5)
    elif int(traj_num) >= 2:
        lc.set_linewidth(1)
    line = ax.add_collection(lc)

plt.colorbar(line, ax=ax, pad=0.01, shrink=plt_sett["shrink"],
             ticks=plt_sett['cmap_ticks']).set_label(label=plt_sett['label'], size=6)

# plot flight track - 11 April
ins = bahamas_ds["RF17"]
track_lons, track_lats = ins["IRS_LON"], ins["IRS_LAT"]
ax.scatter(track_lons[::10], track_lats[::10], c="k", alpha=1, marker=".", s=1, zorder=400,
           label='HALO flight track', transform=data_crs, linestyle="solid")

# highlight case study region
ins_hl = ins.sel(time=slices["RF17"]["above"])
ax.scatter(ins_hl.IRS_LON[::10], ins_hl.IRS_LAT[::10], c=cbc[1], alpha=1, marker=".", s=1, zorder=400,
           transform=ccrs.PlateCarree(), linestyle="solid")


# plot dropsonde locations - 11 April
ds_dict = dropsonde_ds["RF17"]
for i, ds in enumerate(ds_dict.values()):
    ds["alt"] = ds.alt / 1000  # convert altitude to km
    launch_time = pd.to_datetime(ds.launch_time.to_numpy())
    x, y = ds.lon.mean().to_numpy(), ds.lat.mean().to_numpy()
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
ax.legend(handles=handles, labels=labels, framealpha=1, loc=2, title="a) RF 17", alignment="left")

# plot trajectories 12 April in second row first column
ax = fig.add_subplot(gs[0, 1], projection=ccrs.NorthPolarStereo())
ax.coastlines(alpha=0.5)
ax.set_xlim((-2000000, 2000000))
ax.set_ylim((-3000000, 500000))
gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=1, color='gray', alpha=0.5,
                  linestyle=':', x_inline=False, y_inline=False, rotate_labels=False)
gl.top_labels = False
gl.right_labels = False
gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 20))
gl.ylocator = mticker.FixedLocator(np.arange(60, 90, 5))

# Plot the surface pressure - 12 April
ifs = ifs_ds["RF18"].sel(time="2022-04-12 12:00")
pressure_levels = np.arange(900, 1125, 5)
press = ifs.mean_sea_level_pressure / 100  # conversion to hPa
cp = ax.tricontour(ifs.lon, ifs.lat, press, levels=pressure_levels, colors='k', linewidths=0.5,
                   linestyles='solid', alpha=1, transform=data_crs)
# cp.clabel(fontsize=2, inline=1, inline_spacing=1, fmt='%i hPa', rightside_up=True, use_clabeltext=True)
cp.clabel(fontsize=4, inline=1, inline_spacing=4, fmt='%i', rightside_up=True, use_clabeltext=True)

# add seaice edge
ci_levels = [0.8]
cci = ax.tricontour(ifs.lon, ifs.lat, ifs.CI, ci_levels, transform=data_crs, linestyles="--", colors="#332288",
                    linewidths=1)

# add high cloud cover
ifs_cc = ifs.cloud_fraction.where(ifs.pressure_full < 60000, drop=True).sum(dim="level")
ax.tricontourf(ifs.lon, ifs.lat, ifs_cc, levels=24, transform=data_crs, cmap="Blues", alpha=1)

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
    lc = LineCollection(segments, cmap=plt_sett['cmap_sel'], norm=plt_sett['norm'],
                        alpha=1, transform=ccrs.PlateCarree())
    # Set the values used for colormapping
    lc.set_array(var)
    if int(traj_num) == 1:
        lc.set_linewidth(5)
    elif int(traj_num) >= 2:
        lc.set_linewidth(1)
    line = ax.add_collection(lc)

plt.colorbar(line, ax=ax, pad=0.01, shrink=plt_sett["shrink"],
             ticks=plt_sett["cmap_ticks"]).set_label(label=plt_sett['label'], size=6)

# plot flight track - 12 April
ins = bahamas_ds["RF18"]
track_lons, track_lats = ins["IRS_LON"], ins["IRS_LAT"]
ax.scatter(track_lons[::10], track_lats[::10], c="k", alpha=1, marker=".", s=1, zorder=400,
           label='HALO flight track', transform=ccrs.PlateCarree(), linestyle="solid")

# highlight case study region
ins_hl = ins.sel(time=slices["RF18"]["above"])
ax.scatter(ins_hl.IRS_LON[::10], ins_hl.IRS_LAT[::10], c=cbc[1], alpha=1, marker=".", s=1, zorder=400,
           transform=ccrs.PlateCarree(), linestyle="solid")

# plot dropsonde locations - 12 April
ds_dict = dropsonde_ds["RF18"]
for ds in ds_dict.values():
    x, y = ds.lon.mean().to_numpy(), ds.lat.mean().to_numpy()
    cross = ax.plot(x, y, "x", color="orangered", markersize=3, label="Dropsonde", transform=data_crs,
                    zorder=450)
# add time to only a selected range of dropsondes
for i in [1, -5, -4, -2, -1]:
    ds = list(ds_dict.values())[i]
    launch_time = pd.to_datetime(ds.launch_time.to_numpy())
    x, y = ds.lon.mean().to_numpy(), ds.lat.mean().to_numpy()
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
ax.legend(handles=handles, labels=labels, framealpha=1, loc=2, title="b) RF 18", alignment="left")

plt.tight_layout()
figname = f"{plot_path}/HALO-AC3_RF17_RF18_fligh_track_trajectories_plot_overview.png"
plt.savefig(figname, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# %% plot temperature and humidity profiles from IFS and from dropsonde
plt.rc("font", size=7)
_, axs = plt.subplots(1, 4, figsize=(18 * h.cm, 10 * h.cm))
for i, key in enumerate(keys):
    ax = axs[i * 2]
    ifs_plot = ecrad_dicts[key]["v15"].sel(time=slices[key]["case"])

    # Air temperature
    for t in ifs_plot.time:
        ifs_p = ifs_plot.sel(time=t)
        ax.plot(ifs_p.temperature_hl - 273.15, ifs_p.press_height_hl / 1000, color="grey", lw=0.5)
    ds_plot = dropsonde_ds[key]
    times = ["104205", "110137"] if key == "RF17" else ["110321", "110823"]
    for k in times:
        ds = ds_plot[k]
        ds = ds.where(~np.isnan(ds.tdry), drop=True)
        ax.plot(ds.tdry, ds.alt / 1000, label=f"DS {k[:2]}:{k[2:4]} UTC", lw=2)
    ax.set(xlim=(-60, -10), ylim=(0, 12), xlabel="Air temperature (°C)", title=f"{key}")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(base=10))
    ax.plot([], color="grey", label="IFS profiles")
    ax.grid()

    # RH
    ax = axs[i * 2 + 1]
    ifs_plot = ecrad_dicts[key]["v15"].sel(time=slices[key]["case"])
    for t in ifs_plot.time:
        ifs_p = ifs_plot.sel(time=t)
        rh = relative_humidity_from_specific_humidity(ifs_p.pressure_full * u.Pa, ifs_p.t * u.K, ifs_p.q * u("kg/kg"))
        rh_ice = met.relative_humidity_water_to_relative_humidity_ice(rh * 100, ifs_p.t - 273.15)
        ax.plot(rh_ice, ifs_p.press_height_full / 1000, color="grey", lw=0.5)
    ds_plot = dropsonde_ds[key]
    times = ["104205", "110137"] if key == "RF17" else ["110321", "110823"]
    for k in times:
        ds = ds_plot[k]
        ds = ds.where(~np.isnan(ds.rh), drop=True)
        ax.plot(met.relative_humidity_water_to_relative_humidity_ice(ds.rh, ds.tdry),
                ds.alt / 1000, label=f"DS {k[:2]}:{k[2:4]} UTC", lw=2)
    ax.set(xlim=(0, 130), ylim=(0, 12), xlabel="Relative humidity over ice (%)", title=f"{key}")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(base=25))
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

# %% plot scatter plot of above cloud measurements and simulations
plt.rc("font", size=7)
label = ["a)", "b)"]
for v in ["v15", "v18", "v19"]:
    _, axs = plt.subplots(1, 2, figsize=(16 * h.cm, 9 * h.cm))
    for i, key in enumerate(keys):
        ax = axs[i]
        above_sel = (bahamas_ds[key].IRS_ALT > 11000).resample(time="1Min").first()
        bacardi_res = bacardi_ds_res[key]
        bacardi_plot = bacardi_res.where(bacardi_res.alt > 11000)
        ecrad_ds = ecrad_dicts[key][v]
        height_sel = height_sels[key][v]
        ecrad_plot = ecrad_ds.flux_dn_sw.isel(half_level=height_sel).where(above_sel)

        # actual plotting
        rmse = np.sqrt(np.mean((ecrad_plot - bacardi_plot["F_down_solar_diff"]) ** 2)).to_numpy()
        bias = np.nanmean((ecrad_plot - bacardi_plot["F_down_solar_diff"]).to_numpy())
        ax.scatter(bacardi_plot.F_down_solar_diff, ecrad_plot, color=cbc[3])
        ax.axline((0, 0), slope=1, color="k", lw=2, transform=ax.transAxes)
        ax.set(
            aspect="equal",
            xlabel="BACARDI irradiance (W$\,$m$^{-2}$)",
            ylabel="ecRad irradiance (W$\,$m$^{-2}$)",
            xlim=(200, 525),
            ylim=(200, 525),
        )
        ax.grid()
        ax.text(
            0.025,
            0.95,
            f"{label[i]} {key}\n"
            f"n= {sum(~np.isnan(bacardi_plot['F_down_solar'])):.0f}\n"
            f"RMSE: {rmse:.0f} {h.plot_units['flux_dn_sw']}\n"
            f"Bias: {bias:.0f} {h.plot_units['flux_dn_sw']}",
            ha="left",
            va="top",
            transform=ax.transAxes,
            bbox=dict(fc="white", ec="black", alpha=0.8, boxstyle="round"),
        )

    plt.tight_layout()

    figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_bacardi_ecrad_f_down_solar_above_cloud_all_{v}.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot scatter plot of above cloud measurements and simulations - clear sky
plt.rc("font", size=7)
label = ["a)", "b)"]
_, axs = plt.subplots(1, 2, figsize=(16 * h.cm, 9 * h.cm))

for i, key in enumerate(keys):
    ax = axs[i]
    above_sel = (bahamas_ds[key].IRS_ALT > 11000).resample(time="1Min").first()
    bacardi_res = bacardi_ds_res[key]
    bacardi_plot = bacardi_res.where(bacardi_res.alt > 11000)
    ecrad_ds = ecrad_dicts[key]["v15"]
    height_sel = height_sels[key]["v15"]
    ecrad_plot = ecrad_ds.flux_dn_sw_clear.isel(half_level=height_sel).where(above_sel)

    # actual plotting
    rmse = np.sqrt(np.mean((ecrad_plot - bacardi_plot["F_down_solar"]) ** 2)).to_numpy()
    bias = np.nanmean((ecrad_plot - bacardi_plot["F_down_solar"]).to_numpy())
    ax.scatter(bacardi_plot.F_down_solar_sim, ecrad_plot, color=cbc[3])
    ax.axline((0, 0), slope=1, color="k", lw=2, transform=ax.transAxes)
    ax.set(
        aspect="equal",
        xlabel="BACARDI irradiance (W$\,$m$^{-2}$)",
        ylabel="ecRad irradiance (W$\,$m$^{-2}$)",
        xlim=(200, 525),
        ylim=(200, 525),
    )
    ax.grid()
    ax.text(
        0.025,
        0.95,
        f"{label[i]} {key}\n# points: {sum(~np.isnan(bacardi_plot['F_down_solar'])):.0f}\n"
        f"RMSE: {rmse:.0f} {h.plot_units['flux_dn_sw']}\n"
        f"Bias: {bias:.0f} {h.plot_units['flux_dn_sw']}",
        ha="left",
        va="top",
        transform=ax.transAxes,
        bbox=dict(fc="white", ec="black", alpha=0.8, boxstyle="round"),
    )

plt.tight_layout()

figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_bacardi_ecrad_f_down_solar_clear_sky_above_cloud_all.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot scatter plot of below cloud measurements and simulations
plt.rc("font", size=7)
label = ["a)", "b)"]
lims = [(110, 260), (100, 160)]
bacardi_var = "F_down_solar_diff"
for v in ["v15", "v18", "v19"]:
    _, axs = plt.subplots(1, 2, figsize=(16 * h.cm, 9 * h.cm))

    for i, key in enumerate(keys):
        ax = axs[i]
        bacardi_res = bacardi_ds[key].resample(time="1Min")
        bacardi_mean = bacardi_res.mean().sel(time=slices[key]["below"])
        bacardi_std = bacardi_res.std().sel(time=slices[key]["below"])
        bacardi_plot = bacardi_mean
        ecrad_ds = ecrad_dicts[key][v]
        height_sel = ecrad.get_model_level_of_altitude(bacardi_res.mean().alt, ecrad_ds, "half_level")
        ecrad_plot = ecrad_ds["flux_dn_sw"].isel(half_level=height_sel).sel(time=slices[key]["below"])
        ecrad_std = ecrad_ds["flux_dn_sw_std"].isel(half_level=height_sel).sel(time=slices[key]["below"])
        # ecrad_mean = ecrad_orgs[key]["v15"].flux_dn_sw.isel(half_level=height_sel).sel(column=slice(0, 10)).mean(dim="column")
        # ecrad_sum = ecrad_ds.spectral_flux_dn_sw.isel(half_level=height_sel, band_sw=slice(1, 12)).sum(dim="band_sw")

        # actual plotting
        rmse = np.sqrt(np.mean((ecrad_plot - bacardi_plot[bacardi_var]) ** 2)).to_numpy()
        bias = np.nanmean((ecrad_plot - bacardi_plot[bacardi_var]).to_numpy())
        ax.errorbar(bacardi_plot[bacardi_var], ecrad_plot,
                    xerr=bacardi_std[bacardi_var], yerr=ecrad_std,
                    color=cbc[3], fmt="o", capsize=4)
        ax.axline((0, 0), slope=1, color="k", lw=2, transform=ax.transAxes)
        # ticks = ax.get_yticks() if i == 0 else ax.get_xticks()
        # ax.set_yticks(ticks)
        # ax.set_xticks(ticks)
        ax.set(
            aspect="equal",
            xlabel="BACARDI irradiance (W$\,$m$^{-2}$)",
            ylabel="ecRad irradiance (W$\,$m$^{-2}$)",
            xlim=lims[i],
            ylim=lims[i],
        )
        ax.grid()
        ax.text(
            0.025,
            0.95,
            f"{label[i]} {key} {ecrad.version_names[v]}\n# points: {sum(~np.isnan(bacardi_plot['F_down_solar'])):.0f}\n"
            f"RMSE: {rmse:.0f} {h.plot_units['flux_dn_sw']}\n"
            f"Bias: {bias:.0f} {h.plot_units['flux_dn_sw']}",
            ha="left",
            va="top",
            transform=ax.transAxes,
            bbox=dict(fc="white", ec="black", alpha=0.8, boxstyle="round"),
        )

    plt.tight_layout()

    figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_bacardi_ecrad_f_down_solar_below_cloud_{v}.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot scatter plot of below cloud measurements and simulations varcloud
plt.rc("font", size=7)
label = ["a)", "b)"]
lims = [(140, 260), (80, 190)]
for v in ["v16", "v20"]:
    _, axs = plt.subplots(1, 2, figsize=(16 * h.cm, 9 * h.cm))

    for i, key in enumerate(keys):
        ax = axs[i]
        ecrad_ds = ecrad_dicts[key][v]
        bacardi_res = bacardi_ds[key].reindex(time=ecrad_ds.time, method="ffill")
        bacardi_plot = bacardi_res  # .sel(time=slices[key]["below"])
        # yerr = np.abs(ecrad_ds["error"].sel(time=slices[key]["below"]).to_numpy())
        height_sel = ecrad.get_model_level_of_altitude(bacardi_res.alt, ecrad_ds, "half_level")
        ecrad_plot = ecrad_ds.flux_dn_sw.isel(half_level=height_sel)
        # ecrad_mean = ecrad_orgs[key]["v15"].flux_dn_sw.isel(half_level=height_sel).sel(column=slice(0, 10)).mean(dim="column")
        # ecrad_sum = ecrad_ds.spectral_flux_dn_sw.isel(half_level=height_sel, band_sw=slice(1, 12)).sum(dim="band_sw")

        # actual plotting
        rmse = np.sqrt(np.mean((ecrad_plot - bacardi_plot["F_down_solar"]) ** 2)).to_numpy()
        bias = np.nanmean((ecrad_plot - bacardi_plot["F_down_solar"]).to_numpy())
        ax.errorbar(bacardi_plot.F_down_solar, ecrad_plot, xerr=bacardi_plot["error"], color=cbc[3], fmt=".", capsize=4)
        ax.axline((0, 0), slope=1, color="k", lw=2, transform=ax.transAxes)
        # ticks = ax.get_yticks() if i == 0 else ax.get_xticks()
        # ax.set_yticks(ticks)
        # ax.set_xticks(ticks)
        ax.set(
            aspect="equal",
            xlabel="BACARDI irradiance (W$\,$m$^{-2}$)",
            ylabel="ecRad irradiance (W$\,$m$^{-2}$)",
            xlim=lims[i],
            ylim=lims[i],
        )
        ax.grid()
        ax.text(
            0.025,
            0.95,
            f"{label[i]} {key} {ecrad.version_names[v]}\n"
            f"# points: {sum(~np.isnan(bacardi_plot['F_down_solar'])):.0f}\n"
            f"RMSE: {rmse:.0f} {h.plot_units['flux_dn_sw']}\n"
            f"Bias: {bias:.0f} {h.plot_units['flux_dn_sw']}",
            ha="left",
            va="top",
            transform=ax.transAxes,
            bbox=dict(fc="white", ec="black", alpha=0.8, boxstyle="round"),
        )

    plt.tight_layout()

    figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_bacardi_ecrad_f_down_solar_below_cloud_{v}.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot PDF of IWC and re_ice
plt.rc("font", size=7)
legend_labels = ["VarCloud", "IFS"]
_, axs = plt.subplots(2, 2, figsize=(17 * h.cm, 10 * h.cm))

# upper left panel - RF17 IWC
ax = axs[0, 0]
plot_ds = ecrad_dicts["RF17"]
sel_time = slice(pd.to_datetime("2022-04-11 10:49"), pd.to_datetime("2022-04-11 11:04"))
binsize = 0.25
bins = np.arange(-0.25, 5.1, binsize)
for i, v in enumerate(["v16", "v15"]):
    if v == "v16":
        pds = plot_ds[v].iwc
    else:
        iwc, cc = plot_ds[v].iwc.sel(time=sel_time), plot_ds[v].cloud_fraction.sel(time=sel_time)
        pds = iwc.where(cc == 0, iwc / cc)

    pds = pds.to_numpy().flatten() * 1e6
    pds = pds[~np.isnan(pds)]
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i],
        color=cbc[i * 2 + 1],
        histtype="step",
        density=True,
        lw=2,
    )
ax.legend()
ax.grid()
ax.text(0.03, 0.88, "a)", transform=ax.transAxes, bbox=dict(boxstyle="Round", fc="white"))
ax.set(title=f"RF 17 - 11 April 2022 {sel_time.start:%H:%M} - {sel_time.stop:%H:%M} UTC",
       ylabel=f"Probability density function",
       xlabel=f"Ice water content ({h.plot_units['iwc']})")

# lower left panel - RF17 re_ice
ax = axs[1, 0]
binsize = 4
bins = np.arange(0, 100, binsize)
for i, v in enumerate(["v16", "v15"]):
    if v == "v16":
        pds = plot_ds[v].re_ice.to_numpy().flatten() * 1e6
    else:
        pds = plot_ds[v].re_ice.sel(time=sel_time).to_numpy().flatten() * 1e6

    pds = pds[~np.isnan(pds)]
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i],
        color=cbc[i * 2 + 1],
        histtype="step",
        density=True,
        lw=2,
    )
ax.legend()
ax.grid()
ax.text(0.03, 0.88, "c)", transform=ax.transAxes, bbox=dict(boxstyle="Round", fc="white"))
ax.set(ylabel="Probability density function",
       xlabel=f"Ice effective radius ({h.plot_units['re_ice']})")

# upper right panel - RF18 IWC
ax = axs[0, 1]
plot_ds = ecrad_dicts["RF18"]
sel_time = slice(pd.to_datetime("2022-04-12 11:04"), pd.to_datetime("2022-04-12 11:24"))
binsize = 0.25
bins = np.arange(-0.25, 5.1, binsize)
for i, v in enumerate(["v16", "v15"]):
    if v == "v16":
        pds = plot_ds[v].iwc
    else:
        iwc, cc = plot_ds[v].iwc.sel(time=sel_time), plot_ds[v].cloud_fraction.sel(time=sel_time)
        pds = iwc.where(cc == 0, iwc / cc)

    pds = pds.to_numpy().flatten() * 1e6
    pds = pds[~np.isnan(pds)]
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i],
        color=cbc[i * 2 + 1],
        histtype="step",
        density=True,
        lw=2,
    )
ax.legend()
ax.grid()
ax.text(0.03, 0.88, "b)", transform=ax.transAxes, bbox=dict(boxstyle="Round", fc="white"))
ax.set(title=f"RF 18 - 12 April 2022 {sel_time.start:%H:%M} - {sel_time.stop:%H:%M} UTC",
       ylabel=f"",
       xlabel=f"Ice water content ({h.plot_units['iwc']})")

# lower right panel - RF18 re_ice
ax = axs[1, 1]
binsize = 4
bins = np.arange(0, 100, binsize)
for i, v in enumerate(["v16", "v15"]):
    if v == "v16":
        pds = plot_ds[v].re_ice.to_numpy().flatten() * 1e6
    else:
        pds = plot_ds[v].re_ice.sel(time=sel_time).to_numpy().flatten() * 1e6
    pds = pds[~np.isnan(pds)]
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i],
        color=cbc[i * 2 + 1],
        histtype="step",
        density=True,
        lw=2,
    )
ax.legend()
ax.grid()
ax.text(0.03, 0.88, "d)", transform=ax.transAxes, bbox=dict(boxstyle="Round", fc="white"))
ax.set(ylabel="",
       xlabel=f"Ice effective radius ({h.plot_units['re_ice']})")

plt.tight_layout()
figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_IFS_iwc_re_ice_pdf_case_studies.png"
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

# %% plot time series of ecRad - BACARDI difference below cloud
plt.rc("font", size=7)
label = ["a)", "b)"]
bacardi_var = "F_down_solar_diff"
for v in ["v15", "v18", "v19"]:
    _, axs = plt.subplots(1, 2, figsize=(16 * h.cm, 9 * h.cm))

    for i, key in enumerate(keys):
        ax = axs[i]
        bacardi_res = bacardi_ds[key].resample(time="1Min")
        bacardi_mean = bacardi_res.mean().sel(time=slices[key]["below"])
        bacardi_std = bacardi_res.std().sel(time=slices[key]["below"])
        ecrad_ds = ecrad_dicts[key][v]
        height_sel = ecrad.get_model_level_of_altitude(bacardi_res.mean().alt, ecrad_ds, "half_level")
        ecrad_plot = ecrad_ds["flux_dn_sw"].isel(half_level=height_sel).sel(time=slices[key]["below"])
        ecrad_std = ecrad_ds["flux_dn_sw_std"].isel(half_level=height_sel).sel(time=slices[key]["below"])
        plot_ds = ecrad_plot - bacardi_mean[bacardi_var]

        # actual plotting
        ax.plot(plot_ds.time, plot_ds, marker="o")
        h.set_xticks_and_xlabels(ax, pd.Timedelta(30, "Min"))
        ax.set(
            xlabel="Time (UTC)",
            ylabel="ecRad - BACARDI solar downward irradiance (W$\,$m$^{-2}$)",
        )
        ax.grid()

    plt.tight_layout()

    figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_ecrad-bacardi_f_down_solar_below_cloud_{v}.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot PDF of normalized solar downward irradiance below cloud
plt.rc("font", size=7)
label = ["a)", "b)"]

for v in ["v15", "v18", "v19"]:
    _, axs = plt.subplots(1, 2, figsize=(16 * h.cm, 9 * h.cm))
    for i, key in enumerate(keys):
        ax = axs[i]
        bacardi_sel = bacardi_ds[key].resample(time="1Min").mean().sel(time=slices[key]["below"])
        ecrad_ds = ecrad_dicts[key][v].sel(time=slices[key]["below"])
        height_sel = ecrad.get_model_level_of_altitude(
            bacardi_sel.alt, ecrad_ds, "half_level"
        )
        ecrad_plot = ecrad_ds.flux_dn_sw_norm.isel(half_level=height_sel)
        bacardi_plot = bacardi_sel["F_down_solar_diff_norm"]

        # actual plotting
        binsize = 10
        bins = np.arange(np.round(bacardi_plot.min() - 10),
                         np.round(bacardi_plot.max() + 10),
                         binsize)
        hist = ax.hist([bacardi_plot, ecrad_plot], label=["BACARDI", "ecRad"],
                       density=True, histtype="step", bins=bins, lw=2)
        h_distance = h.hellinger_distance(hist[0][0], hist[0][1])
        ax.set(
            xlabel="Normalized solar downward irradiance (W$\,$m$^{-2}$)",
            ylabel="Probability density function",
        )
        ax.legend()
        ax.text(
            0.025,
            0.95,
            f"{label[i]} {key} {ecrad.version_names[v]}\n"
            f"Binsize: {binsize}" + "$\,$W$\,$m$^{-2}$\n"
                                    f"H = {h_distance:.3f}",
            ha="left",
            va="top",
            transform=ax.transAxes,
            bbox=dict(fc="white", ec="black", alpha=0.8, boxstyle="round"),
        )
        ax.grid()

    plt.tight_layout()

    figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_bacardi_ecrad_f_down_solar_norm_PDF_below_cloud_{v}.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot PDF of solar downward irradiance below cloud
plt.rc("font", size=7)
label = ["a)", "b)"]
_, axs = plt.subplots(1, 2, figsize=(16 * h.cm, 9 * h.cm))

for i, key in enumerate(keys):
    ax = axs[i]
    bacardi_sel = bacardi_ds[key].sel(time=slices[key]["below"])
    ecrad_ds = ecrad_dicts[key]["v15"].sel(time=slices[key]["below"])
    height_sel = ecrad.get_model_level_of_altitude(
        bacardi_sel.alt, ecrad_ds, "half_level"
    )
    ecrad_plot = ecrad_ds.flux_dn_sw.isel(half_level=height_sel)
    bacardi_plot = bacardi_sel["F_down_solar_diff"]

    # actual plotting
    binsize = 10
    bins = np.arange(np.round(bacardi_plot.min() - 10),
                     np.round(bacardi_plot.max() + 10),
                     binsize)
    ax.hist([bacardi_plot, ecrad_plot], label=["BACARDI", "ecRad v15"],
            density=True, histtype="step", bins=bins, lw=2)
    ax.set(
        xlabel="Solar downward irradiance (W$\,$m$^{-2}$)",
        ylabel="Probability density function",
    )
    ax.legend()
    ax.text(
        0.025,
        0.95,
        f"{label[i]} {key}\n"
        f"Binsize: {binsize}" + "$\,$W$\,$m$^{-2}$",
        ha="left",
        va="top",
        transform=ax.transAxes,
        bbox=dict(fc="white", ec="black", alpha=0.8, boxstyle="round"),
    )

    ax.grid()

plt.tight_layout()

figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_bacardi_ecrad_f_down_solar_PDF_below_cloud.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot PDF of normalized solar downward irradiance above cloud
plt.rc("font", size=7)
label = ["a)", "b)"]

for v in ["v15"]:
    _, axs = plt.subplots(1, 2, figsize=(16 * h.cm, 9 * h.cm))
    for i, key in enumerate(keys):
        ax = axs[i]
        above_sel = (bahamas_ds[key].IRS_ALT > 11000).resample(time="1Min").first()
        bacardi_res = bacardi_ds[key].resample(time="1Min").mean()
        bacardi_plot = bacardi_res["F_down_solar_norm"].where(bacardi_res.alt > 11000)
        ecrad_ds = ecrad_dicts[key][v]
        height_sel = ecrad.get_model_level_of_altitude(
            bacardi_res.alt, ecrad_ds, "half_level"
        )
        ecrad_plot = ecrad_ds.flux_dn_sw_norm.isel(half_level=height_sel).where(above_sel)

        # actual plotting
        binsize = 10
        lower_edge = np.min([bacardi_plot.min(), ecrad_plot.min()])
        upper_edge = np.max([bacardi_plot.max(), ecrad_plot.max()])
        bins = np.arange(lower_edge - 10,
                         upper_edge + 10,
                         binsize)
        sns.histplot(bacardi_plot, bins=bins, kde=False, stat="density", label="BACARDI", ax=ax)
        sns.histplot(ecrad_plot, bins=bins, kde=False, stat="density", label="ecRad", ax=ax)
        bacardi_hist = np.histogram(bacardi_plot, density=True, bins=bins)
        ecrad_hist = np.histogram(ecrad_plot, density=True, bins=bins)

        h_distance = h.hellinger_distance(bacardi_hist[0], ecrad_hist[0])
        ax.set(
            xlabel="Normalized solar downward irradiance (W$\,$m$^{-2}$)",
            ylabel="Density",
        )
        ax.legend(loc=6)
        ax.text(
            0.025,
            0.95,
            f"{label[i]} {key} {ecrad.version_names[v]}\n"
            f"Binsize: {binsize}" + "$\,$W$\,$m$^{-2}$\n"
                                    f"$H = ${h_distance:.3f}",
            ha="left",
            va="top",
            transform=ax.transAxes,
            bbox=dict(fc="white", ec="black", alpha=0.8, boxstyle="round"),
        )
        ax.grid()

    plt.tight_layout()

    figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_bacardi_ecrad_f_down_solar_norm_PDF_above_cloud_{v}.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot reflectivity of above cloud section
plt.rc("font", size=7)
label = ["a)", "b)"]
bacardi_var = "reflectivity_solar"
ecrad_var = "reflectivity_sw"
for v in ["v15", "v18", "v19"]:
    _, axs = plt.subplots(1, 2, figsize=(16 * h.cm, 9 * h.cm))

    for i, key in enumerate(keys):
        ax = axs[i]
        bacardi_sel = bacardi_ds[key].sel(time=slices[key]["above"])
        bacardi_plot = bacardi_sel[bacardi_var]
        ecrad_ds = ecrad_dicts[key][v]
        height_sel = ecrad.get_model_level_of_altitude(bacardi_sel.alt, ecrad_ds, "half_level")
        ecrad_plot = (ecrad_ds[ecrad_var]
                      .isel(half_level=height_sel)
                      .sel(time=slices[key]["above"]))
        ecrad_std = (ecrad_orgs[key][v][ecrad_var]
                     .std(dim="column")
                     .isel(half_level=height_sel)
                     .sel(time=slices[key]["above"]))

        # actual plotting
        ax.plot(bacardi_plot.time, bacardi_plot, label="BACARDI")
        ax.errorbar(ecrad_plot.time.to_numpy(), ecrad_plot,
                    yerr=ecrad_std.to_numpy(),
                    marker=".", label=f"ecRad {ecrad.version_names[v]}")
        h.set_xticks_and_xlabels(ax, pd.Timedelta(30, "Min"))
        ax.set(
            xlabel="Time (UTC)",
            ylabel="Solar reflectivity",
        )
        ax.legend()
        ax.grid()
        ax.text(0.025, 0.95, f"{label[i]} {key}",
                ha="left",
                va="top",
                transform=ax.transAxes,
                bbox=dict(fc="white", ec="black", alpha=0.8, boxstyle="round"),
                )

    plt.tight_layout()

    figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_ecrad_bacardi_solar_reflectivity_above_cloud_{v}.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot reflectivity of above cloud section all in one plot
plt.rc("font", size=7)
label = ["a)", "b)"]
bacardi_var = "reflectivity_solar"
ecrad_var = "reflectivity_sw"
_, axs = plt.subplots(2, 1, figsize=(16 * h.cm, 9 * h.cm))

for i, key in enumerate(keys):
    ax = axs[i]
    bacardi_sel = bacardi_ds[key].sel(time=slices[key]["above"])
    bacardi_plot = bacardi_sel[bacardi_var]
    # plot BACARDI
    ax.plot(bacardi_plot.time, bacardi_plot, label="BACARDI")
    # plot all ecRad versions
    for v in ["v15", "v18", "v19"]:
        ecrad_ds = ecrad_dicts[key][v]
        height_sel = height_sels[key][v]
        ecrad_plot = (ecrad_ds[ecrad_var]
                      .isel(half_level=height_sel)
                      .sel(time=slices[key]["above"]))
        ecrad_std = (ecrad_orgs[key][v][ecrad_var]
                     .std(dim="column")
                     .isel(half_level=height_sel)
                     .sel(time=slices[key]["above"]))

        ax.errorbar(ecrad_plot.time.to_numpy(), ecrad_plot,
                    yerr=ecrad_std.to_numpy(),
                    marker=".", label=f"ecRad\n{ecrad.version_names[v]}")
    # plot ecRad versions using VarCloud as input
    for v in ["v17", "v21"]:
        ecrad_ds = ecrad_dicts[key][v]
        height_sel = height_sels[key][v]
        ecrad_plot = (ecrad_ds[ecrad_var]
                      .isel(half_level=height_sel)
                      .sel(time=slices[key]["above"]))
        ax.plot(ecrad_plot.time, ecrad_plot,
                marker=".", label=f"ecRad\n{ecrad.version_names[v]}")

    h.set_xticks_and_xlabels(ax, pd.Timedelta(30, "Min"))
    ax.set(
        ylabel="Solar reflectivity",
    )
    ax.grid()
    ax.text(0.025, 0.95, f"{label[i]} {key}",
            ha="left",
            va="top",
            transform=ax.transAxes,
            bbox=dict(fc="white", ec="black", alpha=0.8, boxstyle="round"),
            )

handles, labels = axs[0].get_legend_handles_labels()
labels = [label.replace(" ", "\n") for label in labels]
_.legend(handles=handles, labels=labels, loc="center right", bbox_to_anchor=(1, .5))
axs[1].set(xlabel="Time (UTC)")
plt.tight_layout()
plt.subplots_adjust(right=0.845)
figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_ecrad_bacardi_solar_reflectivity_above_cloud_all.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot PDF of normalized solar downward irradiance below cloud - all ice optics
plt.rc("font", size=7)
label = [["a)", "b)", "c)"], ["d)", "e)", "f)"]]
binsize = 10
_, axs = plt.subplots(2, 3, figsize=(18 * h.cm, 14 * h.cm))
for i, key in enumerate(keys):
    ax = axs[i]
    l = label[i]
    bacardi_sel = bacardi_ds_res[key].sel(time=slices[key]["below"])
    bacardi_plot = bacardi_sel["F_down_solar_diff_norm"]
    bins = np.arange(np.round(bacardi_plot.min() - 10),
                     np.round(bacardi_plot.max() + 10),
                     binsize)
    # BACARDI histogram
    bacardi_hist = np.histogram(bacardi_plot, density=True, bins=bins)
    for ii, v in enumerate(["v15", "v18", "v19"]):
        a = ax[ii]
        ecrad_ds = ecrad_dicts[key][v].sel(time=slices[key]["below"])
        height_sel = ecrad.get_model_level_of_altitude(
            bacardi_sel.alt, ecrad_ds, "half_level"
        )
        ecrad_plot = ecrad_ds.flux_dn_sw_norm.isel(half_level=height_sel)

        # actual plotting
        sns.histplot(bacardi_plot, label="BACARDI", ax=a, stat="density", kde=True, bins=bins)
        sns.histplot(ecrad_plot, label=ecrad.version_names[v], stat="density",
                     kde=True, bins=bins, ax=a, color=cbc[ii + 1])
        hist = np.histogram(ecrad_plot, density=True, bins=bins)
        h_distance = h.hellinger_distance(bacardi_hist[0], hist[0])
        a.set(ylabel="")
        a.legend(loc=1)
        a.text(
            0.04,
            0.95,
            f"{l[ii]} {key}\n"
            f"$H$ = {h_distance:.3f}\n"
            f"n = {len(ecrad_plot):.0f}",
            ha="left",
            va="top",
            transform=a.transAxes,
            bbox=dict(fc="white", ec="black", alpha=0.8, boxstyle="round"),
        )
        a.grid()

    ax[0].set(ylabel="Density")
    ax[1].set(xlabel="Normalized solar downward irradiance (W$\,$m$^{-2}$)")

plt.tight_layout()

figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_bacardi_ecrad_f_down_solar_norm_PDF_below_cloud_ice_optics.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot PDF of normalized solar downward irradiance below cloud - varcloud all ice optics
plt.rc("font", size=7)
label = [["a)", "b)", "c)"], ["d)", "e)", "f)"]]
text_locx = [0.04, 0.65]
text_locy = [0.7, 0.5]
binsize = 10
_, axs = plt.subplots(2, 3, figsize=(18 * h.cm, 14 * h.cm))
for i, key in enumerate(keys):
    ax = axs[i]
    l = label[i]
    ts_s, ts_e = ecrad_dicts[key]["v16"].time[[0, -1]]
    bacardi_sel = bacardi_ds[key].sel(time=slice(ts_s, ts_e))
    bacardi_plot = bacardi_sel["F_down_solar_diff_norm"].resample(time="1s").mean()
    bins = np.arange(np.round(bacardi_plot.min() - 10),
                     np.round(bacardi_plot.max() + 10),
                     binsize)
    # BACARDI histogram
    bacardi_hist = np.histogram(bacardi_plot, density=True, bins=bins)
    x_loc = text_locx[i]
    y_loc = text_locy[i]
    for ii, v in enumerate(["v16", "v20", "v28"]):
        a = ax[ii]
        ecrad_ds = ecrad_dicts[key][v]
        altitude = bacardi_sel.alt.sel(time=ecrad_ds.time)
        height_sel = height_sels[key][v]
        ecrad_plot = ecrad_ds.flux_dn_sw_norm.isel(half_level=height_sel)

        # actual plotting
        sns.histplot(bacardi_plot, label="BACARDI", ax=a, stat="density", kde=True, bins=bins)
        sns.histplot(ecrad_plot, label=ecrad.version_names[v], stat="density",
                     kde=True, bins=bins, ax=a, color=cbc[ii + 1])
        hist = np.histogram(ecrad_plot, density=True, bins=bins)
        h_distance = h.hellinger_distance(bacardi_hist[0], hist[0])
        a.set(ylabel="")
        a.legend(loc=2)
        a.text(
            x_loc,
            y_loc,
            f"{l[ii]} {key}\n"
            f"$H$ = {h_distance:.3f}\n"
            f"n = {len(ecrad_plot):.0f}",
            ha="left",
            va="top",
            transform=a.transAxes,
            bbox=dict(fc="white", ec="black", alpha=0.8, boxstyle="round"),
        )
        a.grid()

    ax[0].set(ylabel="Density")
    ax[1].set(xlabel="Normalized solar downward irradiance (W$\,$m$^{-2}$)")

plt.tight_layout()

figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_bacardi_ecrad_f_down_solar_norm_PDF_below_cloud_ice_optics_varcloud.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot PDF of normalized solar downward irradiance below cloud - varcloud all ice optics 1min
plt.rc("font", size=7)
label = [["a)", "b)", "c)"], ["d)", "e)", "f)"]]
text_locx = [0.04, 0.65]
text_locy = [0.7, 0.5]
binsize = 10
_, axs = plt.subplots(2, 3, figsize=(18 * h.cm, 14 * h.cm))
for i, key in enumerate(keys):
    ax = axs[i]
    l = label[i]
    ts_s, ts_e = ecrad_dicts[key]["v16"].time[[0, -1]]
    bacardi_sel = bacardi_ds_res[key].sel(time=slice(ts_s, ts_e))
    bacardi_plot = bacardi_sel["F_down_solar_diff_norm"]
    bins = np.arange(np.round(bacardi_plot.min() - 10),
                     np.round(bacardi_plot.max() + 10),
                     binsize)
    # BACARDI histogram
    bacardi_hist = np.histogram(bacardi_plot, density=True, bins=bins)
    x_loc = text_locx[i]
    y_loc = text_locy[i]
    for ii, v in enumerate(["v16", "v20", "v28"]):
        a = ax[ii]
        ecrad_ds = ecrad_dicts[key][v]
        height_sel = height_sels[key][v]
        ecrad_plot = ecrad_ds.flux_dn_sw_norm.isel(half_level=height_sel).resample(time="1Min").mean()

        # actual plotting
        sns.histplot(bacardi_plot, label="BACARDI", ax=a, stat="density", kde=True, bins=bins)
        sns.histplot(ecrad_plot, label=ecrad.version_names[v], stat="density",
                     kde=True, bins=bins, ax=a, color=cbc[ii + 1])
        hist = np.histogram(ecrad_plot, density=True, bins=bins)
        h_distance = h.hellinger_distance(bacardi_hist[0], hist[0])
        a.set(ylabel="")
        a.legend(loc=2)
        a.text(
            x_loc,
            y_loc,
            f"{l[ii]} {key}\n"
            f"$H$ = {h_distance:.3f}\n"
            f"n = {len(ecrad_plot):.0f}",
            ha="left",
            va="top",
            transform=a.transAxes,
            bbox=dict(fc="white", ec="black", alpha=0.8, boxstyle="round"),
        )
        a.grid()

    ax[0].set(ylabel="Density")
    ax[1].set(xlabel="Normalized solar downward irradiance (W$\,$m$^{-2}$)")

plt.tight_layout()

figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_bacardi_ecrad_f_down_solar_norm_PDF_below_cloud_ice_optics_varcloud_1min.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot result of albedo sensitivity study as boxplots
labels = ["a)", "b)"]
plt.rc("font", size=7)
_, axs = plt.subplots(1, 2, figsize=(16 * h.cm, 9 * h.cm))
for i, key in enumerate(keys):
    ax = axs[i]
    df = albedo_dfs[key]
    sns.boxplot(df, notch=True, ax=ax)
    ax.text(
        0.03,
        0.98,
        f"{labels[i]} {key}\n"
        f"n = {len(df):.0f}",
        ha="left",
        va="top",
        transform=ax.transAxes,
        bbox=dict(fc="white", ec="black", alpha=0.8, boxstyle="round"),
    )
    ylims = ax.get_ylim()
    ax.set_ylim(ylims[0], ylims[1] + 10)

axs[0].set_ylabel("Solar downward irradiance (W$\,$m$^{-2}$)")
plt.tight_layout()
figname = f"{plot_path}/HALO-AC3_RF17_RF18_albedo_experiment_ecrad_flux_dn_sw_below_cloud_boxplot.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% testing
x = bacardi_ds["RF17"].F_down_solar_diff_norm.sel(time=slices["RF17"]["below"]).dropna(dim="time")
x = ecrad_dicts["RF17"]["v20"].flux_dn_sw_norm.isel(half_level=height_sels["RF17"]["v20"])
q75, q25 = np.percentile(x, [75, 25])
iqr = q75 - q25
print(2 * iqr / np.cbrt(len(x)))
