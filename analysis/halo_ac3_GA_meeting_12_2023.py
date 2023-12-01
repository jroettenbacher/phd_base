#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 21.11.2023

Most plots for the presentation during the AC3 General Assembly / |haloac3| Meeting in Leipzig 06 - 08 December 2023.

"""

# %% module import
import pylim.helpers as h
import pylim.halo_ac3 as meta
import ac3airborne
from ac3airborne.tools import flightphase
import sys
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors, ticker, gridspec, patheffects
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cmasher as cmr
import xarray as xr
import pandas as pd
from tqdm import tqdm
import logging


log = logging.getLogger("pylim")
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

h.set_cb_friendly_colors("petroff_6")
cbc = h.get_cb_friendly_colors("petroff_6")

# %% set paths
campaign = "halo-ac3"
plot_path = "C:/Users/Johannes/Documents/Doktor/conferences_workshops/2023_12_05_AC3_General_Assembly/figures"
h.make_dir(plot_path)
trajectory_path = f"{h.get_path('trajectories', campaign=campaign)}/selection_CC_and_altitude"
keys = ["RF17", "RF18"]
ecrad_versions = ["v15", "v15.1", "v16", "v17", "v18", "v18.1", "v19", "v19.1", "v20", "v21", "v28", "v29"]

# %% read in data
(
    bahamas_ds,
    bacardi_ds,
    bacardi_ds_res,
    ecrad_dicts,
    varcloud_ds,
    radar_ds,
    lidar_ds,
    above_clouds,
    below_clouds,
    slices,
    ecrad_orgs,
    ifs_ds,
    dropsonde_ds,
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
    radar_path = h.get_path("hamp_mira", flight, campaign)
    lidar_path = h.get_path("wales", flight, campaign)

    # filenames
    bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{key}_v1_JR.nc"
    bacardi_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1_JR.nc"
    libradtran_bb_solar_si = f"HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_solar_si_{date}_{key}.nc"
    libradtran_bb_thermal_si = f"HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_thermal_si_{date}_{key}.nc"
    ifs_file = f"ifs_{date}_00_ml_O1280_processed.nc"
    varcloud_file = [f for f in os.listdir(varcloud_path) if f.endswith(".nc")][0]
    dropsonde_files = [f for f in os.listdir(dropsonde_path) if f.endswith("QC.nc")]
    radar_file = f"HALO_HALO_AC3_radar_unified_{key}_{date}_v2.6.nc"
    lidar_file = f"HALO-AC3_HALO_WALES_bsrgl_{date}_{key}_V2.0.nc"

    # read in aircraft data
    bahamas_ds[key] = xr.open_dataset(f"{bahamas_path}/{bahamas_file}")
    bacardi = xr.open_dataset(f"{bacardi_path}/{bacardi_file}")
    radar = xr.open_dataset(f"{radar_path}/{radar_file}")
    lidar = xr.open_dataset(f"{lidar_path}/{lidar_file}")

    lidar = lidar.rename(altitude="height").transpose("time", "height")
    lidar["height"] = lidar.height / 1000
    radar["height"] = radar.height / 1000
    # interpolate lidar data onto radar range resolution
    new_range = radar.height.values
    lidar_r = lidar.interp(height=np.flip(new_range))
    # convert lidar data to radar convention: [time, height], ground = 0m
    lidar_r = lidar_r.assign_coords(height=np.flip(new_range)).isel(height=slice(None, None, -1))
    # create radar mask
    radar["mask"] = ~np.isnan(radar["dBZg"])
    # combine radar and lidar mask
    lidar_mask = lidar_r["flags"] == 0
    lidar_mask = lidar_mask.where(lidar_mask == 0, 2).resample(time="1s").first()
    radar["radar_lidar_mask"] = radar["mask"] + lidar_mask

    radar_ds[key] = radar
    lidar_ds[key] = lidar

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

    # read in resampled BACARDI data
    bacardi_res = xr.open_dataset(f"{bacardi_path}/{bacardi_file.replace('.nc', '_1Min.nc')}")
    # normalize downward irradiance for cos SZA
    for var in ["F_down_solar", "F_down_solar_diff"]:
        bacardi_res[f"{var}_norm"] = bacardi_res[var] / np.cos(np.deg2rad(bacardi_res["sza"]))
    bacardi_ds_res[key] = bacardi_res.copy()

    # read in ecrad data
    ecrad_dict, ecrad_org = dict(), dict()

    for k in ecrad_versions:
        ds = xr.open_dataset(f"{ecrad_path}/ecrad_merged_inout_{date}_{k}.nc")

        if "column" in ds.dims:
            b_ds = bacardi_res.expand_dims(dict(column=np.arange(0, len(ds.column)))).copy()
            ecrad_ds = ds.isel(half_level=ds["aircraft_level"], column=slice(0, 10))
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

        ds["tiwp"] = ds.iwp.where(ds.iwp != np.inf, np.nan).sum(dim="level")
        ds["transmissivity_sw_toa"] = ds["flux_dn_sw"] / ds["flux_dn_sw_clear"].isel(half_level=0)
        ds["transmissivity_sw_above_cloud"] = ds["flux_dn_sw"] / ds["flux_dn_sw_clear"].isel(half_level=73)
        for var in ["flux_dn_sw", "flux_dn_direct_sw", "transmissivity_sw_above_cloud", "transmissivity_sw_toa"]:
            ds[f"{var}_norm"] = ds[var] / ds["cos_solar_zenith_angle"]

        ecrad_dict[k] = ds.copy()

    ecrad_dicts[key] = ecrad_dict
    ecrad_orgs[key] = ecrad_org

    # interpolate standard ecRad simulation onto BACARDI time
    bacardi["ecrad_fdw"] = ecrad_dict["v15"].flux_dn_sw_clear.interp(time=bacardi.time,
                                                                     kwargs={"fill_value": "extrapolate"})
    # calculate transmissivity using ecRad at TOA and above cloud
    bacardi["transmissivity_TOA"] = bacardi["F_down_solar"] / bacardi["ecrad_fdw"].isel(half_level=0)
    bacardi["transmissivity_above_cloud"] = bacardi["F_down_solar"] / bacardi["ecrad_fdw"].isel(half_level=73)
    # normalize transmissivity by cosine of solar zenith angle
    for var in ["transmissivity_TOA", "transmissivity_above_cloud"]:
        bacardi[f"{var}_norm"] = bacardi[var] / np.cos(np.deg2rad(bacardi["sza"]))
    bacardi_ds[key] = bacardi

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
        below_slice = slice(pd.to_datetime("2022-04-11 11:35"), below_cloud["end"])
        case_slice = slice(above_cloud["start"], below_cloud["end"])
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

# %% plot flight track together with trajectories and high cloud cover RF 17
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
map_crs = ccrs.NorthPolarStereo()

plt.rc("font", size=6)
fig = plt.figure(figsize=(18 * h.cm, 8 * h.cm))
gs = gridspec.GridSpec(1, 1)

# plot trajectory map 11 April in first row and first column
ax = fig.add_subplot(gs[0, 0], projection=map_crs)
ax.coastlines(alpha=0.5)
ax.set_xlim((-2000000, 2000000))
ax.set_ylim((-3000000, 500000))
gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=1, color='gray', alpha=0.5,
                  linestyle=':', x_inline=False, y_inline=False, rotate_labels=False)
gl.xlocator = ticker.FixedLocator(np.arange(-180, 180, 20))
gl.ylocator = ticker.FixedLocator(np.arange(60, 90, 5))
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
cci = ax.tricontour(ifs.lon, ifs.lat, ifs.ci, ci_levels, transform=data_crs, linestyles="--", colors="#332288",
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
ax.scatter(ins_hl.IRS_LON[::10], ins_hl.IRS_LAT[::10], c=cbc[3], alpha=1, marker=".", s=1, zorder=400,
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
           Patch(facecolor="royalblue")]  # cloud cover
labels = ["HALO flight track", "Dropsonde", "Sea ice edge", "Mean sea level\npressure (hPa)",
          "High cloud cover\nat 12:00 UTC"]
ax.legend(handles=handles, labels=labels, framealpha=1, loc=2, title="a) RF 17", alignment="left")

# add map inset of case study area
# axins = inset_axes(ax, width="20%", height="40%", axes_class=cartopy.mpl.geoaxes.GeoAxes,
#                    axes_kwargs=dict(projection=map_crs))
# axins.scatter(ins_hl.IRS_LON[::10], ins_hl.IRS_LAT[::10], c=cbc[3], alpha=1, marker=".", s=1, zorder=400,
#               transform=ccrs.PlateCarree(), linestyle="solid")

plt.tight_layout()
figname = f"{plot_path}/HALO-AC3_RF17_fligh_track_trajectories_plot_overview.png"
plt.savefig(figname, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# %% plot flight track together with trajectories and high cloud cover RF 18
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
map_crs = ccrs.NorthPolarStereo()

plt.rc("font", size=6)
fig = plt.figure(figsize=(18 * h.cm, 8 * h.cm))
gs = gridspec.GridSpec(1, 1)

# plot trajectories 12 April
ax = fig.add_subplot(gs[0], projection=map_crs)
ax.coastlines(alpha=0.5)
ax.set_xlim((-2000000, 2000000))
ax.set_ylim((-3000000, 500000))
gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=1, color='gray', alpha=0.5,
                  linestyle=':', x_inline=False, y_inline=False, rotate_labels=False)
gl.top_labels = False
gl.right_labels = False
gl.xlocator = ticker.FixedLocator(np.arange(-180, 180, 20))
gl.ylocator = ticker.FixedLocator(np.arange(60, 90, 5))

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
ax.scatter(ins_hl.IRS_LON[::10], ins_hl.IRS_LAT[::10], c=cbc[3], alpha=1, marker=".", s=1, zorder=400,
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
           Patch(facecolor="royalblue")]  # cloud cover
labels = ["HALO flight track", "Dropsonde", "Sea ice edge", "Mean sea level\npressure (hPa)",
          "High cloud cover\nat 12:00 UTC"]
ax.legend(handles=handles, labels=labels, framealpha=1, loc=2, title="b) RF 18", alignment="left")

plt.tight_layout()
figname = f"{plot_path}/HALO-AC3_RF18_fligh_track_trajectories_plot_overview.png"
plt.savefig(figname, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# %% plot lidar data for case study with radar & lidar mask
for key in keys:
    plot_ds = (lidar_ds[key]["backscatter_ratio"]
               .where((lidar_ds[key].flags == 0)
                      & (lidar_ds[key].backscatter_ratio > 1))
               .sel(time=slices[key]["above"]))
    ct_plot = radar_ds[key]["radar_lidar_mask"].sel(time=slices[key]["above"])
    plt.rc("font", size=12)
    _, ax = plt.subplots(figsize=(20 * h.cm, 6.5 * h.cm))
    plot_ds.plot(x="time", y="height", cmap=cmr.chroma_r, norm=colors.LogNorm(), vmax=100,
                 cbar_kwargs=dict(label="Backscatter Ratio \nat 532$\,$nm", pad=0.01))
    ct_plot.plot.contour(x="time", levels=[2.9], colors=cbc[0])
    ax.plot(bahamas_ds[key].time, bahamas_ds[key]["IRS_ALT"] / 1000, color="k", label="HALO altitude")
    ax.plot([], color=cbc[0], label="Radar & Lidar Mask", lw=2)
    ax.legend(loc=1, ncols=2)
    h.set_xticks_and_xlabels(ax, slices[key]["above"].stop - slices[key]["above"].start)
    ax.set(xlabel="Time (UTC)", ylabel="Altitude (km)", ylim=(0, 12))
    plt.tight_layout()

    figname = f"{plot_path}/HALO_AC3_HALO_{key}_lidar_backscatter_ratio_532_radar_mask_cs.png"
    plt.savefig(figname, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# %% visualize grid points from IFS in respect to flight track
key = "RF17"
ds = ecrad_orgs[key]["v15"]
ds1 = bahamas_ds[key]
# plot points along flight track
data_crs = ccrs.PlateCarree()
plot_crs = ccrs.NorthPolarStereo()
plot_ds = ds.sel(time=slices[key]["case"])
plot_ds1 = ds1.sel(time=slices[key]["case"])
_, ax = plt.subplots(figsize=(8 * h.cm, 8 * h.cm), subplot_kw=dict(projection=plot_crs))
ax.set_extent([-30, 0, 88, 89], crs=data_crs)
# ax.add_feature(cfeature.OCEAN)
# ax.add_feature(cfeature.COASTLINE)
# ax.add_feature(cfeature.LAND)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
# several timesteps
ax.plot(plot_ds1.IRS_LON, plot_ds1.IRS_LAT, c="k", transform=data_crs, label="Flight track")
ax.scatter(plot_ds.lon, plot_ds.lat, c=cbc[0], transform=data_crs)
ax.scatter(plot_ds.lon.sel(column=0), plot_ds.lat.sel(column=0), c=cbc[1], transform=data_crs)
# one time step
plot_ds = plot_ds.sel(time="2022-04-11 11:25", method="nearest")
ax.scatter(plot_ds.lon, plot_ds.lat, transform=data_crs, c=ds.column, marker="^", s=100, cmap="tab10")
ax.legend()
figname = f"{plot_path}/gridpoints.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot PDF of IWC
plt.rc("font", size=11)
legend_labels = ["VarCloud", "IFS"]
binsizes = dict(iwc=0.5, reice=4)
text_loc_x = 0.03
text_loc_y = 0.79
_, axs = plt.subplots(1, 2, figsize=(24 * h.cm, 10 * h.cm))
ylims = {"iwc": (0, 0.75), "reice": (0, 0.095)}
# left panel - RF17 IWC
ax = axs[0]
plot_ds = ecrad_dicts["RF17"]
sel_time = slice(pd.to_datetime("2022-04-11 10:49"), pd.to_datetime("2022-04-11 11:04"))
binsize = binsizes["iwc"]
bins = np.arange(-0.25, 5.1, binsize)
for i, v in enumerate(["v16", "v15.1"]):
    if v == "v16":
        pds = plot_ds[v].iwc
    else:
        iwc, cc = plot_ds[v].iwc.sel(time=sel_time), plot_ds[v].cloud_fraction.sel(time=sel_time)
        pds = iwc.where(cc > 0).where(cc == 0, iwc / cc)

    pds = pds.to_numpy().flatten() * 1e6
    pds = pds[~np.isnan(pds)]
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i] + f" (n={len(pds)})",
        color=cbc[i],
        histtype="step",
        density=True,
        lw=2,
    )
ax.legend()
ax.grid()
ax.text(text_loc_x, text_loc_y,
        "a) RF 17\n"
        f"Binsize: {binsize:.1f}" + "$\,$mg$\,$m$^{-3}$",
        transform=ax.transAxes, bbox=dict(boxstyle="Round", fc="white"))
ax.set(title=f"RF 17 - 11 April 2022 {sel_time.start:%H:%M} - {sel_time.stop:%H:%M} UTC",
       ylabel=f"Probability density function",
       xlabel=f"Ice water content ({h.plot_units['iwc']})",
       ylim=ylims["iwc"])

# right panel - RF18 IWC
ax = axs[1]
plot_ds = ecrad_dicts["RF18"]
sel_time = slice(pd.to_datetime("2022-04-12 11:04"), pd.to_datetime("2022-04-12 11:24"))
binsize = binsizes["iwc"]
bins = np.arange(-0.25, 5.1, binsize)
for i, v in enumerate(["v16", "v15.1"]):
    if v == "v16":
        pds = plot_ds[v].iwc
    else:
        iwc, cc = plot_ds[v].iwc.sel(time=sel_time), plot_ds[v].cloud_fraction.sel(time=sel_time)
        pds = iwc.where(cc > 0).where(cc == 0, iwc / cc)

    pds = pds.to_numpy().flatten() * 1e6
    pds = pds[~np.isnan(pds)]
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i] + f" (n={len(pds)})",
        color=cbc[i],
        histtype="step",
        density=True,
        lw=2,
    )
ax.legend()
ax.grid()
ax.text(text_loc_x, text_loc_y,
        "b) RF 18\n"
        f"Binsize: {binsize:.1f}" + "$\,$mg$\,$m$^{-3}$",
        transform=ax.transAxes, bbox=dict(boxstyle="Round", fc="white"))
ax.set(title=f"RF 18 - 12 April 2022 {sel_time.start:%H:%M} - {sel_time.stop:%H:%M} UTC",
       ylabel=f"",
       xlabel=f"Ice water content ({h.plot_units['iwc']})",
       ylim=ylims["iwc"])


plt.tight_layout()
figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_IFS_iwc_pdf_case_studies.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot PDF of re_ice
plt.rc("font", size=11)
legend_labels = ["VarCloud", "IFS"]
binsizes = dict(iwc=0.5, reice=4)
text_loc_x = 0.03
text_loc_y = 0.87
_, axs = plt.subplots(1, 2, figsize=(24 * h.cm, 10 * h.cm))
ylims = {"iwc": (0, 0.75), "reice": (0, 0.12)}
# left panel - RF17 re_ice
ax = axs[0]
plot_ds = ecrad_dicts["RF17"]
sel_time = slice(pd.to_datetime("2022-04-11 10:49"), pd.to_datetime("2022-04-11 11:04"))
binsize = binsizes["reice"]
bins = np.arange(0, 100, binsize)
for i, v in enumerate(["v16", "v15.1"]):
    if v == "v16":
        pds = plot_ds[v].re_ice.to_numpy().flatten() * 1e6
    else:
        pds = plot_ds[v].re_ice.sel(time=sel_time).to_numpy().flatten() * 1e6

    pds = pds[~np.isnan(pds)]
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i] + f" (n={len(pds)})",
        color=cbc[i],
        histtype="step",
        density=True,
        lw=2,
    )
ax.legend(loc=1)
ax.grid()
ax.text(text_loc_x, text_loc_y,
        "c) RF 17\n"
        f"Binsize: {binsize:.0f}$\,\mu$m",
        transform=ax.transAxes,
        bbox=dict(boxstyle="Round", fc="white", alpha=0.8))
ax.set(ylabel="Probability density function",
       xlabel=f"Ice effective radius ({h.plot_units['re_ice']})",
       ylim=ylims["reice"])

# right panel - RF18 re_ice
ax = axs[1]
plot_ds = ecrad_dicts["RF18"]
sel_time = slice(pd.to_datetime("2022-04-12 11:04"), pd.to_datetime("2022-04-12 11:24"))
binsize = binsizes["reice"]
bins = np.arange(0, 100, binsize)
for i, v in enumerate(["v16", "v15.1"]):
    if v == "v16":
        pds = plot_ds[v].re_ice.to_numpy().flatten() * 1e6
    else:
        pds = plot_ds[v].re_ice.sel(time=sel_time).to_numpy().flatten() * 1e6
    pds = pds[~np.isnan(pds)]
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i] + f" (n={len(pds)})",
        color=cbc[i],
        histtype="step",
        density=True,
        lw=2,
    )
ax.legend()
ax.grid()
ax.text(text_loc_x, text_loc_y,
        "d) RF 18\n"
        f"Binsize: {binsize:.0f}$\,\mu$m",
        transform=ax.transAxes, bbox=dict(boxstyle="Round", fc="white"))
ax.set(ylabel="",
       xlabel=f"Ice effective radius ({h.plot_units['re_ice']})",
       ylim=ylims["reice"])

plt.tight_layout()
figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_IFS_re_ice_pdf_case_studies.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()
