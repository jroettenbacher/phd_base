#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 18.12.2023

Plots for third PhD Talk

- Map of RF 17 and RF 18
- PDF of BACARDI solar transmissivity below cloud

"""

# %% import modules
import pylim.helpers as h
import pylim.halo_ac3 as meta
from pylim import ecrad
import ac3airborne
from ac3airborne.tools import flightphase
import cartopy.crs as ccrs
import cmasher as cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import os
from matplotlib import patheffects
from sklearn.neighbors import BallTree
from tqdm import tqdm

h.set_cb_friendly_colors("petroff_6")
cbc = h.get_cb_friendly_colors("petroff_6")

# %% set paths
campaign = "halo-ac3"
plot_path = "C:/Users/Johannes/Documents/Doktor/ppt_gallery/figures_third_phd_talk"
trajectory_path = f"{h.get_path('trajectories', campaign=campaign)}/selection_CC_and_altitude"
keys = ["RF17", "RF18"]
ecrad_versions = ["v15", "v15.1", "v16", "v17", "v18", "v18.1", "v19", "v19.1", "v20", "v21", "v28", "v29",
                  "v30.1", "v31.1", "v32.1", "v33", "v34", "v35"]

# %% read in data
(
    bahamas_ds,
    bacardi_ds,
    bacardi_ds_res,
    ecrad_dicts,
    varcloud_ds,
    lidar_ds,
    radar_ds,
    above_clouds,
    below_clouds,
    slices,
    ecrad_orgs,
    ifs_ds,
    ifs_ds_sel,
    dropsonde_ds,
    albedo_dfs
) = (
dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict())

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
    dropsonde_path = f"{dropsonde_path}/Level_1" if key == "RF17" else f"{dropsonde_path}/Level_2"
    radar_path = h.get_path("hamp_mira", flight, campaign)
    lidar_path = h.get_path("wales", flight, campaign)

    # filenames
    bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{key}_v1_JR.nc"
    bacardi_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1_JR.nc"
    libradtran_bb_solar_si = f"HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_solar_si_{date}_{key}.nc"
    libradtran_bb_thermal_si = f"HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_thermal_si_{date}_{key}.nc"
    ifs_file = f"ifs_{date}_00_ml_O1280_processed.nc"
    varcloud_file = [f for f in os.listdir(varcloud_path) if f.endswith(".nc")][0]
    dropsonde_files = [f for f in os.listdir(dropsonde_path) if f.endswith(".nc")]
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
        k = file[-11:-5] if key == "RF17" else file[27:35].replace("_", "")
        dropsondes[k] = xr.open_dataset(f"{dropsonde_path}/{file}")
        if key == "RF18":
            dropsondes[k]["ta"] = dropsondes[k].ta - 273.15
            dropsondes[k]["rh"] = dropsondes[k].rh * 100
    dropsonde_ds[key] = dropsondes

    # read in ifs data
    ifs = xr.open_dataset(f"{ifs_path}/{ifs_file}")
    ifs = ifs.set_index(rgrid=["lat", "lon"])
    # filter low clouds according to ECMWF low cloud criterion (pressure higher than 0.8 * surface pressure)
    cloud_data = ifs[["q_liquid", "q_ice", "cloud_fraction", "clwc", "ciwc", "crwc", "cswc"]]
    pressure_filter = ifs.pressure_full.sel(level=137) * 0.8
    low_cloud_filter = ifs.pressure_full < pressure_filter  # False for low clouds
    cloud_data = cloud_data.where(low_cloud_filter, 0)  # replace where False with 0
    ifs.update(cloud_data)
    ifs_ds[key] = ifs.copy(deep=True)

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
        ecrad_org[k] = ds.copy(deep=True)
        # select only center column for direct comparisons
        ds = ds.sel(column=0, drop=True) if "column" in ds.dims else ds
        ecrad_dict[k] = ds.copy(deep=True)

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

    # get IFS data for the case study area
    ifs_lat_lon = np.column_stack((ifs.lat, ifs.lon))
    ifs_tree = BallTree(np.deg2rad(ifs_lat_lon), metric="haversine")
    # generate an array with lat, lon values from the flight position
    bahamas_sel = bahamas_ds[key].sel(time=slices[key]["above"])
    points = np.deg2rad(np.column_stack((bahamas_sel.IRS_LAT.to_numpy(), bahamas_sel.IRS_LON.to_numpy())))
    _, idxs = ifs_tree.query(points, k=10)  # query the tree
    closest_latlons = ifs_lat_lon[idxs]
    # remove duplicates
    closest_latlons = np.unique(closest_latlons
                                .reshape(closest_latlons.shape[0] * closest_latlons.shape[1], 2),
                                axis=0)
    latlon_sel = [(x, y) for x, y in closest_latlons]
    ifs_ds_sel[key] = ifs.sel(rgrid=latlon_sel)

# %% plot flight track together with high cloud cover
plt.rc("font", size=10)
data_crs = ccrs.PlateCarree()
map_crs = ccrs.NorthPolarStereo()
for key in keys:
    ts = "2022-04-11 12:00" if key == "RF17" else "2022-04-12 12:00"
    ifs = ifs_ds[key].sel(time=ts, method="nearest")
    _, ax = plt.subplots(figsize=(12 * h.cm, 9 * h.cm), layout="constrained", subplot_kw=dict(projection=map_crs))

    ax.coastlines(alpha=0.5)
    ax.set_extent([-20, 25, 65, 90])
    gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=1, color='gray', alpha=0.5,
                      linestyle=':', x_inline=False, y_inline=False, rotate_labels=False)
    gl.xlocator = mpl.ticker.FixedLocator(np.arange(-180, 180, 20))
    gl.ylocator = mpl.ticker.FixedLocator(np.arange(60, 90, 5))
    gl.top_labels = False
    gl.right_labels = False

    # add seaice edge
    ci_levels = [0.8]
    ci_var = "ci" if key == "RF17" else "CI"
    cci = ax.tricontour(ifs.lon, ifs.lat, ifs[ci_var], ci_levels, transform=data_crs, linestyles="--", colors="#332288",
                        linewidths=2)

    # add high cloud cover
    ifs_cc = ifs.cloud_fraction.where(ifs.pressure_full < 60000, drop=True).sum(dim="level")
    ifs_cc = ifs_cc / 101  # divide by number of high cloud levels
    hcc = ax.tricontourf(ifs.lon, ifs.lat, ifs_cc, levels=24, transform=data_crs, cmap="Blues", alpha=1)
    cbar = plt.colorbar(hcc, label="Total high cloud fraction")

    # plot flight track
    ins = bahamas_ds[key]
    track_lons, track_lats = ins["IRS_LON"], ins["IRS_LAT"]
    ax.plot(track_lons[::10], track_lats[::10], color="k",
               label='Flight track', transform=data_crs)

    # plot dropsonde locations
    ds_dict = dropsonde_ds[key]
    for i, ds in enumerate(ds_dict.values()):
        launch_time = pd.to_datetime(ds.launch_time.to_numpy()) if key == "RF17" else pd.to_datetime(ds.time[0].to_numpy())
        x, y = ds.lon.mean().to_numpy(), ds.lat.mean().to_numpy()
        cross = ax.plot(x, y, "x", color=cbc[1], markersize=9, transform=data_crs)
        if key == "RF17":
            ax.text(x, y, f"{launch_time:%H:%M}", c="k", fontsize=9, transform=data_crs,
                    path_effects=[patheffects.withStroke(linewidth=0.5, foreground="white")])
        else:
            for i in [1]:
                ds = list(ds_dict.values())[i]
                launch_time = pd.to_datetime(ds.time[0].to_numpy())
                x, y = ds.lon.mean().to_numpy(), ds.lat.mean().to_numpy()
                ax.text(x, y, f"{launch_time:%H:%M}", color="k", fontsize=9, transform=data_crs,
                        path_effects=[patheffects.withStroke(linewidth=0.5, foreground="white")])


    coords = meta.coordinates["Kiruna"]
    ax.plot(coords[0], coords[1], ls="", marker="^", color=cbc[2], label="Kiruna", transform=data_crs)
    ax.plot([], ls="--", color="#332288", label="Sea ice edge")
    ax.plot([], ls="", marker="x", color=cbc[1], label="Dropsonde")
    ax.legend(loc=3)

    figname = f"{plot_path}/HALO-AC3_{key}_fligh_track_IFS_cloud_cover.png"
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# %% plot zoom of case study region RF 18
plt.rc("font", size=5)
fig, ax = plt.subplots(figsize=(2 * h.cm, 2.5 * h.cm),
                        subplot_kw={"projection": map_crs},
                        layout="constrained")

ax.coastlines(alpha=0.5)
ax.set_extent([-20, 22, 87, 90], crs=data_crs)
gl = ax.gridlines(crs=data_crs, draw_labels=False, linewidth=1, color='gray', alpha=0.5,
                  linestyle=':', x_inline=False, y_inline=False, rotate_labels=False)

# add high cloud cover
ifs_cc = ifs.cloud_fraction.where(ifs.pressure_full < 60000, drop=True).sum(dim="level")
ifs_cc = ifs_cc / 101
ax.tricontourf(ifs.lon, ifs.lat, ifs_cc, levels=24, transform=data_crs, cmap="Blues", alpha=1)

# plot flight track
ins = bahamas_ds[key]
track_lons, track_lats = ins["IRS_LON"], ins["IRS_LAT"]
ax.plot(track_lons[::10], track_lats[::10], color="k",
        label='Flight track', transform=data_crs)

# plot dropsonde locations - 12 April
ds_dict = dropsonde_ds["RF18"]
for i in [0, 3, 6, 10, 13]:
    ds = list(ds_dict.values())[i]
    launch_time = pd.to_datetime(ds.time[0].to_numpy())
    lon, lat = ds.lon.dropna(dim="time"), ds.lat.dropna(dim="time")
    x, y = lon[0].to_numpy(), lat[0].to_numpy()
    cross = ax.plot(x, y, "x", color="orangered", markersize=4, label="Dropsonde", transform=data_crs,
                    zorder=450)
    ax.text(x, y+.1, f"{launch_time:%H:%M}", color="k", transform=data_crs, zorder=500,
            path_effects=[patheffects.withStroke(linewidth=0.5, foreground="white")])

figname = f"{plot_path}/HALO-AC3_RF18_fligh_track_trajectories_plot_overview_zoom.png"
plt.savefig(figname, dpi=600, bbox_inches='tight')
plt.show()
plt.close()

# %% vizualise gridpoints from IFS in respect to flight track
key = "RF17"
ds = ecrad_orgs[key]["v15"]
ds1 = bahamas_ds[key]
# plot points along flight track
data_crs = ccrs.PlateCarree()
plot_crs = ccrs.NorthPolarStereo()
plot_ds = ds.sel(time=slices[key]["case"])
plot_ds1 = ds1.sel(time=slices[key]["case"])
plt.rc("font", size=8)
_, ax = plt.subplots(figsize=(7.5 * h.cm, 7.5 * h.cm), subplot_kw=dict(projection=plot_crs))
ax.set_extent([-30, 0, 88, 89], crs=data_crs)
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
gl.bottom_labels = False
# plot flight track
ax.plot(plot_ds1.IRS_LON, plot_ds1.IRS_LAT, c="k", transform=data_crs, label="Flight track")
# several timesteps
ax.scatter(plot_ds.lon, plot_ds.lat, c=cbc[0], transform=data_crs, label="IFS grid points")
# one time step
ts = "2022-04-11 11:25"
halo_plot = plot_ds1.sel(time=ts, method="nearest")
ax.plot(halo_plot.IRS_LON, halo_plot.IRS_LAT, ls="", marker="X", ms="10", transform=data_crs, label="HALO", c=cbc[2])
plot_ds = plot_ds.sel(time=ts, method="nearest")
ax.scatter(plot_ds.lon, plot_ds.lat, transform=data_crs, c=cbc[1])
ax.legend()
figname = f"{plot_path}/{key}_gridpoints.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% vizualise gridpoints from IFS selected for case study areas
key = "RF17"
extents = dict(RF17=[-15, 20, 85.5, 89], RF18=[-25, 17, 87.75, 89.85])
ds = ifs_ds_sel[key]
ds1 = bahamas_ds[key]
# plot points along flight track
data_crs = ccrs.PlateCarree()
plot_crs = ccrs.NorthPolarStereo()
plot_ds = ds.sel(time=slices[key]["above"])
plot_ds1 = ds1.sel(time=slices[key]["case"])
_, ax = plt.subplots(figsize=h.figsize_equal, subplot_kw=dict(projection=plot_crs))
ax.set_extent(extents[key], crs=data_crs)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.plot(plot_ds1.IRS_LON, plot_ds1.IRS_LAT, c="k", transform=data_crs, label="Flight track")
ax.scatter(plot_ds.lon, plot_ds.lat, s=10, c=cbc[0], transform=data_crs,
           label=f"IFS grid points n={len(plot_ds.lon)}")
plot_ds_single = plot_ds1.sel(time=slices[key]["above"].stop, method="nearest")
ax.scatter(plot_ds_single.IRS_LON, plot_ds_single.IRS_LAT, marker="*", s=50, c=cbc[1], label="Start of descent",
           transform=data_crs)
ax.legend()
figname = f"{plot_path}/{key}_case_study_gridpoints.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot lidar data for case studies
for key in keys:
    plot_ds = (lidar_ds[key]["backscatter_ratio"]
               .where((lidar_ds[key].flags == 0)
                      & (lidar_ds[key].backscatter_ratio > 1))
               .sel(time=slices[key]["above"]))
    plt.rc("font", size=8.5)
    _, ax = plt.subplots(figsize=(12.5 * h.cm, 5 * h.cm), layout="constrained")
    plot_ds.plot(x="time", y="height", cmap=cm.rainforest_r, norm=mpl.colors.LogNorm(), vmax=50,
                 cbar_kwargs=dict(label="Lidar\nbackscatter ratio", pad=0.01))
    # ax.plot(bahamas_ds[key].time, bahamas_ds[key]["IRS_ALT"] / 1000, color="grey", label="HALO altitude")
    h.set_xticks_and_xlabels(ax, slices[key]["above"].stop - slices[key]["above"].start)
    ax.set(xlabel="Time (UTC)", ylabel="Altitude (km)", ylim=(0, 12))

    figname = f"{plot_path}/HALO_AC3_HALO_{key}_lidar_backscatter_ratio_532.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot radar data for case studies
for key in keys:
    plot_ds = (radar_ds[key]["dBZg"]
               .sel(time=slices[key]["above"]))
    t_plot = ecrad_dicts[key]["v15.1"].t
    plt.rc("font", size=8.5)
    _, ax = plt.subplots(figsize=(12.5 * h.cm, 5 * h.cm), layout="constrained")
    plot_ds.plot(x="time", y="height", cmap=cm.torch_r, vmin=-45, vmax=15,
                 cbar_kwargs=dict(label="Radar reflectivity (dBZ)", pad=0.01, ticks=range(-40, 20, 10),
                                  extend="both"))
    h.set_xticks_and_xlabels(ax, slices[key]["above"].stop - slices[key]["above"].start)
    ax.set(xlabel="Time (UTC)", ylabel="Altitude (km)", ylim=(0, 12))

    figname = f"{plot_path}/HALO_AC3_HALO_{key}_radar_reflectivity.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot lidar and radar data for case studies
for key in keys:
    lidar_plot = (lidar_ds[key]["backscatter_ratio"]
                  .where((lidar_ds[key].flags == 0)
                         & (lidar_ds[key].backscatter_ratio > 1))
                  .sel(time=slices[key]["above"]))
    radar_plot = (radar_ds[key]["dBZg"]
                  .sel(time=slices[key]["above"]))
    ds = ecrad_dicts[key]["v15.1"].sel(time=slices[key]["case"])
    ifs_plot = ds["t"]
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
    # select only tropopause temperature
    tp_sel = ifs_plot == ifs_plot.min(dim="height")
    tp_height = [tp_sel.sel(time=i).height.where(tp_sel.sel(time=i), drop=True).to_numpy()[0] for i in tp_sel.time]
    plt.rc("font", size=8.5)
    _, axs = plt.subplots(2, 1, figsize=(12.5 * h.cm, 11 * h.cm), layout="constrained")
    lidar_plot.plot(x="time", y="height", cmap=cm.rainforest_r, norm=mpl.colors.LogNorm(), vmax=50,
                    cbar_kwargs=dict(label="Lidar backscatter ratio", pad=0.01, extend="both"), ax=axs[0])
    radar_plot.plot(x="time", y="height", cmap=cm.torch_r, vmin=-45, vmax=15, ax=axs[1],
                    cbar_kwargs=dict(label="Radar reflectivity (dBZ)", pad=0.01, ticks=range(-40, 20, 10),
                                     extend="both"))
    for ax in axs:
        ax.set(xlabel="", ylabel="Altitude (km)", ylim=(0, 12), xticks=[], title=f"Research flight {key[2:]}")
        ax.plot(ifs_plot.time, tp_height, color="k", linestyle="--", label="Tropopause")
    axs[0].legend(loc=1)
    axs[1].set(xlabel="Time (UTC)", title="")
    h.set_xticks_and_xlabels(axs[1], slices[key]["above"].stop - slices[key]["above"].start)

    figname = f"{plot_path}/HALO_AC3_HALO_{key}_lidar_backscatter_ratio_532_radar_reflectivity.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot PDF of BACARDI transmissivity (above cloud simulation) below cloud
od = list()
plt.rc("font", size=12)
ylims = (0, 33)
binsize = 0.01
xlabel, ylabel = "Solar transmissivity", "Probability density function"
_, axs = plt.subplots(1, 2, figsize=(17 * h.cm, 9 * h.cm), layout="constrained")
for i, key in enumerate(keys):
    ax = axs[i]
    bacardi_sel = bacardi_ds[key].sel(time=slices[key]["below"])
    bacardi_plot = bacardi_sel[f"transmissivity_above_cloud"]
    print(f"Mean solar transmissivity: {np.mean(bacardi_plot):.3f}")
    bins = np.arange(0.5, 1.0, binsize)
    # BACARDI histogram
    ax.hist(bacardi_plot, density=True, bins=bins, label="Measurement")
    ax.axvline(bacardi_plot.mean(), ls="--", lw=3, label="Mean", c="k")
    ax.set(title=f"{key.replace('RF1', 'Research flight 1')}",
           xlabel=xlabel,
           ylim=ylims,
           xticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.grid()
    od.append((key, "BACARDI", bacardi_plot.to_numpy(), -np.log(bacardi_plot.to_numpy())))

axs[0].legend()
axs[0].set(ylabel=ylabel)
figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_bacardi_transmissivity_above_cloud_PDF_below_cloud.svg"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot PDF of transmissivity (above cloud simulation) below cloud - all ice optics
plt.rc("font", size=10)
xlims, ylims = (0.45, 1), (0, 33)
binsize = 0.01
xlabel, ylabel = "Solar transmissivity", "Probability density function"
for i, v in enumerate(["v15.1", "v19.1", "v18.1"]):
    _, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 9 * h.cm), layout="constrained")
    bins = np.arange(0.5, 1.0, binsize)
    v_name = ecrad.version_names[v[:3]]
    v_name = f"{v_name}2013" if v_name == "Yi" else v_name
    for ii, key in enumerate(keys):
        ax = axs[ii]
        bacardi_sel = bacardi_ds[key].sel(time=slices[key]["below"])
        bacardi_plot = bacardi_sel[f"transmissivity_above_cloud"]
        ecrad_ds = ecrad_orgs[key][v].sel(time=slices[key]["below"])
        height_sel = ecrad_ds["aircraft_level"]
        ecrad_plot = (ecrad_ds[f"transmissivity_sw_above_cloud"]
                      .isel(half_level=height_sel)
                      .to_numpy()
                      .flatten())
        # calculate optical depth
        od.append((key, v_name, ecrad_plot, -np.log(ecrad_plot)))

        # actual plotting
        sns.histplot(bacardi_plot, label="Measurement", ax=ax, stat="density", kde=False, bins=bins)
        sns.histplot(ecrad_plot, label=f"Model {v_name}", stat="density",
                     kde=False, bins=bins, ax=ax, color=cbc[i + 1])
        ax.set(xlabel=xlabel,
               xlim=xlims,
               xticks=np.arange(0.5, 1.01, 0.1),
               ylabel="",
               ylim=ylims,
               title=f"{key.replace('RF1', 'Research flight 1')}"
               )
        ax.grid()

    axs[1].legend()
    axs[0].set(ylabel=ylabel)

    figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_bacardi_ecrad-{v_name}_transmissivity_above_cloud_PDF_below_cloud.svg"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot PDF of transmissivity (above cloud simulation) below cloud - all ice optics varcloud
plt.rc("font", size=10)
xlims, ylims = (0.45, 1), (0, 33)
binsize = 0.01
xlabel, ylabel = "Solar transmissivity", "Probability density function"
for i, v in enumerate(["v16", "v28", "v20"]):
    _, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 9 * h.cm), layout="constrained")
    bins = np.arange(0.5, 1.0, binsize)
    v_name = ecrad.version_names[v[:3]].replace(" VarCloud", "")
    v_name = f"{v_name}2013" if v_name == "Yi" else v_name
    for ii, key in enumerate(keys):
        ax = axs[ii]
        bacardi_sel = bacardi_ds[key].sel(time=slices[key]["below"])
        bacardi_plot = bacardi_sel[f"transmissivity_above_cloud"]
        ecrad_ds = ecrad_orgs[key][v].sel(time=slices[key]["below"])
        height_sel = ecrad_ds["aircraft_level"]
        ecrad_plot = (ecrad_ds[f"transmissivity_sw_above_cloud"]
                      .isel(half_level=height_sel)
                      .to_numpy()
                      .flatten())

        # calculate optical depth
        od.append((key, ecrad.version_names[v[:3]], ecrad_plot, -np.log(ecrad_plot)))

        # actual plotting
        sns.histplot(bacardi_plot, label="Measurement", ax=ax, stat="density", kde=False, bins=bins)
        sns.histplot(ecrad_plot, label=f"VarCloud {v_name}", stat="density",
                     kde=False, bins=bins, ax=ax, color=cbc[i + 1])
        ax.set(xlabel=xlabel,
               xlim=xlims,
               xticks=np.arange(0.5, 1.01, 0.1),
               ylabel="",
               ylim=ylims,
               title=f"{key.replace('RF1', 'Research flight 1')}"
               )
        ax.grid()

    axs[1].legend()
    axs[0].set(ylabel=ylabel)

    figname = (f"{plot_path}/HALO-AC3_HALO_RF17_RF18_bacardi_ecrad-"
               f"{ecrad.version_names[v[:3]]}_transmissivity_above_cloud_PDF_below_cloud.svg")
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% create dataframe for optical depth
od_df = pd.DataFrame(od, columns=["flight", "source", "solar_transmissivity", "optical_depth"])
od_means = od_df.copy(deep=True)
od_means["od"] = [np.nanmean(od_means["optical_depth"][i]) for i in range(len(od_means))]
print(od_means)

# %% plot PDF of IWC from VarCloud and IFS above cloud
plt.rc("font", size=9)
legend_labels = ["Retrieval", "Model"]
binsizes = dict(iwc=1, reice=4)
_, axs = plt.subplots(1, 2, figsize=(12 * h.cm, 5.6 * h.cm), layout="constrained")
ylims = {"iwc": (0, 0.3), "reice": (0, 0.095)}
# left panel - RF17 IWC
ax = axs[0]
binsize = binsizes["iwc"]
bins = np.arange(0, 20.1, binsize)
iwc_varcloud = ecrad_orgs["RF17"]["v16"].iwc
iwc_ifs, cc = (ifs_ds_sel["RF17"].q_ice.sel(time="2022-04-11 11:00"),
               ifs_ds_sel["RF17"].cloud_fraction.sel(time="2022-04-11 11:00"))
iwc_ifs = iwc_ifs.where(cc > 0).where(cc == 0, iwc_ifs / cc)
for i, pds in enumerate([iwc_varcloud, iwc_ifs]):
    pds = pds.to_numpy().flatten() * 1e6
    pds = pds[~np.isnan(pds)]
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i],
        color=cbc[i],
        histtype="step",
        density=True,
        lw=2,
    )
    print(f"{legend_labels[i]} n={len(pds)}")
ax.grid()
ax.set(ylabel=f"Probability density function",
       xlabel=f"Ice water content ({h.plot_units['iwc']})",
       ylim=ylims["iwc"],
       xticks=range(0, 21, 5),
       title="Research flight 17")

# right panel - RF18 IWC
ax = axs[1]
iwc_varcloud = ecrad_orgs["RF18"]["v16"].iwc
iwc_ifs, cc = (ifs_ds_sel["RF18"].q_ice.sel(time="2022-04-12 11:00"),
               ifs_ds_sel["RF18"].cloud_fraction.sel(time="2022-04-12 11:00"))
iwc_ifs = iwc_ifs.where(cc > 0).where(cc == 0, iwc_ifs / cc)
for i, pds in enumerate([iwc_varcloud, iwc_ifs]):
    pds = pds.to_numpy().flatten() * 1e6
    pds = pds[~np.isnan(pds)]
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i],
        color=cbc[i],
        histtype="step",
        density=True,
        lw=2,
    )
    print(f"{legend_labels[i]} n={len(pds)}")
ax.legend()
ax.grid()
ax.set(ylabel=f"",
       xlabel=f"Ice water content ({h.plot_units['iwc']})",
       ylim=ylims["iwc"],
       xticks=range(0, 21, 5),
       title="Research flight 18")

figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_IFS_iwc_pdf_case_studies.svg"
plt.savefig(figname, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# %% plot PDF of reice from VarCloud and IFS above cloud
plt.rc("font", size=9)
legend_labels = ["Retrieval", "Model"]
binsizes = dict(iwc=1, reice=4)
_, axs = plt.subplots(1, 2, figsize=(12 * h.cm, 5.2 * h.cm), layout="constrained")
ylims = {"iwc": (0, 0.3), "reice": (0, 0.095)}
# left panel - RF17 reice
ax = axs[0]
binsize = binsizes["reice"]
bins = np.arange(0, 100.1, binsize)
reice_varcloud = ecrad_orgs["RF17"]["v16"].re_ice.to_numpy().flatten()
reice_ifs, cc = (ifs_ds_sel["RF17"].sel(time="2022-04-11 12:00", drop=True),
                 ifs_ds_sel["RF17"].cloud_fraction.sel(time="2022-04-11 11:00", drop=True))
reice_ifs = ecrad.apply_ice_effective_radius(reice_ifs)
reice_ifs = (reice_ifs
             .re_ice
             .where(cc > 0)
             .where(~np.isclose(reice_ifs.re_ice, 5.19616e-05))
             .to_numpy().flatten())
for i, pds in enumerate([reice_varcloud, reice_ifs]):
    pds = pds[~np.isnan(pds)] * 1e6
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i],
        color=cbc[i],
        histtype="step",
        density=True,
        lw=2,
    )
    print(f"{legend_labels[i]} n={len(pds)}")
ax.grid()
ax.set(ylabel="Probability density function",
       xlabel=f"Ice effective radius ({h.plot_units['re_ice']})",
       ylim=ylims["reice"],
       xticks=range(0, 101, 20))

# right panel - RF18 reice
ax = axs[1]
reice_varcloud = ecrad_orgs["RF18"]["v16"].re_ice.to_numpy().flatten()
reice_ifs, cc = (ifs_ds_sel["RF18"].sel(time="2022-04-12 12:00", drop=True),
                 ifs_ds_sel["RF18"].cloud_fraction.sel(time="2022-04-12 11:00", drop=True))
reice_ifs = ecrad.apply_ice_effective_radius(reice_ifs)
reice_ifs = (reice_ifs
             .re_ice
             .where(cc > 0)
             .where(~np.isclose(reice_ifs.re_ice, 5.19616e-05))
             .to_numpy().flatten())
for i, pds in enumerate([reice_varcloud, reice_ifs]):
    pds = pds[~np.isnan(pds)] * 1e6
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i],
        color=cbc[i],
        histtype="step",
        density=True,
        lw=2,
    )
    print(f"{legend_labels[i]} n={len(pds)}")
ax.grid()
ax.set(ylabel="",
       xlabel=f"Ice effective radius ({h.plot_units['re_ice']})",
       ylim=ylims["reice"],
       xticks=range(0, 101, 20))

figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_IFS_reice_pdf_case_studies.svg"
plt.savefig(figname, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# %% plot PDF of IWC from IFS above cloud for 11 and 12 UTC
plt.rc("font", size=9)
legend_labels = ["11 UTC", "12 UTC"]
binsizes = dict(iwc=1, reice=4)
_, axs = plt.subplots(1, 2, figsize=(12 * h.cm, 5.6 * h.cm), layout="constrained")
ylims = {"iwc": (0, 0.3), "reice": (0, 0.095)}
# left panel - RF17 IWC
ax = axs[0]
binsize = binsizes["iwc"]
bins = np.arange(0, 20.1, binsize)
iwc_ifs_ls = list()
for t in ["2022-04-11 11:00", "2022-04-11 12:00"]:
    iwc_ifs, cc = ifs_ds_sel["RF17"].q_ice.sel(time=t), ifs_ds_sel["RF17"].cloud_fraction.sel(time=t)
    iwc_ifs_ls.append(iwc_ifs.where(cc > 0).where(cc == 0, iwc_ifs / cc))

for i, pds in enumerate(iwc_ifs_ls):
    pds = pds.to_numpy().flatten() * 1e6
    pds = pds[~np.isnan(pds)]
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i],
        color=cbc[i],
        histtype="step",
        density=True,
        lw=2,
    )
    print(f"{legend_labels[i]} n={len(pds)}")
ax.grid()
ax.set(ylabel=f"Probability density function",
       xlabel=f"Ice water content ({h.plot_units['iwc']})",
       ylim=ylims["iwc"],
       xticks=range(0, 21, 5),
       title="Research flight 17")

# right panel - RF18 IWC
ax = axs[1]
iwc_ifs_ls = list()
for t in ["2022-04-12 11:00", "2022-04-12 12:00"]:
    iwc_ifs, cc = ifs_ds_sel["RF18"].q_ice.sel(time=t), ifs_ds_sel["RF18"].cloud_fraction.sel(time=t)
    iwc_ifs_ls.append(iwc_ifs.where(cc > 0).where(cc == 0, iwc_ifs / cc))

for i, pds in enumerate(iwc_ifs_ls):
    pds = pds.to_numpy().flatten() * 1e6
    pds = pds[~np.isnan(pds)]
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i],
        color=cbc[i],
        histtype="step",
        density=True,
        lw=2,
    )
    print(f"{legend_labels[i]} n={len(pds)}")
ax.legend()
ax.grid()
ax.set(ylabel=f"",
       xlabel=f"Ice water content ({h.plot_units['iwc']})",
       ylim=ylims["iwc"],
       xticks=range(0, 21, 5),
       title="Research flight 18")

figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_IFS_iwc_11_vs_12_pdf_case_studies.svg"
plt.savefig(figname, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# %% plot PDF of reice from IFS and from IFS using VarCloud IWC input above cloud
plt.rc("font", size=9)
legend_labels = ["Retrieval", "Model + Retrieval"]
binsizes = dict(iwc=1, reice=4)
_, axs = plt.subplots(1, 2, figsize=(12 * h.cm, 5.2 * h.cm), layout="constrained")
ylims = {"iwc": (0, 0.3), "reice": (0, 0.095)}
# left panel - RF17 reice
ax = axs[0]
binsize = binsizes["reice"]
bins = np.arange(0, 100.1, binsize)
reice_varcloud = ecrad_orgs["RF17"]["v16"].re_ice.to_numpy().flatten()
# overwrite ciwc and cswc as these variables are used in the calculation of re_ice
ds = ecrad_orgs["RF17"]["v16"].copy(deep=True)
ds["ciwc"] = ds.q_ice
ds["cswc"] = xr.full_like(ds.cswc, 0)  # set cloud snow water content to 0
# calculate re_ice from VarCloud IWC
reice_ifs = ecrad.apply_ice_effective_radius(ds)
reice_ifs = (reice_ifs
             .re_ice
             .where(ds.cloud_fraction > 0)
             .where(~np.isclose(reice_ifs.re_ice, 5.19616e-05))
             .to_numpy().flatten())
for i, pds in enumerate([reice_varcloud, reice_ifs]):
    pds = pds[~np.isnan(pds)] * 1e6
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i],
        color=cbc[i],
        histtype="step",
        density=True,
        lw=2,
    )
    print(f"{legend_labels[i]} n={len(pds)}")
ax.grid()
ax.set(ylabel="Probability density function",
       xlabel=f"Ice effective radius ({h.plot_units['re_ice']})",
       ylim=ylims["reice"])

# right panel - RF18 reice
ax = axs[1]
reice_varcloud = ecrad_orgs["RF18"]["v16"].re_ice.to_numpy().flatten()
# overwrite ciwc and cswc as these variables are used in the calculation of re_ice
ds = ecrad_orgs["RF18"]["v16"].copy(deep=True)
ds["ciwc"] = ds.q_ice
ds["cswc"] = xr.full_like(ds.cswc, 0)  # set cloud snow water content to 0
# calculate re_ice from VarCloud IWC
reice_ifs = ecrad.apply_ice_effective_radius(ds)
reice_ifs = (reice_ifs
             .re_ice
             .where(ds.cloud_fraction > 0)
             .where(~np.isclose(reice_ifs.re_ice, 5.19616e-05))
             .to_numpy().flatten())
for i, pds in enumerate([reice_varcloud, reice_ifs]):
    pds = pds[~np.isnan(pds)] * 1e6
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i],
        color=cbc[i],
        histtype="step",
        density=True,
        lw=2,
    )
    print(f"{legend_labels[i]} n={len(pds)}")
ax.legend()
ax.grid()
ax.set(ylabel="",
       xlabel=f"Ice effective radius ({h.plot_units['re_ice']})",
       ylim=ylims["reice"])

figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_IFS_reice_varcloud_pdf_case_studies.svg"
plt.savefig(figname, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# %% plot IFS cloud fraction lidar/mask comparison
plt.rc("font", size=6.5)
_, axs = plt.subplots(2, 1, figsize=(16 * h.cm, 9 * h.cm), layout="constrained")
for i, key in enumerate(keys):
    ax = axs[i]
    ds = ecrad_dicts[key]["v15"].sel(time=slices[key]["case"])
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
    ifs_plot.plot(x="time", cmap=cm.sapphire, ax=ax,
                  cbar_kwargs=dict(label=f"IFS {h.cbarlabels['cloud_fraction']}",
                                   pad=0.01))
    halo_plot.plot.contour(x="time", levels=[0.9], colors=cbc[1], ax=ax, linewidths=2)
    bahamas_plot = bahamas_ds[key].IRS_ALT.sel(time=slices[key]["case"]) / 1000
    bahamas_plot.plot(x="time", ax=ax, label="HALO altitude", color=cbc[-2], lw=2)
    ax.plot([], color=cbc[1], label="Radar & Lidar Mask", lw=2)
    h.set_xticks_and_xlabels(ax, time_extend)
    ax.set(xlabel="Time (UTC)", ylabel="Height (km)")
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=0, ha="center")
    # add line for turning point/start of second circle
    if key == "RF17":
        ax.axvline(pd.Timestamp("2022-04-11 11:12:26"), 0, 1, ls="--", lw=3, label="Turning point")
        ax.set_xlabel("")
    else:
        ax.axvline(slices[key]["above"].start, 0, 1, ls="--", lw=3, label="Start of pentagon")
        ax.axvline(slices[key]["below"].start, 0, 1, ls="--", lw=3, color=cbc[0])

    ax.legend(ncols=2)

figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_IFS_cloud_fraction_radar_lidar_mask.svg"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot IFS cloud fraction lidar/mask comparison for VarCloud below cloud simulation
key = "RF18"
plt.rc("font", size=6.5)
_, ax = plt.subplots(figsize=(16 * h.cm, 4.5 * h.cm), layout="constrained")
ds = ecrad_dicts[key]["v15.1"].sel(time=slices[key]["case"])
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
# reorder time axis to show how the input to the below cloud varcloud simulation works
sel_time = slice(pd.to_datetime("2022-04-12 11:04"), pd.to_datetime("2022-04-12 11:24"))
sim_time = pd.date_range("2022-04-12 11:41", "2022-04-12 12:14", freq="1s")
start_time_str = str(halo_plot.time[0].astype('datetime64[s]').to_numpy())
end_time_str = str(halo_plot.time[-1].astype('datetime64[s]').to_numpy())
new_index = pd.date_range(start_time_str, end_time_str, freq="1s")
halo_plot = halo_plot.reindex(time=new_index, method="bfill")
halo_plot = halo_plot.sel(time=sel_time)  # select the above cloud time
new_index = pd.date_range(str(halo_plot.time[0].astype('datetime64[s]').to_numpy()),
                          str(halo_plot.time[-1].astype('datetime64[s]').to_numpy()),
                          periods=len(sim_time))
# this "stretches" the varcloud data over the time range of the simulation
halo_plot = halo_plot.reindex(time=new_index, method="nearest")
# replace time axis with the simulation time
halo_plot["time"] = sim_time
time_extend = pd.to_timedelta((ifs_plot.time[-1] - ifs_plot.time[0]).to_numpy())

# plot IFS cloud cover prediction and Radar lidar mask
ifs_plot.plot(x="time", cmap=cm.sapphire,
              cbar_kwargs=dict(label=f"IFS {h.cbarlabels['cloud_fraction']}",
                               pad=0.01)
              , ax=ax)
halo_plot.plot.contour(x="time", levels=[0.9], colors=cbc[1], ax=ax, linewidths=2)
bahamas_plot = bahamas_ds[key].IRS_ALT.sel(time=slices[key]["case"]) / 1000
bahamas_plot.plot(x="time", ax=ax, label="HALO altitude", color=cbc[-2], lw=2)
ax.plot([], color=cbc[1], label="Radar & Lidar Mask", lw=2)
ax.legend()
h.set_xticks_and_xlabels(ax, time_extend)
ax.set(xlabel="Time (UTC)", ylabel="Height (km)")
ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=0, ha="center")

figname = f"{plot_path}/HALO-AC3_HALO_RF18_IFS_cloud_fraction_radar_lidar_mask_varcloud_below_cloud.svg"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot sketch of PSD
psd_x_vals = [50, 10e1, 5e2, 10e2, 5e3, 10e3, 10.1e3]
psd_y_vals = [10e-3, 10e-2, 5e-1, 10e-3, 10e-4, 10e-5, 10e-6]
with plt.xkcd():
    plt.rc("font", size=6)
    _, ax = plt.subplots(figsize=(3.5 * h.cm, 2.5 * h.cm), layout="constrained")
    ax.spines[['top', 'right']].set_visible(False)
    ax.plot(psd_x_vals, psd_y_vals, lw=3)
    ax.set(xscale="log", yscale="log",
           xticks=[],
           yticks=[],
           xlabel="Diameter ($\mu$m)",
           ylabel="dN/dlogD (cm$^{-3}$)")
plt.savefig(f"{plot_path}/psd_sketch.svg")
plt.show()
plt.close()

# %% plot sketch of correlation
cor_x = [10, 20, 30, 40, 50, 60, 70, 80, 90]
cor_y = [0.9, 0.84, 0.83, 0.8, 0.75, 0.7, 0.69, 0.67, 0.65]
with plt.xkcd():
    plt.rc("font", size=8)
    _, ax = plt.subplots(figsize=(3.5 * h.cm, 2.5 * h.cm), layout="constrained")
    ax.spines[['top', 'right']].set_visible(False)
    ax.plot(cor_x, cor_y, lw=3)
    ax.set(xlabel="re$_{ice}$ ($\mu$m)",
           ylabel="$\omega$",
           xticks=[],
           yticks=[])
plt.savefig(f"{plot_path}/cor_sketch.svg")
plt.show()
plt.close()


# %% plot simple plot of solar transmissivity against optical depth
st = np.linspace(0.01, 0.99, 100)
od_ = -np.log(st)
st_m = [0.88, 0.58]
od_m = -np.log(st_m)
_, ax = plt.subplots()
ax.plot(st, od_)
ax.plot(st_m, od_m, "or", label="Mean")
ax.vlines(st_m, 0, od_m)
ax.hlines(od_m, 0, st_m)
for key in keys:
    od_plot = od_df[(od_df["flight"] == key) & (od_df["source"] == "BACARDI")]
    ax.plot(od_plot["solar_transmissivity"].to_numpy()[0],
            od_plot["optical_depth"].to_numpy()[0],
            label=key)
ax.set(xlim=(0), ylim=(0),
       xlabel= "Solar transmissivity", ylabel="Optical depth")
ax.legend()
plt.show()
plt.close()
