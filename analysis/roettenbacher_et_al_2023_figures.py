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
from pylim import ecrad
import ac3airborne
from ac3airborne.tools import flightphase
import cartopy.crs as ccrs
import cmasher as cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import os
from matplotlib import gridspec, patheffects
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from metpy.calc import relative_humidity_from_specific_humidity
from metpy.units import units as u
from scipy.stats import median_abs_deviation
from sklearn.neighbors import BallTree
from tqdm import tqdm

h.set_cb_friendly_colors("petroff_6")
cbc = h.get_cb_friendly_colors("petroff_6")

# %% set paths
campaign = "halo-ac3"
plot_path = "C:/Users/Johannes/Documents/Doktor/manuscripts/_arctic_cirrus/figures"
trajectory_path = f"{h.get_path('trajectories', campaign=campaign)}/selection_CC_and_altitude"
keys = ["RF17", "RF18"]
ecrad_versions = ["v13.2", "v15", "v15.1", "v16", "v17", "v18", "v18.1", "v19", "v19.1", "v20", "v21", "v28", "v29",
                  "v30.1", "v31.1", "v32.1", "v33", "v34", "v35", "v36", "v37", "v38"]

# %% read in data
(
    bahamas_ds,
    bacardi_ds,
    bacardi_ds_res,
    ecrad_dicts,
    varcloud_ds,
    above_clouds,
    below_clouds,
    slices,
    ecrad_orgs,
    ifs_ds,
    ifs_ds_sel,
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
    dropsonde_path = f"{dropsonde_path}/Level_1" if key == "RF17" else f"{dropsonde_path}/Level_2"

    # filenames
    bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{key}_v1_JR.nc"
    bacardi_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1_JR.nc"
    libradtran_bb_solar_si = f"HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_solar_si_{date}_{key}.nc"
    libradtran_bb_thermal_si = f"HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_thermal_si_{date}_{key}.nc"
    ifs_file = f"ifs_{date}_00_ml_O1280_processed.nc"
    varcloud_file = [f for f in os.listdir(varcloud_path) if f.endswith(".nc")][0]
    dropsonde_files = [f for f in os.listdir(dropsonde_path) if f.endswith(".nc")]

    # read in aircraft data
    bahamas_ds[key] = xr.open_dataset(f"{bahamas_path}/{bahamas_file}")
    bacardi = xr.open_dataset(f"{bacardi_path}/{bacardi_file}")

    # read in results of albedo experiment
    albedo_dfs[key] = pd.read_csv(f"{plot_path}/{flight}_boxplot_data.csv")

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
    bacardi["ecrad_fdw"] = ecrad_dict["v15.1"].flux_dn_sw_clear.interp(time=bacardi.time,
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

# %% print time between above and below cloud section
print(f"Time between above and below cloud section")
for key in keys:
    below = below_clouds[key]
    above = above_clouds[key]
    print(f"{key}: {below['start'] - above['end']} HH:MM:SS")
    print(f"{key}: Case study duration: {above['start']} - {below['end']}")

# %% print start location and most northerly point of above cloud leg
for key in keys:
    tmp = bahamas_ds[key].sel(time=slices[key]["above"])[["IRS_LAT", "IRS_LON"]]
    print(f"Start position of above cloud leg for {key}:\n"
          f"Latitude, Longitude: {tmp.IRS_LAT.isel(time=0):.2f}°N, {tmp.IRS_LON.isel(time=0):.2f}°W\n")
    if key == "RF18":
        max_lat_i = np.argmax(tmp.IRS_LAT.to_numpy())
        print(f"Most northerly location of pentagon:\n"
              f"{tmp.IRS_LAT.isel(time=max_lat_i):.2f}°N, {tmp.IRS_LON.isel(time=max_lat_i):.2f}°W")

# %% print range/change of solar zenith angle during case study
for key in keys:
    tmp = bacardi_ds[key].sel(time=slices[key]["case"])
    print(f"Range of solar zenith angle during case study of {key}: {tmp.sza.min():.2f} - {tmp.sza.max():.2f}")
    tmp = bacardi_ds[key].sel(time=slices[key]["above"])
    print(f"Change of solar zenith angle during above cloud section of {key}: {tmp.sza[0]:.2f} - {tmp.sza[-1]:.2f}")


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
bacardi_vars = ["reflectivity_solar", "F_up_solar", "F_down_solar", "F_down_solar_diff", "transmissivity_above_cloud"]
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
                ("v1", "BACARDI", key, var, section, bacardi_min, bacardi_max, bacardi_spread, bacardi_mean, bacardi_std))

# %% calculate statistics from ecRad
ecrad_stats = list()
ecrad_vars = ["reflectivity_sw", "flux_up_sw", "flux_dn_direct_sw", "flux_dn_sw", "transmissivity_sw_above_cloud"]
for version in ["v13.2", "v15.1", "v16", "v18.1", "v19.1", "v20", "v28", "v30.1", "v31.1", "v32.1"]:
    v_name = ecrad.version_names[version[0:3]]
    for key in keys:
        bds = bacardi_ds[key]
        height_sel = ecrad_dicts[key][version]["aircraft_level"]
        ecrad_ds = ecrad_dicts[key][version].isel(half_level=height_sel)
        for evar in ecrad_vars:
            for section in ["above", "below"]:
                time_sel = slices[key][section]
                eds = ecrad_ds[evar].sel(time=time_sel)
                try:
                    ecrad_min = np.min(eds).to_numpy()
                    ecrad_max = np.max(eds).to_numpy()
                    ecrad_spread = ecrad_max - ecrad_min
                    ecrad_mean = eds.mean().to_numpy()
                    ecrad_std = eds.std().to_numpy()
                    ecrad_stats.append(
                        (version, v_name, key, evar, section, ecrad_min, ecrad_max, ecrad_spread, ecrad_mean, ecrad_std)
                    )
                except ValueError:
                    ecrad_stats.append(
                        (version, v_name, key, evar, section, np.nan, np.nan, np.nan, np.nan, np.nan)
                    )

# %% convert statistics to dataframe
columns = ["version", "source", "key", "variable", "section", "min", "max", "spread", "mean", "std"]
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
        v_mean = df[["version", "source", "mean"]][selection]
        print(f"{key}: Mean {var} above cloud:\n{v_mean}")

# %% print mean solar transmissivity for below cloud section
for key in keys:
    for var in ["transmissivity_above_cloud", "transmissivity_sw_above_cloud"]:
        selection = (df["variable"] == var) & (df["key"] == key) & (df["section"] == "below")
        v_mean = df[["version", "source", "mean", "std"]][selection]
        print(f"{key}: Mean {var} below cloud:\n{v_mean}")

# %% print mean flux_dn_sw for above/below cloud section
for key in keys:
    for var in ["flux_dn_sw", "F_down_solar_diff", "transmissivity_above_cloud"]:
        selection = (df["variable"] == var) & (df["key"] == key) & (df["section"] == "below")
        v_mean = df[["version", "source", "mean"]][selection]
        v_std = df[["version", "source", "std"]][selection]
        print(f"{key}: Mean {var} below cloud:\n{v_mean}")
        print(f"{key}: Standard deviation of {var} for below cloud:\n{v_std}")
        selection = (df["variable"] == var) & (df["key"] == key) & (df["section"] == "above")
        v_mean = df[["version", "source", "mean"]][selection]
        print(f"{key}: Mean {var} above cloud:\n{v_mean}")

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
ds = ecrad_orgs["RF17"]["v15"].sel(column=0, drop=True)
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

# %% vizualise gridpoints from IFS in respect to flight track
key = "RF17"
ds = ecrad_orgs[key]["v15"]
ds1 = bahamas_ds[key]
# plot points along flight track
data_crs = ccrs.PlateCarree()
plot_crs = ccrs.NorthPolarStereo()
plot_ds = ds.sel(time=slices[key]["case"])
plot_ds1 = ds1.sel(time=slices[key]["case"])
_, ax = plt.subplots(figsize=h.figsize_equal, subplot_kw=dict(projection=plot_crs))
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

# %% calculate distance between each 10 grid cells along flight track
ds.distance.plot(x="time")
plt.show()
plt.close()

# %% get maximum distance along time
max_dist = ds.distance.sel(column=9).max()
min_dist = ds.distance.sel(column=9).min()
# calculate minimum and maximum circle area
area_max = np.pi * max_dist ** 2
area_min = np.pi * min_dist ** 2
halo_speed = 200  # m/s
print(f"Range of circle area covered by the 9 gridpoints surrounding the closest gridpoint to the HALO flight track:\n"
      f"{area_min:.2f} - {area_max:.2f} km^2\n"
      f"Radius: {min_dist:.2f} - {max_dist:.2f} km\n"
      f"Time HALO needs to cover these circles: "
      f"{min_dist * 2 * 1000 / halo_speed / 60:.2f} - {max_dist * 2 * 1000 / halo_speed / 60:.2f} minutes")

# %% plot IWC before and after dividing by cloud fraction
plot_ds = ecrad_dicts["RF17"]["v15"]
plot_ds["iwc_new"] = plot_ds["iwc"].where(plot_ds["cloud_fraction"] == 0,
                                          plot_ds["iwc"] / plot_ds["cloud_fraction"]) * 1e6
plot_ds = plot_ds["iwc_new"].sel(time=slices["RF17"]["below"])
plot_ds = plot_ds.to_numpy().flatten()

plt.hist(plot_ds, bins=np.arange(0, 10, 0.25), density=True)
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

# %% plot BACARDI six panel plot with above and below cloud measurements and transmissivity - solar
plt.rc("font", size=8)
xlims = [(0, 240), (0, 320)]
ylim_transmissivity = (0.45, 1)
ylim_irradiance = [(100, 279), (80, 260)]
label_xy = (0.03, 0.9)
box_xy = (0.98, 0.9)
_, axs = plt.subplots(3, 2, figsize=(17 * h.cm, 15 * h.cm))

# upper left panel - RF17 BACARDI F above cloud
ax = axs[0, 0]
plot_ds = bacardi_ds["RF17"].sel(time=slices["RF17"]["above"])
plot_ds["distance"] = bahamas_ds["RF17"]["distance"].sel(time=slices["RF17"]["above"])
# set first distance to 0
plot_ds["distance"][0] = 0
# sum up distances to generate a distance axis and convert to km
plot_ds["cum_distance"] = plot_ds["distance"].cumsum() / 1000
# bacardi measurements
for var in ["F_down_solar", "F_up_solar"]:
    ax.plot(plot_ds.cum_distance, plot_ds[var], label=f"{h.bacardi_labels[var]}")
ax.legend(loc=4, fontsize=9)
ax.grid()
ax.text(box_xy[0], box_xy[1], "Above cloud", ha="right",
        transform=ax.transAxes, bbox=dict(boxstyle="Round", fc="white"))
ax.set(title="RF 17 - 11 April 2022",
       ylabel=f"Solar irradiance ({h.plot_units['flux_dn_sw']})",
       ylim=ylim_irradiance[0],
       xlim=xlims[0])

# middle left panel - RF17 BACARDI F below_cloud
ax = axs[1, 0]
plot_ds = bacardi_ds["RF17"].sel(time=slices["RF17"]["below"])
plot_ds["distance"] = bahamas_ds["RF17"]["distance"].sel(time=slices["RF17"]["below"])
# set first distance to 0
plot_ds["distance"][0] = 0
# sum up distances to generate a distance axis and convert to km, flip the distance to show travel in other direction
cum_distance = np.flip(plot_ds["distance"].cumsum().to_numpy() / 1000)
# bacardi measurements
for var in ["F_down_solar", "F_up_solar"]:
    ax.plot(cum_distance, plot_ds[var], label=f"{h.bacardi_labels[var]}")
ax.grid()
ax.text(box_xy[0], box_xy[1], "Below cloud", ha="right",
        transform=ax.transAxes, bbox=dict(boxstyle="Round", fc="white"))
ax.set(ylabel=f"Solar irradiance ({h.plot_units['flux_dn_sw']})",
       ylim=ylim_irradiance[1],
       xlim=xlims[0])

# lower left panel - RF17 transmissivity
ax = axs[2, 0]
# ax.axhline(y=1, color="k")
ax.plot(cum_distance, plot_ds["transmissivity_above_cloud"], label="Solar transmissivity", color=cbc[3])
ax.grid()
ax.text(box_xy[0], box_xy[1], "Below cloud", ha="right",
        transform=ax.transAxes, bbox=dict(boxstyle="Round", fc="white"))
ax.set(ylabel="Solar transmissivity",
       xlabel="Distance (km)",
       ylim=ylim_transmissivity,
       xlim=xlims[0])

# upper right panel - RF18 BACARDI F above cloud
ax = axs[0, 1]
plot_ds = bacardi_ds["RF18"].sel(time=slices["RF18"]["above"])
plot_ds["distance"] = bahamas_ds["RF18"]["distance"].sel(time=slices["RF18"]["above"])
# set first distance to 0
plot_ds["distance"][0] = 0
# sum up distances to generate a distance axis and convert to km
plot_ds["cum_distance"] = plot_ds["distance"].cumsum() / 1000
# bacardi measurements
for var in ["F_down_solar", "F_up_solar"]:
    ax.plot(plot_ds.cum_distance, plot_ds[var], label=f"{h.bacardi_labels[var]}")
ax.grid()
ax.text(box_xy[0], box_xy[1], "Above cloud", ha="right",
        transform=ax.transAxes, bbox=dict(boxstyle="Round", fc="white"))
ax.set(title="RF 18 - 12 April 2022",
       ylim=ylim_irradiance[0],
       xlim=xlims[1])

# middle right panel - RF18 BACARDI F below cloud
ax = axs[1, 1]
plot_ds = bacardi_ds["RF18"].sel(time=slices["RF18"]["below"])
plot_ds["distance"] = bahamas_ds["RF18"]["distance"].sel(time=slices["RF18"]["below"])
# set first distance to 0
plot_ds["distance"][0] = 0
# sum up distances to generate a distance axis and convert to km, flip the distance to show travel in other direction
cum_distance = np.flip(plot_ds["distance"].cumsum().to_numpy() / 1000)
# bacardi measurements
for var in ["F_down_solar", "F_up_solar"]:
    ax.plot(cum_distance, plot_ds[var], label=f"{h.bacardi_labels[var]}")
ax.grid()
ax.text(box_xy[0], box_xy[1], "Below cloud", ha="right",
        transform=ax.transAxes, bbox=dict(boxstyle="Round", fc="white"))
ax.set(ylim=ylim_irradiance[1],
       xlim=xlims[1])

# lower right panel - RF18 transmissivity
ax = axs[2, 1]
# ax.axhline(y=1, color="k")
ax.plot(cum_distance, plot_ds["transmissivity_above_cloud"], label="Solar transmissivity", color=cbc[3])
ax.grid()
# ax.text(label_xy[0], label_xy[1], "(f)", transform=ax.transAxes)
ax.text(box_xy[0], box_xy[1], "Below cloud", ha="right",
        transform=ax.transAxes, bbox=dict(boxstyle="Round", fc="white"))
ax.set(xlabel="Distance (km)",
       ylim=ylim_transmissivity,
       xlim=xlims[1])

# set a-f labels
for ax, label in zip(axs.flatten(), ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]):
    ax.text(label_xy[0], label_xy[1], label, transform=ax.transAxes, fontsize=8)

plt.tight_layout()
figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_BACARDI_case_studies_6panel.pdf"
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
plt.rc("font", size=7)
fig, axs = plt.subplots(2, 1, figsize=(16 * h.cm, 9 * h.cm), layout="constrained")
for i, key in enumerate(keys):
    ax = axs[i]
    ds = ecrad_dicts[key]["v15"].sel(time=slices[key]["case"])
    ifs_plot = ds[["cloud_fraction", "iwc"]]
    bahamas_plot = bahamas_ds[key].IRS_ALT.sel(time=slices[key]["case"]) / 1000
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
    ifs_plot = ifs_plot.where(ifs_plot.cloud_fraction > 0)
    halo_plot = varcloud_ds[key].sel(time=slices[key]["case"]).Varcloud_Input_Mask
    halo_plot = halo_plot.assign_coords(height=halo_plot.height / 1000).sortby("height")
    time_extend = pd.to_timedelta((ifs_plot.time[-1] - ifs_plot.time[0]).to_numpy())

    # plot IFS cloud cover prediction and Radar lidar mask
    pcm = ifs_plot.cloud_fraction.plot(x="time", cmap=cm.sapphire, ax=ax, add_colorbar=False)
    halo_plot.plot.contour(x="time", levels=[0.9], colors=cbc[1], ax=ax, linewidths=2)
    ax.plot([], color=cbc[1], label="Radar & Lidar Mask", lw=2)
    bahamas_plot.plot(x="time", lw=2, color=cbc[-2], label="HALO altitude", ax=ax)
    ax.axvline(x=pd.to_datetime(f"{bahamas_plot.time.dt.date[0]:%Y-%m-%d} 11:30"),
               label="New IFS timestep", lw=2, ls="--")
    h.set_xticks_and_xlabels(ax, time_extend)
    ax.set(xlabel="Time (UTC)", ylabel="Height (km)")
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=0, ha="center")

# place colorbar for both flights
fig.colorbar(pcm, ax=axs[:2], label=f"IFS {h.cbarlabels['cloud_fraction'].lower()}", pad=0.001)
axs[0].legend()
axs[0].set_xlabel("")
axs[0].text(0.03, 0.88, "(a) RF 17", transform=axs[0].transAxes, bbox=dict(boxstyle="Round", fc="white"))
axs[1].text(0.03, 0.88, "(b) RF 18", transform=axs[1].transAxes, bbox=dict(boxstyle="Round", fc="white"))

figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_IFS_cloud_fraction_radar_lidar_mask.pdf"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot flight track together with trajectories and high cloud cover
cmap = mpl.colormaps["tab20b_r"]([20, 20, 0, 3, 4, 7, 8, 11, 12, 15, 16, 19])
cmap[:2] = mpl.colormaps["tab20c"]([7, 4])
cmap = mpl.colors.ListedColormap(cmap)
plt_sett = {
    'label': 'Time relative to release (h)',
    'norm': plt.Normalize(-72, 0),
    'ylim': [-72, 0],
    'cmap_sel': cmap,
    'cmap_ticks': np.arange(-72, 0.1, 12),
    'shrink': 1
}
data_crs = ccrs.PlateCarree()
map_crs = ccrs.NorthPolarStereo()

plt.rc("font", size=8)
fig, axs = plt.subplots(1, 2,
                        figsize=(17 * h.cm, 9 * h.cm),
                        subplot_kw={"projection": map_crs},
                        layout="constrained")

# plot trajectory map 11 April in first row and first column
ax = axs[0]
ax.coastlines(alpha=0.5)
xlim = (-1200000, 1200000)
ylim = (-2500000, 50000)
ax.set(title="(a) RF 17")
# ax.set_extent([-30, 40, 65, 90], crs=map_crs)
ax.set_extent([xlim[0], xlim[1], ylim[0], ylim[1]], crs=map_crs)
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
cp.clabel(fontsize=5, inline=1, inline_spacing=4, fmt='%i', rightside_up=True, use_clabeltext=True)

# add seaice edge
ci_levels = [0.8]
cci = ax.tricontour(ifs.lon, ifs.lat, ifs.ci, ci_levels, transform=data_crs, linestyles="--", colors="#332288",
                    linewidths=1)

# add high cloud cover according to IFS
ifs_cc = ifs.hcc
hcc = ax.tricontourf(ifs.lon, ifs.lat, ifs_cc, levels=np.arange(0.2, 1.01, 0.1), transform=data_crs,
                     cmap="Blues", alpha=0.5)
# ax.tricontour(ifs.lon, ifs.lat, ifs_cc, levels=[0.2], linestyles=":", colors="blue", transform=data_crs,
#               alpha=1, linewidths=0.5)

# add colorbar
axins1 = inset_axes(
    ax,
    width="3%",  # width: 50% of parent_bbox width
    height="25%",  # height: 5%
    loc="upper left",
)
plt.colorbar(hcc, cax=axins1, orientation="vertical", ticks=[0.2, 0.4, 0.6, 0.8, 1])
axins1.yaxis.set_ticks_position("right")
axins1.set_yticklabels([0.2, 0.4, 0.6, 0.8, 1], size=6,
                      path_effects=[patheffects.withStroke(linewidth=0.5, foreground="white")])

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

# plot flight track - 11 April
ins = bahamas_ds["RF17"]
track_lons, track_lats = ins["IRS_LON"], ins["IRS_LAT"]
ax.plot(track_lons[::1000], track_lats[::1000], c="k",
        zorder=400, transform=data_crs)

# highlight case study region
ins_hl = ins.sel(time=slices["RF17"]["above"])
ax.plot(ins_hl.IRS_LON[::100], ins_hl.IRS_LAT[::100], c=cbc[1],
        zorder=400, transform=ccrs.PlateCarree())

# plot dropsonde locations - 11 April
ds_dict = dropsonde_ds["RF17"]
for i, ds in enumerate(ds_dict.values()):
    ds["alt"] = ds.alt / 1000  # convert altitude to km
    launch_time = pd.to_datetime(ds.launch_time.to_numpy())
    x, y = ds.lon.mean().to_numpy(), ds.lat.mean().to_numpy()
    cross = ax.plot(x, y, "x", color="orangered", markersize=4, label="Dropsonde", transform=data_crs,
                    zorder=450)
    ax.text(x, y, f"{launch_time:%H:%M}", c="k", fontsize=7, transform=data_crs, zorder=500,
            path_effects=[patheffects.withStroke(linewidth=0.5, foreground="white")])

# plot trajectories 12 April in second row first column
ax = axs[1]
ax.coastlines(alpha=0.5)
ax.set(title="(b) RF 18")
ax.set_extent([xlim[0], xlim[1], ylim[0], ylim[1]], crs=map_crs)
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
cp.clabel(fontsize=5, inline=1, inline_spacing=4, fmt='%i', rightside_up=True, use_clabeltext=True)

# add seaice edge
ci_levels = [0.8]
cci = ax.tricontour(ifs.lon, ifs.lat, ifs.ci, ci_levels, transform=data_crs, linestyles="--", colors="#332288",
                    linewidths=1)

# add high cloud cover according to IFS
ifs_cc = ifs.hcc
hcc = ax.tricontourf(ifs.lon, ifs.lat, ifs_cc, levels=np.arange(0.2, 1.01, 0.1), transform=data_crs,
                     cmap="Blues", alpha=0.5)
# ax.tricontour(ifs.lon, ifs.lat, ifs_cc, levels=[0.2], linestyles=":", colors="blue", transform=data_crs,
#               alpha=1, linewidths=0.5)

# add colorbar
axins1 = inset_axes(
    ax,
    width="3%",  # width: 50% of parent_bbox width
    height="25%",  # height: 5%
    loc="upper left",
)
cb = plt.colorbar(hcc, cax=axins1, orientation="vertical", ticks=[0.2, 0.4, 0.6, 0.8, 1])
axins1.yaxis.set_ticks_position("right")
axins1.set_yticklabels([0.2, 0.4, 0.6, 0.8, 1], size=6,
                      path_effects=[patheffects.withStroke(linewidth=0.5, foreground="white")])

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

# plot flight track - 12 April
ins = bahamas_ds["RF18"]
track_lons, track_lats = ins["IRS_LON"], ins["IRS_LAT"]
ax.plot(track_lons[::1000], track_lats[::1000], c="k",
        zorder=400, transform=ccrs.PlateCarree())

# highlight case study region
ins_hl = ins.sel(time=slices["RF18"]["above"])
ax.plot(ins_hl.IRS_LON[::100], ins_hl.IRS_LAT[::100], c=cbc[1],
        zorder=400, transform=ccrs.PlateCarree())

# plot dropsonde locations - 12 April
ds_dict = dropsonde_ds["RF18"]
for i, ds in enumerate(ds_dict.values()):
    launch_time = pd.to_datetime(ds.time[0].to_numpy())
    x, y = ds.lon.mean().to_numpy(), ds.lat.mean().to_numpy()
    cross = ax.plot(x, y, "x", color="orangered", markersize=4, transform=data_crs, zorder=450)
for i in [-4]:
    ds = list(ds_dict.values())[i]
    launch_time = pd.to_datetime(ds.time[-1].to_numpy())
    x, y = ds.lon.mean().to_numpy(), ds.lat.mean().to_numpy()
    ax.text(x, y, f"{launch_time:%H:%M}", color="k", fontsize=7, transform=data_crs, zorder=500,
            path_effects=[patheffects.withStroke(linewidth=0.5, foreground="white")])

# make legend for flight track and dropsondes
labels = ["HALO flight track", "Case study section",
          "Dropsonde", "Sea ice edge",
          "Mean sea level pressure (hPa)", "High cloud cover at 12:00 UTC"]
handles = [plt.plot([], ls="-", color="k")[0],  # flight track
           plt.plot([], ls="-", color=cbc[1])[0],  # case study section
           cross[0],  # dropsondes
           plt.plot([], ls="--", color="#332288")[0],  # sea ice edge
           plt.plot([], ls="solid", lw=0.7, color="k")[0],  # isobars
           Patch(facecolor="royalblue", alpha=0.5)]  # cloud cover
fig.legend(handles=handles, labels=labels, framealpha=1, ncols=3,
           loc="outside lower center")

cbar = fig.colorbar(line, pad=0.01, ax=ax,
                    shrink=plt_sett["shrink"],
                    ticks=plt_sett["cmap_ticks"])
cbar.set_label(label=plt_sett['label'])

figname = f"{plot_path}/HALO-AC3_RF17_RF18_fligh_track_trajectories_plot_overview.png"
plt.savefig(figname, dpi=600)
plt.show()
plt.close()

# %% plot zoom of case study region RF 17
data_crs = ccrs.PlateCarree()
map_crs = ccrs.NorthPolarStereo()

plt.rc("font", size=5)
fig, ax = plt.subplots(figsize=(2.5 * h.cm, 2.5 * h.cm),
                        subplot_kw={"projection": map_crs},
                        layout="constrained")

# plot trajectory map 11 April in first row and first column
ax.set_extent([-28, 28, 85, 90], crs=data_crs)
# ax.set_ylim(ylim)
gl = ax.gridlines(crs=data_crs, draw_labels=False, linewidth=1, color='gray', alpha=0.5,
                  linestyle=':', x_inline=False, y_inline=False, rotate_labels=False)

# Plot the surface pressure - 11 April
ifs = ifs_ds["RF17"].sel(time="2022-04-11 12:00")
pressure_levels = np.arange(900, 1125, 5)
press = ifs.mean_sea_level_pressure / 100  # conversion to hPa
cp = ax.tricontour(ifs.lon, ifs.lat, press, levels=pressure_levels, colors='k', linewidths=0.5,
                   linestyles='solid', alpha=1, transform=data_crs)
# cp.clabel(fontsize=2, inline=1, inline_spacing=1, fmt='%i hPa', rightside_up=True, use_clabeltext=True)
cp.clabel(fontsize=4, inline=1, inline_spacing=4, fmt='%i', rightside_up=True, use_clabeltext=True)

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

# plot flight track - 11 April
ins = bahamas_ds["RF17"]
track_lons, track_lats = ins["IRS_LON"], ins["IRS_LAT"]
ax.scatter(track_lons[::20], track_lats[::20], c="k", alpha=1, marker=".", s=3, zorder=400,
           label='HALO flight track', transform=data_crs, linestyle="solid")

# highlight case study region
ins_hl = ins.sel(time=slices["RF17"]["above"])
ax.scatter(ins_hl.IRS_LON[::10], ins_hl.IRS_LAT[::10], c=cbc[3], alpha=1, marker=".", s=1, zorder=400,
           transform=data_crs, linestyle="solid")

# plot dropsonde locations - 11 April
ds_dict = dropsonde_ds["RF17"]
for i, ds in enumerate([ds_dict["104205"], ds_dict["110137"]]):
    ds["alt"] = ds.alt / 1000  # convert altitude to km
    launch_time = pd.to_datetime(ds.launch_time.to_numpy())
    x, y = ds.lon.mean().to_numpy(), ds.lat.mean().to_numpy()
    cross = ax.plot(x, y, "x", color="orangered", markersize=4, label="Dropsonde", transform=data_crs,
                    zorder=450)
    ax.text(x-.5, y+.2, f"{launch_time:%H:%M}", c="k", transform=data_crs, zorder=500,
            path_effects=[patheffects.withStroke(linewidth=0.25, foreground="white")])


figname = f"{plot_path}/HALO-AC3_RF17_fligh_track_trajectories_plot_overview_zoom.png"
plt.savefig(figname, dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# %% plot zoom of case study region RF 18
plt.rc("font", size=5)
fig, ax = plt.subplots(figsize=(2 * h.cm, 2.5 * h.cm),
                        subplot_kw={"projection": map_crs},
                        layout="constrained")

# plot trajectories 12 April in second row first column
ax.coastlines(alpha=0.5)
ax.set_extent([-20, 22, 87, 90], crs=data_crs)
gl = ax.gridlines(crs=data_crs, draw_labels=False, linewidth=1, color='gray', alpha=0.5,
                  linestyle=':', x_inline=False, y_inline=False, rotate_labels=False)

# Plot the surface pressure - 12 April
ifs = ifs_ds["RF18"].sel(time="2022-04-12 12:00")
pressure_levels = np.arange(900, 1125, 5)
press = ifs.mean_sea_level_pressure / 100  # conversion to hPa
cp = ax.tricontour(ifs.lon, ifs.lat, press, levels=pressure_levels, colors='k', linewidths=0.5,
                   linestyles='solid', alpha=1, transform=data_crs)
# cp.clabel(fontsize=2, inline=1, inline_spacing=1, fmt='%i hPa', rightside_up=True, use_clabeltext=True)
cp.clabel(fontsize=4, inline=1, inline_spacing=4, fmt='%i', rightside_up=True, use_clabeltext=True)

# add high cloud cover
ifs_cc = ifs.hcc
hcc = ax.tricontourf(ifs.lon, ifs.lat, ifs_cc, levels=np.arange(0.2, 1.01, 0.1), transform=data_crs,
                     cmap="Blues", alpha=0.5)

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

# plot flight track - 12 April
ins = bahamas_ds["RF18"]
track_lons, track_lats = ins["IRS_LON"], ins["IRS_LAT"]
ax.plot(track_lons[::1000], track_lats[::1000], c="k",
        zorder=400, transform=data_crs)

# highlight case study region
ins_hl = ins.sel(time=slices["RF18"]["above"])
ax.plot(ins_hl.IRS_LON[::20], ins_hl.IRS_LAT[::20], c=cbc[1],
        zorder=400, transform=data_crs)

# plot dropsonde locations - 12 April
ds_dict = dropsonde_ds["RF18"]
for i in [0, -3, -6, 6, 3]:
    ds = list(ds_dict.values())[i]
    launch_time = pd.to_datetime(ds.time[-1].to_numpy())
    x, y = ds.lon.mean().to_numpy(), ds.lat.mean().to_numpy()
    cross = ax.plot(x, y, "x", color="orangered", markersize=4, transform=data_crs, zorder=450)
    launch_time = pd.to_datetime(ds.time[-1].to_numpy())
    x, y = ds.lon.mean().to_numpy(), ds.lat.mean().to_numpy()
    ax.text(x, y, f"{launch_time:%H:%M}", color="k", fontsize=6, transform=data_crs, zorder=500,
            path_effects=[patheffects.withStroke(linewidth=0.5, foreground="white")])

figname = f"{plot_path}/HALO-AC3_RF18_fligh_track_trajectories_plot_overview_zoom.png"
plt.savefig(figname, dpi=600, bbox_inches='tight')
plt.show()
plt.close()

# %% calculate stats for dropsonde comparison
dropsonde_stats = list()
below_cloud_altitude = dict()
for i, key in enumerate(keys):
    below_cloud_altitude[key] = bahamas_ds[key].IRS_ALT.sel(time=slices[key]["below"]).mean(dim="time") / 1000
    print(f"{key}: Mean below cloud altitude of HALO: {below_cloud_altitude[key]:.1f} km")
    ifs_plot = ecrad_dicts[key]["v15.1"].sel(time=slices[key]["case"])
    # add relative humidity over ice
    rh = relative_humidity_from_specific_humidity(ifs_plot.pressure_full * u.Pa, ifs_plot.t * u.K, ifs_plot.q * u("kg/kg"))
    rh_ice = met.relative_humidity_water_to_relative_humidity_ice(rh * 100, ifs_plot.t - 273.15)
    ifs_plot["rh_ice"] = rh_ice

    ds_plot = dropsonde_ds[key].copy()

    times = ["104205", "110137"] if key == "RF17" else ["110321", "110823", "111442", "112014", "112524"]
    # calculate time switch for comaprison with IFS
    date = "20220411" if key == "RF17" else "20220412"
    times_dt = pd.to_datetime([date + t for t in times], format="%Y%m%d%H%M%S")
    t_switch = times_dt[0] + (times_dt[-1] - times_dt[0]) / 2
    # Air temperature
    ds_list = list()
    for k in times:
        ds = ds_plot[k]
        ds = ds.where((~np.isnan(ds.tdry) & ~np.isnan(ds.alt)), drop=True)
        ds["rh_ice"] = met.relative_humidity_water_to_relative_humidity_ice(ds.rh, ds.tdry)
        t_array = xr.DataArray(ds.tdry.to_numpy(), coords=dict(alt=ds.alt.to_numpy()), name="t")
        rh_ice_array = xr.DataArray(ds.rh_ice.to_numpy(), coords=dict(alt=ds.alt.to_numpy()), name="rh_ice")
        dataset = xr.merge([t_array, rh_ice_array])
        ds_list.append(dataset)
    d1_t, d1_rh_ice = list(), list()
    d2_t, d2_rh_ice = list(), list()
    for t in ifs_plot.time:
        ifs_sel = ifs_plot.sel(time=t)
        ifs_sel = ifs_sel.assign_coords(half_level=ifs_sel.press_height_hl, level=ifs_sel.press_height_full)
        if t < t_switch:
            ds = ds_list[0]
            ifs_inp = ifs_sel.interp(half_level=ds.alt)
            d1_t.append(ds.t - (ifs_inp.temperature_hl - 273))
            ifs_inp = ifs_sel.interp(level=ds.alt)
            d1_rh_ice.append(ds.rh_ice - ifs_inp.rh_ice)
        else:
            ds = ds_list[1]
            ifs_inp = ifs_sel.interp(half_level=ds.alt)
            d2_t.append(ds.t - (ifs_inp.temperature_hl - 273))
            ifs_inp = ifs_sel.interp(level=ds.alt)
            d2_rh_ice.append(ds.rh_ice - ifs_inp.rh_ice)

    ds1_t_diff = xr.concat(d1_t, dim="time")
    ds1_t_diff.name = "t"
    ds1_rh_ice_diff = xr.concat(d1_rh_ice, dim="time")
    ds1_rh_ice_diff.name = "rh_ice"
    ds1_diff = xr.merge([ds1_t_diff, ds1_rh_ice_diff])
    ds2_t_diff = xr.concat(d2_t, dim="time")
    ds2_t_diff.name = "t"
    ds2_rh_ice_diff = xr.concat(d2_rh_ice, dim="time")
    ds2_rh_ice_diff.name = "rh_ice"
    ds2_diff = xr.merge([ds2_t_diff, ds2_rh_ice_diff])

    print(f"{key}: IFS maximum temperature difference from dropsonde at {times[0]}:"
          f" {np.abs(ds1_diff.t).max().to_numpy():.2f} K\n"
          f"{key}: IFS maximum temperature difference from dropsonde at {times[1]}:"
          f" {np.abs(ds2_diff.t).max().to_numpy():.2f} K\n"
          f"{key}: IFS maximum RH_ice difference from dropsonde at {times[0]}:"
          f" {np.abs(ds1_diff.rh_ice).max().to_numpy():.2f} %\n"
          f"{key}: IFS maximum RH_ice difference from dropsonde at {times[1]}:"
          f" {np.abs(ds2_diff.rh_ice).max().to_numpy():.2f} %"
          )
# %% plot temperature and humidity profiles from IFS and from dropsonde
below_cloud_altitude = dict()
h.set_cb_friendly_colors("petroff_8")
plt.rc("font", size=7)
_, axs = plt.subplots(1, 4, figsize=(18 * h.cm, 10 * h.cm), layout="constrained")
for i, key in enumerate(keys):
    below_cloud_altitude[key] = bahamas_ds[key].IRS_ALT.sel(time=slices[key]["below"]).mean(dim="time") / 1000
    ax = axs[i * 2]
    ifs_plot = ecrad_dicts[key]["v15.1"].sel(time=slices[key]["case"])
    sf = 1000

    # Air temperature
    for t in ifs_plot.time:
        ifs_p = ifs_plot.sel(time=t)
        ax.plot(ifs_p.temperature_hl - 273.15, ifs_p.press_height_hl / 1000, color="grey", lw=0.5)
    ds_plot = dropsonde_ds[key]
    times = ["104205", "110137"] if key == "RF17" else ["110321", "110823", "111442", "112014", "112524"]
    for k in times:
        ds = ds_plot[k]
        var = "tdry" if key == "RF17" else "ta"
        ds = ds.where(~np.isnan(ds[var]), drop=True)
        ax.plot(ds[var], ds.gpsalt / sf, label=f"DS {k[:2]}:{k[2:4]} UTC", lw=2)
    ax.set(xlim=(-60, -10), ylim=(0, 12), xlabel="Air temperature (°C)", title=f"{key}")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(base=10))
    ax.plot([], color="grey", label="IFS profiles")
    ax.axhline(below_cloud_altitude[key], c="k")
    ax.grid()

    # RH
    ax = axs[i * 2 + 1]
    ifs_plot = ecrad_dicts[key]["v15.1"].sel(time=slices[key]["case"])
    for t in ifs_plot.time:
        ifs_p = ifs_plot.sel(time=t)
        rh = relative_humidity_from_specific_humidity(ifs_p.pressure_full * u.Pa, ifs_p.t * u.K, ifs_p.q * u("kg/kg"))
        rh_ice = met.relative_humidity_water_to_relative_humidity_ice(rh * 100, ifs_p.t - 273.15)
        ax.plot(rh_ice, ifs_p.press_height_full / 1000, color="grey", lw=0.5)
    ds_plot = dropsonde_ds[key]
    times = ["104205", "110137"] if key == "RF17" else ["110321", "110823", "111442", "112014", "112524"]
    for k in times:
        ds = ds_plot[k]
        ds = ds.where(~np.isnan(ds.rh), drop=True)
        ax.plot(met.relative_humidity_water_to_relative_humidity_ice(ds.rh, ds[var]),
                ds.gpsalt / sf, label=f"DS {k[:2]}:{k[2:4]} UTC", lw=2)
    ax.set(xlim=(0, 130), ylim=(0, 12), xlabel="Relative humidity over ice (%)", title=f"{key}")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(base=25))
    ax.plot([], color="grey", label="IFS profiles")
    ax.axhline(below_cloud_altitude[key], c="k")
    ax.legend()
    ax.grid()

axs[0].set_ylabel("Altitude (km)")
axs[0].text(0.02, 0.95, "(a)", transform=axs[0].transAxes)
axs[1].text(0.02, 0.95, "(b)", transform=axs[1].transAxes)
axs[2].text(0.02, 0.95, "(c)", transform=axs[2].transAxes)
axs[3].text(0.02, 0.95, "(d)", transform=axs[3].transAxes)

figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_ifs_dropsonde_t_rh.pdf"
plt.savefig(figname, dpi=300, bbox_inches="tight")
plt.show()
plt.close()
h.set_cb_friendly_colors("petroff_6")

# %% plot scatter plot of above cloud measurements and simulations
plt.rc("font", size=7)
label = ["(a)", "(b)"]
for v in ["v15.1", "v18.1", "v19.1"]:
    _, axs = plt.subplots(1, 2, figsize=(16 * h.cm, 8 * h.cm))
    for i, key in enumerate(keys):
        ax = axs[i]
        above_sel = (bahamas_ds[key].IRS_ALT > 11000).resample(time="1Min").first()
        bacardi_res = bacardi_ds_res[key]
        bacardi_plot = bacardi_res.where(bacardi_res.alt > 11000)
        ecrad_ds = ecrad_dicts[key][v]
        height_sel = ecrad_dicts[key][v].aircraft_level
        ecrad_plot = ecrad_ds.flux_dn_sw.isel(half_level=height_sel).where(above_sel)

        # actual plotting
        rmse = np.sqrt(np.mean((bacardi_plot["F_down_solar_diff"] - ecrad_plot) ** 2)).to_numpy()
        bias = np.nanmean((bacardi_plot["F_down_solar_diff"] - ecrad_plot).to_numpy())
        ax.scatter(bacardi_plot.F_down_solar_diff, ecrad_plot, color=cbc[3])
        ax.axline((0, 0), slope=1, color="k", lw=2, transform=ax.transAxes)
        ax.set(
            aspect="equal",
            xlabel="Measured irradiance (W$\,$m$^{-2}$)",
            ylabel="Simulated irradiance (W$\,$m$^{-2}$)",
            xlim=(175, 525),
            ylim=(175, 525),
        )
        ax.grid()
        ax.text(
            0.025,
            0.95,
            f"{label[i]} {key.replace('1', ' 1')}\n"
            f"n= {sum(~np.isnan(bacardi_plot['F_down_solar'])):.0f}\n"
            f"RMSE: {rmse:.0f} {h.plot_units['flux_dn_sw']}\n"
            f"Bias: {bias:.0f} {h.plot_units['flux_dn_sw']}",
            ha="left",
            va="top",
            transform=ax.transAxes,
            bbox=dict(fc="white", ec="black", alpha=0.8, boxstyle="round"),
        )

    plt.tight_layout()

    figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_bacardi_ecrad_f_down_solar_above_cloud_all_{v}.pdf"
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
    ecrad_ds = ecrad_dicts[key]["v15.1"]
    height_sel = ecrad_dicts[key]["v15.1"].aircraft_level
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

# %% plot re_ice from VarCloud and IFS
key = "RF18"
sel_time = slice(pd.to_datetime("2022-04-11 10:49"), pd.to_datetime("2022-04-11 11:04"))
sel_time = slice(pd.to_datetime("2022-04-12 11:04"), pd.to_datetime("2022-04-12 11:24"))
varcloud = ecrad_dicts[key]["v16"].reset_coords("column")
ifs = ecrad_dicts[key]["v15"].sel(time=sel_time)

# varcloud.re_ice.plot(x="time", label="VarCloud", cmap=cm.sapphire)
# ifs.re_ice.where(varcloud.re_ice.resample(time="1Min").mean().to_numpy() > 0).plot(x="time", label="IFS", cmap=cm.flamingo)
ifs.cloud_fraction.where(ifs.cloud_fraction > 0.01).plot(x="time", label="IFS", cmap=cm.sapphire)
plt.show()
plt.close()

# %% plot PDF of IWC and re_ice
plt.rc("font", size=8)
legend_labels = ["VarCloud", "IFS"]
binsizes = dict(iwc=1, reice=4)
binedges = dict(iwc=20, reice=100)
text_loc_x = 0.05
text_loc_y = 0.9
_, axs = plt.subplots(2, 2, figsize=(17 * h.cm, 10 * h.cm), layout="constrained")
ylims = {"iwc": (0, 0.3), "reice": (0, 0.095)}
# upper left panel - RF17 IWC
ax = axs[0, 0]
plot_ds = ecrad_dicts["RF17"]
# sel_time = slice(pd.to_datetime("2022-04-11 10:49"), pd.to_datetime("2022-04-11 11:04"))
sel_time = slices["RF17"]["below"]
bins = np.arange(0, binedges["iwc"], binsizes["iwc"])
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
ax.text(text_loc_x, text_loc_y, "(a)", transform=ax.transAxes)
ax.set(title=f"RF 17",
       ylabel=f"Probability density function",
       xlabel=f"Ice water content ({h.plot_units['iwc']})",
       ylim=ylims["iwc"])

# lower left panel - RF17 re_ice
ax = axs[1, 0]
bins = np.arange(0, binedges["reice"], binsizes["reice"])
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
    print(f"RF17 Mean reice {v}: {pds.mean():.2f}")
ax.grid()
ax.text(text_loc_x, text_loc_y, "(c)", transform=ax.transAxes)
ax.set(ylabel="Probability density function",
       xlabel=f"Ice effective radius ({h.plot_units['re_ice']})",
       ylim=ylims["reice"])

# upper right panel - RF18 IWC
ax = axs[0, 1]
plot_ds = ecrad_dicts["RF18"]
# sel_time = slice(pd.to_datetime("2022-04-12 11:04"), pd.to_datetime("2022-04-12 11:24"))
sel_time = slices["RF18"]["below"]
bins = np.arange(0, binedges["iwc"], binsizes["iwc"])
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
ax.text(text_loc_x, text_loc_y, "(b)", transform=ax.transAxes)
ax.set(title=f"RF 18",
       ylabel=f"",
       xlabel=f"Ice water content ({h.plot_units['iwc']})",
       ylim=ylims["iwc"])

# lower right panel - RF18 re_ice
ax = axs[1, 1]
bins = np.arange(0, binedges["reice"], binsizes["reice"])
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
    print(f"RF18 Mean reice {v}: {pds.mean():.2f}")
ax.grid()
ax.text(text_loc_x, text_loc_y, "(d)", transform=ax.transAxes)
ax.set(ylabel="",
       xlabel=f"Ice effective radius ({h.plot_units['re_ice']})",
       ylim=ylims["reice"])

figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_IFS_iwc_re_ice_pdf_case_studies.pdf"
plt.savefig(figname, dpi=300, bbox_inches="tight")
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
        height_sel = ecrad_dicts[key][v].aircraft_level
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
        height_sel = ecrad_dicts[key][v].aircraft_level
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
        height_sel = ecrad_dicts[key][v].aircraft_level
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
        height_sel = ecrad_dicts[key][v].aircraft_level
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

# %% plot PDF of transmissivity (above cloud simulation) below cloud all columns - all ice optics
transmissivity_stats = list()
plt.rc("font", size=7.5)
label = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"]]
ylims = [(0, 36), (0, 36)]
legend_loc = [3, 1]
sf = 1
norm = ""
binsize = 0.01 * sf
xlabel = "Solar Transmissivity" if norm == "" else "Normalized Transmissivity"
_, axs = plt.subplots(2, 3, figsize=(18 * h.cm, 14 * h.cm), layout="constrained")
for i, key in enumerate(keys):
    ax = axs[i]
    l = label[i]
    bacardi_sel = bacardi_ds[key].sel(time=slices[key]["below"])
    bacardi_plot = bacardi_sel[f"transmissivity_above_cloud{norm}"].resample(time="1Min").mean() * sf
    bins = np.arange(0.5, 1.0, binsize)
    # BACARDI histogram
    bacardi_hist = np.histogram(bacardi_plot, density=True, bins=bins)

    # save statistics
    transmissivity_stats.append((key, "BACARDI", "Mean", bacardi_plot.mean().to_numpy()))
    transmissivity_stats.append((key, "BACARDI", "Median", bacardi_plot.median().to_numpy()))

    for ii, v in enumerate(["v15.1", "v19.1", "v18.1"]):
        v_name = ecrad.version_names[v[:3]]
        v_name = f"{v_name}2013" if v_name == "Yi" else v_name
        a = ax[ii]
        ecrad_ds = ecrad_orgs[key][v].sel(time=slices[key]["below"])
        height_sel = ecrad_ds["aircraft_level"]
        ecrad_plot = ecrad_ds[f"transmissivity_sw_above_cloud{norm}"].isel(half_level=height_sel) * sf

        # save statistics
        transmissivity_stats.append((key, v_name, "Mean", ecrad_plot.mean().to_numpy()))
        transmissivity_stats.append((key, v_name, "Median", ecrad_plot.median().to_numpy()))
        # actual plotting
        sns.histplot(bacardi_plot, label="BACARDI", ax=a, stat="density", kde=False, bins=bins, element="step")
        sns.histplot(ecrad_plot.to_numpy().flatten(), label=v_name, stat="density", element="step",
                     kde=False, bins=bins, ax=a, color=cbc[ii + 1])
        # add mean
        a.axvline(bacardi_plot.mean(), color=cbc[0], lw=3, ls="--")
        a.axvline(ecrad_plot.mean(), color=cbc[ii + 1], lw=3, ls="--")
        # a.plot(bacardi_plot.mean(), 15, color=cbc[0], ls="", marker="*")
        # a.plot(ecrad_plot.mean(), 15, color=cbc[ii + 1], ls="", marker="*")
        a.plot([], ls="--", color="k", label="Mean")  # label for means
        # textbox
        hist = np.histogram(ecrad_plot, density=True, bins=bins)
        # w = wasserstein_distance(bacardi_hist[0], hist[0]) * sf
        a.set(ylabel="",
              ylim=ylims[i],
              xlim=(0.45, 1)
              )
        handles, labels = a.get_legend_handles_labels()
        order = [1, 0, 2]
        handles = [handles[idx] for idx in order]
        labels = [labels[idx] for idx in order]
        if key == "RF17":
            a.legend(handles, labels, loc=legend_loc[i])
        a.text(
            0.02,
            0.95,
            f"{l[ii]}",
            ha="left",
            va="top",
            transform=a.transAxes,
        )
        a.grid()

    ax[0].set(ylabel="Probability density function")
    ax[1].set(title=f"{key.replace('1', ' 1')} (n = {len(ecrad_plot.to_numpy().flatten()):.0f})")
    if key == "RF18":
        ax[1].set(xlabel=xlabel)

figname = (f"{plot_path}/HALO-AC3_HALO_RF17_RF18_bacardi_ecrad_transmissivity_above_cloud{norm}_PDF"
           f"_below_cloud_ice_optics"
           f"_all_columns.pdf")
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot PDF of transmissivity (above cloud simulation) below cloud  - varcloud all ice optics
plt.rc("font", size=7.5)
label = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"]]
ylims = [(0, 36), (0, 36)]
legend_loc = [6, 1]
sf = 1
norm = ""
binsize = 0.01 * sf
xlabel = "Solar Transmissivity" if norm == "" else "Normalized Transmissivity"
_, axs = plt.subplots(2, 3, figsize=(18 * h.cm, 14 * h.cm), layout="constrained")
for i, key in enumerate(keys):
    ax = axs[i]
    l = label[i]
    bacardi_sel = bacardi_ds[key].sel(time=slices[key]["below"])
    bacardi_plot = bacardi_sel[f"transmissivity_above_cloud{norm}"].resample(time="1Min").mean() * sf
    bins = np.arange(0.5, 1.0, binsize)
    # BACARDI histogram
    bacardi_hist = np.histogram(bacardi_plot, density=True, bins=bins)
    for ii, v in enumerate(["v16", "v28", "v20"]):
        v_name = ecrad.version_names[v]
        v_name = f"Yi2013 VarCloud" if "Yi" in v_name else v_name
        a = ax[ii]
        ecrad_ds = ecrad_dicts[key][v].sel(time=slices[key]["below"])
        height_sel = ecrad_ds["aircraft_level"]
        ecrad_plot = ecrad_ds[f"transmissivity_sw_above_cloud{norm}"].isel(half_level=height_sel) * sf

        # save statistics
        transmissivity_stats.append((key, v_name, "Mean", ecrad_plot.mean().to_numpy()))
        transmissivity_stats.append((key, v_name, "Median", ecrad_plot.median().to_numpy()))

        # actual plotting
        sns.histplot(bacardi_plot, label="BACARDI", ax=a, stat="density", kde=False, bins=bins)
        sns.histplot(ecrad_plot, label=v_name.replace(" ", "\n"), stat="density",
                     kde=False, bins=bins, ax=a, color=cbc[ii + 1])
        # add mean
        a.axvline(bacardi_plot.mean(), color=cbc[0], lw=3, ls="--")
        a.axvline(ecrad_plot.mean(), color=cbc[ii + 1], lw=3, ls="--")
        a.plot([], ls="--", color="k", label="Mean")  # label for means
        # textbox
        hist = np.histogram(ecrad_plot, density=True, bins=bins)
        # w = wasserstein_distance(bacardi_hist[0], hist[0]) * sf
        a.set(ylabel="",
              ylim=ylims[i],
              xlim=(0.45, 1)
              )
        handles, labels = a.get_legend_handles_labels()
        order = [1, 2, 0]
        handles = [handles[idx] for idx in order]
        labels = [labels[idx] for idx in order]
        if key == "RF17":
            a.legend(handles, labels, loc=legend_loc[i])
        a.text(
            0.02,
            0.95,
            f"{l[ii]}",
            ha="left",
            va="top",
            transform=a.transAxes
        )
        a.grid()

    ax[0].set(ylabel="Probability density function")
    ax[1].set(title=f"{key.replace('1', ' 1')} (n = {len(ecrad_plot):.0f})")
    if key == "RF18":
        ax[1].set(xlabel=xlabel)

figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_bacardi_ecrad_f_down_solar_norm_PDF_below_cloud_ice_optics_varcloud.pdf"
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
        height_sel = ecrad_ds["aircraft_level"]
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

# %% print transmissivity statistics
transmissivity_df = pd.DataFrame(transmissivity_stats)
print(transmissivity_df)

# %% satellite image together with flight tracks
_, axs = plt.subplots(1, 2, figsize=h.figsize)
plot_crs = ccrs.NorthPolarStereo()
data_crs = ccrs.Geodetic()

# %% plot IFS cloud fraction lidar/mask comparison with below cloud track and histograms of cloud base and ceiling
plt.rc("font", size=6.5)
fig = plt.figure(figsize=(18 * h.cm, 9 * h.cm), layout="constrained")
gs = gridspec.GridSpec(2, 3, figure=fig)
axs = list()
# cloud fraction plus lidar mask
axs.append(fig.add_subplot(gs[0, :-1]))
axs.append(fig.add_subplot(gs[1, :-1]))
# histograms of iwc
axs.append(fig.add_subplot(gs[0, -1]))
axs.append(fig.add_subplot(gs[1, -1]))
for i, key in enumerate(keys):
    ax = axs[i]
    ds = ecrad_dicts[key]["v15"].sel(time=slices[key]["case"])
    ifs_plot = ds[["cloud_fraction", "iwc"]]
    ifs_cth = ds.ceil / 1000
    # ifs_cbh = ds.cbh / 1000
    bahamas_plot = bahamas_ds[key].IRS_ALT.sel(time=slices[key]["case"]) / 1000
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
    ifs_plot = ifs_plot.where(ifs_plot.cloud_fraction > 0)
    halo_plot = varcloud_ds[key].sel(time=slices[key]["case"]).Varcloud_Input_Mask
    halo_plot = halo_plot.assign_coords(height=halo_plot.height / 1000).sortby("height")
    time_extend = pd.to_timedelta((ifs_plot.time[-1] - ifs_plot.time[0]).to_numpy())

    # plot IFS cloud cover prediction and Radar lidar mask
    pcm = ifs_plot.cloud_fraction.plot(x="time", cmap=cm.sapphire, ax=ax, add_colorbar=False)
    halo_plot.plot.contour(x="time", levels=[0.9], colors=cbc[1], ax=ax, linewidths=2)
    ax.plot([], color=cbc[1], label="Radar & Lidar Mask", lw=2)
    bahamas_plot.plot(x="time", lw=2, color=cbc[-2], label="HALO altitude", ax=ax)
    ax.axvline(x=pd.to_datetime(f"{bahamas_plot.time.dt.date[0]:%Y-%m-%d} 11:30"),
               label="New IFS timestep", lw=2, ls="--")
    h.set_xticks_and_xlabels(ax, time_extend)
    ax.set(xlabel="Time (UTC)", ylabel="Height (km)")
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=0, ha="center")

    # plot IWC histograms over height
    ax = axs[2 + i]
    binsize = 0.5  # km
    bins = np.arange(0, 12, binsize)
    iwc_df = ifs_plot.iwc.to_dataframe().dropna()
    sns.histplot(iwc_df, y="height", stat="density", bins=bins, ax=ax)
    ax.set(ylabel="", xlim=(0, 0.51))
    ax.margins(y=0)

# place colorbar for both flights
fig.colorbar(pcm, ax=axs[:2], label=f"IFS {h.cbarlabels['cloud_fraction']}", pad=0.001)
axs[0].legend()
axs[0].set_xlabel("")
axs[2].set_xlabel("")
axs[3].set_xlabel("Ice water content density")
axs[0].text(0.03, 0.88, "(a) RF 17", transform=axs[0].transAxes, bbox=dict(boxstyle="Round", fc="white"))
axs[1].text(0.03, 0.88, "(c) RF 18", transform=axs[1].transAxes, bbox=dict(boxstyle="Round", fc="white"))
axs[2].text(0.03, 0.88, "(b) RF 17", transform=axs[2].transAxes, bbox=dict(boxstyle="Round", fc="white"))
axs[3].text(0.03, 0.88, "(d) RF 18", transform=axs[3].transAxes, bbox=dict(boxstyle="Round", fc="white"))

figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_IFS_cloud_fraction_radar_lidar_mask_hists.pdf"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% testing
cover = np.arange(60, 100, 5) / 100
theta = np.deg2rad(90) - np.arccos(np.sqrt(cover))
radius = np.arange(1, 11, 0.5)
dA = 2 * np.pi * radius[0] ** 2 * (1 - np.cos(theta))
plt.plot(cover * 100, np.rad2deg(theta), marker="o")
plt.xlabel("Fraction of irradiance (%)")
plt.ylabel("Viewing zentih angle (°)")
plt.grid()
plt.show()
plt.plot()
plt.close()

# %% plot BACARDI footprint
_, ax = plt.subplots(figsize=h.figsize_wide)
for c in cover:
    theta = np.deg2rad(90) - np.arccos(np.sqrt(c))
    A = np.pi * (np.tan(theta) * radius) ** 2
    ax.plot(radius, A, marker="o", label=f"{c:.0%}")
ax.legend(title="Fraction of irradiance", ncols=2)
ax.set(xlabel="Distance to aircraft (km)",
       ylabel="Footprint of sensor (km$^2$)")
ax.grid()
plt.show()
plt.plot()
plt.close()

# %% plot radius of BACARDI footprint
_, ax = plt.subplots(figsize=h.figsize_wide)
for c in cover:
    theta = np.deg2rad(90) - np.arccos(np.sqrt(c))
    r = np.tan(theta) * radius
    ax.plot(radius, r, marker="o", label=f"{c:.0%}")
ax.legend(title="Fraction of irradiance", ncols=2)
ax.set(xlabel="Distance to aircraft (km)",
       ylabel="Radius of footprint of sensor (km)")
ax.grid()
plt.show()
plt.plot()
plt.close()
# %% plot PDF of transmissivity (above cloud simulation) below cloud - Fu-IFS above and below cloud section
plt.rc("font", size=7)
label = [["a)", "b)", "c)"], ["d)", "e)", "f)"]]
ylims = [(0, 50), (0, 50)]
legend_loc = [3, 1]
sf = 1
norm = ""
binsize = 0.01 * sf
xlabel = "Solar Transmissivity" if norm == "" else "Normalized Transmissivity"
_, axs = plt.subplots(1, 2, figsize=(18 * h.cm, 14 * h.cm))
for i, key in enumerate(keys):
    ax = axs[i]
    l = label[i]
    bacardi_sel = bacardi_ds[key].sel(time=slices[key]["below"])
    bacardi_plot = bacardi_sel[f"transmissivity_above_cloud{norm}"].resample(time="1Min").mean() * sf
    bins = np.arange(np.round(bacardi_plot.min() - binsize, 2),
                     np.round(bacardi_plot.max() + binsize, 2),
                     binsize)

    ecrad_ds = ecrad_dicts[key]["v15.1"].sel(time=slices[key]["below"])
    height_sel = ecrad_ds["aircraft_level"]
    ecrad_plot = ecrad_ds[f"transmissivity_sw_above_cloud{norm}"].isel(half_level=height_sel) * sf

    # actual plotting
    # sns.histplot(bacardi_plot, label="BACARDI", ax=ax, stat="density", kde=False, bins=bins)
    sns.histplot(ecrad_plot, label=f"{ecrad.version_names['v15']} (n={len(ecrad_plot)})", stat="density",
                 kde=False, bins=bins, ax=ax, color=cbc[1])
    # add mean
    # ax.axvline(bacardi_plot.mean(), color=cbc[0], lw=3, ls="--")
    ax.axvline(ecrad_plot.mean(), color=cbc[1], lw=3, ls="--")

    # add below cloud transmissivity of above cloud section to Fu-IFS panel (a, d)
    ecrad_ds = ecrad_dicts[key]["v15.1"].sel(time=slices[key]["above"])
    height_sel = np.unique(height_sel)
    ecrad_plot = ecrad_ds[f"transmissivity_sw_above_cloud{norm}"].isel(half_level=height_sel) * sf
    sns.histplot(ecrad_plot.to_numpy().flatten(), label=f"Fu-IFS\n$11\,$UTC (n={len(ecrad_plot)})", stat="density",
                 kde=False, bins=bins, ax=ax, color=cbc[2])
    # add mean
    ax.axvline(ecrad_plot.mean(), color=cbc[2], lw=3, ls="--")
    ax.plot([], ls="--", color="k", label="Mean")  # label for means

    # textbox
    ax.set(ylabel="",
           ylim=ylims[i],
           xlabel=xlabel,
           xlim=(0.45, 1)
           )
    handles, labels = ax.get_legend_handles_labels()
    order = [1, 2, 0]
    handles = [handles[idx] for idx in order]
    labels = [labels[idx] for idx in order]
    ax.legend(handles, labels, loc=legend_loc[i])
    ax.text(
        0.04,
        0.95,
        f"{l[ii]} {key}\n",
        # f"$W$ = {w:.1f}\n"
        # f"n = {len(ecrad_plot):.0f}",
        ha="left",
        va="top",
        transform=ax.transAxes,
        bbox=dict(fc="white", ec="black", alpha=0.9, boxstyle="round"),
    )
    ax.grid()

axs[0].set(ylabel="Density")


plt.tight_layout()

figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_bacardi_ecrad_transmissivity_above_cloud{norm}_PDF_below_cloud_11_12UTC.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot PDF of IWC difference for 10 surrounding grid points
plt.rc("font", size=8)
legend_labels = ["11 UTC", "12 UTC"]
text_labels = ["(a)", "(b)"]
binsizes = dict(iwc=0.25, reice=4)
_, axs = plt.subplots(1, 2, figsize=(17 * h.cm, 10 * h.cm))
ylims = {"iwc": (0, 1.5), "reice": (0, 0.095)}
stats = list()
for i, key in enumerate(keys):
    ax = axs[i]
    plot_ds = ecrad_orgs[key]
    sel_time = slices[key]["case"]
    date = "2022-04-11" if key == "RF17" else "2022-04-12"
    binsize = binsizes["iwc"]
    bins = np.arange(-5.1, 5.1, binsize)
    for ii, v in enumerate(["v30.1"]):
        iwc, cc = plot_ds[v].iwc.sel(time=sel_time), plot_ds[v].cloud_fraction.sel(time=sel_time)
        iwc_plot = iwc.where(cc > 0).where(cc == 0, iwc / cc)
        iwc_diffs = list()
        iwc_plot_c0 = iwc_plot.sel(column=0)
        iwc_plot_c0 = iwc_plot_c0.where(~np.isnan(iwc_plot_c0), 0)
        for iii in iwc_plot.column[1:]:
            iwc_diffs.append(iwc_plot_c0 - iwc_plot.sel(column=iii))

        iwc_plot = xr.concat(iwc_diffs, dim="column")

        # 11 UTC
        pds = (iwc_plot
               .where(iwc_plot.time < pd.to_datetime(f"{date} 11:30"))
               .to_numpy()).flatten() * 1e6
        pds = pds[~np.isnan(pds)]
        mad = median_abs_deviation(pds)
        median = np.median(pds)
        stats.append((key, "11UTC", "MAD", mad))
        stats.append((key, "11UTC", "Median", median))
        ax.hist(
            pds,
            bins=bins,
            label=legend_labels[0] + f" (n={len(pds)})",
            color=cbc[0],
            histtype="step",
            density=True,
            lw=2,
        )
        # 12 UTC
        pds = (iwc_plot
               .where(iwc_plot.time > pd.to_datetime(f"{date} 11:30"))
               .to_numpy()).flatten() * 1e6
        pds = pds[~np.isnan(pds)]
        stats.append((key, "12UTC", "mad", median_abs_deviation(pds)))
        stats.append((key, "12UTC", "median", np.median(pds)))
        ax.hist(
            pds,
            bins=bins,
            label=legend_labels[1] + f" (n={len(pds)})",
            color=cbc[1],
            histtype="step",
            density=True,
            lw=2,
        )
    ax.legend()
    ax.grid()
    ax.text(0.03, 0.93,
            f"{text_labels[i]} {key.replace('1', ' 1')}",
            transform=ax.transAxes,
            bbox=dict(boxstyle="Round", fc="white"),
            )
    ax.set(title="",
           ylabel="",
           xlabel=f"Ice water content ({h.plot_units['iwc']})",
           ylim=ylims["iwc"])

axs[0].set(ylabel="Probability density function")

figname = f"{plot_path}/HALO_AC3_RF17_RF18_IFS_IWC_variability.pdf"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot PDF of IWC for 11 and 12 UTC for each column
plt.rc("font", size=8)
legend_labels = ["11 UTC", "12 UTC"]
text_labels = ["(a)", "(b)"]
binsizes = dict(iwc=1, reice=4)
ylims = {"iwc": (0, 0.3), "reice": (0, 0.095)}
stats = list()
for c in range(0, 10):
    _, axs = plt.subplots(1, 2, figsize=(17 * h.cm, 10 * h.cm))
    for i, key in enumerate(keys):
        ax = axs[i]
        plot_ds = ecrad_orgs[key]
        sel_time = slices[key]["case"]
        date = "2022-04-11" if key == "RF17" else "2022-04-12"
        binsize = binsizes["iwc"]
        bins = np.arange(0, 20.1, binsize)
        v = "v15.1"
        iwc, cc = plot_ds[v].iwc.sel(time=sel_time), plot_ds[v].cloud_fraction.sel(time=sel_time)
        iwc_plot = iwc.where(cc > 0).where(cc == 0, iwc / cc)
        iwc_plot = iwc_plot.sel(column=c)

        # 11 UTC
        pds = (iwc_plot
               .where(iwc_plot.time < pd.to_datetime(f"{date} 11:30"))
               .to_numpy()).flatten() * 1e6
        pds = pds[~np.isnan(pds)]
        mad = median_abs_deviation(pds)
        median = np.median(pds)
        stats.append((key, c, "11UTC", "MAD", mad))
        stats.append((key, c, "11UTC", "Median", median))
        ax.hist(
            pds,
            bins=bins,
            label=legend_labels[0] + f" (n={len(pds)})",
            color=cbc[0],
            histtype="step",
            density=True,
            lw=2,
        )

        # 12 UTC
        pds = (iwc_plot
               .where(iwc_plot.time > pd.to_datetime(f"{date} 11:30"))
               .to_numpy()).flatten() * 1e6
        pds = pds[~np.isnan(pds)]
        stats.append((key, c, "12UTC", "MAD", median_abs_deviation(pds)))
        stats.append((key, c, "12UTC", "Median", np.median(pds)))
        ax.hist(
            pds,
            bins=bins,
            label=legend_labels[1] + f" (n={len(pds)})",
            color=cbc[1],
            histtype="step",
            density=True,
            lw=2,
        )

        ax.legend()
        ax.grid()
        ax.text(0.03, 0.93,
                f"{text_labels[i]} {key.replace('1', ' 1')}",
                transform=ax.transAxes,
                bbox=dict(boxstyle="Round", fc="white"),
                )
        ax.set(title="",
               ylabel="",
               xlabel=f"Ice water content ({h.plot_units['iwc']})",
               ylim=ylims["iwc"])

    axs[0].set(ylabel="Probability density function")

    figname = f"{plot_path}/HALO_AC3_RF17_RF18_IFS_IWC_change_11_to_12UTC_column{c}.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot PDF of IWC for 11 and 12 UTC with error bars
plt.rc("font", size=8)
legend_labels = ["11 UTC", "12 UTC"]
text_labels = ["(a)", "(b)"]
binsizes = dict(iwc=1, reice=4)
_, axs = plt.subplots(1, 2, figsize=(17 * h.cm, 10 * h.cm))
ylims = {"iwc": (0, 0.3), "reice": (0, 0.095)}
stats = list()
for i, key in enumerate(keys):
    ax = axs[i]
    plot_ds = ecrad_orgs[key]
    sel_time = slices[key]["case"]
    date = "2022-04-11" if key == "RF17" else "2022-04-12"
    binsize = binsizes["iwc"]
    bins = np.arange(0, 20.1, binsize)
    v = "v15.1"
    iwc, cc = plot_ds[v].iwc.sel(time=sel_time), plot_ds[v].cloud_fraction.sel(time=sel_time)
    iwc_plot = iwc.where(cc > 0).where(cc == 0, iwc / cc)
    iwc_plot0 = iwc_plot.sel(column=0)

    # 11 UTC
    pds = (iwc_plot0
           .where(iwc_plot0.time < pd.to_datetime(f"{date} 11:30"))
           .to_numpy()).flatten() * 1e6
    pds = pds[~np.isnan(pds)]
    y, binedges = np.histogram(pds, bins=bins, density=True)
    bincenters = 0.5 * (binedges[1:] + binedges[:-1])
    # get max and min counts from all 9 surrounding grid cells
    hists = list()
    for c in iwc_plot.column[1:]:
        tmp = (iwc_plot
               .sel(column=c)
               .where(iwc_plot.time < pd.to_datetime(f"{date} 11:30"))
               .to_numpy()).flatten() * 1e6
        tmp = tmp[~np.isnan(tmp)]
        y, _ = np.histogram(tmp, bins=bins, density=True)
        hists.append(y)

    stacked_hists = np.vstack(hists)
    min_counts = np.min(stacked_hists, axis=0)
    max_counts = np.max(stacked_hists, axis=0)
    yerror = np.stack((y - min_counts, np.abs(y - max_counts)))

    ax.errorbar(
        bincenters,
        y,
        yerr=yerror,
        label=legend_labels[0] + f" (n={len(pds)})",
        color=cbc[0],
        drawstyle="steps-mid",
        lw=2,
        capsize=2
    )

    # 12 UTC
    pds = (iwc_plot0
           .where(iwc_plot0.time > pd.to_datetime(f"{date} 11:30"))
           .to_numpy()).flatten() * 1e6
    pds = pds[~np.isnan(pds)]
    y, binedges = np.histogram(pds, bins=bins, density=True)
    bincenters = 0.5 * (binedges[1:] + binedges[:-1])
    # get max and min counts from all 9 surrounding grid cells
    hists = list()
    for c in iwc_plot.column[1:]:
        tmp = (iwc_plot
               .sel(column=c)
               .where(iwc_plot.time > pd.to_datetime(f"{date} 11:30"))
               .to_numpy()).flatten() * 1e6
        tmp = tmp[~np.isnan(tmp)]
        y, _ = np.histogram(tmp, bins=bins, density=True)
        hists.append(y)

    stacked_hists = np.vstack(hists)
    min_counts = np.min(stacked_hists, axis=0)
    max_counts = np.max(stacked_hists, axis=0)
    yerror = np.stack((y - min_counts, np.abs(y - max_counts)))

    ax.errorbar(
        bincenters,
        y,
        yerr=yerror,
        label=legend_labels[1] + f" (n={len(pds)})",
        color=cbc[1],
        drawstyle="steps-mid",
        lw=2,
        capsize=2
    )

    ax.legend()
    ax.grid()
    ax.text(0.03, 0.93,
            f"{text_labels[i]} {key.replace('1', ' 1')}",
            transform=ax.transAxes,
            bbox=dict(boxstyle="Round", fc="white"),
            )
    ax.set(title="",
           ylabel="",
           xlabel=f"Ice water content ({h.plot_units['iwc']})",
           ylim=ylims["iwc"])

axs[0].set(ylabel="Probability density function")

figname = f"{plot_path}/HALO_AC3_RF17_RF18_IFS_IWC_change_uncertainty_11_to_12UTC.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot PDF of IWC for 11 and 12 UTC for all columns
plt.rc("font", size=8)
legend_labels = ["11 UTC", "12 UTC"]
text_labels = ["(a)", "(b)"]
binsizes = dict(iwc=1, reice=4)
_, axs = plt.subplots(1, 2, figsize=(17 * h.cm, 10 * h.cm))
ylims = {"iwc": (0, 0.3), "reice": (0, 0.095)}
stats = list()
for i, key in enumerate(keys):
    ax = axs[i]
    plot_ds = ecrad_orgs[key]
    sel_time = slices[key]["case"]
    date = "2022-04-11" if key == "RF17" else "2022-04-12"
    binsize = binsizes["iwc"]
    bins = np.arange(0, 20.1, binsize)
    v = "v15.1"
    iwc, cc = plot_ds[v].iwc.sel(time=sel_time), plot_ds[v].cloud_fraction.sel(time=sel_time)
    iwc_plot = iwc.where(cc > 0).where(cc == 0, iwc / cc)

    # 11 UTC
    pds = (iwc_plot
           .where(iwc_plot.time < pd.to_datetime(f"{date} 11:30"))
           .to_numpy()).flatten() * 1e6
    pds = pds[~np.isnan(pds)]
    ax.hist(
        pds,
        bins=bins,
        density=True,
        histtype="step",
        label=legend_labels[0] + f" (n={len(pds)})",
        color=cbc[0],
        lw=2,
    )

    # 12 UTC
    pds = (iwc_plot
           .where(iwc_plot.time > pd.to_datetime(f"{date} 11:30"))
           .to_numpy()).flatten() * 1e6
    pds = pds[~np.isnan(pds)]
    ax.hist(
        pds,
        bins=bins,
        density=True,
        histtype="step",
        label=legend_labels[1] + f" (n={len(pds)})",
        color=cbc[1],
        lw=2,
    )

    ax.legend()
    ax.grid()
    ax.text(0.03, 0.93,
            f"{text_labels[i]} {key.replace('1', ' 1')}",
            transform=ax.transAxes,
            bbox=dict(boxstyle="Round", fc="white"),
            )
    ax.set(title="",
           ylabel="",
           xlabel=f"Ice water content ({h.plot_units['iwc']})",
           ylim=ylims["iwc"])

axs[0].set(ylabel="Probability density function")

figname = f"{plot_path}/HALO_AC3_RF17_RF18_IFS_IWC_change_all_columns_11_to_12UTC.pdf"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot PDF of IWC from IFS above cloud for 11 and 12 UTC
plt.rc("font", size=9)
legend_labels = ["11 UTC", "12 UTC"]
binsizes = dict(iwc=1, reice=4)
_, axs = plt.subplots(1, 2, figsize=(17 * h.cm, 10 * h.cm), layout="constrained")
ylims = {"iwc": (0, 0.22), "reice": (0, 0.095)}
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
    print(f"RF 17/n{legend_labels[i]}: n={len(pds)}, mean={np.mean(pds):.2f}, median={np.median(pds):.2f}")
ax.grid()
ax.set(ylabel=f"Probability density function",
       xlabel=f"Ice water content ({h.plot_units['iwc']})",
       ylim=ylims["iwc"],
       xticks=range(0, 21, 5),
       title="")
ax.text(0.03, 0.93,
        f"(a) RF 17",
        transform=ax.transAxes,
        bbox=dict(boxstyle="Round", fc="white"),
        )

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
    print(f"RF 18/n{legend_labels[i]}: n={len(pds)}, mean={np.mean(pds):.2f}, median={np.median(pds):.2f}")
ax.legend()
ax.grid()
ax.set(ylabel=f"",
       xlabel=f"Ice water content ({h.plot_units['iwc']})",
       ylim=ylims["iwc"],
       xticks=range(0, 21, 5),
       title="")
ax.text(0.03, 0.93,
        f"(b) RF 18",
        transform=ax.transAxes,
        bbox=dict(boxstyle="Round", fc="white"),
        )

figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_IFS_iwc_11_vs_12_pdf_case_studies.pdf"
plt.savefig(figname, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# %% plot box plots of IWC from IFS above cloud for 11 and 12 UTC
plt.rc("font", size=9)
legend_labels = ["11 UTC", "12 UTC"]
binsizes = dict(iwc=1, reice=4)
_, axs = plt.subplots(1, 2, figsize=(17 * h.cm, 10 * h.cm), layout="constrained")
ylims = (0, 50)
# left panel - RF17 IWC
ax = axs[0]
binsize = binsizes["iwc"]
bins = np.arange(0, 20.1, binsize)
iwc_ifs_ls = list()
for t in ["2022-04-11 11:00", "2022-04-11 12:00"]:
    iwc_ifs, cc = ifs_ds_sel["RF17"].q_ice.sel(time=t), ifs_ds_sel["RF17"].cloud_fraction.sel(time=t)
    iwc_ifs = (iwc_ifs
               .where(cc > 0)
               .where(cc == 0, iwc_ifs / cc)
               .to_numpy().flatten() * 1e6)
    iwc_ifs_ls.append(iwc_ifs[~np.isnan(iwc_ifs)])

for i, pds in enumerate(iwc_ifs_ls):
    ax.boxplot(pds,
               positions=[i + 1],
               notch=True,
               patch_artist=True,
               boxprops={"facecolor": cbc[i]},
               medianprops={"color": cbc[2]})
ax.grid()
ax.set(ylabel=f"Ice water content ({h.plot_units['iwc']})",
       xlabel=f"IFS time step",
       ylim=ylims,
       title="")
ax.text(0.03, 0.93,
        f"(a) RF 17",
        transform=ax.transAxes,
        bbox=dict(boxstyle="Round", fc="white"),
        )

# right panel - RF18 IWC
ax = axs[1]
iwc_ifs_ls = list()
for t in ["2022-04-12 11:00", "2022-04-12 12:00"]:
    iwc_ifs, cc = ifs_ds_sel["RF18"].q_ice.sel(time=t), ifs_ds_sel["RF18"].cloud_fraction.sel(time=t)
    iwc_ifs = (iwc_ifs
               .where(cc > 0)
               .where(cc == 0, iwc_ifs / cc)
               .to_numpy().flatten() * 1e6)
    iwc_ifs_ls.append(iwc_ifs[~np.isnan(iwc_ifs)])

for i, pds in enumerate(iwc_ifs_ls):
    ax.violinplot(pds,
               positions=[i + 1],)
               # notch=True,
               # patch_artist=True,
               # boxprops={"facecolor": cbc[i]},
               # medianprops={"color": cbc[2]})
ax.grid()
ax.set(ylabel=f"Ice water content ({h.plot_units['iwc']})",
       xlabel=f"IFS time step",
       # xticklabels=legend_labels,
       ylim=ylims,
       title="")
ax.text(0.03, 0.93,
        f"(b) RF 18",
        transform=ax.transAxes,
        bbox=dict(boxstyle="Round", fc="white"),
        )

figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_IFS_iwc_11_vs_12_box_plot_case_studies.pdf"
plt.savefig(figname, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# %% vizualise gridpoints from IFS selected for case study areas
key = "RF18"
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

# %% print statistics of IWC comparison
iwc_df = pd.DataFrame(stats, columns=["Flight", "Column", "Time", "Statistic", "Value"])
print(iwc_df)
iwc_df.to_csv(f"{plot_path}/IFS_IWC_statistics.csv")

# %% plot sea ice fraction along fligh track for case study period
key = "RF18"
plot_ds = ecrad_dicts[key]["v15.1"].ci.sel(time=slices[key]["case"])

_, ax = plt.subplots(figsize=h.figsize_wide)
plot_ds.plot(x="time", ax=ax)
plt.show()
plt.close()

# %% print statistics of sea ice cover for the case studies
for key in keys:
    plot_ds = ecrad_dicts[key]["v15.1"].ci.sel(time=slices[key]["case"])
    print(f"{key} Sea ice fraction for case study\n"
          f"Max: {plot_ds.max():.3f}\n"
          f"Min: {plot_ds.min():.3f}\n"
          f"Mean: {plot_ds.mean():.3f}")

# %% calculate the broadband surface albedo by a weighted average
print(h.ci_bands)
dates = dict(RF17="2022-04-11", RF18="2022-04-12")
for key in keys:
    ds = ecrad_dicts[key]["v15.1"].sel(half_level=0.5, time=f"{dates[key]} 12:00")
    spectral_weights = ds.spectral_flux_dn_sw / ds.flux_dn_sw
    weights = np.empty(6)
    weights[0] = spectral_weights.sel(band_sw=13)
    weights[1] = spectral_weights.sel(band_sw=[12, 11]).sum()
    weights[2] = spectral_weights.sel(band_sw=10)
    weights[3] = spectral_weights.sel(band_sw=[9, 8]).sum()
    weights[4] = spectral_weights.sel(band_sw=[7, 6, 5, 4, 3]).sum()
    weights[5] = spectral_weights.sel(band_sw=[2, 1, 14]).sum()

    ecrad_dicts[key]["v15.1"]["bb_sw_albedo"] = (ecrad_dicts[key]["v15.1"].sw_albedo * weights).sum(dim="sw_albedo_band")
    bacardi_bb_albedo = bacardi_ds[key]['reflectivity_solar'].sel(time=slices[key]["below"])
    ecrad_bb_albedo = ecrad_dicts[key]["v15.1"]["bb_sw_albedo"].sel(time=slices[key]["case"])

    print(f"{key}\n"
          f"IFS broadband albedo:\n"
          f"Mean: {ecrad_bb_albedo.mean():.3f}\n"
          f"Max: {ecrad_bb_albedo.max():.3f}\n"
          f"Min: {ecrad_bb_albedo.min():.3f}\n"
          f"BACARDI broadband albedo:\n"
          f"Mean: {bacardi_bb_albedo.mean():.3f}\n"
          f"Max: {bacardi_bb_albedo.max():.3f}\n"
          f"Min: {bacardi_bb_albedo.min():.3f}")

# %% testing
