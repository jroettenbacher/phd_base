#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 01.02.2024

Here the data used in thesis_figures.py is prepared.
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
save_path = "C:/Users/Johannes/Documents/Doktor/manuscripts/_arctic_cirrus/figures"
trajectory_path = f"{h.get_path('trajectories', campaign=campaign)}/selection_CC_and_altitude"
keys = ["RF17", "RF18"]
ecrad_versions = [f"v{x}" for x in [13, 13.1, 13.2, 15, 15.1, 16, 17, 18, 18.1, 19,
                                    19.1, 20, 21, 28, 29, 30.1, 31.1, 32.1, 33,
                                    34, 35, 36, 37, 38]]

# %% read in data
(
    bahamas_ds, bacardi_ds, bacardi_ds_res, ecrad_dicts, varcloud_ds, above_clouds, below_clouds,
    slices, ecrad_orgs, ifs_ds, ifs_ds_sel, dropsonde_ds, albedo_dfs
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
    # interpolate simulation onto BACARDI time
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

