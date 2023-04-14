#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 27-03-2023

Analysis for poster at EGU General Assembly 2023-04-25

- trajectories overview plot with dropsonde profiles
- radar lidar mask for whole case study period of RF17



"""

# %% module import
import pylim.helpers as h
from pylim import reader
from pylim import meteorological_formulas as met
import pylim.halo_ac3 as meta
import ac3airborne
from ac3airborne.tools import flightphase
import sys

sys.path.append('./larda')
from larda.pyLARDA.spec2mom_limrad94 import despeckle
from metpy.units import units
from metpy.calc import density
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib import colors
from matplotlib.collections import LineCollection
from matplotlib import patheffects
from matplotlib.patches import Patch
import cmasher as cmr
import cartopy.crs as ccrs
import xarray as xr
import pandas as pd
from tqdm import tqdm
import logging

mpl.use('module://backend_interagg')

log = logging.getLogger("pylim")
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

cm = 1 / 2.54
cbc = h.get_cb_friendly_colors()

# %% set up paths and meta data
campaign = "halo-ac3"
key = "RF17"
flight = meta.flight_names[key]
date = flight[9:17]
smart_path = h.get_path("calibrated", flight, campaign)
ecrad_path = f"{h.get_path('ecrad', campaign=campaign)}/{date}"
libradtran_path = h.get_path("libradtran", flight, campaign)
bahamas_path = h.get_path("bahamas", flight, campaign)
bacardi_path = h.get_path("bacardi", flight, campaign)
dropsonde_path = f"{h.get_path('dropsondes', flight, campaign)}/.."
trajectory_path = f"{h.get_path('trajectories', campaign=campaign)}/selection_CC_and_altitude"
era5_path = h.get_path("era5", flight, campaign)
radar_path = h.get_path("hamp_mira", flight, campaign)
lidar_path = h.get_path("wales", flight, campaign)
plot_path = f"{h.get_path('plot', campaign=campaign)}/{flight}/EGU2023"

# file names
calibrated_file = f"HALO-AC3_HALO_SMART_spectral-irradiance-Fdw_{date}_{key}_v1.0.nc"
bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{key}_v1.nc"
bacardi_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1.nc"
bacardi_res_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1_1s.nc"
ecrad_v1 = f"ecrad_merged_inout_{date}_v1.nc"
ecrad_v8 = f"ecrad_merged_output_{date}_v8.nc"
ecrad_v10 = f"ecrad_merged_output_{date}_v10.nc"
ecrad_inputv2 = f"ecrad_merged_input_{date}_v2.nc"
ecrad_inputv3 = f"ecrad_merged_input_{date}_v3.nc"
dropsonde_file = f"HALO-AC3_HALO_Dropsondes_quickgrid_{date}.nc"
era5_files = [os.path.join(era5_path, f) for f in os.listdir(era5_path) if f"P{date}" in f]
era5_files.sort()
radar_file = f"radar_{date}_v1.6.nc"
lidar_file = f"HALO-AC3_HALO_WALES_bsrgl_{date}_{key}_V1_1s.nc"

# set options and credentials for HALO-AC3 cloud and intake catalog
kwds = {'simplecache': dict(same_names=True)}
credentials = {"user": os.environ.get("AC3_CLOUD_USER"), "password": os.environ.get("AC3_CLOUD_PASSWORD")}
cat = ac3airborne.get_intake_catalog()

# %% get flight segmentation and select below and above cloud section
fl_segments = ac3airborne.get_flight_segments()["HALO-AC3"]["HALO"][f"HALO-AC3_HALO_{key}"]
segments = flightphase.FlightPhaseFile(fl_segments)
above_cloud, below_cloud = dict(), dict()
above_cloud["start"] = segments.select("name", "high level 7")[0]["start"]
above_cloud["end"] = segments.select("name", "high level 8")[0]["end"]
below_cloud["start"] = segments.select("name", "high level 9")[0]["start"]
below_cloud["end"] = segments.select("name", "high level 10")[0]["end"]
above_slice = slice(above_cloud["start"], above_cloud["end"])
below_slice = slice(below_cloud["start"], below_cloud["end"])
# above cloud time with thin cirrus below
sel_time = slice(above_cloud["start"], pd.to_datetime("2022-04-11 11:04"))

# %% read in data
smart_ds = xr.open_dataset(f"{smart_path}/{calibrated_file}")
bahamas = reader.read_bahamas(f"{bahamas_path}/{bahamas_file}")
bacardi_ds = xr.open_dataset(f"{bacardi_path}/{bacardi_file}")
bacardi_ds_res = xr.open_dataset(f"{bacardi_path}/{bacardi_res_file}")
ecrad_ds_v1 = xr.open_dataset(f"{ecrad_path}/{ecrad_v1}")
ecrad_ds_v8 = xr.open_dataset(f"{ecrad_path}/{ecrad_v8}")
ecrad_ds_v10 = xr.open_dataset(f"{ecrad_path}/{ecrad_v10}")
ecrad_ds_inputv2 = xr.open_dataset(f"{ecrad_path}/{ecrad_inputv2}")
ecrad_ds_inputv3 = xr.open_dataset(f"{ecrad_path}/{ecrad_inputv3}")
dropsonde_ds = xr.open_dataset(f"{dropsonde_path}/{dropsonde_file}")
dropsonde_ds["alt"] = dropsonde_ds.alt / 1000  # convert altitude to km
ins = cat["HALO-AC3"]["HALO"]["GPS_INS"][f"HALO-AC3_HALO_{key}"](storage_options=kwds, **credentials).to_dask()
era5_ds = xr.open_mfdataset(era5_files).sel(time=f"2022-04-11T12:00")
radar_ds = xr.open_dataset(f"{radar_path}/{radar_file}")
# filter -888 values
radar_ds["dBZg"] = radar_ds.dBZg.where(np.isnan(radar_ds.radar_flag) & ~radar_ds.dBZg.isin(-888))
lidar_ds_res = xr.open_dataset(f"{lidar_path}/{lidar_file}").reset_coords("altitude")
# convert lidar data to radar convention: [time, height], ground = 0m
lidar_ds_res = lidar_ds_res.rename(range="height").transpose("time", "height")
lidar_height = lidar_ds_res.height
# flip height coordinate
lidar_ds_res = lidar_ds_res.assign_coords(height=np.flip(lidar_height)).isel(height=slice(None, None, -1))

# %% plotting meta
time_extend = pd.to_timedelta((ins.time[-1] - ins.time[0]).values)  # get time extend for x-axis labeling
time_extend_cs = below_cloud["end"] - above_cloud["start"]  # time extend for case study
time_extend_ac = (above_cloud["end"] - above_cloud["start"])
time_extend_bc = below_cloud["end"] - below_cloud["start"]
case_slice = slice(above_cloud["start"], below_cloud["end"])
plt.rcdefaults()
h.set_cb_friendly_colors()

# %% prepare ecrad data
ecrad_ds = ecrad_ds_v1.sel(column=16,
                           drop=True)  # select center column which corresponds to grid cell closest to aircraft
ecrad_ds_v8 = xr.merge([ecrad_ds_v8, ecrad_ds_inputv2])
ecrad_ds_v10 = xr.merge([ecrad_ds_v10, ecrad_ds_inputv3])
ecrad_dict = dict(v1=ecrad_ds, v8=ecrad_ds_v8, v10=ecrad_ds_v10)

for k in ecrad_dict:
    ds = ecrad_dict[k].copy()
    ds = ds.assign_coords({"band_sw": range(1, 15), "band_lw": range(1, 17), "sw_albedo_band": range(1, 7)})

    # replace default values with nan
    ds["re_ice"] = ds.re_ice.where(ds.re_ice != 5.19616e-05, np.nan)
    ds["re_liquid"] = ds.re_liquid.where(ds.re_liquid != 4.000001e-06, np.nan)

    # convert kg/kg to kg/m³
    air_density = density(ds.pressure_full * units.Pa, ds.t * units.K, ds.q * units("kg/kg"))
    ds["iwc"] = ds["q_ice"] * units("kg/kg") * air_density

    # calculate radiative effect
    ds["cre_sw"] = (ds.flux_dn_sw - ds.flux_up_sw) - (ds.flux_dn_sw_clear - ds.flux_up_sw_clear)
    ds["cre_lw"] = (ds.flux_dn_lw - ds.flux_up_lw) - (ds.flux_dn_lw_clear - ds.flux_up_lw_clear)
    ds["cre_total"] = ds["cre_sw"] + ds["cre_lw"]

    ecrad_dict[k] = ds.copy()

# %% get height level of actual flight altitude in ecRad model, this determines only the index of the level
ds = ecrad_dict["v1"]
bahamas_tmp = bahamas.sel(time=ds.time, method="nearest")
ecrad_timesteps = len(ds.time)
aircraft_height_level = np.zeros(ecrad_timesteps)

for i in tqdm(range(ecrad_timesteps)):
    aircraft_height_level[i] = h.arg_nearest(ds["pressure_hl"][i, :].values, bahamas_tmp.PS[i].values * 100)

aircraft_height_level = aircraft_height_level.astype(int)
height_level_da = xr.DataArray(aircraft_height_level, dims=["time"], coords={"time": ds.time})
aircraft_height = ds["pressure_hl"].isel(half_level=height_level_da)
aircraft_height_da = xr.DataArray(aircraft_height, dims=["time"], coords={"time": ds.time},
                                  name="aircraft_height", attrs={"unit": "Pa"})

# %% era5 calculate pressure on all model levels
pressure = era5_ds.hyam + era5_ds.hybm * era5_ds.PS * 100
# select only relevant model levels and swap dimension names
era5_ds["pressure"] = pressure.sel(nhym=slice(39, 137)).swap_dims({"nhym": "lev"})

# %% calculate radiative effect BACARDI
ds = ecrad_dict["v1"].isel(half_level=height_level_da, drop=True)
cre_solar = (bacardi_ds_res.F_down_solar - bacardi_ds_res.F_up_solar) - (ds.flux_dn_sw_clear - ds.flux_up_sw_clear)
cre_ter = (bacardi_ds_res.F_down_terrestrial - bacardi_ds_res.F_up_terrestrial) - (
            ds.flux_dn_lw_clear - ds.flux_up_lw_clear)
cre_total = cre_solar + cre_ter

# %% create radar mask and despeckle radar data
radar_ds["mask"] = ~np.isnan(radar_ds["dBZg"])
radar_mask = ~radar_ds["mask"].values
for n in tqdm(range(2)):
    # despeckle 2 times
    radar_mask = despeckle(radar_mask, 50)  # despeckle again
    # plt.pcolormesh(radar_mask.T)
    # plt.title(n + 1)
    # plt.savefig(f"{plot_path}/tmp/radar_despeckle_{n + 1}.png")
    # plt.close()

radar_ds["spklmask"] = (["time", "height"], radar_mask)

# %% use despeckle the reverse way to fill signal gaps in radar data and add it as a mask
radar_mask = ~radar_ds["spklmask"].values
n = 0
for n in tqdm(range(17)):
    # fill gaps 17 times
    radar_mask = despeckle(radar_mask, 50)  # fill gaps again
    # plt.pcolormesh(radar_mask.T)
    # plt.title(n + 1)
    # plt.savefig(f"{plot_path}/tmp/radar_fill_gaps_{n + 1}.png")
    # plt.close()

radar_ds["fill_mask"] = (["time", "height"], radar_mask)

# %% despeckle lidar data and add a speckle mask to the data set
lidar_ds_res["mask"] = ~(lidar_ds_res.backscatter_ratio > 1.1)
lidar_mask = lidar_ds_res["mask"].values
n = 0
for n in tqdm(range(17)):
    # despeckle 17 times
    lidar_mask = despeckle(lidar_mask, 50)  # despeckle again
    # plt.pcolormesh(lidar_mask.T)
    # plt.title(n + 1)
    # plt.savefig(f"{plot_path}/tmp/despeckle_{n + 1}.png")
    # plt.close()

lidar_ds_res["spklmask"] = (["time", "height"], lidar_mask)

# %% use despeckle the reverse way to fill signal gaps in lidar data and add it as a mask
lidar_mask = ~lidar_ds_res["spklmask"].values
n = 0
for n in tqdm(range(17)):
    # fill gaps 17 times
    lidar_mask = despeckle(lidar_mask, 50)  # fill gaps again
    # plt.pcolormesh(lidar_mask.T)
    # plt.title(n + 1)
    # plt.savefig(f"{plot_path}/tmp/fill_gaps_{n + 1}.png")
    # plt.close()

lidar_ds_res["fill_mask"] = (["time", "height"], lidar_mask)

# %% despeckle fill mask
lidar_mask = ~lidar_ds_res["fill_mask"].values
for n in tqdm(range(8)):
    # plt.pcolormesh(lidar_mask.T)
    # plt.title(n + 1)
    # plt.savefig(f"{plot_path}/tmp/fill_gaps_despeckled_{n + 1}.png")
    # plt.close()
    lidar_mask = despeckle(lidar_mask, 50)  # despeckle again

lidar_ds_res["fill_mask"] = (["time", "height"], ~lidar_mask)

# %% interpolate lidar data onto radar range resolution
new_range = radar_ds.height.values
lidar_ds_res_r = lidar_ds_res.interp(height=new_range)

# %% combine radar and lidar mask
lidar_mask = lidar_ds_res_r["fill_mask"]
lidar_mask = lidar_mask.where(lidar_mask == 0, 2)
radar_lidar_mask = radar_ds["mask"] + lidar_mask

# %% plot map of trajectories with surface pressure, flight track, dropsonde locations and high cloud cover + RH_ice profiles from radiosonde
cmap = mpl.colormaps["tab20b_r"]([20, 20, 0, 3, 4, 7, 8, 11, 12, 15, 16, 19])
cmap[:2] = mpl.colormaps["tab20c"]([7, 4])
cmap = mpl.colors.ListedColormap(cmap)
plt_sett = {
    'TIME': {
        'label': 'Time Relative to Release (h)',
        'norm': plt.Normalize(-72, 0),
        'ylim': [-72, 0],
        'cmap_sel': cmap,
        'cmap_ticks': np.arange(-72, 0.1, 12)
    }
}
var_name = "TIME"
data_crs = ccrs.PlateCarree()
h.set_cb_friendly_colors()

plt.rc("font", size=16)
fig = plt.figure(figsize=(42 * cm, 18 * cm))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

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
pressure_levels = np.arange(900, 1125, 5)
E5_press = era5_ds.MSL / 100
cp = ax.contour(E5_press.lon, E5_press.lat, E5_press, levels=pressure_levels, colors='k', linewidths=1,
                linestyles='solid', alpha=1, transform=data_crs)
cp.clabel(fontsize=8, inline=1, inline_spacing=4, fmt='%i hPa', rightside_up=True, use_clabeltext=True)

# add seaice edge
ci_levels = [0.8]
E5_ci = era5_ds.CI
cci = ax.contour(E5_ci.lon, E5_ci.lat, E5_ci, ci_levels, transform=data_crs, linestyles="--", colors="#332288",
                 linewidths=3)

# add high cloud cover
E5_cc = era5_ds.CC.where(era5_ds.pressure < 60000, drop=True).sum(dim="lev")
# E5_cc = E5_cc.where(E5_cc > 1)
ax.contourf(E5_cc.lon, E5_cc.lat, E5_cc, levels=24, transform=data_crs, cmap="Blues", alpha=1)

# plot trajectories - 11 April
header_line = [2]  # header-line of .1 files is always line #2 (counting from 0)
date_h = f"20220411_07"
# get filenames
fname_traj = "traj_CIRR_HALO_" + date_h + ".1"
try:
    trajs = np.loadtxt(f"{trajectory_path}/{fname_traj}", dtype="f", skiprows=5)
    times = trajs[:, 0]
    # generate object to only load specific header line
    gen = h.generate_specific_rows(f"{trajectory_path}/{fname_traj}", userows=header_line)
    header = np.loadtxt(gen, dtype="str", unpack=True)
    header = header.tolist()  # convert to list

    # convert to lower char.
    for j in range(len(header)):
        header[j] = header[j].upper()  # convert to lower

    var_index = header.index(var_name.upper())
except (StopIteration, IOError) as e:
    print("\t>>>Skipping file, probably empty<<<")
else:
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
                        alpha=1, transform=data_crs)
    # Set the values used for colormapping
    lc.set_array(var)
    if int(traj_num) == 1:
        lc.set_linewidth(5)
    elif int(traj_num) >= 2:
        lc.set_linewidth(1)
    line = ax.add_collection(lc)

if var_name != "TIME":
    plt.colorbar(line, ax=ax).set_label(label=plt_sett[var_name]['label'])
if var_name == "TIME":
    plt.colorbar(line, ax=ax, pad=0.01,
                 ticks=np.arange(-120, 0.1, 12)).set_label(label=plt_sett[var_name]['label'])

# plot flight track - 11 April
track_lons, track_lats = ins["lon"], ins["lat"]
ax.scatter(track_lons[::10], track_lats[::10], c="#888888", alpha=1, marker=".", s=4, zorder=400,
           label='HALO flight track', transform=data_crs, linestyle="solid")

# plot dropsonde locations - 11 April
for i in range(dropsonde_ds.lon.shape[0]):
    launch_time = pd.to_datetime(dropsonde_ds.launch_time[i].values)
    x, y = dropsonde_ds.lon[i].mean().values, dropsonde_ds.lat[i].mean().values
    cross = ax.plot(x, y, "x", color="orangered", markersize=12, label="Dropsonde", transform=data_crs,
                    zorder=450)
    ax.text(x, y, f"{launch_time:%H:%M}", c="k", fontsize=12, transform=data_crs, zorder=500,
            path_effects=[patheffects.withStroke(linewidth=0.25, foreground="white")])

# make legend for flight track and dropsondes - 11 April
handles = [plt.plot([], ls="-", color="#888888", lw=3)[0],  # flight track
           cross[0],  # dropsondes
           plt.plot([], ls="--", color="#332288", lw=3)[0],  # sea ice edge
           Patch(facecolor="royalblue")]  # cloud cover
labels = ["HALO flight track", "Dropsonde", "Sea Ice Edge", "High Cloud Cover\nat 12:00 UTC"]
ax.legend(handles=handles, labels=labels, framealpha=1, loc=2, fontsize=14)

ax.text(-72.5, 73, "(a)", size=18, transform=data_crs, ha="center", va="center",
        bbox=dict(boxstyle="round", ec="grey", fc="white"))

# plot dropsonde profiles in row 1 and column 2
rh_ice = met.relative_humidity_water_to_relative_humidity_ice(dropsonde_ds.rh, dropsonde_ds.T - 273.15)
labels = [f"{lt.replace('2022-04-11T', '')} UTC" for lt in np.datetime_as_string(rh_ice.launch_time.values, unit="m")]
ax = fig.add_subplot(gs[0, 1])
rh_ice.plot.line(y="alt", alpha=0.5, label=labels, lw=2, ax=ax)
rh_ice.mean(dim="launch_time").plot(y="alt", lw=3, label="Mean", c="k", ax=ax)

# plot vertical line at 100%
ax.axvline(x=100, color="#661100", lw=3)
ax.set_xlabel("Relative Humidity over Ice (%)")
ax.set_ylabel("Altitude (km)")
ax.grid()
ax.legend(bbox_to_anchor=(1, 1.01), loc="upper left", fontsize=14)
ax.text(0.1, 0.95, "(b)", size=18, transform=ax.transAxes, ha="center", va="center",
        bbox=dict(boxstyle="round", ec="grey", fc="white"))

plt.tight_layout()
figname = f"{plot_path}/{flight}_trajectories_dropsonde_rh_ice.png"
plt.savefig(figname, format='png', dpi=300, bbox_inches='tight')
print(figname)
plt.close()

# %% plot combined radar and lidar mask for case study
plot_ds = radar_lidar_mask.sel(time=sel_time)
plot_ds["height"] = plot_ds.height / 1000
clabel = [x[0] for x in h._CLABEL["detection_status"]]
cbar = [x[1] for x in h._CLABEL["detection_status"]]
clabel = list([clabel[-1], clabel[5], clabel[1], clabel[3]])
cbar = list([cbar[-1], cbar[5], cbar[1], cbar[3]])
cmap = colors.ListedColormap(cbar)
plt.rc("font", size=16)
_, ax = plt.subplots(figsize=(41 * cm, 10 * cm))
pcm = plot_ds.plot(x="time", y="height", cmap=cmap, vmin=-0.5, vmax=len(cbar) - 0.5,
                   cbar_kwargs=dict(pad=0.01))
pcm.colorbar.set_ticks(np.arange(len(clabel)), labels=clabel)
ax.plot(bahamas.time, bahamas["IRS_ALT"] / 1000, color="k", label="HALO altitude")
ax.legend(loc=2)
h.set_xticks_and_xlabels(ax, time_extend_bc)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Altitude (km)")
plt.tight_layout()
figname = f"{plot_path}/{flight}_radar_lidar_mask_cs.png"
plt.savefig(figname, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# %% set ecRad plotting values
var = "re_ice"
v = "v8"
band = None

# %% prepare data set for plotting
band_str = f"_band{band}" if band is not None else ""

# kwarg dicts
alphas = dict()
ct_fontsize = dict()
ct_lines = dict(ciwc=[1, 5, 10, 15], cswc=[1, 5, 10, 15], q_ice=[1, 5, 10, 15], clwc=[1, 5, 10, 15],
                iwc=[1, 5, 10, 15])
linewidths = dict()
robust = dict(iwc=False)
cb_ticks = dict()
vmaxs = dict()
vmins = dict(iwp=0)
xlabels = dict(v1="v1", v8="v8", v10="v10", diffv8="Difference v1 - v8", diffv10="Difference v1 - v10")

# set kwargs
alpha = alphas[var] if var in alphas else 1
cmap = h.cmaps[var] if var in h.cmaps else cmr.rainforest
cmap = plt.get_cmap(cmap).copy()
cmap.set_bad(color="white")
ct_fs = ct_fontsize[var] if var in ct_fontsize else 8
lines = ct_lines[var] if var in ct_lines else None
lw = linewidths[var] if var in linewidths else 1
norm = h.norms[var] if var in h.norms else None
robust = robust[var] if var in robust else True
ticks = cb_ticks[var] if var in cb_ticks else None
if norm is None:
    vmax = vmaxs[var] if var in vmaxs else None
    vmin = vmins[var] if var in vmins else None
else:
    vmax, vmin = None, None

if "diff" in v:
    cmap = cmr.fusion
    norm = colors.TwoSlopeNorm(vcenter=0)

# prepare ecrad dataset for plotting
sf = h.scale_factors[var] if var in h.scale_factors else 1
if v == "diffv8":
    # calculate difference between simulations
    ds = ecrad_dict["v8"].copy()
    ecrad_ds_diff = ecrad_dict["v1"][var] - ds[var]
    ecrad_plot = ecrad_ds_diff.where((ds[var] != 0) | (~np.isnan(ds[var]))) * sf
elif v == "diffv10":
    # calculate difference between simulations
    ds = ecrad_dict["v10"].copy()
    ecrad_ds_diff = ecrad_dict["v1"][var] - ds[var]
    ecrad_plot = ecrad_ds_diff.where((ds[var] != 0) | (~np.isnan(ds[var]))) * sf
else:
    ds = ecrad_dict[v]
    ecrad_plot = ds[var] * sf

# add new z axis mean pressure altitude
if "half_level" in ecrad_plot.dims:
    new_z = ds["press_height_hl"].mean(dim="time") / 1000
else:
    new_z = ds["press_height_full"].mean(dim="time") / 1000

ecrad_plot_new_z = list()
for t in tqdm(ecrad_plot.time, desc="New Z-Axis"):
    tmp_plot = ecrad_plot.sel(time=t)
    if "half_level" in tmp_plot.dims:
        tmp_plot = tmp_plot.assign_coords(
            half_level=ds["press_height_hl"].sel(time=t, drop=True).to_numpy() / 1000)
        tmp_plot = tmp_plot.rename(half_level="height")

    else:
        tmp_plot = tmp_plot.assign_coords(
            level=ds["press_height_full"].sel(time=t, drop=True).to_numpy() / 1000)
        tmp_plot = tmp_plot.rename(level="height")

    tmp_plot = tmp_plot.interp(height=new_z.to_numpy())
    ecrad_plot_new_z.append(tmp_plot)

ecrad_plot = xr.concat(ecrad_plot_new_z, dim="time")
# filter very low to_numpy()
ecrad_plot = ecrad_plot.where(np.abs(ecrad_plot) > 0.001)

# select time height slice
time_sel = sel_time
if len(ecrad_plot.dims) > 2:
    dim3 = "band_sw"
    dim3 = dim3 if dim3 in ecrad_plot.dims else None
    ecrad_plot = ecrad_plot.sel({"time": time_sel, "height": slice(12, 0), f"{dim3}": band})
else:
    ecrad_plot = ecrad_plot.sel(time=time_sel, height=slice(12, 0))

time_extend = pd.to_timedelta((ecrad_plot.time[-1] - ecrad_plot.time[0]).to_numpy())

# %% plot 2D IFS variables along flight track
_, ax = plt.subplots(figsize=(22 * cm, 12 * cm))
# ecrad 2D field
ecrad_plot.plot(x="time", y="height", cmap=cmap, ax=ax, robust=robust, vmin=vmin, vmax=vmax, alpha=alpha, norm=norm,
                cbar_kwargs={"pad": 0.04, "label": f"{h.cbarlabels[var]} ({h.plot_units[var]})",
                             "ticks": ticks})
if lines is not None:
    # add contour lines
    ct = ax.contour(ecrad_plot.time, ecrad_plot.height, ecrad_plot.to_numpy().T, levels=lines, linestyles="--",
                    colors="k",
                    linewidths=lw)
    ct.clabel(fontsize=ct_fs, inline=1, inline_spacing=0, fmt='%i', rightside_up=True, use_clabeltext=True)

ax.set_title(f"")
ax.set_ylabel("Altitude (km)")
ax.set_xlabel("Time (UTC)")
h.set_xticks_and_xlabels(ax, time_extend)
plt.tight_layout()
figname = f"{plot_path}/{flight}_ecrad_{v}_{var}{band_str}_along_track.png"
plt.savefig(figname, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# %% plot histogram of particle sizes
time_sel = sel_time
plot_v1 = (ecrad_dict["v1"].re_ice.sel(time=time_sel).to_numpy() * 1e6).flatten()
plot_v8 = (ecrad_dict["v8"].re_ice.sel(time=time_sel).to_numpy() * 1e6).flatten()
t1, t2 = time_sel.start, time_sel.stop
binsize = 2
bins = np.arange(10, 71, binsize)
_, ax = plt.subplots(figsize=(22 * cm, 12 * cm))
hist = ax.hist(plot_v1, bins=bins, label="IFS", histtype="step", lw=4)
ax.hist(plot_v8, bins=bins, label="VarCloud", histtype="step", lw=4)
ax.legend()
ax.text(0.78, 0.67, f"Binsize: {binsize} $\mu$m", transform=ax.transAxes, bbox=dict(boxstyle="round", fc="white"))
ax.grid()
ax.set(xlabel=r"Ice effective radius ($\mu$m)", ylabel="Number of Occurrence")
plt.tight_layout()
figname = f"{plot_path}/{flight}_ecrad_v1_v8_re_ice_psd.png"
plt.savefig(figname, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# %% plot BACARDI and ecRad below cloud
ds10, ds1 = ecrad_dict["v10"], ecrad_dict["v1"]
time_sel = slice(ds10.time[0], ds10.time[-1])
hl_sel = height_level_da.sel(time=ds10.time, method="nearest").assign_coords(time=ds10.time)
ecrad_plot = ds10.isel(half_level=hl_sel)
ecrad_plot1 = ds1.isel(half_level=height_level_da).sel(time=time_sel)
bacardi_lat = bacardi_ds["lat"].sel(time=time_sel)
bacardi_lon = bacardi_ds["lon"].sel(time=time_sel)
ecrad_lat = bacardi_lat.reindex(time=ecrad_plot.time.to_numpy(), method="ffill")
ecrad1_lat = bacardi_lat.reindex(time=ecrad_plot1.time.to_numpy(), method="ffill")
bacardi_plot = bacardi_ds["F_down_solar"].sel(time=time_sel)
bacardi_error = bacardi_plot * 0.03

plt.rc("font", size=19)
_, ax = plt.subplots(figsize=(41 * cm, 21 * cm))
ax.plot(bacardi_plot.time, bacardi_plot, label="$F_{\downarrow , solar}$ BACARDI", lw=4)
ax.fill_between(bacardi_plot.time, bacardi_plot + bacardi_error, bacardi_plot - bacardi_error, color=cbc[0],
                alpha=0.5, label="BACARDI uncertainty")
ax.plot(ecrad_plot.time, ecrad_plot.flux_dn_sw, label="$F_{\downarrow , solar}$ ecRad Varcloud", lw=4)
ax.plot(ecrad_plot1.time, ecrad_plot1.flux_dn_sw, marker="o", label="$F_{\downarrow , solar}$ ecRad IFS", lw=4,
        markersize=10, color=cbc[3])
ax.legend(loc=4)
ax.grid()
h.set_xticks_and_xlabels(ax, pd.to_timedelta((ds10.time[-1] - ds10.time[0]).to_numpy()))
ax.set_ylabel(f"Broadband Irradiance ({h.plot_units['flux_dn_sw']})")
ax.set_xlabel("Time (UTC)", labelpad=-15)

# add latitude and longitude axis
axs2 = ax.twiny(), ax.twiny()
xlabels = ["Latitude (°N)", "Longitude (°E)"]
for i, ax2 in enumerate(axs2):
    # Move twinned axis ticks and label from top to bottom
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")

    # Offset the twin axis below the host
    ax2.spines["bottom"].set_position(("axes", -0.09 * (i + 1)))

    # Turn on the frame for the twin axis, but then hide all
    # but the bottom spine
    ax2.set_frame_on(True)
    ax2.patch.set_visible(False)

    for sp in ax2.spines.values():
        sp.set_visible(False)
    ax2.spines["bottom"].set_visible(True)

    ticklocs = ax.xaxis.get_ticklocs()  # get tick locations
    ts = pd.to_datetime(mpl.dates.num2date(ticklocs)).tz_localize(None)  # convert matplotlib dates to pandas dates
    xticklabels = [bacardi_lat.sel(time=ts).to_numpy(), bacardi_lon.sel(time=ts).to_numpy()]  # get xticklables
    ax2.set_xticks(np.linspace(0.05, 0.95, len(ts)))
    ax2.set_xticklabels(np.round(xticklabels[i], 2))
    ax2.set_xlabel(xlabels[i], labelpad=-20)

plt.tight_layout()
figname = f"{plot_path}/{flight}_ecrad_varcloud_BACARDI_F_down_solar_below_cloud.png"
plt.savefig(figname, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# %% plot radiative effect
ds10, ds1 = ecrad_dict["v10"], ecrad_dict["v1"]
ds10_res = ds10.resample(time="1Min").first()
time_sel = slice(ds10.time[0], ds10.time[-1])
hl_sel = height_level_da.sel(time=ds10.time, method="nearest").assign_coords(time=ds10.time)
ecrad_plot = ds10["cre_total"].isel(half_level=hl_sel)
ecrad_plot1 = ds1["cre_total"].isel(half_level=height_level_da).sel(time=time_sel)
ecrad_plot2 = ds10_res["cre_total"].isel(half_level=height_level_da.sel(time=ds10_res.time))
bacardi_plot = cre_total.sel(time=time_sel)

plt.rc("font", size=19)
_, ax = plt.subplots(figsize=(41 * cm, 21 * cm))
ax.plot(bacardi_plot.time, bacardi_plot, label="$CRE_{total}$ BACARDI", marker="o", ls="", markersize=10)
ax.plot(ecrad_plot.time, ecrad_plot, label="$CRE_{total}$ ecRad Varcloud", lw=4)
ax.plot(ecrad_plot2.time, ecrad_plot2, marker="o", label="$CRE_{total}$ ecRad Varcloud 1Min", lw=4,
        markersize=10)
ax.plot(ecrad_plot1.time, ecrad_plot1, marker="o", label="$CRE_{total}$ ecRad IFS", lw=4, markersize=10)
ax.legend(loc=4)
ax.grid()
h.set_xticks_and_xlabels(ax, pd.to_timedelta((ds10.time[-1] - ds10.time[0]).to_numpy()))
ax.set(xlabel="Time (UTC)", ylabel=f"Radiative Effect ({h.plot_units['flux_dn_sw']})")
plt.tight_layout()
figname = f"{plot_path}/{flight}_ecrad_varcloud_BACARDI_cre_total_below_cloud.png"
plt.savefig(figname, dpi=300, bbox_inches="tight")
plt.show()
plt.close()
