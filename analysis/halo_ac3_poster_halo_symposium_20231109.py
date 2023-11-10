#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 10-07-2023

Analysis and plots for poster at HALO symposium 09-10-2023

- case study map plot with trajectories, IFS high cloud cover, ERA5 surface pressure and sea ice edge
- Above cloud lidar plot and below cloud BACARDI solar downward irradiance
- time series of below cloud transmissivity from BACARDI and ecRad simulations with different ice optic parameterizations
- PDFs of solar transmissivity calculated from BACARDI and ecRad using either VarCloud as input or different ice optic parameterizations
- PDFs of IWC and re_ice between IFS and VarCloud

"""


# %% module import
import pylim.halo_ac3 as meta
import pylim.helpers as h
from pylim import reader, ecrad
import ac3airborne
from ac3airborne.tools import flightphase
import sys
sys.path.append('./larda')
from larda.pyLARDA.spec2mom_limrad94 import despeckle
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import colors
from matplotlib.collections import LineCollection
from matplotlib import patheffects
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmasher as cmr
import cartopy.crs as ccrs
import xarray as xr
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import logging

mpl.use('module://backend_interagg')

log = logging.getLogger("pylim")
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

cbc = h.get_cb_friendly_colors()

# %% set up paths and meta data
campaign = "halo-ac3"
key = "RF17"
flight = meta.flight_names[key]
date = flight[9:17]
ecrad_path = f"{h.get_path('ecrad', campaign=campaign)}/{date}"
bahamas_path = h.get_path("bahamas", flight, campaign)
bacardi_path = h.get_path("bacardi", flight, campaign)
dropsonde_path = f"{h.get_path('dropsondes', flight, campaign)}/.."
trajectory_path = f"{h.get_path('trajectories', campaign=campaign)}/selection_CC_and_altitude"
era5_path = h.get_path("era5", flight, campaign)
ifs_path = f"{h.get_path('ifs', flight, campaign)}/{date}"
radar_path = h.get_path("hamp_mira", flight, campaign)
lidar_path = h.get_path("wales", flight, campaign)
plot_path = f"C:/Users/Johannes/Documents/Doktor/conferences_workshops/2023_11_HALO_Symposium/figures"

# file names
bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{key}_v1.nc"
bacardi_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1.nc"
bacardi_res_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1_1s.nc"
dropsonde_file = f"HALO-AC3_HALO_Dropsondes_quickgrid_{date}.nc"
era5_files = [os.path.join(era5_path, f) for f in os.listdir(era5_path) if f"P{date}" in f]
era5_files.sort()
radar_file = f"radar_{date}_v1.6.nc"
lidar_file = f"HALO-AC3_HALO_WALES_bsrgl_{date}_{key}_V2.0.nc"

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
sel_time_below = slice(pd.to_datetime("2022-04-11 11:35"), pd.to_datetime("2022-04-11 11:50"))

# %% read in data
bahamas = reader.read_bahamas(f"{bahamas_path}/{bahamas_file}")
bacardi_ds = xr.open_dataset(f"{bacardi_path}/{bacardi_file}")
bacardi_ds_res = xr.open_dataset(f"{bacardi_path}/{bacardi_res_file}")
dropsonde_ds = xr.open_dataset(f"{dropsonde_path}/{dropsonde_file}")
dropsonde_ds["alt"] = dropsonde_ds.alt / 1000  # convert altitude to km
ins = cat["HALO-AC3"]["HALO"]["GPS_INS"][f"HALO-AC3_HALO_{key}"](storage_options=kwds, **credentials).to_dask()
era5_ds = xr.open_mfdataset(era5_files).sel(time=f"2022-04-11T11:00")
ifs = xr.open_dataset(f"{ifs_path}/ifs_20220411_00_ml_O1280_processed.nc").set_index(rgrid=["lat", "lon"])
radar_ds = xr.open_dataset(f"{radar_path}/{radar_file}")
radar_ds["height"] = radar_ds.height / 1000
# filter -888 values
radar_ds["dBZg"] = radar_ds.dBZg.where(np.isnan(radar_ds.radar_flag) & ~radar_ds.dBZg.isin(-888))

# %% read in lidar data V2
lidar_ds = xr.open_dataset(f"{lidar_path}/{lidar_file}")
lidar_ds["altitude"] = lidar_ds["altitude"] / 1000
lidar_ds = lidar_ds.rename(altitude="height").transpose("time", "height")
# convert lidar data to radar convention: [time, height], ground = 0m
lidar_height = lidar_ds.height

# %% plotting meta
time_extend = pd.to_timedelta((ins.time[-1] - ins.time[0]).values)  # get time extend for x-axis labeling
time_extend_cs = below_cloud["end"] - above_cloud["start"]  # time extend for case study
time_extend_ac = (above_cloud["end"] - above_cloud["start"])
time_extend_bc = below_cloud["end"] - below_cloud["start"]
case_slice = slice(above_cloud["start"], below_cloud["end"])
plt.rcdefaults()
h.set_cb_friendly_colors()

# %% read in and select closest column ecrad data
ecrad_versions = ["v15", "v15.1", "v16", "v17", "v18", "v19", "v20"]
ecrad_dict = dict()

for k in ecrad_versions:
    ds = xr.open_dataset(f"{ecrad_path}/ecrad_merged_inout_{date}_{k}.nc")
    if "column" in ds.dims:
        for var in ["flux_dn_sw", "transmissivity_sw_above_cloud"]:
            ds[f"{var}_std"] = ds[var].std(dim="column")
        ds = ds.sel(column=0, drop=True)

    ds["tiwp"] = ds.iwp.where(ds.iwp != np.inf, np.nan).sum(dim="level")

    ecrad_dict[k] = ds.copy()

# %% BACARDI calculate transmissivity
bacardi_ds["ecrad_fdw"] = ecrad_dict["v15"].flux_dn_sw.interp(time=bacardi_ds.time,
                                                              kwargs={"fill_value": "extrapolate"})
bacardi_ds["transmissivity_above_cloud"] = bacardi_ds["F_down_solar"] / bacardi_ds["ecrad_fdw"].isel(half_level=73)

# %% era5 calculate pressure on all model levels
pressure = era5_ds.hyam + era5_ds.hybm * era5_ds.PS * 100
# select only relevant model levels and swap dimension names
era5_ds["pressure"] = pressure.sel(nhym=slice(39, 137)).swap_dims({"nhym": "lev"})

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

# %% interpolate lidar data onto radar range resolution
new_range = radar_ds.height.values
lidar_ds_r = lidar_ds.interp(height=np.flip(new_range))
lidar_ds_r = lidar_ds_r.assign_coords(height=np.flip(new_range)).isel(height=slice(None, None, -1))
lidar_ds_r = lidar_ds_r.assign_coords(time=lidar_ds_r.time.dt.round("1s"))

# %% combine radar and lidar mask
lidar_mask = lidar_ds_r["backscatter_ratio"] > 1.2
lidar_mask = lidar_mask.where(lidar_mask == 0, 2).resample(time="1s").first()
radar_lidar_mask = radar_ds["mask"] + lidar_mask

# %% plot map of trajectories with surface pressure, flight track, dropsonde locations and high cloud cover
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
ifs_plot = ifs.sel(time="2022-04-11 11:00")
h.set_cb_friendly_colors()

plt.rc("font", size=16)
fig = plt.figure(figsize=(42 * h.cm, 18 * h.cm))

# plot trajectory map 11 April in first row and first column
ax = fig.add_subplot(111, projection=ccrs.NorthPolarStereo())
ax.coastlines(alpha=0.5)
ax.set_xlim((-1800000, 1300000))
ax.set_ylim((-2500000, 450000))
gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=1, color='gray', alpha=0.5,
                  linestyle=':', x_inline=False, y_inline=False, rotate_labels=False)
gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 20))
gl.ylocator = mticker.FixedLocator(np.arange(60, 90, 5))
gl.top_labels = False
gl.right_labels = False

# Plot the surface pressure - 11 April
pressure_levels = np.arange(900, 1125, 5)
e5_press = era5_ds.MSL / 100
cp = ax.contour(e5_press.lon, e5_press.lat, e5_press, levels=pressure_levels, colors='k', linewidths=1,
                linestyles='solid', alpha=1, transform=data_crs)
cp.clabel(fontsize=8, inline=1, inline_spacing=4, fmt='%i hPa', rightside_up=True, use_clabeltext=True)

# add seaice edge
ci_levels = [0.8]
e5_ci = era5_ds.CI
cci = ax.contour(e5_ci.lon, e5_ci.lat, e5_ci, ci_levels, transform=data_crs, linestyles="--", colors="#332288",
                 linewidths=3)

# add high cloud cover
# E5_cc = era5_ds.CC.where(era5_ds.pressure < 60000, drop=True).sum(dim="lev")
ifs_cc = ifs_plot.cloud_fraction.where(ifs_plot.pressure_full < 60000, drop=True).sum(dim="level")
# E5_cc = E5_cc.where(E5_cc > 1)
ax.tricontourf(ifs_cc.lon, ifs_cc.lat, ifs_cc, levels=24, transform=data_crs, cmap="Blues", alpha=1)

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
ax.scatter(track_lons[::10], track_lats[::10], c="k", alpha=1, marker=".", s=4, zorder=400,
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
handles = [plt.plot([], ls="-", color="#000000", lw=3)[0],  # flight track
           cross[0],  # dropsondes
           plt.plot([], ls="--", color="#332288", lw=3)[0],  # sea ice edge
           Patch(facecolor="royalblue")]  # cloud cover
labels = ["HALO flight track", "Dropsonde", "Sea Ice Edge", "High Cloud Cover\nat 11:00 UTC"]
ax.legend(handles=handles, labels=labels, framealpha=1, loc=2, fontsize=14)

plt.tight_layout()
figname = f"{plot_path}/{flight}_trajectories_high_cc_map.png"
plt.savefig(figname, format='png', dpi=300, bbox_inches='tight')
print(figname)
plt.show()
plt.close()

# %% plot lidar data for case study together with BACARDI data in second panel
plot_ds = lidar_ds_r["backscatter_ratio"].where((lidar_ds_r.flags == 0) & (lidar_ds_r.backscatter_ratio > 1)).sel(
    time=sel_time).set_index(time="latitude")
# plot_ds = plot_ds.assign_coords(time=np.arange(len(plot_ds.time)))
# xticklabels_all = [(f"{lat:4.2f}", f"{lon:4.2f}") for lat, lon in zip(plot_ds.latitude.to_numpy(), plot_ds.longitude.to_numpy())]
ct_plot = radar_lidar_mask.sel(time=sel_time).assign_coords(time=plot_ds.time.to_numpy())
# ct_plot = ct_plot.assign_coords(time=np.arange(len(ct_plot.time)))
bacardi_plot = bacardi_ds.sel(time=below_slice).set_index(time="lat").sel(time=plot_ds.time, method="nearest")
plt.rc("font", size=19)
_, axs = plt.subplots(2, figsize=(43 * h.cm, 20 * h.cm))
ax = axs[0]
plot_ds.plot(x="time", y="height", cmap=cmr.get_sub_cmap(cmr.chroma_r, 0, 1), norm=colors.LogNorm(), vmax=100,
             cbar_kwargs=dict(label="Backscatter ratio \nat 532$\,$nm", pad=0.01), ax=ax)
ct_plot.plot.contour(x="time", levels=[2.9], colors=cbc[4], ax=ax)
ax.plot([], color=cbc[4], label="Radar & Lidar Mask", lw=2)
ax.legend(loc=3)
ax.set(xlabel="", ylabel="Altitude (km)", ylim=(4.5, 8))

# second row
ax = axs[1]
plot_ds = bacardi_plot["F_down_solar"]
plot_ds.plot(x="time", lw=3, ax=ax)
ax.set(xlabel="Latitude (°N)", ylabel="Solar downward\nirradiance (W$\,$m$^{-2}$)")
ax.grid()
ax.margins(x=0, y=0.1)
ax.fill_between(plot_ds.time, plot_ds + plot_ds * 0.03, plot_ds - plot_ds * 0.03, color=cbc[0],
                alpha=0.5, label="BACARDI uncertainty")
divider2 = make_axes_locatable(ax)
cax2 = divider2.append_axes("right", size="15%", pad=0.5)
cax2.axis('off')

plt.tight_layout()

figname = f"{plot_path}/{flight}_lidar_backscatter_ratio_532_radar_mask_cs_BACARDI.png"
plt.savefig(figname, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# %% plot BACARDI and ecRad below cloud including Yi
ds16, ds15, ds18, ds19 = ecrad_dict["v16"], ecrad_dict["v15"], ecrad_dict["v18"], ecrad_dict["v19"]
time_sel = slice(ds16.time[0], ds16.time[-1])
ecrad_plot = ds16.isel(half_level=ds16.aircraft_level)
ecrad_plot1 = ds15.isel(half_level=ds15.aircraft_level).sel(time=time_sel)
ecrad_plot2 = ds18.isel(half_level=ds18.aircraft_level).sel(time=time_sel)
ecrad_plot3 = ds19.isel(half_level=ds19.aircraft_level).sel(time=time_sel)
bacardi_lat = bacardi_ds["lat"].sel(time=time_sel)
bacardi_lon = bacardi_ds["lon"].sel(time=time_sel)
bacardi_plot = bacardi_ds["transmissivity_above_cloud"].sel(time=time_sel)
bacardi_error = bacardi_plot * 0.03

plt.rc("font", size=19)
_, ax = plt.subplots(figsize=(41 * h.cm, 21 * h.cm))
ax.plot(bacardi_plot.time, bacardi_plot,
        label="BACARDI", lw=4)
ax.fill_between(bacardi_plot.time, bacardi_plot + bacardi_error, bacardi_plot - bacardi_error,
                color=cbc[0], alpha=0.5, label="BACARDI uncertainty")
ax.plot(ecrad_plot.time, ecrad_plot.transmissivity_sw_above_cloud,
        label="ecRad VarCloud Fu-IFS", lw=4)
ax.errorbar(ecrad_plot1.time.to_numpy(), ecrad_plot1.transmissivity_sw_above_cloud,
            yerr=ecrad_plot1.transmissivity_sw_above_cloud_std,
            label="ecRad IFS Fu-IFS", lw=4, marker="o", markersize=10, color=cbc[3], capsize=10)
ax.errorbar(ecrad_plot2.time.to_numpy(), ecrad_plot2.transmissivity_sw_above_cloud,
            yerr=ecrad_plot2.transmissivity_sw_above_cloud_std,
            label="ecRad IFS Baran2016", lw=4, marker="o", markersize=10, color=cbc[7], capsize=10)
ax.errorbar(ecrad_plot3.time.to_numpy(), ecrad_plot3.transmissivity_sw_above_cloud,
            yerr=ecrad_plot3.transmissivity_sw_above_cloud_std,
            label="ecRad IFS Yi2013", lw=4, marker="o", markersize=10, color=cbc[8], capsize=10)
ax.legend(loc=4, ncols=2)
ax.grid()
h.set_xticks_and_xlabels(ax, pd.to_timedelta((ds16.time[-1] - ds16.time[0]).to_numpy()))
ax.set_ylabel(f"Solar transmissivity")
ax.set_xlabel("Time (UTC)", labelpad=-15)
ax.set_ylim(0.66, 1.01)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))

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
figname = f"{plot_path}/{flight}_ecrad_varcloud_BACARDI_transmissivity_solar_below_cloud_2.png"
plt.savefig(figname, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# %% plot PDF of transmissivity
ds16, ds15, ds18, ds19 = ecrad_dict["v16"], ecrad_dict["v15"], ecrad_dict["v18"], ecrad_dict["v19"]
time_sel = slice(ds16.time[0], ds16.time[-1])
ecrad_plot = ds16.isel(half_level=ds16.aircraft_level).transmissivity_sw_above_cloud
ecrad_plot1 = ds15.isel(half_level=ds15.aircraft_level).sel(time=time_sel).transmissivity_sw_above_cloud
ecrad_plot2 = ds18.isel(half_level=ds18.aircraft_level).sel(time=time_sel).transmissivity_sw_above_cloud
ecrad_plot3 = ds19.isel(half_level=ds19.aircraft_level).sel(time=time_sel).transmissivity_sw_above_cloud
bacardi_plot = bacardi_ds["transmissivity_above_cloud"].sel(time=time_sel)
binsize = binsize = 0.01
bins = np.arange(np.round(bacardi_plot.min() - binsize, 2),
                 np.round(bacardi_plot.max() + binsize, 2),
                 binsize)

plt.rc("font", size=19)
_, axs = plt.subplots(1, 2, figsize=(41 * h.cm, 16 * h.cm))

# Left Panel BACARDI and VarCloud
ax = axs[0]
sns.histplot(bacardi_plot, label="BACARDI", stat="density", kde=False, bins=bins, ax=ax)
sns.histplot(ecrad_plot, label="ecRad\nVarCloud Fu-IFS", stat="density",
             kde=False, bins=bins, ax=ax, color=cbc[1])
# add mean
ax.axvline(bacardi_plot.mean(), color=cbc[0], lw=5, ls="--", ymax=0.9,
           path_effects=[patheffects.withStroke(linewidth=9, foreground="k")])
ax.axvline(ecrad_plot.mean(), color=cbc[1], lw=5, ls="--", ymax=0.9,
           path_effects=[patheffects.withStroke(linewidth=9, foreground="k")])
# for legend entry
ax.plot([], ls="--", lw=5, color="k", label="Mean")
ax.set(xlabel="Solar transmissivity")
ax.set_title("Different input", fontweight="bold")
ax.grid()
handles, labels = ax.get_legend_handles_labels()
order = [1, 2, 0]
handles = [handles[i] for i in order]
labels = [labels[i] for i in order]
ax.legend(handles, labels)

# Right Panel BACARDI and different ice optics
ax = axs[1]
sns.histplot(bacardi_plot, label="BACARDI", stat="density", kde=False, bins=bins, ax=ax)
sns.histplot(ecrad_plot1, label="IFS Fu-IFS", stat="density",
             kde=False, bins=bins, ax=ax, color=cbc[3])
sns.histplot(ecrad_plot2, label="IFS Baran2016", stat="density",
             kde=False, bins=bins, ax=ax, color=cbc[7])
sns.histplot(ecrad_plot3, label="IFS Yi2013", stat="density",
             kde=False, bins=bins, ax=ax, color=cbc[8])
# add mean
ax.axvline(bacardi_plot.mean(), color=cbc[0], lw=5, ls="--", ymax=0.4,
           path_effects=[patheffects.withStroke(linewidth=9, foreground="k")])
ax.axvline(ecrad_plot1.mean(), color=cbc[3], lw=5, ls="--", ymax=0.9,
           path_effects=[patheffects.withStroke(linewidth=9, foreground="k")])
ax.axvline(ecrad_plot2.mean(), color=cbc[7], lw=5, ls="--", ymax=0.9,
           path_effects=[patheffects.withStroke(linewidth=9, foreground="k")])
ax.axvline(ecrad_plot3.mean(), color=cbc[8], lw=5, ls="--", ymax=0.9,
           path_effects=[patheffects.withStroke(linewidth=9, foreground="k")])
# for legend entry
ax.plot([], ls="--", lw=5, color="k", label="Mean")
ax.plot([], ls="", color="w", label="ecRad")
ax.set(xlabel="Solar transmissivity")
ax.set_title("Different parameterization", fontweight="bold")
ax.grid()
# reorder legend entries
handles, labels = ax.get_legend_handles_labels()
order = [2, 1, 3, 4, 5, 0]
handles = [handles[i] for i in order]
labels = [labels[i] for i in order]
ax.legend(handles, labels)

plt.tight_layout()
figname = f"{plot_path}/{flight}_ecrad_BACARDI_transmissivity_solar_2panels.png"
plt.savefig(figname, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# %% print mean values
print(f"Mean solar transmissivity\n"
      f"BACARDI: {bacardi_plot.mean():.2f}\n"
      f"VarCloud: {ecrad_plot.mean():.2f}\n"
      f"IFS Fu-IFS: {ecrad_plot1.mean():.2f}\n"
      f"IFS Baran2016: {ecrad_plot2.mean():.2f}\n"
      f"IFS Yi2013: {ecrad_plot3.mean():.2f}")

# %% plot PDF of IWC retrieved and predicted
time_sel = sel_time
plot_v1 = (ecrad_dict["v15.1"].iwc
           .where(ecrad_dict["v15.1"].cloud_fraction > 0)
           .where(ecrad_dict["v15.1"].cloud_fraction == 0, ecrad_dict["v15.1"].iwc / ecrad_dict["v15.1"].cloud_fraction)
           .sel(time=time_sel)
           .to_numpy() * 1e6).flatten()
plot_v1 = plot_v1[~np.isnan(plot_v1)]
plot_v8 = (ecrad_dict["v16"].iwc.to_numpy() * 1e6).flatten()
plot_v8 = plot_v8[~np.isnan(plot_v8)]
binsize = 0.25
bins = np.arange(0, 5.1, binsize)
plt.rc("font", size=19)
_, ax = plt.subplots(figsize=(22 * h.cm, 13 * h.cm))
ax.hist(plot_v8, bins=bins, label="VarCloud", histtype="step", lw=4, color=cbc[1], density=True)
ax.hist(plot_v1, bins=bins, label="IFS", histtype="step", lw=4, color=cbc[5], density=True)
ax.legend()
ax.text(0.05, 0.85, f"Binsize: {binsize}$\,$" + "mg$\,$m$^{-3}$", transform=ax.transAxes,
        bbox=dict(boxstyle="round", fc="white"))
ax.grid()
ax.set(xlabel=r"Ice water content (mg$\,$m$^{-3}$)",
       ylabel="Probability density function",
       ylim=(0, 0.75))
plt.tight_layout()
figname = f"{plot_path}/{flight}_ecrad_v15_v17_iwc_pdf_v3.png"
plt.savefig(figname, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# %% plot PDF of re_ice retrieved and predicted
time_sel = sel_time
plot_v1 = (ecrad_dict["v15"].re_ice.sel(time=time_sel).to_numpy() * 1e6).flatten()
plot_v8 = (ecrad_dict["v16"].re_ice.to_numpy() * 1e6).flatten()
binsize = 2
bins = np.arange(10, 71, binsize)
plt.rc("font", size=19)
_, ax = plt.subplots(figsize=(22 * h.cm, 13 * h.cm))
ax.hist(plot_v8, bins=bins, label="VarCloud", histtype="step", lw=4, color=cbc[1], density=True)
ax.hist(plot_v1, bins=bins, label="IFS", histtype="step", lw=4, color=cbc[5], density=True)
ax.legend()
ax.text(0.7, 0.63, f"Binsize: {binsize}$\,$" + "$\mu$m", transform=ax.transAxes,
        bbox=dict(boxstyle="round", fc="white"))
ax.grid()
ax.set(xlabel=r"Ice effective radius ($\mu$m)",
       ylabel="Probability density function")
plt.tight_layout()
figname = f"{plot_path}/{flight}_ecrad_v15_v17_re_ice_pdf_v3.png"
plt.savefig(figname, dpi=300, bbox_inches="tight")
plt.show()
plt.close()
