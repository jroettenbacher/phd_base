#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 11.04.2024

Plots for Measurement chapter
"""

# %% import modules
import os

import cartopy.crs as ccrs
import cmasher as cmr
import dill
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker, patheffects, colors
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
from skimage import io
import seaborn as sns
import xarray as xr

import pylim.halo_ac3 as meta
import pylim.helpers as h
from pylim import ecrad
from pylim.bahamas import preprocess_bahamas

cbc = h.get_cb_friendly_colors('petroff_6')

# %% set paths
campaign = 'halo-ac3'
keys = ['RF17', 'RF18']
ecrad_versions = ['v15.1']
save_path = 'C:/Users/Johannes/Documents/Doktor/manuscripts/_thesis/data'
plot_path = 'C:/Users/Johannes/Documents/Doktor/manuscripts/_thesis/figure'
bacardi_all_path = h.get_path('all', campaign=campaign, instrument='BACARDI')
bahamas_all_path = h.get_path('all', campaign=campaign, instrument='BAHAMAS')
trajectory_path = f"{h.get_path('trajectories', campaign=campaign)}/selection_CC_and_altitude"

# %% read in data
(
    bahamas_ds, bacardi_ds, bacardi_ds_res, varcloud_ds, above_clouds,
    below_clouds, slices, ifs_ds, ifs_ds_sel, dropsonde_ds, lidar_ds, radar_ds,
    sat_imgs
) = (dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict())

left, right, bottom, top = 0, 1000000, -1000000, 0
sat_img_extent = (left, right, bottom, top)
# read in dropsonde data
dropsonde_path = f"{h.get_path('all', campaign=campaign, instrument='dropsondes')}/Level_3"
dropsonde_file = "merged_HALO_P5_beta_v2.nc"
dds = xr.open_dataset(f"{dropsonde_path}/{dropsonde_file}")

for key in keys:
    flight = meta.flight_names[key]
    date = flight[9:17]
    urldate = pd.to_datetime(date).strftime('%Y-%m-%d')
    bacardi_path = h.get_path('bacardi', flight, campaign)
    bahamas_path = h.get_path('bahamas', flight, campaign)
    ifs_path = f'{h.get_path('ifs', flight, campaign)}/{date}'
    ecrad_path = f'{h.get_path('ecrad', flight, campaign)}/{date}'
    varcloud_path = h.get_path('varcloud', flight, campaign)
    radar_path = h.get_path("hamp_mira", flight, campaign)
    lidar_path = h.get_path("wales", flight, campaign)

    # filenames
    bahamas_file = f'HALO-AC3_HALO_BAHAMAS_{date}_{key}_v1_JR.nc'
    bacardi_file = f'HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1_JR_v2.nc'
    ifs_file = f"ifs_{date}_00_ml_O1280_processed.nc"
    ifs_sel_file = f'ifs_{date}_00_ml_O1280_processed_sel_JR.nc'
    varcloud_file = [f for f in os.listdir(varcloud_path) if f.endswith('_JR.nc')][0]
    radar_file = f"HALO_HALO_AC3_radar_unified_{key}_{date}_v2.6.nc"
    lidar_file = f"HALO-AC3_HALO_WALES_bsrgl_{date}_{key}_V2.0.nc"
    satfile = f'{save_path}/{key}_MODIS_Terra_CorrectedReflectance_Bands367.png'
    sat_url = f'https://gibs.earthdata.nasa.gov/wms/epsg3413/best/wms.cgi?\
    version=1.3.0&service=WMS&request=GetMap&\
    format=image/png&STYLE=default&bbox={left},{bottom},{right},{top}&CRS=EPSG:3413&\
    HEIGHT=8192&WIDTH=8192&TIME={urldate}&layers=MODIS_Terra_CorrectedReflectance_Bands367'

    # read in aircraft data
    bahamas_ds[key] = xr.open_dataset(f'{bahamas_path}/{bahamas_file}')
    bacardi = xr.open_dataset(f'{bacardi_path}/{bacardi_file}')
    bacardi_ds[key] = bacardi
    # read in resampled BACARDI data
    bacardi_res = xr.open_dataset(f'{bacardi_path}/{bacardi_file.replace('_v2.nc', '_1Min_v2.nc')}')
    bacardi_ds_res[key] = bacardi_res

    # read in satellite image
    try:
        sat_imgs[key] = io.imread(satfile)
    except FileNotFoundError:
        sat_imgs[key] = io.imread(sat_url)

    # split up dropsonde data into RF17 and RF18
    dropsonde_ds[key] = dds.where(dds.launch_time.dt.date == pd.to_datetime(date).date(), drop=True)

    # read in radar & lidar data
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

    # read in ifs data
    ifs = xr.open_dataset(f"{ifs_path}/{ifs_file}").set_index(rgrid=["lat", "lon"])
    # filter low clouds according to ECMWF low cloud criterion (pressure higher than 0.8 * surface pressure)
    cloud_data = ifs[["q_liquid", "q_ice", "cloud_fraction", "clwc", "ciwc", "crwc", "cswc"]]
    pressure_filter = ifs.pressure_full.sel(level=137) * 0.8
    low_cloud_filter = ifs.pressure_full < pressure_filter  # False for low clouds
    cloud_data = cloud_data.where(low_cloud_filter, 0)  # replace where False with 0
    ifs.update(cloud_data)
    ifs_ds[key] = ifs.copy(deep=True)
    # read in ifs data along flight path
    ifs_ds_sel[key] = xr.open_dataset(f'{ifs_path}/{ifs_sel_file}').set_index(rgrid=['lat', 'lon'])

    # read in time slices
    loaded_objects = list()
    filenames = [f'{key}_slices.pkl', f'{key}_above_cloud.pkl', f'{key}_below_cloud.pkl']
    for filename in filenames:
        with open(f'{save_path}/{filename}', 'rb') as f:
            loaded_objects.append(dill.load(f))

    slices[key] = loaded_objects[0]
    above_clouds[key] = loaded_objects[1]
    below_clouds[key] = loaded_objects[2]

# read in stats
stats = pd.read_csv(f'{save_path}/halo-ac3_bacardi_ecrad_statistics.csv')

# %% read in all BAHAMAS and BACARDI data
all_files = os.listdir(bacardi_all_path)
all_files.sort()
all_files = [os.path.join(bacardi_all_path, file) for file in all_files[2:] if file.endswith("JR.nc")]
bacardi_ds_all = xr.open_mfdataset(all_files)
# bahamas
all_files = [f for f in os.listdir(bahamas_all_path) if f.startswith("HALO")]
all_files.sort()
all_files = [os.path.join(bahamas_all_path, file) for file in all_files[1:]]
bahamas_ds_all = xr.open_mfdataset(all_files, preprocess=preprocess_bahamas)

# %% plot BACARDI misalignment error
theta = np.arange(60, 90, 5)
d_theta = np.arange(0, 3.5, 0.5)
# Create a DataFrame
df = pd.DataFrame({'theta': np.repeat(theta, len(d_theta)),
                   'd_theta': np.tile(d_theta, len(theta))})
# Calculate psi
df['psi'] = (np.cos(np.deg2rad(df['theta'] + df['d_theta']))
             / np.cos(np.deg2rad(df['theta'])))
df['error'] = (df['psi'] - 1) * 100

plt.rc('font', size=9)
_, ax = plt.subplots(figsize=(9 * h.cm, 9 * h.cm), layout='constrained')
sns.lineplot(df, x='d_theta', y='error',
             hue='theta', palette=cbc,
             style='theta', markers=True, dashes=False, markersize=8,
             ax=ax)
ax.legend(title='Solar zenith\n   angle (°)')
ax.grid()
ax.set(
    xlabel='Horizontal misalignment (°)',
    ylabel='Irradiance deviation $\\Phi_F$ (%)'
)
figname = f'{plot_path}/03_phi_f.pdf'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot BACARDI F_dw_solar deviation from libRadtran simulation
print('To Be Done')
# %% plot flight track together with trajectories and high cloud cover - for colorbar
cmap = mpl.colormaps["tab20b_r"]([20, 20, 20, 20, 4, 7, 8, 11, 12, 15, 0, 3])
cmap[:4] = mpl.colormaps["tab20c"]([11, 8, 7, 4])
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

plt.rc("font", size=10)
fig, axs = plt.subplots(1, 2,
                        figsize=(17.75 * h.cm, 9.5 * h.cm),
                        subplot_kw={"projection": map_crs},
                        layout="constrained")

# plot trajectory map 11 April in first row and first column
ax = axs[0]
ax.coastlines(alpha=0.5)
xlim = (-1200000, 1200000)
ylim = (-2500000, 50000)
ax.set_title("(a) RF 17 - 11 April 2022", fontsize=10)
# ax.set_extent([-30, 40, 65, 90], crs=map_crs)
ax.set_extent([xlim[0], xlim[1], ylim[0], ylim[1]], crs=map_crs)
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
cp.clabel(fontsize=7, inline=1, inline_spacing=4, fmt='%i', rightside_up=True, use_clabeltext=True)

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
    loc="lower left",
    bbox_to_anchor=(-0.15, 0.75, 1, 1),
    bbox_transform=ax.transAxes,
    borderpad=0,
)
plt.colorbar(hcc, cax=axins1, orientation="vertical", ticks=[0.2, 0.4, 0.6, 0.8, 1])
axins1.yaxis.set_ticks_position("right")
axins1.set_yticklabels([0.2, 0.4, 0.6, 0.8, 1], size=8,
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
ds_ds = dropsonde_ds["RF17"]
x, y = ds_ds.lon.isel(alt=-1), ds_ds.lat.isel(alt=-1)
launch_times = pd.to_datetime(ds_ds.launch_time.to_numpy())
cross = ax.scatter(x, y, marker="x", c="orangered", s=8, label="Dropsonde",
                   transform=data_crs,
                   zorder=450)
for i, lt in enumerate(launch_times):
    ax.text(x[i], y[i], f"{launch_times[i]:%H:%M}", c="k", fontsize=8,
            transform=data_crs, zorder=500,
            path_effects=[patheffects.withStroke(linewidth=0.5, foreground="white")])

# plot trajectories 12 April in second row first column
ax = axs[1]
ax.coastlines(alpha=0.5)
ax.set_title("(b) RF 18 - 12 April 2022", fontsize=10)
ax.set_extent([xlim[0], xlim[1], ylim[0], ylim[1]], crs=map_crs)
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
cp.clabel(fontsize=7, inline=1, inline_spacing=4, fmt='%i', rightside_up=True, use_clabeltext=True)

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
    loc="lower left",
    bbox_to_anchor=(-0.15, 0.75, 1, 1),
    bbox_transform=ax.transAxes,
    borderpad=0,
)
cb = plt.colorbar(hcc, cax=axins1, orientation="vertical", ticks=[0.2, 0.4, 0.6, 0.8, 1])
axins1.yaxis.set_ticks_position("right")
axins1.set_yticklabels([0.2, 0.4, 0.6, 0.8, 1], size=8,
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
ds_ds = dropsonde_ds["RF18"]
x, y = ds_ds.lon.isel(alt=-1), ds_ds.lat.isel(alt=-1)
launch_times = pd.to_datetime(ds_ds.launch_time.to_numpy())
cross = ax.scatter(x, y, marker="x", c="orangered", s=8, label="Dropsonde",
                   transform=data_crs,
                   zorder=450)
ax.text(x[10], y[10], f"{launch_times[10]:%H:%M}", c="k", fontsize=8,
        transform=data_crs, zorder=500,
        path_effects=[patheffects.withStroke(linewidth=0.5, foreground="white")])

# make legend for flight track and dropsondes
labels = ["HALO flight track", "Case study section",
          "Dropsonde", "Sea ice edge",
          "Mean sea level pressure (hPa)", "High cloud cover at 12:00 UTC"]
handles = [plt.plot([], ls="-", color="k")[0],  # flight track
           plt.plot([], ls="-", color=cbc[1])[0],  # case study section
           cross,  # dropsondes
           plt.plot([], ls="--", color="#332288")[0],  # sea ice edge
           plt.plot([], ls="solid", lw=0.7, color="k")[0],  # isobars
           Patch(facecolor=mpl.colormaps['Blues'](0.9), alpha=0.5)]  # cloud cover
fig.legend(handles=handles, labels=labels, framealpha=1, ncols=3,
           loc="outside lower center")

cbar = fig.colorbar(line, pad=0.01, ax=ax,
                    shrink=plt_sett["shrink"],
                    ticks=plt_sett["cmap_ticks"])
cbar.set_label(label=plt_sett['label'])

figname = f"{plot_path}/03_HALO-AC3_RF17_RF18_flight_track_trajectories_plot_overview_cb.png"
plt.savefig(figname, dpi=600)
plt.show()
plt.close()

# %% plot zoom of case study region RF 18
plt.rc("font", size=6)
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
ds_ds = dropsonde_ds["RF18"]
ds_ds = ds_ds.where(ds_ds.launch_time > pd.Timestamp('2022-04-12 10:30'), drop=True)
x, y = ds_ds.lon.isel(alt=-1), ds_ds.lat.isel(alt=-1)
launch_times = pd.to_datetime(ds_ds.launch_time.to_numpy())
for i, lt in enumerate(launch_times):
    cross = ax.scatter(x[i], y[i], marker="x", c="orangered", s=8,
                       label="Dropsonde",
                       transform=data_crs,
                       zorder=450)
    ax.text(x[i], y[i], f"{lt:%H:%M}", c="k", fontsize=7,
            transform=data_crs, zorder=500,
            path_effects=[patheffects.withStroke(linewidth=0.5, foreground="white")])

figname = f"{plot_path}/03_HALO-AC3_RF18_flight_track_trajectories_plot_overview_zoom.png"
plt.savefig(figname, dpi=600, bbox_inches='tight')
plt.show()
plt.close()

# %% plot flight track together with trajectories and high cloud cover and satellite images
cmap = mpl.colormaps["tab20b_r"]([20, 20, 20, 20, 4, 7, 8, 11, 12, 15, 0, 3])
cmap[:4] = mpl.colormaps["tab20c"]([11, 8, 7, 4])
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
sat_crs = ccrs.NorthPolarStereo(central_longitude=-45)
extent = sat_img_extent

plt.rc("font", size=10)
fig, axs = plt.subplot_mosaic(
    "AB;CD",
    figsize=(18 * h.cm, 18.5 * h.cm),
    per_subplot_kw={
        ("A", "B"): {"projection": map_crs},
        ("C", "D"): {"projection": sat_crs}
    },
    # layout="constrained"
)

# plot trajectory map 11 April in first row and first column
ax = axs["A"]
ax.coastlines(alpha=0.5)
xlim = (-1200000, 1200000)
ylim = (-2500000, 50000)
ax.text(0, 1.03, "(a)", transform=ax.transAxes)
ax.set_title("RF 17 - 11 April 2022", fontsize=10)
# ax.set_extent([-30, 40, 65, 90], crs=map_crs)
ax.set_extent([xlim[0], xlim[1], ylim[0], ylim[1]], crs=map_crs)
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
cp.clabel(fontsize=7, inline=1, inline_spacing=4, fmt='%i', rightside_up=True, use_clabeltext=True)

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
    bbox_to_anchor=(-0.16, 0, 1, 1),
    bbox_transform=ax.transAxes,
    borderpad=0,
)
plt.colorbar(hcc, cax=axins1, orientation="vertical", ticks=[0.2, 0.4, 0.6, 0.8, 1])
axins1.yaxis.set_ticks_position("right")
axins1.set_yticklabels([0.2, 0.4, 0.6, 0.8, 1], size=8,
                       path_effects=[patheffects.withStroke(linewidth=0.5, foreground="white")])

# # plot trajectories - 11 April
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
ds_ds = dropsonde_ds["RF17"]
x, y = ds_ds.lon.isel(alt=-1), ds_ds.lat.isel(alt=-1)
cross = ax.scatter(x, y, marker="x", c=cbc[2], s=8, label="Dropsonde",
                   transform=data_crs,
                   zorder=450)

# plot trajectories 12 April in second row first column
ax = axs["B"]
ax.coastlines(alpha=0.5)
ax.text(0, 1.03, "(b)", transform=ax.transAxes)
ax.set_title("RF 18 - 12 April 2022", fontsize=10)
ax.set_extent([xlim[0], xlim[1], ylim[0], ylim[1]], crs=map_crs)
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
cp.clabel(fontsize=7, inline=1, inline_spacing=4, fmt='%i', rightside_up=True, use_clabeltext=True)

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

# add colorbar for high cloud cover
axins1 = inset_axes(
    ax,
    width="3%",  # width: 50% of parent_bbox width
    height="25%",  # height: 5%
    loc="upper left",
    bbox_to_anchor=(-0.16, 0, 1, 1),
    bbox_transform=ax.transAxes,
    borderpad=0,
)
cb = plt.colorbar(hcc, cax=axins1, orientation="vertical", ticks=[0.2, 0.4, 0.6, 0.8, 1])
axins1.yaxis.set_ticks_position("right")
axins1.set_yticklabels([0.2, 0.4, 0.6, 0.8, 1], size=8,
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

# add colorbar for trajectories
axins2 = inset_axes(
    ax,
    width="5%",  # width: 50% of parent_bbox width
    height="100%",  # height: 5%
    loc="upper left",
    bbox_to_anchor=(1.02, 0, 1, 1),
    bbox_transform=ax.transAxes,
    borderpad=0,
)
cbar = plt.colorbar(line, cax=axins2,
                    orientation="vertical",
                    shrink=plt_sett["shrink"],
                    ticks=plt_sett["cmap_ticks"])
cbar.set_label(label=plt_sett['label'])
axins2.yaxis.set_ticks_position("right")

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
ds_ds = dropsonde_ds["RF18"]
x, y = ds_ds.lon.isel(alt=-1), ds_ds.lat.isel(alt=-1)
launch_times = pd.to_datetime(ds_ds.launch_time.to_numpy())
cross = ax.scatter(x, y, marker="x", c=cbc[2], s=10, label="Dropsonde",
                   transform=data_crs,
                   zorder=450)
ax.text(x[10], y[10], f"{launch_times[10]:%H:%M}", c="k", fontsize=8,
        transform=data_crs, zorder=500,
        path_effects=[patheffects.withStroke(linewidth=0.5, foreground="white")])

# RF17/RF18 satellite image
sat_axs = list(map(axs.get, ["C", "D"]))
label = ["(c)", "(d)"]
for i, key in enumerate(keys):
    ax = sat_axs[i]
    # satellite
    ax.imshow(sat_imgs[key], extent=extent, origin='upper')
    # bahamas
    ax.plot(bahamas_ds[key].IRS_LON, bahamas_ds[key].IRS_LAT,
            color='k', transform=data_crs)
    # dropsondes
    ds = dropsonde_ds[key]
    launch_time = pd.to_datetime(ds.launch_time.to_numpy())
    x, y = ds.lon.mean(dim='alt').to_numpy(), ds.lat.mean(dim='alt').to_numpy()
    cross = ax.plot(x, y, 'X', color=cbc[2], markersize=7, transform=data_crs,
                    zorder=450)
    if key == 'RF17':
        for ii, lt in enumerate(launch_time):
            ax.text(x[ii], y[ii], f'{lt:%H:%M}', c='k', transform=data_crs, zorder=500,
                    path_effects=[patheffects.withStroke(linewidth=0.5, foreground='white')])
    else:
        ds = ds.where(ds.launch_time.isin(ds.launch_time[[1, 6]]), drop=True)
        launch_time = pd.to_datetime(ds.launch_time.to_numpy())
        x, y = ds.lon.mean(dim='alt').to_numpy(), ds.lat.mean(dim='alt').to_numpy()
        for ii, lt in enumerate(launch_time):
            ax.text(x[ii], y[ii], f"{lt:%H:%M}", color="k", transform=data_crs, zorder=500,
                    path_effects=[patheffects.withStroke(linewidth=0.5, foreground="white")])

    ax.coastlines(color='k', linewidth=1)
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                      linestyle=':')
    gl.top_labels = False
    gl.right_labels = False

    ax.set(
        xlim=(left, right),
        ylim=(bottom, top),
    )
    ax.text(0, 1.03, label[i], transform=ax.transAxes)

# make legend for flight track and dropsondes
labels = ["HALO flight track", "Case study section",
          "Dropsonde", "Sea ice edge",
          "Mean sea level pressure (hPa)", "High cloud cover at 12:00 UTC"]
handles = [plt.plot([], ls="-", color="k")[0],  # flight track
           plt.plot([], ls="-", color=cbc[1])[0],  # case study section
           plt.plot([],  ls='', marker='X', color=cbc[2], markersize=7)[0],  # dropsondes
           plt.plot([], ls="--", color="#332288")[0],  # sea ice edge
           plt.plot([], ls="solid", lw=0.7, color="k")[0],  # isobars
           Patch(facecolor=mpl.colormaps['Blues'](0.9), alpha=0.5)]  # cloud cover
fig.legend(handles=handles, labels=labels, framealpha=1, ncols=3,
           loc="lower center")

fig.subplots_adjust(right=0.9, bottom=0.15)
# fig_width, fig_height = fig.get_size_inches()
# new_fig_width = fig_width + 2 * h.cm  # add 1 cm to the width
# fig.set_size_inches(new_fig_width, fig_height)
figname = f"{plot_path}/03_HALO-AC3_RF17_RF18_flight_track_trajectories_plot_overview.png"
plt.savefig(figname, dpi=600)
plt.show()
plt.close()

# %% plot satellite image together with flight track
labels = ['(a)', '(b)']
date_title = ['11 April 2022', '12 April 2022']
data_crs = ccrs.PlateCarree()
plot_crs = ccrs.NorthPolarStereo(central_longitude=-45)
extent = sat_img_extent
plt.rc('font', size=10)
_, axs = plt.subplots(1, 2, figsize=(18 * h.cm, 9 * h.cm),
                      subplot_kw={'projection': plot_crs},
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    # satellite
    ax.imshow(sat_imgs[key], extent=extent, origin='upper')
    # bahamas
    ax.plot(bahamas_ds[key].IRS_LON, bahamas_ds[key].IRS_LAT,
            color='k', transform=data_crs, label='HALO flight track')
    # dropsondes
    ds = dropsonde_ds[key]
    launch_time = pd.to_datetime(ds.launch_time.to_numpy())
    x, y = ds.lon.mean(dim='alt').to_numpy(), ds.lat.mean(dim='alt').to_numpy()
    cross = ax.plot(x, y, 'X', color=cbc[0], markersize=7, transform=data_crs,
                    zorder=450)
    if key == 'RF17':
        for ii, lt in enumerate(launch_time):
            ax.text(x[ii], y[ii], f'{lt:%H:%M}', c='k', transform=data_crs, zorder=500,
                    path_effects=[patheffects.withStroke(linewidth=0.5, foreground='white')])
    else:
        ds = ds.where(ds.launch_time.isin(ds.launch_time[[1, 6]]), drop=True)
        launch_time = pd.to_datetime(ds.launch_time.to_numpy())
        x, y = ds.lon.mean(dim='alt').to_numpy(), ds.lat.mean(dim='alt').to_numpy()
        for ii, lt in enumerate(launch_time):
            ax.text(x[ii], y[ii], f"{lt:%H:%M}", color="k", transform=data_crs, zorder=500,
                    path_effects=[patheffects.withStroke(linewidth=0.5, foreground="white")])
    # add legend artist
    ax.plot([], label='Dropsonde', ls='', marker='X', color=cbc[0], markersize=7)

    ax.coastlines(color='k', linewidth=1)
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    gl.right_labels = False

    ax.set(
        xlim=(left, right),
        ylim=(bottom, top),
    )
    ax.set_title(f'{labels[i]} {key.replace("1", " 1")} - {date_title[i]}', fontsize=10)


axs[1].legend()
figname = f'{plot_path}/03_HALO-AC3_RF17_RF18_MODIS_Bands367_flight_track.pdf'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% solar irradiance - plot BACARDI six panel plot with above and below cloud measurements and transmissivity
plt.rc("font", size=9)
xlims = [(0, 240), (0, 320)]
ylim_transmissivity = (0.45, 1)
ylim_irradiance = [(100, 279), (80, 260)]
label_xy = (0.03, 0.9)
box_xy = (0.98, 0.9)
_, axs = plt.subplots(3, 2, figsize=(18 * h.cm, 15 * h.cm),
                      layout="constrained")

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
ax.legend(loc=4, ncol=2)
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
ax.plot(cum_distance, plot_ds["transmissivity_above_cloud"],
        label="Solar transmissivity",
        color=cbc[3])
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
# sum up distances to generate a distance axis and convert to km
cum_distance = plot_ds["distance"].cumsum().to_numpy() / 1000
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
ax.plot(cum_distance, plot_ds["transmissivity_above_cloud"],
        label="Solar transmissivity", color=cbc[3])
ax.grid()
# ax.text(label_xy[0], label_xy[1], "(f)", transform=ax.transAxes)
ax.text(box_xy[0], box_xy[1], "Below cloud", ha="right",
        transform=ax.transAxes, bbox=dict(boxstyle="Round", fc="white"))
ax.set(xlabel="Distance (km)",
       ylim=ylim_transmissivity,
       xlim=xlims[1])

# set a-f labels
for ax, label in zip(axs.flatten(), ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]):
    ax.text(label_xy[0], label_xy[1], label, transform=ax.transAxes)

figname = f"{plot_path}/03_HALO-AC3_HALO_RF17_RF18_BACARDI_case_studies_6panel.pdf"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% terrestrial irradiance - plot BACARDI terrestrial fluxes - 6 panel figure
plt.rc('font', size=9)
xlims = [(0, 240), (0, 320)]
ylim_net = (-175, 0)
ylim_irradiance = [(0, 280), (0, 280)]
yticks = ticker.MultipleLocator(50)
label_xy = (0.03, 0.9)
box_xy = (0.98, 0.9)
_, axs = plt.subplots(3, 2, figsize=(17 * h.cm, 15 * h.cm), layout='constrained')

# upper left panel - RF17 BACARDI F above cloud
ax = axs[0, 0]
plot_ds = bacardi_ds['RF17'].sel(time=slices['RF17']['above'])
plot_ds['distance'] = bahamas_ds['RF17']['distance'].sel(time=slices['RF17']['above'])
# set first distance to 0
plot_ds['distance'][0] = 0
# sum up distances to generate a distance axis and convert to km
plot_ds['cum_distance'] = plot_ds['distance'].cumsum() / 1000
# bacardi measurements
for i, var in enumerate(['F_down_terrestrial', 'F_up_terrestrial']):
    ax.plot(plot_ds.cum_distance, plot_ds[var],
            label=h.bacardi_labels[var],
            color=cbc[2 + i])
ax.legend(loc=5)
ax.grid()
ax.text(box_xy[0], box_xy[1], 'Above cloud', ha='right',
        transform=ax.transAxes, bbox=dict(boxstyle='Round', fc='white'))
ax.set(title='RF 17 - 11 April 2022',
       ylabel=f'Terrestrial\nirradiance ({h.plot_units['flux_dn_sw']})',
       ylim=ylim_irradiance[0],
       xlim=xlims[0])
ax.yaxis.set_major_locator(yticks)

# middle left panel - RF17 BACARDI F below_cloud
ax = axs[1, 0]
plot_ds = bacardi_ds['RF17'].sel(time=slices['RF17']['below'])
plot_ds['distance'] = bahamas_ds['RF17']['distance'].sel(time=slices['RF17']['below'])
# set first distance to 0
plot_ds['distance'][0] = 0
# sum up distances to generate a distance axis and convert to km, flip the distance to show travel in other
# direction
cum_distance = np.flip(plot_ds['distance'].cumsum().to_numpy() / 1000)
# bacardi measurements
for i, var in enumerate(['F_down_terrestrial', 'F_up_terrestrial']):
    ax.plot(cum_distance, plot_ds[var],
            label=h.bacardi_labels[var],
            color=cbc[2 + i])
ax.grid()
ax.text(box_xy[0], box_xy[1], 'Below cloud', ha='right',
        transform=ax.transAxes, bbox=dict(boxstyle='Round', fc='white'))
ax.set(ylabel=f'Terrestrial\nirradiance ({h.plot_units['flux_dn_sw']})',
       ylim=ylim_irradiance[1],
       xlim=xlims[0])
ax.yaxis.set_major_locator(yticks)

# lower left panel - RF17 net terrestrial above and below cloud
ax = axs[2, 0]
ax.plot(cum_distance, plot_ds['F_net_terrestrial'],
        color=cbc[4], label='Below cloud')
plot_ds = bacardi_ds['RF17'].sel(time=slices['RF17']['above'])
plot_ds['distance'] = bahamas_ds['RF17']['distance'].sel(time=slices['RF17']['above'])
# set first distance to 0
plot_ds['distance'][0] = 0
# sum up distances to generate a distance axis and convert to km
plot_ds['cum_distance'] = plot_ds['distance'].cumsum() / 1000
ax.plot(plot_ds.cum_distance, plot_ds['F_net_terrestrial'],
        color=cbc[5], label='Above cloud')
ax.grid()
ax.legend()
ax.set(ylabel=f'Net terrestrial\nirradiance ({h.plot_units['flux_dn_lw']})',
       xlabel='Distance (km)',
       ylim=ylim_net,
       xlim=xlims[0])

# upper right panel - RF18 BACARDI F above cloud
ax = axs[0, 1]
plot_ds = bacardi_ds['RF18'].sel(time=slices['RF18']['above'])
plot_ds['distance'] = bahamas_ds['RF18']['distance'].sel(time=slices['RF18']['above'])
# set first distance to 0
plot_ds['distance'][0] = 0
# sum up distances to generate a distance axis and convert to km
plot_ds['cum_distance'] = plot_ds['distance'].cumsum() / 1000
# bacardi measurements
for i, var in enumerate(['F_down_terrestrial', 'F_up_terrestrial']):
    ax.plot(plot_ds.cum_distance, plot_ds[var],
            label=f'{h.bacardi_labels[var]}',
            color=cbc[2 + i])
ax.grid()
ax.text(box_xy[0], box_xy[1], 'Above cloud', ha='right',
        transform=ax.transAxes, bbox=dict(boxstyle='Round', fc='white'))
ax.set(title='RF 18 - 12 April 2022',
       ylim=ylim_irradiance[0],
       xlim=xlims[1])
ax.yaxis.set_major_locator(yticks)

# middle right panel - RF18 BACARDI F below cloud
ax = axs[1, 1]
plot_ds = bacardi_ds['RF18'].sel(time=slices['RF18']['below'])
plot_ds['distance'] = bahamas_ds['RF18']['distance'].sel(time=slices['RF18']['below'])
# set first distance to 0
plot_ds['distance'][0] = 0
# sum up distances to generate a distance axis and convert to km
plot_ds['cum_distance'] = plot_ds['distance'].cumsum() / 1000
# bacardi measurements
for i, var in enumerate(['F_down_terrestrial', 'F_up_terrestrial']):
    ax.plot(plot_ds['cum_distance'], plot_ds[var].to_numpy(),
            label=f'{h.bacardi_labels[var]}',
            color=cbc[2 + i])
ax.grid()
ax.text(box_xy[0], box_xy[1], 'Below cloud', ha='right',
        transform=ax.transAxes, bbox=dict(boxstyle='Round', fc='white'))
ax.set(ylim=ylim_irradiance[1],
       xlim=xlims[1])
ax.yaxis.set_major_locator(yticks)

# lower right panel - RF18 net irradiance
ax = axs[2, 1]
ax.plot(plot_ds['cum_distance'], plot_ds['F_net_terrestrial'],
        color=cbc[4], label='Below cloud')
plot_ds = bacardi_ds['RF18'].sel(time=slices['RF18']['above'])
plot_ds['distance'] = bahamas_ds['RF18']['distance'].sel(time=slices['RF18']['above'])
# set first distance to 0
plot_ds['distance'][0] = 0
# sum up distances to generate a distance axis and convert to km
plot_ds['cum_distance'] = plot_ds['distance'].cumsum() / 1000
ax.plot(plot_ds['cum_distance'], plot_ds['F_net_terrestrial'],
        color=cbc[5], label='Above cloud')
ax.grid()
ax.legend()
ax.set(xlabel='Distance (km)',
       ylim=ylim_net,
       xlim=xlims[1])

# set a-f labels
for ax, label in zip(axs.flatten(), ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']):
    ax.text(label_xy[0], label_xy[1], label, transform=ax.transAxes)

figname = f'{plot_path}/03_HALO-AC3_RF17_RF18_BACARDI_terrestrial_case_studies_6panel.pdf'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% terrestrial irradiance - print statistics
sel_vars = ['F_down_terrestrial', 'F_up_terrestrial', 'F_net_terrestrial']
selection = (stats['version'].isin(['v1', 'v15.1'])
             & (stats['variable'].isin(sel_vars)
                | stats['variable'].isin(['flux_dn_lw', 'flux_up_lw', 'flux_net_lw'])))
df_print = stats[selection]
# %% plot - lidar & radar reflectivity
ylim = (0, 10)
plt.rc('font', size=9)
_, axs = plt.subplot_mosaic([['top_left', 'top_right'],
                             ['bottom_left', 'bottom_right']],
                            figsize=(17 * h.cm, 10 * h.cm), layout='constrained')
# lidar RF17
ax = axs['top_left']
key = 'RF17'
plot_ds = (lidar_ds[key]['backscatter_ratio']
           .where((lidar_ds[key].flags == 0)
                  & (lidar_ds[key].backscatter_ratio > 1))
           .sel(time=slices[key]['above']))
plot_ds.plot(x='time', y='height', cmap=cmr.rainforest_r, norm=colors.LogNorm(),
             vmax=100, add_colorbar=False,
             ax=ax)
# add below cloud altitude of HALO
altitude = bahamas_ds[key]['IRS_ALT'].sel(time=slices[key]['below']).to_numpy() / 1000
new_t = pd.date_range(plot_ds.time.to_numpy()[0], plot_ds.time.to_numpy()[-1], periods=len(altitude))
ax.plot(new_t, altitude, color='k', label='HALO below\n cloud altitude')
h.set_xticks_and_xlabels(ax, slices[key]['above'].stop - slices[key]['above'].start)
ax.set(
    title='RF17 - 11 April 2024',
    xlabel='',
    ylabel='Altitude (km)',
    ylim=ylim)
ax.legend(loc=3)

# radar RF17
ax = axs['bottom_left']
key = 'RF17'
plot_ds = (radar_ds[key]['dBZg']
           .sel(time=slices[key]['above']))
plot_ds.plot(x='time', y='height', cmap=cmr.torch_r, vmax=0, vmin=-50,
             add_colorbar=False,
             ax=ax)
# add below cloud altitude of HALO
altitude = bahamas_ds[key]['IRS_ALT'].sel(time=slices[key]['below']).to_numpy() / 1000
new_t = pd.date_range(plot_ds.time.to_numpy()[0], plot_ds.time.to_numpy()[-1], periods=len(altitude))
ax.plot(new_t, altitude, color='k')
h.set_xticks_and_xlabels(ax, slices[key]['above'].stop - slices[key]['above'].start)
ax.set(
    xlabel='Time (UTC)',
    ylabel='Altitude (km)',
    ylim=ylim)

# lidar RF18
ax = axs['top_right']
key = 'RF18'
plot_ds = (lidar_ds[key]['backscatter_ratio']
           .where((lidar_ds[key].flags == 0)
                  & (lidar_ds[key].backscatter_ratio > 1))
           .sel(time=slices[key]['above']))
plot_ds.plot(x='time', y='height', cmap=cmr.rainforest_r, norm=colors.LogNorm(),
             vmax=100, cbar_kwargs=dict(label='Backscatter ratio\nat 532$\\,$nm'),
             ax=ax)
# add below cloud altitude of HALO
altitude = bahamas_ds[key]['IRS_ALT'].sel(time=slices[key]['below']).to_numpy() / 1000
new_t = pd.date_range(plot_ds.time.to_numpy()[0], plot_ds.time.to_numpy()[-1], periods=len(altitude))
ax.plot(new_t, altitude, color='k')
h.set_xticks_and_xlabels(ax, slices[key]['above'].stop - slices[key]['above'].start)
ax.set(
    title='RF18 - 12 April 2024',
    xlabel='',
    ylabel='',
    ylim=ylim)

# radar RF18
ax = axs['bottom_right']
key = 'RF18'
plot_ds = (radar_ds[key]['dBZg']
           .sel(time=slices[key]['above']))
plot_ds.plot(x='time', y='height', cmap=cmr.torch_r, vmax=0, vmin=-50,
             cbar_kwargs=dict(label='Equivalent reflectivity\nfactor (dBZ)'),
             ax=ax)
# add below cloud altitude of HALO
altitude = bahamas_ds[key]['IRS_ALT'].sel(time=slices[key]['below']).to_numpy() / 1000
new_t = pd.date_range(plot_ds.time.to_numpy()[0], plot_ds.time.to_numpy()[-1], periods=len(altitude))
ax.plot(new_t, altitude, color='k')
h.set_xticks_and_xlabels(ax, slices[key]['above'].stop - slices[key]['above'].start)
ax.set(
    xlabel='Time (UTC)',
    ylabel='',
    ylim=ylim)

for k, label in zip(axs, ['(a)', '(b)', '(c)', '(d)']):
    axs[k].text(0.01, 0.91, label, transform=axs[k].transAxes)

figname = f'{plot_path}/03_HALO-AC3_HALO_RF17_RF18_radar_lidar_backscatter.png'
plt.savefig(figname, dpi=600)
plt.show()
plt.close()

# %% find mean cloud base height and top in case studies
for key in keys:
    print(f'{key}\n')
    mask = radar_ds[key]['radar_lidar_mask'] != 3
    cloud_props_radar, bases_tops_radar = h.find_bases_tops(mask.to_numpy(), mask.height.to_numpy())
    bases_tops_radar = xr.DataArray(bases_tops_radar, dims=["time", "height"],
                                    coords=dict(time=mask.time, height=mask.height))

    tmp = bases_tops_radar.sel(time=slices[key]['above'])
    tmp.plot(x='time', ylim=(3, 8.5))
    plt.title(f'{key} - Bases and Tops')
    plt.show()
    plt.close()
    cbh = tmp.height.where(tmp == -1)
    cbh = cbh.min(dim='height').dropna(dim='time')
    print(f'Mean cloud base height at beginning: {cbh[:60].mean().to_numpy():.2f} km')
    print(f'Mean cloud base height at end: {cbh[-60:].mean().to_numpy():.2f} km')
    cbh.plot(x='time', ylim=(3, 7))
    plt.title(f'{key} - Cloud base height')
    plt.show()
    plt.close()
    cth = tmp.height.where(tmp == 1)
    print(f'Mean cloud top height for above cloud section: {cth.max(dim='height').mean():.2f} km')
