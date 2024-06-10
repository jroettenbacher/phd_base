#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 10.06.2024

Plots for Manfred's 15-minute talk at the IRS in China 2024


"""

# %% import modules
import os

import cartopy.crs as ccrs
import cmasher as cm
import dill
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib import patheffects
from sklearn.neighbors import BallTree
from tqdm import tqdm

import pylim.halo_ac3 as meta
import pylim.helpers as h
import pylim.meteorological_formulas as met
from pylim import ecrad

h.set_cb_friendly_colors('petroff_6')
cbc = h.get_cb_friendly_colors('petroff_6')

# %% set paths
campaign = 'halo-ac3'
plot_path = 'C:/Users/Johannes/Documents/Doktor/ppt_gallery/IRS2024_for_Manfred'
save_path = h.get_path('plot', campaign=campaign)
trajectory_path = f'{h.get_path('trajectories', campaign=campaign)}/selection_CC_and_altitude'
keys = ['RF17', 'RF18']
ecrad_versions = ['v15', 'v15.1', 'v16', 'v17', 'v18', 'v18.1', 'v19', 'v19.1', 'v20', 'v21', 'v28', 'v29',
                  'v30.1', 'v31.1', 'v32.1', 'v33', 'v34', 'v35']

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
    dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(),
    dict(), dict(), dict(), dict(), dict(), dict()
)

for key in keys:
    flight = meta.flight_names[key]
    date = flight[9:17]
    bacardi_path = h.get_path('bacardi', flight, campaign)
    bahamas_path = h.get_path('bahamas', flight, campaign)
    libradtran_path = h.get_path('libradtran', flight, campaign)
    libradtran_exp_path = h.get_path('libradtran_exp', flight, campaign)
    ifs_path = f'{h.get_path("ifs", flight, campaign)}/{date}'
    ecrad_path = f'{h.get_path("ecrad", flight, campaign)}/{date}'
    varcloud_path = h.get_path('varcloud', flight, campaign)
    dropsonde_path = h.get_path('dropsondes', flight, campaign)
    dropsonde_path = f'{dropsonde_path}/Level_1' if key == 'RF17' else f'{dropsonde_path}/Level_2'
    radar_path = h.get_path('hamp_mira', flight, campaign)
    lidar_path = h.get_path('wales', flight, campaign)

    # filenames
    bahamas_file = f'HALO-AC3_HALO_BAHAMAS_{date}_{key}_v1_JR.nc'
    bacardi_file = f'HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1_JR.nc'
    libradtran_bb_solar_si = f'HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_solar_si_{date}_{key}.nc'
    libradtran_bb_thermal_si = f'HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_thermal_si_{date}_{key}.nc'
    ifs_file = f'ifs_{date}_00_ml_O1280_processed.nc'
    varcloud_file = [f for f in os.listdir(varcloud_path) if f.endswith('.nc')][0]
    dropsonde_files = [f for f in os.listdir(dropsonde_path) if f.endswith('.nc')]
    radar_file = f'HALO_HALO_AC3_radar_unified_{key}_{date}_v2.6.nc'
    lidar_file = f'HALO-AC3_HALO_WALES_bsrgl_{date}_{key}_V2.0.nc'

    # read in aircraft data
    bahamas_ds[key] = xr.open_dataset(f'{bahamas_path}/{bahamas_file}')
    bacardi = xr.open_dataset(f'{bacardi_path}/{bacardi_file}')
    radar = xr.open_dataset(f'{radar_path}/{radar_file}')
    lidar = xr.open_dataset(f'{lidar_path}/{lidar_file}')

    lidar = lidar.rename(altitude='height').transpose('time', 'height')
    lidar['height'] = lidar.height / 1000
    radar['height'] = radar.height / 1000
    # interpolate lidar data onto radar range resolution
    new_range = radar.height.values
    lidar_r = lidar.interp(height=np.flip(new_range))
    # convert lidar data to radar convention: [time, height], ground = 0m
    lidar_r = lidar_r.assign_coords(height=np.flip(new_range)).isel(height=slice(None, None, -1))
    # create radar mask
    radar['mask'] = ~np.isnan(radar['dBZg'])
    # combine radar and lidar mask
    lidar_mask = lidar_r['flags'] == 0
    lidar_mask = lidar_mask.where(lidar_mask == 0, 2).resample(time='1s').first()
    radar['radar_lidar_mask'] = radar['mask'] + lidar_mask

    radar_ds[key] = radar
    lidar_ds[key] = lidar

    # read in dropsonde data
    dropsondes = dict()
    for file in dropsonde_files:
        k = file[-11:-5] if key == 'RF17' else file[27:35].replace('_', '')
        dropsondes[k] = xr.open_dataset(f'{dropsonde_path}/{file}')
        if key == 'RF18':
            dropsondes[k]['ta'] = dropsondes[k].ta - 273.15
            dropsondes[k]['rh'] = dropsondes[k].rh * 100
    dropsonde_ds[key] = dropsondes

    # read in ifs data
    ifs = xr.open_dataset(f'{ifs_path}/{ifs_file}')
    ifs = ifs.set_index(rgrid=['lat', 'lon'])
    # filter low clouds according to ECMWF low cloud criterion (pressure higher than 0.8 * surface pressure)
    cloud_data = ifs[['q_liquid', 'q_ice', 'cloud_fraction', 'clwc', 'ciwc', 'crwc', 'cswc']]
    pressure_filter = ifs.pressure_full.sel(level=137) * 0.8
    low_cloud_filter = ifs.pressure_full < pressure_filter  # False for low clouds
    cloud_data = cloud_data.where(low_cloud_filter, 0)  # replace where False with 0
    ifs.update(cloud_data)
    ifs_ds[key] = ifs.copy(deep=True)

    # read in varcloud data
    varcloud = xr.open_dataset(f'{varcloud_path}/{varcloud_file}')
    varcloud = varcloud.swap_dims(time='Time', height='Height').rename(Time='time', Height='height')
    varcloud = varcloud.rename(Varcloud_Cloud_Ice_Water_Content='iwc', Varcloud_Cloud_Ice_Effective_Radius='re_ice')
    varcloud_ds[key] = varcloud

    # read in libRadtran simulation
    bb_sim_solar_si = xr.open_dataset(f'{libradtran_exp_path}/{libradtran_bb_solar_si}')
    bb_sim_thermal_si = xr.open_dataset(f'{libradtran_exp_path}/{libradtran_bb_thermal_si}')
    # interpolate simualtion onto BACARDI time
    bb_sim_solar_si_inp = bb_sim_solar_si.interp(time=bacardi.time)
    bb_sim_thermal_si_inp = bb_sim_thermal_si.interp(time=bacardi.time)

    # calculate transmissivity BACARDI/libRadtran
    bacardi['F_down_solar_sim_si'] = bb_sim_solar_si_inp.fdw
    bacardi['F_down_terrestrial_sim_si'] = bb_sim_thermal_si_inp.edn
    bacardi['F_up_terrestrial_sim_si'] = bb_sim_thermal_si_inp.eup
    bacardi['transmissivity_solar'] = bacardi['F_down_solar'] / bb_sim_solar_si_inp.fdw
    bacardi['transmissivity_terrestrial'] = bacardi['F_down_terrestrial'] / bb_sim_thermal_si_inp.edn
    # calculate reflectivity
    bacardi['reflectivity_solar'] = bacardi['F_up_solar'] / bacardi['F_down_solar']
    bacardi['reflectivity_terrestrial'] = bacardi['F_up_terrestrial'] / bacardi['F_down_terrestrial']

    # calculate radiative effect from BACARDI and libRadtran sea ice simulation
    bacardi['F_net_solar'] = bacardi['F_down_solar'] - bacardi['F_up_solar']
    bacardi['F_net_terrestrial'] = bacardi['F_down_terrestrial'] - bacardi['F_up_terrestrial']
    bacardi['F_net_solar_sim'] = bb_sim_solar_si_inp.fdw - bb_sim_solar_si_inp.eup
    bacardi['F_net_terrestrial_sim'] = bb_sim_thermal_si_inp.edn - bb_sim_thermal_si_inp.eup
    bacardi['CRE_solar'] = bacardi['F_net_solar'] - bacardi['F_net_solar_sim']
    bacardi['CRE_terrestrial'] = bacardi['F_net_terrestrial'] - bacardi['F_net_terrestrial_sim']
    bacardi['CRE_total'] = bacardi['CRE_solar'] + bacardi['CRE_terrestrial']
    bacardi['F_down_solar_error'] = np.abs(bacardi['F_down_solar'] * 0.03)
    # normalize downward irradiance for cos SZA
    for var in ['F_down_solar', 'F_down_solar_diff']:
        bacardi[f'{var}_norm'] = bacardi[var] / np.cos(np.deg2rad(bacardi['sza']))
    # filter data for motion flag
    bacardi_org = bacardi.copy()
    bacardi = bacardi.where(bacardi.motion_flag)
    # overwrite variables which do not need to be filtered with original values
    for var in ['alt', 'lat', 'lon', 'sza', 'saa', 'attitude_flag', 'segment_flag', 'motion_flag']:
        bacardi[var] = bacardi_org[var]

    # read in resampled BACARDI data
    bacardi_res = xr.open_dataset(f'{bacardi_path}/{bacardi_file.replace('.nc', '_1Min.nc')}')
    # normalize downward irradiance for cos SZA
    for var in ['F_down_solar', 'F_down_solar_diff']:
        bacardi_res[f'{var}_norm'] = bacardi_res[var] / np.cos(np.deg2rad(bacardi_res['sza']))
    bacardi_ds_res[key] = bacardi_res.copy()

    # read in ecrad data
    ecrad_dict, ecrad_org = dict(), dict()

    for k in ecrad_versions:
        ds = xr.open_dataset(f'{ecrad_path}/ecrad_merged_inout_{date}_{k}.nc')
        ecrad_org[k] = ds.copy(deep=True)
        # select only center column for direct comparisons
        ds = ds.sel(column=0, drop=True) if 'column' in ds.dims else ds
        ecrad_dict[k] = ds.copy(deep=True)

    ecrad_dicts[key] = ecrad_dict
    ecrad_orgs[key] = ecrad_org

    # interpolate standard ecRad simulation onto BACARDI time
    bacardi['ecrad_fdw'] = ecrad_dict['v15.1'].flux_dn_sw_clear.interp(time=bacardi.time,
                                                                       kwargs={'fill_value': 'extrapolate'})
    # calculate transmissivity using ecRad at TOA and above cloud
    bacardi['transmissivity_TOA'] = bacardi['F_down_solar'] / bacardi['ecrad_fdw'].isel(half_level=0)
    bacardi['transmissivity_above_cloud'] = bacardi['F_down_solar'] / bacardi['ecrad_fdw'].isel(half_level=73)
    # normalize transmissivity by cosine of solar zenith angle
    for var in ['transmissivity_TOA', 'transmissivity_above_cloud']:
        bacardi[f'{var}_norm'] = bacardi[var] / np.cos(np.deg2rad(bacardi['sza']))
    bacardi_ds[key] = bacardi

    # read in time slices
    loaded_objects = list()
    filenames = [f'{key}_slices.pkl', f'{key}_above_cloud.pkl', f'{key}_below_cloud.pkl']
    for filename in filenames:
        with open(f'{save_path}/{filename}', 'rb') as f:
            loaded_objects.append(dill.load(f))

    slices[key] = loaded_objects[0]
    above_clouds[key] = loaded_objects[1]
    below_clouds[key] = loaded_objects[2]

    # get IFS data for the case study area
    ifs_lat_lon = np.column_stack((ifs.lat, ifs.lon))
    ifs_tree = BallTree(np.deg2rad(ifs_lat_lon), metric='haversine')
    # generate an array with lat, lon values from the flight position
    bahamas_sel = bahamas_ds[key].sel(time=slices[key]['above'])
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
plt.rc('font', size=10)
data_crs = ccrs.PlateCarree()
map_crs = ccrs.NorthPolarStereo()
for key in keys:
    ts = '2022-04-11 12:00' if key == 'RF17' else '2022-04-12 12:00'
    ifs = ifs_ds[key].sel(time=ts, method='nearest')
    _, ax = plt.subplots(figsize=(12 * h.cm, 9 * h.cm),
                         layout='constrained',
                         subplot_kw=dict(projection=map_crs))

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
    ci_var = 'ci'
    cci = ax.tricontour(ifs.lon, ifs.lat, ifs[ci_var], ci_levels, transform=data_crs, linestyles='--', colors='#332288',
                        linewidths=2)

    # add high cloud cover
    ifs_cc = ifs.cloud_fraction.where(ifs.pressure_full < 60000, drop=True).sum(dim='level')
    ifs_cc = ifs_cc / 101  # divide by number of high cloud levels
    hcc = ax.tricontourf(ifs.lon, ifs.lat, ifs_cc, levels=24, transform=data_crs, cmap='Blues', alpha=1)
    cbar = plt.colorbar(hcc, label='Total high cloud fraction')

    # plot flight track
    ins = bahamas_ds[key]
    track_lons, track_lats = ins['IRS_LON'], ins['IRS_LAT']
    ax.plot(track_lons[::10], track_lats[::10], color='k',
            label='Flight track', transform=data_crs)

    # plot dropsonde locations
    ds_dict = dropsonde_ds[key]
    for i, ds in enumerate(ds_dict.values()):
        launch_time = pd.to_datetime(ds.launch_time.to_numpy()) if key == 'RF17' else pd.to_datetime(
            ds.time[0].to_numpy())
        x, y = ds.lon.mean().to_numpy(), ds.lat.mean().to_numpy()
        cross = ax.plot(x, y, 'x', color=cbc[1], markersize=9, transform=data_crs)
        if key == 'RF17':
            ax.text(x, y, f'{launch_time:%H:%M}', c='k', fontsize=9, transform=data_crs,
                    path_effects=[patheffects.withStroke(linewidth=0.5, foreground='white')])
        else:
            for i in [1]:
                ds = list(ds_dict.values())[i]
                launch_time = pd.to_datetime(ds.time[0].to_numpy())
                x, y = ds.lon.mean().to_numpy(), ds.lat.mean().to_numpy()
                ax.text(x, y, f'{launch_time:%H:%M}', color='k', fontsize=9, transform=data_crs,
                        path_effects=[patheffects.withStroke(linewidth=0.5, foreground='white')])

    coords = meta.coordinates['Kiruna']
    ax.plot(coords[0], coords[1], ls='', marker='^', color=cbc[2], label='Kiruna', transform=data_crs)
    ax.plot([], ls='--', color='#332288', label='Sea ice edge')
    ax.plot([], ls='', marker='x', color=cbc[1], label='Dropsonde')
    ax.legend(loc=3)

    figname = f'{plot_path}/HALO-AC3_{key}_fligh_track_IFS_cloud_cover.png'
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot zoom of case study region RF 18
key = 'RF18'
ts = '2022-04-12 12:00'
ifs = ifs_ds[key].sel(time=ts, method='nearest')
plt.rc('font', size=5)
fig, ax = plt.subplots(figsize=(2 * h.cm, 2.5 * h.cm),
                       subplot_kw={'projection': map_crs},
                       layout='constrained')

ax.coastlines(alpha=0.5)
ax.set_extent([-20, 22, 87, 90], crs=data_crs)
gl = ax.gridlines(crs=data_crs, draw_labels=False, linewidth=1, color='gray', alpha=0.5,
                  linestyle=':', x_inline=False, y_inline=False, rotate_labels=False)

# add high cloud cover
ifs_cc = ifs.cloud_fraction.where(ifs.pressure_full < 60000, drop=True).sum(dim='level')
ifs_cc = ifs_cc / 101
ax.tricontourf(ifs.lon, ifs.lat, ifs_cc, levels=24, transform=data_crs, cmap='Blues', alpha=1)

# plot flight track
ins = bahamas_ds[key]
track_lons, track_lats = ins['IRS_LON'], ins['IRS_LAT']
ax.plot(track_lons[::10], track_lats[::10], color='k',
        label='Flight track', transform=data_crs)

# plot dropsonde locations - 12 April
ds_dict = dropsonde_ds['RF18']
for i in [0, 3, 6, 10, 13]:
    ds = list(ds_dict.values())[i]
    launch_time = pd.to_datetime(ds.time[0].to_numpy())
    lon, lat = ds.lon.dropna(dim='time'), ds.lat.dropna(dim='time')
    x, y = lon[0].to_numpy(), lat[0].to_numpy()
    cross = ax.plot(x, y, 'x', color='orangered', markersize=4, label='Dropsonde', transform=data_crs,
                    zorder=450)
    ax.text(x, y + .1, f'{launch_time:%H:%M}', color='k', transform=data_crs, zorder=500,
            path_effects=[patheffects.withStroke(linewidth=0.5, foreground='white')])

figname = f'{plot_path}/HALO-AC3_RF18_flight_track_trajectories_plot_overview_zoom.png'
plt.savefig(figname, dpi=600)
plt.show()
plt.close()

# %% visualise grid points from IFS selected for case study areas
key = 'RF17'
extents = dict(RF17=[-15, 20, 85.5, 89], RF18=[-25, 17, 87.75, 89.85])
ds = ifs_ds_sel[key]
ds1 = bahamas_ds[key]
# plot points along flight track
data_crs = ccrs.PlateCarree()
plot_crs = ccrs.NorthPolarStereo()
plot_ds = ds.sel(time=slices[key]['above'])
plot_ds1 = ds1.sel(time=slices[key]['case'])
_, ax = plt.subplots(figsize=h.figsize_equal, subplot_kw=dict(projection=plot_crs))
ax.set_extent(extents[key], crs=data_crs)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.plot(plot_ds1.IRS_LON, plot_ds1.IRS_LAT, c='k', transform=data_crs, label='Flight track')
ax.scatter(plot_ds.lon, plot_ds.lat, s=10, c=cbc[0], transform=data_crs,
           label=f'IFS grid points n={len(plot_ds.lon)}')
plot_ds_single = plot_ds1.sel(time=slices[key]['above'].stop, method='nearest')
ax.scatter(plot_ds_single.IRS_LON, plot_ds_single.IRS_LAT, marker='*', s=50, c=cbc[1], label='Start of descent',
           transform=data_crs)
ax.legend()
figname = f'{plot_path}/{key}_case_study_gridpoints.png'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot lidar data for case studies
for key in keys:
    plot_ds = (lidar_ds[key]['backscatter_ratio']
               .where((lidar_ds[key].flags == 0)
                      & (lidar_ds[key].backscatter_ratio > 1))
               .sel(time=slices[key]['above']))
    plt.rc('font', size=8.5)
    _, ax = plt.subplots(figsize=(12.5 * h.cm, 5 * h.cm), layout='constrained')
    plot_ds.plot(x='time', y='height', cmap=cm.rainforest_r, norm=mpl.colors.LogNorm(), vmax=50,
                 cbar_kwargs=dict(label='Lidar\nbackscatter ratio', pad=0.01))
    # ax.plot(bahamas_ds[key].time, bahamas_ds[key]['IRS_ALT'] / 1000, color='grey', label='HALO altitude')
    h.set_xticks_and_xlabels(ax, slices[key]['above'].stop - slices[key]['above'].start)
    ax.set(xlabel='Time (UTC)', ylabel='Altitude (km)', ylim=(0, 12))

    figname = f'{plot_path}/HALO_AC3_HALO_{key}_lidar_backscatter_ratio_532.png'
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot radar data for case studies
for key in keys:
    plot_ds = (radar_ds[key]['dBZg']
               .sel(time=slices[key]['above']))
    t_plot = ecrad_dicts[key]['v15.1'].t
    plt.rc('font', size=8.5)
    _, ax = plt.subplots(figsize=(12.5 * h.cm, 5 * h.cm), layout='constrained')
    plot_ds.plot(x='time', y='height', cmap=cm.torch_r, vmin=-45, vmax=15,
                 cbar_kwargs=dict(label='Radar reflectivity (dBZ)', pad=0.01, ticks=range(-40, 20, 10),
                                  extend='both'))
    h.set_xticks_and_xlabels(ax, slices[key]['above'].stop - slices[key]['above'].start)
    ax.set(xlabel='Time (UTC)', ylabel='Altitude (km)', ylim=(0, 12))

    figname = f'{plot_path}/HALO_AC3_HALO_{key}_radar_reflectivity.png'
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot lidar and radar data for case studies
for key in keys:
    lidar_plot = (lidar_ds[key]['backscatter_ratio']
                  .where((lidar_ds[key].flags == 0)
                         & (lidar_ds[key].backscatter_ratio > 1))
                  .sel(time=slices[key]['above']))
    radar_plot = (radar_ds[key]['dBZg']
                  .sel(time=slices[key]['above']))
    ds = ecrad_dicts[key]['v15.1'].sel(time=slices[key]['case'])
    ifs_plot = ds['t']
    # add new z axis mean pressure altitude
    if 'half_level' in ifs_plot.dims:
        new_z = ds['press_height_hl'].mean(dim='time') / 1000
    else:
        new_z = ds['press_height_full'].mean(dim='time') / 1000

    ifs_plot_new_z = list()
    for t in tqdm(ifs_plot.time, desc='New Z-Axis'):
        tmp_plot = ifs_plot.sel(time=t)
        if 'half_level' in tmp_plot.dims:
            tmp_plot = tmp_plot.assign_coords(
                half_level=ds['press_height_hl'].sel(time=t, drop=True).to_numpy() / 1000)
            tmp_plot = tmp_plot.rename(half_level='height')
        else:
            tmp_plot = tmp_plot.assign_coords(
                level=ds['press_height_full'].sel(time=t, drop=True).to_numpy() / 1000)
            tmp_plot = tmp_plot.rename(level='height')

        tmp_plot = tmp_plot.interp(height=new_z.to_numpy())
        ifs_plot_new_z.append(tmp_plot)

    ifs_plot = xr.concat(ifs_plot_new_z, dim='time').sortby('height').sel(height=slice(0, 12))
    # select only tropopause temperature
    tp_sel = ifs_plot == ifs_plot.min(dim='height')
    tp_height = [tp_sel.sel(time=i).height.where(tp_sel.sel(time=i), drop=True).to_numpy()[0] for i in tp_sel.time]
    plt.rc('font', size=8.5)
    _, axs = plt.subplots(2, 1, figsize=(12.5 * h.cm, 11 * h.cm), layout='constrained')
    lidar_plot.plot(x='time', y='height', cmap=cm.rainforest_r, norm=mpl.colors.LogNorm(), vmax=50,
                    cbar_kwargs=dict(label='Lidar backscatter ratio', pad=0.01, extend='both'), ax=axs[0])
    radar_plot.plot(x='time', y='height', cmap=cm.torch_r, vmin=-45, vmax=15, ax=axs[1],
                    cbar_kwargs=dict(label='Radar reflectivity (dBZ)', pad=0.01, ticks=range(-40, 20, 10),
                                     extend='both'))
    for ax in axs:
        h.set_xticks_and_xlabels(ax, slices[key]['above'].stop - slices[key]['above'].start)
        ax.set(xlabel='',
               ylabel='Altitude (km)',
               ylim=(0, 12),
               xticklabels='',
               title=f'Research flight {key[2:]}')
        ax.plot(ifs_plot.time, tp_height, color='k', linestyle='--', label='Tropopause')

    h.set_xticks_and_xlabels(axs[1], slices[key]['above'].stop - slices[key]['above'].start)
    axs[0].legend(loc=1)
    axs[1].set(xlabel='Time (UTC)', title='')

    figname = f'{plot_path}/HALO_AC3_HALO_{key}_lidar_backscatter_ratio_532_radar_reflectivity.png'
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% solar irradiance - plot below cloud solar irradiance RF 17
key = 'RF17'
bacardi_plot = bacardi_ds[key].sel(time=slices[key]['below'])
bacardi_error = bacardi_plot * 0.03
var1, var2 = 'F_down_solar', 'F_up_solar'
plt.rc('font', size=12)
_, ax = plt.subplots(figsize=h.figsize_wide, layout='constrained')
ax.plot(bacardi_plot.time, bacardi_plot[var1],
        label=f'{h.bacardi_labels[var1]}', c=cbc[0])
ax.fill_between(bacardi_error.time,
                bacardi_plot[var1] + bacardi_error[var1],
                bacardi_plot[var1] - bacardi_error[var1],
                color=cbc[0], alpha=0.5)
ax.plot(bacardi_plot.time, bacardi_plot[var2],
        label=f'{h.bacardi_labels[var2]}', c=cbc[1])
ax.fill_between(bacardi_error.time,
                bacardi_plot[var2] + bacardi_error[var2],
                bacardi_plot[var2] - bacardi_error[var2],
                color=cbc[1], alpha=0.5)
ax.set_ylim(100, 260)
ax.grid()
ax.legend()
h.set_xticks_and_xlabels(ax, pd.Timedelta('15min'))
ax.set_xlabel('Time (UTC)')
ax.set_ylabel('Broadband solar irradiance (W$\\,$m$^{-2}$)')

figname = f'{plot_path}/HALO-AC3_HALO_{key}_BACARDI_solar_irradiance_below.png'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% solar irradiance - plot below cloud terrestrial irradiance RF 17
key = 'RF17'
bacardi_plot = bacardi_ds[key].sel(time=slices[key]['below'])
bacardi_error = bacardi_plot * 0.03
var1, var2 = 'F_down_terrestrial', 'F_up_terrestrial'
plt.rc('font', size=12)
_, ax = plt.subplots(figsize=h.figsize_wide, layout='constrained')
ax.plot(bacardi_plot.time, bacardi_plot[var1],
        label=f'{h.bacardi_labels[var1]}', c=cbc[2])
ax.fill_between(bacardi_error.time,
                bacardi_plot[var1] + bacardi_error[var1],
                bacardi_plot[var1] - bacardi_error[var1],
                color=cbc[2], alpha=0.5)
ax.plot(bacardi_plot.time, bacardi_plot[var2],
        label=f'{h.bacardi_labels[var2]}', c=cbc[3])
ax.fill_between(bacardi_error.time,
                bacardi_plot[var2] + bacardi_error[var2],
                bacardi_plot[var2] - bacardi_error[var2],
                color=cbc[3], alpha=0.5)
ax.set_ylim(0, 240)
ax.grid()
ax.legend()
h.set_xticks_and_xlabels(ax, pd.Timedelta('15min'))
ax.set_xlabel('Time (UTC)')
ax.set_ylabel('Broadband terrestrial irradiance (W$\\,$m$^{-2}$)')

figname = f'{plot_path}/HALO-AC3_HALO_{key}_BACARDI_terrestrial_irradiance_below.png'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% solar irradiance - plot BACARDI four panel plot with above and below cloud measurements
plt.rc('font', size=11.5)
xlims = [(0, 240), (0, 320)]
ylim_transmissivity = (0.45, 1)
ylim_irradiance = [(100, 279), (80, 260)]
label_xy = (0.03, 0.9)
box_xy = (0.98, 0.9)
_, axs = plt.subplots(2, 2, figsize=(22 * h.cm, 12 * h.cm),
                      layout='constrained')

# upper left panel - RF17 BACARDI F above cloud
ax = axs[0, 0]
plot_ds = bacardi_ds['RF17'].sel(time=slices['RF17']['above'])
plot_ds['distance'] = bahamas_ds['RF17']['distance'].sel(time=slices['RF17']['above'])
# set first distance to 0
plot_ds['distance'][0] = 0
# sum up distances to generate a distance axis and convert to km
plot_ds['cum_distance'] = plot_ds['distance'].cumsum() / 1000
# bacardi measurements
for var in ['F_down_solar', 'F_up_solar']:
    ax.plot(plot_ds.cum_distance, plot_ds[var], label=f'{h.bacardi_labels[var]}')
ax.legend(loc=4, ncol=2)
ax.grid()
ax.text(box_xy[0], box_xy[1], 'Above cloud', ha='right',
        transform=ax.transAxes, bbox=dict(boxstyle='Round', fc='white'))
ax.set(title='RF 17 - 11 April 2022',
       ylabel=f'Solar irradiance ({h.plot_units['flux_dn_sw']})',
       ylim=ylim_irradiance[0],
       xlim=xlims[0])

# lower left panel - RF17 BACARDI F below_cloud
ax = axs[1, 0]
plot_ds = bacardi_ds['RF17'].sel(time=slices['RF17']['below'])
plot_ds['distance'] = bahamas_ds['RF17']['distance'].sel(time=slices['RF17']['below'])
# set first distance to 0
plot_ds['distance'][0] = 0
# sum up distances to generate a distance axis and convert to km, flip the distance to show travel in other direction
cum_distance = np.flip(plot_ds['distance'].cumsum().to_numpy() / 1000)
# bacardi measurements
for var in ['F_down_solar', 'F_up_solar']:
    ax.plot(cum_distance, plot_ds[var], label=f'{h.bacardi_labels[var]}')
ax.grid()
ax.text(box_xy[0], box_xy[1], 'Below cloud', ha='right',
        transform=ax.transAxes, bbox=dict(boxstyle='Round', fc='white'))
ax.set(ylabel=f'Solar irradiance ({h.plot_units['flux_dn_sw']})',
       xlabel='Distance (km)',
       ylim=ylim_irradiance[1],
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
for var in ['F_down_solar', 'F_up_solar']:
    ax.plot(plot_ds.cum_distance, plot_ds[var], label=f'{h.bacardi_labels[var]}')
ax.grid()
ax.text(box_xy[0], box_xy[1], 'Above cloud', ha='right',
        transform=ax.transAxes, bbox=dict(boxstyle='Round', fc='white'))
ax.set(title='RF 18 - 12 April 2022',
       xlabel='',
       ylim=ylim_irradiance[0],
       xlim=xlims[1])

# lower right panel - RF18 BACARDI F below cloud
ax = axs[1, 1]
plot_ds = bacardi_ds['RF18'].sel(time=slices['RF18']['below'])
plot_ds['distance'] = bahamas_ds['RF18']['distance'].sel(time=slices['RF18']['below'])
# set first distance to 0
plot_ds['distance'][0] = 0
# sum up distances to generate a distance axis and convert to km
cum_distance = plot_ds['distance'].cumsum().to_numpy() / 1000
# bacardi measurements
for var in ['F_down_solar', 'F_up_solar']:
    ax.plot(cum_distance, plot_ds[var], label=f'{h.bacardi_labels[var]}')
ax.grid()
ax.text(box_xy[0], box_xy[1], 'Below cloud', ha='right',
        transform=ax.transAxes, bbox=dict(boxstyle='Round', fc='white'))
ax.set(ylabel='',
       xlabel='Distance (km)',
       ylim=ylim_irradiance[1],
       xlim=xlims[1])

# set a-f labels
for ax, label in zip(axs.flatten(), ['(a)', '(b)', '(c)', '(d)']):
    ax.text(label_xy[0], label_xy[1], label, transform=ax.transAxes)

figname = f'{plot_path}/HALO-AC3_HALO_RF17_RF18_BACARDI_solar_irradiance_4panel.png'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% solar irradiance - plot BACARDI six panel plot with above and below cloud measurements and transmissivity
plt.rc('font', size=9)
xlims = [(0, 240), (0, 320)]
ylim_transmissivity = (0.45, 1)
ylim_irradiance = [(100, 279), (80, 260)]
label_xy = (0.03, 0.9)
box_xy = (0.98, 0.9)
_, axs = plt.subplots(3, 2, figsize=(18 * h.cm, 15 * h.cm),
                      layout='constrained')

# upper left panel - RF17 BACARDI F above cloud
ax = axs[0, 0]
plot_ds = bacardi_ds['RF17'].sel(time=slices['RF17']['above'])
plot_ds['distance'] = bahamas_ds['RF17']['distance'].sel(time=slices['RF17']['above'])
# set first distance to 0
plot_ds['distance'][0] = 0
# sum up distances to generate a distance axis and convert to km
plot_ds['cum_distance'] = plot_ds['distance'].cumsum() / 1000
# bacardi measurements
for var in ['F_down_solar', 'F_up_solar']:
    ax.plot(plot_ds.cum_distance, plot_ds[var], label=f'{h.bacardi_labels[var]}')
ax.legend(loc=4, ncol=2)
ax.grid()
ax.text(box_xy[0], box_xy[1], 'Above cloud', ha='right',
        transform=ax.transAxes, bbox=dict(boxstyle='Round', fc='white'))
ax.set(title='RF 17 - 11 April 2022',
       ylabel=f'Solar irradiance ({h.plot_units['flux_dn_sw']})',
       ylim=ylim_irradiance[0],
       xlim=xlims[0])

# middle left panel - RF17 BACARDI F below_cloud
ax = axs[1, 0]
plot_ds = bacardi_ds['RF17'].sel(time=slices['RF17']['below'])
plot_ds['distance'] = bahamas_ds['RF17']['distance'].sel(time=slices['RF17']['below'])
# set first distance to 0
plot_ds['distance'][0] = 0
# sum up distances to generate a distance axis and convert to km, flip the distance to show travel in other direction
cum_distance = np.flip(plot_ds['distance'].cumsum().to_numpy() / 1000)
# bacardi measurements
for var in ['F_down_solar', 'F_up_solar']:
    ax.plot(cum_distance, plot_ds[var], label=f'{h.bacardi_labels[var]}')
ax.grid()
ax.text(box_xy[0], box_xy[1], 'Below cloud', ha='right',
        transform=ax.transAxes, bbox=dict(boxstyle='Round', fc='white'))
ax.set(ylabel=f'Solar irradiance ({h.plot_units['flux_dn_sw']})',
       ylim=ylim_irradiance[1],
       xlim=xlims[0])

# lower left panel - RF17 transmissivity
ax = axs[2, 0]
# ax.axhline(y=1, color='k')
ax.plot(cum_distance, plot_ds['transmissivity_above_cloud'], label='Solar transmissivity', color=cbc[3])
ax.grid()
ax.text(box_xy[0], box_xy[1], 'Below cloud', ha='right',
        transform=ax.transAxes, bbox=dict(boxstyle='Round', fc='white'))
ax.set(ylabel='Solar transmissivity',
       xlabel='Distance (km)',
       ylim=ylim_transmissivity,
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
for var in ['F_down_solar', 'F_up_solar']:
    ax.plot(plot_ds.cum_distance, plot_ds[var], label=f'{h.bacardi_labels[var]}')
ax.grid()
ax.text(box_xy[0], box_xy[1], 'Above cloud', ha='right',
        transform=ax.transAxes, bbox=dict(boxstyle='Round', fc='white'))
ax.set(title='RF 18 - 12 April 2022',
       ylim=ylim_irradiance[0],
       xlim=xlims[1])

# middle right panel - RF18 BACARDI F below cloud
ax = axs[1, 1]
plot_ds = bacardi_ds['RF18'].sel(time=slices['RF18']['below'])
plot_ds['distance'] = bahamas_ds['RF18']['distance'].sel(time=slices['RF18']['below'])
# set first distance to 0
plot_ds['distance'][0] = 0
# sum up distances to generate a distance axis and convert to km
cum_distance = plot_ds['distance'].cumsum().to_numpy() / 1000
# bacardi measurements
for var in ['F_down_solar', 'F_up_solar']:
    ax.plot(cum_distance, plot_ds[var], label=f'{h.bacardi_labels[var]}')
ax.grid()
ax.text(box_xy[0], box_xy[1], 'Below cloud', ha='right',
        transform=ax.transAxes, bbox=dict(boxstyle='Round', fc='white'))
ax.set(ylim=ylim_irradiance[1],
       xlim=xlims[1])

# lower right panel - RF18 transmissivity
ax = axs[2, 1]
# ax.axhline(y=1, color='k')
ax.plot(cum_distance, plot_ds['transmissivity_above_cloud'],
        label='Solar transmissivity', color=cbc[3])
ax.grid()
# ax.text(label_xy[0], label_xy[1], '(f)', transform=ax.transAxes)
ax.text(box_xy[0], box_xy[1], 'Below cloud', ha='right',
        transform=ax.transAxes, bbox=dict(boxstyle='Round', fc='white'))
ax.set(xlabel='Distance (km)',
       ylim=ylim_transmissivity,
       xlim=xlims[1])

# set a-f labels
for ax, label in zip(axs.flatten(), ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']):
    ax.text(label_xy[0], label_xy[1], label, transform=ax.transAxes)

figname = f'{plot_path}/03_HALO-AC3_HALO_RF17_RF18_BACARDI_case_studies_6panel.pdf'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% terrestrial irradiance - plot BACARDI terrestrial fluxes - 4 panel figure
plt.rc('font', size=12)
xlims = [(0, 240), (0, 320)]
ylim_net = (-175, 0)
ylim_irradiance = [(0, 280), (0, 280)]
yticks = mticker.MultipleLocator(50)
label_xy = (0.03, 0.9)
box_xy = (0.98, 0.9)
_, axs = plt.subplots(2, 2, figsize=(22 * h.cm, 12 * h.cm),
                      layout='constrained')

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
    ax.plot(plot_ds.cum_distance, plot_ds[var], label=h.bacardi_labels[var], color=cbc[2+i])
ax.legend(loc=5)
ax.grid()
ax.text(box_xy[0], box_xy[1], 'Above cloud', ha='right',
        transform=ax.transAxes, bbox=dict(boxstyle='Round', fc='white'))
ax.set(title='RF 17 - 11 April 2022',
       ylabel=f'Terrestrial\nirradiance ({h.plot_units['flux_dn_sw']})',
       ylim=ylim_irradiance[0],
       xlim=xlims[0])
ax.yaxis.set_major_locator(yticks)

# lower left panel - RF17 BACARDI F below_cloud
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
    ax.plot(cum_distance, plot_ds[var], label=h.bacardi_labels[var], color=cbc[2+i])
ax.grid()
ax.text(box_xy[0], box_xy[1], 'Below cloud', ha='right',
        transform=ax.transAxes, bbox=dict(boxstyle='Round', fc='white'))
ax.yaxis.set_major_locator(yticks)
ax.set(ylabel=f'Terrestrial\nirradiance ({h.plot_units['flux_dn_sw']})',
       ylim=ylim_irradiance[1],
       xlim=xlims[0],
       xlabel='Distance (km)',
       )

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
    ax.plot(plot_ds.cum_distance, plot_ds[var], label=f'{h.bacardi_labels[var]}', color=cbc[2+i])
ax.grid()
ax.text(box_xy[0], box_xy[1], 'Above cloud', ha='right',
        transform=ax.transAxes, bbox=dict(boxstyle='Round', fc='white'))
ax.set(title='RF 18 - 12 April 2022',
       ylim=ylim_irradiance[0],
       xlim=xlims[1])
ax.yaxis.set_major_locator(yticks)

# lower right panel - RF18 BACARDI F below cloud
ax = axs[1, 1]
plot_ds = bacardi_ds['RF18'].sel(time=slices['RF18']['below'])
plot_ds['distance'] = bahamas_ds['RF18']['distance'].sel(time=slices['RF18']['below'])
# set first distance to 0
plot_ds['distance'][0] = 0
# sum up distances to generate a distance axis and convert to km
plot_ds['cum_distance'] = plot_ds['distance'].cumsum() / 1000
# bacardi measurements
for i, var in enumerate(['F_down_terrestrial', 'F_up_terrestrial']):
    ax.plot(plot_ds['cum_distance'], plot_ds[var].to_numpy(), label=f'{h.bacardi_labels[var]}', color=cbc[2+i])
ax.grid()
ax.text(box_xy[0], box_xy[1], 'Below cloud', ha='right',
        transform=ax.transAxes, bbox=dict(boxstyle='Round', fc='white'))
ax.yaxis.set_major_locator(yticks)
ax.set(ylim=ylim_irradiance[1],
       xlim=xlims[1],
       xlabel='Distance (km)',
       )

# set a-d labels
for ax, label in zip(axs.flatten(), ['(a)', '(b)', '(c)', '(d)']):
    ax.text(label_xy[0], label_xy[1], label, transform=ax.transAxes)

figname = f'{plot_path}/HALO-AC3_RF17_RF18_BACARDI_terrestrial_irradiance_4panel.png'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% terrestrial irradiance - plot BACARDI net terrestrial irradiance
plt.rc('font', size=16)
xlims = [(0, 240), (0, 320)]
ylim_net = (-175, 0)
yticks = mticker.MultipleLocator(50)
label_xy = (0.03, 0.9)
box_xy = (0.98, 0.9)
_, axs = plt.subplots(1, 2, figsize=h.figsize_wide, layout='constrained')

# left panel - RF17 net terrestrial above and below cloud
ax = axs[0]
plot_ds = bacardi_ds['RF17'].sel(time=slices['RF17']['below'])
plot_ds['distance'] = bahamas_ds['RF17']['distance'].sel(time=slices['RF17']['below'])
# set first distance to 0
plot_ds['distance'][0] = 0
# sum up distances to generate a distance axis and convert to km, flip the distance to show travel in other
# direction
cum_distance = np.flip(plot_ds['distance'].cumsum().to_numpy() / 1000)
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
ax.set(title='RF 17 - 11 April 2022',
       ylabel=f'Net terrestrial irradiance ({h.plot_units['flux_dn_lw']})',
       xlabel='Distance (km)',
       ylim=ylim_net,
       xlim=xlims[0])

# right panel - RF18 net irradiance
ax = axs[1]
plot_ds = bacardi_ds['RF18'].sel(time=slices['RF18']['below'])
plot_ds['distance'] = bahamas_ds['RF18']['distance'].sel(time=slices['RF18']['below'])
# set first distance to 0
plot_ds['distance'][0] = 0
# sum up distances to generate a distance axis and convert to km
plot_ds['cum_distance'] = plot_ds['distance'].cumsum() / 1000
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
ax.set(title='RF 18 - 12 April 2022',
       xlabel='Distance (km)',
       ylim=ylim_net,
       xlim=xlims[1])

# set a,b labels
for ax, label in zip(axs.flatten(), ['(a)', '(b)']):
    ax.text(label_xy[0], label_xy[1], label, transform=ax.transAxes)

figname = f'{plot_path}/HALO-AC3_RF17_RF18_BACARDI_net_terrestrial_irradiance.png'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()


# %% terrestrial irradiance - plot BACARDI terrestrial fluxes - comparison above below
plt.rc('font', size=14)
xlims = [(0, 240), (0, 320)]
ylim_irradiance = [(100, 280), (0, 230)]
yticks = mticker.MultipleLocator(50)
label_xy = (0.03, 0.9)
box_xy = (0.98, 0.85)
_, axs = plt.subplots(2, 2, figsize=(22 * h.cm, 12 * h.cm),
                      layout='constrained')

key = 'RF17'
# upper left panel - RF17 BACARDI Fup
ax = axs[0, 0]
var = 'F_up_terrestrial'
# above cloud
plot_ds = bacardi_ds[key].sel(time=slices[key]['above'])
plot_ds['distance'] = bahamas_ds[key]['distance'].sel(time=slices[key]['above'])
# set first distance to 0
plot_ds['distance'][0] = 0
# sum up distances to generate a distance axis and convert to km
plot_ds['cum_distance'] = plot_ds['distance'].cumsum() / 1000
# bacardi measurements
ax.plot(plot_ds.cum_distance, plot_ds[var],
        label='Above cloud',
        color=cbc[2])
# below cloud
plot_ds = bacardi_ds[key].sel(time=slices[key]['below'])
plot_ds['distance'] = bahamas_ds[key]['distance'].sel(time=slices[key]['below'])
# set first distance to 0
plot_ds['distance'][0] = 0
# sum up distances to generate a distance axis and convert to km,
# flip the distance to show travel in other direction
cum_distance = np.flip(plot_ds['distance'].cumsum().to_numpy() / 1000)
# bacardi measurements
ax.plot(cum_distance, plot_ds[var],
        label='Below cloud',
        color=cbc[3])
# aesthetics
ax.legend(loc=4)
ax.grid()
ax.text(box_xy[0], box_xy[1], '$F_{\\uparrow}$', ha='right',
        transform=ax.transAxes, bbox=dict(boxstyle='Round', fc='white'))
ax.set(title='RF 17 - 11 April 2022',
       ylabel=f'Upward terrestrial\nirradiance ({h.plot_units['flux_dn_sw']})',
       ylim=ylim_irradiance[0],
       xlim=xlims[0])
ax.yaxis.set_major_locator(yticks)

# lower left panel - RF17 BACARDI Fdown
ax = axs[1, 0]
var = 'F_down_terrestrial'
# above cloud
plot_ds = bacardi_ds[key].sel(time=slices[key]['above'])
plot_ds['distance'] = bahamas_ds[key]['distance'].sel(time=slices[key]['above'])
# set first distance to 0
plot_ds['distance'][0] = 0
# sum up distances to generate a distance axis and convert to km
plot_ds['cum_distance'] = plot_ds['distance'].cumsum() / 1000
# bacardi measurements
ax.plot(plot_ds.cum_distance, plot_ds[var],
        label='Above cloud',
        color=cbc[2])
# below cloud
plot_ds = bacardi_ds[key].sel(time=slices[key]['below'])
plot_ds['distance'] = bahamas_ds[key]['distance'].sel(time=slices[key]['below'])
# set first distance to 0
plot_ds['distance'][0] = 0
# sum up distances to generate a distance axis and convert to km, flip the distance to show travel in other
# direction
cum_distance = np.flip(plot_ds['distance'].cumsum().to_numpy() / 1000)
# bacardi measurements
ax.plot(cum_distance, plot_ds[var], label='Below cloud', color=cbc[3])
ax.grid()
ax.text(box_xy[0], box_xy[1], '$F_{\\downarrow}$', ha='right',
        transform=ax.transAxes, bbox=dict(boxstyle='Round', fc='white'))
ax.yaxis.set_major_locator(yticks)
ax.set(ylabel=f'Downward terrestrial\nirradiance ({h.plot_units['flux_dn_sw']})',
       ylim=ylim_irradiance[1],
       xlim=xlims[0],
       xlabel='Distance (km)',
       )

key = 'RF18'
# upper right panel - RF18 BACARDI Fup
var = 'F_up_terrestrial'
ax = axs[0, 1]
# above cloud
plot_ds = bacardi_ds[key].sel(time=slices[key]['above'])
plot_ds['distance'] = bahamas_ds[key]['distance'].sel(time=slices[key]['above'])
# set first distance to 0
plot_ds['distance'][0] = 0
# sum up distances to generate a distance axis and convert to km
plot_ds['cum_distance'] = plot_ds['distance'].cumsum() / 1000
# bacardi measurements
ax.plot(plot_ds.cum_distance, plot_ds[var],
        label='Above cloud',
        color=cbc[2])
# below cloud
plot_ds = bacardi_ds[key].sel(time=slices[key]['below'])
plot_ds['distance'] = bahamas_ds[key]['distance'].sel(time=slices[key]['below'])
# set first distance to 0
plot_ds['distance'][0] = 0
# sum up distances to generate a distance axis and convert to km
plot_ds['cum_distance'] = plot_ds['distance'].cumsum() / 1000
# bacardi measurements
ax.plot(plot_ds['cum_distance'], plot_ds[var],
        label='Below cloud',
        color=cbc[3])
# aesthetics
ax.grid()
ax.text(box_xy[0], box_xy[1], '$F_{\\uparrow}$', ha='right',
        transform=ax.transAxes, bbox=dict(boxstyle='Round', fc='white'))
ax.set(title='RF 18 - 12 April 2022',
       ylim=ylim_irradiance[0],
       xlim=xlims[0])
ax.yaxis.set_major_locator(yticks)

# lower right panel - RF18 BACARDI Fdown
var = 'F_down_terrestrial'
ax = axs[1, 1]
# above cloud
plot_ds = bacardi_ds[key].sel(time=slices[key]['above'])
plot_ds['distance'] = bahamas_ds[key]['distance'].sel(time=slices[key]['above'])
# set first distance to 0
plot_ds['distance'][0] = 0
# sum up distances to generate a distance axis and convert to km
plot_ds['cum_distance'] = plot_ds['distance'].cumsum() / 1000
# bacardi measurements
ax.plot(plot_ds.cum_distance, plot_ds[var],
        label='Above cloud',
        color=cbc[2])
# below cloud
plot_ds = bacardi_ds[key].sel(time=slices[key]['below'])
plot_ds['distance'] = bahamas_ds[key]['distance'].sel(time=slices[key]['below'])
# set first distance to 0
plot_ds['distance'][0] = 0
# sum up distances to generate a distance axis and convert to km
plot_ds['cum_distance'] = plot_ds['distance'].cumsum() / 1000
# bacardi measurements
ax.plot(plot_ds['cum_distance'], plot_ds[var],
        label='Below cloud', color=cbc[3])
# aesthetics
ax.grid()
ax.text(box_xy[0], box_xy[1], '$F_{\\downarrow}$', ha='right',
        transform=ax.transAxes, bbox=dict(boxstyle='Round', fc='white'))
ax.yaxis.set_major_locator(yticks)
ax.set(
       ylim=ylim_irradiance[1],
       xlim=xlims[0],
       xlabel='Distance (km)',
       )

# set a-d labels
for ax, label in zip(axs.flatten(), ['(a)', '(b)', '(c)', '(d)']):
    ax.text(label_xy[0], label_xy[1], label, transform=ax.transAxes)

figname = f'{plot_path}/HALO-AC3_RF17_RF18_BACARDI_terrestrial_irradiance_up_down_comparison_4panel.png'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot PDF of BACARDI transmissivity (above cloud simulation) below cloud
od = list()
plt.rc('font', size=12)
ylims = (0, 33)
binsize = 0.01
xlabel, ylabel = 'Solar transmissivity', 'Probability density function'
_, axs = plt.subplots(1, 2, figsize=(17 * h.cm, 9 * h.cm), layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    bacardi_sel = bacardi_ds[key].sel(time=slices[key]['below'])
    bacardi_plot = bacardi_sel[f'transmissivity_above_cloud']
    print(f'Mean solar transmissivity: {np.mean(bacardi_plot):.3f}')
    bins = np.arange(0.5, 1.0, binsize)
    # BACARDI histogram
    ax.hist(bacardi_plot, density=True, bins=bins, label='Measurement')
    ax.axvline(bacardi_plot.mean(), ls='--', lw=3, label='Mean', c='k')
    ax.set(title=f'{key.replace('RF1', 'Research flight 1')}',
           xlabel=xlabel,
           ylim=ylims,
           xticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1])
    ax.grid()
    od.append((key, 'BACARDI', bacardi_plot.to_numpy(), -np.log(bacardi_plot.to_numpy())))

axs[0].legend()
axs[0].set(ylabel=ylabel)
figname = f'{plot_path}/HALO-AC3_HALO_RF17_RF18_bacardi_transmissivity_above_cloud_PDF_below_cloud.png'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot PDF of transmissivity (above cloud simulation) below cloud - all ice optics
plt.rc('font', size=10)
xlims, ylims = (0.45, 1), (0, 33)
binsize = 0.01
xlabel, ylabel = 'Solar transmissivity', 'Probability density function'
for i, v in enumerate(['v15.1', 'v19.1', 'v18.1']):
    _, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 9 * h.cm), layout='constrained')
    bins = np.arange(0.5, 1.0, binsize)
    v_name = ecrad.version_names[v[:3]]
    v_name = f'{v_name}2013' if v_name == 'Yi' else v_name
    for ii, key in enumerate(keys):
        ax = axs[ii]
        bacardi_sel = bacardi_ds[key].sel(time=slices[key]['below'])
        bacardi_plot = bacardi_sel[f'transmissivity_above_cloud']
        ecrad_ds = ecrad_orgs[key][v].sel(time=slices[key]['below'])
        height_sel = ecrad_ds['aircraft_level']
        ecrad_plot = (ecrad_ds[f'transmissivity_sw_above_cloud']
                      .isel(half_level=height_sel)
                      .to_numpy()
                      .flatten())
        # calculate optical depth
        od.append((key, v_name, ecrad_plot, -np.log(ecrad_plot)))

        # actual plotting
        sns.histplot(bacardi_plot, label='Measurement', ax=ax, stat='density', kde=False, bins=bins)
        sns.histplot(ecrad_plot, label=f'Model', stat='density',
                     kde=False, bins=bins, ax=ax, color=cbc[i + 1])
        ax.set(xlabel=xlabel,
               xlim=xlims,
               xticks=np.arange(0.5, 1.01, 0.1),
               ylabel='',
               ylim=ylims,
               title=f'{key.replace('RF1', 'Research flight 1')}'
               )
        ax.grid()

    axs[1].legend()
    axs[0].set(ylabel=ylabel)

    figname = (f'{plot_path}/HALO-AC3_HALO_RF17_RF18_bacardi_ecrad-'
               f'{v_name}_transmissivity_above_cloud_PDF_below_cloud.svg')
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot PDF of transmissivity (above cloud simulation) below cloud - all ice optics varcloud
plt.rc('font', size=10)
xlims, ylims = (0.45, 1), (0, 33)
binsize = 0.01
xlabel, ylabel = 'Solar transmissivity', 'Probability density function'
for i, v in enumerate(['v16', 'v28', 'v20']):
    _, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 9 * h.cm), layout='constrained')
    bins = np.arange(0.5, 1.0, binsize)
    v_name = ecrad.version_names[v[:3]].replace(' VarCloud', '')
    v_name = f'{v_name}2013' if v_name == 'Yi' else v_name
    for ii, key in enumerate(keys):
        ax = axs[ii]
        bacardi_sel = bacardi_ds[key].sel(time=slices[key]['below'])
        bacardi_plot = bacardi_sel[f'transmissivity_above_cloud']
        ecrad_ds = ecrad_orgs[key][v].sel(time=slices[key]['below'])
        height_sel = ecrad_ds['aircraft_level']
        ecrad_plot = (ecrad_ds[f'transmissivity_sw_above_cloud']
                      .isel(half_level=height_sel)
                      .to_numpy()
                      .flatten())

        # calculate optical depth
        od.append((key, ecrad.version_names[v[:3]], ecrad_plot, -np.log(ecrad_plot)))

        # actual plotting
        sns.histplot(bacardi_plot, label='Measurement', ax=ax, stat='density', kde=False, bins=bins)
        sns.histplot(ecrad_plot, label=f'VarCloud {v_name}', stat='density',
                     kde=False, bins=bins, ax=ax, color=cbc[i + 1])
        ax.set(xlabel=xlabel,
               xlim=xlims,
               xticks=np.arange(0.5, 1.01, 0.1),
               ylabel='',
               ylim=ylims,
               title=f'{key.replace('RF1', 'Research flight 1')}'
               )
        ax.grid()

    axs[1].legend()
    axs[0].set(ylabel=ylabel)

    figname = (f'{plot_path}/HALO-AC3_HALO_RF17_RF18_bacardi_ecrad-'
               f'{ecrad.version_names[v[:3]]}_transmissivity_above_cloud_PDF_below_cloud.svg')
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot optical depth along track above cloud from IFS and VarCloud using IWC and reff
od_list = list()
for key in keys:
    plt.rc("font", size=12)
    _, ax = plt.subplots(figsize=h.figsize_wide, layout="constrained")
    v = "v15.1"
    v_name = ecrad.get_version_name(v[:3])
    ds = ecrad_orgs[key][v].sel(time=slices[key]["above"])
    col_flag = "column" in ds.dims
    b_ext = met.calculate_extinction_coefficient_solar(ds.iwc, ds.re_ice)
    # replace vertical coordinate by pressure height to be able to integrate over altitude
    new_z_coord = ds.press_height_full.mean(dim="time")
    new_z_coord = new_z_coord.mean(dim="column").to_numpy() if col_flag else new_z_coord.to_numpy()
    b_ext = b_ext.assign_coords(level=new_z_coord)
    b_ext = b_ext.sortby("level")  # sort vertical coordinate in ascending order
    # replace nan with 0 for integration
    b_ext = b_ext.where(~np.isnan(b_ext), 0)
    od = b_ext.integrate("level").mean(dim="column")
    ax.plot(od.time, od, label=v_name)
    od_list.append((key, v, "Mean", np.mean(od.to_numpy())))
    od_list.append((key, v, "Median", np.median(od.to_numpy())))
    od_list.append((key, v, "Std", np.std(od.to_numpy())))

    # plot VarCloud data in original resolution
    ds = varcloud_ds[key].sel(time=slices[key]["above"])
    b_ext = met.calculate_extinction_coefficient_solar(ds.iwc, ds.re_ice)
    b_ext = b_ext.sortby("height")  # sort vertical coordinate in ascending order
    # replace nan with 0 for integration
    b_ext = b_ext.where(~np.isnan(b_ext), 0)
    od = b_ext.integrate("height")
    ax.plot(od.time, od, label="VarCloud")
    od_list.append((key, "VarCloud", "Mean", np.mean(od.to_numpy())))
    od_list.append((key, "VarCloud", "Median", np.median(od.to_numpy())))
    od_list.append((key, "VarCloud", "Std", np.std(od.to_numpy())))

    h.set_xticks_and_xlabels(ax, pd.Timedelta(30, "Min"))
    ax.set(title=f"{key} - Optical depth from IWC and ice effective radius above cloud section",
           xlabel="Time (UTC)",
           ylabel="Optical depth")
    ax.grid()
    ax.legend()
    figname = f"{plot_path}/{key}_optical_depth_IFS_vs_VarCloud.png"
    plt.savefig(figname, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# %% print optical depth statistics
od_df = pd.DataFrame(od_list, columns=["key", "source", "stat", "optical_depth"])
print(od_df)

