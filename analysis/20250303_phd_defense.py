#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 18.02.2025

Plots and facts for my PhD defense
"""
# %% import modules

import os

import cartopy.crs as ccrs
import cmasher as cmr
import dill
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib import ticker, colors
from metpy.calc import relative_humidity_from_specific_humidity
from metpy.units import units as u
from skimage import io
from tqdm import tqdm

import pylim.halo_ac3 as meta
import pylim.helpers as h
import pylim.meteorological_formulas as met
from pylim import ecrad
from pylim.bahamas import preprocess_bahamas

cbc = h.get_cb_friendly_colors('petroff_6')

# %% set paths
campaign = 'halo-ac3'
revision = 'R2'  # BACARDI revision
keys = ['RF17', 'RF18']
ecrad_versions = [f'v{x}' for x in [13, 13.1, 13.2, 15, 15.1, 16.1, 18.1, 19.1,
                                    22.1, 24.1, 26, 27, 30.1, 31.1, 32.1,
                                    36, 37, 38, 39.2, 40.2, 41.2, 42.2]]

save_path = f'C:/Users/Johannes/Documents/Doktor/defense/data{revision}'
plot_path = f'C:/Users/Johannes/Documents/Doktor/defense/figureR2'
bacardi_all_path = h.get_path('all', campaign=campaign, instrument='BACARDI')
bahamas_all_path = h.get_path('all', campaign=campaign, instrument='BAHAMAS')
trajectory_path = f"{h.get_path('trajectories', campaign=campaign)}/selection_CC_and_altitude"

# %% define variables for multiple use
date_title = ['11 April 2022', '12 April 2022']
panel_label = ['(a)', '(b)']
xlim = (0.35, 1)  # xlim for violin plots

# %% read in data
(
    bahamas_ds, bacardi_ds, bacardi_ds_res, varcloud_ds, above_clouds,
    below_clouds, slices, ifs_ds, ifs_ds_sel, dropsonde_ds, lidar_ds, radar_ds,
    sat_imgs, ecrad_dicts, ecrad_orgs
) = (
dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict())

left, right, bottom, top = 0, 1000000, -1000000, 0
sat_img_extent = (left, right, bottom, top)
# read in dropsonde data
dropsonde_path = f"{h.get_path('all', campaign=campaign, instrument='dropsondes')}/Level_3"
dropsonde_file = 'merged_HALO_P5_beta_v2.nc'
dds = xr.open_dataset(f'{dropsonde_path}/{dropsonde_file}')

for key in keys:
    flight = meta.flight_names[key]
    date = flight[9:17]
    bacardi_path = h.get_path('bacardi', flight, campaign)
    bacardi_file = f'HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_{revision}_JR_v2.nc'
    bacardi = xr.open_dataset(f'{bacardi_path}/{bacardi_file}')
    bacardi_ds[key] = bacardi
    # read in resampled BACARDI data
    bacardi_res = xr.open_dataset(f'{bacardi_path}/{bacardi_file.replace('_v2.nc', '_1Min_v2.nc')}')
    bacardi_ds_res[key] = bacardi_res

for key in keys:
    flight = meta.flight_names[key]
    date = flight[9:17]
    urldate = pd.to_datetime(date).strftime('%Y-%m-%d')
    bahamas_path = h.get_path('bahamas', flight, campaign)
    ifs_path = f'{h.get_path('ifs', flight, campaign)}/{date}'
    ecrad_path = f'{h.get_path('ecrad', flight, campaign)}/{date}'
    varcloud_path = h.get_path('varcloud', flight, campaign)
    radar_path = h.get_path('hamp_mira', flight, campaign)
    lidar_path = h.get_path('wales', flight, campaign)

    # filenames
    bahamas_file = f'HALO-AC3_HALO_BAHAMAS_{date}_{key}_v1_JR.nc'
    ifs_file = f'ifs_{date}_00_ml_O1280_processed.nc'
    ifs_sel_file = f'ifs_{date}_00_ml_O1280_processed_sel_JR.nc'
    varcloud_file = [f for f in os.listdir(varcloud_path) if f.endswith('_JR.nc')][0]
    radar_file = f'HALO_HALO_AC3_radar_unified_{key}_{date}_v2.7.nc'
    lidar_file = f'HALO-AC3_HALO_WALES_bsrgl_{date}_{key}_V2.0.nc'
    satfile = f'{save_path}/{key}_MODIS_Terra_CorrectedReflectance_Bands367.png'
    sat_url = f'https://gibs.earthdata.nasa.gov/wms/epsg3413/best/wms.cgi?\
    version=1.3.0&service=WMS&request=GetMap&\
    format=image/png&STYLE=default&bbox={left},{bottom},{right},{top}&CRS=EPSG:3413&\
    HEIGHT=8192&WIDTH=8192&TIME={urldate}&layers=MODIS_Terra_CorrectedReflectance_Bands367'

    # read in aircraft data
    bahamas_ds[key] = xr.open_dataset(f'{bahamas_path}/{bahamas_file}')

    # read in satellite image
    try:
        sat_imgs[key] = io.imread(satfile)
    except FileNotFoundError:
        sat_imgs[key] = io.imread(sat_url)

    # split up dropsonde data into RF17 and RF18
    dropsonde_ds[key] = dds.where(dds.launch_time.dt.date == pd.to_datetime(date).date(), drop=True)

    # read in radar & lidar data
    radar = xr.open_dataset(f'{radar_path}/{radar_file}')
    lidar = xr.open_dataset(f'{lidar_path}/{lidar_file}')
    varcloud_ds[key] = xr.open_dataset(f'{varcloud_path}/{varcloud_file}')

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

    # read in ifs data
    ifs = xr.open_dataset(f'{ifs_path}/{ifs_file}').set_index(rgrid=['lat', 'lon'])
    # filter low clouds according to ECMWF low cloud criterion (pressure higher than 0.8 * surface pressure)
    cloud_data = ifs[['q_liquid', 'q_ice', 'cloud_fraction', 'clwc', 'ciwc', 'crwc', 'cswc']]
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

    # read in ecrad data
    ecrad_dict, ecrad_org = dict(), dict()

    for k in ecrad_versions:
        ds = xr.open_dataset(f'{ecrad_path}/ecrad_merged_inout_{date}_{k}.nc')
        # add net terrestrial flux if necessary
        if 'flux_net_lw' not in ds:
            ds['flux_net_lw'] = ds['flux_dn_lw'] - ds['flux_up_lw']
        # select only center column for direct comparisons
        ecrad_org[k] = ds.copy(deep=True)
        ds = ds.sel(column=0, drop=True) if 'column' in ds.dims else ds
        ecrad_dict[k] = ds.copy(deep=True)

    ecrad_dicts[key] = ecrad_dict
    ecrad_orgs[key] = ecrad_org

# read in stats
stats = pd.read_csv(f'{save_path}/halo-ac3_bacardi_{revision}_ecrad_statistics.csv')

# %% read in all BAHAMAS and BACARDI data
all_files = os.listdir(bacardi_all_path)
all_files.sort()
all_files = [os.path.join(bacardi_all_path, file) for file in all_files[2:] if file.endswith(f'{revision}_JR.nc')]
bacardi_ds_all = xr.open_mfdataset(all_files)
# bahamas
all_files = [f for f in os.listdir(bahamas_all_path) if f.startswith('HALO')]
all_files.sort()
all_files = [os.path.join(bahamas_all_path, file) for file in all_files[1:]]
bahamas_ds_all = xr.open_mfdataset(all_files, preprocess=preprocess_bahamas)

# %% read in effective diameter from Dela Torre Castro 2023
ed = pd.read_csv(f'{save_path}/dataCIRRUS_HLpaper2sec.csv')
ed['effective_radius'] = ed['ED'] / 2
lat_bins = np.arange(np.min(ed['latitude'].round(0)),
                     np.max(ed['latitude'].round(0)) + 1,
                     1)
ed['latitude_bin'] = pd.cut(ed['latitude'], bins=lat_bins)
stats_ed = ed.groupby('latitude_bin')['effective_radius'].median().reset_index()
stats_ed['mid_latitude'] = stats_ed['latitude_bin'].cat.categories.mid
stats_ed['effective_radius'] = stats_ed['effective_radius'].where(stats_ed['effective_radius'] > 5)

# %% calculate relation and deviation between simulated and measured solar downward irradiance
bacardi_ds_all['relation'] = bacardi_ds_all['F_down_solar'] / bacardi_ds_all['F_down_solar_sim']
bacardi_ds_all['deviation'] = (bacardi_ds_all['relation'] - 1) * 100

# %% viewing direction of halo: 0 = towards sun, 180 = away from sun
heading = bahamas_ds_all.IRS_HDG
viewing_dir = bacardi_ds_all.saa - heading
bacardi_ds_all['viewing_dir'] = viewing_dir.where(viewing_dir > 0, viewing_dir + 360)

# %% solar transmissivity - prepare data for box/violin plot
ecrad_var = 'transmissivity_sw_above_cloud'
label = 'transmissivity_sw'
bacardi_var = 'transmissivity_above_cloud'
filepath = f'{save_path}/halo-ac3_{label}_boxplot_data.csv'
df = pd.DataFrame()
for key in keys:
    dfs = list()
    dfs.append(df)
    for v in ecrad_versions:
        dfs.append(pd.DataFrame({'values': (ecrad_orgs[key][v][ecrad_var]
                                            .isel(half_level=ecrad_dicts[key][v].aircraft_level)
                                            .sel(time=slices[key]['below'])
                                            .to_numpy()
                                            .flatten()),
                                 'label': v,
                                 'key': key}))

    dfs.append(pd.DataFrame({'values': (bacardi_ds_res[key][bacardi_var]
                                        .sel(time=slices[key]['below'])
                                        .dropna('time')
                                        .to_pandas()
                                        .reset_index(drop=True)),
                             'label': 'BACARDI',
                             'key': key}))

    dfs.append(pd.DataFrame({'values': (bacardi_ds[key][bacardi_var]
                                        .sel(time=slices[key]['below'])
                                        .dropna('time')
                                        .to_pandas()
                                        .reset_index(drop=True)),
                             'label': 'BACARDI_org',
                             'key': key}))

    df = pd.concat(dfs)

df = df.reset_index(drop=True)
df.to_csv(filepath, index=False)

# %% solar transmissivity - get statistics
sw_stats = (df
            .groupby(['key', 'label'])['values']
            .describe()
            .sort_values(['key', 'mean'], ascending=[True, False]))
sw_stats = sw_stats.reset_index()  # reset index for nice csv output
name = list()
aerosol = list()
for v in sw_stats['label']:
    try:
        n = ecrad.get_version_name(v[:3])
        name.append(n)
        a = 'On' if v[:3] in ecrad.aerosol_on else 'Off'
        aerosol.append(a)
    except ValueError:
        name.append(v)
        aerosol.append('Off')
sw_stats = sw_stats.assign(name=name, aerosol=aerosol)
# add deviation column
bacardi_means = sw_stats[sw_stats['label'] == 'BACARDI'].set_index('key')['mean']
sw_stats['deviation'] = sw_stats.apply(
    lambda row: (row['mean'] - bacardi_means[row['key']]) / bacardi_means[row['key']] * 100, axis=1)
sw_stats.to_csv(f'{save_path}/HALO-AC3_transmissivity_sw_stats.csv',
                index=False)
sw_stats = sw_stats.set_index(['key', 'label'])

# %% terrestrial downward flux - prepare data for violin plots
ecrad_var = ['flux_dn_lw', 'flux_up_lw']
bacardi_var = ['F_down_terrestrial', 'F_up_terrestrial']
sections = ['case', 'above', 'below']
filepath = f'{save_path}/halo-ac3_flux_lw_boxplot_data.csv'
dfs = list()
for key in keys:
    for sec in sections:
        for var in ecrad_var:
            for v in ecrad_versions:
                try:
                    dfs.append(
                        pd.DataFrame({'values': (ecrad_orgs[key][v][var]
                                                 .isel(half_level=ecrad_dicts[key][v].aircraft_level)
                                                 .sel(time=slices[key][sec])
                                                 .to_numpy()
                                                 .flatten()),
                                      'label': v,
                                      'var': var,
                                      'section': sec,
                                      'key': key}))
                except ValueError:
                    continue
        for var in bacardi_var:
            dfs.append(pd.DataFrame({'values': (bacardi_ds_res[key][var]
                                                .sel(time=slices[key][sec])
                                                .dropna('time')
                                                .to_pandas()
                                                .reset_index(drop=True)),
                                     'label': 'BACARDI',
                                     'var': var,
                                     'section': sec,
                                     'key': key}))

            dfs.append(pd.DataFrame({'values': (bacardi_ds[key][var]
                                                .sel(time=slices[key][sec])
                                                .dropna('time')
                                                .to_pandas()
                                                .reset_index(drop=True)),
                                     'label': 'BACARDI_org',
                                     'var': var,
                                     'section': sec,
                                     'key': key}))

df_lw = pd.concat(dfs)

df_lw['var'] = df_lw['var'].replace(dict(F_down_terrestrial='flux_dn_lw', F_up_terrestrial='flux_up_lw'))
df_lw.to_csv(filepath, index=False)
df_lw = df_lw.reset_index(drop=True)

# %% terrestrial - get statistics
lw_stats = (df_lw
            .groupby(['key', 'label', 'var', 'section'])['values']
            .describe()
            .sort_values(['key', 'mean'], ascending=[True, False]))
lw_stats = lw_stats.reset_index()
lw_stats['var'] = lw_stats['var'].replace({'F_up_terrestrial': 'flux_up_lw', 'F_down_terrestrial': 'flux_dn_lw'})
name = list()
aerosol = list()
for v in lw_stats['label']:
    try:
        n = ecrad.get_version_name(v[:3])
        name.append(n)
        a = 'On' if v[:3] in ecrad.aerosol_on else 'Off'
        aerosol.append(a)
    except ValueError:
        name.append(v)
        aerosol.append('Off')
lw_stats = lw_stats.assign(name=name, aerosol=aerosol)
# add deviation column
bacardi_means = lw_stats[lw_stats['label'] == 'BACARDI'].set_index(['key', 'var', 'section'])['mean']
lw_stats['deviation'] = lw_stats.apply(
    lambda row: (row['mean'] - bacardi_means.get((row['key'], row['var'], row['section']))) / bacardi_means.get(
        (row['key'], row['var'], row['section'])) * 100, axis=1)
lw_stats.to_csv(f'{save_path}/HALO-AC3_terrestrial_irradiance_stats.csv',
                index=False)
lw_stats = lw_stats.set_index(['key', 'var', 'label'])

# %% plot flight track together with high cloud cover
plt.rc("font", size=10)
data_crs = ccrs.PlateCarree()
map_crs = ccrs.NorthPolarStereo()
for key in keys:
    ts = "2022-04-11 12:00" if key == "RF17" else "2022-04-12 12:00"
    ifs = ifs_ds[key].sel(time=ts, method="nearest")
    _, ax = plt.subplots(figsize=(12 * h.cm, 9 * h.cm), layout="constrained",
                         subplot_kw=dict(projection=map_crs))

    ax.coastlines(alpha=0.5)
    ax.set_extent([-20, 25, 65, 90])
    gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=1, color='gray', alpha=0.5,
                      linestyle=':', x_inline=False, y_inline=False, rotate_labels=False)
    gl.xlocator = ticker.FixedLocator(np.arange(-180, 180, 20))
    gl.ylocator = ticker.FixedLocator(np.arange(60, 90, 5))
    gl.top_labels = False
    gl.right_labels = False

    # add seaice edge
    ci_levels = [0.8]
    cci = ax.tricontour(ifs.lon, ifs.lat, ifs['ci'], ci_levels, transform=data_crs, linestyles="--", colors="#332288",
                        linewidths=2)

    # add total column ice water
    # ifs_cc = ifs.cloud_fraction.where(ifs.pressure_full < 60000, drop=True).sum(dim="level")
    # ifs_cc = ifs_cc / 101  # divide by number of high cloud levels
    cmap = colors.ListedColormap(['white', 'lightblue'])
    hcc = ax.tricontourf(ifs.lon, ifs.lat, ifs.hcc,
                         levels=3,
                         extend='both',
                         transform=data_crs,
                         cmap=cmap,
                         )
    cbar = plt.colorbar(hcc, label="IFS high cloud cover")
    cbar.set_ticks([])

    # plot windfield
    ifs_500 = ifs.isel(level=96, rgrid=slice(0, len(ifs.rgrid), 100))
    ax.quiver(ifs_500.lon, ifs_500.lat, ifs_500.u, ifs_500.v, transform=ccrs.PlateCarree(), scale=300)

    # plot flight track
    ins = bahamas_ds[key]
    track_lons, track_lats = ins["IRS_LON"], ins["IRS_LAT"]
    ax.plot(track_lons[::10], track_lats[::10], color="k",
            label='Flight track', transform=data_crs)

    # plot dropsonde locations
    ds_ds = dropsonde_ds[key]
    launch_time = pd.to_datetime(ds_ds.launch_time.to_numpy())
    x, y = ds_ds.lon.mean(dim='alt').to_numpy(), ds_ds.lat.mean(dim='alt').to_numpy()
    cross = ax.plot(x, y, "x", color=cbc[1], markersize=9, transform=data_crs)
    # ax.text(x, y, f"{launch_time.strftime('%H:%M')}", c="k", fontsize=9, transform=data_crs,
    #         path_effects=[patheffects.withStroke(linewidth=0.5, foreground="white")])

    coords = meta.coordinates["Kiruna"]
    ax.plot(coords[0], coords[1], ls="", marker="^", color=cbc[2], label="Kiruna", transform=data_crs)
    ax.plot([], ls="--", color="#332288", label="Sea ice edge")
    ax.plot([], ls="", marker="x", color=cbc[1], label="Dropsonde")
    ax.legend(loc=3)

    figname = f"{plot_path}/HALO-AC3_{key}_flight_track_IFS_cloud_cover.png"
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# %% print statistics of deviation
print_ds = (bacardi_ds_all
            .where(bacardi_ds_all.alt >= 10000)
            )
bins = np.arange(60, 95, 5)
grouped_ds = print_ds.groupby_bins('sza', bins=bins, labels=bins[:-1])
grouped_mean = grouped_ds.mean()
print(grouped_mean.deviation.to_pandas())
print(grouped_ds.max().deviation.to_pandas())
print(grouped_ds.min().deviation.to_pandas())

# %% print statistics of deviation for case studies
print_ds = (bacardi_ds_all
            .where(bacardi_ds_all.alt >= 10000)
            .sel(time=slice('2022-04-11', '2022-04-13'))
            )
bins = np.arange(60, 95, 5)
grouped_ds = print_ds.groupby_bins('sza', bins=bins, labels=bins[:-1])
grouped_mean = grouped_ds.mean()
print(grouped_mean.deviation.to_pandas())
print(grouped_ds.max().deviation.to_pandas())
print(grouped_ds.min().deviation.to_pandas())

# %% plot lidar data for case studies
for key in keys:
    plot_ds = (lidar_ds[key]["backscatter_ratio"]
               .where((lidar_ds[key].flags == 0)
                      & (lidar_ds[key].backscatter_ratio > 1))
               .sel(time=slices[key]["above"]))
    plt.rc("font", size=8.5)
    _, ax = plt.subplots(figsize=(12.5 * h.cm, 4 * h.cm), layout="constrained")
    plot_ds.plot(x="time", y="height", cmap=cmr.rainforest_r, norm=mpl.colors.LogNorm(), vmax=50,
                 cbar_kwargs=dict(label="Lidar\nbackscatter ratio", pad=0.01))

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

    # plot tropopause heigth
    ax.plot(ifs_plot.time, tp_height, color="k", linestyle="--", label="Tropopause")

    h.set_xticks_and_xlabels(ax, slices[key]["above"].stop - slices[key]["above"].start)
    ax.set(xlabel="", ylabel="Altitude (km)", ylim=(0, 12))
    ax.legend(loc=1)

    figname = f"{plot_path}/HALO_AC3_HALO_{key}_lidar_backscatter_ratio_532.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot varcloud microphysical data for case studies
for key in keys:
    ds_plot = varcloud_ds[key]['iwc'].sel(time=slices[key]['above']) * 1e6
    ds_plot['height'] = ds_plot['height'] / 1000
    _, ax = plt.subplots(figsize=(12.5 * h.cm, 4 * h.cm), layout='constrained')
    ds_plot.plot(x='time', cmap=cmr.get_sub_cmap(cmr.freeze, 0.3, 1), vmax=50,
                 cbar_kwargs=dict(label="Retrieved ice water\ncontent (mg$\\,$m$^{-3}$)", pad=0.01),
                 ax=ax)
    ax.set(
        xlabel='Time (UTC)',
        ylabel='Altitude (km)',
        ylim=(0, 12),
    )
    h.set_xticks_and_xlabels(ax, slices[key]["above"].stop - slices[key]["above"].start)
    figname = f'{plot_path}/HALO_AC3_HALO_{key}_iwc_varcloud.png'
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot time series of optical depth
for key in keys:
    ds_plot = varcloud_ds[key].sel(time=slices[key]['above'])
    bext = met.calculate_extinction_coefficient_solar(ds_plot['iwc'], ds_plot['re_ice'])
    bext = bext.sortby('height')
    # replace nan with 0 for integration
    bext = bext.where(~np.isnan(bext), 0)
    od = bext.integrate("height")
    od = od.where(od > 0.006)
    print(f'{key}: {np.mean(od)}')
    _, ax = plt.subplots(figsize=(11 * h.cm, 2.5 * h.cm), layout='constrained')
    od.plot(x='time')
    ax.set(
        xlabel='',
        ylabel='Optical\n depth',
        ylim=(0, 2.5),
        yticks=(0, 1, 2)
    )
    ax.grid()
    h.set_xticks_and_xlabels(ax, slices[key]["above"].stop - slices[key]["above"].start)
    figname = f'{plot_path}/HALO_AC3_HALO_{key}_od_varcloud.png'
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot scatter plot of above cloud measurements and simulations
plt.rc('font', size=10)
for v in ['v15.1']:
    _, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 7.5 * h.cm),
                          layout='constrained')
    for i, key in enumerate(keys):
        ax = axs[i]
        above_sel = (bahamas_ds[key].IRS_ALT > 11000).resample(time='1Min').first()
        bacardi_res = bacardi_ds_res[key]
        bacardi_plot = bacardi_res.where(bacardi_res.alt > 11000)
        ecrad_ds = ecrad_dicts[key][v]
        height_sel = ecrad_dicts[key][v].aircraft_level
        ecrad_plot = ecrad_ds.flux_dn_sw.isel(half_level=height_sel).where(above_sel)

        # actual plotting
        rmse = np.sqrt(np.mean((bacardi_plot['F_down_solar_diff'] - ecrad_plot) ** 2)).to_numpy()
        bias = np.nanmean((bacardi_plot['F_down_solar_diff'] - ecrad_plot).to_numpy())
        ax.scatter(bacardi_plot.F_down_solar_diff, ecrad_plot, color=cbc[3])
        ax.axline((0, 0), slope=1, color='k', lw=2, transform=ax.transAxes)
        ax.set(
            aspect='equal',
            xlabel='Measured irradiance (W$\\,$m$^{-2}$)',
            ylabel='Simulated irradiance (W$\\,$m$^{-2}$)',
            xlim=(175, 525),
            ylim=(175, 525),
        )
        ax.grid()
        ax.text(
            0.025,
            0.95,
            f'{panel_label[i]}\n'
            f'n= {sum(~np.isnan(bacardi_plot['F_down_solar'])):.0f}\n'
            f'RMSE: {rmse:.0f} {h.plot_units['flux_dn_sw']}\n'
            f'Bias: {bias:.0f} {h.plot_units['flux_dn_sw']}',
            ha='left',
            va='top',
            transform=ax.transAxes,
            bbox=dict(fc='white', ec='black', alpha=0.8, boxstyle='round'),
        )
        ax.set_title(f'{key.replace('1', ' 1')} - {date_title[i]}', fontsize=10)

    figname = f'{plot_path}/05_HALO-AC3_HALO_RF17_RF18_bacardi_{revision}_ecrad_f_down_solar_above_cloud_all_{v}.png'
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% solar transmissivity - plot violinplot BACARDI vs ecRad
sel_ver = ['BACARDI', 'v15.1']
_, axs = plt.subplots(2, 1, figsize=(15 * h.cm, 9 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df[(df.key == key)
                 & (df.label.isin(sel_ver))]
    df_plot['label'] = (df_plot['label']
                        .astype('category')
                        .cat.reorder_categories(sel_ver))
    sns.violinplot(df_plot, x='values', y='label', hue='label', ax=ax)
    ax.set(
        xlabel='',
        ylabel='',
        yticklabels=['BACARDI',
                     'ecRad Reference\nsimulation (v15.1)'],
        xlim=xlim,
    )
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.89, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[1].set(xlabel='Solar transmissivity')
figname = f'05_HALO_AC3_RF17_RF18_transmissivity_sw_BACARDI_ecRad_boxplot_v15.1.png'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% BACARDI vs ecRad - print stats
df_print = sw_stats.loc[pd.IndexSlice[:, ['BACARDI_org'] + sel_ver], :].sort_values('key')
print(df_print)

# %% plot comparison of above cloud irradiance between libRadtran and ecRad
_, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 9 * h.cm), layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    date = '2022-04-11' if key == 'RF17' else '2022-04-12'
    plot_df = (bacardi_ds[key][['ecrad_fdw', 'F_down_solar_sim', 'alt', 'lat']]
               .sel(time=slice(bacardi_ds[key].time[0] + pd.Timedelta(1, 'h'),
                               bacardi_ds[key].time[-1] - pd.Timedelta(1, 'h')))
               .to_pandas()
               .dropna())
    rmse = np.mean(np.sqrt((plot_df['F_down_solar_sim'] - plot_df['ecrad_fdw']) ** 2))
    bias = np.mean(plot_df['F_down_solar_sim'] - plot_df['ecrad_fdw'])
    ax.scatter(plot_df['F_down_solar_sim'], plot_df['ecrad_fdw'], color=cbc[3])
    ax.axline((0, 0), slope=1, color='k', lw=2, transform=ax.transAxes)
    ax.grid()
    ax.set(
        aspect='equal',
        xlabel='libRadtran $F^{\\downarrow}_{\\text{sol}}$' + f' ({h.plot_units['flux_dn_sw']})',
        ylabel='',
        xlim=(150, 500),
        ylim=(150, 500)
    )
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.text(
        0.05,
        0.95,
        f'{panel_label[i]} {key.replace('F', 'F ')}\n'
        f'n = {sum(~np.isnan(plot_df['F_down_solar_sim'])):,.0f}\n'
        f'RMSE: {rmse:.0f} {h.plot_units['flux_dn_sw']}\n'
        f'Bias: {np.abs(bias):.0f} {h.plot_units['flux_dn_sw']}',
        ha='left',
        va='top',
        transform=ax.transAxes,
        bbox=dict(fc='white', ec='black', alpha=0.8, boxstyle='round'),
    )

axs[0].set(
    ylabel='ecRad $F^{\\downarrow}_{\\text{sol}}$' f' ({h.plot_units['flux_dn_sw']})',
)
figname = f'04_libRadtran_vs_ecRad_cloud-free.png'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()
# %% plot f_dn_lw BACARDI vs. ecRad - violinplot
sel_ver = ['BACARDI', 'v15.1']
var = ['flux_dn_lw', 'F_down_terrestrial']
section = 'below'
h.set_cb_friendly_colors('petroff_6')
plt.rc('font', size=10)
_, axs = plt.subplots(2, 1, figsize=(15 * h.cm, 9 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    sel = ((df_lw['key'] == key)
           & (df_lw['label'].isin(sel_ver))
           & (df_lw['section'] == section)
           & (df_lw['var'].isin(var)))
    df_plot = df_lw[sel]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='values', y='label', hue='label',
                   ax=ax, order=sel_ver)
    ax.set(xlabel='',
           ylabel='',
           xlim=(75, 200),
           yticklabels=['BACARDI',
                        'ecRad Reference\nFu-IFS (v15.1)',
                        ])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.89, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[1].set(
    xlabel=f'Downward terrestrial irradiance below cloud ({h.plot_units["flux_net_lw"]})'
)
figname = f'05_HALO_AC3_RF17_RF18_flux_dn_lw_BACARDI_{revision}_ecRad_violin.png'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% BACARDI vs ecRad - print stats
df_print = lw_stats.loc[pd.IndexSlice[:, 'flux_dn_lw', ['BACARDI_org'] + sel_ver], :].sort_values('key')
print(df_print[df_print.section == 'below'])

# %% plot temperature and humidity profiles from IFS and from dropsonde
below_cloud_altitude = dict()
h.set_cb_friendly_colors('petroff_8')
plt.rc('font', size=10)
_, axs = plt.subplots(1, 4, figsize=(18 * h.cm, 11 * h.cm), layout='constrained')
for i, key in enumerate(keys):
    below_cloud_altitude[key] = bahamas_ds[key].IRS_ALT.sel(time=slices[key]['below']).mean(dim='time') / 1000
    ax = axs[i * 2]
    ifs_plot = ecrad_dicts[key]['v15.1'].sel(time=slices[key]['case'])
    sf = 1000

    # Air temperature
    for t in ifs_plot.time:
        ifs_p = ifs_plot.sel(time=t)
        ax.plot(ifs_p.temperature_hl - 273.15, ifs_p.press_height_hl / 1000, color='grey', lw=0.5)
    ds_plot = dropsonde_ds[key]
    times = ['104205', '110137'] if key == 'RF17' else ['110321', '110823', '111442', '112014', '112524']
    date = '20220411' if key == 'RF17' else '20220412'
    times_dt = pd.to_datetime([date + t for t in times], format='%Y%m%d%H%M%S')
    for k in times_dt:
        ds = ds_plot.where(ds_plot.launch_time == k, drop=True)
        ds = ds.where(~np.isnan(ds['ta']), drop=True)
        ax.plot(ds['ta'][0] - 273.15, ds.alt / sf, label=f'DS {k:%H:%M} UTC', lw=2)
    ax.set(
        xlim=(-60, -10),
        ylim=(0, 12),
        xlabel='Air temperature (°C)',
    )
    ax.set_title(f'{key.replace('1', ' 1')} - {date_title[i]}', fontsize=10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=15))
    ax.plot([], color='grey', label='IFS profiles')
    ax.axhline(below_cloud_altitude[key], c='k')
    ax.grid()

    # RH
    ax = axs[i * 2 + 1]
    ifs_plot = ecrad_dicts[key]['v15.1'].sel(time=slices[key]['case'])
    for t in ifs_plot.time:
        ifs_p = ifs_plot.sel(time=t)
        rh = relative_humidity_from_specific_humidity(ifs_p.pressure_full * u.Pa, ifs_p.t * u.K, ifs_p.q * u('kg/kg'))
        rh_ice = met.relative_humidity_water_to_relative_humidity_ice(rh * 100, ifs_p.t - 273.15)
        ax.plot(rh_ice, ifs_p.press_height_full / 1000, color='grey', lw=0.5)
    ds_plot = dropsonde_ds[key]
    for k in times_dt:
        ds = ds_plot.where(ds_plot.launch_time == k, drop=True)
        ds = ds.where(~np.isnan(ds.rh), drop=True)
        ax.plot(met.relative_humidity_water_to_relative_humidity_ice(ds.rh * 100, ds['ta'] - 273.15)[0],
                ds.alt / sf, label=f'DS {k:%H:%M} UTC', lw=2)
    ax.set(
        xlim=(0, 130),
        ylim=(0, 12),
        xlabel='Relative humidity \nover ice (%)',
    )
    ax.set_title(f'{key.replace('1', ' 1')} - {date_title[i]}', fontsize=10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=25))
    ax.plot([], color='grey', label='IFS profiles')
    ax.axhline(below_cloud_altitude[key], c='k')
    ax.legend(fontsize=7)
    ax.grid()

axs[0].set_ylabel('Altitude (km)')
axs[0].text(0.02, 0.95, '(a)', transform=axs[0].transAxes)
axs[1].text(0.02, 0.95, '(b)', transform=axs[1].transAxes)
axs[2].text(0.02, 0.95, '(c)', transform=axs[2].transAxes)
axs[3].text(0.02, 0.95, '(d)', transform=axs[3].transAxes)

figname = f'{plot_path}/05_HALO-AC3_HALO_RF17_RF18_ifs_dropsonde_t_rh.png'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot PDF of IWC from IFS above cloud for 11 and 12 UTC
plt.rc('font', size=10)
legend_labels = ['11:00$\\,$UTC', '12:00$\\,$UTC']
binsizes = dict(iwc=1, reice=4)
_, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 8 * h.cm), layout='constrained')
ylims = {'iwc': (0, 0.22), 'reice': (0, 0.095)}
# left panel - RF17 IWC
ax = axs[0]
binsize = binsizes['iwc']
bins = np.arange(0, 20.1, binsize)
iwc_ifs_ls = list()
for t in ['2022-04-11 11:00', '2022-04-11 12:00']:
    iwc_ifs, cc = ifs_ds_sel['RF17'].q_ice.sel(time=t), ifs_ds_sel['RF17'].cloud_fraction.sel(time=t)
    iwc_ifs_ls.append(iwc_ifs.where(cc > 0).where(cc == 0, iwc_ifs / cc))

for i, pds in enumerate(iwc_ifs_ls):
    pds = pds.to_numpy().flatten() * 1e6
    pds = pds[~np.isnan(pds)]
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i],
        color=cbc[i],
        histtype='step',
        density=True,
        lw=2,
    )
    print(f'RF 17/n{legend_labels[i]}: n={len(pds)}, mean={np.mean(pds):.2f}, median={np.median(pds):.2f}')
ax.grid()
ax.set(ylabel=f'Probability density function',
       xlabel=f'Ice water content ({h.plot_units['iwc']})',
       ylim=ylims['iwc'],
       xticks=range(0, 21, 5),
       )
ax.set_title('RF 17 - 11 April 2022', fontsize=10)
ax.text(0.05, 0.93,
        f'(a)',
        transform=ax.transAxes,
        )

# right panel - RF18 IWC
ax = axs[1]
iwc_ifs_ls = list()
for t in ['2022-04-12 11:00', '2022-04-12 12:00']:
    iwc_ifs, cc = ifs_ds_sel['RF18'].q_ice.sel(time=t), ifs_ds_sel['RF18'].cloud_fraction.sel(time=t)
    iwc_ifs_ls.append(iwc_ifs.where(cc > 0).where(cc == 0, iwc_ifs / cc))

for i, pds in enumerate(iwc_ifs_ls):
    pds = pds.to_numpy().flatten() * 1e6
    pds = pds[~np.isnan(pds)]
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i],
        color=cbc[i],
        histtype='step',
        density=True,
        lw=2,
    )
    print(f'RF 18/n{legend_labels[i]}: n={len(pds)}, mean={np.mean(pds):.2f}, median={np.median(pds):.2f}')
ax.legend()
ax.grid()
ax.set(ylabel=f'',
       xlabel=f'Ice water content ({h.plot_units['iwc']})',
       ylim=ylims['iwc'],
       xticks=range(0, 21, 5),
       )
ax.text(0.05, 0.93,
        f'(b)',
        transform=ax.transAxes,
        )
ax.set_title('RF 18 - 12 April 2022', fontsize=10)

figname = f'{plot_path}/05_HALO-AC3_HALO_RF17_RF18_IFS_iwc_11_vs_12_pdf_case_studies.png'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot IFS cloud fraction lidar/mask comparison
var = 'cloud_fraction'
key = 'RF17'
plt.rc('font', size=10)
fig, ax = plt.subplots(figsize=(12.5 * h.cm, 5 * h.cm), layout='constrained')

ds = ecrad_dicts[key]['v15'].sel(time=slices[key]['case'])
ifs_plot = ds[[var]]
bahamas_plot = bahamas_ds[key].IRS_ALT.sel(time=slices[key]['case']) / 1000
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
ifs_plot = ifs_plot.where(ifs_plot[var] > 0)  # > 1e-9) * 1e6
halo_plot = varcloud_ds[key].sel(time=slices[key]['case']).Varcloud_Input_Mask
halo_plot = halo_plot.assign_coords(height=halo_plot.height / 1000).sortby('height')
time_extend = pd.to_timedelta((ifs_plot.time[-1] - ifs_plot.time[0]).to_numpy())

# plot IFS cloud cover prediction and Radar lidar mask
pcm = ifs_plot[var].plot(x='time', cmap=cmr.sapphire, ax=ax, add_colorbar=False)
halo_plot.plot.contour(x='time', levels=[0.9], colors=cbc[1], ax=ax, linewidths=2)
ax.plot([], color=cbc[1], label='Radar & Lidar Mask', lw=2)
bahamas_plot.plot(x='time', lw=2, color=cbc[-2], label='HALO flight altitude', ax=ax)
ax.axvline(pd.Timestamp("2022-04-11 11:12:26"), 0, 1, ls="--", lw=3, label="Turning point")
h.set_xticks_and_xlabels(ax, time_extend)
ax.set(xlabel='Time (UTC)', ylabel='Height (km)')
# add line for turning point

# place colorbar for both flights
fig.colorbar(pcm, ax=axs[:2], label=f'IFS {h.cbarlabels[var].lower()} {h.plot_units[var]}', pad=0.001)
ax.legend(fontsize=7, ncols=2)

figname = f'{plot_path}/05_HALO-AC3_HALO_RF17_IFS_{var}_radar_lidar_mask.png'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% ice optics parameterizations - plot violinplot of solar transmissivity
sel_ver = ['BACARDI', 'v15.1', 'v19.1', 'v18.1']
plt.rc('font', size=10)
_, axs = plt.subplots(2, 1, figsize=(15 * h.cm, 10 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df[(df.key == key)
                 & (df.label.isin(sel_ver))]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='values', y='label', hue='label', ax=ax,
                   order=sel_ver)
    ax.set(xlabel='',
           ylabel='',
           yticklabels='',
           xlim=xlim,
           )
    ax.set(yticklabels=['BACARDI',
                        'ecRad Reference\nFu-IFS (v15.1)',
                        'ecRad Yi2013\n(v19.1)',
                        'ecRad Baran2016\n(v18.1)',
                        ])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.89, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[1].set(
    xlabel='Solar transmissivity'
)
figname = f'05_HALO_AC3_RF17_RF18_transmissivity_sw_BACARDI_{revision}_ecRad_ice_optics_violin.png'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% ice optics parameterization - print stat
df_print = sw_stats.loc[pd.IndexSlice[:, ['BACARDI_org'] + sel_ver], :].sort_values('key')
print(df_print)

# %% ice optics parameterizations - plot violinplot of downward terrestrial irradiance
sel_ver = ['BACARDI', 'v15.1', 'v19.1', 'v18.1']
plt.rc('font', size=10)
_, axs = plt.subplots(2, 1, figsize=(15 * h.cm, 10 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df_lw[(df_lw.key == key)
                    & (df_lw.label.isin(sel_ver))
                    & (df_lw['var'] == 'flux_dn_lw')
                    & (df_lw.section == 'below')
                    ]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='values', y='label', hue='label', ax=ax,
                   order=sel_ver)
    ax.set(xlabel='',
           ylabel='',
           yticklabels='',
           xlim=(75, 200),
           )
    ax.set(yticklabels=['BACARDI',
                        'ecRad Reference\nFu-IFS (v15.1)',
                        'ecRad Yi2013\n(v19.1)',
                        'ecRad Baran2016\n(v18.1)',
                        ])
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.89, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[1].set(
    xlabel='Downward terrestrial irradiance below cloud (W$\\,$m$^-2$)'
)
figname = f'05_HALO_AC3_RF17_RF18_flux_dn_lw_BACARDI_{revision}_ecRad_ice_optics_violin.png'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% ice optics parameterization - print stat
df_print = lw_stats.loc[pd.IndexSlice[:, 'flux_dn_lw', ['BACARDI_org'] + sel_ver], :].sort_values('key')
print(df_print[df_print.section == 'below'])

# %% ice effective radius - plot reice with and without cosine dependence and using VarCloud IWC as input for case study clouds
plt.rc('font', size=10)
legend_labels = ['Off (IWC IFS)', 'On (IWC IFS)', 'Off (IWC VarCloud)', 'On (IWC VarCloud)']
linestyles = ['solid', 'solid', 'dashed', 'dashed']
binsizes = dict(iwc=1, reice=4)
binedges = dict(iwc=20, reice=100)
text_loc_x = 0.05
text_loc_y = 0.95
ylims = {'iwc': (0, 0.3), 'reice': (0, 0.25)}
_, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 10 * h.cm), layout='constrained')

# left panel - RF17 re_ice
ax = axs[0]
plot_ds = ecrad_orgs['RF17']
sel_time = slices['RF17']['below']
bins = np.arange(0, binedges['reice'], binsizes['reice'])
for i, v in enumerate(['v39.2', 'v15.1', 'v41.2', 'v16.1']):
    pds = plot_ds[v].re_ice.sel(time=sel_time).to_numpy().flatten() * 1e6
    pds = pds[~np.isnan(pds)]
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i],
        color=cbc[i],
        linestyle=linestyles[i],
        histtype='step',
        density=True,
        lw=2,
    )
    print(f'RF17 Mean reice {v}: {pds.mean():.2f}\n'
          f'n={len(pds)}')

ax.grid()
ax.text(text_loc_x, text_loc_y, '(a)',
        transform=ax.transAxes,
        )
ax.set(ylabel='Probability density function',
       xlabel=f'Ice effective radius ({h.plot_units['re_ice']})',
       ylim=ylims['reice'])
ax.set_title('RF 17 - 11 April 2024', fontsize=10)
ax.legend(title='Cosine dependence (input)')

# right panel - RF18 re_ice
ax = axs[1]
plot_ds = ecrad_orgs['RF18']
sel_time = slices['RF18']['below']
bins = np.arange(0, binedges['reice'], binsizes['reice'])
for i, v in enumerate(['v39.2', 'v15.1', 'v41.2', 'v16.1']):
    pds = plot_ds[v].re_ice.sel(time=sel_time).to_numpy().flatten() * 1e6
    pds = pds[~np.isnan(pds)]
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i],
        color=cbc[i],
        linestyle=linestyles[i],
        histtype='step',
        density=True,
        lw=2,
    )
    print(f'RF18 Mean reice {v}: {pds.mean():.2f}\n'
          f'n={len(pds)}')

ax.grid()
ax.text(text_loc_x, text_loc_y, '(b)',
        transform=ax.transAxes,
        )
ax.set(ylabel='',
       xlabel=f'Ice effective radius ({h.plot_units['re_ice']})',
       ylim=ylims['reice'])
ax.set_title('RF 18 - 12 April 2024', fontsize=10)

figname = f'{plot_path}/05_HALO-AC3_HALO_RF17_RF18_IFS_re_ice_pdf_case_studies_cosine_dependence.png'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% ice effective radius - plot violinplot of solar transmissivity cosine dependence
sel_ver = ['BACARDI', 'v15.1', 'v39.2']
plt.rc('font', size=10)
_, axs = plt.subplots(2, 1, figsize=(15 * h.cm, 9 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df[(df.key == key)
                 & (df.label.isin(sel_ver))]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='values', y='label', hue='label', ax=ax,
                   order=sel_ver)
    ax.set(xlabel='',
           ylabel='',
           yticklabels='',
           xlim=xlim,
           )
    ax.set(yticklabels=['BACARDI',
                        'ecRad Reference\nCosine (v15.1)',
                        'ecRad \nNo cosine (v39.2)',
                        ])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.89, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[1].set(
    xlabel='Solar transmissivity'
)
figname = f'05_HALO_AC3_RF17_RF18_transmissivity_sw_BACARDI_{revision}_ecRad_no_cosine_violin.png'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% plot deviation from measurement for ice optics params experiment
h.set_cb_friendly_colors('petroff_6')
sel_ver = ['v15.1', 'v19.1', 'v18.1']
fig, axs = plt.subplots(2, 2, figsize=(20 * h.cm, 10 * h.cm),
                        constrained_layout=True)
for i, key in enumerate(keys):
    ax = axs[i, 0]
    df_plot = sw_stats.loc[(key, sel_ver), :]
    ax.grid()
    sns.barplot(df_plot, y='label', x='deviation', errorbar=None, ax=ax, hue='label', )
    ax.set(xlabel='',
           ylabel='',
           xlim=(-11.5, 0),
           yticklabels=['ecRad Reference\nFu-IFS (v15.1)',
                        'ecRad Yi2013\n(v19.1)',
                        'ecRad Baran2016\n(v18.1)',
                        ])
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    print(df_plot['deviation'])

for i, key in enumerate(keys):
    ax = axs[i, 1]
    df_plot = lw_stats[lw_stats['section'] == 'below'].loc[(key, 'flux_dn_lw', sel_ver), :]
    ax.grid()
    sns.barplot(df_plot, y='label', x='deviation', errorbar=None, ax=ax, hue='label', )
    ax.set(xlabel='',
           ylabel='',
           yticklabels='',
           xlim=(0, 15)
           )
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    print(df_plot['deviation'])

axs[1, 0].set(xlabel='Deviation (%)')
axs[1, 1].set(xlabel='Deviation (%)')
figname = f'{plot_path}/HALO_AC3_RF17_RF18_BACARDI_{revision}_ecrad_deviation_ice_optics.png'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot deviation from measurement for VarCloud experiment
h.set_cb_friendly_colors('petroff_6')
sel_ver = ['v15.1', 'v36', 'v37', 'v38']
fig, axs = plt.subplots(2, 2, figsize=(20 * h.cm, 10 * h.cm),
                        constrained_layout=True)
for i, key in enumerate(keys):
    ax = axs[i, 0]
    df_plot = sw_stats.loc[(key, sel_ver), :]
    ax.grid()
    sns.barplot(df_plot, y='label', x='deviation', errorbar=None, ax=ax, hue='label', )
    ax.set(xlabel='',
           ylabel='',
           xlim=(-13, 4),
           yticklabels=['ecRad Reference\nFu-IFS (v15.1)',
                        'ecRad VarCloud\nFu-IFS (v36)',
                        'ecRad VarCloud\nYi2013 (v37)',
                        'ecRad VarCloud\nBaran2016 (v38)'
                        ])
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    print(df_plot['deviation'])

for i, key in enumerate(keys):
    ax = axs[i, 1]
    df_plot = lw_stats[lw_stats['section'] == 'below'].loc[(key, 'flux_dn_lw', sel_ver), :]
    ax.grid()
    sns.barplot(df_plot, y='label', x='deviation', errorbar=None, ax=ax, hue='label', )
    ax.set(xlabel='',
           ylabel='',
           yticklabels='',
           xlim=(0, 17.5)
           )
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    print(df_plot['deviation'])

axs[1, 0].set(xlabel='Deviation (%)')
axs[1, 1].set(xlabel='Deviation (%)')
figname = f'{plot_path}/HALO_AC3_RF17_RF18_BACARDI_{revision}_ecrad_deviation_varcloud.png'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()
# %% plot minimum ice effective radius from Sun2001 parameterization together with median ed from delatorre2023
latitudes = np.arange(0, 91)
de2re = 0.64952  # from suecrad.f90
min_ice = 60
min_diameter_um = 20 + (min_ice - 20) * np.cos((np.deg2rad(latitudes)))
min_radius_um = de2re * min_diameter_um

plt.rc('font', size=10)
_, ax = plt.subplots(figsize=(15 * h.cm, 6 * h.cm), layout='constrained')
ax.plot(latitudes, min_radius_um, '-', label='Minimum $r_{\\text{eff, ice}}$ Sun (2001)')
stats_ed.plot(x='mid_latitude', y='effective_radius', ax=ax,
              label='Mean $r_{\\text{eff, ice}}$\nDe La Torre Castro et al. (2023)')
ax.set(xlabel='Latitude (°N)',
       ylabel='Ice effective radius ($\\mathrm{\\mu}$m)',
       # ylim=(10, 40),
       xlim=0)
# ax.xaxis.set_major_locator(ticker.MultipleLocator(15))
# ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax.grid()
ax.legend()
plt.savefig(f'{plot_path}/02_reice_min_latitude.png', dpi=300)
plt.show()
plt.close()

# %% ice effective radius - plot reice with and without cosine dependence and using VarCloud IWC as input for case
# study clouds
plt.rc('font', size=10)
legend_labels = ['Off (IWC IFS)', 'On (IWC IFS)', 'Off (IWC VarCloud)', 'On (IWC VarCloud)']
linestyles = ['solid', 'solid', 'dashed', 'dashed']
binsizes = dict(iwc=1, reice=4)
binedges = dict(iwc=20, reice=100)
text_loc_x = 0.05
text_loc_y = 0.95
ylims = {'iwc': (0, 0.3), 'reice': (0, 0.25)}
_, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 10 * h.cm), layout='constrained')

# left panel - RF17 re_ice
ax = axs[0]
plot_ds = ecrad_orgs['RF17']
sel_time = slices['RF17']['below']
bins = np.arange(0, binedges['reice'], binsizes['reice'])
for i, v in enumerate(['v39.2', 'v15.1', 'v41.2', 'v16.1']):
    pds = plot_ds[v].re_ice.sel(time=sel_time).to_numpy().flatten() * 1e6
    pds = pds[~np.isnan(pds)]
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i],
        color=cbc[i],
        linestyle=linestyles[i],
        histtype='step',
        density=True,
        lw=2,
    )
    print(f'RF17 Mean reice {v}: {pds.mean():.2f}\n'
          f'n={len(pds)}')

ax.grid()
ax.text(text_loc_x, text_loc_y, '(a)',
        transform=ax.transAxes,
        )
ax.set(ylabel='Probability density function',
       xlabel=f'Ice effective radius ({h.plot_units['re_ice']})',
       ylim=ylims['reice'])
ax.set_title('RF 17 - 11 April 2024', fontsize=10)
ax.legend(title='Cosine dependence (input)')

# right panel - RF18 re_ice
ax = axs[1]
plot_ds = ecrad_orgs['RF18']
sel_time = slices['RF18']['below']
bins = np.arange(0, binedges['reice'], binsizes['reice'])
for i, v in enumerate(['v39.2', 'v15.1', 'v41.2', 'v16.1']):
    pds = plot_ds[v].re_ice.sel(time=sel_time).to_numpy().flatten() * 1e6
    pds = pds[~np.isnan(pds)]
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i],
        color=cbc[i],
        linestyle=linestyles[i],
        histtype='step',
        density=True,
        lw=2,
    )
    print(f'RF18 Mean reice {v}: {pds.mean():.2f}\n'
          f'n={len(pds)}')

ax.grid()
ax.text(text_loc_x, text_loc_y, '(b)',
        transform=ax.transAxes,
        )
ax.set(ylabel='',
       xlabel=f'Ice effective radius ({h.plot_units['re_ice']})',
       ylim=ylims['reice'])
ax.set_title('RF 18 - 12 April 2024', fontsize=10)

figname = f'{plot_path}/05_HALO-AC3_HALO_RF17_RF18_IFS_re_ice_pdf_case_studies_cosine_dependence.png'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% ice effective radius - plot violinplot of solar transmissivity cosine dependence
sel_ver = ['BACARDI', 'v15.1', 'v39.2', 'v42.2']
plt.rc('font', size=10)
_, axs = plt.subplots(2, 1, figsize=(15 * h.cm, 10 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df[(df.key == key)
                 & (df.label.isin(sel_ver))]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='values', y='label', hue='label', ax=ax,
                   order=sel_ver)
    ax.set(xlabel='',
           ylabel='',
           xlim=xlim,
           yticklabels=['BACARDI',
                        'ecRad Reference\nCosine (v15.1)',
                        'ecRad \nNo cosine (v39.2)',
                        'ecRad VarCloud\nNo cosine (v41.2)'
                        ])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.89, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[1].set(
    xlabel='Solar transmissivity'
)
figname = f'05_HALO_AC3_RF17_RF18_transmissivity_sw_BACARDI_{revision}_ecRad_no_cosine_violin.png'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()
# %% plot deviation from measurement for no cosine VarCloud experiment
h.set_cb_friendly_colors('petroff_6')
sel_ver = ['v15.1', 'v39.2', 'v42.2']
fig, axs = plt.subplots(2, 2, figsize=(20 * h.cm, 10 * h.cm),
                        constrained_layout=True)
for i, key in enumerate(keys):
    ax = axs[i, 0]
    df_plot = sw_stats.loc[(key, sel_ver), :]
    ax.grid()
    sns.barplot(df_plot, y='label', x='deviation', errorbar=None, ax=ax, hue='label', )
    ax.set(xlabel='',
           ylabel='',
           xlim=(-11, 6),
           yticklabels=['ecRad Reference\nCosine (v15.1)',
                        'ecRad \nNo cosine (v39.2)',
                        'ecRad VarCloud\nNo cosine (v41.2)'
                        ])
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    print(df_plot['deviation'])

for i, key in enumerate(keys):
    ax = axs[i, 1]
    df_plot = lw_stats[lw_stats['section'] == 'below'].loc[(key, 'flux_dn_lw', sel_ver), :]
    ax.grid()
    sns.barplot(df_plot, y='label', x='deviation', errorbar=None, ax=ax, hue='label', )
    ax.set(xlabel='',
           ylabel='',
           yticklabels='',
           xlim=(0, 15)
           )
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    print(df_plot['deviation'])

axs[1, 0].set(xlabel='Solar transmissivity deviation (%)')
axs[1, 1].set(xlabel='Terrestrial downward irradiance deviation (%)')
figname = f'{plot_path}/HALO_AC3_RF17_RF18_BACARDI_{revision}_ecrad_deviation_varcloud_no_cosine.png'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot a dropsonde temperature and humidity profile
_, ax = plt.subplots()
plt.show()
plt.close()
# %% plot BACARDI F_dw_solar deviation from libRadtran simulation polar plot
plot_ds = (bacardi_ds_all
           .where(bacardi_ds_all.alt >= 10000)
           # .isel(time=slice(0, len(bacardi_ds_all.time), 100))
           )
h.set_cb_friendly_colors()
plt.rc('font', size=12)
fig, ax = plt.subplots(figsize=h.figsize_wide,
                       layout='constrained',
                       subplot_kw={'projection': 'polar'})
scatter = ax.scatter(np.deg2rad(plot_ds['viewing_dir']), plot_ds['sza'],
                     c=plot_ds['deviation'], s=1,
                     cmap=cmr.pride, norm=colors.CenteredNorm(vcenter=0, halfrange=10))
fig.colorbar(scatter,
             label=r'$F^{\downarrow}_{\text{sol}}$ Deviation from simulation (%)',
             format='%2.1f',
             extend='both')
ax.set_rmax(91)
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi / 2)
ax.set_rorigin(55)
ax.set_rticks([60, 65, 70, 75, 80, 85, 90])
# add aircraft at center of plot
ax.text(0.506, 0.51, '\u2708',
        fontsize=45, ha='center', va='center', rotation=90,
        rotation_mode='anchor', transform=ax.transAxes)

figname = f'{plot_path}/03_BACARDI_vs_simulation_sza_polar_percent.png'
plt.savefig(figname, dpi=300, bbox_inches='tight')
plt.show()
plt.close()
