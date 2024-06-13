#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 30.05.2024

Here the figures from Chapter 5 (Results) Section 5.3 (Terrestrial Radiative Effect) of my thesis are created.



"""
# %% import modules
import os
import dill

import cmasher as cmr
import cmcrameri.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import colors
from metpy.calc import relative_humidity_from_specific_humidity
from metpy.units import units as u
import numpy as np
import pandas as pd
from skimage import io
from sklearn.neighbors import BallTree
import seaborn as sns
from tqdm import tqdm
import xarray as xr

import pylim.halo_ac3 as meta
import pylim.helpers as h
import pylim.meteorological_formulas as met
from pylim import ecrad

h.set_cb_friendly_colors('petroff_6')
cbc = h.get_cb_friendly_colors('petroff_6')

# %% set paths
campaign = 'halo-ac3'
save_path = 'C:/Users/Johannes/Documents/Doktor/manuscripts/_thesis/data'
plot_path = 'C:/Users/Johannes/Documents/Doktor/manuscripts/_thesis/figure'
trajectory_path = f'{h.get_path('trajectories', campaign=campaign)}/selection_CC_and_altitude'
keys = ['RF17', 'RF18']
ecrad_versions = [f'v{x}' for x in [13, 13.1, 13.2, 15, 15.1, 16.1, 18.1, 19.1,
                                    22.1, 24.1, 26, 27, 30.1, 31.1, 32.1,
                                    36, 37, 38, 39.2, 40.2, 41.2, 42.2]]

# %% read in data
(
    bahamas_ds, bacardi_ds, bacardi_ds_res, bacardi_ds_res_1s, ecrad_dicts,
    varcloud_ds, above_clouds, below_clouds, slices,
    ecrad_orgs, ifs_ds, ifs_ds_sel, dropsonde_ds, albedo_dfs, sat_imgs
) = (dict(), dict(), dict(), dict(), dict(),
     dict(), dict(), dict(), dict(),
     dict(), dict(), dict(), dict(), dict(), dict())

left, right, bottom, top = 0, 1000000, -1000000, 0
sat_img_extent = (left, right, bottom, top)
# read in dropsonde data
dropsonde_path = f'{h.get_path('all', campaign=campaign, instrument='dropsondes')}/Level_3'
dropsonde_file = 'merged_HALO_P5_beta_v2.nc'
dds = xr.open_dataset(f'{dropsonde_path}/{dropsonde_file}')

for key in keys:
    flight = meta.flight_names[key]
    date = flight[9:17]
    urldate = pd.to_datetime(date).strftime('%Y-%m-%d')
    bacardi_path = h.get_path('bacardi', flight, campaign)
    bahamas_path = h.get_path('bahamas', flight, campaign)
    ifs_path = f'{h.get_path('ifs', flight, campaign)}/{date}'
    ecrad_path = f'{h.get_path('ecrad', flight, campaign)}/{date}'
    varcloud_path = h.get_path('varcloud', flight, campaign)

    # filenames
    bahamas_file = f'HALO-AC3_HALO_BAHAMAS_{date}_{key}_v1_JR.nc'
    bacardi_file = f'HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1_JR.nc'
    ifs_file = f'ifs_{date}_00_ml_O1280_processed.nc'
    varcloud_file = [f for f in os.listdir(varcloud_path) if f.endswith('.nc')][0]
    satfile = f'{save_path}/{key}_MODIS_Terra_CorrectedReflectance_Bands367.png'
    sat_url = f'https://gibs.earthdata.nasa.gov/wms/epsg3413/best/wms.cgi?\
version=1.3.0&service=WMS&request=GetMap&\
format=image/png&STYLE=default&bbox={left},{bottom},{right},{top}&CRS=EPSG:3413&\
HEIGHT=8192&WIDTH=8192&TIME={urldate}&layers=MODIS_Terra_CorrectedReflectance_Bands367'

    # read in aircraft data
    bahamas_ds[key] = xr.open_dataset(f'{bahamas_path}/{bahamas_file}')
    bacardi = xr.open_dataset(f'{bacardi_path}/{bacardi_file}')
    # read in resampled BACARDI data
    bacardi_res = xr.open_dataset(f'{bacardi_path}/{bacardi_file.replace('.nc', '_1Min_v2.nc')}')
    bacardi_ds_res[key] = bacardi_res.copy(deep=True)
    bacardi_res_1s = xr.open_dataset(f'{bacardi_path}/{bacardi_file.replace('.nc', '_1s_v2.nc')}')
    bacardi_ds_res_1s[key] = bacardi_res_1s.copy(deep=True)

    # read in results of albedo experiment
    albedo_dfs[key] = pd.read_csv(f'{save_path}/{flight}_boxplot_data.csv')

    dropsonde_ds[key] = dds.where(dds.launch_time.dt.date == pd.to_datetime(date).date(), drop=True)

    # read in satellite image
    try:
        sat_imgs[key] = io.imread(satfile)
    except FileNotFoundError:
        sat_imgs[key] = io.imread(sat_url)

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

    # filter BACARDI data for motion flag
    bacardi_org = bacardi.copy(deep=True)
    bacardi = bacardi.where(bacardi.motion_flag)
    # overwrite variables which do not need to be filtered with original values
    for var in ['alt', 'lat', 'lon', 'sza', 'saa', 'attitude_flag', 'segment_flag', 'motion_flag']:
        bacardi[var] = bacardi_org[var]

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

    # interpolate standard ecRad simulation onto BACARDI time
    bacardi['ecrad_fdw'] = ecrad_dict['v15.1'].flux_dn_sw_clear.interp(time=bacardi.time,
                                                                       kwargs={'fill_value': 'extrapolate'})
    # calculate transmissivity using ecRad at TOA and above cloud
    bacardi['transmissivity_TOA'] = bacardi['F_down_solar'] / bacardi['ecrad_fdw'].isel(half_level=0)
    bacardi['transmissivity_above_cloud'] = bacardi['F_down_solar'] / bacardi['ecrad_fdw'].isel(half_level=73)
    bacardi_ds[key] = bacardi

    # get flight segmentation and select below and above cloud section
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

# %% define variables for multiple use
date_title = ['11 April 2022', '12 April 2022']
panel_label = ['(a)', '(b)']

# %% prepare data for violin plots
ecrad_var = 'flux_net_lw'
label = 'flux_net_lw'
bacardi_var = 'F_net_terrestrial'
filepath = f'{save_path}/halo-ac3_{label}_boxplot_data.csv'
df_lw = pd.DataFrame()
for key in keys:
    dfs = list()
    dfs.append(df_lw)
    for v in ecrad_versions:
        dfs.append(pd.DataFrame({'values': (ecrad_orgs[key][v][ecrad_var]
                                            .isel(half_level=ecrad_dicts[key][v].aircraft_level)
                                            .sel(time=slices[key]['below'])
                                            .to_numpy()
                                            .flatten()),
                                 'label': v,
                                 'key': key}))

    dfs.append(pd.DataFrame({'values': (bacardi_ds_res_1s[key][bacardi_var]
                                        .sel(time=slices[key]['below'])
                                        .dropna('time')
                                        .to_pandas()
                                        .reset_index(drop=True)),
                             'label': 'BACARDI_1s',
                             'key': key}))

    dfs.append(pd.DataFrame({'values': (bacardi_ds_res[key][bacardi_var]
                                        .sel(time=slices[key]['below'])
                                        .dropna('time')
                                        .to_pandas()
                                        .reset_index(drop=True)),
                             'label': 'BACARDI',
                             'key': key}))

    # add variable to original BACARDI data
    bacardi_ds[key][bacardi_var] = bacardi_ds[key]['F_down_terrestrial'] - bacardi_ds[key]['F_up_terrestrial']
    dfs.append(pd.DataFrame({'values': (bacardi_ds[key][bacardi_var]
                                        .sel(time=slices[key]['below'])
                                        .dropna('time')
                                        .to_pandas()
                                        .reset_index(drop=True)),
                             'label': 'BACARDI_org',
                             'key': key}))

    df_lw = pd.concat(dfs)

df_lw.to_csv(filepath, index=False)
df_lw = df_lw.reset_index(drop=True)

# %% get statistics
lw_stats = (df_lw
            .groupby(['key', 'label'])['values']
            .describe()
            .sort_values(['key', 'mean'], ascending=[True, False]))
versions = [v for v in lw_stats.index.get_level_values('label') if v.startswith('v')]
df_save = lw_stats.reset_index()
name = list()
aerosol = list()
for v in df_save['label']:
    try:
        n = ecrad.get_version_name(v[:3])
        name.append(n)
        a = 'On' if v[:3] in ecrad.aerosol_on else 'Off'
        aerosol.append(a)
    except ValueError:
        name.append(v)
        aerosol.append('Off')
df_save = df_save.assign(name=name, aerosol=aerosol)
df_save.to_csv(f'{save_path}/HALO-AC3_net_terrestrial_irradiance_stats.csv',
               index=False)

# %% violinplot
sel_ver = ['BACARDI', 'v15.1']
h.set_cb_friendly_colors('petroff_6')
plt.rc('font', size=10)
_, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 6 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df_lw[(df_lw.key == key)
                 & (df_lw.label.isin(sel_ver))]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='values', y='label', hue='label',
                   ax=ax, order=sel_ver)
    ax.set(xlabel=f'Net terrestrial irradiance\n({h.plot_units["flux_net_lw"]})',
           ylabel='',
           yticklabels='',
           xlim=(-140, -40),
           )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.91, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[0].set(yticklabels=['BACARDI',
                        'ecRad Reference\nFu-IFS (v15.1)',
                        ])
figname = f'05_HALO_AC3_RF17_RF18_flux_net_lw_BACARDI_ecRad_violin.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% print statistics
df_print = lw_stats.loc[pd.IndexSlice[:, ['BACARDI_org'] + sel_ver], :].sort_values('key')
print(df_print)

# %% plot BACARDI vs. ecRad terrestrial upward below cloud
plt.rc('font', size=10)
label = ['(a)', '(b)']
for v in ['v15.1', 'v18.1', 'v19.1']:
    _, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 9 * h.cm),
                          layout='constrained')
    for i, key in enumerate(keys):
        ax = axs[i]
        time_sel = slices[key]['below']
        bacardi_res = bacardi_ds_res[key]
        bacardi_plot = bacardi_res.sel(time=time_sel)
        ecrad_ds = ecrad_dicts[key][v]
        height_sel = ecrad_dicts[key][v].aircraft_level
        ecrad_plot = (ecrad_ds.flux_up_lw
                      .isel(half_level=height_sel)
                      .sel(time=time_sel))

        # actual plotting
        rmse = np.sqrt(np.mean((bacardi_plot['F_up_terrestrial'] - ecrad_plot) ** 2)).to_numpy()
        bias = np.nanmean((bacardi_plot['F_up_terrestrial'] - ecrad_plot).to_numpy())
        ax.scatter(bacardi_plot['F_up_terrestrial'], ecrad_plot, color=cbc[3])
        ax.axline((0, 0), slope=1, color='k', lw=2, transform=ax.transAxes)
        ax.set(
            aspect='equal',
            xlabel=r'Measured irradiance (W$\,$m$^{-2}$)',
            ylabel=r'Simulated irradiance (W$\,$m$^{-2}$)',
            xlim=(210, 230),
            ylim=(210, 230),
        )
        ax.grid()
        ax.text(
            0.025,
            0.95,
            f'{label[i]} {key.replace('1', ' 1')}\n'
            f'n= {sum(~np.isnan(bacardi_plot['F_up_terrestrial'])):.0f}\n'
            f'RMSE: {rmse:.0f} {h.plot_units['flux_up_lw']}\n'
            f'Bias: {bias:.0f} {h.plot_units['flux_up_lw']}',
            ha='left',
            va='top',
            transform=ax.transAxes,
            bbox=dict(fc='white', ec='black', alpha=0.8, boxstyle='round'),
        )

    figname = f'{plot_path}/05_HALO-AC3_RF17_RF18_bacardi_ecrad_f_up_lw_below_cloud_{v}.pdf'
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot BACARDI vs. ecRad terrestrial downward below cloud
plt.rc('font', size=10)
label = ['(a)', '(b)']
# xlims = [(85, 110), (130, 180)]
xlims = [(175, 200), (155, 180)]
for v in ['v15.1', 'v18.1', 'v19.1']:
    _, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 9 * h.cm),
                          layout='constrained')
    for i, key in enumerate(keys):
        ax = axs[i]
        time_sel = slices[key]['above']
        bacardi_res = bacardi_ds_res[key]
        bacardi_plot = bacardi_res.sel(time=time_sel)
        ecrad_ds = ecrad_dicts[key][v]
        height_sel = ecrad_dicts[key][v].aircraft_level
        ecrad_plot = (ecrad_ds.flux_up_lw
                      .isel(half_level=height_sel)
                      .sel(time=time_sel))

        # actual plotting
        rmse = np.sqrt(np.mean((bacardi_plot['F_up_terrestrial'] - ecrad_plot) ** 2)).to_numpy()
        bias = np.nanmean((bacardi_plot['F_up_terrestrial'] - ecrad_plot).to_numpy())
        ax.scatter(bacardi_plot['F_up_terrestrial'], ecrad_plot, color=cbc[3])
        ax.axline((0, 0), slope=1, color='k', lw=2, transform=ax.transAxes)
        ax.set(
            aspect='equal',
            xlabel=r'Measured irradiance (W$\,$m$^{-2}$)',
            ylabel=r'Simulated irradiance (W$\,$m$^{-2}$)',
            xlim=xlims[i],
            ylim=xlims[i],
        )
        ax.grid()
        ax.text(
            0.05,
            0.95,
            f'{label[i]} {key.replace('1', ' 1')}\n'
            f'n= {sum(~np.isnan(bacardi_plot['F_up_terrestrial'])):.0f}\n'
            f'RMSE: {rmse:.0f} {h.plot_units['flux_up_lw']}\n'
            f'Bias: {bias:.0f} {h.plot_units['flux_up_lw']}',
            ha='left',
            va='top',
            transform=ax.transAxes,
            bbox=dict(fc='white', ec='black', alpha=0.8, boxstyle='round'),
        )

    figname = f'{plot_path}/05_HALO-AC3_RF17_RF18_bacardi_ecrad_f_up_lw_above_cloud_{v}.pdf'
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% print mean below-cloud terrestrial upward irradiance
for key in keys:
    time_sel = slices[key]['below']
    bacardi_res = bacardi_ds_res[key]
    bacardi_plot = bacardi_res.sel(time=time_sel)
    ecrad_ds = ecrad_dicts[key][v]
    height_sel = ecrad_dicts[key][v].aircraft_level
    ecrad_plot = (ecrad_ds.flux_up_lw
                  .isel(half_level=height_sel)
                  .sel(time=time_sel))
    print(f'{key}\n'
          f'Mean F up BACARDI: {bacardi_plot["F_up_terrestrial"].mean():.2f}\n'
          f'Std F up BACARDI: {bacardi_plot["F_up_terrestrial"].std():.2f}\n'
          f'Mean F up ecRad: {ecrad_plot.mean():.2f}\n'
          f'Std F up ecRad: {ecrad_plot.std():.2f}\n'
          )

# %% print mean below-cloud terrestrial downward irradiance
for key in keys:
    time_sel = slices[key]['below']
    bacardi_res = bacardi_ds_res[key]
    bacardi_plot = bacardi_res.sel(time=time_sel)
    ecrad_ds = ecrad_dicts[key][v]
    height_sel = ecrad_dicts[key][v].aircraft_level
    ecrad_plot = (ecrad_ds.flux_dn_lw
                  .isel(half_level=height_sel)
                  .sel(time=time_sel))
    print(f'{key}\n'
          f'Mean F down BACARDI: {bacardi_plot["F_down_terrestrial"].mean():.2f}\n'
          f'Std F down BACARDI: {bacardi_plot["F_down_terrestrial"].std():.2f}\n'
          f'Mean F down ecRad: {ecrad_plot.mean():.2f}\n'
          f'Std F down ecRad: {ecrad_plot.std():.2f}\n'
          )

# %% sea ice albedo experiment plot violinplot
sel_ver = ['BACARDI', 'v15.1', 'v13', 'v13.2']
h.set_cb_friendly_colors('petroff_6')
plt.rc('font', size=10)
_, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 7.5 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df_lw[(df_lw.key == key)
                 & (df_lw.label.isin(sel_ver))]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='values', y='label', hue='label',
                   ax=ax, order=sel_ver)
    ax.set(xlabel=f'Net terrestrial irradiance\n({h.plot_units["flux_net_lw"]})',
           ylabel='',
           yticklabels='',
           # xlim=(-140, -40),
           )
    # ax.xaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.94, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[0].set_yticklabels(['BACARDI',
                        'ecRad Reference\nsimulation (v15.1)',
                        'ecRad Open ocean\nsimulation (v13)',
                        'ecRad Measured albedo\nsimulation (v13.2)'],)
figname = f'05_HALO_AC3_RF17_RF18_flux_net_lw_BACARDI_ecRad_albedo_violin.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% sea ice albedo experiment plot skin temperature
sel_ver = ['v15.1', 'v13', 'v13.2']
var = 'skin_temperature'
h.set_cb_friendly_colors('petroff_6')
plt.rc('font', size=10)
_, axs = plt.subplots(2, 3, figsize=(15 * h.cm, 7.5 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    for ii, v in enumerate(sel_ver):
        a = ax[ii]
        ds = ecrad_orgs[key][v][var].sel(time=slices[key]['below']).to_numpy().flatten()
        sns.histplot(ds, label=v, stat='density', element='step',
                     kde=False, ax=a)
        a.grid()
        a.text(0.5, 0.87, f'{v}', transform=a.transAxes)

    ax[1].set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
axs[1][1].set_xlabel('Skin temperature (K)')
figname = f'05_HALO_AC3_RF17_RF18_{var}_pdf_ecRad_albedo_experiment.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% 3-D effect violinplot
sel_ver = ['BACARDI', 'v15.1', 'v22.1', 'v18.1', 'v24.1']
h.set_cb_friendly_colors('petroff_6')
plt.rc('font', size=10)
_, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 10 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df_lw[(df_lw.key == key)
                 & (df_lw.label.isin(sel_ver))]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='values', y='label', hue='label',
                   ax=ax, order=sel_ver)
    ax.set(xlabel=f'Net terrestrial irradiance\n({h.plot_units["flux_net_lw"]})',
           ylabel='',
           yticklabels='',
           xlim=(-140, -40),
           )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.94, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[0].set_yticklabels(['BACARDI',
                        'ecRad Reference\nFu-IFS (v15.1)',
                        'ecRad 3D on\nFu-IFS (v22.1)',
                        'ecRad Reference\nBaran2016 (v18.1)',
                        'ecRad 3D on\nBaran2016 (v24.1)'])
figname = f'05_HALO_AC3_RF17_RF18_flux_net_lw_BACARDI_ecRad_3d_effects_violin.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% 3-D effect print statistics
df_print = lw_stats.loc[pd.IndexSlice[:, ['BACARDI_org'] + sel_ver], :].sort_values('key')
print(df_print)

# %% aerosol plot violinplot
sel_ver = ['BACARDI', 'v15.1', 'v30.1']
h.set_cb_friendly_colors('petroff_6')
plt.rc('font', size=10)
_, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 7.5 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df_lw[(df_lw.key == key)
                 & (df_lw.label.isin(sel_ver))]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='values', y='label', hue='label',
                   ax=ax, order=sel_ver)
    ax.set(xlabel=f'Net terrestrial irradiance\n({h.plot_units["flux_net_lw"]})',
           ylabel='',
           yticklabels='',
           xlim=(-140, -40),
           )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.94, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[0].set(yticklabels=['BACARDI',
                        'ecRad Reference\nFu-IFS (v15.1)',
                        'ecRad aerosol on\nFu-IFS (v30.1)',
                        ])
figname = f'05_HALO_AC3_RF17_RF18_flux_net_lw_BACARDI_ecRad_aerosol_violin.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% aerosol plot violinplot - all ice optics
sel_ver = ['BACARDI', 'v15.1', 'v30.1', 'v19.1', 'v31.1', 'v18.1', 'v32.1']
h.set_cb_friendly_colors('petroff_6')
plt.rc('font', size=10)
_, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 10 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df_lw[(df_lw.key == key)
                 & (df_lw.label.isin(sel_ver))]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='values', y='label', hue='label',
                   ax=ax, order=sel_ver)
    ax.set(xlabel=f'Net terrestrial irradiance\n({h.plot_units["flux_net_lw"]})',
           ylabel='',
           yticklabels='',
           xlim=(-140, -40),
           )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.94, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[0].set(yticklabels=['BACARDI',
                        'ecRad Reference\nFu-IFS (v15.1)',
                        'ecRad aerosol on\nFu-IFS (v30.1)',
                        'ecRad Reference\nYi2013 (v19.1)',
                        'ecRad aerosol on\nYi2013 (v31.1)',
                        'ecRad Reference\nBaran2016 (v18.1)',
                        'ecRad aerosol on\nBaran2016 (v32.1)',
                        ])
figname = f'05_HALO_AC3_RF17_RF18_flux_net_lw_BACARDI_ecRad_aerosol_violin_all.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% aerosol - plot total aerosol mass mixing ratio
plt.rc('font', size=9)
fig, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 7.5 * h.cm),
                        layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    ds = ecrad_orgs[key]['v30.1']
    mmr = (ds.aerosol_mmr
           .mean(dim='column')
           .sum(dim='aer_type')
           .sel(time=slices[key]['case']) * 1e9)
    mmr = mmr.where(mmr > 0)
    pcm = mmr.plot(x='time', cmap=cm.batlow,
                   vmin=0.0003, vmax=60,
                   add_colorbar=False,
                   norm=colors.LogNorm(),
                   # cbar_kwargs={
                   #     'label': 'Aerosol mass mixing ratio (mg$\\,$kg$^{-1}$)'},
                   ax=ax,
                   )
    ax.set(
        xlabel='Time (UTC)',
        ylabel='',
        ylim=(137, 0),
    )
    ax.set_title(f'{key.replace("1", " 1")} - {date_title[i]}', fontsize=9)
    h.set_xticks_and_xlabels(ax, time_extend=pd.Timedelta(1, 'h'))

axs[0].set_ylabel('Model level')
fig.colorbar(pcm, ax=axs.ravel().tolist(),
             label='Aerosol mass mixing ratio ($\\mu$g$\\,$kg$^{-1}$)')
figname = f'HALO-AC3_RF17_RF18_case_IFS_total_aerosol_mmr.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% aerosol - plot each aerosol species
plt.rc('font', size=9)
for type in ecrad_orgs['RF17']['v30.1'].aer_type:
    fig, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 7.5 * h.cm),
                            layout='constrained')
    for i, key in enumerate(keys):
        ax = axs[i]
        ds = ecrad_orgs[key]['v30.1']
        mmr = (ds.aerosol_mmr
               .mean(dim='column')
               .sel(aer_type=type,
                    time=slices[key]['case'])
               * 1e9)
        mmr = mmr.where(mmr > 0)
        pcm = mmr.plot(x='time', cmap=cm.batlow,
                       vmin=0.0003, vmax=60,
                       add_colorbar=False,
                       norm=colors.LogNorm(),
                       # cbar_kwargs={
                       #     'label': 'Aerosol mass mixing ratio (mg$\\,$kg$^{-1}$)'},
                       ax=ax,
                       )
        ax.set(
            xlabel='Time (UTC)',
            ylabel='',
            ylim=(137, 0),
        )
        ax.set_title(f'{key.replace("1", " 1")} - {date_title[i]}', fontsize=9)
        h.set_xticks_and_xlabels(ax, time_extend=pd.Timedelta(1, 'h'))

    axs[0].set_ylabel('Model level')
    fig.colorbar(pcm, ax=axs.ravel().tolist(),
                 label='Aerosol mass mixing ratio ($\\mu$g$\\,$kg$^{-1}$)')
    figname = f'HALO-AC3_RF17_RF18_case_IFS_aerosol_mmr_{type.to_numpy()}.png'
    plt.savefig(f'{plot_path}/{figname}', dpi=300)
    plt.show()
    plt.close()

# %% aerosol - plot total aerosol mass mixing ratio
plt.rc('font', size=9)
fig, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 7.5 * h.cm),
                        layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    ds = ecrad_orgs[key]['v30.1']
    mmr = (ds.aerosol_mmr
           .mean(dim='column')
           .sum(dim='aer_type')
           .sel(time=slices[key]['case']) * 1e9)
    mmr = mmr.where(mmr > 0)
    pcm = mmr.plot(x='time', cmap=cm.batlow,
                   vmin=0.0003, vmax=60,
                   add_colorbar=False,
                   norm=colors.LogNorm(),
                   # cbar_kwargs={
                   #     'label': 'Aerosol mass mixing ratio (mg$\\,$kg$^{-1}$)'},
                   ax=ax,
                   )
    ax.set(
        xlabel='Time (UTC)',
        ylabel='',
        ylim=(137, 0),
    )
    ax.set_title(f'{key.replace("1", " 1")} - {date_title[i]}', fontsize=9)
    h.set_xticks_and_xlabels(ax, time_extend=pd.Timedelta(1, 'h'))

axs[0].set_ylabel('Model level')
fig.colorbar(pcm, ax=axs.ravel().tolist(),
             label='Aerosol mass mixing ratio ($\\mu$g$\\,$kg$^{-1}$)')
figname = f'HALO-AC3_RF17_RF18_case_IFS_total_aerosol_mmr.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% aerosol print stats
df_print = lw_stats.loc[pd.IndexSlice[:, ['BACARDI_org'] + sel_ver], :].sort_values('key')
print(df_print)

# %% ice optics - plot violinplot
sel_ver = ['BACARDI', 'v15.1', 'v19.1', 'v18.1']
h.set_cb_friendly_colors('petroff_6')
plt.rc('font', size=10)
_, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 7.5 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df_lw[(df_lw.key == key)
                 & (df_lw.label.isin(sel_ver))]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='values', y='label', hue='label',
                   ax=ax, order=sel_ver)
    ax.set(xlabel=f'Net terrestrial irradiance\n({h.plot_units["flux_net_lw"]})',
           ylabel='',
           yticklabels='',
           xlim=(-140, -40),
           )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.94, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[0].set(yticklabels=['BACARDI',
                        'ecRad Reference\nFu-IFS (v15.1)',
                        'ecRad Reference\nYi2013 (v19.1)',
                        'ecRad Reference\nBaran2016 (v18.1)',
                        ],)
figname = f'05_HALO_AC3_RF17_RF18_flux_net_lw_BACARDI_ecRad_ice_optics_violin.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% ice optics - print stats
df_print = lw_stats.loc[pd.IndexSlice[:, ['BACARDI_org'] + sel_ver], :].sort_values('key')
print(df_print)

# %% VarCloud - plot violinplot
sel_ver = ['BACARDI', 'v15.1', 'v36']
h.set_cb_friendly_colors('petroff_6')
plt.rc('font', size=10)
_, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 7.5 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df_lw[(df_lw.key == key)
                 & (df_lw.label.isin(sel_ver))]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='values', y='label', hue='label',
                   ax=ax, order=sel_ver)
    ax.set(xlabel=f'Net terrestrial irradiance\n({h.plot_units["flux_net_lw"]})',
           ylabel='',
           yticklabels='',
           xlim=(-140, -20),
           )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.94, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[0].set(yticklabels=['BACARDI',
                        'ecRad Reference\nFu-IFS (v15.1)',
                        'ecRad VarCloud\nFu-IFS (v36)',
                        ],)
figname = f'05_HALO_AC3_RF17_RF18_flux_net_lw_BACARDI_ecRad_varcloud_violin.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% VarCloud - plot violinplot - all ice optics
sel_ver = ['BACARDI', 'v15.1', 'v36', 'v19.1', 'v37', 'v18.1', 'v38']
xlims = [(-140, -90), (-140, -10)]
locator = [10, 25]
h.set_cb_friendly_colors('petroff_6')
plt.rc('font', size=10)
_, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 10 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df_lw[(df_lw.key == key)
                 & (df_lw.label.isin(sel_ver))]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='values', y='label', hue='label',
                   ax=ax, order=sel_ver)
    ax.set(xlabel=f'Net terrestrial irradiance\n({h.plot_units["flux_net_lw"]})',
           ylabel='',
           yticklabels='',
           xlim=xlims[i],
           )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(locator[i]))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.94, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[0].set(yticklabels=['BACARDI',
                        'ecRad Reference\nFu-IFS (v15.1)',
                        'ecRad VarCloud\nFu-IFS (v36)',
                        'ecRad Reference\nYi2013 (v19.1)',
                        'ecRad VarCloud\nYi2013 (v37)',
                        'ecRad Reference\nBaran2016 (v18.1)',
                        'ecRad VarCloud\nBaran2016 (v38)',
                        ],)
figname = f'05_HALO_AC3_RF17_RF18_flux_net_lw_BACARDI_ecRad_varcloud_violin_all.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()
# %% VarCloud - print stats
df_print = lw_stats.loc[pd.IndexSlice[:, ['BACARDI_org'] + sel_ver], :].sort_values('key')
print(df_print)
# %% re_ice latitude dependence - plot violinplot
sel_ver = ['BACARDI', 'v15.1', 'v39.2']
h.set_cb_friendly_colors('petroff_6')
plt.rc('font', size=10)
_, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 7.5 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df_lw[(df_lw.key == key)
                 & (df_lw.label.isin(sel_ver))]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='values', y='label', hue='label',
                   ax=ax, order=sel_ver)
    ax.set(xlabel=f'Net terrestrial irradiance\n({h.plot_units["flux_net_lw"]})',
           ylabel='',
           yticklabels='',
           xlim=(-140, -20),
           )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.94, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[0].set(yticklabels=['BACARDI',
                        'ecRad Reference\nFu-IFS (v15.1)',
                        'ecRad no latitude\nFu-IFS (v39.2)',
                        ],)
figname = f'05_HALO_AC3_RF17_RF18_flux_net_lw_BACARDI_ecRad_re_ice_latitude_violin.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% re_ice latitude - print stats
df_print = lw_stats.loc[pd.IndexSlice[:, ['BACARDI_org'] + sel_ver], :].sort_values('key')
print(df_print)