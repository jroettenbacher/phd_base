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
from skimage import io, measure
from sklearn.neighbors import BallTree
import seaborn as sns
from tqdm import tqdm
import xarray as xr

import pylim.halo_ac3 as meta
import pylim.helpers as h
import pylim.meteorological_formulas as met
from pylim import ecrad
from pylim.meteorological_formulas import calculate_absorption_coefficient_terrestrial

h.set_cb_friendly_colors('petroff_6')
cbc = h.get_cb_friendly_colors('petroff_6')

# %% set paths
campaign = 'halo-ac3'
save_path = 'C:/Users/Johannes/Documents/Doktor/manuscripts/_thesis/data'
plot_path = 'C:/Users/Johannes/Documents/Doktor/manuscripts/_thesis/figure'
trajectory_path = f'{h.get_path('trajectories', campaign=campaign)}/selection_CC_and_altitude'
keys = ['RF17', 'RF18']
ecrad_versions = [f'v{x}' for x in [13, 13.1, 13.2, 15, 15.1, 16, 18.1, 19.1, 20,
                                    22.1, 24.1, 26, 27, 28, 30.1, 31.1, 32.1,
                                    36, 37, 38, 39.2, 40.2, 41.2, 42.2, 43.1, 44]]

# %% read in data
(
    bahamas_ds, bacardi_ds, bacardi_ds_res, bacardi_ds_res_1s, ecrad_dicts,
    varcloud_ds, above_clouds, below_clouds, slices,
    ecrad_orgs, ifs_ds, ifs_ds_sel, dropsonde_ds, albedo_dfs, sat_imgs,
    rad_props
) = (dict(), dict(), dict(), dict(), dict(),
     dict(), dict(), dict(), dict(), dict(),
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

    # read in radiative properties
    rad_props_dict = dict()
    for k in ecrad_versions:
        try:
            rp = xr.open_dataset(f'{ecrad_path}/radiative_properties_merged_{date}_{k}.nc')
            rad_props_dict[k] = rp.copy(deep=True)
        except FileNotFoundError:
            continue

    rad_props[key] = rad_props_dict

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
                    dfs.append(
                        pd.DataFrame({'values': np.array([np.nan]),
                                      'label': v,
                                      'var': var,
                                      'section': sec,
                                      'key': key}))

        for var in bacardi_var:
            dfs.append(pd.DataFrame({'values': (bacardi_ds_res_1s[key][var]
                                                .sel(time=slices[key][sec])
                                                .dropna('time')
                                                .to_pandas()
                                                .reset_index(drop=True)),
                                     'label': 'BACARDI_1s',
                                     'var': var,
                                     'section': sec,
                                     'key': key}))

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

df_lw.to_csv(filepath, index=False)
df_lw = df_lw.reset_index(drop=True)

# %% get statistics
lw_stats = (df_lw
            .groupby(['key', 'label', 'var', 'section'])['values']
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
df_save.to_csv(f'{save_path}/HALO-AC3_terrestrial_irradiance_stats.csv',
               index=False)

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
    ax.xaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.89, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[1].set(
    xlabel=f'Downward terrestrial irradiance ({h.plot_units["flux_net_lw"]})'
)
figname = f'05_HALO_AC3_RF17_RF18_flux_dn_lw_BACARDI_ecRad_violin.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% print statistics
df_print = lw_stats.loc[pd.IndexSlice[:, ['BACARDI_org'] + sel_ver], :].sort_values('key')
print(df_print)

# %% plot f_up_lw BACARDI vs. ecRad - violinplot
sel_ver = ['BACARDI', 'v15.1']
var = ['flux_up_lw', 'F_up_terrestrial']
section = 'above'
xlims = dict(above=(150, 200), below=(200, 240), case=(125, 260))
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
           xlim=xlims[section],
           yticklabels=['BACARDI',
                        'ecRad Reference\nFu-IFS (v15.1)',
                        # 'ecRad Fu-IFS (v43.1)'
                        ])
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.89, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[1].set(
    xlabel=f'Upward terrestrial irradiance ({h.plot_units["flux_net_lw"]})'
)
figname = f'05_HALO_AC3_RF17_RF18_{var[0]}_BACARDI_ecRad_{section}_violin.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% print stats
df_print = df_lw[(df_lw['label'].isin(sel_ver))
                 & (df_lw['section'] == section)
                 & (df_lw['var'].isin(var))]
df_print = (df_print
            .groupby(['key', 'label', 'var', 'section'])['values']
            .describe()
            .sort_values(['key', 'mean'], ascending=[True, False]))
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
            f'Bias: {np.abs(bias):.0f} {h.plot_units['flux_up_lw']}',
            ha='left',
            va='top',
            transform=ax.transAxes,
            bbox=dict(fc='white', ec='black', alpha=0.8, boxstyle='round'),
        )

    figname = f'{plot_path}/05_HALO-AC3_RF17_RF18_bacardi_ecrad_f_up_lw_below_cloud_{v}.pdf'
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot BACARDI vs. ecRad terrestrial upward above cloud
plt.rc('font', size=10)
label = ['(a)', '(b)']
lims = (150, 200)
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
            xlim=lims,
            ylim=lims,
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

    figname = f'{plot_path}/05_HALO-AC3_RF17_RF18_bacardi_ecrad_f_up_lw_above_cloud_{v}.png'
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot BACARDI vs. ecRad terrestrial downward below cloud
plt.rc('font', size=10)
label = ['(a)', '(b)']
xlims = [(85, 110), (130, 180)]
# xlims = [(175, 200), (155, 180)]
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
        ecrad_plot = (ecrad_ds.flux_dn_lw
                      .isel(half_level=height_sel)
                      .sel(time=time_sel))

        # actual plotting
        rmse = np.sqrt(np.mean((bacardi_plot['F_down_terrestrial'] - ecrad_plot) ** 2)).to_numpy()
        bias = np.nanmean((bacardi_plot['F_down_terrestrial'] - ecrad_plot).to_numpy())
        ax.scatter(bacardi_plot['F_down_terrestrial'], ecrad_plot, color=cbc[3])
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
            0.95,
            0.05,
            f'{label[i]} {key.replace('1', ' 1')}\n'
            f'n= {sum(~np.isnan(bacardi_plot['F_down_terrestrial'])):.0f}\n'
            f'RMSE: {rmse:.0f} {h.plot_units['flux_up_lw']}\n'
            f'Bias: {bias:.0f} {h.plot_units['flux_up_lw']}',
            ha='right',
            va='bottom',
            transform=ax.transAxes,
            bbox=dict(fc='white', ec='black', alpha=0.8, boxstyle='round'),
        )

    figname = f'{plot_path}/05_HALO-AC3_RF17_RF18_bacardi_ecrad_f_dn_lw_below_cloud_{v}.png'
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
section = 'below'
var = ['flux_dn_lw', 'F_down_terrestrial']
h.set_cb_friendly_colors('petroff_6')
plt.rc('font', size=10)
_, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 7.5 * h.cm),
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
sel_ver = ['BACARDI', 'v15.1', 'v22.1']
section = 'below'
var = ['flux_dn_lw', 'F_down_terrestrial']
h.set_cb_friendly_colors('petroff_6')
plt.rc('font', size=10)
_, axs = plt.subplots(2, 1, figsize=(15 * h.cm, 10 * h.cm),
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
                        'ecRad 3D on\nFu-IFS (v22.1)',
                        'ecRad Reference\nBaran2016 (v18.1)',
                        'ecRad 3D on\nBaran2016 (v24.1)']
           )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.89, panel_label[i], transform=ax.transAxes)
    ax.grid()


axs[1].set(
    xlabel=f'Downward terrestrial irradiance ({h.plot_units["flux_net_lw"]})'
)
figname = f'07_HALO_AC3_RF17_RF18_flux_dn_lw_BACARDI_ecRad_3d_effects_violin.png'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% 3-D effect print statistics
df_print = lw_stats.loc[pd.IndexSlice[:, ['BACARDI_org'] + sel_ver], :].sort_values('key')
print(df_print)

# %% aerosol plot violinplot
sel_ver = ['BACARDI', 'v15.1', 'v30.1']
section = 'below'
var = ['flux_dn_lw', 'F_down_terrestrial']
h.set_cb_friendly_colors('petroff_6')
plt.rc('font', size=10)
_, axs = plt.subplots(2, 1, figsize=(15 * h.cm, 10 * h.cm),
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
           yticklabels=['BACARDI',
                        'ecRad Reference\nFu-IFS (v15.1)',
                        'ecRad aerosol on\nFu-IFS (v30.1)',
                        ],
           xlim=(75, 200),
           )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.89, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[1].set(
    xlabel=f'Downward terrestrial irradiance ({h.plot_units["flux_net_lw"]})'
)
figname = f'07_HALO_AC3_RF17_RF18_flux_dn_lw_BACARDI_ecRad_aerosol_violin.pdf'
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
section = 'below'
var = ['flux_dn_lw', 'F_down_terrestrial']
h.set_cb_friendly_colors('petroff_6')
plt.rc('font', size=10)
_, axs = plt.subplots(2, 1, figsize=(15 * h.cm, 11 * h.cm),
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
           yticklabels=['BACARDI',
                        'ecRad Reference\nFu-IFS (v15.1)',
                        'ecRad \nYi2013 (v19.1)',
                        'ecRad \nBaran2016 (v18.1)',
                        ],
           xlim=(75, 200),
           )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.89, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[1].set(
    xlabel=f'Downward terrestrial irradiance ({h.plot_units["flux_net_lw"]})'
)
figname = f'05_HALO_AC3_RF17_RF18_flux_dn_lw_BACARDI_ecRad_ice_optics_violin.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% ice optics - print stats
df_print = (lw_stats.loc[pd.IndexSlice[:, ['BACARDI_org'] + sel_ver, var, 'below'], :].sort_values('key'))
print(df_print)

# %% VarCloud - plot violinplot
sel_ver = ['BACARDI', 'v15.1', 'v36']
h.set_cb_friendly_colors('petroff_6')
plt.rc('font', size=10)
_, axs = plt.subplots(2, 1, figsize=(15 * h.cm, 10 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df_lw[(df_lw.key == key)
                 & (df_lw.label.isin(sel_ver))]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='values', y='label', hue='label',
                   ax=ax, order=sel_ver)
    ax.set(xlabel='',
           ylabel='',
           yticklabels=['BACARDI',
                        'ecRad Reference\nFu-IFS (v15.1)',
                        'ecRad VarCloud\nFu-IFS (v36)',
                        ],
           xlim=(75, 200),
           )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.89, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[1].set(xlabel=f'Downward terrestrial irradiance ({h.plot_units["flux_net_lw"]})')
figname = f'05_HALO_AC3_RF17_RF18_flux_dn_lw_BACARDI_ecRad_varcloud_violin.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% VarCloud - plot violinplot - all ice optics
sel_ver = ['BACARDI', 'v15.1', 'v36', 'v37', 'v38']
h.set_cb_friendly_colors('petroff_6')
plt.rc('font', size=10)
_, axs = plt.subplots(2, 1, figsize=(15 * h.cm, 11 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df_lw[(df_lw.key == key)
                 & (df_lw.label.isin(sel_ver))]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='values', y='label', hue='label',
                   ax=ax, order=sel_ver)
    ax.set(xlabel='',
           ylabel='',
           yticklabels=['BACARDI',
                        'ecRad Reference\nFu-IFS (v15.1)',
                        'ecRad VarCloud\nFu-IFS (v36)',
                        # 'ecRad Reference\nYi2013 (v19.1)',
                        'ecRad VarCloud\nYi2013 (v37)',
                        # 'ecRad Reference\nBaran2016 (v18.1)',
                        'ecRad VarCloud\nBaran2016 (v38)',
                        ],
           xlim=(75, 210),
           )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.89, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[1].set(
    xlabel=f'Downward terrestrial irradiance ({h.plot_units["flux_net_lw"]})',
)
figname = f'05_HALO_AC3_RF17_RF18_flux_dn_lw_BACARDI_ecRad_varcloud_violin_all.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()
# %% VarCloud - print stats
df_print = lw_stats.loc[pd.IndexSlice[:, ['BACARDI_org'] + sel_ver], :].sort_values('key')
print(df_print)
# %% re_ice latitude dependence - plot violinplot
sel_ver = ['BACARDI', 'v15.1', 'v39.2']
section = 'below'
var = ['flux_dn_lw', 'F_down_terrestrial']
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
           yticklabels=['BACARDI',
                        'ecRad Reference\nCosine (v15.1)',
                        'ecRad \nNo cosine (v39.2)',
                        ],
           xlim=(75, 200),
           )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.89, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[1].set(xlabel=f'Downward terrestrial irradiance ({h.plot_units["flux_net_lw"]})')
figname = f'05_HALO_AC3_RF17_RF18_flux_dn_lw_BACARDI_ecRad_re_ice_latitude_violin.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% re_ice latitude - print stats
df_print = lw_stats.loc[pd.IndexSlice[:, ['BACARDI_org'] + sel_ver], :].sort_values('key')
print(df_print)
# %% best sim - VarCloud 3D aerosol Yi2013 violinplot
sel_ver = ['BACARDI', 'v28', 'v44']
section = 'below'
var = ['flux_dn_lw', 'F_down_terrestrial']
h.set_cb_friendly_colors('petroff_6')
plt.rc('font', size=10)
_, axs = plt.subplots(2, 1, figsize=(15 * h.cm, 10 * h.cm),
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
                        'ecRad VarCloud\nYi2013 (v28)',
                        'ecRad 3D Aerosol on\nYi2013 (v44)',
                        ]
           )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.89, panel_label[i], transform=ax.transAxes)
    ax.grid()


axs[1].set(
    xlabel=f'Downward terrestrial irradiance ({h.plot_units["flux_net_lw"]})'
)
figname = f'07_HALO_AC3_RF17_RF18_flux_dn_lw_BACARDI_ecRad_varcloud_3d_aerosol_violin.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% detailed investigation of temperature profile and in cloud temperature
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
    ax.xaxis.set_major_locator(mticker.MultipleLocator(base=15))
    ax.plot([], color='grey', label='IFS profiles')
    ax.axhline(below_cloud_altitude[key], c='k')
    ax.set(
        ylim=(3.5, 8)
    )
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
        ylim=(3.5, 8),
        xlabel='Relative humidity \nover ice (%)',
    )
    ax.set_title(f'{key.replace('1', ' 1')} - {date_title[i]}', fontsize=10)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(base=25))
    ax.plot([], color='grey', label='IFS profiles')
    ax.axhline(below_cloud_altitude[key], c='k')
    ax.legend(fontsize=7)
    ax.grid()

axs[0].set_ylabel('Altitude (km)')
axs[0].text(0.02, 0.95, '(a)', transform=axs[0].transAxes)
axs[1].text(0.02, 0.95, '(b)', transform=axs[1].transAxes)
axs[2].text(0.02, 0.95, '(c)', transform=axs[2].transAxes)
axs[3].text(0.02, 0.95, '(d)', transform=axs[3].transAxes)

# figname = f'{plot_path}/05_HALO-AC3_HALO_RF17_RF18_ifs_dropsonde_t_rh.pdf'
# plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% get cloud base height from VarCloud data
height_timeseries = dict()
for key in keys:
    time_sel = slices[key]['case']
    ds_vc = varcloud_ds[key].sel(time=time_sel)
    mask = ds_vc.Varcloud_Input_Mask != 1
    # mask = np.isnan(ds_vc.iwc)
    props, bases_tops = h.find_bases_tops(np.flip(mask.to_numpy(), axis=1),
                                          np.flip(mask.height.to_numpy()))
    bases_tops = xr.DataArray(bases_tops,
                              dims=['time', 'height'],
                              coords=dict(time=mask.time, height=np.flip(mask.height.to_numpy())))
    # extract a time series of cloud base
    heights_series = []

    # Step 2: Loop through each time step and extract the corresponding heights where values are 1
    for t in bases_tops['time']:
        # Extract the values for this time step
        time_slice = bases_tops.sel(time=t)

        # Find the heights where the value is 1
        heights = bases_tops['height'].where(time_slice == -1, drop=True)

        # Append the height or NaN if no 1's found in that time slice
        if len(heights) > 0:
            heights_series.append(heights.values[0])  # Take the first occurrence if multiple
        else:
            heights_series.append(np.nan)  # No value found, append NaN

    # Step 3: Create a time series (DataArray) from the heights list
    height_timeseries[key] = xr.DataArray(heights_series, dims='time', coords={'time': bases_tops['time']})

# %% plot cloud base height VarCloud vs. IFS (plus ceiling)
h.set_cb_friendly_colors('petroff_6')
plt.rc('font', size=10)
fig, axs = plt.subplots(2, 1, layout='constrained',
                      figsize=(15 * h.cm, 11 * h.cm))
for i, key in enumerate(['RF17', 'RF18']):
    time_sel = slices[key]['above']
    ds = ecrad_dicts[key]['v15.1'].sel(time=time_sel)
    ax = axs[i]
    ((height_timeseries[key]
      .where((height_timeseries[key] > 2000) & (height_timeseries[key] < 7000)) / 1000)
     .plot(ls='', marker='x', label='VarCloud cloud base height', ax=ax))
    ((ds['ceil'] / 1000)
     .plot(ls='', marker='.', label='IFS ceiling altitude', ax=ax))
    ((ds['cbh'].where(ds['cbh'] > 2000) / 1000)
     .plot(ls='', marker='^', label='IFS cloud base height', ax=ax))
    ax.set(
        xlabel='',
        ylabel='',
        ylim=(2.5, 7.5),
        xlim=(slices[key]['above'].start, slices[key]['above'].stop)
    )
    h.set_xticks_and_xlabels(ax, pd.Timedelta(10, 'min'))
    ax.grid()
    ax.text(0.025, 0.91, f'{panel_label[i]} {key}', transform=ax.transAxes)

axs[1].legend(ncol=2, bbox_to_anchor=(0.5, -0.7), loc='lower center')
axs[1].set(
    xlabel='Time (UTC)'
)
fig.supylabel('                  Height (km)', size=10)
fig.savefig(f'{plot_path}/HALO-AC3_RF17_RF18_cbh_varcloud_IFS.pdf')
plt.show()
plt.close()

# %% extract temperature at cloud base for IFS and VarCloud
key = 'RF17'
cbt_ifs, cbt_vc = dict(), dict()
for key in keys:
    time_sel = slices[key]['above']
    ds = ecrad_dicts[key]['v15.1'].sel(time=time_sel)
    hl_sel = np.abs(ds.press_height_hl - ds.ceil).argmin(dim='half_level')
    cbt_ifs[key] = ds.temperature_hl.isel(half_level=hl_sel) - 273.15
    hl_sel_vc = (np.abs(
        ds.press_height_hl -
        height_timeseries[key]
        .interp(time=ds.time))
                 .dropna(dim='time')
                 .argmin(dim='half_level'))
    cbt_vc[key] = (ds['temperature_hl']
                   .sel(time=hl_sel_vc.time)
                   .isel(half_level=hl_sel_vc)) - 273.15

# %% plot cloud base temperature from IFS cloud base height and VarCloud cloud base height
h.set_cb_friendly_colors('petroff_6')
fig, axs = plt.subplots(2,
                      layout='constrained',
                      figsize=(15 * h.cm, 9 * h.cm))
for i, key in enumerate(keys):
    ax = axs[i]
    cbt_ifs[key].plot(x='time', label='IFS', ax=ax)
    cbt_vc[key].plot(x='time', label='VarCloud', ax=ax)
    ax.set(
        xlabel='',
        ylabel='',
        ylim=(-50, -20)
    )
    h.set_xticks_and_xlabels(ax, pd.Timedelta(15, 'min'))
    ax.grid()
    ax.text(0.02, 0.9, f'{panel_label[i]} {key}', transform=ax.transAxes)

axs[0].legend(loc=9)
axs[1].set(
    xlabel='Time (UTC)',
)
fig.supylabel('Cloud base temperature (°C)', size=10)
fig.savefig(f'{plot_path}/05_HAlO-AC3_RF17_RF18_cbt_varcloud_IFS.pdf')
plt.show()
plt.close()

# %% plot optical depth lw from the radiative properties file
key = 'RF17'
ds = rad_props[key]['v15.1']
var = 'od_lw_cloud'
dim = 'band_lw' if var == 'od_lw_cloud' else 'gpoint_lw'
_, axs = plt.subplots(3, layout='constrained')
# 2D time height plot of optical depth
ax = axs[0]
((ds[var]
 .sel(column=0,
      time=slices[key]['case'],
      # level=slice(70, 140)
      )
 .sum(dim=dim))
 .plot(x='time', robust=True, ax=ax,
       cbar_kwargs=dict(label='LW optical thickness'))
 )
ax.set(
    xlabel='',
    ylabel='Model Level',
    ylim=(110, 70)
)
# integrated time series of optical depth
ax = axs[1]
(ds[var]
 .sel(column=0,
      time=slices[key]['case'],
      # level=slice(70, 140)
      )
 .sum(dim=[dim, 'level'])
 .plot(x='time', ax=ax))
ax.set(
    xlabel='',
    ylabel='Integrated \noptical thickness',
)
ax.grid()
# integrated emissivity
ax = axs[2]
em = 1 - np.exp(-1.66 * ds[var].sum(dim=dim))
em = em.where(em > 0)
em_plot = (em.sel(column=0,
                  time=slices[key]['case'],
                  # level=slice(70, 140)
                  )
           )
(em_plot
 .where(em_plot > 0)
 .plot(x='time', ax=ax, vmax=1, vmin=0,
       cbar_kwargs=dict(label='Emissivity')))
ax.set(
    xlabel='Time (UTC)',
    ylabel='Model Level',
    ylim=(110, 70)
)
ax.grid()
plt.savefig(f'{plot_path}/HALO-AC3_{key}_{var}_3rows.png', dpi=300)
plt.show()
plt.close()

# %% plot optical depth sw from the radiative properties file
key = 'RF17'
ds = rad_props[key]['v15.1']
var = 'od_sw_cloud'
_, axs = plt.subplots(2, layout='constrained')
# 2D time height plot of optical depth
ax = axs[0]
(ds[var]
 .where(ds[var] > 0)
 .sel(column=0,
      time=slices[key]['case'],
      # level=slice(70, 140)
      )
 .sum(dim='band_sw')
 .plot(x='time', robust=True, ax=ax,
       cbar_kwargs=dict(label='SW optical thickness'))
 )
ax.set(
    xlabel='',
    ylabel='Model Level',
    ylim=(138, 0)
)
# integrated time series of optical depth
ax = axs[1]
(ds[var]
 .sel(column=0,
      time=slices[key]['case'],
      # level=slice(70, 140)
      )
 .sum(dim=['band_sw', 'level'])
 .plot(x='time', ax=ax))
ax.set(
    xlabel='',
    ylabel='Integrated \noptical thickness',
)
ax.grid()

plt.savefig(f'{plot_path}/HALO-AC3_RF17_od_sw_cloud_2rows.png', dpi=300)
plt.show()
plt.close()

# %% calculate approximated absorption coefficient and optical depth
od_ifs, od_varcloud = dict(), dict()
for key in keys:
    # IFS
    ds = ecrad_dicts[key]['v15.1'].sel(time=slices[key]['case'])
    b_abs = calculate_absorption_coefficient_terrestrial(ds.iwc, ds.re_ice)
    new_z_coord = ds.press_height_full.mean(dim='time')
    new_z_coord = new_z_coord.to_numpy()
    b_abs = b_abs.assign_coords(level=new_z_coord)
    b_abs = b_abs.sortby('level')  # sort vertical coordinate in ascending order
    level_height = np.diff(new_z_coord)
    od_ifs[key] = b_abs[:, 1:] * -level_height
    # VarCloud
    ds = varcloud_ds[key].sel(time=slices[key]['case'])
    b_abs = calculate_absorption_coefficient_terrestrial(ds.iwc, ds.re_ice)
    b_abs = b_abs.sortby('height')  # sort vertical coordinate in ascending order
    od_varcloud[key] = b_abs * b_abs.height.diff(dim='height')

# %% calculate ice optics from fu lw
od_lw, od_lw_vc, od_scat_lw, od_scat_lw_vc = dict(), dict(), dict(), dict()
for key in keys:
    ds = ecrad_dicts[key]['v15.1'].sel(time=slices[key]['case'])
    od_lw[key], od_scat_lw[key], _ = ecrad.calc_ice_optics_fu_lw(ds.iwp, ds.re_ice)
    ds_vc = varcloud_ds[key]
    da = np.abs(ds_vc.height.diff(dim="height"))
    varcloud_ds[key]['iwp'] = ds_vc.iwc * da
    iwp = varcloud_ds[key]['iwp'].sel(time=slices[key]['case'])
    od_lw_vc[key], od_scat_lw_vc[key] = dict(), dict()
    od_lw_vc[key]['v15.1'], od_scat_lw_vc[key]['v15.1'], _ = ecrad.calc_ice_optics_fu_lw(iwp, ds_vc.re_ice.isel(height=slice(1, 154)))
    od_lw_vc[key]['v19.1'], od_scat_lw_vc[key]['v19.1'], _ = ecrad.calc_ice_optics_yi(
        'lw', iwp, ds_vc.re_ice.isel(height=slice(1, 154))
    )

# %% plot calculated emissivity from IFS and VarCloud
key = 'RF18'
_, axs = plt.subplots(2, layout='constrained')
# IFS
em = 1 - np.exp(-od_ifs[key])
ax = axs[0]
em.sel(time=slices[key]['above']).plot(x='time', ax=ax, vmin=0, vmax=0.2)
# VarCloud
em = 1 - np.exp(-od_varcloud[key])
ax = axs[1]
em.sel(time=slices[key]['above']).plot(x='time', ax=ax, vmin=0, vmax=0.2)
for ax in axs:
    ax.set(
        ylim=(3000, 8500)
    )
plt.savefig(f'{plot_path}/HALO-AC3_{key}_emissivity_IFS_VarCloud.png')
plt.show()
plt.close()

# %% plot difference of calculated optical depth from IFS and ecRad radiative properties optical depth
key = 'RF17'
_, axs = plt.subplots(2, layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    calc_od = od_ifs[key]
    ecrad_od = (rad_props[key]['v15.1']['od_lw_cloud']
                .sel(time=slices[key]['case'],
                     column=0,
                     level=slice(1, 138))
                .sum(dim='band_lw') / 16
                )
    ecrad_od = (ecrad_od
    .where(ecrad_od > 0)
    .assign_coords(
        level=(ecrad_dicts[key]['v15.1']['press_height_full']
               .sel(level=slice(2, 138),
                    time=slices[key]['case'])
               .mean(dim='time'))))
    diff = calc_od - ecrad_od

    diff.plot(x='time', ax=ax, vmax=0.55)
    ax.set(
        ylim=(3000, 8500)
    )
plt.savefig(f'{plot_path}/HALO-AC3_RF17_RF18_od_IFS_vs_ecRad.png')
plt.show()
plt.close()

# %% plot calculated optical depth from IFS with ecRad radiative properties optical depth
key = 'RF17'
vmax = 0.25 if key == 'RF17' else 0.3
calc_od = od_ifs[key]
ecrad_od = (rad_props[key]['v15.1']['od_lw_cloud']
            .sel(time=slices[key]['case'],
                 column=0,
                 level=slice(1, 138))
            .mean(dim='band_lw')
            )
ecrad_od = (ecrad_od
.where(ecrad_od > 0)
.assign_coords(
    level=(ecrad_dicts[key]['v15.1']['press_height_full']
           .sel(level=slice(2, 138)))))
_, axs = plt.subplots(2, layout='constrained')
calc_od.plot(x='time', ax=axs[0], vmax=vmax)
ecrad_od.plot(x='time', ax=axs[1], vmax=vmax)
for ax in axs:
    ax.set(
        ylim=(3000, 8500)
    )
plt.savefig(f'{plot_path}/HALO-AC3_{key}_od_lw_cloud_IFS_vs_ecRad.png')
plt.show()
plt.close()

# %% plot emissivity according to ice optics parameterization od_lw
for band in range(16):
    _, axs = plt.subplots(2, layout='constrained')
    for i, key in enumerate(keys):
        ax = axs[i]
        ds = ecrad_dicts[key]['v15.1'].sel(time=slices[key]['case'])
        od_lw, od_scat_lw, g_lw = ecrad.calc_ice_optics_fu_lw(ds.iwp, ds.re_ice)
        em = 1 - np.exp(-1.66 * od_lw.sel(band_lw=band))
        em.plot(x='time', ax=ax, vmin=0, vmax=1)
        ax.set(
            title=f'LW Band {band}',
            ylim=(105, 80)
        )
    plt.savefig(f'{plot_path}/HALO-AC3_RF17_RF18_od_lw_fu_band_{band}.png')
    # plt.show()
    plt.close()

# %% plot sum of emissivity according to ice optics parameterization od_lw
_, axs = plt.subplots(2, layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    em = 1 - np.exp(-1.66 * od_lw[key].sum(dim='band_lw') / 16)
    em = em.where(em > 0)
    em.plot(x='time', ax=ax, vmin=0, vmax=1)
    ax.set(
        ylim=(105, 80)
    )
plt.savefig(f'{plot_path}/HALO-AC3_RF17_RF18_od_lw_fu_sum.png')
plt.show()
plt.close()

# %% plot difference of od_lw from fu and from radiative properties file
_, axs = plt.subplots(2, layout='constrained')
for i, key in enumerate(keys):
    od_lw_fu = od_lw[key]
    od_lw_scat_fu = od_scat_lw[key]
    od_lw_fu = od_lw_fu - od_lw_scat_fu
    ecrad_od = (rad_props[key]['v15.1']['od_lw_cloud']
                .sel(time=slices[key]['case'],
                     column=0,
                     )
                .assign_coords(level=range(1, 138))
                )
    diff = (od_lw_fu - ecrad_od).mean(dim='band_lw')
    ax = axs[i]
    diff.plot(x='time', ax=ax, vmin=0, vmax=0.5)
    ax.set(
        ylim=(105, 80)
    )
plt.savefig(f'{plot_path}/HALO-AC3_RF17_RF18_mean_od_lw_diff_fu_ecrad.png')
plt.show()
plt.close()

# %% plot od_lw calculated via fu from VarCloud data
key = 'RF17'
plot_vc = od_lw_vc[key] - od_scat_lw_vc[key]
plot_ec = od_lw[key] - od_scat_lw[key]
_, axs = plt.subplots(2, layout='constrained')
plot_vc.sum(dim='band_lw').where(od_lw_vc[key].sel(band_lw=0) > 0).plot(x='time', robust=True, ax=axs[0])
plot_ec.sum(dim='band_lw').where(od_lw[key].sel(band_lw=0) > 0).plot(x='time', robust=True, ax=axs[1])
plt.show()
plt.close()

# %% plot emissivity from Fu-IFS using Varcloud and from radiative properties file
plt.rc('font', size=9)
v = 'v15.1'
layout=(
    '''
    ABc
    DEc
    '''
)
fig, axs = plt.subplot_mosaic(layout,
    layout='constrained',
    figsize=(17 * h.cm, 10 * h.cm),
    width_ratios=[1, 1, 0.05])
for i, key in enumerate(keys):
    time_sel = slices[key]['above']
    new_z = (ecrad_dicts[key][v]['press_height_full']
             .sel(time=time_sel)
             .mean(dim='time') / 1e3
             )
    od_ecrad = (rad_props[key][v]['od_lw_cloud']
                .sel(time=time_sel,
                     column=0)
                .sum(dim='band_lw')
                .assign_coords(level=new_z)
                )
    od_vc = (od_lw_vc[key][v] - od_scat_lw_vc[key][v]).sel(time=time_sel).sum(dim='band_lw')
    em_ecrad = 1 - np.exp(-1.66 * od_ecrad)
    em_vc = 1 - np.exp(-1.66 * od_vc)
    em_vc = em_vc.assign_coords(height=em_vc.height / 1e3)
    # plot emissivity ecrad (first row)
    ax = axs['A' if i == 0 else 'B']
    im = (em_ecrad
          .where(em_ecrad > 0)
          .plot(x='time', ax=ax, vmin=0, vmax=1, add_colorbar=False, cmap='cividis')
          )
    h.set_xticks_and_xlabels(ax, pd.Timedelta(10, 'min'))
    # plot emissivity VarCloud (second row)
    ax = axs['D' if i == 0 else 'E']
    (em_vc
     .where(em_vc > 0)
     .plot(x='time', ax=ax, vmin=0, vmax=1, add_colorbar=False, cmap='cividis')
     )
    h.set_xticks_and_xlabels(ax, pd.Timedelta(10, 'min'))

for k, label in zip(['A', 'B', 'D', 'E'], ['(a)', '(b)', '(c)', '(d)']):
    axs[k].set(
        xlabel='',
        ylabel='',
        ylim=(0, 10)
    )
    axs[k].text(0.01, 0.91, label, transform=axs[k].transAxes)

axs['A'].set(ylabel='Altitude (km)')
axs['A'].set_title('RF 17 - 11 April 2024', size=9)
axs['B'].set_title('RF 18 - 12 April 2024', size=9)
axs['D'].set(xlabel='Time (UTC)', ylabel='Altitude (km)')
axs['E'].set(xlabel='Time (UTC)')


# Create a colorbar for the whole figure on the right side, spanning both rows
fig.colorbar(im, cax=axs['c'], label='Broadband emissivity $\\epsilon$')
plt.savefig(f'{plot_path}/05_HALO-AC3_RF17_RF18_emissivity_ecrad_{v}_varcloud.png', dpi=300)
plt.show()
plt.close()

# %% plot emissivity from radiative properties files v15.1 and v16
plt.rc('font', size=9)
layout=(
    '''
    ABc
    DEc
    '''
)
fig, axs = plt.subplot_mosaic(layout,
    layout='constrained',
    figsize=(17 * h.cm, 10 * h.cm),
    width_ratios=[1, 1, 0.05])
ks = np.array([['A', 'B'], ['D', 'E']])
vers = ['v15.1', 'v44']
for i, key in enumerate(keys):
    time_sel = slices[key]['below']
    for j, v in enumerate(vers):
        k = ks[j, i]
        new_z = (ecrad_dicts[key][v]['press_height_full']
                 .sel(time=time_sel)
                 .mean(dim='time') / 1e3
                 )
        if 'column' in rad_props[key][v].dims:
            od_ecrad = (rad_props[key][v]['od_lw_cloud']
                        .sel(time=time_sel,
                             column=0)
                        .sum(dim='band_lw')
                        .assign_coords(level=new_z)
                        )
        else:
            od_ecrad = (rad_props[key][v]['od_lw_cloud']
                        .sel(time=time_sel)
                        .sum(dim='band_lw')
                        .assign_coords(level=new_z)
                        )
        em_ecrad = 1 - np.exp(-1.66 * od_ecrad)

        # plot emissivity ecrad v15.1 (first row)
        ax = axs[k]
        im = (em_ecrad
              .where(em_ecrad > 0)
              .plot(x='time', ax=ax, vmin=0, vmax=1, add_colorbar=False, cmap='cividis')
              )
        h.set_xticks_and_xlabels(ax, pd.Timedelta(15, 'min'))


for k, label in zip(['A', 'B', 'D', 'E'], ['(a)', '(b)', '(c)', '(d)']):
    axs[k].set(
        title='',
        xlabel='',
        ylabel='',
        ylim=(0, 10)
    )
    axs[k].text(0.01, 0.91, label, transform=axs[k].transAxes)

axs['A'].set(ylabel='Altitude (km)')
axs['A'].set_title('RF 17 - 11 April 2024', size=9)
axs['B'].set_title('RF 18 - 12 April 2024', size=9)
axs['D'].set(xlabel='Time (UTC)', ylabel='Altitude (km)')
axs['E'].set(xlabel='Time (UTC)')


# Create a colorbar for the whole figure on the right side, spanning both rows
fig.colorbar(im, cax=axs['c'], label='Broadband Emissivity $\\epsilon$')
plt.savefig(f'{plot_path}/HALO-AC3_RF17_RF18_emissivity_ecrad_{vers[0]}_{vers[1]}.png', dpi=300)
plt.show()
plt.close()

# %% plot emissivity from Fu-IFS using height interpolated Varcloud data and from radiative properties file
plt.rc('font', size=9)
layout=(
    '''
    ABc
    DEc
    '''
)
fig, axs = plt.subplot_mosaic(layout,
    layout='constrained',
    figsize=(17 * h.cm, 10 * h.cm),
    width_ratios=[1, 1, 0.05])
for i, key in enumerate(keys):
    time_sel = slices[key]['above']
    new_z = (ecrad_dicts[key]['v15.1']['press_height_full']
             .sel(time=time_sel)
             .mean(dim='time') / 1e3
             )
    od_ecrad = (rad_props[key]['v15.1']['od_lw_cloud']
                .sel(time=time_sel,
                     column=0)
                .sum(dim='band_lw')
                .assign_coords(level=new_z)
                )
    od_vc = ((od_lw_vc[key] - od_scat_lw_vc[key])
             .sel(time=time_sel)
             .sum(dim='band_lw')
             .assign_coords(height=od_lw_vc[key].height / 1e3))
    # od_vc = od_vc.interp(height=new_z.to_numpy())
    em_ecrad = 1 - np.exp(-1.66 * od_ecrad)
    em_vc = 1 - np.exp(-1.66 * od_vc)
    em_vc = em_vc
    # plot emissivity ecrad (first row)
    ax = axs['A' if i == 0 else 'B']
    im = (em_ecrad
          .where(em_ecrad > 0)
          .plot(x='time', ax=ax, vmin=0, vmax=1, add_colorbar=False, cmap='cividis')
          )
    h.set_xticks_and_xlabels(ax, pd.Timedelta(10, 'min'))
    # plot emissivity VarCloud (second row)
    ax = axs['D' if i == 0 else 'E']
    (em_vc
     .where(em_vc > 0)
     .plot(x='time', ax=ax, vmin=0, vmax=1, add_colorbar=False, cmap='cividis')
     )
    h.set_xticks_and_xlabels(ax, pd.Timedelta(10, 'min'))

for k, label in zip(['A', 'B', 'D', 'E'], ['(a)', '(b)', '(c)', '(d)']):
    axs[k].set(
        xlabel='',
        ylabel='',
        ylim=(0, 10)
    )
    axs[k].text(0.01, 0.91, label, transform=axs[k].transAxes)

axs['A'].set(ylabel='Altitude (km)')
axs['A'].set_title('RF 17 - 11 April 2024', size=9)
axs['B'].set_title('RF 18 - 12 April 2024', size=9)
axs['D'].set(xlabel='Time (UTC)', ylabel='Altitude (km)')
axs['E'].set(xlabel='Time (UTC)')


# Create a colorbar for the whole figure on the right side, spanning both rows
fig.colorbar(im, cax=axs['c'], label='Broadband Emissivity $\\epsilon$')
plt.savefig(f'{plot_path}/HALO-AC3_RF17_RF18_emissivity_ecrad_varcloud_inp.png', dpi=300)
plt.show()
plt.close()

# %% plot difference of flux_dn_lw in profile at dropsonde locations
plt.rc('font', size=10)
ds_plot = list()
var = 'flux_dn_lw'
ds_times = list()
for key in keys:
    ts = dropsonde_ds[key].launch_time.sortby('launch_time').to_numpy()
    ts = ts[2:4] if key == 'RF17' else ts[11:]
    ds_times.append(ts)

ds_times = np.concatenate(ds_times)

fig = plt.figure(figsize=(15 * h.cm, 12 * h.cm), layout='constrained')
subfigs = fig.subfigures(2, 1)
axs0 = subfigs[0].subplots(1, 3)
axs1 = subfigs[1].subplots(1, 4)

for (i, t), label in zip(enumerate(ds_times[:3]), ['(a)', '(b)', '(c)']):
    key = 'RF17' if pd.to_datetime(t).date() == pd.Timestamp('2022-04-11').date() else 'RF18'
    ax = axs0[i]
    ds_new = ecrad_dicts[key]['v43.1'].sel(time=t, method='nearest')
    ds_old = ecrad_dicts[key]['v15.1'].sel(time=t, method='nearest')
    ds_plot = ds_new - ds_old
    (ds_plot[var]
     .assign_coords(half_level=ds_new.press_height_hl / 1e3)
     .plot(y='half_level', ax=ax))
    ax.axhline((ds_new['press_height_hl'] / 1e3)
               .isel(half_level=98 if key == 'RF17' else 103),
               color='k',
               label='Below cloud\nflight altitude'
               )
    ax.text(0.1, 0.9, label, transform=ax.transAxes)
    ax.grid()
    ax.set(
        xlabel='',
        ylabel='',
        xlim=(-7, 2),
        ylim=(0, 10)
    )
    ax.set_title(f'{pd.to_datetime(t):%Y-%m-%d\n %H:%M} UTC', size=10)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(base=2))

# second row
for (i, t), label in zip(enumerate(ds_times[3:]), ['(d)', '(e)', '(f)', '(g)']):
    key = 'RF18'
    ax = axs1[i]
    ds_new = ecrad_dicts[key]['v43.1'].sel(time=t, method='nearest')
    ds_old = ecrad_dicts[key]['v15.1'].sel(time=t, method='nearest')
    ds_plot = ds_new - ds_old
    (ds_plot[var]
     .assign_coords(half_level=ds_new.press_height_hl / 1e3)
     .plot(y='half_level', ax=ax))
    ax.axhline((ds_new['press_height_hl'] / 1e3)
               .isel(half_level=103),
               color='k',
               label='Below cloud\nflight altitude'
               )
    ax.text(0.1, 0.9, label, transform=ax.transAxes)
    ax.grid()
    ax.set(
        xlabel='',
        ylabel='',
        xlim=(-7, 2),
        ylim=(0, 10)
    )
    ax.set_title(f'{pd.to_datetime(t):%Y-%m-%d\n %H:%M} UTC', size=10)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(base=2))

axs0[0].set(
    ylabel='Height (km)'
)
axs1[0].set(
    ylabel='Height (km)'
)
fig.supxlabel(f'Difference in downward terrestrial irradiance ({h.plot_units[var]})',
              size=10)
figname = f'05_HALO-AC3_RF17_RF18_flux_dn_lw_IFS_vs_dropsonde.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% select downward terrestrial irradiance at aircraft level
h.set_cb_friendly_colors('petroff_8')
ds = ecrad_dicts[key]['v15.1'].sel(time=slices[key]['case'])
for i in range(10):
    ds.flux_dn_lw.isel(half_level=ds.aircraft_level + i).plot(label=i)
plt.legend()
plt.show()
plt.close()
# %% plot iwp from varcloud and IFS
key = 'RF17'
time_sel = slices[key]['above']
_, axs = plt.subplots(2, layout='constrained')
plot_vc = varcloud_ds[key]['iwp']
plot_vc = (plot_vc
           # .where(~np.isnan(plot_vc), 0)
           .sel(time=time_sel)
           # .integrate(coord='height') * -1
           ) * 1e3
plot_ec = ecrad_dicts[key]['v15.1']['iwp'].sel(time=time_sel)*1e3
plot_ec = plot_ec.where(plot_ec != np.inf, 0) #  .sum(dim='level')

plot_vc.plot(x='time', ax=axs[0], vmax=10)
plot_ec.plot(x='time', ax=axs[1], vmax=10)
for ax in axs:
    ax.set(
        # ylim=(0, 0.17)
    )
    ax.grid()

plt.show()
plt.close()
# %% plot height resolution
ec_ds = ecrad_dicts[key]['v15.1'].press_height_full.isel(time=300)
(ec_ds.assign_coords(level=ec_ds.to_numpy()).diff(dim='level') * -1).plot(marker='.', ls='')
(varcloud_ds[key].height.diff(dim='height') * -1).plot()
plt.ylim(0, 400)
plt.xlim(0, 1e4)
plt.show()
plt.close()
