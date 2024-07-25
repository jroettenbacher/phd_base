#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 30.05.2024

Here the figures from Chapter 5 (Results) Section 5.4 (Cloud radiative effect) of my thesis are created.



"""
# %% import modules
import os

import dill
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from skimage import io
from sklearn.neighbors import BallTree

import pylim.halo_ac3 as meta
import pylim.helpers as h
from pylim import ecrad

h.set_cb_friendly_colors('petroff_6')
cbc = h.get_cb_friendly_colors('petroff_6')

# %% set paths
campaign = 'halo-ac3'
save_path = 'C:/Users/Johannes/Documents/Doktor/manuscripts/_thesis/data'
plot_path = 'C:/Users/Johannes/Documents/Doktor/manuscripts/_thesis/figure'
keys = ['RF17', 'RF18']
ecrad_versions = [f'v{x}' for x in [13, 13.1, 13.2, 15, 15.1, 18.1, 19.1,
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
    bacardi_file = f'HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1_JR_v2.nc'
    ifs_file = f'ifs_{date}_00_ml_O1280_processed_sel_JR.nc'
    varcloud_file = [f for f in os.listdir(varcloud_path) if f.endswith('JR.nc')][0]
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
    bacardi_ds_res[key] = bacardi_res.copy(deep=True)
    bacardi_res_1s = xr.open_dataset(f'{bacardi_path}/{bacardi_file.replace('_v2.nc', '_1s_v2.nc')}')
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
    ifs_ds[key] = ifs.copy(deep=True)

    # read in varcloud data
    varcloud = xr.open_dataset(f'{varcloud_path}/{varcloud_file}')
    varcloud_ds[key] = varcloud

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
ecrad_vars = ['cre_sw', 'cre_lw', 'cre_total']
label = 'cre'
bacardi_vars = ['CRE_solar', 'CRE_terrestrial', 'CRE_total']
filepath = f'{save_path}/halo-ac3_{label}_boxplot_data.csv'
df = pd.DataFrame()
for key in keys:
    time_sel = slices[key]['below']
    dfs = list()
    dfs.append(df)
    for v in ecrad_versions:
        height_sel = ecrad_dicts[key][v].aircraft_level
        for i, ecrad_var in enumerate(ecrad_vars):
            dfs.append(pd.DataFrame({'values': (ecrad_orgs[key][v][ecrad_var]
                                                .isel(half_level=height_sel)
                                                .sel(time=time_sel)
                                                .to_numpy()
                                                .flatten()),
                                     'label': v,
                                     'key': key,
                                     'variable': bacardi_vars[i]}))
    for bacardi_var in bacardi_vars:
        dfs.append(pd.DataFrame({'values': (bacardi_ds_res_1s[key][bacardi_var]
                                            .sel(time=slices[key]['below'])
                                            .dropna('time')
                                            .to_pandas()
                                            .reset_index(drop=True)),
                                 'label': 'BACARDI_1s',
                                 'key': key,
                                 'variable': bacardi_var}))

        dfs.append(pd.DataFrame({'values': (bacardi_ds_res[key][bacardi_var]
                                            .sel(time=slices[key]['below'])
                                            .dropna('time')
                                            .to_pandas()
                                            .reset_index(drop=True)),
                                 'label': 'BACARDI_1Min',
                                 'key': key,
                                 'variable': bacardi_var}))

        # add variable to original BACARDI data
        dfs.append(pd.DataFrame({'values': (bacardi_ds[key][bacardi_var]
                                            .sel(time=slices[key]['below'])
                                            .dropna('time')
                                            .to_pandas()
                                            .reset_index(drop=True)),
                                 'label': 'BACARDI_org',
                                 'key': key,
                                 'variable': bacardi_var}))

    df = pd.concat(dfs)

df.to_csv(filepath, index=False)
df = df.reset_index(drop=True)

# %% calculate statistics and save data frame to csv
cre_stats = (df
             .groupby(['key', 'label', 'variable'])['values']
             .describe()
             .sort_values(['key', 'label'], ascending=[True, True]))
versions = [v for v in cre_stats.index.get_level_values('label') if v.startswith('v')]
df_save = cre_stats.reset_index()
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
df_save.to_csv(f'{save_path}/HALO-AC3_cre_stats.csv',
               index=False)

# %% cre - plot BACARDI/libRadtran net radiative effect
plt.rc('font', size=10)
ylims = (-50, 70)
_, axs = plt.subplots(2, 1, figsize=(15 * h.cm, 9 * h.cm), layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    plot_ds = bacardi_ds[key].sel(time=slices[key]['case'])
    time_extend = pd.to_timedelta((plot_ds.time[-1] - plot_ds.time[0]).to_numpy())
    ax.plot(plot_ds.time, plot_ds['CRE_solar'], label=h.bacardi_labels['CRE_solar'])
    ax.plot(plot_ds.time, plot_ds['CRE_terrestrial'], label=h.bacardi_labels['CRE_terrestrial'])
    ax.plot(plot_ds.time, plot_ds['CRE_total'], label=h.bacardi_labels['CRE_total'])
    ax.axhline(y=0, color='k')
    h.set_xticks_and_xlabels(ax, time_extend)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.grid()
    ax.set(ylabel=f'Cloud radiative\neffect ({h.plot_units['cre_sw']})',
           ylim=ylims)

axs[0].text(0.03, 0.88, '(a)', transform=axs[0].transAxes)
axs[1].text(0.03, 0.88, '(b)', transform=axs[1].transAxes)
axs[1].set_xlabel('Time (UTC)')
axs[0].legend(loc=1, ncols=3)

figname = f'{plot_path}/05_HALO-AC3_RF17_RF18_BACARDI_libRadtran_CRE.pdf'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% cre - plot BACARDI/ecRad net radiative effect
plt.rc('font', size=10)
ylims = (-200, 50)
_, axs = plt.subplots(2, 1, figsize=(15 * h.cm, 9 * h.cm), layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    plot_ds = bacardi_ds[key].sel(time=slices[key]['case'])
    time_extend = pd.to_timedelta((plot_ds.time[-1] - plot_ds.time[0]).to_numpy())
    ax.plot(plot_ds.time, plot_ds['CRE_solar_ecrad'], label=h.bacardi_labels['CRE_solar'])
    ax.plot(plot_ds.time, plot_ds['CRE_terrestrial_ecrad'], label=h.bacardi_labels['CRE_terrestrial'])
    ax.plot(plot_ds.time, plot_ds['CRE_total_ecrad'], label=h.bacardi_labels['CRE_total'])
    ax.axhline(y=0, color='k')
    h.set_xticks_and_xlabels(ax, time_extend)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.grid()
    ax.set(ylabel=f'Cloud radiative\neffect ({h.plot_units['cre_sw']})',
           ylim=ylims)

axs[0].text(0.03, 0.88, '(a)', transform=axs[0].transAxes)
axs[1].text(0.03, 0.88, '(b)', transform=axs[1].transAxes)
axs[1].set_xlabel('Time (UTC)')
axs[0].legend(loc=1, ncols=3)

figname = f'{plot_path}/05_HALO-AC3_RF17_RF18_BACARDI_ecrad_CRE.pdf'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% cre - plot ecRad net radiative effect
v = 'v15.1'
plt.rc('font', size=10)
ylims = (-50, 70)
_, axs = plt.subplots(2, 1, figsize=(15 * h.cm, 9 * h.cm), layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    ecrad_ds = ecrad_dicts[key][v]
    height_sel = ecrad_dicts[key][v].aircraft_level
    time_sel = slices[key]['case']
    plot_ds = (ecrad_ds
               .isel(half_level=height_sel)
               .sel(time=time_sel))
    time_extend = pd.to_timedelta((plot_ds.time[-1] - plot_ds.time[0]).to_numpy())
    ax.plot(plot_ds.time, plot_ds['cre_sw'], label=h.bacardi_labels['CRE_solar'])
    ax.plot(plot_ds.time, plot_ds['cre_lw'], label=h.bacardi_labels['CRE_terrestrial'])
    ax.plot(plot_ds.time, plot_ds['cre_total'], label=h.bacardi_labels['CRE_total'])
    ax.axhline(y=0, color='k')
    h.set_xticks_and_xlabels(ax, time_extend)
    ax.grid()
    ax.set(ylabel=f'Cloud radiative\neffect ({h.plot_units['cre_sw']})',
           ylim=ylims)

axs[0].text(0.03, 0.88, '(a)', transform=axs[0].transAxes)
axs[1].text(0.03, 0.88, '(b)', transform=axs[1].transAxes)
axs[1].set_xlabel('Time (UTC)')
axs[0].legend(loc=1, ncols=3)

figname = f'{plot_path}/05_HALO-AC3_RF17_RF18_ecrad_CRE.pdf'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot violinplot of cre_total
sel_ver = ['BACARDI_1Min', 'v15.1']
h.set_cb_friendly_colors('petroff_6')
plt.rc('font', size=10)
_, axs = plt.subplots(2, 1, figsize=(15 * h.cm, 10 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df[(df.key == key)
                 & (df.variable == 'CRE_total')
                 & (df.label.isin(sel_ver))]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='values', y='label', hue='label',
                   ax=ax, order=sel_ver)
    ax.set(xlabel='',
           ylabel='',
           yticklabels=['BACARDI',
                        # 'BACARDI 1s',
                        # 'BACARDI 1Min',
                        'ecRad Reference\nFu-IFS (v15.1)',
                        ],
           xlim=(-50, 70),
           )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.89, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[1].set(xlabel=f'Total cloud radiative effect ({h.plot_units['cre_total']})')
figname = f'05_HALO_AC3_RF17_RF18_cre_total_BACARDI_ecRad_violin.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% plot violinplot of cre_solar
sel_ver = ['BACARDI_org', 'BACARDI_1s', 'BACARDI_1Min', 'v15.1']
h.set_cb_friendly_colors('petroff_6')
plt.rc('font', size=10)
_, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 7.5 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df[(df.key == key)
                 & (df.variable == 'CRE_solar')
                 & (df.label.isin(sel_ver))]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='values', y='label', hue='label',
                   ax=ax, order=sel_ver)
    ax.set(xlabel=f'Solar cloud radiative\neffect ({h.plot_units['cre_total']})',
           ylabel='',
           yticklabels='',
           xlim=(-50, 50),
           )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.94, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[0].set(yticklabels=['BACARDI',
                        'BACARDI 1s',
                        'BACARDI 1Min',
                        'ecRad Reference\nFu-IFS (v15.1)',
                        ])
figname = f'05_HALO_AC3_RF17_RF18_cre_solar_BACARDI_ecRad_violin.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% plot violinplot of cre_terrestrial
sel_ver = ['BACARDI_org', 'BACARDI_1s', 'BACARDI_1Min', 'v15.1']
h.set_cb_friendly_colors('petroff_6')
plt.rc('font', size=10)
_, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 7.5 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df[(df.key == key)
                 & (df.variable == 'CRE_terrestrial')
                 & (df.label.isin(sel_ver))]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='values', y='label', hue='label',
                   ax=ax, order=sel_ver)
    ax.set(xlabel=f'Terrestrial cloud radiative\neffect ({h.plot_units['cre_total']})',
           ylabel='',
           yticklabels='',
           xlim=(-10, 80),
           )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(25))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.94, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[0].set(yticklabels=['BACARDI',
                        'BACARDI 1s',
                        'BACARDI 1Min',
                        'ecRad Reference\nFu-IFS (v15.1)',
                        ])
figname = f'05_HALO_AC3_RF17_RF18_cre_terrestrial_BACARDI_ecRad_violin.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% print cre statistics
df_print = cre_stats.loc[pd.IndexSlice[:, ['BACARDI_org'] + sel_ver], :].sort_values(['key', 'variable', 'label'])
print(df_print)
# %% plot violinplot of cre_total for all versions
sel_ver = ['BACARDI_1Min', 'v15.1', 'v19.1', 'v18.1', 'v13.2', 'v30.1', 'v31.1',
           'v32.1', 'v36', 'v37', 'v38', 'v39.2', 'v40.2']
h.set_cb_friendly_colors('petroff_6')
plt.rc('font', size=10)
_, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 20 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df[(df.key == key)
                 & (df.variable == 'CRE_total')
                 & (df.label.isin(sel_ver))]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='values', y='label', hue='label',
                   ax=ax, order=sel_ver)
    ax.set(xlabel=f'Total cloud radiative\neffect ({h.plot_units['cre_total']})',
           ylabel='',
           yticklabels='',
           xlim=(-50, 70),
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
                        'ecRad Measured albedo\nsimulation (v13.2)',
                        'ecRad aerosol\nFu-IFS (v30.1)',
                        'ecRad aerosol\nYi2013 (v31.1)',
                        'ecRad aersosl\nBaran2016 (v32.1)',
                        'ecRad VarCloud\nFu-IFS (v36)',
                        'ecRad VarCloud\nYi2013 (v37)',
                        'ecRad VarCloud\nBaran2016 (v38)',
                        'ecRad no cosine\nFu-IFS (v39.2)',
                        'ecRad no cosine\nYi2013 (v40.2)',
                        ])
figname = f'05_HALO_AC3_RF17_RF18_cre_total_BACARDI_ecRad_violin_all.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% plot violinplot of cre_terrestrial for all versions
sel_ver = ['BACARDI_1Min', 'v15.1', 'v19.1', 'v18.1', 'v13.2', 'v30.1', 'v31.1',
           'v32.1', 'v36', 'v37', 'v38', 'v39.2', 'v40.2']
h.set_cb_friendly_colors('petroff_6')
plt.rc('font', size=10)
_, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 20 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df[(df.key == key)
                 & (df.variable == 'CRE_terrestrial')
                 & (df.label.isin(sel_ver))]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='values', y='label', hue='label',
                   ax=ax, order=sel_ver)
    ax.set(xlabel=f'Terrestrial cloud radiative\neffect ({h.plot_units['cre_total']})',
           ylabel='',
           yticklabels='',
           xlim=(-50, 70),
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
                        'ecRad Measured albedo\nsimulation (v13.2)',
                        'ecRad aerosol\nFu-IFS (v30.1)',
                        'ecRad aerosol\nYi2013 (v31.1)',
                        'ecRad aersosl\nBaran2016 (v32.1)',
                        'ecRad VarCloud\nFu-IFS (v36)',
                        'ecRad VarCloud\nYi2013 (v37)',
                        'ecRad VarCloud\nBaran2016 (v38)',
                        'ecRad no cosine\nFu-IFS (v39.2)',
                        'ecRad no cosine\nYi2013 (v40.2)',
                        ])
figname = f'05_HALO_AC3_RF17_RF18_cre_terrestrial_BACARDI_ecRad_violin_all.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% plot violinplot of cre_solar for all versions
sel_ver = ['BACARDI_1Min', 'v15.1', 'v19.1', 'v18.1', 'v13.2', 'v30.1', 'v31.1',
           'v32.1', 'v36', 'v37', 'v38', 'v39.2', 'v40.2']
h.set_cb_friendly_colors('petroff_6')
plt.rc('font', size=10)
_, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 20 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df[(df.key == key)
                 & (df.variable == 'CRE_solar')
                 & (df.label.isin(sel_ver))]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='values', y='label', hue='label',
                   ax=ax, order=sel_ver)
    ax.set(xlabel=f'Solar cloud radiative\neffect ({h.plot_units['cre_total']})',
           ylabel='',
           yticklabels='',
           xlim=(-50, 70),
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
                        'ecRad Measured albedo\nsimulation (v13.2)',
                        'ecRad aerosol\nFu-IFS (v30.1)',
                        'ecRad aerosol\nYi2013 (v31.1)',
                        'ecRad aersosl\nBaran2016 (v32.1)',
                        'ecRad VarCloud\nFu-IFS (v36)',
                        'ecRad VarCloud\nYi2013 (v37)',
                        'ecRad VarCloud\nBaran2016 (v38)',
                        'ecRad no cosine\nFu-IFS (v39.2)',
                        'ecRad no cosine\nYi2013 (v40.2)',
                        ])
figname = f'05_HALO_AC3_RF17_RF18_cre_solar_BACARDI_ecRad_violin_all.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% print cre statistics
df_print = cre_stats.loc[pd.IndexSlice[:, ['BACARDI_org'] + sel_ver], :].sort_values(['key', 'variable', 'label'])
print(df_print)

# %% plot all terrestrial fluxes making up the cre
plt.rc('font', size=10)
key = 'RF17'
_, ax = plt.subplots(figsize=(15 * h.cm, 9 * h.cm), layout='constrained')
plot_ds = bacardi_ds[key].sel(time=slices[key]['case'])
time_extend = pd.to_timedelta((plot_ds.time[-1] - plot_ds.time[0]).to_numpy())
ax.plot(plot_ds.time, plot_ds['CRE_terrestrial'], label=h.bacardi_labels['CRE_terrestrial'])
ax.plot(plot_ds.time, plot_ds['F_up_terrestrial'], label=h.bacardi_labels['F_up_terrestrial'])
ax.plot(plot_ds.time, plot_ds['F_down_terrestrial'], label=h.bacardi_labels['F_down_terrestrial'])
ax.plot(plot_ds.time, plot_ds['F_up_terrestrial_sim_si'], label=f"{h.bacardi_labels['F_up_terrestrial']} sim")
ax.plot(plot_ds.time, plot_ds['F_down_terrestrial_sim_si'], label=f"{h.bacardi_labels['F_down_terrestrial']} sim")
ax.axhline(y=0, color='k')
h.set_xticks_and_xlabels(ax, time_extend)
ax.yaxis.set_major_locator(mticker.MultipleLocator(25))
ax.grid()
ax.legend()
ax.set(ylabel=f'Terrestrial irradiance ({h.plot_units['cre_sw']})')

axs[0].text(0.03, 0.88, '(a)', transform=axs[0].transAxes)
axs[1].set_xlabel('Time (UTC)')
axs[0].legend(loc=1, ncols=3)

# figname = f'{plot_path}/05_HALO-AC3_RF17_RF18_BACARDI_libRadtran_CRE.pdf'
# plt.savefig(figname, dpi=300)
plt.show()
plt.close()
