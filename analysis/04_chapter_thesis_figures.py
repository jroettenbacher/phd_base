#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 22.03.2024

Here all figures from chapter 4 of my thesis are created.

- violin plot for sea ice albedo experiment
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

import pylim.halo_ac3 as meta
import pylim.helpers as h
from pylim import ecrad

h.set_cb_friendly_colors('petroff_6')
cbc = h.get_cb_friendly_colors('petroff_6')

# %% set paths
campaign = 'halo-ac3'
save_path = 'C:/Users/Johannes/Documents/Doktor/manuscripts/_thesis/data'
plot_path = 'C:/Users/Johannes/Documents/Doktor/manuscripts/_thesis/figure'
trajectory_path = f'{h.get_path('trajectories', campaign=campaign)}/selection_CC_and_altitude'
keys = ['RF17', 'RF18']
ecrad_versions = [f'v{x}' for x in [13, 13.1, 13.2, 15.1, 16, 18.1, 19.1, 20,
                                    22, 24, 26, 27, 28, 30.1, 31.1, 32.1]]

# %% read in data
(
    bahamas_ds, bacardi_ds, bacardi_ds_res, ecrad_dicts, varcloud_ds, above_clouds,
    below_clouds, slices, ecrad_orgs, ifs_ds_sel, dropsonde_ds
) = (dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict())

for key in keys:
    flight = meta.flight_names[key]
    date = flight[9:17]
    bacardi_path = h.get_path('bacardi', flight, campaign)
    bahamas_path = h.get_path('bahamas', flight, campaign)
    ifs_path = f'{h.get_path('ifs', flight, campaign)}/{date}'
    ecrad_path = f'{h.get_path('ecrad', flight, campaign)}/{date}'
    varcloud_path = h.get_path('varcloud', flight, campaign)
    dropsonde_path = h.get_path('dropsondes', flight, campaign)
    dropsonde_path = f'{dropsonde_path}/Level_1' if key == 'RF17' else f'{dropsonde_path}/Level_2'

    # filenames
    bahamas_file = f'HALO-AC3_HALO_BAHAMAS_{date}_{key}_v1_JR.nc'
    bacardi_file = f'HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1_JR_v2.nc'
    ifs_file = f'ifs_{date}_00_ml_O1280_processed_sel_JR.nc'
    varcloud_file = [f for f in os.listdir(varcloud_path) if f.endswith('_JR.nc')][0]
    dropsonde_files = [f for f in os.listdir(dropsonde_path) if f.endswith('.nc')]

    # read in aircraft data
    bahamas_ds[key] = xr.open_dataset(f'{bahamas_path}/{bahamas_file}')
    bacardi = xr.open_dataset(f'{bacardi_path}/{bacardi_file}')
    bacardi_ds[key] = bacardi
    # read in resampled BACARDI data
    bacardi_res = xr.open_dataset(f'{bacardi_path}/{bacardi_file.replace('_v2.nc', '_1Min_v2.nc')}')
    bacardi_ds_res[key] = bacardi_res

    # read in ifs data
    ifs_ds_sel[key] = xr.open_dataset(f'{ifs_path}/{ifs_file}').set_index(rgrid=['lat', 'lon'])

    # read in ecrad data
    ecrad_dict, ecrad_org = dict(), dict()

    for k in ecrad_versions:
        ds = xr.open_dataset(f'{ecrad_path}/ecrad_merged_inout_{date}_{k}.nc')
        # add net terrestrial irradiance
        ds['flux_net_lw'] = ds['flux_dn_lw'] - ds['flux_up_lw']
        ecrad_org[k] = ds.copy(deep=True)
        # select only center column for direct comparisons
        ds = ds.sel(column=0, drop=True) if 'column' in ds.dims else ds
        ecrad_dict[k] = ds.copy(deep=True)

    ecrad_dicts[key] = ecrad_dict
    ecrad_orgs[key] = ecrad_org

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

# %% solar transmissivity - prepare data for box/violin plot
ecrad_var = 'transmissivity_sw_above_cloud'
label = 'transmissivity_sw'
bacardi_var = 'transmissivity_above_cloud'
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

    dfs.append(pd.DataFrame({'values': (bacardi_ds[key][bacardi_var]
                                        .sel(time=slices[key]['below'])
                                        .dropna('time')
                                        .to_pandas()
                                        .reset_index(drop=True)),
                             'label': 'BACARDI',
                             'key': key}))
    df = pd.concat(dfs)

df = df.reset_index(drop=True)

# %% solar transmissivity - get statistics
st_stats = (df
            .groupby(['key', 'label'])['values']
            .describe()
            .sort_values(['key', 'mean'], ascending=[True, False]))

# %% 3D effects - plot violinplot of solar transmissivity
h.set_cb_friendly_colors('petroff_6')
_, axs = plt.subplots(1, 2, figsize=h.figsize_wide, layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df[(df.key == key)
                 & (df.label.isin(['BACARDI', 'v15.1', 'v22', 'v18.1', 'v24']))]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='label', y='values', hue='label',
                   order=['BACARDI', 'v15.1', 'v22', 'v18.1', 'v24'],
                   ax=ax)
    ax.set(xlabel='', ylabel='',
           xticklabels=['BACARDI',
                        'ecRad Reference\nFu-IFS (v15.1)',
                        'ecRad 3D on\nFu-IFS (v22)',
                        'ecRad Reference\nBaran2016 (v18.1)',
                        'ecRad 3D on\nBaran2016 (v24)'],
           ylim=(0.45, 1),
           title=key.replace('1', ' 1'))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.grid()

axs[0].set(ylabel=f'{h.cbarlabels[label]}')
figname = f'03_HALO_AC3_RF17_RF18_{bacardi_var}_BACARDI_ecRad_3d_effects_violin.png'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% 3D effects - plot boxplot with all simulations
h.set_cb_friendly_colors('cartocolor')
_, axs = plt.subplots(1, 2, figsize=h.figsize_wide, layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df[(df.key == key)
                 & (df.label.isin(['BACARDI'] + ecrad_versions))]
    sns.boxplot(df_plot, x='label', y='values', notch=True, hue='label',
                order=['BACARDI', 'v15.1', 'v22', 'v18.1', 'v24', 'v16', 'v26', 'v20', 'v27'],
                ax=ax)
    ax.set(xlabel='', ylabel='',
           xticklabels=['BACARDI',
                        'Fu-IFS (v15.1)',
                        '3D on Fu-IFS (v22)',
                        'Baran2016 (v18.1)',
                        '3D on Baran2016 (v24)',
                        'VarCloud Fu-IFS (v16)',
                        'VarCloud 3D Fu-IFS (v26)',
                        'VarCloud Baran2016 (v20)',
                        'VarCloud 3D Baran2016 (v27)'],
           ylim=(0.45, 1),
           title=key.replace('1', ' 1'))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid()

axs[0].set(ylabel=f'{h.cbarlabels[label]}')
figname = f'03_HALO_AC3_RF17_RF18_{bacardi_var}_BACARDI_ecRad_3d_effects_boxplot_all.png'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% sea ice - plot violinplot of below cloud transmissivity
h.set_cb_friendly_colors('petroff_6')
_, axs = plt.subplots(1, 2, figsize=h.figsize_wide, layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df[(df.key == key)
                 & (df.label.isin(['BACARDI', 'v15.1', 'v13', 'v13.2']))]
    df_plot['label'] = (df_plot['label']
                        .astype('category')
                        .cat.reorder_categories(['BACARDI', 'v15.1', 'v13', 'v13.2']))
    sns.violinplot(df_plot, x='label', y='values', hue='label',
                   ax=ax)
    ax.set(xlabel='', ylabel='',
           xticklabels=['BACARDI',
                        'ecRad Reference\nsimulation (v15.1)',
                        'ecRad Open ocean\nsimulation (v13)',
                        'ecRad Measured albedo\nsimulation (v13.2)'],
           ylim=(0.4, 1),
           title=key.replace('1', ' 1'))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.grid()

axs[0].set(ylabel='Solar transmissivity')
figname = f'03_HALO_AC3_RF17_RF18_{bacardi_var}_BACARDI_ecRad_albedo_violin.png'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% aerosol - plot violinplot of solar transmissivity
plt.rc('font', size=10)
_, axs = plt.subplots(1, 2, figsize=h.figsize_wide, layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df[(df.key == key)
                 & (df.label.isin(['BACARDI', 'v15.1', 'v30.1']))]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='label', y='values', hue='label', ax=ax)
    ax.set(xlabel='', ylabel='',
           xticklabels=['BACARDI',
                        'ecRad Reference\nFu-IFS (v15.1)',
                        'ecRad aerosol on\nFu-IFS (v30.1)',
                        ],
           ylim=(0.45, 1),
           title=key.replace('1', ' 1'))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.grid()

axs[0].set(ylabel=f'{h.cbarlabels[label]}')
figname = f'03_HALO_AC3_RF17_RF18_{bacardi_var}_BACARDI_ecRad_aerosol_violin.png'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% aerosol - plot violinplot of solar transmissivity for all simulations
plt.rc('font', size=10)
_, axs = plt.subplots(1, 2, figsize=h.figsize_wide, layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df[(df.key == key)
                 & (df.label.isin(['BACARDI', 'v15.1', 'v30.1', 'v31.1', 'v32.1']))]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='label', y='values', hue='label', ax=ax)
    ax.set(xlabel='', ylabel='',
           xticklabels=['BACARDI',
                        'ecRad Reference\nFu-IFS (v15.1)',
                        'ecRad aerosol on\nFu-IFS (v30.1)',
                        'ecRad aerosol on\nYi2013 (v31.1)',
                        'ecRad aerosol on\nBaran2016 (v32.1)',
                        ],
           ylim=(0.45, 1),
           title=key.replace('1', ' 1'))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.grid()

axs[0].set(ylabel=f'{h.cbarlabels[label]}')
figname = f'03_HALO_AC3_RF17_RF18_{bacardi_var}_BACARDI_ecRad_aerosol_violin_all.png'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% aerosol - print stats
st_stats[st_stats.index.isin(['BACARDI', 'v15.1', 'v30.1', 'v31.1', 'v32.1'], level=1)]

# %% terrestrial irradiance - plot BACARDI vs. ecRad terrestrial downward above cloud
plt.rc('font', size=10)
label = ['(a)', '(b)']
for v in ['v15.1', 'v18.1', 'v19.1']:
    _, axs = plt.subplots(1, 2, figsize=(16 * h.cm, 8 * h.cm), layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
above_sel = (bahamas_ds[key].IRS_ALT > 11000).resample(time='1Min').first()
bacardi_res = bacardi_ds_res[key]
bacardi_plot = bacardi_res.where(bacardi_res.alt > 11000)
ecrad_ds = ecrad_dicts[key][v]
height_sel = ecrad_dicts[key][v].aircraft_level
ecrad_plot = ecrad_ds.flux_dn_lw.isel(half_level=height_sel).where(above_sel)

# actual plotting
rmse = np.sqrt(np.mean((bacardi_plot['F_down_terrestrial'] - ecrad_plot) ** 2)).to_numpy()
bias = np.nanmean((bacardi_plot['F_down_terrestrial'] - ecrad_plot).to_numpy())
ax.scatter(bacardi_plot['F_down_terrestrial'], ecrad_plot, color=cbc[3])
ax.axline((0, 0), slope=1, color='k', lw=2, transform=ax.transAxes)
ax.set(
    aspect='equal',
    xlabel=r'Measured irradiance (W$\,$m$^{-2}$)',
    ylabel=r'Simulated irradiance (W$\,$m$^{-2}$)',
    xlim=(20, 40),
    ylim=(20, 40),
)
ax.grid()
ax.text(
    0.025,
    0.95,
    f'{label[i]} {key.replace('1', ' 1')}\n'
    f'n= {sum(~np.isnan(bacardi_plot['F_down_terrestrial'])):.0f}\n'
    f'RMSE: {rmse:.0f} {h.plot_units['flux_dn_lw']}\n'
    f'Bias: {bias:.0f} {h.plot_units['flux_dn_lw']}',
    ha='left',
    va='top',
    transform=ax.transAxes,
    bbox=dict(fc='white', ec='black', alpha=0.8, boxstyle='round'),
)

figname = f'{plot_path}/03_HALO-AC3_RF17_RF18_bacardi_ecrad_f_down_terrestrial_above_cloud_all_{v}.png'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% terrestrial irradiance - plot BACARDI terrestrial fluxes - 6 panel figure
plt.rc('font', size=10)
xlims = [(0, 240), (0, 320)]
ylim_net = (-175, 0)
ylim_irradiance = [(0, 280), (0, 280)]
yticks = mticker.MultipleLocator(50)
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
for var in ['F_down_terrestrial', 'F_up_terrestrial']:
    ax.plot(plot_ds.cum_distance, plot_ds[var], label=h.bacardi_labels[var])
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
for var in ['F_down_terrestrial', 'F_up_terrestrial']:
    ax.plot(cum_distance, plot_ds[var], label=h.bacardi_labels[var])
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
        color=cbc[2], label='Below cloud')
plot_ds = bacardi_ds['RF17'].sel(time=slices['RF17']['above'])
plot_ds['distance'] = bahamas_ds['RF17']['distance'].sel(time=slices['RF17']['above'])
# set first distance to 0
plot_ds['distance'][0] = 0
# sum up distances to generate a distance axis and convert to km
plot_ds['cum_distance'] = plot_ds['distance'].cumsum() / 1000
ax.plot(plot_ds.cum_distance, plot_ds['F_net_terrestrial'],
        color=cbc[3], label='Above cloud')
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
for var in ['F_down_terrestrial', 'F_up_terrestrial']:
    ax.plot(plot_ds.cum_distance, plot_ds[var], label=f'{h.bacardi_labels[var]}')
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
for var in ['F_down_terrestrial', 'F_up_terrestrial']:
    ax.plot(plot_ds['cum_distance'], plot_ds[var].to_numpy(), label=f'{h.bacardi_labels[var]}')
ax.grid()
ax.text(box_xy[0], box_xy[1], 'Below cloud', ha='right',
        transform=ax.transAxes, bbox=dict(boxstyle='Round', fc='white'))
ax.set(ylim=ylim_irradiance[1],
       xlim=xlims[1])
ax.yaxis.set_major_locator(yticks)

# lower right panel - RF18 net irradiance
ax = axs[2, 1]
ax.plot(plot_ds['cum_distance'], plot_ds['F_net_terrestrial'],
        color=cbc[2], label='Below cloud')
plot_ds = bacardi_ds['RF18'].sel(time=slices['RF18']['above'])
plot_ds['distance'] = bahamas_ds['RF18']['distance'].sel(time=slices['RF18']['above'])
# set first distance to 0
plot_ds['distance'][0] = 0
# sum up distances to generate a distance axis and convert to km
plot_ds['cum_distance'] = plot_ds['distance'].cumsum() / 1000
ax.plot(plot_ds['cum_distance'], plot_ds['F_net_terrestrial'],
        color=cbc[3], label='Above cloud')
ax.grid()
ax.legend()
ax.set(xlabel='Distance (km)',
       ylim=ylim_net,
       xlim=xlims[1])

# set a-f labels
for ax, label in zip(axs.flatten(), ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']):
    ax.text(label_xy[0], label_xy[1], label, transform=ax.transAxes)

figname = f'{plot_path}/03_HALO-AC3_RF17_RF18_BACARDI_terrestrial_case_studies_6panel.png'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% terrestrial irradiance - print statistics
sel_vars = ['F_down_terrestrial', 'F_up_terrestrial', 'F_net_terrestrial']
selection = (stats['version'].isin(['v1', 'v15.1'])
             & (stats['variable'].isin(sel_vars)
                | stats['variable'].isin(['flux_dn_lw', 'flux_up_lw', 'flux_net_lw'])))
df_print = stats[selection]

# %% terrestrial irradiance - plot PDF of net terrestrial irradiance below cloud
plt.rc('font', size=10)
label = [['(a)', '(b)', '(c)'], ['(d)', '(e)', '(f)']]
ylims = [(0, 0.3), (0, 0.12)]
xlims = (-150, -25)
legend_loc = ['upper right']
binsize = 1
xlabel = r'Net terrestrial irradiance (W$\,$m$^{-2}$)'
_, axs = plt.subplots(2, 3, figsize=(18 * h.cm, 14 * h.cm), layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    l = label[i]
    bacardi_sel = bacardi_ds[key].sel(time=slices[key]['below'])
    bacardi_plot = bacardi_sel['F_net_terrestrial'].resample(time='1Min').mean()
    bins = np.arange(-150, -25, binsize)
    # BACARDI histogram
    bacardi_hist = np.histogram(bacardi_plot, density=True, bins=bins)

    for ii, v in enumerate(['v15.1', 'v19.1', 'v18.1']):
        v_name = ecrad.get_version_name(v[:3])
        a = ax[ii]
        ecrad_ds = ecrad_orgs[key][v].sel(time=slices[key]['below'])
        height_sel = ecrad_ds['aircraft_level']
        ecrad_plot = ecrad_ds['flux_net_lw'].isel(half_level=height_sel)

        # actual plotting
        sns.histplot(bacardi_plot, label='BACARDI', ax=a, stat='density', kde=False, bins=bins, element='step')
        sns.histplot(ecrad_plot.to_numpy().flatten(), label=v_name, stat='density', element='step',
                     kde=False, bins=bins, ax=a, color=cbc[ii + 1])
        # add mean
        a.axvline(bacardi_plot.mean(), color=cbc[0], lw=3, ls='--')
        a.axvline(ecrad_plot.mean(), color=cbc[ii + 1], lw=3, ls='--')
        a.plot([], ls='--', color='k', label='Mean')  # label for means
        a.set(ylabel='',
              ylim=ylims[i],
              xlim=xlims
              )
        handles, labels = a.get_legend_handles_labels()
        order = [1, 0, 2]
        handles = [handles[idx] for idx in order]
        labels = [labels[idx] for idx in order]
        if key == 'RF17':
            a.legend(handles, labels, loc=legend_loc[i])
        a.text(
            0.02,
            0.95,
            f'{l[ii]}',
            ha='left',
            va='top',
            transform=a.transAxes,
        )
        a.grid()

    ax[0].set(ylabel='Probability density function')
    ax[1].set(title=f'{key.replace('1', ' 1')} '
                f'(n = {len(ecrad_plot.to_numpy().flatten()):.0f})')
    if key == 'RF18':
        ax[1].set(xlabel=xlabel)

figname = (f'{plot_path}/03_HALO-AC3_HALO_RF17_RF18_bacardi_ecrad_F_net_terr_PDF'
           f'_below_cloud_ice_optics'
           f'_all_columns.pdf')
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% terrestrial irradiance - plot PDF of net terrestrial irradiance below cloud VarCloud
plt.rc('font', size=10)
label = [['(a)', '(b)', '(c)'], ['(d)', '(e)', '(f)']]
ylims = [(0, 0.3), (0, 0.12)]
xlims = (-150, -25)
legend_loc = ['upper right']
binsize = 1
xlabel = r'Net terrestrial irradiance (W$\,$m$^{-2}$)'
_, axs = plt.subplots(2, 3, figsize=(18 * h.cm, 14 * h.cm), layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    l = label[i]
    bacardi_sel = bacardi_ds[key].sel(time=slices[key]['below'])
    bacardi_plot = bacardi_sel['F_net_terrestrial'].resample(time='1Min').mean()
    bins = np.arange(-150, -25, binsize)
    # BACARDI histogram
    bacardi_hist = np.histogram(bacardi_plot, density=True, bins=bins)

    for ii, v in enumerate(['v16', 'v28', 'v20']):
        v_name = ecrad.get_version_name(v[:3])
        v_name = v_name.replace(' VarCloud', '')
        a = ax[ii]
        ecrad_ds = ecrad_orgs[key][v].sel(time=slices[key]['below'])
        height_sel = ecrad_ds['aircraft_level']
        ecrad_plot = ecrad_ds['flux_net_lw'].isel(half_level=height_sel)

        # actual plotting
        sns.histplot(bacardi_plot, label='BACARDI', ax=a, stat='density', kde=False, bins=bins, element='step')
        sns.histplot(ecrad_plot.to_numpy().flatten(), label=v_name, stat='density', element='step',
                     kde=False, bins=bins, ax=a, color=cbc[ii + 1])
        # add mean
        a.axvline(bacardi_plot.mean(), color=cbc[0], lw=3, ls='--')
        a.axvline(ecrad_plot.mean(), color=cbc[ii + 1], lw=3, ls='--')
        a.plot([], ls='--', color='k', label='Mean')  # label for means
        a.set(ylabel='',
              ylim=ylims[i],
              xlim=xlims
              )
        handles, labels = a.get_legend_handles_labels()
        order = [1, 0, 2]
        handles = [handles[idx] for idx in order]
        labels = [labels[idx] for idx in order]
        if key == 'RF17':
            a.legend(handles, labels, loc=legend_loc[i])
        a.text(
            0.02,
            0.95,
            f'{l[ii]}',
            ha='left',
            va='top',
            transform=a.transAxes,
        )
        a.grid()

    ax[0].set(ylabel='Probability density function')
    ax[1].set(title=f'{key.replace('1', ' 1')} '
                    f'(n = {len(ecrad_plot.to_numpy().flatten()):.0f})')
    if key == 'RF18':
        ax[1].set(xlabel=xlabel)

figname = (f'{plot_path}/03_HALO-AC3_HALO_RF17_RF18_bacardi_ecrad_F_net_terr_PDF'
           f'_below_cloud_ice_optics_VarCloud'
           f'_all_columns.pdf')
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% terrestrial irradiance - print stats for all simulations
sel_vars = ['F_net_terrestrial']
selection = (stats['version'].isin(['v1', 'v15.1', 'v16', 'v19.1', 'v28', 'v18.1', 'v20'])
             & (stats['section'] == 'below')
             & (stats['variable'].isin(sel_vars)
                | stats['variable'].isin(['flux_net_lw'])))
df_print = stats[selection]

# %% terrestrial irradiance - plot PDF of downward terrestrial irradiance below cloud
plt.rc('font', size=10)
label = [['(a)', '(b)', '(c)'], ['(d)', '(e)', '(f)']]
ylims = [(0, 0.3), (0, 0.12)]
xlims = (80, 200)
legend_loc = ['lower right']
binsize = 1
xlabel = r'Terrestrial downward irradiance (W$\,$m$^{-2}$)'
_, axs = plt.subplots(2, 3, figsize=(18 * h.cm, 14 * h.cm), layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
l = label[i]
bacardi_sel = bacardi_ds[key].sel(time=slices[key]['below'])
bacardi_plot = bacardi_sel['F_down_terrestrial'].resample(time='1Min').mean()
bins = np.arange(xlims[0], xlims[1], binsize)
# BACARDI histogram
bacardi_hist = np.histogram(bacardi_plot, density=True, bins=bins)

for ii, v in enumerate(['v15.1', 'v19.1', 'v18.1']):
    v_name = ecrad.get_version_name(v[:3])
a = ax[ii]
ecrad_ds = ecrad_orgs[key][v].sel(time=slices[key]['below'])
height_sel = ecrad_ds['aircraft_level']
ecrad_plot = ecrad_ds['flux_dn_lw'].isel(half_level=height_sel)

# actual plotting
sns.histplot(bacardi_plot, label='BACARDI', ax=a, stat='density', kde=False, bins=bins, element='step')
sns.histplot(ecrad_plot.to_numpy().flatten(), label=v_name, stat='density', element='step',
             kde=False, bins=bins, ax=a, color=cbc[ii + 1])
# add mean
a.axvline(bacardi_plot.mean(), color=cbc[0], lw=3, ls='--')
a.axvline(ecrad_plot.mean(), color=cbc[ii + 1], lw=3, ls='--')
a.plot([], ls='--', color='k', label='Mean')  # label for means
a.set(ylabel='',
      ylim=ylims[i],
      xlim=xlims
      )
handles, labels = a.get_legend_handles_labels()
order = [1, 0, 2]
handles = [handles[idx] for idx in order]
labels = [labels[idx] for idx in order]
if key == 'RF17':
    a.legend(handles, labels, loc=legend_loc[i])
a.text(
    0.02,
    0.95,
    f'{l[ii]}',
    ha='left',
    va='top',
    transform=a.transAxes,
)
a.grid()

ax[0].set(ylabel='Probability density function')
ax[1].set(title=f'{key.replace('1', ' 1')} '
                f'(n = {len(ecrad_plot.to_numpy().flatten()):.0f})')
if key == 'RF18':
    ax[1].set(xlabel=xlabel)

figname = (f'{plot_path}/03_HALO-AC3_HALO_RF17_RF18_bacardi_ecrad_F_down_terr_PDF'
           f'_below_cloud_ice_optics'
           f'_all_columns.pdf')
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% terrestrial irradiance - plot PDF of upward terrestrial irradiance below cloud
plt.rc('font', size=10)
label = [['(a)', '(b)', '(c)'], ['(d)', '(e)', '(f)']]
ylims = [(0, 0.3), (0, 0.12)]
xlims = (210, 240)
legend_loc = ['lower right']
binsize = 1
xlabel = r'Terrestrial upward irradiance (W$\,$m$^{-2}$)'
_, axs = plt.subplots(2, 3, figsize=(18 * h.cm, 14 * h.cm), layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
l = label[i]
bacardi_sel = bacardi_ds[key].sel(time=slices[key]['below'])
bacardi_plot = bacardi_sel['F_up_terrestrial'].resample(time='1Min').mean()
bins = np.arange(xlims[0], xlims[1], binsize)
# BACARDI histogram
bacardi_hist = np.histogram(bacardi_plot, density=True, bins=bins)

for ii, v in enumerate(['v15.1', 'v19.1', 'v18.1']):
    v_name = ecrad.get_version_name(v[:3])
a = ax[ii]
ecrad_ds = ecrad_orgs[key][v].sel(time=slices[key]['below'])
height_sel = ecrad_ds['aircraft_level']
ecrad_plot = ecrad_ds['flux_up_lw'].isel(half_level=height_sel)

# actual plotting
sns.histplot(bacardi_plot, label='BACARDI', ax=a, stat='density', kde=False, bins=bins, element='step')
sns.histplot(ecrad_plot.to_numpy().flatten(), label=v_name, stat='density', element='step',
             kde=False, bins=bins, ax=a, color=cbc[ii + 1])
# add mean
a.axvline(bacardi_plot.mean(), color=cbc[0], lw=3, ls='--')
a.axvline(ecrad_plot.mean(), color=cbc[ii + 1], lw=3, ls='--')
a.plot([], ls='--', color='k', label='Mean')  # label for means
a.set(ylabel='',
      ylim=ylims[i],
      xlim=xlims
      )
handles, labels = a.get_legend_handles_labels()
order = [1, 0, 2]
handles = [handles[idx] for idx in order]
labels = [labels[idx] for idx in order]
if key == 'RF17':
    a.legend(handles, labels, loc=legend_loc[i])
a.text(
    0.02,
    0.95,
    f'{l[ii]}',
    ha='left',
    va='top',
    transform=a.transAxes,
)
a.grid()

ax[0].set(ylabel='Probability density function')
ax[1].set(title=f'{key.replace('1', ' 1')} '
                f'(n = {len(ecrad_plot.to_numpy().flatten()):.0f})')
if key == 'RF18':
    ax[1].set(xlabel=xlabel)

figname = (f'{plot_path}/03_HALO-AC3_HALO_RF17_RF18_bacardi_ecrad_F_up_terr_PDF'
           f'_below_cloud_ice_optics'
           f'_all_columns.pdf')
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% cre - plot BACARDI net radiative effect
plt.rc('font', size=10)
ylims = (-50, 70)
_, axs = plt.subplots(2, 1, figsize=(16 * h.cm, 9 * h.cm), layout='constrained')
for i, k in enumerate(keys):
    ax = axs[i]
    plot_ds = bacardi_ds[k].sel(time=slices[k]['case'])
    time_extend = pd.to_timedelta((plot_ds.time[-1] - plot_ds.time[0]).to_numpy())
    ax.plot(plot_ds.time, plot_ds['CRE_solar'], label=h.bacardi_labels['CRE_solar'])
    ax.plot(plot_ds.time, plot_ds['CRE_terrestrial'], label=h.bacardi_labels['CRE_terrestrial'])
    ax.plot(plot_ds.time, plot_ds['CRE_total'], label=h.bacardi_labels['CRE_total'])
    ax.axhline(y=0, color='k')
    h.set_xticks_and_xlabels(ax, time_extend)
    ax.grid()
    ax.set(ylabel=f'Cloud radiative\neffect ({h.plot_units['cre_sw']})',
           ylim=ylims)

axs[0].text(0.03, 0.88, '(a)', transform=axs[0].transAxes)
axs[1].text(0.03, 0.88, '(b)', transform=axs[1].transAxes)
axs[1].set_xlabel('Time (UTC)')
axs[0].legend(loc=1, ncols=3)

figname = f'{plot_path}/03_HALO-AC3_RF17_RF18_BACARDI_libRadtran_CRE.png'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% cre - plot PDF
pass

# %% testing
plot_ds[var].plot(x='time')
plt.show()
plt.close()
