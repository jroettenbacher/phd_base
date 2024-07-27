#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 11.04.2024

Plots for Model chapter
"""
import os

# %% import modules
import cmasher as cm
import dill
import matplotlib.pyplot as plt
from matplotlib import ticker
from metpy.constants import Rd
from metpy.units import units as u
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

import pylim.helpers as h
import pylim.halo_ac3 as meta
from pylim import ecrad

cbc = h.get_cb_friendly_colors('petroff_6')

# %% set paths
ecrad_path = 'E:/ecrad_sensitivity_studies'
save_path = 'C:/Users/Johannes/Documents/Doktor/manuscripts/_thesis/data'
plot_path = 'C:/Users/Johannes/Documents/Doktor/manuscripts/_thesis/figure'

campaign = 'halo-ac3'
keys = ['RF17', 'RF18']

# %% read in data
ecrad_dict = dict()
for v in ['v1', 'v2', 'v3']:
    ecrad_dict[v] = xr.open_dataset(f'{ecrad_path}/ecrad_merged_inout_{v}.nc')

ecrad_dict['diff_v2'] = ecrad_dict['v1'] - ecrad_dict['v2']
ecrad_dict['diff_v3'] = ecrad_dict['v1'] - ecrad_dict['v3']

# %% read in BACARDI data
bacardi_ds, ecrad_ds, slices, above_clouds, below_clouds = dict(), dict(), dict(), dict(), dict()
for key in keys:
    flight = meta.flight_names[key]
    date = flight[9:17]
    bacardi_path = h.get_path('bacardi', flight=flight, campaign=campaign)
    ecrad_path2 = f'{h.get_path("ecrad", campaign=campaign)}/{date}'
    ecrad_file = f'ecrad_merged_inout_{date}_v15.1.nc'
    bacardi_file = f'HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1_JR_v2.nc'

    bacardi_ds[key] = xr.open_dataset(f'{bacardi_path}/{bacardi_file}')
    ecrad_ds[key] = xr.open_dataset(f'{ecrad_path2}/{ecrad_file}')

    # get flight segmentation and select below and above cloud section
    loaded_objects = list()
    filenames = [f'{key}_slices.pkl', f'{key}_above_cloud.pkl', f'{key}_below_cloud.pkl']
    for filename in filenames:
        with open(f'{save_path}/{filename}', 'rb') as f:
            loaded_objects.append(dill.load(f))

    slices[key] = loaded_objects[0]
    above_clouds[key] = loaded_objects[1]
    below_clouds[key] = loaded_objects[2]

# %% plot profiles of sensitivity study - ice optics
plt.rc('font', size=9)
fig, axs = plt.subplots(1, 3, figsize=(15 * h.cm, 9 * h.cm), layout='constrained')

re_ice = '1.5e-05'
q_ices = ['0.0001', '1e-05', '5e-06']
vs = ['v1', 'v2', 'v3']
lats = ['30', '80']
titles = ['Fu-IFS', 'Yi2013', 'Baran2016']
ls = ['-', '--', ':']
labels = ['(a)', '(b)', '(c)']
for i, v in enumerate(vs):
    ax = axs[i]
    ax.grid()
    ax.set(
        xlim=(75, 250),
        ylim=(0, 10),
        title=titles[i]
    )
    ax.fill_between(np.array([0, 250]), 5, 8, alpha=0.3, color='grey')
    ax.text(0.02, 0.95, labels[i], transform=ax.transAxes)
    for ii, lat in enumerate(lats):
        for iii, q_ice in enumerate(q_ices):
            ecrad_plot = ecrad_dict[v].sel(q_ice_dim=q_ice,
                                           latitude_dim=lat,
                                           re_ice_dim=re_ice,
                                           column=0)
            ax.plot(ecrad_plot.flux_dn_sw.to_numpy(),
                    ecrad_plot.altitude.to_numpy(),
                    color=cbc[ii],
                    ls=ls[iii],
                    )

# create nice legend
axs[0].plot([], color=None, ls='', label='Latitude (°N)')
axs[0].plot([], color=cbc[0], label='30')
axs[0].plot([], color=cbc[1], label='80')
axs[0].plot([], color=None, ls='', label='IWC (g/kg)')
axs[0].plot([], color='k', ls=ls[0], label='0.1')
axs[0].plot([], color='k', ls=ls[1], label='0.01')
axs[0].plot([], color='k', ls=ls[2], label='0.005')
fig.legend(loc='outside right upper')

# label the cloud
axs[0].annotate('Cirrus', xy=(120, 8), xytext=(100, 9),
                arrowprops=dict(arrowstyle='->'),
                )
axs[0].set(ylabel='Altitude (km)')
axs[1].set(xlabel=r'Solar downward irradiance (W$\,$m$^{-2}$)')

figname = f'{plot_path}/04_ecrad_sensitivity_study_flux_dn_sw_ice_optics.pdf'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot profiles of sensitivity study - IWC
plt.rc('font', size=9)
fig, axs = plt.subplots(1, 3, figsize=(15 * h.cm, 9 * h.cm), layout='constrained')

re_ice = '1.5e-05'
q_ices = ['0.0001', '1e-05', '5e-06']
vs = ['diff_v2', 'diff_v3']
lats = ['30', '80']
titles = [r'$0.1\,$g kg$^{-1}$', r'$0.01\,$g kg$^{-1}$', r'$0.005\,$g kg$^{-1}$']
ls = ['-', '--']
labels = ['(a)', '(b)', '(c)']
altitude = ecrad_dict['v1']['altitude'].sel(q_ice_dim=q_ices[0], re_ice_dim=re_ice, latitude_dim='30', column=0)

for i, q_ice in enumerate(q_ices):
    ax = axs[i]
    ax.grid()
    ax.set(
        xlim=(-10, 10),
        ylim=(0, 10),
        title=titles[i]
    )
    ax.text(0.02, 0.95, labels[i], transform=ax.transAxes)
    ax.fill_between(np.array([-10, 10]), 5, 8, alpha=0.3, color='grey')
    for ii, v in enumerate(vs):
        for iii, lat in enumerate(lats):
            ecrad_plot = ecrad_dict[v].sel(q_ice_dim=q_ice,
                                           latitude_dim=lat,
                                           re_ice_dim=re_ice,
                                           column=0)
            ax.plot(ecrad_plot.flux_dn_sw.to_numpy(),
                    altitude.to_numpy(),
                    color=cbc[ii],
                    ls=ls[iii],
                    )

# create nice legend
axs[0].plot([], color=None, ls='', label='Ice Optics')
axs[0].plot([], color=cbc[0], label='Yi2013')
axs[0].plot([], color=cbc[1], label='Baran2016')
axs[0].plot([], color=None, ls='', label='Latitude')
axs[0].plot([], color='k', ls=ls[0], label='30°$\\,$N')
axs[0].plot([], color='k', ls=ls[1], label='80°$\\,$N')
fig.legend(loc='outside right upper')

axs[0].set(ylabel='Altitude (km)')
axs[1].set(xlabel=r'Solar downward irradiance (W$\,$m$^{-2}$)')

figname = f'{plot_path}/04_ecrad_sensitivity_study_flux_dn_sw_iwc.pdf'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot difference between ice optics
re_ice = '1.5e-05'
q_ice = '1e-05'
vs = ['diff_v2', 'diff_v3']
lats = ['30', '80']
half_level = 44.5  # below cloud altitude (5km)
labels = ['Yi2013', 'Baran2016']
ls = ['-', '--']
altitude = ecrad_dict['v1']['altitude'].sel(q_ice_dim=q_ice, re_ice_dim=re_ice, latitude_dim='30', column=0) / 1000

_, ax = plt.subplots(figsize=(15 * h.cm, 6 * h.cm), layout='constrained')
for i, v in enumerate(vs):
    for ii, lat in enumerate(lats):
        flux_dn_sw = ecrad_dict[v]['flux_dn_sw'].sel(q_ice_dim=q_ice, re_ice_dim=re_ice, latitude_dim=lat, column=0)#half_level=half_level)
        ax.plot(flux_dn_sw.to_numpy(),
                altitude.to_numpy(),
                color=cbc[i],
                ls=ls[ii])

ax.set(
    ylabel='Altitude (km)',
    xlabel='Solar downward\n irradiance (W$\\,$m$^{-2}$)',
    title='',
    xlim=(-6, 6),
    ylim=(0, 10),
)
# create nice legend
ax.plot([], color=None, ls='', label='Latitude (°N)')
ax.plot([], color='k', ls=ls[0], label='30')
ax.plot([], color='k', ls=ls[1], label='80')
ax.plot([], color=None, ls='', label='Ice Optics')
ax.plot([], color=cbc[0], label=labels[0])
ax.plot([], color=cbc[1], label=labels[1])
ax.grid()
ax.legend()
plt.show()
plt.close()

# %% get some statistics
stats = list()
variables = ['transmissivity_sw_above_cloud', 'flux_dn_sw']
for v in vs:
    for lat in lats:
        for q_ice in q_ices:
            for label, half_level in zip(['above cloud', 'below cloud'], [41.5, 44.5]):
                for var in variables:
                    ds = ecrad_dict[v].sel(q_ice_dim=q_ice,
                                           latitude_dim=lat,
                                           re_ice_dim=re_ice,
                                           column=0,
                                           half_level=half_level)
                    stats.append((v, lat, q_ice, label, var, ds[var].to_numpy()))

df = pd.DataFrame(stats, columns=['version', 'latitude', 'q_ice', 'altitude', 'variable', 'value'])

# %% get more statistics
stats = list()
variables = ['flux_dn_sw']
for v in vs:
    for lat in lats:
        for q_ice in q_ices:
            for var in variables:
                ds = ecrad_dict[v].sel(q_ice_dim=q_ice,
                                       latitude_dim=lat,
                                       re_ice_dim=re_ice,
                                       column=0,
                                       half_level=slice(42, 50),
                                       )
                stats.append((v, lat, q_ice, 'mean', var, np.mean(ds[var].to_numpy())))
                stats.append((v, lat, q_ice, 'median', var, np.median(ds[var].to_numpy())))
                stats.append((v, lat, q_ice, 'min', var, np.min(ds[var].to_numpy())))
                stats.append((v, lat, q_ice, 'max', var, np.max(ds[var].to_numpy())))

df_stat = pd.DataFrame(stats, columns=['version', 'latitude', 'q_ice', 'stat', 'variable', 'value'])

# %% print

# %% calculate difference of above to below cloud solar downward irradiance
pivot_df = (df[df.variable == 'flux_dn_sw']
            .pivot_table(index=['version', 'latitude', 'q_ice', 'variable'],
                         columns='altitude',
                         values='value')
            .reset_index())
pivot_df['difference'] = pivot_df['above cloud'] - pivot_df['below cloud']

# %% plot scatter plot of difference
plot_df = df[(df.variable == 'transmissivity_sw_above_cloud') & (df.altitude == 'below cloud')]
plot_df.loc[:, ['latitude', 'value', 'q_ice']] = plot_df[['latitude', 'value', 'q_ice']].apply(pd.to_numeric)
fig, ax = plt.subplots(figsize=(10 * h.cm, 10 * h.cm), layout='constrained')
sns.scatterplot(plot_df, x='q_ice', y='value', hue='latitude', style='version', ax=ax)
plt.show()
plt.close()


# %% plot comparison of above cloud irradiance between libRadtran and ecRad
label = ['a)', 'b)']
_, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 9 * h.cm), layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    date = '2022-04-11' if key == 'RF17' else '2022-04-12'
    plot_df = (bacardi_ds[key][['ecrad_fdw', 'F_down_solar_sim', 'alt', 'lat']]
               # .sel(time=slices[key]['case'])
               .to_pandas()
               .dropna())
    rmse = np.mean(np.sqrt((plot_df['F_down_solar_sim'] - plot_df['ecrad_fdw'])**2))
    bias = np.mean(plot_df['F_down_solar_sim'] - plot_df['ecrad_fdw'])
    ax.scatter(plot_df['F_down_solar_sim'], plot_df['ecrad_fdw'], color=cbc[3])
    ax.axline((0, 0), slope=1, color='k', lw=2, transform=ax.transAxes)
    ax.grid()
    ax.set(
        aspect='equal',
        xlabel='libRadtran $F^{\\downarrow}_{\\text{solar}}$' + f' ({h.plot_units['flux_dn_sw']})',
        ylabel='',
        xlim=(150, 550),
        ylim=(150, 550)
    )
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.text(
        0.05,
        0.95,
        f'{label[i]} {key.replace('F', 'F ')}\n'
        f'$n$ = {sum(~np.isnan(plot_df['F_down_solar_sim'])):,.0f}\n'
        f'RMSE: {rmse:.0f} {h.plot_units['flux_dn_sw']}\n'
        f'Bias: {bias:.0f} {h.plot_units['flux_dn_sw']}',
        ha='left',
        va='top',
        transform=ax.transAxes,
        bbox=dict(fc='white', ec='black', alpha=0.8, boxstyle='round'),
    )

axs[0].set(
    ylabel='ecRad $F^{\\downarrow}_{\\text{solar}}$' f' ({h.plot_units['flux_dn_sw']})',
)
figname = f'04_libRadtran_vs_ecRad_cloud-free.pdf'
plt.savefig(f'{plot_path}/{figname}')
plt.show()
plt.close()

# %% plot comparison of above cloud irradiance between libRadtran and ecRad - time series
label = ['a)', 'b)']
_, axs = plt.subplots(2, 1, figsize=(15 * h.cm, 9 * h.cm), layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    plot_df = (bacardi_ds[key][['ecrad_fdw', 'F_down_solar_sim', 'alt', 'lat']]
               # .sel(time=slices[key]['case'])
               .to_pandas()
               .dropna())
    plot_df['difference'] = plot_df['F_down_solar_sim'] - plot_df['ecrad_fdw']
    rmse = np.mean(np.sqrt((plot_df['F_down_solar_sim'] - plot_df['ecrad_fdw'])**2))
    bias = np.mean(plot_df['F_down_solar_sim'] - plot_df['ecrad_fdw'])
    # ax.plot(plot_df.index, plot_df['F_down_solar_sim'], label='libRadtran')
    # ax.plot(plot_df.index, plot_df['ecrad_fdw'], label='ecRad')
    ax.plot(plot_df.index, plot_df['difference'])
    ax.grid()
    ax.set(
        xlabel='Time (UTC)',
        ylabel='Downward solar irradiance (wm2)',
    )

figname = f'04_libRadtran_vs_ecRad_cloud-free_time_series.pdf'
plt.savefig(f'{plot_path}/{figname}')
plt.show()
plt.close()
