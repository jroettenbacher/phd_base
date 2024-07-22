#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 11.04.2024

Plots for Model chapter
"""

# %% import modules
import pylim.helpers as h
from pylim import ecrad
import cmasher as cm
import matplotlib.pyplot as plt
from matplotlib import ticker
from metpy.constants import Rd
from metpy.units import units as u
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

cbc = h.get_cb_friendly_colors('petroff_6')

# %% set paths
ecrad_path = 'E:/ecrad_sensitivity_studies'
save_path = 'C:/Users/Johannes/Documents/Doktor/manuscripts/_thesis/data'
plot_path = 'C:/Users/Johannes/Documents/Doktor/manuscripts/_thesis/figure'

# %% read in data
ecrad_dict = dict()
for v in ['v1', 'v2', 'v3']:
    ecrad_dict[v] = xr.open_dataset(f'{ecrad_path}/ecrad_merged_inout_{v}.nc')

ecrad_dict['diff_v2'] = ecrad_dict['v1'] - ecrad_dict['v2']
ecrad_dict['diff_v3'] = ecrad_dict['v1'] - ecrad_dict['v3']
ed = pd.read_csv(f'{save_path}/median_ed_delatorre2023.csv')

# %% plot minimum ice effective radius from Sun2001 parameterization together with median ed from delatorre2023
latitudes = np.arange(0, 91)
de2re = 0.64952  # from suecrad.f90
min_ice = 60
min_diameter_um = 20 + (min_ice - 20) * np.cos((np.deg2rad(latitudes)))
min_radius_um = de2re * min_diameter_um

plt.rc('font', size=9)
_, ax = plt.subplots(figsize=(15 * h.cm, 6 * h.cm), layout='constrained')
ax.plot(latitudes, min_radius_um, '.', ms=10)
ed.plot(x='mid_latitude', y='effective_diameter', ax=ax)
ax.set(xlabel='Latitude (°N)',
       ylabel='Minimum\n ice effective radius ($\\mu m$)',
       # ylim=(10, 40),
       xlim=0)
# ax.xaxis.set_major_locator(ticker.MultipleLocator(15))
# ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax.grid()
# plt.savefig(f'{plot_path}/04_reice_min_latitude.pdf', dpi=300)
plt.show()
plt.close()

# %% calculate ice effective radius for different IWC and T combinations covering the Arctic to the Tropics
lats = [0, 18, 36, 54, 72, 90]
iwc_kgkg = np.logspace(-5.5, -2.5, base=10, num=100)
t = np.arange(182, 273)
empty_array = np.empty((len(t), len(iwc_kgkg), len(lats)))
for i, temperature in enumerate(t):
    for j, iwc in enumerate(iwc_kgkg):
        for k, lat in enumerate(lats):
            empty_array[i, j, k] = ecrad.ice_effective_radius(25000, temperature, 1, iwc, 0, lat)

da = xr.DataArray(empty_array * 1e6, coords=[('temperature', t), ('iwc_kgkg', iwc_kgkg * 1e3), ('Latitude', lats, {'units': '°N'})],
                  name='re_ice')

# convert kg kg-1 to g m-3
air_density = 25000 * u('Pa') / (Rd * 182 * u('K'))
air_density = air_density.to(u('g/m3'))
iwc_gm3 = iwc_kgkg * air_density

# %% plot re_ice iwc t combinations
latitudes = [0, 18, 36, 54, 72, 90]
de2re = 0.64952  # from suecrad.f90
min_ice = 60
min_diameter_um = 20 + (min_ice - 20) * np.cos((np.deg2rad(latitudes)))
min_radius_um = de2re * min_diameter_um

plt.rc('font', size=9)
g = da.plot(col='Latitude', col_wrap=3,
            cbar_kwargs=dict(label=r'Ice effective radius ($\mu$m)'),
            cmap=cm.get_sub_cmap(cm.rainforest, 0.25, 1),
            figsize=(15 * h.cm, 10 * h.cm))
for i, ax in enumerate(g.axs.flat[::3]):
    ax.set_ylabel('Temperature (K)')
for i, ax in enumerate(g.axs.flat[3:]):
    ax.set_xlabel('IWC (g/kg)')
# 1st - 3rd panel
for i, ax in enumerate(g.axs.flat[:3]):
    cl = ax.contour(da.iwc_kgkg, da.temperature, da.isel(Latitude=i).to_numpy(),
                    levels=[min_radius_um[i], 45, 60, 80, 100],
                    colors='k', linestyles='--')
    ax.clabel(cl, fmt='%.0f', inline=True, fontsize=8)
# 4th and 5th panel
for i, ax in enumerate(g.axs.flat[3:-1]):
    cl = ax.contour(da.iwc_kgkg, da.temperature, da.isel(Latitude=i+3).to_numpy(),
                    levels=[min_radius_um[i+3], 40, 60, 80, 100],
                    colors='k', linestyles='--')
    ax.clabel(cl, fmt='%.0f', inline=True, fontsize=8)
# 6th panel
ax = g.axs.flat[-1]
cl = ax.contour(da.iwc_kgkg, da.temperature, da.isel(Latitude=5).to_numpy(),
                levels=[min_radius_um[5], 20, 40, 60, 80, 100],
                colors='k', linestyles='--')
ax.clabel(cl, fmt='%.0f', inline=True, fontsize=8)
# all panels
for i, ax in enumerate(g.axs.flat):
    ax.hlines(233, 0, 1e-2, color=cbc[2], ls='-')
    ax.vlines(1e-2, 0, 233, color=cbc[2], ls='-')
    ax.set_xscale('log')
    ax.set_title(f'Latitude = {da.Latitude[i].to_numpy()}' + r'$\,^{\circ}$N',
                 size=9)

figname = f'{plot_path}/02_re_ice_parameterization_T-IWC-Lat_log.pdf'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

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

