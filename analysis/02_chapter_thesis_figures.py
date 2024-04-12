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
import xarray as xr

cbc = h.get_cb_friendly_colors('petroff_6')

# %% set paths
ecrad_path = 'E:/ecrad_sensitivity_studies'
save_path = 'C:/Users/Johannes/Documents/Doktor/manuscripts/_thesis/data'
plot_path = 'C:/Users/Johannes/Documents/Doktor/manuscripts/_thesis/figure'

# %% plot minimum ice effective radius from Sun2001 parameterization
latitudes = np.arange(0, 91)
de2re = 0.64952  # from suecrad.f90
min_ice = 60
min_diameter_um = 20 + (min_ice - 20) * np.cos((np.deg2rad(latitudes)))
min_radius_um = de2re * min_diameter_um

plt.rc('font', size=9)
# _, ax = plt.subplots(figsize=(15 * h.cm, 7 * h.cm), layout='constrained')
# ax.plot(min_radius_um, latitudes, '.')
# ax.set(ylabel='Latitude (°N)', xlabel='Minimum ice effective radius ($\\mu m$)')
# ax.yaxis.set_major_locator(ticker.MultipleLocator(15))
# ax.grid()
# plt.show()
# plt.close()

_, ax = plt.subplots(figsize=(15 * h.cm, 6 * h.cm), layout='constrained')
ax.plot(latitudes, min_radius_um, '.', ms=10)
ax.set(xlabel='Latitude (°N)', ylabel='Minimum\n ice effective radius ($\\mu m$)',
       ylim=(10, 40), xlim=0)
ax.xaxis.set_major_locator(ticker.MultipleLocator(15))
ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax.grid()
plt.savefig(f'{plot_path}/02_reice_min_latitude.pdf', dpi=300)
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
