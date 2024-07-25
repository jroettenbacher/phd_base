#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 14.03.2024

Plot of

- single scattering albedo
- scattering phase function

"""

# %% import modules
import os

import cmasher as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from metpy.constants import Rd
from metpy.units import units as u
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

import pylim.helpers as h
from pylim import ecrad

cbc = h.get_cb_friendly_colors('petroff_6')

# %% set paths
if os.getcwd().startswith('/'):
    # we are on the server
    lib_path = '/projekt_agmwend/Documents/Scattering_Libraries/IceScatPropLib'
else:
    lib_path = 'E:/Scattering_Libraries/IceScatPropLib'

plot_path = 'C:/Users/Johannes/Documents/Doktor/manuscripts/_thesis/figure'
data_path = 'C:/Users/Johannes/Documents/Doktor/manuscripts/_thesis/data'

wavelengths = ['Data_0.2_15.25', 'Data_16.4_99.0']
shapes = ['column_8elements', 'droxtal',
          'hollow_bullet_rosette', 'hollow_column',
          'plate', 'plate_5elements', 'plate_10elements',
          'solid_bullet_rosette', 'solid_column']
roughness = [f'Rough{x}' for x in ['000', '003', '050']]
# columns:
# wavelength (um),
# maximum dimension of particle size (um) (d_max),
# volume of particle (um^3),
# projected area (um^2) (a_proj),
# extinction efficiency (q_ext),
# single-scattering albedo (omega),
# asymmetry factor (g)
columns = ['wavelength', 'd_max', 'volume', 'a_proj', 'q_ext', 'omega', 'g']

# %% read in single scattering properties
isca_filename = f'{plot_path}/../data/single_scat_props_sel.csv'
if not os.path.isfile(isca_filename):
    merge_list = list()
    for s in shapes:
        for r in roughness:
            filepath_solar = f'{lib_path}/{wavelengths[0]}/{s}/{r}/isca.dat'
            filepath_ter = f'{lib_path}/{wavelengths[1]}/{s}/{r}/isca.dat'
            df1 = pd.read_csv(filepath_solar, sep='\\s+', header=None, names=columns)
            df2 = pd.read_csv(filepath_ter, sep='\\s+', header=None, names=columns)
            df = pd.concat([df1, df2], ignore_index=True)
            df['shape'] = s
            df['roughness'] = r[-3:]
            df['reff'] = np.round((3 / 4) * (df['volume'] / df['a_proj']), 0)
            merge_list.append(df)

    # read in single scattering properties of droplets
    filepath_liquid = f'{lib_path}/../Water_droplets_Scat_Prop'
    df1 = pd.read_csv(f'{filepath_liquid}/{wavelengths[0]}/isca.dat', sep='\\s+', names=columns)
    df2 = pd.read_csv(f'{filepath_liquid}/{wavelengths[1]}/isca.dat', sep='\\s+', names=columns)
    df = pd.concat([df1, df2], ignore_index=True)
    df['shape'] = 'sphere'
    df['roughness'] = '000'
    df['reff'] = np.round((df['d_max'] ** 3) / (df['d_max'] ** 2) / 2, 0)
    merge_list.append(df)

    df = pd.concat(merge_list, ignore_index=True)
    df['wavelength'] = df['wavelength'] * 1e3  # convert µm to nm
    df.to_csv(isca_filename, index=False)
else:
    df = pd.read_csv(isca_filename, header=0, sep=',')

# %% read in phase function
merge_list = list()
rough = '000'
shapes_sel = ['plate', 'solid_column', 'droxtal', 'column_8elements']
size_sel = [40]
wl_sel = .5
tmp_filename = f'{plot_path}/../data/P11_selection.csv'
if not os.path.isfile(tmp_filename):
    for s in shapes_sel:
        for r in roughness[0:1]:
            filepath_solar = f'{lib_path}/{wavelengths[0]}/{s}/{r}/P11.dat'
            filepath_ter = f'{lib_path}/{wavelengths[1]}/{s}/{r}/P11.dat'
            df1_p = pd.read_csv(filepath_solar, sep='\\s+', header=0)
            df2_p = pd.read_csv(filepath_ter, sep='\\s+', header=0)
            df1_p['wavelength'] = df1['wavelength']  # add wavelength column
            df1_p['d_max'] = df1['d_max']  # add size column
            df2_p['wavelength'] = df2['wavelength']  # add wavelength column
            df2_p['d_max'] = df2['d_max']  # add size column
            df_p = pd.concat([df1_p, df2_p], ignore_index=True)
            df_p['shape'] = s
            df_p['roughness'] = r[-3:]
            selection = (df_p['shape'].isin(shapes_sel)
                         & (df_p['roughness'] == rough)
                         & (df_p['d_max'].isin(size_sel))
                         & (df_p['wavelength'] == wl_sel))
            merge_list.append(df_p[selection])

    # read in phase function of droplets
    filepath_liquid = f'{lib_path}/../Water_droplets_Scat_Prop'
    df1_p = pd.read_csv(f'{filepath_liquid}/{wavelengths[0]}/P11.dat', sep='\\s+', header=0)
    df2_p = pd.read_csv(f'{filepath_liquid}/{wavelengths[1]}/P11.dat', sep='\\s+', header=0)
    df1_p['wavelength'] = df1['wavelength']  # add wavelength column
    df1_p['d_max'] = df1['d_max']  # add size column
    df2_p['wavelength'] = df2['wavelength']  # add wavelength column
    df2_p['d_max'] = df2['d_max']  # add size column
    df_p = pd.concat([df1_p, df2_p], ignore_index=True)
    df_p['shape'] = 'liquid_droplet'
    df_p['roughness'] = '000'
    selection = ((df_p['roughness'] == rough)
                 & (df_p['wavelength'] == wl_sel))
    merge_list.append(df_p[selection])

    concat_list = []
    for df_con in merge_list:
        concat_list.append(pd.melt(df_con,
                                   id_vars=['shape', 'roughness', 'd_max', 'wavelength'],
                                   value_name='P11', var_name='angle'))
    df_plot = pd.concat(concat_list, ignore_index=True)
    df_plot['angle'] = pd.to_numeric(df_plot.angle)
    df_plot.to_csv(tmp_filename, index=False)
else:
    df_plot = pd.read_csv(tmp_filename, header=0, sep=',')

# %% read in effective diameter from Dela Torre Castro 2023
ed = pd.read_csv(f'{data_path}/median_ed_delatorre2023.csv')
ed['effective_radius'] = ed['effective_diameter'] / 2

# %% plot single scattering albedo
h.set_cb_friendly_colors('petroff_6')
rough = 0
wl_range = (300, 5000)  # nm
shapes_sel = ['droxtal', 'plate', 'solid_column', 'column_8elements']
selection1 = ((df['roughness'] == rough)
              & (df['wavelength'] > wl_range[0])
              & (df['wavelength'] < wl_range[1])
              & df['shape'].isin(shapes_sel)
              & (df['d_max'] == 40)
              )

selection2 = ((df['roughness'] == rough)
              & (df['wavelength'] > wl_range[0])
              & (df['wavelength'] < wl_range[1])
              & (df['shape'].isin(['plate', 'solid_column']))
              & (df['d_max'].isin([10, 20, 40, 80]))
              )
plt.rc('font', size=9)
ylim = (0.2, 1.01)
fig, axs = plt.subplots(2, 1, layout='constrained',
                        figsize=(15 * h.cm, 9 * h.cm))

# first row - single scattering albedo vs wavelength for different shapes
ax = axs[0]
hue_order = ['plate', 'solid_column', 'droxtal', 'column_8elements']
sns.lineplot(data=df[selection1],
             x='wavelength',
             y='omega',
             hue='shape',
             hue_order=hue_order,
             ax=ax)
ax.text(0.02, 0.85, '(a)', transform=ax.transAxes)
ax.grid()
ax.set(xlabel='',
       ylabel='',
       xticklabels=[],
       ylim=ylim)
# ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
handles, labels = ax.get_legend_handles_labels()
labels = [x.capitalize().replace('_', ' ') for x in labels]
labels[-1] = "Column aggregate"
ax.legend(handles=handles, labels=labels,
          loc='lower left')

# second row - single scattering albedo vs wavelength for different sizes
ax = axs[1]
hue_order = ['plate', 'solid_column']
sns.lineplot(data=df[selection2],
             x='wavelength',
             y='omega',
             hue='shape',
             hue_order=hue_order,
             style='d_max',
             ax=ax)
ax.text(0.02, 0.85, '(b)', transform=ax.transAxes)
ax.grid()
ax.set(xlabel='Wavelength (nm)',
       ylabel='',
       ylim=ylim)
# ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))
handles, labels = ax.get_legend_handles_labels()
labels = [x.capitalize().replace('_', ' ') for x in labels]
labels[3] = r'$D_{\text{max}}$ ($\mu m$)'
handles.insert(3, plt.plot([], ls='')[0])
labels.insert(3, '')
handles.insert(3, plt.plot([], ls='')[0])
labels.insert(3, '')
ax.legend(handles=handles, labels=labels, ncols=2)
fig.supylabel(r'Single scattering albedo $\tilde{\omega}$', size=9)

figname = f'{plot_path}/01_single_scattering_albedo.pdf'
plt.savefig(figname, bbox_inches='tight')
plt.show()
plt.close()

# %% plot phase function for different ice crystal shapes
df_plot = pd.read_csv(tmp_filename, header=0, sep=',')
sel = (df_plot['shape'].isin(['plate', 'solid_column', 'droxtal', 'column_8elements'])
       & df_plot['d_max'].isin([40]))
df_plot = df_plot[sel]
hue_order = ['plate', 'solid_column', 'droxtal', 'column_8elements']
plt.rc('font', size=9)
h.set_cb_friendly_colors('petroff_6')
fig, axs = plt.subplot_mosaic([['left', 'right']], sharey=True,
                              width_ratios=[1, 5],
                              layout='constrained',
                              figsize=(15 * h.cm, 9 * h.cm))
# zoom in to first 5 deg
ax = axs['left']
sns.lineplot(data=df_plot[df_plot.angle < 5], x='angle', y='P11',
             hue='shape',
             hue_order=hue_order,
             # style='d_max',
             legend=False, ax=ax)
ax.text(0.02, 0.96, '(a)', transform=ax.transAxes)
ax.set(xlabel='',
       ylabel='Phase function $\\mathcal{P}$',
       yscale='log',
       xlim=(0, 5),
       xticks=(0, 1, 2, 3, 4, 5)
       )
ax.grid()
# rest of phase function
ax = axs['right']
sns.lineplot(data=df_plot[df_plot.angle >= 5], x='angle', y='P11',
             hue='shape',
             hue_order=hue_order,
             # style='d_max',
             ax=ax)
ax.text(0.01, 0.96, '(b)', transform=ax.transAxes)
handles, labels = ax.get_legend_handles_labels()
labels = [x.capitalize().replace('_', ' ') for x in labels]
labels[-1] = "Column aggregate"
ax.legend(handles=handles, labels=labels)
ax.grid()
ax.set(xlabel='',
       xlim=(5, 180),
       ylabel='',
       yscale='log'
       )
fig.supxlabel('Scattering angle $\\vartheta$ (deg)', size=9)

figname = f'{plot_path}/01_phase_function.pdf'
plt.savefig(figname, bbox_inches='tight')
plt.show()
plt.close()
# %% plot single scattering albedo for terrestrial wavelengths
h.set_cb_friendly_colors('petroff_6')
rough = 0
wl_range = (200, 100000)  # nm
shapes_sel = ['droxtal', 'plate', 'solid_column', 'column_8elements']
selection1 = ((df['roughness'] == rough)
              & (df['wavelength'] > wl_range[0])
              & (df['wavelength'] < wl_range[1])
              & df['shape'].isin(shapes_sel)
              & (df['d_max'] == 40)
              )

selection2 = ((df['roughness'] == rough)
              & (df['wavelength'] > wl_range[0])
              & (df['wavelength'] < wl_range[1])
              & (df['shape'].isin(['plate']))#, 'solid_column']))
              & (df['d_max'].isin([2, 5, 10, 100, 1000]))
              )

df_plot = df[selection1].copy()
df_plot['wavelength'] = df_plot.wavelength / 1000  # convert to mum
plt.rc('font', size=10)
ylim = (0.0, 1.01)
fig, axs = plt.subplots(2, 1, layout='constrained',
                        figsize=(15 * h.cm, 9 * h.cm))

# first row - single scattering albedo vs wavelength for different shapes
ax = axs[0]
hue_order = ['plate', 'solid_column', 'droxtal', 'column_8elements']
sns.lineplot(data=df_plot,
             x='wavelength',
             y='omega',
             hue='shape',
             hue_order=hue_order,
             ax=ax)
ax.text(0.02, 0.85, '(a)', transform=ax.transAxes)
ax.grid()
ax.set(xlabel='',
       ylabel='',
       xticklabels=[],
       ylim=ylim,
       xscale='log',
       )
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
handles, labels = ax.get_legend_handles_labels()
labels = [x.capitalize().replace('_', '\n') for x in labels]
labels[-1] = "Column\naggregate"
ax.legend(handles=handles, labels=labels,
          loc='upper left', bbox_to_anchor=(1.01, 1.01))

# second row - single scattering albedo vs wavelength for different sizes
df_plot = df[selection2].copy()
df_plot['wavelength'] = df_plot.wavelength / 1000  # convert to mum
ax = axs[1]
hue_order = ['plate', 'solid_column']
sns.lineplot(data=df_plot,
             x='wavelength',
             y='omega',
             hue='shape',
             hue_order=hue_order,
             style='d_max',
             ax=ax)
ax.text(0.02, 0.85, '(b)', transform=ax.transAxes)
ax.grid()
ax.set(xlabel=r'Wavelength ($\mu$m)',
       ylabel='',
       ylim=ylim,
       xscale='log',
       )
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))


handles, labels = ax.get_legend_handles_labels()
labels[3] = r'$D_{\text{max}}$'
ax.legend(handles=handles[3:], labels=labels[3:],
          bbox_to_anchor=(1.01, 1.01), loc='upper left')
fig.supylabel(r'Single scattering albedo $\tilde{\omega}$', size=10)

figname = f'{plot_path}/01_single_scattering_albedo_NIR.pdf'
plt.savefig(figname, bbox_inches='tight')
plt.show()
plt.close()

# %% plot single scattering albedo for terrestrial wavelengths
h.set_cb_friendly_colors('petroff_6')
rough = 0
wl_range = (200, 100000)  # nm
shapes_sel = ['droxtal', 'plate', 'solid_column', 'column_8elements']
selection1 = ((df['roughness'] == rough)
              & (df['wavelength'] > wl_range[0])
              & (df['wavelength'] < wl_range[1])
              & df['shape'].isin(shapes_sel)
              & (df['d_max'] == 40)
              )

selection2 = ((df['roughness'] == rough)
              & (df['wavelength'] > wl_range[0])
              & (df['wavelength'] < wl_range[1])
              & (df['shape'].isin(['plate']))#, 'solid_column']))
              & (df['d_max'].isin([2, 5, 10, 100, 1000]))
              )

df_plot = df[selection1].copy()
df_plot['wavelength'] = df_plot.wavelength / 1000  # convert to mum
plt.rc('font', size=10)
ylim = (0.0, 5)
fig, axs = plt.subplots(2, 1, layout='constrained',
                        figsize=(15 * h.cm, 9 * h.cm))

# first row - single scattering albedo vs wavelength for different shapes
ax = axs[0]
hue_order = ['plate', 'solid_column', 'droxtal', 'column_8elements']
sns.lineplot(data=df_plot,
             x='wavelength',
             y='q_ext',
             hue='shape',
             hue_order=hue_order,
             ax=ax)
ax.text(0.02, 0.85, '(a)', transform=ax.transAxes)
ax.grid()
ax.set(xlabel='',
       ylabel='',
       xticklabels=[],
       ylim=ylim,
       xscale='log',
       )
# ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))
handles, labels = ax.get_legend_handles_labels()
labels = [x.capitalize().replace('_', '\n') for x in labels]
labels[-1] = "Column\naggregate"
ax.legend(handles=handles, labels=labels,
          loc='upper left', bbox_to_anchor=(1.01, 1.01))

# second row - single scattering albedo vs wavelength for different sizes
df_plot = df[selection2].copy()
df_plot['wavelength'] = df_plot.wavelength / 1000  # convert to mum
ax = axs[1]
hue_order = ['plate', 'solid_column']
sns.lineplot(data=df_plot,
             x='wavelength',
             y='q_ext',
             hue='shape',
             hue_order=hue_order,
             style='d_max',
             ax=ax)
ax.text(0.02, 0.85, '(b)', transform=ax.transAxes)
ax.grid()
ax.set(xlabel=r'Wavelength ($\mu$m)',
       ylabel='',
       ylim=ylim,
       xscale='log',
       )
# ax.yaxis.set_major_locator(mticker.MultipleLocator(0.25))


handles, labels = ax.get_legend_handles_labels()
labels[3] = r'$D_{\text{max}}$'
ax.legend(handles=handles[3:], labels=labels[3:],
          bbox_to_anchor=(1.01, 1.01), loc='upper left')
fig.supylabel(r'Extinction efficiency $Q_{\text{ext}}$', size=10)

figname = f'{plot_path}/01_extinction_efficiency_NIR.pdf'
plt.savefig(figname, bbox_inches='tight')
plt.show()
plt.close()


# %% extinction efficiency
import miepython

# Constants
wavelengths = np.logspace(-1, 2, 400)  # Wavelengths from 0.1 to 100 micrometers
radius_values = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]  # Different droplet radii in micrometers
n_water = 1.33  # Refractive index of water droplets (simplified as constant)


# Function to calculate extinction efficiency using Mie theory
def extinction_efficiency(radius, wavelength, n_water):
    x = 2 * np.pi * radius / wavelength
    m = n_water
    qext, qsca, qback, g = miepython.mie(m, x)
    return qext


# Calculate extinction efficiencies
ext_efficiencies = {radius: [] for radius in radius_values}
for radius in radius_values:
    for wavelength in wavelengths:
        qext = extinction_efficiency(radius, wavelength, n_water)
        ext_efficiencies[radius].append(qext)

# Plotting
plt.figure(figsize=(10, 6))
for radius in radius_values:
    plt.plot(wavelengths, ext_efficiencies[radius], label=f'r = {radius} μm')

plt.xscale('log')
# plt.yscale('log')
plt.xlabel('Wavelength (μm)')
plt.ylabel('Extinction Efficiency')
plt.title('Extinction Efficiency of Water Droplets')
plt.legend()
plt.grid(True)
plt.show()
# %% plot minimum ice effective radius from Sun2001 parameterization together with median ed from delatorre2023
latitudes = np.arange(0, 91)
de2re = 0.64952  # from suecrad.f90
min_ice = 60
min_diameter_um = 20 + (min_ice - 20) * np.cos((np.deg2rad(latitudes)))
min_radius_um = de2re * min_diameter_um

plt.rc('font', size=10)
_, ax = plt.subplots(figsize=(15 * h.cm, 6 * h.cm), layout='constrained')
ax.plot(latitudes, min_radius_um, '-', label='Minimum $r_{\\text{eff, ice}}$ Sun (2001)')
ed.plot(x='mid_latitude', y='effective_radius', ax=ax, label='Mean $r_{\\text{eff, ice}}$\nDe La Torre Castro et al. (2023)')
ax.set(xlabel='Latitude (°N)',
       ylabel='Ice effective radius ($\\mu m$)',
       # ylim=(10, 40),
       xlim=0)
# ax.xaxis.set_major_locator(ticker.MultipleLocator(15))
# ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax.grid()
ax.legend()
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
