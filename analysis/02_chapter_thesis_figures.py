#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 14.03.2024

Plot of

- single scattering albedo
- scattering phase function

"""

# %% import modules
import pylim.helpers as h
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# %% set paths
if os.getcwd().startswith('/'):
    # we are on the server
    lib_path = '/projekt_agmwend/Documents/Scattering_Libraries/IceScatPropLib'
else:
    lib_path = 'E:/Scattering_Libraries/IceScatPropLib'

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

# %% read in data
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
        merge_list.append(df)

# read in single scattering properties of droplets
filepath_liquid = f'{lib_path}/../Water_droplets_Scat_Prop'
df1 = pd.read_csv(f'{filepath_liquid}/{wavelengths[0]}/isca.dat', sep='\\s+', names=columns)
df2 = pd.read_csv(f'{filepath_liquid}/{wavelengths[1]}/isca.dat', sep='\\s+', names=columns)
df = pd.concat([df1, df2], ignore_index=True)
df['shape'] = 'sphere'
df['roughness'] = '000'
merge_list.append(df)

df = pd.concat(merge_list, ignore_index=True)
df['wavelength'] = df['wavelength'] * 1e3  # convert µm to nm

# %% plot single scattering albedo
h.set_cb_friendly_colors('petroff_8')
rough = '000'  # µm
wl_range = (300, 2500)  # nm
shapes_sel = ['droxtal', 'plate', 'solid_column']
selection1 = ((df['roughness'] == rough)
              & (df['wavelength'] > wl_range[0])
              & (df['wavelength'] < wl_range[1])
              & (((df['shape'] == 'sphere') & (df['d_max'] == 10))
                 | ((df['shape'].isin(shapes_sel)) & (df['d_max'] == 40)))
              )

selection2 = ((df['roughness'] == rough)
              & (df['wavelength'] > wl_range[0])
              & (df['wavelength'] < wl_range[1])
              & (df['shape'].isin(['sphere', 'solid_column']))
              & (df['d_max'].isin([10, 20, 30]))
              )

_, axs = plt.subplots(2, 1, layout='constrained')

# first row - single scattering albedo vs wavelength for different shapes
ax = axs[0]
sns.lineplot(data=df[selection1],
             x='wavelength',
             y='omega',
             hue='shape',
             ax=ax)
ax.grid()
ax.set(xlabel='',
       ylabel='',
       ylim=(0.83, 1.01))
handles, labels = ax.get_legend_handles_labels()
labels = [x.capitalize().replace('_', ' ') for x in labels]
ax.legend(title='Shape',
          handles=handles, labels=labels)

# second row - single scattering albedo vs wavelength for different sizes
ax = axs[1]
sns.lineplot(data=df[selection2],
             x='wavelength',
             y='omega',
             hue='shape',
             style='d_max',
             ax=ax)
ax.grid()
ax.set(xlabel='Wavelength (nm)',
       ylabel='',
       ylim=(0.83, 1.01))
handles, labels = ax.get_legend_handles_labels()
labels = [x.capitalize().replace('_', ' ') for x in labels]
ax.legend(handles=handles, labels=labels)
_.set_label(r'Single scattering albedo $\tilde{\omega}$')

plt.show()
plt.close()


# %% calculate effective radius for single particle
reff = (3/2) * (df['volume'] / df['a_proj'])