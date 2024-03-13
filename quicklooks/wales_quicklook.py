#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 28.02.2024

Plot a quicklook of the WALES lidar data from HALO.
"""
import pylim.helpers as h
import cmasher as cmr
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import xarray as xr

# %% set paths
plot_path = 'C:/Users/Johannes/Documents/Doktor/campaigns/CIRRUS-HL/case_studies/Flight_20210629a'
wales_path = 'E:/CIRRUS-HL/01_Flights/Flight_20210629a/WALES'
file = 'bsri'
wales_file = f'CIRRUS-HL_F05_20210629a_ADLR_WALES_{file}_v1.nc'
wl_dict = dict(bsrgl='532nm', bsrg='532nm', bsri='1064nm')

# %% read in data
ds = xr.open_dataset(f'{wales_path}/{wales_file}')
ds['altitude'] = ds['altitude'] / 1000
ds['range'] = ds.altitude
ds = ds.rename(range='height')

# %% filter data for looks
# ds['backscatter_ratio'] = ds['backscatter_ratio'].where(ds['backscatter_ratio'] > 0.001)

# %% plot backscatter ratio
t_sel = slice(pd.to_datetime('2021-06-29 13:50'),
              pd.to_datetime('2021-06-29 14:15'))
lidar_plot = ds.backscatter_ratio.sel(time=t_sel)
plt.rc('font', size=12)
_, ax = plt.subplots(figsize=h.figsize_wide, layout='constrained')
cmap = cmr.rainforest_r
cmap.set_bad(color='black')
lidar_plot.plot(x='time', y='height', robust=False, cmap=cmap, ax=ax,
                norm=colors.LogNorm(vmin=0.1, vmax=200),
                cbar_kwargs=(dict(label=f'Backscatter Ratio at {wl_dict[file]}')))
h.set_xticks_and_xlabels(ax, pd.Timedelta('30min'))
ax.set(xlabel='Time (UTC)',
       ylabel='Altitude (km)',
       title='',
       ylim=(0, 13))
figname = f'{plot_path}/WALES_backscatter_ratio_{wl_dict[file]}.png'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot profile
plot_ds = ds.backscatter_ratio.sel(time='2021-06-29 14:03:30', method='nearest')
_, ax = plt.subplots(figsize=h.figsize_equal)
plot_ds.plot(y='height', ax=ax)
ax.set(xlabel='Backscatter Ratio at 1064 nm',
       ylabel='Altitude (km)',
       title='')
plt.show()
plt.close()
