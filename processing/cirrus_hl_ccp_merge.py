#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 19.07.2024

Combine all daily CCP files and add the latitude from BAHAMAS to them.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

import pylim.helpers as h
from pylim.bahamas import preprocess_bahamas

# %% set paths
bahamas_path = h.get_path('all', campaign='cirrus-hl', instrument='BAHAMAS')
ccp_path = h.get_path('all', campaign='cirrus-hl', instrument='CCP')
save_path = 'C:/Users/Johannes/Documents/Doktor/manuscripts/_thesis/data'

ccp_files = [os.path.join(ccp_path, f) for f in os.listdir(ccp_path) if f.endswith('nc')]
ccp_files.sort()

bahamas_files = [os.path.join(bahamas_path, f) for f in os.listdir(bahamas_path) if f.startswith('C')]

# %% read in data
ccp = xr.open_mfdataset(ccp_files)
bahamas = xr.open_mfdataset(bahamas_files, preprocess=preprocess_bahamas)

# %% select relevant variables
lat = bahamas.IRS_LAT.rename('latitude')
ed = ccp['ed,_m,_effective_diameter'].rename('effective_diameter')
ed.attrs.update(long_name='effective diameter',
                title='ratio of 3rd and 2nd moment of the distribution of the particle maximum dimension',
                units='m')

# %% merge
ds = xr.merge([lat, ed], join='inner')

# %% prepare dataframe
df = ds.to_pandas()
df['effective_diameter'] = df['effective_diameter'] * 1e6  # convert to mum
df.loc[df['effective_diameter'] < 15, 'effective_diameter'] = np.nan
df = df.dropna()
df['category'] = np.where(df['latitude'] > 60, 'arctic', 'mid-latitude')
lat_bins = np.arange(np.min(df['latitude'].round(0)),
                     np.max(df['latitude'].round(0))+1,
                     1)
df['latitude_bin'] = pd.cut(df['latitude'], bins=lat_bins)


# %% plot dataframe
sns.boxplot(x='category', y='effective_diameter', data=df)
plt.show()
plt.close()

# %% get statistics
df.groupby('category')['effective_diameter'].count()
df.groupby('category')['effective_diameter'].median()

# %% look at values in one bin
df.where(df['latitude_bin'] == pd.Interval(71, 72, closed='right')).dropna()

# %% stats of binned data
stats = df.groupby('latitude_bin')['effective_diameter'].median().reset_index()
stats.rename(columns={'effective_diameter': 'median'}, inplace=True)
stats['min'] = df.groupby('latitude_bin')['effective_diameter'].min().reset_index().iloc[:, 1]
stats['max'] = df.groupby('latitude_bin')['effective_diameter'].max().reset_index().iloc[:, 1]
stats['mid_latitude'] = stats['latitude_bin'].cat.categories.mid

# %% save to csv for plotting
stats.to_csv(f'{save_path}/median_ed_delatorre2023.csv', index=False)
