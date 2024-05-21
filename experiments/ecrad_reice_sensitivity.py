#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 29.04.2024

Investigate sensitivity of solar downward flux below cirrus to the ice effective radius.

"""

# %% import modules
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import xarray as xr

# %% set paths
base_dir = 'E:/HALO-AC3/08_ecrad/reice_sensitivity/'
plot_path = 'C:/Users/Johannes/Documents/Doktor/campaigns/HALO-AC3/case_studies/ecrad_reice_sensitivity/'
key = 'RF17'
# below cloud half level
half_level = 93.5 if key == 'RF17' else 97.5

# %% merge ecrad input and output files along new dimension reice_dim
filename = f'{base_dir}/ecrad_merged_inout_reice_sensitivity_{key}.nc'
if os.path.exists(filename):
    ds = xr.open_dataset(filename)
else:
    infiles = [os.path.join(f'{base_dir}/ecrad_input_{key}', f) for f in os.listdir(f'{base_dir}/ecrad_input_{key}')]
    outfiles = [os.path.join(f'{base_dir}/ecrad_output_{key}', f) for f in os.listdir(f'{base_dir}/ecrad_output_{key}')]
    infiles.sort()
    outfiles.sort()
    reice_dim_coord = [float(re.findall(r'_(\d{1,3})\.nc', value)[0]) for value in infiles]

    ds_input = xr.open_mfdataset(infiles, combine='nested', concat_dim='reice_dim')
    ds_output = xr.open_mfdataset(outfiles, combine='nested', concat_dim='reice_dim')

    ds = xr.merge([ds_input, ds_output])
    # replace fill value with nan
    ds['re_ice'] = ds['re_ice'].where(ds.re_ice != 5.1961601e-05, np.nan)
    # replace reference simulation (0) coordinate with its actual value
    reice_dim_coord[0] = ds.re_ice.isel(reice_dim=0).mean() * 1e6
    ds = ds.assign_coords(reice_dim=reice_dim_coord)
    ds.to_netcdf(filename)

# %% plot flux_dn_sw below cloud against reice
plot_ds = ds['flux_dn_sw'].sel(half_level=half_level, column=0).sortby('reice_dim')
plot_ds.plot(x='reice_dim', marker='o', ls='')
plt.xlabel('Ice effective radius (µm)')
plt.ylabel('Solar downward irradiance (W/$m^2$)')
plt.show()
plt.close()

# %% calculate correlation coefficient
cor_ds = (ds['flux_dn_sw']
          .sel(half_level=93.5, column=0)
          .isel(reice_dim=slice(1, -1))
          .sortby('reice_dim')
          .compute())
np.corrcoef(cor_ds.to_numpy(), cor_ds.reice_dim.to_numpy())

# %% plot percentage difference to reference calculation
plot_ds = (ds['flux_dn_sw']
           .sel(half_level=93.5, column=0)
           .sortby('reice_dim')
           .compute())
plot_ds = (plot_ds / plot_ds.sel(reice_dim=18.1, method='nearest') - 1) * 100
plot_ds = plot_ds.assign_coords(reice_dim=plot_ds.reice_dim / 18.11)
plot_ds.plot(x='reice_dim', marker='o', ls='')
# plt.xlim(0.95, 2)
plt.show()
plt.close()
