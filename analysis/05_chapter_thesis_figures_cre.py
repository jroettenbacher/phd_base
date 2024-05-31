#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 30.05.2024

Here the figures from Chapter 5 (Results) Section 5.4 (Cloud radiative effect) of my thesis are created.



"""
# %% import modules
import os
import dill

import cmasher as cmr
import cmcrameri.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import colors
from metpy.calc import relative_humidity_from_specific_humidity
from metpy.units import units as u
import numpy as np
import pandas as pd
from skimage import io
from sklearn.neighbors import BallTree
import seaborn as sns
from tqdm import tqdm
import xarray as xr

import pylim.halo_ac3 as meta
import pylim.helpers as h
import pylim.meteorological_formulas as met
from pylim import ecrad

h.set_cb_friendly_colors('petroff_6')
cbc = h.get_cb_friendly_colors('petroff_6')

# %% set paths
campaign = 'halo-ac3'
save_path = 'C:/Users/Johannes/Documents/Doktor/manuscripts/_thesis/data'
plot_path = 'C:/Users/Johannes/Documents/Doktor/manuscripts/_thesis/figure'
trajectory_path = f'{h.get_path('trajectories', campaign=campaign)}/selection_CC_and_altitude'
keys = ['RF17', 'RF18']
ecrad_versions = [f'v{x}' for x in [13, 13.1, 13.2, 15, 15.1, 16.1, 18.1, 19.1,
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
    bacardi_file = f'HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1_JR.nc'
    ifs_file = f'ifs_{date}_00_ml_O1280_processed.nc'
    varcloud_file = [f for f in os.listdir(varcloud_path) if f.endswith('.nc')][0]
    satfile = f'{save_path}/{key}_MODIS_Terra_CorrectedReflectance_Bands367.png'
    sat_url = f'https://gibs.earthdata.nasa.gov/wms/epsg3413/best/wms.cgi?\
version=1.3.0&service=WMS&request=GetMap&\
format=image/png&STYLE=default&bbox={left},{bottom},{right},{top}&CRS=EPSG:3413&\
HEIGHT=8192&WIDTH=8192&TIME={urldate}&layers=MODIS_Terra_CorrectedReflectance_Bands367'

    # read in aircraft data
    bahamas_ds[key] = xr.open_dataset(f'{bahamas_path}/{bahamas_file}')
    bacardi = xr.open_dataset(f'{bacardi_path}/{bacardi_file}')
    # read in resampled BACARDI data
    bacardi_res = xr.open_dataset(f'{bacardi_path}/{bacardi_file.replace('.nc', '_1Min_v2.nc')}')
    bacardi_ds_res[key] = bacardi_res.copy(deep=True)
    bacardi_res_1s = xr.open_dataset(f'{bacardi_path}/{bacardi_file.replace('.nc', '_1s_v2.nc')}')
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
    # filter low clouds according to ECMWF low cloud criterion (pressure higher than 0.8 * surface pressure)
    cloud_data = ifs[['q_liquid', 'q_ice', 'cloud_fraction', 'clwc', 'ciwc', 'crwc', 'cswc']]
    pressure_filter = ifs.pressure_full.sel(level=137) * 0.8
    low_cloud_filter = ifs.pressure_full < pressure_filter  # False for low clouds
    cloud_data = cloud_data.where(low_cloud_filter, 0)  # replace where False with 0
    ifs.update(cloud_data)
    ifs_ds[key] = ifs.copy(deep=True)

    # read in varcloud data
    varcloud = xr.open_dataset(f'{varcloud_path}/{varcloud_file}')
    varcloud = varcloud.swap_dims(time='Time', height='Height').rename(Time='time', Height='height')
    varcloud = varcloud.rename(Varcloud_Cloud_Ice_Water_Content='iwc', Varcloud_Cloud_Ice_Effective_Radius='re_ice')
    varcloud_ds[key] = varcloud

    # filter BACARDI data for motion flag
    bacardi_org = bacardi.copy(deep=True)
    bacardi = bacardi.where(bacardi.motion_flag)
    # overwrite variables which do not need to be filtered with original values
    for var in ['alt', 'lat', 'lon', 'sza', 'saa', 'attitude_flag', 'segment_flag', 'motion_flag']:
        bacardi[var] = bacardi_org[var]

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

    # interpolate standard ecRad simulation onto BACARDI time
    bacardi['ecrad_fdw'] = ecrad_dict['v15.1'].flux_dn_sw_clear.interp(time=bacardi.time,
                                                                       kwargs={'fill_value': 'extrapolate'})
    # calculate transmissivity using ecRad at TOA and above cloud
    bacardi['transmissivity_TOA'] = bacardi['F_down_solar'] / bacardi['ecrad_fdw'].isel(half_level=0)
    bacardi['transmissivity_above_cloud'] = bacardi['F_down_solar'] / bacardi['ecrad_fdw'].isel(half_level=73)
    bacardi_ds[key] = bacardi

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
