#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 22.03.2024

Here all figures from Chapter 5 (Results) of my thesis are created.

- dropsonde profile comparison with IFS above cloud profiles
- IFS cloud fraction with lidar/radar mask
- PDF of IWC at 11 and 12 UTC
- scatter plot of above cloud measurements and simulations
- violin plot for sea ice albedo experiment
- violin plot of aerosol simulations
- violin plot of ice optics comparison
"""
# %% import modules
import os
import dill

import cmasher as cmr
import cmcrameri.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

# %% plot temperature and humidity profiles from IFS and from dropsonde
below_cloud_altitude = dict()
h.set_cb_friendly_colors('petroff_8')
plt.rc('font', size=10)
_, axs = plt.subplots(1, 4, figsize=(18 * h.cm, 11 * h.cm), layout='constrained')
for i, key in enumerate(keys):
    below_cloud_altitude[key] = bahamas_ds[key].IRS_ALT.sel(time=slices[key]['below']).mean(dim='time') / 1000
    ax = axs[i * 2]
    ifs_plot = ecrad_dicts[key]['v15.1'].sel(time=slices[key]['case'])
    sf = 1000

    # Air temperature
    for t in ifs_plot.time:
        ifs_p = ifs_plot.sel(time=t)
        ax.plot(ifs_p.temperature_hl - 273.15, ifs_p.press_height_hl / 1000, color='grey', lw=0.5)
    ds_plot = dropsonde_ds[key]
    times = ['104205', '110137'] if key == 'RF17' else ['110321', '110823', '111442', '112014', '112524']
    date = '20220411' if key == 'RF17' else '20220412'
    times_dt = pd.to_datetime([date + t for t in times], format='%Y%m%d%H%M%S')
    for k in times_dt:
        ds = ds_plot.where(ds_plot.launch_time == k, drop=True)
        ds = ds.where(~np.isnan(ds['ta']), drop=True)
        ax.plot(ds['ta'][0] - 273.15, ds.alt / sf, label=f'DS {k:%H:%M} UTC', lw=2)
    ax.set(
        xlim=(-60, -10),
        ylim=(0, 12),
        xlabel='Air temperature (°C)',
    )
    ax.set_title(f'{key.replace('1', ' 1')} - {date_title[i]}', fontsize=10)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(base=15))
    ax.plot([], color='grey', label='IFS profiles')
    ax.axhline(below_cloud_altitude[key], c='k')
    ax.grid()

    # RH
    ax = axs[i * 2 + 1]
    ifs_plot = ecrad_dicts[key]['v15.1'].sel(time=slices[key]['case'])
    for t in ifs_plot.time:
        ifs_p = ifs_plot.sel(time=t)
        rh = relative_humidity_from_specific_humidity(ifs_p.pressure_full * u.Pa, ifs_p.t * u.K, ifs_p.q * u('kg/kg'))
        rh_ice = met.relative_humidity_water_to_relative_humidity_ice(rh * 100, ifs_p.t - 273.15)
        ax.plot(rh_ice, ifs_p.press_height_full / 1000, color='grey', lw=0.5)
    ds_plot = dropsonde_ds[key]
    for k in times_dt:
        ds = ds_plot.where(ds_plot.launch_time == k, drop=True)
        ds = ds.where(~np.isnan(ds.rh), drop=True)
        ax.plot(met.relative_humidity_water_to_relative_humidity_ice(ds.rh * 100, ds['ta'] - 273.15)[0],
                ds.alt / sf, label=f'DS {k:%H:%M} UTC', lw=2)
    ax.set(
        xlim=(0, 130),
        ylim=(0, 12),
        xlabel='Relative humidity \nover ice (%)',
    )
    ax.set_title(f'{key.replace('1', ' 1')} - {date_title[i]}', fontsize=10)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(base=25))
    ax.plot([], color='grey', label='IFS profiles')
    ax.axhline(below_cloud_altitude[key], c='k')
    ax.legend(fontsize=7)
    ax.grid()

axs[0].set_ylabel('Altitude (km)')
axs[0].text(0.02, 0.95, '(a)', transform=axs[0].transAxes)
axs[1].text(0.02, 0.95, '(b)', transform=axs[1].transAxes)
axs[2].text(0.02, 0.95, '(c)', transform=axs[2].transAxes)
axs[3].text(0.02, 0.95, '(d)', transform=axs[3].transAxes)

figname = f'{plot_path}/05_HALO-AC3_HALO_RF17_RF18_ifs_dropsonde_t_rh.pdf'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()
h.set_cb_friendly_colors('petroff_6')

# %% plot IFS cloud fraction lidar/mask comparison
var = 'cloud_fraction'
plt.rc('font', size=10)
fig, axs = plt.subplots(2, 1, figsize=(16 * h.cm, 11 * h.cm), layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    ds = ecrad_dicts[key]['v15'].sel(time=slices[key]['case'])
    ifs_plot = ds[[var]]
    bahamas_plot = bahamas_ds[key].IRS_ALT.sel(time=slices[key]['case']) / 1000
    # add new z axis mean pressure altitude
    if 'half_level' in ifs_plot.dims:
        new_z = ds['press_height_hl'].mean(dim='time') / 1000
    else:
        new_z = ds['press_height_full'].mean(dim='time') / 1000

    ifs_plot_new_z = list()
    for t in tqdm(ifs_plot.time, desc='New Z-Axis'):
        tmp_plot = ifs_plot.sel(time=t)
        if 'half_level' in tmp_plot.dims:
            tmp_plot = tmp_plot.assign_coords(
                half_level=ds['press_height_hl'].sel(time=t, drop=True).to_numpy() / 1000)
            tmp_plot = tmp_plot.rename(half_level='height')
        else:
            tmp_plot = tmp_plot.assign_coords(
                level=ds['press_height_full'].sel(time=t, drop=True).to_numpy() / 1000)
            tmp_plot = tmp_plot.rename(level='height')

        tmp_plot = tmp_plot.interp(height=new_z.to_numpy())
        ifs_plot_new_z.append(tmp_plot)

    ifs_plot = xr.concat(ifs_plot_new_z, dim='time').sortby('height').sel(height=slice(0, 12))
    ifs_plot = ifs_plot.where(ifs_plot[var] > 0)  # > 1e-9) * 1e6
    halo_plot = varcloud_ds[key].sel(time=slices[key]['case']).Varcloud_Input_Mask
    halo_plot = halo_plot.assign_coords(height=halo_plot.height / 1000).sortby('height')
    time_extend = pd.to_timedelta((ifs_plot.time[-1] - ifs_plot.time[0]).to_numpy())

    # plot IFS cloud cover prediction and Radar lidar mask
    pcm = ifs_plot[var].plot(x='time', cmap=cmr.sapphire, ax=ax, add_colorbar=False)
    halo_plot.plot.contour(x='time', levels=[0.9], colors=cbc[1], ax=ax, linewidths=2)
    ax.plot([], color=cbc[1], label='Radar & Lidar Mask', lw=2)
    bahamas_plot.plot(x='time', lw=2, color=cbc[-2], label='HALO altitude', ax=ax)
    ax.axvline(x=pd.to_datetime(f'{bahamas_plot.time.dt.date[0]:%Y-%m-%d} 11:30'),
               label='New IFS time step', lw=2, ls='--')
    h.set_xticks_and_xlabels(ax, time_extend)
    ax.set(xlabel='Time (UTC)', ylabel='Height (km)')
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=0, ha='center')

# place colorbar for both flights
fig.colorbar(pcm, ax=axs[:2], label=f'IFS {h.cbarlabels[var].lower()} {h.plot_units[var]}', pad=0.001)
axs[0].legend(fontsize=8)
axs[0].set_xlabel('')
axs[0].text(0.01, 0.85, '(a) RF 17', transform=axs[0].transAxes)
axs[1].text(0.01, 0.85, '(b) RF 18', transform=axs[1].transAxes)

figname = f'{plot_path}/05_HALO-AC3_HALO_RF17_RF18_IFS_{var}_radar_lidar_mask.pdf'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot PDF of IWC from IFS above cloud for 11 and 12 UTC
plt.rc('font', size=10)
legend_labels = ['11 UTC', '12 UTC']
binsizes = dict(iwc=1, reice=4)
_, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 8 * h.cm), layout='constrained')
ylims = {'iwc': (0, 0.22), 'reice': (0, 0.095)}
# left panel - RF17 IWC
ax = axs[0]
binsize = binsizes['iwc']
bins = np.arange(0, 20.1, binsize)
iwc_ifs_ls = list()
for t in ['2022-04-11 11:00', '2022-04-11 12:00']:
    iwc_ifs, cc = ifs_ds_sel['RF17'].q_ice.sel(time=t), ifs_ds_sel['RF17'].cloud_fraction.sel(time=t)
    iwc_ifs_ls.append(iwc_ifs.where(cc > 0).where(cc == 0, iwc_ifs / cc))

for i, pds in enumerate(iwc_ifs_ls):
    pds = pds.to_numpy().flatten() * 1e6
    pds = pds[~np.isnan(pds)]
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i],
        color=cbc[i],
        histtype='step',
        density=True,
        lw=2,
    )
    print(f'RF 17/n{legend_labels[i]}: n={len(pds)}, mean={np.mean(pds):.2f}, median={np.median(pds):.2f}')
ax.grid()
ax.set(ylabel=f'Probability density function',
       xlabel=f'Ice water content ({h.plot_units['iwc']})',
       ylim=ylims['iwc'],
       xticks=range(0, 21, 5),
       )
ax.set_title('RF 17 - 11 April 2022', fontsize=10)
ax.text(0.05, 0.93,
        f'(a)',
        transform=ax.transAxes,
        )

# right panel - RF18 IWC
ax = axs[1]
iwc_ifs_ls = list()
for t in ['2022-04-12 11:00', '2022-04-12 12:00']:
    iwc_ifs, cc = ifs_ds_sel['RF18'].q_ice.sel(time=t), ifs_ds_sel['RF18'].cloud_fraction.sel(time=t)
    iwc_ifs_ls.append(iwc_ifs.where(cc > 0).where(cc == 0, iwc_ifs / cc))

for i, pds in enumerate(iwc_ifs_ls):
    pds = pds.to_numpy().flatten() * 1e6
    pds = pds[~np.isnan(pds)]
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i],
        color=cbc[i],
        histtype='step',
        density=True,
        lw=2,
    )
    print(f'RF 18/n{legend_labels[i]}: n={len(pds)}, mean={np.mean(pds):.2f}, median={np.median(pds):.2f}')
ax.legend()
ax.grid()
ax.set(ylabel=f'',
       xlabel=f'Ice water content ({h.plot_units['iwc']})',
       ylim=ylims['iwc'],
       xticks=range(0, 21, 5),
       )
ax.text(0.05, 0.93,
        f'(b)',
        transform=ax.transAxes,
        )
ax.set_title('RF 18 - 12 April 2022', fontsize=10)

figname = f'{plot_path}/05_HALO-AC3_HALO_RF17_RF18_IFS_iwc_11_vs_12_pdf_case_studies.pdf'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot scatter plot of above cloud measurements and simulations
plt.rc('font', size=10)
label = ['(a)', '(b)']
date_title = ['11 April 2022', '12 April 2022']
for v in ['v15.1']:
    _, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 7.5 * h.cm),
                          layout='constrained')
    for i, key in enumerate(keys):
        ax = axs[i]
        above_sel = (bahamas_ds[key].IRS_ALT > 11000).resample(time='1Min').first()
        bacardi_res = bacardi_ds_res[key]
        bacardi_plot = bacardi_res.where(bacardi_res.alt > 11000)
        ecrad_ds = ecrad_dicts[key][v]
        height_sel = ecrad_dicts[key][v].aircraft_level
        ecrad_plot = ecrad_ds.flux_dn_sw.isel(half_level=height_sel).where(above_sel)

        # actual plotting
        rmse = np.sqrt(np.mean((bacardi_plot['F_down_solar_diff'] - ecrad_plot) ** 2)).to_numpy()
        bias = np.nanmean((bacardi_plot['F_down_solar_diff'] - ecrad_plot).to_numpy())
        ax.scatter(bacardi_plot.F_down_solar_diff, ecrad_plot, color=cbc[3])
        ax.axline((0, 0), slope=1, color='k', lw=2, transform=ax.transAxes)
        ax.set(
            aspect='equal',
            xlabel='Measured irradiance (W$\\,$m$^{-2}$)',
            ylabel='Simulated irradiance (W$\\,$m$^{-2}$)',
            xlim=(175, 525),
            ylim=(175, 525),
        )
        ax.grid()
        ax.text(
            0.025,
            0.95,
            f'{label[i]}\n'
            f'n= {sum(~np.isnan(bacardi_plot['F_down_solar'])):.0f}\n'
            f'RMSE: {rmse:.0f} {h.plot_units['flux_dn_sw']}\n'
            f'Bias: {bias:.0f} {h.plot_units['flux_dn_sw']}',
            ha='left',
            va='top',
            transform=ax.transAxes,
            bbox=dict(fc='white', ec='black', alpha=0.8, boxstyle='round'),
        )
        ax.set_title(f'{key.replace('1', ' 1')} - {date_title[i]}', fontsize=10)

    figname = f'{plot_path}/05_HALO-AC3_HALO_RF17_RF18_bacardi_ecrad_f_down_solar_above_cloud_all_{v}.pdf'
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% solar transmissivity - prepare data for box/violin plot
ecrad_var = 'transmissivity_sw_above_cloud'
label = 'transmissivity_sw'
bacardi_var = 'transmissivity_above_cloud'
filepath = f'{save_path}/halo-ac3_{label}_boxplot_data.csv'
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

    dfs.append(pd.DataFrame({'values': (bacardi_ds_res_1s[key][bacardi_var]
                                        .sel(time=slices[key]['below'])
                                        .dropna('time')
                                        .to_pandas()
                                        .reset_index(drop=True)),
                             'label': 'BACARDI_1s',
                             'key': key}))

    dfs.append(pd.DataFrame({'values': (bacardi_ds_res[key][bacardi_var]
                                        .sel(time=slices[key]['below'])
                                        .dropna('time')
                                        .to_pandas()
                                        .reset_index(drop=True)),
                             'label': 'BACARDI',
                             'key': key}))

    dfs.append(pd.DataFrame({'values': (bacardi_ds[key][bacardi_var]
                                        .sel(time=slices[key]['below'])
                                        .dropna('time')
                                        .to_pandas()
                                        .reset_index(drop=True)),
                             'label': 'BACARDI_org',
                             'key': key}))

    df = pd.concat(dfs)

df = df.reset_index(drop=True)
df.to_csv(filepath, index=False)

# %% solar transmissivity - get statistics
st_stats = (df
            .groupby(['key', 'label'])['values']
            .describe()
            .sort_values(['key', 'mean'], ascending=[True, False]))
versions = [v for v in st_stats.index.get_level_values('label') if v.startswith('v')]
df_save = st_stats.reset_index()
name = list()
aerosol = list()
for v in df_save['label']:
    try:
        n = ecrad.get_version_name(v[:3])
        name.append(n)
        a = 'On' if v[:3] in ecrad.aerosol_on else 'Off'
        aerosol.append(a)
    except ValueError:
        name.append(v)
        aerosol.append('Off')
df_save = df_save.assign(name=name, aerosol=aerosol)
df_save.to_csv(f'{save_path}/HALO-AC3_transmissivity_sw_stats.csv',
               index=False)

# %% plot violinplot with all simulations
sel_ver = ['BACARDI_org', 'BACARDI'] + ecrad_versions
_, axs = plt.subplots(1, 2, figsize=h.figsize_wide,
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df[(df.key == key)
                 & (df.label.isin(sel_ver))]
    sns.violinplot(df_plot, x='values', y='label', hue='label', ax=ax)
    ax.set(xlabel='Solar transmissivity',
           ylabel='',
           xlim=(0.35, 1),
           )
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.95, panel_label[i], transform=ax.transAxes)
    ax.grid()

figname = f'A_HALO_AC3_RF17_RF18_transmissivity_sw_BACARDI_ecRad_boxplot_all.png'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% plot violinplot BACARDI vs ecRad
sel_ver = ['BACARDI', 'v15.1']
_, axs = plt.subplots(2, 1, figsize=(15 * h.cm, 9 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df[(df.key == key)
                 & (df.label.isin(sel_ver))]
    df_plot['label'] = (df_plot['label']
                        .astype('category')
                        .cat.reorder_categories(sel_ver))
    sns.violinplot(df_plot, x='values', y='label', hue='label', ax=ax)
    ax.set(
        xlabel='',
        ylabel='',
        yticklabels=['BACARDI',
                     'ecRad Reference\nsimulation (v15.1)'],
        xlim=(0.35, 1),
    )
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.89, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[1].set(xlabel='Solar transmissivity')
figname = f'05_HALO_AC3_RF17_RF18_transmissivity_sw_BACARDI_ecRad_boxplot_v15.1.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% BACARDI vs ecRad - print stats
df_print = st_stats.loc[pd.IndexSlice[:, ['BACARDI_org'] + sel_ver], :].sort_values('key')
print(df_print)

# %% sea ice - plot violinplot of below cloud transmissivity
sel_ver = ['BACARDI', 'v15.1', 'v13', 'v13.2']
h.set_cb_friendly_colors('petroff_6')
plt.rc('font', size=10)
_, axs = plt.subplots(2, 1, figsize=(15 * h.cm, 10 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df[(df.key == key)
                 & (df.label.isin(sel_ver))]
    df_plot['label'] = (df_plot['label']
                        .astype('category')
                        .cat.reorder_categories(sel_ver))
    sns.violinplot(df_plot, x='values', y='label', hue='label',
                   ax=ax)
    ax.set(xlabel='',
           ylabel='',
           yticklabels='',
           xlim=(0.35, 1),
           )
    ax.set_yticklabels(['BACARDI',
                        'ecRad Reference\nsimulation (v15.1)',
                        'ecRad Open ocean\nsimulation (v13)',
                        'ecRad Measured albedo\nsimulation (v13.2)'], )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.89, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[1].set(
    xlabel='Solar transmissivity'
)

figname = f'05_HALO_AC3_RF17_RF18_transmissivity_sw_BACARDI_ecRad_albedo_violin.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% 3-D effects - plot violinplot of solar transmissivity
sel_ver = ['BACARDI', 'v15.1', 'v22.1']
h.set_cb_friendly_colors('petroff_6')
plt.rc('font', size=10)
_, axs = plt.subplots(2, 1, figsize=(15 * h.cm, 9 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df[(df.key == key)
                 & (df.label.isin(sel_ver))]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='values', y='label', hue='label',
                   order=sel_ver, ax=ax)
    ax.set(
        xlabel='',
        ylabel='',
        yticklabels='',
        xlim=(0.45, 1),
           )
    ax.set_yticklabels(['BACARDI',
                        'ecRad Reference\nFu-IFS (v15.1)',
                        'ecRad 3D on\nFu-IFS (v22.1)',
                        ])
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.89, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[1].set(
    xlabel='Solar transmissivity'
)
figname = f'05_HALO_AC3_RF17_RF18_transmissivity_sw_BACARDI_ecRad_3d_effects_violin.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% 3-D effects - print stats
df_print = st_stats.loc[pd.IndexSlice[:, ['BACARDI_org'] + sel_ver], :].sort_values('key')
print(df_print)

# %% aerosol - plot violinplot of solar transmissivity
sel_ver = ['BACARDI', 'v15.1', 'v30.1']
plt.rc('font', size=10)
_, axs = plt.subplots(2, 1, figsize=(15 * h.cm, 9 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df[(df.key == key)
                 & (df.label.isin(sel_ver))]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='values', y='label', hue='label', ax=ax,
                   order=sel_ver)
    ax.set(xlabel='',
           ylabel='',
           yticklabels='',
           xlim=(0.45, 1),
           )
    ax.set(yticklabels=['BACARDI',
                        'ecRad Reference\nFu-IFS (v15.1)',
                        'ecRad aerosol on\nFu-IFS (v30.1)',
                        ])
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.89, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[1].set(
    xlabel='Solar transmissivity'
)
figname = f'05_HALO_AC3_RF17_RF18_transmissivity_sw_BACARDI_ecRad_aerosol_violin.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% aerosol - plot violinplot of solar transmissivity for all simulations
sel_ver = ['BACARDI', 'v15.1', 'v30.1', 'v19.1', 'v31.1', 'v18.1', 'v32.1']
plt.rc('font', size=10)
_, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 13 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df[(df.key == key)
                 & (df.label.isin(sel_ver))]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='values', y='label', hue='label', ax=ax,
                   order=sel_ver)
    ax.set(xlabel='Solar transmissivity',
           ylabel='',
           yticklabels='',
           xlim=(0.45, 1),
           )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.96, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[0].set(yticklabels=['BACARDI',
                        'ecRad Reference\nFu-IFS (v15.1)',
                        'ecRad aerosol on\nFu-IFS (v30.1)',
                        'ecRad Reference\nYi2013 (v19.1)',
                        'ecRad aerosol on\nYi2013 (v31.1)',
                        'ecRad Reference\nBaran2016 (v18.1)',
                        'ecRad aerosol on\nBaran2016 (v32.1)',
                        ],)
figname = f'A_HALO_AC3_RF17_RF18_transmissivity_sw_BACARDI_ecRad_aerosol_violin_all.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% aerosol - print stats
st_stats[st_stats.index.isin(['BACARDI_org', 'BACARDI', 'v15.1', 'v30.1', 'v31.1', 'v32.1'], level=1)]

# %% ice optics parameterizations - plot violinplot of solar transmissivity
sel_ver = ['BACARDI', 'v15.1', 'v19.1', 'v18.1']
plt.rc('font', size=10)
_, axs = plt.subplots(2, 1, figsize=(15 * h.cm, 10 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df[(df.key == key)
                 & (df.label.isin(sel_ver))]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='values', y='label', hue='label', ax=ax,
                   order=sel_ver)
    ax.set(xlabel='',
           ylabel='',
           yticklabels='',
           xlim=(0.45, 1),
           )
    ax.set(yticklabels=['BACARDI',
                        'ecRad Reference\nFu-IFS (v15.1)',
                        'ecRad Reference\nYi2013 (v19.1)',
                        'ecRad Reference\nBaran2016 (v18.1)',
                        ])
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.89, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[1].set(
    xlabel='Solar transmissivity'
)
figname = f'05_HALO_AC3_RF17_RF18_transmissivity_sw_BACARDI_ecRad_ice_optics_violin.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% ice optics parameterization - print stat
df_print = st_stats.loc[pd.IndexSlice[:, ['BACARDI_org'] + sel_ver], :].sort_values('key')
print(df_print)

# %% ice effective radius - plot reice with and without cosine dependence and using VarCloud IWC as input for case study clouds
plt.rc('font', size=10)
legend_labels = ['Off (IWC IFS)', 'On (IWC IFS)', 'Off (IWC VarCloud)', 'On (IWC VarCloud)']
linestyles = ['solid', 'solid', 'dashed', 'dashed']
binsizes = dict(iwc=1, reice=4)
binedges = dict(iwc=20, reice=100)
text_loc_x = 0.05
text_loc_y = 0.95
ylims = {'iwc': (0, 0.3), 'reice': (0, 0.25)}
_, axs = plt.subplots(1, 2, figsize=(15 * h.cm, 10 * h.cm), layout='constrained')

# left panel - RF17 re_ice
ax = axs[0]
plot_ds = ecrad_orgs['RF17']
sel_time = slices['RF17']['below']
bins = np.arange(0, binedges['reice'], binsizes['reice'])
for i, v in enumerate(['v39.2', 'v15.1', 'v41.2', 'v16.1']):
    pds = plot_ds[v].re_ice.sel(time=sel_time).to_numpy().flatten() * 1e6
    pds = pds[~np.isnan(pds)]
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i],
        color=cbc[i],
        linestyle=linestyles[i],
        histtype='step',
        density=True,
        lw=2,
    )
    print(f'RF17 Mean reice {v}: {pds.mean():.2f}\n'
          f'n={len(pds)}')

ax.grid()
ax.text(text_loc_x, text_loc_y, '(a)',
        transform=ax.transAxes,
        )
ax.set(ylabel='Probability density function',
       xlabel=f'Ice effective radius ({h.plot_units['re_ice']})',
       ylim=ylims['reice'])
ax.set_title('RF 17 - 11 April 2024', fontsize=10)
ax.legend(title='Cosine dependence (input)')

# right panel - RF18 re_ice
ax = axs[1]
plot_ds = ecrad_orgs['RF18']
sel_time = slices['RF18']['below']
bins = np.arange(0, binedges['reice'], binsizes['reice'])
for i, v in enumerate(['v39.2', 'v15.1', 'v41.2', 'v16.1']):
    pds = plot_ds[v].re_ice.sel(time=sel_time).to_numpy().flatten() * 1e6
    pds = pds[~np.isnan(pds)]
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i],
        color=cbc[i],
        linestyle=linestyles[i],
        histtype='step',
        density=True,
        lw=2,
    )
    print(f'RF18 Mean reice {v}: {pds.mean():.2f}\n'
          f'n={len(pds)}')

ax.grid()
ax.text(text_loc_x, text_loc_y, '(b)',
        transform=ax.transAxes,
        )
ax.set(ylabel='',
       xlabel=f'Ice effective radius ({h.plot_units['re_ice']})',
       ylim=ylims['reice'])
ax.set_title('RF 18 - 12 April 2024', fontsize=10)

figname = f'{plot_path}/05_HALO-AC3_HALO_RF17_RF18_IFS_re_ice_pdf_case_studies_cosine_dependence.pdf'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% ice effective radius - plot violinplot of solar transmissivity cosine dependence
sel_ver = ['BACARDI', 'v15.1', 'v39.2']
plt.rc('font', size=10)
_, axs = plt.subplots(2, 1, figsize=(15 * h.cm, 9 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df[(df.key == key)
                 & (df.label.isin(sel_ver))]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='values', y='label', hue='label', ax=ax,
                   order=sel_ver)
    ax.set(xlabel='',
           ylabel='',
           yticklabels='',
           xlim=(0.45, 1),
           )
    ax.set(yticklabels=['BACARDI',
                        'ecRad Reference\nCosine (v15.1)',
                        'ecRad Reference\nNo cosine (v39.2)',
                        ])
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.89, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[1].set(
    xlabel='Solar transmissivity'
)
figname = f'05_HALO_AC3_RF17_RF18_transmissivity_sw_BACARDI_ecRad_no_cosine_violin.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% ice effective radius - print statistics
df_print = st_stats.loc[pd.IndexSlice[:, ['BACARDI_org'] + sel_ver], :].sort_values('key')
print(df_print)

# %% VarCloud - plot PDF of IWC, IWP and reice
plt.rc('font', size=9)
legend_labels = ['VarCloud', 'IFS']
binsizes = dict(iwc=1, tiwp=10, reice=4)
binedges = dict(iwc=20, tiwp=200, reice=100)
text_loc_x = 0.05
text_loc_y = 0.9
ylims = {'iwc': (0, 0.3), 'tiwp': (0, 0.115), 'reice': (0, 0.095)}
_, axs = plt.subplots(3, 2, figsize=(15 * h.cm, 15 * h.cm),
                      layout='constrained')

# upper left panel - RF17 IWC
ax = axs[0, 0]
plot_ds = ecrad_orgs['RF17']
# sel_time = slice(pd.to_datetime('2022-04-11 10:49'), pd.to_datetime('2022-04-11 11:04'))
sel_time = slices['RF17']['below']
bins = np.arange(0, binedges['iwc'], binsizes['iwc'])
for i, v in enumerate(['v36', 'v15.1']):
    if v == 'v36':
        pds = plot_ds[v].iwc
    else:
        iwc, cc = plot_ds[v].iwc.sel(time=sel_time), plot_ds[v].cloud_fraction.sel(time=sel_time)
        pds = iwc.where(cc > 0).where(cc == 0, iwc / cc)

    pds = pds.to_numpy().flatten() * 1e6
    pds = pds[~np.isnan(pds)]
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i] + f' (n={len(pds)})',
        color=cbc[i],
        histtype='step',
        density=True,
        lw=2,
    )
ax.legend()
ax.grid()
ax.text(text_loc_x, text_loc_y, '(a)', transform=ax.transAxes)
ax.set(
    ylabel='Probability density function',
    xlabel=f'Ice water content ({h.plot_units['iwc']})',
    ylim=ylims['iwc'])
ax.set_title('RF 17 - 11 April 2022', fontsize=9)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))

# middle left panel - RF17 IWP
ax = axs[1, 0]
bins = np.arange(0, binedges['tiwp'], binsizes['tiwp'])
for i, v in enumerate(['v36', 'v15.1']):
    if v == 'v36':
        pds = plot_ds[v].tiwp
    else:
        tiwp, cc = plot_ds[v].tiwp.sel(time=sel_time), plot_ds[v].cloud_fraction.sel(time=sel_time)
        pds = tiwp

    pds = pds.to_numpy().flatten() * 1e3
    pds = pds[~np.isnan(pds)]
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i] + f' (n={len(pds)})',
        color=cbc[i],
        histtype='step',
        density=True,
        lw=2,
    )
ax.legend()
ax.grid()
ax.text(text_loc_x, text_loc_y, '(c)', transform=ax.transAxes)
ax.set(
    ylabel='Probability density function',
    xlabel=f'Ice water path ({h.plot_units['iwp']})',
    ylim=ylims['tiwp'],
)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.02))

# lower left panel - RF17 re_ice
ax = axs[2, 0]
bins = np.arange(0, binedges['reice'], binsizes['reice'])
for i, v in enumerate(['v36', 'v15.1']):
    if v == 'v36':
        pds = plot_ds[v].re_ice.to_numpy().flatten() * 1e6
    else:
        pds = plot_ds[v].re_ice.sel(time=sel_time).to_numpy().flatten() * 1e6

    pds = pds[~np.isnan(pds)]
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i] + f'\n(n={len(pds)})',
        color=cbc[i],
        histtype='step',
        density=True,
        lw=2,
    )
    print(f'RF17 Mean reice {v}: {pds.mean():.2f}')
ax.legend()
ax.grid()
ax.text(text_loc_x, text_loc_y, '(e)', transform=ax.transAxes)
ax.set(ylabel='Probability density function',
       xlabel=f'Ice effective radius ({h.plot_units['re_ice']})',
       ylim=ylims['reice'])

# upper right panel - RF18 IWC
ax = axs[0, 1]
plot_ds = ecrad_orgs['RF18']
# sel_time = slice(pd.to_datetime('2022-04-12 11:04'), pd.to_datetime('2022-04-12 11:24'))
sel_time = slices['RF18']['case']
bins = np.arange(0, binedges['iwc'], binsizes['iwc'])
for i, v in enumerate(['v36', 'v15.1']):
    if v == 'v36':
        pds = plot_ds[v].iwc
    else:
        iwc, cc = plot_ds[v].iwc.sel(time=sel_time), plot_ds[v].cloud_fraction.sel(time=sel_time)
        pds = iwc.where(cc > 0).where(cc == 0, iwc / cc)

    pds = pds.to_numpy().flatten() * 1e6
    pds = pds[~np.isnan(pds)]
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i] + f' (n={len(pds)})',
        color=cbc[i],
        histtype='step',
        density=True,
        lw=2,
    )
ax.legend()
ax.grid()
ax.text(text_loc_x, text_loc_y, '(b)', transform=ax.transAxes)
ax.set(
    ylabel='',
    xlabel=f'Ice water content ({h.plot_units['iwc']})',
    ylim=ylims['iwc'])
ax.set_title('RF 18 - 12 April 2022', fontsize=9)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))

# middle right panel - RF18 IWP
ax = axs[1, 1]
bins = np.arange(0, binedges['tiwp'], binsizes['tiwp'])
for i, v in enumerate(['v36', 'v15.1']):
    if v == 'v36':
        pds = plot_ds[v].tiwp
    else:
        tiwp, cc = plot_ds[v].tiwp.sel(time=sel_time), plot_ds[v].cloud_fraction.sel(time=sel_time)
        pds = tiwp

    pds = pds.to_numpy().flatten() * 1e3
    pds = pds[~np.isnan(pds)]
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i] + f' (n={len(pds)})',
        color=cbc[i],
        histtype='step',
        density=True,
        lw=2,
    )
ax.legend()
ax.grid()
ax.text(text_loc_x, text_loc_y, '(d)', transform=ax.transAxes)
ax.set(
   ylabel='',
   xlabel=f'Ice water path ({h.plot_units['iwp']})',
   ylim=ylims['tiwp'],
   )
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.02))

# lower right panel - RF18 re_ice
ax = axs[2, 1]
bins = np.arange(0, binedges['reice'], binsizes['reice'])
for i, v in enumerate(['v36', 'v15.1']):
    if v == 'v36':
        pds = plot_ds[v].re_ice.to_numpy().flatten() * 1e6
    else:
        pds = plot_ds[v].re_ice.sel(time=sel_time).to_numpy().flatten() * 1e6
    pds = pds[~np.isnan(pds)]
    ax.hist(
        pds,
        bins=bins,
        label=legend_labels[i] + f' (n={len(pds)})',
        color=cbc[i],
        histtype='step',
        density=True,
        lw=2,
    )
    print(f'RF18 Mean reice {v}: {pds.mean():.2f}')
ax.legend()
ax.grid()
ax.text(text_loc_x, text_loc_y, '(f)', transform=ax.transAxes)
ax.set(ylabel='',
       xlabel=f'Ice effective radius ({h.plot_units['re_ice']})',
       ylim=ylims['reice'])

figname = f'{plot_path}/05_HALO-AC3_HALO_RF17_RF18_IFS_iwc_iwp_re_ice_pdf_case_studies.pdf'
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% VarCloud - plot violinplot of solar transmissivity
sel_ver = ['BACARDI_1s', 'v36', 'v37', 'v38']
plt.rc('font', size=10)
_, axs = plt.subplots(2, 1, figsize=(15 * h.cm, 10 * h.cm),
                      layout='constrained')
for i, key in enumerate(keys):
    ax = axs[i]
    df_plot = df[(df.key == key)
                 & (df.label.isin(sel_ver))]
    df_plot['label'] = df_plot['label'].astype('category')
    sns.violinplot(df_plot, x='values', y='label', hue='label', ax=ax,
                   order=sel_ver)
    ax.set(xlabel='',
           ylabel='',
           yticklabels='',
           xlim=(0.45, 1),
           )
    ax.set(yticklabels=['BACARDI',
                        'ecRad VarCloud\nFu-IFS (v36)',
                        'ecRad VarCloud\nYi2013 (v37)',
                        'ecRad VarCloud\nBaran2016 (v38)',
                        ])
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.set_title(key.replace('1', ' 1') + ' - ' + date_title[i],
                 fontsize=10)
    ax.text(0.01, 0.89, panel_label[i], transform=ax.transAxes)
    ax.grid()

axs[1].set(
    xlabel='Solar transmissivity'
)
figname = f'05_HALO_AC3_RF17_RF18_transmissivity_sw_BACARDI_ecRad_varcloud_violin.pdf'
plt.savefig(f'{plot_path}/{figname}', dpi=300)
plt.show()
plt.close()

# %% testing
plot_ds[var].plot(x='time')
plt.show()
plt.close()
