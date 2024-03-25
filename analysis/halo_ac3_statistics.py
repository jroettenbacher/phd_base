#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 05.02.2024

Create a data frame with statistics of different variables for the above and below cloud sections of RF 17 and RF 18
"""
# %% import modules
import pylim.helpers as h
import pylim.halo_ac3 as meta
from pylim import ecrad
import ac3airborne
from ac3airborne.tools import flightphase
import numpy as np
import pandas as pd
import xarray as xr

# %% set paths
campaign = 'halo-ac3'
outpath = h.get_path('plot', campaign=campaign)
keys = ['RF17', 'RF18']
ecrad_versions = [f'v{x}' for x in [13, 13.1, 13.2, 15.1, 18.1, 19.1, 30.1, 31.1, 32.1]
                  + np.arange(15, 30).tolist()
                  + np.arange(33, 39).tolist()]
ecrad_versions.sort()

# %% read in data
(
    bahamas_ds, bacardi_ds, ecrad_dicts, above_clouds,
    below_clouds, slices, ecrad_orgs
) = (dict(), dict(), dict(), dict(), dict(), dict(), dict())

for key in keys:
    flight = meta.flight_names[key]
    date = flight[9:17]
    bacardi_path = h.get_path('bacardi', flight, campaign)
    bahamas_path = h.get_path('bahamas', flight, campaign)
    libradtran_path = h.get_path('libradtran', flight, campaign)
    libradtran_exp_path = h.get_path('libradtran_exp', flight, campaign)
    ecrad_path = f'{h.get_path("ecrad", flight, campaign)}/{date}'

    # filenames
    bahamas_file = f'HALO-AC3_HALO_BAHAMAS_{date}_{key}_v1_JR.nc'
    bacardi_file = f'HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1_JR.nc'
    libradtran_bb_solar_si = f'HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_solar_si_{date}_{key}.nc'
    libradtran_bb_thermal_si = f'HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_thermal_si_{date}_{key}.nc'

    # read in aircraft data
    bahamas_ds[key] = xr.open_dataset(f'{bahamas_path}/{bahamas_file}')
    bacardi = xr.open_dataset(f'{bacardi_path}/{bacardi_file}')

    # read in libRadtran simulation
    bb_sim_solar_si = xr.open_dataset(f'{libradtran_exp_path}/{libradtran_bb_solar_si}')
    bb_sim_thermal_si = xr.open_dataset(f'{libradtran_exp_path}/{libradtran_bb_thermal_si}')
    # interpolate simualtion onto BACARDI time
    bb_sim_solar_si_inp = bb_sim_solar_si.interp(time=bacardi.time)
    bb_sim_thermal_si_inp = bb_sim_thermal_si.interp(time=bacardi.time)

    # calculate transmissivity BACARDI/libRadtran
    bacardi['F_down_solar_sim_si'] = bb_sim_solar_si_inp.fdw
    bacardi['F_down_terrestrial_sim_si'] = bb_sim_thermal_si_inp.edn
    bacardi['F_up_terrestrial_sim_si'] = bb_sim_thermal_si_inp.eup
    bacardi['transmissivity_solar'] = bacardi['F_down_solar'] / bb_sim_solar_si_inp.fdw
    bacardi['transmissivity_terrestrial'] = bacardi['F_down_terrestrial'] / bb_sim_thermal_si_inp.edn
    # calculate reflectivity
    bacardi['reflectivity_solar'] = bacardi['F_up_solar'] / bacardi['F_down_solar']
    bacardi['reflectivity_terrestrial'] = bacardi['F_up_terrestrial'] / bacardi['F_down_terrestrial']

    # calculate radiative effect from BACARDI and libRadtran sea ice simulation
    bacardi['F_net_solar'] = bacardi['F_down_solar'] - bacardi['F_up_solar']
    bacardi['F_net_terrestrial'] = bacardi['F_down_terrestrial'] - bacardi['F_up_terrestrial']
    bacardi['F_net_solar_sim'] = bb_sim_solar_si_inp.fdw - bb_sim_solar_si_inp.eup
    bacardi['F_net_terrestrial_sim'] = bb_sim_thermal_si_inp.edn - bb_sim_thermal_si_inp.eup
    bacardi['CRE_solar'] = bacardi['F_net_solar'] - bacardi['F_net_solar_sim']
    bacardi['CRE_terrestrial'] = bacardi['F_net_terrestrial'] - bacardi['F_net_terrestrial_sim']
    bacardi['CRE_total'] = bacardi['CRE_solar'] + bacardi['CRE_terrestrial']
    bacardi['F_down_solar_error'] = np.abs(bacardi['F_down_solar'] * 0.03)
    # normalize downward irradiance for cos SZA
    for var in ['F_down_solar', 'F_down_solar_diff']:
        bacardi[f'{var}_norm'] = bacardi[var] / np.cos(np.deg2rad(bacardi['sza']))
    # filter data for motion flag
    bacardi_org = bacardi.copy()
    bacardi = bacardi.where(bacardi.motion_flag)
    # overwrite variables which do not need to be filtered with original values
    for var in ['alt', 'lat', 'lon', 'sza', 'saa', 'attitude_flag', 'segment_flag', 'motion_flag']:
        bacardi[var] = bacardi_org[var]

    # read in ecrad data
    ecrad_dict, ecrad_org = dict(), dict()

    for k in ecrad_versions:
        ds = xr.open_dataset(f'{ecrad_path}/ecrad_merged_inout_{date}_{k}.nc')
        ecrad_org[k] = ds.copy(deep=True)
        # select only center column for direct comparisons
        ds = ds.sel(column=0, drop=True) if 'column' in ds.dims else ds
        ecrad_dict[k] = ds.copy(deep=True)

    ecrad_dicts[key] = ecrad_dict
    ecrad_orgs[key] = ecrad_org

    # interpolate standard ecRad simulation onto BACARDI time
    bacardi['ecrad_fdw'] = (ecrad_dict['v15.1']
                            .flux_dn_sw_clear
                            .interp(time=bacardi.time,
                                    kwargs={'fill_value': 'extrapolate'}))
    # calculate transmissivity using ecRad at TOA and above cloud
    bacardi['transmissivity_TOA'] = bacardi['F_down_solar'] / bacardi['ecrad_fdw'].isel(half_level=0)
    bacardi['transmissivity_above_cloud'] = bacardi['F_down_solar'] / bacardi['ecrad_fdw'].isel(half_level=73)
    bacardi_ds[key] = bacardi

    # get flight segmentation and select below and above cloud section
    fl_segments = ac3airborne.get_flight_segments()['HALO-AC3']['HALO'][f'HALO-AC3_HALO_{key}']
    segments = flightphase.FlightPhaseFile(fl_segments)
    above_cloud, below_cloud = dict(), dict()
    if key == 'RF17':
        above_cloud['start'] = segments.select('name', 'high level 7')[0]['start']
        above_cloud['end'] = segments.select('name', 'high level 8')[0]['end']
        below_cloud['start'] = segments.select('name', 'high level 9')[0]['start']
        below_cloud['end'] = segments.select('name', 'high level 10')[0]['end']
        above_slice = slice(above_cloud['start'], above_cloud['end'])
        below_slice = slice(pd.to_datetime('2022-04-11 11:35'), below_cloud['end'])
        case_slice = slice(above_cloud['start'], below_cloud['end'])
    else:
        above_cloud['start'] = segments.select('name', 'polygon pattern 1')[0]['start']
        above_cloud['end'] = segments.select('name', 'polygon pattern 1')[0]['parts'][-1]['start']
        below_cloud['start'] = segments.select('name', 'polygon pattern 2')[0]['start']
        below_cloud['end'] = segments.select('name', 'polygon pattern 2')[0]['end']
        above_slice = slice(above_cloud['start'], above_cloud['end'])
        below_slice = slice(below_cloud['start'], below_cloud['end'])
        case_slice = slice(above_cloud['start'], below_cloud['end'])

    above_clouds[key] = above_cloud
    below_clouds[key] = below_cloud
    slices[key] = dict(case=case_slice, above=above_slice, below=below_slice)

# %% print time between above and below cloud section
print(f'Time between above and below cloud section')
for key in keys:
    below = below_clouds[key]
    above = above_clouds[key]
    print(f'{key}: {below['start'] - above['end']} HH:MM:SS')
    print(f'{key}: Case study duration: {above['start']} - {below['end']}')

# %% print start location and most northerly point of above cloud leg
for key in keys:
    tmp = bahamas_ds[key].sel(time=slices[key]['above'])[['IRS_LAT', 'IRS_LON']]
    print(f'Start position of above cloud leg for {key}:\n'
          f'Latitude, Longitude: {tmp.IRS_LAT.isel(time=0):.2f}°N, {tmp.IRS_LON.isel(time=0):.2f}°W\n')
    if key == 'RF18':
        max_lat_i = np.argmax(tmp.IRS_LAT.to_numpy())
        print(f'Most northerly location of pentagon:\n'
              f'{tmp.IRS_LAT.isel(time=max_lat_i):.2f}°N, {tmp.IRS_LON.isel(time=max_lat_i):.2f}°W')

# %% print range/change of solar zenith angle during case study
for key in keys:
    tmp = bacardi_ds[key].sel(time=slices[key]['case'])
    print(f'Range of solar zenith angle during case study of {key}: {tmp.sza.min():.2f} - {tmp.sza.max():.2f}')
    tmp = bacardi_ds[key].sel(time=slices[key]['above'])
    print(f'Change of solar zenith angle during above cloud section of {key}: {tmp.sza[0]:.2f} - {tmp.sza[-1]:.2f}')


# %% calculate statistics from BACARDI
not_bacardi_vars = ['alt', 'lat', 'lon', 'sza', 'saa', 'attitude_flag',
                    'segment_flag', 'motion_flag']
bacardi_vars = [var for var in bacardi_ds['RF17'].data_vars if var not in not_bacardi_vars]
bacardi_stats = list()
for key in keys:
    for var in bacardi_vars:
        for section in ['above', 'below']:
            time_sel = slices[key][section]
            ds = bacardi_ds[key][var].sel(time=time_sel)
            bacardi_min = np.min(ds).to_numpy()
            bacardi_max = np.max(ds).to_numpy()
            bacardi_spread = bacardi_max - bacardi_min
            bacardi_mean = ds.mean().to_numpy()
            bacardi_median = ds.median().to_numpy()
            bacardi_std = ds.std().to_numpy()
            bacardi_stats.append(
                ('v1', 'BACARDI', key, var, section,
                 bacardi_min, bacardi_max, bacardi_spread,
                 bacardi_mean, bacardi_median, bacardi_std))

# %% calculate statistics from ecRad
ecrad_stats = list()
ecrad_vars = ['flux_up_lw', 'flux_dn_lw', 'flux_up_lw_clear', 'flux_dn_lw_clear',
              'flux_up_sw', 'flux_dn_sw', 'flux_up_sw_clear', 'flux_dn_sw_clear',
              'flux_dn_direct_sw_clear',
              'flux_net_sw', 'flux_net_lw',
              'transmissivity_sw_toa', 'transmissivity_sw_above_cloud',
              'reflectivity_sw', 'reflectivity_lw',
              'cre_sw', 'cre_lw', 'cre_total']
for version in ecrad_versions:
    v_name = ecrad.get_version_name(version[:3])
    for key in keys:
        height_sel = ecrad_dicts[key][version]['aircraft_level']
        ecrad_ds = ecrad_dicts[key][version].isel(half_level=height_sel)
        for evar in ecrad_vars:
            for section in ['above', 'below']:
                time_sel = slices[key][section]
                eds = ecrad_ds[evar].sel(time=time_sel)
                try:
                    ecrad_min = np.min(eds).to_numpy()
                    ecrad_max = np.max(eds).to_numpy()
                    ecrad_spread = ecrad_max - ecrad_min
                    ecrad_mean = eds.mean().to_numpy()
                    ecrad_median = eds.median().to_numpy()
                    ecrad_std = eds.std().to_numpy()
                    ecrad_stats.append(
                        (version, v_name, key, evar, section,
                         ecrad_min, ecrad_max, ecrad_spread,
                         ecrad_mean, ecrad_median, ecrad_std)
                    )
                except ValueError:
                    ecrad_stats.append(
                        (version, v_name, key, evar, section,
                         np.nan, np.nan, np.nan, np.nan, np.nan)
                    )

# %% convert statistics to dataframe
columns = ['version', 'source', 'key', 'variable', 'section', 'min', 'max', 'spread', 'mean', 'median', 'std']
ecrad_df = pd.DataFrame(ecrad_stats, columns=columns)
bacardi_df = pd.DataFrame(bacardi_stats, columns=columns)
df = pd.concat([ecrad_df, bacardi_df]).reset_index(drop=True)
df.to_csv(f'{outpath}/{campaign}_bacardi_ecrad_statistics.csv', index=False)
