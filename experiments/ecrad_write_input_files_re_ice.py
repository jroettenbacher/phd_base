#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 06-07-2023

Use a processed IFS output file on a O1280 grid and generate several ecRad input files for one time step with differing re_ice values.

**Required User Input:**

All options can be set in the script or given as command line key=value pairs.
The first possible option is the default.

* campaign, (default: 'halo-ac3')
* key, flight key (default: 'RF17')
* init_time, initalization time of IFS run (00, 12, yesterday)
* o3_source, which ozone concentration to use? (one of '47r1', 'ifs', 'constant', 'sonde')
* trace_gas_source, which trace gas concentrations to use? (one of '47r1', 'constant')
* aerosol_source, which aerosol concentrations to use? (one of '47r1', 'ADS')

**Output:**

* well documented ecRad input files in netCDF format for one time step with differing re_ice values

"""

if __name__ == '__main__':
    # %% module import
    import pylim.helpers as h
    import pylim.meteorological_formulas as met
    from pylim.ecrad import apply_liquid_effective_radius, apply_ice_effective_radius
    import numpy as np
    from sklearn.neighbors import BallTree
    import xarray as xr
    import os
    import pandas as pd
    import time
    from tqdm import tqdm

    start = time.time()
    g = 9.80665  # acceleration due to gravity [m s-2]

    # %% read in command line arguments
    args = h.read_command_line_args()
    campaign = args['campaign'] if 'campaign' in args else 'halo-ac3'
    key = args['key'] if 'key' in args else 'RF18'
    init_time = args['init'] if 'init' in args else '00'
    o3_source = args['o3_source'] if 'o3_source' in args else '47r1'
    trace_gas_source = args['trace_gas_source'] if 'trace_gas_source' in args else '47r1'
    aerosol_source = args['aerosol_source'] if 'aerosol_source' in args else '47r1'
    filter_low_clouds = h.strtobool(args['filter_low_clouds']) if 'filter_low_clouds' in args else True

    if campaign == 'halo-ac3':
        import pylim.halo_ac3 as meta

        flight = meta.flight_names[key]
        date = flight[9:17]
    else:
        import pylim.cirrus_hl as meta

        flight = key
        date = flight[7:15]

    # setup logging
    __file__ = None if '__file__' not in locals() else __file__
    log = h.setup_logging('./logs', __file__, key)
    # print options to user
    log.info(f'Options set: \ncampaign: {campaign}\nkey: {key}\nflight: {flight}\ndate: {date}\n'
             f'init time: {init_time}\n'
             f'O3 source: {o3_source}\nTrace gas source: {trace_gas_source}\n'
             f'Aerosol source: {aerosol_source}\n'
             f'Filter low level clouds: {filter_low_clouds}\n'
             )

    # %% set paths
    ifs_path = os.path.join(h.get_path('ifs', campaign=campaign), date)
    ecrad_path = os.path.join(h.get_path('ecrad', campaign=campaign), 'reice_sensitivity', f'ecrad_input_{key}')
    cams_path = h.get_path('cams', campaign=campaign)
    trace_gas_file = f'trace_gas_mm_climatology_2020_{trace_gas_source}_{date}.nc'
    aerosol_file = f'aerosol_mm_climatology_2020_{aerosol_source}_{date}.nc'

    # create output path
    os.makedirs(ecrad_path, exist_ok=True)

    # %% read in intermediate files from read_ifs
    if init_time == 'yesterday':
        ifs_date = int(date) - 1
        init_time = 12
    else:
        ifs_date = date

    nav_data_ip = pd.read_csv(f'{ifs_path}/nav_data_ip_{date}.csv', index_col='time', parse_dates=True)
    data_ml = xr.open_dataset(f'{ifs_path}/ifs_{ifs_date}_{init_time}_ml_O1280_processed.nc')
    data_ml = data_ml.set_index(rgrid=['lat', 'lon'])

    # %% select below cloud time step
    if key == 'RF17':
        timestamp_str = '2022-04-11 11:40'
        timestamp = pd.Timestamp(timestamp_str)
    else:
        timestamp_str = '2022-04-12 12:00'
        timestamp = pd.Timestamp(timestamp_str)
    data_ml = data_ml.sel(time=timestamp, method='nearest')
    nav_data_ip = nav_data_ip.loc[timestamp_str]

    # %% read in trace gas and aerosol data
    trace_gas = xr.open_dataset(f'{cams_path}/{trace_gas_file}')
    aerosol = xr.open_dataset(f'{cams_path}/{aerosol_file}')

    # %% find closest grid point along flight track
    ifs_lat_lon = np.column_stack((data_ml.lat, data_ml.lon))
    ifs_tree = BallTree(np.deg2rad(ifs_lat_lon), metric='haversine')
    # generate an array with lat, lon values from the flight position
    points = np.deg2rad(np.column_stack((nav_data_ip.lat, nav_data_ip.lon)))
    dist, idxs = ifs_tree.query(points, k=1)  # query the tree
    closest_latlons = ifs_lat_lon[idxs].flatten()
    # a sphere with radius 1 is assumed so multiplying by Earth's radius gives the distance in km
    distances = dist.flatten() * 6371

    # %% filter low clouds according to ECMWF low cloud criterion (pressure higher than 0.8 * surface pressure)
    if filter_low_clouds:
        cloud_data = data_ml[['q_liquid', 'q_ice', 'cloud_fraction', 'clwc', 'ciwc', 'crwc', 'cswc']]
        pressure_filter = data_ml.pressure_full.sel(level=137) * 0.8
        low_cloud_filter = data_ml.pressure_full < pressure_filter  # False for low clouds
        cloud_data = cloud_data.where(low_cloud_filter, 0)  # replace where False with 0
        data_ml.update(cloud_data)

    # %% calculate sw_albedo_direct to account for direct reflection of solar incoming radiation above ocean and sea ice
    ci_albedo_direct = met.calculate_direct_sea_ice_albedo_ebert(nav_data_ip.cos_sza)
    # create xr.DataArray with open ocean albedo after Taylor et al. 1996 for all spectral bands
    open_ocean_albedo_taylor = (xr.DataArray(nav_data_ip.open_ocean_albedo_taylor)
                                .expand_dims(sw_albedo_band=len(ci_albedo_direct))
                                )

    # %% select one time step and set variables
    latlon_sel = (closest_latlons[0], closest_latlons[1])
    ds = data_ml.sel(rgrid=latlon_sel)
    sod = nav_data_ip.seconds

    ds['cos_solar_zenith_angle'] = xr.DataArray(nav_data_ip.cos_sza,
                                                attrs=dict(unit='1',
                                                           long_name='Cosine of the solar zenith angle'))

    # add sw_albedo_direct
    if np.isnan(ds.ci):
        sw_albedo_direct = xr.full_like(ds.ci, 0.2)  # constant direct surface albedo for land areas
    else:
        sw_albedo_direct = (ds.ci * ci_albedo_direct
                            + (1. - ds.ci) * open_ocean_albedo_taylor)
    sw_albedo_direct.attrs = dict(unit=1, long_name='Banded direct short wave albedo')
    ds['sw_albedo_direct'] = sw_albedo_direct

    # interpolate trace gas data onto ifs full pressure levels
    new_pressure = ds.pressure_full.to_numpy()
    tg = (trace_gas
          .sel(time=timestamp)
          .interp(level=new_pressure,
                  kwargs={'fill_value': 0}))

    # read out trace gases from trace gas file
    tg_vars = ['cfc11_vmr', 'cfc12_vmr', 'ch4_vmr', 'co2_vmr', 'n2o_vmr', 'o3_vmr']
    for var in tg_vars:
        ds[var] = tg[var].assign_coords(level=ds.level)

    # overwrite the trace gases with the variables corresponding to trace_gas_source
    if trace_gas_source == 'constant':
        for var in tg_vars:
            ds[var] = ds[f'{var}_{trace_gas_source}']
    # overwrite ozone according to o3_source
    if o3_source != '47r1':
        ds['o3_vmr'] = ds[f'o3_vmr_{o3_source}']

    # calculate pressure difference between each level,
    # use half level pressure as the full level pressure corresponds to the pressure at the center of the layer
    delta_p = np.diff(ds['pressure_hl'])
    # calculate layer mass
    layer_mass = delta_p / g
    # interpolate aerosol dataset to ifs full pressure levels,
    # and turn it into a data array with one new dimension: aer_type
    aerosol_kgm2 = (aerosol
                    .sel(time=timestamp)
                    .assign(level=(aerosol
                                   .sel(time=timestamp)['full_level_pressure']
                                   .to_numpy()))
                    .interp(level=new_pressure,
                            kwargs={'fill_value': 0})
                    .drop_vars(['half_level_pressure', 'full_level_pressure',
                                'half_level_delta_pressure'])
                    .to_array(dim='aer_type')
                    .assign_coords(aer_type=np.arange(1, 12),
                                   level=ds.level)
                    .reset_coords('time', drop=True))
    # convert layer integrated mass to mass mixing ratio
    aerosol_mmr = aerosol_kgm2 / layer_mass
    aerosol_mmr.attrs = dict(units='kg kg-1',
                             long_name='Aerosol mass mixing ratio',
                             short_name='aerosol_mmr',
                             comment='Aerosol MMR converted from layer-integrated mass for 11 species.\n'
                                     '1: Sea salt, bin 1, 0.03-0.5 micron, OPAC\n'
                                     '2: Sea salt, bin 2, 0.50-5.0 micron, OPAC\n'
                                     '3: Sea salt, bin 3, 5.0-20.0 micron, OPAC\n'
                                     '4: Desert dust, bin 1, 0.03-0.55 micron, (SW) Dubovik et al. 2002 (LW) Fouquart et al. 1987\n'
                                     '5: Desert dust, bin 2, 0.55-0.90 micron, (SW) Dubovik et al. 2002 (LW) Fouquart et al. 1987\n'
                                     '6: Desert dust, bin 3, 0.90-20.0 micron, (SW) Dubovik et al. 2002 (LW) Fouquart et al. 1987\n'
                                     '7: Hydrophilic organic matter, OPAC\n'
                                     '8: Hydrophobic organic matter, OPAC (hydrophilic at RH=20%)\n'
                                     '9: Black carbon, OPAC\n'
                                     '10: Black carbon, OPAC\n'
                                     '11: Stratospheric sulfate (hydrophilic ammonium sulfate at RH 20%-30%',
                             )
    # TODO: Change description to match actual species and not only the radiative description
    ds['aerosol_mmr'] = aerosol_mmr

    # calculate effective radius for all levels
    ds = apply_liquid_effective_radius(ds)
    ds = apply_ice_effective_radius(ds)

    # turn lat, lon, time into variables for cleaner output and to avoid later problems when merging data
    ds = ds.reset_coords(['rgrid', 'lat', 'lon', 'time']).drop_dims('reduced_points')
    ds = ds.expand_dims('column')  # expand dims to add column
    ds = ds.drop_vars('rgrid')
    # some variables should not have column as a dimension
    variables = ['o2_vmr']
    for var in variables:
        ds[var] = ds[var].sel(column=0)
    # add distance to aircraft location for each point
    ds['distance'] = xr.DataArray(distances, dims='column',
                                  attrs=dict(long_name='distance', units='km',
                                             description='Haversine distance to aircraft location'))

    # write one reference file
    ds = ds.transpose('column', ...)  # move column to the first dimension
    ds = ds.astype(np.float32)  # change type from double to float32

    ds.to_netcdf(
        path=f'{ecrad_path}/ecrad_input_standard_{sod:7.1f}_sod_0.nc',
        format='NETCDF4_CLASSIC')

# %% loop through reice values
    for value in tqdm(np.arange(13, 101)):
        ds['re_ice'] = ds.re_ice.where(ds.re_ice == 5.1961601e-05, value * 1e-6)
        ds = ds.transpose('column', ...)  # move column to the first dimension
        ds = ds.astype(np.float32)  # change type from double to float32

        ds.to_netcdf(
            path=f'{ecrad_path}/ecrad_input_standard_{sod}_sod_{value}.nc',
            format='NETCDF4_CLASSIC')

    log.info(f'Done with writing ecRad input files: {pd.to_timedelta((time.time() - start), unit='second')} (hr:min:sec)')
