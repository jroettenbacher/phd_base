#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 09-04-2023

 and generate one ecRad input file for sensitivity studies

**Output:**

* well documented ecRad input file in netCDF format

"""

if __name__ == "__main__":
    # %% module import
    import pylim.helpers as h
    import pylim.meteorological_formulas as met
    from pylim.ecrad import apply_ice_effective_radius, apply_liquid_effective_radius
    import metpy.constants as constants
    from metpy.units import units as u
    import numpy as np
    import os
    import pandas as pd
    import time
    import xarray as xr

    start = time.time()

    # %% read in command line arguments
    args = h.read_command_line_args()

    # setup logging
    __file__ = None if '__file__' not in locals() else __file__
    log = h.setup_logging('./logs', __file__)

    # %% set paths
    if os.getcwd().startswith('C'):
        path = 'E:/ecrad_sensitivity_studies'
    else:
        path = '/projekt_agmwend/home_rad/jroettenbacher/ecrad_sensitivity_studies'

    ecrad_path = f'{path}/ecrad_input'
    atm_file = 'afglsw.dat'

    # %% set constants for conversion from cm-3 to kg kg-1
    NA = 6.02214076e23 * u('mol-1')  # Avogadro constant
    air_M = constants.Md  # Molar mass of dry air
    H_M = 1.00794e-3 * u('kg/mol')  # Molar mass hydrogen
    C_M = 12.0107e-3 * u('kg/mol')  # Molar mass carbon
    O_M = 15.9994e-3 * u('kg/mol')  # Molar mass oxygen
    N_M = 14.00674e-3 * u('kg/mol')  # Molar mass nitrogen
    O2_M = 2 * O_M  # Molar mass of oxygen
    O3_M = 3 * O_M  # Molar mass of ozone
    H2O_M = 2 * H_M + O_M  # Molar mass of water
    CO2_M = C_M + 2 * O_M  # Molar mass of carbon dioxide
    NO2_M = N_M + 2 * O_M  # Molar mass of Nitrogendioxide

    # read in atmosphere file
    atm = pd.read_table(f'{path}/{atm_file}', skiprows=1, header=0,
                        skipinitialspace=True, sep=r'\s+')

    # %% convert cm-3 to kg kg-1
    atm_new = atm.iloc[:, :3].copy(deep=True)
    atm_new['air(kg)'] = atm['air(cm-3)'] * u('cm-3') * air_M / NA
    atm_new['o3_mmr'] = atm['o3(cm-3)'] * u('cm-3') * O3_M / NA / atm_new['air(kg)']
    atm_new['o2_mmr'] = atm['o2(cm-3)'] * u('cm-3') * O2_M / NA / atm_new['air(kg)']
    atm_new['h2o_mmr'] = atm['h2o(cm-3)'] * u('cm-3') * H2O_M / NA / atm_new['air(kg)']
    atm_new['co2_mmr'] = atm['co2(cm-3)'] * u('cm-3') * CO2_M / NA / atm_new['air(kg)']
    atm_new['no2_mmr'] = atm['no2(cm-3)'] * u('cm-3') * NO2_M / NA / atm_new['air(kg)']

    # %% calculate pressure and temperature on full levels
    s = atm_new['p(mb)']
    for i in np.arange(0.5, len(atm_new.index) - 1):
        s[i] = np.nan  # fill every second value with nan to interpolate over it
    pressure_full = (s
                     .sort_index()
                     .reset_index(drop=True)
                     .interpolate(method='cubic')[1::2])
    s = atm_new['T(K)']
    for i in np.arange(0.5, len(atm_new.index) - 1):
        s[i] = np.nan  # fill every second value with nan to interpolate over it
    t = (s
         .sort_index()
         .reset_index(drop=True)
         .interpolate(method='cubic')[1::2])

    # %% set albedo and cos sza
    sw_albedo = (h.ci_albedo_da
                 .interp(time=f'2022-04-11')  # interpolate sea ice albedo to date
                 .drop_vars('time')
                 .expand_dims('column'))
    sw_albedo.attrs = dict(units=1, long_name="Banded short wave albedo")
    cos_sza = np.cos(np.deg2rad(80))

    # %% experiment settings
    # define artificial ice cloud
    cbase = 5  # cloud base (km)
    ctop = 8  # cloud top (km)
    q_ice_set = [0.5e-5, 1e-5, 1e-4, 1e-3]  # ice water mixing ratio (kg kg-1)
    re_ice_set = [15e-6]  # ice effective radius (m)
    latitude = [80, 60, 30]  # latitude (deg N)
    for lat in latitude:
        for qice in q_ice_set:
            # %% define cloud in atmosphere
            cond = (atm_new['z(km)'] > cbase) & (atm_new['z(km)'] < ctop)
            q_ice = pd.Series(0, index=atm_new.index)
            q_ice = q_ice.where(~cond, qice)
            re_ice = q_ice.where(~cond, re_ice_set[0])
            cloud_fraction = q_ice.where(~cond, 1)

            # %% create DataSet with relevant variables for ecRad
            ds = xr.Dataset(
                {
                    'altitude': (['column', 'half_level'], [atm_new['z(km)']], {'units': 'km'}),
                    'skin_temperature': (['column'], [253], {'units': 'K'}),
                    'cos_solar_zenith_angle': (['column'], [cos_sza], dict(units='1')),
                    'sw_albedo': sw_albedo,
                    'sw_albedo_direct': met.calculate_direct_sea_ice_albedo_ebert(cos_sza).expand_dims('column'),
                    'lw_emissivity': (['column', 'lw_emiss_band'], [[0.98, 0.99]],
                                      dict(units="1", long_name="Longwave surface emissivity")),
                    'pressure_full': (['column', 'level'], [pressure_full * 100],
                                      dict(units='Pa', long_name='Pressure on full levels')),
                    'pressure_hl': (['column', 'half_level'], [atm_new['p(mb)'] * 100],
                                    dict(units='Pa', long_name='Pressure on half levels')),
                    't': (['column', 'level'], [t], dict(units='K', long_name='Temperature on full levels')),
                    'temperature_hl': (['column', 'half_level'], [atm_new['T(K)']],
                                       dict(units='K', long_name='Temperature on half levels')),
                    'q': (['column', 'level'], [atm_new['h2o_mmr'].iloc[:-1]],
                                dict(units='kg kg-1', long_name='Specific humidity')),
                    'o3_mmr': (['column', 'level'], [atm_new['o3_mmr'].iloc[:-1]],
                               dict(units='kg kg-1', long_name='Ozone mass mixing ratio')),
                    'q_liquid': (['column', 'level'], [np.repeat(0, len(atm_new.index) - 1)],
                                 dict(units='kg kg-1', long_name='Liquid cloud mass mixing ratio')),
                    'clwc': (['column', 'level'], [np.repeat(0, len(atm_new.index) - 1)],
                             dict(units='kg kg-1', long_name='Cloud liquid water content')),
                    'crwc': (['column', 'level'], [np.repeat(0, len(atm_new.index) - 1)],
                             dict(units='kg kg-1', long_name='Cloud rain water content')),
                    're_liquid': (['column', 'level'], [np.repeat(0, len(atm_new.index) - 1)],
                                  dict(units='m', long_name='Liquid cloud effective radius')),
                    'q_ice': (['column', 'level'], [q_ice.iloc[:-1]],
                              dict(units='kg kg-1', long_name='Ice cloud mass mixing ratio')),
                    'ciwc': (['column', 'level'], [q_ice.iloc[:-1]],
                             dict(units='kg kg-1', long_name='Cloud ice water content')),
                    'cswc': (['column', 'level'], [np.repeat(0, len(atm_new.index) - 1)],
                             dict(units='kg kg-1', long_name='Cloud snow water content')),
                    're_ice': (['column', 'level'], [re_ice.iloc[:-1]],
                               dict(units='m', long_name='Ice cloud effective radius')),
                    'cloud_fraction': (['column', 'level'], [cloud_fraction.iloc[:-1]],
                                       dict(units='1', long_name='Cloud fraction')),
                    'fractional_std': (['column'], [1]),
                    'lat': (['column'], [lat], dict(units='degree North', long_name='Latitude')),
                },
                attrs={'settings': f'q_ice={qice}, '
                                   f're_ice={re_ice_set[0]}, '
                                   f'latitude={lat}'})

            # %% calculate effective radius for all levels
            ds = apply_ice_effective_radius(ds)
            ds.re_ice.attrs['comment'] = 'Defined by Sun (2001) parameterization'
            # ds = apply_liquid_effective_radius(ds)

            # %% final adjustments to file structure
            ds = ds.transpose("column", ...)  # move column to the first dimension
            ds = ds.astype(np.float32)  # change type from double to float32

            # %% save to netcdf
            ds.to_netcdf(
                path=f"{ecrad_path}/ecrad_input_q_ice_{qice}_re_ice_{re_ice_set[0]}_latitude_{lat}.nc",
                format='NETCDF4_CLASSIC')

    log.info(f"Done with writing input file(s): {pd.to_timedelta((time.time() - start), unit='second')} (hr:min:sec)")
