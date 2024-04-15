#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 10.04.2024

Runs ecRad for a sensitivity study.
Included post-processing:

- merge input and output files and add new dimensions
- calculate additional variables

"""
import pylim.helpers as h
from pylim.ecrad import calculate_pressure_height, calc_ice_optics_fu_sw, calc_ice_optics_yi, calc_ice_optics_baran2016
import argparse
import glob
from metpy.constants import Cp_d, g
from metpy.calc import density, mixing_ratio_from_specific_humidity
from metpy.units import units as un
import numpy as np
import subprocess
import os
import xarray as xr


def parse_args():
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('-v', '--version', default='v1', help='Namelist version (default: v1)')
    return parser.parse_args()


def main():
    args = parse_args()
    version = args.version

    ecrad = '/projekt_agmwend/Modelle/ECMWF_ECRAD/src/ecrad-1.5.0/bin/ecrad'
    base_dir = '/projekt_agmwend/home_rad/jroettenbacher/ecrad_sensitivity_studies'
    # setup logging and return input to user
    log = h.setup_logging('./logs', __file__)
    log.info('Options set:'
             f'version: {version}')

    inpath = os.path.join(base_dir, 'ecrad_input')
    outpath = os.path.join(base_dir, 'ecrad_output')
    rad_prop_outpath = os.path.join(base_dir, f'radiative_properties')
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(rad_prop_outpath, exist_ok=True)

    namelist = f'{base_dir}/IFS_namelist_jr_sensitivity_{version}.nam'

    reg_file = f'ecrad_input_*.nc'

    os.chdir(inpath)

    # delete all radiative property files before running ecRad to avoid errors
    subprocess.run(['rm', 'radiative_properties*'])

    file_list = sorted(glob.glob(reg_file))
    n_files = len(file_list)
    log.info(f'Number of files to calculate: {n_files}')

    for i, file_path in enumerate(file_list):
        filename = os.path.basename(file_path)
        log.info(f'infile: {filename}')
        outfilename = os.path.join(outpath,
                                   (filename
                                    .replace('input', 'output')
                                    .replace('.nc', f'_{version}.nc')
                                    ))
        log.info(f'outfile: {outfilename}')
        log.info(f'Processing filenumber: {i + 1} of {n_files}')

        subprocess.run([ecrad, namelist, filename, outfilename])

        if os.path.isfile('radiative_properties.nc'):
            subprocess.run(['mv', 'radiative_properties.nc', f'{rad_prop_outpath}/radiative_properties_{version}_{i}.nc'])
            log.info(f'Moved radiative_properties.nc to {rad_prop_outpath}')
        elif os.path.isfile('radiative_properties_*.nc'):
            subprocess.run(['mv'] + glob.glob('radiative_properties_*.nc') + [f'{rad_prop_outpath}/'])
            log.info(f'Moved all radiative_properties files to {rad_prop_outpath}')
        else:
            log.info('No radiative_properties files to move found!')

    log.info('> Done with ecRad simulations.')
    os.chdir('/projekt_agmwend/home_rad/jroettenbacher/phd_base/processing')  # cd to working directory

    # read in input and output files and merge them by adding new dimensions according to the settings
    ds_list = list()
    for f in file_list:
        ds_in = xr.open_dataset(f'{inpath}/{f}')
        outfile = f.replace('input', 'output').replace('.nc', f'_{version}.nc')
        ds_out = xr.open_dataset(f'{outpath}/{outfile}')
        for setting in ds_in.attrs['settings'].split(','):
            key, value = setting.split('=')
            ds_in = ds_in.expand_dims({f'{key.strip()}_dim': [value.strip()]})
            ds_out = ds_out.expand_dims({f'{key.strip()}_dim': [value.strip()]})

        # merge output and input files
        ds_list.append(xr.merge([ds_in, ds_out]))

    ds = xr.merge(ds_list)  # merge all DataSets

    # %% calculate additional variables
    ds = ds.assign_coords(half_level=np.arange(0.5, len(ds.half_level)),
                          level=np.arange(1, len(ds.level) + 1),
                          band_sw=range(1, 15),
                          band_lw=range(1, 17))
    ds["re_ice"] = ds.re_ice.where(ds.re_ice != 5.19616e-05, np.nan)
    for var in ["ciwc", "cswc", "clwc", "crwc", "q_ice", "q_liquid"]:
        ds[var] = ds[var].where(ds[var] != 0, np.nan)

    ds = calculate_pressure_height(ds)

    # calculate transmissivity and reflectivity
    ds["transmissivity_sw"] = ds["flux_dn_sw"] / ds["flux_dn_sw_clear"]
    ds["transmissivity_sw_toa"] = ds["flux_dn_sw"] / ds["flux_dn_sw_clear"].isel(half_level=0)
    ds["transmissivity_sw_above_cloud"] = ds["flux_dn_sw"] / ds["flux_dn_sw_clear"].isel(half_level=41)
    ds["reflectivity_sw"] = ds["flux_up_sw"] / ds["flux_dn_sw"]
    ds["spectral_transmissivity_sw"] = ds["spectral_flux_dn_sw"] / ds["spectral_flux_dn_sw_clear"]
    ds["spectral_reflectivity_sw"] = ds["spectral_flux_up_sw"] / ds["spectral_flux_dn_sw"]
    # terrestrial
    ds["transmissivity_lw"] = ds["flux_dn_lw"] / ds["flux_dn_lw_clear"]
    ds["reflectivity_lw"] = ds["flux_up_lw"] / ds["flux_dn_lw"]
    ds["spectral_transmissivity_lw"] = ds["spectral_flux_dn_lw"] / ds["spectral_flux_dn_lw_clear"]
    ds["spectral_reflectivity_lw"] = ds["spectral_flux_up_lw"] / ds["spectral_flux_dn_lw"]

    # calculate net fluxes
    ds["flux_net_sw"] = ds["flux_dn_sw"] - ds["flux_up_sw"]
    ds["flux_net_lw"] = ds["flux_dn_lw"] - ds["flux_up_lw"]
    ds["spectral_flux_net_sw"] = ds["spectral_flux_dn_sw"] - ds["spectral_flux_up_sw"]
    ds["spectral_flux_net_lw"] = ds["spectral_flux_dn_lw"] - ds["spectral_flux_up_lw"]

    # calculate cloud radiative effect
    ds["cre_sw"] = (ds.flux_dn_sw - ds.flux_up_sw) - (ds.flux_dn_sw_clear - ds.flux_up_sw_clear)  # solar
    # spectral cre
    ds["spectral_cre_sw"] = (ds.spectral_flux_dn_sw - ds.spectral_flux_up_sw) - (
            ds.spectral_flux_dn_sw_clear - ds.spectral_flux_up_sw_clear)
    # terrestrial
    ds["cre_lw"] = (ds.flux_dn_lw - ds.flux_up_lw) - (ds.flux_dn_lw_clear - ds.flux_up_lw_clear)
    # spectral cre
    ds["spectral_cre_lw"] = (ds.spectral_flux_dn_lw - ds.spectral_flux_up_lw) - (
            ds.spectral_flux_dn_lw_clear - ds.spectral_flux_up_lw_clear)
    # cre_total
    ds["cre_total"] = ds.cre_sw + ds.cre_lw
    # spectral cre net
    ds["spectral_cre_total"] = ds.spectral_cre_sw + ds.spectral_cre_lw

    # calculate IWP and LWP
    da = (ds.pressure_hl
          .diff(dim="half_level")
          .rename(half_level="level")
          .assign_coords(level=ds.level.to_numpy()))
    factor = da * un.Pa / (g * ds.cloud_fraction)
    iwp = (factor * ds.q_ice * un("kg/kg")).metpy.convert_units("kg/m^2")
    ds["iwp"] = iwp.metpy.dequantify().where(iwp != np.inf, np.nan)
    ds["iwp"].attrs = {"units": "kg m^-2",
                       "long_name": "Ice water path",
                       "description": "Ice water path derived from q_ice"}
    ds["tiwp"] = ds.iwp.where(ds.iwp != np.inf, np.nan).sum(dim="level")
    ds["tiwp"].attrs = {"units": "kg m^-2", "long_name": "Total ice water path",
                        "description": "Total ice water path derived from q_ice"}

    ciwp = (factor * ds.ciwc * un("kg/kg")).metpy.convert_units("kg/m^2")
    ds["ciwp"] = ciwp.metpy.dequantify().where(ciwp != np.inf, np.nan)
    ds["ciwp"].attrs = {"units": "kg m^-2",
                        "long_name": "Cloud ice water path",
                        "description": "Cloud ice water path derived from ciwc"}
    ds["tciwp"] = ds.ciwp.where(ds.ciwp != np.inf, np.nan).sum(dim="level")
    ds["tciwp"].attrs = {"units": "kg m^-2", "long_name": "Total cloud ice water path",
                         "description": "Total cloud ice water path derived from ciwc"}

    lwp = (factor * ds.q_liquid * un("kg/kg")).metpy.convert_units("kg/m^2")
    ds["lwp"] = lwp.metpy.dequantify().where(lwp != np.inf, np.nan)
    ds["lwp"].attrs = {"units": "kg m^-2",
                       "long_name": "Liquid water path",
                       "description": "Liquid water path derived from q_liquid"}
    ds["tlwp"] = ds.lwp.where(ds.lwp != np.inf, np.nan).sum(dim="level")
    ds["tlwp"].attrs = {"units": "kg m^-2", "long_name": "Total liquid water path"}

    clwp = (factor * ds.clwc * un("kg/kg")).metpy.convert_units("kg/m^2")
    ds["clwp"] = clwp.metpy.dequantify().where(clwp != np.inf, np.nan)
    ds["clwp"].attrs = {"units": "kg m^-2",
                        "long_name": "Cloud liquid water path",
                        "description": "Cloud liquid water path derived from clwc"}
    ds["tclwp"] = ds.clwp.where(ds.clwp != np.inf, np.nan).sum(dim="level")
    ds["tclwp"].attrs = {"units": "kg m^-2", "long_name": "Total cloud liquid water path",
                         "description": "Total cloud liquid water path derived from clwc"}

    # calculate density
    pressure = ds["pressure_full"] * un.Pa
    temperature = ds["t"] * un.K
    mixing_ratio = mixing_ratio_from_specific_humidity(ds["q"] * un("kg/kg"))
    ds["air_density"] = density(pressure, temperature, mixing_ratio)
    ds["air_density"].attrs = {"units": "kg m^-3", "long_name": "Air density"}

    # convert kg/kg to kg/m³
    ds["iwc"] = ds["q_ice"] * un("kg/kg") * ds["air_density"]
    ds["iwc"].attrs = {"units": "kg m^-3",
                       "long_name": "Ice water content",
                       "description": "Ice water content derived from q_ice"}

    # TODO: calculate relative humidity over water and over ice

    # calculate bulk optical properties
    if version in ['v1']:
        ice_optics = calc_ice_optics_fu_sw(ds.iwp, ds.re_ice)
    elif version in ['v2']:
          ice_optics = calc_ice_optics_yi("sw", ds.iwp, ds.re_ice)
    elif version in ['v3']:
        ice_optics = calc_ice_optics_baran2016("sw", ds.iwp, ds.q_ice, ds.t)
    else:
        raise ValueError(f"No parameterization for {version} defined!")

    ds["od"] = ice_optics[0]
    ds["scat_od"] = ice_optics[1]
    ds["g"] = ice_optics[2]
    ds["g_mean"] = ds["g"].mean(dim="band_sw")

    # get array to normalize each band value by its bandwidth
    dx = list()
    for i, band in enumerate(h.ecRad_bands.keys()):
        h12 = h.ecRad_bands[band]
        dx.append(h12[1] - h12[0])

    dx_array = xr.DataArray(dx, coords=dict(band_sw=range(1, 15)))

    # integrate over band_sw
    for var in ["od", "scat_od"]:
        try:
            ret = ds[var] / dx_array
            ds[f"{var}_int"] = ret.sum(dim="band_sw")
        except KeyError:
            print(f"{var} not found in ds")

    # calculate heating rates, solar
    fdw_top = ds.flux_dn_sw.sel(half_level=slice(len(ds.level))).to_numpy() * un("W/m2")
    fup_top = ds.flux_up_sw.sel(half_level=slice(len(ds.level))).to_numpy() * un("W/m2")
    fdw_bottom = ds.flux_dn_sw.sel(half_level=slice(1, len(ds.half_level))).to_numpy() * un("W/m2")
    fup_bottom = ds.flux_up_sw.sel(half_level=slice(1, len(ds.half_level))).to_numpy() * un("W/m2")
    z_top = ds.press_height_hl.sel(half_level=slice(len(ds.level))).to_numpy() * un.m
    z_bottom = ds.press_height_hl.sel(half_level=slice(1, len(ds.half_level))).to_numpy() * un.m
    heating_rate = (1 / (ds.air_density * Cp_d)) * (
            ((fdw_top - fup_top) - (fdw_bottom - fup_bottom)) / (z_top - z_bottom))
    ds["heating_rate_sw"] = heating_rate.metpy.convert_units("K/day")
    # terrestrial
    fdw_top = ds.flux_dn_lw.sel(half_level=slice(len(ds.level))).to_numpy() * un("W/m2")
    fup_top = ds.flux_up_lw.sel(half_level=slice(len(ds.level))).to_numpy() * un("W/m2")
    fdw_bottom = ds.flux_dn_lw.sel(half_level=slice(1, len(ds.half_level))).to_numpy() * un("W/m2")
    fup_bottom = ds.flux_up_lw.sel(half_level=slice(1, len(ds.half_level))).to_numpy() * un("W/m2")
    z_top = ds.press_height_hl.sel(half_level=slice(len(ds.level))).to_numpy() * un.m
    z_bottom = ds.press_height_hl.sel(half_level=slice(1, len(ds.half_level))).to_numpy() * un.m
    heating_rate = (1 / (ds.air_density * Cp_d)) * (
            ((fdw_top - fup_top) - (fdw_bottom - fup_bottom)) / (z_top - z_bottom))
    ds["heating_rate_lw"] = heating_rate.metpy.convert_units("K/day")
    # net heating rate
    ds["heating_rate_net"] = ds.heating_rate_sw + ds.heating_rate_lw

    ds.attrs.pop('settings')  # remove the settings attribute as it is not needed anymore
    # save to netCDF
    ds.to_netcdf(f'{base_dir}/ecrad_merged_inout_{version}.nc')


if __name__ == '__main__':
    main()
