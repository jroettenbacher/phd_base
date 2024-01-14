#!\usr\bin\env python
"""
| *author*: Johannes Röttenbacher
| *created*: 20.04.2023

**Bundle calculation of additional variables for ecRad input and output files**

This script calculates a few additional variables which are often needed when working with ecRad in- and output data.
After that it also saves a file containing the mean over all columns.

It can be run via the command line and accepts several keyword arguments.

**Run like this:**

.. code-block:: shell

    python ecrad_processing.py date=yyyymmdd base_dir="./data_jr" iv=v1 ov=v1

This would merge the version 1 merged input files with the version 1 merged output files which can be found in ``{base_dir}/{date}`` and add some variables.

**User Input:**

* date (yyyymmdd)
* base_dir (directory, default: ecrad directory for halo-ac3 campaign)
* iv (vx, default:v1) input file version
* ov (vx, default:v1) output file version

**Output:**

* log file
* merged input and output file: ``{base_dir}/ecrad_merged_inout_{yyyymmdd}_{ov}.nc``

"""

# %% run script
if __name__ == "__main__":
    # %% import modules
    import pylim.helpers as h
    from pylim import ecrad
    import xarray as xr
    import numpy as np
    from metpy.units import units as un
    from metpy.calc import density, mixing_ratio_from_specific_humidity
    from metpy.constants import Cp_d, g
    import os
    import time
    import re

    start = time.time()
    # %% read in command line arguments or set defaults
    args = h.read_command_line_args()
    date = args["date"] if "date" in args else None
    key = args["key"] if "key" in args else None
    if date is None and key is None:
        raise ValueError("One of 'date' or 'key' needs to be given!")
    iv = args["iv"] if "iv" in args else "v1"
    ov = args["ov"] if "ov" in args else "v1"
    campaign = args["campaign"] if "campaign" in args else "halo-ac3"
    base_dir = args["base_dir"] if "base_dir" in args else h.get_path("ecrad", campaign=campaign)

    if campaign == "halo-ac3":
        import pylim.halo_ac3 as meta

        if key is not None:
            flight = meta.flight_names[key]
            date = flight[9:17]
    elif campaign == "cirrus-hl":
        import pylim.cirrus_hl as meta

        if key is not None:
            flight = key
            date = flight[7:15]
    else:
        raise ValueError(f"No metadata defined for campaign = {campaign}!")

    # %% setup logging
    try:
        file = __file__
    except NameError:
        file = None
    log = h.setup_logging("./logs", file, f"input{iv}_output{ov}_{date}")
    log.info(f"The following options have been passed:\n"
             f"campaign: {campaign}\n"
             f"key: {key}\n"
             f"date: {date} (defined via key if possible)\n"
             f"iv: {iv}\n"
             f"ov: {ov}\n"
             f"base_dir: {base_dir}\n")

    # create input path according to given base_dir and date
    inpath = os.path.join(base_dir, date)

    # %% read in merged in- and outfile
    input_file = f"{inpath}/ecrad_merged_input_{date}_{iv}.nc"
    output_file = f"{inpath}/ecrad_merged_output_{date}_{ov}.nc"
    outfile = f"{inpath}/ecrad_merged_inout_{date}_{ov}.nc"
    # merge in and out file, reading in outfile directly stops us from overwriting it
    ecrad_in = xr.open_dataset(input_file)
    ecrad_out = xr.open_dataset(output_file)
    ds = xr.merge([ecrad_out, ecrad_in])

    # %% calculate additional variables
    ds = ds.assign_coords(half_level=np.arange(0.5, 138.5))
    ds["band_sw"] = range(1, 15)
    ds["band_lw"] = range(1, 17)
    ds["re_ice"] = ds.re_ice.where(ds.re_ice != 5.19616e-05, np.nan)
    ds["re_liquid"] = ds.re_liquid.where(ds.re_liquid != 4.e-06, np.nan)
    for var in ["ciwc", "cswc", "q_ice"]:
        ds[var] = ds[var].where(ds[var] != 0, np.nan)

    if "press_height_hl" not in ds:
        ds = ecrad.calculate_pressure_height(ds)

    if "column" in ds.dims:
        # add some statistics
        ds["flux_dn_sw_std"] = ds["flux_dn_sw"].std(dim="column")

    # calculate transmissivity and reflectivity
    ds["transmissivity_sw"] = ds["flux_dn_sw"] / ds["flux_dn_sw_clear"]
    ds["transmissivity_sw_toa"] = ds["flux_dn_sw"] / ds["flux_dn_sw_clear"].isel(half_level=0)
    ds["transmissivity_sw_above_cloud"] = ds["flux_dn_sw"] / ds["flux_dn_sw_clear"].isel(half_level=73)
    ds["reflectivity_sw"] = ds["flux_up_sw"] / ds["flux_dn_sw"]
    ds["spectral_transmissivity_sw"] = ds["spectral_flux_dn_sw"] / ds["spectral_flux_dn_sw_clear"]
    ds["spectral_reflectivity_sw"] = ds["spectral_flux_up_sw"] / ds["spectral_flux_dn_sw"]
    # terrestrial
    ds["transmissivity_lw"] = ds["flux_dn_lw"] / ds["flux_dn_lw_clear"]
    ds["reflectivity_lw"] = ds["flux_up_lw"] / ds["flux_dn_lw"]
    ds["spectral_transmissivity_lw"] = ds["spectral_flux_dn_lw"] / ds["spectral_flux_dn_lw_clear"]
    ds["spectral_reflectivity_lw"] = ds["spectral_flux_up_lw"] / ds["spectral_flux_dn_lw"]

    # normalize by solar zenith angle
    for var in ["flux_dn_sw", "flux_dn_direct_sw", "transmissivity_sw_above_cloud", "transmissivity_sw_toa"]:
        ds[f"{var}_norm"] = ds[var] / ds["cos_solar_zenith_angle"]

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

    # calculate IWP
    da = ds.pressure_hl.diff(dim="half_level").rename(half_level="level").assign_coords(level=ds.level.to_numpy())
    factor = da * un.Pa / (g * ds.cloud_fraction)
    iwp = (factor * ds.q_ice * un("kg/kg")).metpy.convert_units("kg/m^2")
    ds["iwp"] = iwp.metpy.dequantify().where(iwp != np.inf, np.nan)
    ds["iwp"].attrs = {"units": "kg m^-2",
                       "long_name": "Ice water path",
                       "description": "Ice water path derived from q_ice"}
    ds["tiwp"] = ds.iwp.where(ds.iwp != np.inf, np.nan).sum(dim="level")
    ds["tiwp"].attrs = {"units": "kg m^-2", "long_name": "Total ice water path"}

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

    #TODO: calculate relative humidity over water and over ice

    # calculate bulk optical properties
    ov_short = re.sub(r"\.[0-9]", "", ov)  # remove the .x
    if ov_short in ecrad.ice_optic_parameterizations["fu"]:
        ice_optics = ecrad.calc_ice_optics_fu_sw(ds.iwp, ds.re_ice)
    elif ov_short == "v2":
        ice_optics = ecrad.calc_ice_optics_baran2017("sw", ds.iwp, ds.q_ice, ds.t)
    elif ov_short in ecrad.ice_optic_parameterizations["yi"]:
        ice_optics = ecrad.calc_ice_optics_yi("sw", ds.iwp, ds.re_ice)
    elif ov_short in ecrad.ice_optic_parameterizations["baran2016"]:
        ice_optics = ecrad.calc_ice_optics_baran2016("sw", ds.iwp, ds.q_ice, ds.t)
    else:
        raise ValueError(f"No parameterization for {ov_short} defined!")

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
    fdw_top = ds.flux_dn_sw.sel(half_level=slice(137)).to_numpy() * un("W/m2")
    fup_top = ds.flux_up_sw.sel(half_level=slice(137)).to_numpy() * un("W/m2")
    fdw_bottom = ds.flux_dn_sw.sel(half_level=slice(1, 138)).to_numpy() * un("W/m2")
    fup_bottom = ds.flux_up_sw.sel(half_level=slice(1, 138)).to_numpy() * un("W/m2")
    z_top = ds.press_height_hl.sel(half_level=slice(137)).to_numpy() * un.m
    z_bottom = ds.press_height_hl.sel(half_level=slice(1, 138)).to_numpy() * un.m
    heating_rate = (1 / (ds.air_density * Cp_d)) * (
            ((fdw_top - fup_top) - (fdw_bottom - fup_bottom)) / (z_top - z_bottom))
    ds["heating_rate_sw"] = heating_rate.metpy.convert_units("K/day")
    # terrestrial
    fdw_top = ds.flux_dn_lw.sel(half_level=slice(137)).to_numpy() * un("W/m2")
    fup_top = ds.flux_up_lw.sel(half_level=slice(137)).to_numpy() * un("W/m2")
    fdw_bottom = ds.flux_dn_lw.sel(half_level=slice(1, 138)).to_numpy() * un("W/m2")
    fup_bottom = ds.flux_up_lw.sel(half_level=slice(1, 138)).to_numpy() * un("W/m2")
    z_top = ds.press_height_hl.sel(half_level=slice(137)).to_numpy() * un.m
    z_bottom = ds.press_height_hl.sel(half_level=slice(1, 138)).to_numpy() * un.m
    heating_rate = (1 / (ds.air_density * Cp_d)) * (
            ((fdw_top - fup_top) - (fdw_bottom - fup_bottom)) / (z_top - z_bottom))
    ds["heating_rate_lw"] = heating_rate.metpy.convert_units("K/day")
    # net heating rate
    ds["heating_rate_net"] = ds.heating_rate_sw + ds.heating_rate_lw

    # try to get model level of flight altitude if possible
    try:
        bahamas_path = h.get_path("bahamas", flight=flight, campaign=campaign)
        bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{key}_v1_JR.nc"
        bahamas_ds = xr.open_dataset(f"{bahamas_path}/{bahamas_file}")
        bahamas_ds = bahamas_ds.sel(time=ds.time, method="nearest")

        if "column" in ds.dims:
            ds["aircraft_level"] = ecrad.get_model_level_of_altitude(bahamas_ds.IRS_ALT,
                                                                     ds.sel(column=0),
                                                                     "half_level")
        else:
            ds["aircraft_level"] = ecrad.get_model_level_of_altitude(bahamas_ds.IRS_ALT,
                                                                     ds,
                                                                     "half_level")

    except NameError as e:
        log.info(e)
    except FileNotFoundError as e:
        log.info(e)

    # %% save to netCDF
    ds.to_netcdf(outfile, format="NETCDF4_CLASSIC")
    log.info(f"Saved {outfile}")

    # %% take the mean over all columns and save it
    if "column" in ds.dims:
        mean_outfile = f"{inpath}/ecrad_merged_inout_{date}_{ov}_mean.nc"
        ds_mean = ds.mean(dim="column")
        ds_mean.to_netcdf(mean_outfile, format="NETCDF4_CLASSIC")
        log.info(f"Saved {mean_outfile}")

    log.info(f"Done with ecrad_processing in: {h.seconds_to_fstring(time.time() - start)} [h:mm:ss]")
