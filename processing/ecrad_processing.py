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

    start = time.time()
    # %% read in command line arguments or set defaults
    args = h.read_command_line_args()
    date = args["date"] if "date" in args else None
    if date is None:
        raise ValueError("'date' needs to be given!")
    iv = args["iv"] if "iv" in args else "v1"
    ov = args["ov"] if "ov" in args else "v1"
    base_dir = args["base_dir"] if "base_dir" in args else h.get_path("ecrad", campaign="halo-ac3")

    # %% setup logging
    try:
        file = __file__
    except NameError:
        file = None
    log = h.setup_logging("./logs", file, f"input{iv}_output{ov}_{date}")
    log.info(f"The following options have been passed:\n"
             f"iv: {iv}\n"
             f"ov: {ov}\n"
             f"base_dir: {base_dir}\n"
             f"date: {date}\n")

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

    # calculate transmissivity and reflectivity
    ds["transmissivity_sw"] = ds["flux_dn_sw"] / ds["flux_dn_sw_clear"]
    ds["reflectivity_sw"] = ds["flux_up_sw"] / ds["flux_dn_sw"]
    ds["spectral_transmissivity_sw"] = ds["spectral_flux_dn_sw"] / ds["spectral_flux_dn_sw_clear"]
    ds["spectral_reflectivity_sw"] = ds["spectral_flux_up_sw"] / ds["spectral_flux_dn_sw"]
    # terrestrial
    ds["transmissivity_lw"] = ds["flux_dn_lw"] / ds["flux_dn_lw_clear"]
    ds["reflectivity_lw"] = ds["flux_up_lw"] / ds["flux_dn_lw"]
    ds["spectral_transmissivity_lw"] = ds["spectral_flux_dn_lw"] / ds["spectral_flux_dn_lw_clear"]
    ds["spectral_reflectivity_lw"] = ds["spectral_flux_up_lw"] / ds["spectral_flux_dn_lw"]

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
    factor = da * un.Pa  / (g * ds.cloud_fraction)
    iwp = (factor * ds.ciwc * un("kg/kg")).metpy.convert_units("kg/m^2")
    ds["iwp"] = iwp.metpy.dequantify().where(iwp != np.inf, np.nan)
    ds["iwp"].attrs = {"units": "kg m^-2", "long_name": "Ice water path"}

    # calculate density
    pressure = ds["pressure_full"] * un.Pa
    temperature = ds["t"] * un.K
    mixing_ratio = mixing_ratio_from_specific_humidity(ds["q"] * un("kg/kg"))
    ds["air_density"] = density(pressure, temperature, mixing_ratio)
    ds["air_density"].attrs = {"units": "kg m^-3", "long_name": "Air density"}

    # convert kg/kg to kg/m³
    ds["iwc"] = ds["q_ice"] * un("kg/kg") * ds["air_density"]
    ds["iwc"].attrs = {"units": "kg m^-3", "long_name": "Ice water content"}

    # calculate bulk optical properties
    if ov in ["v1", "v5", "v8", "v10", "v11", "v12", "v13", "v13.1", "v13.2", "v14", "v15", "v16", "v17", "v22", "v23", "v26"]:
        ice_optics = ecrad.calc_ice_optics_fu_sw(ds.iwp, ds.re_ice)
    elif ov == "v2":
        ice_optics = ecrad.calc_ice_optics_baran2017("sw", ds.iwp, ds.q_ice, ds.t)
    elif ov in ["v4", "v19"]:
        ice_optics = ecrad.calc_ice_optics_yi("sw", ds.iwp, ds.re_ice)
    elif ov in ["v6", "v7", "v9", "v18", "v20", "v21", "v24", "v25", "v27"]:
        ice_optics = ecrad.calc_ice_optics_baran2016("sw", ds.iwp, ds.q_ice, ds.t)
    else:
        raise ValueError(f"No parameterization for {ov} defined")

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
