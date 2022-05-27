#!/usr/bin/env python
"""Given a SMART input file write a well documented netCDF file.

Two options:

    1. One spectrometer = one file
    2. One flight = one file

Option 2 might result in a huge file but would be easier to distribute.
With option 1 one could still merge all single files quite easily with xarray.
Go with option 1 for now.

The netCDF file could be writen as a standard output from smart_calibrate_measurement.py or as a separate step in this script. Start with this script and write a function that can then be used in the calibration script.

author: Johannes Röttenbacher
"""

if __name__ == "__main__":
    # %% module import
    import pylim.helpers as h
    from pylim import reader, smart
    from pylim.halo_ac3 import smart_lookup, take_offs_landings
    from pylim.cirrus_hl import lookup as smart_lookup, take_offs_landings as take_offs_landings
    import os
    import pandas as pd
    from datetime import datetime
    import logging

    # %% set up logging
    log = logging.getLogger("pylim")
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)

    # %% set user variables
    campaign = "cirrus-hl"
    date = "20210629"
    flight_key = "Flight_20210629a"
    flight = f"HALO-AC3_{date}_HALO_{flight_key}" if campaign == "halo-ac3" else flight_key
    prop_channel = "Fdw_VNIR"
    # prop_channel = "Fdw_SWIR"
    to, td = take_offs_landings[flight_key]

    # %% get paths and read in files
    smart_dir = h.get_path("calibrated", flight=flight, campaign=campaign)
    pixel_wl_dir = h.get_path("pixel_wl", campaign=campaign)
    smart_file = [f for f in os.listdir(smart_dir) if prop_channel in f][0]
    # read in smart calibrated data
    cal_data = reader.read_smart_cor(smart_dir, smart_file)
    # read in pixel to wavelength mapping
    pixel_wl = reader.read_pixel_to_wavelength(pixel_wl_dir, smart_lookup[prop_channel])

    # %% set negative values to 0
    cal_data[cal_data < 0] = 0

    # %% prepare dataframe for conversion to xr.Dataset
    cal_data_long = pd.melt(cal_data, var_name="pixel", value_name="irradiance", ignore_index=False)  # convert to long
    # merge wavelength to data and set it as a multi index together with time
    cal_data_long = cal_data_long.reset_index().merge(pixel_wl, how="left", on="pixel").set_index(["time", "wavelength"])

    ds = cal_data_long.to_xarray()  # convert to xr.DataSet

    # %% calculate broadband irradiance
    ds[f"{prop_channel}_bb"] = ds.irradiance.sum(axis=1)

    # %% create metadata for ncfile
    var_attrs = {
        f"{prop_channel}": dict(
            long_name="Spectral downward irradiance (SMART)",
            standard_name="solar_irradiance_per_unit_wavelength",
            units="W m-2 nm-1"),
        f"{prop_channel}_bb": dict(
            long_name="Broadband downward irradiance (SMART)",
            units="W m-2",
            comment="Summed over all available wavelengths"),
        "pixel": dict(
            long_name="Spectrometer pixel",
            units="1"),
        "wavelength":dict(
            long_name="wavelength",
            standard_name="radiation_wavelength",
            units="nm"),
    }

    global_attrs = dict(
        title="Spectral downward irradiance measured by SMART",
        campaign_id=f"{campaign.swapcase()}",
        platform_id="HALO",
        instrument_id="SMART",
        version_id="0.1",
        description="Calibrated SMART measurements corrected for dark current",
        institution="Leipzig Institute for Meteorology, Leipzig, Germany",
        history=f"created {datetime.strftime(datetime.utcnow(), '%c UTC')}",
        contact="Johannes Röttenbacher, johannes.roettenbacher@uni-leipzig.de",
        PI="André Ehrlich, a.ehrlich@uni-leipzig.de"
    )

    encoding = dict(time=dict(units="seconds since 2017-01-01 00:00:00 UTC", _FillValue=None))

    ds = ds.rename(dict(irradiance=prop_channel))
    # drop time dimension from pixel
    ds["pixel"] = ds.pixel.isel(time=0, drop=True)

    for var in var_attrs:
        encoding[var] = dict(_FillValue=None)
        ds[var].attrs = var_attrs[var]

    ds.attrs = global_attrs
    # cut data to flight times
    ds = ds.sel(time=slice(to, td))

    # %% create ncfile
    date_str, prop, direction = smart.get_info_from_filename(smart_file)
    outfile = f"{campaign.swapcase()}_HALO_SMART_{direction}_{prop}_{date}_{flight_key}.nc"
    outpath = os.path.join(smart_dir, outfile)
    ds.to_netcdf(outpath, format="NETCDF4_CLASSIC", encoding=encoding)
    log.info(f"Saved {outpath}")
