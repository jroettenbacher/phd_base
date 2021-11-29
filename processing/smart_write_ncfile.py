#!/usr/bin/env python
"""Given a SMART input file write a well documented netCDF file.

Two options:

    1. One spectrometer = one file
    2. One flight = one file

Option 2 might result in a huge file but would be easier to distribute.
With option 1 one could still merge all single files quite easily with xarray.
Go with option 1 for now.

The netCDF file could be writen as a standard output from smart_calibrate_measurement.py or as a separate step in this
 script. Start with this script and write a function that can then be used in the calibration script.

author: Johannes Röttenbacher
"""

if __name__ == "__main__":
    # %% module import
    import pylim.helpers as h
    from pylim import reader
    from pylim import cirrus_hl, smart
    import os
    import pandas as pd
    import logging

    # %% set up logging
    log = logging.getLogger("pylim")
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)

    # %% set user variables
    campaign = "cirrus-hl"
    flight = "Flight_20210629a"
    prop_channel = "Fdw_VNIR"

    # %% get paths and read in files
    smart_dir = h.get_path("calibrated", flight=flight, campaign=campaign)
    pixel_wl_dir = h.get_path("pixel_wl", campaign=campaign)
    smart_file = [f for f in os.listdir(smart_dir) if prop_channel in f][0]
    # read in smart calibrated data
    cal_data = reader.read_smart_cor(smart_dir, smart_file)
    # read in pixel to wavelength mapping
    pixel_wl = reader.read_pixel_to_wavelength(pixel_wl_dir, cirrus_hl.lookup[prop_channel])

    # %% set negative values to 0
    cal_data[cal_data < 0] = 0

    # %% prepare dataframe for conversion to xr.Dataset
    cal_data_long = pd.melt(cal_data, var_name="pixel", value_name="irradiance", ignore_index=False)  # convert to long
    # merge wavelength to data and set it as a multi index together with time
    cal_data_long = cal_data_long.reset_index().merge(pixel_wl, how="left", on="pixel").set_index(["time", "wavelength"])

    ds = cal_data_long.to_xarray()  # convert to xr.DataSet

    # %% create metadata for ncfile
    ds.irradiance.attrs = dict(long_name="spectral downward irradiance", units="W m-2")
    ds = ds.rename(dict(irradiance=prop_channel[:3]))
    ds["pixel"] = ds.pixel.isel(time=0, drop=True)
    # TODO: Add global attributes and standard name
    # TODO: create attributes according to prop_channel

    # %% create ncfile
    date_str, prop, direction = smart.get_info_from_filename(smart_file)
    outfile = f"{campaign}_SMART_{direction}_{prop}_{date_str}.nc"
    outpath = os.path.join(smart_dir, outfile)
    ds.to_netcdf(outpath, format="NETCDF4_CLASSIC",
                 encoding={"time": {"units": "seconds since 2021-01-01"}})
    log.info(f"Saved {outpath}")
