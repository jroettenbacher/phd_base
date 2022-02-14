#!/usr/bin/env python
"""Given a flight and campaign write a well documented netCDF file containing only certain wavelengths from HALO-SMART
for quicklooks.

Two options:

    1. One spectrometer = one file
    2. One flight = one file

Use option 2 because only a couple of wavelengths will be used.

This script can be used during campaigns to quickly generate files for quicklooks and sharing with other groups.

*author*: Johannes Röttenbacher
"""

if __name__ == "__main__":
# %% module import
    import pylim.helpers as h
    from pylim import reader
    from pylim import cirrus_hl, smart
    import os
    import pandas as pd
    import xarray as xr
    from datetime import datetime
    import logging

# %% set up logging
    log = logging.getLogger("pylim")
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)

# %% set user variables
    campaign = "cirrus-hl"
    flight = "Flight_20210629a"

# %% get paths and read in files
    smart_dir = h.get_path("calibrated", flight=flight, campaign=campaign)
    pixel_wl_dir = h.get_path("pixel_wl", campaign=campaign)
    swir_file = [f for f in os.listdir(smart_dir) if "Fdw_SWIR" in f and f.endswith("norm.dat")][0]
    vnir_file = [f for f in os.listdir(smart_dir) if "Fdw_VNIR" in f and f.endswith("norm.dat")][0]
    # read in smart calibrated data
    swir = reader.read_smart_cor(smart_dir, swir_file)
    vnir = reader.read_smart_cor(smart_dir, vnir_file)
    # read in pixel to wavelength mapping
    pixel_wl_swir = reader.read_pixel_to_wavelength(pixel_wl_dir, cirrus_hl.lookup["Fdw_SWIR"])
    pixel_wl_vnir = reader.read_pixel_to_wavelength(pixel_wl_dir, cirrus_hl.lookup["Fdw_VNIR"])

# %% set negative values to 0
    # cal_data[cal_data < 0] = 0

# %% prepare dataframe for conversion to xr.Dataset
    # swir_long = pd.melt(swir, var_name="pixel", value_name="irradiance", ignore_index=False)  # convert to long
    # vnir_long = pd.melt(vnir, var_name="pixel", value_name="irradiance", ignore_index=False)  # convert to long
    # # merge wavelength to data and set it as a multi index together with time
    # swir_long = swir_long.reset_index().merge(pixel_wl_swir, how="left", on="pixel").set_index(["time", "wavelength"])
    # vnir_long = vnir_long.reset_index().merge(pixel_wl_vnir, how="left", on="pixel").set_index(["time", "wavelength"])

# %% extract six specific wavelengths which corresponds with standard satellite wavelengths averaged over +-5nm
    wl_422 = vnir.iloc[:, 278:289].mean(axis=1)
    wl_532 = vnir.iloc[:, 410:421].mean(axis=1)
    wl_648 = vnir.iloc[:, 550:561].mean(axis=1)
    wl_858 = vnir.iloc[:, 807:818].mean(axis=1)
    wl_1238 = swir.iloc[:, 55:61].mean(axis=1)
    wl_1638 = swir.iloc[:, 130:135].mean(axis=1)

# %% calculate broadband irradiance
    wl_all = vnir.sum(axis=1) + swir.sum(axis=1)

# %% merge all products
    cal_data = pd.concat([wl_422, wl_532, wl_648, wl_858, wl_1238, wl_1638, wl_all], join='outer', axis=1)

# %% create metadata for ncfile
    var_attrs = dict(
        F_down_solar_wl_422=dict(
            long_name='Spectral downward solar irradiance (422 nm) (SMART)',
            units='W m-2 nm-1',
            comment='Averaged for wavelength band +-5 nm'),
        F_down_solar_wl_532=dict(
            long_name='Spectral downward solar irradiance (532 nm) (SMART)',
            units='W m-2 nm-1',
            comment='Averaged for wavelength band +-5 nm'),
        F_down_solar_wl_648=dict(
            long_name='Spectral downward solar irradiance (648 nm) (SMART)',
            units='W m-2 nm-1',
            comment='Averaged for wavelength band +-5 nm'),
        F_down_solar_wl_858=dict(
            long_name='Spectral downward solar irradiance (858 nm) (SMART)',
            units='W m-2 nm-1',
            comment='Averaged for wavelength band +-5 nm'),
        F_down_solar_wl_1238=dict(
            long_name='Spectral downward solar irradiance (1238 nm) (SMART)',
            units='W m-2 nm-1',
            comment='Averaged for wavelength band +-5 nm'),
        F_down_solar_wl_1638=dict(
            long_name='Spectral downward solar irradiance (1638 nm) (SMART)',
            units='W m-2 nm-1',
            comment='Averaged for wavelength band +-5 nm'),
        F_down_solar_bb=dict(
            long_name='Broadband downward solar irradiance (180 - 2200 nm) (SMART)',
            units='W m-2',
            comment='Summed over all available wavelengths')
    )

    global_attrs = dict(
        title="Preliminary spectral downward irradiance measured by HALO-SMART",
        campaign_id=f"{campaign.swapcase()}",
        platform_id="HALO",
        instrument_id="HALO-SMART",
        version_id="quicklook",
        description="Calibrated SMART measurements corrected for dark current",
        institution="Leipzig Institute for Meteorology, Leipzig, Germany",
        history=f"created {datetime.strftime(datetime.utcnow(), '%c UTC')}",
        contact="Johannes Röttenbacher, johannes.roettenbacher@uni-leipzig.de",
        PI="André Ehrlich, a.ehrlich@uni-leipzig.de"
    )

    # set units according to campaign
    if campaign == "cirrus-hl":
        units = "seconds since 2017-01-01 00:00:00 UTC"
    elif campaign == "halo-ac3":
        units = "seconds since 2017-01-01 00:00:00 UTC"
    else:
        raise ValueError(f"Campaign {campaign} unknown!")

# TODO: simplify setting the encoding of the FillValue -> loop -> function
    encoding = dict(
        time=dict(units=units, _FillValue=None),
        F_down_solar_wl_422=dict(_FillValue=-999.0),
        F_down_solar_wl_532=dict(_FillValue=-999.0),
        F_down_solar_wl_648=dict(_FillValue=-999.0),
        F_down_solar_wl_858=dict(_FillValue=-999.0),
        F_down_solar_wl_1238=dict(_FillValue=-999.0),
        F_down_solar_wl_1638=dict(_FillValue=-999.0)
    )

    # change column names
    cal_data.columns = var_attrs.keys()
    ds = cal_data.to_xarray()
    for var in ds:
        ds[var].attrs = var_attrs[var]

    ds.attrs = global_attrs

# %% create ncfile
    date_str, prop, direction = smart.get_info_from_filename(swir_file)
    outfile = f"{campaign.swapcase()}_HALO_SMART_spectral_irradiance_{direction}_ql_{date_str.replace('_', '')}.nc"
    outpath = os.path.join(smart_dir, outfile)
    ds.to_netcdf(outpath, format="NETCDF4_CLASSIC", encoding=encoding)
    log.info(f"Saved {outpath}")
