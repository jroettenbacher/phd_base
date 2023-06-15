#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 14.06.2023

Create a smaller subset of SMART data to be uploaded to the HALO database.

"""
import pandas as pd

if __name__ == "__main__":
    # %% module import
    import pylim.helpers as h
    import pylim.cirrus_hl as meta
    import os
    import xarray as xr
    from datetime import datetime

    # %% set user variables
    version = "v1.0"  # set version
    campaign = "cirrus-hl"
    for key in ["Flight_20210629a", "Flight_20210629b", "Flight_20210705a", "Flight_20210712a", "Flight_20210715a",
                "Flight_20210715b", "Flight_20210719a", "Flight_20210719b"]:
        # key = f"Flight_20210629a"
        date = key[7:16]
        flight = key[9:16]
        flight_nr = meta.flight_numbers[key]

        # %% setup logging
        try:
            file = __file__
        except NameError:
            file = None
        log = h.setup_logging("./logs", file, key)
        log.info(f"Creating subset of SMART data...\nOptions Given:\nversion: {version}\ncampaign: {campaign}\n"
                 f"flight: {key}\nScript started: {datetime.utcnow():%c UTC}\n")

        # %% get paths and read in files
        smart_dir = h.get_path("calibrated", flight=key, campaign=campaign)
        fdw_file = f"CIRRUS-HL_{flight_nr}_{date}_HALO_SMART_spectral-irradiance-Fdw_v1.0.nc"
        fup_file = f"CIRRUS-HL_{flight_nr}_{date}_HALO_SMART_spectral-irradiance-Fup_v1.0.nc"
        # read in smart calibrated data
        fdw = xr.open_dataset(f"{smart_dir}/{fdw_file}")
        fup = xr.open_dataset(f"{smart_dir}/{fup_file}")
        ds = xr.merge([fdw, fup], compat="override")

    # %% create metadata for ncfile
        var_attrs_fdw = dict(
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
                comment='Integrated over all available wavelengths'),
        )

        var_attrs_fup = dict(
            F_up_solar_wl_422=dict(
                long_name='Spectral upward solar irradiance (422 nm) (SMART)',
                units='W m-2 nm-1',
                comment='Averaged for wavelength band +-5 nm'),
            F_up_solar_wl_532=dict(
                long_name='Spectral upward solar irradiance (532 nm) (SMART)',
                units='W m-2 nm-1',
                comment='Averaged for wavelength band +-5 nm'),
            F_up_solar_wl_648=dict(
                long_name='Spectral upward solar irradiance (648 nm) (SMART)',
                units='W m-2 nm-1',
                comment='Averaged for wavelength band +-5 nm'),
            F_up_solar_wl_858=dict(
                long_name='Spectral upward solar irradiance (858 nm) (SMART)',
                units='W m-2 nm-1',
                comment='Averaged for wavelength band +-5 nm'),
            F_up_solar_wl_1238=dict(
                long_name='Spectral upward solar irradiance (1238 nm) (SMART)',
                units='W m-2 nm-1',
                comment='Averaged for wavelength band +-5 nm'),
            F_up_solar_wl_1638=dict(
                long_name='Spectral upward solar irradiance (1638 nm) (SMART)',
                units='W m-2 nm-1',
                comment='Averaged for wavelength band +-5 nm'),
            F_up_solar_bb=dict(
                long_name='Broadband upward solar irradiance (180 - 2200 nm) (SMART)',
                units='W m-2',
                comment='Integrated over all available wavelengths')
        )

        global_attrs = dict(
            title="Selected wavelengths of spectral upward and downward irradiance measured by SMART",
            project="SPP 1294 HALO",
            mission=f"{campaign.swapcase()}",
            platform="HALO",
            instrument="SMART",
            version=version,
            description="Selected wavelengths of calibrated HALO-SMART measurements corrected for dark current and cosine "
                        "response, and resampled to 1Hz resolution combined with the SMART INS data. "
                        "Downward irradiance measurements are actively stabilized.",
            institution="Leipzig Institute for Meteorology, Leipzig, Germany",
            date_created=f"{pd.to_datetime(date[:-1]):%Y-%m-%d}",
            date_modified=f"{datetime.strftime(datetime.utcnow(), '%c UTC')}",
            history=f"created {datetime.strftime(datetime.utcnow(), '%c UTC')}",
            contact="Johannes Röttenbacher, johannes.roettenbacher@uni-leipzig.de",
            PI="André Ehrlich, a.ehrlich@uni-leipzig.de",
            Conventions="CF-1.9",
            license="CC BY 4.0",
        )

        # set units according to campaign
        if campaign == "cirrus-hl":
            units = "seconds since 2017-01-01 00:00:00 UTC"
        elif campaign == "halo-ac3":
            units = "seconds since 2017-01-01 00:00:00 UTC"
        else:
            raise ValueError(f"Campaign {campaign} unknown!")

        encoding = dict(time=dict(units=units, _FillValue=None))

    # %% extract six specific wavelengths which corresponds with standard satellite wavelengths averaged over +-5nm
        wavelengths = [422, 532, 648, 858, 1238, 1638, 0]
        for var_dw, var_up, wl in zip(var_attrs_fdw, var_attrs_fup, wavelengths):
            if wl != 0:
                ds[var_dw] = ds["Fdw_cor"].sel(wavelength=slice(wl-5, wl+5)).mean(dim="wavelength")
                ds[var_up] = ds["Fup_cor"].sel(wavelength=slice(wl - 5, wl + 5)).mean(dim="wavelength")
                # add meta data
                ds[var_dw].attrs = var_attrs_fdw[var_dw]
                ds[var_up].attrs = var_attrs_fup[var_up]
            else:
                ds[var_dw] = ds["Fdw_cor"].integrate(coord="wavelength")
                ds[var_up] = ds["Fup_cor"].integrate(coord="wavelength")
                # add meta data
                ds[var_dw].attrs = var_attrs_fdw[var_dw]
                ds[var_up].attrs = var_attrs_fup[var_up]

    # %% remove unwanted variables by dropping unnecessary dimension
        ds = ds.drop_dims("wavelength")

    # %% add global meta data and encoding
        ds.attrs = global_attrs
        for var in ds:
            encoding[var] = dict(_FillValue=None)

    # %% create ncfile
        outfile = f"{campaign.swapcase()}_{flight_nr}_{date}_HALO_SMART_spectral_irradiance_subset_{version}.nc"
        outpath = os.path.join(smart_dir, outfile)
        ds.to_netcdf(outpath, format="NETCDF3_CLASSIC", encoding=encoding)
        log.info(f"Saved {outpath}")
