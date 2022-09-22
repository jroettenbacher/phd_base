#!/usr/bin/env python
"""Calculate attitude correction for BACARDI and for SMART from BAHAMAS data

**Rationale**

SMART is stabilized, however the stabilization does not work perfectly all the time.
By calculating the attitude correction for an unstabilized case and applying this backwards on a libRadtran simulation can tell us more about, how well the stabilization worked and the impact of its imperfection.
The attitude angles have to be interpolated onto the libRadtran time steps before the correction can be calculated.
The attitude correction can also be used for the BACARDI data.

Save the attitude correction in the respective time resolutions for BACARDI, SMART and libRadtran.

*author*: Johannes Röttenbacher
"""
if __name__ == "__main__":
    # %% import modules
    import pylim.helpers as h
    from pylim.bacardi import fdw_attitude_correction
    from pylim import reader
    import os
    import xarray as xr
    import numpy as np
    from datetime import datetime

    # %% find data and read it in
    campaign = "halo-ac3"
    date = "20220321"
    flight_key = "RF08"
    flight = f"HALO-AC3_{date}_HALO_{flight_key}"
    smart_path = h.get_path("calibrated", flight, campaign)
    bahamas_path = h.get_path("bahamas", flight, campaign)
    bacardi_path = h.get_path("bacardi", flight, campaign)
    libradtran_path = h.get_path("libradtran", flight, campaign)
    bahamas_file = f"QL_HALO-AC3_HALO_BAHAMAS_{date}_{flight_key}_v1.nc"
    bacardi_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{flight_key}.nc"
    swir_file = [f for f in os.listdir(smart_path) if "Fdw_SWIR" in f and f.endswith("nc")][0]
    vnir_file = [f for f in os.listdir(smart_path) if "Fdw_VNIR" in f and f.endswith("nc")][0]
    libradtran_file = f"HALO-AC3_HALO_libRadtran_clearsky_simulation_smart_{date}_{flight_key}.nc"
    bahamas = reader.read_bahamas(f"{bahamas_path}/{bahamas_file}")
    bacardi = xr.open_dataset(f"{bacardi_path}/{bacardi_file}")
    libradtran = xr.open_dataset(f"{libradtran_path}/{libradtran_file}")

    # %% read in SMART files
    df_swir = xr.open_dataset(f"{smart_path}/{swir_file}")
    df_vnir = xr.open_dataset(f"{smart_path}/{vnir_file}")
    # rename variables to merge them
    df_vnir = df_vnir.rename(dict(Fdw_VNIR="Fdw"))
    df_swir = df_swir.rename(dict(Fdw_SWIR="Fdw"))
    # merge SWIR and VNIR
    df_smart = xr.merge([df_swir, df_vnir])
    df_smart["Fdw_bb"] = df_smart["Fdw_VNIR_bb"] + df_smart["Fdw_SWIR_bb"]

    # %% calculate attitude correction
    fdw = np.ones_like(bahamas.TS)  # generate dummy downward irradiance
    fdir = 1  # dummy direct fraction
    roll = bahamas.IRS_PHI.values
    pitch = bahamas.IRS_THE.values * -1  # invert pitch to fit function convention
    yaw = bahamas.IRS_HDG.values
    saa = bacardi.saa.values
    sza = bacardi.sza.values
    # BACARDI time
    _, factor = fdw_attitude_correction(fdw, roll, pitch, yaw, sza, saa, fdir)

    # %% interpolate BAHAMAS and BACARDI to SMART timestamps
    bahamas_inp = bahamas.interp_like(df_smart, assume_sorted=True)
    bacardi_inp = bacardi.interp_like(df_smart, assume_sorted=True)

    fdw = np.ones_like(bahamas_inp.TS)  # generate dummy downward irradiance
    fdir = 1  # dummy direct fraction
    roll = bahamas_inp.IRS_PHI.values
    pitch = bahamas_inp.IRS_THE.values * -1  # invert pitch to fit function convention
    yaw = bahamas_inp.IRS_HDG.values
    saa = bacardi_inp.saa.values
    sza = bacardi_inp.sza.values
    _, factor_smart = fdw_attitude_correction(fdw, roll, pitch, yaw, sza, saa, fdir)

    # %% interpolate BAHAMAS and BACARDI to libradtran timestamps
    bahamas_inp = bahamas.interp_like(libradtran, assume_sorted=True)
    bacardi_inp = bacardi.interp_like(libradtran, assume_sorted=True)

    fdw = np.ones_like(bahamas_inp.TS)  # generate dummy downward irradiance
    fdir = 1  # dummy direct fraction
    roll = bahamas_inp.IRS_PHI.values
    pitch = bahamas_inp.IRS_THE.values * -1  # invert pitch to fit function convention
    yaw = bahamas_inp.IRS_HDG.values
    saa = bacardi_inp.saa.values
    sza = bacardi_inp.sza.values
    _, factor_libradtran = fdw_attitude_correction(fdw, roll, pitch, yaw, sza, saa, fdir)

    # %% prepare variable for netCDF export BACARDI
    ds = xr.Dataset(data_vars=dict(correction_factor=(["time"], factor)), coords=dict(time=bacardi.time))
    ds["correction_factor"].attrs = dict(long_name="Attitude correction factor", units="1",
                                         comment="Attitude correction factor for direct downward irradiance. Fdw_cor = direct_fraction * Fdw * factor + (1 - direct_fraction) * Fdw")
    global_attrs = dict(
        title="Attitude correction factor for direct downward irradiance from BACARDI",
        campaign_id=f"{campaign.swapcase()}",
        platform_id="HALO",
        instrument_id="BACARDI",
        version_id="v1",
        description="Use to correct the direct downward irradiance measurement from BACARDI: Fdw_cor = direct_fraction * Fdw * factor + (1 - direct_fraction) * Fdw",
        institution="Leipzig Institute for Meteorology, Leipzig, Germany",
        history=f"created {datetime.strftime(datetime.utcnow(), '%c UTC')}",
        contact="Johannes Röttenbacher, johannes.roettenbacher@uni-leipzig.de",
        PI="André Ehrlich, a.ehrlich@uni-leipzig.de"
    )
    ds.attrs = global_attrs

    # %% write to nc file
    filename = f"{bacardi_path}/HALO-AC3_HALO_BACARDI_attitude_correction_factor_{date}_{flight_key}.nc"
    ds.to_netcdf(filename,
                 encoding=dict(time=dict(units="seconds since 2017-01-01", _FillValue=None)))
    print(f"Saved {filename}")

    # %% prepare smart nc file
    ds = xr.Dataset(data_vars=dict(correction_factor=(["time"], factor_smart)), coords=dict(time=df_smart.time))
    ds["correction_factor"].attrs = dict(long_name="Attitude correction factor", units="1",
                                         comment="Attitude correction factor for direct downward irradiance. Fdw_cor = direct_fraction * Fdw * factor + (1 - direct_fraction) * Fdw")
    global_attrs = dict(
        title="Attitude correction factor for direct downward irradiance from SMART",
        campaign_id=f"{campaign.swapcase()}",
        platform_id="HALO",
        instrument_id="SMART",
        version_id="v1",
        description="Use to correct the direct downward irradiance measurement from SMART: Fdw_cor = direct_fraction * Fdw * factor + (1 - direct_fraction) * Fdw",
        institution="Leipzig Institute for Meteorology, Leipzig, Germany",
        history=f"created {datetime.strftime(datetime.utcnow(), '%c UTC')}",
        contact="Johannes Röttenbacher, johannes.roettenbacher@uni-leipzig.de",
        PI="André Ehrlich, a.ehrlich@uni-leipzig.de"
    )
    ds.attrs = global_attrs
    # %% write to nc file
    filename = f"{smart_path}/HALO-AC3_HALO_SMART_attitude_correction_factor_{date}_{flight_key}.nc"
    ds.to_netcdf(filename,
                 encoding=dict(time=dict(units="seconds since 2017-01-01", _FillValue=None)))
    print(f"Saved {filename}")

    # %% prepare libradtran nc file
    ds = xr.Dataset(data_vars=dict(correction_factor=(["time"], factor_libradtran)), coords=dict(time=libradtran.time))
    ds["correction_factor"].attrs = dict(long_name="Attitude correction factor", units="1",
                                         comment="Attitude correction factor for direct downward irradiance. Fdw_cor = direct_fraction * Fdw * factor + (1 - direct_fraction) * Fdw")
    global_attrs = dict(
        title="Attitude correction factor for direct downward irradiance from libRadtran",
        campaign_id=f"{campaign.swapcase()}",
        platform_id="HALO",
        instrument_id="libRadtran",
        version_id="v1",
        description="Use to simulate the direct downward irradiance as measured by an unstabilized SMART instrument: Fdw_cor = Fdw * factor",
        institution="Leipzig Institute for Meteorology, Leipzig, Germany",
        history=f"created {datetime.strftime(datetime.utcnow(), '%c UTC')}",
        contact="Johannes Röttenbacher, johannes.roettenbacher@uni-leipzig.de",
        PI="André Ehrlich, a.ehrlich@uni-leipzig.de"
    )
    ds.attrs = global_attrs
    # %% write to nc file
    filename = f"{libradtran_path}/HALO-AC3_HALO_libRadtran_attitude_correction_factor_{date}_{flight_key}.nc"
    ds.to_netcdf(filename,
                 encoding=dict(time=dict(units="seconds since 2017-01-01", _FillValue=None)))
    print(f"Saved {filename}")
