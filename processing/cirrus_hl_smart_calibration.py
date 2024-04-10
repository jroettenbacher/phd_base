#!/usr/bin/env python
"""Complete calibration of the SMART files for the CIRRUS-HL campaign

The dark current corrected SMART measurement files are calibrated and filtered.

- read in dark current corrected files
- calibrate with matching transfer calibration
- correct measurement for cosine dependence of inlet
- add some metadata such as sza and saa
- add stabilization flag for Fdw
- add SMART IMS data
- write to netcdf file (VNIR and SWIR seperate)
- merge VNIR and SWIR data

*author*: Johannes Röttenbacher
"""
if __name__ == "__main__":
    # %% module import
    from pylim import helpers as h
    from pylim import cirrus_hl as campaign_meta
    from pylim import smart, reader, bacardi
    import os
    import pandas as pd
    import numpy as np
    import xarray as xr
    from datetime import datetime
    from tqdm import tqdm
    import logging

    log = logging.getLogger("pylim")
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)

    # %% set some options
    stabilized_flights = campaign_meta.stabilized_flights
    unstabilized_flights = campaign_meta.unstabilized_flights
    flights = list(campaign_meta.flight_numbers.keys())[1:]  # run all flights
    # flights = ["Flight_20210713a"]  # uncomment for single flight
    for flight in tqdm(flights):

        __file__ = None if "__file__" not in locals() else __file__
        log = h.setup_logging("./logs", __file__, flight)
        prop = "Fdw"  # Fup or Fdw
        normalize = True  # use normalized calibration factor (counts are divided by the integration time)
        lab_calib = "after"  # before or after, set which lab calibration to use for the transfer calibration
        t_int_asp06 = 300  # give integration time of field measurement for ASP06
        t_int_asp07 = 300  # give integration time of field measurement for ASP07

    # %% set paths
        cor_data_dir = h.get_path("data", flight)
        inpath = cor_data_dir
        calib_data_dir = h.get_path("calibrated", flight)
        outpath = calib_data_dir
        calib_dir = f"{h.get_path('calib')}/transfer_calibs_{lab_calib}_campaign"  # path to transfer calibration files
        pixel_dir = h.get_path("pixel_wl")
        hori_dir = h.get_path("horidata", flight)
        bacardi_dir = h.get_path("bacardi", flight)
        bahamas_dir = h.get_path("bahamas", flight)
        libradtran_dir = h.get_path("libradtran", flight)
        cosine_dir = h.get_path("cosine")

    # %% get metadata
        transfer_calib_date = campaign_meta.transfer_calibs[flight]
        date = f"{transfer_calib_date[:4]}_{transfer_calib_date[4:6]}_{transfer_calib_date[6:]}"  # reformat date to match file name
        norm = "_norm" if normalize else ""
        to, td = campaign_meta.take_offs_landings[flight]
        flight_number = campaign_meta.flight_numbers[flight]

    # %% read in dark current corrected measurement files
        files = [f for f in os.listdir(inpath) if prop in f]
        for file in files:
            date_str, channel, direction = smart.get_info_from_filename(file)
            inlet = campaign_meta.smart_lookup[direction]
            date_str = date if len(date) > 0 else date_str  # overwrite date_str if date is given
            spectrometer = campaign_meta.smart_lookup[f"{direction}_{channel}"]
            pixel_wl = reader.read_pixel_to_wavelength(pixel_dir, spectrometer)  # read in pixel to wavelength mapping
            t_int = t_int_asp06 if "ASP06" in spectrometer else t_int_asp07  # select relevant integration time
            measurement = reader.read_smart_cor(inpath, file)
            # cut measurement to take off and landing times
            measurement = measurement[to:td]

    # %% read in matching transfer calibration file from same day or from given day with matching t_int
            cali_file = f"{calib_dir}/{date_str}_{spectrometer}_{direction}_{channel}_{t_int}ms_transfer_calib{norm}.dat"
            log.info(f"Calibration file used:\n {cali_file}")
            cali = pd.read_csv(cali_file)
            # convert to long format
            m_long = measurement.melt(var_name="pixel", value_name="counts", ignore_index=False)
            if normalize:
                m_long["counts"] = m_long["counts"] / t_int

            # merge field calibration factor to long df on pixel column
            df = m_long.join(cali.set_index(cali.pixel)["c_field"], on="pixel")

    # %% calibrate measurement with transfer calibration
            df[direction] = df["counts"] * df["c_field"]  # calculate calibrated radiance/irradiance
            df = df[~np.isnan(df[f"{prop}"])]  # remove rows with nan (dark current measurements)
            df = df.reset_index().merge(pixel_wl, how="left", on="pixel").set_index(["time", "wavelength"])
            df = df.drop("pixel", axis=1)  # drop pixel column
            # extract field calibration factor from df to avoid resampling it
            c_field = df["c_field"].to_xarray().isel(time=0, drop=True)  # drop time dimension from field calibration factor
            df = df.drop("c_field", axis=1)
            # resample to 1 second time resolution and keep wavelength as an index
            df = df.groupby([pd.Grouper(freq="1S", level="time"), pd.Grouper(level="wavelength")]).mean()

            ds = df.to_xarray()  # convert to xr.DataSet
            ds["c_field"] = c_field

    # %% correct measurement for cosine dependence of inlet
            bacardi_file = f"CIRRUS-HL_{flight_number}_{flight[7:]}_ADLR_BACARDI_BroadbandFluxes_v1.1.nc"
            libradtran_file = f"CIRRUS-HL_HALO_libRadtran_clearsky_simulation_smart_spectral_{flight[7:-1]}_{flight}.nc"
            bacardi_ds = xr.open_dataset(f"{bacardi_dir}/{bacardi_file}")
            libradtran = xr.open_dataset(f"{libradtran_dir}/{libradtran_file}")
            # extract sza from BACARDI file
            sza = bacardi_ds["sza"]
            # extract direct fraction from libRadtran
            f_dir = libradtran["direct_fraction"]
            # interpolate to SMART time
            sza = sza.interp_like(ds.time)
            f_dir = f_dir.interp_like(ds.time, method="nearest")
            f_dir = f_dir.interp_like(ds.wavelength)  # interpolate to SMART wavelength
            f_dir = f_dir.where(f_dir <= 1, 1)  # replace values higher 1 with 1
            # replace values which seem to low (4 standard deviations below the mean) with the mean
            f_dir = f_dir.where(f_dir > (f_dir.mean() - 4 * f_dir.std()), f_dir.mean())

    # %% read in cosine correction factors
            cosine_file = f"HALO_SMART_{inlet}_cosine_correction_factors.csv"
            cosine_diffuse_file = f"HALO_SMART_{inlet}_diffuse_cosine_correction_factors.csv"
            cosine_cor = pd.read_csv(f"{cosine_dir}/{cosine_file}")
            cosine_diffuse_cor = pd.read_csv(f"{cosine_dir}/{cosine_diffuse_file}")
            # preprocess cosine correction
            cosine_cor = cosine_cor[cosine_cor["property"] == channel]  # select only the relevant property/channel
            cosine_diffuse_cor = cosine_diffuse_cor[cosine_diffuse_cor["property"] == channel]  # select only the relevant property/channel
            pos_sel = cosine_cor["position"] == "normal"  # leave out azimuth dependence for now
            # take mean of normal and turned position
            mean_k_cos = (cosine_cor.loc[pos_sel].loc[:, "k_cos"].values + cosine_cor.loc[~pos_sel].loc[:, "k_cos"].values) / 2
            cosine_cor = cosine_cor.loc[pos_sel]  # select only normal position
            cosine_cor["k_cos"] = mean_k_cos  # overwrite correction factor with mean

    # %% create netCDF file to merge with smart measurements
            cosine_cor = cosine_cor.merge(pixel_wl, on="pixel")  # merge wavelength to dataframe
            cosine_diffuse_cor = cosine_diffuse_cor.merge(pixel_wl, on="pixel")  # merge wavelength to dataframe
            # drop useless columns
            cosine_cor = cosine_cor.drop(["prop", "position", "direction", "property", "channel"], axis=1)
            cosine_diffuse_cor = cosine_diffuse_cor.drop(["prop", "direction", "property", "channel", "pixel"], axis=1)
            cosine_cor.set_index(["wavelength", "angle"], inplace=True)
            cosine_diffuse_cor.set_index(["wavelength"], inplace=True)
            cosine_ds = cosine_cor.to_xarray()
            cosine_diffuse_ds = cosine_diffuse_cor.to_xarray()
            cosine_ds["pixel"] = cosine_ds["pixel"].isel(angle=0, drop=True)  # remove angle dimension from pixel
            # select matching correction factors to the sza
            k_cos = cosine_ds["k_cos"].sel(angle=sza, method="nearest", drop=True)
            k_cos = k_cos.drop("angle")  # remove angle dimension
            # filter to high correction factors which come from low sensitivity of the spectrometers at lower wavelengths and
            # lower output of the calibration standard
            max_cor = 0.1  # maximum correction
            # replace correction factors above or below maximum correction with maximum/minimum correction factor
            k_cos = k_cos.where(k_cos < 1 + max_cor, 1 + max_cor).where(k_cos > 1 - max_cor, 1 - max_cor)
            k_cos_diff = cosine_diffuse_ds["k_cos_diff"]
            k_cos_diff = k_cos_diff.where(k_cos_diff < 1 + max_cor, 1 + max_cor).where(k_cos_diff > 1 - max_cor, 1 - max_cor)

    # %% add sza, saa and correction factor to dataset
            ds["sza"] = sza
            ds["saa"] = bacardi_ds["saa"].interp_like(ds.time)
            ds["k_cos_diff"] = k_cos_diff

            # save intermediate output file as backup
            # ds.to_netcdf(f"{outpath}/CIRRUS-HL_HALO_SMART_{direction}_{channel}_{flight[7:-1]}_{flight}_v0.5.nc")

    #  %% correct for cosine response of inlet
            if prop == "Fup":
                # only diffuse radiation -> use only diffuse cosine correction factor
                ds[f"{prop}_cor"] = ds[f"{prop}"] * ds["k_cos_diff"]
            else:
                ds["k_cos"] = k_cos
                ds["direct_fraction"] = f_dir
                # combine direct and diffuse cosine correction factor
                ds[f"{prop}_cor"] = f_dir * ds["k_cos"] * ds[f"{prop}"] + (1 - f_dir) * ds["k_cos_diff"] * ds[f"{prop}"]
                ds[f"{prop}_cor_diff"] = ds["k_cos_diff"] * ds[f"{prop}"]  # correct for cosine assuming only diffuse radiation

    # %% create stabilization flag for Fdw
            if prop == "Fdw":
                stabbi_threshold = 0.1
                try:
                    hori_file = [f for f in os.listdir(hori_dir) if ".dat" in f][0]
                    horidata = reader.read_stabbi_data(f"{hori_dir}/{hori_file}")
                    horidata.index.name = "time"
                    # interpolate to SMART data
                    horidata_ds = horidata.to_xarray()
                    horidata_ds = horidata_ds.interp_like(ds.time)
                    abs_diff = np.abs(horidata_ds["TARGET3"] - horidata_ds["POSN3"])  # difference between roll target and actuall roll
                    stabbi_flag = (abs_diff > stabbi_threshold).astype(int)
                    ds["stabilization_flag"] = xr.DataArray(stabbi_flag, coords=dict(time=ds.time))
                except IndexError:
                    # no stabilization data file can be found -> Stabbi was fixed on 12.07.2021
                    ds["stabilization_flag"] = xr.DataArray(np.ones(len(ds.time), dtype=int) + 1, coords=dict(time=ds.time))
                # save intermediate output
                # ds.to_netcdf(f"{outpath}/CIRRUS-HL_HALO_SMART_{direction}_{channel}_{flight[7:-1]}_{flight}_v0.9.nc")

    # %% filter output
            ds[f"{prop}"] = ds[f"{prop}"].where(ds[f"{prop}"] > 0, np.nan)  # set values < 0 to nan
            ds[f"{prop}_cor"] = ds[f"{prop}_cor"].where(ds[f"{prop}_cor"] > 0, np.nan)  # set values < 0 to nan
            if prop == "Fdw":
                ds[f"{prop}_cor_diff"] = ds[f"{prop}_cor_diff"].where(ds[f"{prop}_cor"] > 0, np.nan)  # set values < 0 to nan

    # %% prepare meta data
            if prop == "Fdw" and flight in stabilized_flights:
                var_attributes = dict(
                    counts=dict(long_name="Dark current corrected spectrometer counts", units="1"),
                    c_field=dict(long_name="Field calibration factor", units="1",
                                 comment=f"Field calibration factor calculated from transfer calibration on {transfer_calib_date}"),
                    direct_fraction=dict(long_name="direct fraction of downward irradiance", units="1",
                                         comment="Calculated from a libRadtran clearsky simulation"),
                    Fdw=dict(long_name="Spectral downward solar irradiance", units="W m-2 nm-1",
                             standard_name="solar_irradiance_per_unit_wavelength",
                             comment="Actively stabilized"),
                    k_cos=dict(long_name="Direct cosine correction factor along track", units="1",
                               comment="Fdw_cor = direct_fraction * k_cos * Fdw + (1 - direct_fraction) * k_cos_diff * Fdw"),
                    k_cos_diff=dict(long_name="Diffuse cosine correction factor", units="1",
                                    comment="Fup_cor = Fup * k_cos_diff"),
                    Fdw_cor=dict(long_name="Spectral downward solar irradiance",
                                 units="W m-2 nm-1",
                                 standard_name="solar_irradiance_per_unit_wavelength",
                                 comment="Actively stabilized and corrected for cosine response of the inlet"),
                    Fdw_cor_diff=dict(long_name="Spectral downward solar irradiance",
                                      units="W m-2 nm-1",
                                      standard_name="solar_irradiance_per_unit_wavelength",
                                      comment="Actively stabilized and corrected for cosine response of the inlet assuming "
                                              "only diffuse radiation"),
                    stabilization_flag=dict(long_name="Stabilization flag", units="1", standard_name="status_flag",
                                            flag_values=[0, 1, 2],
                                            flag_meanings=f"roll_stabilization_performance_good "
                                                          f"roll_stabilization_performance_bad "
                                                          f"stabilization_turned_off",
                                            comment=f"Good Performance: Offset between target and actual roll <= {stabbi_threshold} deg,"
                                                    f"Bad Performance: Offset between target and actual roll > {stabbi_threshold} deg"),
                    wavelength=dict(long_name="Center wavelength of spectrometer pixel", units="nm"))
            elif prop == "Fdw" and flight in unstabilized_flights:
                var_attributes = dict(
                    counts=dict(long_name="Dark current corrected spectrometer counts", units="1"),
                    c_field=dict(long_name="Field calibration factor", units="1",
                                 comment=f"Field calibration factor calculated from transfer calibration on {transfer_calib_date}"),
                    direct_fraction=dict(long_name="direct fraction of downward irradiance", units="1",
                                         comment="Calculated from a libRadtran clearsky simulation"),
                    Fdw=dict(long_name="Spectral downward solar irradiance", units="W m-2 nm-1",
                             standard_name="solar_irradiance_per_unit_wavelength",
                             comment="Not stabilized"),
                    k_cos=dict(long_name="Direct cosine correction factor along track", units="1",
                               comment="Fdw_cor = direct_fraction * k_cos * Fdw + (1 - direct_fraction) * k_cos_diff * Fdw"),
                    k_cos_diff=dict(long_name="Diffuse cosine correction factor", units="1",
                                    comment="Fup_cor = Fup * k_cos_diff"),
                    Fdw_cor=dict(long_name="Spectral downward solar irradiance",
                                 units="W m-2 nm-1",
                                 standard_name="solar_irradiance_per_unit_wavelength",
                                 comment="Attitude corrected (max 40%) and corrected for cosine response of the inlet (max 10%)"),
                    Fdw_cor_diff=dict(long_name="Spectral downward solar irradiance",
                                      units="W m-2 nm-1",
                                      standard_name="solar_irradiance_per_unit_wavelength",
                                      comment="Corrected for cosine response of the inlet (max 10%) assuming only diffuse "
                                              "radiation"),
                    stabilization_flag=dict(long_name="Stabilization flag", units="1", standard_name="status_flag",
                                            flag_values=2, flag_meanings="stabilization_turned_off"),
                    wavelength=dict(long_name="Center wavelength of spectrometer pixel", units="nm"))
            else:
                var_attributes = dict(
                    counts=dict(long_name="Dark current corrected spectrometer counts", units="1"),
                    c_field=dict(long_name="Field calibration factor", units="1",
                                 comment=f"Field calibration factor calculated from transfer calibration on {transfer_calib_date}"),
                    Fup=dict(long_name="Spectral upward solar irradiance", units="W m-2 nm-1",
                             standard_name="solar_irradiance_per_unit_wavelength"),
                    k_cos_diff=dict(long_name="Diffuse cosine correction factor", units="1",
                                    comment="Fup_cor = Fup * k_cos_diff"),
                    Fup_cor=dict(long_name="Spectral upward solar irradiance", units="W m-2 nm-1",
                                 standard_name="solar_irradiance_per_unit_wavelength",
                                 comment="Corrected for cosine response of the inlet assuming only diffuse radiation"),
                    wavelength=dict(long_name="Center wavelength of spectrometer pixel", units="nm"))

            global_attrs = dict(
                title="Spectral irradiance measured by SMART",
                project="SPP 1294 HALO",
                mission="CIRRUS-HL",
                ongoing_subset=f"{flight[7:]}",
                platform="HALO",
                instrument="Spectral Modular Airborne Radiation measurement sysTem (SMART)",
                version="1.0",
                description="Calibrated SMART measurements corrected for dark current and cosine response of inlet",
                institution="Leipzig Institute for Meteorology, Leipzig, Germany",
                date_created=f"{datetime.strftime(datetime.utcnow(), '%c UTC')}",
                history=f"created {datetime.strftime(datetime.utcnow(), '%c UTC')}",
                contact="Johannes Röttenbacher, johannes.roettenbacher@uni-leipzig.de",
                PI="André Ehrlich, a.ehrlich@uni-leipzig.de",
                Conventions="CF-1.8"
            )

            ds.attrs = global_attrs  # assign global attributes

            encoding = dict(time=dict(units="seconds since 2021-01-01 00:00:00 UTC", _FillValue=None),
                            wavelength=dict(_FillValue=None))
            for var in ds:
                encoding[var] = dict(_FillValue=None)  # remove the default _FillValue attribute from each variable
            for var in var_attributes:
                ds[var].attrs = var_attributes[var]

            # set scale factor to reduce file size
            if prop == "Fdw":
                for var in ["Fdw", "Fdw_cor", "Fdw_cor_diff"]:
                    encoding[var] = dict(dtype="int16", scale_factor=0.0001, _FillValue=-999)
            else:
                for var in ["Fup", "Fup_cor"]:
                    encoding[var] = dict(dtype="int16", scale_factor=0.0001, _FillValue=-999)

            outfile = f"{outpath}/CIRRUS-HL_HALO_SMART_{direction}_{channel}_{flight[7:-1]}_{flight}_v1.0.nc"
            ds.to_netcdf(outfile, format="NETCDF4_CLASSIC", encoding=encoding)
            log.info(f"Saved {outfile}")

    # %% merge SWIR and VNIR file
        ds_vnir = xr.open_dataset(f"{outpath}/CIRRUS-HL_HALO_SMART_{prop}_VNIR_{flight[7:-1]}_{flight}_v1.0.nc")
        ds_swir = xr.open_dataset(f"{outpath}/CIRRUS-HL_HALO_SMART_{prop}_SWIR_{flight[7:-1]}_{flight}_v1.0.nc")

        if prop == "Fdw":
            # from 900/950nm onward use the SWIR data
            ds_vnir = ds_vnir.sel(wavelength=slice(300, 899))
            ds_swir = ds_swir.sel(wavelength=slice(900, 2100))

            # interpolate VNIR wavelength to 1nm resolution
            ds_vnir = ds_vnir.interp(wavelength=range(300, 900), kwargs={"fill_value": "extrapolate"})
            # interpolate SWIR wavelength to 5nm resolution
            ds_swir = ds_swir.interp(wavelength=range(900, 2100, 5), kwargs={"fill_value": "extrapolate"})

            # list faulty pixels
            faulty_pixels = [316, 318, 321, 325, 328, 329, 330, 334, 337, 338, 345, 348, 350, 355, 370, 1410, 1415]
        else:
            # from 950nm onward use the SWIR data
            ds_vnir = ds_vnir.sel(wavelength=slice(300, 949))
            ds_swir = ds_swir.sel(wavelength=slice(950, 2100))

            # interpolate VNIR wavelength to 1nm resolution
            ds_vnir = ds_vnir.interp(wavelength=range(300, 950), kwargs={"fill_value": "extrapolate"})
            # interpolate SWIR wavelength to 5nm resolution
            ds_swir = ds_swir.interp(wavelength=range(950, 2100, 5), kwargs={"fill_value": "extrapolate"})

            # list faulty pixels
            faulty_pixels = [1730, 1735]

        ds = xr.merge([ds_vnir, ds_swir])  # merge vnir and swir
        # remove faulty pixels
        ds = ds.where(~ds.wavelength.isin(faulty_pixels), drop=True)
        # drop introduced wavelength dimension from 1D variables, remove them
        ds[["sza", "saa"]] = ds[["sza", "saa"]].isel(wavelength=0, drop=True)
        # replace negative values with 0
        ds[f"{prop}"] = ds[f"{prop}"].where(ds[f"{prop}"] > 0, np.nan)  # set values < 0 to nan
        ds[f"{prop}_cor"] = ds[f"{prop}_cor"].where(ds[f"{prop}_cor"] > 0, np.nan)  # set values < 0 to nan
        if prop == "Fdw":
            ds[f"{prop}_cor_diff"] = ds[f"{prop}_cor_diff"].where(ds[f"{prop}_cor"] > 0, np.nan)  # set values < 0 to 0

        if prop == "Fdw":
            ds["stabilization_flag"] = ds["stabilization_flag"].isel(wavelength=0, drop=True)

    # %% add SMART IMS data
        if flight != "Flight_20210705a" and flight != "Flight_20210729a":
            gps_attrs = dict(
                lat=dict(
                    long_name="latitude",
                    units="degrees_north",
                    comment="GPS latitude measured by the SMART IMS"),
                lon=dict(
                    long_name="longitude",
                    units="degrees_east",
                    comment="GPS longitude measured by the SMART IMS"),
                alt=dict(
                    long_name="altitude",
                    units="m",
                    comment="GPS altitude measured by the SMART IMS"),
                v_east=dict(
                    long_name="Eastward velocity",
                    units="m s^-1",
                    comment="Eastward velocity component as derived from the GPS sensor."),
                v_north=dict(
                    long_name="Northward velocity",
                    units="m s^-1",
                    comment="Northward velocity component as derived from the GPS sensor."),
                v_up=dict(
                    long_name="Upward velocity",
                    units="m s^-1",
                    comment="Vertical velocity as derived from the GPS sensor."),
                vel=dict(
                    long_name="Ground speed",
                    units="m s^-1",
                    comment="Ground speed calculated from northward and eastward velocity component: vel = sqrt(v_north^2 + v_east^2)")
            )
            ims_attrs = dict(
                roll=dict(
                    long_name="Roll angle",
                    units="degrees",
                    comment="Roll angle: positive = left wing up"),
                pitch=dict(
                    long_name="Pitch angle",
                    units="degrees",
                    comment="Pitch angle: positive = nose up"),
                yaw=dict(
                    long_name="Yaw angle",
                    units="degrees",
                    comment="0 = East, 90 = North, 180 = West, -90 = South, range: -180 to 180")
            )
            hori_files = os.listdir(hori_dir)
            nav_filepaths = [f"{hori_dir}/{f}" for f in hori_files if "Nav_IMS" in f]
            gps_filepaths = [f"{hori_dir}/{f}" for f in hori_files if "Nav_GPSPos" in f]
            vel_filepaths = [f"{hori_dir}/{f}" for f in hori_files if "Nav_GPSVel" in f]
            ims = pd.concat([reader.read_nav_data(f) for f in nav_filepaths])
            gps = pd.concat([reader.read_ins_gps_pos(f) for f in gps_filepaths])
            vel = pd.concat([reader.read_ins_gps_vel(f) for f in vel_filepaths])
            gps = gps.merge(vel, how="outer", on="time")
            # IMS pitch is opposite to BAHAMAS pitch, switch signs so that they follow the same convention
            ims["pitch"] = -ims["pitch"]
            # calculate velocity from northerly and easterly part
            gps["vel"] = np.sqrt(gps["v_east"]**2 + gps["v_north"]**2)
            gps = gps.to_xarray()
            ims = ims.to_xarray()

            for var in gps_attrs:
                ds[var] = gps[var].interp_like(ds, method="nearest")
                ds[var].attrs = gps_attrs[var]
            for var in ims_attrs:
                ds[var] = ims[var].interp_like(ds, method="nearest")
                ds[var].attrs = ims_attrs[var]
        else:
            # read in bahamas data and add lat lon and altitude and add to SMART data (Flight_20210705a)
            bahamas_filepath = os.path.join(bahamas_dir, f"CIRRUSHL_{flight_number}_{flight[7:]}_ADLR_BAHAMAS_v1.nc")
            bahamas_ds = reader.read_bahamas(bahamas_filepath)
            # rename variables to fit SMART IMS naming convention
            bahamas_ds = bahamas_ds.rename_vars(
                dict(IRS_LAT="lat", IRS_LON="lon", IRS_ALT="alt", IRS_EWV="v_east", IRS_NSV="v_north", IRS_VV="v_up",
                     IRS_GS="vel", IRS_PHI="roll", IRS_THE="pitch", IRS_HDG="yaw"))
            bahamas_attrs = dict(
                lat=dict(
                    long_name="latitude",
                    units="degrees_north",
                    comment="GPS latitude measured by BAHAMAS"),
                lon=dict(
                    long_name="longitude",
                    units="degrees_east",
                    comment="GPS longitude measured by BAHAMAS"),
                alt=dict(
                    long_name="altitude",
                    units="m",
                    comment="GPS altitude measured by BAHAMAS"),
                v_east=dict(
                    long_name="Eastward velocity",
                    units="m s^-1",
                    comment="Eastward velocity component from BAHAMAS."),
                v_north=dict(
                    long_name="Northward velocity",
                    units="m s^-1",
                    comment="Northward velocity component  from BAHAMAS."),
                v_up=dict(
                    long_name="Upward velocity",
                    units="m s^-1",
                    comment="Vertical velocity  from BAHAMAS."),
                vel=dict(
                    long_name="Ground speed",
                    units="m s^-1",
                    comment="IRS Groundspeed from corrected IGI data (BAHAMAS)"),
                roll=dict(
                    long_name="Roll angle",
                    units="degrees",
                    comment="Roll angle: positive = left wing up (BAHAMAS)"),
                pitch=dict(
                    long_name="Pitch angle",
                    units="degrees",
                    comment="Pitch angle: positive = nose up (BAHAMAS)"),
                yaw=dict(
                    long_name="Yaw angle",
                    units="degrees",
                    comment="0 = East, 90 = North, 180 = West, -90 = South, range: -180 to 180 (converted from 0-360 with 0 = North, BAHAMAS)")
            )
            yaw = bahamas_ds["yaw"] - 90  # convert East from 90 to 0°
            yaw = yaw.where(yaw > 180, yaw * -1)  # switch signs for 1st, 3rd and 4th quadrant
            yaw = yaw.where(yaw < 180, yaw - 180)  # reduce values in 2nd quadrant from 270 to 180 to 90 to 180
            bahamas_ds["yaw"] = yaw
            for var in bahamas_attrs:
                ds[var] = bahamas_ds[var].interp_like(ds, method="nearest")
                ds[var].attrs = bahamas_attrs[var]

    # %% correct attitude with current roll and pitch offset
        if prop == "Fdw" and flight in unstabilized_flights:
            # convert smart from -180 - 180 to 0 - 360
            yaw = ds.yaw - 90
            yaw = yaw.where(yaw < 0, 360 - yaw)
            yaw = yaw.where(yaw > 0, yaw * -1)
            fdw_cor, factor = bacardi.fdw_attitude_correction(ds.Fdw_cor.values, ds["roll"].values, ds.pitch.values,
                                                              yaw.values, ds.sza.values, ds.saa.values,
                                                              ds.direct_fraction, -1.4, 2.9)
            # only correct irradiance by a maximum of 40%
            ds[f"{prop}_cor"] = ds[f"{prop}_cor"].where((np.abs(factor - 1) > 0.4)[:, None], fdw_cor)
            ds["attitude_correction_factor"] = xr.DataArray(factor, coords={"time": ds.time},
                                                            attrs=dict(long_name="Attitude correction factor", units="1",
                                                                       comment="Correction factor for attitude of aircraft for a fixed inlet."
                                                                               "fdw_cor = fdir * fdw * factor + (1 - fdir) * fdw, fdir=direct fraction"))
    # %% set encoding and save file
        encoding = dict(time=dict(units="seconds since 2021-01-01 00:00:00 UTC", _FillValue=None),
                        wavelength=dict(_FillValue=None))
        for var in ds:
            encoding[var] = dict(_FillValue=None)  # remove the default _FillValue attribute from each variable

        # set scale factor to reduce file size
        if prop == "Fdw":
            for var in ["Fdw", "Fdw_cor", "Fdw_cor_diff", "counts"]:
                encoding[var] = dict(dtype="int16", scale_factor=0.0001, _FillValue=-999)
        else:
            for var in ["Fup", "Fup_cor", "counts"]:
                encoding[var] = dict(dtype="int16", scale_factor=0.0001, _FillValue=-999)

        filename = f"{outpath}/CIRRUS-HL_{flight_number}_{flight[7:]}_HALO_SMART_spectral-irradiance-{prop}_v1.0.nc"
        ds.to_netcdf(filename, format="NETCDF4_CLASSIC", encoding=encoding)
        log.info(f"Saved {filename}")
