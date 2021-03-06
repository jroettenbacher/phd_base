#!/usr/bin/env python
"""Complete calibration of the SMART files for the HALO-AC3 campaign

The dark current corrected SMART measurement files are calibrated and filtered.

- read in dark current corrected files
- calibrate with matching transfer calibration
- correct measurement for cosine dependence of inlet
- add some metadata such as sza and saa
- add stabilization flag
- write to netcdf file (VNIR and SWIR separate)
- merge VNIR and SWIR data

*author*: Johannes Röttenbacher
"""

# %% module import
from pylim import helpers as h
from pylim import halo_ac3 as campaign_meta
from pylim import smart, reader
import os
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
import logging

log = logging.getLogger("pylim")
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

# %% set some options
campaign = "halo-ac3"
flight = "HALO-AC3_20220411_HALO_RF17"
date = flight[9:17]
key = flight[-4:]
prop = "Fdw"
normalize = True  # use normalized calibration factor (counts are divided by the integration time)
lab_calib = "after"  # before or after, set which lab calibration to use for the transfer calibration
t_int_asp06 = 300  # give integration time of field measurement for ASP06

# %% set paths
cor_data_dir = h.get_path("data", flight, campaign)
inpath = cor_data_dir
calib_data_dir = h.get_path("calibrated", flight, campaign)
outpath = calib_data_dir
calib_dir = f"{h.get_path('calib', campaign=campaign)}/transfer_calibs_{lab_calib}_campaign"  # path to transfer calibration files
pixel_dir = h.get_path("pixel_wl")
ql_dir = h.get_path("quicklooks", flight, campaign)
hori_dir = h.get_path("horidata", flight, campaign)
bahamas_dir = h.get_path("bahamas", flight, campaign)
bacardi_dir = h.get_path("bacardi", flight, campaign)
libradtran_dir = h.get_path("libradtran", flight, campaign)
cosine_dir = h.get_path("cosine")

# %% get metadata
transfer_calib_date = campaign_meta.transfer_calibs[key]
t_date = f"{transfer_calib_date[:4]}_{transfer_calib_date[4:6]}_{transfer_calib_date[6:]}"  # reformat date to match file name
norm = "_norm" if normalize else ""
to, td = campaign_meta.take_offs_landings[key]

# %% read in dark current corrected measurement files
files = [f for f in os.listdir(inpath) if prop in f]
for file in files:
    date_str, channel, direction = smart.get_info_from_filename(file)
    inlet = campaign_meta.smart_lookup[direction]
    date_str = t_date if len(t_date) > 0 else date_str  # overwrite date_str if date for transfer calib is given
    spectrometer = campaign_meta.smart_lookup[f"{direction}_{channel}"]
    pixel_wl = reader.read_pixel_to_wavelength(pixel_dir, spectrometer)  # read in pixel to wavelength mapping
    measurement = reader.read_smart_cor(inpath, file)
    # cut measurement to take off and landing times
    measurement = measurement.loc[to.strftime("%Y-%m-%d %H:%M:%S"):td.strftime("%Y-%m-%d %H:%M:%S")]

    # %% read in matching transfer calibration file from same day or from given day with matching t_int
    cali_file = f"{calib_dir}/{date_str}_{spectrometer}_{direction}_{channel}_{t_int_asp06}ms_transfer_calib{norm}.dat"
    log.info(f"Calibration file used:\n {cali_file}")
    cali = pd.read_csv(cali_file)
    # convert to long format
    m_long = measurement.melt(var_name="pixel", value_name="counts", ignore_index=False)
    if normalize:
        m_long["counts"] = m_long["counts"] / t_int_asp06

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
    bacardi_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}.nc"
    libradtran_file = f"HALO-AC3_HALO_libRadtran_clearsky_simulation_smart_{date}_{key}.nc"
    bacardi = xr.open_dataset(f"{bacardi_dir}/{bacardi_file}")
    libradtran = xr.open_dataset(f"{libradtran_dir}/{libradtran_file}")
    # extract sza from BACARDI file
    sza = bacardi["sza"]
    # extract direct fraction from libRadtran
    f_dir = libradtran["direct_fraction"]
    # interpolate to SMART time
    sza = sza.interp_like(ds.time)
    f_dir = f_dir.interp_like(ds.time, method="nearest")
    f_dir = f_dir.interp_like(ds.wavelength)  # interpolate to SMART wavelength
    f_dir = f_dir.where(f_dir <= 1, 1)  # replace values higher 1 with 1
    # replace values which seem to low (4 standard deviations below the mean) with the mean
    f_dir = f_dir.where(f_dir > (f_dir.mean() - 4 * f_dir.std()), f_dir.mean())

    # read in cosine correction factors
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

    # create netCDF file to merge with smart measurements
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
    # filter to high correction factors which come from low sensitivity of the spectrometers at lower wavelengths and
    # lower output of the calibration standard
    k_cos = k_cos.where(k_cos < 1.1, 1.1)  # allow a maximum correction of 10%

    # add sza, saa and correction factor to dataset
    ds["sza"] = sza
    ds["saa"] = bacardi["saa"].interp_like(ds.time)
    ds["k_cos_diff"] = cosine_diffuse_ds.k_cos_diff

    # save intermediate output file as backup
    # ds.to_netcdf(f"{outpath}/CIRRUS-HL_HALO_SMART_{direction}_{channel}_{flight[7:-1]}_{flight}_v0.5.nc")

    # correct for cosine response of inlet
    ds["k_cos"] = k_cos
    ds["direct_fraction"] = f_dir
    # combine direct and diffuse cosine correction factor
    ds[f"{prop}_cor"] = f_dir * ds["k_cos"] * ds[f"{prop}"] + (1 - f_dir) * ds["k_cos_diff"] * ds[f"{prop}"]
    ds[f"{prop}_cor_diff"] = ds["k_cos_diff"] * ds[f"{prop}"]  # correct for cosine assuming only diffuse radiation

    # %% create stabilization flag for Fdw
    hori_file = [f for f in os.listdir(hori_dir) if ".dat" in f][0]
    horidata = reader.read_stabbi_data(f"{hori_dir}/{hori_file}")
    horidata.index.name = "time"
    # interpolate to SMART data
    horidata_ds = horidata.to_xarray()
    horidata_ds = horidata_ds.interp_like(ds.time)
    abs_diff = np.abs(horidata_ds["TARGET3"] - horidata_ds["POSN3"])  # difference between roll target and actuall roll
    stabbi_threshold = 0.1
    stabbi_flag = (abs_diff > stabbi_threshold).astype(int)
    ds["stabilization_flag"] = stabbi_flag
    # save intermediate output
    # ds.to_netcdf(f"{outpath}/CIRRUS-HL_HALO_SMART_{direction}_{channel}_{flight[7:-1]}_{flight}_v0.9.nc")

    # %% filter output
    ds[f"{prop}"] = ds[f"{prop}"].where(ds[f"{prop}"] > 0, 0)  # set values < 0 to 0
    ds[f"{prop}_cor"] = ds[f"{prop}_cor"].where(ds[f"{prop}_cor"] > 0, 0)  # set values < 0 to 0
    ds[f"{prop}_cor_diff"] = ds[f"{prop}_cor_diff"].where(ds[f"{prop}_cor"] > 0, 0)  # set values < 0 to 0

    # %% write to netCDF file

    var_attributes = dict(
        counts=dict(long_name="Dark current corrected spectrometer counts", units="1"),
        c_field=dict(long_name="Field calibration factor", units="1",
                     comment=f"Field calibration factor calculated from transfer calibration on {transfer_calib_date}"),
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
                          comment="Actively stabilized and corrected for cosine response of the inlet assuming"
                                  "only diffuse radiation"),
        stabilization_flag=dict(long_name="Stabilization flag", units="1",
                                comment=f"0: Roll Stabilization performed good "
                                        f"(Offset between target and actual roll <= {stabbi_threshold} deg), "
                                        f"1: Roll Stabilization was not performing good "
                                        f"(Offset between target and actual roll > {stabbi_threshold} deg)"),
        # Fdw_flagged=dict(long_name="Stabilization flagged, cosine corrected spectral downward solar irradiance",
        #                  units="W m-2 nm-1", standard_name="solar_irradiance_per_unit_wavelength",
        #                  comment="Actively stabilized, corrected for cosine response of the inlet and "
        #                          "filtered for stabilization performance"),
        wavelength=dict(long_name="Center wavelength of spectrometer pixel", units="nm"))

    global_attrs = dict(
        title="Spectral irradiance measured by SMART",
        project="(AC)³ and SPP 1294 HALO",
        mission="HALO-(AC)³",
        ongoing_subset=flight,
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

    encoding = dict(time=dict(units="seconds since 2017-01-01 00:00:00 UTC", _FillValue=None),
                    wavelength=dict(_FillValue=None))
    for var in ds:
        encoding[var] = dict(_FillValue=None)  # remove the default _FillValue attribute from each variable
    for var in var_attributes:
        ds[var].attrs = var_attributes[var]

    # set scale factor to reduce file size
    for var in ["Fdw", "Fdw_cor", "Fdw_cor_diff"]:
        encoding[var] = dict(dtype="int16", scale_factor=0.01, _FillValue=-999)

    outfile = f"{outpath}/HALO-AC3_HALO_SMART_spectral_irradiance_{direction}_{channel}_{date}_{key}_v1.0.nc"
    ds.to_netcdf(outfile, format="NETCDF4_CLASSIC", encoding=encoding)
    log.info(f"Saved {outfile}")

# %% merge SWIR and VNIR file
ds_vnir = xr.open_dataset(f"{outpath}/HALO-AC3_HALO_SMART_spectral_irradiance_{prop}_VNIR_{date}_{key}_v1.0.nc")
ds_swir = xr.open_dataset(f"{outpath}/HALO-AC3_HALO_SMART_spectral_irradiance_{prop}_SWIR_{date}_{key}_v1.0.nc")

# from 900 onward use the SWIR data
ds_vnir = ds_vnir.sel(wavelength=slice(300, 899))
ds_swir = ds_swir.sel(wavelength=slice(900, 2100))

# interpolate VNIR wavelength to 1nm resolution
ds_vnir = ds_vnir.interp(wavelength=range(300, 900), kwargs={"fill_value": "extrapolate"})
# interpolate SWIR wavelength to 5nm resolution
ds_swir = ds_swir.interp(wavelength=range(900, 2100, 5), kwargs={"fill_value": "extrapolate"})

# list faulty pixels
faulty_pixels = [316, 318, 321, 325, 328, 329, 330, 334, 337, 338, 345, 348, 350, 355, 370, 1410, 1415]

ds = xr.merge([ds_vnir, ds_swir])  # merge vnir and swir
# remove faulty pixels
ds = ds.where(~ds.wavelength.isin(faulty_pixels), drop=True)
# drop introduced wavelength dimension from 1D variables
ds[["sza", "saa"]] = ds[["sza", "saa"]].isel(wavelength=0, drop=True)
if prop == "Fdw":
    ds["stabilization_flag"] = ds["stabilization_flag"].isel(wavelength=0, drop=True)

encoding = dict(time=dict(units="seconds since 2021-01-01 00:00:00 UTC", _FillValue=None),
                wavelength=dict(_FillValue=None))
for var in ds:
    encoding[var] = dict(_FillValue=None)  # remove the default _FillValue attribute from each variable

# set scale factor to reduce file size
for var in ["Fdw", "Fdw_cor", "Fdw_cor_diff", "counts"]:
    encoding[var] = dict(dtype="int16", scale_factor=0.01, _FillValue=-999)

filename = f"{outpath}/HALO-AC3_HALO_SMART_spectral_irradiance_{prop}_{date}_{key}_v1.0.nc"
ds.to_netcdf(filename, format="NETCDF4_CLASSIC", encoding=encoding)
log.info(f"Saved {filename}")
