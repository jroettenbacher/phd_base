#!/usr/bin/env python
"""Data extraction from IFS

*author*: Hanno Müller, Johannes Röttenbacher
"""

# %% module import
from pylim import reader
from pylim import helpers as h
import numpy as np
import xarray as xr
import glob
import os
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import logging
import time
from scipy.interpolate import interp1d

start = time.time()
log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

# %% add solar functions
from pylim.solar_position import get_sza
from pylim.ecrad import calc_pressure, cloud_overlap_decorr_len

# %% read in command line arguments
args = h.read_command_line_args()
ozone_flag = args["ozone"] if "ozone" in args else "sonde"
# set interpolate flag
date = args["date"] if "date" in args else '20210629'
init_time = args["init"] if "init" in args else "00"
flight = args["flight"] if "flight" in args else 'Flight_20210629a'
aircraft = args["aircraft"] if "aircraft" in args else "halo"
campaign = args["campaign"] if "campaign" in args else "cirrus-hl"
dt_day = datetime.strptime(date, '%Y%m%d')  # convert date to date time for further use
# print options to user
log.info(f"Options set: \ncampaign: {campaign}\naircraft: {aircraft}\nflight: {flight}\ndate: {date}"
         f"\ninit time: {init_time}\nozone: {ozone_flag}")

# %% set paths
path_ifs_raw = h.get_path(campaign, "ifs_raw")
path_ifs_raw = f"{path_ifs_raw}/{date}"
path_ifs_output = os.path.join(h.get_path(campaign, "ifs"), date)
path_horidata = h.get_path(campaign, "horidata", flight)
path_ecrad = os.path.join(h.get_path(campaign, "ecrad"), date)
path_ozone = h.get_path(campaign, "ozone")
# create output path
h.make_dir(path_ecrad)
h.make_dir(path_ifs_output)

# %% read ifs files
if init_time == "yesterday":
    ifs_date = int(date) - 1
    init_time = 12
else:
    ifs_date = date
ml_file = f"{path_ifs_raw}/ifs_{ifs_date}_{init_time}_ml.nc"
data_ml = xr.open_dataset(ml_file)
srf_file = f"{path_ifs_raw}/ifs_{ifs_date}_{init_time}_sfc.nc"
data_srf = xr.open_dataset(srf_file)
# test if longitude coordinates are equal
try:
    np.testing.assert_array_equal(data_ml.lon, data_srf.lon)
except AssertionError:
    log.debug("longitude coordinates are not equal, replace srf lon with ml lon")
    data_srf = data_srf.assign_coords(lon=data_ml.lon)

# %% read navigation file
log.info(f"Processed flight: {flight}")

if aircraft == "halo":
    horidata_files = [f for f in os.listdir(path_horidata) if "IMS" in f or "Pos" in f]
    log.info(f"Einzulesende navigation data files: {horidata_files}")
    nav_data = reader.read_nav_data(path_horidata)
    nav_data["time"] = nav_data.index
else:
    horidata_file = glob.glob(os.path.join(path_horidata, "Polar5*.nav"))[0]
    log.info(f"Einzulesendes navigation data file: {horidata_file}")
    colnames = ["time", "lon", "lat", "alt", "vel", "pitch", "roll", "yaw", "sza", "saa"]
    nav_data = pd.read_csv(horidata_file, sep="\s+", skiprows=3, names=colnames, header=None)
    nav_data["seconds"] = nav_data.time * 3600  # convert time to seconds of day

# %% select every step row of nav_data to interpolate IFS data on
step = 2
nav_data_ip = nav_data[::step]
idx = len(nav_data_ip)

# %% convert every nav_data time to datetime
if aircraft == "halo":
    dt_nav_data = nav_data_ip.index.to_pydatetime()
else:
    dt_nav_data = []
    for i in range(0, len(nav_data_ip.time)):
        dt_nav_data.append(dt_day + timedelta(seconds=nav_data_ip.time.iloc[i]))

# %% calculate cosine of solar zenith angle along flight track

# initialize some arrays
sza = np.empty(idx)
cos_sza = np.empty(idx)
closest_lats, closest_lons = list(), list()

for i in tqdm(range(0, idx), desc="Calc cos_sza"):
    # aircraft position
    sod = nav_data_ip.seconds.iloc[i]
    xpos = nav_data_ip.lon.iloc[i]
    ypos = nav_data_ip.lat.iloc[i]
    t_idx = np.abs(data_srf.time - np.datetime64(dt_nav_data[i])).argmin()
    lat_idx = np.abs(data_srf.lat - ypos).argmin()
    lon_idx = np.abs(data_srf.lon - xpos).argmin()
    p_surf_nearest = data_srf.SP[t_idx, lat_idx, lon_idx].values / 100  # hPa
    t_surf_nearest = data_srf.SKT[t_idx, lat_idx, lon_idx].values - 273.15  # degree Celsius
    sza[i] = get_sza(sod / 3600, ypos, xpos, dt_day.year, dt_day.month, dt_day.day, p_surf_nearest, t_surf_nearest)
    cos_sza[i] = np.cos(sza[i] / 180. * np.pi)
    closest_lats.append(lat_idx.values)
    closest_lons.append(lon_idx.values)

# %% add cos_sza to navdata
nav_data_ip = nav_data_ip.assign(cos_sza=cos_sza)

# %% calculate decorrelation length to put into namelist
decorr_len, b, c = cloud_overlap_decorr_len(np.median(nav_data_ip.lat), 1)  # operational scheme 1
log.info(f"Decorrelation length for namelist file: {decorr_len}")

# %% select only closest (+-10) lats and lons from datasets to reduce size in memory
closest_lats = np.unique(closest_lats)
closest_lons = np.unique(closest_lons)
# include 10 points in both direction to allow for a rectangle selection around the closest lat lon point and to avoid
# edge cases
lat_sel = np.arange(closest_lats.min() - 10, closest_lats.max() + 10)
lon_sel = np.arange(closest_lons.min() - 10, closest_lons.max() + 10)
data_ml = data_ml.isel(lat=lat_sel, lon=lon_sel)
data_srf = data_srf.isel(lat=lat_sel, lon=lon_sel)

# %% calculate pressure and modify datasets
data_ml = calc_pressure(data_ml)
data_ml = data_ml.sel(lev_2=1).reset_coords("lev_2", drop=True)  # drop lev_2 dimension
data_ml = data_ml.rename({"nhyi": "half_level", "lev": "level"})  # rename variables and thus dimensions

# calculate lw_em from ratio to blackbody temperature
sigma = 5.670374419e-8
bb_radiation = sigma * data_srf.SKT ** 4
# unit of surface thermal radiation is W m^-2 s, thus it needs to be divided with the seconds after the hour
seconds_after_hour = data_srf.time.dt.hour * 3600
seconds_after_hour[-1] = 25 * 3600  # replace the last value (0 UTC next day) with 25 hrs after start
lw_up_radiation = (data_srf.STRD - data_srf.STR) / seconds_after_hour
lw_em_ratio = lw_up_radiation / bb_radiation
# correct lw_em_ratios > 0.98 (probably due to difference between skin and sea surface temperature)
lw_em_ratio = lw_em_ratio.where(lw_em_ratio < 0.98, 0.98)

# calculate shortwave albedo according to sea ice concentration and shot wave band albedo climatology for sea ice
month_idx = dt_day.month - 1
open_ocean_albedo = 0.06
sw_albedo_bands = list()
for i in range(h.ci_albedo.shape[1]):
    sw_albedo_bands.append(data_srf.CI * h.ci_albedo[month_idx, i] + (1. - data_srf.CI) * open_ocean_albedo)

sw_albedo = xr.concat(sw_albedo_bands, dim="sw_albedo_band")
sw_albedo.attrs = dict(unit=1, long_name="Banded short wave albedo")
# set sw_albedo to constant 0.2 when over land
sw_albedo.where(data_srf.LSM < 0.5, 0.2)


data_srf = data_srf[["SKT", "U10M", "V10M", "LSM", "CI"]]  # select only relevant variables
data_ml = data_ml.drop_vars(["lnsp", "hyam", "hybm", "hyai", "hybi"])  # drop unnecessary variables
# interpolate temperature on half levels
data_ml["temperature_hl"] = data_ml.t.interp(level=np.insert((data_ml.level + 0.5).values, 0, 0.5),
                                             kwargs={"fill_value": "extrapolate"}).rename(level="half_level")
# replace last (and wrong) extrapolated value with surface temp
data_ml["temperature_hl"][:, -1, :, :] = data_srf.SKT.values
# first (0.5 half level) point is also extrapolated but not as important as surface temperature
n_levels = len(data_ml.level)  # get number of levels

# %% prepare datasets for merge and write to ncfile
data_ml = data_ml.expand_dims("column")  # add the new dimension "column"
# rename surface variables and add new dimension "column"
data_srf = data_srf.rename({"U10M": "u_wind_10m",
                            "V10M": "v_wind_10m",
                            "SKT": "skin_temperature"}
                           ).expand_dims("column")

# %% add a few more necessary variables to the dataset
if ozone_flag == "sonde":
    log.info(f"ozone flag set to {ozone_flag}\ninterpolating sonde measurement onto IFS full pressure levels...")
    # read the corresponding ozone file for the flight
    ozone_file = h.ozone_files[flight]
    # interpolate ozone sonde data on IFS full pressure levels
    ozone_sonde = reader.read_ozone_sonde(f"{path_ozone}/{ozone_file}")
    ozone_sonde["Press"] = ozone_sonde["Press"] * 100  # convert hPa to Pa
    # create interpolation function
    f_ozone = interp1d(ozone_sonde["Press"], ozone_sonde["o3_vmr"], fill_value="extrapolate", bounds_error=False)
    ozone_interp = f_ozone(data_ml.pressure_full.isel(lat=0, lon=0, time=0, column=0))
    # copy interpolated ozone concentration over time, lat and longitude dimension
    o3_t = np.concatenate([ozone_interp[..., None]] * data_ml.dims["time"], axis=1)
    o3_lat = np.concatenate([o3_t[..., None]] * data_ml.dims["lat"], axis=2)
    o3_lon = np.concatenate([o3_lat[..., None]] * data_ml.dims["lon"], axis=3)
    # create DataArray
    o3_vmr = xr.DataArray(o3_lon, coords=data_ml.pressure_full.coords, dims=["level", "time", "lat", "lon"])
    o3_vmr = o3_vmr.where(o3_vmr > 0, 0)  # set negative values to 0
    o3_vmr = o3_vmr.where(~np.isnan(o3_vmr), 0)  # set nan values to 0
    o3_vmr.attrs = dict(unit="1", long_name="Ozone volume mass mixing ratio")
    data_ml["o3_vmr"] = o3_vmr

else:
    assert ozone_flag == "default", f"ozone flag set to unrecognized value {ozone_flag}! Check input!"
    data_ml["o3_mmr"] = xr.DataArray(np.expand_dims(np.repeat([1.587701e-7], n_levels), axis=0),
                                     dims=["column", "level"],
                                     attrs=dict(unit="1", long_name="Ozone mass mixing ratio"))

data_ml["cfc11_vmr"] = xr.DataArray(2.51e-10, attrs=dict(unit="1", long_name="CFC11 volume mixing ratio"))
data_ml["cfc12_vmr"] = xr.DataArray(5.38e-10, attrs=dict(unit="1", long_name="CFC12 volume mixing ratio"))
data_ml["ch4_vmr"] = xr.DataArray(1.774e-6, attrs=dict(unit="1", long_name="CH4 volume mixing ratio"))
data_ml["co2_vmr"] = xr.DataArray(0.000379, attrs=dict(unit="1", long_name="CO2 volume mixing ratio"))
data_ml["fractional_std"] = xr.DataArray(np.expand_dims(np.repeat([1.], n_levels), axis=0), dims=["column", "level"])
data_ml["inv_cloud_effective_size"] = xr.DataArray(np.expand_dims(np.repeat([0.0013], n_levels), axis=0),
                                                   dims=["column", "level"])
data_ml["lw_emissivity"] = xr.DataArray(lw_em_ratio, dims=["time", "lat", "lon"],
                                        attrs=dict(unit="1", long_name="Longwave surface emissivity"))
data_ml["n2o_vmr"] = xr.DataArray(3.19e-7, attrs=dict(unit="1", long_name="N2O volume mixing ratio"))
data_ml["o2_vmr"] = xr.DataArray(0.209488, attrs=dict(unit="1", long_name="Oxygen volume mixing ratio"))
data_ml["overlap_param"] = xr.DataArray(np.expand_dims(np.repeat([1.], n_levels - 1), axis=0),
                                        dims=["column", "mid_level"], coords={"mid_level": np.arange(1.5, 137, 1)},
                                        attrs=dict(unit="1", long_name="Cloud overlap parameter"))
data_ml["q_ice"] = data_ml.ciwc + data_ml.cswc
data_ml["q_liquid"] = data_ml.clwc + data_ml.crwc
data_ml["sw_albedo"] = sw_albedo
data_ml = data_ml.merge(data_srf)

# %% rename variables for ecrad
data_ml = data_ml.rename(dict(cc="cloud_fraction"))

# write to history attribute
data_ml.attrs["history"] = data_ml.attrs["history"] + f" {datetime.today().strftime('%c')}: " \
                                                      f"formatted file to serve as input to ecRad (read_ifs.py)"
data_ml.attrs["contact"] = f"hanno.mueller@uni-leipzig.de, johannes.roettenbacher@uni-leipzig.de"

# %% write intermediate output files
filename = f"{path_ifs_output}/ifs_{ifs_date}_{init_time}_ml_processed.nc"
data_ml.to_netcdf(filename, format='NETCDF4_CLASSIC')
log.info(f"Saved {filename}")
csv_filename = f"{path_ifs_output}/nav_data_ip_{date}.csv"
nav_data_ip.to_csv(csv_filename)
log.info(f"Saved {csv_filename}")

log.info(f"Done with date {date}: {h.seconds_to_fstring(time.time() - start)} [hr:min:sec]")
