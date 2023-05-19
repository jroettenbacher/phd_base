#!/usr/bin/env python
"""Data extraction from IFS along the flight track

**TODO:**

- [x] more precise read in of navigation data
- [x] test if it is necessary to generate one file per time step |rarr| it is
- [ ] Include an option to interpolate in space
- [x] check why file 1919-1925 + 6136-6180 + 7184-7194 cause a floating-point exception when processed with ecrad (discovered 2021-04-26) -> probably has to do the way ecrad is called (see execute_IFS.sh)

**Input:**

* IFS model level (ml) file
* IFS surface (sfc) file
* SMART horidata with flight track or BAHAMAS file

**Required User Input:**

Can be passed via the command line (except step).

* ozone_flag ('default' or 'sonde')
* date (yyyymmdd)
* init_time (00, 12 or yesterday)
* flight (e.g. 'Flight_20210629a' or 'HALO-AC3_20220412_HALO_RF18')
* aircraft ('halo')
* campaign ('cirrus-hl' or 'halo-ac3')
* step, one can choose the time resolution on which to interpolate the IFS data on (e.g. '1Min')
* use_bahamas, whether to use BAHAMAS or the SMART INS for navigation data (True/False)

**Output:**

* processed IFS file for input to :ref:`processing:ecrad_write_input_files.py`
* decorrelation length for ecRad namelist file |rarr| manually change that in the namelist file


*author*: Hanno Müller, Johannes Röttenbacher
"""

if __name__ == "__main__":
    # %% module import
    from pylim import reader
    from pylim import helpers as h
    from pylim.solar_position import get_sza
    from pylim.ecrad import calc_pressure, cloud_overlap_decorr_len, calculate_pressure_height
    import numpy as np
    import xarray as xr
    import glob
    import os
    import pandas as pd
    from datetime import datetime, timedelta
    from tqdm import tqdm
    import time
    from scipy.interpolate import interp1d
    from distutils.util import strtobool

    start = time.time()

    # %% read in command line arguments
    args = h.read_command_line_args()
    flight_key = args["flight_key"] if "flight_key" in args else "RF17"
    campaign = args["campaign"] if "campaign" in args else "halo-ac3"
    init_time = args["init"] if "init" in args else "00"
    aircraft = args["aircraft"] if "aircraft" in args else "halo"
    use_bahamas = strtobool(args["use_bahamas"]) if "use_bahamas" in args else True
    ozone_flag = args["ozone"] if "ozone" in args else "sonde"
    if campaign == "halo-ac3":
        import pylim.halo_ac3 as meta
    else:
        import pylim.cirrus_hl as meta
    flight = meta.flight_names[flight_key]
    date = flight[9:17] if campaign == "halo-ac3" else flight[7:15]
    dt_day = datetime.strptime(date, '%Y%m%d')  # convert date to date time for further use
    # setup logging
    try:
        file = __file__
    except NameError:
        file = None
    log = h.setup_logging("./logs", file, flight_key)

    # print options to user
    log.info(f"Options set: \ncampaign: {campaign}\naircraft: {aircraft}\nflight: {flight}\ndate: {date}"
             f"\ninit time: {init_time}\nozone: {ozone_flag}\nuse_bahamas: {use_bahamas}")

    # %% set paths
    path_ifs_raw = f"{h.get_path('ifs_raw', campaign=campaign)}/{date}"
    path_ifs_output = os.path.join(h.get_path("ifs", campaign=campaign), date)
    path_horidata = h.get_path("horidata", flight, campaign)
    path_ecrad = os.path.join(h.get_path("ecrad", campaign=campaign), date)
    path_ozone = h.get_path("ozone", campaign=campaign)
    bahamas_path = h.get_path("bahamas", flight, campaign)
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
    # hack while only incomplete ml file is available
    # data_srf = data_srf.sel(lat=slice(data_ml.lat.max(), data_ml.lat.min()))

    # test if longitude coordinates are equal
    try:
        np.testing.assert_array_equal(data_ml.lon, data_srf.lon)
    except AssertionError:
        log.debug("longitude coordinates are not equal, replace srf lon with ml lon")
        data_srf = data_srf.assign_coords(lon=data_ml.lon)
    # try:
    #     np.testing.assert_array_equal(data_ml.lat, data_srf.lat)
    # except AssertionError:
    #     log.debug("latitude coordinates are not equal, replace srf lat with ml lat")
    #     data_srf = data_srf.assign_coords(lat=data_ml.lat)

    # %% read navigation file
    log.info(f"Processed flight: {flight}")

    if aircraft == "halo":
        if use_bahamas:
            bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{flight_key}_v1.nc"
            nav_data = reader.read_bahamas(f"{bahamas_path}/{bahamas_file}")
            nav_data = nav_data.to_dataframe()
            nav_data.rename(columns={"IRS_LAT": "lat", "IRS_LON": "lon"}, inplace=True)
            nav_data = nav_data.loc[:, ["lon", "lat"]]
        else:
            gps_file = [f for f in os.listdir(path_horidata) if "Pos" in f][0]
            log.info(f"Einzulesendes GPS navigation file: {gps_file}")
            nav_data = reader.read_ins_gps_pos(f"{path_horidata}/{gps_file}")
    else:
        horidata_file = glob.glob(os.path.join(path_horidata, "Polar5*.nav"))[0]
        log.info(f"Einzulesendes navigation data file: {horidata_file}")
        colnames = ["time", "lon", "lat", "alt", "vel", "pitch", "roll", "yaw", "sza", "saa"]
        nav_data = pd.read_csv(horidata_file, sep="\s+", skiprows=3, names=colnames, header=None)
        nav_data["seconds"] = nav_data.time * 3600  # convert time to seconds of day

    # %% select every step row of nav_data to interpolate IFS data on
    step = "1Min"  # User input
    nav_data_ip = nav_data.resample(step).mean()
    idx = len(nav_data_ip)
    nav_data_ip["seconds"] = nav_data_ip.index.hour * 60 * 60 + nav_data_ip.index.minute * 60

    # %% convert every nav_data time to datetime
    if aircraft == "halo":
        dt_nav_data = nav_data_ip.index.to_pydatetime()
    else:
        dt_nav_data = []
        for i in range(0, len(nav_data_ip.time)):
            dt_nav_data.append(dt_day + timedelta(seconds=nav_data_ip.time.iloc[i]))

    # %% calculate cosine of solar zenith angle along flight track
    sza = np.empty(idx)  # initialize some arrays
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
        p_surf_nearest = data_srf.SP.isel(time=t_idx, lat=lat_idx, lon=lon_idx).values / 100  # hPa
        t_surf_nearest = data_srf.SKT.isel(time=t_idx, lat=lat_idx, lon=lon_idx).values - 273.15  # degree Celsius
        sza[i] = get_sza(sod / 3600, ypos, xpos, dt_day.year, dt_day.month, dt_day.day, p_surf_nearest, t_surf_nearest)
        cos_sza[i] = np.cos(sza[i] / 180. * np.pi)
        closest_lats.append(lat_idx.values)
        closest_lons.append(lon_idx.values)

    # %% add cos_sza to navdata
    nav_data_ip = nav_data_ip.assign(cos_sza=cos_sza)

    # %% calculate decorrelation length to put into namelist
    decorr_len, b, c = cloud_overlap_decorr_len(nav_data_ip.lat, 1)  # operational scheme 1
    decorr_file = f"{path_ifs_output}/{date}_decorrelation_length.csv"
    decorr_len.to_csv(decorr_file)
    log.info(f"Decorrelation length saved in {decorr_file}")

    # %% select only closest (+-10) lats and lons from datasets to reduce size in memory
    closest_lats = np.unique(closest_lats)
    closest_lons = np.unique(closest_lons)
    # include 10 points in both direction to allow for a rectangle selection around the closest lat lon point
    # to avoid edge cases
    lat_sel = np.arange(closest_lats.min() - 10, closest_lats.max() + 10)
    lon_sel = np.arange(closest_lons.min() - 10, closest_lons.max() + 10)
    # make sure that all latitude indices are available in the file
    lat_sel = lat_sel[np.where(lat_sel < data_ml.sizes["lat"])[0]]
    # drop indices < 0
    lat_sel = lat_sel[np.where(lat_sel >= 0)[0]]
    data_ml = data_ml.isel(lat=lat_sel, lon=lon_sel)
    data_srf = data_srf.isel(lat=lat_sel, lon=lon_sel)

    # %% calculate pressure and modify datasets
    data_ml = calc_pressure(data_ml)
    data_ml = data_ml.sel(lev_2=1).reset_coords("lev_2", drop=True)  # drop lev_2 dimension
    data_ml = data_ml.rename({"nhyi": "half_level", "lev": "level"})  # rename variables and thus dimensions
    data_ml = data_ml.assign_coords(half_level=np.insert((data_ml.level + 0.5).values, 0, 0.5))
    # transpose pressure data so time is the first dimension
    for var in ["pressure_hl", "pressure_full"]:
        data_ml[var] = data_ml[var].transpose("time", ...)


    # %% calculate lw_em from ratio to blackbody temperature
    # sigma = 5.670374419e-8  # Stefan-Boltzmann constant W⋅m−2⋅K−4
    # bb_radiation = sigma * data_srf.SKT ** 4
    # unit of surface thermal radiation is W m^-2 s, thus it needs to be divided with the seconds after the hour
    # seconds_after_hour = data_srf.time.dt.hour * 3600
    # seconds_after_hour[-1] = 24 * 3600  # replace the last value (0 UTC next day) with 24 hrs after start
    # lw_up_radiation = (data_srf.STRD - data_srf.STR) / seconds_after_hour
    # lw_em_ratio = lw_up_radiation / bb_radiation
    # correct lw_em_ratios > 0.98 (the longwave emissivity for sea ice is assumed to be constant at 0.98,
    # IFS documentation Part IV, Table 2.6)
    # lw_em_ratio = lw_em_ratio.where(lw_em_ratio < 0.98, 0.98)
    # data_ml["lw_emissivity"] = xr.DataArray(lw_em_ratio, dims=["time", "lat", "lon"],
    #                                         attrs=dict(unit="1", long_name="Longwave surface emissivity"))

    # %% set longwave emissivity as described in IFS documentation Part IV Chapter 2.8.5
    lw_em_shape = data_srf.SKT.shape + (2,)
    lw_em_ratio = np.ones(lw_em_shape)
    # The thermal emissivity of the surface outside the 800–1250 cm−1 spectral region is assumed to be 0.99 everywhere
    lw_em_ratio[:, :, :, 0] = 0.99
    # In the window region, the spectral emissivity is constant for open water, sea ice, the interception layer and
    # exposed snow tiles.
    lw_em_ratio[:, :, :, 1] = 0.98  # see Table 2.6
    data_ml["lw_emissivity"] = xr.DataArray(lw_em_ratio, dims=["time", "lat", "lon", "lw_emiss_band"],
                                            attrs=dict(unit="1", long_name="Longwave surface emissivity"))
    # %% calculate shortwave albedo according to sea ice concentration and shortwave band albedo climatology for sea ice
    month_idx = dt_day.month - 1
    open_ocean_albedo = 0.06
    sw_albedo_bands = list()
    for i in range(h.ci_albedo.shape[1]):
        sw_albedo_bands.append(data_srf.CI * h.ci_albedo[month_idx, i] + (1. - data_srf.CI) * open_ocean_albedo)

    sw_albedo = xr.concat(sw_albedo_bands, dim="sw_albedo_band")
    sw_albedo.attrs = dict(unit=1, long_name="Banded short wave albedo")
    sw_albedo = sw_albedo.transpose("time", ...)  # transpose so time is first dimension
    # set sw_albedo to constant 0.2 when over land
    sw_albedo = sw_albedo.where(data_srf.LSM < 0.5, 0.2)

    # %% select only relevant variables
    data_srf = data_srf[["SKT", "U10M", "V10M", "LSM", "CI", "MSL"]]
    data_ml = data_ml.drop_vars(["lnsp", "hyam", "hybm", "hyai", "hybi"])

    # %% interpolate temperature on half levels according to IFS Documentation Part IV Section 2.8.1
    n_levels = len(data_ml.level)  # get number of levels
    t_hl = list()
    for i in tqdm(range(n_levels-1), desc="Temperature Interpolation on Half Levels"):
        t_k0 = data_ml.t.isel(level=i, drop=True)
        t_k1 = data_ml.t.isel(level=i+1, drop=True)
        p_k0 = data_ml.pressure_full.isel(level=i, drop=True)
        p_k1 = data_ml.pressure_full.isel(level=i+1, drop=True)
        p_k05 = data_ml.pressure_hl.isel(half_level=i+1)
        t_k0_weight = p_k0 * (p_k1 - p_k05) / (p_k05 * (p_k1 - p_k0))
        t_k1_weight = p_k1 * (p_k05 - p_k0) / (p_k05 * (p_k1 - p_k0))
        t_k05 = (t_k0 * t_k0_weight) + (t_k1 * t_k1_weight)
        t_hl.append(t_k05)

    t_1375 = data_srf.SKT.expand_dims(half_level=[137.5]).isel(half_level=0)
    t_hl.append(t_1375)  # set surface temperature to skin temperature
    diff_t15_t1 = t_hl[0] - data_ml.t.sel(level=1, drop=True)
    t_05 = (data_ml.t.sel(level=1, drop=True) - diff_t15_t1).assign_coords(half_level=0.5)
    t_hl.insert(0, t_05)  # interpolate 0.5 half level
    data_ml["temperature_hl"] = xr.concat(t_hl, dim="half_level").transpose("time", ...)

    # %% calculate pressure height for model half and full levels
    data_ml = calculate_pressure_height(data_ml)

    # %% rename surface variables
    data_srf = data_srf.rename({"U10M": "u_wind_10m",
                                "V10M": "v_wind_10m",
                                "SKT": "skin_temperature",
                                "MSL": "mean_sea_level_pressure"}
                               )

    # %% add trace gases
    if ozone_flag == "sonde":
        log.info(f"ozone flag set to {ozone_flag}\ninterpolating sonde measurement onto IFS full pressure levels...")
        # read the corresponding ozone file for the flight
        ozone_file = h.ozone_files[flight_key]
        # interpolate ozone sonde data on IFS full pressure levels
        ozone_sonde = reader.read_ozone_sonde(f"{path_ozone}/{ozone_file}")
        ozone_sonde["Press"] = ozone_sonde["Press"] * 100  # convert hPa to Pa
        # create interpolation function
        f_ozone = interp1d(ozone_sonde["Press"], ozone_sonde["o3_vmr"], fill_value="extrapolate", bounds_error=False)
        ozone_interp = f_ozone(data_ml.pressure_full.isel(lat=0, lon=0, time=0))
        # copy interpolated ozone concentration over time, lat and longitude dimension
        o3_t = np.concatenate([ozone_interp[..., None]] * data_ml.dims["time"], axis=1)
        o3_lat = np.concatenate([o3_t[..., None]] * data_ml.dims["lat"], axis=2)
        o3_lon = np.concatenate([o3_lat[..., None]] * data_ml.dims["lon"], axis=3)
        # create DataArray
        o3_vmr = xr.DataArray(o3_lon, coords=data_ml.pressure_full.coords, dims=["level", "time", "lat", "lon"])
        o3_vmr = o3_vmr.transpose("time", ...)
        o3_vmr = o3_vmr.where(o3_vmr > 0, 0)  # set negative values to 0
        o3_vmr = o3_vmr.where(~np.isnan(o3_vmr), 0)  # set nan values to 0
        o3_vmr.attrs = dict(unit="1", long_name="Ozone volume mass mixing ratio")
        data_ml["o3_vmr"] = o3_vmr

    else:
        assert ozone_flag == "default", f"ozone flag set to unrecognized value {ozone_flag}! Check input!"
        data_ml["o3_mmr"] = xr.DataArray(np.repeat([1.587701e-7], n_levels),
                                         dims=["level"],
                                         attrs=dict(unit="1", long_name="Ozone mass mixing ratio"))
    # constants according to IFS Documentation Part IV Section 2.8.4
    data_ml["n2o_vmr"] = xr.DataArray(0.31e-6, attrs=dict(unit="1", long_name="N2O volume mixing ratio"))
    data_ml["cfc11_vmr"] = xr.DataArray(280e-12, attrs=dict(unit="1", long_name="CFC11 volume mixing ratio"))
    data_ml["cfc12_vmr"] = xr.DataArray(484e-12, attrs=dict(unit="1", long_name="CFC12 volume mixing ratio"))
    # other constants
    data_ml["o2_vmr"] = xr.DataArray(0.209488, attrs=dict(unit="1", long_name="Oxygen volume mixing ratio"))
    # monthly surface mean value from https://gml.noaa.gov/ccgg/trends_ch4/
    data_ml["ch4_vmr"] = xr.DataArray(1900e-9, attrs=dict(unit="1", long_name="CH4 volume mixing ratio"))
    # monthly mean CO2 from the Keeling curve https://keelingcurve.ucsd.edu/
    data_ml["co2_vmr"] = xr.DataArray(416e-6, attrs=dict(unit="1", long_name="CO2 volume mixing ratio"))
    #TODO: get profile from CAMS
    #TODO: Add NO2 profile from CAMS

    # %% add cloud properties
    data_ml["fractional_std"] = xr.DataArray(np.repeat([1.], n_levels),
                                             dims=["level"])  # set to 1 according to ecRad documentation
    # only needed for 3D effects in SPARTACUS of a known cloud scene. For general IFS input use parameterization
    # data_ml["inv_cloud_effective_size"] = xr.DataArray(np.expand_dims(np.repeat([0.0013], n_levels), axis=0),
    #                                                    dims=["column", "level"])
    # sum up cloud ice and cloud snow water content according to IFS documentation Part IV Section 2.8.2 (ii)
    data_ml["q_ice"] = data_ml.ciwc + data_ml.cswc
    # sum up cloud liquid and cloud rain water content
    data_ml["q_liquid"] = data_ml.clwc + data_ml.crwc
    data_ml["sw_albedo"] = sw_albedo

    # %% merge surface and multilevel file
    data_ml = data_ml.merge(data_srf)

    # %% rename variables for ecrad
    data_ml = data_ml.rename({"cc": "cloud_fraction"})

    # write to history attribute
    data_ml.attrs["history"] = data_ml.attrs["history"] + f" {datetime.today().strftime('%c')}: " \
                                                          f"formatted file to serve as input to ecRad (ecrad_read_ifs.py)"
    data_ml.attrs["contact"] = f"johannes.roettenbacher@uni-leipzig.de, hanno.mueller@uni-leipzig.de"

    # %% write output files
    filename = f"{path_ifs_output}/ifs_{ifs_date}_{init_time}_ml_processed.nc"
    data_ml.to_netcdf(filename, format='NETCDF4_CLASSIC')
    log.info(f"Saved {filename}")
    csv_filename = f"{path_ifs_output}/nav_data_ip_{date}.csv"
    nav_data_ip.to_csv(csv_filename)
    log.info(f"Saved {csv_filename}")

    log.info(f"Done with date {date}: {pd.to_timedelta((time.time() - start), unit='second')} (hr:min:sec)")
