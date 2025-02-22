#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 05-09-2023

Scale the BACARDI measured broadband albedo for the below cloud section with the sw_albedo calculated according to :cite:t:`ebert1993`.

**Required User Input:**

All options can be set in the script or given as command line key=value pairs.
The first possible option is the default.

* key (RF17), flight key
* t_interp (False), interpolate time or use the closest time step
* init_time (00, 12, yesterday), initialization time of the IFS model run

**Output:**

* well documented ecRad input file in netCDF format for each time step below cloud with sw_albedo according to the BACARDI measurement

"""

if __name__ == "__main__":
    # %% module import
    import pylim.helpers as h
    from pylim.ecrad import apply_liquid_effective_radius, apply_ice_effective_radius
    import ac3airborne
    from ac3airborne.tools import flightphase
    from sklearn.neighbors import BallTree
    import numpy as np
    import xarray as xr
    import os
    import pandas as pd
    from datetime import datetime
    from tqdm import tqdm
    import time

    start = time.time()
    # %% read in command line arguments
    campaign = "halo-ac3"
    version = "v5.2"
    args = h.read_command_line_args()
    key = args["key"] if "key" in args else "RF17"
    # set interpolate flag
    t_interp = h.strtobool(args["t_interp"]) if "t_interp" in args else False  # interpolate between timesteps?
    init_time = args["init"] if "init" in args else "00"
    trace_gas_source = args["trace_gas_source"] if "trace_gas_source" in args else "47r1"

    if campaign == "halo-ac3":
        import pylim.halo_ac3 as meta

        flight = meta.flight_names[key]
        date = flight[9:17]
    else:
        import pylim.cirrus_hl as meta

        flight = key
        date = flight[7:15]
    dt_day = datetime.strptime(date, '%Y%m%d')  # convert date to date time for further use
    # setup logging
    __file__ = None if '__file__' not in locals() else __file__
    log = h.setup_logging('./logs', __file__, key)
    # print options to user
    log.info(f"Options set: \ncampaign: {campaign}\nflight: {flight}\ndate: {date}"
             f"\ninit time: {init_time}\nt_interp: {t_interp}\nversion: {version}\n"
             f"Trace gas source: {trace_gas_source}\n")

    # %% set paths
    path_ifs_output = os.path.join(h.get_path("ifs", campaign=campaign), date)
    path_ecrad = os.path.join(h.get_path("ecrad", campaign=campaign), date, "ecrad_input")
    cams_path = h.get_path("cams", campaign=campaign)
    trace_gas_file = f"trace_gas_mm_climatology_2020_{trace_gas_source}_{date}.nc"
    bacardi_path = h.get_path("bacardi", flight, campaign)
    bacardi_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}.nc"
    # create output path
    h.make_dir(path_ecrad)

    # %% read in trace gas data
    trace_gas = xr.open_dataset(f"{cams_path}/{trace_gas_file}")

    # %% read in file from read_ifs
    if init_time == "yesterday":
        ifs_date = int(date) - 1
        init_time = 12
    else:
        ifs_date = date

    nav_data_ip = pd.read_csv(f"{path_ifs_output}/nav_data_ip_{date}.csv", index_col="time", parse_dates=True)
    data_ml = xr.open_dataset(f"{path_ifs_output}/ifs_{ifs_date}_{init_time}_ml_O1280_processed.nc")
    data_ml = data_ml.set_index(rgrid=["lat", "lon"])

    # %% read in BACARDI data and calculate diffuse short wave albedo
    bacardi_ds = xr.open_dataset(f"{bacardi_path}/{bacardi_file}")
    bacardi_ds["sw_albedo"] = bacardi_ds["F_up_solar"] / bacardi_ds["F_down_solar_diff"]

    # %% get flight segmentation and retrieve below and above cloud section
    fl_segments = ac3airborne.get_flight_segments()["HALO-AC3"]["HALO"][f"HALO-AC3_HALO_{key}"]
    segments = flightphase.FlightPhaseFile(fl_segments)
    above_cloud, below_cloud = dict(), dict()
    if key == "RF17":
        above_cloud["start"] = segments.select("name", "high level 7")[0]["start"]
        above_cloud["end"] = segments.select("name", "high level 8")[0]["end"]
        below_cloud["start"] = segments.select("name", "high level 9")[0]["start"]
        below_cloud["end"] = segments.select("name", "high level 10")[0]["end"]
        above_slice = slice(above_cloud["start"], above_cloud["end"])
        below_slice = slice(below_cloud["start"], below_cloud["end"])
        case_slice = slice(pd.to_datetime("2022-04-11 10:30"), pd.to_datetime("2022-04-11 12:29"))
    else:
        above_cloud["start"] = segments.select("name", "polygon pattern 1")[0]["start"]
        above_cloud["end"] = segments.select("name", "polygon pattern 1")[0]["parts"][-1]["start"]
        below_cloud["start"] = segments.select("name", "polygon pattern 2")[0]["start"]
        below_cloud["end"] = segments.select("name", "polygon pattern 2")[0]["end"]
        above_slice = slice(above_cloud["start"], above_cloud["end"])
        below_slice = slice(below_cloud["start"], below_cloud["end"])
        case_slice = slice(above_cloud["start"], below_cloud["end"])

    # %% select below cloud time slice
    nav_data_ip = nav_data_ip[below_cloud["start"]:below_cloud["end"]]
    # resample below cloud measurements to 1 minute resolution
    bacardi_below = bacardi_ds.sel(time=below_slice).resample(time="1Min").mean()

    # %% select lat and lon closest to flightpath
    idx = len(nav_data_ip)
    dt_nav_data = nav_data_ip.index.to_pydatetime()
    ifs_lat_lon = np.column_stack((data_ml.lat, data_ml.lon))
    ifs_tree = BallTree(np.deg2rad(ifs_lat_lon), metric="haversine")
    # generate an array with lat, lon values from the flight position
    points = np.deg2rad(np.column_stack((nav_data_ip.lat.to_numpy(), nav_data_ip.lon.to_numpy())))
    dist, idxs = ifs_tree.query(points, k=10)  # query the tree
    closest_latlons = ifs_lat_lon[idxs]
    # a sphere with radius 1 is assumed so multiplying by Earth's radius gives the distance in km
    distances = dist * 6371

    # %% filter low clouds according to ECMWF low cloud criterion (pressure higher than 0.8 * surface pressure)
    cloud_data = data_ml[["q_liquid", "q_ice", "cloud_fraction", "clwc", "ciwc", "crwc", "cswc"]]
    pressure_filter = data_ml.pressure_full.sel(level=137) * 0.8
    low_cloud_filter = data_ml.pressure_full < pressure_filter  # False for low clouds
    cloud_data = cloud_data.where(low_cloud_filter, 0)  # replace where False with 0
    data_ml.update(cloud_data)

    # %% loop through time steps and write one file per time step
    for i in tqdm(range(idx), desc="Time loop"):
        # select the 10 nearest grid points around closest grid point
        latlon_sel = [(x, y) for x, y in closest_latlons[i]]
        ds = data_ml.sel(rgrid=latlon_sel)
        dt_time = dt_nav_data[i]
        sw_albedo = bacardi_below.sw_albedo.sel(time=dt_time)

        if t_interp:
            ds = ds.interp(time=dt_time)  # interpolate to time step
            ending = "_inp"
        else:
            ds = ds.sel(time=dt_time, method="nearest")  # select closest time step
            ending = ""

        # get relative difference from all sw albedo bands to the first band in the IFS to scale the constant albedo from BACARDI
        sw_albedo = xr.full_like(ds["sw_albedo"], sw_albedo)
        diffs = (ds["sw_albedo"][0, ...] - ds["sw_albedo"][1:, ...]) / ds["sw_albedo"][0, :]
        sw_albedo[1:6, ...] = sw_albedo[1:6, ...] - diffs * sw_albedo[0, ...]
        ds["sw_albedo"] = sw_albedo

        # add cos solar zenith angle to the data set
        n_rgrid = len(ds.rgrid)
        cos_sza = np.full(n_rgrid, fill_value=nav_data_ip.cos_sza.iloc[i])
        ds["cos_solar_zenith_angle"] = xr.DataArray(cos_sza,
                                                    dims=["rgrid"],
                                                    attrs=dict(unit="1",
                                                               long_name="Cosine of the solar zenith angle"))

        # interpolate trace gas data onto ifs full pressure levels
        new_pressure = ds.pressure_full.isel(rgrid=0).to_numpy()
        tg = (trace_gas
              .isel(time=i)
              .interp(level=new_pressure,
                      kwargs={"fill_value": 0}))

        # read out trace gases from trace gas file
        tg_vars = ["cfc11_vmr", "cfc12_vmr", "ch4_vmr", "co2_vmr", "n2o_vmr", "o3_vmr"]
        for var in tg_vars:
            ds[var] = tg[var].assign_coords(level=ds.level)

        # overwrite the trace gases with the variables corresponding to trace_gas_source
        if trace_gas_source == "constant":
            for var in tg_vars:
                ds[var] = ds[f"{var}_{trace_gas_source}"]

        # calculate effective radius for all levels
        ds = apply_ice_effective_radius(ds)
        ds = apply_liquid_effective_radius(ds)
        # reset the MultiIndex
        ds = ds.reset_index(["rgrid", "lat", "lon"])
        # overwrite the MultiIndex object with simple integers as column numbers
        # otherwise it can not be saved to a netCDF file
        ds["rgrid"] = np.arange(n_rgrid)
        # turn lat, lon, time into variables for cleaner output and to avoid later problems when merging data
        ds = ds.reset_coords(["lat", "lon", "time"]).drop_dims("reduced_points")

        for var in ["lat", "lon"]:
            ds[var] = xr.DataArray(ds[var].to_numpy(), dims="rgrid")
        ds = ds.rename(rgrid="column")  # rename rgrid to column for ecrad
        # some variables now need to have the dimension column as well
        variables = ["fractional_std"] + tg_vars
        for var in variables:
            ds[var] = ds[var].expand_dims(dim={"column": np.arange(n_rgrid)})
        # add distance to aircraft location for each point
        ds["distance"] = xr.DataArray(distances[i, :], dims="column",
                                      attrs=dict(long_name="distance", units="km",
                                                 description="Haversine distance to aircraft location"))
        ds = ds.transpose("column", ...)  # move column to the first dimension
        ds = ds.astype(np.float32)  # change type from double to float32

        ds.to_netcdf(
            path=f"{path_ecrad}/ecrad_input_standard_{nav_data_ip.seconds.iloc[i]:7.1f}_sod{ending}_{version}.nc",
            format='NETCDF4_CLASSIC')

    log.info(f"Done with date {date}: {pd.to_timedelta((time.time() - start), unit='second')} (hr:min:sec)")
