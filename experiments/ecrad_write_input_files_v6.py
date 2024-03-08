#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 06-07-2023

Use a processed IFS output file on a O1280 grid and generate one ecRad input file for each time step

**Required User Input:**

All options can be set in the script or given as command line key=value pairs.
The first possible option is the default.

* campaign, (default: 'halo-ac3')
* key, flight key (default: 'RF17')
* t_interp, interpolate IFS data in time? (default: False)
* init_time, initalization time of IFS run (00, 12, yesterday)
* o3_source, which ozone concentration to use? (one of '47r1', 'ifs', 'constant', 'sonde')
* trace_gas_source, which trace gas concentrations to use? (one of '47r1', 'constant')
* aerosol_source, which aersol concentrations to use? (one of '47r1', 'ADS')

**Output:**

* well documented ecRad input file in netCDF format for each time step

"""

if __name__ == "__main__":
    # %% module import
    import pylim.helpers as h
    import pylim.meteorological_formulas as met
    from pylim.ecrad import apply_ice_effective_radius, apply_liquid_effective_radius
    import numpy as np
    from sklearn.neighbors import BallTree
    import xarray as xr
    import os
    import pandas as pd
    import time
    from tqdm import tqdm

    start = time.time()

    # %% read in command line arguments
    args = h.read_command_line_args()
    campaign = args["campaign"] if "campaign" in args else "halo-ac3"
    key = args["key"] if "key" in args else "RF17"
    t_interp = h.strtobool(args["t_interp"]) if "t_interp" in args else False
    init_time = args["init"] if "init" in args else "00"
    o3_source = args["o3_source"] if "o3_source" in args else "47r1"
    trace_gas_source = args["trace_gas_source"] if "trace_gas_source" in args else "47r1"
    aerosol_source = args["aerosol_source"] if "aerosol_source" in args else "47r1"
    filter_low_clouds = h.strtobool(args["filter_low_clouds"]) if "filter_low_clouds" in args else True
    version = "v6" if not filter_low_clouds else "v6.1"

    if campaign == "halo-ac3":
        import pylim.halo_ac3 as meta

        flight = meta.flight_names[key]
        date = flight[9:17]
    else:
        import pylim.cirrus_hl as meta

        flight = key
        date = flight[7:15]

    # setup logging
    log = h.setup_logging('./logs', __file__, key)
    # print options to user
    log.info(f"Options set: \ncampaign: {campaign}\nkey: {key}\nflight: {flight}\ndate: {date}\n"
             f"init time: {init_time}\nt_interp: {t_interp}\nversion: {version}\n"
             f"O3 source: {o3_source}\nTrace gas source: {trace_gas_source}\n"
             f"Aerosol source: {aerosol_source}\n"
             f"Filter low level clouds: {filter_low_clouds}\n"
             )

    # %% set paths
    ifs_path = os.path.join(h.get_path("ifs", campaign=campaign), date)
    ecrad_path = os.path.join(h.get_path("ecrad", campaign=campaign), date, "ecrad_input")
    cams_path = h.get_path("cams", campaign=campaign)
    trace_gas_file = f"trace_gas_mm_climatology_2020_{trace_gas_source}_{date}.nc"
    aerosol_file = f"aerosol_mm_climatology_2020_{aerosol_source}_{date}.nc"

    # create output path
    h.make_dir(ecrad_path)

    # %% read in intermediate files from read_ifs
    if init_time == "yesterday":
        ifs_date = int(date) - 1
        init_time = 12
    else:
        ifs_date = date

    nav_data_ip = pd.read_csv(f"{ifs_path}/nav_data_ip_{date}.csv", index_col="time", parse_dates=True)
    data_ml = xr.open_dataset(f"{ifs_path}/ifs_{ifs_date}_{init_time}_ml_O1280_processed.nc")
    data_ml = data_ml.set_index(rgrid=["lat", "lon"])

    # %% read in trace gas and aerosol data
    trace_gas = xr.open_dataset(f"{cams_path}/{trace_gas_file}")
    aerosol = xr.open_dataset(f"{cams_path}/{aerosol_file}")

    # %% find closest grid points along flight track
    ifs_lat_lon = np.column_stack((data_ml.lat, data_ml.lon))
    ifs_tree = BallTree(np.deg2rad(ifs_lat_lon), metric="haversine")
    # generate an array with lat, lon values from the flight position
    points = np.deg2rad(np.column_stack((nav_data_ip.lat.to_numpy(), nav_data_ip.lon.to_numpy())))
    dist, idxs = ifs_tree.query(points, k=10)  # query the tree
    closest_latlons = ifs_lat_lon[idxs]
    # a sphere with radius 1 is assumed so multiplying by Earth's radius gives the distance in km
    distances = dist * 6371

    # %% filter low clouds according to ECMWF low cloud criterion (pressure higher than 0.8 * surface pressure)
    if filter_low_clouds:
        cloud_data = data_ml[["q_liquid", "q_ice", "cloud_fraction", "clwc", "ciwc", "crwc", "cswc"]]
        pressure_filter = data_ml.pressure_full.sel(level=137) * 0.8
        low_cloud_filter = data_ml.pressure_full < pressure_filter  # False for low clouds
        cloud_data = cloud_data.where(low_cloud_filter, 0)  # replace where False with 0
        data_ml.update(cloud_data)

    # %% calculate sw_albedo_direct to account for direct reflection of solar incoming radiation above ocean and sea ice
    ci_albedo_direct = met.calculate_direct_sea_ice_albedo_ebert(nav_data_ip.cos_sza.to_xarray())
    # create xr.DataArray with open ocean albedo after Taylor et al. 1996 for all spectral bands
    open_ocean_albedo_taylor = (nav_data_ip.open_ocean_albedo_taylor
                                .to_xarray()
                                .expand_dims(sw_albedo_band=len(ci_albedo_direct))
                                )

    # %% loop through time steps and write one file per time step
    idx = len(nav_data_ip)
    dt_nav_data = nav_data_ip.index.to_pydatetime()
    for i in tqdm(range(0, idx)):
        # select the 10 nearest grid points around closest grid point
        latlon_sel = [(x, y) for x, y in closest_latlons[i]]
        ds = data_ml.sel(rgrid=latlon_sel)
        dt_time = dt_nav_data[i]

        if t_interp:
            ds = ds.interp(time=dt_time)  # interpolate to time step
            ending = "_inp"
        else:
            ds = ds.sel(time=dt_time, method="nearest")  # select closest time step
            ending = ""

        n_rgrid = len(ds.rgrid)
        cos_sza = np.full(n_rgrid, fill_value=nav_data_ip.cos_sza[i])
        ds["cos_solar_zenith_angle"] = xr.DataArray(cos_sza,
                                                    dims=["rgrid"],
                                                    attrs=dict(unit="1",
                                                               long_name="Cosine of the solar zenith angle"))

        # add sw_albedo_direct
        sw_albedo_direct = (ds.ci * ci_albedo_direct.isel(time=i)
                            + (1. - ds.ci) * open_ocean_albedo_taylor.isel(time=i))
        sw_albedo_direct.attrs = dict(unit=1, long_name="Banded direct short wave albedo")
        ds["sw_albedo_direct"] = sw_albedo_direct

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
        # overwrite ozone according to o3_source
        if o3_source != "47r1":
            ds["o3_vmr"] = ds[f"o3_vmr_{o3_source}"]

        # interpolate aerosol dataset to ifs full pressure levels
        # and turn it into a data array with one new dimension: aer_type
        aerosol_mmr = (aerosol
                       .isel(time=i)
                       .assign(level=(aerosol
                                      .isel(time=i)["full_level_pressure"]
                                      .to_numpy()))
                       .interp(level=new_pressure,
                               kwargs={"fill_value": 0})
                       .drop_vars(["half_level_pressure", "full_level_pressure",
                                   "half_level_delta_pressure"])
                       .to_array(dim="aer_type")
                       .assign_coords(aer_type=np.arange(1, 12),
                                      level=ds.level)
                       .reset_coords("time", drop=True))
        ds["aerosol_mmr"] = aerosol_mmr

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
        variables = ["fractional_std", "aerosol_mmr"] + tg_vars
        for var in variables:
            ds[var] = ds[var].expand_dims(dim={"column": np.arange(n_rgrid)})
        # add distance to aircraft location for each point
        ds["distance"] = xr.DataArray(distances[i, :], dims="column",
                                      attrs=dict(long_name="distance", units="km",
                                                 description="Haversine distance to aircraft location"))
        ds = ds.transpose("column", ...)  # move column to the first dimension
        ds = ds.astype(np.float32)  # change type from double to float32

        ds.to_netcdf(
            path=f"{ecrad_path}/ecrad_input_standard_{nav_data_ip.seconds[i]:7.1f}_sod{ending}_{version}.nc",
            format='NETCDF4_CLASSIC')

    log.info(f"Done with date {date}: {pd.to_timedelta((time.time() - start), unit='second')} (hr:min:sec)")
