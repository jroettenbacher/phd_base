#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 06-07-2023

Use a processed IFS output file on a O1280 grid and generate one ecRad input file for each time step

**Required User Input:**

* step: at which intervals should the IFS data be interpolated on the aircraft data (default: 1Min from :ref:`processing:ecrad_read_ifs.py`)

**Output:**

* well documented ecRad input file in netCDF format for each time step

"""

if __name__ == "__main__":
    # %% module import
    import pylim.helpers as h
    from pylim.ecrad import apply_ice_effective_radius, apply_liquid_effective_radius
    import numpy as np
    from sklearn.neighbors import BallTree
    import xarray as xr
    import os
    import pandas as pd
    import time
    from distutils.util import strtobool
    from tqdm import tqdm

    start = time.time()

    # %% read in command line arguments
    campaign = "halo-ac3"
    version = "v6"
    args = h.read_command_line_args()
    key = args["key"] if "key" in args else "RF17"
    # set interpolate flag
    t_interp = strtobool(args["t_interp"]) if "t_interp" in args else False  # interpolate between timesteps?
    init_time = args["init"] if "init" in args else "00"
    if campaign == "halo-ac3":
        import pylim.halo_ac3 as meta

        flight = meta.flight_names[key]
        date = flight[9:17]
    else:
        import pylim.cirrus_hl as meta

        flight = key
        date = flight[7:15]

    # setup logging
    try:
        file = __file__
    except NameError:
        file = None
    log = h.setup_logging("./logs", file, key)
    # print options to user
    log.info(f"Options set: \ncampaign: {campaign}\nflight: {flight}\ndate: {date}"
             f"\ninit time: {init_time}\nt_interp: {t_interp}")

    # %% set paths
    path_ifs_output = os.path.join(h.get_path("ifs", campaign=campaign), date)
    path_ecrad = os.path.join(h.get_path("ecrad", campaign=campaign), date, "ecrad_input")
    # create output path
    h.make_dir(path_ecrad)

    # %% read in intermediate files from read_ifs
    if init_time == "yesterday":
        ifs_date = int(date) - 1
        init_time = 12
    else:
        ifs_date = date

    nav_data_ip = pd.read_csv(f"{path_ifs_output}/nav_data_ip_{date}.csv", index_col="time", parse_dates=True)
    data_ml = xr.open_dataset(f"{path_ifs_output}/ifs_{ifs_date}_{init_time}_ml_O1280_processed.nc")
    data_ml = data_ml.set_index(rgrid=["lat", "lon"])

    # %% loop through time steps and write one file per time step
    idx = len(nav_data_ip)
    dt_nav_data = nav_data_ip.index.to_pydatetime()
    ifs_lat_lon = np.column_stack((data_ml.lat, data_ml.lon))
    ifs_tree = BallTree(np.deg2rad(ifs_lat_lon), metric="haversine")
    # generate an array with lat, lon values from the flight position
    points = np.deg2rad(np.column_stack((nav_data_ip.lat.to_numpy(), nav_data_ip.lon.to_numpy())))
    dist, idxs = ifs_tree.query(points, k=33)  # query the tree
    closest_latlons = ifs_lat_lon[idxs]
    # a sphere with radius 1 is assumed so multiplying by Earth's radius gives the distance in km
    distances = dist * 6371

    for i in tqdm(range(0, idx)):
        # select the 33 nearest grid points around closest grid point
        latlon_sel = [(x, y) for x, y in closest_latlons[i]]
        ds_sel = data_ml.sel(rgrid=latlon_sel)
        dt_time = dt_nav_data[i]

        if t_interp:
            dsi_ml_out = ds_sel.interp(time=dt_time)  # interpolate to time step
            ending = "_inp"
        else:
            dsi_ml_out = ds_sel.sel(time=dt_time, method="nearest")  # select closest time step
            ending = ""

        n_rgrid = len(ds_sel.rgrid)
        cos_sza = np.full(n_rgrid, fill_value=nav_data_ip.cos_sza[i])
        dsi_ml_out["cos_solar_zenith_angle"] = xr.DataArray(cos_sza,
                                                            dims=["rgrid"],
                                                            attrs=dict(unit="1",
                                                                       long_name="Cosine of the solar zenith angle"))

        # calculate effective radius for all levels
        dsi_ml_out = apply_ice_effective_radius(dsi_ml_out)
        dsi_ml_out = apply_liquid_effective_radius(dsi_ml_out)
        # reset the MultiIndex
        dsi_ml_out = dsi_ml_out.reset_index(["rgrid", "lat", "lon"])
        # overwrite the MultiIndex object with simple integers as column numbers
        # otherwise it can not be saved to a netCDF file
        dsi_ml_out["rgrid"] = np.arange(n_rgrid)
        # turn lat, lon, time into variables for cleaner output and to avoid later problems when merging data
        dsi_ml_out = dsi_ml_out.reset_coords(["lat", "lon", "time"]).drop_dims("reduced_points")

        for var in ["lat", "lon"]:
            dsi_ml_out[var] = xr.DataArray(dsi_ml_out[var].to_numpy(), dims="rgrid")
        dsi_ml_out = dsi_ml_out.rename(rgrid="column")  # rename rgrid to column for ecrad
        # some variables now need to have the dimension column as well
        variables = ["fractional_std"]
        for var in variables:
            dsi_ml_out[var] = dsi_ml_out[var].expand_dims(dim={"column": np.arange(n_rgrid)})
        # add distance to aircraft location for each point
        dsi_ml_out["distance"] = xr.DataArray(distances[i, :], dims="column",
                                              attrs=dict(long_name="distance", units="km",
                                                         description="Haversine distance to aircraft location"))
        dsi_ml_out = dsi_ml_out.transpose("column", ...)  # move column to the first dimension
        dsi_ml_out = dsi_ml_out.astype(np.float32)  # change type from double to float32

        dsi_ml_out.to_netcdf(
            path=f"{path_ecrad}/ecrad_input_standard_{nav_data_ip.seconds[i]:7.1f}_sod{ending}_v6.nc",
            format='NETCDF4_CLASSIC')

    log.info(f"Done with date {date}: {pd.to_timedelta((time.time() - start), unit='second')} (hr:min:sec)")
