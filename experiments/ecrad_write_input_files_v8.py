#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 23-03-2023

Use a processed IFS output file and the Varcloud retrieval from Florian Ewald, LMU and generate one ecRad input file for each time step along the flight path of HALO.
Instead of the *CIWC* and :math:`r_{eff, ice}` from the IFS use the retrieved variables from the Varcloud lidar/radar retrieval.

**Required User Input:**

All options can be set in the script or given as command line key=value pairs.
The first possible option is the default.

* version (v8), modifies the version number at the end of the input filename
* campaign (halo-ac3, cirrus-hl)
* key (RF17), flight key
* t_interp (False), interpolate time or use the closest time step
* init_time (00, 12, yesterday), initialization time of the IFS model run

**Output:**

* well documented ecRad input file in netCDF format for each time step with retrieved *CIWC* and :math:`re_{eff, ice}`

"""
if __name__ == "__main__":
    # %% module import
    import pylim.helpers as h
    import pylim.solar_position as sp
    from pylim.ecrad import apply_liquid_effective_radius
    from metpy.calc import density, mixing_ratio_from_specific_humidity
    from metpy.units import units
    import numpy as np
    import xarray as xr
    from sklearn.neighbors import BallTree
    import os
    import pandas as pd
    from datetime import datetime
    from tqdm import tqdm
    import time
    from distutils.util import strtobool

    start = time.time()
    # %% read in command line arguments
    version = "v8"
    args = h.read_command_line_args()
    campaign = args["campaign"] if "campaign" in args else "halo-ac3"
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

    dt_day = datetime.strptime(date, '%Y%m%d')  # convert date to date time for further use
    # setup logging
    try:
        file = __file__
    except NameError:
        file = None
    log = h.setup_logging("./logs", file, key)
    # print options to user
    log.info(f"Options set: \ncampaign: {campaign}\nflight: {flight}\ndate: {date}"
             f"\ninit time: {init_time}\nt_interp: {t_interp}\nversion: {version}\n")

    # %% set paths
    path_ifs_output = os.path.join(h.get_path("ifs", campaign=campaign), date)
    path_ecrad = os.path.join(h.get_path("ecrad", campaign=campaign), date, "ecrad_input")
    path_varcloud = h.get_path("varcloud", flight, campaign)
    file_varcloud = [f for f in os.listdir(path_varcloud) if f.endswith(".nc")][0]
    # create output path
    h.make_dir(path_ecrad)

    # %% read in intermediate files from read_ifs
    if init_time == "yesterday":
        ifs_date = int(date) - 1
        init_time = 12
    else:
        ifs_date = date

    data_ml = xr.open_dataset(f"{path_ifs_output}/ifs_{ifs_date}_{init_time}_ml_O1280_processed.nc")
    data_ml = data_ml.set_index(rgrid=["lat", "lon"])
    varcloud_ds = xr.open_dataset(f"{path_varcloud}/{file_varcloud}").swap_dims(time="Time", height="Height").rename(
        Time="time")

    # %% resample varcloud data to minutely resolution
    varcloud_ds = varcloud_ds.resample(time="1min").mean()

    # %% select lat and lon closest to flightpath
    lats, lons, times = varcloud_ds.Latitude, varcloud_ds.Longitude, varcloud_ds.time
    idx = len(times)
    ifs_lat_lon = np.column_stack((data_ml.lat, data_ml.lon))
    ifs_tree = BallTree(np.deg2rad(ifs_lat_lon), metric="haversine")  # build the kd tree for nearest neighbour look up
    # generate an array with lat, lon values from the flight position
    points = np.deg2rad(np.column_stack((lats.to_numpy(), lons.to_numpy())))
    dist, idxs = ifs_tree.query(points, k=1)  # query the tree
    closest_latlons = ifs_lat_lon[idxs.flatten()]
    # a sphere with radius 1 is assumed so multiplying by Earth's radius gives the distance in km
    distances = dist.flatten() * 6371

    # %% loop through time steps and write one file per time step
    for i in tqdm(range(idx), desc="Time loop"):
        latlon_sel = (closest_latlons[i][0], closest_latlons[i][1])
        t = times[i]
        ds_sel = data_ml.sel(rgrid=latlon_sel).reset_coords(["rgrid", "lat", "lon"]).drop_vars("rgrid").drop_dims("reduced_points")
        if t_interp:
            dsi_ml_out = ds_sel.interp(time=times[i]).reset_coords("time")  # interpolate to time step
            ending = "_inp"
        else:
            dsi_ml_out = ds_sel.sel(time=t, method="nearest").reset_coords("time")  # select closest time step
            ending = ""

        varcloud_sel = varcloud_ds.sel(time=t)
        # interpolate varcloud height to model height
        varcloud_sel = varcloud_sel.interp(Height=dsi_ml_out.press_height_full).reset_coords(["time", "Height"])
        # convert kg/m3 to kg/kg
        mixing_ratio = mixing_ratio_from_specific_humidity(dsi_ml_out["q"] * units("kg/kg"))
        air_density = density(dsi_ml_out.pressure_full * units.Pa, dsi_ml_out.t * units.K, mixing_ratio)
        q_ice = varcloud_sel["Varcloud_Cloud_Ice_Water_Content"] * units("kg/m3") / air_density
        # overwrite ice water content
        dsi_ml_out["q_ice"] = q_ice.metpy.dequantify().where(~np.isnan(q_ice), 0)

        # add cos_sza for the grid point using model data for the thermodynamics and aircraft data for the location
        sod = (t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second).to_numpy()
        p_surf_nearest = dsi_ml_out.pressure_hl.isel(half_level=137).to_numpy() / 100  # hPa
        t_surf_nearest = dsi_ml_out.temperature_hl.isel(half_level=137).to_numpy() - 273.15  # degree Celsius
        ypos = varcloud_sel.Latitude.to_numpy()
        xpos = varcloud_sel.Longitude.to_numpy()
        sza = sp.get_sza(sod / 3600, ypos, xpos, dt_day.year, dt_day.month, dt_day.day, p_surf_nearest, t_surf_nearest)
        cos_sza = np.cos(sza / 180. * np.pi)

        dsi_ml_out["cos_solar_zenith_angle"] = xr.DataArray(cos_sza,
                                                            attrs=dict(unit="1",
                                                                       long_name="Cosine of the solar zenith angle"))

        # assign effective radius
        re_ice = varcloud_sel["Varcloud_Cloud_Ice_Effective_Radius"]
        dsi_ml_out["re_ice"] = re_ice.where(~np.isnan(re_ice), 51.9616 * 1e-6)  # replace nan with default value
        dsi_ml_out = apply_liquid_effective_radius(dsi_ml_out)
        dsi_ml_out = dsi_ml_out.expand_dims("column", axis=0)
        # remove column dim from dimensionless variables
        for var in ["co2_vmr", "n2o_vmr", "ch4_vmr", "o2_vmr", "cfc11_vmr", "cfc12_vmr", "time"]:
            dsi_ml_out[var] = dsi_ml_out[var].isel(column=0)
        n_column = dsi_ml_out.dims["column"]  # get number of columns
        dsi_ml_out["column"] = np.arange(n_column)

        # add distance to aircraft location
        dsi_ml_out["distance"] = xr.DataArray(np.expand_dims(distances[i], axis=0), dims="column",
                                              attrs=dict(long_name="distance", units="km",
                                                         description="Haversine distance to aircraft location"))

        dsi_ml_out = dsi_ml_out.transpose("column", ...)  # move column to the first dimension
        dsi_ml_out = dsi_ml_out.astype("float32")  # change type from double to float32

        dsi_ml_out.to_netcdf(
            path=f"{path_ecrad}/ecrad_input_standard_{sod:7.1f}_sod{ending}_v8.nc",
            format='NETCDF4_CLASSIC')

    log.info(f"Done with date {date}: {pd.to_timedelta((time.time() - start), unit='second')} (hr:min:sec)")
