#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 10-07-2023

Use a processed IFS output file on a O1280 grid and generate one ecRad input file for each time step.
Use the VarCloud retrieval for the below cloud section.

**Required User Input:**

* key (flight key, eg. RF17)
* init_time (init time of IFS data, eg. 00)

**Output:**

* well documented ecRad input file in netCDF format for each time step

"""

if __name__ == "__main__":
    # %% module import
    import pylim.helpers as h
    import pylim.solar_position as sp
    from pylim.ecrad import apply_liquid_effective_radius
    from pylim import reader
    from metpy.calc import density
    from metpy.units import units
    import numpy as np
    import scipy.spatial as ssp
    import xarray as xr
    import os
    import pandas as pd
    import time
    from tqdm import tqdm
    # from joblib import cpu_count, delayed, Parallel

    start = time.time()

    # %% read in command line arguments
    campaign = "halo-ac3"
    version = "v7"
    args = h.read_command_line_args()
    key = args["key"] if "key" in args else "RF17"
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
             f"\ninit time: {init_time}\nversion: {version}\n")

    # %% set paths
    path_ifs_output = os.path.join(h.get_path("ifs", campaign=campaign), date)
    path_ecrad = os.path.join(h.get_path("ecrad", campaign=campaign), date, "ecrad_input")
    path_varcloud = h.get_path("varcloud", flight, campaign)
    file_varcloud = [f for f in os.listdir(path_varcloud) if f.endswith(".nc")][0]
    path_bahamas = h.get_path("bahamas", flight, campaign)
    file_bahamas = f"HALO-AC3_HALO_BAHAMAS_{date}_{key}_v1.nc"
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
    bahamas_ds = reader.read_bahamas(f"{path_bahamas}/{file_bahamas}")

    # %% select only case study time which features the cloud that HALO also underpassed
    sel_time = slice(pd.to_datetime("2022-04-11 10:49"), pd.to_datetime("2022-04-11 11:04"))
    sim_time = pd.date_range("2022-04-11 11:35", "2022-04-11 11:50", freq="1s")

    # %% resample varcloud data to secondly resolution
    new_index = pd.date_range(str(varcloud_ds.time[0].astype('datetime64[s]').to_numpy()),
                              str(varcloud_ds.time[-1].astype('datetime64[s]').to_numpy()), freq="1s")
    varcloud_ds = varcloud_ds.reindex(time=new_index, method="bfill")
    varcloud_ds = varcloud_ds.sel(time=sel_time)
    # reverse varcloud data as the last retrieval point is the first point that HALO underpasses on its way back
    varcloud_ds = varcloud_ds.sortby("time", ascending=False)

    # %% select bahamas data for simulation
    bahamas_ds = bahamas_ds.sel(time=sim_time)

# %% loop through time steps and write one file per time step
    lats, lons = bahamas_ds.IRS_LAT.sel(time=sim_time), bahamas_ds.IRS_LON.sel(time=sim_time)
    idx = len(varcloud_ds.time)
    ifs_lat_lon = np.column_stack((data_ml.lat, data_ml.lon))
    ifs_tree = ssp.KDTree(ifs_lat_lon)  # build the kd tree for nearest neighbour look up
    # generate an array with lat, lon values from the flight position
    points = np.column_stack((lats.to_numpy(), lons.to_numpy()))
    dist, idxs = ifs_tree.query(points, k=1)  # query the tree
    closest_latlons = ifs_tree.data[idxs]


    def write_ecrad_input_file(data_ml, varcloud_ds, bahamas_ds, closest_latlons, sim_time, path_ecrad, i):
        """
        Helper function to be called in parallel to speed up file creation.
        Variables are from outer script.

        Returns: ecRad input files

        """
        # select the nearest grid point to flight path
        latlon_sel = (closest_latlons[i][0], closest_latlons[i][1])
        t = sim_time[i]
        ds_sel = data_ml.sel(rgrid=latlon_sel)
        dsi_ml_out = ds_sel.sel(time=t, method="nearest").reset_coords("time")
        varcloud_sel = varcloud_ds.isel(time=i)
        # interpolate varcloud height to model height
        varcloud_sel = varcloud_sel.interp(Height=dsi_ml_out.press_height_full).reset_coords(["time", "Height"])
        # convert kg/m3 to kg/kg
        air_density = density(dsi_ml_out.pressure_full * units.Pa, dsi_ml_out.t * units.K,
                              dsi_ml_out.q * units("kg/kg"))
        q_ice = varcloud_sel["Varcloud_Cloud_Ice_Water_Content"] * units("kg/m3") / air_density
        # overwrite ice water content
        dsi_ml_out["q_ice"] = q_ice.metpy.dequantify().where(~np.isnan(q_ice), 0)

        # add cos_sza for the grid point using only model data
        sod = t.hour * 3600 + t.minute * 60 + t.second
        p_surf_nearest = dsi_ml_out.pressure_hl.isel(half_level=137).to_numpy() / 100  # hPa
        t_surf_nearest = dsi_ml_out.temperature_hl.isel(half_level=137).to_numpy() - 273.15  # degree Celsius
        ypos = dsi_ml_out.lat.to_numpy()
        xpos = dsi_ml_out.lon.to_numpy()
        sza = sp.get_sza(sod / 3600, ypos, xpos, t.year, t.month, t.day, p_surf_nearest, t_surf_nearest)
        cos_sza = np.cos(sza / 180. * np.pi)

        dsi_ml_out["cos_solar_zenith_angle"] = xr.DataArray(cos_sza,
                                                            attrs=dict(unit="1",
                                                                       long_name="Cosine of the solar zenith angle"))

        # assign effective radius
        re_ice = varcloud_sel["Varcloud_Cloud_Ice_Effective_Radius"]
        dsi_ml_out["re_ice"] = re_ice.where(~np.isnan(re_ice), 51.9616 * 1e-6)  # replace nan with default value
        dsi_ml_out = apply_liquid_effective_radius(dsi_ml_out)
        # reset coordinates, drop unused dimension and unused variable
        dsi_ml_out = dsi_ml_out.reset_coords(["rgrid", "lat", "lon"]).drop_dims("reduced_points").drop_vars("rgrid")
        dsi_ml_out = dsi_ml_out.expand_dims("column", axis=0)
        # remove column dim from dimensionless variables
        for var in ["co2_vmr", "n2o_vmr", "ch4_vmr", "o2_vmr", "cfc11_vmr", "cfc12_vmr", "time"]:
            dsi_ml_out[var] = dsi_ml_out[var].isel(column=0)
        n_column = dsi_ml_out.dims["column"]  # get number of columns
        dsi_ml_out["column"] = np.arange(1, n_column + 1)
        # overwrite lat lon values, somehow this is necessary
        for var in ["lat", "lon"]:
            dsi_ml_out[var] = xr.DataArray(dsi_ml_out[var].to_numpy(), dims="columns")

        dsi_ml_out = dsi_ml_out.transpose("column", ...)  # move column to the first dimension
        dsi_ml_out = dsi_ml_out.astype(np.float32)  # change type from double to float32

        dsi_ml_out.to_netcdf(
            path=f"{path_ecrad}/ecrad_input_standard_{sod:7.1f}_sod_v7.nc",
            format='NETCDF4_CLASSIC')

        return None


    # somehow this throws a PickleError
    # Parallel(n_jobs=cpu_count() - 2)(delayed(write_ecrad_input_file)(data_ml, varcloud_ds, bahamas_ds, closest_latlons,
    #                                                                  sim_time.to_numpy(), path_ecrad, i)
    #                                  for i in tqdm(range(0, idx)))

    for i in tqdm(range(0, idx)):
        write_ecrad_input_file(data_ml, varcloud_ds, bahamas_ds, closest_latlons, sim_time, path_ecrad, i)

    log.info(f"Done with date {date}: {pd.to_timedelta((time.time() - start), unit='second')} (hr:min:sec)")
