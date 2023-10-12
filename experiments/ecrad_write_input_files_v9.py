#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 10-10-2023

Use a processed IFS output file and a processed CAMS file and generate one ecRad input file for each time step along the flight path of HALO.
Include aerosols and trace gases from a monthly mean CAMS climatology.

**Required User Input:**

All options can be set in the script or given as command line key=value pairs.
The first possible option is the default.

* version (v9), modifies the version number at the end of the input filename
* campaign (halo-ac3, cirrus-hl)
* key (RF17), flight key
* t_interp (False), interpolate time or use the closest time step
* init_time (00, 12, yesterday), initialization time of the IFS model run

**Output:**

* well documented ecRad input file in netCDF format for each time step CAMS monthly mean climatology of aerosols and trace gases

"""
if __name__ == "__main__":
    # %% module import
    import pylim.helpers as h
    from pylim.ecrad import apply_ice_effective_radius, apply_liquid_effective_radius
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
    version = "v9"
    args = h.read_command_line_args()
    campaign = args["campaign"] if "campaign" in args else "halo-ac3"
    key = args["key"] if "key" in args else "RF17"
    # set interpolate flag
    t_interp = strtobool(args["t_interp"]) if "t_interp" in args else False  # interpolate between timesteps?
    init_time = args["init"] if "init" in args else "00"
    cams_source = "ADS"

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
             f"\ninit time: {init_time}\nt_interp: {t_interp}\nversion: {version}\n"
             f"CAMS source: {cams_source}\n")

    # %% set paths
    path_ifs_output = os.path.join(h.get_path("ifs", campaign=campaign), date)
    path_cams = h.get_path("cams", campaign=campaign)
    cams_file = f"cams_mm_climatology_2020_{cams_source}_{date}.nc"
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

    nav_data_ip = pd.read_csv(f"{path_ifs_output}/nav_data_ip_{date}.csv", index_col="time", parse_dates=True)
    data_ml = xr.open_dataset(f"{path_ifs_output}/ifs_{ifs_date}_{init_time}_ml_O1280_processed.nc")
    data_ml = data_ml.set_index(rgrid=["lat", "lon"])
    cams_ml = xr.open_dataset(f"{path_cams}/{cams_file}")
    cams_ml["level"] = cams_ml.level * 100  # convert to Pa

    # %% select ifs grid cell closest to flightpath
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

    # %% loop through time steps and write one file per time step
    for i in tqdm(range(idx), desc="Time loop"):
        # select the 10 nearest grid points around closest grid point
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

        # interpolate cams data onto ifs full pressure levels
        new_pressure = dsi_ml_out.pressure_full.isel(rgrid=0).to_numpy()
        cams = (cams_ml
                .isel(time=i)
                .interp(level=new_pressure,
                        kwargs={"fill_value": 0}))
        # read out trace gases from cams
        ecrad_vars = ["co2_mmr", "ch4_mmr", "o3_mmr"]
        for cams_var, ecrad_var in zip(["co2", "ch4", "go3"], ecrad_vars):
            dsi_ml_out[ecrad_var] = cams[cams_var].assign_coords(level=dsi_ml_out.level)
            cams = cams.drop_vars(cams_var).copy()

        # turn cams dataset into a data array with one new dimension: aer_type
        aerosol_mmr = (cams
                       .to_array(dim="aer_type")
                       .sortby("aer_type")
                       .assign_coords(aer_type=np.arange(1, 12),
                                      level=dsi_ml_out.level)
                       .reset_coords("time", drop=True))
        dsi_ml_out["aerosol_mmr"] = aerosol_mmr

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

        for var in ["fractional_std", "aerosol_mmr"] + ecrad_vars:
            dsi_ml_out[var] = dsi_ml_out[var].expand_dims(dim={"column": np.arange(n_rgrid)})
        # add distance to aircraft location for each point
        dsi_ml_out["distance"] = xr.DataArray(distances[i, :], dims="column",
                                              attrs=dict(long_name="distance", units="km",
                                                         description="Haversine distance to aircraft location"))
        dsi_ml_out = dsi_ml_out.transpose("column", ...)  # move column to the first dimension
        dsi_ml_out = dsi_ml_out.astype(np.float32)  # change type from double to float32
        # drop unneccesary variables
        vars_to_drop = ["ch4_vmr", "co2_vmr", "o3", "o3_vmr"]
        dsi_ml_out = dsi_ml_out.drop_vars(vars_to_drop)

        dsi_ml_out.to_netcdf(
            path=f"{path_ecrad}/ecrad_input_standard_{nav_data_ip.seconds[i]:7.1f}_sod{ending}_{version}.nc",
            format='NETCDF4_CLASSIC')

    log.info(f"Done with date {date}: {pd.to_timedelta((time.time() - start), unit='second')} (hr:min:sec)")
