#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 05-04-2023

Instead of summing up Cloud Ice Water Content (ciwc) and Cloud Snow Water Content (cswc) for the ice mass mixing ratio (|q-ice|), use only ciwc as |q-ice|.

**Required User Input:**

All options can be set in the script or given as command line key=value pairs.
The first possible option is the default.

* key (RF17), flight key
* t_interp (False), interpolate time or use the closest time step
* init_time (00, 12, yesterday), initialization time of the IFS model run

**Output:**

* well documented ecRad input file in netCDF format for each time step with q_ice = ciwc

"""

if __name__ == "__main__":
    # %% module import
    import pylim.helpers as h
    import pylim.solar_position as sp
    from pylim.ecrad import apply_liquid_effective_radius, apply_ice_effective_radius
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
    version = "v4"
    args = h.read_command_line_args()
    key = args["key"] if "key" in args else "RF17"
    # set interpolate flag
    t_interp = h.strtobool(args["t_interp"]) if "t_interp" in args else False  # interpolate between timesteps?
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
    log = h.setup_logging('./logs', __file__, key)
    # print options to user
    log.info(f"Options set: \ncampaign: {campaign}\nflight: {flight}\ndate: {date}"
             f"\ninit time: {init_time}\nt_interp: {t_interp}")

    # %% set paths
    path_ifs_output = os.path.join(h.get_path("ifs", campaign=campaign), date)
    path_ecrad = os.path.join(h.get_path("ecrad", campaign=campaign), date, "ecrad_input")
    # create output path
    h.make_dir(path_ecrad)

    # %% read in file from read_ifs
    if init_time == "yesterday":
        ifs_date = int(date) - 1
        init_time = 12
    else:
        ifs_date = date

    nav_data_ip = pd.read_csv(f"{path_ifs_output}/nav_data_ip_{date}.csv", index_col="time", parse_dates=True)
    data_ml = xr.open_dataset(f"{path_ifs_output}/ifs_{ifs_date}_{init_time}_ml_processed.nc")
    # set cswc to 0 as ciwc and cswc are added up in the calculation of the effective radius
    data_ml["cswc"] = xr.zeros_like(data_ml["cswc"])
    # overwrite q_ice with ciwc
    data_ml["q_ice"] = data_ml["ciwc"].copy()

    # %% loop through time steps and write one file per time step
    idx = len(nav_data_ip)
    dt_nav_data = nav_data_ip.index.to_pydatetime()

    for i in tqdm(range(0, idx), desc="Time loop"):
        # select a 3 x 11 lat/lon grid around closest grid point
        lat_id = h.arg_nearest(data_ml.lat, nav_data_ip.lat.iat[i])
        lon_id = h.arg_nearest(data_ml.lon, nav_data_ip.lon.iat[i])
        lat_circle = np.arange(lat_id - 1, lat_id + 2)
        lon_circle = np.arange(lon_id - 5, lon_id + 6)
        # make sure latitude indices are available in data
        lat_circle = lat_circle[np.where(lat_circle < data_ml.sizes["lat"])[0]]
        ds_sel = data_ml.isel(lat=lat_circle, lon=lon_circle)

        if t_interp:
            dsi_ml_out = ds_sel.interp(time=dt_nav_data[i])  # interpolate to time step
            ending = "_inp"
        else:
            dsi_ml_out = ds_sel.sel(time=dt_nav_data[i], method="nearest")  # select closest time step
            ending = ""

        cos_sza = np.full((len(lat_circle), len(lon_circle)), fill_value=nav_data_ip.cos_sza.iloc[i])
        dsi_ml_out["cos_solar_zenith_angle"] = xr.DataArray(cos_sza,
                                                            dims=["lat", "lon"],
                                                            attrs=dict(unit="1",
                                                                       long_name="Cosine of the solar zenith angle"))

        # calculate effective radius for all levels
        dsi_ml_out = apply_ice_effective_radius(dsi_ml_out)
        dsi_ml_out = apply_liquid_effective_radius(dsi_ml_out)
        # stack lat, lon to a multi index named column, reset the index,
        # turn lat, lon, time into variables for cleaner output and to avoid later problems when merging data
        # this turns the two dimensions lat lon into one new dimension column with which ecrad can work
        dsi_ml_out = dsi_ml_out.stack(column=("lat", "lon")).reset_index("column").reset_coords(["lat", "lon", "time"])
        # overwrite the MultiIndex object with simple integers as column numbers
        # otherwise it can not be saved to a netCDF file
        n_column = dsi_ml_out.dims["column"]  # get number of columns
        dsi_ml_out["column"] = np.arange(n_column)
        # some variables now need to have the dimension column as well
        variables = ["fractional_std"]
        for var in variables:
            arr = dsi_ml_out[var].values
            dsi_ml_out[var] = dsi_ml_out[var].expand_dims(dim={"column": n_column})
        dsi_ml_out = dsi_ml_out.transpose("column", ...)  # move column to the first dimension
        dsi_ml_out = dsi_ml_out.astype(np.float32)  # change type from double to float32

        dsi_ml_out.to_netcdf(
            path=f"{path_ecrad}/ecrad_input_standard_{nav_data_ip.seconds.iloc[i]:7.1f}_sod{ending}_{version}.nc",
            format='NETCDF4_CLASSIC')

    log.info(f"Done with date {date}: {pd.to_timedelta((time.time() - start), unit='second')} (hr:min:sec)")
