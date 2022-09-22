#!/usr/bin/env python
"""Calculate statistics from the ECMWF IFS output along the flight track

Using the BAHAMAS down sampled data (0.5Hz) and a defined circle around the closest IFS grid point to the aircraft
location, statistics from the IFS output are calculated and saved to new netCDF files for use in the case studies.

*Input*:

- IFS data as returned by read_ifs.py
- BAHAMAS data
- ecRad output data

*Output*:

NetCDF files in the IFS input directory with the statistics.

TODO: Add option to vary amount of surrounding grid cells

*author*: Johannes Röttenbacher
"""
# %% functions
import xarray as xr


def calculate_ifs_along_flighttrack(ifs: xr.Dataset, lat_circles: list, lon_circles: list,
                                    aircraft_height_level: list, bahamas_time: xr.DataArray, i: int):
    """
    Calculate the mean and standard deviation of each IFS variable along the flight track using the surrounding grid
    cells.

    Args:
        ifs: IFS dataset
        lat_circles: list of arrays with each holding the lat indices of the point of interest and the surrounding
        grid cells
        lon_circles: list of arrays with each holding the lon indices of the point of interest and the surrounding
        grid cells
        aircraft_height_level: index of the aircraft height level in the model
        bahamas_time: DataArray with the bahamas time subsample to 0.5Hz as used for the ecRad modelling
        i: time index

    Returns: A tuple with the mean and the standard deviation of all variables in ifs

    """
    ifs_grid_sel = ifs.isel(lat=lat_circles[i], lon=lon_circles[i],
                            level=aircraft_height_level[i], half_level=aircraft_height_level[i],
                            mid_level=aircraft_height_level[i], column=0).stack(column=("lat", "lon")).reset_index(
        "column").reset_coords(["lat", "lon"])
    ifs_grid_inp = ifs_grid_sel.interp(time=bahamas_time).isel(time=i)

    return ifs_grid_inp.mean(dim="column"), ifs_grid_inp.std(dim="column")


if __name__ == "__main__":
    # %% import modules
    import datetime
    import pylim.helpers as h
    from pylim import reader
    import os
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    import multiprocessing as mp
    import datetime
    import logging
    # %% set up logging
    start = pd.to_datetime(datetime.datetime.now())
    log = logging.getLogger(__name__)
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)

    # %% set options and paths
    flight = "Flight_20210629a"
    date = "20210629"
    bahamas_dir = h.get_path("bahamas", flight)
    ecrad_dir = os.path.join(h.get_path("ecrad"), date)
    ifs_dir = os.path.join(h.get_path("ifs", flight), date)
    log.info(f"Options set:\nflight: {flight}\ndate: {date}\nSkript started: {start}")
    # %% read in IFS model data
    ifs_file = f"ifs_{date}_00_ml_processed.nc"
    ifs = xr.open_dataset(f"{ifs_dir}/{ifs_file}")

    # %% read in bahamas file
    file = [f for f in os.listdir(bahamas_dir) if f.endswith(".nc")][0]
    bahamas = reader.read_bahamas(os.path.join(bahamas_dir, file))

    # %% read in ecRad output
    ecrad_output_file = [f for f in os.listdir(ecrad_dir) if f"output_{date}" in f][0]
    ecrad_output = xr.open_dataset(f"{ecrad_dir}/{ecrad_output_file}")
    # assign coordinates to band_sw
    ecrad_output = ecrad_output.assign_coords({"band_sw": range(1, 15), "band_lw": range(1, 17),
                                               "half_level": range(138)})
    # select only the center column for the analysis
    if ecrad_output.dims["column"] > 1:
        ecrad_output = ecrad_output.isel(column=ecrad_output.dims["column"] // 2)

    # %% select only relevant time (bahamas time range)
    ecrad_output = ecrad_output.where(ecrad_output.time == bahamas.time, drop=True)

    # %% select only bahamas data corresponding to ecRad model output
    bahamas_sel = bahamas.sel(time=ecrad_output.time)

    # %% calculate pressure height
    log.info("Calculate pressure height from model...")
    q_air = 1.292
    g_geo = 9.81
    pressure_hl = ecrad_output["pressure_hl"]
    ecrad_output["press_height"] = -(pressure_hl[:, 137]) * np.log(pressure_hl[:, :] / pressure_hl[:, 137]) / (
            q_air * g_geo)
    # replace TOA height (calculated as infinity) with nan
    ecrad_output["press_height"] = ecrad_output["press_height"].where(ecrad_output["press_height"] != np.inf, np.nan)

    # %% get height level of actual flight altitude in ecRad model, this determines only the index of the level
    ecrad_timesteps = len(ecrad_output.time)
    aircraft_height_level = np.zeros(ecrad_timesteps)

    for i in tqdm(range(ecrad_timesteps), desc="Find aircraft height level in model"):
        aircraft_height_level[i] = h.arg_nearest(ecrad_output["press_height"][i, :].values, bahamas_sel.IRS_ALT[i].values)

    aircraft_height_level = aircraft_height_level.astype(int)

    # %% select a 3 x 11 lat/lon grid around closest grid points to flight track
    lat_id = [h.arg_nearest(ifs.lat, lat) for lat in bahamas_sel.IRS_LAT.values]
    lon_id = [h.arg_nearest(ifs.lon, lon) for lon in bahamas_sel.IRS_LON.values]
    lat_circles = [np.arange(lat - 1, lat + 2) for lat in lat_id]
    lon_circles = [np.arange(lon - 5, lon + 6) for lon in lon_id]

    # %% calculate mean and std for each grid point along flight track and save it to a new variable
    bahamas_time = bahamas_sel.time  # time steps which to interpolate IFS data on
    n_processes = mp.cpu_count() - 4  # number of cpus to use
    nr_time = len(bahamas_time)  # number of time steps to calculate
    iterations = range(0, int(nr_time))
    log.info(f"Number of time steps to calculate: {nr_time}")

    # %% open a Pool of n_processes to calculate the means and standard deviations
    log.info("Starting calculation...")
    with mp.Pool(n_processes) as pool:
        means, stds = zip(*pool.starmap(calculate_ifs_along_flighttrack,
                                        [(ifs, lat_circles, lon_circles, aircraft_height_level, bahamas_time, i)
                                         for i in tqdm(iterations, desc="Sending tasks to processes")])
                          )
    # pool.map returns a list of the outputs off the function which is a tuple, * unpacks this list of tuples into to
    # tuples and zip zips them, so they can be assigned to a variable each
    # the result are two lists one with the means and one with the standard deviations
    log.info("Done with calculating the means and standard deviations")

    # %% concatenate all means and standard deviations into new datasets
    ifs_means = xr.concat(means, dim="time")
    ifs_stds = xr.concat(stds, dim="time")

    # save means and stds for future use
    ifs_means.to_netcdf(f"{ifs_dir}/ifs_means_{date}_00_ml_processed.nc")
    ifs_stds.to_netcdf(f"{ifs_dir}/ifs_stds_{date}_00_ml_processed.nc")

    log.info(f"Saved outfiles to {ifs_dir}\n"
             f"Time elapsed: {pd.to_datetime(datetime.datetime.now()) - start}")
