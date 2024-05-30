#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 10.10.2023

Select closest points of CAMS global reanalysis and global greenhouse gas reanalysis data to flight track.
Read in CAMS from different sources (ADS, Copernicus Knowledge Base (47r1)).
The Copernicus Atmospheric Data Store provides the monthly means of trace gase concentrations and aerosol concentrations.
They are the basis for the files available via the Copernicus Knowledge Base.
However, these files have been processed a bit before their use in the IFS.
See :ref:`processing:IFS/CAMS Download` for more details and the links to the files.

**Required User Input:**

- source (47r1, ADS)
- year (2020, 2019)
- date (20220411)

**Input:**

- monthly mean CAMS data
- greenhouse gas time series (needed if source is '47r1')

**Output:**

- trace gas and aerosol monthly climatology interpolated to flight day along flight track

"""
if __name__ == "__main__":
    # %% import modules
    import pylim.helpers as h
    import numpy as np
    import pandas as pd
    import xarray as xr
    from sklearn.neighbors import BallTree
    from datetime import datetime

    # %% set source and paths
    args = h.read_command_line_args()
    campaign = args["campaign"] if "campaign" in args else "halo-ac3"
    source = args["source"] if "source" in args else "47r1"
    date = args["date"] if "date" in args else "20220411"
    climatology_year = "2020"

    # setup logging
    __file__ = None if '__file__' not in locals() else __file__
    log = h.setup_logging("./logs", __file__, f"{date}")
    log.info(f"The following options have been passed:\n"
             f"campaign: {campaign}\n"
             f"source: {source}\n"
             f"date: {date}\n")

    cams_path = h.get_path("cams_raw", campaign=campaign)
    cams_output_path = h.get_path("cams", campaign=campaign)
    # IFS path for flight track file
    ifs_path = h.get_path("ifs", campaign=campaign)
    if source == "ADS":
        aerosol_file = f"cams_eac4_global_reanalysis_mm_{climatology_year}_pl.nc"
        trace_gas_file = f"cams_global_ghg_reanalysis_mm_{climatology_year}_pl.nc"
    elif source == "47r1":
        aerosol_file = "aerosol_cams_3d_climatology_47r1.nc"
        trace_gas_file = "greenhouse_gas_climatology_46r1.nc"
        scaling_file = "greenhouse_gas_timeseries_CMIP6_SSP370_CFC11equiv_47r1.nc"
        scaling_ds = xr.open_dataset(f"{cams_path}/{scaling_file}", decode_times=False).isel(time=int(date[0:4]))
    else:
        raise ValueError(f"Unknown source: {source}")

    # %% read in data
    nav_data_ip = pd.read_csv(f"{ifs_path}/{date}/nav_data_ip_{date}.csv", index_col="time", parse_dates=True)
    aerosol = xr.open_dataset(f"{cams_path}/{aerosol_file}")
    trace_gas = xr.open_dataset(f"{cams_path}/{trace_gas_file}")

    # %% calculate pressure at full model level for aerosol file
    # add half the difference between the pressure at the base and top of the layer to the pressure at the base of the layer
    aerosol["full_level_pressure"] = (aerosol.half_level_pressure
                                      + 0.5 * aerosol.half_level_delta_pressure)

    # %% linearly interpolate in time and rename dimensions
    new_time_axis = pd.date_range(f"{date[0:4]}-01-15", f"{date[0:4]}-12-15", freq=pd.offsets.SemiMonthBegin(2))
    date_dt = pd.to_datetime(date)
    if source == "ADS":
        aerosol = (aerosol
                   .assign_coords(time=new_time_axis)
                   .interp(time=date_dt)
                   .rename({"latitude": "lat", "longitude": "lon"}))
        trace_gas = (trace_gas
                     .assign_coords(time=new_time_axis)
                     .interp(time=date_dt)
                     .rename({"latitude": "lat", "longitude": "lon"}))
    elif source == "47r1":
        aerosol = (aerosol
                   .assign_coords(month=new_time_axis)
                   .interp(month=date_dt)
                   .rename(month="time"))
        trace_gas = (trace_gas
                     .assign_coords(month=new_time_axis)
                     .interp(month=date_dt)
                     .rename(latitude="lat", month="time"))

        # scale trace house gases so that their annual-mean surface concentrations
        # match the values appropriate for the current year
        for var in scaling_ds:
            scale_factor = (scaling_ds[var] / trace_gas[var].attrs["surface_mean"]).to_numpy()
            trace_gas[var] = np.multiply(trace_gas[var], scale_factor)  # use numpy function to conserve attributes
            trace_gas[var].attrs["comment1"] = (f"Scaled to annual-mean surface concentration of "
                                                f"{date[0:4]} provided in {scaling_file}")

    # %% create array of aircraft locations
    points = np.deg2rad(
        np.column_stack(
            (nav_data_ip.lat.to_numpy(), nav_data_ip.lon.to_numpy())))

    # %% select closest points along flight track from aerosol data
    aerosol_latlon = aerosol.stack(latlon=["lat", "lon"])  # combine lat and lon into one dimension
    # make an array with all lat lon combinations
    aerosol_lat_lon = np.array([np.array(element) for element in aerosol_latlon["latlon"].to_numpy()])
    # build the look-up tree
    aerosol_tree = BallTree(np.deg2rad(aerosol_lat_lon), metric="haversine")
    # query the tree for the closest CAMS grid points to the flight track
    dist, idx = aerosol_tree.query(points, k=1)
    # select only the closest grid points along the flight track
    aerosol_sel = aerosol_latlon.isel(latlon=idx.flatten())
    # reset index and make lat, lon and time (month) a variable
    aerosol_sel = (aerosol_sel
                   .reset_index(["latlon", "lat", "lon"])
                   .reset_coords(["lat", "lon", "time"])
                   .drop_vars(["time", "lat", "lon"])
                   .rename(latlon="time", lev="level")
                   .assign(time=nav_data_ip.index.to_numpy()))  # replace latlon with time as a dimension/coordinate

    # %% select zonal mean closest to flight track from greenhouse gas data
    trace_gas_sel = (trace_gas
                     .sel(lat=nav_data_ip.lat.to_numpy(), method="nearest")
                     .drop_vars("time")
                     .rename(lat="time")
                     .assign(time=nav_data_ip.index.to_numpy())
                     .assign(level=trace_gas.pressure))

    # %% save files to netcdf
    history_str = (f"\n{datetime.today().strftime('%c')}: "
                   f"formatted file to serve as input to ecRad"
                   f" using ecrad_cams_preprocessing.py")
    try:
        aerosol_sel.attrs["history"] = aerosol_sel.attrs["history"] + history_str
        trace_gas_sel.attrs["history"] = trace_gas_sel.attrs["history"] + history_str
    except KeyError:
        aerosol_sel.attrs["history"] = history_str
        trace_gas_sel.attrs["history"] = history_str

    aerosol_outfile = f"aerosol_mm_climatology_{climatology_year}_{source}_{date}.nc"
    trace_gas_outfile = f"trace_gas_mm_climatology_{climatology_year}_{source}_{date}.nc"
    aerosol_sel.to_netcdf(f"{cams_output_path}/{aerosol_outfile}",
                          format='NETCDF4_CLASSIC')
    trace_gas_sel.to_netcdf(f"{cams_output_path}/{trace_gas_outfile}",
                            format='NETCDF4_CLASSIC')

    log.info(f"\nSaved {aerosol_outfile}\nand {trace_gas_outfile}\nto {cams_output_path}")
