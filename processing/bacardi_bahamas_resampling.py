#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 19.09.2023

Resample BACARDI and BAHAMAS data to one minute and one second resolution.

"""
# %% import modules
import pylim.helpers as h
import pylim.halo_ac3 as meta
from pylim import reader
import ac3airborne
from ac3airborne.tools import flightphase
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

cbc = h.get_cb_friendly_colors()

# %% set paths
campaign = "halo-ac3"
# keys = "RF17"  # run for single flight
keys = [f"RF{i:02}" for i in range(3, 19)]  # run for all flights

# %% run resampling for all flights in keys
for key in tqdm(keys):
    flight = meta.flight_names[key]
    date = flight[9:17]

    bahamas_path = h.get_path("bahamas", flight, campaign)
    bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{key}_v1.nc"
    bacardi_path = h.get_path("bacardi", flight, campaign)
    bacardi_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1_JR.nc"

    # %% read in BACARDI and BAHAMAS data
    try:
        bahamas_ds = reader.read_bahamas(f"{bahamas_path}/{bahamas_file}")
        bacardi_ds = xr.open_dataset(f"{bacardi_path}/{bacardi_file}")
    except FileNotFoundError as e:
        print(f"Skipping {key} because of {e}")
        continue

    for resampling_time in ["1s", "1Min"]:
        # resample fast with pandas
        df_h = bahamas_ds.to_dataframe().resample(resampling_time).mean()  # what we want (quickly), but in Pandas form
        vals = [xr.DataArray(data=df_h[c], dims=['time'], coords={'time': df_h.index}, attrs=bahamas_ds[c].attrs) for c in
                df_h.columns]
        bahamas_out = xr.Dataset(dict(zip(df_h.columns, vals)), attrs=bahamas_ds.attrs)
        # BACARDI - filter for motion before resampling
        ds = bacardi_ds.where(bacardi_ds.motion_flag)
        # overwrite variables which do not need to be filtered with original values
        for var in ["alt", "lat", "lon", "sza", "saa", "attitude_flag", "segment_flag", "motion_flag"]:
            ds[var] = bacardi_ds[var]
        df_h = ds.to_dataframe().resample(resampling_time).mean()  # what we want (quickly), but in Pandas form
        vals = [xr.DataArray(data=df_h[c], dims=['time'], coords={'time': df_h.index}, attrs=ds[c].attrs) for c in
                df_h.columns]
        bacardi_out = xr.Dataset(dict(zip(df_h.columns, vals)), attrs=ds.attrs)

        # write to new file
        bahamas_outfile = bahamas_file.replace(".nc", f"_{resampling_time}.nc")
        bacardi_outfile = bacardi_file.replace(".nc", f"_{resampling_time}.nc")

        bahamas_out.to_netcdf(f"{bahamas_path}/{bahamas_outfile}", format="NETCDF4_CLASSIC")
        bacardi_out.to_netcdf(f"{bacardi_path}/{bacardi_outfile}", format="NETCDF4_CLASSIC")
        #TODO: remove fill value from all variables, add time encoding as suitable for HALO-AC3