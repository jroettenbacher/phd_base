#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 17.10.2023

Add a couple of more variables to BAHAMAS

"""
# %% import modules
import pylim.helpers as h
from pylim.reader import read_bahamas
from pylim.bahamas import calculate_distances
import xarray as xr

# %% set options and paths
campaign = "halo-ac3"
keys = ["RF17", "RF18"]
for key in keys:
    print(f"Processing flight {key}")
    if campaign == "halo-ac3":
        import pylim.halo_ac3 as meta
        flight = meta.flight_names[key]
        date = flight[9:17]
    else:
        import pylim.cirrus_hl as meta
        flight = key
        date = flight[7:15]

    bahamas_path = h.get_path("bahamas", flight, campaign)
    bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{key}_v1.nc"
    outfile = bahamas_file.replace(".nc", "_JR.nc")

    # %% read in data
    ds = read_bahamas(f"{bahamas_path}/{bahamas_file}")

    # %% add variables
    ds = calculate_distances(ds)  # calculate geodesic distance

    # calculate distance according to ground speed
    sampling_frequency = 0.1
    gs_distance = (ds.IRS_GS * sampling_frequency).to_numpy()
    ds["gs_distance"] = xr.DataArray(gs_distance, dims="time",
                                     attrs={"units": "m", "long_name": "Ground distance",
                                            "description": "Distance travelled according to ground speed.\n"
                                                           " d = GS * sampling_frequency"})

    # %% write to new netcdf file
    ds.to_netcdf(f"{bahamas_path}/{outfile}")

    # %% resample to secondly resolution and save to new netcdf file
    (ds.resample(time="1s").mean()
     .to_netcdf(f"{bahamas_path}/{outfile.replace('.nc', '_1s.nc')}"))
