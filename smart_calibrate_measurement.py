#!/usr/bin/env python
"""Calibrate measurement files with the transfer calibration
author: Johannes Roettenbacher
"""

# %% import modules and set paths
import os
import pandas as pd
import smart
from smart import lookup
from functions_jr import make_dir

flight = "flight_00"  # set flight folder
_, _, calib_path, data_path, _ = smart.set_paths()
calibrated_path = smart.get_path("calibrated")
inpath = f"{data_path}/{flight}"
outpath = f"{calibrated_path}/{flight}"
make_dir(outpath)  # create outpath if necessary

# %% read in dark current corrected measurement files
files = [f for f in os.listdir(inpath)]
for file in files:
    date_str, channel, direction = smart.get_info_from_filename(file)
    spectrometer = lookup[f"{direction}_{channel}"]
    measurement = smart.read_smart_cor(f"{data_path}/{flight}", file)
    measurement[measurement.values < 0] = 0  # set negative values to 0

    # %% read in matching transfer calibration file from same day
    cali_file = f"{calib_path}/{date_str}_{spectrometer}_{direction}_{channel}_transfer_calib.dat"
    print(f"Calibration file used:\n {cali_file}")
    cali = pd.read_csv(cali_file)
    # convert to long format
    m_long = measurement.melt(var_name="pixel", value_name="counts", ignore_index=False)
    # merge field calibration factor to long df on pixel column
    df = m_long.join(cali.set_index(cali.pixel)["c_field"], on="pixel")
    df[direction] = df["counts"] * df["c_field"]  # calculate calibrated radiance/irradiance

    # %% save wide format calibrated measurement
    df_out = df.pivot(columns="pixel", values=direction)  # convert to wide format (row=time, column=pixel)
    outfile = f"{outpath}/{file.replace('.dat', '_calibrated.dat')}"
    df_out.to_csv(outfile, sep="\t")
    print(f"Saved {outfile}")
