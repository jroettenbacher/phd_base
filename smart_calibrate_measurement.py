#!/usr/bin/env python
"""Calibrate measurement files with the transfer calibration
author: Johannes Roettenbacher
"""

# %% import modules and set paths
import os
import numpy as np
import pandas as pd
import smart
from smart import lookup
from functions_jr import make_dir
import logging

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

# %% set user given parameters
flight = "flight_02"  # set flight folder
t_int_asp06 = 300  # give integration time of field measurement for ASP06
t_int_asp07 = 300  # give integration time of field measurement for ASP07
normalize = True  # normalize counts with integration time
# give date of transfer calib to use for calibrating measurement if not same as measurement date else set to ""
date = ""

# %% set paths
norm = "_norm" if normalize else ""
_, _, calib_path, data_path, _ = smart.set_paths()
calibrated_path = smart.get_path("calibrated")
inpath = f"{data_path}/{flight}"
outpath = f"{calibrated_path}/{flight}"
make_dir(outpath)  # create outpath if necessary

# %% read in dark current corrected measurement files
files = [f for f in os.listdir(inpath)]
for file in files:
    date_str, channel, direction = smart.get_info_from_filename(file)
    date_str = date if len(date) > 0 else date_str  # overwrite date_str if date is given
    spectrometer = lookup[f"{direction}_{channel}"]
    t_int = t_int_asp06 if "ASP06" in spectrometer else t_int_asp07  # select relevant integration time
    measurement = smart.read_smart_cor(f"{data_path}/{flight}", file)
    # measurement[measurement.values < 0] = 0  # set negative values to 0

    # %% read in matching transfer calibration file from same day or from given day with matching t_int
    cali_file = f"{calib_path}/{date_str}_{spectrometer}_{direction}_{channel}_{t_int}ms_transfer_calib{norm}.dat"
    log.info(f"Calibration file used:\n {cali_file}")
    cali = pd.read_csv(cali_file)
    # convert to long format
    m_long = measurement.melt(var_name="pixel", value_name="counts", ignore_index=False)
    if normalize:
        m_long["counts"] = m_long["counts"] / t_int

    # merge field calibration factor to long df on pixel column
    df = m_long.join(cali.set_index(cali.pixel)["c_field"], on="pixel")
    df[direction] = df["counts"] * df["c_field"]  # calculate calibrated radiance/irradiance

    # %% save wide format calibrated measurement
    # remove rows where the index is nan
    df = df[~np.isnan(df.index)]
    df_out = df.pivot(columns="pixel", values=direction)  # convert to wide format (row=time, column=pixel)
    outname = "_calibrated_norm.dat" if normalize else "_calibrated.dat"
    outfile = f"{outpath}/{file.replace('.dat', outname)}"
    df_out.to_csv(outfile, sep="\t")
    log.info(f"Saved {outfile}")
