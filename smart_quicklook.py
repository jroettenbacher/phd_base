#!/usr/bin/env python
"""Plot and save SMART quicklooks of dark current corrected and calibrated measurements for one flight
author: Johannes Roettenbacher
"""

# %% import modules and set paths
import os
from functions_jr import make_dir
import smart
import pandas as pd
import numpy as np

raw_path, pixel_path, _, data_path, plot_path = smart.set_paths()
flight = "flight_00"
ql_path = f"{plot_path}/quicklooks/{flight}"
make_dir(ql_path)
calibrated_path = smart.get_path("calibrated")
# %% read in raw files
inpath = f"{raw_path}/{flight}"
raw_files = os.listdir(inpath)
outpath = f"{ql_path}"
make_dir(outpath)
for file in raw_files:
    smart.plot_smart_data(file, "all", path=inpath, plot_path=outpath, save_fig=True)
    try:
        smart.plot_smart_data(file, [500], path=inpath, plot_path=outpath, save_fig=True)
    except AssertionError:
        smart.plot_smart_data(file, [1500], path=inpath, plot_path=outpath, save_fig=True)

# %% read in dark current corrected files
inpath = f"{data_path}/{flight}"
data_files = os.listdir(inpath)
outpath = f"{ql_path}"
make_dir(outpath)
for file in data_files:
    smart.plot_smart_data(file, "all", path=inpath, plot_path=outpath, save_fig=True)
    try:
        smart.plot_smart_data(file, [500], path=inpath, plot_path=outpath, save_fig=True)
    except AssertionError:
        smart.plot_smart_data(file, [1500], path=inpath, plot_path=outpath, save_fig=True)

# %% read in calibrated files
inpath = f"{calibrated_path}/{flight}"
calib_files = os.listdir(inpath)
outpath = f"{ql_path}/calibrated"
make_dir(outpath)
for file in calib_files:
    # plot the time average of the flight
    smart.plot_smart_data(file, "all", path=inpath, plot_path=outpath, save_fig=True)
    try:
        # plot a time series of 500 nm
        smart.plot_smart_data(file, [500], path=inpath, plot_path=outpath, save_fig=True)
    except AssertionError:
        # plot a time series of 1500 nm
        smart.plot_smart_data(file, [1500], path=inpath, plot_path=outpath, save_fig=True)

#
