#!/usr/bin/env python
"""Plot and save SMART quicklooks for one flight
author: Johannes Roettenbacher
"""

# %% import modules and set paths
import os
import smart
import pandas as pd
import numpy as np

raw_path, pixel_path, _, data_path, plot_path = smart.set_paths()
ql_path = f"{plot_path}/quicklooks"
calibrated_path = smart.get_path("calibrated")
flight = "flight_00"
# %% read in raw files
inpath = f"{raw_path}/{flight}"
raw_files = os.listdir(inpath)
for file in raw_files:
    smart.plot_smart_data(file, "all", path=inpath, plot_path=ql_path, save_fig=True)
    try:
        smart.plot_smart_data(file, [500], path=inpath, plot_path=ql_path, save_fig=True)
    except AssertionError:
        smart.plot_smart_data(file, [1500], path=inpath, plot_path=ql_path, save_fig=True)

# %% read in dark current corrected files
inpath = f"{data_path}/{flight}"
data_files = os.listdir(inpath)
for file in data_files:
    smart.plot_smart_data(file, "all", path=inpath, plot_path=ql_path, save_fig=True)
    try:
        smart.plot_smart_data(file, [500], path=inpath, plot_path=ql_path, save_fig=True)
    except AssertionError:
        smart.plot_smart_data(file, [1500], path=inpath, plot_path=ql_path, save_fig=True)

# %% read in calibrated files
inpath = f"{calibrated_path}/{flight}"
calib_files = os.listdir(inpath)
for file in calib_files:
    smart.plot_smart_data(file, "all", path=inpath, plot_path=ql_path, save_fig=True)
    try:
        smart.plot_smart_data(file, [500], path=inpath, plot_path=ql_path, save_fig=True)
    except AssertionError:
        smart.plot_smart_data(file, [1500], path=inpath, plot_path=ql_path, save_fig=True)


