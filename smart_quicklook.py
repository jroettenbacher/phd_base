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
import matplotlib.pyplot as plt

raw_path, pixel_path, _, data_path, plot_path = smart.set_paths()
flight = "Flight_20210705a"
ql_path = f"{plot_path}/quicklooks/{flight}"
make_dir(ql_path)
calibrated_path = smart.get_path("calibrated")
# %% read in raw files
inpath = f"{raw_path}/{flight}"
raw_files = os.listdir(inpath)
outpath = f"{ql_path}/raw"
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
        smart.plot_smart_data(file, [550], path=inpath, plot_path=outpath, save_fig=True)
    except AssertionError:
        smart.plot_smart_data(file, [1200], path=inpath, plot_path=outpath, save_fig=True)

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
        smart.plot_smart_data(file, [550], path=inpath, plot_path=outpath, save_fig=True)
    except AssertionError:
        # plot a time series of 1500 nm
        smart.plot_smart_data(file, [1200], path=inpath, plot_path=outpath, save_fig=True)

# %% plot Fdw with yaw angle - read in and prepare data

path = smart.get_path("calibrated")
nav_path = "C:/Users/Johannes/Documents/Doktor/campaigns/CIRRUS-HL/ASP04/NavCommand/20210625"
nav_file = "Nav_IMS0000.Asc"
flight = "flight_02"
inpath = f"{path}/{flight}"
nav_inpath = f"{nav_path}/{nav_file}"
file = "2021_06_25_10_08.Fdw_VNIR_cor_calibrated_norm.dat"
df = smart.read_smart_cor(inpath, file)
with open(nav_inpath) as f:
    time_info = f.readlines()[1]
start_time = pd.to_datetime(time_info[11:31], format="%m/%d/%Y %H:%M:%S")
start_date = pd.Timestamp(year=start_time.year, month=start_time.month, day=start_time.day)
header = ["marker", "seconds", "roll", "pitch", "yaw", "AccS_X", "AccS_Y", "AccS_Z", "OmgS_X", "OmgS_Y", "OmgS_Z"]
nav = pd.read_csv(nav_inpath, sep="\s+", skiprows=13, header=None, names=header)
nav["time"] = pd.to_datetime(nav["seconds"], origin=start_date, unit="s")
nav = nav.set_index("time")
nav = nav.resample("S").mean()
# nav["yaw"] = nav["yaw"] * 100

# %% plot Fdw with yaw angle
# ax = smart.plot_smart_data(file, wavelength=[550], path=inpath)
fig, ax = plt.subplots()
time_range = df["2021-06-25 11:30":"2021-06-25 13:00"].index
nav_plot = nav[time_range[0]:time_range[-1]]
df.iloc[:, 550][time_range].plot(ax=ax, label="Fdw Irradiance")
ax.set_ylabel("Irradiance (W$\\,$m$^{-2}\\,$nm$^{-1}$)")
ax2 = ax.twinx()
ax2.plot(nav_plot.index, nav_plot["yaw"], label="IMS yaw", c="red")
ax2.set_ylabel("Yaw Angle [deg]")
ax.grid()
ax.legend()
ax2.legend()
plt.title("2021-06-25 Calibrated Irradiance measurement and IMS yaw angle")
plt.tight_layout()
plt.savefig("20210625_smart_fdw_yaw_zoom.png", dpi=100)
plt.close()
