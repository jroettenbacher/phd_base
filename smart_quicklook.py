#!/usr/bin/env python
"""Plot and save SMART quicklooks of dark current corrected and calibrated measurements for one flight
author: Johannes Roettenbacher
"""

# %% import modules
import os
from functions_jr import make_dir
import smart
from cirrus_hl import lookup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%  set paths
flight = "Flight_20210728a"
raw_path = smart.get_path("raw", flight)
pixel_path = smart.get_path("pixel_wl")
data_path = smart.get_path("data", flight)
calibrated_path = smart.get_path("calibrated", flight)
plot_path = smart.get_path("plot")
ql_path = f"{plot_path}/quicklooks/{flight}"
make_dir(ql_path)
# %% read in raw files
raw_files = os.listdir(raw_path)
outpath = f"{ql_path}/raw"
make_dir(outpath)
for file in raw_files:
    smart.plot_smart_data(file, "all", path=raw_path, plot_path=outpath, save_fig=True)
    try:
        smart.plot_smart_data(file, [500], path=raw_path, plot_path=outpath, save_fig=True)
    except AssertionError:
        smart.plot_smart_data(file, [1500], path=raw_path, plot_path=outpath, save_fig=True)

# %% read in dark current corrected files
data_files = os.listdir(data_path)
outpath = f"{ql_path}"
make_dir(outpath)
for file in data_files:
    smart.plot_smart_data(flight, file, "all", path=data_path, plot_path=outpath, save_fig=True)
    try:
        smart.plot_smart_data(flight, file, [550], path=data_path, plot_path=outpath, save_fig=True)
    except AssertionError:
        smart.plot_smart_data(flight, file, [1200], path=data_path, plot_path=outpath, save_fig=True)

# read in calibrated files
calib_files = os.listdir(calibrated_path)
outpath = f"{ql_path}/calibrated"
make_dir(outpath)
for file in calib_files:
    # plot the time average of the flight
    smart.plot_smart_data(flight, file, "all", path=calibrated_path, plot_path=outpath, save_fig=True)
    try:
        # plot a time series of 500 nm
        smart.plot_smart_data(flight, file, [550], path=calibrated_path, plot_path=outpath, save_fig=True)
    except AssertionError:
        # plot a time series of 1500 nm
        smart.plot_smart_data(flight, file, [1200], path=calibrated_path, plot_path=outpath, save_fig=True)

# %% plot Fdw with IMS and stabbi angles - read in and prepare data

flight = "Flight_20210628a"
path = smart.get_path("calibrated")
horipath = smart.get_path("horidata")
hori_dir = os.path.join(horipath, flight)
hori_file = [f for f in os.listdir(hori_dir) if f.endswith("dat")][0]  # select stabbi file
# read stabbi data and make PCTIME a datetime column and set it as index
horidata = pd.read_csv(f"{hori_dir}/{hori_file}", skipinitialspace=True, sep="\t")
horidata["PCTIME"] = pd.to_datetime(horidata["DATE"] + " " + horidata["PCTIME"], format='%Y/%m/%d %H:%M:%S.%f')
horidata.set_index("PCTIME", inplace=True)
nav_file = [f for f in os.listdir(hori_dir) if "IMS" in f][0]  # select IMS file
nav_df = smart.read_nav_data(f"{hori_dir}/{nav_file}")
nav_df = nav_df.resample("S").mean()  # resample IMS data to one second to save RAM
inpath = f"{path}/{flight}"  # get inpath for SMART data
file = [f for f in os.listdir(inpath) if "Fdw_VNIR" in f][0]  # select Fdw VNIR file
# get info from filename and select pixel to wavelength file
date_str, channel, direction = smart.get_info_from_filename(file)
pixel_wl = smart.read_pixel_to_wavelength(smart.get_path("pixel_wl"), lookup[f"{direction}_{channel}"])
pixel_nr, wl = smart.find_pixel(pixel_wl, 550)
# read in SMART data and select specified wavelength
df = smart.read_smart_cor(inpath, file).iloc[:, pixel_nr]

# %% read in BACARDI simulated stuff
bacardi_path = "C:/Users/Johannes/Documents/Doktor/campaigns/CIRRUS-HL/simulations/Flight_20210628a/BBR_Fdn_clear_sky_Flight_20210628a_R0_ds_high.dat"
bacardi = pd.read_csv(bacardi_path, sep="\s+", skiprows=34)
bacardi["time"] = pd.to_datetime(bacardi["sod"], origin="2021-06-28", unit="s")
bacardi.set_index("time", inplace=True)

# %% plot Fdw with angles
date_str = date_str.replace("_", "-")
fig, ax = plt.subplots(figsize=(7, 5))
# select a specific time range
time_range = df[f"{date_str} 09:30":f"{date_str} 11:30"].index
nav_plot = nav_df[time_range[0]:time_range[-1]]
df_plot = df.loc[time_range]
hori_plot = horidata.loc[time_range[0]:time_range[-1]]
bacardi_plot = bacardi.loc[time_range[0]:time_range[-1]]
df_plot.plot(ax=ax, label="Fdw Irradiance")
# ax.plot(bacardi_plot["F_dw"]/1000, label="Fdw broadband / 100")
ax.set_ylabel("Irradiance (W$\\,$m$^{-2}\\,$nm$^{-1}$)")
ax2 = ax.twinx()
ax2.plot(nav_plot.index, nav_plot["roll"], label="IMS roll", c="red")
# ax2.plot(nav_plot.index, nav_plot["pitch"], label="IMS pitch", c="red")
ax2.plot(nav_plot.index, nav_plot["yaw"]/10, label="IMS yaw / 10", c="orange")
ax2.plot(hori_plot["POSN3"], label="Stabilization Roll", c="pink")
# ax2.plot(hori_plot["POSN4"], label="Stabilization Pitch", c="pink")
ax2.set_ylabel("Roll Angle [deg]")
ax.grid()
ax.legend(loc=1)
ax2.legend(loc=2)
plt.title(f"{date_str} Calibrated Irradiance measurement and position angles")
plt.tight_layout()
# plt.show()
plt.savefig(f"{horipath}/plots/troubleshooting/{date_str}_smart_{direction}_angles.png", dpi=100)
plt.close()
