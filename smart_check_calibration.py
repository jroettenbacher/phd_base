#!/usr/bin/env python
"""Go through all calibrations and check the quality of the calibration
author: Johannes Röttenbacher
"""

# %% module import
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bahamas import read_bahamas
from cirrus_hl import lookup, transfer_calibs
from smart import get_path, plot_smart_data, read_smart_raw
from functions_jr import set_cb_friendly_colors
import logging

log = logging.getLogger("smart")
log.setLevel(logging.INFO)

# %% set paths
calib_path = get_path("calib")
plot_path = f"{get_path('plot')}/quality_check_calibration"

# %% list all files from one spectrometer
prop = "Fup_SWIR"
files = [f for f in os.listdir(calib_path) if lookup[prop] in f]

# %% select only normalized and transfer calib files
files = [f for f in files if "norm" in f]
files = [f for f in files if "transfer" in f]
files.sort()
if prop == "Fdw_SWIR" or prop == "Fdw_VNIR" or prop == "Fup_VNIR":
    files.pop(2)  # remove 500ms file from the 16th
elif prop == "Fup_SWIR":
    files.pop(1)  # remove 500ms file from the 16th
else:
    pass

# %% compare 300ms and 500ms normalized measurements for Fdw_SWIR
# file_300, file_500 = files[1], files[2]  # TODO

# %% read in transfer calib file and add timestamp from filename
date_strs = [f[0:10] for f in files]  # extract date strings from filenames
df = pd.DataFrame()
for f, date_str in zip(files, date_strs):
    df_tmp = pd.read_csv(f"{calib_path}/{f}")
    df_tmp["date"] = np.repeat(date_str, len(df_tmp))
    df = pd.concat([df, df_tmp])

df = df.reset_index(drop=True)

# %% plot relation between lab calib measurement and each transfer calib measurement
colors = plt.cm.tab20.colors  # get tab20 colors
plt.rc('axes', prop_cycle=(mpl.cycler('color', colors)))  # Set the default color cycle
plt.rc('font', family="serif", size=14)
zoom = False  # zoom in on y axis

fig, ax = plt.subplots(figsize=(10, 6))
for date_str in date_strs:
    df[df["date"] == date_str].sort_values(["wavelength"]).plot(x="wavelength", y="rel_ulli", label=date_str, ax=ax)

if zoom:
    ax.set_ylim((0, 10))
    zoom = "_zoom"
else:
    zoom = ""
ax.set_ylabel("$S_{ulli, lab} / S_{ulli, field}$")
ax.set_xlabel("Wavenlength (nm)")
ax.set_title(f"Relation between Lab Calib measurement \nand Transfer Calib measurement - {prop}")
ax.grid()
ax.legend(bbox_to_anchor=(1.04, 1.1), loc="upper left")
plt.tight_layout()
# plt.show()
plt.savefig(f"{plot_path}/SMART_calib_rel_lab-field_{prop}{zoom}.png", dpi=100)
plt.close()

# %% take the average over n pixels and prepare data frame for plotting
if "SWIR" in prop:
    wl1, wl2 = 1400, 1420  # set wavelengths for averaging
else:
    wl1, wl2 = 550, 570
df_ts = df[df["wavelength"].between(wl1, wl2)]
df_mean = df_ts.groupby("date").mean().reset_index()
# df_mean["dt"] = pd.to_datetime(df_mean.index.values, format="%Y_%m_%d")
# df_mean.set_index(df_mean["dt"], inplace=True)

# %% plot a time series of the calibration factor
fig, ax = plt.subplots(figsize=(10, 6))
df_mean.plot(y="c_field", ax=ax, label="$c_{field}$")
df_mean.plot(y="c_lab", c="#117733", ax=ax, label="$c_{lab}$")
# ax.set_ylim((1, 2.5))
ax.set_xticks(df_mean.index.values)
ax.set_xticklabels(df_mean.date.values, fontsize=14, rotation=45, ha="right")
# ax.tick_params(axis="x", labelsize=12)
ax.set_ylabel("Calibration Factor")
ax.set_xlabel("Date")
ax.set_title(f"{prop} - Evolution of the Field Calibration Factor\n for the mean of {wl1} - {wl2} nm")
ax.grid()
ax.legend()
plt.tight_layout()
# plt.show()
plt.savefig(f"{plot_path}/SMART_calib_factors_{prop}.png", dpi=100)
plt.close()

# %% investigate the last four days in more detail because they look wrong, plot calibration files
mpl.rcdefaults()  # set default plotting options
flight = "Flight_20210719a"  # "Flight_20210721a"  # "Flight_20210721b" "Flight_20210723a" "Flight_20210728a" "Flight_20210729a"
transfer_cali_date = transfer_calibs[flight]
instrument = lookup[prop][:5]
calibration = f"{instrument}_transfer_calib_{transfer_cali_date}"
trans_calib_path = f"{calib_path}/{calibration}/Tint_300ms"
trans_calib_path_dark = f"{calib_path}/{calibration}/dark_300ms"
trans_calib_files = [f for f in os.listdir(trans_calib_path) if prop in f]
trans_calib_files_dark = [f for f in os.listdir(trans_calib_path_dark) if prop in f]
plot_paths = [plot_path, f"{plot_path}/dark"]
for path, filenames, p_path in zip([trans_calib_path, trans_calib_path_dark],
                                   [trans_calib_files, trans_calib_files_dark], plot_paths):
    for filename in filenames:
        log.info(f"Plotting {path}/{filename}")
        plot_smart_data(flight, filename, wavelength="all", path=path, plot_path=p_path, save_fig=True)

# %% plot mean dark current for SWIR over flight; read in all raw files
flight = "Flight_20210728a"
props = ["Fdw_SWIR", "Fup_SWIR"]
dfs, dfs_plot, files_dict = dict(), dict(), dict()
raw_path = get_path("raw", flight)
bahamas_path = get_path("bahamas", flight)
bahamas_file = [f for f in os.listdir(bahamas_path) if f.endswith(".nc")][0]
bahamas_ds = read_bahamas(f"{bahamas_path}/{bahamas_file}")
for prop in props:
    files_dict[prop] = [f for f in os.listdir(raw_path) if prop in f]
    dfs[prop] = pd.concat([read_smart_raw(raw_path, file) for file in files_dict[prop]])
    # select only rows where the shutter is closed and take mean over all pixels
    dfs_plot[prop] = dfs[prop][dfs[prop]["shutter"] == 0].iloc[:, 2:].mean(axis=1)

# plot mean dark current over flight
set_cb_friendly_colors()
fig, axs = plt.subplots(nrows=2, sharex="all", figsize=(10, 6))
for prop in props:
    dfs_plot[prop].plot(ax=axs[0], ylabel="Netto Counts", label=f"{prop}")
bahamas_ds["IRS_ALT_km"] = bahamas_ds["IRS_ALT"] / 1000
bahamas_ds["IRS_ALT_km"].plot(ax=axs[1], label="BAHAMAS Altitude", color="#DDCC77")
axs[0].set_ylim((1500, 4000))
axs[1].set_ylabel("Altitude (km)")
axs[1].set_xlabel("Time (UTC)")
for ax in axs:
    ax.legend()
    ax.grid()
fig.suptitle(f"{flight} - Mean Dark Current")
# plt.show()
plt.savefig(f"{plot_path}/{flight}_SWIR_mean_dark_current.png", dpi=100)
plt.close()

# %% plot mean dark current for VNIR over flight; read in all raw files
flight = "Flight_20210723a"
props = ["Fdw_VNIR", "Fup_VNIR"]
dfs, dfs_plot, files_dict = dict(), dict(), dict()
raw_path = get_path("raw", flight)
bahamas_path = get_path("bahamas", flight)
bahamas_file = [f for f in os.listdir(bahamas_path) if f.endswith(".nc")][0]
bahamas_ds = read_bahamas(f"{bahamas_path}/{bahamas_file}")
for prop in props:
    files_dict[prop] = [f for f in os.listdir(raw_path) if prop in f]
    dfs[prop] = pd.concat([read_smart_raw(raw_path, file) for file in files_dict[prop]])
    # select only columns where no signal is measured in the VNIR, drop t_int and shutter column
    dfs_plot[prop] = dfs[prop].iloc[:, 2:150].mean(axis=1)

# plot mean dark current over flight VNIR
set_cb_friendly_colors()
fig, axs = plt.subplots(nrows=2, sharex="all", figsize=(10, 6))
for prop in props:
    dfs_plot[prop].plot(ax=axs[0], ylabel="Netto Counts", label=f"{prop}")
bahamas_ds["IRS_ALT_km"] = bahamas_ds["IRS_ALT"] / 1000
bahamas_ds["IRS_ALT_km"].plot(ax=axs[1], label="BAHAMAS Altitude", color="#DDCC77")
axs[0].set_ylim((90, 230))
axs[1].set_ylabel("Altitude (km)")
axs[1].set_xlabel("Time (UTC)")
for ax in axs:
    ax.legend()
    ax.grid()
fig.suptitle(f"{flight} - Mean Dark Current")
# plt.show()
plt.savefig(f"{plot_path}/{flight}_VNIR_mean_dark_current.png", dpi=100)
plt.close()
