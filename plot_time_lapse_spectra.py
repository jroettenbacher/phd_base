#!/usr/bin/env python
"""Script to plot SMART spectra to each GoPro picture
author: Johannes Röttenbacher
"""

# %% import modules
import os
import logging
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import smart
from smart import lookup
from functions_jr import make_dir
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.WARNING)

# %% select flight and set paths
date = 20210707
flight = f"Flight_{date}a"
gopro_path = smart.get_path("gopro")
# read the GoPro csv
gopro_times = pd.read_csv(f"{gopro_path}/{flight}_timestamps_sel.csv", index_col="datetime", parse_dates=True)

# %% read SMART data
calib_path = smart.get_path("calibrated")
smart_path = f"{calib_path}/{flight}"
fdw_file = [f for f in os.listdir(smart_path) if "Fdw_VNIR" in f][0]
fup_file = [f for f in os.listdir(smart_path) if "Fup_VNIR" in f][0]
fdw = smart.read_smart_cor(smart_path, fdw_file)
fup = smart.read_smart_cor(smart_path, fup_file)

# %% get pixel to wavelength file for x-axis
pixel_wl_path = smart.get_path("pixel_wl")
channel = "VNIR"
direction = "Fdw"
pixel_wl = smart.read_pixel_to_wavelength(pixel_wl_path, lookup[f"{direction}_{channel}"])
wavelengths = pixel_wl["wavelength"]

# %% select the closest SMART timestamps to the picture time
idxs = [fdw.index.get_loc(value, method="nearest") for value in gopro_times.index.values]
fdw_sel = fdw.iloc[idxs, :].set_index(gopro_times.index)
idxs = [fup.index.get_loc(value, method="nearest") for value in gopro_times.index.values]
fup_sel = fup.iloc[idxs, :].set_index(gopro_times.index)

# %% set values < 0 to 0
fdw_sel[fdw_sel < 0] = 0
fup_sel[fup_sel < 0] = 0

# %% calculate albedo
albedo = fup_sel.divide(fdw_sel)

# %% plot spectral albedo, fdw and fup
plot_path = f"{smart.get_path('plot')}/time_lapse/{flight}"
make_dir(plot_path)


def plot_timelapse_spectra(wavelengths: list, albedo: pd.DataFrame, fdw: pd.DataFrame, fup: pd.DataFrame,
                           gopro_times: pd.DataFrame, idx: int, channel: str, plot_path: str):
    """
    Plot the upward and downward irradiance together with the albedo
    Args:
        wavelengths: array with wavelengths corresponding to each pixel of the spectrometer
        albedo: calculated albedo
        fdw: calibrated downward irradiance
        fup: calibrated upward irradiance
        gopro_times: data frame with gopro times and picture numbers used in plot_maps.py
        idx: which timestep (index) to plot
        channel: VNIR or SWIR (only for plot title at the moment, should come from script above)
        plot_path: where to save figure

    Returns: Saves a figure

    """
    number = gopro_times.number[idx]  # get image number for outfile name
    timestep = fdw_sel.index[idx]  # get timestep to select corresponding spectra
    # set plot properties
    font = {'weight': 'bold', 'size': 26}
    matplotlib.rc('font', **font)

    fig, ax = plt.subplots(figsize=(13, 6))
    l_albedo = ax.plot(wavelengths, albedo.loc[timestep], label="Albedo", c="k", linewidth=5)  # plot albedo
    ax.set_ylabel("Albedo")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylim((0, 1))
    ax2 = ax.twinx()
    # plot irradiance
    l_fdw = ax2.plot(wavelengths, fdw.loc[timestep], label="Fdw", linewidth=5)
    l_fup = ax2.plot(wavelengths, fup.loc[timestep], label="Fup", linewidth=5)
    ax2.set_ylabel("Irradiance (W$\\,$m$^{-2}\\,$nm$^{-1}$)")
    ax2.set_ylim((-0.1, 2.5))
    # create legend
    lns = l_albedo + l_fdw + l_fup
    labs = [line.get_label() for line in lns]
    ax.legend(lns, labs, loc=2)
    ax.grid()
    plt.title(f"{channel} Spectrum and Albedo for {timestep}\nDark Current Corrected and Calibrated")
    timestep_name = timestep.strftime("%Y%m%d")
    fig_name = f"{plot_path}/{timestep_name}_{channel}_spectrum_albedo_{number:04}.png"
    plt.savefig(fig_name, dpi=100, bbox_inches="tight")
    log.info(f"Saved {fig_name}")
    plt.close()


# plot_timelapse_spectra(wavelengths, albedo, fdw_sel, fup_sel, gopro_times, 0, channel, plot_path)
Parallel(n_jobs=cpu_count()-2)(delayed(plot_timelapse_spectra)
                               (wavelengths, albedo, fdw_sel, fup_sel, gopro_times, idx, channel, plot_path)
                               for idx in tqdm(range(len(gopro_times))))

# %% plot timeseries with moving marker
plot_path = f"{smart.get_path('plot')}/time_lapse_ts/{flight}"
make_dir(plot_path)
wl = 550
nr, wl = smart.find_pixel(pixel_wl, wl)
number = gopro_times.number[idx]  # get image number for outfile name
# set plot properties
font = {'weight': 'bold', 'size': 26}
matplotlib.rc('font', **font)

fig, ax = plt.subplots(figsize=(13, 6))
l_albedo = ax.plot(albedo.iloc[:, nr], label="Albedo", c="k", linewidth=5)  # plot albedo
ax.set_ylabel("Albedo")
ax.set_xlabel("Time (UTC)")
ax.set_ylim((0, 1))
ax2 = ax.twinx()
# plot irradiance
l_fdw = ax2.plot(fdw.iloc[:, nr], label="Fdw", linewidth=5)
l_fup = ax2.plot(fup.iloc[:, nr], label="Fup", linewidth=5)
ax2.set_ylabel("Irradiance (W$\\,$m$^{-2}\\,$nm$^{-1}$)")
ax2.set_ylim((-0.1, 2.5))
# create legend
lns = l_albedo + l_fdw + l_fup
labs = [line.get_label() for line in lns]
ax.legend(lns, labs, loc=2)
ax.grid()
plt.title(f"{channel} Spectrum and Albedo for {timestep}\nDark Current Corrected and Calibrated")
timestep_name = timestep.strftime("%Y%m%d")
fig_name = f"{plot_path}/{timestep_name}_{channel}_spectrum_albedo_{number:04}.png"
plt.savefig(fig_name, dpi=100, bbox_inches="tight")
log.info(f"Saved {fig_name}")
plt.close()
