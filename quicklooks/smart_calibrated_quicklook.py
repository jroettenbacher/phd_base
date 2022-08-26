#!/usr/bin/env python
"""Plot and save SMART quicklooks of dark current corrected and calibrated measurements for one flight
author: Johannes Roettenbacher
"""

# %% import modules
import pylim.helpers as h
from pylim import cirrus_hl as meta
from pylim import reader, smart, bahamas
import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
import cartopy
import cartopy.crs as ccrs

# %% set paths
campaign = "cirrus-hl"  # adjust bahamas filename when using for HALO-AC3
flight = "Flight_20210625a"
prop = "Fdw"
wavelengths = [422, 532, 648, 858, 1240, 1640]  # five wavelengths to plot individually
calibrated_path = h.get_path("calibrated", flight, campaign)  # path to calibrated nc files
plot_path = calibrated_path  # output path of plot
cm = 1 / 2.54  # conversion factor for centimeter to inch

# %% get metadata
flight_no = meta.flight_numbers[flight]

# %% read in calibrated files
file = f"CIRRUS-HL_HALO_SMART_{prop}_{flight[7:-1]}_{flight}_v1.0.nc"
filepath = os.path.join(calibrated_path, file)
ds = xr.open_dataset(filepath)
Fup_cor = ds.Fup_cor  # extract corrected Fup
Fup_cor_flat = ds.Fup_cor.values.flatten()  # flatten 2D array for statistics
time_range = pd.to_timedelta((Fup_cor.time[-1] - Fup_cor.time[0]).values)  # get time range for time axis formatting

# %% set plotting aesthetics
plt.rcdefaults()
font = {"size": 14, "family": "serif"}
plt.rc("font", **font)

# %% calculate statistics
Fmin, Fmax, Fmean, Fmedian, Fstd = Fup_cor.min(), Fup_cor.max(), Fup_cor.mean(), Fup_cor.median(), Fup_cor.std()
stats_text = f"Statistics (W$\,$m$^{{-2}}$)\nMin: {Fmin:.2f}\nMax: {Fmax:.2f}\nMean: {Fmean:.2f}\nMedian: {Fmedian:.2f}" \
             f"\nStd: {Fstd:.2f}"

# %% plot Fup Boxplot
fig, ax = plt.subplots(figsize=(7.5, 3))
ax.boxplot(Fup_cor_flat, vert=False, labels=[""], widths=0.9)
ax.set_xlabel("Irradiance (W$\,$m$^{-2}$)")
plt.tight_layout()
plt.show()
plt.close()

# %% plot wavelength-time plot
fig, ax = plt.subplots(figsize=(7.5, 7.5))
Fup_cor.plot(x="time", robust=True, cmap="inferno",
             cbar_kwargs={"location": "bottom", "label": "Irradiance (W$\,$m$^{-2}\,$nm$^{-1}$)", "pad": 0.11})
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Wavelength (nm)")
h.set_xticks_and_xlabels(ax, time_range)
plt.tight_layout()
plt.show()
plt.close()

# %% plot 5 wavelengths
h.set_cb_friendly_colors()
fig, ax = plt.subplots(figsize=(5, 7.5))
for wavelength in wavelengths:
    Fup_cor.sel(wavelength=wavelength).plot(y="time", label=f"{wavelength}$\,$nm")
ax.set_ylabel("Time (UTC)")
ax.set_xlabel("Irradiance (W$\,$m$^{-2}\,$nm$^{-1}$)")
ax.set_title("")
ax.grid()
h.set_yticks_and_ylabels(ax, time_range)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
plt.tight_layout()
plt.show()
plt.close()

# %% plot sza and saa
h.set_cb_friendly_colors()
fig, ax = plt.subplots(figsize=(7.5, 3))
ds.sza.plot(ax=ax, label="Solar Zenith Angle")
ax2 = ax.twinx()
ds.saa.plot(ax=ax2, label="Solar Azimuth Angle", color="#CC6677")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Solar zenith angle (°)")
ax2.set_ylabel("Solar azimuth angle (°)")
ax.grid()
h.set_xticks_and_xlabels(ax, time_range)
handles, _ = ax.get_legend_handles_labels()
handles2, _ = ax2.get_legend_handles_labels()
handles.append(handles2[0])
ax.legend(handles=handles)
ax2.spines["left"].set_color("#88CCEE")
ax2.spines["right"].set_color("#CC6677")
ax2.spines["left"].set_linewidth(3)
ax2.spines["right"].set_linewidth(3)
ax.tick_params(colors="#88CCEE", axis="y", which="both")
ax2.tick_params(colors="#CC6677", axis="y", which="both")
plt.tight_layout()
plt.show()
plt.close()

# %% Plot Fup overview plot
fig = plt.figure(figsize=(25 * cm, 30 * cm), constrained_layout=True)
gs0 = fig.add_gridspec(1, 2, width_ratios=[3, 1])
gs01 = gs0[0].subgridspec(3, 1, height_ratios=[1, 3, 1])
gs02 = gs0[1].subgridspec(3, 1, height_ratios=[1, 4, 1])

# boxplot
ax = fig.add_subplot(gs01[0])
ax.boxplot(Fup_cor_flat, vert=False, labels=[""], widths=0.9)
ax.set_xlabel("Irradiance (W$\,$m$^{-2}$)")
ax.set_ylabel("Boxplot of all\n values")
ax.grid()

# textbox with statistics
ax = fig.add_subplot(gs02[0])
ax.axis("off")  # hide axis
ax.text(-0.4, 0.9, stats_text, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)

# wavelength-time plot
ax = fig.add_subplot(gs01[1])
Fup_cor.plot(ax=ax, x="time", robust=True, cmap="inferno",
             cbar_kwargs={"location": "bottom", "label": "Irradiance (W$\,$m$^{-2}\,$nm$^{-1}$)", "pad": 0.01})
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Wavelength (nm)")
h.set_xticks_and_xlabels(ax, time_range)

# 5 wavelength plot
ax1 = fig.add_subplot(gs02[1])
for wavelength in wavelengths:
    Fup_cor.sel(wavelength=wavelength).plot(ax=ax1, y="time", label=f"{wavelength}$\,$nm")
ax1.set_ylabel("Time (UTC)")
ax1.set_xlabel("Irradiance \n(W$\,$m$^{-2}\,$nm$^{-1}$)")
ax1.set_title("")
ax1.grid()
ax1.invert_yaxis()
ax1.tick_params(axis="y", labelrotation=45)
h.set_yticks_and_ylabels(ax1, time_range)

# legend for 5 wavelength plit
ax = fig.add_subplot(gs02[2])
ax.axis("off")
handles, labels = ax1.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=[0.1, 1], ncol=1)

# sza and saa
ax = fig.add_subplot(gs01[2])
ds.sza.plot(ax=ax, label="Solar Zenith Angle")
ax2 = ax.twinx()
ds.saa.plot(ax=ax2, label="Solar Azimuth Angle", color="#CC6677")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("")
ax2.set_ylabel("")
ax.grid()
h.set_xticks_and_xlabels(ax, time_range)
handles, _ = ax.get_legend_handles_labels()
handles2, _ = ax2.get_legend_handles_labels()
handles.append(handles2[0])
ax.legend(handles=handles)
ax2.spines["left"].set_color("#88CCEE")
ax2.spines["right"].set_color("#CC6677")
ax2.spines["left"].set_linewidth(3)
ax2.spines["right"].set_linewidth(3)
ax.tick_params(colors="#88CCEE", axis="y", which="both")
ax2.tick_params(colors="#CC6677", axis="y", which="both", direction="in", pad=-30)

fig.suptitle(f"SMART upward Irradiance for {flight} - {flight_no}")
plt.savefig(f"{plot_path}/CIRRUS-HL_{flight_no}_SMART_calibrated-Fup_quicklook_{flight[7:-1]}.png", dpi=300)
plt.show()
plt.close()

# %% plot Fdw overview plot with stabbi angles
