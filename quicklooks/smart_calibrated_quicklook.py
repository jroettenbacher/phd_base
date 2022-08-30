#!/usr/bin/env python
"""Plot and save SMART quicklooks of dark current corrected and calibrated measurements for one flight
author: Johannes Roettenbacher
"""

# %% import modules
import pylim.helpers as h
from pylim import cirrus_hl as meta
from pylim import bahamas
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
file = f"CIRRUS-HL_{flight_no}_{flight[7:]}_HALO_SMART_spectral-irradiance-{prop}_v1.0.nc"
filepath = os.path.join(calibrated_path, file)
ds = xr.open_dataset(filepath)
F_cor = ds[f"{prop}_cor"]  # extract corrected F
F_cor_flat = F_cor.values.flatten()  # flatten 2D array for statistics
time_range = pd.to_timedelta((F_cor.time[-1] - F_cor.time[0]).values)  # get time range for time axis formatting

# %% set plotting aesthetics
plt.rcdefaults()
h.set_cb_friendly_colors()
font = {"size": 14, "family": "serif"}
plt.rc("font", **font)

# %% calculate statistics
Fmin, Fmax, Fmean, Fmedian, Fstd = F_cor.min(), F_cor.max(), F_cor.mean(), F_cor.median(), F_cor.std()
stats_text = f"Statistics \n(W$\,$m$^{{-2}}$)\nMax: {Fmax:.2f}\nMean: {Fmean:.2f}\nMedian: {Fmedian:.2f}" \
             f"\nStd: {Fstd:.2f}"

# %% Plot Fup overview plot
if prop == "Fup":
    fig = plt.figure(figsize=(25 * cm, 40 * cm), constrained_layout=True)
    gs0 = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[3, 1])
    gs01 = gs0[0, 0].subgridspec(3, 1, height_ratios=[1, 3, 1])
    gs02 = gs0[0, 1].subgridspec(3, 1, height_ratios=[1, 4, 1])

    # 5 wavelength plot - first subrow, first column
    ax1 = fig.add_subplot(gs01[0])
    for wavelength in wavelengths:
        F_cor.sel(wavelength=wavelength).plot(ax=ax1, x="time", label=f"{wavelength}$\,$nm")
    ax1.set_xlabel("")
    ax1.set_ylabel("Irradiance \n(W$\,$m$^{-2}\,$nm$^{-1}$)")
    ax1.set_title("")
    ax1.grid()
    h.set_xticks_and_xlabels(ax1, time_range)

    # wavelength-time plot - second subrow, first column
    ax = fig.add_subplot(gs01[1])
    F_cor.plot(ax=ax, x="time", robust=True, cmap="inferno",
                 cbar_kwargs={"location": "bottom", "label": "Irradiance (W$\,$m$^{-2}\,$nm$^{-1}$)", "pad": 0.01})
    ax.set_xlabel("")
    ax.set_ylabel("Wavelength (nm)")
    h.set_xticks_and_xlabels(ax, time_range)

    # sza and saa - third subrow, first column
    ax = fig.add_subplot(gs01[2])
    ds.sza.plot(ax=ax, label="SZA")
    ax2 = ax.twinx()
    ds.saa.plot(ax=ax2, label="SAA", color="#CC6677")
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

    # legend for 5 wavelength plot - first subrow, second column
    ax = fig.add_subplot(gs02[0])
    ax.axis("off")
    handles, labels = ax1.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=[0.2, 1.1], ncol=1)

    # boxplot - second subrow, second column
    ax = fig.add_subplot(gs02[1])
    ax.boxplot(F_cor_flat, vert=True, labels=[""], widths=0.7)
    ax.set_ylabel("Irradiance (W$\,$m$^{-2}$)")
    ax.grid()

    # textbox with statistics - third subrow, second column
    ax = fig.add_subplot(gs02[2])
    ax.axis("off")  # hide axis
    ax.text(-0.2, 0.9, stats_text, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)

    # map of flight track - second row, both columns
    data_crs = ccrs.PlateCarree()
    props = bahamas.plot_props[flight]  # get plot properties
    x_edmo, y_edmo = meta.coordinates["EDMO"]
    airport = meta.stop_over_locations[flight] if flight in meta.stop_over_locations else None
    # get extent of map plot
    pad = 2
    llcrnlat = ds.lat.min(skipna=True) - pad
    llcrnlon = ds.lon.min(skipna=True) - pad
    urcrnlat = ds.lat.max(skipna=True) + pad
    urcrnlon = ds.lon.max(skipna=True) + pad
    extent = [llcrnlon, urcrnlon, llcrnlat, urcrnlat]
    ax = fig.add_subplot(gs0[1, :], projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS)
    ax.set_extent(extent, crs=data_crs)
    gl = ax.gridlines(crs=data_crs, draw_labels=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    # gl.left_labels = False
    # plot flight track
    points = ax.scatter(ds["lon"], ds["lat"], s=2, c="#6699CC", transform=data_crs)
    # add point for EDMO and optional second airport
    ax.plot(x_edmo, y_edmo, 'ok')
    ax.text(x_edmo + 0.1, y_edmo + 0.1, "EDMO", fontsize=8,
            path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])
    if airport is not None:
        x2, y2 = meta.coordinates[airport]
        ax.plot(x2, y2, 'ok')
        ax.text(x2 + 0.1, y2 + 0.1, airport, fontsize=8,
                path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])

    fig.suptitle(f"SMART upward Irradiance for {flight} - {flight_no}")
    plt.savefig(f"{plot_path}/CIRRUS-HL_{flight_no}_SMART_calibrated-Fup_quicklook_{flight[7:-1]}.png", dpi=300)
    plt.show()
    plt.close()

# %% plot Fdw overview plot with role angles and stabbi flag
if prop == "Fdw":
    fig = plt.figure(figsize=(25 * cm, 45 * cm), constrained_layout=True)
    gs0 = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[3, 1])
    gs01 = gs0[0, 0].subgridspec(4, 1, height_ratios=[1, 3, 1, 1])
    gs02 = gs0[0, 1].subgridspec(4, 1, height_ratios=[1, 4, 1, 1])

    # 5 wavelength plot - first subrow, first column
    ax1 = fig.add_subplot(gs01[0])
    for wavelength in wavelengths:
        F_cor.sel(wavelength=wavelength).plot(ax=ax1, x="time", label=f"{wavelength}$\,$nm")
    ax1.set_xlabel("")
    ax1.set_ylabel("Irradiance \n(W$\,$m$^{-2}\,$nm$^{-1}$)")
    ax1.set_title("")
    ax1.grid()
    h.set_xticks_and_xlabels(ax1, time_range)

    # wavelength-time plot - second subrow, first column
    ax = fig.add_subplot(gs01[1])
    F_cor.plot(ax=ax, x="time", robust=True, cmap="inferno",
                 cbar_kwargs={"location": "bottom", "label": "Irradiance (W$\,$m$^{-2}\,$nm$^{-1}$)", "pad": 0.01})
    ax.set_xlabel("")
    ax.set_ylabel("Wavelength (nm)")
    h.set_xticks_and_xlabels(ax, time_range)

    # sza and saa - third subrow, first column
    ax = fig.add_subplot(gs01[2])
    ds.sza.plot(ax=ax, label="SZA")
    ax2 = ax.twinx()
    ds.saa.plot(ax=ax2, label="SAA", color="#CC6677")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax2.set_ylabel("")
    ax.grid()
    h.set_xticks_and_xlabels(ax, time_range)
    # add legend
    handles, _ = ax.get_legend_handles_labels()
    handles2, _ = ax2.get_legend_handles_labels()
    handles.append(handles2[0])
    ax.legend(handles=handles)
    # color y axis
    ax2.spines["left"].set_color("#88CCEE")
    ax2.spines["right"].set_color("#CC6677")
    ax2.spines["left"].set_linewidth(3)
    ax2.spines["right"].set_linewidth(3)
    ax.tick_params(colors="#88CCEE", axis="y", which="both")
    ax2.tick_params(colors="#CC6677", axis="y", which="both", direction="in", pad=-30)

    # roll angle with stabbi flag - fourth subrow, first column
    ax = fig.add_subplot(gs01[3])
    stabbi = ds.stabilization_flag
    stabbi_working = stabbi.where(stabbi == 0, drop=True) + 2
    stabbi_not_working = stabbi.where(stabbi == 1, drop=True) - 1.05 + 2
    ds["roll"].plot(ax=ax, x="time", color="#999933")
    stabbi_working.plot(ls="", marker="s", markersize=2, label="working", color="#44AA99")
    stabbi_not_working.plot(ls="", marker="s", markersize=2, label="not working", color="#882255")
    h.set_xticks_and_xlabels(ax, time_range)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Roll Angle (°)")
    ax.grid()
    ax.set_ylim([-2, 2.02])
    handles_roll, labels_roll = ax.get_legend_handles_labels()

    # legend for 5 wavelength plot - first subrow, second column
    ax = fig.add_subplot(gs02[0])
    ax.axis("off")
    handles, labels = ax1.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=[0.2, 1.1], ncol=1)

    # boxplot - second subrow, second column
    ax = fig.add_subplot(gs02[1])
    ax.boxplot(F_cor_flat, vert=True, labels=[""], widths=0.7)
    ax.set_ylabel("Irradiance (W$\,$m$^{-2}$)")
    ax.grid()

    # textbox with statistics - third subrow, second column
    ax = fig.add_subplot(gs02[2])
    ax.axis("off")  # hide axis
    ax.text(-0.2, 0.9, stats_text, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)

    # legend for roll angle and Stabbi
    ax = fig.add_subplot(gs02[3])
    ax.axis("off")
    ax.legend(handles=handles_roll, labels=labels_roll, markerscale=6, title="Stabilization", loc='upper center',
              bbox_to_anchor=[0.2, 1.1])
    # map of flight track - second row, both columns
    data_crs = ccrs.PlateCarree()
    props = bahamas.plot_props[flight]  # get plot properties
    x_edmo, y_edmo = meta.coordinates["EDMO"]
    airport = meta.stop_over_locations[flight] if flight in meta.stop_over_locations else None
    # get extent of map plot
    pad = 2
    llcrnlat = ds.lat.min(skipna=True) - pad
    llcrnlon = ds.lon.min(skipna=True) - pad
    urcrnlat = ds.lat.max(skipna=True) + pad
    urcrnlon = ds.lon.max(skipna=True) + pad
    extent = [llcrnlon, urcrnlon, llcrnlat, urcrnlat]
    ax = fig.add_subplot(gs0[1, :], projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS)
    ax.set_extent(extent, crs=data_crs)
    gl = ax.gridlines(crs=data_crs, draw_labels=True, x_inline=False, y_inline=False)
    gl.top_labels = False
    # gl.left_labels = False
    # plot flight track
    points = ax.scatter(ds["lon"], ds["lat"], s=2, c="#6699CC", transform=data_crs)
    # add point for EDMO and optional second airport
    ax.plot(x_edmo, y_edmo, 'ok')
    ax.text(x_edmo + 0.1, y_edmo + 0.1, "EDMO", fontsize=8,
            path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])
    if airport is not None:
        x2, y2 = meta.coordinates[airport]
        ax.plot(x2, y2, 'ok')
        ax.text(x2 + 0.1, y2 + 0.1, airport, fontsize=8,
                path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])

    fig.suptitle(f"SMART downward Irradiance for {flight} - {flight_no}")
    plt.savefig(f"{plot_path}/CIRRUS-HL_{flight_no}_SMART_calibrated-Fdw_quicklook_{flight[7:-1]}.png", dpi=300)
    plt.show()
    plt.close()
