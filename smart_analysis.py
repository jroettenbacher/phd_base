#!/usr/bin/env python
"""Script to do some analysis on SMART data
author: Johannes Röttenbacher
"""

# %% import libraries
import smart
from smart import lookup
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% list files
flight = "flight_03"
calibrated_path = smart.get_path("calibrated")
pixel_wl_path = smart.get_path("pixel_wl")
inpath = os.path.join(calibrated_path, flight)
all_files = os.listdir(inpath)
fdw_files = [f for f in all_files if "Fdw" in f]
fup_files = [f for f in all_files if "Fup" in f]

# %% get pixel to wavelength file
channel = "SWIR"
direction = "Fdw"
pixel_wl = smart.read_pixel_to_wavelength(pixel_wl_path, lookup[f"{direction}_{channel}"])
# %% read in files
fdw_file = fdw_files[0] if channel in fdw_files[0] else fdw_files[1]
fup_file = fup_files[0] if channel in fup_files[0] else fup_files[1]
fdw = smart.read_smart_cor(inpath, fdw_file)
fup = smart.read_smart_cor(inpath, fup_file)

# %% remove values < 0
fdw_clean = fdw[fdw > 0]
fup_clean = fup[fup > 0]

# %% calculate albedo
albedo = fup_clean / fdw_clean

# %% select time range and wavelength
begin = "2021-06-26 8:15"
end = "2021-06-26 14:15"
wavelength = 1200
albedo_sel = albedo[begin:end]
fdw_sel = fdw_clean[begin:end]
fup_sel = fup_clean[begin:end]
pixel_nr, wl = smart.find_pixel(pixel_wl, wavelength)
# %% plot spectral albedo time series
fig, ax = plt.subplots()
albedo_sel.plot(y=pixel_nr, ax=ax, title=f"Spectral Albedo (Fup / Fdw) at {wl} nm", label="Albedo", c="g")
ax.set_ylabel("Spectral Albedo")
ax.set_xlabel("Time (UTC)")
ax2 = ax.twinx()
fdw_sel.plot(y=pixel_nr, ax=ax2, label="Fdw")
fup_sel.plot(y=pixel_nr, ax=ax2, label="Fup")
ax2.set_ylabel("Irradiance (W$\\,$m$^{-2}\\,$nm$^{-1}$)")
plt.grid()
plt.show()
plt.close()

# %% plot average albedo
albedo_avg = albedo_sel.median(axis=1)
fdw_avg = fdw_sel.median(axis=1)
fup_avg = fup_sel.median(axis=1)
fig, ax = plt.subplots()
albedo_avg.plot(ax=ax, title=f"{channel} Median Albedo (Fup / Fdw)", label="Albedo", c="g")
ax.set_ylabel("Albedo")
ax.set_xlabel("Time (UTC)")
ax.legend()
ax2 = ax.twinx()
fup_avg.plot(ax=ax2, label="Fup")
fdw_avg.plot(ax=ax2, label="Fdw")
ax2.set_ylabel("Irradiance (W$\\,$m$^{-2}\\,$nm$^{-1}$)")
ax2.legend()
plt.grid()
plt.show()
plt.close()

# %% plot spectral albedo
timestep = "2021-06-26 12:00:58.84"
wavelengths = pixel_wl["wavelength"]
fig, ax = plt.subplots()
ax.plot(wavelengths, albedo.loc[timestep], label="Albedo", c="g")
ax.set_ylabel("Albedo")
ax.set_xlabel("Wavelength (nm)")
ax.legend()
ax2 = ax.twinx()
ax2.plot(wavelengths, fdw.loc[timestep], label="Fdw")
ax2.plot(wavelengths, fup.loc[timestep], label="Fup")
ax2.set_ylabel("Irradiance (W$\\,$m$^{-2}\\,$nm$^{-1}$)")
ax2.legend()
ax.grid()
plt.show()
plt.close()
