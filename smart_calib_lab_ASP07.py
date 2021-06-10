#!/usr/bin/env python
"""Script to read in calibration files and calculate calibration factors for lab calibration of ASP07
1. read in 1000W lamp file, plot it and save to data file
2. set channel to work with
3. set which folder pair to work with
4. read in calibration lamp measurements
5. read in pixel to wavelength mapping and interpolate lamp output onto pixel/wavelength of spectrometer
6. plot lamp measurements
7. read in ulli sphere measurements
8. plot ulli measurements
9. write dat file with all information
author: Johannes Roettenbacher
"""
# %%
import smart
from smart import lookup
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# %% set paths
raw_path, pixel_path, calib_path, data_path, plot_path = smart.set_paths()
panel_path = smart.get_path("panel")
# %% read in lamp and panel files
lamp = smart.read_lamp_file(plot=False, save_file=False, save_fig=False)
columns = ["wavelength", "reflectance"]
panels = dict()
panels["VNIR"] = pd.read_csv(f"{panel_path}/panels_VIS_8_(ASP_07).dat", skiprows=15, usecols=[0, 3], names=columns,
                             header=None, sep="\s+")
panels["SWIR"] = pd.read_csv(f"{panel_path}/panels_PGS_4_(ASP_07).dat", skiprows=15, usecols=[0, 3], names=columns,
                             header=None, sep="\s+")
# %% read in ASP07 dark current corrected lamp measurement data and relate pixel to wavelength
channel = "SWIR"  # set channel to work on (VNIR or SWIR)
base = "ASP_07_Calib_Lab_20210318"
panel = panels[channel]
panel["pixel"] = np.flip(panel.index + 1)
folders = ["Cal_Plate", "Ulli_trans"]
dirpath = os.path.join(calib_path, base, folders[0])
dirpath_ulli = os.path.join(calib_path, base, folders[1])
lamp_measurement = [f for f in os.listdir(dirpath) if f.endswith(f"{channel}_cor.dat")]
ulli_measurement = [f for f in os.listdir(dirpath_ulli) if f.endswith(f"{channel}_cor.dat")]
filename = lamp_measurement[0]
date_str, channel, direction = smart.get_info_from_filename(filename)
lab_calib = smart.read_smart_cor(dirpath, filename)
# set negative counts to 0
lab_calib[lab_calib.values < 0] = 0

# %% add counts, lamp irradiance and lamp irradiance multiplied with panel reflectance to data frame
spectrometer = lookup[f"{direction}_{channel}"]
panel["S0"] = lab_calib.mean().reset_index(drop=True)  # take mean over time of calib measurement
# interpolate lamp irradiance on pixel wavelength
lamp_func = interp1d(lamp["Wavelength"], lamp["Irradiance"], fill_value="extrapolate")
panel["F0"] = lamp_func(panel["wavelength"])
panel["F_ref"] = panel["F0"] * panel["reflectance"]  # this is the actual measurable radiance
panel["c_lab"] = panel["F_ref"] / panel["S0"]  # calculate lab calibration factor W/m^2/sr/counts
panel[panel.values < 0] = 0  # set values < 0 to 0

# %% plot counts and radiance of lamp lab calibration
fig, ax = plt.subplots()
ax.plot(panel["wavelength"], panel["F_ref"], color="orange", label="Reflected Radiance")
ax.set_title(f"1000W Lamp and Reflectance Panel Laboratory Calibration \n"
             f"{date_str.replace('_', '-')} {spectrometer} {direction} {channel}")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Radiance (W$\\,$sr$^{-1}\\,$m$^{-2}$)")
ax2 = ax.twinx()
ax2.plot(panel["wavelength"], panel["S0"], label="Counts")
ax2.set_ylabel("Counts")
# ask matplotlib for the plotted objects and their labels
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)
ax.grid()
plt.tight_layout()
figname = f"{plot_path}/{date_str}_{spectrometer}_{direction}_{channel}_lamp_lab_calib.png"
plt.savefig(figname, dpi=100)
print(f"Saved {figname}")
plt.show()
plt.close()

# %% read in Ulli transfer measurement from lab
ulli_file = ulli_measurement[0]
ulli = smart.read_smart_cor(f"{calib_path}/{base}/{folders[1]}", ulli_file)
ulli[ulli.values < 0] = 0  # set negative counts to 0
panel["S_ulli"] = ulli.mean().reset_index(drop=True)  # take mean over time of calib measurement
panel["F_ulli"] = panel["S_ulli"] * panel["c_lab"]  # calculate radiance measured from Ulli

# %% plot Ulli transfer measurement from laboratory
fig, ax = plt.subplots()
ax.plot(panel["wavelength"], panel["F_ulli"], color="orange", label="Radiance")
ax.set_title(f"Ulli Transfer Sphere Laboratory Calibration\n{spectrometer} {direction} {channel}")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Radiance (W$\\,$sr$^{-1}\\,$m$^{-2}$)")
ax2 = ax.twinx()
ax2.plot(panel["wavelength"], panel["S_ulli"], label="Counts")
ax2.set_ylabel("Counts")
# ask matplotlib for the plotted objects and their labels
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)
ax.grid()
plt.tight_layout()
figname = f"{plot_path}/{date_str}_{spectrometer}_{direction}_{channel}_ulli_lab_calib.png"
plt.savefig(figname, dpi=100)
print(f"Saved {figname}")
plt.show()
plt.close()

# %% save lamp and panel and ulli measurement from lab to file
csvname = f"{calib_path}/{date_str}_{spectrometer}_{direction}_{channel}_lab_calib.dat"
panel.to_csv(csvname, index=False)
print(f"Saved {csvname}")
