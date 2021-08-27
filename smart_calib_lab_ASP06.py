#!/usr/bin/env python
"""Script to read in calibration files and calculate calibration factors for lab calibration of ASP06
1. read in 1000W lamp file, plot it and save to data file
2. set channel to work with
3. set which folder pair to work with
4. read in calibration lamp measurements
5. read in pixel to wavelength mapping and interpolate lamp output onto pixel/wavelength of spectrometer
6. plot lamp measurements
7. read in ulli sphere measurements
8. plot ulli measurements
9. write dat file with all information
author: Johannes Roettenbacher"""
# %%
import smart
from smart import lookup, get_path
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# %% set paths
pixel_path, calib_path, plot_path = get_path("pixel_wl"), get_path("calib"), get_path("plot")

# %% read lamp file
lamp = smart.read_lamp_file(plot=False, save_file=False, save_fig=False)

# %% read in ASP06 dark current corrected lamp measurement data and relate pixel to wavelength
channel = "SWIR"  # set channel to work on (VNIR or SWIR)
normalize = True  # normalize counts to integration time
t_int = 300  # integration time of calibration measurement
base = "ASP06_Calib_Lab_20210809"
folder_pairs = [["calib_J3_4", "Ulli_trans_J3_4"], ["calib_J5_6", "Ulli_trans_J5_6"]]
folders = folder_pairs[1]  # which folder pair to work on (0 or 1)
dirpath = os.path.join(calib_path, base, folders[0])
dirpath_ulli = os.path.join(calib_path, base, folders[1])
lamp_measurement = [f for f in os.listdir(dirpath) if f.endswith(f"{channel}_cor.dat")]
ulli_measurement = [f for f in os.listdir(dirpath_ulli) if f.endswith(f"{channel}_cor.dat")]
filename = lamp_measurement[0]
date_str, channel, direction = smart.get_info_from_filename(filename)
lab_calib = smart.read_smart_cor(dirpath, filename)
# set negative counts to 0
lab_calib[lab_calib.values < 0] = 0

# %% read in pixel to wavelength file
spectrometer = lookup[f"{direction}_{channel}"]
pixel_wl = smart.read_pixel_to_wavelength(pixel_path, spectrometer)
pixel_wl["S0"] = lab_calib.mean().reset_index(drop=True)  # take mean over time of calib measurement
if normalize:
    pixel_wl["S0"] = pixel_wl["S0"] / t_int  # normalize counts by integration time
    ylabel2, norm = "Normalized Counts", "_norm"
else:
    ylabel2, norm = "Counts", ""
# interpolate lamp irradiance on pixel wavelength
lamp_func = interp1d(lamp["Wavelength"], lamp["Irradiance"], fill_value="extrapolate")
pixel_wl["F0"] = lamp_func(pixel_wl["wavelength"])
pixel_wl["c_lab"] = pixel_wl["F0"] / pixel_wl["S0"]  # calculate lab calibration factor W/m^2/counts
pixel_wl[pixel_wl.values < 0] = 0  # set values < 0 to 0

# %% plot counts and irradiance of lamp lab calibration
fig, ax = plt.subplots()
ax.plot(pixel_wl["wavelength"], pixel_wl["F0"], color="orange", label="Irradiance")
ax.set_title(f"1000W Lamp Laboratory Calibration\n"
             f"{date_str.replace('_', '-')} {spectrometer} {direction} {channel}")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Irradiance (W$\\,$m$^{-2}$)")
ax2 = ax.twinx()
ax2.plot(pixel_wl["wavelength"], pixel_wl["S0"], label="counts")
ax2.set_ylabel(ylabel2)
# ask matplotlib for the plotted objects and their labels
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)
ax.grid()
plt.tight_layout()
figname = f"{plot_path}/{date_str}_{spectrometer}_{direction}_{channel}_lamp_lab_calib{norm}.png"
plt.savefig(figname, dpi=100)
print(f"Saved {figname}")
plt.show()
plt.close()

# %% read in Ulli transfer measurement from lab
ulli_file = ulli_measurement[0]
ulli = smart.read_smart_cor(f"{calib_path}/{base}/{folders[1]}", ulli_file)
ulli[ulli.values < 0] = 0  # set negative counts to 0
pixel_wl["S_ulli"] = ulli.mean().reset_index(drop=True)  # take mean over time of calib measurement
if normalize:
    pixel_wl["S_ulli"] = pixel_wl["S_ulli"] / t_int

pixel_wl["F_ulli"] = pixel_wl["S_ulli"] * pixel_wl["c_lab"]  # calculate irradiance measured from Ulli

# %% plot Ulli transfer measurement from laboratory
fig, ax = plt.subplots()
ax.plot(pixel_wl["wavelength"], pixel_wl["F_ulli"], color="orange", label="Irradiance")
ax.set_title(f"Ulli Transfer Sphere Laboratory Calibration \n{spectrometer} {direction} {channel}")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Irradiance (W$\\,$m$^{-2}$)")
ax2 = ax.twinx()
ax2.plot(pixel_wl["wavelength"], pixel_wl["S_ulli"], label="counts")
ax2.set_ylabel(ylabel2)
# ask matplotlib for the plotted objects and their labels
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)
ax.grid()
plt.tight_layout()
figname = f"{plot_path}/{date_str}_{spectrometer}_{direction}_{channel}_ulli_lab_calib{norm}.png"
plt.savefig(figname, dpi=100)
print(f"Saved {figname}")
plt.show()
plt.close()

# %% save lamp and ulli measurement from lab to file
csvname = f"{calib_path}/{date_str}_{spectrometer}_{direction}_{channel}_lab_calib{norm}.dat"
pixel_wl.to_csv(csvname, index=False)
print(f"Saved {csvname}")
