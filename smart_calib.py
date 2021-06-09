#!/usr/bin/env python
"""Script to read in calibration files and calculate calibration factors for lab calibration and transfer calibrations
1. read in 1000W lamp file, plot it and save to data file
2. set channel to work with
3. set which folder pair to work with
4. read in calibration lamp measurements
5. read in pixel to wavelength mapping and interpolate lamp output onto pixel/wavelength of spectrometer
6. plot lamp measurements
7. read in ulli sphere measurements
8. plot ulli measurements
9. write dat file with all information"""
# %%
# TODO: functionise tranfer calib read in
# TODO: add plots to folder
# TODO: functionise read in functions
# TODO: save one file with pixel, c_field, etc in calib for each spectrometer
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
lamp_path = smart.get_path("lamp")

# %% read in lamp file
lamp_file = "F1587i01_19.std"
names = ["Irradiance"]
lamp = pd.read_csv(os.path.join(lamp_path, lamp_file), skiprows=1, header=None, names=names)
lamp["Wavelength"] = np.arange(250, 2501)
# convert from W/cm^2 to W/m^2; cm = m * 10^-2 => cm^2 = (m * 10^-2)^2 = m^2 * 10^-4 => W*10^4/m^2
lamp["Irradiance"] = lamp["Irradiance"] * 10000

# %% plot lamp calibration
lamp.plot(x="Wavelength", y="Irradiance", ylabel="Irradiance $(W\\,m^{-2})$", xlabel="Wavelenght (nm)",
          legend=False, title="1000W Lamp F-1587 interpolated on 1nm steps")
plt.grid()
# plt.savefig(f"{plot_path}/1000W_Lamp_F1587_1nm_19.png", dpi=100)
plt.show()
plt.close()

# %% save lamp file in calib folder
lamp.to_csv(f"{calib_path}/1000W_lamp_F1587_1nm_19.dat", index=False)

# %% read in ASP06/ASP07 dark current corrected lamp measurement data and relate pixel to wavelength
channel = "SWIR"  # set channel to work on (VNIR or SWIR)
base = "ASP_06_Calib_Lab_20210329"
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
# interpolate lamp irradiance on pixel wavelength
lamp_func = interp1d(lamp["Wavelength"], lamp["Irradiance"], fill_value="extrapolate")
pixel_wl["F0"] = lamp_func(pixel_wl["wavelength"])
pixel_wl["c_lab"] = pixel_wl["F0"] / pixel_wl["S0"]  # calculate lab calibration factor
pixel_wl[pixel_wl.values < 0] = 0  # set values < 0 to 0

# %% plot counts and irradiance of lamp lab calibration
fig, ax = plt.subplots()
ax.plot(pixel_wl["wavelength"], pixel_wl["F0"], color="orange", label="Irradiance")
ax.set_title(f"1000W Lamp Laboratory Calibration of {spectrometer}")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Irradiance $(W\\,m^{-2})$")
ax2 = ax.twinx()
ax2.plot(pixel_wl["wavelength"], pixel_wl["S0"], label="counts")
ax2.set_ylabel("Counts")
# ask matplotlib for the plotted objects and their labels
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)
plt.grid()
plt.savefig(f"{plot_path}/{date_str}_{spectrometer}_{direction}_{channel}_lamp_lab_calib.png", dpi=100)
plt.show()
plt.close()

# %% read in Ulli transfer measurement from lab
ulli_file = ulli_measurement[0]
ulli = smart.read_smart_cor(f"{calib_path}/{base}/{folders[1]}", ulli_file)
ulli[ulli.values < 0] = 0  # set negative counts to 0
pixel_wl["S_ulli"] = ulli.mean().reset_index(drop=True)  # take mean over time of calib measurement
pixel_wl["F_ulli"] = pixel_wl["S_ulli"] * pixel_wl["c_lab"]  # calculate irradiance measured by Ulli

# %% plot Ulli transfer measurement from laboratory
fig, ax = plt.subplots()
ax.plot(pixel_wl["wavelength"], pixel_wl["F_ulli"], color="orange", label="Irradiance")
ax.set_title(f"Ulli Transfer Sphere Laboratory Calibration of {lookup[f'{direction}_{channel}']}")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Irradiance $(W\\,m^{-2})$")
ax2 = ax.twinx()
ax2.plot(pixel_wl["wavelength"], pixel_wl["S_ulli"], label="counts")
ax2.set_ylabel("Counts")
# ask matplotlib for the plotted objects and their labels
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)
plt.grid()
plt.savefig(f"{plot_path}/{date_str}_{spectrometer}_{direction}_{channel}_ulli_lab_calib.png", dpi=100)
plt.show()
plt.close()

# %% save lamp and ulli measurement from lab to file
pixel_wl.to_csv(f"{calib_path}/{date_str}_{spectrometer}_{direction}_{channel}_lab_calib.dat", index=False)

# %% read in Ulli transfer measurement from field
field_folder = "ASP_06_transfer_calib_20210604"
t_int = 500
field_file = "2021_06_04_13_40.Fdw_VNIR_cor.dat"
ulli_field = smart.read_smart_cor(f"{calib_path}/{field_folder}/Tint_{t_int}ms", field_file)
ulli_field[ulli_field.values < 0] = 0  # set negative counts to 0
pixel_wl["S_ulli_field"] = ulli_field.mean().reset_index(drop=True)  # take mean over time of calib measurement
pixel_wl["c_field"] = pixel_wl["F_ulli"] / pixel_wl["S_ulli_field"]  # calculate field calibration factor

# calculate relation between S_ulli_lab and S_ulli_field
pixel_wl["rel_ulli"] = pixel_wl["S_ulli"] / pixel_wl["S_ulli_field"]

# read in dark current corrected measurement files
measurement = smart.read_smart_cor(f"{data_path}/flight_00", "2021_03_29_11_15.Fdw_VNIR_cor.dat")
# set negative values to 0
measurement[measurement.values < 0] = 0
# convert to long format
m_long = measurement.melt(var_name="pixel", value_name="counts", ignore_index=False)
df = m_long.merge(pixel_wl.loc[:, ["pixel", "c_field"]], on="pixel", right_index=True)
df["F_x"] = df["counts"] * df["c_field"]
df.pivot(columns="pixel", values="F_x")
