#!/usr/bin/env python

import smart
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# absolute calibration of measurement files

plot_path = smart.get_path("plot")
# read in lamp file
lamp_path = smart.get_path("lamp")
lamp_file = "F1587i01_19.std"
names = ["Irradiance"]
lamp = pd.read_csv(os.path.join(lamp_path, lamp_file), skiprows=1, header=None, names=names)
lamp["Wavelength"] = np.arange(250, 2501)
# convert from W/cm^2 to W/m^2; cm = m * 10^-2 => cm^2 = (m * 10^-2)^2 = m^2 * 10^-4 => W*10^4/m^2
lamp["Irradiance"] = lamp["Irradiance"] * 10000
# plot lamp calibration
lamp.plot(x="Wavelength", y="Irradiance", ylabel="Irradiance $(W\\,m^{-2})$", xlabel="Wavelenght (nm)",
          legend=False, title="1000W Lamp F-1587 interpolated on 1nm steps")
plt.grid()
# plt.savefig(f"{plot_path}/1000W_Lamp_F1587_1nm_19.png", dpi=100)
plt.show()
plt.close()

# TODO: vary channel on ASP06
# TODO: functionise tranfer calib read in
# TODO: add plots to folder
# TODO: functionise read in functions
# TODO: save one file with pixel, c_field, etc in calib for each spectrometer

# read in ASP06 dark current corrected lamp measurement data and relate pixel to wavelength
base_dir, pixel_path, calib_path, data_path, _ = smart.set_paths()
filename = "2021_03_29_11_15.Fdw_VNIR_cor.dat"
date_str, channel, direction = smart.get_info_from_filename(filename)
lab_calib = smart.read_smart_cor(f"{calib_path}/ASP_06_Calib_Lab_20210329/calib_J3_4", filename)
# read in pixel to wavelength file
pixel_wl = smart.read_pixel_to_wavelength(pixel_path, smart.lookup[f"{direction}_{channel}"])
# set negative counts to 0
lab_calib[lab_calib.values < 0] = 0
# take mean over time
pixel_wl["S0"] = lab_calib.mean().reset_index(drop=True)
# interpolate irradiance on pixel wavelength
func = interp1d(lamp["Wavelength"], lamp["Irradiance"], fill_value="extrapolate")
pixel_wl["F0"] = func(pixel_wl["wavelength"])
# calculate calibration factor
pixel_wl["c_lab"] = pixel_wl["F0"] / pixel_wl["S0"]
# plot stuff
fig, ax = plt.subplots()
ax.plot(pixel_wl["wavelength"], pixel_wl["F0"], color="orange", label="Irradiance")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Irradiance $(W\\,m^{-2})$")
ax2 = ax.twinx()
ax2.plot(pixel_wl["wavelength"], pixel_wl["S0"], label="counts")
ax2.set_ylabel("Counts")
ax.legend()
ax2.legend()
plt.grid()
plt.show()
plt.close()

# read in Ulli transfer measurement from lab
ulli_file = "2021_03_29_11_17.Fdw_VNIR_cor.dat"
ulli = smart.read_smart_cor(f"{calib_path}/ASP_06_Calib_Lab_20210329/Ulli_trans_J3_4", ulli_file)
# set negative counts to 0
ulli[ulli.values < 0] = 0
# take mean
pixel_wl["S_ulli"] = ulli.mean().reset_index(drop=True)
pixel_wl["F_ulli"] = pixel_wl["S_ulli"] * pixel_wl["c_lab"]

# read in Ulli transfer measurement from field
field_file = "2021_06_04_13_40.Fdw_VNIR_cor.dat"
ulli_field = smart.read_smart_cor(f"{calib_path}/ASP_06_transfer_calib_20210604/Tint_500ms", field_file)
# set negative counts to 0
ulli_field[ulli_field.values < 0] = 0
# take mean
pixel_wl["S_ulli_field"] = ulli_field.mean().reset_index(drop=True).apply(np.floor)
pixel_wl["c_field"] = pixel_wl["F_ulli"] / pixel_wl["S_ulli_field"]

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
