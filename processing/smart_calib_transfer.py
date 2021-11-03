#!/usr/bin/env python
"""Script to calculate field calibration factor and save a file for each PC and channel
1. set user defined variables and list field calibration files and lab calibration files
2. for each field cali file:
    2.1 read field file and corresponding lab calibration file
    2.2 calculate field calibration factor and the relation between lab and field counts of the ulli sphere
    2.3 plot transfer calib measurements and save the plot
    2.4 save transfer calibration file in calib folder

author: Johannes Roettenbacher
"""

# %% module import and set paths
import pylim.helpers as h
from pylim import reader, smart
from pylim.cirrus_hl import lookup
import os
import pandas as pd
import matplotlib.pyplot as plt

# %% set variables
field_folder = "ASP06_transfer_calib_20210729"  # transfer calib folder
lab_cali_date = "2021_03_29"  # set lab calib to relate measurement to
calib_path = h.get_path("calib")
plot_path = h.get_path("plot")
t_int = 300  # integration time of transfer calibration measurement
normalize = True  # normalize counts by integration time
norm = "_norm" if normalize else ""
# # loop through all transfer calib folders
# field_folders = [d for d in next(os.walk(calib_path))[1] if "transfer_calib" in d]
# for field_folder in field_folders[20:]:
# list transfer calibration files
field_cali_files = [f for f in os.listdir(f"{calib_path}/{field_folder}/Tint_{t_int}ms") if f.endswith("cor.dat")]
# list lab calibration files for selecting the right one
lab_cali_files = [f for f in os.listdir(calib_path) if f.endswith(f"lab_calib{norm}.dat") and lab_cali_date in f]

# %% read in Ulli transfer measurement from field
for field_file in field_cali_files:
    date_str, channel, direction = smart.get_info_from_filename(field_file)
    ulli_field = reader.read_smart_cor(f"{calib_path}/{field_folder}/Tint_{t_int}ms", field_file)
    ulli_field[ulli_field.values < 0] = 0  # set negative counts to 0
    # read corresponding cali file
    lab_file = [f for f in lab_cali_files if f"{direction}_{channel}" in f][0]
    print(f"Lab cali file used: {lab_file}")
    lab_df = pd.read_csv(f"{calib_path}/{lab_file}")
    lab_df["S_ulli_field"] = ulli_field.mean().reset_index(drop=True)  # take mean over time of field calib measurement
    if normalize:
        lab_df["S_ulli_field"] = lab_df["S_ulli_field"] / t_int
        ylabel = "Normalized Counts"
    else:
        ylabel = "Counts"
    lab_df["c_field"] = lab_df["F_ulli"] / lab_df["S_ulli_field"]  # calculate field calibration factor
    # calculate relation between S_ulli_lab and S_ulli_field
    lab_df["rel_ulli"] = lab_df["S_ulli"] / lab_df["S_ulli_field"]

    # %% plot transfer calib measurement
    spectrometer = lookup[f'{direction}_{channel}']
    fig, ax = plt.subplots()
    ax.plot(lab_df["wavelength"], lab_df["S_ulli"], label="Counts in lab")
    ax.plot(lab_df["wavelength"], lab_df["S_ulli_field"], label="Counts in field")
    ax.set_title(f"Ulli Transfer Sphere Lab and Transfer Calibration {date_str.replace('_', '-')} \n"
                 f"{spectrometer} {direction} {channel}")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(ylabel)
    ax2 = ax.twinx()
    ax2.plot(lab_df["wavelength"], lab_df["rel_ulli"], label="Lab / Field", color="green")
    ax2.set_ylabel("$S_{ulli, lab} / S_{ulli, field}$")
    ax2.set_ylim((0, lab_df["rel_ulli"].median() + 3))
    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)
    ax.grid()
    plt.tight_layout()
    figname = f"{plot_path}/{date_str}_{spectrometer}_{direction}_{channel}_{t_int}ms_ulli_transfer_calib{norm}.png"
    plt.savefig(figname, dpi=100)
    plt.show()
    plt.close()
    print(f"Saved {figname}")

    # %% write output file
    outfile = f"{calib_path}/{date_str}_{spectrometer}_{direction}_{channel}_{t_int}ms_transfer_calib{norm}.dat"
    lab_df.to_csv(outfile, index=False)
    print(f"Saved {outfile}")
