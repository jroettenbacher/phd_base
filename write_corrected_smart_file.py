#!/usr/bin/env python
"""Script to correct SMART measurement for dark current and save it to a new file
input: raw smart measurements
output: corrected smart measurements
author: Johannes Roettenbacher
"""
import os
import smart
from functions_jr import make_dir

# Set paths in config.toml
raw_path, _, calib_path, data_path, _ = smart.set_paths()
flight = "flight_00"  # which flight doe the files in raw belong to?
files = [file for file in os.listdir(raw_path) if os.path.isfile(os.path.join(raw_path, file))]
outdir = os.path.join(data_path, flight)
make_dir(outdir)

# folder = "ASP_07_transfer_calib_20210607"  # name specific folder if you only want to run this one, else set this to ""
# # correct calibration files for dark current
# for dirpath, dirs, files in os.walk(os.path.join(calib_path, folder)):
#     for file in files:
#         if file.endswith("IR.dat"):
#             print(file)
#             smart_cor = smart.correct_smart_dark_current(file, option=2, path=dirpath)
#             smart_cor.to_csv(f"{dirpath}/{file.replace('.dat', '_cor.dat')}", sep="\t", float_format="%.0f")

for file in files:
    smart_cor = smart.correct_smart_dark_current(file, option=2)
    outfile = f"{outdir}/{file.replace('.dat', '_cor.dat')}"
    smart_cor.to_csv(outfile, sep="\t", float_format="%.0f")
    print(f"Saved {outfile}")
