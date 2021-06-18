#!/usr/bin/env python
"""Script to correct SMART measurement for dark current and save it to a new file
input: raw smart measurements
output: dark current corrected smart measurements
author: Johannes Roettenbacher
"""
import os
import smart
from functions_jr import make_dir

# Set paths in config.toml
raw_path, _, calib_path, data_path, _ = smart.set_paths()
flight = "flight_00"  # which flight do the files in raw belong to?
inpath = os.path.join(raw_path, flight)
files = [file for file in os.listdir(inpath) if os.path.isfile(os.path.join(inpath, file))]
outdir = os.path.join(data_path, flight)
make_dir(outdir)

folder = "flight_00"  # name specific folder if you only want to run this one, else set this to ""
# date of transfer cali with dark current measurements to use for VNIR, set to "" if not needed
transfer_cali_date = "20210616"

# # correct calibration files for dark current
# for dirpath, dirs, files in os.walk(os.path.join(calib_path, folder)):
#     for file in files:
#         if file.endswith("IR.dat"):
#             print(f"Working on {dirpath}/{file}")
#
#             if len(transfer_cali_date) > 0:
#                 smart_cor = smart.correct_smart_dark_current(file, option=2, path=dirpath, date=transfer_cali_date)
#             else:
#                 smart_cor = smart.correct_smart_dark_current(file, option=2, path=dirpath)
#
#             outname = f"{dirpath}/{file.replace('.dat', '_cor.dat')}"
#             smart_cor.to_csv(outname, sep="\t", float_format="%.0f")
#             print(f"Saved {outname}")

for file in files:
    print(f"Working on {file}")
    if len(transfer_cali_date) > 0:
        smart_cor = smart.correct_smart_dark_current(file, option=2, path=inpath, date=transfer_cali_date)
    else:
        smart_cor = smart.correct_smart_dark_current(file, option=2, path=inpath)
    outfile = f"{outdir}/{file.replace('.dat', '_cor.dat')}"
    smart_cor.to_csv(outfile, sep="\t", float_format="%.0f")
    print(f"Saved {outfile}")
