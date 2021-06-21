#!/usr/bin/env python
"""Script to correct SMART measurement for dark current and save it to a new file
input: raw smart measurements
output: dark current corrected smart measurements
author: Johannes Roettenbacher
"""
import os
import smart
from functions_jr import make_dir
import logging
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.WARNING)

# User input
flight = "flight_00"  # which flight do the files in raw belong to?
# date of transfer cali with dark current measurements to use for VNIR, set to "" if not needed
transfer_cali_date = "20210616"

# Set paths in config.toml
raw_path, _, calib_path, data_path, _ = smart.set_paths()
inpath = os.path.join(raw_path, flight)
outdir = os.path.join(data_path, flight)
make_dir(outdir)
# create list of input files and add a progress bar to it
files = tqdm([file for file in os.listdir(inpath) if os.path.isfile(os.path.join(inpath, file))])

# # correct calibration files for dark current
# folder = "flight_00"  # name specific folder if you only want to run this one, else set this to ""
# for dirpath, dirs, files in os.walk(os.path.join(calib_path, folder)):
#     for file in files:
#         if file.endswith("IR.dat"):
#             log.info(f"Working on {dirpath}/{file}")
#
#             if len(transfer_cali_date) > 0:
#                 smart_cor = smart.correct_smart_dark_current(file, option=2, path=dirpath, date=transfer_cali_date)
#             else:
#                 smart_cor = smart.correct_smart_dark_current(file, option=2, path=dirpath)
#
#             outname = f"{dirpath}/{file.replace('.dat', '_cor.dat')}"
#             smart_cor.to_csv(outname, sep="\t", float_format="%.0f")
#             log.info(f"Saved {outname}")


def make_dark_cur_cor_file(file: str, inpath: str, transfer_cali_date: str, outdir: str) -> None:
    """
    Function to write a dark current corrected file to the given input file
    Args:
        file: Standard SMART file name
        inpath: input directory
        transfer_cali_date: date of transfer calibration to use for the dark current measurements ("" or yyyymmdd)
        outdir: output directory

    Returns: Writes dark current corrected file to data_path in config.toml

    """
    log.info(f"Working on {file}")
    if len(transfer_cali_date) > 0:
        smart_cor = smart.correct_smart_dark_current(file, option=2, path=inpath, date=transfer_cali_date)
    else:
        smart_cor = smart.correct_smart_dark_current(file, option=2, path=inpath)
    outfile = f"{outdir}/{file.replace('.dat', '_cor.dat')}"
    smart_cor.to_csv(outfile, sep="\t", float_format="%.0f")
    log.info(f"Saved {outfile}")

# run job in parallel
Parallel(n_jobs=cpu_count()-2)(delayed(make_dark_cur_cor_file)(file, inpath, transfer_cali_date, outdir) for file in files)
