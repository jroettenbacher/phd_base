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
log.setLevel(logging.INFO)

# User input
flight = "Flight_20210723a"  # which flight do the files in raw belong to?
# date of transfer cali with dark current measurements to use for VNIR, set to "" if not needed
transfer_cali_date = smart.transfer_calibs[flight]

# Set paths in config.toml
inpath = smart.get_path("raw", flight)
outdir = smart.get_path("data", flight)
make_dir(outdir)
# create list of input files and add a progress bar to it
files = tqdm([file for file in os.listdir(inpath) if os.path.isfile(os.path.join(inpath, file))])
files_debug = [file for file in os.listdir(inpath) if os.path.isfile(os.path.join(inpath, file))]


def make_dark_cur_cor_file(flight: str, file: str, inpath: str, transfer_cali_date: str, outdir: str) -> None:
    """
    Function to write a dark current corrected file to the given input file
    Args:
        flight: to which flight does the file belong to? (e.g. Flight_20210707a)
        file: Standard SMART file name
        inpath: input directory
        transfer_cali_date: date of transfer calibration to use for the dark current measurements ("" or yyyymmdd)
        outdir: output directory

    Returns: Writes dark current corrected file to data_path in config.toml

    """
    log.info(f"Working on {file}")
    if len(transfer_cali_date) > 0:
        smart_cor = smart.correct_smart_dark_current(flight, file, option=2, path=inpath, date=transfer_cali_date)
    else:
        smart_cor = smart.correct_smart_dark_current(flight, file, option=2, path=inpath)
    outfile = f"{outdir}/{file.replace('.dat', '_cor.dat')}"
    smart_cor.to_csv(outfile, sep="\t", float_format="%.0f")
    log.info(f"Saved {outfile}")


# test one file
# file = files_debug[0]
# make_dark_cur_cor_file(file, inpath, transfer_cali_date, outdir)
# for file in files_debug:
#     make_dark_cur_cor_file(file, inpath, transfer_cali_date, outdir)
# run job in parallel
Parallel(n_jobs=cpu_count()-2)(delayed(make_dark_cur_cor_file)(flight, file, inpath, transfer_cali_date, outdir)
                               for file in files)

print("Done with smart_write_dark_current_corrected_file.py")
