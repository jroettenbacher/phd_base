#!/usr/bin/env python
"""Script to correct SMART transfer calibration measurement for dark current and save it to a new file and merge the
minutely files
input: raw smart transfer calibration measurements
output: dark current corrected and merged smart measurements
author: Johannes Roettenbacher
"""
import os
import smart
import pandas as pd
import logging

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

# User input
folder = "ASP06_transfer_calib_20210729"

# Set paths in config.toml
calib_path = smart.get_path("calib")

# merge VNIR dark measurement files before correcting the calib files
property = ["Iup", "Fup", "Fdw"]
for dirpath, dirs, files in os.walk(os.path.join(calib_path, folder)):
    log.info(f"Working on {dirpath}")
    if "dark" in dirpath:
        for prop in property:
            try:
                vnir_dark_files = [f for f in files if f.endswith(f"{prop}_VNIR.dat")]
                df = pd.concat([pd.read_csv(f"{dirpath}/{file}", sep="\t", header=None) for file in vnir_dark_files])
                # delete all minutely files
                for file in vnir_dark_files:
                    os.remove(os.path.join(dirpath, file))
                    log.info(f"Deleted {dirpath}/{file}")
                outname = f"{dirpath}/{vnir_dark_files[0]}"
                df.to_csv(outname, sep="\t", index=False, header=False)
                log.info(f"Saved {outname}")
            except ValueError:
                pass

# correct all calibration measurement files for the dark current
for dirpath, dirs, files in os.walk(os.path.join(calib_path, folder)):
    log.info(f"Working on {dirpath}")
    for file in files:
        if file.endswith("IR.dat"):
            log.info(f"Working on {dirpath}/{file}")
            smart_cor = smart.correct_smart_dark_current("", file, option=2, path=dirpath)
            outname = f"{dirpath}/{file.replace('.dat', '_cor.dat')}"
            smart_cor.to_csv(outname, sep="\t", float_format="%.0f")
            log.info(f"Saved {outname}")

# merge minutely corrected files to one file
channels = ["SWIR", "VNIR"]
for dirpath, dirs, files in os.walk(os.path.join(calib_path, folder)):
    log.info(f"Working on {dirpath}")
    for channel in channels:
        for prop in property:
            try:
                filename = [file for file in files if file.endswith(f"{prop}_{channel}_cor.dat")]
                df = pd.concat([pd.read_csv(f"{dirpath}/{file}", sep="\t", index_col="time") for file in filename])
                # delete all minutely files
                for file in filename:
                    os.remove(os.path.join(dirpath, file))
                    log.info(f"Deleted {dirpath}/{file}")
                outname = f"{dirpath}/{filename[0]}"
                df.to_csv(outname, sep="\t")
                log.info(f"Saved {outname}")
            except ValueError:
                pass
