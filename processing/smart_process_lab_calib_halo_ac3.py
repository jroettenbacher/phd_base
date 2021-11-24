#!/usr/bin/env python
"""Correct the lab calibration files for the dark current and merge the minutely files

The calibration was still done with the naming convention of CIRRUS-HL.
There are two calibrations available both use the VN11 inlet (ASP02), because it has a nicer cosine response, and the optical fiber 22b.
One calibration was done with VN11 attached to J3 and J4 on ASP06 and the other with VN11 attached to J5 and J6.
For HALO-AC3 J5 and J6 will be the channels used.
Thus, only the Fup measurements are of interest.
The Fdw measurements are kept for completeness.
Just be aware of the fact, that the file naming convention is changed for HALO-AC3.
J5 and J6 will then be called Fdw_SWIR and Fdw_VNIR.
So although they are called Fup the calibration measurements are actually for Fdw.

author: Johannes Röttenbacher
"""
# %% module import
import pylim.helpers as h
from pylim import reader, smart
import os
import re
import pandas as pd
import logging

# %% setup logging
log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.DEBUG)

# %% User input
folder = "ASP06_lab_calibration_before"
campaign = "halo-ac3"

# Set paths in config.toml
calib_path = h.get_path("calib", campaign=campaign)

# %% merge VNIR dark measurement files before correcting the calib files
property = ["Fup"]
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
                log.debug(f"Encountered ValueError for {dirpath} and {prop}")
                pass

# %% correct all calibration measurement files for the dark current
for dirpath, dirs, files in os.walk(os.path.join(calib_path, folder)):
    log.info(f"Working on {dirpath}")
    dirname = os.path.basename(dirpath)
    parent = os.path.dirname(dirpath)
    for file in files:
        if file.endswith("SWIR.dat"):
            log.info(f"Working on {dirpath}/{file}")
            smart_cor = smart.correct_smart_dark_current("", file, option=2, path=dirpath)
            outname = f"{dirpath}/{file.replace('.dat', '_cor.dat')}"
            smart_cor.to_csv(outname, sep="\t", float_format="%.0f")
            log.info(f"Saved {outname}")

        if file.endswith("VNIR.dat"):
            log.info(f"Working on {dirpath}/{file}")
            # get integration time from directory name to replace it with the dark string
            int_time = re.search(r"\d{3}ms", dirname)[0]
            _, channel, direction = smart.get_info_from_filename(file)
            new_dirname = dirname.replace(int_time, f"dark_{int_time}") if "dark" not in dirname else dirname
            dark_dir = os.path.join(parent, new_dirname)
            # there is only one merged dark current file for VNIR
            dark_file = [f for f in os.listdir(dark_dir) if direction in f and channel in f][0]
            measurement = reader.read_smart_raw(dirpath, file)
            dark_current = reader.read_smart_raw(dark_dir, dark_file)
            dark_current = dark_current.iloc[:, 2:].mean()
            measurement = measurement.where(measurement.shutter == 1).iloc[:, 2:]  # only use data when shutter is open
            smart_cor = measurement - dark_current
            outname = f"{dirpath}/{file.replace('.dat', '_cor.dat')}"
            smart_cor.to_csv(outname, sep="\t", float_format="%.0f")
            log.info(f"Saved {outname}")

# %% merge minutely corrected files to one file
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
                log.debug(f"Encountered ValueError for '{dirpath}', channel '{channel}' and property '{property}'")
                pass
