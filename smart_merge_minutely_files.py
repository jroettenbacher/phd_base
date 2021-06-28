#!/usr/bin/env python
"""
ASP06 and ASP07 were configured to write minutely files during calibration.
This script
* merges the minutely files into one file,
* deletes the minutely files,
* saves the merged file to the first found filename.
author: Johannes Röttenbacher
"""

from smart import set_paths
import os
import pandas as pd
import logging
from tqdm import tqdm

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

# combine minutely files to one file
raw_path, _, calib_path, data_path, _ = set_paths()
directory = data_path
folder = "flight_03"
channels = ["SWIR", "VNIR"]
property = ["Iup", "Fup", "Fdw"]
for dirpath, dirs, files in os.walk(os.path.join(directory, folder)):
    for prop in property:
        for channel in channels:
            try:
                filename = [file for file in files if file.endswith(f"{prop}_{channel}_cor.dat")]
                df = pd.concat([pd.read_csv(f"{dirpath}/{file}", sep="\t", index_col="time") for file in tqdm(filename)])
                # delete all minutely files
                for file in filename:
                    os.remove(os.path.join(dirpath, file))
                outname = f"{dirpath}/{filename[0]}"
                df.to_csv(outname, sep="\t")
                log.info(f"Saved {outname}")
            except ValueError:
                pass
