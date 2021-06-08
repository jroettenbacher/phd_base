#!/usr/bin/env python
"""
ASP_07 was configured to write minutely files during calibration.
This script
* merges the minutely files into one file,
* deletes the minutely files,
* saves the merged file to the first found filename.
author: Johannes Röttenbacher
"""

from smart import set_paths
import os
import pandas as pd

# combine minutely files from ASP_07 to one file
_, _, calib_path, _, _ = set_paths()
folder = "ASP_07_transfer_calib_20210607"
channels = ["SWIR", "VNIR"]
for dirpath, dirs, files in os.walk(os.path.join(calib_path, folder)):
    for channel in channels:
        try:
            # channel = "VNIR"  # replace SWIR with VNIR and run again
            filename = [file for file in files if file.endswith(f"{channel}.dat")]
            df = pd.concat([pd.read_csv(f"{dirpath}/{file}", sep="\t", header=None) for file in filename])
            # delete all minutely files
            for file in filename:
                os.remove(os.path.join(dirpath, file))
            outname = f"{dirpath}/{filename[0]}"
            df.to_csv(outname, sep="\t", index=False, header=False)
            print(f"Saved {outname}")
        except ValueError:
            pass
