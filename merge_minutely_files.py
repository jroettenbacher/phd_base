#!/usr/bin/env python
"""
ASP_07 was configured to write minutely files during calibration. This script merges the minutely files into one file
and attaches _merged to the first found filename.
The minutely files are then manually deleted and the _merged is deleted from the merged files.
author: Johannes Röttenbacher
"""

from smart import set_paths
import os
import pandas as pd

# combine minutely files from ASP_07 to one file
_, _, calib_path, _, _ = set_paths()
for dirpath, dirs, files in os.walk(os.path.join(calib_path, "ASP_07_Calib_Lab_20210318")):
    try:
        filename = [file for file in files if file.endswith("SWIR.dat")]  # replace SWIR with VNIR and run again
        df = pd.concat([pd.read_csv(f"{dirpath}/{file}", sep="\s+", header=None) for file in files
                        if file.endswith("SWIR.dat")])
        df.to_csv(f"{dirpath}/{filename[0].replace('.dat', '_merged.dat')}", sep="\t",
                  index=False, header=False)
    except ValueError:
        pass
