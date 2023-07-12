#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 04.07.2023

Preprocess the big CAMS data files to generate monthly means using cdo on the ecs-login node.

1. Before running this script run: `module load python3` in the console
2. Call this script with: `python3 halo_ac3_preprocess_cams_data_on_ecmwf.py`
"""

# %% import modules
import os
from tqdm import tqdm
from cdo import *
cdo = Cdo()

# %% set paths
base_path = f"{os.environ['SCRATCH']}/scratch_jr/cams_data"

# %% loop through yearly folders
years = range(2003, 2021)
for year in tqdm(years, desc="Years"):
    os.chdir(f"{base_path}/{year}")
    infiles = [f for f in os.listdir() if f.endswith("grb")]
    for infile in tqdm(infiles, desc="Files"):
        outfile = infile.replace(".grb", "_monmean.grb")
        cdo.monmean(input=infile, output=outfile, options="-v")
