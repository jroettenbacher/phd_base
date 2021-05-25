#!/usr/bin/env python
"""Script to correct SMART measurement for dark current and save it to a new file
input: raw smart measurements
output: corrected smart measurements
author: Johannes Roettenbacher
"""
import os
import smart

# Set paths in config.toml
raw_path, _, _, data_path = smart.set_paths()
files = os.listdir(raw_path)

for file in files:
    smart_cor = smart.correct_smart_dark_current(file, option=2)
    smart_cor.to_csv(f"{data_path}/{file.replace('.dat', '_cor.dat')}", sep="\t", float_format="%.0f")

