#!/usr/bin/env python
"""Calibrate measurement files with the transfer calibration
author: Johannes Roettenbacher"""

# read in dark current corrected measurement files
measurement = smart.read_smart_cor(f"{data_path}/flight_00", "2021_03_29_11_15.Fdw_VNIR_cor.dat")
# set negative values to 0
measurement[measurement.values < 0] = 0
# convert to long format
m_long = measurement.melt(var_name="pixel", value_name="counts", ignore_index=False)
df = m_long.merge(lab_df.loc[:, ["pixel", "c_field"]], on="pixel", right_index=True)
df["F_x"] = df["counts"] * df["c_field"]
df.pivot(columns="pixel", values="F_x")