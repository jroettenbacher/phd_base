#!/usr/bin/env python
"""Compare measurements of BAHAMAS and SMART IMS

To evaluate the performance of the SMART build in IMS the data is compared to the BAHAMAS measurements.

*author*: Johannes Röttenbacher
"""

# %% import modules
import pylim.helpers as h
from pylim import reader
import os
import pandas as pd
import holoviews as hv
from holoviews import opts

hv.extension('bokeh')

# %% set paths
campaign = "halo-ac3"
flight = "HALO-AC3_20220225_HALO_RF00"
output_format = "html"  # png or html, html gives an interactive plot
plot_path = f"{h.get_path('plot', flight, campaign)}/{flight}"
horipath = h.get_path("horidata", flight, campaign)
horifile = [f for f in os.listdir(horipath) if "IMS0000" in f][0]
bahamas_path = h.get_path("bahamas", flight, campaign)
bahamas_file = [f for f in os.listdir(bahamas_path) if f.endswith("nc") and "QL" in f][0]

# %% read in files
ims = reader.read_nav_data(f"{horipath}/{horifile}")
bahamas = reader.read_bahamas(f"{bahamas_path}/{bahamas_file}")
# convert pitch of the IMS to the same convention as pitch of the bahamas file (positive = nose up)
ims["pitch"] = -ims["pitch"]
bahamas = bahamas.to_dataframe()  # convert dataset to dataframe

# %% Calculate difference between IMS and BAHAMAS
# resample and reindex IMS data to bahamas resolution (10Hz)
ims_10Hz = ims.reindex_like(ims.resample("100ms").asfreq(), method="nearest")
# select the same timesteps
bahamas_sel = (ims_10Hz.index >= bahamas.index[0]) & (ims_10Hz.index <= bahamas.index[-1])
ims_10Hz = ims_10Hz.loc[bahamas_sel]
# calculate difference
roll_diff = pd.DataFrame(ims_10Hz["roll"] - bahamas["IRS_PHI"], columns=["roll"])
pitch_diff = pd.DataFrame(ims_10Hz["pitch"] - bahamas["IRS_THE"], columns=["pitch"])
# calculate mean
rf_mean, rf_median, rf_std = roll_diff.mean(), roll_diff.median(), roll_diff.std()
roll_stats = f"Mean: {rf_mean.values[0]:.4f}\nMedian: {rf_median.values[0]:.4f}\nStd: {rf_std.values[0]:.4f}"
pf_mean, pf_median, pf_std = pitch_diff.mean(), pitch_diff.median(), pitch_diff.std()
pitch_stats = f"Mean: {pf_mean.values[0]:.4f}\nMedian: {pf_median.values[0]:.4f}\nStd: {rf_std.values[0]:.4f}"

# %% reset datetime index for plotting with holoviews
nav_data = ims.reset_index()
bahamas_df = bahamas.reset_index()

# annotate INS data
roll = hv.Dimension('roll', label='Roll Angle', unit='deg')
pitch = hv.Dimension('pitch', label='Pitch Angle', unit='deg')
yaw = hv.Dimension('yaw', label='Yaw Angle', unit='deg')
time = hv.Dimension('time', label='Time', unit='UTC')
# annotate INS data
phi = hv.Dimension('IRS_PHI', label='Roll Angle', unit='deg')
the = hv.Dimension('IRS_THE', label='Pitch Angle', unit='deg')
heading = hv.Dimension('IRS_HDG', label='Yaw Angle', unit='deg')
# annotate difference
roll_dim = hv.Dimension("roll", label="Roll Difference", unit="deg")
pitch_dim = hv.Dimension("pitch", label="Pitch Difference", unit="deg")

# %% create layout
ts = pd.Timestamp(2022, 2, 25, 8)  # timestamp for text location
layout = hv.Curve(nav_data, time, roll, label="IMU") * hv.Curve(bahamas_df, time, phi, label="BAHAMAS") + \
         hv.Curve(roll_diff, time, roll_dim, label="IMU - BAHAMAS") * hv.Text(ts, 0.7, roll_stats) +\
         hv.Curve(nav_data, time, pitch, label="IMU") * hv.Curve(bahamas_df, time, the, label="BAHAMAS") +\
         hv.Curve(pitch_diff, time, pitch_dim, label="IMU - BAHAMAS") * hv.Text(ts, 0, pitch_stats) +\
         hv.Curve(nav_data, time, yaw, label="IMU") * hv.Curve(bahamas_df, time, heading, label="BAHAMAS")
layout.opts(
    opts.Curve(responsive=True, height=300, tools=["hover"], show_grid=True,
               fontsize={'title': 16, 'labels': 14, 'xticks': 12, 'yticks': 12, 'legend': 12}),
    opts.Overlay(legend_position="left")
)
layout.opts(title=f"{flight} SMART IMU Measurements vs BAHAMAS Measurements")
layout.cols(1)
figname = f"{plot_path}/{flight}_SMART_IMU_vs_BAHAMAS.{output_format}"
hv.save(layout, figname)
print(f"Saved {figname}")