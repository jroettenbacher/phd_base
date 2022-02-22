#!/usr/bin/env python

# %% library import
import pylim.helpers as h
from pylim import reader
import os
import pandas as pd
import matplotlib.pyplot as plt
import holoviews as hv
from holoviews import opts

hv.extension('bokeh')

# %% Define flight and paths
flight = "HALO-AC3_FD_00_HALO_Flight_00_20220221"
path = h.get_path('horidata', flight, campaign="halo-ac3")
ql_path = h.get_path("quicklooks", flight, campaign="halo-ac3")
h.make_dir(ql_path)
output_format = "html"  # png or html, html gives an interactive plot

# %% working section
# file = [f for f in os.listdir(path) if f.endswith('dat')]
#
# df = pd.read_csv(f"{path}/{file[0]}", sep="\t", skipinitialspace=True, index_col="PCTIME", parse_dates=True)
# start_id = 8000
# end_id = start_id + 3600
# fig, ax = plt.subplots()
# df.iloc[start_id:end_id].plot(y="TARGET3", ax=ax, label="Roll Target")
# df.iloc[start_id:end_id].plot(y="POSN3", ax=ax, label="Roll Position")
# plt.grid()
# plt.savefig(f"{path}/../plots/{flight}_Roll_target-position.png", dpi=100)
# plt.show()
# plt.close()
#
# fig, ax = plt.subplots()
# df.iloc[start_id:end_id].plot(y="TARGET4", ax=ax, label="Pitch Target")
# df.iloc[start_id:end_id].plot(y="POSN4", ax=ax, label="Pitch Position")
# plt.grid()
# # plt.savefig(f"{path}/../plots/{flight}_Pitch_target-position.png", dpi=100)
# plt.show()
# plt.close()

# %% Holoviews Dashboard of NavCommand quicklook

horipath = h.get_path("horidata", flight, campaign="halo-ac3")
nav_files = [f for f in os.listdir(horipath) if "Nav_IMS" in f]
nav_paths = [os.path.join(horipath, f) for f in nav_files]
nav_data = pd.concat([reader.read_nav_data(f) for f in nav_paths])
# resample file from 20Hz to 1Hz
ims_1Hz = nav_data.resample("1s").asfreq()  # create a dataframe with a 1Hz index
# reindex original dataframe and use the nearest values for the full seconds
nav_data = nav_data.reindex_like(ims_1Hz, method="nearest")
# convert to hv.Table
nav_data = nav_data.reset_index()
nav_table = hv.Table(nav_data)
# annotate data
roll = hv.Dimension('roll', label='Roll Angle', unit='deg')
pitch = hv.Dimension('pitch', label='Pitch Angle', unit='deg')
yaw = hv.Dimension('yaw', label='Yaw Angle', unit='deg')
time = hv.Dimension('time', label='Time', unit='UTC')
layout = hv.Curve(nav_data, time, roll) + hv.Curve(nav_data, time, pitch) + hv.Curve(nav_data, time, yaw)
layout.opts(
    opts.Curve(responsive=True, height=300, tools=["hover"], show_grid=True,
               fontsize={'title': 16, 'labels': 14, 'xticks': 12, 'yticks': 12, 'legend': 12})
)
layout.opts(title=f"{flight} SMART INS Measurements")
layout.cols(1)
figname = f"{ql_path}/{flight}_NavCommand.{output_format}"
hv.save(layout, figname)
print(f"Saved {figname}")

# %% Holoviews Dashboard of Stabilization Platform data

horipath = h.get_path("horidata", flight, campaign="halo-ac3")
hori_files = [f for f in os.listdir(horipath) if f.endswith("dat")]
horidata = pd.concat([pd.read_csv(f"{horipath}/{f}", skipinitialspace=True, sep="\t") for f in hori_files])
horidata["PCTIME"] = pd.to_datetime(horidata["DATE"] + " " + horidata["PCTIME"], format='%Y/%m/%d %H:%M:%S.%f')
horidata_hv = hv.Table(horidata)
# annotate data
roll_target = hv.Dimension('TARGET3', label='Target Roll Angle', unit='deg')
roll = hv.Dimension('POSN3', label='Actual Roll Angle', unit='deg')
pitch_target = hv.Dimension('TARGET4', label='Target Pitch Angle', unit='deg')
pitch = hv.Dimension('POSN4', label='Actual Pitch Angle', unit='deg')
time = hv.Dimension('PCTIME', label='Time', unit='UTC')
layout = hv.Curve(horidata_hv, time, roll_target, label="Roll Target").opts(color="green") \
         * hv.Curve(horidata_hv, time, roll, label="Actual Roll").opts(color="red", ylabel="Roll Angle (deg)")\
         + hv.Curve(horidata_hv, time, pitch_target, label="Pitch Target").opts(color="green") \
         * hv.Curve(horidata_hv, time, pitch, label="Actual Pitch").opts(color="red", ylabel="Pitch Angele (deg)")
layout.opts(
    opts.Curve(responsive=True, height=350, show_grid=True, tools=["hover"],
               fontsize={'title': 16, 'labels': 14, 'xticks': 12, 'yticks': 12, 'legend': 12}),
    opts.Overlay(legend_position="right", legend_offset=(0, 100))
)
layout.opts(title=f"{flight} SMART Stabilization Table Measurements")
layout.cols(1)
figname = f"{ql_path}/{flight}_horidata.{output_format}"
hv.save(layout, figname)
print(f"Saved {figname}")
