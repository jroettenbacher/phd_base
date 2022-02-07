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
flight = "Flight_20210629a"
path = h.get_path('horidata', flight)
ql_path = h.get_path("quicklooks", flight)

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

horipath = h.get_path("horidata", flight)
nav_file = [f for f in os.listdir(horipath) if "IMS" in f][0]
nav_path = os.path.join(horipath, nav_file)
nav_data = reader.read_nav_data(nav_path)
# resample file from 20Hz to 1Hz
nav_data = nav_data.resample("S").mean()
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
layout.opts(title=f"{flight} SMART NS Measurements")
layout.cols(1)
figname = f"{ql_path}/{flight}_NavCommand.html"
hv.save(layout, figname)
print(f"Saved {figname}")

# %% Holoviews Dashboard of Stabilization Platform data

horipath = h.get_path("horidata", flight)
hori_file = [f for f in os.listdir(horipath) if f.endswith("dat")][0]
horidata = pd.read_csv(f"{horipath}/{hori_file}", skipinitialspace=True, sep="\t")
horidata["PCTIME"] = pd.to_datetime(horidata["DATE"] + " " + horidata["PCTIME"], format='%Y/%m/%d %H:%M:%S.%f')
horidata_hv = hv.Table(horidata)
# annotate data
roll_target = hv.Dimension('TARGET3', label='Target Roll Angle', unit='deg')
roll = hv.Dimension('POSN3', label='Actual Roll Angle', unit='deg')
pitch_target = hv.Dimension('TARGET4', label='Target Pitch Angle', unit='deg')
pitch = hv.Dimension('POSN4', label='Actual Pitch Angle', unit='deg')
time = hv.Dimension('PCTIME', label='Time', unit='UTC')
layout = hv.Curve(horidata_hv, time, roll_target, label="Roll Target").opts(color="green") \
         * hv.Curve(horidata_hv, time, roll, label="Actual Roll").opts(color="red")\
         + hv.Curve(horidata_hv, time, pitch_target, label="Pitch Target").opts(color="green") \
         * hv.Curve(horidata_hv, time, pitch, label="Actual Pitch").opts(color="red")
layout.opts(
    opts.Curve(responsive=True, height=350, show_grid=True, tools=["hover"],
               fontsize={'title': 16, 'labels': 14, 'xticks': 12, 'yticks': 12, 'legend': 12}),
    opts.Overlay(legend_position="right", legend_offset=(0, 100))
)
layout.opts(title=f"{flight} SMART Stabilization Table Measurements")
layout.cols(1)
figname = f"{ql_path}/{flight}_horidata.html"
hv.save(layout, figname)
print(f"Saved {figname}")
