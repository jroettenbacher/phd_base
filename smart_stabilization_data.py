#!/usr/bin/env python

# %% library import
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import holoviews as hv
from holoviews import opts

import smart

hv.extension('bokeh')

# %% Working section
date = 20210625
path = f"C:/Users/Johannes/Documents/Doktor/campaigns/CIRRUS-HL/ASP04/HARPDATA/{date}"

file = [f for f in os.listdir(path) if f.endswith('dat')]

df = pd.read_csv(f"{path}/{file[0]}", sep="\t", skipinitialspace=True, index_col="PCTIME", parse_dates=True)
start_id = 8000
end_id = start_id + 3600
fig, ax = plt.subplots()
df.iloc[start_id:end_id].plot(y="TARGET3", ax=ax, label="Roll Target")
df.iloc[start_id:end_id].plot(y="POSN3", ax=ax, label="Roll Position")
plt.grid()
plt.savefig(f"{path}/../plots/{date}_Roll_target-position.png", dpi=100)
plt.show()
plt.close()

fig, ax = plt.subplots()
df.iloc[start_id:end_id].plot(y="TARGET4", ax=ax, label="Pitch Target")
df.iloc[start_id:end_id].plot(y="POSN4", ax=ax, label="Pitch Position")
plt.grid()
plt.savefig(f"{path}/../plots/{date}_Pitch_target-position.png", dpi=100)
plt.show()
plt.close()

# %% Holoviews Dashboard of NavCommand quicklook


def read_nav_data(nav_path: str) -> pd.DataFrame:
    """
    Reader function for Navigation data file from the INS
    Args:
        nav_path: path to file including filename

    Returns: pandas DataFrame with headers and a DateTimeIndex

    """
    # read out the time start time information given in the file
    with open(nav_path) as f:
        time_info = f.readlines()[1]
    start_time = pd.to_datetime(time_info[11:31], format="%m/%d/%Y %H:%M:%S")
    # define the start date of the measurement
    start_date = pd.Timestamp(year=start_time.year, month=start_time.month, day=start_time.day)
    header = ["marker", "seconds", "roll", "pitch", "yaw", "AccS_X", "AccS_Y", "AccS_Z", "OmgS_X", "OmgS_Y", "OmgS_Z"]
    nav = pd.read_csv(nav_path, sep="\s+", skiprows=13, header=None, names=header)
    nav["time"] = pd.to_datetime(nav["seconds"], origin=start_date, unit="s")
    nav = nav.set_index("time")

    return nav


date = "20210626"  # User Input
horipath = smart.get_path("horidata")
nav_dir = os.path.join(horipath, f"NavCommand/{date}")
nav_file = [f for f in os.listdir(nav_dir) if "IMS" in f][0]
nav_path = os.path.join(nav_dir, nav_file)
nav_data = read_nav_data(nav_path)
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
    opts.Curve(width=900, height=200, tools=["hover"], show_grid=True)
)
layout.opts(title=f"{date} INS Measurements")
layout.cols(1)
outpath = "C:/Users/Johannes/Documents/Doktor/campaigns/CIRRUS-HL/horidata/plots"
hv.save(layout, f"{outpath}/{date}_NavCommand.html")

# %% Holoviews Dashboard of Stabilization Platform data

horipath = smart.get_path("horidata")
hori_file = [f for f in os.listdir(horipath) if date[2:] in f and f.endswith("dat")][0]
horidata = pd.read_csv(f"{horipath}/{hori_file}", skipinitialspace=True, sep="\t")
file_date = pd.to_datetime(horidata["DATE"].iloc[0])
horidata["PCTIME"] = pd.to_datetime(horidata["DATE"] + " " + horidata["PCTIME"], format='%Y/%m/%d %H:%M:%S.%f')
horidata_hv = hv.Table(horidata)
# annotate data
roll_target = hv.Dimension('TARGET3', label='Target Roll Angle', unit='deg')
roll = hv.Dimension('POSN3', label='Actual Roll Angle', unit='deg')
pitch_target = hv.Dimension('TARGET4', label='Target Pitch Angle', unit='deg')
pitch = hv.Dimension('POSN4', label='Actual Pitch Angle', unit='deg')
time = hv.Dimension('PCTIME', label='Time', unit='UTC')
layout = hv.Curve(horidata_hv, time, roll_target, label="Roll Target").opts(color="green") * hv.Curve(horidata_hv, time, roll, label="Actual Roll").opts(color="red")\
         + hv.Curve(horidata_hv, time, pitch_target, label="Pitch Target").opts(color="green") * hv.Curve(horidata_hv, time, pitch, label="Actual Pitch").opts(color="red")
layout.opts(
    opts.Curve(width=900, height=300, show_grid=True, tools=["hover"]),
    opts.Overlay(legend_position="right", legend_offset=(0, 100))
)
layout.opts(title=f"{file_date:%Y-%m-%d} Stabilization Table Measurements")
layout.cols(1)
outpath = "C:/Users/Johannes/Documents/Doktor/campaigns/CIRRUS-HL/horidata/plots"
hv.save(layout, f"{outpath}/{file_date:%Y%m%d}_horidata.html")
