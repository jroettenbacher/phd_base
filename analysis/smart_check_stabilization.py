#!/usr/bin/env python
"""Investigate active stabilization of the upward facing SMART inlet during CIRRUS-HL

Notes:

* The SMART IMS records data at 200 Hz and BAHAMAS records data at 10 Hz.
* SMART IMS pitch is negative = nose up, BAHAMAS pitch is negative = nose down -> switch sign of IMS
* The stabilization of F_down was shut off and a ring was inserted to stop the inlet from freely moving on the 11.07.2021.

Results:
* Somewhat constant offset between IMS and BAHAMAS of about 0.8 deg for roll (IMS > BAHAMAS)
* Pitch only shows higher deviations from zero during take off and landing


*author*: Johannes Röttenbacher
"""

# %% import modules
import pandas as pd
from pylim import reader, smart
import pylim.helpers as h
import os
import matplotlib.pyplot as plt

# %% functions


def read_stabilization_table(filepath: str) -> pd.DataFrame:
    """Read in Stabilization table data from the HALO SMART stabilization

    Args:
        filepath: complete path to .dat file

    Returns: time series with stabilization table data

    """
    df = pd.read_csv(filepath, skipinitialspace=True, sep="\t")
    df["time"] = pd.to_datetime(df["DATE"] + " " + df["PCTIME"], format='%Y/%m/%d %H:%M:%S.%f')
    df.set_index("time", inplace=True)

    return df
# %% read in SMART hori and nav data and BAHAMAS data
flight = "Flight_20210629a"
hori_path = h.get_path("horidata", flight)
nav_file = [f for f in os.listdir(hori_path) if "IMS" in f][0]
nav_filepath = os.path.join(hori_path, nav_file)
hori_file = [f for f in os.listdir(hori_path) if f.endswith("dat")][0]
hori_filepath = os.path.join(hori_path, hori_file)
bahamas_path = h.get_path("bahamas", flight)
bahamas_file = [f for f in os.listdir(bahamas_path) if f.startswith("CIRRUSHL") and f.endswith(".nc")][0]
bahamas_filepath = os.path.join(bahamas_path, bahamas_file)

nav_data = reader.read_nav_data(nav_filepath)
hori_data = read_stabilization_table(hori_filepath)
bahamas_ds = reader.read_bahamas(bahamas_filepath)

# %% cut IMS data to start and end with BAHAMAS data
bahamas_start, bahamas_end = bahamas_ds.time[0].values, bahamas_ds.time[-1].values
# hori_data = hori_data[(hori_data.index > bahamas_start) & (hori_data.index < bahamas_end)]
nav_data = nav_data[(nav_data.index > bahamas_start) & (nav_data.index < bahamas_end)]

# %% resample IMS data to 10 Hz
nav_data_10hz = nav_data.resample("0.1S").asfreq()  # create a dataframe with the correct index
# reindex the original df to the 10Hz index and choose the closest values to fill in
nav_data = nav_data.reindex_like(nav_data_10hz, method="nearest")

# %% convert BAHAMAS data set to a data frame and merge both data frames
bahamas_df = bahamas_ds.to_dataframe()
df = nav_data.merge(bahamas_df, on="time")

# %% calculate difference between SMART INS measured roll and pitch and BAHAMAS measured roll and pitch
df["pitch"] = -df["pitch"]  # switch sign of IMS pitch
df["roll_diff"] = df["roll"] - df["IRS_PHI"]
df["pitch_diff"] = df["pitch"] - df["IRS_THE"]

# %% get time extend for nice x-axis formatting
time_extend = df.index[-1] - df.index[0]
plt.rcParams["font.size"] = 12

# %% plot the differences
fig, axs = plt.subplots(nrows=2, figsize=(12, 6))
ax = axs[0]  # roll difference
ax.plot(df.index, df["roll_diff"])
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Roll Difference \n(IMS - BAHAMAS) (deg)")
ax = axs[1]  # pitch difference
ax.plot(df.index, df["pitch_diff"])
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Pitch Difference \n(IMS - BAHAMAS) (deg)")
for ax in axs:
    h.set_xticks_and_xlabels(ax, time_extend)
    ax.grid()
plt.tight_layout()
plt.show()
plt.close()

# %% plot both roll and pitch measurements in one plot
fig, axs = plt.subplots(nrows=2, figsize=(12, 6))
ax = axs[0]  # roll difference
ax.plot(df.index, df["roll"], label="IMS")
ax.plot(df.index, df["IRS_PHI"], label="BAHAMAS")
ax.set_ylabel("Roll Angle (deg)")
ax = axs[1]  # pitch difference
ax.plot(df.index, df["pitch"], label="IMS")
ax.plot(df.index, df["IRS_THE"], label="BAHAMAS")
ax.set_ylabel("Pitch Angle (deg)")
for ax in axs:
    ax.set_xlabel("Time (UTC)")
    h.set_xticks_and_xlabels(ax, time_extend)
    ax.grid()
    ax.legend()
plt.tight_layout()
plt.show()
plt.close()

# %% plot inclinometer data
fig, axs = plt.subplots(nrows=2, figsize=(12, 6))
ax = axs[0]
ax.plot(hori_data.index, hori_data["INCL2ROL"])
ax.plot(hori_data.index, hori_data["INCL2ROL"].rolling(window="5min").mean(), label="5min rolling mean")
ax.set_ylabel("Roll Angle (deg)")
ax = axs[1]
ax.plot(hori_data.index, hori_data["INCL2PIT"])
ax.plot(hori_data.index, hori_data["INCL2PIT"].rolling(window="5min").mean(), label="5min rolling mean")
ax.set_ylabel("Pitch Angle (deg)")
for ax in axs:
    ax.set_xlabel("Time (UTC)")
    h.set_xticks_and_xlabels(ax, time_extend)
    ax.grid()
    ax.legend()

fig.suptitle("SMART Inclinometer data")
plt.tight_layout()
plt.show()
plt.close()
