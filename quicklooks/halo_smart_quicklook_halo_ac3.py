#!/usr/bin/env python
"""Produces the standard SMART quicklooks for HALO-(AC3)3


*author*: Johannes Röttenbacher
"""

# %% import modules and set paths
import pylim.helpers as h
from pylim.halo_ac3 import take_offs_landings
from pylim import reader, smart
import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

campaign = "halo-ac3"
flight = "HALO-AC3_20220411"
flight_key = flight[19:] if campaign == "halo-ac3" else flight
halo_smart_path = h.get_path("calibrated", flight, campaign)
halo_smart_file = [f for f in os.listdir(halo_smart_path) if f.endswith(".nc")][0]
halo_smart_filepath = os.path.join(halo_smart_path, halo_smart_file)
horidata_path = h.get_path("horidata", flight, campaign)
ins_file = [f for f in os.listdir(horidata_path) if f.endswith(".nc")][0]
ins_filepath = os.path.join(horidata_path, ins_file)
libradtran_path = h.get_path("libradtran", flight, campaign)
libradtran_file = [f for f in os.listdir(libradtran_path) if "smart" in f][0]
libradtran_filepath = os.path.join(libradtran_path, libradtran_file)
halo_smart_spectra = reader.read_smart_cor(halo_smart_path, "2022_02_25_07_25.Fdw_VNIR_cor_calibrated_norm.dat")
ql_path = h.get_path("quicklooks", flight, campaign)

# %%
halo_smart = xr.open_dataset(halo_smart_filepath)
ins = xr.open_dataset(ins_filepath)
sim = xr.open_dataset(libradtran_filepath)

# %% cut all files to start and end with the flight
flight_time = take_offs_landings[flight_key]
timedelta = pd.to_datetime(flight_time[1]) - pd.to_datetime(flight_time[0])
time_sel = (flight_time[0] < halo_smart.time) & (halo_smart.time < flight_time[1])
halo_smart = halo_smart.sel(time=time_sel)
time_sel = (flight_time[0] < ins.time) & (ins.time < flight_time[1])
ins = ins.sel(time=time_sel)
time_sel = (flight_time[0] < sim.time) & (sim.time < flight_time[1])
sim = sim.sel(time=time_sel)


# %%
halo_smart = halo_smart.resample(time="1s").mean()
halo_smart = halo_smart.sel(time=halo_smart.time[1:])

# %%
roll_filter = 5
pitch_filter = 2
pitch_offset = 2  # pitch offset from 0
attitude_filter = ((np.abs(ins["roll"]) < roll_filter) & (np.abs(ins.pitch - pitch_offset) < pitch_filter)).values

# %% filter SMART data for high roll and pitch values
halo_smart_filtered = halo_smart.where(attitude_filter)

# %% 532 nm time series
fig, ax = plt.subplots()
ax.plot(halo_smart_filtered.time[100:-1200], halo_smart_filtered.F_down_solar_wl_532[100:-1200], label="532nm")
# ins.pitch.plot()
ax.grid()
ax.legend()
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Spectral Solar Downward Irradiance (W$\,$m$^{-2}\,$nm$^{-1}$)")
h.set_xticks_and_xlabels(ax, timedelta)
plt.tight_layout()
# plt.show()
plt.savefig(f"{ql_path}/HALO-SMART_irradiance-532nm_{flight[9:]}.png", dpi=100)
plt.close()
