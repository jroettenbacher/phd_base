#!/usr/bin/env python
"""Plot integrated SMART measurement together with libRadtran broadband simulation

*author*: Johannes Röttenbacher
"""

# %% import modules
import pylim.helpers as h
from pylim import reader
import os
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import logging

log = logging.getLogger("pylim")
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

# %% set up paths and get files
campaign = "halo-ac3"
date = "20220321"
flight_key = "RF08"
flight = f"HALO-AC3_{date}_HALO_{flight_key}"
smart_path = h.get_path("calibrated", flight, campaign=campaign)
libradtran_path = h.get_path("libradtran", flight, campaign)
horipath = h.get_path("horidata", flight, campaign)
plot_path = f"{h.get_path('plot', flight, campaign)}/stabbi"
h.make_dir(plot_path)
# get files
swir_file = [f for f in os.listdir(smart_path) if "SWIR" in f and f.endswith("nc")][0]
vnir_file = [f for f in os.listdir(smart_path) if "VNIR" in f and f.endswith("nc")][0]
libradtran_file = f"HALO-AC3_HALO_libRadtran_clearsky_simulation_wl500-600_{date}_{flight_key}.nc"
horifile = [f for f in os.listdir(horipath) if f.endswith("nc")][0]
stabbi_file = [f for f in os.listdir(horipath) if f.endswith("dat")][0]

# %% read in data
sim = xr.open_dataset(f"{libradtran_path}/{libradtran_file}")
horidata = xr.open_dataset(f"{horipath}/{horifile}")
stabbi = reader.read_stabbi_data(f"{horipath}/{stabbi_file}")
df_swir = xr.open_dataset(f"{smart_path}/{swir_file}")
df_vnir = xr.open_dataset(f"{smart_path}/{vnir_file}")
# rename variables to merge them
df_vnir = df_vnir.rename(dict(Fdw_VNIR="Fdw"))
df_swir = df_swir.rename(dict(Fdw_SWIR="Fdw"))

# %% merge datasets
df_all = xr.merge([df_swir, df_vnir])
df_all["Fdw_bb"] = df_all["Fdw_VNIR_bb"] + df_all["Fdw_SWIR_bb"]

# %% plot SMART measurement and libRadtran simulation
h.set_cb_friendly_colors()
# x_sel = (pd.Timestamp(2022, 3, 29, 10), pd.Timestamp(2021, 6, 29, 12, 15))
x_sel = (pd.to_datetime(sim.time[0].values), pd.to_datetime(sim.time[-1].values))
plt.rc('font', size=20)
plt.rc('lines', linewidth=3)
fig, axs = plt.subplots(nrows=2, figsize=(13, 9))

ax = axs[0]
# SMART measurements
ax.plot(df_all["time"], df_all["Fdw_bb"], label=r"F$_\downarrow$ SMART")
# libRadtran simulations
ax.plot(sim["time"], sim["fdw"], label=r"F$_\downarrow$ libRadtran", c="#117733", ls="--")
ax.legend()
ax.set_ylabel(r"Irradiance (W$\,$m$^{-2}$)")
ax.set_title(f"Integrated SMART Measurement "
             f"{df_all.wavelength[0].values:.2f} - {df_all.wavelength[-1].values:.2f} nm\n"
             f"and libRadtran Clear Sky Simulation")

# second row
ax = axs[1]
# Offset from target
ax.plot(stabbi.index, stabbi["POSN3"]-stabbi["TARGET3"], c="#DDCC77")
ax.set_ylabel("Offset from \nTarget Roll Angle (deg)")
ax.set_xlabel("Time (UTC)")
ax.set_ylim(-2, 2)

# aesthetics
for ax in axs:
    ax.set_xlim(x_sel)
    h.set_xticks_and_xlabels(ax, x_sel[-1] - x_sel[0])
    ax.grid()

plt.tight_layout()
# plt.show()
figname = f"{plot_path}/HALO-AC3_HALO_SMART_wl500-600_libRadtran_timeseries_{date}_{flight_key}.png"
plt.savefig(figname, dpi=100)
log.info(f"Saved {figname}")
plt.close()
