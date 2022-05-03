#!/usr/bin/env python
"""Analyze the SMART measurements during the radiation square

*author*: Johannes Röttenbacher
"""

# %%
import os
import pylim.helpers as h
from pandas import Timestamp as Ts
import xarray as xr
import matplotlib.pyplot as plt

radiation_square = dict(s_leg=(Ts(2022, 2, 25, 9, 17), Ts(2022, 2, 25, 9, 19, 45)),
                        w_leg=(Ts(2022, 2, 25, 9, 21, 7), Ts(2022, 2, 25, 9, 23, 30)),
                        n_leg=(Ts(2022, 2, 25, 9, 25), Ts(2022, 2, 25, 9, 27, 40)),
                        e_leg=(Ts(2022, 2, 25, 9, 31, 20), Ts(2022, 2, 25, 9, 33, 50)))

# %% set up paths and get files
date = "20220225"
flight_key = "RF00"
flight = f"HALO-AC3_{date}_HALO_{flight_key}"
campaign = "halo-ac3"
smart_path = h.get_path("calibrated", flight, campaign=campaign)
libradtran_path = h.get_path("libradtran", flight, campaign)
horipath = h.get_path("horidata", flight, campaign)
swir_file = [f for f in os.listdir(smart_path) if "SWIR" in f and f.endswith("nc")][0]
vnir_file = [f for f in os.listdir(smart_path) if "VNIR" in f and f.endswith("nc")][0]
libradtran_file = f"HALO-AC3_HALO_libRadtran_clearsky_simulation_wl500-600_{date}_{flight_key}.nc"
horifile = [f for f in os.listdir(horipath) if f.endswith("nc")][0]

# %% read in SMART files
df_swir = xr.open_dataset(f"{smart_path}/{swir_file}")
df_vnir = xr.open_dataset(f"{smart_path}/{vnir_file}")
# rename variables to merge them
df_vnir = df_vnir.rename(dict(Fdw_VNIR="Fdw"))
df_swir = df_swir.rename(dict(Fdw_SWIR="Fdw"))

# %% merge datasets
df_all = xr.merge([df_swir, df_vnir])
df_all["Fdw_bb"] = df_all["Fdw_VNIR_bb"] + df_all["Fdw_SWIR_bb"]

# %% read in libRadtran simulation and horidata
sim = xr.open_dataset(f"{libradtran_path}/{libradtran_file}")
ins = xr.open_dataset(f"{horipath}/{horifile}")

# %% select the right time from the data
timeslice = slice(radiation_square["s_leg"][0], radiation_square["e_leg"][-1])
df_all = df_all.sel(time=timeslice)
sim = sim.sel(time=timeslice)

# %% integrate SMART over 500-600nm
df_all["Fdw_int"] = df_all["Fdw"].sel(wavelength=slice(500, 600)).sum(dim="wavelength")

# %% plot the broadband irradiance together with the direction

fig, ax = plt.subplots()
ax.plot(df_all.time, df_all.Fdw_int, label="SMART integrated irradiance (500-600nm)")
ax.plot(sim.time, sim.fdw, label="libRadtran clear sky simulation")
ax.grid()
# ax2 = ax.twinx()
ax.legend()
plt.show()
plt.close()
