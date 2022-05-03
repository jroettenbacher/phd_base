#!/usr/bin/env python
"""Correct the SMART measurement for the cosine response of the inlet

Each wavelength has to be corrected with its own factor depending on the solar zenith angle.

*author*: Johannes Röttenbacher
"""

# %% import modules
import pylim.helpers as h
from pylim import smart
import intake
import os
import pandas as pd
import xarray as xr
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# %% set paths
campaign = "halo-ac3"
date = "20220321"
flight_key = "RF08"
flight = f"HALO-AC3_{date}_HALO_{flight_key}"
inlet = "VN05"
cat = intake.open_catalog("./ac3airborne-intake/catalog.yaml")["HALO-AC3"]["HALO"]
kwds = {'simplecache': dict(cache_storage=h.get_path("bacardi", flight, campaign), same_names=True)}
credentials = {"storage_options": kwds,
               "user": os.environ.get("AC3_CLOUD_USER"), "password": os.environ.get("AC3_CLOUD_PASSWORD")}

smart_path = h.get_path("calibrated", flight, campaign)
smart_vnir_file = f"HALO-AC3_HALO_SMART_Fdw_VNIR_{date}_{flight_key}.nc"
smart_swir_file = f"HALO-AC3_HALO_SMART_Fdw_SWIR_{date}_{flight_key}.nc"
cosine_path = "C:/Users/Johannes/Documents/Doktor/instruments/SMART/cosine_correction_factors"
cosine_file = f"HALO_SMART_{inlet}_cosine_correction_factors.csv"
libradtran_path = h.get_path("libradtran", flight, campaign)
libradtran_file = f"HALO-AC3_HALO_libRadtran_clearsky_simulation_wl500-600_{date}_{flight_key}.nc"

# %% read in files
cosine_cor = pd.read_csv(f"{cosine_path}/{cosine_file}")
vnir = xr.open_dataset(f"{smart_path}/{smart_vnir_file}")
swir = xr.open_dataset(f"{smart_path}/{smart_swir_file}")
smart_all = smart.merge_vnir_swir_nc(vnir, swir)
libradtran = xr.open_dataset(f"{libradtran_path}/{libradtran_file}")
bacardi = cat["BACARDI"]["HALO-AC3_HALO_RF08"](**credentials).to_dask()

# %% resample BACARDI to SMART resolution
smart_all["sza"] = bacardi.sza.interp(time=smart_all.time)

# %% preprocess cosine correction
vnir_cos_cor = cosine_cor[cosine_cor["prop"] == "Fdw_VNIR"]
swir_cos_cor = cosine_cor[cosine_cor["prop"] == "Fdw_SWIR"]
pos_sel = vnir_cos_cor["position"] == "normal"
mean_k_cos = (vnir_cos_cor.loc[pos_sel].loc[:, "k_cos"].values + vnir_cos_cor.loc[~pos_sel].loc[:, "k_cos"].values) / 2
vnir_cos_cor = vnir_cos_cor.loc[pos_sel]
vnir_cos_cor["k_cos"] = mean_k_cos
pos_sel = swir_cos_cor["position"] == "normal"
mean_k_cos = (swir_cos_cor.loc[pos_sel].loc[:, "k_cos"].values + swir_cos_cor.loc[~pos_sel].loc[:, "k_cos"].values) / 2
swir_cos_cor = swir_cos_cor.loc[pos_sel]
swir_cos_cor["k_cos"] = mean_k_cos

# %% correct only one wavelength
pixel = smart_all["pixel"][500]
cos_cor = vnir_cos_cor.loc[vnir_cos_cor["pixel"] == pixel.values].loc[:, ["k_cos", "angle"]]
# create function of k_cos depending on angle
cos_cor_func = interp1d(cos_cor["angle"], cos_cor["k_cos"])
k_cos_500 = cos_cor_func(smart_all["sza"])

# %% correct pixel 500 for cosine response
wl500_cor = smart_all["Fdw"].sel(wavelength=slice(500, 600)) / np.repeat(np.expand_dims(k_cos_500, axis=1), 120, axis=1)
wl500_cor_int = wl500_cor.sum(dim="wavelength")

# %% plot corrected Fdw with libRadtran simulation
h.set_cb_friendly_colors()
fig, ax = plt.subplots()
ax.plot(smart_all.time, smart_all.Fdw.sel(wavelength=slice(500, 600)).sum(dim="wavelength"), label="SMART 500-600nm uncorrected")
ax.plot(wl500_cor_int.time, wl500_cor_int, label="SMART 500-600nm cosine corrected")
ax.plot(libradtran.time, libradtran.fdw, "--", label="libRadtran simulation")
h.set_xticks_and_xlabels(ax, pd.Timedelta((smart_all.time[-1] - smart_all.time[0]).values))
ax.grid()
ax.legend(loc=2)
plt.show()
plt.close()
