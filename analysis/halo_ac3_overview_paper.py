#!/usr/bin/env python
"""Figures for the HALO-(AC)³ overview paper by Wendisch et al.

*author*: Johannes Röttenbacher
"""

# %% import modules
import pylim.helpers as h
import pylim.halo_ac3 as meta
import ac3airborne
import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# %% set paths and filenames
campaign = "halo-ac3"
key = "RF08"
flight = meta.flight_names[key]
date = flight[9:17]
# plot_path = f"{h.get_path('plot', campaign=campaign)}/../manuscripts/2023_HALO-AC3_overview"
plot_path = f"C:/Users/Johannes/Documents/Doktor/manuscripts/2023_halo-ac3_overview"
bacardi_path = h.get_path("bacardi", flight, campaign)
libradtran_path = h.get_path("libradtran", flight, campaign)
ecrad_path = f"{h.get_path('ecrad', flight, campaign)}/{date}"
wales_path = f"{h.get_path('wales', flight, campaign)}"

# filenames
bacardi_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1.nc"
libradtran_file = f"HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_thermal_{date}_{key}.nc"
ecrad_file = f"ecrad_merged_inout_{date}_v1_mean.nc"
wales_file = f"HALO-AC3_HALO_WALES_wv_{date}_{key}_V2.0.nc"

# set options and credentials for HALO-AC3 cloud and intake catalog
kwds = {'simplecache': dict(same_names=True)}
credentials = {"user": os.environ.get("AC3_CLOUD_USER"), "password": os.environ.get("AC3_CLOUD_PASSWORD")}
cat = ac3airborne.get_intake_catalog()

# %% read in data
bacardi_ds = xr.open_dataset(f"{bacardi_path}/{bacardi_file}")
libradtran_ds = xr.open_dataset(f"{libradtran_path}/{libradtran_file}")
wales_ds = xr.open_dataset(f"{wales_path}/{wales_file}")

# %% calculate BACARDI net irradiance
bacardi_ds["net_terrestrial"] = bacardi_ds.F_down_terrestrial - bacardi_ds.F_up_terrestrial
bacardi_ds["net_solar"] = bacardi_ds.F_down_solar - bacardi_ds.F_up_solar
bacardi_ds["net_total"] = bacardi_ds.net_solar + bacardi_ds.net_terrestrial


# %% select BACARDI data to match libRadtran time
bacardi_ds_sel = bacardi_ds.sel(time=libradtran_ds.time)

# %% calculate radiative effect from BACARDI/libRadtran
bacardi_ds_sel["cre_terrestrial"] = bacardi_ds_sel.net_terrestrial - (libradtran_ds.edn - libradtran_ds.eup)
# bacardi_ds_sel["cre_solar"] = bacardi_ds_sel.net_solar - (libradtran_ds.edn - libradtran_ds.eup)

# %% plot BACARDI data
bacardi_plot = bacardi_ds_sel.sel(time=slice(pd.to_datetime("2022-03-21 11:10"), pd.to_datetime("2022-03-21 11:20")))
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(bacardi_plot.time, bacardi_plot.net_terrestrial, label="BACARDI net thermal-infrared irradiance")
ax.plot(bacardi_plot.time, bacardi_plot.net_solar, label="BACARDI net solar irradiance")
ax.plot(bacardi_plot.time, bacardi_plot.cre_terrestrial, label="BACARDI/libRadtran CRE thermal-infrared")
ax.plot(bacardi_plot.time, bacardi_plot.F_up_terrestrial, label="BACARDI F$_{up}$ thermal-infrared")
ax.plot(bacardi_plot.time, bacardi_plot.F_down_terrestrial, label="BACARDI F$_{down}$ thermal-infrared")
ax.plot(bacardi_plot.time, bacardi_plot.F_up_solar, label="BACARDI $F_{up}$ solar")
ax.plot(bacardi_plot.time, bacardi_plot.F_down_solar, label="BACARDI F$_{down}$ solar")
h.set_xticks_and_xlabels(ax, pd.to_timedelta(bacardi_plot.time[-1].to_numpy() - bacardi_plot.time[0].to_numpy()))
ax.set(xlabel="Time (UTC)", ylabel="Irradiance (W$\,$m$^{-2}$)")
ax.grid()
ax.legend()
figname = f"{plot_path}/HALO-AC3_{date}_HALO_{key}_BACARDI.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot BACARDI data for two panel plot with VELOX
cbc = h.get_cb_friendly_colors("petroff_6")
plt.rc("font", size=16.5)
begin, end = pd.to_datetime("2022-03-21 11:13:45"), pd.to_datetime("2022-03-21 11:18:45")
bacardi_plot = bacardi_ds.sel(time=slice(begin, end))
_, axs = plt.subplots(2, 1, figsize=(30*h.cm, 14*h.cm), layout="constrained")
# axs[0].plot(bacardi_plot.time, bacardi_plot.net_solar,
#         label="$F_{\mathrm{net, solar}}$", color=cbc[-1])
axs[0].plot(bacardi_plot.time, bacardi_plot.F_down_solar,
        label="$F_{\\uparrow,\mathrm{solar}}$", color=cbc[-2])
axs[1].plot(bacardi_plot.time, bacardi_plot.net_terrestrial,
        label="$F_{\mathrm{net, thermal-infrared}}$", color=cbc[1])
# ax.plot(bacardi_plot.time, bacardi_plot.net_total,
#         label="$F_{\mathrm{net, total}}$", color=cbc[2])
for ax in axs:
    h.set_xticks_and_xlabels(ax, pd.to_timedelta(4, "Minutes"))
    ax.set(
           ylabel="Irradiance (W$\,$m$^{-2}$)",
           xlim=(begin, end),)
    ax.grid()
    ax.legend(loc=1, fontsize=14)
axs[1].set(xlabel="Time (UTC)")
figname = f"{plot_path}/HALO-AC3_{date}_HALO_{key}_BACARDI.png"
# plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% export BACARDI data
bacardi_ds_export = bacardi_ds.sel(time=slice(pd.to_datetime("2022-03-21 11:10"), pd.to_datetime("2022-03-21 11:20")))
bacardi_ds_export = bacardi_ds_export.to_pandas()
bacardi_ds_export.to_csv(f"{plot_path}/HALO-AC3_{date}_HALO_{key}_BACARDI.csv")
