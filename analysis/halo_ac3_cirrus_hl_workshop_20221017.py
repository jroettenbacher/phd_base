#!/usr/bin/env python
"""Prepare plots for CIRRUS-HL workshop 17. and 18. November 2022

- show case study from |haloac3|
- RF17 Arctic Dragon

*author*: Johannes Röttenbacher
"""

# %% import modules
import pylim.helpers as h
import pylim.meteorological_formulas as met
import ac3airborne
from ac3airborne.tools import flightphase
import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
from tqdm import tqdm

cm = 1 / 2.54
# %% set paths
campaign = "halo-ac3"
date = "20220411"
halo_key = "RF17"
halo_flight = f"HALO-AC3_{date}_HALO_{halo_key}"

plot_path = f"{h.get_path('plot', halo_flight, campaign)}/{halo_flight}"
smart_path = h.get_path("calibrated", halo_flight, campaign)
calibrated_file = f"HALO-AC3_HALO_SMART_spectral-irradiance-Fdw_{date}_{halo_key}_v1.0.nc"
libradtran_path = h.get_path("libradtran", halo_flight, campaign)
libradtran_spectral = f"HALO-AC3_HALO_libRadtran_clearsky_simulation_smart_spectral_{date}_{halo_key}.nc"
libradtran_bb_solar = f"HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_solar_{date}_{halo_key}.nc"
libradtran_bb_thermal = f"HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_thermal_{date}_{halo_key}.nc"
libradtran_bb_solar_si = f"HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_solar_si_{date}_{halo_key}.nc"
libradtran_bb_thermal_si = f"HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_thermal_si_{date}_{halo_key}.nc"
bahamas_path = h.get_path("bahamas", halo_flight, campaign)
bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{halo_key}_v1.nc"
bacardi_path = h.get_path("bacardi", halo_flight, campaign)
bacardi_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{halo_key}_R1_1s.nc"
ecrad_path = f"{h.get_path('ecrad', campaign=campaign)}/{date}"
ecrad_file = "ecrad_inout_20220411_v1_mean.nc"

# flight tracks from halo-ac3 cloud
kwds = {'simplecache': dict(cache_storage='E:/HALO-AC3/cloud', same_names=True)}
credentials = {"user": os.environ.get("AC3_CLOUD_USER"), "password": os.environ.get("AC3_CLOUD_PASSWORD")}
cat = ac3airborne.get_intake_catalog()
ins = cat["HALO-AC3"]["HALO"]["GPS_INS"][f"HALO-AC3_HALO_{halo_key}"](storage_options=kwds, **credentials).to_dask()

# %% get flight segmentation and select below and above cloud section
meta = ac3airborne.get_flight_segments()["HALO-AC3"]["HALO"][f"HALO-AC3_HALO_{halo_key}"]
segments = flightphase.FlightPhaseFile(meta)
above_cloud, below_cloud = dict(), dict()
above_cloud["start"] = segments.select("name", "high level 7")[0]["start"]
above_cloud["end"] = segments.select("name", "high level 8")[0]["end"]
below_cloud["start"] = segments.select("name", "high level 9")[0]["start"]
below_cloud["end"] = segments.select("name", "high level 10")[0]["end"]
above_slice = slice(above_cloud["start"], above_cloud["end"])
below_slice = slice(below_cloud["start"], below_cloud["end"])

# %% read in HALO smart data
smart_ds = xr.open_dataset(f"{smart_path}/{calibrated_file}")

# %% read in libRadtran simulation
spectral_sim = xr.open_dataset(f"{libradtran_path}/{libradtran_spectral}")
bb_sim_solar = xr.open_dataset(f"{libradtran_path}/{libradtran_bb_solar}")
bb_sim_thermal = xr.open_dataset(f"{libradtran_path}/{libradtran_bb_thermal}")
bb_sim_solar_si = xr.open_dataset(f"{libradtran_path}/{libradtran_bb_solar_si}")
bb_sim_thermal_si = xr.open_dataset(f"{libradtran_path}/{libradtran_bb_thermal_si}")

# %% read in BACARDI data
bacardi_ds = xr.open_dataset(f"{bacardi_path}/{bacardi_file}")

# %% read in ecrad data
ecrad_ds = xr.open_dataset(f"{ecrad_path}/{ecrad_file}")

# %% filter values which are not stabilized
smart_ds_filtered = smart_ds.where(smart_ds.stabilization_flag == 0)

# %% resample data to minutely resolution
ins = ins.resample(time="1Min").asfreq()
ins = ins.where(ins.time <= smart_ds_filtered.time)
smart_ds_filtered = smart_ds_filtered.resample(time="1Min").asfreq()
spectral_sim = spectral_sim.resample(time="1Min").asfreq()
bb_sim_solar = bb_sim_solar.resample(time="1Min").asfreq()
bb_sim_thermal = bb_sim_thermal.resample(time="1Min").asfreq()
bb_sim_solar_si = bb_sim_solar_si.resample(time="1Min").asfreq()
bb_sim_thermal_si = bb_sim_thermal_si.resample(time="1Min").asfreq()
bacardi_ds_res = bacardi_ds.resample(time="1Min").asfreq()

# %% plotting aesthetics
h.set_cb_friendly_colors()
time_extend = pd.to_timedelta((ins.time[-1] - ins.time[0]).values)  # get time extend for x-axis labeling
time_extend_cs = below_cloud["end"] - above_cloud["start"]  # time extend for case study

# %% plot BACARDI measurements of below and above cloud section
bacardi_ds_slice = bacardi_ds.sel(time=slice(above_cloud["start"], below_cloud["end"]))
labels = dict(F_down_solar=r"$F_{\downarrow, solar}$", F_down_terrestrial=r"$F_{\downarrow, terrestrial}$",
              F_up_solar=r"$F_{\uparrow, solar}$", F_up_terrestrial=r"$F_{\uparrow, terrestrial}$")
fig, ax = plt.subplots(figsize=(8, 4))
for var in ["F_down_solar", "F_down_terrestrial", "F_up_solar", "F_up_terrestrial"]:
    ax.plot(bacardi_ds_slice[var].time, bacardi_ds_slice[var], label=labels[var])

ax.axvline(x=above_cloud["end"], label="End above cloud section", color="#6699CC")
ax.axvline(x=below_cloud["start"], label="Start below cloud section", color="#888888")
ax.grid()
ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=3)
h.set_xticks_and_xlabels(ax, (below_cloud["end"] - above_cloud["start"]))
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Broadband Irradiance (W$\,$m$^{-2}$)")
ax.set_title("BACARDI broadband irradiance")
plt.tight_layout()
plt.subplots_adjust(bottom=0.25)
figname = f"{plot_path}/{halo_flight}_BACARDI_bb_irradiance_above_below.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% get height level of actual flight altitude in ecRad model, this determines only the index of the level
press_height = ecrad_ds[["pressure_hl", "temperature_hl"]]
p_array_list = list()
for time in tqdm(press_height.time):
    tmp = press_height.sel(time=time, drop=True)
    press_height_new = met.barometric_height(tmp["pressure_hl"], tmp["temperature_hl"])
    p_array = xr.DataArray(data=press_height_new[None, :], dims=["time", "half_level"],
                           coords={"half_level": (["half_level"], np.flip(tmp.half_level.values)),
                                   "time": np.array([time.values])},
                           name="pressure_height")
    p_array_list.append(p_array)

ecrad_ds["press_height"] = xr.merge(p_array_list).pressure_height
ecrad_ds["press_height"] = ecrad_ds["press_height"].where(~np.isnan(ecrad_ds["press_height"]), 80000)
ins_tmp = ins.sel(time=ecrad_ds.time, method="nearest")
ecrad_timesteps = len(ecrad_ds.time)
aircraft_height_level = np.zeros(ecrad_timesteps)

for i in tqdm(range(ecrad_timesteps)):
    aircraft_height_level[i] = h.arg_nearest(ecrad_ds["press_height"][i, :].values, ins_tmp.alt[i].values)

aircraft_height_level = aircraft_height_level.astype(int)
height_level_da = xr.DataArray(aircraft_height_level, dims=["time"], coords={"time": ecrad_ds.time})
aircraft_height = ecrad_ds["press_height"].isel(half_level=height_level_da)

# %% prepare dataset for plotting
ecrad_plot = ecrad_ds["flux_dn_sw"]
new_z = ecrad_ds["press_height"].mean(dim="time") / 1000
ecrad_plot_new_z = list()
for t in tqdm(ecrad_plot.time):
    tmp_plot = ecrad_plot.sel(time=t)
    tmp_plot = tmp_plot.assign_coords(half_level=ecrad_ds["press_height"].sel(time=t, drop=True).values / 1000)
    tmp_plot = tmp_plot.rename(half_level="height")
    tmp_plot = tmp_plot.interp(height=new_z.values)
    ecrad_plot_new_z.append(tmp_plot)

ecrad_plot = xr.concat(ecrad_plot_new_z, dim="time")
aircraft_height_plot = aircraft_height / 1000

# %% plot aircraft track through IFS model
plt.rcdefaults()
h.set_cb_friendly_colors()
plt.rc('font', size=6)
cmap = cmr.get_sub_cmap(cmr.sunburst, 0.25, 1)
_, ax = plt.subplots(figsize=(13 * cm, 7 * cm))
# ecrad output broadband solar downward flux
ecrad_plot.plot(x="time", y="height", cmap=cmap, ax=ax, robust=True,
                cbar_kwargs={"pad": 0.04, "label": f"Broadband Shortwave Downward\n Irradiance (W$\,$m$^{-2}$)"})
# aircraft altitude through model
aircraft_height_plot.plot(x="time", color="k", ax=ax, label="HALO altitude")

ax.set_ylabel("Altitude (km)")
ax.set_xlabel("Time (UTC)")
ax.set_ylim(0, 12)
h.set_xticks_and_xlabels(ax, time_extend)
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_ecrad_Fdw.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()
# %% plot ecRad broadband fluxes above and below cloud
ecrad_ds_slice = ecrad_ds.isel(half_level=height_level_da)
ecrad_ds_slice = ecrad_ds_slice.sel(time=slice(above_cloud["start"], below_cloud["end"]))
labels = dict(flux_dn_sw=r"$F_{\downarrow, solar}$", flux_dn_lw=r"$F_{\downarrow, terrestrial}$",
              flux_up_sw=r"$F_{\uparrow, solar}$", flux_up_lw=r"$F_{\uparrow, terrestrial}$")
plt.rc("font", size=12)
_, ax = plt.subplots(figsize=(22 * cm, 11 * cm))
for var in ["flux_dn_sw", "flux_dn_lw", "flux_up_sw"]:
    ax.plot(ecrad_ds_slice[var].time, ecrad_ds_slice[var], label=labels[var])

ax.axvline(x=above_cloud["end"], label="End above cloud section", color="#6699CC")
ax.axvline(x=below_cloud["start"], label="Start below cloud section", color="#888888")
ax.grid()
ax.legend(bbox_to_anchor=(0.2, 0), loc="lower left", bbox_transform=fig.transFigure, ncol=3)
h.set_xticks_and_xlabels(ax, (below_cloud["end"] - above_cloud["start"]))
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Broadband Irradiance (W$\,$m$^{-2}$)")
ax.set_title("ecRad broadband irradiance")
plt.tight_layout()
plt.subplots_adjust(bottom=0.28)
figname = f"{plot_path}/{halo_flight}_ecRad_bb_irradiance_above_below.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot 2D histogram of below cloud solar downward irradiance measurements
bacardi_plot = bacardi_ds_res.sel(time=below_slice)
ecrad_plot = ecrad_ds.isel(half_level=height_level_da).sel(time=below_slice)
_, ax = plt.subplots(figsize=(12 * cm, 11 * cm))
hist = ax.hist2d(bacardi_plot["F_down_solar"], ecrad_plot["flux_dn_sw"], bins=15, cmap=cmr.rainforest.reversed())
ax.axline((0, 0), slope=1, color="k", lw=2, transform=ax.transAxes)
plt.colorbar(hist[3], label="Counts", shrink=0.8)
ax.set_ylim((130, 240))
ax.set_xlim((130, 240))
ax.set_aspect('equal')
ax.set_xlabel("BACARDI Irradiance  (W$\,$m$^-{2}$)")
ax.set_ylabel("ecRad Irradiance  (W$\,$m$^-{2}$)")
ax.set_title("Solar Downward Irradiance")
plt.tight_layout()
plt.savefig(f"{plot_path}/{halo_flight}_Fdw_solar_bacardi_vs_ecrad.png", dpi=300)
plt.close()

# %% plot 2D histogram of below cloud terrestrial downward measurements
bacardi_plot = bacardi_ds_res.sel(time=below_slice)
ecrad_plot = ecrad_ds.isel(half_level=height_level_da).sel(time=below_slice)
_, ax = plt.subplots(figsize=(12 * cm, 11 * cm))
hist = ax.hist2d(bacardi_plot["F_down_terrestrial"], ecrad_plot["flux_dn_lw"], bins=15, cmap=cmr.rainforest.reversed())
ax.axline((0, 0), slope=1, color="k", lw=2, transform=ax.transAxes)
plt.colorbar(hist[3], label="Counts", shrink=0.8)
ax.set_ylim((80, 130))
ax.set_xlim((80, 130))
ax.set_aspect('equal')
ax.set_xlabel("BACARDI Irradiance  (W$\,$m$^-{2}$)")
ax.set_ylabel("ecRad Irradiance  (W$\,$m$^-{2}$)")
ax.set_title("Terrestrial Downward Irradiance")
plt.tight_layout()
plt.savefig(f"{plot_path}/{halo_flight}_Fdw_terrestrial_bacardi_vs_ecrad.png", dpi=300)
plt.close()

# %% plot 2D histogram of below cloud solar upward irradiance measurements
bacardi_plot = bacardi_ds_res.sel(time=below_slice)
ecrad_plot = ecrad_ds.isel(half_level=height_level_da).sel(time=below_slice)
_, ax = plt.subplots(figsize=(12 * cm, 11 * cm))
hist = ax.hist2d(bacardi_plot["F_up_solar"], ecrad_plot["flux_up_sw"], bins=15, cmap=cmr.rainforest.reversed())
ax.axline((0, 0), slope=1, color="k", lw=2, transform=ax.transAxes)
plt.colorbar(hist[3], label="Counts", shrink=0.8)
ax.set_ylim((100, 170))
ax.set_xlim((100, 170))
ax.set_aspect('equal')
ax.set_xlabel("BACARDI Irradiance  (W$\,$m$^-{2}$)")
ax.set_ylabel("ecRad Irradiance  (W$\,$m$^-{2}$)")
ax.set_title("Solar Upward Irradiance")
plt.tight_layout()
plt.savefig(f"{plot_path}/{halo_flight}_Fup_solar_bacardi_vs_ecrad.png", dpi=300)
plt.close()
