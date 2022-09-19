#!/usr/bin/env python
"""Case Study for RF17 2022-04-11

Arctic Dragon. Flight to the North with a west-east cross-section above and below the cirrus.

- prepare plots for supervision committee meeting

*author*: Johannes Röttenbacher
"""

# %% import modules
import pylim.helpers as h
from pylim.halo_ac3 import coordinates
from pylim import smart
import ac3airborne
from ac3airborne.tools.get_amsr2_seaice import get_amsr2_seaice
from ac3airborne.tools import flightphase
import os
import xarray as xr
import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
import holoviews as hv
from holoviews import opts
import cartopy
import cartopy.crs as ccrs

hv.extension("bokeh")

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
bacardi_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{halo_key}_R1.nc"

# flight tracks from halo-ac3 cloud
kwds = {'simplecache': dict(cache_storage='E:/HALO-AC3/cloud', same_names=True)}
credentials = {"user": os.environ.get("AC3_CLOUD_USER"), "password": os.environ.get("AC3_CLOUD_PASSWORD")}
cat = ac3airborne.get_intake_catalog()
ins = cat["HALO-AC3"]["HALO"]["GPS_INS"][f"HALO-AC3_HALO_{halo_key}"](storage_options=kwds, **credentials).to_dask()

# %% get flight segmentation and select below and above cloud section
meta = ac3airborne.get_flight_segments()["HALO-AC3"]["HALO"][f"HALO-AC3_HALO_{halo_key}"]
segments = flightphase.FlightPhaseFile(meta)
above_cloud, below_cloud = dict(), dict()
if "RF17" in halo_flight:
    above_cloud["start"] = segments.select("name", "high level 7")[0]["start"]
    above_cloud["end"] = segments.select("name", "high level 8")[0]["end"]
    below_cloud["start"] = segments.select("name", "high level 9")[0]["start"]
    below_cloud["end"] = segments.select("name", "high level 10")[0]["end"]
    above_slice = slice(above_cloud["start"], above_cloud["end"])
    below_slice = slice(below_cloud["start"], below_cloud["end"])
else:
    above_cloud["start"] = segments.select("name", "polygon pattern 1")[0]["start"]
    above_cloud["end"] = segments.select("name", "polygon pattern 1")[0]["parts"][-1]["start"]
    below_cloud["start"] = segments.select("name", "polygon pattern 2")[0]["start"]
    below_cloud["end"] = segments.select("name", "polygon pattern 2")[0]["end"]
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

# %% filter values which are not stabilized
smart_ds_filtered = smart_ds.where(smart_ds.stabilization_flag == 0)

# %% resample data to minutely resolution
ins = ins.resample(time="1Min").asfreq()
ins = ins.where(ins.time <= smart_ds_filtered.time)
smart_ds_filtered = smart_ds_filtered.resample(time="1Min").asfreq()
spectral_sim = spectral_sim.resample(time="1Min").asfreq()

# %% plotting aesthetics
h.set_cb_friendly_colors()
time_extend = pd.to_timedelta((ins.time[-1] - ins.time[0]).values)  # get time extend for x-axis labeling
time_extend_cs = below_cloud["end"] - above_cloud["start"]  # time extend for case study

# %% relation between simulation and measurement
relation = smart_ds_filtered.Fdw.sel(wavelength=500) / spectral_sim.fdw.sel(wavelength=500)

# %% viewing direction of halo: 0 = towards sun, 180 = away from sun
heading = ins.yaw * -1  # reverse turning direction of yaw angle
# replace negative values with positive angles to convert range from -180 - 180 to 0 - 360
heading = heading.where(heading > 0, heading + 360)
heading = (heading + 90)  # change from mathematical convention to meteorological convention (East = 0° -> North = 0°)
heading = heading.where(heading < 360, heading - 360)
viewing_dir = ins.saa - heading
viewing_dir = viewing_dir.where(viewing_dir > 0, viewing_dir + 360)

# %% merge information in data frame
df1 = viewing_dir.to_dataframe(name="viewing_dir")
df2 = relation.to_dataframe(name="relation")
df = df1.merge(df2, on="time")
df["sza"] = ins.sza
# df = df[df.relation > 0.7]
df = df.sort_values(by="viewing_dir")

# %% plot relation as function of viewing direction as polar plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.scatter(np.deg2rad(df["viewing_dir"]), df["relation"], label="0 = facing sun\n180 = facing away from sun")
ax.set_rmax(1.2)
ax.set_rticks([0.8, 1, 1.2])  # Less radial ticks
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi / 2)
ax.grid(True)

ax.set_title("Relation between SMART Fdw measurement and libRadtran simulation\n"
             "(500 nm)\n"
             " according to viewing direction of HALO with respect to the sun")
ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure)
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
figname = f"{plot_path}/{halo_flight}_inlet_directional_dependence.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot relation as time series
df = df.sort_values(by="time", axis=0)
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df.index, df.viewing_dir, label="Viewing direction", color="#88CCEE")
ax.scatter(df.index, df.relation * 100, label="Relation * 100", color="#CC6677")
ax.axhline(y=100, label="Relation = 1", color="#DDCC77")
ax.axhline(y=0, label="Towards sun", color="#117733")
ax.axhline(y=180, label="Away from sun", color="#332288")
ax.fill_between(df.index, 0, 1, where=((below_cloud["start"] < df.index) & (df.index < below_cloud["end"])),
                transform=ax.get_xaxis_transform(), label="below cloud", color="green", alpha=0.5)
ax.legend()
h.set_xticks_and_xlabels(ax, time_extend)
ax.grid()
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Viewing direction with respect to sun (deg)")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_viewing_direction_time_series.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot relation as function of SZA
fig, ax = plt.subplots(figsize=(10, 5))
plt.rcdefaults()
ax.scatter(df["sza"], df["relation"])
df_tmp = df[((below_cloud["start"] < df.index) & (df.index < below_cloud["end"]))]
ax.scatter(df_tmp["sza"], df_tmp["relation"], label="below cloud")
ax.grid()
ax.set_xlabel("Solar Zenith Angle (deg)")
ax.set_ylabel("Relation")
ax.set_title("Relation between SMART Fdw measurement and libRadtran simulation\n"
             "(500 nm)\n"
             " according to solar zenith angle")
ax.legend()
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_inlet_sza_dependence.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot simulation and measurement for different wavelengths
wavelength = 1450
h.set_cb_friendly_colors()
fig, ax = plt.subplots(figsize=(8, 4))
smart_ds_filtered.Fdw.sel(wavelength=wavelength).plot(ax=ax, label="SMART")
spectral_sim.fdw.sel(wavelength=wavelength).plot(ax=ax, label="libRadtran")
# smart_ds_filtered.Fdw.sel(wavelength=800).plot(ax=ax, label="SMART 800nm")
# spectral_sim.fdw.sel(wavelength=800).plot(ax=ax, label="libRadtran 800nm")
ax.grid()
h.set_xticks_and_xlabels(ax, time_extend)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Spectral Solar Irradiance (W$\,$m$^{-2}\,$nm$^{-1}$)")
ax.set_title(f"SMART - libRadtran comparison - {wavelength} nm")
ax.legend()
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_SMART-libRadtran_comp_{wavelength}.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot measurements of below and above cloud section
h.set_cb_friendly_colors()
smart_above = smart_ds_filtered.Fdw.sel(time=above_slice).mean(dim="time", skipna=True)
smart_below = smart_ds_filtered.Fdw.sel(time=below_slice).mean(dim="time", skipna=True)
fig, ax = plt.subplots(figsize=(8, 4))
# Fdw SMART
ax.plot(smart_above.wavelength, smart_above, label="SMART above")
ax.plot(smart_below.wavelength, smart_below, label="SMART below")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Spectral Downward Irradiance (W$\,$m$^{-2}\,$nm$^{-1}$)")
ax.set_title(f"Mean SMART spectra above and below cloud - {halo_flight}")
ax.grid()
ax.legend()
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_SMART_spectra_above_vs_below.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot BACARDI measurements of below and above cloud section
bacardi_ds_slice = bacardi_ds.sel(time=slice(above_cloud["start"], below_cloud["end"]))
fig, ax = plt.subplots(figsize=(8, 4))
for var in ["F_down_solar", "F_down_terrestrial", "F_up_solar", "F_up_terrestrial"]:
    ax.plot(bacardi_ds_slice[var].time, bacardi_ds_slice[var], label=f"{var}")

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

# %% calculate statistics from BACARDI measurements
bacardi_above, bacardi_below = dict(), dict()
for var in ["F_down_solar", "F_down_terrestrial", "F_up_solar", "F_up_terrestrial"]:
    bacardi_above[var] = bacardi_ds[var].sel(time=above_slice).mean(skipna=True)
    bacardi_below[var] = bacardi_ds[var].sel(time=below_slice).mean(skipna=True)

# %% calculate CRE with libradtran
cre_solar = (bacardi_ds.F_down_solar - bacardi_ds.F_up_solar) - (bb_sim_solar.fdw - bb_sim_solar.eup)
cre_terrestrial = (bacardi_ds.F_down_terrestrial - bacardi_ds.F_up_terrestrial) - (bb_sim_thermal.edn - bb_sim_thermal.eup)

# cut above and below cloud sections and plot it
cre_solar_above = cre_solar.sel(time=above_slice)
cre_terrestrial_above = cre_terrestrial.sel(time=above_slice)
cre_solar_below = cre_solar.sel(time=below_slice)
cre_terrestrial_below = cre_terrestrial.sel(time=below_slice)

# %% plot CRE above and below cloud
h.set_cb_friendly_colors()

fig, ax = plt.subplots(figsize=(8, 4))
# above
cre_solar_above.plot(ax=ax, label="CRE solar above")
cre_terrestrial_above.plot(ax=ax, label="CRE terrestrial above")
# below
cre_solar_below.plot(ax=ax, label="CRE solar below")
cre_terrestrial_below.plot(ax=ax, label="CRE terrestrial below")

ax.legend()
ax.grid()
h.set_xticks_and_xlabels(ax, time_extend_cs)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Cloud Radiative Effect (W$\,$m$^{-2}$)")
ax.set_title(f"BACARDI/libRadtran cloud radiative effect {halo_flight}")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_bacardi_libradtran_cre_above_below.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% calcalate cre statistics
cre_solar_above.mean(skipna=True)
cre_terrestrial_above.mean(skipna=True)
cre_solar_below.mean(skipna=True)
cre_terrestrial_below.mean(skipna=True)

# %% calculate CRE with BACARDI and sea ice simulation
bacardi_ds_above = bacardi_ds.sel(time=above_slice)
bacardi_ds_below = bacardi_ds.sel(time=below_slice)
Fdw_clear = bacardi_ds_above.F_down_solar.mean(skipna=True)
Fup_clear = bb_sim_solar_si.sel(time=above_slice).eup.mean(skipna=True)
Fdw_cloudy = bacardi_ds_below.F_down_solar.mean(skipna=True)
Fup_cloudy = bacardi_ds_below.F_up_solar.mean(skipna=True)
cre_solar = (Fdw_cloudy - Fup_cloudy) - (Fdw_clear - Fup_clear)

Fdw_clear = bacardi_ds_above.F_down_terrestrial.mean(skipna=True)
Fup_clear = bb_sim_thermal_si.sel(time=above_slice).eup.mean(skipna=True)
Fdw_cloudy = bacardi_ds_below.F_down_solar.mean(skipna=True)
Fup_cloudy = bacardi_ds_below.F_up_solar.mean(skipna=True)
cre_terrestrial = (Fdw_cloudy - Fup_cloudy) - (Fdw_clear - Fup_clear)

# %% plot CRE above and below cloud
h.set_cb_friendly_colors()

fig, ax = plt.subplots(figsize=(8, 4))
# above
cre_solar_above.plot(ax=ax, label="CRE solar above")
cre_terrestrial_above.plot(ax=ax, label="CRE terrestrial above")
# below
cre_solar_below.plot(ax=ax, label="CRE solar below")
cre_terrestrial_below.plot(ax=ax, label="CRE terrestrial below")

ax.legend()
ax.grid()
h.set_xticks_and_xlabels(ax, time_extend_cs)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Cloud Radiative Effect (W$\,$m$^{-2}$)")
ax.set_title(f"BACARDI/libRadtran cloud radiative effect {halo_flight}")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_bacardi_libradtran_cre_above_below.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# use measured F_up as real
# F_dw_meas = bacardi_ds.F_down_solar
# F_up_meas = bacardi_ds.F_up_solar
# cre_solar = (F_dw_meas - F_up_meas) - (bbr_sim.F_dw - F_up_meas)
# F_dw_meas = bacardi_ds.F_down_terrestrial.sel(time=bbr_sim_ter.index)
# F_up_meas = bacardi_ds.F_up_terrestrial.sel(time=bbr_sim_ter.index)
# cre_terrestrial = (F_dw_meas - F_up_meas) - (bbr_sim_ter.F_dw - F_up_meas)
# cre_net = cre_solar + cre_terrestrial
# apply rolling average
# cre_solar = cre_solar.rolling(time=2).mean(skipna=True)
# cre_terrestrial = cre_terrestrial.rolling(time=2).mean(skipna=True)
# cre_net = cre_net.rolling(time=2).mean(skipna=True)