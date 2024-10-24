#!/usr/bin/env python
"""Case Study for RF18 2022-04-12

Flight to the North with to Pentagrams above and below cloud and dropsondes

*author*: Johannes Röttenbacher
"""

# %% import modules
import pylim.helpers as h
import pylim.meteorological_formulas as met
import pylim.halo_ac3 as meta
from pylim import smart, reader, ecrad
import ac3airborne
from ac3airborne.tools.get_amsr2_seaice import get_amsr2_seaice
from ac3airborne.tools import flightphase
import sys

sys.path.append('./larda')
from larda.pyLARDA.spec2mom_limrad94 import despeckle
import os
import xarray as xr
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib import patheffects, colors
from matplotlib.patches import Patch
from matplotlib.collections import LineCollection
import cartopy
import cartopy.crs as ccrs
from tqdm import tqdm
import cmasher as cmr

cm = 1 / 2.54
cbc = h.get_cb_friendly_colors()
matplotlib.use('module://backend_interagg')

# %% set paths
campaign = "halo-ac3"
halo_key = "RF18"
halo_flight = meta.flight_names[halo_key]
date = halo_flight[9:17]

plot_path = f"{h.get_path('plot', halo_flight, campaign)}/{halo_flight}"
smart_path = h.get_path("calibrated", halo_flight, campaign)
calibrated_file = f"HALO-AC3_HALO_SMART_spectral-irradiance-Fdw_{date}_{halo_key}_v1.0.nc"
libradtran_path = h.get_path("libradtran", halo_flight, campaign)
libradtran_exp_path = h.get_path("libradtran_exp", campaign=campaign)
libradtran_spectral = f"HALO-AC3_HALO_libRadtran_clearsky_simulation_smart_spectral_{date}_{halo_key}.nc"
libradtran_bb_solar = f"HALO-AC3_HALO_libRadtran_simulation_seaice_2_solar_{date}_{halo_key}.nc"
libradtran_bb_thermal = f"HALO-AC3_HALO_libRadtran_simulation_seaice_2_thermal_{date}_{halo_key}.nc"
libradtran_bb_solar1 = f"HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_solar_si_{date}_{halo_key}.nc"
libradtran_bb_thermal1 = f"HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_thermal_si_{date}_{halo_key}.nc"
bahamas_path = h.get_path("bahamas", halo_flight, campaign)
bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{halo_key}_v1_1s.nc"
# bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{halo_key}_v1.nc"
bacardi_path = h.get_path("bacardi", halo_flight, campaign)
bacardi_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{halo_key}_R1_1s.nc"
# bacardi_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{halo_key}_R1.nc"
# era5_path = h.get_path("era5", campaign=campaign)
# trajectory_path = f"{h.get_path('trajectories', campaign=campaign)}/selection_CC_and_altitude"
ifs_path = f"{h.get_path('ifs', campaign=campaign)}"
ecrad_path = f"{h.get_path('ecrad', campaign=campaign)}/{date}"
ecrad_file = f"ecrad_merged_inout_{date}_v1_mean.nc"
# radar_path = h.get_path("hamp_mira", halo_flight, campaign)
# radar_file = f"radar_{date}_v1.6.nc"
# lidar_path = h.get_path("wales", halo_flight, campaign)
# bsrgl_file = "HALO-AC3_HALO_WALES_bsrgl_{date}_{halo_key}_V1.nc"
era5_path = "E:/HALO-AC3/10_ERA5"

# set up metadata for access to HALO-AC3 cloud
kwds = {'simplecache': dict(same_names=True)}
credentials = {"user": os.environ.get("AC3_CLOUD_USER"), "password": os.environ.get("AC3_CLOUD_PASSWORD")}
cat = ac3airborne.get_intake_catalog()["HALO-AC3"]["HALO"]

# %% read in data from HALO-AC3 cloud
ins = cat["GPS_INS"][f"HALO-AC3_HALO_{halo_key}"](storage_options=kwds, **credentials).to_dask()
dropsonde_ds = cat["DROPSONDES_GRIDDED"][f"HALO-AC3_HALO_{halo_key}"](storage_options=kwds, **credentials).to_dask()
dropsonde_ds["alt"] = dropsonde_ds.alt / 1000  # convert altitude to km

# %% get flight segmentation and select below and above cloud section
segmentation = ac3airborne.get_flight_segments()["HALO-AC3"]["HALO"][f"HALO-AC3_HALO_{halo_key}"]
segments = flightphase.FlightPhaseFile(segmentation)
above_cloud, below_cloud = dict(), dict()
above_cloud["start"] = segments.select("name", "polygon pattern 1")[0]["start"]
above_cloud["end"] = segments.select("name", "polygon pattern 1")[0]["parts"][-1]["start"]
below_cloud["start"] = segments.select("name", "polygon pattern 2")[0]["start"]
below_cloud["end"] = segments.select("name", "polygon pattern 2")[0]["end"]
above_slice = slice(above_cloud["start"], above_cloud["end"])
below_slice = slice(below_cloud["start"], below_cloud["end"])
case_slice = slice(above_cloud["start"], below_cloud["end"])

# %% read in HALO smart data
smart_ds = xr.open_dataset(f"{smart_path}/{calibrated_file}")
# smart_std = smart_ds.resample(time="1Min").std()
# smart_std.to_netcdf(f"{smart_path}/{calibrated_file.replace('.nc', '_std.nc')}")
smart_std = xr.open_dataset(f"{smart_path}/{calibrated_file.replace('.nc', '_std.nc')}")

# %% read in BACARDI and BAHAMAS data and resample to 1 sec
bacardi_ds = xr.open_dataset(f"{bacardi_path}/{bacardi_file}")
bahamas_ds = xr.open_dataset(f"{bahamas_path}/{bahamas_file}")
sza = bacardi_ds.sza
bahamas_unfiltered = bahamas_ds[["IRS_ALT", "RELHUM", "TS", "MIXRATIO", "MIXRATIOV", "PS", "QC", "IRS_PHI", "IRS_THE"]]
bahamas_unfiltered["RHice"] = met.relative_humidity_water_to_relative_humidity_ice(bahamas_unfiltered["RELHUM"],
                                                                                   bahamas_unfiltered["TS"] - 273.16)
# bahamas_ds = reader.read_bahamas(f"{bahamas_path}/{bahamas_file}")
# bacardi_ds = bacardi_ds.resample(time="1S").mean()
# bahamas_ds = bahamas_ds.resample(time="1S").mean()
# bacardi_ds.to_netcdf(f"{bacardi_path}/{bacardi_file.replace('.nc', '_1s.nc')}")
# bahamas_ds.to_netcdf(f"{bahamas_path}/{bahamas_file.replace('.nc', '_1s.nc')}")
print("")

# %% read in libRadtran simulation
spectral_sim = xr.open_dataset(f"{libradtran_path}/{libradtran_spectral}")
bb_sim_solar = xr.open_dataset(f"{libradtran_exp_path}/{libradtran_bb_solar}")
bb_sim_thermal = xr.open_dataset(f"{libradtran_exp_path}/{libradtran_bb_thermal}")
# simulations only to 3600 nm
bb_sim_solar1 = xr.open_dataset(f"{libradtran_exp_path}/{libradtran_bb_solar1}")
bb_sim_thermal1 = xr.open_dataset(f"{libradtran_exp_path}/{libradtran_bb_thermal1}")

# %% read in ecrad data
ecrad_ds = xr.open_dataset(f"{ecrad_path}/{ecrad_file}")
# replace numeric nan values with nan
ecrad_ds["re_ice"] = ecrad_ds.re_ice.where(ecrad_ds.re_ice != 5.196162e-05, np.nan)
ecrad_ds["re_liquid"] = ecrad_ds.re_liquid.where(ecrad_ds.re_liquid != 4.000001e-06, np.nan)

# %% filter values which are not stabilized or which exceeded certain motion threshold
stabbi_filter = smart_ds.stabilization_flag == 0
smart_ds_filtered = smart_ds.where(stabbi_filter)
smart_std = smart_std.where(stabbi_filter)

roll_center = np.abs(bahamas_ds["IRS_PHI"].median())  # -> 0.06...
roll_threshold = 0.5
# pitch is not centered on 0 thus we need to calculate the difference to the center and compare that to the threshold
# the pitch center changes during flight due to loss of weight (fuel) and speed
# use the 30-minute rolling median as center to calculate the threshold
rolling_pitch = bahamas_ds["IRS_THE"].rolling(time=1800, center=True, min_periods=60)
pitch_center = rolling_pitch.median()
pitch_threshold = rolling_pitch.std().median().round(2)
# True -> keep value, False -> drop value (Nan)
roll_filter = np.abs(bahamas_ds["IRS_PHI"]) < roll_threshold
pitch_filter = np.abs(bahamas_ds["IRS_THE"] - pitch_center) < pitch_threshold
motion_filter = roll_filter & pitch_filter
# bacardi_ds = bacardi_ds.where(motion_filter)
# bahamas_ds = bahamas_ds.where(motion_filter)

# %% filter values from bacardi which correspond to turns or descents
selected_segments = segments.findSegments(above_cloud["start"].strftime('%Y-%m-%d %H:%M:%S'),
                                          below_cloud["end"].strftime('%Y-%m-%d %H:%M:%S'))
starts, ends = list(), list()
for dic in selected_segments:
    if "short_turn" in dic["kinds"] or "roll_maneuver" in dic["kinds"] or "large_descent" in dic["kinds"]:
        starts.append(dic["start"])
        ends.append(dic["end"])
starts, ends = pd.to_datetime(starts), pd.to_datetime(ends)
for i in range(len(starts)):
    sel_time = (bacardi_ds.time > starts[i]) & (bacardi_ds.time < ends[i])
    bacardi_ds = bacardi_ds.where(~sel_time)

# %% cut data to smart time
time_slice = slice(smart_ds.time[0].values, smart_ds.time[-1].values)
bahamas_ds = bahamas_ds.sel(time=time_slice)
# ecrad_ds = ecrad_ds.sel(time=time_slice)
bacardi_ds = bacardi_ds.sel(time=time_slice)

# %% calculate albedo from BACARDI and libRadtran and from ecRad
bacardi_ds["albedo_solar"] = bacardi_ds["F_up_solar"] / bacardi_ds["F_down_solar"]
bacardi_ds["albedo_solar_cls"] = bacardi_ds["F_up_solar"] / bacardi_ds["F_down_solar_sim"]
bacardi_ds["albedo_terrestrial"] = bacardi_ds["F_up_terrestrial"] / bacardi_ds["F_down_terrestrial"]
# ecrad_ds["albedo_sw"] = ecrad_ds["flux_up_sw"] / ecrad_ds["flux_dn_sw"]
# ecrad_ds["albedo_sw_cls"] = ecrad_ds["flux_up_sw"] / ecrad_ds["flux_dn_sw_clear"]
# ecrad_ds["albedo_lw"] = ecrad_ds["flux_up_lw"] / ecrad_ds["flux_dn_lw"]
# ecrad_ds["albedo_lw_cls"] = ecrad_ds["flux_up_sw"] / ecrad_ds["flux_dn_lw_clear"]
print("")

# %% calculate transmissivity BACARDI/libRadtran and ecRad
bacardi_ds["transmissivity_solar"] = bacardi_ds["F_down_solar"] / bacardi_ds["F_down_solar_sim"]
bacardi_ds["transmissivity_terrestrial"] = bacardi_ds["F_down_terrestrial"] / bb_sim_thermal.edn
# ecrad_ds["transmissivity_sw"] = ecrad_ds["flux_dn_sw"] / ecrad_ds["flux_dn_sw_clear"]
# ecrad_ds["transmissivity_lw"] = ecrad_ds["flux_dn_lw"] / ecrad_ds["flux_dn_lw_clear"]
print("")

# %% calculate standard deviation of transmissivity below cloud
bacardi_ds["transmissivity_solar_std"] = bacardi_ds["transmissivity_solar"].sel(time=below_slice).std()

# %% resample data to minutely resolution
ins_res = ins.resample(time="1Min").asfreq()
smart_ds = smart_ds.resample(time="1Min").asfreq()
spectral_sim = spectral_sim.resample(time="1Min").asfreq()
bb_sim_solar = bb_sim_solar.resample(time="1Min").asfreq()
bb_sim_thermal = bb_sim_thermal.resample(time="1Min").asfreq()
bb_sim_solar1 = bb_sim_solar1.resample(time="1Min").asfreq()
bb_sim_thermal1 = bb_sim_thermal1.resample(time="1Min").asfreq()
bacardi_ds_res = bacardi_ds.resample(time="1Min").mean().sel(time=time_slice)
# bacardi_std = bacardi_ds.resample(time="1Min").std()
# bacardi_std.to_netcdf(f"{bacardi_path}/{bacardi_file.replace('.nc', '_std.nc')}")
bacardi_std = xr.open_dataset(f"{bacardi_path}/HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{halo_key}_R1_1s_std.nc")

# %% band SMART and libRadtran data to ecRad bands
# nr_bands = len(h.ecRad_bands)
# smart_banded = np.empty((nr_bands, smart_ds.dims["time"]))
# ifs_sim_banded = np.empty((nr_bands, ifs_sim.dims["time"]))
# ifs_sim_banded2 = np.empty((nr_bands, ifs_sim.dims["time"]))
# for i, band in enumerate(h.ecRad_bands):
#     wl1 = h.ecRad_bands[band][0]
#     wl2 = h.ecRad_bands[band][1]
#     smart_banded[i, :] = smart_ds.Fdw_cor.sel(wavelength=slice(wl1, wl2)).integrate(coord="wavelength")
#     ifs_sim_banded[i, :] = ifs_sim.fdw.sel(wavelength=slice(wl1, wl2)).integrate(coord="wavelength")
#     ifs_sim_banded2[i, :] = ifs_sim.fdw.sel(wavelength=slice(wl1, wl2)).sum(dim="wavelength")
#
# smart_banded = xr.DataArray(smart_banded, coords={"ecrad_band": range(1, 15), "time": smart_ds.time}, name="Fdw_cor")
# ifs_sim_banded = xr.DataArray(ifs_sim_banded, coords={"ecrad_band": range(1, 15), "time": ifs_sim.time}, name="fdw")
# ifs_sim_banded2 = xr.DataArray(ifs_sim_banded2, coords={"ecrad_band": range(1, 15), "time": ifs_sim.time}, name="fdw")
print("")

# %% create a boolean mask for above cloud time steps in 1 min data for whole flight
# end_ascend = pd.to_datetime(segments.select("name", "high level 1")[0]["start"])
# end_ascend2 = pd.to_datetime(segments.select("name", "high level 5")[0]["start"])
# t = pd.to_datetime(bacardi_ds_res.time)
# above_sel = xr.DataArray(((t > end_ascend) & (t < above_cloud["end"])) | (t > end_ascend2),
#                          coords={"time": bacardi_ds_res.time})

# %% create boolean array for cirrus times for minutely BACARDI data
# bt = bacardi_ds_res.time
# hc_st = cloudy_times["starts"]
# hc_end = cloudy_times["ends"]
# lsm = ecrad_ds.LSM < 0.01  # get a boolean land sea mask (0 = sea -> True = sea)
# ci_mask = ecrad_ds.CI > 0.8  # get a boolean sea ice mask (1 = sea ice -> True = ci)
# mciz_mask = (ecrad_ds.CI > 0.15) & (ecrad_ds.CI < 0.8)
# cirrus_only = ((hc_st[1] < bt) & (hc_end[1] > bt)) | ((hc_st[2] < bt) & (hc_end[2] > bt))
# cirrus_over_ci = cirrus_only & ci_mask
# cirrus_over_sea = ((cirrus_only & ~ci_mask) & ~mciz_mask) & lsm
print("")

# %% plotting variables
time_extend = pd.to_timedelta((ins.time[-1] - ins.time[0]).values)  # get time extend for x-axis labeling
time_extend_cs = below_cloud["end"] - above_cloud["start"]  # time extend for case study
time_extend_ac = above_cloud["end"] - above_cloud["start"]
time_extend_bc = below_cloud["end"] - below_cloud["start"]
plt.rcdefaults()
h.set_cb_friendly_colors()
plt.rc("font", size=12)

# %% calculate pressure height for ecrad data
if "press_height_full" not in ecrad_ds:
    ecrad_ds = ecrad.calculate_pressure_height(ecrad_ds)

# %% get height level of actual flight altitude in ecRad model on half levels
ins_tmp = ins_res.sel(time=ecrad_ds.time, method="nearest")
ecrad_timesteps = len(ecrad_ds.time)
aircraft_height_level = np.zeros(ecrad_timesteps)

for i in tqdm(range(ecrad_timesteps)):
    aircraft_height_level[i] = h.arg_nearest(ecrad_ds["press_height_hl"][i, :].values, ins_tmp.alt[i].values)

aircraft_height_level = aircraft_height_level.astype(int)
height_level_da = xr.DataArray(aircraft_height_level, dims=["time"], coords={"time": ecrad_ds.time})
aircraft_height = ecrad_ds["press_height_hl"].isel(half_level=height_level_da)

# %% get height level of actual flight altitude in ecRad model on full levels
aircraft_height_level_full = np.zeros(ecrad_timesteps)

for i in tqdm(range(ecrad_timesteps)):
    aircraft_height_level_full[i] = h.arg_nearest(ecrad_ds["press_height_full"][i, :].values, ins_tmp.alt[i].values)

aircraft_height_level_full = aircraft_height_level_full.astype(int)
height_level_da_full = xr.DataArray(aircraft_height_level_full, dims=["time"], coords={"time": ecrad_ds.time})
aircraft_height_full = ecrad_ds["press_height_full"].isel(level=height_level_da_full)


# %% calculate cloud radiative effect from BACARDI
bacardi_ds_res["cre_solar"] = (bacardi_ds_res.F_down_solar - bacardi_ds_res.F_up_solar) - (
        bb_sim_solar.fdw - bb_sim_solar.eup)
bacardi_ds_res["cre_terrestrial"] = (bacardi_ds_res.F_down_terrestrial - bacardi_ds_res.F_up_terrestrial) - (
        bb_sim_thermal.edn - bb_sim_thermal.eup)
bacardi_ds_res["cre_total"] = bacardi_ds_res["cre_solar"] + bacardi_ds_res["cre_terrestrial"]

# %% calculate broadband cloud radiative effect from ecRad
ecrad_ds["cre_sw"] = (ecrad_ds.flux_dn_sw - ecrad_ds.flux_up_sw) - (ecrad_ds.flux_dn_sw_clear
                                                                    - ecrad_ds.flux_up_sw_clear)
ecrad_ds["cre_lw"] = (ecrad_ds.flux_dn_lw - ecrad_ds.flux_up_lw) - (ecrad_ds.flux_dn_lw_clear
                                                                    - ecrad_ds.flux_up_lw_clear)
ecrad_ds["cre_total"] = ecrad_ds["cre_sw"] + ecrad_ds["cre_lw"]

# %% plotting dictionaries and lists for BACARDI and ecrad
labels = dict(F_down_solar=r"$F_{\downarrow, solar}$", F_down_terrestrial=r"$F_{\downarrow, terrestrial}$",
              F_up_solar=r"$F_{\uparrow, solar}$", F_up_terrestrial=r"$F_{\uparrow, terrestrial}$")

# prepare metadata for comparing ecRad and BACARDI
titles = ["Solar Downward Irradiance", "Terrestrial Downward Irradiance", "Solar Upward Irradiance",
          "Terrestrial Upward Irradiance"]
names = ["Fdw_solar", "Fdw_terrestrial", "Fup_solar", "Fup_terrestrial"]
bacardi_vars = ["F_down_solar", "F_down_terrestrial", "F_up_solar", "F_up_terrestrial"]
ecrad_vars = ["flux_dn_sw", "flux_dn_lw", "flux_up_sw", "flux_up_lw"]

# %% calculate difference between BACARDI and ecRad
# bacardi_plot = bacardi_ds_res.where(above_sel)
# ecrad_plot = ecrad_ds.isel(half_level=height_level_da).where(above_sel)
# ecrad_bacardi = xr.Dataset()
# for x, y in zip(bacardi_vars, ecrad_vars):
#     bacardi_ecrad = (bacardi_plot[x] - ecrad_plot[y]).sel(time=above_slice)
#     ecrad_bacardi[x] = ecrad_plot[y] - bacardi_plot[x]
#     print(f"BACARDI {x} - ecRad {y} = {bacardi_ecrad.mean():.2f} Wm^-2")

# %% plot BAHAMAS roll angle for above cloud section
# plot_ds = bahamas_unfiltered.sel(time=above_slice)
# plot_ds1 = bahamas_unfiltered.where(pitch_filter).sel(time=above_slice)
# selected_segments = segments.findSegments(above_cloud["start"].strftime('%Y-%m-%d %H:%M:%S'),
#                                           above_cloud["end"].strftime('%Y-%m-%d %H:%M:%S'))
# 
# fig, ax = plt.subplots(figsize=h.figsize_wide)
# ax.plot(plot_ds.time, plot_ds["IRS_PHI"], label="Roll angle", ls="", marker=".")
# ax.plot(plot_ds1.time, plot_ds1["IRS_PHI"], label="Pitch filtered roll angle", ls="", marker=".")
# for i, dic in enumerate(selected_segments[0]["parts"]):
#     ax.axvline(dic["start"], label=f"start {dic['name']}", color=cbc[i + 1])
# 
# ax.axhline(roll_threshold, color=cbc[-1])
# ax.axhline(-roll_threshold, color=cbc[-1])
# ax.set_ylim(-10, 5)
# ax.set_ylim(-1.5, 1.5)  # for zoom plot
# ax.grid()
# ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0), ncol=3, bbox_transform=fig.transFigure)
# ax.set_xlabel("Time (UTC)")
# ax.set_ylabel("Roll Angle (°)")
# ax.set_title("BAHAMAS Roll Angle Above Cloud")
# h.set_xticks_and_xlabels(ax, time_extend_ac)
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.35)
# figname = f"{plot_path}/{halo_flight}_BAHAMAS_roll_angle_above_cloud_zoom.png"
# plt.savefig(figname, dpi=300)
# plt.show()
# plt.close()

# %% plot BAHAMAS pitch angle for above cloud section
# plot_ds = bahamas_unfiltered.sel(time=above_slice)
# plot_ds1 = bahamas_unfiltered.where(roll_filter).sel(time=above_slice)
# plot_ds2 = pitch_center.sel(time=above_slice)
# selected_segments = segments.findSegments(above_cloud["start"].strftime('%Y-%m-%d %H:%M:%S'),
#                                           above_cloud["end"].strftime('%Y-%m-%d %H:%M:%S'))
# 
# fig, ax = plt.subplots(figsize=h.figsize_wide)
# ax.plot(plot_ds.time, plot_ds["IRS_THE"], label="Pitch angle", ls="", marker=".")
# ax.plot(plot_ds1.time, plot_ds1["IRS_THE"], label="Roll filtered pitch angle", ls="", marker=".")
# for i, dic in enumerate(selected_segments[0]["parts"]):
#     ax.axvline(dic["start"], label=f"start {dic['name']}", color=cbc[i + 1])
# 
# ax.plot(plot_ds2.time, plot_ds2, color=cbc[-1], ls="--")
# ax.plot(plot_ds2.time, plot_ds2 + pitch_threshold, color=cbc[-1])
# ax.plot(plot_ds2.time, plot_ds2 - pitch_threshold, color=cbc[-1])
# # ax.set_ylim(-10, 5)
# ax.set_ylim(2, 3)  # for zoom plot
# ax.grid()
# ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0), ncol=3, bbox_transform=fig.transFigure)
# ax.set_xlabel("Time (UTC)")
# ax.set_ylabel("Pitch Angle (°)")
# ax.set_title("BAHAMAS Pitch Angle Above Cloud")
# h.set_xticks_and_xlabels(ax, time_extend_ac)
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.35)
# figname = f"{plot_path}/{halo_flight}_BAHAMAS_pitch_angle_above_cloud_zoom.png"
# plt.savefig(figname, dpi=300)
# plt.show()
# plt.close()

# %% plot BAHAMAS pitch angle for case study section
plot_ds = bahamas_unfiltered.sel(time=case_slice)
plot_ds1 = bahamas_unfiltered.where(roll_filter).sel(time=case_slice)
plot_ds2 = pitch_center.sel(time=case_slice)
selected_segments = segments.findSegments(above_cloud["start"].strftime('%Y-%m-%d %H:%M:%S'),
                                          above_cloud["end"].strftime('%Y-%m-%d %H:%M:%S'))

fig, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(plot_ds.time, plot_ds["IRS_THE"], label="Pitch angle", ls="", marker=".")
ax.plot(plot_ds1.time, plot_ds1["IRS_THE"], label="Roll filtered pitch angle", ls="", marker=".")
for i, dic in enumerate(selected_segments[0]["parts"]):
    ax.axvline(dic["start"], label=f"start {dic['name']}", color=cbc[i + 1])

ax.plot(plot_ds2.time, plot_ds2, color=cbc[-1], ls="--")
ax.plot(plot_ds2.time, plot_ds2 + pitch_threshold, color=cbc[-1])
ax.plot(plot_ds2.time, plot_ds2 - pitch_threshold, color=cbc[-1])
ax.set_ylim(-2.5, 3)
# ax.set_ylim(2, 3)  # for zoom plot
ax.grid()
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0), ncol=3, bbox_transform=fig.transFigure)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Pitch Angle (°)")
ax.set_title("BAHAMAS Pitch Angle Case Study")
h.set_xticks_and_xlabels(ax, time_extend_cs)
plt.tight_layout()
plt.subplots_adjust(bottom=0.35)
figname = f"{plot_path}/{halo_flight}_BAHAMAS_pitch_angle_case_study.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot BAHAMAS pitch angle for whole flight
plot_ds = bahamas_unfiltered
plot_ds_filtered = bahamas_unfiltered.where(roll_filter)
rolling_median = plot_ds.IRS_THE.rolling(time=1800, min_periods=2, center=True).median()
selected_segments = segments.findSegments(above_cloud["start"].strftime('%Y-%m-%d %H:%M:%S'),
                                          above_cloud["end"].strftime('%Y-%m-%d %H:%M:%S'))

fig, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(plot_ds.time, plot_ds["IRS_THE"], label="Pitch angle", ls="", marker="."),
ax.plot(plot_ds_filtered.time, plot_ds_filtered["IRS_THE"], label="Roll filtered pitch angle", ls="", marker="."),
ax.plot(rolling_median.time, rolling_median, label="Rolling Median (30min)", ls="-.")
ax.axhline(2.27688217, label="Median", color=cbc[-1], ls="--")
ax.axhline(2.27688217 + 0.37, color=cbc[-1], label="Static pitch threshold")
ax.axhline(2.27688217 - 0.37, color=cbc[-1])
ax.axvline(above_cloud["start"], label=f"Start case study", color=cbc[3])
ax.axvline(below_cloud["end"], label=f"End case study", color=cbc[4])
ax.set_ylim(1.5, 3)  # for zoom plot
ax.grid()
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0), ncol=3, bbox_transform=fig.transFigure)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Pitch Angle (°)")
ax.set_title(f"{halo_key} BAHAMAS Pitch Angle Above Cloud")
h.set_xticks_and_xlabels(ax, time_extend)
plt.tight_layout()
plt.subplots_adjust(bottom=0.3)
figname = f"{plot_path}/{halo_flight}_BAHAMAS_pitch_angle_all_zoom.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot BACARDI measurements of below and above cloud section
bacardi_ds_slice = bacardi_ds.sel(time=slice(above_cloud["start"], below_cloud["end"]))
fig, ax = plt.subplots(figsize=h.figsize_wide)
for var in ["F_down_solar", "F_down_terrestrial", "F_up_solar", "F_up_terrestrial"]:
    ax.plot(bacardi_ds_slice[var].time, bacardi_ds_slice[var], label=labels[var])

ax.axvline(x=above_cloud["end"], label="End above cloud section", color="#6699CC")
ax.axvline(x=below_cloud["start"], label="Start below cloud section", color="#888888")
ax.grid()
ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=3)
h.set_xticks_and_xlabels(ax, (below_cloud["end"] - above_cloud["start"]))
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Broadband Irradiance (W$\,$m$^{-2}$)")
ax.set_title(f"{halo_key} BACARDI broadband irradiance case study")
plt.tight_layout()
plt.subplots_adjust(bottom=0.3)
figname = f"{plot_path}/{halo_flight}_BACARDI_bb_irradiance_above_below.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot BACARDI measurements of above cloud section solar only with uncertainty and libRadtran simulation
plot_ds = bacardi_ds.sel(time=above_slice)
sza_plot = sza.sel(time=above_slice)
plot_ds1 = bb_sim_solar.sel(time=above_slice)
plot_ds2 = bb_sim_solar1.sel(time=above_slice)
bacardi_error = plot_ds * 0.03
var1, var2 = "F_down_solar", "F_up_solar"
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(plot_ds.time, plot_ds[var1], label=f"{labels[var1]} BACARDI", c=cbc[0])
ax.fill_between(bacardi_error.time, plot_ds[var1] + bacardi_error[var1], plot_ds[var1] - bacardi_error[var1],
                color=cbc[0], alpha=0.5)
ax.plot(plot_ds.time, plot_ds[var2], label=f"{labels[var2]} BACARDI", c=cbc[2])
ax.fill_between(bacardi_error.time, plot_ds[var2] + bacardi_error[var2], plot_ds[var2] - bacardi_error[var2],
                color=cbc[2], alpha=0.5)
ax.plot(plot_ds1.time, plot_ds1.fdw, label=f"{labels[var1]} libRadtran", c=cbc[4])
ax.plot(plot_ds2.time, plot_ds2.fdw, label=f"{labels[var1]} libRadtran -3600nm", c=cbc[-3])
for i, dic in enumerate(selected_segments[0]["parts"]):
    ax.axvline(dic["start"], color=cbc[i + 1])
    ax.annotate(dic["name"], (dic["start"], 255 - 3 * i), size=6)

ax.set_ylim(100, 260)
ax.grid()
h.set_xticks_and_xlabels(ax, time_extend_ac)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Broadband Irradiance (W$\,$m$^{-2}$)")
ax.set_title("BACARDI and libRadtran solar broadband irradiance above cloud")
ax2 = ax.twinx()
ax2.plot(sza_plot.time, sza_plot, label="SZA", c=cbc[3])
ax2.set_ylabel("Solar Zenith Angle (°)")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
h1.append(h2[0])
l1.append(l2[0])
ax2.legend(h1, l1, loc=4, ncol=2)
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_BACARDI_libRadtran_solar_irradiance_above.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot BACARDI measurements of below cloud section solar only
bacardi_plot = bacardi_ds.sel(time=below_slice)
bacardi_error = bacardi_plot * 0.03
var1, var2 = "F_down_solar", "F_up_solar"
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(bacardi_plot.time, bacardi_plot[var1], label=labels[var1], c=cbc[0])
ax.fill_between(bacardi_error.time, bacardi_plot[var1] + bacardi_error[var1], bacardi_plot[var1] - bacardi_error[var1],
                color=cbc[0], alpha=0.5)
ax.plot(bacardi_plot.time, bacardi_plot[var2], label=labels[var2], c=cbc[2])
ax.fill_between(bacardi_error.time, bacardi_plot[var2] + bacardi_error[var2], bacardi_plot[var2] - bacardi_error[var2],
                color=cbc[2], alpha=0.5)
ax.set_ylim(80, 175)
ax.grid()
ax.legend()
h.set_xticks_and_xlabels(ax, time_extend_ac)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Broadband Irradiance (W$\,$m$^{-2}$)")
ax.set_title(f"{halo_key} BACARDI broadband irradiance below cloud")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_BACARDI_solar_irradiance_below.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot BACARDI measurements of below cloud section upward only
bacardi_plot = bacardi_ds.sel(time=below_slice)
bacardi_error = bacardi_plot * 0.03
var1, var2 = "F_up_solar", "F_up_terrestrial"
_, ax = plt.subplots(figsize=h.figsize_wide)
# ax.plot(bacardi_plot.time, bacardi_plot["F_down_solar"], label=labels["F_down_solar"], c=cbc[0])
ax.plot(bacardi_plot.time, bacardi_plot[var1], label=labels[var1], c=cbc[2])
ax.fill_between(bacardi_error.time, bacardi_plot[var1] + bacardi_error[var1], bacardi_plot[var1] - bacardi_error[var1],
                color=cbc[2], alpha=0.5)
ax.plot(bacardi_plot.time, bacardi_plot[var2], label=labels[var2], c=cbc[1])
ax.fill_between(bacardi_error.time, bacardi_plot[var2] + bacardi_error[var2], bacardi_plot[var2] - bacardi_error[var2],
                color=cbc[1], alpha=0.5)
ax.set_ylim(80, 260)
ax.grid()
ax.legend()
h.set_xticks_and_xlabels(ax, time_extend_ac)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Broadband Irradiance (W$\,$m$^{-2}$)")
ax.set_title(f"{halo_key} BACARDI broadband irradiance below cloud")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_BACARDI_upward_irradiance_below.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot BACARDI, simulation in BACARDI and bb libRadtran simulation
h.set_cb_friendly_colors()
plot_ds = bacardi_ds.sel(time=below_slice)
libradtran = bb_sim_solar1.sel(time=below_slice)
labels = dict(F_down_solar=r"$F_{\downarrow, solar}$", F_down_terrestrial=r"$F_{\downarrow, terrestrial}$",
              F_up_solar=r"$F_{\uparrow, solar}$", F_up_terrestrial=r"$F_{\uparrow, terrestrial}$",
              F_down_solar_sim=r"$F_{\downarrow, solar, sim}$")
fig, ax = plt.subplots(figsize=h.figsize_wide)
for var in ["F_down_solar", "F_down_solar_sim"]:
    ax.plot(plot_ds[var].time, plot_ds[var], label=labels[var])
ax.plot(libradtran.time, libradtran.fdw, label=r"$F_{\downarrow, solar, bbsim}$")
ax.grid()
ax.legend()
h.set_xticks_and_xlabels(ax, (below_cloud["end"] - above_cloud["start"]))
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Broadband Irradiance (W$\,$m$^{-2}$)")
ax.set_title("BACARDI broadband irradiance below cloud")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_BACARDI+libRadtran_bb_irradiance_below.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot BACARDI Fdw and libRadtran simulation Fdw below cloud
h.set_cb_friendly_colors()
bacardi_plot = bacardi_ds.sel(time=below_slice)
libradtran = bb_sim_solar1.sel(time=below_slice)
bacardi_error = bacardi_plot * 0.03
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(bacardi_plot.time, bacardi_plot["F_down_solar"], label=labels["F_down_solar"], marker=".")
ax.fill_between(bacardi_error.time, bacardi_plot["F_down_solar"] + bacardi_error["F_down_solar"],
                bacardi_plot["F_down_solar"] - bacardi_error["F_down_solar"], color=cbc[0], alpha=0.5)
ax.plot(libradtran.time, libradtran.fdw, label=r"$F_{\downarrow, solar, clear sky}$", marker=".", c=cbc[5])
ax.grid()
ax.legend()
h.set_xticks_and_xlabels(ax, time_extend_ac)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Broadband Irradiance (W$\,$m$^{-2}$)")
ax.set_title("BACARDI broadband irradiance below cloud")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_BACARDI_libRadtran_Fdw_below.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot BACARDI Fup and libRadtran simulation Fdw
h.set_cb_friendly_colors()
bacardi_plot = bacardi_ds.sel(time=below_slice)
libradtran = bb_sim_solar1.sel(time=below_slice)
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(bacardi_plot.time, bacardi_plot["F_up_solar"], label=labels["F_up_solar"], marker=".", c=cbc[2])
ax.plot(libradtran.time, libradtran.fdw, label=r"$F_{\downarrow, solar, clear sky}$", marker=".", c=cbc[5])
ax.grid()
ax.legend()
h.set_xticks_and_xlabels(ax, time_extend_ac)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Broadband Irradiance (W$\,$m$^{-2}$)")
ax.set_title("BACARDI broadband irradiance below cloud")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_BACARDI_Fup_libRadtran_Fdw_below.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# # %% plot BACARDI Fdw and libRadtran clearsky simulation Fdw above cloud time series
# h.set_cb_friendly_colors()
# bacardi_plot = bacardi_ds.where(above_sel)
# libradtran = bb_sim_solar1.where(above_sel)
# plt.rc("font", size=12)
# _, ax = plt.subplots(figsize=h.figsize_wide)
# ax.plot(bacardi_plot.time, bacardi_plot["F_down_solar"], label=labels["F_down_solar"], marker=".", ls="")
# ax.plot(libradtran.time, libradtran.fdw, label=r"$F_{\downarrow, solar, clear sky}$", marker=".", c=cbc[5], ls="")
# ax.grid()
# ax.legend()
# h.set_xticks_and_xlabels(ax, time_extend)
# ax.set_xlabel("Time (UTC)")
# ax.set_ylabel("Broadband Irradiance (W$\,$m$^{-2}$)")
# ax.set_title(f"BACARDI and libRadtran broadband irradiance above cloud whole flight - {halo_key}")
# plt.tight_layout()
# figname = f"{plot_path}/{halo_flight}_BACARDI_libRadtran_Fdw_above_all_2.png"
# plt.savefig(figname, dpi=300)
# plt.show()
# plt.close()

# # %% plot BACARDI Fdw and libRadtran clearsky simulation Fdw above cloud scatter
# h.set_cb_friendly_colors()
# bacardi_plot = bacardi_ds_res.where(above_sel)
# libradtran = bb_sim_solar1.where(above_sel)
# rmse = np.sqrt(np.mean((libradtran.fdw - bacardi_plot["F_down_solar"]) ** 2))
# bias = np.mean((libradtran.fdw - bacardi_plot["F_down_solar"]))
# plt.rc("font", size=12)
# _, ax = plt.subplots(figsize=h.figsize_equal)
# ax.scatter(bacardi_plot["F_down_solar"], libradtran.fdw, c=cbc[3])
# ax.axline((0, 0), slope=1, color="k", lw=2, transform=ax.transAxes)
# ax.grid()
# ax.set_xlabel("BACARDI Broadband Irradiance (W$\,$m$^{-2}$)")
# ax.set_ylabel("libRadtran Broadband Irradiance (W$\,$m$^{-2}$)")
# ax.set_title(f"Solar downward irradiance\nabove cloud whole flight - {halo_key}")
# ax.text(0.025, 0.95, f"# points: {sum(~np.isnan(bacardi_plot['F_down_solar'])):.0f}\n"
#                      f"RMSE: {rmse.values:.2f}" + " W$\,$m$^{-2}$\n"
#                                                   f"Bias: {bias.values:.2f}" + " W$\,$m$^{-2}$",
#         ha='left', va='top', transform=ax.transAxes,
#         bbox=dict(fc='white', ec='black', alpha=0.8, boxstyle='round'))
# plt.tight_layout()
# figname = f"{plot_path}/{halo_flight}_BACARDI_libRadtran_Fdw_above_all_scatter_2.png"
# plt.savefig(figname, dpi=300)
# plt.show()
# plt.close()

# # %% plot difference between libRadtran clearsky Fdw simulation and BACARDI Fdw above cloud time series
# h.set_cb_friendly_colors()
# bacardi_plot = bacardi_ds_res.where(above_sel)
# libradtran = bb_sim_solar.where(above_sel)
# libradtran_bacardi = libradtran.fdw - bacardi_plot["F_down_solar"]
# rmse = np.sqrt(np.mean((libradtran.fdw - bacardi_plot["F_down_solar"]) ** 2))
# bias = np.mean((libradtran.fdw - bacardi_plot["F_down_solar"]))
# plt.rc("font", size=12)
# _, ax = plt.subplots(figsize=h.figsize_wide)
# ax.plot(libradtran_bacardi.time, libradtran_bacardi, c=cbc[3], ls="", marker=".")
# ax.grid()
# ax.set_xlabel("Time (UTC)")
# ax.set_ylabel("libRadtran - BACARDI\n Broadband Irradiance Difference (W$\,$m$^{-2}$)")
# ax.set_title(f"Solar downward irradiance difference (290 - 5000nm)\nabove cloud whole flight - {halo_key}")
# ax.text(0.025, 0.95, f"# points: {sum(~np.isnan(bacardi_plot['F_down_solar'])):.0f}\n"
#                      f"RMSE: {rmse.values:.2f}" + " W$\,$m$^{-2}$\n"
#                                                   f"Bias: {bias.values:.2f}" + " W$\,$m$^{-2}$",
#         ha='left', va='top', transform=ax.transAxes,
#         bbox=dict(fc='white', ec='black', alpha=0.8, boxstyle='round'))
# h.set_xticks_and_xlabels(ax, time_extend)
# plt.tight_layout()
# figname = f"{plot_path}/{halo_flight}_BACARDI_libRadtran_Fdw_difference_above_all_2.png"
# plt.savefig(figname, dpi=300)
# plt.show()
# plt.close()

# %% plot transmissivity calculated from BACARDI and libRadtran
h.set_cb_friendly_colors()
bacardi_plot = bacardi_ds["transmissivity_solar"].sel(time=below_slice)
bacardi_error = bacardi_plot * 0.03
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(bacardi_plot.time, bacardi_plot, label="Transmissivity", marker=".", c=cbc[3])
ax.fill_between(bacardi_error.time, bacardi_plot + bacardi_error, bacardi_plot - bacardi_error,
                color=cbc[3], alpha=0.5)
ax.axhline(y=1, color="#888888")
ax.grid()
h.set_xticks_and_xlabels(ax, time_extend_ac)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Transmissivity")
ax.set_title(f"Cloud Transmissivity - {halo_key} \nCalculated from BACARDI Measurements and libRadtran Simulations")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_BACARDI_libRadtran_transmissivity.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% Relation of BACARDI to simulation depending on viewing angle of HALO
relation_bacardi_libradtran = bacardi_ds["F_down_solar"] / bacardi_ds["F_down_solar_sim"]

# %% viewing direction of halo: 0 = towards sun, 180 = away from sun
heading = bahamas_ds.IRS_HDG
viewing_dir = bacardi_ds.saa - heading
viewing_dir = viewing_dir.where(viewing_dir > 0, viewing_dir + 360)

# %% merge information in dataframe
df1 = viewing_dir.to_dataframe(name="viewing_dir")
df2 = relation_bacardi_libradtran.to_dataframe(name="relation")
df = df1.merge(df2, on="time")
df["sza"] = bacardi_ds.sza
df["roll"] = bahamas_ds["IRS_PHI"].where(motion_filter)
# df = df[df.relation > 0.7]
df = df.sort_values(by="viewing_dir")

# %% plot relation between BACARDI measurement and simulation depending on viewing angle as polarplot
h.set_cb_friendly_colors()
plt.rc("font", size=12, family="serif")
_, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': 'polar'})
ax.scatter(np.deg2rad(df["viewing_dir"]), df["relation"], label="0 = facing sun\n180 = facing away from sun")
df_plot = df[((below_cloud["start"] < df.index) & (df.index < below_cloud["end"]))]
ax.scatter(np.deg2rad(df_plot["viewing_dir"]), df_plot["relation"], label="below cloud")
ax.set_rmax(1.2)
ax.set_rticks([0.8, 1, 1.2])  # Less radial ticks
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi / 2)
ax.grid(True)
ax.set_title("Relation between BACARDI Fdw measurement and libRadtran simulation\n"
             " according to viewing direction of HALO with respect to the sun")
ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure)
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
figname = f"{plot_path}/{halo_flight}_BACARDI_inlet_directional_dependence.png"
# plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot relation as function of SZA
h.set_cb_friendly_colors()
plt.rc("font", size=14, family="serif")
_, ax = plt.subplots(figsize=(10, 6))
df_tmp = df[((below_cloud["start"] < df.index) & (df.index < below_cloud["end"]))]
ax.scatter(df_tmp["sza"], df_tmp["relation"], label="below cloud")
df_tmp = df[~((below_cloud["start"] < df.index) & (df.index < below_cloud["end"]))]
ax.scatter(df_tmp["sza"], df_tmp["relation"])
ax.grid()
ax.set_xlabel("Solar Zenith Angle (deg)")
ax.set_ylabel("Relation")
ax.set_title("Relation between BACARDI Fdw measurement and libRadtran simulation\n"
             " in relation to solar zenith angle", size=16)
ax.legend()
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_BACARDI_inlet_sza_dependence.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot relation as function of SZA, exclude below cloud section
h.set_cb_friendly_colors()
plt.rc("font", size=14, family="serif")
_, ax = plt.subplots(figsize=(10, 6))
df_tmp = df[~((below_cloud["start"] < df.index) & (df.index < below_cloud["end"]))]
mean_rel = df_tmp.relation.mean()
ax.axhline(y=mean_rel, c=cbc[2], lw=2, label=f"Mean {mean_rel:.3f}")
ax.scatter(df_tmp["sza"], df_tmp["relation"])
ax.grid()
ax.set_xlabel("Solar Zenith Angle (deg)")
ax.set_ylabel("Relation")
ax.set_title("Relation between BACARDI Fdw measurement and libRadtran simulation\n"
             " in relation to solar zenith angle (excluding below cloud section)", size=16)
ax.legend()
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_BACARDI_inlet_sza_dependence_without_below_cloud.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot relation as function of time (exclude below cloud section)
h.set_cb_friendly_colors()
plt.rc("font", size=14, family="serif")
plot_df = df.sort_values(by="time")
_, ax = plt.subplots(figsize=(10, 6))
plot_df = plot_df[~((below_cloud["start"] < plot_df.index) & (plot_df.index < below_cloud["end"]))]
ax.scatter(plot_df.index, plot_df["relation"])
ax.grid()
h.set_xticks_and_xlabels(ax, time_extend)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Relation")
ax.set_title("Relation between BACARDI Fdw measurement and libRadtran simulation\n"
             "(excluding below cloud)", size=16)
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_BACARDI_simulation_relation_without_below_cloud.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% dig into measurements with relation > 1
relation_filter = df.sort_values(by="time")["relation"] > 1
relation_filter = relation_filter.between_time("12:00", "11:20")
bacardi_plot = bahamas_ds.where(relation_filter.to_xarray())
h.set_cb_friendly_colors()
plt.rc("font", size=12, family="serif")
_, ax = plt.subplots(figsize=(10, 6))
ax.scatter(bacardi_plot.time, bacardi_plot["IRS_THE"])
ax.grid()
h.set_xticks_and_xlabels(ax, time_extend)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Pitch Angle (degrees)")
ax.set_title("BAHAMAS Pitch Angle for measurements when the Relation was greater 1")
plt.tight_layout()
# figname = f"{plot_path}/{halo_flight}_BAHAMAS_pitch_angle_rel_1.png"
# plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot standard deviation of minutely average of BACARDI measurement as a measure of heterogeneity
h.set_cb_friendly_colors()
plt.rc("font", size=12, family="serif")
bacardi_plot = bacardi_std.sel(time=below_slice)
_, ax = plt.subplots(figsize=(8, 4))
ax.plot(bacardi_plot.time, bacardi_plot.F_down_solar, lw=2)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Minutely Standard Deviation of\n Solar Downward Broadband\nIrradiance (W$\,$m$^{-2}$)")
h.set_xticks_and_xlabels(ax, time_extend_cs)
ax.grid()
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_BACARDI_minutely_std_below_cloud.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot minutely average of BACARDI measurement with standard deviation
h.set_cb_friendly_colors()
plt.rc("font", size=12, family="serif")
_, ax = plt.subplots(figsize=(8, 4))
plot_ds = bacardi_std.sel(time=below_slice)
plot_ds2 = bacardi_ds_res.sel(time=below_slice)
ax.plot(plot_ds2.time, plot_ds2.F_down_solar, lw=2, label=labels["F_down_solar"])
ax.plot(plot_ds2.time, plot_ds2.F_down_terrestrial, lw=2, label=labels["F_down_terrestrial"])
ax.fill_between(plot_ds.time, plot_ds2.F_down_solar - plot_ds.F_down_solar,
                plot_ds2.F_down_solar + plot_ds.F_down_solar, alpha=0.3, label="Standard Deviation")
ax.fill_between(plot_ds.time, plot_ds2.F_down_terrestrial - plot_ds.F_down_terrestrial,
                plot_ds2.F_down_terrestrial + plot_ds.F_down_terrestrial, alpha=0.3)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Downward Broadband\nIrradiance (W$\,$m$^{-2}$)")
ax.set_title(f"Minutely Mean of BACARDI Measurement - {halo_key}")
h.set_xticks_and_xlabels(ax, time_extend_cs)
ax.grid()
ax.legend(fontsize=10)
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_BACARDI_Fdw+std_below_cloud.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot SMART spectra with standard deviation for below cloud
h.set_cb_friendly_colors()
plt.rc("font", size=12, family="serif")
plot_ds = smart_ds_filtered.sel(time=below_slice).mean(dim="time")
plot_std = smart_ds_filtered.sel(time=below_slice).std(dim="time")
plot_ds2 = smart_ds_filtered.sel(time=above_slice).mean(dim="time")
plot_std2 = smart_ds_filtered.sel(time=above_slice).std(dim="time")
_, ax = plt.subplots(figsize=(8, 4))
ax.plot(plot_ds.wavelength, plot_ds["Fdw"], lw=2, label="Below Cloud")
ax.plot(plot_ds2.wavelength, plot_ds2["Fdw"], lw=2, label="Above Cloud")
ax.fill_between(plot_ds.wavelength, plot_ds.Fdw - plot_std.Fdw, plot_ds.Fdw + plot_std.Fdw,
                alpha=0.5, label="Standard Deviation")
ax.fill_between(plot_ds2.wavelength, plot_ds2.Fdw - plot_std2.Fdw, plot_ds2.Fdw + plot_std2.Fdw,
                alpha=0.5)
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Spectral Downward \nIrradiance (W$\,$m$^{-2}\,$nm$^{-1}$)")
ax.set_title(f"SMART Average Spectra - {halo_key}")
ax.grid()
ax.legend(fontsize=10)
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_SMART_mean+Fdw+std_above+below_cloud.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot minutely SMART spectra below cloud with standard deviation
h.set_cb_friendly_colors()
plt.rc("font", size=12, family="serif")
ds = smart_ds_filtered.sel(time=below_slice)
times = pd.date_range(ds.time[0].values, ds.time[-1].values, freq="1Min")
for t in times:
    plot_ds = ds.sel(time=t)
    plot_std = smart_std.sel(time=t)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(plot_ds.wavelength, plot_ds["Fdw"], lw=2, label="Below Cloud")
    ax.fill_between(plot_ds.wavelength, plot_ds.Fdw - plot_std.Fdw, plot_ds.Fdw + plot_std.Fdw,
                    alpha=0.5, label="Standard Deviation")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Spectral Downward \nIrradiance (W$\,$m$^{-2}\,$nm$^{-1}$)")
    ax.set_title(f"SMART Minutely Averaged Spectra - {halo_key} - {t:%Y-%m-%d %H:%M}")
    ax.grid()
    ax.legend(fontsize=10)
    plt.tight_layout()
    figname = f"{plot_path}/below_cloud_spectra/{halo_flight}_SMART_Fdw+std_{t:%H%M}.png"
    fig.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot minutely SMART spectra above cloud with standard deviation
h.set_cb_friendly_colors()
plt.rc("font", size=12, family="serif")
ds = smart_ds_filtered.sel(time=above_slice)
times = pd.date_range(ds.time[0].values, ds.time[-1].values, freq="1Min")
for t in times:
    plot_ds = ds.sel(time=t)
    plot_std = smart_std.sel(time=t)
    time1 = pd.to_datetime(t)
    _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(plot_ds.wavelength, plot_ds["Fdw"], lw=2, label="Above Cloud")
    ax.fill_between(plot_ds.wavelength, plot_ds.Fdw - plot_std.Fdw, plot_ds.Fdw + plot_std.Fdw,
                    alpha=0.5, label="Standard Deviation")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Spectral Downward \nIrradiance (W$\,$m$^{-2}\,$nm$^{-1}$)")
    ax.set_title(f"SMART Minutely Averaged Spectra - {halo_key} - {time1:%Y-%m-%d %H:%M}")
    ax.grid()
    ax.legend(fontsize=10)
    plt.tight_layout()
    figname = f"{plot_path}/above_cloud_spectra/{halo_flight}_SMART_Fdw+std_{time1:%H%M}.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot Dropsondes and map in one plot
rh_ice = met.relative_humidity_water_to_relative_humidity_ice(dropsonde_ds.rh, dropsonde_ds.T - 273.15)
labels = [f"{lt.replace('2022-04-11T', '')}Z" for lt in np.datetime_as_string(rh_ice.launch_time.values, unit="m")]
h.set_cb_friendly_colors()
plt.rc("font", family="serif", size=10)
fig = plt.figure(figsize=(18 * cm, 10.125 * cm))
ax = fig.add_subplot(121)
rh_ice.plot.line(y="alt", alpha=0.5, label=labels, lw=1, ax=ax)
rh_ice.mean(dim="launch_time").plot(y="alt", lw=3, label="Mean", c="k", ax=ax)

# plot vertical line at 100%
ax.axvline(x=100, color="#661100", lw=2)
ax.set_xlabel("Relative Humidity over Ice (%)")
ax.set_ylabel("Altitude (km)")
ax.grid()
ax.legend(bbox_to_anchor=(1, 1.01), loc="upper left")

# plot map
orig_map = plt.cm.get_cmap('Blues')  # getting the original colormap using cm.get_cmap() function
reversed_map = orig_map.reversed()  # reversing the original colormap using reversed() function
# select position and time data
lon, lat, altitude, times = ins["lon"], ins["lat"], ins["alt"], ins["time"]
data_crs = ccrs.PlateCarree()
extent = [-15, 30, 68, 90]
ax = fig.add_subplot(122, projection=ccrs.NorthPolarStereo())
ax.coastlines()
ax.add_feature(cartopy.feature.BORDERS)
ax.set_extent(extent, crs=data_crs)
gl = ax.gridlines(crs=data_crs, draw_labels=True, x_inline=False, y_inline=False)
gl.bottom_labels = False
gl.left_labels = False

# add sea ice extent
seaice = get_amsr2_seaice(f"{(pd.to_datetime(date) - pd.Timedelta(days=0)):%Y%m%d}")
seaice = seaice.seaice
ax.pcolormesh(seaice.lon, seaice.lat, seaice, transform=data_crs, cmap=reversed_map)

# plot flight track
points = ax.scatter(lon, lat, s=1, c="orange", transform=data_crs)

# plot dropsonde launch locations
for i in range(dropsonde_ds.lon.shape[0]):
    launch_time = pd.to_datetime(dropsonde_ds.launch_time[i].values)
    x, y = dropsonde_ds.lon[i].mean().values, dropsonde_ds.lat[i].mean().values
    ax.plot(x, y, "x", color="red", markersize=8, label="Dropsonde", transform=data_crs)
    # ax.text(x, y, f"{i+1:02d}", c="white", fontsize=8, transform=data_crs,
    #         path_effects=[patheffects.withStroke(linewidth=0.5, foreground="black")])  # RF09, RF11, RF12, RF14, RF18
    ax.text(x, y, f"{launch_time:%H:%M}", c="white", fontsize=10, transform=data_crs,
            path_effects=[patheffects.withStroke(linewidth=0.5, foreground="black")])

handles, labels = ax.get_legend_handles_labels()
ax.legend([handles[0]], [labels[0]], loc=3)

# Kiruna
x_kiruna, y_kiruna = meta.coordinates["Kiruna"]
ax.plot(x_kiruna, y_kiruna, ".", color="#117733", markersize=8, transform=data_crs)
ax.text(x_kiruna + 0.1, y_kiruna + 0.1, "Kiruna", fontsize=11, transform=data_crs)
# Longyearbyen
x_longyear, y_longyear = meta.coordinates["Longyearbyen"]
ax.plot(x_longyear, y_longyear, ".", color="#117733", markersize=8, transform=data_crs)
ax.text(x_longyear + 0.1, y_longyear + 0.1, "Longyearbyen", fontsize=11, transform=data_crs,
        path_effects=[patheffects.withStroke(linewidth=0.5, foreground="white")])

plt.tight_layout()

figname = f"{plot_path}/{halo_flight}_dropsonde_rh_map.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# # %% plot trajectories and dropsondes in overview plot
# plt.rcdefaults()
# plt_sett = {
#     'TIME': {
#         'label': 'Time Relative to Release (h)',
#         'norm': plt.Normalize(-120, 0),
#         'ylim': [-120, 0],
#         'cmap_sel': 'tab20b_r',
#     }
# }
# var_name = "TIME"
# data_crs = ccrs.PlateCarree()
# h.set_cb_friendly_colors()
#
# plt.rc("font", size=6)
# fig = plt.figure(figsize=(20 * cm, 10 * cm))
# gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
#
# # plot trajectory map 11 April in first row and first column
# ax = fig.add_subplot(gs[0, 0], projection=ccrs.NorthPolarStereo())
# ax.coastlines(alpha=0.5)
# ax.set_xlim((-2000000, 2000000))
# ax.set_ylim((-3000000, 500000))
# gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=1, color='gray', alpha=0.5,
#                   linestyle=':', x_inline=False, y_inline=False, rotate_labels=False)
# gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 20))
# gl.ylocator = mticker.FixedLocator(np.arange(60, 90, 5))
# gl.top_labels = False
# gl.right_labels = False
#
# # read in ERA5 data - 11 April
# era5_files = [os.path.join(era5_path, f) for f in os.listdir(era5_path) if f"P20220412" in f]
# era5_files.sort()
# era5_ds = xr.open_mfdataset(era5_files).sel(time=f"2022-04-12T12:00")
# # calculate pressure on all model levels
# pressure = era5_ds.hyam + era5_ds.hybm * era5_ds.PS * 100
# # select only relevant model levels and swap dimension names
# era5_ds["pressure"] = pressure.sel(nhym=slice(39, 137)).swap_dims({"nhym": "lev"})
#
# # Plot the surface pressure - 11 April
# pressure_levels = np.arange(900, 1125, 5)
# E5_press = era5_ds.MSL / 100  # conversion to hPa
# cp = ax.contour(E5_press.lon, E5_press.lat, E5_press, levels=pressure_levels, colors='k', linewidths=0.7,
#                 linestyles='solid', alpha=1, transform=data_crs)
# cp.clabel(fontsize=4, inline=1, inline_spacing=4, fmt='%i hPa', rightside_up=True, use_clabeltext=True)
#
# # add seaice edge
# ci_levels = [0.8]
# E5_ci = era5_ds.CI
# cci = ax.contour(E5_ci.lon, E5_ci.lat, E5_ci, ci_levels, transform=data_crs, linestyles="--",
#                  colors="#332288")
#
# # add high cloud cover
# E5_cc = era5_ds.CC.where(era5_ds.pressure < 55000, drop=True).sum(dim="lev")
# ax.contourf(E5_cc.lon, E5_cc.lat, E5_cc, levels=20, transform=data_crs, cmap="Blues", alpha=1)
#
# # plot trajectories - 11 April
# header_line = [2]  # header-line of .1 files is always line #2 (counting from 0)
# date_h = f"20220411_07"
# # get filenames
# fname_traj = "traj_CIRR_HALO_" + date_h + ".1"
# trajs = np.loadtxt(f"{trajectory_path}/{fname_traj}", dtype="f", skiprows=5)
# times = trajs[:, 0]
# # generate object to only load specific header line
# gen = h.generate_specific_rows(f"{trajectory_path}/{fname_traj}", userows=header_line)
# header = np.loadtxt(gen, dtype="str", unpack=True)
# header = header.tolist()  # convert to list
# # convert to upper char
# for j in range(len(header)):
#     header[j] = header[j].upper()
#
# print("\tTraj_select.1 could be opened, processing...")
#
# # get the time step of the trajectories # here: manually set
# dt = 0.01
# traj_single_len = 4320  # int(tmax/dt)
# traj_overall_len = int(len(times))
# traj_num = int(traj_overall_len / (traj_single_len + 1))  # +1 for the empty line after
# # each traj
# var_index = header.index(var_name.upper())
#
# for k in range(traj_single_len + 1):
#     # reduce to hourly? --> [::60]
#     lon = trajs[k * (traj_single_len + 1):(k + 1) * (traj_single_len + 1), 1][::60]
#     lat = trajs[k * (traj_single_len + 1):(k + 1) * (traj_single_len + 1), 2][::60]
#     var = trajs[k * (traj_single_len + 1):(k + 1) * (traj_single_len + 1), var_index][::60]
#     x, y = lon, lat
#     points = np.array([x, y]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)
#     lc = LineCollection(segments, cmap=plt_sett[var_name]['cmap_sel'], norm=plt_sett[var_name]['norm'],
#                         alpha=1, transform=data_crs)
#     # Set the values used for colormapping
#     lc.set_array(var)
#     if int(traj_num) == 1:
#         lc.set_linewidth(5)
#     elif int(traj_num) >= 2:
#         lc.set_linewidth(1)
#     line = ax.add_collection(lc)
#
# plt.colorbar(line, ax=ax, pad=0.01,
#              ticks=np.arange(-120, 0.1, 12)).set_label(label=plt_sett[var_name]['label'], size=6)
#
# # plot flight track - 11 April
# track_lons, track_lats = ins["lon"], ins["lat"]
# ax.scatter(track_lons[::10], track_lats[::10], c="k", alpha=1, marker=".", s=1, zorder=400,
#            label='HALO flight track', transform=data_crs, linestyle="solid")
#
# # plot dropsonde locations - 11 April
# for i in range(dropsonde_ds.lon.shape[0]):
#     launch_time = pd.to_datetime(dropsonde_ds.launch_time[i].values)
#     x, y = dropsonde_ds.lon[i].mean().values, dropsonde_ds.lat[i].mean().values
#     cross = ax.plot(x, y, "x", color="orangered", markersize=3, label="Dropsonde", transform=data_crs,
#                     zorder=450)
#     ax.text(x, y, f"{launch_time:%H:%M}", c="k", fontsize=4, transform=data_crs, zorder=500,
#             path_effects=[patheffects.withStroke(linewidth=0.25, foreground="white")])
#
# # make legend for flight track and dropsondes - 11 April
# handles = [plt.plot([], ls="-", color="k")[0],  # flight track
#            cross[0],  # dropsondes
#            plt.plot([], ls="--", color="#332288")[0],  # sea ice edge
#            Patch(facecolor="royalblue")]  # cloud cover
# labels = ["HALO flight track", "Dropsonde", "Sea Ice Edge", "High Cloud Cover\nat 12:00 UTC"]
# ax.legend(handles=handles, labels=labels, framealpha=1, loc=2)
#
# title = f"11 April 2022"
# ax.set_title(title, fontsize=7)
# ax.text(-72.5, 73, "(a)", size=7, transform=data_crs, ha="center", va="center",
#         bbox=dict(boxstyle="round", ec="grey", fc="white"))
#
# # plot dropsonde profiles in row 1 and column 2
# rh_ice = met.relative_humidity_water_to_relative_humidity_ice(dropsonde_ds.rh, dropsonde_ds.T - 273.15)
# labels = [f"{lt.replace('2022-04-11T', '')} UTC" for lt in np.datetime_as_string(rh_ice.launch_time.values, unit="m")]
# ax = fig.add_subplot(gs[0, 1])
# rh_ice.plot.line(y="alt", alpha=0.5, label=labels, lw=1, ax=ax)
# rh_ice.mean(dim="launch_time").plot(y="alt", lw=2, label="Mean", c="k", ax=ax)
#
# # plot vertical line at 100%
# ax.axvline(x=100, color="#661100", lw=2)
# ax.set_xlabel("Relative Humidity over Ice (%)")
# ax.set_ylabel("Altitude (km)")
# ax.grid()
# ax.legend(bbox_to_anchor=(1, 1.01), loc="upper left")
# ax.text(0.1, 0.95, "(b)", size=7, transform=ax.transAxes, ha="center", va="center",
#         bbox=dict(boxstyle="round", ec="grey", fc="white"))
#
# figname = f"{plot_path}/{halo_flight}_trajectories_dropsonde_plot_overview.png"
# print(figname)
# plt.tight_layout()
# plt.savefig(figname, format='png', dpi=300)  # , bbox_inches='tight')
# print("\t\t\t ...saved as: " + str(figname))
# plt.show()
# plt.close()

# %% prepare metadata for plotting IFS data in ecrad dataset
variable = "ciwc"
units = h.plot_units
scale_factor = h.scale_factors
colorbarlabel = h.cbarlabels
# pcm kwargs
alphas = dict(ciwc=0.8)
norms = dict(t=colors.TwoSlopeNorm(vmin=196, vcenter=238, vmax=280))
robust = dict(cloud_fraction=False)
vmaxs = dict(ciwc=15)
# colorbar kwargs
cb_ticks = dict(t=[198, 208, 218, 228, 238, 248, 258, 268, 278])
ct_lines = dict(ciwc=[1, 5, 10, 15], t=range(198, 278, 10))
linewidths = dict(ciwc=0.5, t=1)
ct_fontsize = dict(ciwc=6, t=8)
cmaps = dict(t=cmr.prinsenvlag.reversed(), ciwc=cmr.get_sub_cmap("cmr.freeze", .25, 0.85), cloud_fraction=cmr.neutral,
             re_ice=cmr.cosmic.reversed(), flux_dn_sw=cmr.get_sub_cmap("cmr.torch", 0.2, 1))
# set kwargs
alpha = alphas[variable] if variable in alphas else 1
cmap = cmaps[variable] if variable in cmaps else cmr.rainforest
cmap = plt.get_cmap(cmap).copy()
cmap.set_bad(color="white")
ct_fs = ct_fontsize[variable] if variable in ct_fontsize else 8
lines = ct_lines[variable] if variable in ct_lines else None
lw = linewidths[variable] if variable in linewidths else 1
norm = norms[variable] if variable in norms else None
robust = robust[variable] if variable in robust else True
ticks = cb_ticks[variable] if variable in cb_ticks else None
vmax = vmaxs[variable] if variable in vmaxs else None

# %% prepare ecrad dataset for plotting
sf = scale_factor[variable] if variable in scale_factor else 1
ecrad_plot = ecrad_ds[variable] * sf
# add new z axis mean pressure altitude
new_z = ecrad_ds["press_height_full"].mean(dim="time") / 1000
ecrad_plot_new_z = list()
for t in tqdm(ecrad_plot.time):
    tmp_plot = ecrad_plot.sel(time=t)
    if "half_level" in tmp_plot.dims:
        tmp_plot = tmp_plot.assign_coords(half_level=ecrad_ds["press_height_hl"].sel(time=t, drop=True).values / 1000)
        tmp_plot = tmp_plot.rename(half_level="height")
        aircraft_height_plot = aircraft_height / 1000
    else:
        tmp_plot = tmp_plot.assign_coords(level=ecrad_ds["press_height_full"].sel(time=t, drop=True).values / 1000)
        tmp_plot = tmp_plot.rename(level="height")
        aircraft_height_plot = aircraft_height_full / 1000

    tmp_plot = tmp_plot.interp(height=new_z.values)
    ecrad_plot_new_z.append(tmp_plot)

ecrad_plot = xr.concat(ecrad_plot_new_z, dim="time")
# filter very low values
ecrad_plot = ecrad_plot.where(ecrad_plot > 0.001)

# %% plot aircraft track through IFS model
plt.rcdefaults()
h.set_cb_friendly_colors()
plt.rc('font', size=12)
_, ax = plt.subplots(figsize=(22 * cm, 7 * cm))
# ecrad 2D field
ecrad_plot.plot(x="time", y="height", cmap=cmap, ax=ax, robust=robust, vmax=vmax, alpha=alpha, norm=norm,
                cbar_kwargs={"pad": 0.04, "label": f"{colorbarlabel[variable]}\n ({units[variable]})",
                             "ticks": ticks})
if lines is not None:
    # add contour lines
    ct = ax.contour(ecrad_plot.time, ecrad_plot.height, ecrad_plot.values.T, levels=lines, linestyles="--", colors="k",
                    linewidths=lw)
    ct.clabel(fontsize=ct_fs, inline=1, inline_spacing=4, fmt='%i', rightside_up=True, use_clabeltext=True)
# aircraft altitude through model
aircraft_height_plot.plot(x="time", color="k", ax=ax, label="HALO altitude")

ax.set_ylabel("Altitude (km)")
ax.set_xlabel("Time (UTC)")
ax.set_xlim(case_slice.start, case_slice.stop)
ax.set_ylim(0, 13)
ax.legend(loc=0)
h.set_xticks_and_xlabels(ax, time_extend_cs)
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_ecrad_{variable}_along_track.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot IFS surface properties along flight path
plt.rcdefaults()
h.set_cb_friendly_colors()
plt.rc('font', size=12)
variable = "LSM"
ylabels = dict(skin_temperature="Skin Temperature (K)", CI="Sea Ice Concentration (%)", LSM="Land Sea Mask")
ecrad_plot = ecrad_ds[variable]
_, ax = plt.subplots(figsize=h.figsize_wide)
# ecrad 1D field
ax.plot(ecrad_plot.time, ecrad_plot, lw=3)
ax.set_ylabel(ylabels[variable])
ax.set_xlabel("Time (UTC)")
h.set_xticks_and_xlabels(ax, time_extend)
ax.grid()
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_ecrad_{variable}_along_track.png"
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
for var in ["flux_dn_sw", "flux_dn_lw", "flux_up_sw", "flux_up_lw"]:
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
binsize = 10
bins_array = np.arange(120, 250, binsize)
bacardi_plot = bacardi_ds_res.sel(time=below_slice)
ecrad_plot = ecrad_ds.isel(half_level=height_level_da).sel(time=below_slice)
_, ax = plt.subplots(figsize=(12 * cm, 11 * cm))
hist = ax.hist2d(bacardi_plot["F_down_solar"], ecrad_plot["flux_dn_sw"], cmap=cmr.rainforest.reversed(),
                 bins=bins_array)
ax.axline((0, 0), slope=1, color="k", lw=2, transform=ax.transAxes)
plt.colorbar(hist[3], label="Counts", shrink=0.78)
ax.set_ylim((120, 240))
ax.set_xlim((120, 240))
ax.set_aspect('equal')
ax.set_xlabel("BACARDI Irradiance  (W$\,$m$^{-2}$)")
ax.set_ylabel("ecRad Irradiance  (W$\,$m$^{-2}$)")
ax.set_title("Solar Downward Irradiance")
ax.text(0.01, 0.95, f"binsize: {binsize}" + "$\,$W$\,$m$^{-2}$" + f"\n# points: {hist[0].sum():.0f}",
        horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
plt.tight_layout()
plt.savefig(f"{plot_path}/{halo_flight}_Fdw_solar_bacardi_vs_ecrad.png", dpi=300)
plt.show()
plt.close()

# %% plot 2D histogram of below cloud terrestrial downward measurements
binsize = 2
bins_array = np.arange(80, 132, binsize)
bacardi_plot = bacardi_ds_res.sel(time=below_slice)
ecrad_plot = ecrad_ds.isel(half_level=height_level_da).sel(time=below_slice)
_, ax = plt.subplots(figsize=(12 * cm, 11 * cm))
hist = ax.hist2d(bacardi_plot["F_down_terrestrial"], ecrad_plot["flux_dn_lw"], cmap=cmr.rainforest.reversed(),
                 bins=bins_array)
ax.axline((0, 0), slope=1, color="k", lw=2, transform=ax.transAxes)
plt.colorbar(hist[3], label="Counts", shrink=0.8)
ax.set_ylim((80, 130))
ax.set_xlim((80, 130))
ax.set_aspect('equal')
ax.set_xlabel("BACARDI Irradiance  (W$\,$m$^{-2}$)")
ax.set_ylabel("ecRad Irradiance  (W$\,$m$^{-2}$)")
ax.set_title("Terrestrial Downward Irradiance")
ax.text(0.01, 0.95, f"binsize: {binsize}" + "$\,$W$\,$m$^{-2}$" + f"\n# points: {hist[0].sum():.0f}",
        horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
plt.tight_layout()
plt.savefig(f"{plot_path}/{halo_flight}_Fdw_terrestrial_bacardi_vs_ecrad.png", dpi=300)
plt.show()
plt.close()

# %% plot 2D histogram of below cloud solar upward irradiance measurements
binsize = 5
bins_array = np.arange(95, 175, binsize)
bacardi_plot = bacardi_ds_res.sel(time=below_slice)
ecrad_plot = ecrad_ds.isel(half_level=height_level_da).sel(time=below_slice)
_, ax = plt.subplots(figsize=(12 * cm, 11 * cm))
hist = ax.hist2d(bacardi_plot["F_up_solar"], ecrad_plot["flux_up_sw"], bins=bins_array, cmap=cmr.rainforest.reversed())
ax.axline((0, 0), slope=1, color="k", lw=2, transform=ax.transAxes)
plt.colorbar(hist[3], label="Counts", shrink=0.8)
ax.set_ylim((95, 170))
ax.set_xlim((95, 170))
ax.set_aspect('equal')
ax.set_xlabel("BACARDI Irradiance  (W$\,$m$^{-2}$)")
ax.set_ylabel("ecRad Irradiance  (W$\,$m$^{-2}$)")
ax.set_title("Solar Upward Irradiance")
ax.text(0.01, 0.95, f"binsize: {binsize}" + "$\,$W$\,$m$^{-2}$" + f"\n# points: {hist[0].sum():.0f}",
        horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
plt.tight_layout()
plt.savefig(f"{plot_path}/{halo_flight}_Fup_solar_bacardi_vs_ecrad.png", dpi=300)
plt.show()
plt.close()

# %% plot 2D histogram of below cloud terrestrial upward measurements
binsize = 1
bins_array = np.arange(210, 221, binsize)
bacardi_plot = bacardi_ds_res.sel(time=below_slice)
ecrad_plot = ecrad_ds.isel(half_level=height_level_da).sel(time=below_slice)
_, ax = plt.subplots(figsize=(12 * cm, 11 * cm))
hist = ax.hist2d(bacardi_plot["F_up_terrestrial"], ecrad_plot["flux_up_lw"], cmap=cmr.rainforest.reversed(),
                 bins=bins_array)
ax.axline((0, 0), slope=1, color="k", lw=2, transform=ax.transAxes)
plt.colorbar(hist[3], label="Counts", shrink=0.79)
ax.set_ylim((210, 220))
ax.set_xlim((210, 220))
ax.set_aspect('equal')
ax.set_xlabel("BACARDI Irradiance  (W$\,$m$^{-2}$)")
ax.set_ylabel("ecRad Irradiance  (W$\,$m$^{-2}$)")
ax.set_title("Terrestrial Upward Irradiance")
ax.text(0.01, 0.95, f"binsize: {binsize}" + "$\,$W$\,$m$^{-2}$" + f"\n# points: {hist[0].sum():.0f}",
        horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
plt.tight_layout()
plt.savefig(f"{plot_path}/{halo_flight}_Fup_terrestrial_bacardi_vs_ecrad.png", dpi=300)
plt.show()
plt.close()

# %% plot scatterplot of below cloud measurements
bacardi_plot = bacardi_ds_res.sel(time=below_slice)
ecrad_plot = ecrad_ds.isel(half_level=height_level_da).sel(time=below_slice)
lims = [(100, 160), (130, 180), (80, 140), (220, 230)]
for (i, x), y in zip(enumerate(bacardi_vars), ecrad_vars):
    rmse = np.sqrt(np.mean((ecrad_plot[y] - bacardi_plot[x]) ** 2))
    bias = np.mean((ecrad_plot[y] - bacardi_plot[x]))
    _, ax = plt.subplots(figsize=(12 * cm, 12 * cm))
    ax.scatter(bacardi_plot[x], ecrad_plot[y], c=cbc[3])
    ax.axline((0, 0), slope=1, color="k", lw=2, transform=ax.transAxes)
    ax.set_ylim(lims[i])
    ax.set_xlim(lims[i])
    ticks = ax.get_yticks() if i == 0 else ax.get_xticks()
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)
    ax.set_aspect('equal')
    ax.set_xlabel("BACARDI Irradiance (W$\,$m$^{-2}$)")
    ax.set_ylabel("ecRad Irradiance (W$\,$m$^{-2}$)")
    ax.set_title(f"{titles[i]}\nbelow cloud")
    ax.grid()
    ax.text(0.025, 0.95, f"# points: {sum(~np.isnan(bacardi_plot[x])):.0f}\n"
                         f"RMSE: {rmse.values:.2f}" + " W$\,$m$^{-2}$\n"
                                                      f"Bias: {bias.values:.2f}" + " W$\,$m$^{-2}$",
            ha='left', va='top', transform=ax.transAxes,
            bbox=dict(fc='white', ec='black', alpha=0.8, boxstyle='round'))
    plt.tight_layout()
    plt.savefig(f"{plot_path}/{halo_flight}_{names[i]}_bacardi_vs_ecrad_scatter_below_cloud.png", dpi=300)
    plt.show()
    plt.close()

# %% plot scatterplot of above cloud measurements
bacardi_plot = bacardi_ds_res.sel(time=above_slice)
ecrad_plot = ecrad_ds.isel(half_level=height_level_da).sel(time=above_slice)
lims = [(200, 270), (25, 35), (150, 200), (175, 195)]
for (i, x), y in zip(enumerate(bacardi_vars), ecrad_vars):
    rmse = np.sqrt(np.mean((ecrad_plot[y] - bacardi_plot[x]) ** 2))
    bias = np.mean((ecrad_plot[y] - bacardi_plot[x]))
    _, ax = plt.subplots(figsize=(12 * cm, 12 * cm))
    ax.scatter(bacardi_plot[x], ecrad_plot[y], c=cbc[3])
    ax.axline((0, 0), slope=1, color="k", lw=2, transform=ax.transAxes)
    ax.set_ylim(lims[i])
    ax.set_xlim(lims[i])
    ticks = ax.get_yticks() if i == 0 else ax.get_xticks()
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)
    ax.set_aspect('equal')
    ax.set_xlabel("BACARDI Irradiance (W$\,$m$^{-2}$)")
    ax.set_ylabel("ecRad Irradiance (W$\,$m$^{-2}$)")
    ax.set_title(f"{titles[i]}\nabove cloud")
    ax.grid()
    ax.text(0.025, 0.95, f"# points: {sum(~np.isnan(bacardi_plot[x])):.0f}\n"
                         f"RMSE: {rmse.values:.2f}" + " W$\,$m$^{-2}$\n"
                                                      f"Bias: {bias.values:.2f}" + " W$\,$m$^{-2}$",
            ha='left', va='top', transform=ax.transAxes,
            bbox=dict(fc='white', ec='black', alpha=0.8, boxstyle='round'))
    plt.tight_layout()
    plt.savefig(f"{plot_path}/{halo_flight}_{names[i]}_bacardi_vs_ecrad_scatter_above_cloud.png", dpi=300)
    plt.show()
    plt.close()

# %% plot scatterplot above cloud whole flight
bacardi_plot = bacardi_ds_res.where(above_sel)
ecrad_plot = ecrad_ds.isel(half_level=height_level_da).where(above_sel)
lims = [(200, 550), (25, 35), (0, 350), (150, 240)]
for i in range(4):
    x, y = bacardi_vars[i], ecrad_vars[i]
    rmse = np.sqrt(np.mean((ecrad_plot[y] - bacardi_plot[x]) ** 2))
    bias = np.mean((ecrad_plot[y] - bacardi_plot[x]))
    _, ax = plt.subplots(figsize=(12 * cm, 10 * cm))
    scatter = ax.scatter(bacardi_plot[x], ecrad_plot[y], c=bacardi_plot.sza)
    plt.colorbar(scatter, shrink=1, label="Solar Zenith Angle (°)")
    ax.axline((0, 0), slope=1, color="k", lw=2, transform=ax.transAxes)
    ax.set_ylim(lims[i])
    ax.set_xlim(lims[i])
    ticks = ax.get_yticks() if i == 0 else ax.get_xticks()
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)
    ax.set_aspect('equal')
    ax.set_xlabel("BACARDI Irradiance (W$\,$m$^{-2}$)")
    ax.set_ylabel("ecRad Irradiance (W$\,$m$^{-2}$)")
    ax.set_title(f"{titles[i]}\nabove cloud whole flight {halo_key}")
    ax.grid()
    ax.text(0.025, 0.95, f"# points: {sum(~np.isnan(bacardi_plot[x])):.0f}\n"
                         f"RMSE: {rmse.values:.2f}" + " W$\,$m$^{-2}$\n"
                                                      f"Bias: {bias.values:.2f}" + " W$\,$m$^{-2}$",
            ha='left', va='top', transform=ax.transAxes,
            bbox=dict(fc='white', ec='black', alpha=0.8, boxstyle='round'))
    plt.tight_layout()
    plt.savefig(f"{plot_path}/{halo_flight}_{names[i]}_bacardi_vs_ecrad_scatter_above_cloud_all.png", dpi=300)
    plt.show()
    plt.close()

# %% plot timeseries of difference between ecRad and BACARDI
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(ecrad_bacardi.time, ecrad_bacardi["F_down_solar"], c=cbc[3], marker="o", ls="")
ax2 = ax.twinx()
ax2.plot(bacardi_plot.time, bacardi_plot["sza"], label="SZA", lw=3)
# ax2.plot(ecrad_plot.time, np.rad2deg(np.arccos(ecrad_plot["cos_solar_zenith_angle"])), label="SZA ecRad")
h.set_xticks_and_xlabels(ax, time_extend)
ax.grid()
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("ecRad - BACARDI\nIrradiance Difference (W$\,$m$^{-2}$)")
ax.set_title(f"{titles[0]} Difference\nabove cloud whole flight {halo_key}")
ax2.legend()
ax2.set_ylabel("Solar Zenith Angle (°)")
plt.tight_layout()
plt.savefig(f"{plot_path}/{halo_flight}_{names[0]}_ecrad-bacardi_timeseries_above_cloud_all.png", dpi=300)
plt.show()
plt.close()

# %% calculate correlation coefficient between SZA and ecrad BACARDI difference
drop_nans = ~np.isnan(ecrad_bacardi["F_down_solar"])
ecrad_bacardi_raw = ecrad_bacardi.where(drop_nans, drop=True)
sza = bacardi_plot["sza"].where(drop_nans, drop=True)
pearsonr(sza, ecrad_bacardi_raw["F_down_solar"])

# %% plot Fdw solar along the flight track ecRad and BACARDI
bacardi_plot = bacardi_ds_res
ecrad_plot = ecrad_ds.isel(half_level=height_level_da)
i = 0
_, ax = plt.subplots(figsize=h.figsize_wide)
for i in range(0, 4, 2):
    x, y = bacardi_vars[i], ecrad_vars[i]
    ax.plot(ecrad_plot.time, ecrad_plot[y], c=cbc[i + 4], label=f"ecRad {labels[x]}", ls="", marker=".")
    ax.plot(bacardi_plot.time, bacardi_plot[x], c=cbc[i], label=f"BACARDI {labels[x]}")
ax.axvline(below_cloud["start"], 0, 1, color=cbc[11])
ax.axvline(below_cloud["end"], 0, 1, color=cbc[11])
ax.annotate("Below Cloud",
            xy=(0.52, 0.6), xycoords='axes fraction',
            xytext=(25, 2), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
            )
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Broadband Irradiance (W$\,$m$^{-2}$)")
ax.set_title(f"Solar Broadband Irradiance - BACARDI and ecRad {halo_key}")
h.set_xticks_and_xlabels(ax, time_extend)
ax.grid()
ax.legend(ncol=2)
plt.tight_layout()
plt.savefig(f"{plot_path}/{halo_flight}_bacardi_ecrad_fluxes_time_series.png", dpi=300)
plt.show()
plt.close()

# %% plot Fdw terrestrial along the flight track
bacardi_plot = bacardi_ds_res.where(above_sel)
ecrad_plot = ecrad_ds.isel(half_level=height_level_da)
ecrad_plot = ecrad_plot.where(above_sel)
lims = [(200, 550), (25, 35), (150, 200), (175, 195)]
i = 1
x, y = bacardi_vars[1], ecrad_vars[1]
rmse = np.sqrt(np.mean((ecrad_plot[y] - bacardi_plot[x]) ** 2))
bias = np.mean((ecrad_plot[y] - bacardi_plot[x]))
_, ax = plt.subplots(figsize=(12 * cm, 12 * cm))
ax.scatter(bacardi_plot[x], ecrad_plot[y], c=cbc[3])
ax.axline((0, 0), slope=1, color="k", lw=2, transform=ax.transAxes)
ax.set_ylim(lims[i])
ax.set_xlim(lims[i])
ticks = ax.get_yticks() if i == 0 else ax.get_xticks()
ax.set_yticks(ticks)
ax.set_xticks(ticks)
ax.set_aspect('equal')
ax.set_xlabel("BACARDI Irradiance (W$\,$m$^{-2}$)")
ax.set_ylabel("ecRad Irradiance (W$\,$m$^{-2}$)")
ax.set_title(f"{titles[i]}\nabove cloud whole flight {halo_key}")
ax.text(0.01, 0.95, f"# points: {sum(~np.isnan(bacardi_plot[x])):.0f}\n"
                    f"RMSE: {rmse.values:.2f}" + " W$\,$m$^{-2}$\n"
                                                 f"Bias: {bias.values:.2f}" + " W$\,$m$^{-2}$",
        horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
plt.tight_layout()
plt.savefig(f"{plot_path}/{halo_flight}_{names[i]}_bacardi_vs_ecrad_scatter_above_cloud_all.png", dpi=300)
plt.show()
plt.close()

# %% plot timeseries of difference between ecRad and BACARDI Fdw_terrestrial
ecrad_bacardi = ecrad_plot[y] - bacardi_plot[x]
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(ecrad_bacardi.time, ecrad_bacardi, c=cbc[3], marker="o")
ax2 = ax.twinx()
ax2.plot(bacardi_plot.time, bacardi_plot["sza"], label="SZA")
ax2.plot(ecrad_plot.time, np.rad2deg(np.arccos(ecrad_plot["cos_solar_zenith_angle"])), label="SZA ecRad")
h.set_xticks_and_xlabels(ax, time_extend)
ax.grid()
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("ecRad - BACARDI\nIrradiance Difference (W$\,$m$^{-2}$)")
ax.set_title(f"{titles[i]}\nabove cloud whole flight {halo_key}")
ax2.legend()
plt.tight_layout()
plt.savefig(f"{plot_path}/{halo_flight}_{names[i]}_ecrad-bacardi_timeseries_above_cloud_all.png", dpi=300)
plt.show()
plt.close()

# %% plot transmissivity ecRad vs BACARDI along track below cloud
var = ["transmissivity_solar", "transmissivity_sw"]
bacardi_plot = bacardi_ds_res.sel(time=below_slice)
bacardi_plot2 = bacardi_ds.sel(time=below_slice)
ecrad_plot = ecrad_ds.isel(half_level=height_level_da).sel(time=below_slice)
bacardi_error = bacardi_plot2[var[0]] * 0.03
rmse = np.sqrt(np.mean((ecrad_plot[var[1]] - bacardi_plot[var[0]]) ** 2))
bias = np.mean((ecrad_plot[var[1]] - bacardi_plot[var[0]]))
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(bacardi_plot2.time, bacardi_plot2[var[0]], c=cbc[3], label="BACARDI full resolution")
ax.fill_between(bacardi_error.time, bacardi_plot2[var[0]] + bacardi_error, bacardi_plot2[var[0]] - bacardi_error,
                color=cbc[3], alpha=0.5)
ax.plot(bacardi_plot.time, bacardi_plot[var[0]], c=cbc[5], label="BACARDI resampled", marker="o")
ax.plot(ecrad_plot.time, ecrad_plot[var[1]], c=cbc[4], label="ecRad", marker="o", ls="")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Transmissivity")
ax.grid()
ax.axhline(1, color="grey")
ax.text(0.025, 0.97, f"# points: {sum(~np.isnan(bacardi_plot[var[0]])):.0f}\nRMSE: {rmse.values:.2f}\n"
                     f"Bias: {bias.values:.2f}",
        ha='left', va='top', transform=ax.transAxes,
        bbox=dict(fc='white', ec='black', alpha=0.8, boxstyle='round'))
ax.legend(loc=4)
h.set_xticks_and_xlabels(ax, time_extend_bc)
plt.tight_layout()
plt.savefig(f"{plot_path}/{halo_flight}_bacardi_vs_ecrad_{var[0]}_time_series_below_cloud.png", dpi=300)
plt.show()
plt.close()

# %% plot transmissivity ecRad vs BACARDI scatter below cloud
bacardi_plot = bacardi_ds_res.sel(time=below_slice)
ecrad_plot = ecrad_ds.isel(half_level=height_level_da).sel(time=below_slice)
_, ax = plt.subplots(figsize=h.figsize_equal)
ax.scatter(bacardi_plot[var[0]], ecrad_plot[var[1]], c=cbc[3])
ax.axline((0, 0), slope=1, color="k", lw=2, transform=ax.transAxes)
ax.set_ylim(0.65, 1.05)
ax.set_xlim(0.65, 1.05)
ticks = ax.get_yticks() if i == 0 else ax.get_xticks()
ax.set_yticks(ticks)
ax.set_xticks(ticks)
ax.set_aspect('equal')
ax.set_xlabel("BACARDI Transmissivity")
ax.set_ylabel("ecRad Transmissivity")
ax.grid()
ax.text(0.025, 0.95, f"# points: {sum(~np.isnan(bacardi_plot[var[0]])):.0f}\nRMSE: {rmse.values:.2f}\n"
                     f"Bias: {bias.values:.2f}",
        ha='left', va='top', transform=ax.transAxes,
        bbox=dict(fc='white', ec='black', alpha=0.8, boxstyle='round'))
plt.tight_layout()
plt.savefig(f"{plot_path}/{halo_flight}_bacardi_vs_ecrad_{var[0]}_scatter_below_cloud.png", dpi=300)
plt.show()
plt.close()

# %% plot difference in Fup solar between ecRad and BACARDI along track
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(ecrad_bacardi.time, ecrad_bacardi["F_up_solar"], c=cbc[2], marker="o", ls="")
# ax2 = ax.twinx()
# ax2.plot(bacardi_plot.time, bacardi_plot["sza"], label="SZA", lw=3)
# ax2.plot(ecrad_plot.time, np.rad2deg(np.arccos(ecrad_plot["cos_solar_zenith_angle"])), label="SZA ecRad")
h.set_xticks_and_xlabels(ax, time_extend)
ax.grid()
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("ecRad - BACARDI\nIrradiance Difference (W$\,$m$^{-2}$)")
ax.set_title(f"{titles[2]}\nabove cloud whole flight {halo_key}")
# ax2.legend()
# ax2.set_ylabel("Solar Zenith Angle (°)")
plt.tight_layout()
plt.savefig(f"{plot_path}/{halo_flight}_{names[2]}_ecrad-bacardi_timeseries_above_cloud_all.png", dpi=300)
plt.show()
plt.close()

# %% Relation of BACARDI to ecRad simulation depending on viewing angle of HALO only above cloud
bacardi_plot = bacardi_ds_res.where(above_sel)
ecrad_plot = ecrad_ds.isel(half_level=height_level_da)
ecrad_plot = ecrad_plot.where(above_sel)
relation_bacardi_ecrad = bacardi_plot["F_down_solar"] / ecrad_plot["flux_dn_sw"]

# %% viewing direction of halo: 0 = towards sun, 180 = away from sun
heading = bahamas_ds.IRS_HDG.sel(time=ecrad_plot.time, method="nearest")
viewing_dir = bacardi_plot.saa - heading.values
viewing_dir = viewing_dir.where(viewing_dir > 0, viewing_dir + 360)

# %% merge information in dataframe
df1 = viewing_dir.to_dataframe(name="viewing_dir")
# df2 = relation_bacardi_ecrad.to_dataframe(name="relation")
df2 = ecrad_bacardi["F_down_solar"].to_dataframe(name="difference")
df = df1.merge(df2, on="time")
df["sza"] = bacardi_plot.sza
# df = df[df.relation > 0.7]
df = df.sort_values(by="viewing_dir")

# %% plot difference between ecRad simulation and BACARDI measurement depending on viewing angle as polarplot
h.set_cb_friendly_colors()
plt.rc("font", size=12, family="serif")
_, ax = plt.subplots(figsize=h.figsize_wide, subplot_kw={'projection': 'polar'})
ax.scatter(np.deg2rad(df["viewing_dir"]), df["difference"], label="0 = facing sun\n180 = facing away from sun")
# df_plot = df[((below_cloud["start"] < df.index) & (df.index < below_cloud["end"]))]
# ax.scatter(np.deg2rad(df_plot["viewing_dir"]), df_plot["relation"], label="below cloud")
ax.set_rmax(10)
# ax.set_rticks([0.8, 1, 1.2])  # Less radial ticks
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi / 2)
ax.grid(True)
ax.set_title("Difference between ecRad simulation and BACARDI Fdw measurement\n"
             " according to viewing direction of HALO with respect to the sun")
ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure)
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
figname = f"{plot_path}/{halo_flight}_BACARDI_inlet_directional_dependence_ecRad_difference.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot relation as function of SZA
h.set_cb_friendly_colors()
plt.rc("font", size=14, family="serif")
_, ax = plt.subplots(figsize=(10, 6))
df_tmp = df
ax.scatter(df_tmp["sza"], df_tmp["relation"])
# df_tmp = df[~((below_cloud["start"] < df.index) & (df.index < below_cloud["end"]))]
# ax.scatter(df_tmp["sza"], df_tmp["relation"])
ax.grid()
ax.set_xlabel("Solar Zenith Angle (deg)")
ax.set_ylabel("Relation")
ax.set_title("Relation between BACARDI Fdw measurement and ecRad simulation\n"
             " in relation to solar zenith angle - only above cloud", size=16)
ax.legend()
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_BACARDI_inlet_sza_dependence_ecRad.png"
# plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% compare Fdw solar along the flight track between ecRad band 1-13 and BACARDI
bacardi_plot = bacardi_ds_res.where(above_sel)
ecrad_plot = ecrad_ds.isel(half_level=height_level_da).where(above_sel)
lims = [(200, 550), (25, 35), (150, 200), (175, 195)]
i = 0
x, y = bacardi_vars[0], "flux_dn_sw_2"
rmse = np.sqrt(np.mean((ecrad_plot[y] - bacardi_plot[x]) ** 2))
bias = np.mean((ecrad_plot[y] - bacardi_plot[x]))
_, ax = plt.subplots(figsize=(12 * cm, 12 * cm))
ax.scatter(bacardi_plot[x], ecrad_plot[y], c=cbc[3])
ax.axline((0, 0), slope=1, color="k", lw=2, transform=ax.transAxes)
ax.set_ylim(lims[i])
ax.set_xlim(lims[i])
ticks = ax.get_yticks() if i == 0 else ax.get_xticks()
ax.set_yticks(ticks)
ax.set_xticks(ticks)
ax.set_aspect('equal')
ax.set_xlabel("BACARDI Irradiance (W$\,$m$^{-2}$)")
ax.set_ylabel("ecRad Irradiance (band 1-13) (W$\,$m$^{-2}$)")
ax.set_title(f"{titles[i]}\nabove cloud whole flight {halo_key}")
ax.grid()
ax.text(0.01, 0.95, f"# points: {sum(~np.isnan(bacardi_plot[x])):.0f}\n"
                    f"RMSE: {rmse.values:.2f}" + " W$\,$m$^{-2}$\n"
                                                 f"Bias: {bias.values:.2f}" + " W$\,$m$^{-2}$",
        horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
plt.tight_layout()
plt.savefig(f"{plot_path}/{halo_flight}_{names[i]}_bacardi_vs_ecrad_banded_scatter_above_cloud_all.png", dpi=300)
plt.show()
plt.close()

# %% compare ecRad and ecRad clearsky simulations only above cloud at flight level - time series
ecrad_plot = ecrad_ds.isel(half_level=height_level_da).where(above_sel)
clearsky_diff = ecrad_plot["flux_dn_sw"] - ecrad_plot["flux_dn_sw_clear"]
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(ecrad_plot.time, ecrad_plot["flux_dn_sw"], label="F$_{\downarrow}$ solar")
ax.plot(ecrad_plot.time, ecrad_plot["flux_dn_sw_clear"], label="F$_{\downarrow}$ solar clearsky")
ax2 = ax.twinx()
ax2.plot(clearsky_diff.time, clearsky_diff, label="ecRad - clearsky ecRad", color=cbc[3])
ax.legend()
ax2.legend(loc=2)
h.set_xticks_and_xlabels(ax, time_extend)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Broadband Irradiance (W$\,$m$^{-2}$)")
ax2.set_ylabel("ecRad - ecRad clearsky (W$\,$m$^{-2}$)")
ax.set_title("ecRad Solar Downward Irradiance {halo_key} above cloud along track")
ax.grid()
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_ecRad_Fdw_solar_clearsky_comparison.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% compare ecRad and ecRad clearsky simulations only above cloud at flight level - scatter plot
ecrad_plot = ecrad_ds.isel(half_level=height_level_da).where(above_sel)
mean_diff = np.mean(ecrad_plot["flux_dn_sw"] - ecrad_plot["flux_dn_sw_clear"])
_, ax = plt.subplots(figsize=(12 * cm, 12 * cm))
ax.scatter(ecrad_plot["flux_dn_sw"], ecrad_plot["flux_dn_sw_clear"], color=cbc[3])
ax.axline((0, 0), slope=1, color="k", lw=2, transform=ax.transAxes)
ax.set_ylim(200, 550)
ax.set_xlim(200, 550)
ticks = ax.get_yticks()
ax.set_yticks(ticks)
ax.set_xticks(ticks)
ax.set_aspect("equal")
ax.text(0.01, 0.95, f"Mean difference: {mean_diff.values:.2f} " + "W$\,$m$^{-2}$", transform=ax.transAxes)
ax.set_title("Solar Downward Irradiance \n{halo_key} above cloud along track")
ax.set_xlabel("ecRad Irradiance (W$\,$m$^{-2}$)")
ax.set_ylabel("ecRad clearsky Irradiance (W$\,$m$^{-2}$)")
ax.grid()
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_ecRad_Fdw_solar_clearsky_comparison_scatter.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot scatterplot of below cloud measurements - repeated
bacardi_plot = bacardi_ds_res.sel(time=below_slice)
ecrad_plot = ecrad_ds.isel(half_level=height_level_da).sel(time=below_slice)
titles = ["Solar Downward Irradiance", "Terrestrial Downward Irradiance", "Solar Upward Irradiance",
          "Terrestrial Upward Irradiance"]
names = ["Fdw_solar", "Fdw_terrestrial", "Fup_solar", "Fup_terrestrial"]
lims = [(120, 240), (80, 130), (95, 170), (210, 220)]
for (i, x), y in zip(enumerate(["F_down_solar", "F_down_terrestrial", "F_up_solar", "F_up_terrestrial"]),
                     ["flux_dn_sw_2", "flux_dn_lw", "flux_up_sw_2", "flux_up_lw"]):
    rmse = np.sqrt(np.mean((ecrad_plot[y] - bacardi_plot[x]) ** 2))
    bias = np.mean((ecrad_plot[y] - bacardi_plot[x]))
    _, ax = plt.subplots(figsize=(12 * cm, 12 * cm))
    ax.scatter(bacardi_plot[x], ecrad_plot[y], c=cbc[3])
    ax.axline((0, 0), slope=1, color="k", lw=2, transform=ax.transAxes)
    ax.set_ylim(lims[i])
    ax.set_xlim(lims[i])
    ticks = ax.get_yticks() if i == 0 else ax.get_xticks()
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)
    ax.set_aspect('equal')
    ax.set_xlabel("BACARDI Irradiance (W$\,$m$^{-2}$)")
    ax.set_ylabel("ecRad Irradiance (W$\,$m$^{-2}$)")
    ax.set_title(f"{titles[i]}\nbelow cloud")
    ax.grid()
    ax.text(0.01, 0.95, f"# points: {sum(~np.isnan(bacardi_plot[x])):.0f}\n"
                        f"RMSE: {rmse.values:.2f}" + " W$\,$m$^{-2}$\n"
                                                     f"Bias: {bias.values:.2f}" + " W$\,$m$^{-2}$",
            horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(f"{plot_path}/{halo_flight}_{names[i]}_bacardi_vs_ecrad_scatter_below_cloud_2.png", dpi=300)
    plt.show()
    plt.close()

# %% plot scatterplot of below cloud measurements - repeated
bacardi_plot = bacardi_ds_res.sel(time=above_slice)
ecrad_plot = ecrad_ds.isel(half_level=height_level_da).sel(time=above_slice)
titles = ["Solar Downward Irradiance", "Terrestrial Downward Irradiance", "Solar Upward Irradiance",
          "Terrestrial Upward Irradiance"]
names = ["Fdw_solar", "Fdw_terrestrial", "Fup_solar", "Fup_terrestrial"]
lims = [(200, 270), (25, 35), (150, 200), (175, 195)]
for (i, x), y in zip(enumerate(["F_down_solar", "F_down_terrestrial", "F_up_solar", "F_up_terrestrial"]),
                     ["flux_dn_sw_2", "flux_dn_lw", "flux_up_sw_2", "flux_up_lw"]):
    rmse = np.sqrt(np.mean((ecrad_plot[y] - bacardi_plot[x]) ** 2))
    bias = np.mean((ecrad_plot[y] - bacardi_plot[x]))
    _, ax = plt.subplots(figsize=(12 * cm, 12 * cm))
    ax.scatter(bacardi_plot[x], ecrad_plot[y], c=cbc[3])
    ax.axline((0, 0), slope=1, color="k", lw=2, transform=ax.transAxes)
    ax.set_ylim(lims[i])
    ax.set_xlim(lims[i])
    ticks = ax.get_yticks() if i == 0 else ax.get_xticks()
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)
    ax.set_aspect('equal')
    ax.set_xlabel("BACARDI Irradiance (W$\,$m$^{-2}$)")
    ax.set_ylabel("ecRad Irradiance (W$\,$m$^{-2}$)")
    ax.set_title(f"{titles[i]}\nbelow cloud")
    ax.grid()
    ax.text(0.01, 0.95, f"# points: {sum(~np.isnan(bacardi_plot[x])):.0f}\n"
                        f"RMSE: {rmse.values:.2f}" + " W$\,$m$^{-2}$\n"
                                                     f"Bias: {bias.values:.2f}" + " W$\,$m$^{-2}$",
            horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(f"{plot_path}/{halo_flight}_{names[i]}_bacardi_vs_ecrad_scatter_above_cloud_2.png", dpi=300)
    plt.show()
    plt.close()

# %% compare ecRad with libRadtran simulations
ecrad_plot = ecrad_ds.isel(half_level=height_level_da)
ifs_sim["fdw_bb"] = ifs_sim["fdw"].sum(dim="lambda")

# %% plot time series of Fdw ecRad and libRadtran
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(ecrad_plot.time, ecrad_plot["flux_dn_sw_2"], label="F$_{\downarrow , sw}$ ecRad")
ax.plot(ifs_sim.time, ifs_sim["fdw_bb"], label="F$_{\downarrow , sw}$ libradtran")
ax.legend()
ax.grid()
h.set_xticks_and_xlabels(ax, time_extend)
ax.set_title("Downward solar irradiance from ecRad and libRadtran simulations using IFS input")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Simulated Broadband Irradiance (W$\,$m$^{-2}$)")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_ecRad_libRadtran_Fdw_solar_time_series.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot difference between ecRad and libRadtran as scatter plot - only above cloud
ifs_sim_plot = ifs_sim.where(above_sel)
ecrad_plot = ecrad_plot.where(above_sel)
lims = [(150, 550), (25, 35), (150, 200), (175, 195)]
i = 0
x, y = "fdw_bb", "flux_dn_sw_2"
rmse = np.sqrt(np.mean((ecrad_plot[y] - ifs_sim_plot[x]) ** 2))
bias = np.mean((ecrad_plot[y] - ifs_sim_plot[x]))
_, ax = plt.subplots(figsize=(12 * cm, 12 * cm))
ax.scatter(ifs_sim_plot[x], ecrad_plot[y], c=cbc[3])
ax.axline((0, 0), slope=1, color="k", lw=2, transform=ax.transAxes)
ax.set_ylim(lims[i])
ax.set_xlim(lims[i])
ticks = ax.get_yticks() if i == 0 else ax.get_xticks()
ax.set_yticks(ticks)
ax.set_xticks(ticks)
ax.set_aspect('equal')
ax.set_xlabel("libRadtran Irradiance (W$\,$m$^{-2}$)")
ax.set_ylabel("ecRad Irradiance (W$\,$m$^{-2}$)")
ax.set_title(f"Downward Solar Irradiance\nabove cloud whole flight {halo_key}")
ax.grid()
ax.text(0.01, 0.95, f"# points: {sum(~np.isnan(ifs_sim_plot[x])):.0f}\n"
                    f"RMSE: {rmse.values:.2f}" + " W$\,$m$^{-2}$\n"
                                                 f"Bias: {bias.values:.2f}" + " W$\,$m$^{-2}$",
        horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
plt.tight_layout()
# plt.savefig(f"{plot_path}/{halo_flight}_ecrad_vs_libRadtran_Fdw_solar_scatter_above_cloud_all.png", dpi=300)
plt.show()
plt.close()

# %% plot CRE ecRad
ecrad_plot = ecrad_ds.isel(half_level=height_level_da).sel(time=case_slice)
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(ecrad_plot.time, ecrad_plot.cre_sw, label="CRE$_{sw}$", color=cbc[2], marker=".")
ax.plot(ecrad_plot.time, ecrad_plot.cre_lw, label="CRE$_{lw}$", color=cbc[3], marker=".")
ax.plot(ecrad_plot.time, ecrad_plot.cre_total, label="CRE$_{net}$", color=cbc[5], marker=".")
ax.axvline(x=above_cloud["end"], label="End above cloud section", color=cbc[-2])
ax.axvline(x=below_cloud["start"], label="Start below cloud section", color=cbc[-1])
ax.legend()
ax.grid()
h.set_xticks_and_xlabels(ax, time_extend_cs)
ax.set_title("Cloud radiative effect from ecRad simulation")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Cloud Radiative Effect (W$\,$m$^{-2}$)")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_ecRad_cre_time_series.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot solar CRE components ecRad
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(ecrad_plot.time, ecrad_plot.flux_dn_sw_2, label="F$_{\downarrow, sw}$", c=cbc[1], marker=".")
ax.plot(ecrad_plot.time, ecrad_plot.flux_up_sw_2, label="F$_{\\uparrow, sw}$", c=cbc[0], marker=".")
ax.plot(ecrad_plot.time, ecrad_plot.flux_dn_sw_clear_2, label="F$_{\downarrow, sw, cls}$", c=cbc[1], ls="--",
        marker=".")
ax.plot(ecrad_plot.time, ecrad_plot.flux_up_sw_clear_2, label="F$_{\\uparrow, sw, cls}$", c=cbc[0], ls="--", marker=".")
ax.axvline(x=above_cloud["end"], label="End above cloud section", color=cbc[-2])
ax.axvline(x=below_cloud["start"], label="Start below cloud section", color=cbc[-1])
ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=3)
ax.grid()
h.set_xticks_and_xlabels(ax, time_extend_cs)
ax.set_title("Broadband solar irradiance fluxes from ecRad simulation")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Broadband Irradiance (W$\,$m$^{-2}$)")
plt.tight_layout()
plt.subplots_adjust(bottom=0.28)
figname = f"{plot_path}/{halo_flight}_ecRad_sw_flux_case_study_time_series.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot lw CRE components ecRad
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(ecrad_plot.time, ecrad_plot.flux_dn_lw, label="F$_{\downarrow, lw}$", c=cbc[1], marker=".")
ax.plot(ecrad_plot.time, ecrad_plot.flux_up_lw, label="F$_{\\uparrow, lw}$", c=cbc[0], marker=".")
ax.plot(ecrad_plot.time, ecrad_plot.flux_dn_lw_clear, label="F$_{\downarrow, lw, cls}$", c=cbc[1], ls="--", marker=".")
ax.plot(ecrad_plot.time, ecrad_plot.flux_up_lw_clear, label="F$_{\\uparrow, lw, cls}$", c=cbc[0], ls="--", marker=".")
ax.axvline(x=above_cloud["end"], label="End above cloud section", color=cbc[-2])
ax.axvline(x=below_cloud["start"], label="Start below cloud section", color=cbc[-1])
ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=3)
ax.grid()
h.set_xticks_and_xlabels(ax, time_extend_cs)
ax.set_title("Broadband long wave irradiance fluxes from ecRad simulation")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Broadband Irradiance (W$\,$m$^{-2}$)")
plt.tight_layout()
plt.subplots_adjust(bottom=0.28)
figname = f"{plot_path}/{halo_flight}_ecRad_lw_flux_case_study_time_series.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot CRE BACARDI
bacardi_plot = bacardi_ds_res.sel(time=case_slice)
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(bacardi_plot.time, bacardi_plot.cre_solar, label="CRE$_{solar}$", color=cbc[2], marker=".")
ax.plot(bacardi_plot.time, bacardi_plot.cre_terrestrial, label="CRE$_{terrestrial}$", color=cbc[3], marker=".")
ax.plot(bacardi_plot.time, bacardi_plot.cre_total, label="CRE$_{net}$", color=cbc[5], marker=".")
ax.axvline(x=above_cloud["end"], label="End above cloud section", color=cbc[-2])
ax.axvline(x=below_cloud["start"], label="Start below cloud section", color=cbc[-1])
ax.legend()
ax.grid()
h.set_xticks_and_xlabels(ax, time_extend_cs)
ax.set_title("Cloud radiative effect from BACARDI and libRadtran simulation")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Cloud Radiative Effect (W$\,$m$^{-2}$)")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_BACARDI_libRadtran_cre_time_series.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot solar CRE components BACARDI, libRadtran
sim_plot = bb_sim_solar_si.sel(time=case_slice)
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(bacardi_plot.time, bacardi_plot.F_down_solar, label="F$_{\downarrow, solar}$", c=cbc[1], marker=".")
ax.plot(bacardi_plot.time, bacardi_plot.F_up_solar, label="F$_{\\uparrow, solar}$", c=cbc[0], marker=".")
ax.plot(sim_plot.time, sim_plot.fdw, label="F$_{\downarrow, solar, cls}$", c=cbc[1], ls="--")
ax.plot(sim_plot.time, sim_plot.eup, label="F$_{\\uparrow, solar, cls}$", c=cbc[0], ls="--")
ax.axvline(x=above_cloud["end"], label="End above cloud section", color=cbc[-2])
ax.axvline(x=below_cloud["start"], label="Start below cloud section", color=cbc[-1])
ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=3)
ax.grid()
h.set_xticks_and_xlabels(ax, time_extend_cs)
ax.set_title("Broadband solar irradiance fluxes from BACARDI measurement and libRadtran simulation")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Broadband Irradiance (W$\,$m$^{-2}$)")
plt.tight_layout()
plt.subplots_adjust(bottom=0.28)
figname = f"{plot_path}/{halo_flight}_BACARDI_libRadtran_solar_flux_case_study_time_series.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot lw CRE components BACARDI, libRadtran
sim_plot = bb_sim_thermal_si.sel(time=case_slice)
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(bacardi_plot.time, bacardi_plot.F_down_terrestrial, label="F$_{\downarrow, terrestrial}$", c=cbc[1], marker=".")
ax.plot(bacardi_plot.time, bacardi_plot.F_up_terrestrial, label="F$_{\\uparrow, terrestrial}$", c=cbc[0], marker=".")
ax.plot(sim_plot.time, sim_plot.edn, label="F$_{\downarrow, terrestrial, cls}$", c=cbc[1], ls="--")
ax.plot(sim_plot.time, sim_plot.eup, label="F$_{\\uparrow, terrestrial, cls}$", c=cbc[0], ls="--")
ax.axvline(x=above_cloud["end"], label="End above cloud section", color=cbc[-2])
ax.axvline(x=below_cloud["start"], label="Start below cloud section", color=cbc[-1])
ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=3)
ax.grid()
h.set_xticks_and_xlabels(ax, time_extend_cs)
ax.set_title("Broadband long wave irradiance fluxes from BACARDI measurement and libRadtran simulation")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Broadband Irradiance (W$\,$m$^{-2}$)")
plt.tight_layout()
plt.subplots_adjust(bottom=0.28)
figname = f"{plot_path}/{halo_flight}_BACARDI_libRadtran_terrestrial_flux_case_study_time_series.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% compare CRE ecRad vs BACARDI
bacardi_plot = bacardi_ds_res.sel(time=below_slice)
ecrad_plot = ecrad_ds.isel(half_level=height_level_da).sel(time=below_slice)
lims = (-30, 40)
var = "cre_total"
rmse = np.sqrt(np.mean((ecrad_plot[var] - bacardi_plot[var]) ** 2))
bias = np.mean((ecrad_plot[var] - bacardi_plot[var]))
_, ax = plt.subplots(figsize=(12 * cm, 12 * cm))
ax.scatter(bacardi_plot[var], ecrad_plot[var], c=cbc[3])
ax.axline((0, 0), slope=1, color="k", lw=2, transform=ax.transAxes)
ax.set_ylim(lims)
ax.set_xlim(lims)
ticks = ax.get_yticks()
ax.set_yticks(ticks)
ax.set_xticks(ticks)
ax.set_aspect('equal')
ax.set_xlabel("BACARDI/libRadtran CRE (W$\,$m$^{-2}$)")
ax.set_ylabel("ecRad CRE (W$\,$m$^{-2}$)")
ax.set_title(f"Cloud radiative effect \nbelow cloud {halo_key}")
ax.grid()
ax.text(0.01, 0.95, f"# points: {sum(~np.isnan(bacardi_plot[var])):.0f}\n"
                    f"RMSE: {rmse.values:.2f}" + " W$\,$m$^{-2}$\n"
                                                 f"Bias: {bias.values:.2f}" + " W$\,$m$^{-2}$",
        ha='left', va='top', transform=ax.transAxes)
plt.tight_layout()
plt.savefig(f"{plot_path}/{halo_flight}_bacardi_vs_ecrad_cre_scatter_below_cloud.png", dpi=300)
plt.show()
plt.close()

# %% statistics of CRE below and above cloud
bacardi_plot = bacardi_ds_res.sel(time=below_slice)
ecrad_plot = ecrad_ds.isel(half_level=height_level_da).sel(time=below_slice)
print(f"BACARDI below cloud mean CRE solar: {bacardi_plot['cre_solar'].mean():.2f} Wm^-2")
print(f"ecRad below cloud mean CRE solar: {ecrad_plot['cre_sw'].mean():.2f} Wm^-2")
bacardi_plot = bacardi_ds_res.sel(time=above_slice)
ecrad_plot = ecrad_ds.isel(half_level=height_level_da).sel(time=above_slice)
print(f"BACARDI above cloud mean CRE solar: {bacardi_plot['cre_solar'].mean():.2f} Wm^-2")
print(f"ecRad above cloud mean CRE solar: {ecrad_plot['cre_sw'].mean():.2f} Wm^-2")

# %% plot BACARDI and ecRad CRE in one plot
bacardi_plot = bacardi_ds_res.sel(time=case_slice)
ecrad_plot = ecrad_ds.isel(half_level=height_level_da).sel(time=case_slice)
_, ax = plt.subplots(figsize=h.figsize_wide)
# ax.plot(bacardi_plot.time, bacardi_plot.cre_solar, label="CRE$_{solar}$", color=cbc[2])
# ax.plot(bacardi_plot.time, bacardi_plot.cre_terrestrial, label="CRE$_{terrestrial}$", color=cbc[3])
ax.plot(bacardi_plot.time, bacardi_plot.cre_total, label="BACARDI CRE$_{net}$", color=cbc[5], marker=".")
ax.plot(ecrad_plot.time, ecrad_plot.cre_total, label="ecRad CRE$_{net}$", color=cbc[6], marker=".")
ax.axvline(x=above_cloud["end"], label="End above cloud section", color=cbc[-2])
ax.axvline(x=below_cloud["start"], label="Start below cloud section", color=cbc[-1])
ax.legend()
ax.grid()
h.set_xticks_and_xlabels(ax, time_extend_cs)
ax.set_title("Net cloud radiative effect from BACARDI and ecRad")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Cloud Radiative Effect (W$\,$m$^{-2}$)")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_BACARDI_vs_ecrad_cre_total_time_series.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% create PDF of BACARDI and ecRad measurements above cirrus clouds
bacardi_plot = bacardi_ds_res.where(cirrus_only)
ecrad_plot = ecrad_ds.isel(half_level=height_level_da).where(cirrus_only)
bin_width = 5
bins = np.arange(75, 280, bin_width)
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.hist(ecrad_plot["flux_up_sw"], bins=bins, label="ecRad F$_{\\uparrow, sw}$", fc=(0, 0, 0, 0), ec=cbc[0],
        histtype="step", density=True)
ax.hist(bacardi_plot["F_up_solar"], bins=bins, label="BACARDI F$_{\\uparrow, solar}$", fc=(0, 0, 0, 0), ec=cbc[1],
        histtype="step", density=True)
ax.legend(loc=2)
ax.set_title("Probability density function above cirrus (single layer)")
ax.set_xlabel("Broadband Irradiance (W$\,$m$^{-2}$)")
ax.set_ylabel("Normalized PDF")
plt.show()
plt.close()

# %% create PDF of BACARDI and ecRad Fdw below cirrus clouds
bacardi_plot = bacardi_ds_res.sel(time=below_slice)
ecrad_plot = ecrad_ds.isel(half_level=height_level_da).sel(time=below_slice)
bin_width = 5
bins = np.arange(75, 280, bin_width)
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.hist(ecrad_plot["flux_dn_sw_2"], bins=bins, label="ecRad F$_\downarrow, sw$", fc=(0, 0, 0, 0), ec=cbc[0],
        histtype="step", density=True)
ax.hist(bacardi_plot["F_up_solar"], bins=bins, label="BACARDI F$_\downarrow, solar$", fc=(0, 0, 0, 0), ec=cbc[1],
        histtype="step", density=True)
ax.legend(loc=1)
ax.set_title("Probability density function below cirrus (single layer)")
ax.set_xlabel("Broadband Irradiance (W$\,$m$^{-2}$)")
ax.set_ylabel("Normalized PDF")
plt.show()
plt.close()

# %% create PDF of BACARDI and ecRad Fup below cirrus clouds
bacardi_plot = bacardi_ds_res.sel(time=below_slice)
ecrad_plot = ecrad_ds.isel(half_level=height_level_da).sel(time=below_slice)
bin_width = 5
bins = np.arange(75, 280, bin_width)
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.hist(ecrad_plot["flux_up_sw_2"], bins=bins, label="ecRad F$_\\uparrow, sw$", fc=(0, 0, 0, 0), ec=cbc[0],
        histtype="step", density=True)
ax.hist(bacardi_plot["F_down_solar"], bins=bins, label="BACARDI F$_\\uparrow, solar$", fc=(0, 0, 0, 0), ec=cbc[1],
        histtype="step", density=True)
ax.legend(loc=1)
ax.set_title("Probability density function below cirrus (single layer)")
ax.set_xlabel("Broadband Irradiance (W$\,$m$^{-2}$)")
ax.set_ylabel("Normalized PDF")
plt.show()
plt.close()

# %% separate BACARDI and ecRad CRE above ocean and above sea ice
bacardi_ci = bacardi_ds_res["cre_solar"].where(cirrus_over_ci)
bacardi_ocean = bacardi_ds_res["cre_solar"].where(cirrus_over_sea)
ecrad_ci = ecrad_ds.isel(half_level=height_level_da)["cre_sw"].where(cirrus_over_ci)
ecrad_ocean = ecrad_ds.isel(half_level=height_level_da)["cre_sw"].where(cirrus_over_sea)

# %% plot PDF of solar CRE BACARDI above ocean and sea ice
bin_width = 10
bins = np.arange(-130, 40, bin_width)
hist_ds = xr.Dataset(dict(ci=bacardi_ci, ocean=bacardi_ocean))
hist_array = np.array([bacardi_ci.values, bacardi_ocean.values]).T
_, ax = plt.subplots(figsize=(18 * cm, 12 * cm))
ax.hist(hist_array, bins=bins, lw=2, density=True, histtype="step", label=["Sea ice", "Ocean"])
ax.text(0.02, 0.65, f"# points ocean: {bacardi_ocean.count().values}\n# points sea ice: {bacardi_ci.count().values}\n"
                    f"bin width: {bin_width}" + "W$\,$m$^{-2}$",
        transform=ax.transAxes, bbox=dict(fc='white', ec='black', alpha=0.8, boxstyle="round"))
ax.legend(loc=2)
ax.grid()
ax.set_title("Solar cloud radiative effect above cirrus from BACARDI")
ax.set_xlabel("Solar Cloud Radiative Effect (W$\,$m$^{-2}$)")
ax.set_ylabel("Normalized PDF")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_bacardi_cre_solar_PDF_ci_vs_ocean.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot PDF of solar CRE ecRad above ocean and sea ice
bin_width = 10
bins = np.arange(-130, 40, bin_width)
hist_array = np.array([ecrad_ci.values, ecrad_ocean.values]).T
_, ax = plt.subplots(figsize=(18 * cm, 12 * cm))
ax.hist(hist_array, bins=bins, lw=2, density=True, histtype="step", label=["Sea ice", "Ocean"], color=[cbc[2], cbc[3]])
ax.text(0.02, 0.65, f"# points ocean: {ecrad_ocean.count().values}\n# points sea ice: {ecrad_ci.count().values}\n"
                    f"bin width: {bin_width}" + "W$\,$m$^{-2}$",
        transform=ax.transAxes, bbox=dict(fc='white', ec='black', alpha=0.8, boxstyle="round"))
ax.legend(loc=2)
ax.grid()
ax.set_title("Solar cloud radiative effect above cirrus from ecRad")
ax.set_xlabel("Solar Cloud Radiative Effect (W$\,$m$^{-2}$)")
ax.set_ylabel("Normalized PDF")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_ecrad_cre_solar_PDF_ci_vs_ocean.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot solar CRE above cirrus BACARDI vs ecRad above ocean
bin_width = 10
bins = np.arange(-130, 20, bin_width)
hist_array = np.array([ecrad_ocean.values, bacardi_ocean.values]).T
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.hist(hist_array, bins=bins, lw=2, density=True, histtype="step", label=["ecRad", "BACARDI"])
ax.text(0.02, 0.65, f"# points BACARDI: {bacardi_ocean.count().values}\n# points ecRad: {ecrad_ocean.count().values}\n"
                    f"bin width: {bin_width}" + "W$\,$m$^{-2}$",
        transform=ax.transAxes, bbox=dict(fc='white', ec='black', alpha=0.8, boxstyle="round"))
ax.legend(loc=2)
ax.grid()
ax.set_title("Solar cloud radiative effect above cirrus above ocean BACARDI vs ecRad")
ax.set_xlabel("Solar Cloud Radiative Effect (W$\,$m$^{-2}$)")
ax.set_ylabel("Normalized PDF")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_bacardi_vs_ecrad_cre_solar_PDF_ocean.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot solar CRE above cirrus BACARDI vs ecRad above sea ice
bin_width = 5
bins = np.arange(-60, 40, bin_width)
hist_array = np.array([ecrad_ci.values, bacardi_ci.values]).T
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.hist(hist_array, bins=bins, lw=2, density=True, histtype="step", label=["ecRad", "BACARDI"])
ax.text(0.02, 0.65, f"# points BACARDI: {bacardi_ci.count().values}\n# points ecRad: {ecrad_ci.count().values}\n"
                    f"bin width: {bin_width}" + "W$\,$m$^{-2}$",
        transform=ax.transAxes, bbox=dict(fc='white', ec='black', alpha=0.8, boxstyle="round"))
ax.legend(loc=2)
ax.grid()
ax.set_title("Solar cloud radiative effect above cirrus above sea ice BACARDI vs ecRad")
ax.set_xlabel("Solar Cloud Radiative Effect (W$\,$m$^{-2}$)")
ax.set_ylabel("Normalized PDF")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_bacardi_vs_ecrad_cre_solar_PDF_ci.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot BACARDI albedo
bacardi_plot = bacardi_ds.sel(time=below_slice)
_, ax = plt.subplots(figsize=h.figsize_wide)
# ax.plot(bacardi_plot.time, bacardi_plot.albedo_solar, label="Solar Albedo", color=cbc[3], marker=".")
ax.plot(bacardi_plot.time, bacardi_plot.albedo_solar_cls, label="Clearsky Solar Albedo", color=cbc[3], marker=".")
ax.legend()
ax.grid()
ax.set_ylim(0.5, 1)
h.set_xticks_and_xlabels(ax, time_extend_ac)
ax.set_title("Below cloud albedo - BACARDI")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Albedo")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_BACARDI_albedo_time_series_below_cloud.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot solar albedo from BACARDI and ecRad above cloud whole flight
bacardi_plot = bacardi_ds_res.where(above_sel)
ecrad_plot = ecrad_ds.isel(half_level=height_level_da).where(above_sel)
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(bacardi_plot.time, bacardi_plot.albedo_solar, label="BACARDI solar albedo", color=cbc[3], marker=".")
ax.plot(ecrad_plot.time, ecrad_plot.albedo_sw, label="ecRad solar albedo", color=cbc[5], marker=".")
ax.legend()
ax.grid()
ax.set_ylim(0, 1)
h.set_xticks_and_xlabels(ax, time_extend)
ax.set_title("Solar albedo above cloud whole flight {halo_key}")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Albedo")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_BACARDI_ecRad_sw_albedo_time_series_above_cloud_all.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot terrestrial albedo from BACARDI and ecRad above cloud whole flight
bacardi_plot = bacardi_ds_res.where(above_sel)
ecrad_plot = ecrad_ds.isel(half_level=height_level_da).where(above_sel)
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(bacardi_plot.time, bacardi_plot.albedo_terrestrial, label="BACARDI terrestrial albedo", color=cbc[3],
        marker=".")
ax.plot(ecrad_plot.time, ecrad_plot.albedo_lw, label="ecRad terrestrial albedo", color=cbc[5], marker=".")
ax.legend()
ax.grid()
h.set_xticks_and_xlabels(ax, time_extend)
ax.set_title("Terrestrial albedo above cloud whole flight {halo_key}")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Albedo")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_BACARDI_ecRad_lw_albedo_time_series_above_cloud_all.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot sea ice albedo parameterization from Ebert and Curry 1992
grid_y = np.array([0.185, 0.25, 0.44, 0.69, 1.19, 2.38, 4.00])
grid_x = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])
cmap = cmr.get_sub_cmap("Greens", 0, 1).reversed()
_, ax = plt.subplots(figsize=h.figsize_wide)
pcm = ax.pcolormesh(grid_x, grid_y, h.ci_albedo.T, cmap=cmap, shading="flat")
for y in grid_y:
    ax.axhline(y, ls="--", color="grey")
# ax.set_yscale("log")
ax.set_xlim(0, 365)
ax.set_ylim(0.185, 4)
ax.set_xlabel("Day of Year")
ax.set_ylabel("Wavelength ($\\mu$m)")
# ax.set_yticks(grid_y)
# ax.set_yticklabels(grid_y)
plt.colorbar(pcm, label="Albedo")
figname = f"{plot_path}/sea_ice_albedo_ifs_ebert_and_curry_1993.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot cirrus flags
ci_flag = h.make_flag(cirrus_over_ci, "Sea Ice")
mciz_flag = h.make_flag(mciz_mask, "MIZ")
ocean_flag = h.make_flag(cirrus_over_sea, "Ocean")
cirrus_flag = h.make_flag(cirrus_only, "Only Cirrus")
_, ax = plt.subplots(figsize=(24 * cm, 6 * cm))
ax.plot(cirrus_over_ci.where(cirrus_over_ci, drop=True).time, ci_flag, marker="o", markersize=9, ls="", color=cbc[6])
ax.plot(mciz_mask.where(mciz_mask, drop=True).time, mciz_flag, marker="o", markersize=9, ls="", color=cbc[1])
ax.plot(cirrus_over_sea.where(cirrus_over_sea, drop=True).time, ocean_flag, marker="o", markersize=9, ls="",
        color=cbc[3])
ax.plot(cirrus_only.where(cirrus_only, drop=True).time, cirrus_flag, marker="o", markersize=9, ls="", color=cbc[4])
ax.set_title("Above cirrus whole flight {halo_key}")
ax.set_xlabel("Time (UTC)")
h.set_xticks_and_xlabels(ax, time_extend)
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_cirrus_flag.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot solar CRE components BACARDI, libRadtran with flags colored
sim_plot = bb_sim_solar_si
bacardi_plot = bacardi_ds_res
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(bacardi_plot.time, bacardi_plot.F_down_solar, label="F$_{\downarrow, solar}$", c=cbc[0], marker=".")
ax.plot(bacardi_plot.time, bacardi_plot.F_up_solar, label="F$_{\\uparrow, solar}$", c=cbc[2], marker=".")
ax.plot(sim_plot.time, sim_plot.fdw, label="F$_{\downarrow, solar, cls}$", c=cbc[0], ls="--")
ax.plot(sim_plot.time, sim_plot.eup, label="F$_{\\uparrow, solar, cls}$", c=cbc[2], ls="--")
# ax.axvline(x=above_cloud["end"], label="End above cloud section", color=cbc[-2])
# ax.axvline(x=below_cloud["start"], label="Start below cloud section", color=cbc[-1])
ax.fill_between(cirrus_over_sea.time, 500, 0, where=cirrus_over_sea, label="Ocean", color=cbc[3], alpha=1)
ax.fill_between(mciz_mask.time, 500, 0, where=mciz_mask, label="MIZ", color=cbc[1], alpha=1)
ax.fill_between(cirrus_over_ci.time, 500, 0, where=cirrus_over_ci, label="Sea Ice", color=cbc[6], alpha=1)
ax.fill_between(cirrus_only.time, 525, 500, where=cirrus_only, label="Cirrus only", color=cbc[4], alpha=1)
ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=4)
ax.grid()
h.set_xticks_and_xlabels(ax, time_extend)
ax.set_title("Broadband solar irradiance fluxes\nfrom BACARDI measurement and libRadtran simulation")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Broadband Irradiance (W$\,$m$^{-2}$)")
ax.set_xlim(pd.to_datetime("2022-04-11 9:30"), pd.to_datetime("2022-04-11 14:30"))
plt.tight_layout()
plt.subplots_adjust(bottom=0.28)
figname = f"{plot_path}/{halo_flight}_BACARDI_libRadtran_solar_flux_time_series.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot sw CRE components ecRad
ecrad_plot = ecrad_ds.isel(half_level=height_level_da)
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(ecrad_plot.time, ecrad_plot.flux_dn_sw, label="F$_{\downarrow, sw}$", c=cbc[0], marker=".")
ax.plot(ecrad_plot.time, ecrad_plot.flux_up_sw, label="F$_{\\uparrow, sw}$", c=cbc[2], marker=".")
ax.plot(ecrad_plot.time, ecrad_plot.flux_dn_sw_clear, label="F$_{\downarrow, sw, cls}$", c=cbc[0], ls="--")
ax.plot(ecrad_plot.time, ecrad_plot.flux_up_sw_clear, label="F$_{\\uparrow, sw, cls}$", c=cbc[2], ls="--")
# ax.axvline(x=above_cloud["end"], label="End above cloud section", color=cbc[-2])
# ax.axvline(x=below_cloud["start"], label="Start below cloud section", color=cbc[-1])
ax.fill_between(cirrus_over_sea.time, 500, 0, where=cirrus_over_sea, label="Ocean", color=cbc[3], alpha=1)
ax.fill_between(mciz_mask.time, 500, 0, where=mciz_mask, label="MIZ", color=cbc[1], alpha=1)
ax.fill_between(cirrus_over_ci.time, 500, 0, where=cirrus_over_ci, label="Sea Ice", color=cbc[6], alpha=1)
ax.fill_between(cirrus_only.time, 525, 500, where=cirrus_only, label="Cirrus only", color=cbc[4], alpha=1)
ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=4)
ax.grid()
h.set_xticks_and_xlabels(ax, time_extend)
ax.set_title("Broadband solar irradiance fluxes from ecRad simulation")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Broadband Irradiance (W$\,$m$^{-2}$)")
ax.set_xlim(pd.to_datetime("2022-04-11 9:30"), pd.to_datetime("2022-04-11 14:30"))
plt.tight_layout()
plt.subplots_adjust(bottom=0.28)
figname = f"{plot_path}/{halo_flight}_ecRad_sw_flux_time_series.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()
# %% plot terrestrial Fup comparison between BACARDI and ecRad
bacardi_plot = bacardi_ds.sel(time=below_slice)
ecrad_plot = ecrad_ds.isel(half_level=height_level_da).sel(time=below_slice)
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(bacardi_plot.time, bacardi_plot.F_up_terrestrial, label=f"BACARDI {labels['F_up_terrestrial']}", color=cbc[3])
ax.plot(ecrad_plot.time, ecrad_plot.flux_up_lw, label=f"ecRad {labels['F_up_terrestrial']}", color=cbc[5], marker=".")
ax.set_ylim(200, 230)
ax.legend()
ax.grid()
h.set_xticks_and_xlabels(ax, time_extend_bc)
ax.set_title("Terrestrial upward irradiance below cloud {halo_key}")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel(f"Irradiance ({units['flux_dn_sw']})")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_BACARDI_ecRad_Fup_terrestrial_time_series_below_cloud.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot radar data below cloud
radar_plot = radar_ds.sel(time=below_slice)
radar_plot.dBZg.plot(x="time", y="height", cmap="viridis", robust=True, figsize=h.figsize_wide)
radar_ds.alt.sel(time=below_slice).plot(label="HALO altitude", c=cbc[-1])
plt.ylim(0, 5000)
plt.show()
plt.close()

# %% plot radar flag below cloud
radar_flag_plot = radar_flag.sel(time=below_slice)
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(radar_flag_plot.time, radar_flag_plot, marker="o", markersize=2, ls="", color=cbc[6])
ax.set_ylim(0, 10)
ax.grid()
plt.show()
plt.close()

# %% plot radar data according to in cloud flag
radar_plot = radar_ds.dBZg.where(in_cloud)
radar_plot.plot(x="time", y="height", cmap="viridis", robust=True, figsize=h.figsize_wide)
radar_ds.alt.sel(time=below_slice).plot(label="HALO altitude", c=cbc[-1])
plt.ylim(0, 5000)
plt.show()
plt.close()

# %% plot 1s lidar data whole flight
lidar_plot = lidar_ds_res.backscatter_ratio
_, ax = plt.subplots(figsize=h.figsize_wide)
lidar_plot.plot(x="time", y="height", robust=True, cmap="plasma", ax=ax)
ax.plot(bahamas_unfiltered.time, bahamas_unfiltered["IRS_ALT"], color="k", label="HALO altitude")
ax.legend(loc=2)
h.set_xticks_and_xlabels(ax, time_extend)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Altitude (m)")
ax.set_title(f"{halo_key} WALES backscatter ratio 532 nm low sensitivity\nresampled to 1s")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_WALES_backscatter_ratio_532_ls_1s.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot original lidar data whole flight -> will fill gaps forward (bad)
lidar_plot = lidar_ds.backscatter_ratio
_, ax = plt.subplots(figsize=h.figsize_wide)
lidar_plot.plot(x="time", y="range", robust=True, cmap="plasma", ax=ax)
ax.plot(bahamas_unfiltered.time, bahamas_unfiltered["IRS_ALT"], color="k", label="HALO altitude")
ax.legend(loc=2)
ax.invert_yaxis()
h.set_xticks_and_xlabels(ax, time_extend)
ax.set_title(f"{halo_key} WALES backscatter ratio 532 nm low sensitivity\noriginal")
plt.show()
plt.close()

# %% plot radar data whole flight
radar_plot = radar_ds
_, ax = plt.subplots(figsize=h.figsize_wide)
radar_plot.dBZg.plot(x="time", y="height", cmap="viridis", robust=True, ax=ax)
radar_ds.alt.plot(label="HALO altitude", c=cbc[-1], ax=ax)
ax.legend()
h.set_xticks_and_xlabels(ax, time_extend)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Altitude (m)")
ax.set_title(f"{halo_key} MIRA reflectivity factor")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_MIRA_reflectivity_factor.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot filtered 1s lidar data whole flight as used for cloud mask
for mask_value in [1.1]:  # , 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]:
    lidar_plot = lidar_ds_res.backscatter_ratio.where(lidar_ds_res.backscatter_ratio > mask_value)
    _, ax = plt.subplots(figsize=h.figsize_wide)
    lidar_plot.plot(x="time", y="height", robust=True, cmap="plasma", ax=ax)
    ax.plot(bahamas_unfiltered.time, bahamas_unfiltered["IRS_ALT"], color="k", label="HALO altitude")
    ax.legend(loc=2)
    h.set_xticks_and_xlabels(ax, time_extend)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title(
        f"{halo_key} WALES backscatter ratio 532 nm low sensitivity\nresampled to 1s, masked with {mask_value}")
    plt.tight_layout()
    figname = f"{plot_path}/{halo_flight}_WALES_backscatter_ratio_532_ls_1s_cloud_mask_{mask_value}.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot despeckled lidar data
lidar_plot = lidar_ds_res.backscatter_ratio.where(~lidar_ds_res["mask"]).where(~lidar_ds_res["spklmask"])
_, ax = plt.subplots(figsize=h.figsize_wide)
lidar_plot.plot(x="time", y="height", robust=True, cmap="plasma", ax=ax)
ax.plot(bahamas_unfiltered.time, bahamas_unfiltered["IRS_ALT"], color="k", label="HALO altitude")
ax.legend(loc=2)
h.set_xticks_and_xlabels(ax, time_extend)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Altitude (m)")
ax.set_title(f"{halo_key} WALES backscatter ratio 532 nm low sensitivity\nresampled to 1s, masked with 1.1, despeckled")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_WALES_backscatter_ratio_532_ls_1s_despeckled.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot interpolated lidar data
lidar_mask = (lidar_ds_res_r["mask"] & lidar_ds_res_r["spklmask"])
lidar_plot = lidar_ds_res_r.backscatter_ratio.where(~lidar_mask)
_, ax = plt.subplots(figsize=h.figsize_wide)
lidar_plot.plot(x="time", y="height", robust=True, cmap="plasma", ax=ax)
ax.plot(bahamas_unfiltered.time, bahamas_unfiltered["IRS_ALT"], color="k", label="HALO altitude")
ax.legend(loc=2)
h.set_xticks_and_xlabels(ax, time_extend)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Altitude (m)")
ax.set_title(
    f"{halo_key} WALES backscatter ratio 532 nm low sensitivity\nresampled to 1s, masked with 1.1, despeckled, interpolated")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_WALES_backscatter_ratio_532_ls_1s_30m.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% combine radar and lidar mask
lidar_mask = lidar_ds_res_r["fill_mask"]
lidar_mask = lidar_mask.where(lidar_mask == 0, 2)
radar_lidar_mask = radar_ds["mask"] + lidar_mask

# %% plot combined radar and lidar mask
plot_ds = radar_lidar_mask
clabel = [x[0] for x in h._CLABEL["detection_status"]]
cbar = [x[1] for x in h._CLABEL["detection_status"]]
clabel = list([clabel[-1], clabel[5], clabel[1], clabel[3]])
cbar = list([cbar[-1], cbar[5], cbar[1], cbar[3]])
cmap = colors.ListedColormap(cbar)
_, ax = plt.subplots(figsize=h.figsize_wide)
pcm = plot_ds.plot(x="time", y="height", cmap=cmap, vmin=-0.5, vmax=len(cbar) - 0.5)
pcm.colorbar.set_ticks(np.arange(len(clabel)), labels=clabel)
ax.plot(bahamas_unfiltered.time, bahamas_unfiltered["IRS_ALT"], color="k", label="HALO altitude")
ax.legend(loc=2)
h.set_xticks_and_xlabels(ax, time_extend)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Altitude (m)")
ax.set_title(f"{halo_key} combined radar lidar mask")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_radar_lidar_mask.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% find cloud bases and tops from lidar mask
mask = ~lidar_ds_res["fill_mask"]
cloud_props_lidar, bases_tops = h.find_bases_tops(mask.values, mask.height.values)
bases_tops = xr.DataArray(bases_tops, dims=["time", "height"], coords=dict(time=mask.time, height=mask.height))

# %% plot bases tops
plot_ds = bases_tops
clabel = list(["bases", "no data", "tops"])
cbar = list([cbc[1], "#ffffff", cbc[-2]])
cmap = colors.ListedColormap(cbar)
_, ax = plt.subplots(figsize=h.figsize_wide)
pcm = plot_ds.plot(x="time", y="height", cmap=cmap, vmin=-1.5, vmax=len(cbar) - 1.5)
pcm.colorbar.set_ticks(np.arange(len(clabel)) - 1, labels=clabel)
ax.plot(bahamas_unfiltered.time, bahamas_unfiltered["IRS_ALT"], color="k", label="HALO altitude")
ax.legend(loc=2)
h.set_xticks_and_xlabels(ax, time_extend)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Altitude (m)")
ax.set_title(f"{halo_key} cloud bases and tops")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_bases_tops_lidar.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% find cloud bases and tops from lidar mask resampled to 30m
mask = ~lidar_ds_res_r["fill_mask"]
cloud_props, bases_tops = h.find_bases_tops(mask.values, mask.height.values)
bases_tops = xr.DataArray(bases_tops, dims=["time", "height"], coords=dict(time=mask.time, height=mask.height))

# %% plot bases tops
plot_ds = bases_tops
clabel = list(["bases", "no data", "tops"])
cbar = list([cbc[1], "#ffffff", cbc[-2]])
cmap = colors.ListedColormap(cbar)
_, ax = plt.subplots(figsize=h.figsize_wide)
pcm = plot_ds.plot(x="time", y="height", cmap=cmap, vmin=-1.5, vmax=len(cbar) - 1.5)
pcm.colorbar.set_ticks(np.arange(len(clabel)) - 1, labels=clabel)
ax.plot(bahamas_unfiltered.time, bahamas_unfiltered["IRS_ALT"], color="k", label="HALO altitude")
ax.legend(loc=2)
h.set_xticks_and_xlabels(ax, time_extend)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Altitude (m)")
ax.set_title(f"{halo_key} cloud bases and tops")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_bases_tops.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% investigate bases tops
bases_tops.sum()  # if it adds up to 0 every base has a corresponding top
(bases_tops == 1).sum(dim="height").plot(x="time")
plt.show()
plt.close()

# %% find cloud bases and tops from radar mask
mask = ~radar_ds["fill_mask"]
cloud_props_radar, bases_tops_radar = h.find_bases_tops(mask.values, mask.height.values)
bases_tops_radar = xr.DataArray(bases_tops_radar, dims=["time", "height"],
                                coords=dict(time=mask.time, height=mask.height))

# %% plot bases tops from radar
plot_ds = bases_tops_radar
clabel = list(["bases", "no data", "tops"])
cbar = list([cbc[1], "#ffffff", cbc[-2]])
cmap = colors.ListedColormap(cbar)
_, ax = plt.subplots(figsize=h.figsize_wide)
pcm = plot_ds.plot(x="time", y="height", cmap=cmap, vmin=-1.5, vmax=len(cbar) - 1.5)
pcm.colorbar.set_ticks(np.arange(len(clabel)) - 1, labels=clabel)
ax.plot(bahamas_unfiltered.time, bahamas_unfiltered["IRS_ALT"], color="k", label="HALO altitude")
ax.legend(loc=2)
h.set_xticks_and_xlabels(ax, time_extend)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Altitude (m)")
ax.set_title(f"{halo_key} cloud bases and tops")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_bases_tops_radar.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% investigate bases tops
bases_tops_radar.sum()  # if it adds up to 0 every base has a corresponding top
(bases_tops_radar == 1).sum(dim="height").plot(x="time")
plt.show()
plt.close()

# %% plot BAHAMAS along track data
bahamas_plot = bahamas_unfiltered.sel(time=case_slice)
_, ax = plt.subplots(figsize=h.figsize_wide)
ax.plot(bahamas_plot.time, bahamas_plot.RHice, label="RH$_{ice}$")
ax.plot(bahamas_plot.time, bahamas_plot.RELHUM, label="RH$_{water}$")
ax.axvline(x=above_cloud["end"], label="End Above Cloud", color=cbc[2])
ax.axvline(x=below_cloud["start"], label="Start Below Cloud", color=cbc[3])
ax.grid()
ax.legend(loc=2)
ax.set_title("BAHAMAS Relative Humidity for case study")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Relative Humidity (%)")
h.set_xticks_and_xlabels(ax, time_extend_cs)
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_BAHAMAS_RH_case_study.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot combined radar and lidar mask with dropsonde locations
plot_ds = radar_lidar_mask
clabel = [x[0] for x in h._CLABEL["detection_status"]]
cbar = [x[1] for x in h._CLABEL["detection_status"]]
clabel = list([clabel[-1], clabel[5], clabel[1], clabel[3]])
cbar = list([cbar[-1], cbar[5], cbar[1], cbar[3]])
cmap = colors.ListedColormap(cbar)
_, ax = plt.subplots(figsize=h.figsize_wide)
pcm = plot_ds.plot(x="time", y="height", cmap=cmap, vmin=-0.5, vmax=len(cbar) - 0.5)
pcm.colorbar.set_ticks(np.arange(len(clabel)), labels=clabel)
ax.plot(bahamas_unfiltered.time, bahamas_unfiltered["IRS_ALT"], color="k", label="HALO altitude")
ax.vlines(dropsonde_ds.launch_time.values, 0, 1, transform=ax.get_xaxis_transform(), color=cbc[1], label="Dropsonde")
ax.legend(loc=2)
h.set_xticks_and_xlabels(ax, time_extend)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Altitude (m)")
ax.set_title(f"{halo_key} combined radar lidar mask")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_radar_lidar_mask_dropsondes.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% group lidar cloud bases and tops in three altitude layers (low, mid, high)
altitude_layers = (0, 1000, 4000, 12000)
layer_labels = ["low(-1km)", "mid(-4km)", "high(-12km)"]
bases = bases_tops == -1
bases_per_layer_lidar = bases.groupby_bins("height", bins=altitude_layers, labels=layer_labels).sum()

# %% plot bases per layer
plot_ds = bases_per_layer_lidar
cbar = cbc[:7]
cmap = colors.ListedColormap(cbar)
time_axis = np.append(plot_ds.time[0] - pd.Timedelta("1S"), plot_ds.time)
X, Y = np.meshgrid(time_axis, np.array([0, 1, 4, 12]))
_, ax = plt.subplots(figsize=h.figsize_wide)
pcm = ax.pcolormesh(X, Y, plot_ds.values.T, cmap=cmap, vmin=-0.5, vmax=len(cbar) - 0.5)
plt.colorbar(pcm)
ax.set_yticks((1, 4, 12))
ax.set_title("Number of cloud bases in different altitude layers (Lidar)")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Altitude (km)")
h.set_xticks_and_xlabels(ax, time_extend)
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_bases_tops_layered.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% resample height grouped cloud bases into minutely bins
bases_binned_lidar = bases_per_layer_lidar.resample(time="1Min").median()

# %% plot minutely median of altitude binned bases
plot_ds = bases_binned_lidar
cbar = cbc[:7]
cmap = colors.ListedColormap(cbar)
time_axis = np.append(plot_ds.time[0] - pd.Timedelta("1S"), plot_ds.time)
X, Y = np.meshgrid(time_axis, np.array([0, 1, 4, 12]))
_, ax = plt.subplots(figsize=h.figsize_wide)
pcm = ax.pcolormesh(X, Y, plot_ds.values.T, cmap=cmap, vmin=-0.5, vmax=len(cbar) - 0.5)
plt.colorbar(pcm)
ax.set_yticks((1, 4, 12))
ax.set_title("Number of cloud bases in different altitude layers (Lidar) \nminutely median")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Altitude (km)")
h.set_xticks_and_xlabels(ax, time_extend)
plt.tight_layout()
# figname = f"{plot_path}/{halo_flight}_bases_tops_layered_1min.png"
# plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% group radar cloud bases and tops in three altitude layers (low, mid, high)
altitude_layers = (0, 1000, 4000, 12000)
layer_labels = ["low(-1km)", "mid(-4km)", "high(-12km)"]
bases = bases_tops_radar == -1
bases_per_layer = bases.groupby_bins("height", bins=altitude_layers, labels=layer_labels).sum()

# %% plot bases per layer
cbar = cbc[:7]
cmap = colors.ListedColormap(cbar)
time_axis = np.append(bases_per_layer.time[0] - pd.Timedelta("1S"), bases_per_layer.time)
X, Y = np.meshgrid(time_axis, np.array([0, 1, 4, 12]))
_, ax = plt.subplots(figsize=h.figsize_wide)
pcm = ax.pcolormesh(X, Y, bases_per_layer.values.T, cmap=cmap, vmin=-0.5, vmax=len(cbar) - 0.5)
plt.colorbar(pcm)
ax.set_yticks((1, 4, 12))
ax.set_title("Number of cloud bases in different altitude layers (Radar)")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Altitude (km)")
h.set_xticks_and_xlabels(ax, time_extend)
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_bases_tops_layered_radar.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% resample height grouped cloud bases into minutely bins
bases_binned = bases_per_layer.resample(time="1Min").median()

# %% plot minutely median of altitude binned bases
cbar = cbc[:7]
cmap = colors.ListedColormap(cbar)
time_axis = np.append(bases_binned.time[0] - pd.Timedelta("1S"), bases_binned.time)
X, Y = np.meshgrid(time_axis, np.array([0, 1, 4, 12]))
_, ax = plt.subplots(figsize=h.figsize_wide)
pcm = ax.pcolormesh(X, Y, bases_binned.values.T, cmap=cmap, vmin=-0.5, vmax=len(cbar) - 0.5)
plt.colorbar(pcm)
ax.set_yticks((1, 4, 12))
ax.set_title("Number of cloud bases in different altitude layers\nminutely median (Radar)")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Altitude (km)")
h.set_xticks_and_xlabels(ax, time_extend)
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_bases_tops_layered_1min_radar.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% turn cloud props into a dataframe and filter clouds which only span over two pixels
cloud_props_df = pd.DataFrame([w["width"] for w in cloud_props_lidar], index=lidar_ds_res.time)
cloud_props_df_filtered = cloud_props_df.where(cloud_props_df > 60)
base_height_df = pd.DataFrame([w["val_cb"] for w in cloud_props_lidar], index=lidar_ds_res.time)
base_height_df = base_height_df.where(~np.isnan(cloud_props_df_filtered))
df1 = base_height_df.apply(lambda x: x < 1000).agg("sum", axis=1)
df2 = base_height_df.apply(lambda x: ((x > 1000) & (x < 4000))).agg("sum", axis=1)
df3 = base_height_df.apply(lambda x: x > 4000).agg("sum", axis=1)
df = pd.concat([df1, df2, df3], axis=1)
df.columns = layer_labels
df = df.stack()
df.index.names = ["time", "layer"]
ds = df.to_xarray()

# %% plot filtered altitude binned cloud bases
plot_ds = ds
cbar = cbc[:7]
cmap = colors.ListedColormap(cbar)
time_axis = np.append(plot_ds.time[0] - pd.Timedelta("1S"), plot_ds.time)
X, Y = np.meshgrid(time_axis, np.array([0, 1, 4, 12]))
_, ax = plt.subplots(figsize=h.figsize_wide)
pcm = ax.pcolormesh(X, Y, plot_ds.values.T, cmap=cmap, vmin=-0.5, vmax=len(cbar) - 0.5)
plt.colorbar(pcm)
ax.set_yticks((1, 4, 12))
ax.set_title("Number of cloud bases in different altitude layers\nfiltered by cloud depth (Lidar)")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Altitude (km)")
h.set_xticks_and_xlabels(ax, time_extend)
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_bases_tops_layered_lidar_filtered.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()
# %%
