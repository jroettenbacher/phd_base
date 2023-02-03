#!/usr/bin/env python
"""2nd PhD Talk: Case Study for RF17 2022-04-11

Arctic Dragon. Flight to the North with a west-east cross-section above and below the cirrus.
Version 2 with calibrated data.

All plots for 2nd PhD Talk:
- BACARDI solar Fdw, Fup above cloud with SZA
- BACARDI solar Fdw, Fup below cloud
- BACARDI terrestrial, solar Fup below cloud
- BACARDI solar Fdw and libradtran clearsky Fdw below cloud
- BACARDI/libRadtran transmissivity below cloud
- ecRad IFS input CIWC for case study with HALO altitude
- scatterplot between ecRad and BACARDI solar and terrestrial up- and downward irradiance below cloud
- scatterplot between ecRad and BACARDI solar and terrestrial up- and downward irradiance above cloud whole flight colored by SZA
- BACARDI and ecRad solar irradiance along flight track
- BACARDI full resolution and resampled and ecRad transmissivity below cloud
- sea ice albedo parameterization after Ebert and Curry 1993


*author*: Johannes Röttenbacher
"""

# %% import modules
import pylim.helpers as h
import pylim.meteorological_formulas as met
import pylim.halo_ac3 as meta
import ac3airborne
from ac3airborne.tools import flightphase
import os
import xarray as xr
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm import tqdm
import cmasher as cmr

cm = 1 / 2.54
cbc = h.get_cb_friendly_colors()

# %% set paths
campaign = "halo-ac3"
date = "20220411"
halo_key = "RF17"
halo_flight = f"HALO-AC3_{date}_HALO_{halo_key}"

plot_path = f"{h.get_path('plot', halo_flight, campaign)}/{halo_flight}"
smart_path = f"{h.get_path('calibrated', halo_flight, campaign)}"
smart_file = f"HALO-AC3_HALO_SMART_spectral-irradiance-Fdw_{date}_{halo_key}_v1.0.nc"
libradtran_exp_path = h.get_path("libradtran_exp", campaign=campaign)
libradtran_bb_solar = f"HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_solar_{date}_{halo_key}.nc"
libradtran_bb_solar_si = f"HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_solar_si_{date}_{halo_key}.nc"
bahamas_path = h.get_path("bahamas", halo_flight, campaign)
bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{halo_key}_v1_1s.nc"
bacardi_path = h.get_path("bacardi", halo_flight, campaign)
bacardi_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{halo_key}_R1_1s.nc"
ifs_path = f"{h.get_path('ifs', campaign=campaign)}"
ecrad_path = f"{h.get_path('ecrad', campaign=campaign)}/{date}"
ecrad_file = "ecrad_merged_inout_20220411_v1_mean.nc"

# set up metadata for access to HALO-AC3 cloud
kwds = {'simplecache': dict(same_names=True)}
credentials = {"user": os.environ.get("AC3_CLOUD_USER"), "password": os.environ.get("AC3_CLOUD_PASSWORD")}
cat = ac3airborne.get_intake_catalog()["HALO-AC3"]["HALO"]

# %% get flight segmentation and select below and above cloud section
segmentation = ac3airborne.get_flight_segments()["HALO-AC3"]["HALO"][f"HALO-AC3_HALO_{halo_key}"]
segments = flightphase.FlightPhaseFile(segmentation)
above_cloud, below_cloud = dict(), dict()
if "RF17" in halo_flight:
    above_cloud["start"] = segments.select("name", "high level 7")[0]["start"]
    above_cloud["end"] = segments.select("name", "high level 8")[0]["end"]
    below_cloud["start"] = segments.select("name", "high level 9")[0]["start"]
    below_cloud["end"] = segments.select("name", "high level 10")[0]["end"]
    above_slice = slice(above_cloud["start"], above_cloud["end"])
    below_slice = slice(below_cloud["start"], below_cloud["end"])
    case_slice = slice(above_cloud["start"], below_cloud["end"])
else:
    above_cloud["start"] = segments.select("name", "polygon pattern 1")[0]["start"]
    above_cloud["end"] = segments.select("name", "polygon pattern 1")[0]["parts"][-1]["start"]
    below_cloud["start"] = segments.select("name", "polygon pattern 2")[0]["start"]
    below_cloud["end"] = segments.select("name", "polygon pattern 2")[0]["end"]
    above_slice = slice(above_cloud["start"], above_cloud["end"])
    below_slice = slice(below_cloud["start"], below_cloud["end"])
    case_slice = slice(above_cloud["start"], below_cloud["end"])

cloudy_times = meta.cloudy_times["RF17"]["high"]

# %% read in SMART data
smart_ds = xr.open_dataset(f"{smart_path}/{smart_file}")
time_slice = slice(smart_ds.time[0].values, smart_ds.time[-1].values)

# %% read in data from HALO-AC3 cloud
ins = cat["GPS_INS"][f"HALO-AC3_HALO_{halo_key}"](storage_options=kwds, **credentials).to_dask()

# %% read in BACARDI and BAHAMAS data resampled to 1 sec
bacardi_ds = xr.open_dataset(f"{bacardi_path}/{bacardi_file}")
bahamas_ds = xr.open_dataset(f"{bahamas_path}/{bahamas_file}")

# %% read in libRadtran simulation
bb_sim_solar_si = xr.open_dataset(f"{libradtran_exp_path}/{libradtran_bb_solar_si}")
fdw_inp = bb_sim_solar_si.fdw.interp(time=bacardi_ds.time)  # interpolate simulated fdw onto bacardi time

# %% read in ecrad data
ecrad_ds = xr.open_dataset(f"{ecrad_path}/{ecrad_file}")
# replace numeric nan values with nan
ecrad_ds["re_ice"] = ecrad_ds.re_ice.where(ecrad_ds.re_ice != 5.196162e-05, np.nan)
ecrad_ds["re_liquid"] = ecrad_ds.re_liquid.where(ecrad_ds.re_liquid != 4.000001e-06, np.nan)

# %% cut to SMART time
ecrad_ds = ecrad_ds.sel(time=time_slice)
bacardi_ds = bacardi_ds.sel(time=time_slice)
bahamas_ds = bahamas_ds.sel(time=time_slice)
bb_sim_solar_si = bb_sim_solar_si.sel(time=time_slice)
fdw_inp = fdw_inp.sel(time=time_slice)
sza = bacardi_ds.sza

# %% filter values which exceeded certain motion threshold
roll_center = np.abs(bahamas_ds["IRS_PHI"].median())  # -> 0.05...
roll_threshold = 0.5
# pitch is not centered on 0 thus we need to calculate the difference to the center and compare that to the threshold
pitch_center = np.abs(bahamas_ds["IRS_THE"].median())
pitch_threshold = 0.34
# True -> keep value, False -> drop value (Nan)
roll_filter = np.abs(bahamas_ds["IRS_PHI"]) < roll_threshold
pitch_filter = np.abs(bahamas_ds["IRS_THE"] - pitch_center) < pitch_threshold
motion_filter = roll_filter & pitch_filter
bacardi_ds = bacardi_ds.where(motion_filter)
bahamas_ds = bahamas_ds.where(motion_filter)

# %% filter values from bacardi which correspond to turns or descents
selected_segments = segments.findSegments(above_cloud["start"].strftime('%Y-%m-%d %H:%M:%S'),
                                          below_cloud["end"].strftime('%Y-%m-%d %H:%M:%S'))
starts, ends = list(), list()
for dic in selected_segments:
    if "short_turn" in dic["kinds"] or "large_descent" in dic["kinds"]:
        starts.append(dic["start"])
        ends.append(dic["end"])
starts, ends = pd.to_datetime(starts), pd.to_datetime(ends)
for i in range(len(starts)):
    sel_time = (bacardi_ds.time > starts[i]) & (bacardi_ds.time < ends[i])
    bacardi_ds = bacardi_ds.where(~sel_time)

# %% calculate albedo from BACARDI and libRadtran and from ecRad
bacardi_ds["albedo_solar"] = bacardi_ds["F_up_solar"] / bacardi_ds["F_down_solar"]
bacardi_ds["albedo_solar_cls"] = bacardi_ds["F_up_solar"] / fdw_inp
bacardi_ds["albedo_terrestrial"] = bacardi_ds["F_up_terrestrial"] / bacardi_ds["F_down_terrestrial"]
ecrad_ds["albedo_sw"] = ecrad_ds["flux_up_sw"] / ecrad_ds["flux_dn_sw"]
ecrad_ds["albedo_sw_cls"] = ecrad_ds["flux_up_sw"] / ecrad_ds["flux_dn_sw_clear"]
ecrad_ds["albedo_lw"] = ecrad_ds["flux_up_lw"] / ecrad_ds["flux_dn_lw"]
ecrad_ds["albedo_lw_cls"] = ecrad_ds["flux_up_sw"] / ecrad_ds["flux_dn_lw_clear"]

# %% calculate transmissivity BACARDI/libRadtran and ecRad
bacardi_ds["transmissivity_solar"] = bacardi_ds["F_down_solar"] / fdw_inp
ecrad_ds["transmissivity_sw"] = ecrad_ds["flux_dn_sw"] / ecrad_ds["flux_dn_sw_clear"]

# %% calculate standard deviation of transmissivity below cloud
bacardi_ds["transmissivity_solar_std"] = bacardi_ds["transmissivity_solar"].sel(time=below_slice).std()

# %% resample data to minutely resolution
bb_sim_solar_si = bb_sim_solar_si.resample(time="1Min").asfreq()
bacardi_ds_res = bacardi_ds.resample(time="1Min").mean()
bacardi_ds_res = bacardi_ds_res.sel(time=time_slice)

# %% create a boolean mask for above cloud time steps in 1 min data for whole flight
end_ascend = pd.to_datetime(segments.select("name", "high level 1")[0]["start"])
end_ascend2 = pd.to_datetime(segments.select("name", "high level 11")[0]["start"])
t = pd.to_datetime(bacardi_ds_res.time)
above_sel = xr.DataArray(((t > end_ascend) & (t < above_cloud["end"])) | (t > end_ascend2),
                         coords={"time": bacardi_ds_res.time})

# %% plotting variables
time_extend = pd.to_timedelta((bahamas_ds.time[-1] - bahamas_ds.time[0]).values)  # get time extend for x-axis labeling
time_extend_cs = below_cloud["end"] - above_cloud["start"]  # time extend for case study
time_extend_ac = (above_cloud["end"] - above_cloud["start"])
time_extend_bc = below_cloud["end"] - below_cloud["start"]
h.set_cb_friendly_colors()
plt.rc("font", size=12)
figsize_wide = (24 * cm, 12 * cm)
figsize_equal = (12 * cm, 12 * cm)

# %% get height level of actual flight altitude in ecRad model on half levels
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

ecrad_ds["press_height_hl"] = xr.merge(p_array_list).pressure_height
ecrad_ds["press_height_hl"] = ecrad_ds["press_height_hl"].where(~np.isnan(ecrad_ds["press_height_hl"]), 80000)
ins_tmp = ins.sel(time=ecrad_ds.time, method="nearest")
ecrad_timesteps = len(ecrad_ds.time)
aircraft_height_level = np.zeros(ecrad_timesteps)

for i in tqdm(range(ecrad_timesteps)):
    aircraft_height_level[i] = h.arg_nearest(ecrad_ds["press_height_hl"][i, :].values, ins_tmp.alt[i].values)

aircraft_height_level = aircraft_height_level.astype(int)
height_level_da = xr.DataArray(aircraft_height_level, dims=["time"], coords={"time": ecrad_ds.time})
aircraft_height = ecrad_ds["press_height_hl"].isel(half_level=height_level_da)

# %% get height level of actual flight altitude in ecRad model on full levels
press_height = ecrad_ds[["pressure_full", "t"]]
p_array_list = list()
for time in tqdm(press_height.time):
    tmp = press_height.sel(time=time, drop=True)
    press_height_new = met.barometric_height(tmp["pressure_full"], tmp["t"])
    p_array = xr.DataArray(data=press_height_new[None, :], dims=["time", "level"],
                           coords={"level": (["level"], np.flip(tmp.level.values)),
                                   "time": np.array([time.values])},
                           name="pressure_height")
    p_array_list.append(p_array)

ecrad_ds["press_height_full"] = xr.merge(p_array_list).pressure_height
ecrad_ds["press_height_full"] = ecrad_ds["press_height_full"].where(~np.isnan(ecrad_ds["press_height_full"]), 80000)
aircraft_height_level_full = np.zeros(ecrad_timesteps)

for i in tqdm(range(ecrad_timesteps)):
    aircraft_height_level_full[i] = h.arg_nearest(ecrad_ds["press_height_full"][i, :].values, ins_tmp.alt[i].values)

aircraft_height_level_full = aircraft_height_level_full.astype(int)
height_level_da_full = xr.DataArray(aircraft_height_level_full, dims=["time"], coords={"time": ecrad_ds.time})
aircraft_height_full = ecrad_ds["press_height_full"].isel(level=height_level_da_full)

# %% plotting dictionaries for BACARDI
labels = dict(F_down_solar=r"$F_{\downarrow, solar}$", F_down_terrestrial=r"$F_{\downarrow, terrestrial}$",
              F_up_solar=r"$F_{\uparrow, solar}$", F_up_terrestrial=r"$F_{\uparrow, terrestrial}$")

# %% prepare metadata for comparing ecRad and BACARDI
titles = ["Solar Downward Irradiance", "Terrestrial Downward Irradiance", "Solar Upward Irradiance",
          "Terrestrial Upward Irradiance"]
names = ["Fdw_solar", "Fdw_terrestrial", "Fup_solar", "Fup_terrestrial"]
bacardi_vars = ["F_down_solar", "F_down_terrestrial", "F_up_solar", "F_up_terrestrial"]
ecrad_vars = ["flux_dn_sw", "flux_dn_lw", "flux_up_sw", "flux_up_lw"]

# %% calculate difference between BACARDI and ecRad
bacardi_plot = bacardi_ds_res.where(above_sel)
ecrad_plot = ecrad_ds.isel(half_level=height_level_da).where(above_sel)
ecrad_bacardi = xr.Dataset()
for x, y in zip(bacardi_vars, ecrad_vars):
    bacardi_ecrad = (bacardi_plot[x] - ecrad_plot[y]).sel(time=above_slice)
    ecrad_bacardi[x] = ecrad_plot[y] - bacardi_plot[x]
    print(f"BACARDI {x} - ecRad {y} = {bacardi_ecrad.mean():.2f} Wm^-2")

# %% plot BACARDI measurements of above cloud section solar only
bacardi_ds_slice = bacardi_ds.sel(time=above_slice)
bacardi_plot2 = bacardi_ds.sel(time=above_slice)
bacardi_error = bacardi_plot2 * 0.03
var1, var2 = "F_down_solar", "F_up_solar"
sza_plot = sza.sel(time=above_slice)
plt.rc("font", size=10)
_, ax = plt.subplots(figsize=(18 * cm, 8 * cm))
ax.plot(bacardi_ds_slice.time, bacardi_ds_slice[var1], label=labels[var1], c=cbc[0])
ax.fill_between(bacardi_error.time, bacardi_plot2[var1] + bacardi_error[var1],
                bacardi_plot2[var1] - bacardi_error[var1],
                color=cbc[0], alpha=0.5)
ax.plot(bacardi_ds_slice.time, bacardi_ds_slice[var2], label=labels[var2], c=cbc[2])
ax.fill_between(bacardi_error.time, bacardi_plot2[var2] + bacardi_error[var2],
                bacardi_plot2[var2] - bacardi_error[var2],
                color=cbc[2], alpha=0.5)
ax.set_ylim(100, 260)
ax.grid()
h.set_xticks_and_xlabels(ax, time_extend_ac)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Broadband Irradiance (W$\,$m$^{-2}$)")
ax.set_title("BACARDI broadband irradiance above cloud")
ax2 = ax.twinx()
ax2.plot(sza_plot.time, sza_plot, label="SZA", c=cbc[3])
ax2.set_ylabel("Solar Zenith Angle (°)")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
h1.append(h2[0])
l1.append(l2[0])
ax2.legend(h1, l1, loc=4)
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_BACARDI_solar_irradiance_above.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot BACARDI measurements of below cloud section solar only
bacardi_plot = bacardi_ds.sel(time=below_slice)
bacardi_error = bacardi_plot * 0.03
var1, var2 = "F_down_solar", "F_up_solar"
plt.rc("font", size=10)
_, ax = plt.subplots(figsize=(16 * cm, 8 * cm))
ax.plot(bacardi_plot.time, bacardi_plot[var1], label=labels[var1], c=cbc[0])
ax.fill_between(bacardi_error.time, bacardi_plot[var1] + bacardi_error[var1], bacardi_plot[var1] - bacardi_error[var1],
                color=cbc[0], alpha=0.5)
ax.plot(bacardi_plot.time, bacardi_plot[var2], label=labels[var2], c=cbc[2])
ax.fill_between(bacardi_error.time, bacardi_plot[var2] + bacardi_error[var2], bacardi_plot[var2] - bacardi_error[var2],
                color=cbc[2], alpha=0.5)
ax.set_ylim(100, 260)
ax.grid()
ax.legend()
h.set_xticks_and_xlabels(ax, time_extend_ac)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Broadband Irradiance (W$\,$m$^{-2}$)")
ax.set_title("BACARDI broadband irradiance below cloud")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_BACARDI_solar_irradiance_below.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot BACARDI measurements of below cloud section upward only
bacardi_plot = bacardi_ds.sel(time=below_slice)
bacardi_error = bacardi_plot * 0.03
var1, var2 = "F_up_solar", "F_up_terrestrial"
plt.rc("font", size=10)
_, ax = plt.subplots(figsize=(16 * cm, 8 * cm))
# ax.plot(bacardi_plot.time, bacardi_plot["F_down_solar"], label=labels["F_down_solar"], c=cbc[0])
ax.plot(bacardi_plot.time, bacardi_plot[var1], label=labels[var1], c=cbc[2])
ax.fill_between(bacardi_error.time, bacardi_plot[var1] + bacardi_error[var1], bacardi_plot[var1] - bacardi_error[var1],
                color=cbc[2], alpha=0.5)
ax.plot(bacardi_plot.time, bacardi_plot[var2], label=labels[var2], c=cbc[1])
ax.fill_between(bacardi_error.time, bacardi_plot[var2] + bacardi_error[var2], bacardi_plot[var2] - bacardi_error[var2],
                color=cbc[1], alpha=0.5)
ax.set_ylim(100, 260)
ax.grid()
ax.legend()
h.set_xticks_and_xlabels(ax, time_extend_ac)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Broadband Irradiance (W$\,$m$^{-2}$)")
ax.set_title("BACARDI broadband irradiance below cloud")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_BACARDI_upward_irradiance_below.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot BACARDI Fdw and libRadtran simulation Fdw below cloud
h.set_cb_friendly_colors()
bacardi_plot = bacardi_ds.sel(time=below_slice)
libradtran = bb_sim_solar_si.sel(time=below_slice)
bacardi_error = bacardi_plot * 0.03
plt.rc("font", size=12)
_, ax = plt.subplots(figsize=figsize_wide)
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

# %% plot transmissivity calculated from BACARDI and libRadtran
h.set_cb_friendly_colors()
bacardi_plot = bacardi_ds["transmissivity_solar"].sel(time=below_slice)
bacardi_error = bacardi_plot * 0.03
_, ax = plt.subplots(figsize=figsize_wide)
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

# %% prepare metadata for plotting IFS data in ecrad dataset
variable = "ciwc"
units = dict(cloud_fraction="", clwc="mg$\,$kg$^{-1}$", ciwc="mg$\,$kg$^{-1}$", cswc="g$\,$kg$^{-1}$",
             crwc="g$\,$kg$^{-1}$", t="K", q="g$\,$kg$^{-1}$", re_ice="$\mu$m", flux_dn_sw="W$\,$m$^{-2}$")
scale_factor = dict(cloud_fraction=1, clwc=1e6, ciwc=1e6, cswc=1000, crwc=1000, t=1, q=1000, re_ice=1e6)
colorbarlabel = dict(cloud_fraction="Cloud Fraction", clwc="Cloud Liquid Water Content", ciwc="Cloud Ice Water Content",
                     cswc="Cloud Snow Water Content", crwc="Cloud Rain Water Content", t="Temperature",
                     q="Specific Humidity", re_ice="Ice Effective Radius", flux_dn_sw="Downward Solar Irradiance")
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
ax.set_xlim(pd.to_datetime("2022-04-11 10:30"), pd.to_datetime("2022-04-11 12:30"))
ax.set_ylim(0, 13)
ax.legend(loc=2)
h.set_xticks_and_xlabels(ax, time_extend_cs)
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_ecrad_{variable}_along_track.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot scatterplot of below cloud measurements
bacardi_plot = bacardi_ds_res.sel(time=below_slice)
ecrad_plot = ecrad_ds.isel(half_level=height_level_da).sel(time=below_slice)
lims = [(120, 240), (80, 130), (95, 170), (210, 220)]
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
    ax.set_title(f"{titles[i]}\nabove cloud whole flight RF17")
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

# %% calculate correlation coefficient between SZA and ecrad BACARDI difference
drop_nans = ~np.isnan(ecrad_bacardi["F_down_solar"])
ecrad_bacardi_raw = ecrad_bacardi.where(drop_nans, drop=True)
sza = bacardi_plot["sza"].where(drop_nans, drop=True)
pearsonr(sza, ecrad_bacardi_raw["F_down_solar"])

# %% plot Fdw solar along the flight track ecRad and BACARDI
bacardi_plot = bacardi_ds_res
ecrad_plot = ecrad_ds.isel(half_level=height_level_da)
i = 0
_, ax = plt.subplots(figsize=figsize_wide)
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
ax.set_title(f"Solar Broadband Irradiance - BACARDI and ecRad RF17")
h.set_xticks_and_xlabels(ax, time_extend)
ax.grid()
ax.legend(ncol=2)
plt.tight_layout()
plt.savefig(f"{plot_path}/{halo_flight}_bacardi_ecrad_fluxes_time_series.png", dpi=300)
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
_, ax = plt.subplots(figsize=figsize_wide)
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

# %% plot sea ice albedo parameterization from Ebert and Curry 1992
grid_y = np.array([0.185, 0.25, 0.44, 0.69, 1.19, 2.38, 4.00])
grid_x = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])
cmap = cmr.get_sub_cmap("Greens", 0, 1).reversed()
_, ax = plt.subplots(figsize=figsize_wide)
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
