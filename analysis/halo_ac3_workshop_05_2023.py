#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 05.05.2023

Most plots for the presentation during the |haloac3| workshop in Leipzig 09 - 11 May 2023.

"""

# %% module import
import pylim.helpers as h
from pylim import reader
import pylim.halo_ac3 as meta
import ac3airborne
from ac3airborne.tools import flightphase
import sys

sys.path.append('./larda')
from larda.pyLARDA.spec2mom_limrad94 import despeckle
from metpy.units import units
from metpy.calc import density
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import cmasher as cmr
import xarray as xr
import pandas as pd
from tqdm import tqdm
import logging

mpl.use('module://backend_interagg')

log = logging.getLogger("pylim")
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

cm = 1 / 2.54
cbc = h.get_cb_friendly_colors()

# %% set up paths and meta data
campaign = "halo-ac3"
key = "RF17"
flight = meta.flight_names[key]
date = flight[9:17]
ecrad_path = f"{h.get_path('ecrad', campaign=campaign)}/{date}"
bahamas_path = h.get_path("bahamas", flight, campaign)
bacardi_path = h.get_path("bacardi", flight, campaign)
radar_path = h.get_path("hamp_mira", flight, campaign)
lidar_path = h.get_path("wales", flight, campaign)
varcloud_path = h.get_path("varcloud", flight, campaign)
plot_path = f"{h.get_path('plot', campaign=campaign)}/{flight}/HALO-AC3_workshop"

# file names
bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{key}_v1.nc"
bacardi_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1.nc"
bacardi_res_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1_1s.nc"
ecrad_v1 = f"ecrad_merged_inout_{date}_v1.nc"
ecrad_v8 = f"ecrad_merged_output_{date}_v8.nc"
ecrad_v10 = f"ecrad_merged_output_{date}_v10.nc"
ecrad_inputv2 = f"ecrad_merged_input_{date}_v2.nc"
ecrad_inputv3 = f"ecrad_merged_input_{date}_v3.nc"
radar_file = f"radar_{date}_v1.6.nc"
lidar_file = f"HALO-AC3_HALO_WALES_bsrgl_{date}_{key}_V2.0.nc"
varcloud_file = [f for f in os.listdir(varcloud_path) if "nc" in f][0]

# set options and credentials for HALO-AC3 cloud and intake catalog
kwds = {'simplecache': dict(same_names=True)}
credentials = {"user": os.environ.get("AC3_CLOUD_USER"), "password": os.environ.get("AC3_CLOUD_PASSWORD")}
cat = ac3airborne.get_intake_catalog()

# %% get flight segmentation and select below and above cloud section
fl_segments = ac3airborne.get_flight_segments()["HALO-AC3"]["HALO"][f"HALO-AC3_HALO_{key}"]
segments = flightphase.FlightPhaseFile(fl_segments)
above_cloud, below_cloud = dict(), dict()
above_cloud["start"] = segments.select("name", "high level 7")[0]["start"]
above_cloud["end"] = segments.select("name", "high level 8")[0]["end"]
below_cloud["start"] = segments.select("name", "high level 9")[0]["start"]
below_cloud["end"] = segments.select("name", "high level 10")[0]["end"]
above_slice = slice(above_cloud["start"], above_cloud["end"])
below_slice = slice(below_cloud["start"], below_cloud["end"])
# above cloud time with thin cirrus below
sel_time = slice(above_cloud["start"], pd.to_datetime("2022-04-11 11:04"))

# %% read in data
bahamas = reader.read_bahamas(f"{bahamas_path}/{bahamas_file}")
bacardi_ds = xr.open_dataset(f"{bacardi_path}/{bacardi_file}")
bacardi_ds_res = xr.open_dataset(f"{bacardi_path}/{bacardi_res_file}")
ecrad_ds_v1 = xr.open_dataset(f"{ecrad_path}/{ecrad_v1}")
ecrad_ds_v8 = xr.open_dataset(f"{ecrad_path}/{ecrad_v8}")
ecrad_ds_v10 = xr.open_dataset(f"{ecrad_path}/{ecrad_v10}")
ecrad_ds_inputv2 = xr.open_dataset(f"{ecrad_path}/{ecrad_inputv2}")
ecrad_ds_inputv3 = xr.open_dataset(f"{ecrad_path}/{ecrad_inputv3}")
ins = cat["HALO-AC3"]["HALO"]["GPS_INS"][f"HALO-AC3_HALO_{key}"](storage_options=kwds, **credentials).to_dask()
radar_ds = xr.open_dataset(f"{radar_path}/{radar_file}")
radar_ds["height"] = radar_ds.height / 1000
varcloud_ds = xr.open_dataset(f"{varcloud_path}/{varcloud_file}")
varcloud_ds = varcloud_ds.swap_dims(time="Time", height="Height").rename(Time="time")
varcloud_ds = varcloud_ds.rename(Varcloud_Cloud_Ice_Water_Content="iwc",
                                 Varcloud_Cloud_Ice_Effective_Radius="re_ice")
varcloud_ds["Height"] = varcloud_ds.Height / 1000
# filter -888 values
radar_ds["dBZg"] = radar_ds.dBZg.where(np.isnan(radar_ds.radar_flag) & ~radar_ds.dBZg.isin(-888))

# %% read in lidar data V2
lidar_ds = xr.open_dataset(f"{lidar_path}/{lidar_file}")
lidar_ds["altitude"] = lidar_ds["altitude"] / 1000
lidar_ds = lidar_ds.rename(altitude="height").transpose("time", "height")
# convert lidar data to radar convention: [time, height], ground = 0m
lidar_height = lidar_ds.height

# %% plotting meta
time_extend = pd.to_timedelta((ins.time[-1] - ins.time[0]).values)  # get time extend for x-axis labeling
time_extend_cs = below_cloud["end"] - above_cloud["start"]  # time extend for case study
time_extend_ac = (above_cloud["end"] - above_cloud["start"])
time_extend_bc = below_cloud["end"] - below_cloud["start"]
case_slice = slice(above_cloud["start"], below_cloud["end"])
plt.rcdefaults()
h.set_cb_friendly_colors()
# prepare metadata for comparing ecRad and BACARDI
titles = ["Solar Downward Irradiance", "Terrestrial Downward Irradiance", "Solar Upward Irradiance",
          "Terrestrial Upward Irradiance"]
names = ["Fdw_solar", "Fdw_terrestrial", "Fup_solar", "Fup_terrestrial"]
bacardi_vars = ["F_down_solar", "F_down_terrestrial", "F_up_solar", "F_up_terrestrial"]
ecrad_vars = ["flux_dn_sw", "flux_dn_lw", "flux_up_sw", "flux_up_lw"]

# %% prepare ecrad data
ecrad_ds = ecrad_ds_v1.mean(dim="column")  # take mean over columns
ecrad_ds_v8 = xr.merge([ecrad_ds_v8, ecrad_ds_inputv2])
ecrad_ds_v10 = xr.merge([ecrad_ds_v10, ecrad_ds_inputv3])
ecrad_dict = dict(v1=ecrad_ds, v8=ecrad_ds_v8, v10=ecrad_ds_v10)

for k in ecrad_dict:
    ds = ecrad_dict[k].copy()
    ds = ds.assign_coords({"band_sw": range(1, 15), "band_lw": range(1, 17), "sw_albedo_band": range(1, 7)})

    # replace default values with nan
    ds["re_ice"] = ds.re_ice.where(np.abs(ds.re_ice - 5.19616e-05) > 1e-10, np.nan)
    ds["re_liquid"] = ds.re_liquid.where(ds.re_liquid != 4.000001e-06, np.nan)

    # convert kg/kg to kg/m³
    air_density = density(ds.pressure_full * units.Pa, ds.t * units.K, ds.q * units("kg/kg"))
    ds["iwc"] = ds["q_ice"] * units("kg/kg") * air_density
    ds["iwc"] = ds["iwc"].where(~np.isnan(ds.re_ice), np.nan)

    # calculate iwp
    factor = ds.pressure_hl.diff(dim="half_level").to_numpy() / (9.80665 * ds.cloud_fraction.to_numpy())
    ds["iwp"] = (["time", "level"], factor * ds.ciwc.to_numpy())
    ds["iwp"] = ds.iwp.where(ds.iwp != np.inf, np.nan)

    # calculate radiative effect
    ds["cre_sw"] = (ds.flux_dn_sw - ds.flux_up_sw) - (ds.flux_dn_sw_clear - ds.flux_up_sw_clear)
    ds["cre_lw"] = (ds.flux_dn_lw - ds.flux_up_lw) - (ds.flux_dn_lw_clear - ds.flux_up_lw_clear)
    ds["cre_total"] = ds["cre_sw"] + ds["cre_lw"]

    ecrad_dict[k] = ds.copy()

# %% get height level of actual flight altitude in ecRad model, this determines only the index of the level
aircraft_height_da, height_level_da = dict(), dict()
for v in ["v1", "v10"]:
    ds = ecrad_dict[v]
    bahamas_tmp = bahamas.sel(time=ds.time, method="nearest")
    ecrad_timesteps = len(ds.time)
    aircraft_height_level = np.zeros(ecrad_timesteps)

    for i in tqdm(range(ecrad_timesteps)):
        aircraft_height_level[i] = h.arg_nearest(ds["pressure_hl"][i, :].values, bahamas_tmp.PS[i].values * 100)

    aircraft_height_level = aircraft_height_level.astype(int)
    height_level_da[v] = xr.DataArray(aircraft_height_level, dims=["time"], coords={"time": ds.time})
    aircraft_height = ds["pressure_hl"].isel(half_level=height_level_da[v])
    aircraft_height_da[v] = xr.DataArray(aircraft_height, dims=["time"], coords={"time": ds.time},
                                         name="aircraft_height", attrs={"unit": "Pa"})

# %% calculate radiative effect BACARDI
ds = ecrad_dict["v1"].isel(half_level=height_level_da["v1"], drop=True)
cre_solar = (bacardi_ds_res.F_down_solar - bacardi_ds_res.F_up_solar) - (ds.flux_dn_sw_clear - ds.flux_up_sw_clear)
cre_ter = (bacardi_ds_res.F_down_terrestrial - bacardi_ds_res.F_up_terrestrial) - (
        ds.flux_dn_lw_clear - ds.flux_up_lw_clear)
cre_total = cre_solar + cre_ter

# %% create radar mask and despeckle radar data
radar_ds["mask"] = ~np.isnan(radar_ds["dBZg"])
radar_mask = ~radar_ds["mask"].values
for n in tqdm(range(2)):
    # despeckle 2 times
    radar_mask = despeckle(radar_mask, 50)  # despeckle again

radar_ds["spklmask"] = (["time", "height"], radar_mask)

# %% use despeckle the reverse way to fill signal gaps in radar data and add it as a mask
radar_mask = ~radar_ds["spklmask"].values
n = 0
for n in tqdm(range(17)):
    # fill gaps 17 times
    radar_mask = despeckle(radar_mask, 50)  # fill gaps again

radar_ds["fill_mask"] = (["time", "height"], radar_mask)

# %% interpolate lidar data onto radar range resolution
new_range = radar_ds.height.values
lidar_ds_r = lidar_ds.interp(height=np.flip(new_range))
lidar_ds_r = lidar_ds_r.assign_coords(height=np.flip(new_range)).isel(height=slice(None, None, -1))

# %% combine radar and lidar mask
lidar_mask = lidar_ds_r["backscatter_ratio"] > 1.2
lidar_mask = lidar_mask.where(lidar_mask == 0, 2).resample(time="1s").first()
radar_lidar_mask = radar_ds["mask"] + lidar_mask

# %% plot lidar data for case study with radar & lidar mask
plot_ds = lidar_ds["backscatter_ratio"].where((lidar_ds.flags == 0) & (lidar_ds.backscatter_ratio > 1)).sel(
    time=sel_time)
ct_plot = radar_lidar_mask.sel(time=sel_time)
plt.rc("font", size=14)
_, ax = plt.subplots(figsize=h.figsize_wide)
plot_ds.plot(x="time", y="height", cmap=cmr.chroma_r, norm=colors.LogNorm(), vmax=100,
             cbar_kwargs=dict(label="Backscatter Ratio \nat 532$\,$nm", pad=0.01))
ct_plot.plot.contour(x="time", levels=[2.9], colors=cbc[4])
ax.plot(bahamas.time, bahamas["IRS_ALT"] / 1000, color="k", label="HALO altitude")
ax.plot([], color=cbc[4], label="Radar & Lidar Mask", lw=2)
ax.legend(loc=2)
h.set_xticks_and_xlabels(ax, time_extend_bc)
ax.set(xlabel="Time (UTC)", ylabel="Altitude (km)", ylim=(0, 12))
plt.tight_layout()

figname = f"{plot_path}/{flight}_lidar_backscatter_ratio_532_radar_mask_cs.png"
plt.savefig(figname, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# %% plot varcloud data for case study
var = "iwc"
plot_ds = varcloud_ds[var].sel(time=sel_time) * h.scale_factors[var]
plt.rc("font", size=12)
fig, ax = plt.subplots(figsize=h.figsize_wide)
plot_ds.plot(x="time", cmap=h.cmaps[var],
             cbar_kwargs=dict(label=f"{h.cbarlabels[var]} ({h.plot_units[var]})", pad=0.01))
h.set_xticks_and_xlabels(ax, time_extend_bc)
ax.set(xlabel="Time (UTC)", ylabel="Altitude (km)", ylim=(0, 12))
plt.tight_layout()

figname = f"{plot_path}/{flight}_varcloud_{var}_cs.png"
plt.savefig(figname, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# %% set ecRad plotting values
var = "iwc"
v = "diffv8"
band = None

# %% prepare data set for plotting
band_str = f"_band{band}" if band is not None else ""

# kwarg dicts
alphas = dict()
ct_fontsize = dict()
ct_lines = dict(ciwc=[1, 5, 10, 15], cswc=[1, 5, 10, 15], q_ice=[1, 5, 10, 15], clwc=[1, 5, 10, 15],
                iwc=[1, 5, 10, 15])
linewidths = dict()
robust = dict(iwc=False)
cb_ticks = dict()
vmaxs = dict()
vmins = dict(iwp=0)
xlabels = dict(v1="v1", v8="v8", v10="v10", diffv8="Difference v1 - v8", diffv10="Difference v1 - v10")

# set kwargs
alpha = alphas[var] if var in alphas else 1
cmap = h.cmaps[var] if var in h.cmaps else cmr.rainforest
cmap = plt.get_cmap(cmap).copy()
cmap.set_bad(color="white")
ct_fs = ct_fontsize[var] if var in ct_fontsize else 8
lines = ct_lines[var] if var in ct_lines else None
lw = linewidths[var] if var in linewidths else 1
norm = h.norms[var] if var in h.norms else None
robust = robust[var] if var in robust else True
ticks = cb_ticks[var] if var in cb_ticks else None
if norm is None:
    vmax = vmaxs[var] if var in vmaxs else None
    vmin = vmins[var] if var in vmins else None
else:
    vmax, vmin = None, None

if "diff" in v:
    cmap = cmr.fusion
    norm = colors.TwoSlopeNorm(vcenter=0)

# prepare ecrad dataset for plotting
sf = h.scale_factors[var] if var in h.scale_factors else 1
if v == "diffv8":
    # calculate difference between simulations
    ds = ecrad_dict["v8"].copy()
    ecrad_ds_diff = ecrad_dict["v1"][var] - ds[var]
    ecrad_plot = ecrad_ds_diff.where((ds[var] != 0) | (~np.isnan(ds[var]))) * sf
elif v == "diffv10":
    # calculate difference between simulations
    ds = ecrad_dict["v10"].copy()
    ecrad_ds_diff = ecrad_dict["v1"][var] - ds[var]
    ecrad_plot = ecrad_ds_diff.where((ds[var] != 0) | (~np.isnan(ds[var]))) * sf
else:
    ds = ecrad_dict[v]
    ecrad_plot = ds[var] * sf

# add new z axis mean pressure altitude
if "half_level" in ecrad_plot.dims:
    new_z = ds["press_height_hl"].mean(dim="time") / 1000
else:
    new_z = ds["press_height_full"].mean(dim="time") / 1000

ecrad_plot_new_z = list()
for t in tqdm(ecrad_plot.time, desc="New Z-Axis"):
    tmp_plot = ecrad_plot.sel(time=t)
    if "half_level" in tmp_plot.dims:
        tmp_plot = tmp_plot.assign_coords(
            half_level=ds["press_height_hl"].sel(time=t, drop=True).to_numpy() / 1000)
        tmp_plot = tmp_plot.rename(half_level="height")

    else:
        tmp_plot = tmp_plot.assign_coords(
            level=ds["press_height_full"].sel(time=t, drop=True).to_numpy() / 1000)
        tmp_plot = tmp_plot.rename(level="height")

    tmp_plot = tmp_plot.interp(height=new_z.to_numpy())
    ecrad_plot_new_z.append(tmp_plot)

ecrad_plot = xr.concat(ecrad_plot_new_z, dim="time")
# filter very low to_numpy()
ecrad_plot = ecrad_plot.where(np.abs(ecrad_plot) > 0.001)

# select time height slice
time_sel = sel_time
if len(ecrad_plot.dims) > 2:
    dim3 = "band_sw"
    dim3 = dim3 if dim3 in ecrad_plot.dims else None
    ecrad_plot = ecrad_plot.sel({"time": time_sel, "height": slice(12, 0), f"{dim3}": band})
else:
    ecrad_plot = ecrad_plot.sel(time=time_sel, height=slice(12, 0))

time_extend = pd.to_timedelta((ecrad_plot.time[-1] - ecrad_plot.time[0]).to_numpy())

# %% plot 2D IFS variables along flight track
_, ax = plt.subplots(figsize=(43 * cm, 10 * cm))
# ecrad 2D field
ecrad_plot.plot(x="time", y="height", cmap=cmap, ax=ax, robust=robust, vmin=vmin, vmax=vmax, alpha=alpha, norm=norm,
                cbar_kwargs={"pad": 0.01, "label": f"{h.cbarlabels[var]} \n({h.plot_units[var]})",
                             "ticks": ticks})
if lines is not None:
    # add contour lines
    ct = ax.contour(ecrad_plot.time, ecrad_plot.height, ecrad_plot.to_numpy().T, levels=lines, linestyles="--",
                    colors="k",
                    linewidths=lw)
    ct.clabel(fontsize=ct_fs, inline=1, inline_spacing=0, fmt='%i', rightside_up=True, use_clabeltext=True)

ax.set_title(f"")
ax.set_ylabel("Altitude (km)")
ax.set_xlabel("Time (UTC)")
h.set_xticks_and_xlabels(ax, time_extend)
plt.tight_layout()
figname = f"{plot_path}/{flight}_ecrad_{v}_{var}{band_str}_along_track.png"
plt.savefig(figname, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# %% plot histogram of particle sizes
time_sel = sel_time
plot_v1 = (ecrad_dict["v1"].re_ice.sel(time=time_sel).to_numpy() * 1e6).flatten()
plot_v8 = (ecrad_dict["v8"].re_ice.sel(time=time_sel).to_numpy() * 1e6).flatten()
t1, t2 = time_sel.start, time_sel.stop
binsize = 2
bins = np.arange(10, 71, binsize)
plt.rc("font", size=10)
_, ax = plt.subplots(figsize=(13 * cm, 9 * cm))
ax.hist(plot_v8, bins=bins, label=f"VarCloud ({plot_v8.shape[0]})", histtype="step", lw=4, color=cbc[1])
ax.hist(plot_v1, bins=bins, label=f"IFS ({plot_v1.shape[0]})", histtype="step", lw=4, color=cbc[3])
ax.legend()
ax.text(0.75, 0.7, f"Binsize: {binsize} $\mu$m", transform=ax.transAxes, bbox=dict(boxstyle="round", fc="white"))
ax.grid()
ax.set(xlabel=r"Ice effective radius ($\mu$m)", ylabel="Number of occurrence")
plt.tight_layout()
figname = f"{plot_path}/{flight}_ecrad_v1_v8_re_ice_hist.png"
plt.savefig(figname, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# %% plot histogram of iwc
time_sel = sel_time
plot_v1 = (ecrad_dict["v1"].iwc.sel(time=time_sel).to_numpy() * 1e6).flatten()
plot_v8 = (ecrad_dict["v8"].iwc.sel(time=time_sel).to_numpy() * 1e6).flatten()
binsize = 0.25
bins = np.arange(0, 10, binsize)
plt.rc("font", size=10)
_, ax = plt.subplots(figsize=(13 * cm, 9 * cm))
ax.hist(plot_v8, bins=bins, label=f"VarCloud ({plot_v8.shape[0]})", histtype="step", lw=4, color=cbc[1],
        density=False)
ax.hist(plot_v1, bins=bins, label=f"IFS ({plot_v1.shape[0]})", histtype="step", lw=4, color=cbc[3],
        density=False)
ax.legend()
ax.text(0.63, 0.7, f"Binsize: {binsize} {h.plot_units['iwc']}", transform=ax.transAxes,
        bbox=dict(boxstyle="round", fc="white"))
ax.grid()
ax.set(xlabel=f"Ice water content ({h.plot_units['iwc']})", ylabel="Number of occurrence")
plt.tight_layout()
figname = f"{plot_path}/{flight}_ecrad_v1_v8_iwc_hist.png"
plt.savefig(figname, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# %% plot BACARDI and ecRad below cloud
ds10, ds1 = ecrad_dict["v10"], ecrad_dict["v1"]
time_sel = slice(ds10.time[0], ds10.time[-1])
ecrad_plot = ds10.isel(half_level=height_level_da["v10"])
ecrad_plot1 = ds1.isel(half_level=height_level_da["v1"]).sel(time=time_sel)
bacardi_lat = bacardi_ds["lat"].sel(time=time_sel)
bacardi_lon = bacardi_ds["lon"].sel(time=time_sel)
ecrad_lat = bacardi_lat.reindex(time=ecrad_plot.time.to_numpy(), method="ffill")
ecrad1_lat = bacardi_lat.reindex(time=ecrad_plot1.time.to_numpy(), method="ffill")
bacardi_plot = bacardi_ds["F_down_solar"].sel(time=time_sel)
bacardi_error = bacardi_plot * 0.03

plt.rc("font", size=10)
_, ax = plt.subplots(figsize=(16 * cm, 10 * cm))
ax.plot(bacardi_plot.time, bacardi_plot, label="$F_{\downarrow , solar}$ BACARDI", lw=2)
ax.fill_between(bacardi_plot.time, bacardi_plot + bacardi_error, bacardi_plot - bacardi_error, color=cbc[0],
                alpha=0.5, label="BACARDI uncertainty")
ax.plot(ecrad_plot.time, ecrad_plot.flux_dn_sw, label="$F_{\downarrow , solar}$ ecRad Varcloud", lw=2)
ax.plot(ecrad_plot1.time, ecrad_plot1.flux_dn_sw, marker="o", label="$F_{\downarrow , solar}$ ecRad IFS", lw=2,
        markersize=6, color=cbc[3])
ax.legend(loc=4)
ax.grid()
h.set_xticks_and_xlabels(ax, pd.to_timedelta((ds10.time[-1] - ds10.time[0]).to_numpy()))
ax.set_ylabel(f"Broadband irradiance ({h.plot_units['flux_dn_sw']})")
ax.set_xlabel("Time (UTC)", labelpad=-12)

# add latitude and longitude axis
axs2 = ax.twiny(), ax.twiny()
xlabels = ["Latitude (°N)", "Longitude (°E)"]
for i, ax2 in enumerate(axs2):
    # Move twinned axis ticks and label from top to bottom
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")

    # Offset the twin axis below the host
    ax2.spines["bottom"].set_position(("axes", -0.08 * (i + 1)))

    # Turn on the frame for the twin axis, but then hide all
    # but the bottom spine
    ax2.set_frame_on(True)
    ax2.patch.set_visible(False)

    for sp in ax2.spines.values():
        sp.set_visible(False)
    ax2.spines["bottom"].set_visible(True)

    ticklocs = ax.xaxis.get_ticklocs()  # get tick locations
    ts = pd.to_datetime(mpl.dates.num2date(ticklocs)).tz_localize(None)  # convert matplotlib dates to pandas dates
    xticklabels = [bacardi_lat.sel(time=ts).to_numpy(), bacardi_lon.sel(time=ts).to_numpy()]  # get xticklables
    ax2.set_xticks(np.linspace(0.05, 0.95, len(ts)))
    ax2.set_xticklabels(np.round(xticklabels[i], 2))
    ax2.set_xlabel(xlabels[i], labelpad=-12)

plt.tight_layout()
figname = f"{plot_path}/{flight}_ecrad_varcloud_BACARDI_F_down_solar_below_cloud.png"
plt.savefig(figname, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# %% calculate differences between ecrad and bacardi
ds1 = ecrad_dict["v1"].isel(half_level=height_level_da["v1"])
ds8 = ecrad_dict["v8"].isel(half_level=height_level_da["v1"].sel(time=ecrad_dict["v8"].time))
ds10 = ecrad_dict["v10"].isel(half_level=height_level_da["v10"].sel(time=ecrad_dict["v10"].time))
ifs_fdn = ds1.flux_dn_sw - bacardi_ds["F_down_solar"]
ifs_fup = ds1.flux_up_sw - bacardi_ds["F_up_solar"]
ifs_fdn_above = ifs_fdn.sel(time=above_slice).to_numpy().flatten()
ifs_fup_above = ifs_fup.sel(time=above_slice).to_numpy().flatten()
ifs_fdn_below = ifs_fdn.sel(time=below_slice).to_numpy().flatten()
ifs_fup_below = ifs_fup.sel(time=below_slice).to_numpy().flatten()
varcloud_fdn_above = ds8.flux_dn_sw - bacardi_ds["F_down_solar"]
varcloud_fdn_below = ds10.flux_dn_sw - bacardi_ds["F_down_solar"]
varcloud_fup_above = ds8.flux_up_sw - bacardi_ds["F_up_solar"]
varcloud_fup_below = ds10.flux_up_sw - bacardi_ds["F_up_solar"]

# %% plot ecRad IFS - BACARDI
_, ax = plt.subplots(figsize=(20 * cm, 20 * cm))
ax.hist([ifs_fup_above, ifs_fup_below, ifs_fdn_above, ifs_fdn_below], density=True, histtype="step", lw=3,
        label=[r"$F_{\uparrow , solar}$ above cloud", r"$F_{\uparrow , solar}$ below cloud",
               r"$F_{\downarrow , solar}$ above cloud", r"$F_{\downarrow , solar}$ below cloud"])
ax.legend(loc=2)
plt.show()
plt.close()

# %% plot ecRad VarCloud - BACARDI
_, ax = plt.subplots(figsize=(20 * cm, 20 * cm))
ax.hist([varcloud_fup_above, varcloud_fup_below, varcloud_fdn_above, varcloud_fdn_below], density=True,
        histtype="step", lw=3, bins=20,
        label=[r"$F_{\uparrow , solar}$ above cloud", r"$F_{\uparrow , solar}$ below cloud",
               r"$F_{\downarrow , solar}$ above cloud", r"$F_{\downarrow , solar}$ below cloud"])
ax.legend(loc=2)
plt.show()
plt.close()

# %% plot ecRad VarCloud - BACARDI and ecRad IFS - BACARDI F down only
bias_varcloud = np.mean(varcloud_fdn_below).to_numpy()
bias_ifs = np.nanmean(ifs_fdn_below)
wm2 = h.plot_units["flux_dn_sw"]
binsize = 4
bins = np.arange(-50, 29, binsize)
_, ax = plt.subplots(figsize=(22 * cm, 13 * cm))
ax.hist([ifs_fdn_below, varcloud_fdn_below], density=True, histtype="step", lw=4, bins=bins, color=[cbc[3], cbc[1]],
        label=["IFS", r"VarCloud"])
ax.text(0.67, 0.63, f"Binsize: {binsize} {h.plot_units['flux_dn_sw']}", transform=ax.transAxes,
        bbox=dict(boxstyle="round", fc="white"))
ax.text(0.015, 0.735, f"Mean Bias ({wm2})\nVarCloud: {bias_varcloud:.2f}\nIFS: {bias_ifs:.2f}", transform=ax.transAxes,
        bbox=dict(boxstyle="round", fc="white", alpha=0.5))
ax.set(xlabel=f"Solar Downward Irradiance ecRad - BACARDI ({wm2})",
       ylabel="PDF")
ax.grid()
ax.legend(loc=1)
plt.tight_layout()

figname = f"{plot_path}/{flight}_ecrad-bacardi_pdf_below_cloud.png"
plt.savefig(figname, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# %% plot scatterplot of below cloud measurements
bacardi_plot = bacardi_ds_res.resample(time="1Min").first().sel(time=below_slice)
ecrad_plot = ecrad_ds.isel(half_level=height_level_da["v1"]).sel(time=below_slice)
plt.rc("font", size=12)
lims = [(120, 240), (80, 130), (95, 170), (210, 220)]
for (i, x), y in zip(enumerate(bacardi_vars), ecrad_vars):
    rmse = np.sqrt(np.mean((ecrad_plot[y] - bacardi_plot[x]) ** 2))
    bias = np.mean((ecrad_plot[y] - bacardi_plot[x]))
    _, ax = plt.subplots(figsize=h.figsize_equal)
    ax.scatter(bacardi_plot[x], ecrad_plot[y], c=cbc[3])
    ax.axline((0, 0), slope=1, color="k", lw=2, transform=ax.transAxes)
    ax.set_ylim(lims[i])
    ax.set_xlim(lims[i])
    ticks = ax.get_yticks() if i == 0 else ax.get_xticks()
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)
    ax.set_aspect('equal')
    ax.set_xlabel("BACARDI irradiance (W$\,$m$^{-2}$)")
    ax.set_ylabel("ecRad irradiance (W$\,$m$^{-2}$)")
    ax.set_title(f"{titles[i]}\nbelow cloud")
    ax.grid()
    ax.text(0.025, 0.95, f"# points: {sum(~np.isnan(bacardi_plot[x])):.0f}\n"
                         f"RMSE: {rmse.values:.2f}" + " W$\,$m$^{-2}$\n"
                                                      f"Bias: {bias.to_numpy():.2f}" + " W$\,$m$^{-2}$",
            ha='left', va='top', transform=ax.transAxes,
            bbox=dict(fc='white', ec='black', alpha=0.8, boxstyle='round'))
    plt.tight_layout()
    plt.savefig(f"{plot_path}/{flight}_{names[i]}_bacardi_vs_ecrad_scatter_below_cloud.png", dpi=300)
    plt.show()
    plt.close()

# %% plot scatterplot of below cloud measurements BACARDI vs ecRad Varcloud
ecrad_plot = ecrad_dict["v10"].isel(half_level=height_level_da["v10"]).sel(time=below_slice)
bacardi_plot = bacardi_ds_res.sel(time=ecrad_plot.time)
plt.rc("font", size=12)
lims = [(120, 240), (80, 130), (95, 170), (210, 220)]
for (i, x), y in zip(enumerate(bacardi_vars), ecrad_vars):
    rmse = np.sqrt(np.mean((ecrad_plot[y] - bacardi_plot[x]) ** 2))
    bias = np.mean((ecrad_plot[y] - bacardi_plot[x]))
    _, ax = plt.subplots(figsize=h.figsize_equal)
    ax.scatter(bacardi_plot[x], ecrad_plot[y], c=cbc[3])
    ax.axline((0, 0), slope=1, color="k", lw=2, transform=ax.transAxes)
    ax.set_ylim(lims[i])
    ax.set_xlim(lims[i])
    ticks = ax.get_yticks() if i == 0 else ax.get_xticks()
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)
    ax.set_aspect('equal')
    ax.set_xlabel("BACARDI irradiance (W$\,$m$^{-2}$)")
    ax.set_ylabel("ecRad irradiance (W$\,$m$^{-2}$)")
    ax.set_title(f"{titles[i]}\nbelow cloud")
    ax.grid()
    ax.text(0.025, 0.95, f"# points: {sum(~np.isnan(bacardi_plot[x])):.0f}\n"
                         f"RMSE: {rmse.to_numpy():.2f}" + " W$\,$m$^{-2}$\n"
                                                      f"Bias: {bias.to_numpy():.2f}" + " W$\,$m$^{-2}$",
            ha='left', va='top', transform=ax.transAxes,
            bbox=dict(fc='white', ec='black', alpha=0.8, boxstyle='round'))
    plt.tight_layout()
    plt.savefig(f"{plot_path}/{flight}_{names[i]}_bacardi_vs_ecrad_v10_scatter_below_cloud.png", dpi=300)
    plt.show()
    plt.close()

