#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 21-02-2023

Compare different ice optic parameterizations for ecRad simulations along flightpath with IFS input.

**Baran2017 (deprecated)**

- Baran2017 has a higher solar transmissivity
- Baran2017 has higher terrestrial absorption (less downward below cloud and more upward above cloud)

Baran2017 is experimental and not documented. Baran2016 should be used.

**Baran2016**



"""
# %% import modules
import pylim.helpers as h
import pylim.halo_ac3 as meta
from pylim import ecrad
import ac3airborne
from ac3airborne.tools import flightphase
from metpy.calc import density, mixing_ratio_from_specific_humidity
from metpy.units import units as un
from metpy.constants import Cp_d
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import cmasher as cmr
from tqdm import tqdm

cb_colors = h.get_cb_friendly_colors()

# %% set paths
campaign = "halo-ac3"
key = "RF17"
flight = meta.flight_names[key]
date = flight[9:17]

plot_path = f"{h.get_path('plot', campaign=campaign)}/{flight}/ecrad_ice_param_comparison"
h.make_dir(plot_path)
ecrad_path = f"{h.get_path('ecrad', campaign=campaign)}/{date}/"
ifs_path = f"{h.get_path('ifs', campaign=campaign)}/{date}"
bahamas_path = h.get_path("bahamas", flight, campaign)
bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{key}_v1_1s.nc"
bacardi_path = h.get_path("bacardi", flight, campaign)
bacardi_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1.nc"
ecrad_versions = ["v15", "v18", "v19"]

# %% read in bahamas and bacardi data
bahamas_ds = xr.open_dataset(f"{bahamas_path}/{bahamas_file}")
bacardi_ds = xr.open_dataset(f"{bacardi_path}/{bacardi_file}")

# %% get flight segments for case study period
segmentation = ac3airborne.get_flight_segments()["HALO-AC3"]["HALO"][f"HALO-AC3_HALO_{key}"]
segments = flightphase.FlightPhaseFile(segmentation)
above_cloud, below_cloud = dict(), dict()
above_cloud["start"] = segments.select("name", "high level 7")[0]["start"]
above_cloud["end"] = segments.select("name", "high level 8")[0]["end"]
below_cloud["start"] = segments.select("name", "high level 9")[0]["start"]
below_cloud["end"] = segments.select("name", "high level 10")[0]["end"]
above_slice = slice(above_cloud["start"], above_cloud["end"])
below_slice = slice(below_cloud["start"], below_cloud["end"])
case_slice = slice(pd.to_datetime("2022-04-11 10:30"), pd.to_datetime("2022-04-11 12:30"))

# %% read in ecrad data and add more variables to each data set
ecrad_dict = dict()
above_cloud_hl, below_cloud_hl = 87.5, 103.5
for v in tqdm(ecrad_versions):
    # merged input and output file with additional variables
    ecrad_file = f"ecrad_merged_inout_{date}_{v}.nc"
    # select only closest column to flight path
    ds = xr.open_dataset(f"{ecrad_path}/{ecrad_file}").sel(column=1, drop=True)
    # calculate spectral absorption by cloud, above cloud - below cloud spectrum
    for var in ["spectral_flux_dn_sw", "spectral_flux_dn_lw", "spectral_flux_up_sw", "spectral_flux_up_lw"]:
        ds_tmp = ds[var]
        ds[f"{var}_diff"] = ds_tmp.sel(half_level=above_cloud_hl) - ds_tmp.sel(half_level=below_cloud_hl)

    ecrad_dict[v] = ds.copy()

# %% get model level of flight altitude for half and full level
level_da = ecrad.get_model_level_of_altitude(bahamas_ds.IRS_ALT, ecrad_dict["v15"], "level")
hlevel_da = ecrad.get_model_level_of_altitude(bahamas_ds.IRS_ALT, ecrad_dict["v15"], "half_level")

# %% read in radiative property files
rad_dict = dict()
for v in tqdm(ecrad_versions):
    rad_file = f"radiative_properties_merged_{date}_{v}.nc"
    ds = xr.open_dataset(f"{ecrad_path}/{rad_file}")
    rad_dict[v] = ds.copy()

# %% calculate difference between Fu-IFS and other parameterizations
ds_fu = ecrad_dict["v15"]
for k in ["v18", "v19"]:
    ds = ecrad_dict[k].copy()
    ds_diff = ds_fu - ds
    ecrad_dict[f"{k}_diff"] = ds_diff.copy()

# %% plotting variables
h.set_cb_friendly_colors()
plt.rc("font", size=12)
time_extend = pd.to_timedelta((ecrad_dict["v15"].time[-1] - ecrad_dict["v15"].time[0]).to_numpy())  # get time extend for x-axis labeling
time_extend_cs = below_cloud["end"] - above_cloud["start"]  # time extend for case study
time_extend_ac = above_cloud["end"] - above_cloud["start"]
time_extend_bc = below_cloud["end"] - below_cloud["start"]
ecrad_xlabels = [str(l).replace(",", " -") for l in h.ecRad_bands.values()]
version_labels = dict(v15="Fu-IFS", v18="Baran2016", v19="Yi", v18_diff="Fu-IFS-Baran2016", v19_diff="Fu-IFS-Yi")
var_dict = dict(flux_dn_sw="F_down_solar", flux_dn_lw="F_down_terrestrial",
                flux_up_sw="F_up_solar", flux_up_lw="F_up_terrestrial")

# %% select options for plotting for ecrad_dict
version = "v18_diff"
variable = "flux_dn_sw"
band = None  # set to None if variable is not banded

# %% prepare metadata for plotting
band_str = f"_band{band}" if band is not None else ""
units = h.plot_units
scale_factor = h.scale_factors
colorbarlabel = h.cbarlabels
# pcm kwargs
alphas = dict(ciwc=0.8)
norms = h.norms
robust = dict(cloud_fraction=False, transmissivity_sw=False, transmissivity_lw=True, reflectivity_sw=False,
              reflectivity_lw=True, cre_total=False)
vmaxs = dict(ciwc=15, clwc=1, flux_dn_sw=400, heating_rate_sw=1.1, heating_rate_lw=1.1, heating_rate_net=1.1)
vmins = dict(heating_rate_lw=-3, heating_rate_net=-1.1)
# colorbar kwargs
cb_ticks = dict(t=[205, 215, 225, 235, 245, 255, 265, 275])
ct_lines = dict(ciwc=[1, 5, 10, 15], t=range(195, 275, 10), re_ice=[10, 15, 17, 20, 25, 30, 35, 40, 60])
linewidths = dict(ciwc=0.5, t=1, re_ice=0.5)
ct_fontsize = dict(ciwc=6, t=9, re_ice=6)
cmaps = h.cmaps
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

if "diff" in version:
    cmap = cmr.fusion
    norm = colors.TwoSlopeNorm(vcenter=0)

if norm is None:
    vmax = vmaxs[variable] if variable in vmaxs else None
    vmin = vmins[variable] if variable in vmins else None
else:
    vmax, vmin = None, None

# prepare ecrad dataset for plotting
sf = scale_factor[variable] if variable in scale_factor else 1
ecrad_v1 = ecrad_dict["v15"]
ecrad_plot = ecrad_dict[version][variable] * sf
# add new z axis mean pressure altitude
if "half_level" in ecrad_plot.dims:
    new_z = ecrad_v1["press_height_hl"].mean(dim="time") / 1000
else:
    new_z = ecrad_v1["press_height_full"].mean(dim="time") / 1000

ecrad_plot_new_z = list()
for t in tqdm(ecrad_plot.time, desc="New Z-Axis"):
    tmp_plot = ecrad_plot.sel(time=t)
    if "half_level" in tmp_plot.dims:
        tmp_plot = tmp_plot.assign_coords(half_level=ecrad_v1["press_height_hl"].sel(time=t, drop=True).to_numpy() / 1000)
        tmp_plot = tmp_plot.rename(half_level="height")

    else:
        tmp_plot = tmp_plot.assign_coords(level=ecrad_v1["press_height_full"].sel(time=t, drop=True).to_numpy() / 1000)
        tmp_plot = tmp_plot.rename(level="height")

    tmp_plot = tmp_plot.interp(height=new_z.to_numpy())
    ecrad_plot_new_z.append(tmp_plot)

ecrad_plot = xr.concat(ecrad_plot_new_z, dim="time")
# filter very low values
ecrad_plot = ecrad_plot.where(np.abs(ecrad_plot) > 0.0001)

# select time height slice
if len(ecrad_plot.dims) > 2:
    dim3 = "band_sw"
    dim3 = dim3 if dim3 in ecrad_plot.dims else None
    ecrad_plot = ecrad_plot.sel({"time": case_slice, "height": slice(13, 0), f"{dim3}": band})
else:
    ecrad_plot = ecrad_plot.sel(time=case_slice, height=slice(13, 0))

# %% plot 2D IFS variables along flight track
_, ax = plt.subplots(figsize=h.figsize_wide)
# ecrad 2D field
ecrad_plot.plot(x="time", y="height", cmap=cmap, ax=ax, robust=robust, vmin=vmin, vmax=vmax, alpha=alpha, norm=norm,
                cbar_kwargs={"pad": 0.04, "label": f"{colorbarlabel[variable]} ({units[variable]})",
                             "ticks": ticks})
if lines is not None:
    # add contour lines
    ct = ax.contour(ecrad_plot.time, ecrad_plot.height, ecrad_plot.to_numpy().T, levels=lines, linestyles="--",
                    colors="k",
                    linewidths=lw)
    ct.clabel(fontsize=ct_fs, inline=1, inline_spacing=0, fmt='%i', rightside_up=True, use_clabeltext=True)

ax.set(title=f"{key} IFS/ecRad input/output along Flight Track - {version_labels[version]}",
       ylabel="Altitude (km)", xlabel="Time (UTC)")
# plot flight track
ax.plot(hlevel_da.time, ecrad_dict["v15"].press_height_hl.isel(half_level=hlevel_da) / 1000, label="HALO altitude", color="k")
# ax.set_xlim(pd.to_datetime("2022-04-11 10:30"), pd.to_datetime("2022-04-11 12:30"))
# ax.set_ylim(0, 13)
ax.legend(loc=2)
h.set_xticks_and_xlabels(ax, time_extend_cs)
plt.tight_layout()

figname = f"{plot_path}/{flight}_ecrad_{version_labels[version]}_{variable}{band_str}_along_track.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot profile from 2D data
time_sel = pd.date_range("2022-04-11 10:45", "2022-04-11 12:00", freq="10min")
_, ax = plt.subplots(figsize=h.figsize_wide)
for ts in time_sel:
    ecrad_1d = ecrad_plot.sel(time=ts, drop=True)
    # ecrad 1D profile
    ecrad_1d.plot(y="height", ax=ax, label=f"{ts:%H:%M}")

ax.set_title(f"{key} ecRad 1D Profiles along Flight Track - {version}")
ax.set_ylabel("Altitude (km)")
ax.set_xlabel(f"{colorbarlabel[variable]} ({units[variable]})")
if norm is None:
    ax.set_xlim(vmin, vmax)
else:
    ax.set_xlim(norm.vmin, norm.vmax)
ax.set_ylim(0, 13)
ax.legend(title="Time (UTC)")
plt.tight_layout()
figname = f"{plot_path}/{flight}_ecrad_{version}_{variable}_1D.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot histogram comparison
variable = "flux_dn_sw"
hists = dict()
all_bins = dict(cre_sw=np.arange(-25, 1, 0.5), cre_lw=np.arange(0, 50), cre_total=np.arange(-2, 28),
                reflectivity_sw=np.arange(0.675, 0.85, 0.01),
                transmissivity_sw=np.arange(0.8, 1.01, 0.01), transmissivity_lw=np.arange(1, 1.51, 0.01),
                flux_dn_sw=np.arange(105, 270, 5), flux_up_sw=np.arange(85, 190, 3),
                flux_dn_lw=np.arange(25, 200, 5), flux_up_lw=np.arange(165, 245, 5),
                g_mean=np.arange(0.78, 0.9, 0.005), od_int=np.arange(0, 3, 0.05), scat_od_int=np.arange(0, 3, 0.05))
bins = all_bins[variable] if variable in all_bins else None
hl_slice = slice(72, 138)  # ~11km to ground
try:
    for k in ecrad_versions:
        hists[k] = ecrad_dict[k][variable].sel(time=case_slice, half_level=hl_slice).to_numpy().flatten()
except KeyError:
    for k in ecrad_versions:
        hists[k] = ecrad_dict[k][variable].sel(time=case_slice, level=hl_slice).to_numpy().flatten()

hist_list = [hists[k] for k in hists]

_, ax = plt.subplots(figsize=h.figsize_wide)
ax.hist(hist_list, bins=bins, label=["Fu-IFS", "Baran2016", "Yi"], histtype="step",
        density=True, lw=2)
ax.set(title=f"Comparison of ecRad output {variable}\n with different ice optic parameterizations for RF17 Case Study",
       xlabel=f"{colorbarlabel[variable]} ({units[variable]})", ylabel="PDF")
ax.legend(loc=0)
figname = f"{plot_path}/ecrad_pdf_{variable}_fu_vs_baran2016_yi_case_study.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot ecrad spectrum sw above - below cloud
version = "v19"
ecrad_plot = ecrad_dict[version]["spectral_flux_dn_sw_diff"].sel(time=case_slice)
labels = [b for b in h.ecRad_bands.values()]
fig, axs = plt.subplots(nrows=2, figsize=h.figsize_wide)
ax = axs[0]
bands = slice(7)
ax.plot(ecrad_plot.time, ecrad_plot.sel(band_sw=bands), label=labels[bands])
ax.legend(bbox_to_anchor=(1.01, 1.19), loc="upper left", title="NIR Bands (nm)")
ax.set(ylim=(0, 13), xticklabels="", title=f"ecRad {version} spectral difference above and below cloud")
ax.grid()
ax = axs[1]
bands = slice(8, 13)
ax.plot(ecrad_plot.time, ecrad_plot.sel(band_sw=bands), label=labels[7:13])
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", title="VIS Bands (nm)")
h.set_xticks_and_xlabels(ax, time_extend_cs)
ax.grid()
ax.set(xlabel="Time (UTC)", ylim=(0, 23))
fig.supylabel(r"Spectral Irradiance (W$\,$m$^{-2}\,$nm$^{-1}$)")
plt.tight_layout()
figname = f"{plot_path}/ecrad_{version}_difference_above_below_spectral_flux_dn_sw_case_study.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot ecrad data at flight altitude and compare with BACARDI
var = "flux_dn_sw"
time_sel = below_slice
bacardi_plot = bacardi_ds[var_dict[var]].sel(time=time_sel)
_, ax = plt.subplots(figsize=h.figsize_wide)
# BACARDI
bacardi_plot.plot(x="time", label="BACARDI", ax=ax)
# ecRad
for v in ecrad_versions:
    try:
        e_plot = ecrad_dict[v][var].isel(half_level=hlevel_da).sel(time=time_sel)
    except KeyError:
        e_plot = ecrad_dict[v][var].isel(level=level_da).sel(time=time_sel)

    e_plot.plot(x="time", label=f"ecRad {version_labels[v]}", ax=ax)

ax.legend()
ax.grid()
h.set_xticks_and_xlabels(ax, time_extend_bc)
ax.set(xlabel="Time (UTC)", ylabel=f"Broadband irradiance ({h.plot_units[var]})")
ax.set_title(f"{key} - Comparison between {h.cbarlabels[var]}")
plt.tight_layout()
figname = f"{plot_path}/{flight}_bacardi_vs_ecrad_{var}_along_track.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot histogram comparison at flight altitude
var = "flux_dn_sw"
time_sel = case_slice
bins = all_bins[var] if var in all_bins else None
hists = list()
hists.append(bacardi_ds[var_dict[var]].sel(time=time_sel).to_numpy().flatten())
_, ax = plt.subplots(figsize=h.figsize_wide)
for v in ecrad_versions:
    try:
        ecrad_hist = ecrad_dict[v][var].isel(half_level=hlevel_da).sel(time=time_sel).to_numpy().flatten()
    except KeyError:
        ecrad_hist = ecrad_dict[v][var].isel(level=level_da).sel(time=time_sel).to_numpy().flatten()

    hists.append(ecrad_hist)

ax.hist(hists, bins=bins, label=["BACARDI", "Fu-IFS", "Baran2016", "Yi"], histtype="step",
        density=True, lw=2)
ax.legend()
ax.grid()
ax.set(title=f"Comparison of ecRad output {var}\n with different ice optic parameterizations for {key} Case Study",
       xlabel=f"{colorbarlabel[var]} ({units[var]})", ylabel="PDF")
plt.tight_layout()
figname = f"{plot_path}/{flight}_bacardi_ecrad_histogram_{var}_along_track.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

