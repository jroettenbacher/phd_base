#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 21-02-2023

Compare different ice optic parameterizations for ecRad simulations along flightpath with IFS input

**Baran2017**

- Baran2017 has a higher solar transmissivity
- Baran2017 has higher terrestrial absorption (less downward below cloud and more upward above cloud)

Baran2017 is experimental and not documented. Baran2016 should be used.

**Baran2016**

- optical properties are the same as for Baran2017


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

cm = 1 / 2.54
cb_colors = h.get_cb_friendly_colors()

# %% set paths
campaign = "halo-ac3"
halo_key = "RF17"
halo_flight = meta.flight_names[halo_key]
date = halo_flight[9:17]

plot_path = f"{h.get_path('plot', campaign=campaign)}/{halo_flight}/ecrad_ice_param_comparison"
h.make_dir(plot_path)
ecrad_path = f"{h.get_path('ecrad', campaign=campaign)}/{date}/"
ifs_path = f"{h.get_path('ifs', campaign=campaign)}/{date}"
# files with mean over 33 surrounding columns
ecrad_fu_file = f"ecrad_merged_inout_{date}_v1_mean.nc"
ecrad_baran_file = f"ecrad_merged_output_{date}_v2_mean.nc"
ecrad_baran2016_file = f"ecrad_merged_output_{date}_v7_mean.nc"

# %% get flight segments for case study period
segmentation = ac3airborne.get_flight_segments()["HALO-AC3"]["HALO"][f"HALO-AC3_HALO_{halo_key}"]
segments = flightphase.FlightPhaseFile(segmentation)
above_cloud, below_cloud = dict(), dict()
above_cloud["start"] = segments.select("name", "high level 7")[0]["start"]
above_cloud["end"] = segments.select("name", "high level 8")[0]["end"]
below_cloud["start"] = segments.select("name", "high level 9")[0]["start"]
below_cloud["end"] = segments.select("name", "high level 10")[0]["end"]
above_slice = slice(above_cloud["start"], above_cloud["end"])
below_slice = slice(below_cloud["start"], below_cloud["end"])
case_slice = slice(pd.to_datetime("2022-04-11 10:30"), pd.to_datetime("2022-04-11 12:30"))

# %% read in ecrad data
ecrad_fu = xr.open_dataset(f"{ecrad_path}/{ecrad_fu_file}")
ecrad_baran = xr.open_dataset(f"{ecrad_path}/{ecrad_baran_file}")
ecrad_baran2016 = xr.open_dataset(f"{ecrad_path}/{ecrad_baran2016_file}")
ecrad_baran = ecrad_baran.assign_coords(half_level=np.arange(0.5, 138.5))
ecrad_baran2016 = ecrad_baran2016.assign_coords(half_level=np.arange(0.5, 138.5))
ecrad_fu["re_ice"] = ecrad_fu.re_ice.where(ecrad_fu.re_ice != 5.196162e-05, np.nan)
ecrad_fu["re_liquid"] = ecrad_fu.re_liquid.where(ecrad_fu.re_liquid != 4.000001e-06, np.nan)
ecrad_fu["ciwc"] = ecrad_fu.ciwc.where(ecrad_fu.ciwc != 0, np.nan)

ecrad_dict = dict(fu=ecrad_fu, baran=ecrad_baran, baran2016=ecrad_baran2016)

# %% calculate transmissivity and reflectivity
for key in ecrad_dict:
    ds = ecrad_dict[key].copy()
    # solar
    ds["transmissivity_sw"] = ds["flux_dn_sw"] / ds["flux_dn_sw_clear"]
    ds["reflectivity_sw"] = ds["flux_up_sw"] / ds["flux_dn_sw"]
    ds["spectral_transmissivity_sw"] = ds["spectral_flux_dn_sw"] / ds["spectral_flux_dn_sw_clear"]
    ds["spectral_reflectivity_sw"] = ds["spectral_flux_up_sw"] / ds["spectral_flux_dn_sw"]
    # terrestrial
    ds["transmissivity_lw"] = ds["flux_dn_lw"] / ds["flux_dn_lw_clear"]
    ds["reflectivity_lw"] = ds["flux_up_lw"] / ds["flux_dn_lw"]
    ds["spectral_transmissivity_lw"] = ds["spectral_flux_dn_lw"] / ds["spectral_flux_dn_lw_clear"]
    ds["spectral_reflectivity_lw"] = ds["spectral_flux_up_lw"] / ds["spectral_flux_dn_lw"]

    ecrad_dict[key] = ds.copy()

# %% calculate cloud radiative effect
for key in ecrad_dict:
    ds = ecrad_dict[key].copy()
    # solar
    ds["cre_sw"] = (ds.flux_dn_sw - ds.flux_up_sw) - (ds.flux_dn_sw_clear - ds.flux_up_sw_clear)
    # spectral cre
    ds["spectral_cre_sw"] = (ds.spectral_flux_dn_sw - ds.spectral_flux_up_sw) - (
            ds.spectral_flux_dn_sw_clear - ds.spectral_flux_up_sw_clear)
    # terrestrial
    ds["cre_lw"] = (ds.flux_dn_lw - ds.flux_up_lw) - (ds.flux_dn_lw_clear - ds.flux_up_lw_clear)
    # spectral cre
    ds["spectral_cre_lw"] = (ds.spectral_flux_dn_lw - ds.spectral_flux_up_lw) - (
            ds.spectral_flux_dn_lw_clear - ds.spectral_flux_up_lw_clear)
    # cre_total
    ds["cre_total"] = ds.cre_sw + ds.cre_lw
    # spectral cre net
    ds["spectral_cre_total"] = ds.spectral_cre_sw + ds.spectral_cre_lw

    ecrad_dict[key] = ds.copy()

# %% calculate density
pressure = ecrad_dict["fu"]["pressure_full"] * un.Pa
temperature = ecrad_dict["fu"]["t"] * un.K
mixing_ratio = mixing_ratio_from_specific_humidity(ecrad_dict["fu"]["q"] * un("kg/kg")).metpy.convert_units("g/kg")
ecrad_dict["fu"]["air_density"] = density(pressure, temperature, mixing_ratio)

# %% calculate heating rates
ds_fu = ecrad_dict["fu"].copy()
for key in ecrad_dict:
    ds = ecrad_dict[key].copy()
    # solar
    fdw_top = ds.flux_dn_sw.sel(half_level=slice(137)).to_numpy() * un("W/m2")
    fup_top = ds.flux_up_sw.sel(half_level=slice(137)).to_numpy() * un("W/m2")
    fdw_bottom = ds.flux_dn_sw.sel(half_level=slice(1, 138)).to_numpy() * un("W/m2")
    fup_bottom = ds.flux_up_sw.sel(half_level=slice(1, 138)).to_numpy() * un("W/m2")
    z_top = ds_fu.press_height_hl.sel(half_level=slice(137)).to_numpy() * un.m
    z_bottom = ds_fu.press_height_hl.sel(half_level=slice(1, 138)).to_numpy() * un.m
    heating_rate = (1 / (ds_fu.air_density * Cp_d)) * (
            ((fdw_top - fup_top) - (fdw_bottom - fup_bottom)) / (z_top - z_bottom))
    ds["heating_rate_sw"] = heating_rate.metpy.convert_units("K/day")
    # terrestrial
    fdw_top = ds.flux_dn_lw.sel(half_level=slice(137)).to_numpy() * un("W/m2")
    fup_top = ds.flux_up_lw.sel(half_level=slice(137)).to_numpy() * un("W/m2")
    fdw_bottom = ds.flux_dn_lw.sel(half_level=slice(1, 138)).to_numpy() * un("W/m2")
    fup_bottom = ds.flux_up_lw.sel(half_level=slice(1, 138)).to_numpy() * un("W/m2")
    z_top = ds_fu.press_height_hl.sel(half_level=slice(137)).to_numpy() * un.m
    z_bottom = ds_fu.press_height_hl.sel(half_level=slice(1, 138)).to_numpy() * un.m
    heating_rate = (1 / (ds_fu.air_density * Cp_d)) * (
            ((fdw_top - fup_top) - (fdw_bottom - fup_bottom)) / (z_top - z_bottom))
    ds["heating_rate_lw"] = heating_rate.metpy.convert_units("K/day")
    # net heating rate
    ds["heating_rate_net"] = ds.heating_rate_sw + ds.heating_rate_lw

    ecrad_dict[key] = ds.copy()

# %% calculate spectral absorption by cloud, above cloud - below cloud spectrum
above_cloud_hl, below_cloud_hl = 87.5, 103.5
for var in ["spectral_flux_dn_sw", "spectral_flux_dn_lw", "spectral_flux_up_sw", "spectral_flux_up_lw"]:
    for key in ecrad_dict:
        ds = ecrad_dict[key].copy()
        ds_tmp = ds[var]
        ds[f"{var}_diff"] = ds_tmp.sel(half_level=above_cloud_hl) - ds_tmp.sel(half_level=below_cloud_hl)
        ecrad_dict[key] = ds.copy()

# %% compute cloud ice water path
ds_fu = ecrad_dict["fu"].copy()
factor = ds_fu.pressure_hl.diff(dim="half_level").to_numpy() / (9.80665 * ds_fu.cloud_fraction.to_numpy())
ds_fu["iwp"] = (["time", "level"], factor * ds_fu.ciwc.to_numpy())
ds_fu["iwp"] = ds_fu.iwp.where(ds_fu.iwp != np.inf, np.nan)
ecrad_dict["fu"] = ds_fu.copy()

# %% calculate ice optics with baran2017 parameterization
ds_fu, ds_baran = ecrad_dict["fu"].copy(), ecrad_dict["baran"].copy()
ice_optics_baran2017 = ecrad.calc_ice_optics_baran2017("sw", ds_fu["iwp"], ds_fu.ciwc, ds_fu.t)
ds_baran["od"] = ice_optics_baran2017[0]
ds_baran["scat_od"] = ice_optics_baran2017[1]
ds_baran["g"] = ice_optics_baran2017[2]
ds_baran["band_sw"] = range(1, 15)
ds_baran["band_lw"] = range(1, 17)
ds_baran["g_mean"] = ds_baran["g"].mean(dim="band_sw")

ecrad_dict["baran"] = ds_baran.copy()

# %% calculate ice optics with baran2016 parameterization
ds_fu, ds_baran2016 = ecrad_dict["fu"].copy(), ecrad_dict["baran2016"].copy()
ice_optics_baran2016 = ecrad.calc_ice_optics_baran2016("sw", ds_fu["iwp"], ds_fu.ciwc, ds_fu.t)
ds_baran2016["od"] = ice_optics_baran2017[0]
ds_baran2016["scat_od"] = ice_optics_baran2017[1]
ds_baran2016["g"] = ice_optics_baran2017[2]
ds_baran2016["band_sw"] = range(1, 15)
ds_baran2016["band_lw"] = range(1, 17)
ds_baran2016["g_mean"] = ds_baran2016["g"].mean(dim="band_sw")

ecrad_dict["baran2016"] = ds_baran2016.copy()

# %% calculate ice optics with fu parameterization sw
ds_fu = ecrad_dict["fu"].copy()
ice_optics_fu = ecrad.calc_ice_optics_fu_sw(ds_fu["iwp"], ds_fu.re_ice)
ds_fu["od"] = ice_optics_fu[0]
ds_fu["scat_od"] = ice_optics_fu[1]
ds_fu["g"] = ice_optics_fu[2]
ds_fu["band_sw"] = range(1, 15)
ds_fu["band_lw"] = range(1, 17)
ds_fu["g_mean"] = ds_fu["g"].mean(dim="band_sw")

ecrad_dict["fu"] = ds_fu.copy()

# %% calculate ice optics with fu parameterization lw
ds_fu = ecrad_dict["fu"].copy()
ice_optics_fu = ecrad.calc_ice_optics_fu_lw(ds_fu["iwp"], ds_fu.re_ice)
ds_fu["od_lw"] = ice_optics_fu[0]
ds_fu["scat_od_lw"] = ice_optics_fu[1]
ds_fu["g_lw"] = ice_optics_fu[2]
ds_fu["band_sw"] = range(1, 15)
ds_fu["band_lw"] = range(1, 17)
ds_fu["od_lw_int"] = ds_fu["od_lw"].integrate(coord="band_lw")
ds_fu["scat_od_lw_int"] = ds_fu["scat_od_lw"].integrate(coord="band_lw")
ds_fu["g_lw_mean"] = ds_fu["g_lw"].mean(dim="band_lw")

ecrad_dict["fu"] = ds_fu.copy()

# %% calculate ice optics with yi parameterization
# ice_optics_yi = ecrad.calc_ice_optics_yi("sw", ds_fu["iwp"], ds_fu.re_ice)
# ds_yi["od"] = ice_optics_yi[0]
# ds_yi["scat_od"] = ice_optics_yi[1]
# ds_yi["g"] = ice_optics_yi[2]
# ds_yi["band_sw"] = range(1, 15)
# ds_yi["band_lw"] = range(1, 17)
# ds_yi["od_int"] = ds_yi["od"].integrate(coord="band_sw")
# ds_yi["scat_od_int"] = ds_yi["scat_od"].integrate(coord="band_sw")
# ds_yi["g_mean"] = ds_yi["g"].mean(dim="band_sw")

# %% calculate difference between Fu-IFS and Baran2017/Baran2016
ds_fu = ecrad_dict["fu"]
for key in ["baran", "baran2016"]:
    ds = ecrad_dict[key].copy()
    ds_diff = ds_fu - ds
    for var in ds_fu:
        if var not in ds_diff:
            ds_diff[var] = ds_fu[var]
    ecrad_dict[f"{key}_diff"] = ds_diff.copy()

# %% integrate over band_sw
for var in ["od", "scat_od"]:
    for key in ecrad_dict:
        try:
            ds = ecrad_dict[key]
            ds[f"{var}_int"] = ds[var].integrate(coord="band_sw")
            ecrad_dict[key] = ds
        except KeyError:
            print(f"{var} not found in {key}")

# %% plotting variables
h.set_cb_friendly_colors()
plt.rc("font", size=12)
time_extend = pd.to_timedelta((ecrad_fu.time[-1] - ecrad_fu.time[0]).to_numpy())  # get time extend for x-axis labeling
time_extend_cs = below_cloud["end"] - above_cloud["start"]  # time extend for case study
time_extend_ac = above_cloud["end"] - above_cloud["start"]
time_extend_bc = below_cloud["end"] - below_cloud["start"]
ecrad_xlabels = [str(l).replace(",", " -") for l in h.ecRad_bands.values()]

# %% prepare metadata for plotting IFS data in ecrad dataset
version = "baran2016_diff"
variable = "scat_od_int"
band = None  # set to None if variable is not banded

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
ecrad_plot = ecrad_dict[version][variable] * sf
# add new z axis mean pressure altitude
if "half_level" in ecrad_plot.dims:
    new_z = ecrad_fu["press_height_hl"].mean(dim="time") / 1000
else:
    new_z = ecrad_fu["press_height_full"].mean(dim="time") / 1000

ecrad_plot_new_z = list()
for t in tqdm(ecrad_plot.time, desc="New Z-Axis"):
    tmp_plot = ecrad_plot.sel(time=t)
    if "half_level" in tmp_plot.dims:
        tmp_plot = tmp_plot.assign_coords(
            half_level=ecrad_fu["press_height_hl"].sel(time=t, drop=True).to_numpy() / 1000)
        tmp_plot = tmp_plot.rename(half_level="height")

    else:
        tmp_plot = tmp_plot.assign_coords(level=ecrad_fu["press_height_full"].sel(time=t, drop=True).to_numpy() / 1000)
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

ax.set(title=f"{halo_key} IFS/ecRad input/output along Flight Track - {version}",
       ylabel="Altitude (km)", xlabel="Time (UTC)")
# ax.set_xlim(pd.to_datetime("2022-04-11 10:30"), pd.to_datetime("2022-04-11 12:30"))
# ax.set_ylim(0, 13)
# ax.legend(loc=2)
h.set_xticks_and_xlabels(ax, time_extend_cs)
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_ecrad_{version}_{variable}{band_str}_along_track.png"
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

ax.set_title(f"{halo_key} ecRad 1D Profiles along Flight Track - {version}")
ax.set_ylabel("Altitude (km)")
ax.set_xlabel(f"{colorbarlabel[variable]} ({units[variable]})")
if norm is None:
    ax.set_xlim(vmin, vmax)
else:
    ax.set_xlim(norm.vmin, norm.vmax)
ax.set_ylim(0, 13)
ax.legend(title="Time (UTC)")
plt.tight_layout()
figname = f"{plot_path}/{halo_flight}_ecrad_{version}_{variable}_1D.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot histogram comparison
variable = "flux_dn_sw"
all_bins = dict(cre_sw=np.arange(-25, 1, 0.5), cre_lw=np.arange(0, 50), cre_total=np.arange(-2, 28),
                reflectivity_sw=np.arange(0.675, 0.85, 0.01),
                transmissivity_sw=np.arange(0.8, 1.01, 0.01), transmissivity_lw=np.arange(1, 1.51, 0.01),
                flux_dn_sw=np.arange(105, 270, 5), flux_up_sw=np.arange(85, 190, 3),
                flux_dn_lw=np.arange(25, 200, 5), flux_up_lw=np.arange(165, 245, 5),
                od_int=np.arange(0, 3, 0.05), scat_od_int=np.arange(0, 3, 0.05))
bins = all_bins[variable] if variable in all_bins else None
hl_slice = slice(72, 138)  # ~11km to ground
try:
    hist_x = ecrad_dict["fu"][variable].sel(time=case_slice, half_level=hl_slice).to_numpy().flatten()
    hist_y = ecrad_dict["baran"][variable].sel(time=case_slice, half_level=hl_slice).to_numpy().flatten()
    hist_z = ecrad_dict["baran2016"][variable].sel(time=case_slice, half_level=hl_slice).to_numpy().flatten()
except KeyError:
    hist_x = ecrad_dict["fu"][variable].sel(time=case_slice, level=hl_slice).to_numpy().flatten()
    hist_y = ecrad_dict["baran"][variable].sel(time=case_slice, level=hl_slice).to_numpy().flatten()
    hist_z = ecrad_dict["baran2016"][variable].sel(time=case_slice, level=hl_slice).to_numpy().flatten()

_, ax = plt.subplots(figsize=h.figsize_wide)
ax.hist([hist_z, hist_y, hist_x], bins=bins, label=["Baran2016", "Baran2017", "Fu-IFS"], histtype="step",
        density=True, lw=2)
ax.set(title=f"Comparison of ecRad Fu-IFS vs Baran2016 vs Baran2017 for RF17 Case Study",
       xlabel=f"{colorbarlabel[variable]} ({units[variable]})", ylabel="PDF")
ax.legend(loc=0)
figname = f"{plot_path}/ecrad_pdf_{variable}_fu_vs_baran2016_baran2017_case_study.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot ecrad spectrum sw above - below cloud
version = "fu"
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