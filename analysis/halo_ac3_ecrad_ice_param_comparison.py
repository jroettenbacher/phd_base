#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 21-02-2023

Compare different ice optic parameterizations for ecRad simulations along flightpath with IFS input

- Baran2017 has a higher solar transmissivity
- Baran2017 has higher terrestrial absorption (less downward below cloud and more upward above cloud)




"""
# %% import modules
import pylim.helpers as h
import pylim.halo_ac3 as meta
from pylim.ecrad import calc_ice_optics_baran2017, calc_ice_optics_fu_sw
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
ecrad_baran = ecrad_baran.assign_coords(half_level=np.arange(0.5, 138.5))
ecrad_fu["re_ice"] = ecrad_fu.re_ice.where(ecrad_fu.re_ice != 5.196162e-05, np.nan)
ecrad_fu["re_liquid"] = ecrad_fu.re_liquid.where(ecrad_fu.re_liquid != 4.000001e-06, np.nan)
ecrad_fu["ciwc"] = ecrad_fu.ciwc.where(ecrad_fu.ciwc != 0, np.nan)

# %% calculate transmissivity and reflectivity
ecrad_fu["transmissivity_sw"] = ecrad_fu["flux_dn_sw"] / ecrad_fu["flux_dn_sw_clear"]
ecrad_baran["transmissivity_sw"] = ecrad_baran["flux_dn_sw"] / ecrad_baran["flux_dn_sw_clear"]
ecrad_fu["reflectivity_sw"] = ecrad_fu["flux_up_sw"] / ecrad_fu["flux_dn_sw"]
ecrad_baran["reflectivity_sw"] = ecrad_baran["flux_up_sw"] / ecrad_baran["flux_dn_sw"]
ecrad_fu["spectral_transmissivity_sw"] = ecrad_fu["spectral_flux_dn_sw"] / ecrad_fu["spectral_flux_dn_sw_clear"]
ecrad_baran["spectral_transmissivity_sw"] = ecrad_baran["spectral_flux_dn_sw"] / ecrad_baran[
    "spectral_flux_dn_sw_clear"]
ecrad_fu["spectral_reflectivity_sw"] = ecrad_fu["spectral_flux_up_sw"] / ecrad_fu["spectral_flux_dn_sw"]
ecrad_baran["spectral_reflectivity_sw"] = ecrad_baran["spectral_flux_up_sw"] / ecrad_baran["spectral_flux_dn_sw"]
# terrestrial
ecrad_fu["transmissivity_lw"] = ecrad_fu["flux_dn_lw"] / ecrad_fu["flux_dn_lw_clear"]
ecrad_baran["transmissivity_lw"] = ecrad_baran["flux_dn_lw"] / ecrad_baran["flux_dn_lw_clear"]
ecrad_fu["reflectivity_lw"] = ecrad_fu["flux_up_lw"] / ecrad_fu["flux_dn_lw"]
ecrad_baran["reflectivity_lw"] = ecrad_baran["flux_up_lw"] / ecrad_baran["flux_dn_lw"]
ecrad_fu["spectral_transmissivity_lw"] = ecrad_fu["spectral_flux_dn_lw"] / ecrad_fu["spectral_flux_dn_lw_clear"]
ecrad_baran["spectral_transmissivity_lw"] = ecrad_baran["spectral_flux_dn_lw"] / ecrad_baran[
    "spectral_flux_dn_lw_clear"]
ecrad_fu["spectral_reflectivity_lw"] = ecrad_fu["spectral_flux_up_lw"] / ecrad_fu["spectral_flux_dn_lw"]
ecrad_baran["spectral_reflectivity_lw"] = ecrad_baran["spectral_flux_up_lw"] / ecrad_baran["spectral_flux_dn_lw"]

# %% calculate cloud radiative effect
ecrad_fu["cre_sw"] = (ecrad_fu.flux_dn_sw - ecrad_fu.flux_up_sw) - (
        ecrad_fu.flux_dn_sw_clear - ecrad_fu.flux_up_sw_clear)
ecrad_baran["cre_sw"] = (ecrad_baran.flux_dn_sw - ecrad_baran.flux_up_sw) - (
        ecrad_baran.flux_dn_sw_clear - ecrad_baran.flux_up_sw_clear)
# spectral cre
ecrad_fu["spectral_cre_sw"] = (ecrad_fu.spectral_flux_dn_sw - ecrad_fu.spectral_flux_up_sw) - (
        ecrad_fu.spectral_flux_dn_sw_clear - ecrad_fu.spectral_flux_up_sw_clear)
ecrad_baran["spectral_cre_sw"] = (ecrad_baran.spectral_flux_dn_sw - ecrad_baran.spectral_flux_up_sw) - (
        ecrad_baran.spectral_flux_dn_sw_clear - ecrad_baran.spectral_flux_up_sw_clear)
# terrestrial
ecrad_fu["cre_lw"] = (ecrad_fu.flux_dn_lw - ecrad_fu.flux_up_lw) - (
        ecrad_fu.flux_dn_lw_clear - ecrad_fu.flux_up_lw_clear)
ecrad_baran["cre_lw"] = (ecrad_baran.flux_dn_lw - ecrad_baran.flux_up_lw) - (
        ecrad_baran.flux_dn_lw_clear - ecrad_baran.flux_up_lw_clear)
# spectral cre
ecrad_fu["spectral_cre_lw"] = (ecrad_fu.spectral_flux_dn_lw - ecrad_fu.spectral_flux_up_lw) - (
        ecrad_fu.spectral_flux_dn_lw_clear - ecrad_fu.spectral_flux_up_lw_clear)
ecrad_baran["spectral_cre_lw"] = (ecrad_baran.spectral_flux_dn_lw - ecrad_baran.spectral_flux_up_lw) - (
        ecrad_baran.spectral_flux_dn_lw_clear - ecrad_baran.spectral_flux_up_lw_clear)
# cre_total
ecrad_fu["cre_total"] = ecrad_fu.cre_sw + ecrad_fu.cre_lw
ecrad_baran["cre_total"] = ecrad_baran.cre_sw + ecrad_baran.cre_lw

# spectral cre net
ecrad_fu["spectral_cre_total"] = ecrad_fu.spectral_cre_sw + ecrad_fu.spectral_cre_lw
ecrad_baran["spectral_cre_total"] = ecrad_baran.spectral_cre_sw + ecrad_baran.spectral_cre_lw

# %% calculate density
pressure = ecrad_fu["pressure_full"] * un.Pa
temperature = ecrad_fu["t"] * un.K
mixing_ratio = mixing_ratio_from_specific_humidity(ecrad_fu["q"] * un("kg/kg")).metpy.convert_units("g/kg")
ecrad_fu["air_density"] = density(pressure, temperature, mixing_ratio)

# %% calculate heating rates
fdw_top = ecrad_fu.flux_dn_sw.sel(half_level=slice(137)).values * un("W/m2")
fup_top = ecrad_fu.flux_up_sw.sel(half_level=slice(137)).values * un("W/m2")
fdw_bottom = ecrad_fu.flux_dn_sw.sel(half_level=slice(1, 138)).values * un("W/m2")
fup_bottom = ecrad_fu.flux_up_sw.sel(half_level=slice(1, 138)).values * un("W/m2")
z_top = ecrad_fu.press_height_hl.sel(half_level=slice(137)).values * un.m
z_bottom = ecrad_fu.press_height_hl.sel(half_level=slice(1, 138)).values * un.m
heating_rate = (1 / (ecrad_fu.air_density * Cp_d)) * (
            ((fdw_top - fup_top) - (fdw_bottom - fup_bottom)) / (z_top - z_bottom))
ecrad_fu["heating_rate_sw"] = heating_rate.metpy.convert_units("K/day")
# terrestrial
fdw_top = ecrad_fu.flux_dn_lw.sel(half_level=slice(137)).values * un("W/m2")
fup_top = ecrad_fu.flux_up_lw.sel(half_level=slice(137)).values * un("W/m2")
fdw_bottom = ecrad_fu.flux_dn_lw.sel(half_level=slice(1, 138)).values * un("W/m2")
fup_bottom = ecrad_fu.flux_up_lw.sel(half_level=slice(1, 138)).values * un("W/m2")
z_top = ecrad_fu.press_height_hl.sel(half_level=slice(137)).values * un.m
z_bottom = ecrad_fu.press_height_hl.sel(half_level=slice(1, 138)).values * un.m
heating_rate = (1 / (ecrad_fu.air_density * Cp_d)) * (
            ((fdw_top - fup_top) - (fdw_bottom - fup_bottom)) / (z_top - z_bottom))
ecrad_fu["heating_rate_lw"] = heating_rate.metpy.convert_units("K/day")
# net heating rate
ecrad_fu["heating_rate_net"] = ecrad_fu.heating_rate_sw + ecrad_fu.heating_rate_lw

# Baran2017
fdw_top = ecrad_baran.flux_dn_sw.sel(half_level=slice(137)).values * un("W/m2")
fup_top = ecrad_baran.flux_up_sw.sel(half_level=slice(137)).values * un("W/m2")
fdw_bottom = ecrad_baran.flux_dn_sw.sel(half_level=slice(1, 138)).values * un("W/m2")
fup_bottom = ecrad_baran.flux_up_sw.sel(half_level=slice(1, 138)).values * un("W/m2")
z_top = ecrad_fu.press_height_hl.sel(half_level=slice(137)).values * un.m
z_bottom = ecrad_fu.press_height_hl.sel(half_level=slice(1, 138)).values * un.m
heating_rate = (1 / (ecrad_fu.air_density * Cp_d)) * (
            ((fdw_top - fup_top) - (fdw_bottom - fup_bottom)) / (z_top - z_bottom))
ecrad_baran["heating_rate_sw"] = heating_rate.metpy.convert_units("K/day")
# terrestrial
fdw_top = ecrad_baran.flux_dn_lw.sel(half_level=slice(137)).values * un("W/m2")
fup_top = ecrad_baran.flux_up_lw.sel(half_level=slice(137)).values * un("W/m2")
fdw_bottom = ecrad_baran.flux_dn_lw.sel(half_level=slice(1, 138)).values * un("W/m2")
fup_bottom = ecrad_baran.flux_up_lw.sel(half_level=slice(1, 138)).values * un("W/m2")
z_top = ecrad_fu.press_height_hl.sel(half_level=slice(137)).values * un.m
z_bottom = ecrad_fu.press_height_hl.sel(half_level=slice(1, 138)).values * un.m
heating_rate = (1 / (ecrad_fu.air_density * Cp_d)) * (
            ((fdw_top - fup_top) - (fdw_bottom - fup_bottom)) / (z_top - z_bottom))
ecrad_baran["heating_rate_lw"] = heating_rate.metpy.convert_units("K/day")
# net heating rate
ecrad_baran["heating_rate_net"] = ecrad_baran.heating_rate_sw + ecrad_baran.heating_rate_lw

# %% calculate difference between Fu-IFS and Baran2017
ecrad_diff = ecrad_fu - ecrad_baran
for var in ecrad_fu:
    if var not in ecrad_diff:
        ecrad_diff[var] = ecrad_fu[var]

# %% calculate spectral absorption by cloud, above cloud - below cloud spectrum
above_cloud_hl, below_cloud_hl = 87.5, 103.5
for var in ["spectral_flux_dn_sw", "spectral_flux_dn_lw", "spectral_flux_up_sw", "spectral_flux_up_lw"]:
    ecrad_tmp = ecrad_fu[var]
    ecrad_fu[f"{var}_diff"] = ecrad_tmp.sel(half_level=above_cloud_hl) - ecrad_tmp.sel(half_level=below_cloud_hl)
    ecrad_tmp = ecrad_baran[var]
    ecrad_baran[f"{var}_diff"] = ecrad_tmp.sel(half_level=above_cloud_hl) - ecrad_tmp.sel(half_level=below_cloud_hl)

# %% compute cloud ice water path
factor = ecrad_fu.pressure_hl.diff(dim="half_level").values / (9.80665 * ecrad_fu.cloud_fraction.values)
ecrad_fu["iwp"] = (["time", "level"], factor * ecrad_fu.ciwc.values)
ecrad_fu["iwp"] = ecrad_fu.iwp.where(ecrad_fu.iwp != np.inf, np.nan)

# %% calculate ice optics with baran2017 parameterization
ice_optics_baran2017 = calc_ice_optics_baran2017("sw", ecrad_fu["iwp"], ecrad_fu.ciwc, ecrad_fu.t)
ecrad_baran["od"] = ice_optics_baran2017[0]
ecrad_baran["scat_od"] = ice_optics_baran2017[1]
ecrad_baran["g"] = ice_optics_baran2017[2]
ecrad_baran["band_sw"] = range(1, 15)
ecrad_baran["band_lw"] = range(1, 17)

# %% calculate ice optics with fu parameterization
ice_optics_fu = calc_ice_optics_fu_sw(ecrad_fu["iwp"], ecrad_fu.re_ice)
ecrad_fu["od"] = ice_optics_fu[0]
ecrad_fu["scat_od"] = ice_optics_fu[1]
ecrad_fu["g"] = ice_optics_fu[2]
ecrad_fu["band_sw"] = range(1, 15)
ecrad_fu["band_lw"] = range(1, 17)

# %% integrate over band_sw
for var in ["od", "scat_od"]:
    ecrad_baran[f"{var}_int"] = ecrad_baran[var].integrate(coord="band_sw")
    ecrad_fu[f"{var}_int"] = ecrad_fu[var].integrate(coord="band_sw")

# %% plotting variables
h.set_cb_friendly_colors()
plt.rc("font", size=12)
figsize_wide = (24 * cm, 12 * cm)
figsize_equal = (12 * cm, 12 * cm)
time_extend = pd.to_timedelta((ecrad_fu.time[-1] - ecrad_fu.time[0]).values)  # get time extend for x-axis labeling
time_extend_cs = below_cloud["end"] - above_cloud["start"]  # time extend for case study
time_extend_ac = above_cloud["end"] - above_cloud["start"]
time_extend_bc = below_cloud["end"] - below_cloud["start"]
ecrad_xlabels = [str(l).replace(",", " -") for l in h.ecRad_bands.values()]

# %% prepare metadata for plotting IFS data in ecrad dataset
version = "baran"
variable = "g"
band = 5  # set to None if variable is not banded
band_str = f"_band{band}" if band is not None else ""
units = dict(cloud_fraction="", clwc="mg$\,$kg$^{-1}$", ciwc="mg$\,$kg$^{-1}$", cswc="g$\,$kg$^{-1}$",
             crwc="g$\,$kg$^{-1}$", t="K", q="g$\,$kg$^{-1}$", re_ice="$\mu$m", re_liquid="$\mu$m",
             heating_rate_sw="K$\,$day$^{-1}$", heating_rate_lw="K$\,$day$^{-1}$", heating_rate_net="K$\,$day$^{-1}$",
             flux_dn_sw="W$\,$m$^{-2}$", flux_dn_lw="W$\,$m$^{-2}$", flux_up_sw="W$\,$m$^{-2}$",
             flux_up_lw="W$\,$m$^{-2}$",
             cre_sw="W$\,$m$^{-2}$", cre_lw="W$\,$m$^{-2}$", cre_total="W$\,$m$^{-2}$",
             transmissivity_sw="", transmissivity_lw="", reflectivity_sw="", reflectivity_lw="",
             od="", scat_od="", g="", od_int="", scat_od_int="", g_int="")
scale_factor = dict(cloud_fraction=1, clwc=1e6, ciwc=1e6, cswc=1000, crwc=1000, t=1, q=1000, re_ice=1e6, re_liquid=1e6)
colorbarlabel = dict(cloud_fraction="Cloud Fraction", clwc="Cloud Liquid Water Content", ciwc="Cloud Ice Water Content",
                     cswc="Cloud Snow Water Content", crwc="Cloud Rain Water Content", t="Temperature",
                     q="Specific Humidity", re_ice="Ice Effective Radius", re_liquid="Liquid Effective Radius",
                     heating_rate_sw="Solar Heating Rate", heating_rate_lw="Terrestrial Heating Rate",
                     heating_rate_net="Net Heating Rate",
                     transmissivity_sw="Solar Transmissivity", transmissivity_lw="Terrestrial Transmissivity",
                     reflectivity_sw="Solar Reflectivity", reflectivity_lw="Terrestrial Reflectivity",
                     flux_dn_sw="Downward Solar Irradiance", flux_up_sw="Upward Solar Irradiance",
                     flux_dn_lw="Downward Terrestrial Irradiance", flux_up_lw="Upward Terrestrial Irradiance",
                     cre_sw="Solar Cloud Radiative Effect", cre_lw="Terrestrial Cloud Radiative Effect",
                     cre_total="Total Cloud Radiative Effect",
                     od=f"Total Optical Depth Band {band}", scat_od=f"Scattering Optical Depth Band {band}",
                     g=f"Asymmetry Factor Band {band}",
                     od_int="Integrated Total Optical Depth", scat_od_int="Integrated Scattering Optical Depth")
# pcm kwargs
alphas = dict(ciwc=0.8)
norms = dict(t=colors.TwoSlopeNorm(vmin=200, vcenter=235, vmax=280), clwc=colors.LogNorm(),
             heating_rate_lw=colors.TwoSlopeNorm(vmin=-3, vcenter=0, vmax=1.5),
             heating_rate_net=colors.TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2),
             od=colors.LogNorm(vmax=10), od_scat=colors.LogNorm(),
             od_int=colors.LogNorm(vmax=10), scat_od_int=colors.LogNorm())
robust = dict(cloud_fraction=False, transmissivity_sw=False, transmissivity_lw=True, reflectivity_sw=False,
              reflectivity_lw=True, cre_total=False)
vmaxs = dict(ciwc=15, clwc=1, flux_dn_sw=400, heating_rate_sw=1.1, heating_rate_lw=1.1, heating_rate_net=1.1)
vmins = dict(heating_rate_lw=-3, heating_rate_net=-1.1)
# colorbar kwargs
cb_ticks = dict(t=[205, 215, 225, 235, 245, 255, 265, 275])
ct_lines = dict(ciwc=[1, 5, 10, 15], t=range(195, 275, 10), re_ice=[10, 15, 17, 20, 25, 30, 35, 40, 60])
linewidths = dict(ciwc=0.5, t=1, re_ice=0.5)
ct_fontsize = dict(ciwc=6, t=9, re_ice=6)
cmaps = dict(t=cmr.prinsenvlag_r, ciwc=cmr.get_sub_cmap("cmr.freeze", .25, 0.85), cloud_fraction=cmr.neutral,
             re_ice=cmr.cosmic_r, re_liquid=cmr.cosmic_r,
             heating_rate_sw=cmr.get_sub_cmap("cmr.ember_r", 0, 0.75), heating_rate_lw=cmr.fusion_r,
             heating_rate_net=cmr.fusion_r,
             cre_total=cmr.fusion_r,
             flux_dn_sw=cmr.get_sub_cmap("cmr.torch", 0.2, 1))
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
if norm is None:
    vmax = vmaxs[variable] if variable in vmaxs else None
    vmin = vmins[variable] if variable in vmins else None
else:
    vmax, vmin = None, None

if version == "diff":
    cmap = cmr.fusion
    norm = colors.TwoSlopeNorm(vcenter=0)

# %% prepare ecrad dataset for plotting
sf = scale_factor[variable] if variable in scale_factor else 1
if version == "baran" and variable in ecrad_baran:
    ecrad_plot = ecrad_baran[variable] * sf
elif version == "diff":
    ecrad_plot = ecrad_diff[variable] * sf
else:
    ecrad_plot = ecrad_fu[variable] * sf
# add new z axis mean pressure altitude
if "half_level" in ecrad_plot.dims:
    new_z = ecrad_fu["press_height_hl"].mean(dim="time") / 1000
else:
    new_z = ecrad_fu["press_height_full"].mean(dim="time") / 1000

ecrad_plot_new_z = list()
for t in tqdm(ecrad_plot.time, desc="New Z-Axis"):
    tmp_plot = ecrad_plot.sel(time=t)
    if "half_level" in tmp_plot.dims:
        tmp_plot = tmp_plot.assign_coords(half_level=ecrad_fu["press_height_hl"].sel(time=t, drop=True).values / 1000)
        tmp_plot = tmp_plot.rename(half_level="height")

    else:
        tmp_plot = tmp_plot.assign_coords(level=ecrad_fu["press_height_full"].sel(time=t, drop=True).values / 1000)
        tmp_plot = tmp_plot.rename(level="height")

    tmp_plot = tmp_plot.interp(height=new_z.values)
    ecrad_plot_new_z.append(tmp_plot)

ecrad_plot = xr.concat(ecrad_plot_new_z, dim="time")
# filter very low values
ecrad_plot = ecrad_plot.where(np.abs(ecrad_plot) > 0.0001)

# %% select time height slice
if len(ecrad_plot.dims) > 2:
    dim3 = "band_sw"
    dim3 = dim3 if dim3 in ecrad_plot.dims else None
    ecrad_plot = ecrad_plot.sel({"time": case_slice, "height": slice(13, 0), f"{dim3}": band})
else:
    ecrad_plot = ecrad_plot.sel(time=case_slice, height=slice(13, 0))

# %% plot 2D IFS variables along flight track
_, ax = plt.subplots(figsize=figsize_wide)
# ecrad 2D field
ecrad_plot.plot(x="time", y="height", cmap=cmap, ax=ax, robust=robust, vmin=vmin, vmax=vmax, alpha=alpha, norm=norm,
                cbar_kwargs={"pad": 0.04, "label": f"{colorbarlabel[variable]} ({units[variable]})",
                             "ticks": ticks})
if lines is not None:
    # add contour lines
    ct = ax.contour(ecrad_plot.time, ecrad_plot.height, ecrad_plot.values.T, levels=lines, linestyles="--", colors="k",
                    linewidths=lw)
    ct.clabel(fontsize=ct_fs, inline=1, inline_spacing=0, fmt='%i', rightside_up=True, use_clabeltext=True)

ax.set_title(f"{halo_key} IFS/ecRad input/output along Flight Track - {version}")
ax.set_ylabel("Altitude (km)")
ax.set_xlabel("Time (UTC)")
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
_, ax = plt.subplots(figsize=figsize_wide)
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

# %% plot histogram comparison fu vs baran2017
variable = "flux_up_sw"
all_bins = dict(cre_sw=np.arange(-25, 1, 0.5), cre_lw=np.arange(0, 50), cre_total=np.arange(-2, 28),
                reflectivity_sw=np.arange(0.675, 0.85, 0.01),
                transmissivity_sw=np.arange(0.8, 1.01, 0.01), transmissivity_lw=np.arange(1, 1.51, 0.01),
                flux_dn_sw=np.arange(105, 270, 5), flux_up_sw=np.arange(85, 190, 3),
                flux_dn_lw=np.arange(25, 200, 5), flux_up_lw=np.arange(165, 245, 5))
bins = all_bins[variable] if variable in all_bins else None
hl_slice = slice(72, 138)  # ~11km to ground
hist_x = ecrad_fu[variable].sel(time=case_slice, half_level=hl_slice).values.flatten()
hist_y = ecrad_baran[variable].sel(time=case_slice, half_level=hl_slice).values.flatten()
_, ax = plt.subplots(figsize=figsize_wide)
ax.hist([hist_y, hist_x], bins=bins, label=["Baran2017", "Fu-IFS"], histtype="step", density=True)
ax.legend(loc=2)
ax.set_title(f"Comparison of ecRad Fu-IFS vs Baran2017 for RF17 Case Study")
ax.set_xlabel(f"{colorbarlabel[variable]} ({units[variable]})")
ax.set_ylabel("PDF")
figname = f"{plot_path}/ecrad_pdf_{variable}_fu_vs_baran2017_case_study.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot ecrad spectrum sw above - below cloud
bands = slice(7)
ecrad_plot = ecrad_fu["spectral_flux_dn_sw_diff"].sel(time=case_slice)
labels = [b for b in h.ecRad_bands.values()]
fig, axs = plt.subplots(nrows=2, figsize=figsize_wide)
ax = axs[0]
ax.plot(ecrad_plot.time, ecrad_plot.sel(band_sw=bands), label=labels[bands])
ax.legend(bbox_to_anchor=(1.01, 1.19), loc="upper left", title="NIR Bands (nm)")
ax.set_xticklabels("")
ax.grid()
ax.set_title("ecRad spectral difference above and below cloud")
ax = axs[1]
bands = slice(7, 13)
ax.plot(ecrad_plot.time, ecrad_plot.sel(band_sw=bands), label=labels[bands])
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", title="VIS Bands (nm)")
h.set_xticks_and_xlabels(ax, time_extend_cs)
ax.grid()
ax.set_xlabel("Time (UTC)")
fig.supylabel(r"Spectral Irradiance (W$\,$m$^{-2}\,$nm$^{-1}$)")
plt.tight_layout()
figname = f"{plot_path}/ecrad_fu_difference_above_below_spectral_flux_dn_sw_case_study.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()
