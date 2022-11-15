#!/usr/bin/env python
"""Analysis for poster at (AC)³ General Assembly 2022-11-22

- take mean if ecRad columns
- calculate pressure and pressure height
- get height of HALO in model
- band SMART data to ecRad bands
- calculate cloud radiative effect from BACARDI and libRadtran data and from ecRad data
- plot aircraft track through ciwc of IFS
- plot CRE for BACARDI and ecRad below cloud
- plot SMART spectra and banded SMART and ecrad irradiance for above and below cloud flight sections
- plot trajectory map with high cloud cover and relative humidity over ice profiles from dropsondes

*author*: Johannes Röttenbacher
"""

# %% module import
import pylim.helpers as h
from pylim import reader
from pylim import meteorological_formulas as met
from pylim.halo_ac3 import coordinates
import ac3airborne
from ac3airborne.tools import flightphase
import numpy as np
import os
from matplotlib import patheffects
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import colors
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import cmasher as cmr
import cartopy.crs as ccrs
import xarray as xr
import pandas as pd
from tqdm import tqdm
import logging

log = logging.getLogger("pylim")
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

# %% set up paths and meta data
date = 20220411
halo_key = "RF17"
flight = f"HALO-AC3_{date}_HALO_{halo_key}"
campaign = "halo-ac3"
smart_path = h.get_path("calibrated", flight, campaign)
ecrad_path = f"{h.get_path('ecrad', campaign=campaign)}/{date}"
libradtran_path = h.get_path("libradtran", flight, campaign)
bahamas_path = h.get_path("bahamas", flight, campaign)
bacardi_path = h.get_path("bacardi", flight, campaign)
dropsonde_path = f"{h.get_path('dropsondes', flight, campaign)}/.."
trajectory_path = f"{h.get_path('trajectories', campaign=campaign)}/selection_CC_and_altitude"
era5_path = "/projekt_agmwend/home_rad/BenjaminK/HALO-AC3/ERA5/ERA5_ml_noOMEGAscale"
plot_path = f"{h.get_path('plot', campaign=campaign)}/{flight}"

# file names
calibrated_file = f"HALO-AC3_HALO_SMART_spectral-irradiance-Fdw_{date}_{halo_key}_v1.0.nc"
bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{halo_key}_v1.nc"
bacardi_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{halo_key}_R1.nc"
ecrad_inout = f"ecrad_merged_inout_{date}_v1.nc"
libradtran_file = "HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_solar_si_20220411_RF17.nc"
dropsonde_file = f"HALO-AC3_HALO_Dropsondes_quickgrid_{date}.nc"
era5_files = [os.path.join(era5_path, f) for f in os.listdir(era5_path) if f"P{date}" in f]
era5_files.sort()

# set options and credentials for HALO-AC3 cloud and intake catalog
kwds = {'simplecache': dict(same_names=True)}
credentials = {"user": os.environ.get("AC3_CLOUD_USER"), "password": os.environ.get("AC3_CLOUD_PASSWORD")}
cat = ac3airborne.get_intake_catalog()

cm = 1 / 2.54

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

# %% read in data
smart_ds = xr.open_dataset(f"{smart_path}/{calibrated_file}")
bahamas = reader.read_bahamas(f"{bahamas_path}/{bahamas_file}")
bacardi_ds = xr.open_dataset(f"{bacardi_path}/{bacardi_file}")
bb_solar_sim = xr.open_dataset(f"{libradtran_path}/{libradtran_file}")
ecrad_fu = xr.open_dataset(f"{ecrad_path}/{ecrad_inout}")
ecrad_fu = ecrad_fu.assign_coords({"band_sw": range(1, 15), "band_lw": range(1, 17), "sw_albedo_band": range(1, 7)})
dropsonde_ds = xr.open_dataset(f"{dropsonde_path}/{dropsonde_file}")
dropsonde_ds["alt"] = dropsonde_ds.alt / 1000  # convert altitude to km
ins = cat["HALO-AC3"]["HALO"]["GPS_INS"][f"HALO-AC3_HALO_{halo_key}"](storage_options=kwds, **credentials).to_dask()
era5_ds = xr.open_mfdataset(era5_files).sel(time=f"2022-04-11T12:00")

# %% take mean of all ecRad columns
mean_file_fu = f"{ecrad_path}/ecrad_inout_{date}_v1_mean.nc"
if os.path.isfile(mean_file_fu):
    ecrad_fu = xr.open_dataset(mean_file_fu)
else:
    # take mean of column and calculate standard deviation
    if ecrad_fu.dims["column"] > 1:
        ecrad_fu_std = ecrad_fu.std(dim="column")
        ecrad_fu = ecrad_fu.mean(dim="column")
        # save intermediate file to save time
        ecrad_fu_std.to_netcdf(f"{ecrad_path}/ecrad_inout_{date}_v1_stds.nc")
        ecrad_fu.to_netcdf(mean_file_fu)

# %% calculate pressure height and pressure
q_air = 1.292  # dry air density at 0°C in kg/m3
g_geo = 9.81  # earth acceleration in m/s^2
pressure_hl = ecrad_fu["pressure_hl"]
ecrad_fu["press_height"] = -(pressure_hl[:, 137]) * np.log(pressure_hl[:, :] / pressure_hl[:, 137]) / (q_air * g_geo)
# replace TOA height (calculated as infinity) with nan
ecrad_fu["press_height"] = ecrad_fu["press_height"].where(ecrad_fu["press_height"] != np.inf, np.nan)

# calculate pressure on all model levels
pressure = era5_ds.hyam + era5_ds.hybm * era5_ds.PS * 100
# select only relevant model levels and swap dimension names
era5_ds["pressure"] = pressure.sel(nhym=slice(39, 137)).swap_dims({"nhym": "lev"})

# %% get height level of actual flight altitude in ecRad model, this determines only the index of the level
bahamas_tmp = bahamas.sel(time=ecrad_fu.time, method="nearest")
ecrad_timesteps = len(ecrad_fu.time)
aircraft_height_level = np.zeros(ecrad_timesteps)

for i in tqdm(range(ecrad_timesteps)):
    aircraft_height_level[i] = h.arg_nearest(ecrad_fu["pressure_hl"][i, :].values, bahamas_tmp.PS[i].values * 100)

aircraft_height_level = aircraft_height_level.astype(int)
height_level_da = xr.DataArray(aircraft_height_level, dims=["time"], coords={"time": ecrad_fu.time})
aircraft_height = ecrad_fu["pressure_hl"].isel(half_level=height_level_da)
aircraft_height_da = xr.DataArray(aircraft_height, dims=["time"], coords={"time": ecrad_fu.time},
                                  name="aircraft_height", attrs={"unit": "Pa"})

# %% create new coordinate for ecRad bands to show spectrum on a continuous scale
ecrad_bands = [np.mean(h.ecRad_bands[band]) for band in h.ecRad_bands]

# %% band SMART data to ecRad bands
nr_bands = len(h.ecRad_bands)
smart_banded = np.empty((nr_bands, smart_ds.dims["time"]))
for i, band in enumerate(h.ecRad_bands):
    wl1 = h.ecRad_bands[band][0]
    wl2 = h.ecRad_bands[band][1]
    smart_banded[i, :] = smart_ds.Fdw_cor.sel(wavelength=slice(wl1, wl2)).integrate(coord="wavelength")

smart_banded = xr.DataArray(smart_banded, coords={"ecrad_band": range(1, 15), "time": smart_ds.time}, name="Fdw_cor")

# %% calculate cloud radiative effect from ecRad and BACARDI
ds = ecrad_fu
cre_solar_sim = (ds.flux_dn_sw - ds.flux_up_sw) - (ds.flux_dn_sw_clear - ds.flux_up_sw_clear)
# cre_terrestrial = (ds.flux_dn_lw - ds.flux_up_lw) - (ds.flux_dn_lw_clear - ds.flux_up_lw_clear)
cre_solar = (bacardi_ds.F_down_solar - bacardi_ds.F_up_solar) - (bb_solar_sim.fdw - bb_solar_sim.eup)

# %% set some variables for the coming plots
ecrad_xlabels = [str(l).replace(",", " -") for l in h.ecRad_bands.values()]

# %% set plotting options for map plot
lon, lat, times = bahamas["IRS_LON"], bahamas["IRS_LAT"], bahamas["time"]
pad = 2
llcrnlat = lat.min(skipna=True) - pad
llcrnlon = lon.min(skipna=True) - pad
urcrnlat = lat.max(skipna=True) + pad
urcrnlon = lon.max(skipna=True) + pad
extent = [llcrnlon, urcrnlon, llcrnlat, urcrnlat]
kiruna = coordinates["Kiruna"]
longyear = coordinates["Longyearbyen"]

# %% plot aircraft track through ecrad input
variable = "ciwc"
units = dict(clwc="g/kg", ciwc="g/kg", q_ice="g/kg", cswc="g/kg", crwc="g/kg", t="K", re_ice="m x $10^{-6}$",
             re_liquid="m x $10^{-6}$", flux_up_lw="W$\,$m$^{-2}$", flux_up_sw="W$\,$m$^{-2}$",
             flux_dn_lw="W$\,$m$^{-2}$", flux_dn_sw="W$\,$m$^{-2}$", flux_up_lw_clear="W$\,$m$^{-2}$")
scale_factor = dict(clwc=1000, ciwc=1000, q_ice=1000, cswc=1000, crwc=1000, t=1, re_ice=1e6, re_liquid=1e6,
                    flux_up_lw=1, flux_up_sw=1, flux_dn_lw=1, flux_dn_sw=1, flux_up_lw_clear=1)
colorbarlabel = dict(clwc="Cloud Liquid Water Content", ciwc="Cloud Ice Water Content", q_ice="Ice and Snow Content",
                     cswc="Cloud Snow Water Content", crwc="Cloud Rain Water Content", t="Temperature",
                     re_ice="Effective Ice Particle Radius", re_liquid="Effective Droplet Radius",
                     flux_up_lw="Terrestrial Upward Irradiance", flux_up_sw="Solar Upward Irradiance",
                     flux_dn_lw="Terrestrial Downward Irradiance", flux_dn_sw="Solar Downward Irradiance",
                     flux_up_lw_clear="Terrestrial Clear Sky Upward Irradiance")
cmaps = dict(t=cmr.fusion, ciwc=cmr.get_sub_cmap("cmr.freeze", .25, 0.85),
             clwc=cmr.get_sub_cmap("cmr.flamingo", .25, .9))
cmap = cmaps[variable] if variable in cmaps else cmr.rainforest
cmap = plt.get_cmap(cmap).copy().reversed()
# cmap.set_under("white")
norms = dict(t=colors.TwoSlopeNorm(vmin=193, vcenter=238, vmax=293))
norm = norms[variable] if variable in norms else None
ticks_d = dict(t=[198, 218, 238, 258, 278, 298])
ticks = ticks_d[variable] if variable in ticks_d else None
# x_sel = (above_cloud["start"], below_cloud["end"])
x_sel = (pd.to_datetime(ecrad_fu.time[0].values), pd.to_datetime(ecrad_fu.time[-1].values))
ecrad_plot = ecrad_fu[variable] * scale_factor[variable]
if "level" in ecrad_plot.dims:
    ecrad_plot = ecrad_plot.assign_coords(level=ecrad_fu["pressure_full"].isel(time=100, drop=True).values / 100)
    ecrad_plot = ecrad_plot.rename(level="height")
else:
    ecrad_plot = ecrad_plot.assign_coords(half_level=ecrad_fu["pressure_hl"].isel(time=100, drop=True).values / 100)
    ecrad_plot = ecrad_plot.rename(half_level="height")
aircraft_height_plot = aircraft_height_da / 100

plt.rcdefaults()
h.set_cb_friendly_colors()
plt.rc('font', size=20)
plt.rc('lines', linewidth=3)
fig, ax = plt.subplots(figsize=(42 * cm, 18 * cm))
# ecrad input
ecrad_plot = ecrad_plot.where(ecrad_plot > 0.001)  # filter very low values
ecrad_plot.plot(x="time", y="height", cmap=cmap, ax=ax, robust=True, norm=norm,
                cbar_kwargs={"pad": 0.01, "label": f"{colorbarlabel[variable]} ({units[variable]})",
                             "ticks": ticks})

# aircraft altitude through model
aircraft_height_plot.plot(x="time", color="k", ax=ax, label="HALO altitude")

ax.legend(loc=2, fontsize=22)
ax.set_xlim(x_sel)
# ax.set_ylim(0, 14)
ax.invert_yaxis()
h.set_xticks_and_xlabels(ax, x_sel[1] - x_sel[0])
ax.tick_params(axis="x", labelsize=18)
ax.tick_params(axis="y", labelsize=18)
ax.grid()
# ax.set_title("IFS Output along HALO Flight Track 11 April 2022", size=24)
ax.set_ylabel("Pressure (hPa)", size=22)
ax.set_xlabel("Time (UTC)", size=22)
plt.tight_layout()
figname = f"{plot_path}/{flight}_IFS_{variable}_HALO_alt.png"
plt.savefig(figname, dpi=300)
log.info(f"Saved {figname}")
plt.show()
plt.close()

# %% plot ecRad and BACARDI radiative effect along flight track for below cloud section and at HALO altitude
plt.rcdefaults()
h.set_cb_friendly_colors()
plt.rc('font', size=20)
plt.rc('lines', linewidth=3)
x_sel = (below_cloud["start"], below_cloud["end"])
# x_sel = (pd.to_datetime(ecrad_fu.time[0].values), pd.to_datetime(ecrad_fu.time[-1].values))
cmap = cmr.rainforest
cmap = plt.get_cmap(cmap).copy().reversed()
_, ax = plt.subplots(figsize=(42 * cm, 18 * cm))
# 0 line
ax.axhline(y=0, color="grey")
# cre ecRad
plot_ds = cre_solar_sim
plot_ds = plot_ds.assign_coords(half_level=ecrad_fu["pressure_hl"].isel(time=100, drop=True).values / 100)
plot_ds = plot_ds.rename(half_level="height")
plot_ds = plot_ds.isel(height=height_level_da).sel(time=below_slice)
ax.scatter(plot_ds.time, plot_ds, label="ecRad", s=300)

# cre bacardi and libradtran
plot_ds = cre_solar.sel(time=below_slice)
ax.scatter(plot_ds.time, plot_ds, label="BACARDI/libRadtran", s=300)

ax.legend(loc=2, fontsize=18)
ax.set_xlim(x_sel)
ax.invert_yaxis()
h.set_xticks_and_xlabels(ax, x_sel[1] - x_sel[0])
ax.tick_params(axis="x", labelsize=18)
ax.tick_params(axis="y", labelsize=18)
ax.grid()
# ax.set_title("Solar Cloud Radiative Effect at HALO Altitude below Cirrus Cloud 11 April 2022", size=24)
ax.set_ylabel("Cloud Radiative Effect (W$\,$m$^{-2}$)", size=22)
ax.set_xlabel("Time (UTC)", size=22)
plt.tight_layout()
figname = f"{plot_path}/{flight}_ecrad_bacardi_cre_at_HALO_alt.png"
plt.savefig(figname, dpi=300)
log.info(f"Saved {figname}")
plt.show()
plt.close()

# %% replace band coordinates in ecRad and smart banded data
ecrad_fu_2 = ecrad_fu.assign_coords(band_sw=ecrad_bands).isel(band_sw=slice(3, 12))
smart_banded_2 = smart_banded.assign_coords(ecrad_band=ecrad_bands).isel(ecrad_band=slice(3, 12))

# %% create list with ecRad band limits for plot
band_limits = [h.ecRad_bands[key] for key in h.ecRad_bands]
band_limits = [item for t in band_limits[3:12] for item in t]
band_limits.sort()
band_limits = band_limits[::2]
band_limits.append(2150)
band_names = [key.replace("and", "") for key in h.ecRad_bands][3:12]

# %% plot banded SMART spectra and ecRad simulation for below and above cloud section
h.set_cb_friendly_colors()
plt.rc("font", size=20)
_, ax = plt.subplots(figsize=(42 * cm, 18 * cm))
ax.vlines(band_limits, ymin=0, ymax=80, colors=["grey"], linestyles="dashed")
# SMART spectra on secondary axis
ax2 = ax.twinx()
plot_ds = smart_ds.Fdw_cor
plot_ds.sel(time=above_slice).mean(dim="time").plot(label="SMART spectral above cloud", alpha=0.8, color="#88CCEE",
                                                    lw=4, ax=ax2)
plot_ds.sel(time=below_slice).mean(dim="time").plot(label="SMART spectral below cloud", alpha=0.8, color="#CC6677",
                                                    lw=4, ax=ax2)
ax2.set_ylabel("Spectral Irradiance $(\mathrm{W\,m}^{-2}\mathrm{\,nm}^{-1})$")
handles2, labels2 = ax2.get_legend_handles_labels()

# banded SMART measurement
smart_banded_2.sel(time=above_slice).mean(dim="time").plot(label="SMART band integrated", marker="X", ls="",
                                                           markersize=18, color="#88CCEE", ax=ax)
smart_banded_2.sel(time=below_slice).mean(dim="time").plot(marker="X", ls="", markersize=18, color="#CC6677", ax=ax)

# ecRad simulation
ecrad_fdw = ecrad_fu_2["spectral_flux_dn_sw"].isel(half_level=height_level_da)
ecrad_fdw.sel(time=above_slice).mean(dim="time").plot(x="band_sw", label="ecRad", marker="d", ls="", markersize=18,
                                                      color="#88CCEE", ax=ax)
ecrad_fdw.sel(time=below_slice).mean(dim="time").plot(marker="d", ls="", markersize=18, color="#CC6677", ax=ax)

for x, name in zip(ecrad_fu_2.band_sw, band_names):
    ax.text(x - 35, 0.5, name, size=14)

# aesthetics
handles, labels = ax.get_legend_handles_labels()
handles = handles + handles2
labels = labels + labels2
ax.legend(handles=handles, labels=labels, loc=1)
# ax.set_title(f"Mean Downward Irradiance above and below Cirrus")
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Banded Spectral Irradiance \n$(\mathrm{W\,m}^{-2}\mathrm{\,nm}^{-1})$")
ax.grid()
plt.tight_layout()
figname = f"{plot_path}/{flight}_smart_ecrad_fdw_banded_above_below.png"
plt.savefig(figname, dpi=200)
log.info(f"Saved {figname}")
plt.show()
plt.close()

# %% plot map of trajectories with surface pressure, flight track, dropsonde locations and high cloud cover + RH_ice profiles from radiosonde
plt_sett = {
    'TIME': {
        'label': 'Time Relative to Release (h)',
        'norm': plt.Normalize(-120, 0),
        'ylim': [-120, 0],
        'cmap_sel': 'tab20b_r',
    }}
var_name = "TIME"
data_crs = ccrs.PlateCarree()
h.set_cb_friendly_colors()

plt.rc("font", size=16)
fig = plt.figure(figsize=(42 * cm, 18 * cm))
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

# plot trajectory map 11 April in first row and first column
ax = fig.add_subplot(gs[0, 0], projection=ccrs.NorthPolarStereo())
ax.coastlines(alpha=0.5)
ax.set_xlim((-2000000, 2000000))
ax.set_ylim((-3000000, 500000))
gl = ax.gridlines(crs=data_crs, draw_labels=True, linewidth=1, color='gray', alpha=0.5,
                  linestyle=':', x_inline=False, y_inline=False, rotate_labels=False)
gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 20))
gl.ylocator = mticker.FixedLocator(np.arange(60, 90, 5))
gl.top_labels = False
gl.right_labels = False

# Plot the surface pressure - 11 April
pressure_levels = np.arange(900, 1125, 5)
E5_press = era5_ds.MSL / 100
cp = ax.contour(E5_press.lon, E5_press.lat, E5_press, levels=pressure_levels, colors='k', linewidths=1,
                linestyles='solid', alpha=1, transform=data_crs)
cp.clabel(fontsize=8, inline=1, inline_spacing=4, fmt='%i hPa', rightside_up=True, use_clabeltext=True)

# add seaice edge
ci_levels = [0.8]
E5_ci = era5_ds.CI
cci = ax.contour(E5_ci.lon, E5_ci.lat, E5_ci, ci_levels, transform=data_crs, linestyles="--", colors="#332288",
                 linewidths=3)

# add high cloud cover
E5_cc = era5_ds.CC.where(era5_ds.pressure < 60000, drop=True).sum(dim="lev")
# E5_cc = E5_cc.where(E5_cc > 1)
ax.contourf(E5_cc.lon, E5_cc.lat, E5_cc, levels=24, transform=data_crs, cmap="Blues", alpha=1)

# plot trajectories - 11 April
header_line = [2]  # header-line of .1 files is always line #2 (counting from 0)
date_h = f"20220411_07"
# get filenames
fname_traj = "traj_CIRR_HALO_" + date_h + ".1"
try:
    trajs = np.loadtxt(f"{trajectory_path}/{fname_traj}", dtype="f", skiprows=5)
    times = trajs[:, 0]
    # generate object to only load specific header line
    gen = h.generate_specific_rows(f"{trajectory_path}/{fname_traj}", userows=header_line)
    header = np.loadtxt(gen, dtype="str", unpack=True)
    header = header.tolist()  # convert to list

    # convert to lower char.
    for j in range(len(header)):
        header[j] = header[j].upper()  # convert to lower

    var_index = header.index(var_name.upper())
except (StopIteration, IOError) as e:
    print("\t>>>Skipping file, probably empty<<<")
else:
    print("\tTraj_select.1 could be opened, processing...")

    # get the time step of the trajectories # here: manually set
    dt = 0.01
    traj_single_len = 4320  # int(tmax/dt)
    traj_overall_len = int(len(times))
    traj_num = int(traj_overall_len / (traj_single_len + 1))  # +1 for the empty line after
    # each traj

for k in range(traj_single_len + 1):
    # reduce to hourly? --> [::60]
    lon = trajs[k * (traj_single_len + 1):(k + 1) * (traj_single_len + 1), 1][::60]
    lat = trajs[k * (traj_single_len + 1):(k + 1) * (traj_single_len + 1), 2][::60]
    var = trajs[k * (traj_single_len + 1):(k + 1) * (traj_single_len + 1), var_index][::60]
    x, y = lon, lat
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=plt_sett[var_name]['cmap_sel'], norm=plt_sett[var_name]['norm'],
                        alpha=1, transform=data_crs)
    # Set the values used for colormapping
    lc.set_array(var)
    if int(traj_num) == 1:
        lc.set_linewidth(5)
    elif int(traj_num) >= 2:
        lc.set_linewidth(1)
    line = ax.add_collection(lc)

if var_name != "TIME":
    plt.colorbar(line, ax=ax).set_label(label=plt_sett[var_name]['label'])
if var_name == "TIME":
    plt.colorbar(line, ax=ax, pad=0.01,
                 ticks=np.arange(-120, 0.1, 12)).set_label(label=plt_sett[var_name]['label'])

# plot flight track - 11 April
track_lons, track_lats = ins["lon"], ins["lat"]
ax.scatter(track_lons[::10], track_lats[::10], c="#888888", alpha=1, marker=".", s=4, zorder=400,
           label='HALO flight track', transform=data_crs, linestyle="solid")

# plot dropsonde locations - 11 April
for i in range(dropsonde_ds.lon.shape[0]):
    launch_time = pd.to_datetime(dropsonde_ds.launch_time[i].values)
    x, y = dropsonde_ds.lon[i].mean().values, dropsonde_ds.lat[i].mean().values
    cross = ax.plot(x, y, "x", color="orangered", markersize=12, label="Dropsonde", transform=data_crs,
                    zorder=450)
    ax.text(x, y, f"{launch_time:%H:%M}", c="k", fontsize=12, transform=data_crs, zorder=500,
            path_effects=[patheffects.withStroke(linewidth=0.25, foreground="white")])

# make legend for flight track and dropsondes - 11 April
handles = [plt.plot([], ls="-", color="#888888", lw=3)[0],  # flight track
           cross[0],  # dropsondes
           plt.plot([], ls="--", color="#332288", lw=3)[0],  # sea ice edge
           Patch(facecolor="royalblue")]  # cloud cover
labels = ["HALO flight track", "Dropsonde", "Sea Ice Edge", "High Cloud Cover\nat 12:00 UTC"]
ax.legend(handles=handles, labels=labels, framealpha=1, loc=2, fontsize=14)

ax.text(-72.5, 73, "(a)", size=18, transform=data_crs, ha="center", va="center",
        bbox=dict(boxstyle="round", ec="grey", fc="white"))

# plot dropsonde profiles in row 1 and column 2
rh_ice = met.relative_humidity_water_to_relative_humidity_ice(dropsonde_ds.rh, dropsonde_ds.T - 273.15)
labels = [f"{lt.replace('2022-04-11T', '')} UTC" for lt in np.datetime_as_string(rh_ice.launch_time.values, unit="m")]
ax = fig.add_subplot(gs[0, 1])
rh_ice.plot.line(y="alt", alpha=0.5, label=labels, lw=2, ax=ax)
rh_ice.mean(dim="launch_time").plot(y="alt", lw=3, label="Mean", c="k", ax=ax)

# plot vertical line at 100%
ax.axvline(x=100, color="#661100", lw=3)
ax.set_xlabel("Relative Humidity over Ice (%)")
ax.set_ylabel("Altitude (km)")
ax.grid()
ax.legend(bbox_to_anchor=(1, 1.01), loc="upper left", fontsize=14)
ax.text(0.1, 0.95, "(b)", size=18, transform=ax.transAxes, ha="center", va="center",
        bbox=dict(boxstyle="round", ec="grey", fc="white"))

figname = f"{plot_path}/{flight}_trajectories_dropsonde_rh_ice.png"
print(figname)
plt.tight_layout()
plt.savefig(figname, format='png', dpi=300)  # , bbox_inches='tight')
print("\t\t\t ...saved as: " + str(figname))
plt.close()
