#!/usr/bin/env python
"""Case Study for RF17 2022-04-11

Arctic Dragon. Flight to the North with a west-east cross-section above and below the cirrus.

*author*: Johannes Röttenbacher
"""

# %% import modules
import pylim.helpers as h
from pylim.halo_ac3 import coordinates
from pylim import smart
import ac3airborne
from ac3airborne.tools.get_amsr2_seaice import get_amsr2_seaice
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
date = "20220328"
halo_key = "RF09"
halo_flight = f"HALO-AC3_{date}_HALO_{halo_key}"

plot_path = f"{h.get_path('plot', halo_flight, campaign)}/{halo_flight}"
h.make_dir(plot_path)
halo_smart_path = h.get_path("calibrated", halo_flight, campaign)
swir_file = f"HALO-AC3_HALO_SMART_Fdw_SWIR_{date}_{halo_key}.nc"
vnir_file = f"HALO-AC3_HALO_SMART_Fdw_VNIR_{date}_{halo_key}.nc"
libradtran_path = h.get_path("libradtran", halo_flight, campaign)
libradtran_file = f"HALO-AC3_HALO_libRadtran_clearsky_simulation_500-600nm_{date}_{halo_key}.nc"
bahamas_path = h.get_path("bahamas", halo_flight, campaign)
bahamas_file = f"QL_HALO-AC3_HALO_BAHAMAS_{date}_{halo_key}_v1.nc"

# flight tracks from halo-ac3 cloud
kwds = {'simplecache': dict(cache_storage='E:/HALO-AC3/cloud', same_names=True)}
credentials = {"user": os.environ.get("AC3_CLOUD_USER"), "password": os.environ.get("AC3_CLOUD_PASSWORD")}
cat = ac3airborne.get_intake_catalog()
HALO_track = cat["HALO-AC3"]["HALO"]["GPS_INS"][f"HALO-AC3_HALO_{halo_key}"](storage_options=kwds, **credentials).to_dask()

# %% read in HALO smart data
halo_swir = xr.open_dataset(f"{halo_smart_path}/{swir_file}")
halo_vnir = xr.open_dataset(f"{halo_smart_path}/{vnir_file}")
halo_all = smart.merge_vnir_swir_nc(halo_vnir, halo_swir)

# %% read in libRadtran simulation
libradtran_sim = xr.open_dataset(f"{libradtran_path}/{libradtran_file}")
# libradtran is simulated with the BAHAMAS time, cut it to SMART time
libradtran_sim = libradtran_sim.where(libradtran_sim.time <= (halo_all.time[-1] + pd.Timedelta(1, "min")), drop=True)

# %% filter high roll angles
HALO_track_inp = HALO_track.interp(time=halo_all.time)
halo_all_filtered = halo_all.where(np.abs(HALO_track_inp["roll"]) < 2)

# %% resample data to minutely resolution
HALO_track = HALO_track.resample(time="1Min").asfreq()
HALO_track = HALO_track.where(HALO_track.time <= halo_all_filtered.time)
halo_all_filtered = halo_all_filtered.resample(time="1Min").asfreq()
libradtran_sim["time"] = halo_all_filtered.time  # replace libRadtran time with SMART time for merging

# %% integrate over wavelengths
halo_500_600 = halo_all_filtered.sel(wavelength=slice(500, 600)).sum(dim="wavelength")
halo_500_600["Fdw"] = halo_500_600.Fdw.where(halo_500_600.Fdw != 0, np.nan)

# %% relation between simulation and measurement
relation = halo_500_600.Fdw / libradtran_sim.fdw

# %% viewing direction of halo: 0 = towards sun, 180 = away from sun
heading = HALO_track.yaw * -1  # reverse turning direction of yaw angle
# replace negative values with positive angles to convert range from -180 - 180 to 0 - 360
heading = heading.where(heading > 0, heading + 360)
heading = (heading + 90)  # change from mathematical convention to meteorological convention (East = 0° -> North = 0°)
heading = heading.where(heading < 360, heading - 360)
viewing_dir = HALO_track.saa - heading
viewing_dir = viewing_dir.where(viewing_dir > 0, viewing_dir + 360)

# %% merge information in data frame
df1 = viewing_dir.to_dataframe(name="viewing_dir")
df2 = relation.to_dataframe(name="relation")
df = df1.merge(df2, on="time")
df["sza"] = HALO_track.sza
df = df.sort_values(by="viewing_dir")
df = df[df.relation != 0.7]

# %% plot relation as function of viewing direction
fig, ax = plt.subplots()
ax.scatter(df["viewing_dir"], df["relation"], label="0 = facing sun\n180 = facing away from sun")
ax.grid(True)
ax.set_title("Relation between SMART Fdw measurement and libRadtran simulation\n"
             "(500-600nm)\n"
             " according to viewing direction of HALO with respect to the sun")
ax.set_xlabel("Viewing direction with respect to sun")
ax.set_ylabel(r"F$_{\downarrow, measured}$ / F$_{\downarrow, simulated}$")
ax.legend()
plt.tight_layout()
# plt.show()
figname= f"{plot_path}/HALO-AC3_{date}_HALO_{halo_key}_inlet_directional_dependence.png"
plt.savefig(figname, dpi=300)
plt.close()

# %% plot relation as function of viewing direction as polar plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.scatter(np.deg2rad(df["viewing_dir"]), df["relation"], label="0 = facing sun\n180 = facing away from sun")
ax.set_rmax(1.5)
ax.set_rticks([0.5, 1, 1.5])  # Less radial ticks
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi/2)
ax.grid(True)

ax.set_title("Relation between SMART Fdw measurement and libRadtran simulation\n"
             "(500-600nm)\n"
             " according to viewing direction of HALO with respect to the sun")
ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure)
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
# plt.show()
figname= f"{plot_path}/HALO-AC3_{date}_HALO_{halo_key}_inlet_directional_dependence_polar.png"
plt.savefig(figname, dpi=300)
plt.close()

# %% plot relation as function of SZA
fig, ax = plt.subplots()
ax.scatter(df["sza"], df["relation"], label="0 = zenith")
ax.grid()
ax.set_title("Relation between SMART Fdw measurement and libRadtran simulation\n"
             "(500-600nm)\n"
             "according to solar zenith angle")
ax.set_ylabel(r"F$_{\downarrow, measured}$ / F$_{\downarrow, simulated}$")
ax.set_xlabel("Solar Zenith Angle")
ax.legend()
plt.tight_layout()
# plt.show()
figname= f"{plot_path}/HALO-AC3_{date}_HALO_{halo_key}_inlet_sza_dependence.png"
plt.savefig(figname, dpi=300)
plt.close()

# %% plot relation as function of SZA as polar plot
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.scatter(np.deg2rad(df["sza"]), df["relation"], label="0 = zenith")
ax.set_rmax(1.5)
ax.set_rticks([0.5, 1, 1.5])  # Less radial ticks
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.set_theta_direction(-1)
ax.set_theta_offset(np.pi/2)
ax.grid(True)

ax.set_title("Relation between SMART Fdw measurement and libRadtran simulation\n"
             "(500-600nm)\n"
             " according to solar zenith angle")
ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure)
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
# plt.show()
figname= f"{plot_path}/HALO-AC3_{date}_HALO_{halo_key}_inlet_sza_dependence_polar.png"
plt.savefig(figname, dpi=300)
plt.close()

# %% plot simulation and measurement
h.set_cb_friendly_colors()
td = pd.to_timedelta((halo_500_600.time[-1] - halo_500_600.time[0]).values)
fig, ax = plt.subplots()
ax.plot(halo_500_600.time, halo_500_600.Fdw, label="SMART")
ax.plot(libradtran_sim.time, libradtran_sim.fdw, label="libRadtran")
h.set_xticks_and_xlabels(ax, td)
ax.grid()
ax.legend()
ax2 = ax.twinx()
ax2.plot(halo_500_600.time, df["viewing_dir"], color="#DDCC77", ls="--", label="Viewing direction")
ax2.set_ylabel("Viewing direction with respect to sun")
ax2.legend()
ax.set_title("Integrated downward Irradiance (500-600nm)")
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Downward Irradiance (W$\,$m$^{-2}$)")
# plt.show()
plt.savefig(f"{plot_path}/HALO-AC3_{date}_HALO_{halo_key}_SMART_Fdw-500-600nm_meas-vs-sim.png", dpi=300)
plt.close()

# %% histogram of viewing direction
fig, ax = plt.subplots()
df["viewing_dir"].hist(bins=36, ax=ax)
ax.text(0.7, 0.8, "Binsize = 10°", transform=fig.transFigure)
ax.set_xlabel("Viewing direction with respect to the sun")
ax.set_ylabel("Counts")
plt.savefig(f"{plot_path}/HALO-AC3_{date}_HALO_{halo_key}_viewing-direction_histogram.png", dpi=300)
plt.close()

# %% correlate viewing direction and sza with relation
df.corr()
