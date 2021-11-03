#!/usr/bin/env python
"""Case study for Cirrus-HL
* 12.07.2021: First day with SMART fixed
author: Johannes Röttenbacher
"""

# %% module import
import pylim.helpers as h
from pylim import reader
from pylim.bahamas import plot_props
from pylim.cirrus_hl import coordinates
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patheffects
import cartopy
import cartopy.crs as ccrs
import pandas as pd
import rasterio
from rasterio.plot import show
import logging
log = logging.getLogger("pylim")
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

# %% 20210712 First day with SMART fixed
print("20210712 SMART fixed")
rs_start = pd.Timestamp(2021, 7, 12, 13, 15)
rs_end = pd.Timestamp(2021, 7, 12, 15, 15)
flight = "Flight_20210712b"
bahamas_dir = h.get_path("bahamas", flight)
bacardi_dir = h.get_path("bacardi", flight)
smart_dir = h.get_path("calibrated", flight)
sat_dir = h.get_path("satellite", flight)
libradtran_dir = h.get_path("libradtran", flight)
if os.getcwd().startswith("C:"):
    outpath = f"C:/Users/Johannes/Documents/Doktor/campaigns/CIRRUS-HL/case_studies/{flight}"
else:
    outpath = f"/projekt_agmwend/home_rad/jroettenbacher/case_studies/{flight}"
h.make_dir(outpath)

# %% find bahamas file and read in bahamas data and satellite picture
file = [f for f in os.listdir(bahamas_dir) if f.endswith(".nc")][0]
bahamas = reader.read_bahamas(f"{bahamas_dir}/{file}")
sat_image = [f for f in os.listdir(sat_dir) if "MODIS" in f][0]
sat_ds = rasterio.open(f"{sat_dir}/{sat_image}")
# bahamas_rs = bahamas_subset.where(np.abs(bahamas_subset["IRS_PHI"]) < 1)  # select only sections with roll < 1°
bahamas_rs = bahamas.sel(time=slice(rs_start, rs_end))  # select subset of radiation square

# %% BAHAMAS: select position and time data and set extent
x_edmo, y_edmo = coordinates["EDMO"]
lon, lat, altitude, times = bahamas_rs["IRS_LON"], bahamas_rs["IRS_LAT"], bahamas_rs["IRS_ALT"], bahamas_rs["time"]
pad = 2
llcrnlat = lat.min(skipna=True) - pad
llcrnlon = lon.min(skipna=True) - pad
urcrnlat = lat.max(skipna=True) + pad
urcrnlon = lon.max(skipna=True) + pad
extent = [llcrnlon, urcrnlon, llcrnlat, urcrnlat]
font = {'weight': 'bold', 'size': 26}
matplotlib.rc('font', **font)
# get plot properties
props = plot_props[flight]
h.set_cb_friendly_colors()

# %% plot bahamas map with sat image
fig, ax = plt.subplots(figsize=(11, 9), subplot_kw={"projection": ccrs.PlateCarree()})
# ax.stock_img()
show(sat_ds, ax=ax)
ax.coastlines(linewidth=3)
ax.add_feature(cartopy.feature.BORDERS, linewidth=3)
ax.set_extent(extent)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
gl.bottom_labels = False
gl.left_labels = False

# plot flight track
points = ax.scatter(lon, lat, c=bahamas_rs["IRS_ALT"] / 1000, linewidth=6)
# add the corresponding colorbar and decide whether to plot it horizontally or vertically
plt.colorbar(points, ax=ax, pad=0.01, location=props["cb_loc"], label="Height (km)", shrink=props["shrink"])

# plot a way point every 15 minutes = 9000 seconds with a time stamp next to it
for long, lati, time_stamp in zip(lon[9000::9000], lat[9000::9000], times[9000::9000]):
    ax.annotate(time_stamp.dt.strftime("%H:%M").values, (long, lati), fontsize=16,
                path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])
    ax.plot(long, lati, '.k', markersize=10)

# plot points with labels and white line around text
# ax.plot(x_edmo, y_edmo, 'ok')
# ax.text(x_edmo + 0.1, y_edmo + 0.1, "EDMO", fontsize=22,
#         path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])
plt.tight_layout(pad=0.1)
fig_name = f"{outpath}/{flight}_bahamas_track_with_sat.png"
# plt.show()
plt.savefig(fig_name, dpi=100)
log.info(f"Saved {fig_name}")
plt.close()

