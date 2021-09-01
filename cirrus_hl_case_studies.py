#!/usr/bin/env python
"""Case studies for Cirrus-HL
* 29.06.2021: cirrus over Atlantic west and north of Iceland
author: Johannes Röttenbacher
"""

# %% module import
from smart import get_path
import logging
from bahamas import plot_props
from cirrus_hl import stop_over_locations, coordinates
import os
import smart
from functions_jr import make_dir
import matplotlib
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

#######################################################################################################################
# 20210629
#######################################################################################################################
# %% set paths
flight = "Flight_20210629a"
bahamas_dir = get_path("bahamas", flight)
outpath = f"C:/Users/Johannes/Documents/campaigns/CIRRUS-HL/case_studies/{flight}"
make_dir(outpath)

# %% find bahamas file and read in bahamas data
file = [f for f in os.listdir(bahamas_dir) if f.endswith(".nc")][0]
bahamas = smart.read_bahamas(f"{bahamas_dir}/{file}")

# %% select further points to plot
x_edmo, y_edmo = coordinates["EDMO"]
airport = stop_over_locations[flight] if flight in stop_over_locations else None
x2, y2 = coordinates[airport]
torshavn_x, torshavn_y = coordinates["Torshavn"]

# %% select position and time data
lon, lat, altitude, times = bahamas["IRS_LON"], bahamas["IRS_LAT"], bahamas["IRS_ALT"], bahamas["TIME"]

# %% set plotting options
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

# %% start plotting
fig, ax = plt.subplots(figsize=props["figsize"], subplot_kw={"projection": ccrs.PlateCarree()})
ax.stock_img()
ax.coastlines()
ax.add_feature(cartopy.feature.BORDERS)
ax.set_extent(extent)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
gl.bottom_labels = False
gl.left_labels = False
# plot a way point every 15 minutes = 9000 seconds with a time stamp next to it
for long, lati, time_stamp in zip(lon[9000::9000], lat[9000::9000], times[9000::9000]):
    ax.annotate(time_stamp.dt.strftime("%H:%M").values, (long, lati), fontsize=10)
    ax.plot(long, lati, '.r', markersize=10)

# plot points with labels
ax.plot(x_edmo, y_edmo, 'ok')
ax.text(x_edmo + 0.1, y_edmo + 0.1, "EDMO", fontsize=16)
ax.plot(x2, y2, 'ok')
ax.text(x2 + 0.1, y2 + 0.1, airport, fontsize=16)
ax.plot(torshavn_x, torshavn_y, 'ok')
ax.text(torshavn_x + 0.1, torshavn_y + 0.1, "Torshavn", fontsize=16)

# plot flight track and color by flight altitude
points = ax.scatter(lon, lat, c=altitude / 1000, s=10)
# add the corresponding colorbar and decide whether to plot it horizontally or vertically
plt.colorbar(points, ax=ax, pad=0.01, location=props["cb_loc"], label="Height (km)", shrink=props["shrink"])
plt.tight_layout(pad=0.1)
fig_name = f"{outpath}/{flight}_bahamas_track.png"
plt.savefig(fig_name, dpi=100)
log.info(f"Saved {fig_name}")
plt.close()
