#!/usr/bin/env python
"""Case studies for Cirrus-HL
* 29.06.2021: cirrus over Atlantic west and north of Iceland
author: Johannes Röttenbacher
"""

# %% module import
import datetime
from smart import get_path
import logging
from bahamas import plot_props
from libradtran import read_libradtran
from cirrus_hl import stop_over_locations, coordinates
import os
import smart
from functions_jr import make_dir, set_cb_friendly_colors
import matplotlib
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import xarray as xr

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

#######################################################################################################################
# 20210629
#######################################################################################################################
# %% set paths
flight = "Flight_20210629a"
bahamas_dir = get_path("bahamas", flight)
bacardi_dir = get_path("bacardi", flight)
outpath = f"C:/Users/Johannes/Documents/Doktor/campaigns/CIRRUS-HL/case_studies/{flight}"
make_dir(outpath)
start_dt = datetime.datetime(2021, 6, 29, 10, 10)
end_dt = datetime.datetime(2021, 6, 29, 11, 54)

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

# %% read ind libradtran and bacardi files
libradtran_file = "BBR_Fdn_clear_sky_Flight_20210629a_R0_ds_high_jr.dat"
bacardi_file = "CIRRUS_HL_F05_20210629a_ADLR_BACARDI_BroadbandFluxes_R0.nc"
bbr_sim = read_libradtran(flight, libradtran_file)
bacardi_ds = xr.open_dataset(f"{bacardi_dir}/{bacardi_file}")

# %% plot libradtran simulations together with BACARDI measurements
set_cb_friendly_colors()
fig, ax = plt.subplots()
bacardi_ds.F_up_solar.plot(x="time", label="F_up solar BACARDI", ax=ax)
bacardi_ds.F_down_solar.plot(x="time", label="F_dw solar BACARDI", ax=ax)
bbr_sim.plot(y="F_dw", ax=ax, ylabel="Broadband irradiance (W$\,$m$^{-2}$)", label="F_dw solar libRadtran")
bbr_sim.plot(y="F_up", ax=ax, label="F_up solar libRadtran")
ax.set_xlabel("Time (UTC)")
ax.grid()
ax.fill_between(bbr_sim.index, 0, 1, where=((start_dt < bbr_sim.index) & (bbr_sim.index < end_dt)),
                transform=ax.get_xaxis_transform(), label="Case Study", color="grey")
ax.legend(bbox_to_anchor=(0.2, 0), loc="lower left", bbox_transform=fig.transFigure, ncol=2)
plt.subplots_adjust(bottom=0.3)
plt.tight_layout()
plt.show()
plt.savefig(f"{outpath}/{flight}_bacardi_libradtran_broadband_irradiance.png", dpi=100)
plt.close()
