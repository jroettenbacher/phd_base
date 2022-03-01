#!/usr/bin/env python
"""Map Quicklook generated from SMART INS data

*author*: Johannes Röttenbacher
"""

# %% modules
import pylim.helpers as h
from pylim.halo_ac3 import coordinates
import os
import xarray as xr
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import pandas as pd
import cartopy
import cartopy.crs as ccrs

# %% user input
campaign = "halo-ac3"
flight = "HALO-AC3_FD00_HALO_RF01_20220225"
flight_key = flight[19:] if campaign == "halo-ac3" else flight
add_satellite = True  # needs to provide a satellite image which has to be manually downloaded
savefig = True

# %% set plot properties
plot_props = dict(RF01_20220225=dict(figsize=(4, 4), cb_loc="bottom", shrink=1, l_loc=1))

# %% set paths and read in files
ql_path = h.get_path("quicklooks", flight, campaign)
horipath = h.get_path("horidata", flight, campaign)
infile = [f for f in os.listdir(horipath) if f.endswith("nc")][0]  # read in the INS quicklook file
if add_satellite:
    sat_path = h.get_path("satellite", flight, campaign)
    sat_file = [f for f in os.listdir(sat_path) if "MODIS" in f][0]
    sat_ds = rasterio.open(f"{sat_path}/{sat_file}")

dropsonde_path = h.get_path("dropsonde", flight, campaign)
dropsonde_files = os.listdir(dropsonde_path)
ins = xr.open_dataset(f"{horipath}/{infile}")

# %% plot map quicklook
orig_map = plt.cm.get_cmap('Blues')  # getting the original colormap using cm.get_cmap() function
reversed_map = orig_map.reversed()  # reversing the original colormap using reversed() function

# select position and time data
lon, lat, altitude, times = ins["lon"], ins["lat"], ins["alt"], ins["time"]
# calculate flight duration
flight_duration = pd.Timedelta((times[-1] - times[0]).values).to_pytimedelta()
# set extent of plot
llcrnlat = 47
llcrnlon = 5
urcrnlat = 58
urcrnlon = 15
extent = [llcrnlon, urcrnlon, llcrnlat, urcrnlat]
# set plotting options
plt.rcdefaults()
font = {'size': 8}
plt.rc('font', **font)
# get plot properties
props = plot_props[flight_key]
# start plotting
fig, ax = plt.subplots(figsize=props["figsize"], subplot_kw={"projection": ccrs.PlateCarree()})
# add satellite image
if add_satellite:
    show(sat_ds, ax=ax)
else:
    ax.stock_image()
ax.coastlines()
ax.add_feature(cartopy.feature.BORDERS)
ax.set_extent(extent, crs=ccrs.PlateCarree())
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False)
gl.bottom_labels = False
gl.left_labels = False

# plot flight track
points = ax.plot(lon, lat, c="orange", linewidth=2)

# add sea ice extent
# ax.pcolormesh(seaice.lon, seaice.lat, seaice, transform=ccrs.PlateCarree(), cmap=reversed_map)
# plot dropsonde launch locations
for i, f in enumerate(dropsonde_files):
    ds = xr.open_dataset(f"{dropsonde_path}/{f}")
    ds_lon, ds_lat = ds.lon[-1], ds.lat[-1]
    ax.plot(ds_lon, ds_lat, "x", color="#CC6677", markersize=6, label="Dropsonde")
    ax.annotate(ds.time.dt.strftime("%H:%M").values[-1], (ds_lon, ds_lat), fontsize=8)

# plot a way point every 15 minutes = 900 seconds with a time stamp next to it
# for long, lati, time_stamp in zip(lon[900::900], lat[900::900], times[900::900]):
#     ax.annotate(time_stamp.dt.strftime("%H:%M").values, (long, lati), fontsize=8)
#     ax.plot(long, lati, '.', color="#D55E00", markersize=6)

# get the coordinates for EDMO and add a label
x_edmo, y_edmo = coordinates["EDMO"]
ax.plot(x_edmo, y_edmo, '.', color="#117733", markersize=8, transform=ccrs.PlateCarree())
ax.text(x_edmo + 0.1, y_edmo + 0.1, "EDMO", fontsize=10, transform=ccrs.PlateCarree())
# plot a second airport label
x2, y2 = coordinates["Leipzig"]
ax.plot(x2, y2, '.', color="#117733", markersize=8, transform=ccrs.PlateCarree())
ax.text(x2 + 0.1, y2 + 0.1, "Leipzig", fontsize=10, transform=ccrs.PlateCarree())
# plot a third location label
x2, y2 = coordinates["Jülich"]
ax.plot(x2, y2, '.', color="#117733", markersize=8, transform=ccrs.PlateCarree())
ax.text(x2 + 0.1, y2 + 0.1, "Jülich", fontsize=10, transform=ccrs.PlateCarree())

# # add wind barbs
# increment = 5000
# lat, lon, u, v = lat[0::increment], lon[0::increment], bahamas.U[::increment] * 1.94384449, bahamas.V[::increment] * 1.94384449
# ax.barbs(lon, lat, u, v, length=6, transform=ccrs.PlateCarree())

# write the flight duration in the lower left corner of the map
ax.text(0, 0.01, f"Duration: {str(flight_duration)[:4]} (hr:min)", transform=ax.transAxes, fontsize=10, color="white")
handles, labels = ax.get_legend_handles_labels()
ax.legend([handles[0]], [labels[0]], loc=1)
plt.tight_layout(pad=0.3)
fig_name = f"{ql_path}/SMART_INS-track_{flight[9:]}.png"
if savefig:
    plt.savefig(fig_name, dpi=100)
    print(f"Saved {fig_name}")
else:
    plt.show()
plt.close()
