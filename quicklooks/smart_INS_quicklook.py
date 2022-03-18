#!/usr/bin/env python
"""Map Quicklook generated from SMART INS data

*author*: Johannes Röttenbacher
"""

# %% modules
import pylim.helpers as h
from pylim.halo_ac3 import coordinates
from ac3airborne.tools.get_amsr2_seaice import get_amsr2_seaice
import os
import xarray as xr
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from matplotlib import patheffects
import pandas as pd
import cartopy
import cartopy.crs as ccrs

# %% user input
campaign = "halo-ac3"
date = "20220316"
flight = f"HALO-AC3_{date}_HALO_RF06"
flight_key = flight[-4:] if campaign == "halo-ac3" else flight
add_satellite = False  # needs to provide a satellite image which has to be manually downloaded
add_seaice = True
add_dropsondes = True
use_gridded_dropsondes = True
savefig = True

# %% set plot properties and get some information from the dicts
defaults = dict(extent=[-15, 30, 68, 81], figsize=(5.5, 5.5), cb_loc="bottom", shrink=1, l_loc=1)
plot_props = dict(RF00=dict(figsize=(4, 4)),
                  RF01=dict(extent=[5, 25, 47, 69]),
                  RF02=dict(extent=[-15, 30, 68, 85]),
                  RF03=dict(extent=[-15, 30, 68, 90], figsize=(4.8, 5.5)),
                  RF04=dict(extent=[-15, 30, 68, 90], figsize=(4.8, 5.5)),
                  RF05=dict(figsize=(5.5, 4)),
                  RF06=dict(extent=[-15, 30, 68, 85]))
x_kiruna, y_kiruna = coordinates["Kiruna"]
x_longyear, y_longyear = coordinates["Longyearbyen"]

# %% set paths and read in files
ql_path = h.get_path("quicklooks", flight, campaign)
h.make_dir(ql_path)
horipath = h.get_path("horidata", flight, campaign)
infile = [f for f in os.listdir(horipath) if f.endswith("nc")][0]  # read in the INS quicklook file
if add_satellite:
    sat_path = h.get_path("satellite", flight, campaign)
    sat_file = [f for f in os.listdir(sat_path) if "MODIS_" in f][0]
    sat_ds = rasterio.open(f"{sat_path}/{sat_file}")

if add_dropsondes:
    dropsonde_path = h.get_path("dropsondes", flight, campaign)
    dropsonde_files = [os.path.join(dropsonde_path, f) for f in os.listdir(dropsonde_path) if f.endswith("nc")]
    # create dataframe with dropsonde launch times and locations
    dropsondes = dict()
    for i in range(len(dropsonde_files)):
        dropsondes[f"{i}"] = xr.open_dataset(dropsonde_files[i])
    launch_times = [pd.to_datetime(dropsondes[var].time[-1].values) for var in dropsondes]
    longitudes = [dropsondes[var].lon.min(skipna=True).values for var in dropsondes]
    latitudes = [dropsondes[var].lat.min(skipna=True).values for var in dropsondes]
    dropsonde_df = pd.DataFrame({"launch_time": launch_times, "lon": longitudes, "lat": latitudes})

    # quickgrid dropsonde file
    gridded_dropsondes = [f for f in os.listdir(f"{dropsonde_path}/..") if f.endswith("nc")][0]
    dropsondes_ds = xr.open_dataset(f"{dropsonde_path}/../{gridded_dropsondes}")

ins = xr.open_dataset(f"{horipath}/{infile}")

# %% plot map quicklook
orig_map = plt.cm.get_cmap('Blues')  # getting the original colormap using cm.get_cmap() function
reversed_map = orig_map.reversed()  # reversing the original colormap using reversed() function

# select position and time data
lon, lat, altitude, times = ins["lon"], ins["lat"], ins["alt"], ins["time"]
# calculate flight duration
flight_duration = pd.Timedelta((times[-1] - times[0]).values).to_pytimedelta()

# set plotting options
plt.rcdefaults()
font = {'size': 10}
plt.rc('font', **font)
data_crs = ccrs.PlateCarree()
props = plot_props[flight_key]  # get plot properties
# read out properties or use default settings
extent = props["extent"] if "extent" in props else defaults["extent"]
figsize = props["figsize"] if "figsize" in props else defaults["figsize"]
cb_loc = props["cb_loc"] if "cb_loc" in props else defaults["cb_loc"]
shrink = props["shrink"] if "shrink" in props else defaults["shrink"]
l_loc = props["l_loc"] if "l_loc" in props else defaults["l_loc"]

# start plotting
fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": ccrs.NorthPolarStereo()})
# add satellite image
if add_satellite:
    show(sat_ds, ax=ax)
else:
    # ax.stock_img()
    # ax.background_img(name='natural-earth-1', resolution='large8192px')
    pass

ax.coastlines()
ax.add_feature(cartopy.feature.BORDERS)
ax.set_extent(extent, crs=data_crs)
gl = ax.gridlines(crs=data_crs, draw_labels=True, x_inline=False, y_inline=False)
gl.bottom_labels = False
gl.left_labels = False

# add sea ice extent
if add_seaice:
    seaice = get_amsr2_seaice(f"{(pd.to_datetime(date) - pd.Timedelta(days=0)):%Y%m%d}")
    seaice = seaice.seaice
    ax.pcolormesh(seaice.lon, seaice.lat, seaice, transform=data_crs, cmap=reversed_map)

# plot flight track
points = ax.scatter(lon, lat, s=1, c="orange", transform=data_crs)

# plot dropsonde launch locations
if add_dropsondes:
    if not use_gridded_dropsondes:
        for i in range(dropsonde_df.shape[0]):
            df = dropsonde_df.iloc[i]
            ax.annotate(f"{df['launch_time']:%H:%M}", (df.lon, df.lat), fontsize=8, transform=data_crs)
            ax.text(df.lon, df.lat, f"{df['launch_time']:%H:%M}", c="white", fontsize=10, transform=data_crs,
                    path_effects=[patheffects.withStroke(linewidth=0.5, foreground="black")])
            ax.plot(df.lon, df.lat, "x", color="red", markersize=8, label="Dropsonde", transform=data_crs)
    else:
        for i in range(dropsondes_ds.lon.shape[0]):
            launch_time = pd.to_datetime(dropsondes_ds.launch_time[i].values)
            x, y = dropsondes_ds.lon[i].mean().values, dropsondes_ds.lat[i].mean().values
            ax.plot(x, y, "x", color="red", markersize=8, label="Dropsonde", transform=data_crs)
            ax.text(x, y, f"{launch_time:%H:%M}", c="white", fontsize=10, transform=data_crs,
                    path_effects=[patheffects.withStroke(linewidth=0.5, foreground="black")])

    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[0]], [labels[0]], loc=l_loc)


# plot some place labels
# x_edmo, y_edmo = coordinates["EDMO"]
# ax.plot(x_edmo, y_edmo, '.', color="#117733", markersize=8, transform=data_crs)
# ax.text(x_edmo + 0.1, y_edmo + 0.1, "EDMO", fontsize=10, transform=data_crs)
# plot a second airport label
# x2, y2 = coordinates["Leipzig"]
# ax.plot(x2, y2, '.', color="#117733", markersize=8, transform=data_crs)
# ax.text(x2 + 0.1, y2 + 0.1, "Leipzig", fontsize=10, transform=data_crs)
# plot a third location label
# x2, y2 = coordinates["Jülich"]
# ax.plot(x2, y2, '.', color="#117733", markersize=8, transform=data_crs)
# ax.text(x2 + 0.1, y2 + 0.1, "Jülich", fontsize=10, transform=data_crs)
# Kiruna
x_kiruna, y_kiruna = coordinates["Kiruna"]
ax.plot(x_kiruna, y_kiruna, '.', color="#117733", markersize=8, transform=data_crs)
ax.text(x_kiruna + 0.1, y_kiruna + 0.1, "Kiruna", fontsize=11, transform=data_crs)
# Longyearbyen
x_longyear, y_longyear = coordinates["Longyearbyen"]
ax.plot(x_longyear, y_longyear, '.', color="#117733", markersize=8, transform=data_crs)
ax.text(x_longyear + 0.1, y_longyear + 0.1, "Longyearbyen", fontsize=11, transform=data_crs,
        path_effects=[patheffects.withStroke(linewidth=0.5, foreground="white")])

# plot most northerly point
# n_lat, n_lon = float(ins.lat.max()), float(ins.lon[ins.lat == ins.lat.max()])
# ax.plot(n_lon, n_lat, 'X', color="#117733", markersize=8, transform=data_crs)
# ax.text(n_lon + 0.1, n_lat + 0.1, f"({n_lat:4.2f}N, {n_lon:4.2f}E)", fontsize=11,
#         transform=data_crs, path_effects=[patheffects.withStroke(linewidth=0.5, foreground="white")])

# # add wind barbs
# increment = 5000
# lat, lon, u, v = lat[0::increment], lon[0::increment], bahamas.U[::increment] * 1.94384449, bahamas.V[::increment] * 1.94384449
# ax.barbs(lon, lat, u, v, length=6, transform=data_crs)

# write the flight duration in the lower left corner of the map
ax.text(0, 0.01, f"Duration: {str(flight_duration)[:4]} (hr:min)", transform=ax.transAxes, fontsize=11, color="white",
        path_effects=[patheffects.withStroke(linewidth=0.5, foreground="black")])
plt.tight_layout(pad=0.3)
fig_name = f"{ql_path}/HALO-AC3_HALO_SMART_INS-track_{date}_{flight_key}.png"
if savefig:
    plt.savefig(fig_name, dpi=100)
    print(f"Saved {fig_name}")
else:
    plt.show()
plt.close()
