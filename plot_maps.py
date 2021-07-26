#!\usr\bin\env python
"""Plotting script for map plots
1. Plot a map of the flight track together with a marker for HALO
author: Johannes Röttenbacher
"""

# %% import libraries
import logging
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import smart
from smart import stop_over_locations
from functions_jr import make_dir
import cartopy.crs as ccrs
import cartopy
from tqdm import tqdm
from joblib import Parallel, cpu_count, delayed

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.WARNING)

# %% set paths
date = 20210707
flight = f"Flight_{date}a"
bahamas_dir = smart.get_path("bahamas", flight)
bahamas_path = f"{bahamas_dir}/{flight}"
gopro_dir = smart.get_path("gopro")
# find bahamas file
file = [f for f in os.listdir(bahamas_dir) if f.endswith(".nc")][0]
# select second airport for map plot according to flight
airport = stop_over_locations[flight] if flight in stop_over_locations else None

# %% read in bahamas data
bahamas = smart.read_bahamas(f"{bahamas_dir}/{file}")
# select only position data
lon = bahamas["IRS_LON"]
lat = bahamas["IRS_LAT"]

# %% find map extend
pad = 2
llcrnlat = lat.min(skipna=True) - pad
llcrnlon = lon.min(skipna=True) - pad
urcrnlat = lat.max(skipna=True) + pad
urcrnlon = lon.max(skipna=True) + pad
extent = [llcrnlon, urcrnlon, llcrnlat, urcrnlat]

# %% select lon and lat values corresponding with the picture timestamps
# get first and last bahamas time step
first_ts, last_ts = pd.to_datetime(bahamas.TIME[0].values), pd.to_datetime(bahamas.TIME[-1].values)
# make strings with the only the time from timestamps
first_ts, last_ts = first_ts.strftime("%H:%M:%S"), last_ts.strftime("%H:%M:%S")
# read timestamps
timestamps = pd.read_csv(f"{gopro_dir}/{date}_timestamps.csv", index_col="datetime", parse_dates=True)
# select range of timestamps
ts_sel = timestamps.between_time(first_ts, last_ts)
# write out pictures used
ts_sel.to_csv(f"{gopro_dir}/{flight}_timestamps_sel.csv", index_label="datetime")

# %% select corresponding lat and lon values
lon_sel = bahamas.IRS_LON.sel(TIME=ts_sel.index)
lat_sel = bahamas.IRS_LAT.sel(TIME=ts_sel.index)
assert len(lon_sel) == len(lat_sel), "Lon and Lat are not of same lenght!"
# %% plot on map


def plot_bahamas_map(flight: str, lon, lat, extent: list, lon1: float, lat1: float, number: int, **kwargs):
    """
    Plot a map of the flight track from BAHAMAS data with the location of HALO.
    Args:
        flight: Flight name (eg. Flight_20210707a)
        lon: array with longitude values for flight track
        lat: array with latitude values for flight track
        extent: list with map extent [llcrnlon, urcrnlon, llcrnlat, urcrnlat]
        lon1: longitude position of HALO
        lat1: latitude position of HALO
        number: GoPro picture number corresponding with the BAHAMAS time step of the HALO position
        **kwargs:
            outpath (str): where to save plot
            airport (str): second airport to label on map

    Returns: Saves a png file

    """
    bahamas_dir = smart.get_path("bahamas", flight)
    outpath = kwargs["outpath"] if "outpath" in kwargs else f"{bahamas_dir}/plots/time_lapse"
    make_dir(outpath)
    airport = kwargs["airport"] if "airport" in kwargs else None
    font = {'weight': 'bold', 'size': 26}
    matplotlib.rc('font', **font)

    # get plot properties
    props = plot_props[flight]
    fig, ax = plt.subplots(figsize=props["figsize"], subplot_kw={"projection": ccrs.PlateCarree()})
    ax.stock_img()
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS)
    ax.set_extent(extent)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    gl.bottom_labels = False
    gl.left_labels = False
    # plot a way point every 30 minutes = 1800 seconds
    for long, lati, nr in zip(lon[18000::18000], lat[18000::18000], range(len(lat[18000::18000]))):
        ax.annotate(nr + 1, (long, lati), fontsize=16)
        ax.plot(long, lati, '.r', markersize=10)
    # plot an airplane marker for HALO
    ax.plot(lon1, lat1, c="k", marker="$\u2708$", markersize=28, label="HALO")
    # get the coordinates for EDMO and ad a label
    x_edmo, y_edmo = smart.coordinates["EDMO"]
    ax.plot(x_edmo, y_edmo, 'ok')
    ax.text(x_edmo + 0.1, y_edmo + 0.1, "EDMO", fontsize=16)
    # plot a second airport label if given
    if airport is not None:
        x2, y2 = smart.coordinates[airport]
        ax.text(x2 + 0.1, y2 + 0.1, airport, fontsize=16)
    # plot flight track and color by flight altitude
    points = ax.scatter(lon, lat, c=bahamas.IRS_ALT/1000, s=10)
    # add the corresponding colorbar and decide whether to plot it horizontally or vertically
    ax.legend(loc=props["l_loc"])
    plt.colorbar(points, ax=ax, pad=0.01, location=props["cb_loc"], label="Height (km)", shrink=props["shrink"])
    plt.tight_layout(pad=0.1)
    fig_name = f"{outpath}/{flight}_map_{number:04}.png"
    plt.savefig(fig_name, dpi=100)
    log.info(f"Saved {fig_name}")
    plt.close()


# %% set plotting options for each flight
plot_props = dict(Flight_20210625a=dict(figsize=(9, 9), cb_loc="left", shrink=1, l_loc=1),
                  Flight_20210626a=dict(figsize=(9.5, 8), cb_loc="bottom", shrink=0.9, l_loc=4),
                  Flight_20210628a=dict(figsize=(10, 9), cb_loc="left", shrink=1, l_loc=4),
                  Flight_20210629a=dict(figsize=(9, 8.2), cb_loc="bottom", shrink=1, l_loc=1),
                  Flight_20210629b=dict(figsize=(8, 8), cb_loc="left", shrink=1, l_loc=3),
                  Flight_20210701a=dict(figsize=(9, 8), cb_loc="bottom", shrink=1, l_loc=2),
                  Flight_20210705a=dict(figsize=(8, 8), cb_loc="left", shrink=1, l_loc=4),
                  Flight_20210705b=dict(figsize=(9, 8), cb_loc="left", shrink=1, l_loc=3),
                  Flight_20210707a=dict(figsize=(10, 7), cb_loc="bottom", shrink=1, l_loc=1),
                  Flight_20210707b=dict(figsize=(10, 7), cb_loc="bottom", shrink=1, l_loc=1))
# %% loop through timesteps
# lon1 = lon[0]
# lat1 = lat[0]
# number = 0
# plot_bahamas_map(flight, lon, lat, extent, lon1, lat1, number, airport=airport)
Parallel(n_jobs=cpu_count()-4)(delayed(plot_bahamas_map)
                               (flight, lon, lat, extent, lon1, lat1, number, airport=airport)
                               for lon1, lat1, number in zip(tqdm(lon_sel), lat_sel, ts_sel.number.values))
