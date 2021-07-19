#!\usr\bin\env python
"""Plotting script for map plots
1. Plot a map of the flight track together with a marker for HALO
author: Johannes Röttenbacher
"""

# %% import libraries
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import smart
from functions_jr import make_dir
import cartopy.crs as ccrs
import cartopy
from tqdm import tqdm
from joblib import Parallel, cpu_count, delayed

# %% set paths
date = 20210707
flight = f"Flight_{date}a"
bahamas_dir = smart.get_path("bahamas")
bahamas_path = f"{bahamas_dir}/{flight}"
gopro_dir = f"C:/Users/Johannes/Documents/Gopro"
# find bahamas file
file = [f for f in os.listdir(bahamas_path) if f.endswith(".nc")][0]
# %% read in bahamas data
bahamas = smart.read_bahamas(f"{bahamas_path}/{file}")
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
ts_sel.to_csv(f"{gopro_dir}/{date}_timestamps_sel.csv", index_label="datetime")

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
    bahamas_dir = smart.get_path("bahamas")
    outpath = kwargs["outpath"] if "outpath" in kwargs else f"{bahamas_dir}/plots/{flight}/time_lapse"
    make_dir(outpath)
    airport = kwargs["airport"] if "airport" in kwargs else None
    font = {'weight': 'bold', 'size': 26}
    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(11, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.stock_img()
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS)
    ax.set_extent(extent)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    gl.bottom_labels = False
    gl.left_labels = False
    # plot a way point every 30 minutes = 1800 seconds
    for long, lati, nr in zip(lon[18000::18000], lat[18000::18000], range(len(lat[18000::18000]))):
        ax.annotate(nr+1, (long, lati), fontsize=16)
        ax.plot(long, lati, '.r', markersize=10)
    # plot an airplane marker for HALO
    ax.plot(lon1, lat1, c="k", marker="$\u2708$", markersize=28, label="HALO")
    # get the coordinates for EDMO and ad a label
    x_edmo, y_edmo = smart.coordinates["EDMO"]
    ax.plot(x_edmo, y_edmo, 'ok')
    ax.text(x_edmo+0.1, y_edmo+0.1, "EDMO", fontsize=16)
    # plot a second airport label if given
    if airport is not None:
        x2, y2 = smart.coordinates[airport]
        ax.text(x2+0.1, y2+0.1, airport, fontsize=16)
    # plot flight track and color by flight altitude
    points = ax.scatter(lon, lat, c=bahamas.IRS_ALT/1000, s=10)
    # add the corresponding colorbar
    plt.colorbar(points, ax=ax, pad=0.01, orientation="horizontal", label="Height (km)")
    ax.legend(loc=1)
    plt.savefig(f"{outpath}/{flight}_map_{number:04}.png", dpi=100, bbox_inches="tight")
    plt.close()


plot_bahamas_map(flight, lon, lat, extent, lon_sel[0], lat_sel[0], ts_sel.number.values[0], airport="Keflavik")
# Parallel(n_jobs=cpu_count()-2)(delayed(plot_bahamas_map)
#                                (bahamas_dir, flight, lon, lat, extent, lon1, lat1, number)
#                                for lon1, lat1, number in zip(tqdm(lon_sel), lat_sel, ts_sel.number.values))
