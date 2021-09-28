#!/usr/bin/env python
"""Script to keep functions to work with BAHAMAS data
author: Johannes Röttenbacher
"""

import matplotlib
import matplotlib.pyplot as plt
import smart
import xarray as xr
from cirrus_hl import stop_over_locations, coordinates
from functions_jr import make_dir
import os
import cartopy.crs as ccrs
import cartopy
import logging

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

# set plotting options for each flight
plot_props = dict(Flight_20210625a=dict(figsize=(9, 9), cb_loc="left", shrink=1, l_loc=1),
                  Flight_20210626a=dict(figsize=(9.5, 8), cb_loc="bottom", shrink=0.9, l_loc=4),
                  Flight_20210628a=dict(figsize=(10, 9), cb_loc="left", shrink=1, l_loc=4),
                  Flight_20210629a=dict(figsize=(9, 8.2), cb_loc="bottom", shrink=1, l_loc=1),
                  Flight_20210629b=dict(figsize=(8, 8), cb_loc="left", shrink=1, l_loc=3),
                  Flight_20210701a=dict(figsize=(9, 8), cb_loc="bottom", shrink=1, l_loc=2),
                  Flight_20210705a=dict(figsize=(8, 8), cb_loc="left", shrink=1, l_loc=4),
                  Flight_20210705b=dict(figsize=(9, 8), cb_loc="left", shrink=1, l_loc=3),
                  Flight_20210707a=dict(figsize=(10, 7), cb_loc="bottom", shrink=1, l_loc=1),
                  Flight_20210707b=dict(figsize=(10, 7), cb_loc="bottom", shrink=1, l_loc=1),
                  Flight_20210708a=dict(figsize=(9, 9), cb_loc="bottom", shrink=1, l_loc=1),
                  Flight_20210712a=dict(figsize=(11, 8), cb_loc="bottom", shrink=1, l_loc=1),
                  Flight_20210712b=dict(figsize=(10.5, 8), cb_loc="bottom", shrink=1, l_loc=1),
                  Flight_20210713a=dict(figsize=(9, 9), cb_loc="left", shrink=1, l_loc=1),
                  Flight_20210715a=dict(figsize=(10, 7), cb_loc="bottom", shrink=1, l_loc=4),
                  Flight_20210715b=dict(figsize=(10, 7), cb_loc="bottom", shrink=1, l_loc=2),
                  Flight_20210719a=dict(figsize=(9, 7.3), cb_loc="bottom", shrink=1, l_loc=3),
                  Flight_20210719b=dict(figsize=(9, 9), cb_loc="bottom", shrink=1, l_loc=3),
                  Flight_20210721a=dict(figsize=(10, 5), cb_loc="bottom", shrink=1, l_loc=2),
                  Flight_20210721b=dict(figsize=(10, 5), cb_loc="bottom", shrink=1, l_loc=4),
                  Flight_20210723a=dict(figsize=(9, 8.5), cb_loc="bottom", shrink=1, l_loc=1),
                  Flight_20210728a=dict(figsize=(9, 9), cb_loc="bottom", shrink=1, l_loc=1),
                  Flight_20210729a=dict(figsize=(9, 9), cb_loc="bottom", shrink=1, l_loc=1))


def plot_bahamas_flight_track(flight: str, **kwargs):
    """
    Plot a map of the flight track from BAHAMAS data with the location of HALO.
    Args:
        flight: Flight name (eg. Flight_20210707a)
        **kwargs:
            outpath (str): where to save plot (default: bahamas_dir/plots)

    Returns: Saves a png file

    """
    bahamas_dir = smart.get_path("bahamas", flight)
    outpath = kwargs["outpath"] if "outpath" in kwargs else f"{bahamas_dir}/plots"
    make_dir(outpath)
    # find bahamas file
    file = [f for f in os.listdir(bahamas_dir) if f.endswith(".nc")][0]
    # read in bahamas data
    bahamas = read_bahamas(f"{bahamas_dir}/{file}")
    # select second airport for map plot according to flight
    airport = stop_over_locations[flight] if flight in stop_over_locations else None
    # select position and time data
    lon, lat, altitude, times = bahamas["IRS_LON"], bahamas["IRS_LAT"], bahamas["IRS_ALT"], bahamas["TIME"]
    # set extent of plot
    pad = 2
    llcrnlat = lat.min(skipna=True) - pad
    llcrnlon = lon.min(skipna=True) - pad
    urcrnlat = lat.max(skipna=True) + pad
    urcrnlon = lon.max(skipna=True) + pad
    extent = [llcrnlon, urcrnlon, llcrnlat, urcrnlat]
    # set plotting options
    font = {'weight': 'bold', 'size': 26}
    matplotlib.rc('font', **font)
    # get plot properties
    props = plot_props[flight]
    # start plotting
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

    # get the coordinates for EDMO and ad a label
    x_edmo, y_edmo = coordinates["EDMO"]
    ax.plot(x_edmo, y_edmo, 'ok')
    ax.text(x_edmo + 0.1, y_edmo + 0.1, "EDMO", fontsize=16)
    # plot a second airport label if given
    if airport is not None:
        x2, y2 = coordinates[airport]
        ax.text(x2 + 0.1, y2 + 0.1, airport, fontsize=16)
    # plot flight track and color by flight altitude
    points = ax.scatter(lon, lat, c=altitude/1000, s=10)
    # add the corresponding colorbar and decide whether to plot it horizontally or vertically
    plt.colorbar(points, ax=ax, pad=0.01, location=props["cb_loc"], label="Height (km)", shrink=props["shrink"])
    plt.tight_layout(pad=0.1)
    fig_name = f"{outpath}/{flight}_bahamas_track.png"
    plt.savefig(fig_name, dpi=100)
    log.info(f"Saved {fig_name}")
    plt.close()


def read_bahamas(bahamas_path: str) -> xr.Dataset:
    """
    Reader function for netcdf BAHAMAS data
    Args:
        bahamas_path: full path to netcdf file

    Returns: xr.DataSet with BAHAMAS data and Time as dimension

    """
    ds = xr.open_dataset(bahamas_path)
    ds = ds.swap_dims({"tid": "TIME"})
    return ds


if __name__ == "__main__":
    # plot flight track with time stamps
    date = 20210629
    number = "a"
    flight = f"Flight_{date}{number}"
    plot_bahamas_flight_track(flight)