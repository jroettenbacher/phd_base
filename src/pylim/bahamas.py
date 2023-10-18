#!/usr/bin/env python
"""Functions to work with BAHAMAS data

- dictionary with plot properties for quicklook maps
- function to plot BAHAMAS map
- get position
- preprocess function for multiple file read in

*author*: Johannes Röttenbacher
"""
from pylim import reader
import pylim.helpers as h
from pylim.cirrus_hl import stop_over_locations, coordinates
import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import os
import cartopy.crs as ccrs
import cartopy
import pandas as pd
from geopy.distance import distance
from tqdm import tqdm
from typing import Union, Tuple
import logging
log = logging.getLogger(__name__)

# set plotting options for each flight
plot_props = dict(Flight_20210624a=dict(figsize=(9.5, 9), cb_loc="left", shrink=1, l_loc=1),
                  Flight_20210625a=dict(figsize=(9.5, 9), cb_loc="left", shrink=1, l_loc=1),
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
                  Flight_20210729a=dict(figsize=(9, 9), cb_loc="bottom", shrink=1, l_loc=1),
                  RF00=dict(figsize=(9, 7), cb_loc="bottom", shrink=1, l_loc=1),
                  RF01=dict(figsize=(6.2, 9), cb_loc="bottom", shrink=1, l_loc=2),
                  RF02=dict(figsize=(6.2, 9), cb_loc="bottom", shrink=1, l_loc=2),
                  RF03=dict(figsize=(6.2, 9), cb_loc="bottom", shrink=1, l_loc=3),
                  RF04=dict(figsize=(6.2, 7.9), cb_loc="bottom", shrink=1, l_loc=3),
                  RF05=dict(figsize=(9, 7.4), cb_loc="bottom", shrink=1, l_loc=3),
                  RF06=dict(figsize=(7.7, 9), cb_loc="bottom", shrink=1, l_loc=3),
                  RF07=dict(figsize=(9, 9), cb_loc="bottom", shrink=1, l_loc=3),
                  RF08=dict(figsize=(8.5, 9), cb_loc="bottom", shrink=1, l_loc=3),
                  RF09=dict(figsize=(7, 8.5), cb_loc="bottom", shrink=1, l_loc=3),
                  RF10=dict(figsize=(5.6, 9), cb_loc="bottom", shrink=1, l_loc=3),
                  RF11=dict(figsize=(7.8, 9), cb_loc="bottom", shrink=1, l_loc=3),
                  RF12=dict(figsize=(6.8, 9), cb_loc="bottom", shrink=1, l_loc=3),
                  RF13=dict(figsize=(7.8, 9), cb_loc="bottom", shrink=1, l_loc=3, extent=[-15, 27, 66.6, 88]),
                  RF14=dict(figsize=(5.6, 9), cb_loc="bottom", shrink=1, l_loc=3),
                  RF15=dict(figsize=(7.8, 9), cb_loc="bottom", shrink=1, l_loc=3),
                  RF16=dict(figsize=(9, 9), cb_loc="bottom", shrink=1, l_loc=3),
                  RF17=dict(figsize=(7, 9), cb_loc="bottom", shrink=1, l_loc=3, extent=[-15, 27, 66.6, 90]),
                  RF18=dict(figsize=(7, 9), cb_loc="bottom", shrink=1, l_loc=3, extent=[-15, 27, 66.6, 90]))


def plot_bahamas_flight_track(flight: str, **kwargs):
    """
    Plot a map of the flight track from BAHAMAS data with the location of HALO.

    Args:
        flight: Flight name (eg. Flight_20210707a)
        **kwargs: outpath (str): where to save plot (default: bahamas_dir/plots)

    Returns: Saves a png file

    """
    bahamas_dir = h.get_path("bahamas", flight)
    outpath = kwargs["outpath"] if "outpath" in kwargs else f"{bahamas_dir}/plots"
    h.make_dir(outpath)
    # read in bahamas data
    bahamas_file = [f for f in os.listdir(bahamas_dir) if f.endswith(".nc")][0]
    bahamas = reader.read_bahamas(f"{bahamas_dir}/{bahamas_file}")
    # select second airport for map plot according to flight
    airport = stop_over_locations[flight] if flight in stop_over_locations else None
    # select position and time data
    lon, lat, altitude, times = bahamas["IRS_LON"], bahamas["IRS_LAT"], bahamas["IRS_ALT"], bahamas["time"]
    # calculate flight duration
    flight_duration = pd.Timedelta((times[-1] - times[0]).values).to_pytimedelta()
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
    # plot a way point every 15 minutes = 9000 milliseconds with a time stamp next to it
    for long, lati, time_stamp in zip(lon[9000::9000], lat[9000::9000], times[9000::9000]):
        ax.annotate(time_stamp.dt.strftime("%H:%M").values, (long, lati), fontsize=10)
        ax.plot(long, lati, '.r', markersize=10)

    # get the coordinates for EDMO and add a label
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
    # write the flight duration in the lower left corner of the map
    ax.text(0, 0.01, f"Duration: {str(flight_duration)[:4]} (hr:min)", transform=ax.transAxes, fontsize=14)
    plt.tight_layout(pad=0.1)
    fig_name = f"{outpath}/{flight}_bahamas_track.png"
    plt.savefig(fig_name, dpi=100)
    log.info(f"Saved {fig_name}")
    plt.close()


def get_position(flight: str,
                 timestamp: Union[datetime.datetime, pd.Timestamp]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given the flight and the exact time, get HALOs position.

    Args:
        flight: Which flight to read in
        timestamp: exact time

    Returns: latitude (deg), longitude (deg), altitude (m)

    """
    bahamas_dir = h.get_path("bahamas", flight)
    bahamas_file = [f for f in os.listdir(bahamas_dir) if f.endswith(".nc")][0]
    bahamas_ds = reader.read_bahamas(f"{bahamas_dir}/{bahamas_file}")
    # convert given datetime to pd.Timestamp and remove any timezone information
    ts = pd.to_datetime(timestamp).tz_convert(None)
    bahamas_ds_sel = bahamas_ds.sel(TIME=ts)

    return bahamas_ds_sel.IRS_LAT.values, bahamas_ds_sel.IRS_LON.values, bahamas_ds_sel.IRS_ALT.values


def preprocess_bahamas(ds: xr.Dataset) -> xr.Dataset:
    """Preprocessing function for xarray.read_mfdataset()

    Returns: Dataset with dimension time

    """
    ds = ds.swap_dims({"tid": "TIME"})
    ds = ds.rename({"TIME": "time"})
    return ds

def calculate_distances(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculate geodesic distance between each aircraft time step

    Args:
        ds: BAHAMAS dataset

    Returns: New dataset with geodesic distances

    """
    loc1 = [(lat, lon) for lat, lon in
            zip(ds.IRS_LAT[:-1].to_numpy(), ds.IRS_LON[:-1].to_numpy())]
    loc2 = [(lat, lon) for lat, lon in
            zip(ds.IRS_LAT[1:].to_numpy(), ds.IRS_LON[1:].to_numpy())]
    distances = [distance(p1, p2).m for p1, p2 in
                 zip(tqdm(loc1, desc="Geodesic distance"), loc2)]
    distances.insert(0, 0)  # add zero as the first value to keep same length of values
    ds["distance"] = xr.DataArray(distances, dims="time",
                                  attrs=dict(units="m", long_name="geodesic distance",
                                             description="Distance between the previous and this point.\n"
                                                         "Does not work for locations in different altitudes!"))

    return ds

