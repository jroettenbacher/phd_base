#!/usr/bin/env python
"""Skript to create sea ice flag from AMSR2 and BAHAMAS data

*author*: Johannes Röttenbacher, Sophie
"""

from ac3airborne.tools.get_amsr2_seaice import get_amsr2_seaice
import os
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import patheffects
import pandas as pd
import cartopy
import cartopy.crs as ccrs

# %% functions


def read_bahamas(bahamas_path: str) -> xr.Dataset:
    """
    Reader function for netcdf BAHAMAS data as provided by DLR.

    Args:
        bahamas_path: full path of netcdf file

    Returns: xr.DataSet with BAHAMAS data and time as dimension

    """
    ds = xr.open_dataset(bahamas_path)
    ds = ds.swap_dims({"tid": "TIME"})
    ds = ds.rename({"TIME": "time"})

    return ds


def make_dir(folder: str) -> None:
    """
    Creates folder if it doesn't exist already.

    Args:
        folder: folder name or full path

    Returns: nothing, but creates a new folder if possible

    """
    try:
        os.makedirs(folder)
    except FileExistsError:
        pass


# %% user input
campaign = "halo-ac3"
date = "20220316"
flight_key = "RF06"
flight = f"HALO-AC3_{date}_HALO_{flight_key}"
add_seaice = True
savefig = True

# %% set plot properties and get some information from the dicts
defaults = dict(extent=[-15, 30, 68, 81], figsize=(5.5, 5.5), cb_loc="bottom", shrink=1, l_loc=1)
plot_props = dict(RF00=dict(figsize=(4, 4)),
                  RF01=dict(extent=[5, 25, 47, 69]),
                  RF02=dict(extent=[-15, 30, 68, 85]),
                  RF03=dict(extent=[-15, 30, 68, 90], figsize=(4.8, 5.5)),
                  RF04=dict(extent=[-15, 30, 68, 90], figsize=(4.8, 5.5)),
                  RF05=dict(figsize=(5.5, 4)),
                  RF06=dict(extent=[-15, 30, 68, 85]),
                  RF07=dict(extent=[-15, 30, 68, 82], figsize=(5.5, 4.8)),
                  RF08=dict(figsize=(5.5, 4.6)),
                  RF09=dict(figsize=(5.5, 4.5)),
                  RF10=dict(figsize=(5.2, 5.5), extent=[-15, 30, 68, 86]),
                  RF11=dict(figsize=(5.5, 4.5)),
                  RF12=dict(figsize=(5.5, 4.5)),
                  RF13=dict(figsize=(4.9, 5.5), extent=[-15, 30, 68, 87.5]),
                  RF14=dict(figsize=(4.7, 5.5), extent=[-15, 30, 68, 88]),
                  RF15=dict(figsize=(5.5, 4.5)),
                  RF16=dict(figsize=(5.5, 4.5)),
                  RF17=dict(figsize=(4.7, 5.5), extent=[-15, 30, 68, 90]),
                  RF18=dict(figsize=(4.5, 5.5), extent=[-15, 30, 68, 90]))

# coordinates for map plots (lon, lat)
coordinates = dict(EDMO=(11.28, 48.08), Keflavik=(-22.6307, 63.976), Kiruna=(20.336, 67.821), Santiago=(-8.418, 42.898),
                   Bergen=(5.218, 60.293), Torshavn=(-6.76, 62.01), Muenchen_Oberschleissheim=(11.55, 48.25),
                   Meiningen=(10.38, 50.56), Lerwick=(-1.18, 60.13), Ittoqqortoormiit=(-21.95, 70.48),
                   Tasiilaq=(-37.63, 65.60), Leipzig=(12.39, 51.33), Jülich=(6.37, 50.92),
                   Longyearbyen=(15.47, 78.25), Norderney=(7.15, 53.71))

x_kiruna, y_kiruna = coordinates["Kiruna"]
x_longyear, y_longyear = coordinates["Longyearbyen"]

# %% set paths and read in files
ql_path = "plots"  # change path for plots
make_dir(ql_path)
bahamas_path = f"E:/HALO-AC3/02_Flights/HALO-AC3_{date}_HALO_{flight_key}/BAHAMAS"  # input path for bahamas file
infile = f"QL_HALO-AC3_HALO_BAHAMAS_{date}_{flight_key}_v1.nc"
bahamas = read_bahamas(f"{bahamas_path}/{infile}")

# read in sea ice
seaice = get_amsr2_seaice(f"{pd.to_datetime(date):%Y%m%d}")
seaice = seaice.seaice

# %% plot map quicklook
orig_map = plt.cm.get_cmap('Blues')  # getting the original colormap using cm.get_cmap() function
reversed_map = orig_map.reversed()  # reversing the original colormap using reversed() function

# select position and time data
lon, lat, altitude, times = bahamas["IRS_LON"], bahamas["IRS_LAT"], bahamas["H"], bahamas["time"]
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

ax.coastlines()
ax.add_feature(cartopy.feature.BORDERS)
ax.set_extent(extent, crs=data_crs)
gl = ax.gridlines(crs=data_crs, draw_labels=True, x_inline=False, y_inline=False)
gl.bottom_labels = False
gl.left_labels = False

# add sea ice extent
if add_seaice:
    ax.pcolormesh(seaice.lon, seaice.lat, seaice, transform=data_crs, cmap=reversed_map)

# plot flight track
points = ax.scatter(lon, lat, s=1, c="orange", transform=data_crs)

# plot some place labels
# Kiruna
x_kiruna, y_kiruna = coordinates["Kiruna"]
ax.plot(x_kiruna, y_kiruna, '.', color="#117733", markersize=8, transform=data_crs)
ax.text(x_kiruna + 0.1, y_kiruna + 0.1, "Kiruna", fontsize=11, transform=data_crs)
# Longyearbyen
x_longyear, y_longyear = coordinates["Longyearbyen"]
ax.plot(x_longyear, y_longyear, '.', color="#117733", markersize=8, transform=data_crs)
ax.text(x_longyear + 0.1, y_longyear + 0.1, "Longyearbyen", fontsize=11, transform=data_crs,
        path_effects=[patheffects.withStroke(linewidth=0.5, foreground="white")])

# write the flight duration in the lower left corner of the map
ax.text(0, 0.01, f"Duration: {str(flight_duration)[:4]} (hr:min)", transform=ax.transAxes, fontsize=11, color="white",
        path_effects=[patheffects.withStroke(linewidth=0.5, foreground="black")])
plt.tight_layout(pad=0.1)
fig_name = f"{ql_path}/HALO-AC3_HALO_BAHAMAS-track_{date}_{flight_key}.png"
if savefig:
    plt.savefig(fig_name, dpi=300)
    print(f"Saved {fig_name}")
else:
    # pass
    plt.show()
# plt.close()

