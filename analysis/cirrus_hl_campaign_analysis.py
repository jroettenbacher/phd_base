#!/usr/bin/env python
"""Analysis script for statistics about the whole campaign

*author*: Johannes Röttenbacher
"""
if __name__ == "__main--":
    # %% module import
    from pylim.cirrus_hl import coordinates
    from pylim.helpers import get_path
    import xarray as xr
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    import cartopy
    import cartopy.crs as ccrs
    import logging
    log = logging.getLogger(__name__)
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)

    # %% define functions


    def preprocess_bahamas(ds: xr.Dataset) -> xr.Dataset:
        # swap dimensions
        ds = ds.swap_dims({"tid": "TIME"})
        ds = ds.rename({"TIME": "time"})
        return ds


    # %% set up paths
    bahamas = "BAHAMAS"
    path = get_path('all', instrument=bahamas)
    plot_path = get_path('plot')

    # %% read in bahamas data
    ds = xr.open_mfdataset(f"{path}/*.nc", preprocess=preprocess_bahamas)

    # %% find max and min lat and lon
    lat_min = ds.IRS_LAT.min().compute().values
    lat_max = ds.IRS_LAT.max().compute().values
    lon_min = ds.IRS_LON.min().compute().values
    lon_max = ds.IRS_LON.max().compute().values
    alt_max = ds.IRS_ALT.max().compute().values
    print(f"Minumum Latitude: {lat_min}°N\nMaximum Latitude: {lat_max}°N\nMinumum Longitude: {lon_min}°E"
          f"\nMaximum Longitude: {lon_max}°E\nMaximum Altitutde: {alt_max}m")

    # %% prepare radiosonde dictionary and remove non radiosonde coordinates
    rs_coordinates = coordinates
    rs_coordinates.pop("EDMO")
    rs_coordinates.pop("Santiago")

    # %% plot radiosonde stations
    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.stock_img()
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS)
    ax.set_extent([-40, 30, 35, 80])
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    gl.bottom_labels = False
    gl.left_labels = False
    # plot each radiosonde station
    for station in rs_coordinates:
        coords = rs_coordinates[station]
        ax.annotate(station, coords, fontsize=12,
                    path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])
        ax.plot(coords[0], coords[1], '.r', markersize=10)

    ax.set_title("CIRRUS-HL - Radiosonde staions close to flight paths")
    plt.tight_layout()
    # plt.show()
    figname = f"{plot_path}/CIRRUS-HL_Radiosonde_stations.png"
    plt.savefig(figname, dpi=100)
    log.info(f"Saved {figname}")
    plt.close()
