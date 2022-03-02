#!\usr\bin\env python
"""Plotting script for map plots to be used in time lapse videos of GoPro

1. Plot a map of the flight track together with a marker for HALO

author: Johannes Röttenbacher
"""
if __name__ == "__main__":
    # %% import libraries
    from pylim import helpers as h
    from pylim.halo_ac3 import coordinates, take_offs_landings
    from pylim.bahamas import plot_props
    import pylim.reader as reader
    import os
    import matplotlib
    import matplotlib.pyplot as plt
    import pandas as pd
    import xarray as xr
    import cartopy.crs as ccrs
    import cartopy
    from tqdm import tqdm
    from joblib import Parallel, cpu_count, delayed
    import logging

    log = logging.getLogger("pylim")
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.WARNING)

    # %% set paths
    campaign = "halo-ac3"
    flight = "HALO-AC3_FD00_HALO_RF01_20220225"
    flight_str = flight[9:] if campaign == "halo-ac3" else flight
    flight_key = flight[19:] if campaign == "halo-ac3" else flight
    date = flight[-8:]
    use_smart_ins = True
    gopro_dir = h.get_path("gopro", campaign=campaign)
    if use_smart_ins:
        horipath = h.get_path("horidata", flight, campaign)
        file = [f for f in os.listdir(horipath) if f.endswith(".nc")][0]
        ins = xr.open_dataset(f"{horipath}/{file}")
        lat, lon, time = ins.lat, ins.lon, ins.time
        # INS data is not cut to actual flight time, do so here
        time_sel = (take_offs_landings[flight_key][0] <= time) & (time <= take_offs_landings[flight_key][1])
        time = time.where(time_sel, drop=True)

    else:
        bahamas_dir = h.get_path("bahamas", flight, campaign)
        # find bahamas file
        file = [f for f in os.listdir(bahamas_dir) if f.endswith(".nc")][0]
        bahamas = reader.read_bahamas(f"{bahamas_dir}/{file}")
        lon, lat, time = bahamas["IRS_LON"], bahamas["IRS_LAT"], bahamas.time

    # select second airport for map
    airport = "Longyearbyen"

    # %% find map extend
    pad = 2
    llcrnlat = lat.min(skipna=True) - pad
    llcrnlon = lon.min(skipna=True) - pad
    urcrnlat = lat.max(skipna=True) + pad
    urcrnlon = lon.max(skipna=True) + pad
    extent = [llcrnlon, urcrnlon, llcrnlat, urcrnlat]

    # %% select lon and lat values corresponding with the picture timestamps
    # get first and last bahamas time step
    first_ts, last_ts = pd.to_datetime(time[0].values), pd.to_datetime(time[-1].values)
    # make strings with the only the time from timestamps
    first_ts, last_ts = first_ts.strftime("%H:%M:%S"), last_ts.strftime("%H:%M:%S")
    # read timestamps
    timestamps = pd.read_csv(f"{gopro_dir}/{flight}_timestamps.csv", index_col="datetime", parse_dates=True)
    # select range of timestamps
    ts_sel = timestamps.between_time(first_ts, last_ts)
    # write out pictures used
    ts_sel.to_csv(f"{gopro_dir}/{flight}_timestamps_sel.csv", index_label="datetime")

    # %% select corresponding lat and lon values
    lon_sel = lon.sel(time=ts_sel.index)
    lat_sel = lat.sel(time=ts_sel.index)
    assert len(lon_sel) == len(lat_sel), "Lon and Lat are not of same lenght!"
    # %% plot on map


    def plot_flight_track(flight: str, campaign: str, lon, lat, extent: list, lon1: float, lat1: float, number: int, **kwargs):
        """
        Plot a map of the flight track from BAHAMAS or INS data with the location of HALO.
        Args:
            flight: Flight name (eg. Flight_20210707a for CIRRUS-HL or RF01_20220225 for HALO-AC3)
            campaign: Campaign keyword (eg. halo-ac3)
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
        bahamas_dir = h.get_path("bahamas", flight, campaign)
        outpath = kwargs["outpath"] if "outpath" in kwargs else f"{bahamas_dir}/plots/time_lapse"
        h.make_dir(outpath)
        airport = kwargs["airport"] if "airport" in kwargs else None
        font = {'weight': 'bold', 'size': 26}
        matplotlib.rc('font', **font)

        # get plot properties
        flight_key = flight[19:] if campaign == "halo-ac3" else flight
        props = plot_props[flight_key]
        projection = ccrs.NorthPolarStereo() if campaign == "halo-ac3" else ccrs.PlateCarree()
        fig, ax = plt.subplots(figsize=props["figsize"], subplot_kw={"projection": ccrs.PlateCarree()})
        ax.stock_img()
        ax.coastlines()
        ax.add_feature(cartopy.feature.BORDERS)
        ax.set_extent(extent)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
        gl.bottom_labels = False
        gl.left_labels = False
        # plot a way point every 30 minutes = 1800 seconds
        td = 1800  # for 1Hz data, for 10Hz data use 18000
        for long, lati, nr in zip(lon[td::td], lat[td::td], range(len(lat[td::td]))):
            ax.annotate(nr + 1, (long, lati), fontsize=16)
            ax.plot(long, lati, '.r', markersize=10)
        # plot an airplane marker for HALO
        ax.plot(lon1, lat1, c="k", marker="$\u2708$", markersize=28, label="HALO")
        # get the coordinates for EDMO and ad a label
        x_edmo, y_edmo = coordinates["EDMO"]
        ax.plot(x_edmo, y_edmo, 'ok')
        ax.text(x_edmo + 0.1, y_edmo + 0.1, "EDMO", fontsize=16)
        # plot a second airport label if given
        if airport is not None:
            x2, y2 = coordinates[airport]
            ax.text(x2 + 0.1, y2 + 0.1, airport, fontsize=16)
        # plot flight track and color by flight altitude
        points = ax.scatter(lon, lat, color="orange", s=10)
        # add the corresponding colorbar and decide whether to plot it horizontally or vertically
        ax.legend(loc=props["l_loc"])
        # plt.colorbar(points, ax=ax, pad=0.01, location=props["cb_loc"], label="Height (km)", shrink=props["shrink"])
        plt.tight_layout(pad=0.1)
        fig_name = f"{outpath}/{flight}_map_{number:04}.png"
        plt.savefig(fig_name, dpi=100)
        log.info(f"Saved {fig_name}")
        plt.close()


    # %% loop through timesteps
    lon1 = lon_sel[0]
    lat1 = lat_sel[0]
    number = ts_sel.number.values[0]
    plot_flight_track(flight, campaign, lon, lat, extent, lon1, lat1, number, airport=airport)
    # Parallel(n_jobs=cpu_count()-4)(delayed(plot_flight_track())
    #                                (flight_key, campaign, lon, lat, extent, lon1, lat1, number, airport=airport)
    #                                for lon1, lat1, number in zip(tqdm(lon_sel), lat_sel, ts_sel.number.values))
