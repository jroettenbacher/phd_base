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
    from ac3airborne.tools.get_amsr2_seaice import get_amsr2_seaice
    import os
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from matplotlib.markers import MarkerStyle
    import pandas as pd
    import xarray as xr
    import cartopy.crs as ccrs
    import cartopy
    from tqdm import tqdm
    from joblib import Parallel, cpu_count, delayed
    from typing import Tuple
    import logging

    log = logging.getLogger("pylim")
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.WARNING)

    # %% set paths
    campaign = "halo-ac3"
    flight = "HALO-AC3_20220313_HALO_RF03"
    date = flight[9:17]
    flight_key = flight[-4:] if campaign == "halo-ac3" else flight
    use_smart_ins = False
    gopro_dir = h.get_path("gopro", campaign=campaign)
    if use_smart_ins:
        horipath = h.get_path("horidata", flight, campaign)
        file = [f for f in os.listdir(horipath) if f.endswith(".nc")][0]
        ins = xr.open_dataset(f"{horipath}/{file}")
        lat, lon, time, heading = ins.lat, ins.lon, ins.time, ins.yaw
        # INS data is not cut to actual flight time, do so here
        time_sel = (take_offs_landings[flight_key][0] <= time) & (time <= take_offs_landings[flight_key][1])
        time = time.where(time_sel, drop=True)

    else:
        bahamas_dir = h.get_path("bahamas", flight, campaign)
        # find bahamas file
        file = [f for f in os.listdir(bahamas_dir) if f.endswith(".nc")][0]
        bahamas = reader.read_bahamas(f"{bahamas_dir}/{file}")
        # subsample to 1 Hz data to reduce computation time
        bahamas = bahamas.resample(dict(time="1s")).nearest()
        lon, lat, time, heading = bahamas["IRS_LON"], bahamas["IRS_LAT"], bahamas.time, bahamas["IRS_HDG"]

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
    heading_sel = heading.sel(time=ts_sel.index)
    # create a list of tuples
    halo_pos = [(lo, la, head) for lo, la, head in zip(lon_sel, lat_sel, heading_sel)]
    assert len(lon_sel) == len(lat_sel), "Lon and Lat are not of same lenght!"
    # %% plot on map


    def plot_flight_track(flight: str, campaign: str, lon, lat, extent: list, halo_pos: Tuple, number: int, **kwargs):
        """
        Plot a map of the flight track from BAHAMAS or INS data with the location of HALO.
        Args:
            flight: Flight name (eg. Flight_20210707a for CIRRUS-HL or HALO-AC3_20220225_HALO_RF00 for HALO-AC3)
            campaign: Campaign keyword (eg. halo-ac3)
            lon: array with longitude values for flight track
            lat: array with latitude values for flight track
            extent: list with map extent [llcrnlon, urcrnlon, llcrnlat, urcrnlat]
            halo_pos: Tuple with (lon, lat, heading) of HALO
            number: GoPro picture number corresponding with the BAHAMAS time step of the HALO position
            **kwargs:
                outpath (str): where to save plot
                airport (str): second airport to label on map

        Returns: Saves a png file

        """
        flight_key = flight[-4:] if campaign == "halo-ac3" else flight
        bahamas_dir = h.get_path("bahamas", flight, campaign)
        outpath = kwargs["outpath"] if "outpath" in kwargs else f"{bahamas_dir}/plots/time_lapse"
        h.make_dir(outpath)
        airport = kwargs["airport"] if "airport" in kwargs else None
        add_seaice = kwargs["add_seaice"] if "add_seaice" in kwargs else True
        font = {'weight': 'bold', 'size': 26}
        matplotlib.rc('font', **font)

        # get plot properties
        props = plot_props[flight_key]
        data_proj = ccrs.PlateCarree()
        projection = ccrs.NorthPolarStereo() if campaign == "halo-ac3" else data_proj
        fig, ax = plt.subplots(figsize=props["figsize"], subplot_kw={"projection": projection})

        if add_seaice:
            orig_map = plt.cm.get_cmap('Blues')  # getting the original colormap using cm.get_cmap() function
            reversed_map = orig_map.reversed()  # reversing the original colormap using reversed() function
            seaice = get_amsr2_seaice(f"{(pd.to_datetime(date) - pd.Timedelta(days=0)):%Y%m%d}")
            seaice = seaice.seaice
            ax.pcolormesh(seaice.lon, seaice.lat, seaice, transform=ccrs.PlateCarree(), cmap=reversed_map)
        else:
            ax.stock_img()

        ax.coastlines()
        ax.add_feature(cartopy.feature.BORDERS)
        ax.set_extent(extent, crs=data_proj)
        gl = ax.gridlines(crs=data_proj, draw_labels=True, x_inline=True)
        gl.bottom_labels = False
        gl.left_labels = False

        # plot flight track and color by flight altitude
        points = ax.scatter(lon, lat, color="orange", s=10, transform=data_proj)
        # add the corresponding colorbar and decide whether to plot it horizontally or vertically
        # plt.colorbar(points, ax=ax, pad=0.01, location=props["cb_loc"], label="Height (km)", shrink=props["shrink"])

        # plot a way point every 30 minutes = 1800 seconds
        td = 1800  # for 1Hz data, for 10Hz data use 18000
        for long, lati, nr in zip(lon[td::td], lat[td::td], range(len(lat[td::td]))):
            ax.text(long, lati, nr + 1, fontsize=16, transform=data_proj,
                    path_effects=[patheffects.withStroke(linewidth=1, foreground="white")])
            ax.plot(long, lati, '.r', markersize=10, transform=data_proj)

        # plot an airplane marker for HALO
        m = MarkerStyle("$\u2708$")
        m._transform.rotate_deg(halo_pos[2])
        ax.plot(halo_pos[0], halo_pos[1], c="k", marker=m, ls="", markersize=28, label="HALO", transform=data_proj)

        # get the coordinates for EDMO/Kiruna and ad a label
        x_kiruna, y_kiruna = coordinates["Kiruna"]
        ax.plot(x_kiruna, y_kiruna, 'ok', transform=data_proj)
        ax.text(x_kiruna - 2, y_kiruna - 0.2, "Kiruna", fontsize=16, transform=data_proj,
                path_effects=[patheffects.withStroke(linewidth=1, foreground="white")])
        # plot a second airport label if given
        if airport is not None:
            x2, y2 = coordinates[airport]
            ax.plot(x2, y2, 'ok', airport, transform=data_proj)
            ax.text(x2, y2, airport, fontsize=16, transform=data_proj,
                    path_effects=[patheffects.withStroke(linewidth=1, foreground="white")])

        ax.legend(loc=props["l_loc"])
        plt.tight_layout(pad=0.1)
        fig_name = f"{outpath}/{flight}_map_{number:04}.png"
        # plt.savefig(fig_name, dpi=100)
        # log.info(f"Saved {fig_name}")
        plt.show()
        plt.close()


    # %% loop through timesteps
    # halo_pos1 = halo_pos[0]
    # number = ts_sel.number.values[0]
    # plot_flight_track(flight, campaign, lon, lat, extent, halo_pos1, number, airport=airport)
    Parallel(n_jobs=cpu_count()-4)(delayed(plot_flight_track)(flight, campaign, lon, lat, extent, halo_pos1, number, airport=airport) for halo_pos1, number in zip(tqdm(halo_pos), ts_sel.number.values))
