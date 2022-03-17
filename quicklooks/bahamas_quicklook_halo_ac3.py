#!/usr/bin/env python
"""Produces the standard BAHAMAS quicklook for |haloac3|

Quicklooks:

- map with sea ice extent and flight track
- TBD: map with MODIS image and flight track
- Movement quicklook
- Meteo quicklook

**Input**:

- flight
- Path to files
- save figure flag

*author:* Johannes Röttenbacher
"""

if __name__ == "__main__":
    # %% import modules and set paths
    import os
    import xarray as xr
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from matplotlib import patheffects
    import cartopy
    import cartopy.crs as ccrs
    from osgeo import gdal, osr
    gdal.UseExceptions()
    import numpy as np
    import pandas as pd
    import datetime
    from metpy.units import units
    from metpy.constants import Rd, g
    from ac3airborne.tools.get_amsr2_seaice import get_amsr2_seaice
    from pylim import reader
    import rasterio
    from rasterio.plot import show

    # %% user input
    flight = "HALO-AC3_20220225_HALO_RF00"
    date = flight[9:17]
    flight_key = flight[-4:]
    savefig = False

    # %% set paths
    data_path = f"E:/HALO-AC3/02_Flights/{flight}"
    plot_path = f"E:/HALO-AC3/02_Flights/{flight}/quicklooks"
    bahamas_path = f"{data_path}/BAHAMAS"
    bahamas_file = [f for f in os.listdir(bahamas_path) if f.endswith("nc")][0]
    bahamas_filepath = os.path.join(bahamas_path, bahamas_file)
    dropsonde_path = f"{data_path}/dropsondes"
    dropsonde_files = [f"{dropsonde_path}/{f}" for f in os.listdir(dropsonde_path) if f.endswith("QC.nc")]
    # modis_path = f"{data_path}/satellite/MODIS/"
    # modis = reader.Modis()
    # modis.get_data("MOD021KM.A2022056.1135.061.2022056194352.hdf", "MOD03.A2022056.1135.061.2022056164637.hdf", path=modis_path)
    # modis.increase_brightness(4)
    # satellite_path = f"{data_path}/satellite"
    # sat_file = "snapshot-2022-02-25.tiff"
    # sat_image = os.path.join(satellite_path, sat_file)


# %% create some dictionaries
    default_extent = [-15, 30, 68, 81]
    default_fs = (4, 4)
    plot_props = dict(RF00=dict(extent=[5, 15, 47, 56], figsize=(5, 5), shrink=0.94, projection=ccrs.PlateCarree()),
                      RF01=dict(extent=[5, 15, 47, 70], figsize=default_fs, shrink=1),
                      RF02=dict(extent=[-15, 30, 68, 85], figsize=(5, 5), shrink=0.79))

    coordinates = dict(EDMO=(11.28, 48.08), Keflavik=(-22.6307, 63.976), Kiruna=(20.336, 67.821), Bergen=(5.218, 60.293),
                       Torshavn=(-6.76, 62.01), Muenchen_Oberschleissheim=(11.55, 48.25), Longyearbyen=(15.46, 78.25),
                       Meiningen=(10.38, 50.56), Lerwick=(-1.18, 60.13), Ittoqqortoormiit=(-21.95, 70.48),
                       Tasiilaq=(-37.63, 65.60))
    x_kiruna, y_kiruna = coordinates["Kiruna"]
    x_longyear, y_longyear = coordinates["Longyearbyen"]

# %% define functions


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


    def set_xticks_and_xlabels(ax: plt.axis, time_extend: datetime.timedelta) -> plt.axis:
        """This function sets the ticks and labels of the x-axis (only when the x-axis is time in UTC).

        Options:
            -   1 days > time_extend > 12 hours:     major ticks every 2 hours, minor ticks every  30 minutes
            -   12 hours > time_extend > 6 hours:     major ticks every 1 hours, minor ticks every  30 minutes
            -   6 hours > time_extend > 2 hour:     major ticks every hour, minor ticks every  15 minutes
            -   2 hours > time_extend > 15 min:     major ticks every 15 minutes, minor ticks every 5 minutes
            -   15 min > time_extend > 5 min:       major ticks every 15 minutes, minor ticks every 5 minutes
            -   else:                               major ticks every minute, minor ticks every 10 seconds

        Args:
            ax: axis in which the x-ticks and labels have to be set
            time_extend: time difference of t_end - t_start (format datetime.timedelta)

        Returns:
            ax - axis with new ticks and labels
        """

        if time_extend > datetime.timedelta(days=30):
            pass
        elif datetime.timedelta(hours=25) > time_extend >= datetime.timedelta(hours=6):
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 2)))
            ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 30)))
        elif datetime.timedelta(hours=12) > time_extend >= datetime.timedelta(hours=6):
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(0, 12, 1)))
            ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 30)))
        elif datetime.timedelta(hours=6) > time_extend >= datetime.timedelta(hours=2):
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=1))
            ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 15)))
        elif datetime.timedelta(hours=2) > time_extend >= datetime.timedelta(minutes=15):
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 15)))
            ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 5)))
        elif datetime.timedelta(minutes=15) > time_extend >= datetime.timedelta(minutes=5):
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 15)))
            ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 5)))
        else:
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 1)))
            ax.xaxis.set_minor_locator(matplotlib.dates.SecondLocator(interval=10))

        return ax


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


    # Values according to the 1976 U.S. Standard atmosphere [NOAA1976]_.
    # List of tuples (height, temperature, pressure, temperature gradient)
    _STANDARD_ATMOSPHERE = [
        (0 * units.km, 288.15 * units.K, 101325 * units.Pa, 0.0065 * units.K / units.m),
        (11 * units.km, 216.65 * units.K, 22632.1 * units.Pa, 0 * units.K / units.m),
        (20 * units.km, 216.65 * units.K, 5474.89 * units.Pa, -0.001 * units.K / units.m),
        (32 * units.km, 228.65 * units.K, 868.019 * units.Pa, -0.0028 * units.K / units.m),
        (47 * units.km, 270.65 * units.K, 110.906 * units.Pa, 0 * units.K / units.m),
        (51 * units.km, 270.65 * units.K, 66.9389 * units.Pa, 0.0028 * units.K / units.m),
        (71 * units.km, 214.65 * units.K, 3.95642 * units.Pa, float("NaN") * units.K / units.m)
    ]
    _HEIGHT, _TEMPERATURE, _PRESSURE, _TEMPERATURE_GRADIENT = 0, 1, 2, 3


    def pressure2flightlevel(pressure):
        r"""
        Conversion of pressure to height (hft) with
        hydrostatic equation, according to the profile of the 1976 U.S. Standard atmosphere [NOAA1976]_.
        Reference:
            H. Kraus, Die Atmosphaere der Erde, Springer, 2001, 470pp., Sections II.1.4. and II.6.1.2.
        Parameters
        ----------
        pressure : `pint.Quantity` or `xarray.DataArray`
            Atmospheric pressure
        Returns
        -------
        `pint.Quantity` or `xarray.DataArray`
            Corresponding height value(s) (hft)
        Notes
        -----
        .. math:: Z = \begin{cases}
                  Z_0 + \frac{T_0 - T_0 \cdot \exp\left(\frac{\Gamma \cdot R}{g\cdot\log(\frac{p}{p0})}\right)}{\Gamma}
                  &\Gamma \neq 0\\
                  Z_0 - \frac{R \cdot T_0}{g \cdot \log(\frac{p}{p_0})} &\text{else}
                  \end{cases}
        """
        is_array = hasattr(pressure.magnitude, "__len__")
        if not is_array:
            pressure = [pressure.magnitude] * pressure.units

        # Initialize the return array.
        z = np.full_like(pressure, np.nan) * units.hft

        for i, ((z0, t0, p0, gamma), (z1, t1, p1, _)) in enumerate(zip(_STANDARD_ATMOSPHERE[:-1],
                                                                       _STANDARD_ATMOSPHERE[1:])):
            p1 = _STANDARD_ATMOSPHERE[i + 1][_PRESSURE]
            indices = (pressure > p1) & (pressure <= p0)
            if i == 0:
                indices |= (pressure >= p0)
            if gamma != 0:
                z[indices] = z0 + 1. / gamma * (t0 - t0 * np.exp(gamma * Rd / g * np.log(pressure[indices] / p0)))
            else:
                z[indices] = z0 - (Rd * t0) / g * np.log(pressure[indices] / p0)

        if np.isnan(z).any():
            raise ValueError("flight level to pressure conversion not "
                             "implemented for z > 71km")

        return z if is_array else z[0]


# %% read in data
    bahamas = read_bahamas(bahamas_filepath)
    seaice = get_amsr2_seaice(date)
    seaice = seaice.seaice
    dropsondes = dict()
    for i in range(len(dropsonde_files)):
        dropsondes[f"{i}"] = xr.open_dataset(dropsonde_files[i])

# %% create dataframe with dropsonde launch times and locations
    launch_times = [pd.to_datetime(dropsondes[var].time[-1].values) for var in dropsondes]
    longitudes = [dropsondes[var].lon.min(skipna=True).values for var in dropsondes]
    latitudes = [dropsondes[var].lat.min(skipna=True).values for var in dropsondes]
    dropsonde_df = pd.DataFrame({"launch_time": launch_times, "lon": longitudes, "lat": latitudes})

# %% plot sea ice conc + flight track
    orig_map = plt.cm.get_cmap('Blues')  # getting the original colormap using cm.get_cmap() function
    reversed_map = orig_map.reversed()  # reversing the original colormap using reversed() function

    # select position and time data
    lon, lat, altitude, times = bahamas["IRS_LON"], bahamas["IRS_LAT"], bahamas["IRS_ALT"], bahamas["time"]
    # calculate flight duration
    flight_duration = pd.Timedelta((times[-1] - times[0]).values).to_pytimedelta()
    # set plot properties
    props = plot_props[flight_key]
    projection = props["projection"] if "projection" in props else ccrs.NorthPolarStereo()
    extent = props["extent"]
    # set plotting options
    plt.rcdefaults()
    font = {'size': 8}
    matplotlib.rc('font', **font)

    # start plotting
    fig, ax = plt.subplots(figsize=props["figsize"], subplot_kw={"projection": projection})

    # add general map features
    ax.stock_img()
    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, x_inline=False, y_inline=False)
    gl.bottom_labels = False
    gl.left_labels = False

    # add sea ice extent
    ax.pcolormesh(seaice.lon, seaice.lat, seaice, transform=ccrs.PlateCarree(), cmap=reversed_map)

    # plot flight track and color by relative humidity
    points = ax.scatter(lon, lat, c=bahamas.RELHUM, cmap="YlGn", norm=plt.Normalize(70, 110), s=4,
                        transform=ccrs.PlateCarree())
    # add the corresponding colorbar and decide whether to plot it horizontally or vertically
    plt.colorbar(points, ax=ax, pad=0, location="bottom", label="Relative Humidity (%)", shrink=props["shrink"])

    # plot a marker for each dropsonde together with the launch time
    for i in range(dropsonde_df.shape[0]):
        df = dropsonde_df.iloc[i]
        ax.annotate(f"{df['launch_time']:%H:%M}", (df.lon, df.lat), fontsize=8)
        ax.plot(df.lon, df.lat, "x", color="#CC6677", markersize=6, label="Dropsonde")

    # plot location of Kiruna and Longyearbyen and add a label
    ax.plot(x_kiruna, y_kiruna, '.r', transform=ccrs.PlateCarree())
    ax.text(x_kiruna + 0.1, y_kiruna + 0.1, "Kiruna", fontsize=10, transform=ccrs.PlateCarree())
    ax.plot(x_longyear, y_longyear, '.r', transform=ccrs.PlateCarree())
    ax.text(x_longyear + 0.1, y_longyear + 0.1, "Longyearbyen", fontsize=10, transform=ccrs.PlateCarree())

    # add wind barbs
    # increment = 5000
    # lat, lon, u, v = lat[0::increment], lon[0::increment], bahamas.U[::increment] * 1.94384449, bahamas.V[::increment] * 1.94384449
    # ax.barbs(lon, lat, u, v, length=6, transform=ccrs.PlateCarree())

    # write the flight duration in the lower left corner of the map
    ax.text(0, 0.01, f"Duration: {str(flight_duration)[:4]} (hr:min)", transform=ax.transAxes, fontsize=10, color="white", path_effects=[patheffects.withStroke(linewidth=1, foreground="black")])

    # add legend for dropsondes
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[0]], [labels[0]], loc=1)

    plt.tight_layout(pad=0.1)
    fig_name = f"{plot_path}/HALO-AC3_HALO_BAHAMAS_track_ql_{date}_{flight_key}.png"
    if savefig:
        plt.savefig(fig_name, dpi=100)
        print(f"Saved {fig_name}")
    else:
        plt.show()
    plt.close()

# %% plot satellite image with track
    #     img = rasterio.open(sat_image)
    #     # extent = [llcrnlon, urcrnlon, llcrnlat, urcrnlat]
    #     fig, ax = plt.subplots(figsize=(11, 9), subplot_kw={"projection": ccrs.NorthPolarStereo()})
    #     show(img, ax=ax, transform=ccrs.PlateCarree())
    #     # ax.stock_img()
    #     # img = ax.imshow(data[:3, :, :].transpose((1, 2, 0)), origin='upper')
    #     # ax.imshow(data[:3, :, :].transpose((1, 2, 0)))
    #     # ax.coastlines(linewidth=3)
    #     ax.add_feature(cartopy.feature.COASTLINE, linewidth=2)
    #     # ax.set_extent(extent, crs=ccrs.Geodetic())
    #     # plot flight track
    #     # points = ax.plot(lon, lat, c="orange", linewidth=6, transform=ccrs.PlateCarree())
    #     gl = ax.gridlines(transform=ccrs.PlateCarree(), draw_labels=True)
    #     plt.show()
    #     plt.close()

# %% plot satellite image with track
    #     projection = ccrs.NorthPolarStereo(central_longitude=7.981409)
    #     # extent = [llcrnlon, urcrnlon, llcrnlat, urcrnlat]
    #     extent = [modis.lon.min(), modis.lon.max(), modis.lat.min(), modis.lat.max()]
    #     central_lon = (extent[1]+extent[0])/2
    #     central_lat = (extent[2]+extent[3])/2
    #     proj = ccrs.NorthPolarStereo(central_longitude=central_lon)
    #     data_crs = ccrs.PlateCarree()
    #     fig, ax = plt.subplots(figsize=(11, 9), subplot_kw={"projection": proj})
    #     # ax.imshow(modis.rgb, extent=extent, origin="lower", transform=ccrs.PlateCarree())
    #     # What you put in for the image doesn't matter because of the color mapping
    #     rgb = 0.299 * modis.rgb[:, :, 0] + 0.587*modis.rgb[:, :, 1] + 0.114* modis.rgb[:, :, 2]
    #     ax.pcolormesh(modis.lon, modis.lat, rgb, transform=ccrs.PlateCarree(), cmap="Blues")
    #     # ax.stock_img()
    #     # img = ax.imshow(data[:3, :, :].transpose((1, 2, 0)), origin='upper')
    #     # ax.imshow(data[:3, :, :].transpose((1, 2, 0)))
    #     # ax.coastlines(linewidth=3)
    #     ax.add_feature(cartopy.feature.COASTLINE, linewidth=3)
    #     # ax.set_extent(extent)
    #     gl = ax.gridlines(transform=ccrs.PlateCarree(), draw_labels=True)
    #     gl.bottom_labels = False
    #     gl.left_labels = False
    #
    #     # plot flight track
    #     # points = ax.plot(lon, lat, c="orange", linewidth=6, transform=ccrs.PlateCarree())
    #
    #     # plot a way point every 15 minutes = 9000 seconds with a time stamp next to it
    #     # for long, lati, time_stamp in zip(lon[9000::9000], lat[9000::9000], times[9000::9000]):
    #     #     ax.annotate(time_stamp.dt.strftime("%H:%M").values, (long, lati), fontsize=16,
    #     #                 path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])
    #     #     ax.plot(long, lati, '.k', markersize=10)
    #
    #     # plot points with labels and white line around text
    #     # ax.plot(x_kiruna, y_kiruna, 'ok')
    #     # ax.text(x_kiruna + 0.1, y_kiruna + 0.1, "EDMO", fontsize=22,
    #     #         path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])
    #     # ax.plot(x_longyear, y_longyear, 'ok')
    #     # ax.text(x_longyear + 0.1, y_longyear + 0.1, "Longyearbyen", fontsize=22,
    #     #         path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])
    #
    #     # ax.legend(loc=3, fontsize=18, markerscale=6)
    #     # plt.tight_layout(pad=0.1)
    #     plt.show()
    #     # fig_name = f"{plot_path}/{flight}_bahamas_track_satellite.png"
    #     # plt.savefig(fig_name, dpi=100)
    #     # print(f"Saved {fig_name}")
    #     plt.close()

    # %% plot bahamas movement quicklook
    plt.rcdefaults()
    cb_colors = ["#6699CC", "#117733", "#CC6677", "#DDCC77", "#D55E00", "#332288"]
    x = bahamas.time
    timedelta = pd.to_datetime(x[-1].values) - pd.to_datetime(x[0].values)
    plt.rcParams['font.size'] = 8
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(15/2.54, 10/2.54))
    # first row
    ax = axs[0]
    ax.plot(x, bahamas.IRS_GS, color=cb_colors[0], label="Ground Speed")
    ax.set_ylim(50, 250)
    ax.set_ylabel("Speed (m$\,$s$^{-1}$)")
    ax.plot(x, bahamas.TAS, color=cb_colors[1], label="True Air Speed")
    # legend for first row
    ax.legend(ncol=2, loc=8)

    # second row
    ax = axs[1]
    ax.plot(x, bahamas.IRS_PHI, color=cb_colors[2], label="Roll")
    ax.plot(x, bahamas.IRS_THE, color=cb_colors[3], label="Pitch")
    ax.set_ylim(-5, 5)
    ax.set_ylabel("Attitude Angle \n(deg)")
    ax.legend(ncol=2, loc=8)

    # third row
    ax = axs[2]
    ax.plot(x, bahamas.IRS_HDG, color=cb_colors[4])
    ax.text(0, 90, "East", va="bottom", ha="left", fontsize=6, transform=ax.get_yaxis_transform())
    ax.text(0, 180, "South", va="bottom", ha="left", fontsize=6, transform=ax.get_yaxis_transform())
    ax.text(0, 270, "West", va="bottom", ha="left", fontsize=6, transform=ax.get_yaxis_transform())
    ax.set_ylim(0, 360)
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_ylabel("True Heading \n(deg)")


    # common settings over all rows
    for ax in axs:
        set_xticks_and_xlabels(ax, timedelta)
        ax.grid()
        ax.set_xlabel("")

    axs[2].set_xlabel("Time (UTC)")

    plt.tight_layout(pad=0.35)
    fig_name = f"{plot_path}/HALO-AC3_HALO_BAHAMAS_movement_ql_{date}_{flight_key}.png"
    if savefig:
        plt.savefig(fig_name, dpi=200)
        print(f"Saved {fig_name}")
    else:
        plt.show()
    plt.close()

# %% calculate flight level
    flv = pressure2flightlevel(bahamas.PS.values * units.hPa)
    p_bot, p_top = 101315, 12045
    flv_limits = pressure2flightlevel([p_bot, p_top] * units.Pa)
    _pres_maj = np.concatenate([np.arange(top * 10, top, -top) for top in (10000, 1000, 100, 10)] + [[10]])
    _pres_min = np.concatenate([np.arange(top * 10, top, -top // 10) for top in (10000, 1000, 100, 10)] + [[10]])

# %% plot bahamas meteo quicklook 2
    ylabels = ["Static Air\nTemperature (K)", "Relative \nHumidity (%)", "Static \nPressure (hPa)"]
    x = bahamas.time  # prepare x axis
    plt.rcParams['font.size'] = 8
    nrows = 2
    fig, axs = plt.subplots(nrows=nrows, figsize=(15/2.54, 10/2.54))
    # first row
    ax = axs[0]
    ax.plot(x, bahamas.TS, color=cb_colors[2], label="Temperature")  # static temperature
    ax.axhline(y=235, color="r", linestyle="--", label="$235\,$K")  # homogeneous freezing threshold
    ax.set_ylim(175, 300)
    ax.set_yticks([200, 225, 250, 275, 300])
    # secondary y axis
    ax2 = ax.twinx()
    ax2.plot(x, bahamas.IRS_ALT / 1000, color=cb_colors[0], label="Altitude")  # add altitude in km
    ax2.set_ylabel("Altitude (km)")
    ax2.set_yticks([0, 2, 4, 6, 8, 10, 12])
    # legend for first row
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, ncol=3, loc=8)

    # second row
    ax = axs[1]
    ax.plot(x, bahamas.RELHUM, color="#117733")
    ax.set_ylim(0, 130)
    ax.set_yticks([0, 25, 50, 75, 100, 125], minor=False)

    # third row
    # ax = axs[2]
    # ax.plot(x, bahamas.PS, label="Pressure")
    # ax.invert_yaxis()
    # ax.set_yscale("log")
    # major_ticks = [1000, 900, 800, 700, 600, 500, 400, 300, 200]
    # minor_ticks = _pres_min[(_pres_min <= p_bot) & (_pres_min >= p_top)]
    # labels = [1000, "", "", 700, "", 500, 400, 300, 200]
    # Draw ticks and tick labels.
    # ax.set_yticks(minor_ticks/100, [], minor=True)
    # ax.set_yticks(major_ticks, minor=False)
    # ax.set_yticklabels(labels, minor=False, fontsize=8)
    # ax.set_ylim(p_bot / 100, p_top / 100)  # Set axis limits7
    # secondary yaxis
    # ax6 = ax.twinx()
    # ax6.plot(x, flv, color="#D55E00", label="Flight Level")
    # ax6.axhline(300, color="k", linewidth=1)
    # ax6.text(0.01, 305, "FL 300", va="bottom", ha="left", fontsize=6, transform=ax6.get_yaxis_transform())
    # ax6.axhline(400, color="k", linewidth=1)
    # ax6.text(0.01, 405, "FL 400", va="bottom", ha="left", fontsize=6, transform=ax6.get_yaxis_transform())
    # ax6.set_ylim(flv_limits)
    # ax6.set_ylabel("Flight Level (hft)")
    # ax6.yaxis.set_major_locator(ticker.MaxNLocator(5, steps=[10]))
    # ax6.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    # legend for third row
    # handles, labels = ax.get_legend_handles_labels()
    # handles2, labels2 = ax6.get_legend_handles_labels()
    # ax.legend(handles + handles2, labels + labels2, ncol=2, loc=8)

    # common settings over all rows
    for ax, ylabel in zip(axs, ylabels):
        ax.set_ylabel(ylabel)
        ax.grid()
        set_xticks_and_xlabels(ax, timedelta)

    axs[nrows-1].set_xlabel("Time (UTC)")
    for ax in axs[0:nrows-1]:
        ax.set_xlabel("")
        ax.set_xticklabels("")

    plt.tight_layout(pad=0.3)
    fig_name = f"{plot_path}/HALO-AC3_HALO_BAHAMAS_meteo_ql_{date}_{flight_key}.png"
    if savefig:
        plt.savefig(fig_name, dpi=200)
        print(f"Saved {fig_name}")
    else:
        plt.show()
    plt.close()
