#!/usr/bin/env python
"""Analysis for LIM Journal contribution 2021

- average spectra over height during staircase pattern

*author*: Johannes Röttenbacher
"""

if __name__ == "__main__":
# %% module import
    import pylim.helpers as h
    from pylim import reader
    from pylim.cirrus_hl import coordinates
    import numpy as np
    import os
    from matplotlib import patheffects
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import cartopy
    import cartopy.crs as ccrs
    import xarray as xr
    import pandas as pd
    import rasterio
    from rasterio.plot import show
    import logging

    log = logging.getLogger("pylim")
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)

# %% set up paths and meta data
    flight = "Flight_20210629a"
    campaign = "cirrus-hl"
    smart_dir = h.get_path("calibrated", flight, campaign)
    bahamas_dir = h.get_path("bahamas", flight, campaign)
    sat_dir = h.get_path("satellite", flight, campaign)
    sat_file = [f for f in os.listdir(sat_dir) if "MODIS" in f][0]
    sat_image = os.path.join(sat_dir, sat_file)
    plot_path = f"{h.get_path('plot')}/{flight}"
    start_dt = pd.Timestamp(2021, 6, 29, 9, 42)
    end_dt = pd.Timestamp(2021, 6, 29, 12, 10)
    cm = 1 / 2.54

# %% read in data
    fdw_vnir = xr.open_dataset(f"{smart_dir}/cirrus-hl_SMART_Fdw_VNIR_2021_06_29.nc")
    fdw_swir = xr.open_dataset(f"{smart_dir}/cirrus-hl_SMART_Fdw_SWIR_2021_06_29.nc")
    fup_vnir = xr.open_dataset(f"{smart_dir}/cirrus-hl_SMART_Fup_VNIR_2021_06_29.nc")
    fup_swir = xr.open_dataset(f"{smart_dir}/cirrus-hl_SMART_Fup_SWIR_2021_06_29.nc")
    # filter VNIR data
    fdw_vnir = fdw_vnir.sel(wavelength=slice(420, 950))
    fup_vnir = fup_vnir.sel(wavelength=slice(420, 950))
    wavelengths_to_drop = fup_vnir.wavelength[fup_vnir.wavelength > 850]
    # filter swir data
    # fup_swir = fup_swir.sel(wavelength=slice(950, 2200))
    # merge VNIR and SWIR channel
    smart_fdw = xr.merge([fdw_vnir.Fdw, fdw_swir.Fdw])
    smart_fup = xr.merge([fup_vnir.Fup, fup_swir.Fup])
    mask = smart_fup.wavelength.isin(wavelengths_to_drop)
    smart_fup = smart_fup.where(~mask, np.nan)
    # smart_fup = xr.combine_by_coords([fup_vnir.Fup, fup_swir.Fup])

    bahamas = reader.read_bahamas(f"{bahamas_dir}/CIRRUSHL_F05_20210629a_ADLR_BAHAMAS_v1.nc")

    sat_ds = rasterio.open(sat_image)

# %% select only relevant times
    smart_fdw = smart_fdw.sel(time=slice(start_dt, end_dt))
    smart_fup = smart_fup.sel(time=slice(start_dt, end_dt))
    bahamas_sel = bahamas.sel(time=slice(start_dt, end_dt))

# %% define staircase sections according to flight levels
    altitude = np.round(bahamas_sel.H, 1)  # rounded to 1 decimal to avoid tiny differences
    idx = np.where(np.diff(altitude) == 0)  # find times with constant altitude
    # draw a quicklook
    altitude[idx].plot(marker="x")
    plt.show()
    plt.close()
    # select times with constant altitude
    times = bahamas_sel.time[idx]
    # find indices where the difference to the next timestep is greater 1 second -> change of altitude
    ids = np.argwhere((np.diff(times) / 10**9).astype(float) > 1)
    ids2 = ids + 1  # add 1 to get the indices at the start of the next section
    ids = np.insert(ids, 0, 0)  # add the start of the first section
    ids = np.append(ids, ids2)  # combine all indices
    ids.sort()  # sort them
    ids = np.delete(ids, [7, 8, 9, 10, -1])  # delete doubles
    times_sel = times[ids]  # select only the start and end times
    # get start and end times
    start_dts = times_sel[::2]
    end_dts = times_sel[1::2]
    # export times for Anna
    start_dts.name = "start"
    end_dts.name = "end"
    start_dts.to_netcdf("start_dts.nc")
    end_dts.to_netcdf("end_dts.nc")

# %% convert to km
    altitude = altitude / 1000

# %% plot sections and altitudes
    h.set_cb_friendly_colors()
    plt.rc("font", family="serif")
    fig, ax = plt.subplots()
    ax.plot(altitude.time, altitude)
    ax.plot(start_dts, altitude.sel(time=start_dts), ls="", marker="x", label="Start times", color="#CC6677")
    ax.plot(end_dts, altitude.sel(time=end_dts), ls="", marker="x", label="End times", color="#DDCC77")
    for i, dt in enumerate(start_dts):
        ax.annotate(f"Section {i+1}", (dt, (altitude.sel(time=dt) + 0.1)))

    ax.set_ylim(8, 12.6)
    ax.yaxis.set_major_locator(MaxNLocator(8, steps=[5], min_n_ticks=8))
    h.set_xticks_and_xlabels(ax, pd.to_timedelta((times[-1] - times[0]).values))
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Altitude (km)")
    ax.grid()
    ax.legend()
    # plt.show()
    plt.savefig(f"{plot_path}/{campaign.swapcase()}_BAHAMAS_altitude_staircase_{flight}.png", dpi=300)
    plt.close()

# %% check altitude differences between start and end of sections
    altitude.sel(time=start_dts).values - altitude.sel(time=end_dts).values  # differences between start and end of section

# %% select SMART Fdw data according to altitude and take averages over height
    sections = dict()
    for i, (st, et) in enumerate(zip(start_dts, end_dts)):
        sections[f"mean_spectra_{i}"] = smart_fdw.sel(time=slice(st, et)).mean(dim="time")

    h.set_cb_friendly_colors()
    plt.rc("font", family="serif")
    labels = ["Section 1 (FL260, 8.3$\,$km)", "Section 2 (FL280, 8.7$\,$km)", "Section 3 (FL300, 9.3$\,$km)",
              "Section 4 (FL320, 10$\,$km)", "Section 5 (FL340, 10.6$\,$km)", "Section 6 (FL360, 11.2$\,$km)",
              "Section 7 (FL390, 12.2$\,$km)"]
    fig, ax = plt.subplots(figsize=(15*cm, 8*cm))
    for section, label in zip(sections, labels):
        fdw = sections[section].Fdw
        fdw.plot(ax=ax, label=label)
        # for x_, y_, label in zip(fdw.wavelength.values, fdw.values, range(len(fdw))):
        #     if label > 700 and label < 790:
        #         plt.annotate(label, (x_, y_))

    ax.legend()
    ax.set_ylim(0, 1.35)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Downward Irradiance (W$\\,$m$^{-2}\\,$nm$^{-1}$)")
    ax.grid()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{plot_path}/{campaign.swapcase()}_SMART_Fdw_staircase_spectra_{flight}.png", dpi=300)
    plt.close()

# %% select SMART Fup data according to altitude and take averages over height
    sections = dict()
    for i, (st, et) in enumerate(zip(start_dts, end_dts)):
        sections[f"mean_spectra_{i}"] = smart_fup.sel(time=slice(st, et)).mean(dim="time")

    h.set_cb_friendly_colors()
    plt.rc("font", family="serif")
    labels = ["Section 1 (FL260, 8.3$\,$km)", "Section 2 (FL280, 8.7$\,$km)", "Section 3 (FL300, 9.3$\,$km)",
              "Section 4 (FL320, 10$\,$km)", "Section 5 (FL340, 10.6$\,$km)", "Section 6 (FL360, 11.2$\,$km)",
              "Section 7 (FL390, 12.2$\,$km)"]
    fig, ax = plt.subplots(figsize=(15*cm, 8*cm))
    for section, label in zip(sections, labels):
        fup = sections[section].Fup
        fup.plot(ax=ax, label=label)
        # for x_, y_, label in zip(fdw.wavelength.values, fdw.values, range(len(fdw))):
        #     if label > 700 and label < 790:
        #         plt.annotate(label, (x_, y_))

    # ax.legend()
    ax.set_ylim(0, 1.35)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Upward Irradiance (W$\\,$m$^{-2}\\,$nm$^{-1}$)")
    ax.grid()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{plot_path}/{campaign.swapcase()}_SMART_Fup_staircase_spectra_{flight}.png", dpi=300)
    plt.close()

# %% plot every single spectra
    # for i in range(fup_swir.dims["time"]):
    #     date_str = pd.to_datetime(str(fup_swir.time[i].values))
    #     date_str = date_str.strftime('%Y%m%d_%H%M%S')
    #     fup_swir.Fup.isel(time=i).plot()
    #     plt.savefig(f"{plot_path}/spectra/SMART_Fup_{date_str}UTC.png")
    #     plt.close()

# %% set plotting options for map plot
    lon, lat, altitude, times = bahamas["IRS_LON"], bahamas["IRS_LAT"], bahamas["IRS_ALT"], bahamas["time"]
    pad = 2
    llcrnlat = lat.min(skipna=True) - pad
    llcrnlon = lon.min(skipna=True) - pad
    urcrnlat = lat.max(skipna=True) + pad
    urcrnlon = lon.max(skipna=True) + pad
    extent = [llcrnlon, urcrnlon, llcrnlat, urcrnlat]
    edmo = coordinates["EDMO"]
    torshavn = coordinates["Torshavn"]
    bergen = coordinates["Bergen"]

# %% plot bahamas map with highlighted below and above cloud sections and sat image
    h.set_cb_friendly_colors()
    plt.rc("font", family="serif")
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    show(sat_ds, ax=ax)
    ax.coastlines(linewidth=1)
    ax.add_feature(cartopy.feature.BORDERS, linewidth=1)
    ax.set_extent(extent)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    gl.bottom_labels = False
    gl.left_labels = False

    # plot flight track
    points = ax.plot(lon, lat, c="orange", linewidth=4)
    # plot staircase section
    for label, st, et in zip(labels, start_dts, end_dts):
        lon_sec = lon.sel(time=slice(st, et))
        lat_sec = lat.sel(time=slice(st, et))
        ax.plot(lon_sec, lat_sec, linewidth=4, label=label)
        ax.annotate(st.dt.strftime("%H:%M").values, (lon_sec[0], lat_sec[0]), fontsize=8,
                    path_effects=[patheffects.withStroke(linewidth=1, foreground="w")])
        ax.plot(lon_sec[0], lat_sec[0], marker=8, c="#CC6677", markersize=8)
        ax.annotate(et.dt.strftime("%H:%M").values, (lon_sec[-1], lat_sec[-1]), xytext=(0, -9), textcoords='offset points',
                    fontsize=8, path_effects=[patheffects.withStroke(linewidth=1, foreground="w")])
        ax.plot(lon_sec[-1], lat_sec[-1], marker=9, c="#DDCC77", markersize=8)

    # plot a way point every 15 minutes = 9000 seconds with a time stamp next to it
    # for long, lati, time_stamp in zip(lon[9000::9000], lat[9000::9000], times[9000::9000]):


    # plot points with labels and white line around text
    ax.plot(edmo[0], edmo[1], 'ok')
    ax.text(edmo[0] + 0.1, edmo[1] + 0.1, "EDMO", fontsize=10,
            path_effects=[patheffects.withStroke(linewidth=1, foreground="w")])
    ax.plot(bergen[0], bergen[1], 'ok')
    ax.text(bergen[0] + 0.1, bergen[1] + 0.1, "Bergen", fontsize=10,
            path_effects=[patheffects.withStroke(linewidth=1, foreground="w")])
    ax.plot(torshavn[0], torshavn[1], 'ok')
    ax.text(torshavn[0] + 0.1, torshavn[1] + 0.1, "Torshavn", fontsize=10,
            path_effects=[patheffects.withStroke(linewidth=1, foreground="w")])

    # make a legend entry for the start and end times
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0, plt.plot([0], ls="", marker=8, markersize=3, color="#CC6677", label="Start Times (UTC)")[0])
    handles.insert(1, plt.plot([0], ls="", marker=9, markersize=3, color="#DDCC77", label="End Times (UTC)")[0])
    ax.legend(handles=handles, loc=3, fontsize=10, markerscale=4)
    plt.tight_layout(pad=0.1)
    fig_name = f"{plot_path}/{campaign.swapcase()}_BAHAMAS_staircase_track_{flight}.png"
    # plt.show()
    plt.savefig(fig_name, dpi=300)
    log.info(f"Saved {fig_name}")
    plt.close()
# %% plot all flight tracks with the natural earth as background
    bahamas_all_dir = h.get_path("all", campaign=campaign, instrument="bahamas")
    bahamas_all_files = [f"{bahamas_all_dir}/{file}" for file in os.listdir(bahamas_all_dir)]
    extent = [-35, 30, 35, 80]
    edmo = coordinates["EDMO"]
    plt.rc("font", family="serif")

    fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
    ax.coastlines(linewidth=1)
    ax.add_feature(cartopy.feature.BORDERS, linewidth=1)
    ax.set_extent(extent)
    ax.background_img(name='BM', resolution='high')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    gl.bottom_labels = False
    gl.left_labels = False

    # plot flight tracks
    cmap = plt.get_cmap("hsv")
    for i in range(len(bahamas_all_files)):
        bahamas_tmp = reader.read_bahamas(bahamas_all_files[i])
        lon, lat = bahamas_tmp.IRS_LON, bahamas_tmp.IRS_LAT
        ax.plot(lon, lat, color=cmap(i/len(bahamas_all_files)), linewidth=2)

    # Add Oberpfaffenhofen
    ax.plot(edmo[0], edmo[1], 'or')
    ax.text(edmo[0] + 0.1, edmo[1] + 0.1, "EDMO", fontsize=10,
            path_effects=[patheffects.withStroke(linewidth=1, foreground="w")])

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{plot_path}/CIRRUS-HL_BAHAMAS_all_tracks.png", dpi=300)
    plt.close()
