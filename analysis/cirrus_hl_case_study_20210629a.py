#!/usr/bin/env python
"""Case study for Cirrus-HL
* 29.06.2021: cirrus over Atlantic west and north of Iceland -> Poster HALO Status Colloquium
author: Johannes Röttenbacher
"""
if __name__ == "__main__":

# %% module import
    import pylim.helpers as h
    from pylim import reader
    from pylim import smart
    from pylim.cirrus_hl import stop_over_locations, coordinates
    from pylim.bahamas import plot_props
    import numpy as np
    import os
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from matplotlib.patches import Patch
    import cartopy
    import cartopy.crs as ccrs
    import xarray as xr
    import pandas as pd
    import rasterio
    from rasterio.plot import show
    from tqdm import tqdm
    import logging
    log = logging.getLogger("pylim")
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)

# %% 20210629
    print(20210629)

# %% set paths and background data
    flight = "Flight_20210629a"
    date = "20210629"
    bahamas_dir = h.get_path("bahamas", flight)
    bacardi_dir = h.get_path("bacardi", flight)
    smart_dir = h.get_path("calibrated", flight)
    sat_dir = h.get_path("satellite", flight)
    horidata_dir = h.get_path("horidata", flight)
    ecrad_dir = os.path.join(h.get_path("ecrad"), date)
    libradtran_dir = h.get_path("libradtran", flight)
    sat_file = [f for f in os.listdir(sat_dir) if "MODIS" in f][0]
    sat_image = os.path.join(sat_dir, sat_file)
    if os.getcwd().startswith("C:"):
        outpath = f"C:/Users/Johannes/Documents/Doktor/campaigns/CIRRUS-HL/case_studies/{flight}"
    else:
        outpath = f"/projekt_agmwend/home_rad/jroettenbacher/case_studies/{flight}"
    h.make_dir(outpath)
    start_dt = pd.Timestamp(2021, 6, 29, 10, 10)
    end_dt = pd.Timestamp(2021, 6, 29, 11, 54)
    below_cloud = (start_dt, pd.Timestamp(2021, 6, 29, 10, 15))
    in_cloud = (pd.Timestamp(2021, 6, 29, 10, 15), pd.Timestamp(2021, 6, 29, 11, 54))
    above_cloud = (pd.Timestamp(2021, 6, 29, 11, 54), pd.Timestamp(2021, 6, 29, 12, 5))

# %% find bahamas file and read in bahamas data
    file = [f for f in os.listdir(bahamas_dir) if f.endswith(".nc")][0]
    bahamas = reader.read_bahamas(os.path.join(bahamas_dir, file))

# %% read in satellite picture
    sat_ds = rasterio.open(sat_image)

# %% select flight sections for plotting
    bahamas_belowcloud = (below_cloud[0] < bahamas.time) & (bahamas.time < below_cloud[1])
    bahamas_abovecloud = (above_cloud[0] < bahamas.time) & (bahamas.time < above_cloud[1])
    bahamas_incloud = (in_cloud[0] < bahamas.time) & (bahamas.time < in_cloud[1])

# %% select further points to plot
    x_edmo, y_edmo = coordinates["EDMO"]
    airport = stop_over_locations[flight] if flight in stop_over_locations else None
    x2, y2 = coordinates[airport]
    torshavn_x, torshavn_y = coordinates["Torshavn"]

# %% select position and time data and set extent
    lon, lat, altitude, times = bahamas["IRS_LON"], bahamas["IRS_LAT"], bahamas["IRS_ALT"], bahamas["time"]
    pad = 2
    llcrnlat = lat.min(skipna=True) - pad
    llcrnlon = lon.min(skipna=True) - pad
    urcrnlat = lat.max(skipna=True) + pad
    urcrnlon = lon.max(skipna=True) + pad
    extent = [llcrnlon, urcrnlon, llcrnlat, urcrnlat]
    font = {'weight': 'bold', 'size': 26}
    matplotlib.rc('font', **font)
    # get plot properties
    props = plot_props[flight]
    h.set_cb_friendly_colors()

# %% plot bahamas map with highlighted below and above cloud sections and sat image
    fig, ax = plt.subplots(figsize=(11, 9), subplot_kw={"projection": ccrs.PlateCarree()})
    # ax.stock_img()
    show(sat_ds, ax=ax)
    ax.coastlines(linewidth=3)
    ax.add_feature(cartopy.feature.BORDERS, linewidth=3)
    ax.set_extent(extent)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True)
    gl.bottom_labels = False
    gl.left_labels = False

    # plot flight track
    points = ax.plot(lon, lat, c="orange", linewidth=6)
    # plot in and below cloud points for case study
    lon_incloud = bahamas["IRS_LON"].where(bahamas_incloud, drop=True)
    lat_incloud = bahamas["IRS_LAT"].where(bahamas_incloud, drop=True)
    ax.plot(lon_incloud, lat_incloud, c="cornflowerblue", linewidth=6, label="inside cloud")
    lon_below = bahamas["IRS_LON"].where(bahamas_belowcloud, drop=True)
    lat_below = bahamas["IRS_LAT"].where(bahamas_belowcloud, drop=True)
    ax.plot(lon_below, lat_below, c="green", linewidth=6, label="below cloud")
    lon_above = bahamas["IRS_LON"].where(bahamas_abovecloud, drop=True)
    lat_above = bahamas["IRS_LAT"].where(bahamas_abovecloud, drop=True)
    ax.plot(lon_above, lat_above, c="red", linewidth=6, label="above cloud")

    # plot a way point every 15 minutes = 9000 seconds with a time stamp next to it
    for long, lati, time_stamp in zip(lon[9000::9000], lat[9000::9000], times[9000::9000]):
        ax.annotate(time_stamp.dt.strftime("%H:%M").values, (long, lati), fontsize=16,
                    path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])
        ax.plot(long, lati, '.k', markersize=10)

    # plot points with labels and white line around text
    ax.plot(x_edmo, y_edmo, 'ok')
    ax.text(x_edmo + 0.1, y_edmo + 0.1, "EDMO", fontsize=22,
            path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])
    ax.plot(x2, y2, 'ok')
    ax.text(x2 + 0.1, y2 + 0.1, airport, fontsize=22,
            path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])
    ax.plot(torshavn_x, torshavn_y, 'ok')
    ax.text(torshavn_x + 0.1, torshavn_y + 0.1, "Torshavn", fontsize=22,
            path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])

    ax.legend(loc=3, fontsize=18, markerscale=6)
    plt.tight_layout(pad=0.1)
    fig_name = f"{outpath}/{flight}_bahamas_track.png"
    # plt.show()
    plt.savefig(fig_name, dpi=100)
    log.info(f"Saved {fig_name}")
    plt.close()

# %% plot bahamas data to check for clouds
    plt.rcdefaults()
    ylabels = ["Static Air\nTemperature (K)", "Relative \nHumidity (%)", "Static \nPressure (hPa)"]
    fig, axs = plt.subplots(nrows=3)
    bahamas.TS.plot(ax=axs[0])
    axs[0].axhline(y=235, color="r", linestyle="--", label="$235\,$K")
    bahamas.RELHUM.plot(ax=axs[1])
    bahamas.PS.plot(ax=axs[2])
    axs[2].invert_yaxis()
    timedelta = pd.to_datetime(bahamas.time[-1].values) - pd.to_datetime(bahamas.time[0].values)

    for ax, ylabel in zip(axs, ylabels):
        ax.set_ylabel(ylabel)
        ax.grid()
        h.set_xticks_and_xlabels(ax, timedelta)
        ax.fill_between(bahamas.time, 0, 1, where=((below_cloud[0] < bahamas.time) & (bahamas.time < below_cloud[1])),
                        transform=ax.get_xaxis_transform(), label="below cloud", color="green", alpha=0.5)
        ax.fill_between(bahamas.time, 0, 1, where=((in_cloud[0] < bahamas.time) & (bahamas.time < in_cloud[1])),
                        transform=ax.get_xaxis_transform(), label="inside cloud", color="grey", alpha=0.5)
        ax.fill_between(bahamas.time, 0, 1, where=((above_cloud[0] < bahamas.time) & (bahamas.time < above_cloud[1])),
                        transform=ax.get_xaxis_transform(), label="above cloud", color="red", alpha=0.5)

    axs[2].set_xlabel("Time (UTC)")
    for ax in axs[0:2]:
        ax.set_xlabel("")
        ax.set_xticklabels("")

    axs[0].legend()
    axs[0].set_ylim((150, 300))
    # axs[2].legend(bbox_to_anchor=(0.05, 0), loc="lower left", bbox_transform=fig.transFigure, ncol=4)
    # plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    # plt.show()
    figname = f"{outpath}/{flight}_bahamas_overview.png"
    plt.savefig(figname, dpi=100)
    log.info(f"Saved {figname}")
    plt.close()

# %% read in libradtran and bacardi files
    libradtran_file = "BBR_Fdn_clear_sky_Flight_20210629a_R0_ds_high.dat"
    libradtran_file_ter = "BBR_Fdn_clear_sky_Flight_20210629a_R0_ds_high_ter.dat"
    libradtran_file_smart = f"{flight}_libRadtran_clearsky_simulation_smart.nc"
    bacardi_file = "CIRRUS_HL_F05_20210629a_ADLR_BACARDI_BroadbandFluxes_R0.nc"
    bbr_sim = reader.read_libradtran(flight, libradtran_file)
    bbr_sim_ter = reader.read_libradtran(flight, libradtran_file_ter)
    bacardi_ds = xr.open_dataset(f"{bacardi_dir}/{bacardi_file}")
    smart_sim = xr.open_dataset(f"{libradtran_dir}/{libradtran_file_smart}")

# %% select flight sections for libRadtran simulations and BACARDI measurements
    bbr_belowcloud = ((below_cloud[0] < bbr_sim.index) & (bbr_sim.index < below_cloud[1]))
    bbr_ter_belowcloud = ((below_cloud[0] < bbr_sim_ter.index) & (bbr_sim_ter.index < below_cloud[1]))
    bacardi_belowcloud = ((below_cloud[0] < bacardi_ds.time) & (bacardi_ds.time < below_cloud[1]))
    bbr_abovecloud = ((above_cloud[0] < bbr_sim.index) & (bbr_sim.index < above_cloud[1]))
    bbr_ter_abovecloud = ((above_cloud[0] < bbr_sim_ter.index) & (bbr_sim_ter.index < above_cloud[1]))
    bacardi_abovecloud = ((above_cloud[0] < bacardi_ds.time) & (bacardi_ds.time < above_cloud[1]))

# %% get mean values for flight sections
    bbr_sim[bbr_belowcloud].mean()
    bbr_sim_ter[bbr_ter_belowcloud].mean()
    bacardi_ds.sel(time=bacardi_belowcloud).mean()
    bbr_sim[bbr_abovecloud].mean()
    bbr_sim_ter[bbr_ter_abovecloud].mean()
    bacardi_ds.sel(time=bacardi_abovecloud).mean()
# %% plot libradtran simulations together with BACARDI measurements (solar + terrestrial)
    plt.rcdefaults()
    h.set_cb_friendly_colors()
    plt.rc('font', size=20)
    plt.rc('lines', linewidth=3)

    x_sel = (pd.Timestamp(2021, 6, 29, 10), pd.Timestamp(2021, 6, 29, 12, 15))
    fig, ax = plt.subplots(figsize=(13, 9))
    # solar radiation
    bacardi_ds.F_up_solar.plot(x="time", label=r"F$_\uparrow$ BACARDI", ax=ax, c="#6699CC", ls="-")
    bacardi_ds.F_down_solar.plot(x="time", label=r"F$_\downarrow$ BACARDI", ax=ax, c="#117733", ls="-")
    bbr_sim.plot(y="F_up", ax=ax, label=r"F$_\uparrow$ libRadtran", c="#6699CC", ls="--",
                 path_effects=[patheffects.withStroke(linewidth=6, foreground="k")])
    bbr_sim.plot(y="F_dw", ax=ax, ylabel="Broadband irradiance (W$\,$m$^{-2}$)", label=r"F$_\downarrow$ libRadtran",
                 c="#117733", ls="--", path_effects=[patheffects.withStroke(linewidth=6, foreground="k")])
    # terrestrial radiation
    bacardi_ds.F_up_terrestrial.plot(x="time", label=r"F$_\uparrow$ BACARDI", ax=ax, c="#CC6677", ls="-")
    bacardi_ds.F_down_terrestrial.plot(x="time", label=r"F$_\downarrow$ BACARDI", ax=ax, c="#f89c20", ls="-")
    bbr_sim_ter.plot(y="F_up", ax=ax, label=r"F$_\uparrow$ libRadtran", c="#CC6677", ls="--",
                     path_effects=[patheffects.withStroke(linewidth=6, foreground="k")])
    bbr_sim_ter.plot(y="F_dw", ax=ax, ylabel="Broadband irradiance (W$\,$m$^{-2}$)", label=r"F$_\downarrow$ libRadtran",
                     c="#f89c20", ls="--", path_effects=[patheffects.withStroke(linewidth=6, foreground="k")])
    ax.set_xlabel("Time (UTC)")
    ax.set_xlim(x_sel)
    h.set_xticks_and_xlabels(ax, x_sel[1] - x_sel[0])
    ax.grid()
    # ax.fill_between(bbr_sim.index, 0, 1, where=((start_dt < bbr_sim.index) & (bbr_sim.index < end_dt)),
    #                 transform=ax.get_xaxis_transform(), label="Case Study", color="grey")
    ax.fill_between(bbr_sim.index, 0, 1, where=bbr_belowcloud,
                    transform=ax.get_xaxis_transform(), label="below cloud", color="green", alpha=0.5)
    ax.fill_between(bbr_sim.index, 0, 1, where=((in_cloud[0] < bbr_sim.index) & (bbr_sim.index < in_cloud[1])),
                    transform=ax.get_xaxis_transform(), label="inside cloud", color="grey", alpha=0.5)
    ax.fill_between(bbr_sim.index, 0, 1, where=bbr_abovecloud,
                    transform=ax.get_xaxis_transform(), label="above cloud", color="red", alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    legend_column_headers = ["Solar", "Terrestrial"]
    handles.insert(0, Patch(color='none', label=legend_column_headers[0]))
    handles.insert(5, Patch(color='none', label=legend_column_headers[1]))
    # add dummy legend entries to get the right amount of rows per column
    handles.append(Patch(color='none', label=""))
    handles.append(Patch(color='none', label=""))
    ax.legend(handles=handles, bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=3)
    plt.subplots_adjust(bottom=0.37)
    plt.tight_layout()
    # plt.show()
    figname = f"{outpath}/{flight}_bacardi_libradtran_broadband_irradiance.png"
    plt.savefig(figname, dpi=100)
    log.info(f"Saved {figname}")
    plt.close()

# %% plot libradtran simulations together with BACARDI measurements (terrestrial)
    plt.rcdefaults()
    x_sel = (pd.Timestamp(2021, 6, 29, 9), pd.Timestamp(2021, 6, 29, 13))
    fig, ax = plt.subplots()
    bacardi_ds.F_up_terrestrial.plot(x="time", label="F_up BACARDI", ax=ax)
    bacardi_ds.F_down_terrestrial.plot(x="time", label="F_dw BACARDI", ax=ax)
    bbr_sim_ter.plot(y="F_dw", ax=ax, ylabel="Broadband irradiance (W$\,$m$^{-2}$)", label="F_dw libRadtran")
    bbr_sim_ter.plot(y="F_up", ax=ax, label="F_up libRadtran")
    ax.set_xlabel("Time (UTC)")
    ax.set_xlim(x_sel)
    h.set_xticks_and_xlabels(ax, x_sel[1] - x_sel[0])
    ax.grid()
    # ax.fill_between(bbr_sim.index, 0, 1, where=((start_dt < bbr_sim.index) & (bbr_sim.index < end_dt)),
    #                 transform=ax.get_xaxis_transform(), label="Case Study", color="grey")
    ax.fill_between(bbr_sim.index, 0, 1, where=bbr_belowcloud,
                    transform=ax.get_xaxis_transform(), label="below cloud", color="green", alpha=0.5)
    ax.fill_between(bbr_sim.index, 0, 1, where=((in_cloud[0] < bbr_sim.index) & (bbr_sim.index < in_cloud[1])),
                    transform=ax.get_xaxis_transform(), label="inside cloud", color="grey", alpha=0.5)
    ax.fill_between(bbr_sim.index, 0, 1, where=bbr_abovecloud,
                    transform=ax.get_xaxis_transform(), label="above cloud", color="red", alpha=0.5)
    ax.legend(bbox_to_anchor=(0.1, 0), loc="lower left", bbox_transform=fig.transFigure, ncol=3)
    plt.subplots_adjust(bottom=0.3)
    plt.tight_layout()
    # plt.show()
    figname= f"{outpath}/{flight}_bacardi_libradtran_broadband_irradiance_terrestrial.png"
    plt.savefig(figname, dpi=100)
    log.info(f"Saved {figname}")
    plt.close()

# %% read in SMART data
    smart_files = [f for f in os.listdir(smart_dir)]
    smart_files.sort()
    fdw_swir = reader.read_smart_cor(smart_dir, smart_files[2])
    fdw_vnir = reader.read_smart_cor(smart_dir, smart_files[3])
    fup_swir = reader.read_smart_cor(smart_dir, smart_files[4])
    fup_vnir = reader.read_smart_cor(smart_dir, smart_files[5])

# %% average smart spectra over different flight sections
    below_cloud_mean_fdw_vnir = fdw_vnir[below_cloud[0]:below_cloud[1]].mean()
    below_cloud_mean_fup_vnir = fup_vnir[below_cloud[0]:below_cloud[1]].mean()
    below_cloud_mean_fdw_swir = fdw_swir[below_cloud[0]:below_cloud[1]].mean()
    below_cloud_mean_fup_swir = fup_swir[below_cloud[0]:below_cloud[1]].mean()

    above_cloud_mean_fdw_vnir = fdw_vnir[above_cloud[0]:above_cloud[1]].mean()
    above_cloud_mean_fup_vnir = fup_vnir[above_cloud[0]:above_cloud[1]].mean()
    above_cloud_mean_fdw_swir = fdw_swir[above_cloud[0]:above_cloud[1]].mean()
    above_cloud_mean_fup_swir = fup_swir[above_cloud[0]:above_cloud[1]].mean()

# %% get pixel to wavelength mapping for each spectrometer
    pixel_wl_dict = dict()
    for filename in smart_files[2:]:
        date_str, channel, direction = smart.get_info_from_filename(filename)
        name = f"{direction}_{channel}"
        spectrometer = smart.lookup[name]
        pixel_wl_dict[name.casefold()] = reader.read_pixel_to_wavelength(h.get_path("pixel_wl"), spectrometer)

# %% prepare data frame for plotting VNIR
    plot_fup_vnir = pixel_wl_dict["fup_vnir"]
    plot_fup_vnir["fup_below_cloud"] = below_cloud_mean_fup_vnir.reset_index(drop=True)
    plot_fup_vnir["fup_above_cloud"] = above_cloud_mean_fup_vnir.reset_index(drop=True)
    plot_fdw_vnir = pixel_wl_dict["fdw_vnir"]
    plot_fdw_vnir["fdw_below_cloud"] = below_cloud_mean_fdw_vnir.reset_index(drop=True)
    plot_fdw_vnir["fdw_above_cloud"] = above_cloud_mean_fdw_vnir.reset_index(drop=True)

    # filter wrong calibrated wavelengths
    min_wl, max_wl = 385, 900
    plot_fup_vnir = plot_fup_vnir[plot_fup_vnir["wavelength"].between(min_wl, max_wl)]
    plot_fdw_vnir = plot_fdw_vnir[plot_fdw_vnir["wavelength"].between(min_wl, max_wl)]

# %% prepare data frame for plotting SWIR
    plot_fup_swir = pixel_wl_dict["fup_swir"]
    plot_fup_swir["fup_below_cloud"] = below_cloud_mean_fup_swir.reset_index(drop=True)
    plot_fup_swir["fup_above_cloud"] = above_cloud_mean_fup_swir.reset_index(drop=True)
    plot_fdw_swir = pixel_wl_dict["fdw_swir"]
    plot_fdw_swir["fdw_below_cloud"] = below_cloud_mean_fdw_swir.reset_index(drop=True)
    plot_fdw_swir["fdw_above_cloud"] = above_cloud_mean_fdw_swir.reset_index(drop=True)

# %% merge VNIR and SWIR data
    plot_fup = pd.concat([plot_fup_vnir, plot_fup_swir], ignore_index=True)
    plot_fdw = pd.concat([plot_fdw_vnir, plot_fdw_swir], ignore_index=True)

# %% sort dataframes by wavelength
    plot_fup.sort_values(by="wavelength", inplace=True)
    plot_fdw.sort_values(by="wavelength", inplace=True)

# %% remove 800 - 950 nm from fup -> calibration problem
    plot_fup.iloc[:, 2:] = plot_fup.iloc[:, 2:].where(~plot_fup["wavelength"].between(850, 950), np.nan)

# %% calculate albedo below and above cloud
    albedo = plot_fup.loc[:, ("pixel", "wavelength")].copy()
    albedo["albedo_below_cloud"] = np.abs(plot_fup["fup_below_cloud"] / plot_fdw["fdw_below_cloud"])
    albedo["albedo_above_cloud"] = np.abs(plot_fup["fup_above_cloud"] / plot_fdw["fdw_above_cloud"])
    albedo = albedo.rename(columns={"fup_below_cloud": "albedo_below_cloud", "fup_above_cloud": "albedo_above_cloud"})
    albedo = albedo[albedo["wavelength"] < 2180]

# %% plot averaged spectra F_up and F_dw
    plt.rcParams.update({'font.size': 14})
    h.set_cb_friendly_colors()
    fig, axs = plt.subplots(figsize=(10, 8), nrows=3)
    plot_fup.plot(x='wavelength', y='fup_below_cloud', ax=axs[0], label="F_up below cloud", linewidth=2)
    plot_fup.plot(x='wavelength', y='fup_above_cloud', ax=axs[0], label="F_up above cloud", linewidth=2)
    # axs[0].fill_between(plot_fup.wavelength, 0.1, 0.6, where=(plot_fup.wavelength.between(800, 1000)),
    #                     label="Calibration offset", color="grey")
    axs[0].set_ylabel("Irradiance (W$\\,$m$^{-2}\\,$nm$^{-1}$)")
    axs[0].set_xlabel("")
    axs[0].grid()
    axs[0].legend()

    # plot f_dw
    plot_fdw.plot(x='wavelength', y='fdw_below_cloud', ax=axs[1], label="F_down below cloud", linewidth=2)
    plot_fdw.plot(x='wavelength', y='fdw_above_cloud', ax=axs[1], label="F_down above cloud", linewidth=2)
    axs[1].set_ylabel("Irradiance (W$\\,$m$^{-2}\\,$nm$^{-1}$)")
    axs[1].set_xlabel("")
    axs[1].grid()
    axs[1].legend()

    # plot albedo
    albedo.plot(x='wavelength', y='albedo_below_cloud', ax=axs[2], label="Albedo below cloud", linewidth=2)
    albedo.plot(x='wavelength', y='albedo_above_cloud', ax=axs[2], label="Albedo above cloud", linewidth=2)
    axs[2].set_ylabel("Albedo")
    axs[2].set_xlabel("")
    axs[2].grid()
    axs[2].legend()
    axs[2].set_ylim((0, 1))

    # fig.supylabel("Irradiance (W$\\,$m$^{-2}\\,$nm$^{-1}$)")
    fig.supxlabel("Wavelength (nm)")
    plt.tight_layout(pad=0.5)
    # plt.show()
    figname = f"{outpath}/{flight}_SMART_average_spectra_albedo.png"
    plt.savefig(figname, dpi=100)
    log.info(f"Saved {figname}")
    plt.close()

# %% read in ecRad output
    ecrad_output_file = [f for f in os.listdir(ecrad_dir) if "output" in f][0]
    ecrad_output = xr.open_dataset(f"{ecrad_dir}/{ecrad_output_file}")
    # assign coordinates to band_sw
    ecrad_output = ecrad_output.assign_coords({"band_sw": range(1, 15), "band_lw": range(1, 17)})

# %% read in SMART horidata
    ims_file = [f for f in os.listdir(horidata_dir) if "IMS" in f][0]
    horidata = reader.read_nav_data(os.path.join(horidata_dir, ims_file))
    # convert data frame to xarray for easier use later
    horidata = horidata.to_xarray()

# %% select only relevant time (bahamas time range)
    ecrad_output = ecrad_output.where(ecrad_output.time == bahamas.time, drop=True)

# %% select flight sections for ecRad
    ecrad_belowcloud = ((below_cloud[0] < ecrad_output.time) & (ecrad_output.time < below_cloud[1]))
    ecrad_abovecloud = ((above_cloud[0] < ecrad_output.time) & (ecrad_output.time < above_cloud[1]))

# %% calculate pressure height
    q_air = 1.292
    g_geo = 9.81
    pressure_hl = ecrad_output["pressure_hl"]
    ecrad_output["press_height"] = -(pressure_hl[:, 137]) * np.log(pressure_hl[:, :] / pressure_hl[:, 137]) / (
            q_air * g_geo)

# %% select bahamas data corresponding to model time
    bahamas_sel = bahamas.sel(time=ecrad_output.time)

# %% get height level of actual flight altitude in ecRad model, this determines only the index of the level
    ecrad_timesteps = len(ecrad_output.time)
    aircraft_height_level = np.zeros(ecrad_timesteps)

    for i in tqdm(range(ecrad_timesteps)):
        aircraft_height_level[i] = h.arg_nearest(ecrad_output["press_height"][i, :].values, bahamas_sel.IRS_ALT[i].values)

    aircraft_height_level = aircraft_height_level.astype(int)

# %% prepare ecRad data for plotting by selecting only the HALO flightlevel
    height_level_da = xr.DataArray(aircraft_height_level, dims=["time"], coords={"time": ecrad_output.time})
    ecrad_dn_sw = ecrad_output["spectral_flux_dn_sw"].isel(half_level=height_level_da)
    ecrad_up_sw = ecrad_output["spectral_flux_up_sw"].isel(half_level=height_level_da)
    ecrad_dn_sw_bb = ecrad_output["flux_dn_sw"].isel(half_level=height_level_da)
    ecrad_up_sw_bb = ecrad_output["flux_up_sw"].isel(half_level=height_level_da)
    ecrad_dn_lw_bb = ecrad_output["flux_dn_lw"].isel(half_level=height_level_da)
    ecrad_up_lw_bb = ecrad_output["flux_up_lw"].isel(half_level=height_level_da)

# %% read in SMART nc files
    fdw_vnir = xr.open_dataset(f"{smart_dir}/cirrus-hl_SMART_Fdw_VNIR_2021_06_29.nc")
    fup_vnir = xr.open_dataset(f"{smart_dir}/cirrus-hl_SMART_Fup_VNIR_2021_06_29.nc")
    fdw_swir = xr.open_dataset(f"{smart_dir}/cirrus-hl_SMART_Fdw_SWIR_2021_06_29.nc")
    fup_swir = xr.open_dataset(f"{smart_dir}/cirrus-hl_SMART_Fup_SWIR_2021_06_29.nc")
    # merge VNIR and SWIR channel
    smart_fdw = xr.merge([fdw_vnir.Fdw, fdw_swir.Fdw])
    smart_fup = xr.merge([fup_vnir.Fup, fup_swir.Fup])

# %% select nearest corresponding horidata to SMART measurements
    hdata = horidata.sel(time=smart_fup.time, method="nearest")

# %% filter SMART data for high motion angles and sensible values (0, 5 Wm^-2nm^-1)
    condition_fdw = xr.DataArray((np.abs(hdata["roll"]) < 4) & (np.abs(hdata["pitch"]) < 4),
                                 coords={"time": smart_fdw.time})
    condition_fup = xr.DataArray((np.abs(hdata["roll"]) < 4) & (np.abs(hdata["pitch"]) < 4),
                                 coords={"time": smart_fup.time})
    smart_fdw["Fdw"] = smart_fdw.Fdw.where((smart_fdw.Fdw > 0) & (smart_fdw.Fdw < 5))
    smart_fup["Fup"] = smart_fup.Fup.where((smart_fup.Fup > 0) & (smart_fup.Fup < 5))

# %% plot SMART timeseries of integrated data together with libRadtran simulation
    plt.rcdefaults()
    h.set_cb_friendly_colors()
    x_sel = (pd.Timestamp(2021, 6, 29, 10), pd.Timestamp(2021, 6, 29, 12, 15))
    plt.rc('font', size=20)
    plt.rc('lines', linewidth=3)
    # integrate F and apply rolling mean for smoothing
    Fup = smart_fup.Fup.sum(dim="wavelength", skipna=True).rolling(time=15, min_periods=1).mean()
    Fdw = smart_fdw.Fdw.sum(dim="wavelength", skipna=True).rolling(time=15, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(13, 9))
    # SMART measurements filtered for high motion angles
    Fup.where(condition_fup).plot(x="time", ax=ax, label=r"F$_\uparrow$ SMART")
    Fdw.where(condition_fdw).plot(x="time", ax=ax, label=r"F$_\downarrow$ SMART")
    # libRadtran simulations
    smart_sim.eup.plot(x="time", ax=ax, label=r"F$_\uparrow$ libRadtran", c="#6699CC", ls="--",
                       path_effects=[patheffects.withStroke(linewidth=6, foreground="k")])
    smart_sim.fdw.plot(x="time", ax=ax, label=r"F$_\downarrow$ libRadtran", c="#117733", ls="--",
                       path_effects=[patheffects.withStroke(linewidth=6, foreground="k")])
    # cloud marks
    ax.fill_between(ecrad_dn_sw.time.values, 0, 1, where=ecrad_belowcloud,
                    transform=ax.get_xaxis_transform(), label="below cloud", color="green", alpha=0.5)
    ax.fill_between(ecrad_dn_sw.time.values, 0, 1,
                    where=((in_cloud[0] < ecrad_dn_sw.time) & (ecrad_dn_sw.time < in_cloud[1])),
                    transform=ax.get_xaxis_transform(), label="inside cloud", color="grey", alpha=0.5)
    ax.fill_between(ecrad_dn_sw.time.values, 0, 1, where=ecrad_abovecloud,
                    transform=ax.get_xaxis_transform(), label="above cloud", color="red", alpha=0.5)
    # aesthetics
    ax.set_xlim(x_sel)
    h.set_xticks_and_xlabels(ax, x_sel[-1] - x_sel[0])
    ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=4)
    ax.set_title(f"Integrated SMART Measurement "
                 f"{smart_fup.wavelength[0].values:.2f} - {smart_fup.wavelength[-1].values:.2f} nm\n"
                 f"and libRadtran Clear Sky Simulation")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(r"Irradiance (W$\,$m$^{-2}$)")
    ax.grid()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    # plt.show()
    figname = f"{outpath}/cirrus-hl_SMART_integrated_libRadtran_timeseries_{flight}.png"
    plt.savefig(figname, dpi=100)
    log.info(f"Saved {figname}")
    plt.close()

# %% plot SMART timeseries of each wavelength data
    plt.rcdefaults()
    h.set_cb_friendly_colors()
    x_sel = (pd.Timestamp(2021, 6, 29, 10), pd.Timestamp(2021, 6, 29, 12, 15))
    plt.rc('font', size=16)
    plt.rc('lines', linewidth=3)
    for wl in tqdm(smart_fdw.wavelength, desc="Wavelength"):
        fig, ax = plt.subplots(figsize=(10, 7))
        # smart_fup.Fup.where((smart_fup.Fup < 5) & (smart_fup.Fup > 0)).sum(dim="wavelength").plot(x="time", ax=ax, label="Upward")
        smart_fdw.Fdw.where((smart_fdw.Fdw < 5) & (smart_fdw.Fdw > 0)).sel(wavelength=wl).plot(x="time", ax=ax, label="Downward")
        ax.legend(loc=4)
        ax.set_xlim(x_sel)
        h.set_xticks_and_xlabels(ax, x_sel[-1] - x_sel[0])

        ax.set_title(f"SMART measurement wavelength: {wl.values:.2f} nm")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel(r"Spectral Irradiance (W\,m$^{-2}\,$nm$^{-1}$)")
        ax.grid()
        plt.tight_layout()
        # plt.show()
        figname = f"{outpath}/smart_wavelength/cirrus-hl_SMART_Fdw_wl{wl.values:07.2f}_{date}.png"
        plt.savefig(figname)
        plt.close()

# %% sum up SMART irradiance over ecRad bands
    nr_bands = len(h.ecRad_bands)
    fdw_banded = np.empty((nr_bands, smart_fdw.time.shape[0]))
    fup_banded = np.empty((nr_bands, smart_fup.time.shape[0]))
    for i, band in enumerate(h.ecRad_bands):
        wl1 = h.ecRad_bands[band][0]
        wl2 = h.ecRad_bands[band][1]
        fdw_banded[i, :] = smart_fdw.Fdw.loc[dict(wavelength=slice(wl1, wl2))].sum(dim="wavelength")
        fup_banded[i, :] = smart_fup.Fup.loc[dict(wavelength=slice(wl1, wl2))].sum(dim="wavelength")

    fdw_banded = xr.DataArray(fdw_banded, coords={"ecrad_band": range(1, 15), "time": smart_fdw.time}, name="Fdw")
    fup_banded = xr.DataArray(fup_banded, coords={"ecrad_band": range(1, 15), "time": smart_fup.time}, name="Fup")

# %% plot ecrad flux in comparison to banded SMART flux
    band = 4
    time_extend = pd.Timedelta((fup_banded.time[-1] - fup_banded.time[0]).values)
    plt.rc('font', size=16)
    plt.rc('lines', linewidth=3)
    fig, ax = plt.subplots(figsize=(10, 7))
    fup_banded.sel(ecrad_band=band).plot(x="time", label=r"F$_\uparrow$ SMART", c="#6699CC", ls="-", ax=ax)
    fdw_banded.sel(ecrad_band=band).plot(x="time", label=r"F$_\downarrow$ SMART", c="#117733", ls="-", ax=ax)
    ecrad_up_sw.sel(band_sw=band).plot(x="time", label=r"F$_\uparrow$ ecRad", c="#CC6677", ls="--", ax=ax)
    ecrad_dn_sw.sel(band_sw=band).plot(x="time", label=r"F$_\downarrow$ ecRad", c="#f89c20", ls="--", ax=ax)
    ax.fill_between(ecrad_dn_sw.time.values, 0, 1, where=ecrad_belowcloud,
                    transform=ax.get_xaxis_transform(), label="below cloud", color="green", alpha=0.5)
    ax.fill_between(ecrad_dn_sw.time.values, 0, 1, where=((in_cloud[0] < ecrad_dn_sw.time) & (ecrad_dn_sw.time < in_cloud[1])),
                    transform=ax.get_xaxis_transform(), label="inside cloud", color="grey", alpha=0.5)
    ax.fill_between(ecrad_dn_sw.time.values, 0, 1, where=ecrad_abovecloud,
                    transform=ax.get_xaxis_transform(), label="above cloud", color="red", alpha=0.5)
    ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=4)

    ax.set_title(f"ecRad Band {band}: {h.ecRad_bands[f'Band{band}']} nm")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Spectral Downward Irradiance \n$(\mathrm{W\,m}^{-2}\mathrm{\,nm}^{-1})$")
    ax.grid()
    h.set_xticks_and_xlabels(ax, time_extend)
    plt.subplots_adjust(bottom=0.27)
    # plt.show()
    figname = f"{outpath}/cirrus-hl_smart_ecrad_band{band}_fdw_comparison_{date}.png"
    plt.savefig(figname, dpi=100)
    log.info(f"Saved {figname}")
    plt.close()

# %% plot aircraft track through model (2D variables)
    variable = "flux_up_lw"
    plt.rcdefaults()
    h.set_cb_friendly_colors()
    plt.rc('font', size=16)
    plt.rc('lines', linewidth=3)
    fig, ax = plt.subplots(figsize=(12, 7))
    ecrad_output[variable].plot(x="time", cmap="afmhot", ax=ax)
    height_level_da.plot(x="time", ax=ax, label="HALO altitude")
    ax.fill_between(ecrad_dn_sw.time.values, 0, 1, where=ecrad_belowcloud,
                    transform=ax.get_xaxis_transform(), label="below cloud", color="green", alpha=0.5)
    ax.fill_between(ecrad_dn_sw.time.values, 0, 1,
                    where=((in_cloud[0] < ecrad_dn_sw.time) & (ecrad_dn_sw.time < in_cloud[1])),
                    transform=ax.get_xaxis_transform(), label="inside cloud", color="grey", alpha=0.5)
    ax.fill_between(ecrad_dn_sw.time.values, 0, 1, where=ecrad_abovecloud,
                    transform=ax.get_xaxis_transform(), label="above cloud", color="red", alpha=0.5)
    ax.legend(loc=2)
    ax.set_title("ecRad Output along HALO Flight Track 29. June 2021")
    ax.set_ylabel("Model Half Level")
    ax.set_xlabel("Time (UTC)")
    ax.invert_yaxis()
    plt.tight_layout()
    # plt.show()
    figname = f"{outpath}/cirrus-hl_ecRad_{variable}_halo_alt_{date}.png"
    plt.savefig(figname, dpi=100)
    log.info(f"Saved {figname}")
    plt.close()


# %% plot ecRad simulations together with BACARDI measurements (solar + terrestrial)
    plt.rcdefaults()
    h.set_cb_friendly_colors()
    plt.rc('font', size=20)
    plt.rc('lines', linewidth=3)

    x_sel = (pd.Timestamp(2021, 6, 29, 10), pd.Timestamp(2021, 6, 29, 12, 15))
    fig, ax = plt.subplots(figsize=(13, 9))
    # solar radiation
    bacardi_ds.F_up_solar.plot(x="time", label=r"F$_\uparrow$ BACARDI", ax=ax, c="#6699CC", ls="-")
    bacardi_ds.F_down_solar.plot(x="time", label=r"F$_\downarrow$ BACARDI", ax=ax, c="#117733", ls="-")
    ecrad_up_sw_bb.plot(x="time", ax=ax, label=r"F$_\uparrow$ ecRad", c="#6699CC", ls="--",
                        path_effects=[patheffects.withStroke(linewidth=6, foreground="k")])
    ecrad_dn_sw_bb.plot(x="time", ax=ax, label=r"F$_\downarrow$ ecRad", c="#117733", ls="--",
                        path_effects=[patheffects.withStroke(linewidth=6, foreground="k")])
    # terrestrial radiation
    bacardi_ds.F_up_terrestrial.plot(x="time", label=r"F$_\uparrow$ BACARDI", ax=ax, c="#CC6677", ls="-")
    bacardi_ds.F_down_terrestrial.plot(x="time", label=r"F$_\downarrow$ BACARDI", ax=ax, c="#f89c20", ls="-")
    ecrad_up_lw_bb.plot(x="time", ax=ax, label=r"F$_\uparrow$ ecRad", c="#CC6677", ls="--",
                        path_effects=[patheffects.withStroke(linewidth=6, foreground="k")])
    ecrad_dn_lw_bb.plot(x="time", ax=ax, label=r"F$_\downarrow$ ecRad", c="#f89c20", ls="--",
                        path_effects=[patheffects.withStroke(linewidth=6, foreground="k")])
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Broadband irradiance (W$\,$m$^{-2}$)")
    ax.set_xlim(x_sel)
    h.set_xticks_and_xlabels(ax, x_sel[1] - x_sel[0])
    ax.grid()
    # ax.fill_between(bbr_sim.index, 0, 1, where=((start_dt < bbr_sim.index) & (bbr_sim.index < end_dt)),
    #                 transform=ax.get_xaxis_transform(), label="Case Study", color="grey")
    ax.fill_between(ecrad_dn_sw_bb.time.values, 0, 1, where=ecrad_belowcloud,
                    transform=ax.get_xaxis_transform(), label="below cloud", color="green", alpha=0.5)
    ax.fill_between(ecrad_dn_sw_bb.time.values, 0, 1, where=((in_cloud[0] < ecrad_dn_sw_bb.time.values)
                                                             & (ecrad_dn_sw_bb.time.values < in_cloud[1])),
                    transform=ax.get_xaxis_transform(), label="inside cloud", color="grey", alpha=0.5)
    ax.fill_between(ecrad_dn_sw_bb.time.values, 0, 1, where=ecrad_abovecloud,
                    transform=ax.get_xaxis_transform(), label="above cloud", color="red", alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    legend_column_headers = ["Solar", "Terrestrial"]
    handles.insert(0, Patch(color='none', label=legend_column_headers[0]))
    handles.insert(5, Patch(color='none', label=legend_column_headers[1]))
    # add dummy legend entries to get the right amount of rows per column
    handles.append(Patch(color='none', label=""))
    handles.append(Patch(color='none', label=""))
    ax.legend(handles=handles, bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=3)
    plt.subplots_adjust(bottom=0.37)
    plt.tight_layout()
    # plt.show()
    figname = f"{outpath}/{flight}_bacardi_ecRad_broadband_irradiance.png"
    plt.savefig(figname, dpi=100)
    log.info(f"Saved {figname}")
    plt.close()
