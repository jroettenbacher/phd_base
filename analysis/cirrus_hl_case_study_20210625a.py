#!/usr/bin/env python
"""Case study for Cirrus-HL
* 25.06.2021: Radiation Square for BACARDI
author: Johannes Röttenbacher
"""
if __name__ == "__main--":
    # %% module import
    import pylim.helpers as h
    from pylim import reader
    from pylim.bacardi import fdw_attitude_correction
    from pylim.cirrus_hl import coordinates
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
    from scipy.interpolate import interp1d
    import logging
    log = logging.getLogger("pylim")
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.DEBUG)

    # %% 20210625 - Radiation Square
    log.info("20210625 - Radiation Square")
    # 90° = W, 180° = S, usw.
    rs_start = pd.Timestamp(2021, 6, 25, 11, 45)
    rs_end = pd.Timestamp(2021, 6, 25, 12, 27)

    # %% set paths
    flight = "Flight_20210625a"
    bahamas_dir = h.get_path("bahamas", flight)
    bacardi_dir = h.get_path("bacardi", flight)
    smart_dir = h.get_path("calibrated", flight)
    sat_dir = h.get_path("satellite", flight)
    if os.getcwd().startswith("C:"):
        outpath = f"C:/Users/Johannes/Documents/Doktor/campaigns/CIRRUS-HL/case_studies/{flight}"
    else:
        outpath = f"/projekt_agmwend/home_rad/jroettenbacher/case_studies/{flight}"
    h.make_dir(outpath)

    # %% find bahamas file and read in bahamas data and satellite picture
    file = [f for f in os.listdir(bahamas_dir) if f.endswith(".nc")][0]
    bahamas = reader.read_bahamas(f"{bahamas_dir}/{file}")
    sat_image = [f for f in os.listdir(sat_dir) if "MODIS" in f][0]
    sat_ds = rasterio.open(f"{sat_dir}/{sat_image}")
    bahamas_subset = bahamas.sel(time=slice(rs_start, rs_end))  # select subset of radiation square
    bahamas_rs = bahamas_subset.where(np.abs(bahamas_subset["IRS_PHI"]) < 1)  # select only sections with roll < 1°

    # %% BAHAMAS: select position and time data and set extent
    x_edmo, y_edmo = coordinates["EDMO"]
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

    # %% plot bahamas map with sat image
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
    points = ax.scatter(lon, lat, c=bahamas["IRS_HDG"], linewidth=6)
    # add the corresponding colorbar and decide whether to plot it horizontally or vertically
    plt.colorbar(points, ax=ax, pad=0.01, location=props["cb_loc"], label="Heading (°)", shrink=props["shrink"])

    # plot a way point every 15 minutes = 9000 seconds with a time stamp next to it
    for long, lati, time_stamp in zip(lon[9000::9000], lat[9000::9000], times[9000::9000]):
        ax.annotate(time_stamp.dt.strftime("%H:%M").values, (long, lati), fontsize=16,
                    path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])
        ax.plot(long, lati, '.k', markersize=10)

    # plot points with labels and white line around text
    ax.plot(x_edmo, y_edmo, 'ok')
    ax.text(x_edmo + 0.1, y_edmo + 0.1, "EDMO", fontsize=22,
            path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])
    plt.tight_layout(pad=0.1)
    fig_name = f"{outpath}/{flight}_bahamas_track_with_sat.png"
    # plt.show()
    plt.savefig(fig_name, dpi=100)
    log.info(f"Saved {fig_name}")
    plt.close()

    # %% plot BAHAMAS data for Radiation Square
    matplotlib.rcdefaults()
    fig, axs = plt.subplots(nrows=1)
    axs.plot(bahamas_subset["time"], bahamas_subset["IRS_HDG"], label="Heading")
    axs.set_ylabel("Heading (°) 0=N")
    axs.set_xlabel("Time (UTC)")
    axs.set_ylim((0, 360))
    axs.grid()
    axs2 = axs.twinx()
    axs2.plot(bahamas_subset["time"], bahamas_subset["IRS_PHI"], color="red", label="Roll")
    axs2.set_ylabel("Roll Angle (°)")
    axs2.set_ylim((-1.5, 1.5))
    axs.set_title(f"BAHAMAS Aircraft Data - {flight}")
    fig.legend(loc=2)
    fig.autofmt_xdate()
    # plt.show()
    fig_name = f"{outpath}/{flight}_roll_heading_rad_square.png"
    plt.savefig(fig_name)
    log.info(f"Saved {fig_name}")
    plt.close()

    # %% select only the relevant flight sections, removing the turns and plot it
    matplotlib.rcdefaults()
    fig, axs = plt.subplots(nrows=1)
    axs.plot(bahamas_rs["time"], bahamas_rs["IRS_HDG"], label="Heading")
    axs.set_ylabel("Heading (°) 0=N")
    axs.set_xlabel("Time (UTC)")
    axs.set_ylim((0, 360))
    axs.grid()
    axs2 = axs.twinx()
    axs2.plot(bahamas_rs["time"], bahamas_rs["IRS_PHI"], color="red", label="Roll")
    axs2.set_ylabel("Roll Angle (°)")
    axs2.set_ylim((-1.5, 1.5))
    axs.set_title(f"BAHAMAS Aircraft Data - {flight}")
    fig.legend(loc=2)
    fig.autofmt_xdate()
    fig_name = f"{outpath}/{flight}_roll_heading_rad_square_filtered.png"
    # plt.show()
    plt.savefig(fig_name)
    log.info(f"Saved {fig_name}")
    plt.close()

    # %% read in uncorrected and corrected BACARDI data and check for offsets
    bacardi_ds = xr.open_dataset(f"{bacardi_dir}/CIRRUS_HL_F02_20210625a_ADLR_BACARDI_BroadbandFluxes_R0.nc")
    bacardi_rs = bacardi_ds.sel(time=slice(rs_start, rs_end))  # select only radiation square data
    bacardi_raw = reader.read_bacardi_raw("QL-CIRRUS-HL_F02_20210625a_ADLR_BACARDI_v1.nc", bacardi_dir)
    bacardi_raw_rs = bacardi_raw.sel(time=slice(rs_start, rs_end))
    bacardi_nooffset = xr.open_dataset(f"{bacardi_dir}/CIRRUS_HL_F02_20210625a_ADLR_BACARDI_BroadbandFluxes_R0_0offset.nc")
    bacardi_nooffset_rs = bacardi_nooffset.sel(time=slice(rs_start, rs_end))
    bacardi_uncor = xr.open_dataset(f"{bacardi_dir}/CIRRUS_HL_F02_20210625a_ADLR_BACARDI_BroadbandFluxes_R0_noattcor.nc")
    bacardi_uncor_rs = bacardi_uncor.sel(time=slice(rs_start, rs_end))
    ylims = (1120, 1200)

    # %% plot all radiation square BACARDI data
    plt.rc('font', size=14)
    plt.rc('lines', linewidth=3)
    plt.rc('font', family="serif")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # solar radiation
    bacardi_rs.F_up_solar.plot(x="time", label=r"$F_{\uparrow}$ BACARDI", ax=ax, c="#6699CC", ls="-")
    bacardi_rs.F_down_solar.plot(x="time", label=r"$F_{\downarrow}$ BACARDI", ax=ax, c="#117733", ls="-")
    # terrestrial radiation
    bacardi_rs.F_up_terrestrial.plot(x="time", label=r"$F_{\uparrow}$ BACARDI", ax=ax, c="#CC6677", ls="-")
    bacardi_rs.F_down_terrestrial.plot(x="time", label=r"$F_{\downarrow}$ BACARDI", ax=ax, c="#f89c20", ls="-")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Irradiance (W$\,$m$^{-2}$)")
    h.set_xticks_and_xlabels(ax, pd.to_timedelta((bacardi_rs.time[-1] - bacardi_rs.time[0]).values))
    ax.grid()
    handles, labels = ax.get_legend_handles_labels()
    legend_column_headers = ["Solar", "Terrestrial"]
    handles.insert(0, Patch(color='none', label=legend_column_headers[0]))
    handles.insert(3, Patch(color='none', label=legend_column_headers[1]))
    ax.legend(handles=handles, bbox_to_anchor=(0.5, 0), loc="lower center", ncol=2, bbox_transform=fig.transFigure)
    plt.subplots_adjust(bottom=0.31)
    plt.tight_layout()
    fig_name = f"{outpath}/CIRRUS_HL_{flight}_bacardi_broadband_irradiance.png"
    # plt.show()
    plt.savefig(fig_name, dpi=100)
    log.info(f"Saved {fig_name}")
    plt.close()

    # %% plot F_dw radiation square BACARDI data old EUREC4A offsets
    plt.rc('font', size=14)
    plt.rc('lines', linewidth=3)
    plt.rc('font', family="serif")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # solar radiation
    bacardi_rs.F_down_solar.plot(x="time", label=r"$F_{\downarrow}$ BACARDI (0.3, 2.55)", ax=ax, c="#117733", ls="-")
    bacardi_rs.F_down_solar_sim.plot(x="time", label=r"$F_{\downarrow}$ libRadtran", ax=ax, c="#332288")
    ax2 = ax.twinx()
    heading = ax2.plot(bahamas_rs["time"], bahamas_rs["IRS_HDG"], label="Heading")
    saa = bacardi_rs.saa.plot(x="time", label="Solar Azimuth Angle")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Irradiance (W$\,$m$^{-2}$)")
    ax.set_ylim(ylims)  # fix y limits for better comparison
    ax2.set_ylabel("Heading (°) 0=N, Solar Azimuth Angle")
    h.set_xticks_and_xlabels(ax, pd.to_timedelta((bacardi_rs.time[-1] - bacardi_rs.time[0]).values))
    ax.grid(axis='x')
    ax2.grid()
    handles, labels = ax.get_legend_handles_labels()
    legend_column_headers = ["Solar (Roll offset, Pitch offset)", "Terrestrial"]
    handles.insert(0, Patch(color='none', label=legend_column_headers[0]))
    handles.insert(3, heading[0])
    handles.insert(4, saa[0])
    ax.legend(handles=handles, bbox_to_anchor=(0.5, 0), loc="lower center", ncol=2, bbox_transform=fig.transFigure)
    plt.subplots_adjust(bottom=0.31)
    plt.tight_layout()
    fig_name = f"{outpath}/CIRRUS_HL_{flight}_bacardi_fdw_saa_heading.png"
    # plt.show()
    plt.savefig(fig_name, dpi=100)
    log.info(f"Saved {fig_name}")
    plt.close()

    # %% plot BACARDI data which has been attitude corrected with no offset
    plt.rc('font', size=14)
    plt.rc('lines', linewidth=3)
    plt.rc('font', family="serif")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # solar radiation
    bacardi_nooffset_rs.F_down_solar.plot(x="time", label=r"$F_{\downarrow}$ BACARDI (0, 0)", ax=ax, c="#117733", ls="-")
    bacardi_nooffset_rs.F_down_solar_sim.plot(x="time", label=r"$F_{\downarrow}$ libRadtran", ax=ax, c="#332288")
    ax2 = ax.twinx()
    heading = ax2.plot(bahamas_rs["time"], bahamas_rs["IRS_HDG"], label="Heading")
    saa = bacardi_nooffset_rs.saa.plot(x="time", label="Solar Azimuth Angle")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Irradiance (W$\,$m$^{-2}$)")
    ax.set_ylim(ylims)  # fix y limits for better comparison
    ax2.set_ylabel("Heading (°) 0=N, Solar Azimuth Angle")
    h.set_xticks_and_xlabels(ax, pd.to_timedelta((bacardi_nooffset_rs.time[-1] - bacardi_nooffset_rs.time[0]).values))
    ax.grid(axis='x')
    ax2.grid()
    handles, labels = ax.get_legend_handles_labels()
    legend_column_headers = ["Solar (Roll offset, Pitch offset)", "Terrestrial"]
    handles.insert(0, Patch(color='none', label=legend_column_headers[0]))
    handles.insert(3, heading[0])
    handles.insert(4, saa[0])
    ax.legend(handles=handles, bbox_to_anchor=(0.5, 0), loc="lower center", ncol=2, bbox_transform=fig.transFigure)
    plt.subplots_adjust(bottom=0.31)
    plt.tight_layout()
    fig_name = f"{outpath}/CIRRUS_HL_{flight}_bacardi_fdw_saa_heading_no_offset.png"
    # plt.show()
    plt.savefig(fig_name, dpi=100)
    log.info(f"Saved {fig_name}")
    plt.close()

    # %% plot BACARDI data with no attitude correction
    plt.rc('font', size=14)
    plt.rc('lines', linewidth=3)
    plt.rc('font', family="serif")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # solar radiation
    bacardi_uncor_rs.F_down_solar.plot(x="time", label=r"$F_{\downarrow}$ BACARDI (None, None)", ax=ax, c="#117733", ls="-")
    bacardi_uncor_rs.F_down_solar_sim.plot(x="time", label=r"$F_{\downarrow}$ libRadtran", ax=ax, c="#332288")
    ax2 = ax.twinx()
    heading = ax2.plot(bahamas_rs["time"], bahamas_rs["IRS_HDG"], label="Heading")
    saa = bacardi_uncor_rs.saa.plot(x="time", label="Solar Azimuth Angle")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Irradiance (W$\,$m$^{-2}$)")
    ax.set_ylim(ylims)  # fix y limits for better comparison
    ax2.set_ylabel("Heading (°) 0=N, Solar Azimuth Angle")
    h.set_xticks_and_xlabels(ax, pd.to_timedelta((bacardi_uncor_rs.time[-1] - bacardi_uncor_rs.time[0]).values))
    ax.grid(axis='x')
    ax2.grid()
    handles, labels = ax.get_legend_handles_labels()
    legend_column_headers = ["Solar (Roll offset, Pitch offset)", "Terrestrial"]
    handles.insert(0, Patch(color='none', label=legend_column_headers[0]))
    handles.insert(3, heading[0])
    handles.insert(4, saa[0])
    ax.legend(handles=handles, bbox_to_anchor=(0.5, 0), loc="lower center", ncol=2, bbox_transform=fig.transFigure)
    plt.subplots_adjust(bottom=0.31)
    plt.tight_layout()
    fig_name = f"{outpath}/CIRRUS_HL_{flight}_bacardi_fdw_saa_heading_no_att_cor.png"
    # plt.show()
    plt.savefig(fig_name, dpi=100)
    log.info(f"Saved {fig_name}")
    plt.close()

    # %% vary roll and pitch offsets and correct F_dw
    roll_offset = -0.15
    pitch_offset = 2.85
    dirdiff = reader.read_libradtran(flight, "BBR_DirectFraction_Flight_20210625a_R0_ds_high.dat")
    dirdiff_rs = dirdiff.loc[rs_start:rs_end]
    # interpolate f_dir on bacardi time
    f_dir_func = interp1d(dirdiff_rs.index.values.astype(float), dirdiff_rs.f_dir, fill_value="extrapolate")
    f_dir_inp = f_dir_func(bacardi_raw_rs.time.values.astype(float))
    F_down_solar_att, factor = fdw_attitude_correction(bacardi_uncor_rs.F_down_solar.values,
                                                       roll=bacardi_raw_rs.IRS_PHI.values,
                                                       pitch=-bacardi_raw_rs.IRS_THE.values,
                                                       yaw=bacardi_raw_rs.IRS_HDG.values, sza=bacardi_uncor_rs.sza.values,
                                                       saa=bacardi_uncor_rs.saa.values, fdir=f_dir_inp,
                                                       r_off=roll_offset, p_off=pitch_offset)

    # %% plot new attitude corrected downward solar irradiance
    plt.rc('font', size=14)
    plt.rc('lines', linewidth=3)
    plt.rc('font', family="serif")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # solar radiation
    bacardi_uncor_rs.F_down_solar.plot(x="time", label=r"$F_{\downarrow}$ BACARDI (None, None)", ax=ax, c="#117733", ls="-")
    bacardi_uncor_rs.F_down_solar_sim.plot(x="time", label=r"$F_{\downarrow}$ simulated clear sky", ax=ax, ls="-", c="#332288")
    ax.plot(bacardi_uncor_rs.time, F_down_solar_att, label=f"F_dw BACARDI ({roll_offset}, {pitch_offset})", ls="-",
            c="#CC6677")
    ax2 = ax.twinx()
    heading = ax2.plot(bahamas_rs["time"], bahamas_rs["IRS_HDG"], label="Heading")
    saa = bacardi_uncor_rs.saa.plot(x="time", label="Solar Azimuth Angle")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Irradiance (W$\,$m$^{-2}$)")
    ax.set_ylim(ylims)  # fix y limits for better comparison (1140, 1170)
    ax2.set_ylabel("Heading (°) 0=N, Solar Azimuth Angle")
    h.set_xticks_and_xlabels(ax, pd.to_timedelta((bacardi_uncor_rs.time[-1] - bacardi_uncor_rs.time[0]).values))
    ax.grid(axis='x')
    ax2.grid()
    handles, labels = ax.get_legend_handles_labels()
    legend_column_headers = ["Solar (Roll offset, Pitch offset)", "Terrestrial"]
    handles.insert(0, Patch(color='none', label=legend_column_headers[0]))
    handles.insert(4, Patch(color='none'))
    handles.insert(5, heading[0])
    handles.insert(6, saa[0])
    ax.legend(handles=handles, bbox_to_anchor=(0.5, 0), loc="lower center", ncol=2, bbox_transform=fig.transFigure)
    plt.subplots_adjust(bottom=0.37)
    plt.tight_layout()
    fig_name = f"{outpath}/CIRRUS_HL_{flight}_bacardi_fdw_saa_heading_new_att_corr_{roll_offset}_{pitch_offset}.png"
    # plt.show()
    plt.savefig(fig_name, dpi=100)
    log.info(f"Saved {fig_name}")
    plt.close()

