#!/usr/bin/env python
"""Case study for Cirrus-HL
* 15.07.2021 SMART fixed - contrail outbreak over Spain - Radiation Square
author: Johannes Röttenbacher
"""
if __name__ == "__main__":
    # %% module import
    import pylim.helpers as h
    from pylim import reader
    from pylim import smart
    from pylim.bacardi import fdw_attitude_correction
    from pylim.cirrus_hl import coordinates, lookup
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
    import logging
    log = logging.getLogger("pylim")
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)

    # %% 20210715 Spain flight for contrail outbreak and Radiation Square candidate
    print("20210715 SMART fixed - contrail outbreak over Spain - Radiation Square")
    rs_start = pd.Timestamp(2021, 7, 15, 8, 0)
    rs_end = pd.Timestamp(2021, 7, 15, 10, 0)
    flight = "Flight_20210715a"
    bacardi_dir = h.get_path("bacardi", flight)
    bahamas_dir = h.get_path("bahamas", flight)
    smart_dir = h.get_path("calibrated", flight)
    pixel_wl_dir = h.get_path("pixel_wl")
    libradtran_dir = h.get_path("libradtran", flight)
    sat_dir = h.get_path("satellite", flight)
    if os.getcwd().startswith("C:"):
        outpath = f"C:/Users/Johannes/Documents/Doktor/campaigns/CIRRUS-HL/case_studies/{flight}"
    else:
        outpath = f"/projekt_agmwend/home_rad/jroettenbacher/case_studies/{flight}"
    h.make_dir(outpath)
    ylims = (600, 1000)  # set ylims for all irradiance plots

    # %% find files and read them in
    bahamas_file = [f for f in os.listdir(bahamas_dir) if f.endswith(".nc")][0]
    bahamas = reader.read_bahamas(f"{bahamas_dir}/{bahamas_file}")
    bahamas_rs = bahamas.sel(time=slice(rs_start, rs_end))  # select subset of radiation square

    bacardi_file = [f for f in os.listdir(bacardi_dir) if f.endswith("R0.nc")][0]
    bacardi = xr.open_dataset(f"{bacardi_dir}/{bacardi_file}")
    bacardi_rs = bacardi.sel(time=slice(rs_start, rs_end))

    sat_image = [f for f in os.listdir(sat_dir) if "MODIS" in f][0]
    sat_ds = rasterio.open(f"{sat_dir}/{sat_image}")

    smart_fdw_vnir_file = [f for f in os.listdir(smart_dir) if "Fdw_VNIR" in f][0]
    smart_fdw_vnir = smart.read_smart_cor(smart_dir, smart_fdw_vnir_file)
    smart_fdw_swir_file = [f for f in os.listdir(smart_dir) if "Fdw_SWIR" in f][0]
    smart_fdw_swir = smart.read_smart_cor(smart_dir, smart_fdw_swir_file)

    fdw_sim_file = [f for f in os.listdir(libradtran_dir) if "smart_bb" in f][0]
    fdw_sim = xr.open_dataset(f"{libradtran_dir}/{fdw_sim_file}")
    fdw_sim_rs = fdw_sim.sel(time=slice(rs_start, rs_end))
    fdw_800_sim_file = [f for f in os.listdir(libradtran_dir) if "800nm" in f][0]
    fdw_800_sim = xr.open_dataset(f"{libradtran_dir}/{fdw_800_sim_file}")
    fdw_800_sim_rs = fdw_800_sim.sel(time=slice(rs_start, rs_end))

    # %% integrate SMART measurements
    smart_fdw_vnir_int = smart_fdw_vnir.sum(axis=1)
    smart_fdw_swir_int = smart_fdw_swir.sum(axis=1, skipna=False)
    smart_fdw = smart_fdw_vnir_int + smart_fdw_swir_int
    smart_fdw_rs = smart_fdw.loc[slice(rs_start, rs_end)]

    # %% select one wavelength with low diffuse part for correction
    pixel_wl = smart.read_pixel_to_wavelength(pixel_wl_dir, lookup["Fdw_VNIR"])
    pixel, wl = smart.find_pixel(pixel_wl, 800)
    smart_800 = smart_fdw_vnir.iloc[:, pixel]
    smart_800_rs = smart_800.loc[slice(rs_start, rs_end)]

    # %% filter high roll angles
    roll_filter = np.abs(bahamas_rs["IRS_PHI"]) < 1
    bahamas_rs_filtered = bahamas_rs.where(roll_filter)  # select only sections with roll < 1°
    bahamas_inp = bahamas_rs.interp(time=smart_fdw_rs.index)  # interpolate bahamas on SMART time
    roll_filter_smart = np.abs(bahamas_inp["IRS_PHI"]) < 1  # create filter for smart data
    smart_fdw_rs_filtered = smart_fdw_rs.where(roll_filter_smart)
    smart_800_rs_filtered = smart_800_rs.where(roll_filter_smart)

    # %% calculate angle towards sun
    angle_towards_sun = np.abs(bahamas_rs.IRS_HDG - fdw_sim_rs.saa)

    # %% BAHAMAS: select position and time data and set extent
    x_santiago, y_santiago = coordinates["Santiago"]
    lon, lat, altitude, times = bahamas_rs["IRS_LON"], bahamas_rs["IRS_LAT"], bahamas_rs["IRS_ALT"], bahamas_rs["time"]
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
    points = ax.scatter(lon, lat, c=bahamas_rs["IRS_ALT"] / 1000, linewidth=6)
    # add the corresponding colorbar and decide whether to plot it horizontally or vertically
    plt.colorbar(points, ax=ax, pad=0.01, location=props["cb_loc"], label="Height (km)", shrink=props["shrink"])

    # plot a way point every 15 minutes = 9000 seconds with a time stamp next to it
    for long, lati, time_stamp in zip(lon[9000::9000], lat[9000::9000], times[9000::9000]):
        ax.annotate(time_stamp.dt.strftime("%H:%M").values, (long, lati), fontsize=16,
                    path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])
        ax.plot(long, lati, '.k', markersize=10)

    # plot points with labels and white line around text
    ax.plot(x_santiago, y_santiago, 'ok')
    ax.text(x_santiago + 0.1, y_santiago + 0.1, "Santiago", fontsize=22,
            path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])
    plt.tight_layout(pad=0.1)
    fig_name = f"{outpath}/{flight}_bahamas_track_with_sat.png"
    # plt.show()
    plt.savefig(fig_name, dpi=100)
    log.info(f"Saved {fig_name}")
    plt.close()

    # %% plot BAHAMAS data for Radiation Square
    matplotlib.rcdefaults()
    h.set_cb_friendly_colors()
    fig, axs = plt.subplots(nrows=1)
    axs.plot(bahamas_rs["time"], bahamas_rs["IRS_HDG"], label="Heading")
    axs.set_ylabel("Heading (°) 0=N")
    axs.set_xlabel("Time (UTC)")
    axs.set_ylim((0, 360))
    axs.grid()
    axs2 = axs.twinx()
    axs2.plot(bahamas_rs["time"], bahamas_rs["IRS_PHI"], label="Roll", c="#117733")
    axs2.set_ylabel("Roll Angle (°)")
    axs2.set_ylim((-1.5, 1.5))
    axs.set_title(f"BAHAMAS Aircraft Data - {flight}")
    fig.legend(loc=2)
    fig.autofmt_xdate()
    fig_name = f"{outpath}/{flight}_roll_heading_rad_square.png"
    plt.show()
    # plt.savefig(fig_name)
    # log.info(f"Saved {fig_name}")
    plt.close()
    # %% plot BAHAMAS filtered for high roll angles and solar azimuth angle
    matplotlib.rcdefaults()
    plt.rc('font', size=14)
    plt.rc('lines', linewidth=3)
    plt.rc('font', family="serif")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bahamas_rs_filtered.IRS_HDG.plot(x="time", label="Heading", ax=ax)
    fdw_sim_rs.saa.plot(x="time", label="Solar Azimuth Angle", ax=ax)
    ax.set_ylabel("Angle (°) 0=N")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylim((0, 360))
    ax.grid()
    ax2 = ax.twinx()
    roll = bahamas_rs_filtered.IRS_PHI.plot(x="time", color="#117733", label="Roll", ax=ax2)
    ax2.set_ylabel("Roll Angle (°)")
    ax2.set_ylim((-1.5, 1.5))
    ax.set_title(f"BAHAMAS Aircraft Data and calculated SAA - {flight}")
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(2, roll[0])
    ax.legend(handles=handles, bbox_to_anchor=(0.5, 0), loc="lower center", ncol=3, bbox_transform=fig.transFigure)
    h.set_xticks_and_xlabels(ax, pd.to_timedelta((bahamas_rs_filtered.time[-1] - bahamas_rs_filtered.time[0]).values))
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    fig_name = f"{outpath}/{flight}_roll_heading_saa_rad_square_filtered.png"
    # plt.show()
    plt.savefig(fig_name)
    log.info(f"Saved {fig_name}")
    plt.close()

    # %% plot integrated SMART data and libRadtran simulation for Radiation Square
    plt.rc('font', size=14)
    plt.rc('lines', linewidth=3)
    plt.rc('font', family="serif")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # solar radiation
    smart_fdw_rs_filtered.plot(label=r"$F_{\downarrow}$ SMART integrated (None, None)", ax=ax, c="#117733", ls="-")
    fdw_sim_rs.fdw.plot(x="time", label=r"$F_{\downarrow}$ simulated clear sky", ax=ax, ls="-", c="#332288")
    ax2 = ax.twinx()
    heading = ax2.plot(bahamas_rs["time"], bahamas_rs_filtered["IRS_HDG"], label="Heading")
    saa = fdw_sim_rs.saa.plot(x="time", label="Solar Azimuth Angle", ax=ax2)
    sza = fdw_sim_rs.sza.plot(x="time", label="Solar Zenith Angle", ax=ax2)
    ats = angle_towards_sun.plot(x="time", label="Angle towards Sun", ax=ax2)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Irradiance (W$\,$m$^{-2}$)")
    ax.set_ylim(ylims)  # fix y limits for better comparison (1140, 1170)
    ax2.set_ylabel("Angle (°)")
    h.set_xticks_and_xlabels(ax, pd.to_timedelta((bahamas_rs.time[-1] - bahamas_rs.time[0]).values))
    ax.grid(axis='x')
    ax2.grid()
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0, Patch(color='none', label="Solar (Roll offset, Pitch offset)"))
    handles.insert(3, Patch(color='none'))
    handles.insert(4, heading[0])
    handles.insert(5, saa[0])
    handles.insert(6, sza[0])
    handles.insert(7, ats[0])
    ax.legend(handles=handles, bbox_to_anchor=(0.5, 0), loc="lower center", ncol=2, bbox_transform=fig.transFigure)
    plt.subplots_adjust(bottom=0.37)
    plt.tight_layout()
    fig_name = f"{outpath}/CIRRUS_HL_{flight}_smart_fdw_saa_sza_heading.png"
    # plt.show()
    plt.savefig(fig_name, dpi=100)
    log.info(f"Saved {fig_name}")
    plt.close()

    # %% attitude correct f_dw with no offsets
    smart_fdw_df = pd.DataFrame(smart_fdw).rename(columns={0: "F_down"})  # make dataframe from series
    # interpolate libradtran simulation and bahamas on smart time
    fdw_sim_inp = fdw_sim.interp(time=smart_fdw.index, kwargs=dict(fill_value="extrapolate"))
    bahamas_inp = bahamas.interp(time=smart_fdw.index, kwargs=dict(fill_value="extrapolate"))
    F_down_att_no_offset, factor = fdw_attitude_correction(smart_fdw_df["F_down"].values,
                                                           roll=bahamas_inp.IRS_PHI.values,
                                                           pitch=-bahamas_inp.IRS_THE.values,
                                                           yaw=bahamas_inp.IRS_HDG.values, sza=fdw_sim_inp.sza.values,
                                                           saa=fdw_sim_inp.saa.values,
                                                           fdir=fdw_sim_inp.direct_fraction.values,
                                                           r_off=0, p_off=0)
    smart_fdw_df["F_down_att_no"] = F_down_att_no_offset

    # %% vary roll and pitch offset and attitude correct f_dw
    roll_offset = -1.4
    pitch_offset = 2.9
    F_down_att, factor = fdw_attitude_correction(smart_fdw_df["F_down"].values,
                                                 roll=bahamas_inp.IRS_PHI.values, pitch=-bahamas_inp.IRS_THE.values,
                                                 yaw=bahamas_inp.IRS_HDG.values, sza=fdw_sim_inp.sza.values,
                                                 saa=fdw_sim_inp.saa.values, fdir=fdw_sim_inp.direct_fraction.values,
                                                 r_off=roll_offset, p_off=pitch_offset)

    smart_fdw_df["F_down_att"] = F_down_att
    smart_fdw_df_rs = smart_fdw_df.loc[slice(rs_start, rs_end)]
    smart_fdw_df_rs_filtered = smart_fdw_df_rs.loc[roll_filter_smart.values]

    # %% plot attitude corrected SMART F_dw measurements
    plt.rc('font', size=14)
    plt.rc('lines', linewidth=3)
    plt.rc('font', family="serif")
    zoom = False
    zoom_str = "_zoom" if zoom else ""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # solar radiation
    smart_fdw_df_rs_filtered.F_down.plot(label=r"$F_{\downarrow}$ SMART (None, None)", ax=ax, c="#117733", ls="-")
    # smart_fdw_df_rs_filtered.F_down_att_no.plot(label=r"$F_{\downarrow}$ SMART (0, 0)", ax=ax, c="#D55E00", ls="-")
    smart_fdw_df_rs_filtered.F_down_att.plot(label=r"$F_{\downarrow}$" + f" SMART ({roll_offset}, {pitch_offset})", ls="-", c="#CC6677")
    fdw_sim_rs.fdw.plot(x="time", label=r"$F_{\downarrow}$ simulated clear sky", ax=ax, ls="-", c="#332288")
    ax2 = ax.twinx()
    heading = bahamas_rs_filtered.IRS_HDG.plot(x="time", label="Heading", ax=ax2)
    ats = angle_towards_sun.plot(x="time", label="Angle towards Sun", ax=ax2)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Irradiance (W$\,$m$^{-2}$)")
    if zoom:
        ax.set_ylim((700, 900))
    else:
        ax.set_ylim(ylims)  # fix y limits for better comparison (1140, 1170)
    ax2.set_ylabel("Angle (°)")
    h.set_xticks_and_xlabels(ax, pd.to_timedelta((bahamas_rs.time[-1] - bahamas_rs.time[0]).values))
    ax.grid(axis='x')
    ax2.grid()
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0, Patch(color='none', label="Solar (Roll offset, Pitch offset)"))
    handles.insert(4, Patch(color='none'))
    handles.insert(6, heading[0])
    handles.insert(7, ats[0])
    ax.legend(handles=handles, bbox_to_anchor=(0.5, 0), loc="lower center", ncol=2, bbox_transform=fig.transFigure)
    plt.subplots_adjust(bottom=0.37)
    plt.tight_layout()
    fig_name = f"{outpath}/CIRRUS_HL_{flight}_smart_fdw_saa_heading_new_att_corr_{roll_offset}_{pitch_offset}{zoom_str}.png"
    # plt.show()
    plt.savefig(fig_name, dpi=100)
    log.info(f"Saved {fig_name}")
    plt.close()

    # %% plot difference between simulation and measurement
    fdw_diff = smart_fdw_df_rs.F_down_att - fdw_sim_inp["fdw"].sel(time=slice(rs_start, rs_end))
    fdw_diff_pc = np.abs(fdw_diff / smart_fdw_df_rs.F_down_att) * 100
    fdw_diff_pc_filtered = fdw_diff_pc.loc[roll_filter_smart.values]
    plt.rc('font', size=14)
    plt.rc('lines', linewidth=3)
    plt.rc('font', family="serif")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fdw_diff_pc_filtered.plot(ax=ax)
    ax.set_ylabel("Difference (%)")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylim((0, 4))
    ax.grid()
    ax.set_title(f"Difference between corrected measurement and clear sky simulation\n"
                 f"Roll offset: {roll_offset}; Pitch offset: {pitch_offset}")
    fig_name = f"{outpath}/CIRRUS_HL_{flight}_difference_smart_cor_simulation_{roll_offset}_{pitch_offset}.png"
    # plt.show()
    plt.savefig(fig_name, dpi=100)
    log.info(f"Saved {fig_name}")
    plt.close()
    # %% plot 800nm SMART data and libRadtran simulation for Radiation Square
    plt.rc('font', size=14)
    plt.rc('lines', linewidth=3)
    plt.rc('font', family="serif")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # solar radiation
    smart_800_rs_filtered.plot(label=r"$F_{\downarrow}$ SMART 800nm (None, None)", ax=ax, c="#117733", ls="-")
    fdw_800_sim_rs.fdw.plot(x="time", label=r"$F_{\downarrow}$ simulated clear sky 800nm", ax=ax, ls="-", c="#332288")
    ax2 = ax.twinx()
    heading = ax2.plot(bahamas_rs["time"], bahamas_rs_filtered["IRS_HDG"], label="Heading")
    saa = fdw_sim_rs.saa.plot(x="time", label="Solar Azimuth Angle", ax=ax2)
    sza = fdw_sim_rs.sza.plot(x="time", label="Solar Zenith Angle", ax=ax2)
    ats = angle_towards_sun.plot(x="time", label="Angle towards Sun", ax=ax2)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Spectral Irradiance (W$\,$m$^{-2}\,$nm$^{-1}$)")
    ax.set_ylim((0.5, 1))  # fix y limits for better comparison (0.5, 1)
    ax2.set_ylabel("Angle (°)")
    h.set_xticks_and_xlabels(ax, pd.to_timedelta((bahamas_rs.time[-1] - bahamas_rs.time[0]).values))
    ax.grid(axis='x')
    ax2.grid()
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0, Patch(color='none', label="Solar (Roll offset, Pitch offset)"))
    handles.insert(3, Patch(color='none'))
    handles.insert(4, heading[0])
    handles.insert(5, saa[0])
    handles.insert(6, sza[0])
    handles.insert(7, ats[0])
    ax.legend(handles=handles, bbox_to_anchor=(0.5, 0), loc="lower center", ncol=2, bbox_transform=fig.transFigure)
    plt.subplots_adjust(bottom=0.37)
    plt.tight_layout()
    fig_name = f"{outpath}/CIRRUS_HL_{flight}_smart_fdw_800nm_saa_sza_heading.png"
    # plt.show()
    plt.savefig(fig_name, dpi=100)
    log.info(f"Saved {fig_name}")
    plt.close()

    # %% attitude correct f_dw with no offsets
    smart_800_df = pd.DataFrame(smart_800).rename(columns={740: "F_down"})  # make dataframe from series
    # interpolate libradtran simulation and bahamas on smart time
    fdw_800_sim_inp = fdw_800_sim.interp(time=smart_800.index, kwargs=dict(fill_value="extrapolate"))
    bahamas_inp = bahamas.interp(time=smart_800.index, kwargs=dict(fill_value="extrapolate"))
    F_down_att_no_offset, factor = fdw_attitude_correction(smart_800_df["F_down"].values,
                                                           roll=bahamas_inp.IRS_PHI.values,
                                                           pitch=-bahamas_inp.IRS_THE.values,
                                                           yaw=bahamas_inp.IRS_HDG.values, sza=fdw_800_sim_inp.sza.values,
                                                           saa=fdw_800_sim_inp.saa.values,
                                                           fdir=fdw_800_sim_inp.direct_fraction.values,
                                                           r_off=0, p_off=0)
    smart_800_df["F_down_att_no"] = F_down_att_no_offset

    # %% vary roll and pitch offset and attitude correct f_dw
    roll_offset = -1.4
    pitch_offset = 2.9
    F_down_att, factor = fdw_attitude_correction(smart_800_df["F_down"].values,
                                                 roll=bahamas_inp.IRS_PHI.values, pitch=-bahamas_inp.IRS_THE.values,
                                                 yaw=bahamas_inp.IRS_HDG.values, sza=fdw_800_sim_inp.sza.values,
                                                 saa=fdw_800_sim_inp.saa.values, fdir=fdw_800_sim_inp.direct_fraction.values,
                                                 r_off=roll_offset, p_off=pitch_offset)

    smart_800_df["F_down_att"] = F_down_att
    smart_800_df_rs = smart_800_df.loc[slice(rs_start, rs_end)]
    smart_800_df_rs_filtered = smart_800_df_rs.loc[roll_filter_smart.values]

    # %% plot attitude corrected SMART F_dw measurements
    plt.rc('font', size=14)
    plt.rc('lines', linewidth=3)
    plt.rc('font', family="serif")
    zoom = False
    zoom_str = "_zoom" if zoom else ""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # solar radiation
    smart_800_df_rs_filtered.F_down.plot(label=r"$F_{\downarrow}$ SMART 800nm (None, None)", ax=ax, c="#117733", ls="-")
    # smart_fdw_df_rs_filtered.F_down_att_no.plot(label=r"$F_{\downarrow}$ SMART (0, 0)", ax=ax, c="#D55E00", ls="-")
    smart_800_df_rs_filtered.F_down_att.plot(label=r"$F_{\downarrow}$" + f" SMART 800nm ({roll_offset}, {pitch_offset})",
                                             ls="-", c="#CC6677")
    fdw_800_sim_rs.fdw.plot(x="time", label=r"$F_{\downarrow}$ simulated clear sky 800nm", ax=ax, ls="-", c="#332288")
    ax2 = ax.twinx()
    heading = bahamas_rs_filtered.IRS_HDG.plot(x="time", label="Heading", ax=ax2)
    ats = angle_towards_sun.plot(x="time", label="Angle towards Sun", ax=ax2)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Spectral Irradiance (W$\,$m$^{-2}\,$nm$^{-1}$)")
    if zoom:
        ax.set_ylim((700, 900))
    else:
        ax.set_ylim((0.5, 1))  # fix y limits for better comparison (0.5, 1)
    ax2.set_ylabel("Angle (°)")
    h.set_xticks_and_xlabels(ax, pd.to_timedelta((bahamas_rs.time[-1] - bahamas_rs.time[0]).values))
    ax.grid(axis='x')
    ax2.grid()
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0, Patch(color='none', label="Solar (Roll offset, Pitch offset)"))
    handles.insert(4, Patch(color='none'))
    handles.insert(6, heading[0])
    handles.insert(7, ats[0])
    ax.legend(handles=handles, bbox_to_anchor=(0.5, 0), loc="lower center", ncol=2, bbox_transform=fig.transFigure)
    plt.subplots_adjust(bottom=0.37)
    plt.tight_layout()
    fig_name = f"{outpath}/CIRRUS_HL_{flight}_smart_800nm_saa_heading_new_att_corr_{roll_offset}_{pitch_offset}{zoom_str}.png"
    # plt.show()
    plt.savefig(fig_name, dpi=100)
    log.info(f"Saved {fig_name}")
    plt.close()

    # %% plot difference between simulation and measurement
    fdw_diff = smart_800_df_rs.F_down_att - fdw_800_sim_inp["fdw"].sel(time=slice(rs_start, rs_end))
    fdw_diff_pc = np.abs(fdw_diff / smart_800_df_rs.F_down_att) * 100
    fdw_diff_pc_filtered = fdw_diff_pc.loc[roll_filter_smart.values]
    plt.rc('font', size=14)
    plt.rc('lines', linewidth=3)
    plt.rc('font', family="serif")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fdw_diff_pc_filtered.plot(ax=ax)
    ax.set_ylabel("Difference (%)")
    ax.set_xlabel("Time (UTC)")
    # ax.set_ylim((0, 4))
    ax.grid()
    ax.set_title(f"Difference between corrected measurement and clear sky simulation (800nm)\n"
                 f"Roll offset: {roll_offset}; Pitch offset: {pitch_offset}")
    fig_name = f"{outpath}/CIRRUS_HL_{flight}_difference_smart_800nm_cor_simulation_{roll_offset}_{pitch_offset}.png"
    # plt.show()
    plt.savefig(fig_name, dpi=100)
    log.info(f"Saved {fig_name}")
    plt.close()
