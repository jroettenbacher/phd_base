#!/usr/bin/env python
"""

| *author*: Johannes Röttenbacher
| *created*: 25-10-2022

BACARDI measurement investigation

Using RF17 the measurement performance of BACARDI is investigated.
The data is filtered using motion angles measured by BAHAMAS.
As the roll angle is centered around 0 (median for RF17: 0.0571659 deg) a simple roll threshold of **0.5** is used here and implemented by comparing the absolute roll angle with the threshold.
The pitch angle of HALO, however, is centered on 2.28 (median for RF17: 2.28605413 deg) and thus the absolute value of the measurement minus the center is compared with the pitch threshold of **0.34**.
The pitch threshold is set by iterating the polar plot showing the relation between the BACARDI measured downward irradiance and the libRadtran clearsky simulation according to viewing angle of HALO (see :numref:`bacardi-polar`).
There is one point in the dataset which shows an abnormal low relation.
This point is removed when the pitch threshold is set to 0.34.
It can thus be said that the low relation value is caused by a pitch motion of HALO.

.. _bacardi-polar:

.. figure:: figures/HALO-AC3_20220411_HALO_RF17_BACARDI_inlet_directional_dependence.png

    Relation between BACARDI measurement and libRadtran clearsky simulation depending on viewing direction of HALO.
    BACARDI data is filtered for aircraft motion.
    Red dots denote the section when HALO was below a cirrus cloud.

Plotting the same relation against the solar zenith angle (:numref:`bacardi-relation-sza-all`) shows that there is some part of the flight when the relation exceeds 1 and HALO is above any cloud.
However, due to the below cloud section not many details can be observed.

.. _bacardi-relation-sza-all:

.. figure:: figures/HALO-AC3_20220411_HALO_RF17_BACARDI_inlet_sza_dependence.png

    Relation between BACARDI measurement and libRadtran clearsky simulation depending on solar zenith angle.
    BACARDI data is filtered for aircraft motion.
    Blue dots denote the section when HALO was below a cirrus cloud.

More details can be seen in :numref:`bacardi-relation-sza`.
Here a maximum deviation of 3% can be observed, which is not much considering the motion thresholds.

.. _bacardi-relation-sza:

.. figure:: figures/HALO-AC3_20220411_HALO_RF17_BACARDI_inlet_sza_dependence_without_below_cloud.png

    Relation between BACARDI measurement and libRadtran clearsky simulation depending on solar zenith angle.
    BACARDI data is filtered for aircraft motion and the below cloud section.

Looking at the time evolution of the relation (:numref:`bacardi-relation-time`) we can see that the high deviations happen close to times when the data is filtered due to high motion angles and just before the start of the below cloud section.

.. _bacardi-relation-time:

.. figure:: figures/HALO-AC3_20220411_HALO_RF17_BACARDI_simulation_relation_without_below_cloud.png

    Relation between BACARDI measurement and libRadtran clearsky simulation depending on time.
    BACARDI data is filtered for aircraft motion and the below cloud section.

The same analysis can be done using the whole BACARDI |haloac3| data set using only the above cloud measurements.


"""

if __name__ == "__main__":
# %% import modules
    import pylim.helpers as h
    from pylim import reader
    from pylim.bahamas import preprocess_bahamas
    import ac3airborne
    from ac3airborne.tools import flightphase
    import xarray as xr
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import colors
    import seaborn as sns
    import os
    import cmasher as cmr

    cbc = h.get_cb_friendly_colors()

# %% set paths
    campaign = "halo-ac3"
    date = "20220411"
    halo_key = "RF17"
    halo_flight = f"HALO-AC3_{date}_HALO_{halo_key}"

    plot_path = f"{h.get_path('plot', halo_flight, campaign)}/{halo_flight}"
    libradtran_path = h.get_path("libradtran", halo_flight, campaign)
    libradtran_spectral = f"HALO-AC3_HALO_libRadtran_clearsky_simulation_smart_spectral_{date}_{halo_key}.nc"
    libradtran_bb_solar = f"HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_solar_{date}_{halo_key}.nc"
    libradtran_bb_thermal = f"HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_thermal_{date}_{halo_key}.nc"
    libradtran_bb_solar_si = f"HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_solar_si_{date}_{halo_key}.nc"
    libradtran_bb_thermal_si = f"HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_thermal_si_{date}_{halo_key}.nc"
    bahamas_path = h.get_path("bahamas", halo_flight, campaign)
    # bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{halo_key}_v1_1s.nc"
    bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{halo_key}_v1.nc"
    bacardi_path = h.get_path("bacardi", halo_flight, campaign)
    # bacardi_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{halo_key}_R1_1s.nc"
    bacardi_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{halo_key}_R1.nc"

# %% get flight segmentation and select below and above cloud section
    meta = ac3airborne.get_flight_segments()["HALO-AC3"]["HALO"][f"HALO-AC3_HALO_{halo_key}"]
    segments = flightphase.FlightPhaseFile(meta)
    above_cloud, below_cloud = dict(), dict()
    if "RF17" in halo_flight:
        above_cloud["start"] = segments.select("name", "high level 7")[0]["start"]
        above_cloud["end"] = segments.select("name", "high level 8")[0]["end"]
        below_cloud["start"] = segments.select("name", "high level 9")[0]["start"]
        below_cloud["end"] = segments.select("name", "high level 10")[0]["end"]
        above_slice = slice(above_cloud["start"], above_cloud["end"])
        below_slice = slice(below_cloud["start"], below_cloud["end"])
    else:
        above_cloud["start"] = segments.select("name", "polygon pattern 1")[0]["start"]
        above_cloud["end"] = segments.select("name", "polygon pattern 1")[0]["parts"][-1]["start"]
        below_cloud["start"] = segments.select("name", "polygon pattern 2")[0]["start"]
        below_cloud["end"] = segments.select("name", "polygon pattern 2")[0]["end"]
        above_slice = slice(above_cloud["start"], above_cloud["end"])
        below_slice = slice(below_cloud["start"], below_cloud["end"])

# %% read in libRadtran simulation
    spectral_sim = xr.open_dataset(f"{libradtran_path}/{libradtran_spectral}")
    bb_sim_solar = xr.open_dataset(f"{libradtran_path}/{libradtran_bb_solar}")
    bb_sim_thermal = xr.open_dataset(f"{libradtran_path}/{libradtran_bb_thermal}")
    bb_sim_solar_si = xr.open_dataset(f"{libradtran_path}/{libradtran_bb_solar_si}")
    bb_sim_thermal_si = xr.open_dataset(f"{libradtran_path}/{libradtran_bb_thermal_si}")

# %% read in BACARDI and BAHAMAS data and resample to 1 sec
    bacardi_ds = xr.open_dataset(f"{bacardi_path}/{bacardi_file}")
    # bahamas_ds = xr.open_dataset(f"{bahamas_path}/{bahamas_file}")
    bahamas_ds = reader.read_bahamas(f"{bahamas_path}/{bahamas_file}")
    # bahamas_ds = reader.read_bahamas(f"{bahamas_path}/{bahamas_file}")
    # bacardi_ds = bacardi_ds.resample(time="1S").mean()
    # bahamas_ds = bahamas_ds.resample(time="1S").mean()
    # bacardi_ds.to_netcdf(f"{bacardi_path}/{bacardi_file.replace('.nc', '_1s.nc')}")
    # bahamas_ds.to_netcdf(f"{bahamas_path}/{bahamas_file.replace('.nc', '_1s.nc')}")

# %% remove landing and final parts of the flight due to very low sun
    time_slice = slice(bahamas_ds.time[0], pd.Timestamp(2022, 4, 11, 15, 30))
    bahamas_ds = bahamas_ds.sel(time=time_slice)
    bacardi_ds = bacardi_ds.sel(time=time_slice)

# %% filter values which exceeded certain motion threshold
    roll_center = np.abs(bahamas_ds["IRS_PHI"].median())
    roll_threshold = 0.5
    pitch_center = np.abs(bahamas_ds["IRS_THE"].median())
    pitch_threshold = 0.34
    # True -> keep value, False -> drop value (Nan)
    roll_filter = np.abs(bahamas_ds["IRS_PHI"]) < roll_threshold
    # pitch is not centered on 0 thus we need to calculate the difference to the center and compare that to the threshold
    pitch_filter = np.abs(bahamas_ds["IRS_THE"] - pitch_center) < pitch_threshold
    motion_filter = roll_filter & pitch_filter
    bacardi_ds = bacardi_ds.where(motion_filter)

# %% Relation of BACARDI to simulation depending on viewing angle of HALO
    relation_bacardi_libradtran = bacardi_ds["F_down_solar"] / bacardi_ds["F_down_solar_sim"]
# %% viewing direction of halo: 0 = towards sun, 180 = away from sun
    heading = bahamas_ds.IRS_HDG
    viewing_dir = bacardi_ds.saa - heading
    viewing_dir = viewing_dir.where(viewing_dir > 0, viewing_dir + 360)

# %% merge information in dataframe
    df1 = viewing_dir.to_dataframe(name="viewing_dir")
    df2 = relation_bacardi_libradtran.to_dataframe(name="relation")
    df = df1.merge(df2, on="time")
    df["sza"] = bacardi_ds.sza
    df["roll"] = bahamas_ds["IRS_PHI"].where(motion_filter)
    # df = df[df.relation > 0.7]
    df = df.sort_values(by="viewing_dir")

# %% plotting aesthetics
    time_extend = pd.to_timedelta(
        (bahamas_ds.time[-1] - bahamas_ds.time[0]).values)  # get time extend for x-axis labeling
    time_extend_cs = below_cloud["end"] - above_cloud["start"]  # time extend for case study

# %% plot relation between BACARDI measurement and simulation depending on viewing angle as polarplot
    h.set_cb_friendly_colors()
    plt.rc("font", size=12, family="serif")
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': 'polar'})
    ax.scatter(np.deg2rad(df["viewing_dir"]), df["relation"], label="0 = facing sun\n180 = facing away from sun")
    df_plot = df[((below_cloud["start"] < df.index) & (df.index < below_cloud["end"]))]
    ax.scatter(np.deg2rad(df_plot["viewing_dir"]), df_plot["relation"], label="below cloud")
    ax.set_rmax(1.2)
    ax.set_rticks([0.8, 1, 1.2])  # Less radial ticks
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2)
    ax.grid(True)
    ax.set_title("Relation between BACARDI Fdw measurement and libRadtran simulation\n"
                 " according to viewing direction of HALO with respect to the sun")
    ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    figname = f"{plot_path}/{halo_flight}_BACARDI_inlet_directional_dependence.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot relation as function of SZA
    h.set_cb_friendly_colors()
    plt.rc("font", size=14, family="serif")
    _, ax = plt.subplots(figsize=(10, 6))
    df_tmp = df[((below_cloud["start"] < df.index) & (df.index < below_cloud["end"]))]
    ax.scatter(df_tmp["sza"], df_tmp["relation"], label="below cloud")
    df_tmp = df[~((below_cloud["start"] < df.index) & (df.index < below_cloud["end"]))]
    ax.scatter(df_tmp["sza"], df_tmp["relation"])
    ax.grid()
    ax.set_xlabel("Solar Zenith Angle (deg)")
    ax.set_ylabel("Relation")
    ax.set_title("Relation between BACARDI Fdw measurement and libRadtran simulation\n"
                 " in relation to solar zenith angle", size=16)
    ax.legend()
    plt.tight_layout()
    figname = f"{plot_path}/{halo_flight}_BACARDI_inlet_sza_dependence.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot relation as function of SZA, exclude below cloud section
    h.set_cb_friendly_colors()
    plt.rc("font", size=14, family="serif")
    _, ax = plt.subplots(figsize=(10, 6))
    df_tmp = df[~((below_cloud["start"] < df.index) & (df.index < below_cloud["end"]))]
    ax.scatter(df_tmp["sza"], df_tmp["relation"])
    ax.grid()
    ax.set_xlabel("Solar Zenith Angle (deg)")
    ax.set_ylabel("Relation")
    ax.set_title("Relation between BACARDI Fdw measurement and libRadtran simulation\n"
                 " in relation to solar zenith angle (excluding below cloud section)", size=16)
    # ax.legend()
    plt.tight_layout()
    figname = f"{plot_path}/{halo_flight}_BACARDI_inlet_sza_dependence_without_below_cloud.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot relation as function of time (exclude below cloud section)
    h.set_cb_friendly_colors()
    plt.rc("font", size=14, family="serif")
    plot_df = df.sort_values(by="time")
    _, ax = plt.subplots(figsize=(10, 6))
    plot_df = plot_df[~((below_cloud["start"] < plot_df.index) & (plot_df.index < below_cloud["end"]))]
    ax.scatter(plot_df.index, plot_df["relation"])
    ax.grid()
    h.set_xticks_and_xlabels(ax, time_extend)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Relation")
    ax.set_title("Relation between BACARDI Fdw measurement and libRadtran simulation\n"
                 "(excluding below cloud)", size=16)
    plt.tight_layout()
    figname = f"{plot_path}/{halo_flight}_BACARDI_simulation_relation_without_below_cloud.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% dig into measurements with relation > 1
    relation_filter = df.sort_values(by="time")["relation"] > 1
    plot_ds = bahamas_ds.where(relation_filter.to_xarray())
    h.set_cb_friendly_colors()
    plt.rc("font", size=12, family="serif")
    _, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(plot_ds.time, plot_ds["IRS_THE"])
    ax.grid()
    h.set_xticks_and_xlabels(ax, time_extend)
    ax.set_xlabel("Time (UTC)")
    plt.tight_layout()
    # figname = f"{plot_path}/{halo_flight}_BACARDI_simulation_relation_without_below_cloud.png"
    # plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% Extension of analysis to the whole data set
    bacardi_all_path = h.get_path("all", campaign=campaign, instrument="BACARDI")
    bahamas_all_path = h.get_path("all", campaign=campaign, instrument="BAHAMAS")
    plot_path = f"{h.get_path('plot', campaign=campaign)}/bacardi_measurement_investigation"
    fig_path = "C:/Users/Johannes/PycharmProjects/phd_base/docs/figures/bacardi_measurement_investigation"

# %% read in data
    all_files = os.listdir(bacardi_all_path)
    all_files.sort()
    all_files = [os.path.join(bacardi_all_path, file) for file in all_files[2:] if file.endswith("JR.nc")]
    bacardi_ds = xr.open_mfdataset(all_files)
    # bahamas
    all_files = [f for f in os.listdir(bahamas_all_path) if f.startswith("HALO")]
    all_files.sort()
    all_files = [os.path.join(bahamas_all_path, file) for file in all_files[1:]]
    bahamas_ds = xr.open_mfdataset(all_files, preprocess=preprocess_bahamas)

# %% calculate relation and deviation between simulated and measured solar downward irradiance
    bacardi_ds["relation"] = bacardi_ds["F_down_solar"] / bacardi_ds["F_down_solar_sim"]
    bacardi_ds["deviation"] = (bacardi_ds["relation"] - 1) * 100

# %% viewing direction of halo: 0 = towards sun, 180 = away from sun
    heading = bahamas_ds.IRS_HDG
    viewing_dir = bacardi_ds.saa - heading
    bacardi_ds["viewing_dir"] = viewing_dir.where(viewing_dir > 0, viewing_dir + 360)

# %% filter data for altitude
    bacardi_ds_fil = bacardi_ds.where(bacardi_ds.alt >= 10000)

# %% plot polar plot with viewing angle and solar zenith angle
    plot_ds = bacardi_ds_fil.isel(time=slice(0, len(bacardi_ds_fil.time), 100))
    h.set_cb_friendly_colors()
    plt.rc("font", size=12)
    fig, ax = plt.subplots(figsize=h.figsize_wide, subplot_kw={'projection': 'polar'})
    scatter = ax.scatter(np.deg2rad(plot_ds["viewing_dir"]), plot_ds["sza"],
                         c=plot_ds["relation"], s=1,
                         cmap=cmr.pride, norm=colors.CenteredNorm(vcenter=1, halfrange=0.1))
    fig.colorbar(scatter,
                 label="$F^{\downarrow}_{solar}$ Ratio (Observed/Simulated)",
                 format="%.2f",
                 extend="both")
    ax.set_rmax(91)
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2)
    ax.set_rorigin(55)
    ax.set_rticks([60, 65, 70, 75, 80, 85, 90])

    plt.tight_layout()
    figname = f"{plot_path}/BACARDI_vs_simulation_sza_polar.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot polar plot with viewing angle and solar zenith angle and deviation from simulation in percent
    plot_ds = bacardi_ds_fil.isel(time=slice(0, len(bacardi_ds_fil.time), 100))
    h.set_cb_friendly_colors()
    plt.rc("font", size=12)
    fig, ax = plt.subplots(figsize=h.figsize_wide, subplot_kw={'projection': 'polar'})
    scatter = ax.scatter(np.deg2rad(plot_ds["viewing_dir"]), plot_ds["sza"],
                         c=plot_ds["deviation"], s=1,
                         cmap=cmr.pride, norm=colors.CenteredNorm(vcenter=0, halfrange=10))
    fig.colorbar(scatter,
                 label="$F^{\downarrow}_{solar}$ Deviation from simulation (%)",
                 format="%2.1f",
                 extend="both")
    ax.set_rmax(91)
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2)
    ax.set_rorigin(55)
    ax.set_rticks([60, 65, 70, 75, 80, 85, 90])

    plt.tight_layout()
    figname = f"{plot_path}/BACARDI_vs_simulation_sza_polar_percent.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot deviation from simulation as function of solar zenith angle
    for key, value in meta.flight_names.items():
        try:
            date = value[9:17]
            plot_ds = bacardi_ds_fil.sel(time=date)
            h.set_cb_friendly_colors()
            plt.rc("font", size=12)
            fig, ax = plt.subplots(figsize=h.figsize_wide)
            scatter = ax.scatter(plot_ds["sza"], plot_ds["deviation"],
                                 c=plot_ds["viewing_dir"],
                                 s=2,
                                 cmap=cmr.horizon)
            cbar = fig.colorbar(scatter,
                                label="Viewing direction (deg)",
                                format="%3.0f",
                                extend="both")
            ax.set(ylim=(-25, 25),
                   ylabel="$F^{\downarrow}_{solar}$ Deviation from simulation (%)",
                   xlabel="Solar zenith angle (deg)",
                   title=value)
            ax.grid()
            plt.tight_layout()
            figname = f"{plot_path}/{value}_BACARDI_vs_simulation_relation_sza_percent.png"
            plt.savefig(figname, dpi=300)
            plt.show()
            plt.close()
        except KeyError:
            pass

# %% plot deviation from simulation as function of solar zenith angle - all flights
    date = value[9:17]
    plot_ds = bacardi_ds_fil
    h.set_cb_friendly_colors()
    plt.rc("font", size=12)
    fig, ax = plt.subplots(figsize=h.figsize_wide)
    scatter = ax.scatter(plot_ds["sza"], plot_ds["deviation"],
                         c=plot_ds["viewing_dir"],
                         s=2,
                         cmap=cmr.horizon)
    fig.colorbar(scatter,
                 label="Viewing direction (deg)",
                 format="%3.0f",
                 extend="both")
    ax.set(ylim=(-25, 25),
           ylabel="$F^{\downarrow}_{solar}$ Deviation from simulation (%)",
           xlabel="Solar zenith angle (deg)")
    ax.grid()
    plt.tight_layout()
    figname = f"{plot_path}/BACARDI_vs_simulation_relation_sza_percent.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% make statistics of deviation
    plot_ds = bacardi_ds_fil.deviation.where(bacardi_ds_fil.sza < 81)
    plot_ds = plot_ds.where(np.abs(plot_ds) < 10)
    sns.histplot(plot_ds, stat="density", color=cbc[1], kde=True)
    plt.axvline(plot_ds.mean(), ls="--")
    plt.show()
    plt.close()


# %% testing
    bacardi_ds_fil["relation"].where(np.abs(bacardi_ds_fil["relation"] - 1) < 0.2).plot(x="time")
    bacardi_ds_fil["F_down_solar"].where(bacardi_ds_fil["F_down_solar"] > 300).plot(x="time")
    plt.show()
    plt.close()
