#!/usr/bin/env python
"""Find correction offsets for SMART

During RF18 the SMART INS system had an error and the stabilization had to be fixed to keep on measuring.
Looking at the stabilization table measurements one can see the values start to drift after the stabilization was turned on again after the radar calibration maneuvers.

.. _stabbi-performance-pitch:

.. figure:: figures/HALO-AC3_20220412_HALO_RF18_stabbi_performance_pitch.png

The stabbi was turned on again at 10:13:22. And shortly after (10:35 UTC) the stabilization table was fixed to the following position:

* Roll **0.07**
* Pitch **-2.38**

This should account for the usual position of the aircraft in flight. (HALO's nose is usually pointing a bit up during flight thus the table needs to counter this)

Looking at the BAHAMAS data filtered for roll angles greater 1 we can see that leg two in the polygon shows some high roll angles.
:numref:`bahamas-overview` also shows that the heading did not stay constant during each leg.

.. _bahamas-overview:

.. figure:: figures/HALO-AC3_20220412_HALO_RF18_BAHAMAS_roll_heading_saa_altitude_rad_square_filtered.png

    BAHAMAS aircraft data filtered for roll angles greater 1 and the calculated solar azimuth angle (first row).
    BAHAMAS altitude (second row).

During the time when SMART was fixed in position the incoming direct solar irradiance can be corrected for the aircraft attitude.
This correction can take an offset between the fixed position of SMART and the aircraft fuselage into account.
To figure out this possible offset one can use a so-called radiation square where HALO flies into four different direction.
Ideally HALO would fly once towards the direction of the sun and then do three 90 degree turns.
From this pattern one can figure out the offset.
However, here we only have a polygon pattern at hand with less than 90 degree turns and also did not fly directly into the direction of the sun (see :numref:`angle-towards-sun`).

.. _angle-towards-sun:

.. figure:: figures/HALO-AC3_20220412_HALO_RF18_BAHAMAS_sun-angle_filtered.png

    Angle of HALO towards sun during the polygon pattern and BAHAMAS heading (HDG) and solar azimuth angle (SAA).

Using the Jupyter notebook ``halo_ac3_20220412_radiation_square_investiagtion.ipynb`` different offsets can be tried out using an interactive widget.
Going through different offsets the fit to the simulation does not get better than using no offset.
Thus, the inlet seems to be well aligned with the horizontal plane of the aircraft and **no offset** is applied to the attitude correction.
Some plots can be found in ``.../case_studies/HALO-AC3_20220412_HALO_RF18/``.

*author*: Johannes Röttenbacher
"""

if __name__ == "__main__":
    # %% import modules
    import pylim.helpers as h
    import pylim.halo_ac3 as meta
    from pylim import reader
    import ac3airborne
    from ac3airborne.tools import flightphase
    import xarray as xr
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    import logging

    log = logging.getLogger(__name__)
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)

    # %% set paths and options
    campaign = "halo-ac3"
    flight_key = "RF18"
    flight = meta.flight_names[flight_key]
    date = flight[9:17]

    smart_dir = h.get_path("calibrated", flight, campaign)
    bahamas_dir = h.get_path("bahamas", flight, campaign)
    libradtran_dir = h.get_path("libradtran", flight, campaign)
    plot_dir = f"{h.get_path('plot', flight, campaign)}/{flight}"
    # file names
    smart_file = f"HALO-AC3_HALO_SMART_spectral-irradiance-Fdw_{date}_{flight_key}_v1.0.nc"
    bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{flight_key}_v1.nc"
    libradtran_file = f"HALO-AC3_HALO_libRadtran_clearsky_simulation_smart_spectral_{date}_{flight_key}.nc"

    # %% flight tracks from halo-ac3 cloud
    kwds = {'simplecache': dict(cache_storage='E:/HALO-AC3/cloud', same_names=True)}
    credentials = {"user": os.environ.get("AC3_CLOUD_USER"), "password": os.environ.get("AC3_CLOUD_PASSWORD")}
    cat = ac3airborne.get_intake_catalog()
    ins = cat["HALO-AC3"]["HALO"]["GPS_INS"][f"HALO-AC3_HALO_{flight_key}"](storage_options=kwds, **credentials).to_dask()

    # %% get flight segmentation and select below and above cloud section
    segments = flightphase.FlightPhaseFile(
        ac3airborne.get_flight_segments()["HALO-AC3"]["HALO"][f"HALO-AC3_HALO_{flight_key}"])
    above_cloud, below_cloud = dict(), dict()
    above_cloud["start"] = segments.select("name", "polygon pattern 1")[0]["start"]
    above_cloud["end"] = segments.select("name", "polygon pattern 1")[0]["parts"][-1]["start"]
    below_cloud["start"] = segments.select("name", "polygon pattern 2")[0]["start"]
    below_cloud["end"] = segments.select("name", "polygon pattern 2")[0]["end"]
    above_slice = slice(above_cloud["start"], above_cloud["end"])
    below_slice = slice(below_cloud["start"], below_cloud["end"])
    polygon_pattern = flightphase.FlightPhaseFile(segments.select("name", "polygon pattern 1")[0]["parts"])

    # %% read in data and select only time of first polygon pattern
    smart_ds = xr.open_dataset(f"{smart_dir}/{smart_file}").sel(time=above_slice)
    bahamas_ds = reader.read_bahamas(f"{bahamas_dir}/{bahamas_file}").sel(time=above_slice)
    sim_ds = xr.open_dataset(f"{libradtran_dir}/{libradtran_file}").sel(time=above_slice)

    # %% get information from BAHAMAS data
    time_extend = pd.to_timedelta((bahamas_ds.time[-1] - bahamas_ds.time[0]).values)
    roll_filter = np.abs(bahamas_ds["IRS_PHI"]) < 1
    bahamas_ds["sun_angle"] = np.abs(bahamas_ds.IRS_HDG - sim_ds.saa.interp_like(bahamas_ds.time, kwargs=dict(fill_value="extrapolate")))  # calculate angle towards sun

    # %% plot BAHAMAS data for Radiation Square
    h.set_cb_friendly_colors()
    plt.rc('font', size=14)
    plt.rc('lines', linewidth=3)
    plt.rc('font', family="serif")
    fig, axs = plt.subplots(figsize=(10,6))
    axs.plot(bahamas_ds["time"], bahamas_ds["IRS_HDG"], label="Heading")
    axs.set_ylabel("Heading (°) 0=N")
    axs.set_xlabel("Time (UTC)")
    axs.set_ylim((0, 360))
    axs.grid()
    axs2 = axs.twinx()
    axs2.plot(bahamas_ds["time"], bahamas_ds["IRS_PHI"], label="Roll", c="#117733")
    axs2.set_ylabel("Roll Angle (°)")
    axs2.set_ylim((-1.5, 1.5))
    axs.set_title(f"BAHAMAS Aircraft Data - {flight}")
    h.set_xticks_and_xlabels(axs, time_extend)
    fig.legend(loc=2)
    # fig_name = f"{plot_dir}/{flight}_roll_heading_rad_square.png"
    # plt.savefig(fig_name)
    # log.info(f"Saved {fig_name}")
    plt.show()
    plt.close()

    # %% plot BAHAMAS filtered for high roll angles and solar azimuth angle
    bahamas_ds_filtered = bahamas_ds.where(roll_filter)
    plt.rc('font', size=14)
    plt.rc('lines', linewidth=3)
    plt.rc('font', family="serif")

    fig, axs = plt.subplots(2, figsize=(10, 6))
    ax = axs[0]
    bahamas_ds_filtered.IRS_HDG.plot(x="time", label="Heading", ax=ax)
    sim_ds.saa.plot(x="time", label="Solar Azimuth Angle", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Angle (°) 0=N")
    ax.set_ylim((0, 360))
    ax.grid()

    # second axis on first row: Roll angle
    ax2 = ax.twinx()
    roll = bahamas_ds_filtered.IRS_PHI.plot(x="time", color="#117733", label="Roll", ax=ax2)
    ax2.set_ylabel("Roll Angle (°)")
    ax2.set_ylim((-1.5, 1.5))
    ax.set_title(f"BAHAMAS Aircraft Data and calculated SAA\n {flight}")
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(2, roll[0])
    ax.legend(handles=handles, bbox_to_anchor=(0.5, 0.45), loc="lower center", ncol=3, bbox_transform=fig.transFigure)
    ax.set_xticklabels("")

    # second row altitude
    ax = axs[1]
    altitude = bahamas_ds_filtered.IRS_ALT / 1000
    ax.plot(altitude.time, altitude, label="Altitude", color="#332288")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Altitude (km)")
    ax.grid()
    h.set_xticks_and_xlabels(ax, time_extend)
    plt.tight_layout()

    fig_name = f"{plot_dir}/{flight}_BAHAMAS_roll_heading_saa_altitude_rad_square_filtered.png"
    plt.savefig(fig_name)
    log.info(f"Saved {fig_name}")
    plt.show()
    plt.close()

    # %% plot angle towards sun
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bahamas_ds_filtered.time, bahamas_ds_filtered["sun_angle"], label="Angle to Sun")
    ax.plot(bahamas_ds_filtered.time, bahamas_ds_filtered["IRS_HDG"], label="Heading")
    ax.plot(sim_ds.time, sim_ds.saa, label="Solar Azimuth Angle")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Angle (°)")
    ax.set_title(f"BAHAMAS Angle Towards Sun (HDG - SAA)\n {flight}")
    ax.legend()
    ax.grid()
    h.set_xticks_and_xlabels(ax, time_extend)
    plt.tight_layout()
    figname = f"{plot_dir}/{flight}_BAHAMAS_sun-angle_filtered.png"
    plt.savefig(figname, dpi=100)
    log.info(f"Saved {figname}")
    plt.show()
    plt.close()
