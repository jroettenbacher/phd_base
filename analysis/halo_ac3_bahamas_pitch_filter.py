#!/usr/bin/env python
"""Investigate pitch angle of HALO measured by BAHAMAS during |haloac3|

HALO's pitch angle changes during the flight as it loses weight by burning fuel or when it changes altitude and speed.
Using a static median pitch value to calculate a pitch threshold for motion filtering measurements is thus not sufficient.
This can be seen in :numref:`pitch-all`, where the ten-minute median pitch angle decreases from 2.8° to 1.8° during RF18.
Using the static pitch thresholds would thus filter the wrong values.
Especially, the above cloud section during the pentagram pattern in the far north, denoted as case study, suffers from this static threshold.
However, :numref:`pitch-all` already shows a problem with a rolling median: During large ascents and descents the median follows the extreme values.
One could either use an absolute threshold for such extreme values or look at the change of pitch over time to filter such events.
The flight segmentation could also help to remove these events.
Using a longer time window also shows good results.

.. _pitch-all:

.. figure:: figures/bahamas_pitch/HALO-AC3_20220412_HALO_RF18_BAHAMAS_pitch_angle_all.png

    BAHAMAS pitch angle during RF18 with whole flight median and 10 minute rolling median.
    Red dots show the roll filtered values.

**Which time range is best for the median?**

.. _median-time:

.. figure:: figures/bahamas_pitch/HALO-AC3_20220412_HALO_RF18_BAHAMAS_pitch_angle_rolling_median.png

    Rolling median of pitch using different time windows.

Judging from RF18 30 minutes seem to be a good time window for the rolling median.
Since it is mostly longer than large ascents and descents they are still averaged over.
For case study like periods, however, it still follows the general trend of the data.

**What threshold is best for the pitch?**

To answer this we can have a look at the rolling standard deviation of the pitch in :numref:`rolling-std`.

.. _rolling-std:

.. figure:: figures/bahamas_pitch/HALO-AC3_20220412_HALO_RF18_BAHAMAS_pitch_angle_all_rolling_std.png

    Pitch and 30-minute rolling median pitch with standard deviation.

We see that the rolling standard deviation is actually quite small during time when HALO is flying steady.
The median value of the 30-minute standard deviation is :math:`0.173^{\circ}`.
Thus, :math:`0.17^{\circ}` looks like a good threshold then.
First result of calculating a dynamic pitch threshold can be seen in :numref:`dynamic-pitch`.

.. _dynamic-pitch:

.. figure:: figures/bahamas_pitch/HALO-AC3_20220412_HALO_RF18_BAHAMAS_pitch_angle_all_dynamic_threshold.png

    Different dynamic pitch thresholds for a 30-minute rolling median.

From this we can see that the :math:`0.5^{\circ}` threshold from the roll angle is too high for the pitch angle.
The :math:`0.1^{\circ}` threshold would cut off to many values during the steady flight sections.

**Evaluation using BACARDI broadband measurements**

##TODO

*author*: Johannes Röttenbacher
"""
if __name__ == "__main__":
# %% import modules
    import pylim.helpers as h
    import pylim.halo_ac3 as meta
    import ac3airborne
    from ac3airborne.tools import flightphase
    import xarray as xr
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    cm = 1 / 2.54
    cbc = h.get_cb_friendly_colors()

# %% set paths
    campaign = "halo-ac3"
    halo_key = "RF18"
    halo_flight = meta.flight_names[halo_key]
    date = halo_flight[9:17]

    plot_path = "./docs/figures/bahamas_pitch"
    bahamas_path = h.get_path("bahamas", halo_flight, campaign)
    bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{halo_key}_v1_1s.nc"

# %% get flight segmentation and select below and above cloud section
    segmentation = ac3airborne.get_flight_segments()["HALO-AC3"]["HALO"][f"HALO-AC3_HALO_{halo_key}"]
    segments = flightphase.FlightPhaseFile(segmentation)
    above_cloud, below_cloud = dict(), dict()
    above_cloud["start"] = segments.select("name", "polygon pattern 1")[0]["start"]
    above_cloud["end"] = segments.select("name", "polygon pattern 1")[0]["parts"][-1]["start"]
    below_cloud["start"] = segments.select("name", "polygon pattern 2")[0]["start"]
    below_cloud["end"] = segments.select("name", "polygon pattern 2")[0]["end"]
    above_slice = slice(above_cloud["start"], above_cloud["end"])
    below_slice = slice(below_cloud["start"], below_cloud["end"])
    case_slice = slice(above_cloud["start"], below_cloud["end"])

# %% read in BACARDI and BAHAMAS data and resample to 1 sec
    bahamas_ds = xr.open_dataset(f"{bahamas_path}/{bahamas_file}")
    bahamas_unfiltered = bahamas_ds[["IRS_ALT", "RELHUM", "TS", "MIXRATIO", "MIXRATIOV", "PS", "QC", "IRS_PHI", "IRS_THE"]]

# %% filter values which exceeded certain motion threshold
    roll_center = np.abs(bahamas_ds["IRS_PHI"].median())  # -> 0.06...
    roll_threshold = 0.5
    # pitch is not centered on 0 thus we need to calculate the difference to the center and compare that to the threshold
    # the pitch center changes during flight due to loss of weight (fuel) and speed
    # use the ten-minute rolling median as center to calculate the threshold
    pitch_center = np.abs(bahamas_ds["IRS_THE"].rolling(time=600, center=True, min_periods=60).median())
    pitch_threshold = 0.5
    # True -> keep value, False -> drop value (Nan)
    roll_filter = np.abs(bahamas_ds["IRS_PHI"]) < roll_threshold
    pitch_filter = np.abs(bahamas_ds["IRS_THE"] - pitch_center) < pitch_threshold
    motion_filter = roll_filter & pitch_filter
    bahamas_ds = bahamas_ds.where(motion_filter)

# %% create filter from flight segmentation which corresponds to turns or descents
    selected_segments = segments.findSegments(above_cloud["start"].strftime('%Y-%m-%d %H:%M:%S'),
                                              below_cloud["end"].strftime('%Y-%m-%d %H:%M:%S'))
    starts, ends = list(), list()
    for dic in selected_segments:
        if "short_turn" in dic["kinds"] or "roll_maneuver" in dic["kinds"] or "large_descent" in dic["kinds"]:
            starts.append(dic["start"])
            ends.append(dic["end"])
    starts, ends = pd.to_datetime(starts), pd.to_datetime(ends)
    for i in range(len(starts)):
        sel_time = (bahamas_ds.time > starts[i]) & (bahamas_ds.time < ends[i])

# %% plotting variables
    time_extend = pd.to_timedelta((bahamas_ds.time[-1] - bahamas_ds.time[0]).values)  # get time extend for x-axis labeling
    time_extend_cs = below_cloud["end"] - above_cloud["start"]  # time extend for case study
    time_extend_ac = above_cloud["end"] - above_cloud["start"]
    time_extend_bc = below_cloud["end"] - below_cloud["start"]
    plt.rcdefaults()
    h.set_cb_friendly_colors()
    plt.rc("font", size=12)
    figsize_wide = (24 * cm, 12 * cm)
    figsize_equal = (12 * cm, 12 * cm)

# %% plot BAHAMAS pitch angle for whole flight
    plot_ds = bahamas_unfiltered
    plot_ds_filtered = bahamas_unfiltered.where(roll_filter)
    rolling_median = plot_ds.IRS_THE.rolling(time=600, min_periods=2, center=True).median()
    pitch_median = plot_ds["IRS_THE"].median()

    fig, ax = plt.subplots(figsize=figsize_wide)
    ax.plot(plot_ds.time, plot_ds["IRS_THE"], label="Pitch angle", ls="", marker="."),
    ax.plot(plot_ds_filtered.time, plot_ds_filtered["IRS_THE"], label="Roll filtered pitch angle", ls="", marker="."),
    ax.plot(rolling_median.time, rolling_median, label="Rolling Median (10min)", ls="-.")
    ax.axhline(pitch_median, label="Median", color=cbc[-1], ls="--")
    ax.axhline(pitch_median + pitch_threshold, color=cbc[-1], label="Static pitch threshold")
    ax.axhline(pitch_median - pitch_threshold, color=cbc[-1])
    ax.axvline(above_cloud["start"], label=f"Start case study", color=cbc[-2])
    ax.axvline(below_cloud["end"], label=f"End case study", color=cbc[-3])
    ax.set_ylim(1.5, 3)
    ax.grid()
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0), ncol=3, bbox_transform=fig.transFigure)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Pitch Angle (°)")
    ax.set_title(f"{halo_key} BAHAMAS Pitch Angle")
    h.set_xticks_and_xlabels(ax, time_extend)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    figname = f"{plot_path}/{halo_flight}_BAHAMAS_pitch_angle_all.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot rolling median of pitch with varying time window
    plot_ds = bahamas_unfiltered

    fig, ax = plt.subplots(figsize=figsize_wide)
    ax.plot(plot_ds.time, plot_ds["IRS_THE"], label="Pitch angle", ls="", marker="."),
    for i, time_window in enumerate([600, 1200, 1800, 2700, 3600, 7200]):
        rolling_median = plot_ds.IRS_THE.rolling(time=time_window, min_periods=2, center=True).median()
        ax.plot(rolling_median.time, rolling_median, label=f"Rolling Median ({int(time_window/60)}min)", ls="-.", lw=3)
    ax.axvline(above_cloud["start"], label=f"Start case study", color=cbc[-2])
    ax.axvline(below_cloud["end"], label=f"End case study", color=cbc[-3])
    ax.set_ylim(1.5, 3)
    ax.grid()
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0), ncol=3, bbox_transform=fig.transFigure)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Pitch Angle (°)")
    ax.set_title(f"{halo_key} BAHAMAS Rolling Median Pitch Angle")
    h.set_xticks_and_xlabels(ax, time_extend)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    figname = f"{plot_path}/{halo_flight}_BAHAMAS_pitch_angle_rolling_median.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot BAHAMAS pitch angle with rolling standard deviation
    plot_ds = bahamas_unfiltered
    rolling = plot_ds.IRS_THE.rolling(time=1800, min_periods=2, center=True)
    rolling_median = rolling.median()
    rolling_std = rolling.std()
    print(rolling_std.median())

    fig, ax = plt.subplots(figsize=figsize_wide)
    ax.plot(plot_ds.time, plot_ds["IRS_THE"], label="Pitch angle", ls="", marker=".")
    ax.plot(rolling_median.time, rolling_median, label="Rolling Median (30min)", ls="-.")
    ax.plot(rolling_std.time, rolling_median + rolling_std, color=cbc[3],
            label=f"Rolling standard deviation (30min)")
    ax.plot(rolling_std.time, rolling_median - rolling_std, color=cbc[3])
    ax.axvline(above_cloud["start"], label=f"Start case study", color=cbc[-2])
    ax.axvline(below_cloud["end"], label=f"End case study", color=cbc[-3])
    ax.set_ylim(1.5, 3)
    ax.grid()
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0), ncol=3, bbox_transform=fig.transFigure)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Pitch Angle (°)")
    ax.set_title(f"{halo_key} BAHAMAS Pitch Angle with Rolling Standard Deviation")
    h.set_xticks_and_xlabels(ax, time_extend)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    figname = f"{plot_path}/{halo_flight}_BAHAMAS_pitch_angle_all_rolling_std.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot BAHAMAS pitch angle with dynamic pitch threshold
    plot_ds = bahamas_unfiltered
    plot_ds_filtered = bahamas_unfiltered.where(roll_filter)
    rolling_median = plot_ds.IRS_THE.rolling(time=1800, min_periods=2, center=True).median()
    pitch_median = plot_ds["IRS_THE"].median()
    pitch_thresholds = [0.1, 0.17, 0.5]

    fig, ax = plt.subplots(figsize=figsize_wide)
    ax.plot(plot_ds.time, plot_ds["IRS_THE"], label="Pitch angle", ls="", marker=".")
    ax.plot(plot_ds_filtered.time, plot_ds_filtered["IRS_THE"], label="Roll filtered pitch angle", ls="", marker=".")
    ax.plot(rolling_median.time, rolling_median, label="Rolling Median (30min)", ls="-.")
    for i, pt in enumerate(pitch_thresholds):
        ax.plot(rolling_median.time, rolling_median + pt, color=cbc[i + 3], label=f"Dynamic pitch threshold ({pt}°)")
        ax.plot(rolling_median.time, rolling_median - pt, color=cbc[i + 3])
    ax.axvline(above_cloud["start"], label=f"Start case study", color=cbc[-2])
    ax.axvline(below_cloud["end"], label=f"End case study", color=cbc[-3])
    ax.set_ylim(1.5, 3)
    ax.grid()
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0), ncol=3, bbox_transform=fig.transFigure)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Pitch Angle (°)")
    ax.set_title(f"{halo_key} BAHAMAS Pitch Angle")
    h.set_xticks_and_xlabels(ax, time_extend)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    figname = f"{plot_path}/{halo_flight}_BAHAMAS_pitch_angle_all_dynamic_threshold.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot BAHAMAS roll and pitch angle for whole flight
    plot_ds = bahamas_unfiltered

    fig, axs = plt.subplots(nrows=2, figsize=figsize_wide)
    # roll angle row one
    ax = axs[0]
    ax.plot(plot_ds.time, plot_ds["IRS_PHI"], label="Roll angle", ls="", marker=".")
    ax.axhline(plot_ds["IRS_PHI"].median(), ls="--", color=cbc[-1])
    ax.axhline(roll_threshold, color=cbc[-1])
    ax.axhline(-roll_threshold, color=cbc[-1])
    ax.set_ylim(-1.5, 1.5)  # for zoom plot
    ax.grid()
    ax.set_ylabel("Roll Angle (°)")
    h.set_xticks_and_xlabels(ax, time_extend)

    # pitch angle row two
    ax = axs[1]
    ax.plot(plot_ds.time, plot_ds["IRS_THE"], label="Pitch angle", ls="", marker=".", c=cbc[1])
    ax.axhline(pitch_median, label="Median", ls="--", color=cbc[-1])
    ax.axhline(pitch_median + pitch_threshold, color=cbc[-1], label="Threshold")
    ax.axhline(pitch_median - pitch_threshold, color=cbc[-1])
    ax.set_ylim(1.5, 3)  # for zoom plot
    ax.grid()
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Pitch Angle (°)")
    h.set_xticks_and_xlabels(ax, time_extend)

    fig.suptitle(f"{halo_key} BAHAMAS Roll and Pitch Angle")
    fig.legend(loc="upper right", bbox_to_anchor=(1, 0.9), bbox_transform=fig.transFigure)
    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    figname = f"{plot_path}/{halo_flight}_BAHAMAS_roll_pitch_angle.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot BAHAMAS pitch angle for above cloud section
    plot_ds = bahamas_unfiltered.sel(time=above_slice)
    plot_ds2 = pitch_center.sel(time=above_slice)
    pitch_threshold = 0.17
    selected_segments = segments.findSegments(above_cloud["start"].strftime('%Y-%m-%d %H:%M:%S'),
                                              above_cloud["end"].strftime('%Y-%m-%d %H:%M:%S'))

    fig, ax = plt.subplots(figsize=figsize_wide)
    ax.plot(plot_ds.time, plot_ds["IRS_THE"], label="Pitch angle", ls="", marker=".")
    ax.plot(plot_ds.time, plot_ds["IRS_THE"].where(roll_filter), label="Roll filtered Pitch angle", ls="", marker=".")
    for i, dic in enumerate(selected_segments[0]["parts"]):
        ax.axvline(dic["start"], label=f"start {dic['name']}", color=cbc[i + 1])

    ax.plot(plot_ds2.time, plot_ds2, color=cbc[-1], ls="--")
    ax.plot(plot_ds2.time, plot_ds2 + pitch_threshold, color=cbc[-1])
    ax.plot(plot_ds2.time, plot_ds2 - pitch_threshold, color=cbc[-1])
    # ax.set_ylim(-10, 5)
    ax.set_ylim(1.5, 3)  # for zoom plot
    ax.grid()
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0), ncol=2, bbox_transform=fig.transFigure)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Pitch Angle (°)")
    ax.set_title("BAHAMAS Pitch Angle Above Cloud")
    h.set_xticks_and_xlabels(ax, time_extend_ac)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.4)
    # figname = f"{plot_path}/{halo_flight}_BAHAMAS_pitch_angle_above_cloud_zoom.png"
    # plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()
