#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 11.09.2023

Filter BACARDI data for aircraft motion using BAHAMAS data

"""
# %% import modules
import pylim.helpers as h
import pylim.halo_ac3 as meta
from pylim import reader
import ac3airborne
from ac3airborne.tools import flightphase
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

cbc = h.get_cb_friendly_colors()

# %% set paths
campaign = "halo-ac3"
# keys = "RF17"  # run for single flight
keys = [f"RF{i:02}" for i in range(3, 19)]  # run for all flights

for key in tqdm(keys):
    flight = meta.flight_names[key]
    date = flight[9:17]

    plot_path = f"{h.get_path('plot', flight, campaign)}/bacarid_attitude_filter"
    bahamas_path = h.get_path("bahamas", flight, campaign)
    bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{key}_v1.nc"
    bacardi_path = h.get_path("bacardi", flight, campaign)
    bacardi_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1.nc"

    # %% get flight segmentation and select below and above cloud section
    segmentation = ac3airborne.get_flight_segments()["HALO-AC3"]["HALO"][f"HALO-AC3_HALO_{key}"]
    segments = flightphase.FlightPhaseFile(segmentation)

    # %% read in BACARDI and BAHAMAS data
    try:
        bahamas_ds = reader.read_bahamas(f"{bahamas_path}/{bahamas_file}")
        bahamas_unfiltered = bahamas_ds[["IRS_ALT", "RELHUM", "TS", "MIXRATIO", "MIXRATIOV", "PS", "QC", "IRS_PHI", "IRS_THE"]]
        bacardi_ds = xr.open_dataset(f"{bacardi_path}/{bacardi_file}")
    except FileNotFoundError as e:
        print(f"Skipping {key} because of {e}")
        continue

    # %% select only BAHAMAS time from BACARDI
    bacardi_ds = bacardi_ds.sel(time=bahamas_ds.time)

    # %% filter values which exceeded certain motion threshold
    roll_center = np.abs(bahamas_ds["IRS_PHI"].median())  # -> 0.06...
    roll_threshold = 0.5
    # pitch is not centered on 0 thus we need to calculate the difference to the center and compare that to the threshold
    # the pitch center changes during flight due to loss of weight (fuel) and speed
    # use the rolling median as center to calculate the threshold
    minutes = 30  # time window of rolling median in minutes
    time_resolution = 10  # time resolution/measurement frequency in Hertz
    nr_timesteps = minutes * 60 * time_resolution  # minutes * seconds
    pitch_center = np.abs(bahamas_ds["IRS_THE"].rolling(time=nr_timesteps, center=True, min_periods=60).median())
    # the actual pitch threshold at any point during the flight is defined as the median of the 30-minute rolling standard deviation
    pitch_threshold = (
        np.round(
            np.median(
                bahamas_ds["IRS_THE"].rolling(
                    time=nr_timesteps, center=True, min_periods=60).std())
            , 3)
    )
    # True -> keep value, False -> drop value (Nan)
    roll_filter = np.abs(bahamas_ds["IRS_PHI"]) < roll_threshold
    pitch_filter = np.abs(bahamas_ds["IRS_THE"] - pitch_center) < pitch_threshold
    attitude_filter = roll_filter & pitch_filter

    # %% create filter from flight segmentation which corresponds to turns or descents
    selected_segments = segments.findSegments(str(bahamas_ds.time[0].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()),
                                              str(bahamas_ds.time[-1].dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()))
    starts, ends = list(), list()
    for dic in selected_segments:
        kinds = dic["kinds"]
        if "short_turn" in kinds or "roll_maneuver" in kinds or "ascent" in kinds or "descent" in kinds:
            starts.append(dic["start"])
            ends.append(dic["end"])
    starts, ends = pd.to_datetime(starts), pd.to_datetime(ends)
    drop_times = xr.full_like(bahamas_ds.time, fill_value=False, dtype=bool)
    for i in range(len(starts)):
        drop_times = ((bahamas_ds.time > starts[i]) & (bahamas_ds.time < ends[i])) | drop_times

    drop_times = ~drop_times

    # %% filter BACARDI data
    bacardi_ds_1 = bacardi_ds.where(attitude_filter)
    bacardi_ds_2 = bacardi_ds.where(drop_times)
    motion_filter = drop_times & attitude_filter
    bacardi_ds_3 = bacardi_ds.where(motion_filter)

    # %% create filter flag variables and add them to the bacardi data set
    bacardi_ds["attitude_flag"] = xr.DataArray(motion_filter, coords={"time": bahamas_ds.time})
    bacardi_ds["segment_flag"] = xr.DataArray(drop_times, coords={"time": bahamas_ds.time})
    bacardi_ds["motion_flag"] = xr.DataArray(motion_filter, coords={"time": bahamas_ds.time})

    # %% plotting variables
    time_extend = pd.to_timedelta((bahamas_ds.time[-1] - bahamas_ds.time[0]).values)  # get time extend for x-axis labeling
    plt.rcdefaults()
    h.set_cb_friendly_colors()
    plt.rc("font", size=12)
    # plotting dictionaries for BACARDI
    labels = h.bacardi_labels

    # %% plot BACARDI measurements
    legend_entries = ["unfiltered", "motion filtered", "segment filtered", "motion+segment filtered"]
    plot_ds = dict()
    for i, ds in enumerate([bacardi_ds, bacardi_ds_1, bacardi_ds_2, bacardi_ds_3]):
        plot_ds[i] = ds #.sel(time=slice(above_cloud["start"], below_cloud["end"]))

    fig, ax = plt.subplots(figsize=h.figsize_wide)
    for k, ds in plot_ds.items():
        for var in ["F_down_solar"]:#, "F_down_terrestrial", "F_up_solar", "F_up_terrestrial"]:
            ax.plot(ds[var].time, ds[var], label=legend_entries[k])

    ax.grid()
    # ax.legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=4)
    ax.legend()
    h.set_xticks_and_xlabels(ax, time_extend)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Broadband Irradiance (W$\,$m$^{-2}$)")
    ax.set_title(f"{key} - BACARDI solar downward broadband irradiance")
    plt.tight_layout()
    # plt.subplots_adjust(bottom=0.3)
    figname = f"{plot_path}/{flight}_BACARDI_F_down_raw_vs_filtered.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

    # %% write new file with filter flags included
    new_filename = bacardi_file.replace(".nc", "_JR.nc")
    bacardi_ds.to_netcdf(f"{bacardi_path}/{new_filename}", format="NETCDF4_CLASSIC")
    #TODO: remove fill value from all variables, add time encoding as suitable for HALO-AC3