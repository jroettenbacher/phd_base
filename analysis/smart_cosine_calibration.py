#!/usr/bin/env python
"""Cosine Response Calibration of irradiance inlets of SMART for CIRRUS-HL and HALO-(AC)3
ASP06 J3, J4 (Fdw) with inlet ASP02 done on 5th November 2021 by Anna Luebke and Johannes Röttenbacher
Lamp used: FEL-1587
The inlet was rotated at 5° increments in both directions away from the lamp. Clockwise is positive.
Each measurement lasted about 30 seconds with 1000 ms integration time.
Data is stored in folders which denote the angle the inlet was turned to.
_turned: The inlet was turned 90deg around the horizontal axis to evaluate al four azimuth directions
_diffuse: A baffle was placed before the inlet to shade it
author: Johannes Röttenbacher
"""

# %% module import
import pylim.helpers as h
from pylim import cirrus_hl, reader, smart
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import logging

log = logging.getLogger("pylim")
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

# %% set paths
calib_path = h.get_path("calib")
folder_name = "ASP06_cosine_calibration"
plot_path = "C:/Users/Johannes/Documents/Doktor/instruments/SMART/cosine_correction"
properties = ["Fdw", "Fup"]
channels = ["VNIR", "SWIR"]

# %% correct measurements for dark current and merge them into one file for each channel
folder = "ASP06_cosine_calibration"

# merge VNIR dark measurement files before correcting the calib files
for dirpath, dirs, files in os.walk(os.path.join(calib_path, folder)):
    log.debug(f"Working on {dirpath}")
    if "diffuse" in dirpath:
        for prop in properties:
            try:
                vnir_dark_files = [f for f in files if f.endswith(f"{prop}_VNIR.dat")]
                df = pd.concat([pd.read_csv(f"{dirpath}/{file}", sep="\t", header=None) for file in vnir_dark_files])
                # delete all minutely files
                for file in vnir_dark_files:
                    os.remove(os.path.join(dirpath, file))
                    log.debug(f"Deleted {dirpath}/{file}")
                outname = f"{dirpath}/{vnir_dark_files[0]}"
                df.to_csv(outname, sep="\t", index=False, header=False)
                log.debug(f"Saved {outname}")
            except ValueError:
                pass

log.info("Merged all diffuse VNIR files")
# correct all calibration measurement files for the dark current
# VNIR: use the diffuse measurements to the corresponding angles as dark current measurements
# if no such measurement is available :
# Option 1: use the first 19:99 pixels from each measurement to scale the 0° dark current measurement as dark current
# Option 2: use the 95deg diffuse measurement
# However, this has the problem that it was probably warmer during that measurement than during the one which is being
# corrected -> higher dark current at 95deg
# SWIR: the shutters on the SWIR spectrometers did not work properly
# SWIR_option 2: use the diffuse measurements also for the correction of the SWIR measured dark current
VNIR_option = 1
SWIR_option = 2
for dirpath, dirs, files in os.walk(os.path.join(calib_path, folder)):
    log.debug(f"Working on {dirpath}")
    dirname = os.path.basename(dirpath)
    parent = os.path.dirname(dirpath)
    for file in files:
        if file.endswith("SWIR.dat"):
            if SWIR_option == 1:
                # use shutter measurements to correct each file
                log.debug(f"Working on {dirpath}/{file}")
                smart_cor = smart.correct_smart_dark_current("", file, option=2, path=dirpath)

            else:
                assert SWIR_option == 2, f"Wrong option '{SWIR_option}' provided!"
                # use diffuse measurement to correct SWIR dark current if possible
                # else use closest diffuse measurement else use 0° measurement
                log.debug(f"Working on {dirpath}/{file}")
                _, channel, direction = smart.get_info_from_filename(file)
                try:
                    new_dirname = dirname.replace(dirname, f"{dirname}_diffuse") if "diffuse" not in dirname else dirname
                    dark_dir = os.path.join(parent, new_dirname)
                    dark_file = [f for f in os.listdir(dark_dir) if direction in f and channel in f][0]

                except (FileNotFoundError, IndexError):
                    log.debug(f"No diffuse {direction}_{channel} file found in {dark_dir}")
                    # find the closest diffuse folder
                    if "turned" in dirname:
                        diffuse_folders = [d for d in os.listdir(parent) if "diffuse" in d and "turned" in d]
                    else:
                        diffuse_folders = [d for d in os.listdir(parent) if "diffuse" in d and "turned" not in d]

                    available_angles = [float(re.search(r"-?\d{1,2}", d)[0]) for d in diffuse_folders]
                    current_angle = float(re.search(r"-?\d{1,2}", dirname)[0])  # get angle from dirname
                    closest_angle = available_angles[h.arg_nearest(available_angles, current_angle)]
                    # generate new dark directory according to closest angle
                    new_dirname = dirname.replace(dirname, f"{dirname}_diffuse").replace(str(int(current_angle)),
                                                                                         str(int(closest_angle)))
                    dark_dir = os.path.join(parent, new_dirname)
                    try:
                        dark_file = [f for f in os.listdir(dark_dir) if direction in f and channel in f][0]
                    except IndexError:
                        log.debug(f"No {direction}_{channel} file found in {dark_dir}")
                        dark_dir = f"{calib_path}/{folder}/0_diffuse"
                        dark_file = [f for f in os.listdir(dark_dir) if direction in f and channel in f][0]

                measurement = reader.read_smart_raw(dirpath, file)
                dark_current = reader.read_smart_raw(dark_dir, dark_file)
                dark_current = dark_current.iloc[:, 2:].mean()
                # only use data when shutter is open
                measurement = measurement.where(measurement.shutter == 1).iloc[:, 2:]
                smart_cor = measurement - dark_current

        elif file.endswith("VNIR.dat"):
            log.debug(f"Working on {dirpath}/{file}")
            _, channel, direction = smart.get_info_from_filename(file)
            new_dirname = dirname.replace(dirname, f"{dirname}_diffuse") if "diffuse" not in dirname else dirname
            dark_dir = os.path.join(parent, new_dirname)
            try:
                dark_file = [f for f in os.listdir(dark_dir) if direction in f and channel in f][0]
                measurement = reader.read_smart_raw(dirpath, file)
                dark_current = reader.read_smart_raw(dark_dir, dark_file)
                dark_current = dark_current.iloc[:, 2:].mean()
                # only use data when shutter is open
                measurement = measurement.where(measurement.shutter == 1).iloc[:, 2:]
                smart_cor = measurement - dark_current
            except (FileNotFoundError, IndexError):
                log.debug(f"No diffuse {direction}_{channel} file found in {dark_dir}")
                if VNIR_option == 1:
                    # scale 0° dark current with current dark pixel measurements
                    dark_dir = f"{calib_path}/{folder}/0_diffuse"
                    dark_file = [f for f in os.listdir(dark_dir) if direction in f and channel in f][0]
                    dark_current = reader.read_smart_raw(dark_dir, dark_file).iloc[:, 2:].mean()  # drop tint and shutter
                    measurement = reader.read_smart_raw(dirpath, file)
                    measurement = measurement.where(measurement.shutter == 1).iloc[:, 2:]
                    dark_scale = dark_current * np.mean(measurement.mean().iloc[19:99]) / np.mean(dark_current.iloc[19: 99])
                    dark_scale = dark_scale - dark_scale.rolling(20, min_periods=1).mean()
                    dark_scale2 = dark_current.rolling(20, min_periods=1).mean() + (
                                np.mean(measurement.mean().iloc[19:99]) - np.mean((dark_current.iloc[19:99])))
                    smart_cor = measurement - dark_scale2 - dark_scale
                else:
                    dark_dir = f"{calib_path}/{folder}/0_diffuse"
                    dark_file = [f for f in os.listdir(dark_dir) if direction in f and channel in f][0]
                    measurement = reader.read_smart_raw(dirpath, file)
                    dark_current = reader.read_smart_raw(dark_dir, dark_file)
                    dark_current = dark_current.iloc[:, 2:].mean()
                    # only use data when shutter is open
                    measurement = measurement.where(measurement.shutter == 1).iloc[:, 2:]
                    smart_cor = measurement - dark_current
        else:
            # skip all _cor.dat files
            continue

        outname = f"{dirpath}/{file.replace('.dat', '_cor.dat')}"
        smart_cor.to_csv(outname, sep="\t", float_format="%.0f")
        log.debug(f"Saved {outname}")

log.info("Corrected all raw files for the dark current")
# merge minutely corrected files to one file
for dirpath, dirs, files in os.walk(os.path.join(calib_path, folder)):
    log.debug(f"Working on {dirpath}")
    for channel in channels:
        for prop in properties:
            try:
                filename = [f for f in files if f.endswith(f"{prop}_{channel}_cor.dat")]
                df = pd.concat([pd.read_csv(f"{dirpath}/{file}", sep="\t", index_col="time") for file in filename])
                # delete all minutely files
                for file in filename:
                    os.remove(os.path.join(dirpath, file))
                    log.debug(f"Deleted {dirpath}/{file}")
                outname = f"{dirpath}/{filename[0]}"
                df.to_csv(outname, sep="\t")
                log.debug(f"Saved {outname}")
            except ValueError:
                pass

log.info("Merged all minutely corrected files")

# %% read in data into dictionary
use_raw = False  # use raw or  dark current corrected files?
measurements = dict(Fdw_VNIR=dict(), Fdw_SWIR=dict(), Fup_VNIR=dict(), Fup_SWIR=dict())
no_measurements = list()
# loop through all combinations of measurements and create a dictionary
for prop in properties:
    for channel in channels:
        for angle in range(-95, 100, 5):
            measurements[f"{prop}_{channel}"][f"{angle}"] = dict()
            for position in ["", "_turned"]:
                # generate more descriptive strings to use as dictionary keys
                position_key = position[1:] if position == "_turned" else "normal"
                measurements[f"{prop}_{channel}"][f"{angle}"][f"{position_key}"] = dict()
                for mtype in ["", "_diffuse"]:
                    # generate input path from options
                    inpath = f"{calib_path}/{folder_name}/{angle}{position}{mtype}"
                    # generate more descriptive dictionary key
                    mtype_key = mtype[1:] if mtype == "_diffuse" else "direct"
                    # not all possible combinations have been measured, account for that with a try/except statement
                    try:
                        if use_raw:
                            # list corresponding files
                            raw_files = [f for f in os.listdir(inpath) if f"{prop}_{channel}" in f and "cor" not in f]
                            measurements[f"{prop}_{channel}"][f"{angle}"][f"{position_key}"][
                                f"{mtype_key}"] = pd.concat([reader.read_smart_raw(inpath, f) for f in raw_files])
                        else:
                            # there is only one corrected file for each prop_channel in each folder
                            cor_file = [f for f in os.listdir(inpath) if f.endswith(f"{prop}_{channel}_cor.dat")][0]
                            measurements[f"{prop}_{channel}"][f"{angle}"][f"{position_key}"][
                                f"{mtype_key}"] = reader.read_smart_cor(inpath, cor_file)
                    except FileNotFoundError:
                        no_measurements.append(f"{prop}_{channel}, {angle}, {position_key}, {mtype_key}")
                    except ValueError:
                        no_measurements.append(f"{prop}_{channel}, {angle}, {position_key}, {mtype_key}")
                    except IndexError:
                        # no cor_file can be found for prop_channel
                        no_measurements.append(f"{prop}_{channel}, {angle}, {position_key}, {mtype_key}")

# %% calculate mean of the three 0° measurements
mean_spectra1, mean_spectra2, mean_spectra3 = dict(), dict(), dict()
# iloc[2:] is not needed for corrected data, leave it in for convenience
mean_spectra1["Fdw_VNIR"] = measurements["Fdw_VNIR"]["0"]["normal"]["direct"].loc[
                            "2021-11-05 12:38":"2021-11-05 12:40"].mean().iloc[2:]
mean_spectra2["Fdw_VNIR"] = measurements["Fdw_VNIR"]["0"]["normal"]["direct"].loc[
                            "2021-11-05 13:10":"2021-11-05 13:13"].mean().iloc[2:]
mean_spectra3["Fdw_VNIR"] = measurements["Fdw_VNIR"]["0"]["normal"]["direct"].loc[
                            "2021-11-05 13:37":"2021-11-05 13:39"].mean().iloc[2:]
mean_spectra1["Fdw_SWIR"] = measurements["Fdw_SWIR"]["0"]["normal"]["direct"].loc[
                            "2021-11-05 12:38":"2021-11-05 12:40"].mean().iloc[2:]
mean_spectra2["Fdw_SWIR"] = measurements["Fdw_SWIR"]["0"]["normal"]["direct"].loc[
                            "2021-11-05 13:10":"2021-11-05 13:13"].mean().iloc[2:]
mean_spectra3["Fdw_SWIR"] = measurements["Fdw_SWIR"]["0"]["normal"]["direct"].loc[
                            "2021-11-05 13:37":"2021-11-05 13:39"].mean().iloc[2:]

mean_spectra1["Fup_VNIR"] = measurements["Fup_VNIR"]["0"]["normal"]["direct"].loc[
                            "2021-11-08 15:26":"2021-11-08 15:28"].mean().iloc[2:]
mean_spectra2["Fup_VNIR"] = measurements["Fup_VNIR"]["0"]["normal"]["direct"].loc[
                            "2021-11-08 16:10":"2021-11-08 16:11"].mean().iloc[2:]
mean_spectra1["Fup_SWIR"] = measurements["Fup_SWIR"]["0"]["normal"]["direct"].loc[
                            "2021-11-08 15:26":"2021-11-08 15:28"].mean().iloc[2:]
mean_spectra2["Fup_SWIR"] = measurements["Fup_SWIR"]["0"]["normal"]["direct"].loc[
                            "2021-11-08 16:10":"2021-11-08 16:11"].mean().iloc[2:]

# %% plot the 0° measurement for all channels separately
h.set_cb_friendly_colors()
for prop in properties:
    for channel in channels:
        fig, ax = plt.subplots()
        mean_spectra1[f"{prop}_{channel}"].plot(ax=ax, label="1st Measurement")
        mean_spectra2[f"{prop}_{channel}"].plot(ax=ax, label="2nd Measurement")
        if prop == "Fdw":
            mean_spectra3[f"Fdw_{channel}"].plot(ax=ax, label="3rd Measurement")
        ax.set_title(f"{prop} {channel} 0° Direct Measurement")
        ax.set_xlabel("Pixel #")
        ax.set_ylabel("Counts")
        ax.grid()
        ax.legend()
        # plt.show()
        figname = f"{plot_path}/{prop}_{channel}_0deg_normal_direct_comparison.png"
        plt.savefig(figname, dpi=100)
        log.info(f"Saved {figname}")
        plt.close()

# %% plot the three 0° measurements for all channels
fig, axs = plt.subplots(nrows=2, ncols=2)
for id1, prop in enumerate(properties):
    for id2, channel in enumerate(channels):
        mean_spectra1[f"{prop}_{channel}"].plot(ax=axs[id1, id2], label="1st Measurement")
        mean_spectra2[f"{prop}_{channel}"].plot(ax=axs[id1, id2], label="2nd Measurement")
        axs[id1, id2].set_title(f"{prop} {channel}")

mean_spectra3[f"Fdw_VNIR"].plot(ax=axs[0, 0], label="3rd Measurement")
mean_spectra3[f"Fdw_SWIR"].plot(ax=axs[0, 1], label="3rd Measurement")
fig.suptitle("0° Direct Measurements")
for ax in axs:
    for a in ax:
        a.set_xlabel("Pixel #")
        a.set_ylabel("Counts")
        a.grid()

# use legend from first plot for the complete figure
axs[0, 0].legend(bbox_to_anchor=(0.5, 0), loc="lower center", bbox_transform=fig.transFigure, ncol=3)
plt.subplots_adjust(bottom=0.2, wspace=0.5, hspace=0.5)
# plt.show()
figname = f"{plot_path}/All_0deg_normal_direct_comparison.png"
plt.savefig(figname, dpi=100)
log.info(f"Saved {figname}")
plt.close()

# %% plot difference between 3rd and 1st measurement at 0°
diff = mean_spectra3["Fdw_VNIR"] - mean_spectra1["Fdw_VNIR"]
# TODO: Do for all channels and in one plot
diff.plot()
plt.title("Fdw VNIR 0° measurement - Difference between 3rd and 1st")
plt.xlabel("Pixel #")
plt.ylabel("Counts")
plt.grid()
plt.tight_layout()
# plt.show()
figname = f"{plot_path}/Fdw_VNIR_0deg_normal_direct_diff3-1.png"
plt.savefig(figname)
log.info(f"Saved {figname}")
plt.close()

# %% calculate the mean for every angle -> one spectra per angle
mean_spectras = deepcopy(measurements)
for channel in mean_spectras:
    for angle in mean_spectras[channel]:
        for position in mean_spectras[channel][f"{angle}"]:
            for mtype in mean_spectras[channel][f"{angle}"][position]:
                df = mean_spectras[channel][f"{angle}"][position][mtype].copy()
                if use_raw:
                    df_mean = df.where(df["shutter"] == 1).mean().iloc[2:]
                else:
                    df_mean = df.mean()
                mean_spectras[channel][f"{angle}"][position][mtype] = df_mean

del measurements  # delete variable from workspace to free memory

# %% calculate difference between positive and negative angles
diff_angles = deepcopy(mean_spectras)
for channel in mean_spectras:
    for angle in range(5, 100, 5):
        # remove negative angles from new dictionary (they are left overs from the copy action above)
        diff_angles[channel].pop(f"-{angle}")
        for position in mean_spectras[channel][f"{angle}"]:
            try:
                # use only direct measurements because not all diffuse measurements have a corresponding counter angle
                df1 = mean_spectras[channel][f"{angle}"][position]["direct"].copy()
                df2 = mean_spectras[channel][f"-{angle}"][position]["direct"].copy()
                df_diff = df1 - df2
                diff_angles[channel][f"{angle}"][position]["direct"] = df_diff
                # remove diffuse measurements because they are of no interest
                diff_angles[channel][f"{angle}"][position].pop("diffuse")
            except KeyError:
                # if no turned measurement was performed for this angle remove the empty dictionary
                if position == "turned":
                    diff_angles[channel][f"{angle}"].pop(position)

# %% plot differences of all angles for each channel and position (6 angles per plot)
prop, channel, position = "Fdw", "VNIR", "normal"  # for testing
rg_start = [5, 35, 65]
rg_end = [35, 65, 100]
plt.rcdefaults()
# define qualitative color map as new color cycle
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20c.colors)
for prop in properties:
    for channel in channels:
        for position in ["normal", "turned"]:
            id1 = 0
            for rgs, rge in zip(rg_start, rg_end):
                fig, ax = plt.subplots()
                for angle in range(rgs, rge, 5):
                    try:
                        df = diff_angles[f"{prop}_{channel}"][f"{angle}"][position]["direct"]
                        df.plot(ax=ax, label=angle)
                    except KeyError:
                        continue
                ax.set_title(f"{prop} {channel} {position} Difference between Positive and Negative Angles")
                ax.set_xlabel("Pixel #")
                ax.set_ylabel("Difference (Counts)")
                ax.legend()
                ax.grid()
                # plt.show()
                figname = f"{prop}_{channel}_{position}_difference_pos_neg_angles_{rg_start[id1]}-{rg_end[id1] - 5}.png"
                plt.savefig(f"{plot_path}/diff_neg_pos_angles/{figname}", dpi=100)
                log.info(f"Saved {figname}")
                plt.close()
                id1 += 1

# %% plot every mean spectra
plt.rcdefaults()
h.set_cb_friendly_colors()
for pair in h.nested_dict_pairs_iterator(mean_spectras):
    prop_channel, angle, position, mtype, df = pair
    df.plot(title=f"{prop_channel} {angle}° {position} {mtype} Mean Spectrum", ylabel="Counts", xlabel="Pixel #")
    plt.grid()
    # plt.show()
    figname = f"{prop_channel}_{angle}deg_{position}_{mtype}_mean_spectrum.png"
    plt.savefig(f"{plot_path}/mean_spectras/{figname}", dpi=100)
    log.info(f"Saved {figname}")
    plt.close()

# %% set values < 1 to 1 to avoid problems with the correction factor calculation
mean_spectras_cor = deepcopy(mean_spectras)
for pair in h.nested_dict_pairs_iterator(mean_spectras_cor):
    prop_channel, angle, position, mtype, df = pair
    df[df < 1] = 1
    mean_spectras_cor[prop_channel][f"{angle}"][position][mtype] = df

del mean_spectras  # delete variable from workspace to free memory

# %% calculate the direct cosine correction factors for each angle
k_cos_dir = dict(Fdw_VNIR=dict(), Fdw_SWIR=dict(), Fup_VNIR=dict(), Fup_SWIR=dict())
for prop in k_cos_dir:
    F_0 = mean_spectras_cor[prop]["0"]["normal"]["direct"]
    for angle in range(-95, 100, 5):
        F_angle = mean_spectras_cor[prop][f"{angle}"]["normal"]["direct"]
        angle_rad = np.deg2rad(angle)
        k_cos_dir[prop][f"{angle}"] = F_0 * np.cos(angle_rad) / F_angle

# %% create dataframe from dictionary
dfs = pd.DataFrame()
for pairs in h.nested_dict_pairs_iterator(mean_spectras_cor):
    prop_channel, angle, position, mtype, df = pairs
    df = pd.DataFrame(df, columns=["counts"]).reset_index()  # reset index to get a column with the pixel numbers
    df = df.rename(columns={"index": "pixel"})  # rename the index column to the pixel numbers
    df = df.assign(prop=prop_channel, angle=angle, position=position, mtype=mtype)
    dfs = pd.concat([dfs, df])

dfs.reset_index(drop=True, inplace=True)

# %% plot actual cosine response
prop_channel, position, mtype = "Fdw_VNIR", "normal", "direct"  # for testing
prop_channels, positions, mtypes = dfs.prop.unique(), dfs.position.unique(), dfs.mtype.unique()

for prop_channel in prop_channels:
    for position in positions:
        for mtype in mtypes:
            # select the corresponding values
            z = dfs.loc[
                (dfs["prop"] == prop_channel) & (dfs["position"] == position) & (dfs["mtype"] == mtype),
                ["angle", "counts", "pixel"]
            ]
            levels = z.angle.unique()  # extract levels from angles
            len_levels = len(levels)
            if len_levels < 8:
                log.info(f"Only {len_levels} angle(s) available")
                # only plot data for which all 39 angles have been measured
                continue
            # convert angles to categorical type to keep order when pivoting
            z["angle"] = pd.Categorical(z["angle"], categories=levels, ordered=True)
            z_new = z.pivot(index="angle", columns="pixel", values="counts")
            # arrange an artificial grid to plot on
            rad = np.arange(0, z_new.shape[1])  # 1024 or 256 pixels
            a = levels.astype(float)  # -95 to 95° in 5° steps
            log.debug(f"Using angles: {a}")
            r, th = np.meshgrid(rad, a)
            # plot
            fig, ax = plt.subplots()
            img = ax.pcolormesh(th, r, z_new, cmap='YlOrRd', shading="nearest")
            ax.grid()
            ax.set_title(f"{prop_channel} {position} {mtype} Cosine Response")
            ax.set_xlabel("Angle (deg)")
            ax.set_ylabel("Pixel #")
            if len_levels != 39:
                ax.set_xticks(a)
                ax.tick_params(axis='x', labelrotation=45)
            plt.colorbar(img, label="Counts")
            plt.tight_layout()
            # plt.show()
            figname = f"{prop_channel}_{position}_{mtype}_cosine_response.png"
            plt.savefig(f"{plot_path}/cosine_response/{figname}", dpi=100)
            log.info(f"Saved {figname}")
            plt.close()

# %% plot theoretical cosine response
dfs["cosine_counts"] = dfs["counts"] * np.cos(np.deg2rad(dfs["angle"].astype(float)))
prop_channel = "Fdw_VNIR"
for prop_channel in prop_channels:
    F0 = dfs.loc[(dfs["prop"] == prop_channel) & (dfs["position"] == "normal") & (dfs["mtype"] == "direct"),
             ["angle", "cosine_counts", "pixel"]]
    levels = dfs.angle.unique()  # extract levels from angles
    # convert angles to categorical type to keep order when pivoting
    F0["angle"] = pd.Categorical(F0["angle"], categories=levels, ordered=True)
    F0_pivot = F0.pivot(index="angle", columns="pixel", values="cosine_counts")
    # arrange an artificial grid to plot on
    rad = np.arange(0, F0_pivot.shape[1])  # 1024 or 256 pixels
    a = levels.astype(float)  # -95 to 95° in 5° steps
    log.debug(f"Using angles: {a}")
    r, th = np.meshgrid(rad, a)
    # plot
    fig, ax = plt.subplots()
    img = ax.pcolormesh(th, r, F0_pivot, cmap='YlOrRd', shading="nearest")
    ax.grid()
    ax.set_title(f"{prop_channel} normal direct Theoretical Cosine Response")
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Pixel #")
    plt.colorbar(img, label="Counts")
    plt.tight_layout()
    # plt.show()
    figname = f"{prop_channel}_normal_direct_theoretical_cosine_response.png"
    plt.savefig(f"{plot_path}/cosine_response/{figname}", dpi=100)
    log.info(f"Saved {figname}")
    plt.close()

# %% create dataframe with cosine correction factors
k_cos_dir_df = pd.DataFrame()
for pairs in h.nested_dict_pairs_iterator(k_cos_dir):
    prop_channel, angle, df = pairs
    df = pd.DataFrame(df, columns=["k_cos"]).reset_index()  # reset index to get a column with the pixel numbers
    df = df.rename(columns={"index": "pixel"})  # rename the index column to the pixel numbers
    df = df.assign(prop=prop_channel, angle=angle)
    k_cos_dir_df = pd.concat([k_cos_dir_df, df])

k_cos_dir_df.reset_index(drop=True, inplace=True)
k_cos_dir_df = k_cos_dir_df[np.abs(k_cos_dir_df["angle"].astype(float)) < 95]

# %% plot cosine correction factors
for prop_channel in k_cos_dir:
    k_cos = k_cos_dir_df.loc[(k_cos_dir_df["prop"] == prop_channel), ["angle", "k_cos", "pixel"]]
    levels = k_cos_dir_df.angle.unique()  # extract levels from angles
    # convert angles to categorical type to keep order when pivoting
    k_cos["angle"] = pd.Categorical(k_cos["angle"], categories=levels, ordered=True)
    k_cos_pivot = k_cos.pivot(index="angle", columns="pixel", values="k_cos")
    # arrange an artificial grid to plot on
    rad = np.arange(0, k_cos_pivot.shape[1])  # 1024 or 256 pixels
    a = levels.astype(float)  # -95 to 95° in 5° steps
    log.debug(f"Using angles: {a}")
    r, th = np.meshgrid(rad, a)
    # plot
    fig, ax = plt.subplots()
    img = ax.pcolormesh(th, r, k_cos_pivot, cmap='coolwarm', shading="nearest")
    ax.grid()
    ax.set_title(f"{prop_channel} normal direct Cosine Corretion Factor")
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Pixel #")
    img.set_clim(0.5, 1.5)
    plt.colorbar(img, label="Correction Factor", extend="both")
    plt.tight_layout()
    # plt.show()
    figname = f"{prop_channel}_normal_direct_cosine_correction_factors.png"
    plt.savefig(f"{plot_path}/correction_factors/{figname}", dpi=100)
    log.info(f"Saved {figname}")
    plt.close()

# %% plot cosine correction factor for each pixel/wavelength
h.set_cb_friendly_colors()
pixel_path = h.get_path("pixel_wl")
for prop_channel in k_cos_dir:
    k_cos = k_cos_dir_df.loc[(k_cos_dir_df["prop"] == prop_channel), ["angle", "k_cos", "pixel"]]
    k_cos["angle"] = k_cos["angle"].astype(float)  # convert angles to float for better plotting
    pixel_wl = reader.read_pixel_to_wavelength(pixel_path, cirrus_hl.lookup[prop_channel])
    k_cos = pd.merge(k_cos, pixel_wl, on="pixel")
    for pixel in k_cos["pixel"].unique():
        k = k_cos.loc[(k_cos["pixel"] == pixel), :]
        fig, ax = plt.subplots()
        k.plot(x="angle", y="k_cos", ax=ax, label=f"{k['wavelength'].iloc[0]} nm",
               title=f"{prop_channel} Cosine correction factor for {k['wavelength'].iloc[0]} nm",
               ylabel="Cosine Correction Factor", xlabel="Angle (°)")
        ax.set_xticks(np.arange(-90, 95, 15))
        ax.set_ylim(0, 2)
        ax.axhline(1, c="k", ls="--")
        ax.grid()
        # plt.show()
        figname = f"{prop_channel}_{pixel}_cosine_correction_factor.png"
        plt.savefig(f"{plot_path}/correction_factors_single/{figname}", dpi=100)
        log.debug(f"Saved {figname}")
        plt.close()

