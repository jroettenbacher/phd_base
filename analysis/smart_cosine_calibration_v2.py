#!/usr/bin/env python
"""Cosine Response Calibration of irradiance inlets of SMART for CIRRUS-HL and HALO-(AC)3 2nd try

* ASP06 J3, J4 (Fdw) with inlet VN05 (ASP_01) done on 16th November 2021 by Anna Luebke and Johannes Röttenbacher and finished on 22th November by Benjamin Kirbus and Johannes
* ASP06 J5, J6 (Fup) with inlet VN11 (ASP_02) done on 22th November 2021 by Benjamin Kirbus and Johannes Röttenbacher

Lamp used: cheap 1000W lamp
optical fiber: 22b
-> 21b has very little output from the SWIR fiber, thus the 22b fiber was chosen for both inlets as it is identical to 21b

The inlet was rotated at 5° increments in both directions away from the lamp. Clockwise is positive.
Each measurement lasted about 42 seconds with 700 ms integration time for VN05 and about 48s with 800ms integration time
for VN11.
Data is stored in folders which denote the angle the inlet was turned to.

* :file:`{angle}_turned`: The inlet was turned 90deg clockwise around the horizontal axis to evaluate all four azimuth directions
* :file:`{angle}_diffuse`: A baffle was placed before the inlet to shade it (dark measurements)

Results Cosine Resopnse Calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See PowerPoint `20211122_SMART_cosine_calibration.pptx` under `instruments/SMART`.


*author*: Johannes Röttenbacher

"""
if __name__ == "__main__":
# %% module import
    import pylim.helpers as h
    from pylim import cirrus_hl, reader, smart
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from copy import deepcopy
    import numpy as np
    import scipy.integrate as integrate
    import scipy.interpolate as interpolate
    from tqdm import tqdm
    import logging

    log = logging.getLogger("pylim")
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)

# %% set paths
    calib_path = h.get_path("calib")
    folder_name = "ASP06_cosine_calibration_v2"
    plot_path = "C:/Users/Johannes/Documents/Doktor/instruments/SMART/cosine_correction_v2"
    outfile_path = "C:/Users/Johannes/Documents/Doktor/instruments/SMART/cosine_correction_factors"
    properties = ["Fdw", "Fup"]
    channels = ["VNIR", "SWIR"]

# %% correct measurements for dark current and merge them into one file for each channel
    folder = folder_name

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
                    log.debug(f"Encountered ValueError for {dirpath} and {prop}")
                    pass

    log.info("Merged all diffuse VNIR files")
    # correct all calibration measurement files for the dark current
    # VNIR: use the diffuse measurements to the corresponding angles as dark current measurements
    # SWIR: the shutters on the SWIR spectrometers did work properly for this calib
    # SWIR_option 1: Use the closed shutter measurements from each file where the shutter should be closed
    # SWIR_option 2: use the diffuse measurements also for the correction of the SWIR measured dark current
    SWIR_option = 1
    for dirpath, dirs, files in os.walk(os.path.join(calib_path, folder)):
        log.debug(f"Working on {dirpath}")
        dirname = os.path.basename(dirpath)
        parent = os.path.dirname(dirpath)
        turned_flag = "turned" in dirname
        for file in files:
            if file.endswith("SWIR.dat"):
                if SWIR_option == 1:
                    # use shutter measurements to correct each file
                    log.debug(f"Working on {dirpath}/{file}")
                    smart_cor = smart.correct_smart_dark_current("", file, option=2, path=dirpath)

                else:
                    assert SWIR_option == 2, f"Wrong option '{SWIR_option}' provided!"
                    # use diffuse measurement to correct SWIR dark current
                    log.debug(f"Working on {dirpath}/{file}")
                    _, channel, direction = smart.get_info_from_filename(file)
                    new_dirname = dirname.replace(dirname, f"{dirname}_diffuse") if "diffuse" not in dirname else dirname
                    dark_dir = os.path.join(parent, new_dirname)
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
                dark_file = [f for f in os.listdir(dark_dir) if direction in f and channel in f][0]
                measurement = reader.read_smart_raw(dirpath, file).iloc[:, 2:]
                dark_current = reader.read_smart_raw(dark_dir, dark_file)
                dark_current = dark_current.iloc[:, 2:].mean()
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
    use_raw = False  # use raw or dark current corrected files?
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

# %% calculate mean of the three 0° measurements
    mean_spectra1, mean_spectra2, mean_spectra3 = dict(), dict(), dict()
    # iloc[2:] is not needed for corrected data, leave it in for convenience
    start1, start2, start3 = "2021-11-16 09:01", "2021-11-16 09:50", "2021-11-16 10:33"
    end1, end2, end3 = "2021-11-16 09:04", "2021-11-16 09:52", "2021-11-16 10:35"
    mean_spectra1["Fdw_VNIR"] = measurements["Fdw_VNIR"]["0"]["normal"]["direct"].loc[start1:end1].mean()  # .iloc[2:]
    mean_spectra2["Fdw_VNIR"] = measurements["Fdw_VNIR"]["0"]["normal"]["direct"].loc[start2:end2].mean()  # .iloc[2:]
    mean_spectra3["Fdw_VNIR"] = measurements["Fdw_VNIR"]["0"]["normal"]["direct"].loc[start3:end3].mean()  # .iloc[2:]
    mean_spectra1["Fdw_SWIR"] = measurements["Fdw_SWIR"]["0"]["normal"]["direct"].loc[start1:end1].mean()  # .iloc[2:]
    mean_spectra2["Fdw_SWIR"] = measurements["Fdw_SWIR"]["0"]["normal"]["direct"].loc[start2:end2].mean()  # .iloc[2:]
    mean_spectra3["Fdw_SWIR"] = measurements["Fdw_SWIR"]["0"]["normal"]["direct"].loc[start3:end3].mean()  # .iloc[2:]

    start1, start2, start3 = "2021-11-22 11:37", "2021-11-22 12:21", "2021-11-22 13:00"
    end1, end2, end3 = "2021-11-22 11:40", "2021-11-22 12:23", "2021-11-22 13:03"
    mean_spectra1["Fup_VNIR"] = measurements["Fup_VNIR"]["0"]["normal"]["direct"].loc[start1:end1].mean()  # .iloc[2:]
    mean_spectra2["Fup_VNIR"] = measurements["Fup_VNIR"]["0"]["normal"]["direct"].loc[start2:end2].mean()  # .iloc[2:]
    mean_spectra3["Fup_VNIR"] = measurements["Fup_VNIR"]["0"]["normal"]["direct"].loc[start3:end3].mean()  # .iloc[2:]
    mean_spectra1["Fup_SWIR"] = measurements["Fup_SWIR"]["0"]["normal"]["direct"].loc[start1:end1].mean()  # .iloc[2:]
    mean_spectra2["Fup_SWIR"] = measurements["Fup_SWIR"]["0"]["normal"]["direct"].loc[start2:end2].mean()  # .iloc[2:]
    mean_spectra3["Fup_SWIR"] = measurements["Fup_SWIR"]["0"]["normal"]["direct"].loc[start3:end3].mean()  # .iloc[2:]

# %% plot the 0° measurement for all channels separately
    h.set_cb_friendly_colors()
    for prop in properties:
        for channel in channels:
            fig, ax = plt.subplots()
            mean_spectra1[f"{prop}_{channel}"].plot(ax=ax, label="1st Measurement")
            mean_spectra2[f"{prop}_{channel}"].plot(ax=ax, label="2nd Measurement")
            mean_spectra3[f"{prop}_{channel}"].plot(ax=ax, label="3rd Measurement")
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
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    for id1, prop in enumerate(properties):
        for id2, channel in enumerate(channels):
            mean_spectra1[f"{prop}_{channel}"].plot(ax=axs[id1, id2], label="1st Measurement")
            mean_spectra2[f"{prop}_{channel}"].plot(ax=axs[id1, id2], label="2nd Measurement")
            mean_spectra3[f"{prop}_{channel}"].plot(ax=axs[id1, id2], label="3nd Measurement")
            axs[id1, id2].set_title(f"{prop} {channel}")

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
    for pair in tqdm(h.nested_dict_pairs_iterator(mean_spectras), desc="Plotting Mean Spectras"):
        prop_channel, angle, position, mtype, df = pair
        df.plot(title=f"{prop_channel} {angle}° {position} {mtype} Mean Spectrum", ylabel="Counts", xlabel="Pixel #")
        plt.grid()
        plt.legend(loc=8)
        # plt.show()
        figname = f"{prop_channel}_{angle}deg_{position}_{mtype}_mean_spectrum.png"
        plt.savefig(f"{plot_path}/mean_spectras/{figname}", dpi=100)
        log.debug(f"Saved {figname}")
        plt.close()

# %% set values < 1 to 1 to avoid problems with the correction factor calculation
    mean_spectras_cor = deepcopy(mean_spectras)
    for pair in h.nested_dict_pairs_iterator(mean_spectras_cor):
        prop_channel, angle, position, mtype, df = pair
        df[df < 1] = 1
        mean_spectras_cor[prop_channel][f"{angle}"][position][mtype] = df

    del mean_spectras  # delete variable from workspace to free memory

# %% create dataframe from dictionary
    dfs = pd.DataFrame()
    for pairs in h.nested_dict_pairs_iterator(mean_spectras_cor):
        prop_channel, angle, position, mtype, df = pairs
        df = pd.DataFrame(df, columns=["counts"]).reset_index()  # reset index to get a column with the pixel numbers
        df = df.rename(columns={"index": "pixel"})  # rename the index column to the pixel numbers
        df = df.assign(prop=prop_channel, angle=angle, position=position, mtype=mtype)
        dfs = pd.concat([dfs, df])

    dfs.reset_index(drop=True, inplace=True)

# %% calculate the direct cosine correction factors for each angle
    k_cos_dir = deepcopy(mean_spectras_cor)
    for pairs in h.nested_dict_pairs_iterator(mean_spectras_cor):
        prop, angle, position, mtype, F_angle = pairs
        if mtype == "direct":
            # remove dictionaries for mtype since only direct measurements are of importance for the cosine correction
            k_cos_dir[prop][angle][position].pop("direct")
            k_cos_dir[prop][angle][position].pop("diffuse")
            F_0 = mean_spectras_cor[prop]["0"][position][mtype]
            angle_rad = np.deg2rad(float(angle))
            k_cos_dir[prop][angle][position] = F_0 * np.cos(angle_rad) / F_angle

# %% create dataframe with cosine correction factors
    k_cos_dir_df = pd.DataFrame()
    for pairs in h.nested_dict_pairs_iterator(k_cos_dir):
        prop_channel, angle, position, df = pairs
        df = pd.DataFrame(df, columns=["k_cos"]).reset_index()  # reset index to get a column with the pixel numbers
        df = df.rename(columns={"index": "pixel"})  # rename the index column to the pixel numbers
        df = df.assign(prop=prop_channel, angle=angle, position=position)
        k_cos_dir_df = pd.concat([k_cos_dir_df, df])

    k_cos_dir_df.reset_index(drop=True, inplace=True)
    # remove angles greater 90°
    k_cos_dir_df = k_cos_dir_df[np.abs(k_cos_dir_df["angle"].astype(float)) < 95]

# %% calculate the diffuse cosine correction factors
    k_list, p_list, prop_list = list(), list(), list()  # define output lists
    props = np.unique(k_cos_dir_df.loc[:, "prop"])  # get all possible properties to loop over
    for prop in props:
        prop_df = k_cos_dir_df[k_cos_dir_df["prop"] == prop]  # select one property
        pixels = np.unique(prop_df.loc[:, "pixel"])  # get all possible pixels to loop over
        for pixel in pixels:
            pixel_df = prop_df[prop_df["pixel"] == pixel]  # select one pixel
            tmp_normal = pixel_df[pixel_df["position"] == "normal"]  # get normal measurements
            tmp_turned = pixel_df[pixel_df["position"] == "turned"]  # get turned measurements
            # select only normal measurements and drop unnecessary columns
            pixel_df = tmp_normal.drop(["pixel", "position", "prop"], axis=1)
            # replace normal positioned cosine correction factor with the mean of normal and turned
            pixel_df.loc[:, "k_cos"] = np.mean([tmp_normal["k_cos"].values, tmp_turned["k_cos"].values], axis=0)
            # convert angle to float and make them absolute to average over positive and negative directions
            pixel_df.loc[:, "angle"] = np.abs(pixel_df.loc[:, "angle"].astype(float))
            # average over both directions
            pixel_df = pixel_df.groupby("angle").mean().reset_index()


            def k_cos_dif(theta, df):
                """
                Function for diffuse cosine correction factor
                Args:
                    theta: angle of incoming radiation (rad)
                    df: DataFrame with angle and corresponding cosine correction factor

                Returns: Diffuse cosine correction factor

                """
                k_cos = interpolate.interp1d(df["angle"], df["k_cos"])  # create interpolation function
                return k_cos(np.rad2deg(theta)) * np.cos(theta) * np.sin(theta)


            # integrate over 0-90 deg
            k_cos_diff = 2 * integrate.quad(k_cos_dif, 0, np.pi/2, args=pixel_df, limit=200)[0]
            # append output to lists
            k_list.append(k_cos_diff)
            p_list.append(pixel)
            prop_list.append(prop)

    df_diff = pd.DataFrame({"k_cos_diff": k_list, "pixel": p_list, "prop": prop_list})  # create dataframe
    # split dataframe by inlet
    VN05_channels, VN11_channels = ["Fdw_VNIR", "Fdw_SWIR"], ["Fup_VNIR", "Fup_SWIR"]
    VN05 = df_diff.loc[df_diff["prop"].isin(VN05_channels)]
    VN11 = df_diff.loc[df_diff["prop"].isin(VN11_channels)]
    # split up information in the prop column
    VN05[["direction", "property"]] = VN05["prop"].str.split("_", expand=True)
    VN11[["direction", "property"]] = VN11["prop"].str.split("_", expand=True)
    VN05["channel"] = [cirrus_hl.smart_lookup[prop] for prop in VN05["prop"]]
    VN11["channel"] = [cirrus_hl.smart_lookup[prop] for prop in VN11["prop"]]
    # save to csv
    VN05.to_csv(f"{outfile_path}/HALO_SMART_VN05_diffuse_cosine_correction_factors.csv", index=False)
    VN11.to_csv(f"{outfile_path}/HALO_SMART_VN11_diffuse_cosine_correction_factors.csv", index=False)

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
                # select min and max of the colorbar depending on property
                if prop_channel == "Fdw_SWIR":
                    vmin, vmax = 0, 24000
                elif prop_channel == "Fup_SWIR":
                    vmin, vmax = 0, 20000
                else:
                    vmin, vmax = 0, 1300
                # plot
                fig, ax = plt.subplots()
                img = ax.pcolormesh(th, r, z_new, vmin=vmin, vmax=vmax, cmap='YlOrRd', shading="nearest")
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
        # select min and max of the colorbar depending on property
        if prop_channel == "Fdw_SWIR":
            vmin, vmax = 0, 24000
        elif prop_channel == "Fup_SWIR":
            vmin, vmax = 0, 20000
        else:
            vmin, vmax = 0, 1300
        # plot
        fig, ax = plt.subplots()
        img = ax.pcolormesh(th, r, F0_pivot, vmin=vmin, vmax=vmax, cmap='YlOrRd', shading="nearest")
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

# %% plot cosine correction factors in 2D and mean, median over pixels
    for prop_channel in k_cos_dir:
        k_tmp = k_cos_dir_df.loc[(k_cos_dir_df["prop"] == prop_channel), :]
        for position in k_tmp["position"].unique():
            k_cos = k_tmp.loc[(k_tmp["position"] == position), ["angle", "k_cos", "pixel"]]
            levels = k_cos.angle.unique()  # extract levels from angles
            # convert angles to categorical type to keep order when pivoting
            k_cos["angle"] = pd.Categorical(k_cos["angle"], categories=levels, ordered=True)
            k_cos_pivot = k_cos.pivot(index="angle", columns="pixel", values="k_cos")
            # arrange an artificial grid to plot on
            rad = np.arange(0, k_cos_pivot.shape[1])  # 1024 or 256 pixels
            a = levels.astype(float)  # -90 to 90° in 5° steps
            log.debug(f"Using angles: {a}")
            r, th = np.meshgrid(rad, a)

            # 2D plot
            fig, ax = plt.subplots()
            img = ax.pcolormesh(th, r, k_cos_pivot, cmap='coolwarm', shading="nearest")
            ax.grid()
            ax.set_title(f"{prop_channel} {position} direct Cosine Correction Factor")
            ax.set_xlabel("Angle (deg)")
            ax.set_ylabel("Pixel #")
            img.set_clim(0, 2)
            plt.colorbar(img, label="Correction Factor", extend="both")
            plt.tight_layout()
            # plt.show()
            figname = f"{prop_channel}_{position}_direct_cosine_correction_factors.png"
            plt.savefig(f"{plot_path}/correction_factors/{figname}", dpi=100)
            log.info(f"Saved {figname}")
            plt.close()

            # mean over all pixels
            fig, ax = plt.subplots()
            if "VNIR" in prop_channel:
                k_cos = k_cos[k_cos["pixel"] > 200]
            k_mean = k_cos.groupby("angle").mean().drop("pixel", axis=1)
            ax.plot(k_mean.index.astype(float), k_mean)
            ax.axhline(1, c="k", ls="--")  # horizontal line at 1
            ax.set_xticks(np.arange(-90, 91, 15))
            ax.grid()
            ax.set_title(f"{prop_channel} {position} direct Mean Cosine Correction Factor")
            ax.set_xlabel("Angle (deg)")
            ax.set_ylabel("Cosine Correction Factor")
            plt.tight_layout()
            # plt.show()
            figname = f"{prop_channel}_{position}_direct_mean_cosine_correction_factors.png"
            plt.savefig(f"{plot_path}/correction_factors/{figname}", dpi=100)
            log.info(f"Saved {figname}")
            plt.close()

            # median over all pixels
            fig, ax = plt.subplots()
            k_mean = k_cos.groupby("angle").median().drop("pixel", axis=1)
            ax.plot(k_mean.index.astype(float), k_mean)
            ax.axhline(1, c="k", ls="--")  # horizontal line at 1
            ax.set_xticks(np.arange(-90, 91, 15))
            ax.grid()
            ax.set_title(f"{prop_channel} {position} direct Median Cosine Correction Factor")
            ax.set_xlabel("Angle (deg)")
            ax.set_ylabel("Cosine Correction Factor")
            plt.tight_layout()
            # plt.show()
            figname = f"{prop_channel}_{position}_direct_median_cosine_correction_factors.png"
            plt.savefig(f"{plot_path}/correction_factors/{figname}", dpi=100)
            log.info(f"Saved {figname}")
            plt.close()

# %% plot cosine correction factor for each pixel/wavelength
    h.set_cb_friendly_colors()
    pixel_path = h.get_path("pixel_wl")
    for prop_channel in k_cos_dir:
        k_cos = k_cos_dir_df.loc[(k_cos_dir_df["prop"] == prop_channel), ["angle", "k_cos", "pixel"]]
        k_cos["angle"] = k_cos["angle"].astype(float)  # convert angles to float for better plotting
        pixel_wl = reader.read_pixel_to_wavelength(pixel_path, cirrus_hl.smart_lookup[prop_channel])
        k_cos = pd.merge(k_cos, pixel_wl, on="pixel")
        for pixel in tqdm(k_cos["pixel"].unique(), desc=f"{prop_channel}", unit=" Pixel"):
            k = k_cos.loc[(k_cos["pixel"] == pixel), :]
            fig, ax = plt.subplots()
            k.plot(x="angle", y="k_cos", ax=ax, label=f"{k['wavelength'].iloc[0]} nm",
                   title=f"{prop_channel} Cosine correction factor for {k['wavelength'].iloc[0]} nm",
                   ylabel="Cosine Correction Factor", xlabel="Angle (°)")
            ax.set_xticks(np.arange(-90, 95, 15))
            ax.set_ylim(0, 2)
            ax.axhline(1, c="k", ls="--")
            ax.grid()
            ax.legend(loc=8)
            # plt.show()
            figname = f"{prop_channel}_{pixel:04d}_cosine_correction_factor.png"
            plt.savefig(f"{plot_path}/correction_factors_single/{figname}", dpi=100)
            log.debug(f"Saved {figname}")
            plt.close()

# %% save correction factors to file, one for each inlet
    out_df = k_cos_dir_df.reset_index(drop=True)
    # separate inlets
    VN05_channels, VN11_channels = ["Fdw_VNIR", "Fdw_SWIR"], ["Fup_VNIR", "Fup_SWIR"]
    VN05 = out_df.loc[out_df["prop"].isin(VN05_channels)]
    VN11 = out_df.loc[out_df["prop"].isin(VN11_channels)]
    # split up information in the prop column
    VN05[["direction", "property"]] = VN05["prop"].str.split("_", expand=True)
    VN11[["direction", "property"]] = VN11["prop"].str.split("_", expand=True)
    VN05["channel"] = [cirrus_hl.smart_lookup[prop] for prop in VN05["prop"]]
    VN11["channel"] = [cirrus_hl.smart_lookup[prop] for prop in VN11["prop"]]
    # save file like this for further use
    VN11.to_csv(f"{outfile_path}/HALO_SMART_VN11_cosine_correction_factors.csv", index=False)
    VN05.to_csv(f"{outfile_path}/HALO_SMART_VN05_cosine_correction_factors.csv", index=False)

# %% interpolate correction factors to 1 deg and save to file, one for each inlet
    out_df = k_cos_dir_df.reset_index(drop=True)
    out_df["angle"] = out_df["angle"].astype(int)  # convert angle from str to int

    tmp = out_df.set_index(["pixel", "prop", "position"])
    func = interpolate.interp1d(tmp["angle"], tmp["k_cos"])
    int_kcos = func(np.arange(-90, 91, 1))
    # separate inlets
    VN05_channels, VN11_channels = ["Fdw_VNIR", "Fdw_SWIR"], ["Fup_VNIR", "Fup_SWIR"]
    VN05 = out_df.loc[out_df["prop"].isin(VN05_channels)]
    VN11 = out_df.loc[out_df["prop"].isin(VN11_channels)]
    # split up information in the prop column
    VN05[["direction", "property"]] = VN05["prop"].str.split("_", expand=True)
    VN11[["direction", "property"]] = VN11["prop"].str.split("_", expand=True)
    VN05["channel"] = [cirrus_hl.smart_lookup[prop] for prop in VN05["prop"]]
    VN11["channel"] = [cirrus_hl.smart_lookup[prop] for prop in VN11["prop"]]

    # save file like this for further use
    VN11.to_csv(f"{outfile_path}/HALO_SMART_VN11_cosine_correction_factors.csv", index=False)
    VN05.to_csv(f"{outfile_path}/HALO_SMART_VN05_cosine_correction_factors.csv", index=False)
