#!/usr/bin/env python
"""Calculate the effective receiving area of an irradiance inlet according to the R^2 law

author: Johannes Röttenbacher
"""
if __name__ == "__main__":
# %% module import
    import pylim.helpers as h
    from pylim import smart, reader
    import os
    import pandas as pd
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from tqdm import tqdm
    import logging
    log = logging.getLogger("pylim")
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)

# %% set paths
    inlet = "VN11"
    campaign = "halo-ac3"
    folder = f"ASP06_{inlet}_effective_receiving_area_v2"
    calib_path = h.get_path("calib", campaign=campaign)
    input_path = os.path.join(calib_path, folder)
    plot_path = f"C:/Users/Johannes/Documents/Doktor/instruments/SMART/effective_receiving_area_{inlet}"
    pixel_wl_path = h.get_path("pixel_wl")

# %% correct all data for the dark current - merge VNIR dark measurement files before correcting the calib files
    for dirpath, dirs, files in os.walk(input_path):
        log.debug(f"Working on {dirpath}")
        if "diffuse" in dirpath:
            try:
                vnir_dark_files = [f for f in files if f.endswith(f"Fdw_VNIR.dat")]
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
    for dirpath, dirs, files in os.walk(input_path):
        log.info(f"Working on {dirpath}")
        parent = os.path.dirname(dirpath)
        for file in files:
            if file.endswith("SWIR.dat"):
                log.info(f"Working on {dirpath}/{file}")
                smart_cor = smart.correct_smart_dark_current("", file, option=2, path=dirpath)

            elif file.endswith("VNIR.dat"):
                log.debug(f"Working on {dirpath}/{file}")
                _, channel, direction = smart.get_info_from_filename(file)
                dark_dir = os.path.join(parent, "30cm_diffuse")
                try:
                    dark_file = [f for f in os.listdir(dark_dir) if direction in f and channel in f][0]
                    measurement = reader.read_smart_raw(dirpath, file).iloc[:, 2:]
                    dark_current = reader.read_smart_raw(dark_dir, dark_file)
                    dark_current = dark_current.iloc[:, 2:].mean()
                    smart_cor = measurement - dark_current
                except (FileNotFoundError, IndexError):
                    log.debug(f"No diffuse {direction}_{channel} file found in {dark_dir}")
                    # scale 0° dark current with current dark pixel measurements
                    dark_dir = f"{parent}/30cm_diffuse"
                    dark_file = [f for f in os.listdir(dark_dir) if direction in f and channel in f][0]
                    dark_current = reader.read_smart_raw(dark_dir, dark_file).iloc[:, 2:].mean()  # drop tint and shutter
                    measurement = reader.read_smart_raw(dirpath, file).iloc[:, 2:]
                    dark_scale = dark_current * np.mean(measurement.mean().iloc[19:99]) / np.mean(dark_current.iloc[19: 99])
                    dark_scale = dark_scale - dark_scale.rolling(20, min_periods=1).mean()
                    dark_scale2 = dark_current.rolling(20, min_periods=1).mean() + (
                            np.mean(measurement.mean().iloc[19:99]) - np.mean((dark_current.iloc[19:99])))
                    smart_cor = measurement - dark_scale2 - dark_scale
            else:
                # skip all _cor.dat files
                continue

            outname = f"{dirpath}/{file.replace('.dat', '_cor.dat')}"
            smart_cor.to_csv(outname, sep="\t", float_format="%.0f")
            log.info(f"Saved {outname}")

    # merge minutely corrected files to one file
    channels = ["SWIR", "VNIR"]
    for dirpath, dirs, files in os.walk(input_path):
        log.info(f"Working on {dirpath}")
        for channel in channels:
            try:
                filename = [file for file in files if file.endswith(f"{channel}_cor.dat")]
                df = pd.concat([pd.read_csv(f"{dirpath}/{file}", sep="\t", index_col="time") for file in filename])
                # delete all minutely files
                for file in filename:
                    os.remove(os.path.join(dirpath, file))
                    log.info(f"Deleted {dirpath}/{file}")
                outname = f"{dirpath}/{filename[0]}"
                df.to_csv(outname, sep="\t")
                log.info(f"Saved {outname}")
            except ValueError:
                pass

# %% read in all spectra and average them to one spectra for each distance measurement
    measurements = dict()
    distances = np.arange(30, 65, 5)
    for channel in ["VNIR", "SWIR"]:
        measurements[channel] = dict()
        for distance in distances:
            # start_id, end_id = (200, 900) if channel == "Fdw_VNIR" else (50, 220)
            inpath = os.path.join(input_path, f"{distance}cm")
            cor_files = [f for f in os.listdir(inpath) if channel in f and f.endswith("_cor.dat")]
            smart_cor = pd.concat([reader.read_smart_cor(inpath, file) for file in cor_files])
            mean_spectra = smart_cor.mean(skipna=True)
            measurements[channel][f"{distance}"] = mean_spectra

# %% convert dictionary to dataframe
    dfs = pd.DataFrame()
    for pairs in h.nested_dict_pairs_iterator(measurements):
        channel, distance, df = pairs
        df = pd.DataFrame(df, columns=["counts"]).reset_index()  # reset index to get a column with the pixel numbers
        df = df.rename(columns={"index": "pixel"})  # rename the index column to the pixel numbers
        df = df.assign(prop=channel, distance=int(distance))
        dfs = pd.concat([dfs, df])

    dfs.reset_index(drop=True, inplace=True)

# %% set negative counts to Nan and calculate 1/sqrt(F)
    dfs["counts"] = dfs["counts"].mask(dfs["counts"] < 0)
    dfs["plot_data"] = 1 / np.sqrt(dfs["counts"])

# %% calculate linear regression for each pixel
    lin_reg = dict()
    for channel in dfs["prop"].unique():
        lin_reg[channel] = dict()
        df_tmp = dfs[dfs["prop"] == channel]
        for pixel in df_tmp["pixel"].unique():
            df_sel = df_tmp[df_tmp["pixel"] == pixel]
            slope, intercept, r_value, p_value, std_err = stats.linregress(df_sel['plot_data'], df_sel['distance'])
            lin_reg[channel][f"{pixel}"] = (slope, intercept)

# %% convert to data frame
    lin_reg_df = pd.DataFrame()
    for pairs in h.nested_dict_pairs_iterator(lin_reg):
        channel, pixel, value = pairs
        df = pd.DataFrame({"prop": channel, "pixel": int(pixel), "slope": value[0], "intercept": value[1]}, index=[0])
        lin_reg_df = pd.concat([lin_reg_df, df])

    lin_reg_df.reset_index(drop=True, inplace=True)

# %% plot each linear relationship
    h.set_cb_friendly_colors()
    for prop in dfs["prop"].unique():
        df_tmp = dfs[dfs["prop"] == prop]
        lin_tmp = lin_reg_df[lin_reg_df["prop"] == prop]
        for pixel in tqdm(df_tmp["pixel"].unique(), desc=prop):
            df_plot = df_tmp[df_tmp["pixel"] == pixel]
            lin_plot = lin_tmp[lin_tmp["pixel"] == pixel]
            x_values = np.linspace(0, df_plot["plot_data"].max())
            y_values = lin_plot["slope"].values * x_values + lin_plot["intercept"].values
            fig, ax = plt.subplots()
            df_plot.plot.scatter(x="plot_data", y="distance", ylabel="Distance (cm)", xlabel=r"$1 / \sqrt{F}}$", ax=ax,
                                 label="Measurements", color="#117733")
            ax.plot(x_values, y_values,
                    label=f"Linear Regression: {lin_plot['slope'].iat[0]:.1f} * x + {lin_plot['intercept'].iat[0]:.2f}")
            ax.grid()
            ax.legend(loc=2)
            ax.set_ylim(-5, 65)
            ax.set_xlim(0, df_plot["plot_data"].max())
            ax.set_title(f"Linear regression of pixel {pixel} from channel {prop}")
            plt.tight_layout()
            # plt.show()
            figname = f"{inlet}_{prop}_{pixel:04d}_linear_regression.png"
            plt.savefig(f"{plot_path}/lin_rel_single/{figname}", dpi=100)
            log.debug(f"Saved {figname}")
            plt.close()

    # %% calculate rolling mean for each channel
    dfs = list()
    for channel in lin_reg_df["prop"].unique():
        df_tmp = lin_reg_df.loc[lin_reg_df["prop"] == channel, :]
        mean_intercept = df_tmp.loc[:, ["intercept"]].rolling(window=10, min_periods=1).mean()
        df_tmp = df_tmp.assign(mean_intercept=mean_intercept)
        dfs.append(df_tmp)

    lin_reg_df_v2 = pd.concat(dfs)

    # %% read in pixel to wavelength mapping to use as x axis for plots
    pixel_wl_VNIR = reader.read_pixel_to_wavelength(pixel_wl_path, "ASP06_J6")
    pixel_wl_SWIR = reader.read_pixel_to_wavelength(pixel_wl_path, "ASP06_J5")
    pixel_wl_VNIR["prop"] = "VNIR"
    pixel_wl_SWIR["prop"] = "SWIR"
    pixel_wl = pd.concat([pixel_wl_VNIR, pixel_wl_SWIR])

    lin_reg_df_v3 = pd.merge(lin_reg_df_v2, pixel_wl, on=["prop", "pixel"])
    # sort by wavelength and reset index accordingly
    lin_reg_df_v3 = lin_reg_df_v3.sort_values(by="wavelength").reset_index(drop=True)

    # %% calculate rolling mean for merged data frame
    lin_reg_df_v3["mean_intercept_all"] = lin_reg_df_v3.loc[:, ["intercept"]].rolling(window=10, min_periods=1).mean()

    # %% plot delta r for each wavelength and the rolling mean
    h.set_cb_friendly_colors()
    df_plot = lin_reg_df_v3.loc[lin_reg_df_v3["intercept"].abs() < 5]
    groups = df_plot.groupby("prop")
    fig, ax = plt.subplots()
    for name, group in groups:
        ax.scatter(group["wavelength"], group["intercept"], label=f"{name} (Median: {group['intercept'].median():.2f})", s=12)

    ax.plot("wavelength", "mean_intercept_all", data=df_plot, label="Rolling mean (window: 10)", linewidth=3, c="#CC6677", )
    ax.set_title(fr"$\Delta r$ (intercept) for each pixel")
    ax.set_ylabel(r"$\Delta r$ (cm)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylim(-5, 5)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(color='none', label=f"Median all: {df_plot['intercept'].median():.2f}"))
    handles.append(Patch(color='none', label=f"Mean Channels: {groups['intercept'].median().mean():.2f}"))
    ax.legend(handles=handles, ncol=2, bbox_to_anchor=(0.5, -0.15), loc="upper center")
    ax.grid()
    plt.tight_layout()
    # plt.show()
    figname = f"{inlet}_effective_receiving_area_delta_r_v2.png"
    plt.savefig(f"{plot_path}/{figname}", dpi=100)
    log.info(f"Saved {figname}")
    plt.close()
