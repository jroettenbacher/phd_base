#!/usr/bin/env python
"""Go through all transfer calibrations and check the quality of the calibration

*author*: Johannes Röttenbacher
"""
if __name__ == "__main__":
# %% module import
    from pylim import reader
    from pylim.halo_ac3 import smart_lookup, transfer_calibs
    from pylim.smart import plot_smart_data
    import pylim.helpers as h
    import os
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import logging

    log = logging.getLogger("pylim")
    log.setLevel(logging.INFO)

# %% set paths
    campaign = "halo-ac3"
    calib_path = h.get_path("calib", campaign=campaign)
    plot_path = f"{h.get_path('plot', campaign=campaign)}/quality_check_calibration"
    if campaign == "halo-ac3":
        lookup = smart_lookup

# %% list all files from one spectrometer
    prop = "Fdw_SWIR"
    files = [f for f in os.listdir(calib_path) if lookup[prop] in f]

# %% select only normalized and transfer calib files
    files = [f for f in files if "norm" in f and "transfer" in f]
    files.sort()
    if campaign == "cirrus-hl":
        if prop == "Fdw_SWIR" or prop == "Fdw_VNIR" or prop == "Fup_VNIR":
            files.pop(2)  # remove 500ms file from the 16th
        elif prop == "Fup_SWIR":
            files.pop(1)  # remove 500ms file from the 16th
        else:
            pass

# %% compare 300ms and 500ms normalized measurements for Fdw_SWIR
    # file_300, file_500 = files[1], files[2]  # TODO

# %% read in transfer calib file and add timestamp from filename
    date_strs = [f[0:10] for f in files]  # extract date strings from filenames
    df = pd.DataFrame()
    for f, date_str in zip(files, date_strs):
        df_tmp = pd.read_csv(f"{calib_path}/{f}")
        df_tmp["date"] = np.repeat(date_str, len(df_tmp))
        df = pd.concat([df, df_tmp])

    df = df.reset_index(drop=True)

# %% set plotting layout options
    colors = plt.cm.tab20.colors  # get tab20 colors
    plt.rc('axes', prop_cycle=(mpl.cycler('color', colors)))  # Set the default color cycle
    plt.rc('font', family="serif", size=14)

# %% plot relation between lab calib measurement and each transfer calib measurement
    zoom = False  # zoom in on y axis

    fig, ax = plt.subplots(figsize=(10, 6))
    for date_str in date_strs:
        df[df["date"] == date_str].sort_values(["wavelength"]).plot(x="wavelength", y="rel_ulli", label=date_str, ax=ax)

    if zoom:
        ax.set_ylim((0, 10))
        zoom = "_zoom"
    else:
        zoom = ""
    ax.set_ylabel("$S_{ulli, lab} / S_{ulli, field}$")
    ax.set_xlabel("Wavenlength (nm)")
    ax.set_title(f"Relation between Lab Calib measurement \nand Transfer Calib measurement - {prop}")
    ax.grid()
    ax.legend(bbox_to_anchor=(1.04, 1.1), loc="upper left")
    plt.tight_layout()
    # plt.show()
    figname = f"{plot_path}/SMART_calib_rel_lab-field_{prop}{zoom}.png"
    plt.savefig(figname, dpi=100)
    print(f"Saved {figname}")
    plt.close()

# %% take the average over n pixels and prepare data frame for plotting
    if "SWIR" in prop:
        wl1, wl2 = 1200, 1250  # set wavelengths for averaging
    else:
        wl1, wl2 = 550, 570
    df_ts = df[df["wavelength"].between(wl1, wl2)]
    df_mean = df_ts.groupby("date").mean().reset_index()
    # df_mean["dt"] = pd.to_datetime(df_mean.index.values, format="%Y_%m_%d")
    # df_mean.set_index(df_mean["dt"], inplace=True)

# %% plot a time series of the calibration factor
    fig, ax = plt.subplots(figsize=(10, 6))
    df_mean.plot(y="c_field", marker="o", ax=ax, label="$c_{field}$")
    # df_mean.plot(y="c_lab", c="#117733", ax=ax, label="$c_{lab}$")
    # ax.set_ylim((1, 2.5))
    # ax.set_yscale("log")
    ax.set_xticks(df_mean.index.values)
    ax.set_xticklabels(df_mean.date.values, fontsize=14, rotation=45, ha="right")
    # ax.tick_params(axis="x", labelsize=12)
    ax.set_ylabel("Calibration Factor")
    ax.set_xlabel("Date")
    ax.set_title(f"{prop} - Evolution of the Field Calibration Factor\n for the mean of {wl1} - {wl2} nm")
    ax.grid()
    ax.legend()
    plt.tight_layout()
    # plt.show()
    figname = f"{plot_path}/SMART_calib_factors_{prop}.png"
    plt.savefig(figname, dpi=100)
    print(f"Saved {figname}")
    plt.close()
