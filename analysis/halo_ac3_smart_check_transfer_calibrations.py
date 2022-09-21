#!/usr/bin/env python
"""Go through all transfer calibrations and check the quality of the calibration


.. figure:: figures/SMART_calib_c_field_after_Fdw_SWIR_zoom.png

    Field calibration factors for all transfer calibrations using the after campaign lab calibration.

It can be seen that the first and last transfer calibration result in quite different calibration factors.
However, the first transfer calibration was done with different spectrometers and the last transfer calibration happened after the optical fibers were detached and reattached to the spectrometers.
The difference is thus not surprising.
The performance of the spectrometers throughout the campaign was stable.
Judging from the steep increase in the field calibration factor at the borders of the spectrometers **950 nm** seems to be a reasonable wavelength to switch from VNIR to SWIR for the combined output data.

.. figure:: figures/SMART_calib_c_field_after_Fdw_VNIR_zoom.png

    Field calibration factors for all transfer calibrations using the after campaign lab calibration.

*author*: Johannes Röttenbacher
"""
if __name__ == "__main__":
# %% module import
    from pylim.halo_ac3 import smart_lookup, transfer_calibs
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
    doc_plot_path = "C:/Users/Johannes/PycharmProjects/phd_base/docs/figures"

# %% list all files from one spectrometer
    prop = "Fdw_VNIR"
    lab_calib = "after"  # which lab calib was used for the calibration of the transfer calib (after or before)
    files = [f for f in os.listdir(f"{calib_path}/transfer_calibs_{lab_calib}_campaign") if smart_lookup[prop] in f]

# %% select only normalized and transfer calib files
    files = [f for f in files if "norm" in f and "transfer" in f]
    files.sort()

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

# %% plot field calibration factor
    zoom = True  # zoom in on y axis

    fig, ax = plt.subplots(figsize=(10, 6))
    for date_str in date_strs:
        df[df["date"] == date_str].sort_values(["wavelength"]).plot(x="wavelength", y="c_field", label=date_str, ax=ax)

    if zoom:
        ax.set_ylim((0, 0.05)) if prop == "Fdw_SWIR" else ax.set_ylim((0, 1))
        zoom = "_zoom"
    else:
        zoom = ""
    ax.set_ylabel("Field Calibration Factor")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_title(f"Field Calibration Factor for each Transfer Calibration - {prop}", size=14)
    ax.grid()
    ax.legend(bbox_to_anchor=(1.04, 1.1), loc="upper left")
    plt.tight_layout()
    xticks = ax.get_xticks()
    ax.set_xticks(np.append(xticks[1:-1], 950))
    figname = f"{plot_path}/SMART_calib_c_field_{lab_calib}_{prop}{zoom}.png"
    figname = f"{doc_plot_path}/SMART_calib_c_field_{lab_calib}_{prop}{zoom}.png"
    plt.savefig(figname, dpi=100)
    print(f"Saved {figname}")
    plt.show()
    plt.close()

# %% plot relation between lab calib measurement and each transfer calib measurement
    zoom = False  # zoom in on y axis

    fig, ax = plt.subplots(figsize=(10, 6))
    for date_str in date_strs:
        df[df["date"] == date_str].sort_values(["wavelength"]).plot(x="wavelength", y="rel_ulli", label=date_str, ax=ax)

    if zoom:
        ax.set_ylim((0, 2))
        zoom = "_zoom"
    else:
        zoom = ""
    ax.set_ylabel("$S_{ulli, lab} / S_{ulli, field}$")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_title(f"Relation between Lab Calib measurement \nand Transfer Calib measurement - {prop}")
    ax.grid()
    ax.legend(bbox_to_anchor=(1.04, 1.1), loc="upper left")
    plt.tight_layout()
    # plt.show()
    figname = f"{plot_path}/SMART_calib_rel_lab-field_{lab_calib}_{prop}{zoom}.png"
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
    figname = f"{plot_path}/SMART_calib_factors_{lab_calib}_{prop}.png"
    plt.savefig(figname, dpi=100)
    print(f"Saved {figname}")
    plt.close()
