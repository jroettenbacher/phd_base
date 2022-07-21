#!/usr/bin/env python
"""Go through all transfer calibrations of CIRRUS-HL and check the quality of the calibration

Part1: Check transfer calibs over the course of the campaign
Part2: Compare transfer calibs calculated with different lab calibrations

Results - Part 1
^^^^^^^^^^^^^^^^^

- 20210629 Fup_SWIR/Fdw_SWIR -> Big increase in counts during calibration -> unstable spectrometer? Discard and use a different calibration.
- 20210711

*author*: Johannes Röttenbacher
"""
if __name__ == "__main__":
# %% module import
    from pylim import reader
    from pylim.cirrus_hl import smart_lookup, transfer_calibs
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
    lab_calib = "before"
    calib_path = f"{h.get_path('calib')}/transfer_calibs_{lab_calib}_campaign"
    plot_path = f"{h.get_path('plot')}/quality_check_calibration"

# %% list all files from one spectrometer
    prop = "Fdw_SWIR"
    files = [f for f in os.listdir(calib_path) if smart_lookup[prop] in f]

# %% select only normalized, 300ms transfer calib files
    files = [f for f in files if "norm" in f and "300ms" in f]
    files = [f for f in files if "transfer" in f]
    files.sort()

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
    zoom = True  # zoom in on y axis
    ylims = dict(Fdw_SWIR=(0, 20), Fup_SWIR=(0, 20))

    fig, ax = plt.subplots(figsize=(10, 6))
    for date_str in date_strs:
        df[df["date"] == date_str].sort_values(["wavelength"]).plot(x="wavelength", y="rel_ulli", label=date_str, ax=ax)

    if zoom:
        ax.set_ylim(ylims[prop])
        zoom = "_zoom"
    else:
        zoom = ""
    ax.set_ylabel("$S_{ulli, lab} / S_{ulli, field}$")
    ax.set_xlabel("Wavenlength (nm)")
    ax.set_title(f"Relation between Lab Calib measurement {lab_calib} \nand Transfer Calib measurement - {prop}")
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
        wl1, wl2 = 1400, 1450  # set wavelengths for averaging
    else:
        wl1, wl2 = 550, 570
    df_ts = df[df["wavelength"].between(wl1, wl2)]
    df_mean = df_ts.groupby("date").mean().reset_index()
    # df_mean["dt"] = pd.to_datetime(df_mean.index.values, format="%Y_%m_%d")
    # df_mean.set_index(df_mean["dt"], inplace=True)

# %% plot a time series of the calibration factor
    fig, ax = plt.subplots(figsize=(10, 6))
    df_mean.plot(y="c_field", ax=ax, label="$c_{field}$")
    # df_mean.plot(y="c_lab", c="#117733", ax=ax, label="$c_{lab}$")
    # ax.set_ylim((1, 2.5))
    ax.set_yscale("log")
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

# %% investigate the last four days in more detail because they look wrong, plot calibration files (Lab View bug!)
    mpl.rcdefaults()  # set default plotting options
    flight = "Flight_20210719a"  # "Flight_20210721a"  # "Flight_20210721b" "Flight_20210723a" "Flight_20210728a" "Flight_20210729a"
    transfer_cali_date = transfer_calibs[flight]
    instrument = smart_lookup[prop][:5]
    calibration = f"{instrument}_transfer_calib_{transfer_cali_date}"
    trans_calib_path = f"{calib_path}/{calibration}/Tint_300ms"
    trans_calib_path_dark = f"{calib_path}/{calibration}/dark_300ms"
    trans_calib_files = [f for f in os.listdir(trans_calib_path) if prop in f]
    trans_calib_files_dark = [f for f in os.listdir(trans_calib_path_dark) if prop in f]
    plot_paths = [plot_path, f"{plot_path}/dark"]
    for path, filenames, p_path in zip([trans_calib_path, trans_calib_path_dark],
                                       [trans_calib_files, trans_calib_files_dark], plot_paths):
        for filename in filenames:
            log.info(f"Plotting {path}/{filename}")
            plot_smart_data(flight, filename, wavelength="all", path=path, plot_path=p_path, save_fig=True)

# %% plot mean dark current for SWIR over flight; read in all raw files
    flight = "Flight_20210728a"
    props = ["Fdw_SWIR", "Fup_SWIR"]
    dfs, dfs_plot, files_dict = dict(), dict(), dict()
    raw_path = h.get_path("raw", flight)
    bahamas_path = h.get_path("bahamas", flight)
    bahamas_file = [f for f in os.listdir(bahamas_path) if f.endswith(".nc")][0]
    bahamas_ds = reader.read_bahamas(f"{bahamas_path}/{bahamas_file}")
    for prop in props:
        files_dict[prop] = [f for f in os.listdir(raw_path) if prop in f]
        dfs[prop] = pd.concat([reader.read_smart_raw(raw_path, file) for file in files_dict[prop]])
        # select only rows where the shutter is closed and take mean over all pixels
        dfs_plot[prop] = dfs[prop][dfs[prop]["shutter"] == 0].iloc[:, 2:].mean(axis=1)

    # plot mean dark current over flight
    h.set_cb_friendly_colors()
    fig, axs = plt.subplots(nrows=2, sharex="all", figsize=(10, 6))
    for prop in props:
        dfs_plot[prop].plot(ax=axs[0], ylabel="Netto Counts", label=f"{prop}")
    bahamas_ds["IRS_ALT_km"] = bahamas_ds["IRS_ALT"] / 1000
    bahamas_ds["IRS_ALT_km"].plot(ax=axs[1], label="BAHAMAS Altitude", color="#DDCC77")
    axs[0].set_ylim((1500, 4000))
    axs[1].set_ylabel("Altitude (km)")
    axs[1].set_xlabel("Time (UTC)")
    for ax in axs:
        ax.legend()
        ax.grid()
    fig.suptitle(f"{flight} - Mean Dark Current")
    plt.show()
    # plt.savefig(f"{plot_path}/{flight}_SWIR_mean_dark_current.png", dpi=100)
    plt.close()

# %% plot mean dark current for VNIR over flight; read in all raw files
    flight = "Flight_20210723a"
    props = ["Fdw_VNIR", "Fup_VNIR"]
    dfs, dfs_plot, files_dict = dict(), dict(), dict()
    raw_path = h.get_path("raw", flight)
    bahamas_path = h.get_path("bahamas", flight)
    bahamas_file = [f for f in os.listdir(bahamas_path) if f.endswith(".nc")][0]
    bahamas_ds = reader.read_bahamas(f"{bahamas_path}/{bahamas_file}")
    for prop in props:
        files_dict[prop] = [f for f in os.listdir(raw_path) if prop in f]
        dfs[prop] = pd.concat([reader.read_smart_raw(raw_path, file) for file in files_dict[prop]])
        # select only columns where no signal is measured in the VNIR, drop t_int and shutter column
        dfs_plot[prop] = dfs[prop].iloc[:, 2:150].mean(axis=1)

    # plot mean dark current over flight VNIR
    h.set_cb_friendly_colors()
    fig, axs = plt.subplots(nrows=2, sharex="all", figsize=(10, 6))
    for prop in props:
        dfs_plot[prop].plot(ax=axs[0], ylabel="Netto Counts", label=f"{prop}")
    bahamas_ds["IRS_ALT_km"] = bahamas_ds["IRS_ALT"] / 1000
    bahamas_ds["IRS_ALT_km"].plot(ax=axs[1], label="BAHAMAS Altitude", color="#DDCC77")
    axs[0].set_ylim((90, 230))
    axs[1].set_ylabel("Altitude (km)")
    axs[1].set_xlabel("Time (UTC)")
    for ax in axs:
        ax.legend()
        ax.grid()
    fig.suptitle(f"{flight} - Mean Dark Current")
    plt.show()
    # plt.savefig(f"{plot_path}/{flight}_VNIR_mean_dark_current.png", dpi=100)
    plt.close()

# %% PART 2: read in lab calibration factor from before and after campaign lab calib
    inlet = smart_lookup[f"{prop}"]
    date_before = "2021_03_29" if "ASP06" in inlet else "2021_03_18"
    date_after = "2021_08_09"
    # read in data
    lab_calib_before = pd.read_csv(f"{calib_path}/../{date_before}_{inlet}_{prop}_lab_calib_norm.dat")
    lab_calib_after = pd.read_csv(f"{calib_path}/../{date_after}_{inlet}_{prop}_lab_calib_norm.dat")

# %% plot lab calibration from before and after campaign
    h.set_cb_friendly_colors()
    # plot variables
    variables = dict(c_lab=dict(title="Laboratory Calibration Factor", ylabel="Laboratory calibration factor",
                                ylim=(0, 0.1), yscale="log"),
                     S0=dict(title="Lamp measurement", ylabel="Dark current corrected counts"),
                     S_ulli=dict(title="Ulli sphere measurement", ylabel="Normalized dark current corrected counts"),
                     F_ulli=dict(title="Irradiance measured from Ulli sphere",
                                 ylabel="Irradiance (W$\,$m$^{-2}\,$nm$^{-1}$)"))
    var = "F_ulli"
    meta = variables[var]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot("wavelength", var, data=lab_calib_before, label="Before Campaign")
    ax.plot("wavelength", var, data=lab_calib_after, label="After Campaign")
    ax.grid()
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(meta["ylabel"])
    ax.set_title(f"{meta['title']} - {prop}" if "title" in meta else f"{prop}")
    ax.set_yscale(meta["yscale"]) if "yscale" in meta else None
    ax.set_ylim(meta["ylim"]) if "ylim" in meta else None
    ax.legend()
    plt.tight_layout()
    # plt.show()
    figname = f"{plot_path}/SMART_lab_calib_comparison_{var}_{prop}.png"
    plt.savefig(figname, dpi=300)
    print(f"Saved {figname}")
    plt.close()

# %% plot relation between before and after
    lab_calib_relation = lab_calib_before / lab_calib_after
    lab_calib_relation["wavelength"] = lab_calib_before["wavelength"]
    var = "F_ulli"
    meta = variables[var]
    fig, ax = plt.subplots()
    ax.plot("wavelength", var, data=lab_calib_relation, label="Relation between before and after")
    ax.grid()
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(meta["ylabel"])
    ax.set_title(meta["title"] if "title" in meta else "")
    # ax.set_ylim(meta["ylim"]) if "ylim" in meta else None
    ax.legend()
    plt.show()
    plt.close()

# %% plot all field calibration factors
    fig, ax = plt.subplots(figsize=(10, 6))
    for date_str in date_strs:
        df[df["date"] == date_str].sort_values(["wavelength"]).plot(x="wavelength", y="c_field", label=date_str, ax=ax)
    ax.set_xlim(890, 2200)
    ax.set_yscale("log")
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("Field calibration factor")
    ax.set_xlabel("Wavenlength (nm)")
    ax.set_title(f"Field calibration factor - {prop}")
    ax.grid()
    ax.legend(bbox_to_anchor=(1.04, 1.1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{plot_path}/SMART_calib_c_field_{prop}_{lab_calib}.png", dpi=300)
    # plt.show()
    plt.close()
