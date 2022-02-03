#!/usr/bin/env python
"""Script to read in calibration files and calculate calibration factors for lab calibration of ASP06

1. set property to work with (SWIR, VNIR), direction = Fup (uses setup names from CIRRUS-HL)
2. read in 1000W lamp file
3. read in calibration lamp measurements
4. read in dark current measurements
5. read in pixel to wavelength mapping and interpolate lamp output onto pixel/wavelength of spectrometer
6. plot lamp measurements
7. read in ulli sphere measurements
8. plot ulli measurements
9. write dat file with all information

The smart lookup from CIRRUS-HL is used because the filenames were not changed before the calibration.
See :ref:`analysis:smart_process_lab_calib_halo_ac3.py` for details.

author: Johannes Roettenbacher
"""
if __name__ == "__main__":
    # %%
    import pylim.helpers as h
    from pylim import reader, smart
    from pylim.cirrus_hl import lookup
    import os
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    import pandas as pd
    import logging

    # %% set up logger
    log = logging.getLogger("pylim")
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)

    # %% User Input
    prop = "SWIR"  # set property to work on (VNIR or SWIR)
    direction = "Fup"  # these are the spectrometers which will measure Fdw during HALO-AC3
    normalize = False  # normalize counts to integration time
    t_int = 300  # integration time of calibration measurement
    base = "ASP06_lab_calibration_before"

    # %% set paths
    campaign = "halo-ac3"
    pixel_path, calib_path = h.get_path("pixel_wl", campaign=campaign), h.get_path("calib", campaign=campaign)
    plot_path = "C:/Users/Johannes/Documents/Doktor/campaigns/HALO-AC3/SMART/plots/lab_calib"

    # %% read lamp file
    lamp = reader.read_lamp_file(plot=False, save_file=False, save_fig=False)

    # %% read in ASP06 dark current corrected lamp measurement data and relate pixel to wavelength
    dirpath = os.path.join(calib_path, base, f"Tint_{t_int}ms")
    dirpath_ulli2 = os.path.join(calib_path, base, f"Ulli_transfer_2_{t_int}ms")
    dirpath_ulli3 = os.path.join(calib_path, base, f"Ulli_transfer_3_{t_int}ms")
    lamp_measurement = [f for f in os.listdir(dirpath) if f.endswith(f"{direction}_{prop}_cor.dat")]
    ulli2_measurement = [f for f in os.listdir(dirpath_ulli2) if f.endswith(f"{direction}_{prop}_cor.dat")]
    ulli3_measurement = [f for f in os.listdir(dirpath_ulli3) if f.endswith(f"{direction}_{prop}_cor.dat")]
    filename = lamp_measurement[0]
    date_str, prop, direction = smart.get_info_from_filename(filename)
    lab_calib = reader.read_smart_cor(dirpath, filename)
    # set negative counts to 0
    lab_calib[lab_calib.values < 0] = 0

    # %% read in dark current measurements
    dirpath_dark = os.path.join(calib_path, base, f"Tint_dark_{t_int}ms")
    dirpath_ulli2_dark = os.path.join(calib_path, base, f"Ulli_transfer_2_dark_{t_int}ms")
    dirpath_ulli3_dark = os.path.join(calib_path, base, f"Ulli_transfer_3_dark_{t_int}ms")

    if prop == "VNIR":
        # there is only ever one VNIR dark current measurement file after smart_process_lab_calib_halo_ac3.py
        lamp_measurement_dark = [f for f in os.listdir(dirpath_dark) if f.endswith(f"{direction}_{prop}.dat")]
        ulli2_measurement_dark = [f for f in os.listdir(dirpath_ulli2_dark) if f.endswith(f"{direction}_{prop}.dat")]
        ulli3_measurement_dark = [f for f in os.listdir(dirpath_ulli3_dark) if f.endswith(f"{direction}_{prop}.dat")]
        lab_calib_dark = reader.read_smart_raw(dirpath_dark, lamp_measurement_dark[0])
        ulli2_dark = reader.read_smart_raw(dirpath_ulli2_dark, ulli2_measurement_dark[0])
        ulli3_dark = reader.read_smart_raw(dirpath_ulli3_dark, ulli3_measurement_dark[0])
        # drop t_int and shutter column
        lab_calib_dark = lab_calib_dark.drop(["t_int", "shutter"], axis=1)
        ulli2_dark = ulli2_dark.drop(["t_int", "shutter"], axis=1)
        ulli3_dark = ulli3_dark.drop(["t_int", "shutter"], axis=1)
    else:
        assert prop == "SWIR", f"'prop' should be either 'VNIR' or 'SWIR' but is {prop}"
        # for SWIR the dark current is measured within each measurement file, use the raw measurement files and extract
        # the dark current from them
        lamp_measurement_dark = [f for f in os.listdir(dirpath) if f.endswith(f"{direction}_{prop}.dat")]
        ulli2_measurement_dark = [f for f in os.listdir(dirpath_ulli2) if f.endswith(f"{direction}_{prop}.dat")]
        ulli3_measurement_dark = [f for f in os.listdir(dirpath_ulli3) if f.endswith(f"{direction}_{prop}.dat")]
        lab_calib_dark = pd.concat([reader.read_smart_raw(dirpath, f) for f in lamp_measurement_dark])
        ulli2_dark = pd.concat([reader.read_smart_raw(dirpath_ulli2, f) for f in ulli2_measurement_dark])
        ulli3_dark = pd.concat([reader.read_smart_raw(dirpath_ulli3, f) for f in ulli3_measurement_dark])
        # keep only rows where the shutter was closed (shutter = 0)
        lab_calib_dark = lab_calib_dark[lab_calib_dark.shutter == 0]
        ulli2_dark = ulli2_dark[ulli2_dark.shutter == 0]
        ulli3_dark = ulli3_dark[ulli3_dark.shutter == 0]


    # %% read in pixel to wavelength file
    spectrometer = lookup[f"{direction}_{prop}"]
    pixel_wl = reader.read_pixel_to_wavelength(pixel_path, spectrometer)
    pixel_wl["S0"] = lab_calib.mean().reset_index(drop=True)  # take mean over time of calib measurement
    pixel_wl["S0_dark"] = lab_calib_dark.mean().reset_index(drop=True)
    if normalize:
        pixel_wl["S0"] = pixel_wl["S0"] / t_int  # normalize counts by integration time
        pixel_wl["S0_dark"] = pixel_wl["S0_dark"] / t_int  # normalize counts by integration time
        ylabel2, norm = "Normalized Counts", f"_{t_int}ms_norm"
    else:
        ylabel2, norm = "Counts", f"_{t_int}ms"
    # interpolate lamp irradiance on pixel wavelength
    lamp_func = interp1d(lamp["Wavelength"], lamp["Irradiance"], fill_value="extrapolate")
    pixel_wl["F0"] = lamp_func(pixel_wl["wavelength"])
    pixel_wl["c_lab"] = pixel_wl["F0"] / pixel_wl["S0"]  # calculate lab calibration factor W/m^2/counts
    pixel_wl[pixel_wl.values < 0] = 0  # set values < 0 to 0

    # %% plot counts and irradiance of lamp lab calibration
    fig, ax = plt.subplots()
    ax.plot(pixel_wl["wavelength"], pixel_wl["F0"], color="#6699CC", label="Irradiance")
    ax.set_title(f"1000W Lamp Laboratory Calibration\n"
                 f"{date_str.replace('_', '-')} {spectrometer} Fdw {prop} {t_int}ms integration time")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Irradiance (W$\\,$m$^{-2}$)")
    ax.set_ylim(-0.01, 0.3)
    ax2 = ax.twinx()
    ax2.plot(pixel_wl["wavelength"], pixel_wl["S0"], color="#117733", label="Counts")
    ax2.plot(pixel_wl["wavelength"], pixel_wl["S0_dark"], color="#DDCC77", label="Dark current counts")
    ax2.set_ylabel(ylabel2)
    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=2)
    ax.grid()
    plt.tight_layout()
    figname = f"{plot_path}/{spectrometer}_Fdw_{prop}_lamp_lab_calib{norm}.png"
    plt.savefig(figname, dpi=100)
    log.info(f"Saved {figname}")
    # plt.show()
    plt.close()

    # %% plot calibration factor
    fig, ax = plt.subplots()
    ax.plot(pixel_wl["wavelength"], pixel_wl["c_lab"], color="#6699CC", label="Calibration")
    ax.set_title("Laboratory Calibration Factor\n"
                 f"{date_str.replace('_', '-')} {spectrometer} Fdw {prop} {t_int}ms integration time")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Calibration Factor")
    ax.set_ylim(0, 7) if normalize else ""
    ax.grid()
    figname = f"{plot_path}/{spectrometer}_Fdw_{prop}_lab_calib_factor{norm}.png"
    plt.savefig(figname, dpi=100)
    log.info(f"Saved {figname}")
    # plt.show()
    plt.close()

    # %% read in Ulli2 and Ulli3 transfer measurement from lab
    ulli2_file = ulli2_measurement[0]
    ulli2 = reader.read_smart_cor(dirpath_ulli2, ulli2_file)
    ulli2[ulli2.values < 0] = 0  # set negative counts to 0
    pixel_wl["S_ulli2"] = ulli2.mean().reset_index(drop=True)  # take mean over time of calib measurement
    pixel_wl["S_ulli2_dark"] = ulli2_dark.mean().reset_index(drop=True)
    if normalize:
        pixel_wl["S_ulli2"] = pixel_wl["S_ulli2"] / t_int
        pixel_wl["S_ulli2_dark"] = pixel_wl["S_ulli2_dark"] / t_int

    pixel_wl["F_ulli2"] = pixel_wl["S_ulli2"] * pixel_wl["c_lab"]  # calculate irradiance measured from Ulli2

    ulli3_file = ulli3_measurement[0]
    ulli3 = reader.read_smart_cor(dirpath_ulli3, ulli3_file)
    ulli3[ulli3.values < 0] = 0  # set negative counts to 0
    pixel_wl["S_ulli3"] = ulli3.mean().reset_index(drop=True)  # take mean over time of calib measurement
    pixel_wl["S_ulli3_dark"] = ulli3_dark.mean().reset_index(drop=True)
    if normalize:
        pixel_wl["S_ulli3"] = pixel_wl["S_ulli3"] / t_int
        pixel_wl["S_ulli3_dark"] = pixel_wl["S_ulli3_dark"] / t_int

    pixel_wl["F_ulli3"] = pixel_wl["S_ulli3"] * pixel_wl["c_lab"]  # calculate irradiance measured from Ulli3

    # %% plot Ulli transfer measurement from laboratory
    fig, ax = plt.subplots()
    ax.plot(pixel_wl["wavelength"], pixel_wl["F_ulli2"], color="#6699CC", label="Irradiance Ulli2", zorder=1)
    ax.plot(pixel_wl["wavelength"], pixel_wl["F_ulli3"], color="#117733", label="Irradiance Ulli3", zorder=1)
    ax.set_title(f"Ulli2 and Ulli3 Transfer Sphere Laboratory Calibration \n"
                 f"{spectrometer} Fdw {prop} {t_int}ms integration time")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Irradiance (W$\\,$m$^{-2}$)")
    ax.set_ylim(-0.01, 0.3)
    ax2 = ax.twinx()
    ax2.plot(pixel_wl["wavelength"], pixel_wl["S_ulli2"], color="#CC6677", label="Counts Ulli2")
    ax2.plot(pixel_wl["wavelength"], pixel_wl["S_ulli3"], color="#DDCC77", label="Counts Ulli3")
    ax2.plot(pixel_wl["wavelength"], pixel_wl["S_ulli2_dark"], color="#D55E00", label="Dark counts Ulli2", zorder=0)
    ax2.plot(pixel_wl["wavelength"], pixel_wl["S_ulli3_dark"], color="#332288", label="Dark counts Ulli3", zorder=0)
    ax2.set_ylabel(ylabel2)
    # move the second axes behind the first one
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.grid()
    ax.legend(lines + lines2, labels + labels2, loc="lower center", bbox_to_anchor=(0.5, 0),
              bbox_transform=fig.transFigure, ncol=3)
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    figname = f"{plot_path}/{spectrometer}_Fdw_{prop}_ullis_lab_calib{norm}.png"
    plt.savefig(figname, dpi=100)
    log.info(f"Saved {figname}")
    # plt.show()
    plt.close()

    # %% save lamp and ulli measurement from lab to file
    csvname = f"{calib_path}/{spectrometer}_Fdw_{prop}_lab_calib{norm}.dat"
    pixel_wl.to_csv(csvname, index=False)
    log.info(f"Saved {csvname}")
