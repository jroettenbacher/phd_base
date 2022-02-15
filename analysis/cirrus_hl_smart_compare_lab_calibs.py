#!/usr/bin/env python
"""CIRRUS-HL: Compare laboratory calibrations of SMART

For CIRRUS-HL two laboratory calibrations were done. Here the two calibrations are compared to each other.

The first lab calibration did not have the final optical fiber and inlet setting as were used during CIRRUS-HL.
Thus, a difference between the after and before campaign calibration is to be expected.
In addition to that, the optical fibers were unplugged after the campaign for the transport of SMART to Leipzig.

Each lab calibration is processed with `smart_process_lab_calib.py` to correct the files for the dark current.
After that the calibration factors are calculated with :py:mod:`analysis.smart_calib_lab_ASP06.py` for ASP06 and :py:mod:`analysis.smart_calib_lab_ASP07.py` for ASP07.

- compare calibration factors

*author*: Johannes Röttenbacher
"""
if __name__ == "__main__":
# %% import modules
    import pylim.helpers as h
    from pylim import reader
    import pylim.cirrus_hl as ch
    import pandas as pd
    import matplotlib.pyplot as plt

    cm = 1 / 2.54  # centimeters in inches

# %% set paths
    calib_path = h.get_path("calib")
    plot_path = f"{h.get_path('plot')}/lab_calib_comparison"
    first_date_asp06 = "2021_03_29"
    first_date_asp07 = "2021_03_18"
    second_date_asp0607 = "2021_08_09"

# %% read in calib files
    asp06_first, asp06_second, asp07_first, asp07_second = dict(), dict(), dict(), dict()
    asp06_props = ["Fdw_SWIR", "Fdw_VNIR", "Fup_SWIR", "Fup_VNIR"]
    asp07_props = ["Iup_SWIR", "Iup_VNIR"]
    for prop in asp06_props:
        asp06_first[prop] = pd.read_csv(f"{calib_path}/{first_date_asp06}_{ch.lookup[prop]}_{prop}_lab_calib.dat")
        asp06_second[prop] = pd.read_csv(f"{calib_path}/{second_date_asp0607}_{ch.lookup[prop]}_{prop}_lab_calib.dat")

    for prop in asp07_props:
        asp07_first[prop] = pd.read_csv(f"{calib_path}/{first_date_asp07}_{ch.lookup[prop]}_{prop}_lab_calib.dat")
        asp07_second[prop] = pd.read_csv(f"{calib_path}/{second_date_asp0607}_{ch.lookup[prop]}_{prop}_lab_calib.dat")

# %% plot calibration factors for ASP06
    h.set_cb_friendly_colors()
    plt.rcParams['font.size'] = 8
    fig, axs = plt.subplots(2, 2, figsize=(18 * cm, 11 * cm), sharex='col')
    # first row, first column: Fdw SWIR
    ax = axs[0, 0]
    ax.plot(asp06_first["Fdw_SWIR"].wavelength, asp06_first["Fdw_SWIR"].c_lab, label=first_date_asp06)
    ax.plot(asp06_second["Fdw_SWIR"].wavelength, asp06_second["Fdw_SWIR"].c_lab, label=second_date_asp0607)
    ax.set_yscale("log")
    ax.set_ylabel("Calibration Factor")
    ax.set_title(f"Fdw SWIR ({ch.lookup['Fdw_SWIR']})")
    ax.legend()
    ax.grid()

    # second row, first column: Fup SWIR
    ax = axs[1, 0]
    ax.plot(asp06_first["Fup_SWIR"].wavelength, asp06_first["Fup_SWIR"].c_lab, label=first_date_asp06)
    ax.plot(asp06_second["Fup_SWIR"].wavelength, asp06_second["Fup_SWIR"].c_lab, label=second_date_asp0607)
    ax.set_yscale("log")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Calibration Factor")
    ax.set_title(f"Fup SWIR ({ch.lookup['Fup_SWIR']})")
    ax.legend()
    ax.grid()

    # first row, second column: Fdw_VNIR
    ax = axs[0, 1]
    ax.plot(asp06_first["Fdw_VNIR"].wavelength, asp06_first["Fdw_VNIR"].c_lab, label=first_date_asp06)
    ax.plot(asp06_second["Fdw_VNIR"].wavelength, asp06_second["Fdw_VNIR"].c_lab, label=second_date_asp0607)
    ax.set_yscale("log")
    ax.set_ylabel("Calibration Factor")
    ax.set_title(f"Fdw VNIR ({ch.lookup['Fdw_VNIR']})")
    ax.legend()
    ax.grid()

    # second row, second column: Fup VNIR
    ax = axs[1, 1]
    ax.plot(asp06_first["Fup_VNIR"].wavelength, asp06_first["Fup_VNIR"].c_lab, label=first_date_asp06)
    ax.plot(asp06_second["Fup_VNIR"].wavelength, asp06_second["Fup_VNIR"].c_lab, label=second_date_asp0607)
    ax.set_yscale("log")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Calibration Factor")
    ax.set_title(f"Fup VNIR ({ch.lookup['Fup_VNIR']})")
    ax.legend()
    ax.grid()

    fig.suptitle("CIRRUS-HL Laboratory Calibrations ASP06")
    plt.tight_layout()
    # plt.show()
    figname = f"{plot_path}/ASP06_lab_calib_comparison.png"
    plt.savefig(figname, dpi=200)
    print(f"Saved {figname}")
    plt.close()

# %% plot calibration factors for ASP07
    h.set_cb_friendly_colors()
    plt.rcParams['font.size'] = 8
    fig, axs = plt.subplots(2, 1, figsize=(18 * cm, 11 * cm))
    # first row: Iup SWIR
    ax = axs[0]
    ax.plot(asp07_first["Iup_SWIR"].wavelength, asp07_first["Iup_SWIR"].c_lab, label=first_date_asp07)
    ax.plot(asp07_second["Iup_SWIR"].wavelength, asp07_second["Iup_SWIR"].c_lab, label=second_date_asp0607)
    ax.set_yscale("log")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Calibration Factor")
    ax.set_title(f"Iup SWIR ({ch.lookup['Iup_SWIR']})")
    ax.legend()
    ax.grid()

    # second row: Iup_VNIR
    ax = axs[1]
    ax.plot(asp07_first["Iup_VNIR"].wavelength, asp07_first["Iup_VNIR"].c_lab, label=first_date_asp07)
    ax.plot(asp07_second["Iup_VNIR"].wavelength, asp07_second["Iup_VNIR"].c_lab, label=second_date_asp0607)
    ax.set_yscale("log")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Calibration Factor")
    ax.set_title(f"Iup VNIR ({ch.lookup['Iup_VNIR']})")
    ax.legend()
    ax.grid()

    fig.suptitle("CIRRUS-HL Laboratory Calibrations ASP07")
    plt.tight_layout()
    # plt.show()
    figname = f"{plot_path}/ASP07_lab_calib_comparison.png"
    plt.savefig(figname, dpi=200)
    print(f"Saved {figname}")
    plt.close()
