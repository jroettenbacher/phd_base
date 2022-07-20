#!/usr/bin/env python
"""CIRRUS-HL: Compare laboratory calibrations of SMART

For CIRRUS-HL two laboratory calibrations were done. Here the two calibrations are compared to each other.

The first lab calibration did not have the final optical fiber and inlet setting as were used during CIRRUS-HL.
Thus, a difference between the after and before campaign calibration is to be expected.
In addition to that, the optical fibers were unplugged after the campaign for the transport of SMART to Leipzig.

Each lab calibration is processed with `smart_process_lab_calib.py` to correct the files for the dark current.
After that the calibration factors are calculated with :py:mod:`analysis.smart_calib_lab_ASP06.py` for ASP06 and :py:mod:`analysis.smart_calib_lab_ASP07.py` for ASP07.

- compare calibration factors
- compare Ulli measurements

Results
^^^^^^^^

.. figure:: figures/ASP06_lab_calib_comparison.png

    Comparison of the lab calibration factors before and after CIRRUS-HL for the four channels on ASP06.

Comparing the calibration factors of ASP06 between the before and after calibration shows that the after calibration (red) has a higher magnitude.
This is due to fewer counts being measured during the after calibration.
A possible cause of this is that the cable used in the after calibration was longer and also showed signs of damage leading to a loss of photons on the way from the inlet to the spectrometer.
To account for that a higher calibration factor is needed to calculate the given irradiance which did not change between the calibrations.
The same result can be observed for the calibrations for ASP07.

.. figure:: figures/ASP07_lab_calib_comparison.png

    Comparison of the lab calibration factors before and after CIRRUS-HL for the two channels on ASP07.

**Ulli measurements**

Taking a look at ASP06's measurements of the Ulbricht sphere (Ulli) shows the same difference in magnitude between the calibrations when comparing the dark current corrected counts.

.. figure:: figures/ASP06_lab_calib_comparison_s-ulli.png

    Dark current corrected count measurements of the Ulli transfer sphere from ASP06.

It can clearly be seen that during the after calibration less photons reached the spectrometers.
However, this should still result in the same irradiance as long as the Ulli sphere is run with the same power source and the same settings.
Just like the output of the 1000W calibration lamp should not change between the calibration, so should the output of the Ulli sphere not change.
Nonetheless, when looking at the calculated irradiance using the corresponding laboratory calibration factors clear differences can be observed for the downward measurement.

.. figure:: figures/ASP06_lab_calib_comparison_f-ulli.png

    Calculated irradiance of the Ulli transfer sphere from ASP06.

Plotting the actual difference between the two shows a more detailed picture.

.. figure:: figures/ASP06_lab_calib_comparison_f-ulli-diff.png

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
        asp06_first[prop] = pd.read_csv(f"{calib_path}/{first_date_asp06}_{ch.smart_lookup[prop]}_{prop}_lab_calib.dat")
        asp06_second[prop] = pd.read_csv(f"{calib_path}/{second_date_asp0607}_{ch.smart_lookup[prop]}_{prop}_lab_calib.dat")

    for prop in asp07_props:
        asp07_first[prop] = pd.read_csv(f"{calib_path}/{first_date_asp07}_{ch.smart_lookup[prop]}_{prop}_lab_calib.dat")
        asp07_second[prop] = pd.read_csv(f"{calib_path}/{second_date_asp0607}_{ch.smart_lookup[prop]}_{prop}_lab_calib.dat")

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
    ax.set_title(f"Fdw SWIR ({ch.smart_lookup['Fdw_SWIR']})")
    ax.legend()
    ax.grid()

    # second row, first column: Fup SWIR
    ax = axs[1, 0]
    ax.plot(asp06_first["Fup_SWIR"].wavelength, asp06_first["Fup_SWIR"].c_lab, label=first_date_asp06)
    ax.plot(asp06_second["Fup_SWIR"].wavelength, asp06_second["Fup_SWIR"].c_lab, label=second_date_asp0607)
    ax.set_yscale("log")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Calibration Factor")
    ax.set_title(f"Fup SWIR ({ch.smart_lookup['Fup_SWIR']})")
    ax.legend()
    ax.grid()

    # first row, second column: Fdw_VNIR
    ax = axs[0, 1]
    ax.plot(asp06_first["Fdw_VNIR"].wavelength, asp06_first["Fdw_VNIR"].c_lab, label=first_date_asp06)
    ax.plot(asp06_second["Fdw_VNIR"].wavelength, asp06_second["Fdw_VNIR"].c_lab, label=second_date_asp0607)
    ax.set_yscale("log")
    ax.set_ylabel("Calibration Factor")
    ax.set_title(f"Fdw VNIR ({ch.smart_lookup['Fdw_VNIR']})")
    ax.legend()
    ax.grid()

    # second row, second column: Fup VNIR
    ax = axs[1, 1]
    ax.plot(asp06_first["Fup_VNIR"].wavelength, asp06_first["Fup_VNIR"].c_lab, label=first_date_asp06)
    ax.plot(asp06_second["Fup_VNIR"].wavelength, asp06_second["Fup_VNIR"].c_lab, label=second_date_asp0607)
    ax.set_yscale("log")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Calibration Factor")
    ax.set_title(f"Fup VNIR ({ch.smart_lookup['Fup_VNIR']})")
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
    ax.set_title(f"Iup SWIR ({ch.smart_lookup['Iup_SWIR']})")
    ax.legend()
    ax.grid()

    # second row: Iup_VNIR
    ax = axs[1]
    ax.plot(asp07_first["Iup_VNIR"].wavelength, asp07_first["Iup_VNIR"].c_lab, label=first_date_asp07)
    ax.plot(asp07_second["Iup_VNIR"].wavelength, asp07_second["Iup_VNIR"].c_lab, label=second_date_asp0607)
    ax.set_yscale("log")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Calibration Factor")
    ax.set_title(f"Iup VNIR ({ch.smart_lookup['Iup_VNIR']})")
    ax.legend()
    ax.grid()

    fig.suptitle("CIRRUS-HL Laboratory Calibrations ASP07")
    plt.tight_layout()
    # plt.show()
    figname = f"{plot_path}/ASP07_lab_calib_comparison.png"
    plt.savefig(figname, dpi=200)
    print(f"Saved {figname}")
    plt.close()

# %% plot Ulli counts for ASP06
    h.set_cb_friendly_colors()
    plt.rcParams['font.size'] = 8
    fig, axs = plt.subplots(2, 2, figsize=(18 * cm, 11 * cm), sharex='col')
    # first row, first column: Fdw SWIR
    ax = axs[0, 0]
    ax.plot(asp06_first["Fdw_SWIR"].wavelength, asp06_first["Fdw_SWIR"].S_ulli, label=first_date_asp06)
    ax.plot(asp06_second["Fdw_SWIR"].wavelength, asp06_second["Fdw_SWIR"].S_ulli, label=second_date_asp0607)
    # ax.set_yscale("log")
    ax.set_title(f"Fdw SWIR ({ch.smart_lookup['Fdw_SWIR']})")

    # second row, first column: Fup SWIR
    ax = axs[1, 0]
    ax.plot(asp06_first["Fup_SWIR"].wavelength, asp06_first["Fup_SWIR"].S_ulli, label=first_date_asp06)
    ax.plot(asp06_second["Fup_SWIR"].wavelength, asp06_second["Fup_SWIR"].S_ulli, label=second_date_asp0607)
    # ax.set_yscale("log")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_title(f"Fup SWIR ({ch.smart_lookup['Fup_SWIR']})")

    # first row, second column: Fdw_VNIR
    ax = axs[0, 1]
    ax.plot(asp06_first["Fdw_VNIR"].wavelength, asp06_first["Fdw_VNIR"].S_ulli, label=first_date_asp06)
    ax.plot(asp06_second["Fdw_VNIR"].wavelength, asp06_second["Fdw_VNIR"].S_ulli, label=second_date_asp0607)
    # ax.set_yscale("log")
    ax.set_title(f"Fdw VNIR ({ch.smart_lookup['Fdw_VNIR']})")

    # second row, second column: Fup VNIR
    ax = axs[1, 1]
    ax.plot(asp06_first["Fup_VNIR"].wavelength, asp06_first["Fup_VNIR"].S_ulli, label=first_date_asp06)
    ax.plot(asp06_second["Fup_VNIR"].wavelength, asp06_second["Fup_VNIR"].S_ulli, label=second_date_asp0607)
    # ax.set_yscale("log")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_title(f"Fup VNIR ({ch.smart_lookup['Fup_VNIR']})")

    for ax in axs:
        for a in ax:
            a.set_ylabel("Counts")
            a.legend()
            a.grid()

    fig.suptitle("CIRRUS-HL Laboratory Calibrations ASP06 - Ulli counts")
    plt.tight_layout()
    # plt.show()
    figname = f"{plot_path}/ASP06_lab_calib_comparison_s-ulli.png"
    plt.savefig(figname, dpi=200)
    print(f"Saved {figname}")
    plt.close()

# %% plot Ulli irradiance measurements for ASP06
    h.set_cb_friendly_colors()
    plt.rcParams['font.size'] = 8
    fig, axs = plt.subplots(2, 2, figsize=(18 * cm, 11 * cm), sharex='col')
    # first row, first column: Fdw SWIR
    ax = axs[0, 0]
    ax.plot(asp06_first["Fdw_SWIR"].wavelength, asp06_first["Fdw_SWIR"].F_ulli, label=first_date_asp06)
    ax.plot(asp06_second["Fdw_SWIR"].wavelength, asp06_second["Fdw_SWIR"].F_ulli, label=second_date_asp0607)
    # ax.set_yscale("log")
    ax.set_title(f"Fdw SWIR ({ch.smart_lookup['Fdw_SWIR']})")

    # second row, first column: Fup SWIR
    ax = axs[1, 0]
    ax.plot(asp06_first["Fup_SWIR"].wavelength, asp06_first["Fup_SWIR"].F_ulli, label=first_date_asp06)
    ax.plot(asp06_second["Fup_SWIR"].wavelength, asp06_second["Fup_SWIR"].F_ulli, label=second_date_asp0607)
    # ax.set_yscale("log")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_title(f"Fup SWIR ({ch.smart_lookup['Fup_SWIR']})")

    # first row, second column: Fdw_VNIR
    ax = axs[0, 1]
    ax.plot(asp06_first["Fdw_VNIR"].wavelength, asp06_first["Fdw_VNIR"].F_ulli, label=first_date_asp06)
    ax.plot(asp06_second["Fdw_VNIR"].wavelength, asp06_second["Fdw_VNIR"].F_ulli, label=second_date_asp0607)
    # ax.set_yscale("log")
    ax.set_title(f"Fdw VNIR ({ch.smart_lookup['Fdw_VNIR']})")

    # second row, second column: Fup VNIR
    ax = axs[1, 1]
    ax.plot(asp06_first["Fup_VNIR"].wavelength, asp06_first["Fup_VNIR"].F_ulli, label=first_date_asp06)
    ax.plot(asp06_second["Fup_VNIR"].wavelength, asp06_second["Fup_VNIR"].F_ulli, label=second_date_asp0607)
    # ax.set_yscale("log")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_title(f"Fup VNIR ({ch.smart_lookup['Fup_VNIR']})")

    for ax in axs:
        for a in ax:
            a.set_ylabel("Irradiance (W$\,$m$^{-2}\,$nm$^{-1}$)")
            a.legend()
            a.grid()

    fig.suptitle("CIRRUS-HL Laboratory Calibrations ASP06 - Ulli Irradiance")
    plt.tight_layout()
    # plt.show()
    figname = f"{plot_path}/ASP06_lab_calib_comparison_f-ulli.png"
    plt.savefig(figname, dpi=200)
    print(f"Saved {figname}")
    plt.close()

# %% plot difference between Ulli irradiance for ASP06
    h.set_cb_friendly_colors()
    plt.rcParams['font.size'] = 8
    fig, axs = plt.subplots(2, 2, figsize=(18 * cm, 11 * cm), sharex='col')
    # first row, first column: Fdw SWIR
    ax = axs[0, 0]
    ax.plot(asp06_first["Fdw_SWIR"].wavelength, asp06_first["Fdw_SWIR"].F_ulli - asp06_second["Fdw_SWIR"].F_ulli)
    ax.set_title(f"Fdw SWIR ({ch.smart_lookup['Fdw_SWIR']})")

    # second row, first column: Fup SWIR
    ax = axs[1, 0]
    ax.plot(asp06_first["Fup_SWIR"].wavelength, asp06_first["Fup_SWIR"].F_ulli - asp06_second["Fup_SWIR"].F_ulli)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_title(f"Fup SWIR ({ch.smart_lookup['Fup_SWIR']})")

    # first row, second column: Fdw_VNIR
    ax = axs[0, 1]
    ax.plot(asp06_first["Fdw_VNIR"].wavelength, asp06_first["Fdw_VNIR"].F_ulli - asp06_second["Fdw_VNIR"].F_ulli)
    ax.set_title(f"Fdw VNIR ({ch.smart_lookup['Fdw_VNIR']})")

    # second row, second column: Fup VNIR
    ax = axs[1, 1]
    ax.plot(asp06_first["Fup_VNIR"].wavelength, asp06_first["Fup_VNIR"].F_ulli - asp06_second["Fup_VNIR"].F_ulli)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_title(f"Fup VNIR ({ch.smart_lookup['Fup_VNIR']})")

    for ax in axs:
        for a in ax:
            a.axhline(y=0, c="k")
            a.set_ylabel("Irradiance (W$\,$m$^{-2}\,$nm$^{-1}$)")
            a.grid()

    fig.suptitle("CIRRUS-HL Laboratory Calibrations ASP06 - Difference in Ulli irradiance (before-after)")
    plt.tight_layout()
    # plt.show()
    figname = f"{plot_path}/ASP06_lab_calib_comparison_f-ulli-diff.png"
    plt.savefig(figname, dpi=200)
    print(f"Saved {figname}")
    plt.close()
