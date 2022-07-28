#!/usr/bin/env python
"""HALO-(AC)³: Compare laboratory calibrations of SMART

For HALO-(AC)³ two laboratory calibrations were done. Here the two calibrations are compared to each other.

**Important:** It was decided to use the other spectrometer pair (J5, J6) because the dark current looks better and does not have as many high offsets on some pixels. Also see the note on the effective receiving area!
Due to technical difficulties the spectrometer pair J3, J4 was used during HALO-AC3. This calibration was therefore used for the transfer calibration. However, a different inlet was used (VN05). The first calib was done with inlet VN11.

**Note on effective receiving area:** Effective receiving area is in front of inlet not inside it, but was interpreted otherwise. Thus, the distance between the effective receiving area and the lamp is off by 44mm!

The second lab calibration after the campaign was done with the correct inlet and the correct channels: VN05 and J3, J4

Each lab calibration is processed with :py:mod:`smart_process_lab_calib_halo_ac3.py` to correct the files for the dark current.
After that the calibration factors are calculated with :py:mod:`analysis.smart_calib_lab_ASP06_halo_ac3.py` for ASP06.

- compare calibration factors
- compare Ulli measurements

Results HALO-(AC)³
^^^^^^^^^^^^^^^^^^

.. figure:: figures/HALO-AC3_ASP06_lab_calib_comparison.png

    Comparison of the lab calibration factors before and after HALO-(AC)³ for the two channels on ASP06.

Comparing the calibration factors of ASP06 between the before and after calibration shows that the after calibration (red) has a lower magnitude in the wavelength range from 250 to 350 nm for the VNIR channel.
This means less signal in the before calibration for those pixels.

**Ulli measurements**

Taking a look at ASP06's measurements of the Ulbricht sphere (Ulli2) shows only a minor difference in magnitude between the calibrations when comparing the dark current corrected and normalized counts.
Especially in the VNIR part between 180 and 350nm a lot of noise can be seen, which might already explain the difference in calibration factors for those wavelengths.

.. figure:: figures/HALO-AC3_ASP06_lab_calib_comparison_s-ulli2.png

    Dark current corrected, normalized count measurements of the Ulli transfer sphere (Ulli2) from ASP06.

.. figure:: figures/HALO-AC3_ASP06_lab_calib_comparison_f-ulli2.png

    Calculated irradiance of the Ulli transfer sphere (Ulli2) from ASP06 measurements.

Plotting the actual difference between the two shows a more detailed picture.

.. figure:: figures/HALO-AC3_ASP06_lab_calib_comparison_f-ulli2-diff.png

    Difference between measured irradiances from Ulli sphere (Ulli2) (before - after).

The after campaign calibration shows a higher irradiance with increasing wavelength in the VNIR channel (J4).

There were several problems with the before calibration on 15. November 2021:

* wrong inlet used (VN11 instead of VN05)
* the effective receiving area was wrongly interpreted leading to an offset of 44mm in the distance between the calibration lamp and the inlet (should be 50cm)
* the power supply for Ulli3 had a voltage limit in place instead of a current limit -> only relevant for the transfer calibration at Oberpfaffenhofen before the campaign

Thus, it is decided to use the after campaign calibration for all transfer calibrations.

*author*: Johannes Röttenbacher
"""
if __name__ == "__main__":
# %% import modules
    import pylim.helpers as h
    from pylim import reader
    import pylim.halo_ac3 as meta
    import pandas as pd
    import matplotlib.pyplot as plt

    cm = 1 / 2.54  # centimeters in inches

# %% set paths
    campaign = "halo-ac3"
    calib_path = h.get_path("calib", campaign=campaign)
    plot_path = f"{h.get_path('plot', campaign=campaign)}/lab_calib_comparison"
    first_date_asp06 = "2021_11_15"
    second_date_asp06 = "2022_05_02"
    t_int = "300ms"

# %% read in calib files
    asp06_first, asp06_second = dict(), dict()
    asp06_props = ["Fdw_SWIR", "Fdw_VNIR"]
    for prop in asp06_props:
        asp06_first[prop] = pd.read_csv(f"{calib_path}/{first_date_asp06}_{meta.smart_lookup[prop]}_{prop}_lab_calib_{t_int}_norm.dat")
        asp06_second[prop] = pd.read_csv(f"{calib_path}/{second_date_asp06}_{meta.smart_lookup[prop]}_{prop}_lab_calib_{t_int}_norm.dat")

# %% plot calibration factors for ASP06
    h.set_cb_friendly_colors()
    plt.rcParams['font.size'] = 8
    fig, axs = plt.subplots(2, 1, figsize=(18 * cm, 11 * cm))
    # first row: Fdw SWIR
    ax = axs[0]
    ax.plot(asp06_first["Fdw_SWIR"].wavelength, asp06_first["Fdw_SWIR"].c_lab, label=first_date_asp06)
    ax.plot(asp06_second["Fdw_SWIR"].wavelength, asp06_second["Fdw_SWIR"].c_lab, label=second_date_asp06)
    ax.set_yscale("log")
    ax.set_ylabel("Calibration Factor")
    ax.set_title(f"Fdw SWIR ({meta.smart_lookup['Fdw_SWIR']})")
    ax.legend()
    ax.grid()

    # second row: Fdw VNIR
    ax = axs[1]
    ax.plot(asp06_first["Fdw_VNIR"].wavelength, asp06_first["Fdw_VNIR"].c_lab, label=first_date_asp06)
    ax.plot(asp06_second["Fdw_VNIR"].wavelength, asp06_second["Fdw_VNIR"].c_lab, label=second_date_asp06)
    ax.set_yscale("log")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Calibration Factor")
    ax.set_title(f"Fdw VNIR ({meta.smart_lookup['Fdw_VNIR']})")
    ax.legend()
    ax.grid()

    fig.suptitle("HALO-(AC)³ Laboratory Calibrations ASP06 - Calibration Factor")
    plt.tight_layout()
    # plt.show()
    figname = f"{plot_path}/HALO-AC3_ASP06_lab_calib_comparison.png"
    plt.savefig(figname, dpi=200)
    print(f"Saved {figname}")
    plt.close()

# %% plot Ulli counts for ASP06
    h.set_cb_friendly_colors()
    plt.rcParams['font.size'] = 8
    fig, axs = plt.subplots(2, 1, figsize=(18 * cm, 11 * cm))
    # first row: Fdw SWIR
    ax = axs[0]
    ax.plot(asp06_first["Fdw_SWIR"].wavelength, asp06_first["Fdw_SWIR"].S_ulli2, label=first_date_asp06)
    ax.plot(asp06_second["Fdw_SWIR"].wavelength, asp06_second["Fdw_SWIR"].S_ulli2, label=second_date_asp06)
    # ax.set_yscale("log")
    ax.set_title(f"Fdw SWIR ({meta.smart_lookup['Fdw_SWIR']})")

    # second row: Fdw VNIR
    ax = axs[1]
    ax.plot(asp06_first["Fdw_VNIR"].wavelength, asp06_first["Fdw_VNIR"].S_ulli2, label=first_date_asp06)
    ax.plot(asp06_second["Fdw_VNIR"].wavelength, asp06_second["Fdw_VNIR"].S_ulli2, label=second_date_asp06)
    ax.set_yscale("log")
    ax.set_title(f"Fdw VNIR ({meta.smart_lookup['Fdw_VNIR']})")
    ax.set_xlabel("Wavelength (nm)")
    for ax in axs:
        ax.set_ylabel("Normalized Counts")
        ax.legend()
        ax.grid()

    fig.suptitle("HALO-(AC)³ Laboratory Calibrations ASP06 - Ulli (2) Normalized Counts")
    plt.tight_layout()
    # plt.show()
    figname = f"{plot_path}/HALO-AC3_ASP06_lab_calib_comparison_s-ulli2.png"
    plt.savefig(figname, dpi=200)
    print(f"Saved {figname}")
    plt.close()

# %% plot Ulli irradiance measurements for ASP06
    h.set_cb_friendly_colors()
    plt.rcParams['font.size'] = 8
    fig, axs = plt.subplots(2, 1, figsize=(18 * cm, 11 * cm))
    # first row: Fdw SWIR
    ax = axs[0]
    ax.plot(asp06_first["Fdw_SWIR"].wavelength, asp06_first["Fdw_SWIR"].F_ulli2, label=first_date_asp06)
    ax.plot(asp06_second["Fdw_SWIR"].wavelength, asp06_second["Fdw_SWIR"].F_ulli2, label=second_date_asp06)
    # ax.set_yscale("log")
    ax.set_title(f"Fdw SWIR ({meta.smart_lookup['Fdw_SWIR']})")

    # second row: Fdw VNIR
    ax = axs[1]
    ax.plot(asp06_first["Fdw_VNIR"].wavelength, asp06_first["Fdw_VNIR"].F_ulli2, label=first_date_asp06)
    ax.plot(asp06_second["Fdw_VNIR"].wavelength, asp06_second["Fdw_VNIR"].F_ulli2, label=second_date_asp06)
    # ax.set_yscale("log")
    ax.set_title(f"Fdw VNIR ({meta.smart_lookup['Fdw_VNIR']})")
    ax.set_xlabel("Wavelength (nm)")

    for ax in axs:
        ax.set_ylabel("Irradiance (W$\,$m$^{-2}\,$nm$^{-1}$)")
        ax.legend()
        ax.grid()

    fig.suptitle("HALO-(AC)³ Laboratory Calibrations ASP06 - Ulli (2) Irradiance")
    plt.tight_layout()
    # plt.show()
    figname = f"{plot_path}/HALO-AC3_ASP06_lab_calib_comparison_f-ulli2.png"
    plt.savefig(figname, dpi=200)
    print(f"Saved {figname}")
    plt.close()

# %% plot difference between Ulli irradiance for ASP06
    h.set_cb_friendly_colors()
    plt.rcParams['font.size'] = 8
    fig, axs = plt.subplots(2, 1, figsize=(18 * cm, 11 * cm))
    # first row: Fdw SWIR
    ax = axs[0]
    ax.plot(asp06_first["Fdw_SWIR"].wavelength, asp06_first["Fdw_SWIR"].F_ulli2 - asp06_second["Fdw_SWIR"].F_ulli2)
    ax.set_title(f"Fdw SWIR ({meta.smart_lookup['Fdw_SWIR']})")

    # second row: Fdw VNIR
    ax = axs[1]
    ax.plot(asp06_first["Fdw_VNIR"].wavelength, asp06_first["Fdw_VNIR"].F_ulli2 - asp06_second["Fdw_VNIR"].F_ulli2)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_title(f"Fdw VNIR ({meta.smart_lookup['Fdw_VNIR']})")

    for ax in axs:
        ax.axhline(y=0, c="k")
        ax.set_ylabel("Irradiance (W$\,$m$^{-2}\,$nm$^{-1}$)")
        ax.grid()

    fig.suptitle("HALO-(AC)³ Laboratory Calibrations ASP06 - Difference in Ulli (2) irradiance (before-after)")
    plt.tight_layout()
    # plt.show()
    figname = f"{plot_path}/HALO-AC3_ASP06_lab_calib_comparison_f-ulli2-diff.png"
    plt.savefig(figname, dpi=200)
    print(f"Saved {figname}")
    plt.close()
