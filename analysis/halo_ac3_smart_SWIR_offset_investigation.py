#!/usr/bin/env python
"""Investigate offset of SWIR spectrometer during |haloac3|

After the first final calibration for |haloac3| (September 2022) it was discovered that the SWIR spectrometer data consistently shows too low values compared to the VNIR spectrometer.
The VNIR spectrometer is closer to the libRadtran clearsky simulation and can thus be considered to be "true".
To investigate this offset RF14 is used as it was a flight with a lot of long straight legs.
One of which was heading far north and thus experiencing high solar zenith angles.


**Steps:**

- Checked all SWIR dark current |rarr| no change over flight (see ``.../quicklooks/SMART_SWIR_dark_current``)
- VNIR dark current comes from transfer calibration |rarr| looks good (see :numref:`dark-current`)
- Plot minutely averaged spectra |rarr| offset between spectrometers gets smaller with higher wavelengths (see ``.../case_studies/...RF14/spectra``)
- Check mean difference between VNIR and SWIR for different wavelengths (see :numref:`table`)
- Look at other flights and plot the spectra of those (RF17)
- look at difference between VNIR and SWIR for 985nm for all flights (see :numref:`diff-985-all`)

.. _dark-current:

.. figure:: figures/2022_04_07_09_08.Fdw_VNIR.dat_dark_current.png

    Dark current used for correction of VNIR measurements during RF14.


Dark current correction works as expected.
The dark current is also correctly written into the calibrated netCDF files.
The field calibration factor looks stable between 950 and 2100 nm for the SWIR spectrometer.
The VNIR field calibration factor shows a strong increase towards higher wavelengths.

.. figure:: figures/HALO-AC3_20220407_HALO_RF14_c_field_VNIR_SWIR.png

    Field calibration factor for the VNIR and the SWIR spectrometer for RF14.
    Wavelength cutoffs at 400, 950 and 2100 nm.

Looking at minutely averaged spectra from both spectrometers shows that the offset between the two is less for higher wavelengths.
Thus, 975nm is tried as a cutoff between the two.
The difference between the two 975nm measurements is shown below.
It can be seen that the VNIR spectrometer is usually between 0.02 and 0.05 :math:`W\,m^{-2}\,nm^{-1}` higher than the SWIR spectrometer.

.. figure:: figures/HALO-AC3_20220407_HALO_RF14_diff_VNIR-SWIR_975nm.png

    Difference between VNIR and SWIR spectrometer for 975nm during RF14.
    Exact wavelengths differ slightly, 974.8nm and 972.8nm (VNIR, SWIR).

Looking at different wavelengths and calculating the mean difference yields the following results fro RF14.

.. _table:

.. table:: Mean difference between selected wavelengths measured with the VNIR and the SWIR spectrometer at the closest corresponding wavelength fro RF14.

     ============= ================== ================== ==================
      Wavelength    VNIR wavelength    SWIR wavelength    Mean difference
     ============= ================== ================== ==================
      950           949.72             946.88             0.029
      955           954.80             953.39             0.026
      960           959.85             959.88             0.028
      965           964.85             966.36             0.020
      970           969.82             972.81             0.016
      975           974.75             972.81             0.021
      980           980.34             979.25             0.012
      985           985.18             985.68             **0.008**
      990           989.98             992.09             0.013
      995           994.73             992.09             0.010
      990           989.98             992.09             0.013
      995           994.73             992.09             0.010
     ============= ================== ================== ==================

The mean difference between 985nm is lowest.
At this wavelength also the VNIR and SWIR wavelengths are close to each other, only 0.5nm difference.
However, at 960nm the difference is only 0.08nm but the difference in measurement is 0.028 :math:`W\,m^{-2}\,nm^{-1}`.

The analysis is repeated for each flight and the difference between the 985nm pixel from the two spectrometers is shown in :numref:`diff-985-all`.
The single txt files for all wavelengths for one flight can be found in the respective case study folders.
One csv file with all flights has been saved in the ``case_studies`` folder (``mean_diff_VNIR-SWIR.csv``).

.. _diff-985-all:

.. figure:: figures/HALO-AC3_mean_diff_VNIR-SWIR_985nm.png

    Mean difference between the 985nm measurement from the VNIR (985.18nm) and the SWIR (985.68nm) spectrometer for each research flight.


This plot is produced for each wavelength between 950 and 995nm.
They can be found in ``.../case_studies/spectrometer_overlap``.
Looking through those plots it becomes apparent that the 990nm wavelength shows the lowest difference between the two spectrometers on average.

.. _diff-990-all:

.. figure:: figures/HALO-AC3_mean_diff_VNIR-SWIR_990nm.png

    Mean difference between the 990nm measurement from the VNIR (989.98nm) and the SWIR (992.09nm) spectrometer for each research flight.

Due to this result the wavelength at which the two spectra are merged is set to **990** nm.

*author*: Johannes Röttenbacher
"""

if __name__ == "__main__":
# %% import modules
    import pylim.helpers as h
    import pylim.halo_ac3 as meta
    from pylim import reader, smart
    import os
    import xarray as xr
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import logging

    log = logging.getLogger("pylim")
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)

# %% set up paths and get files
    campaign = "halo-ac3"
    flight_key = "RF14"
    flight = meta.flight_names[flight_key] if campaign == "halo-ac3" else flight_key
    transfer_calib_date = meta.transfer_calibs[flight_key]
    date = flight[9:17] if campaign == "halo-ac3" else flight[7:-1]
    smart_raw = h.get_path("raw", flight, campaign)
    smart_dir = h.get_path("calibrated", flight, campaign=campaign)
    plot_path = f"{h.get_path('plot', campaign=campaign)}/{flight}"
    h.make_dir(plot_path)

# %% correct dark current of SWIR raw files and plot dark current for each file
    files = [f for f in os.listdir(smart_raw) if "SWIR" in f]
    for file in files:
        smart_cor = smart.correct_smart_dark_current(flight, file, campaign=campaign, option=3, path=smart_raw,
                                                     date=transfer_calib_date, plot=True, save_fig=True)

    # plot dark current for on VNIR file -> the same dark current is used for each file
    file = "2022_04_07_09_08.Fdw_VNIR.dat"
    # file = "2022_04_11_09_08.Fdw_VNIR.dat"  # RF17
    smart_cor = smart.correct_smart_dark_current(flight, file, campaign=campaign, option=3, path=smart_raw,
                                                 date=transfer_calib_date, plot=True, save_fig=True)

# %% plot dark current corrected signal from .dat files
    fig, ax = plt.subplots()
    smart.plot_smart_data(campaign, flight, "2022_04_07_08_12.Fdw_VNIR_cor.dat", "all", ax=ax)
    smart.plot_smart_data(campaign, flight, "2022_04_07_08_12.Fdw_SWIR_cor.dat", "all", ax=ax)
    ax.set_title("Dark current corrected time averaged spectra for SWIR and VNIR")
    plt.show()
    plt.close()

# %% plot single SMART measurement from RF14
    fig, ax = plt.subplots(figsize=(10, 6))
    smart.plot_smart_data(campaign, flight, "2022_04_07_12_16.Fdw_VNIR.dat", wavelength="all", ax=ax)
    plt.show()

# %% read in calibrated files
    swir_file = [f for f in os.listdir(smart_dir) if "SWIR" in f and f.endswith("v1.0.nc")][0]
    vnir_file = [f for f in os.listdir(smart_dir) if "VNIR" in f and f.endswith("v1.0.nc")][0]
    swir_ds = xr.open_dataset(f"{smart_dir}/{swir_file}")
    vnir_ds = xr.open_dataset(f"{smart_dir}/{vnir_file}")

# %% plot dark current corrected signal from calibrated files
    fig, ax = plt.subplots()
    vnir_ds["counts"].mean(dim="time").plot(label="VNIR", ax=ax)
    swir_ds["counts"].mean(dim="time").plot(label="SWIR", ax=ax)
    ax.grid()
    ax.legend()
    ax.set_title("Dark current corrected time averaged spectra for SWIR and VNIR")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Dark current corrected normalized Counts")
    plt.tight_layout()
    plt.show()
    plt.close()

# %% plot field calibration factors
    h.set_cb_friendly_colors()
    plt.rc("font", size=14)
    fig, ax = plt.subplots(figsize=(10, 6))
    vnir_ds["c_field"].plot(label="VNIR", ax=ax, lw=3)
    swir_ds["c_field"].plot(label="SWIR", ax=ax, lw=3)
    ax.axvline(x=320, color="#888888", label="Wavelength cutoffs", lw=2)
    ax.axvline(x=990, color="#888888", lw=2)
    ax.axvline(x=2100, color="#888888", lw=2)
    ax.set_yscale("log")
    ax.grid()
    ax.legend()
    ax.set_title(f"Field calibration factor for SWIR and VNIR - {flight_key}")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Field Calibration Factor")
    plt.tight_layout()
    plt.savefig(f"{plot_path}/{flight}_c_field_VNIR_SWIR.png")
    plt.show()
    plt.close()

# %% average spectra over one minute
    vnir_1min = vnir_ds["Fdw"].resample(time="1min").mean(skipna=True)
    swir_1min = swir_ds["Fdw"].resample(time="1min").mean(skipna=True)

# %% plot minutely averaged calibrated spectra
    h.make_dir(f"{plot_path}/spectra")
    h.set_cb_friendly_colors()
    plt.rc("font", size=14)
    for ts in vnir_1min.time:
        ts_dt = pd.to_datetime(ts.values)
        fig, ax = plt.subplots(figsize=(10, 6))
        vnir_1min.sel(time=ts).plot(label="VNIR", ax=ax, lw=3)
        swir_1min.sel(time=ts).plot(label="SWIR", ax=ax, lw=3)
        ax.axvline(x=400, color="#888888", label="Wavelength cutoffs", lw=2)
        ax.axvline(x=975, color="#888888", lw=2)
        ax.axvline(x=2100, color="#888888", lw=2)
        ax.grid()
        ax.legend()
        ax.set_title(f"Average Calibrated SWIR and VNIR Spectrum - {ts_dt:%Y-%m-%d %H:%M} UTC")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Spectral Irradiance (W$\,$m$^{-2}\,$nm$^{-1}$)")
        plt.tight_layout()
        plt.savefig(f"{plot_path}/spectra/{flight}_{ts_dt:%H%M}_VNIR-SWIR-spectra.png")
        # plt.show()
        plt.close()

# %% calculate difference between 985nm from swir and vnir
    wavelength = 985
    diff_wl = vnir_ds["Fdw"].sel(wavelength=wavelength, method="nearest") - swir_ds["Fdw"].sel(wavelength=wavelength, method="nearest")
    log.info(f"Mean Difference {wavelength}: {diff_wl.mean(skipna=True):.3f}")

# %% plot difference between vnir and swir for specific wavelength
    h.set_cb_friendly_colors()
    plt.rc("font", size=14)
    fig, ax = plt.subplots(figsize=(10, 6))
    diff_wl.plot(ax=ax, lw=2, marker="o",)
    h.set_xticks_and_xlabels(ax, pd.to_timedelta((diff_wl.time[-1] - diff_wl.time[0]).values))
    ax.grid()
    ax.set_title(f"Difference between VNIR and SWIR for {wavelength} nm")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Spectral Irradiance (W$\,$m$^{-2}\,$nm$^{-1}$)")
    plt.tight_layout()
    plt.savefig(f"{plot_path}/{flight}_diff_VNIR-SWIR_{wavelength}nm.png")
    plt.show()
    plt.close()

# %% calculate difference between wavelengths from swir and vnir and write mean to a text file
    with open(f"{plot_path}/{flight_key}_mean_diff_VNIR-SWIR.txt", "w") as ofile:
        ofile.write("Flight, Wavelength, VNIR wavelength, SWIR wavelength, Mean difference\n")
        wavelengths = np.arange(950, 1000, 5)
        for wl in wavelengths:
            vnir_wl = vnir_ds["Fdw"].sel(wavelength=wl, method="nearest")
            swir_wl = swir_ds["Fdw"].sel(wavelength=wl, method="nearest")
            diff_wl = np.nanmean(vnir_wl - swir_wl)
            ofile.write(f"{flight_key}, {wl}, {vnir_wl.wavelength.values:.2f}, {swir_wl.wavelength.values:.2f}, {diff_wl:.3f}\n")

# %% read in all text files and make one file from them
    txt_paths = list()
    for k in meta.flight_names.keys():
        p_path = f"{h.get_path('plot', campaign=campaign)}/{meta.flight_names[k]}"
        txt_paths.append(f"{p_path}/{k}_mean_diff_VNIR-SWIR.txt")
    df = pd.concat([pd.read_csv(p) for p in txt_paths[3:-1]])
    df.reset_index(inplace=True, drop=True)
    df.columns = df.columns.str.lstrip()
    df.to_csv(f"{h.get_path('plot', campaign=campaign)}/mean_diff_VNIR-SWIR.csv", index=False)

# %% plot the difference for each flight and each wavelength
    h.set_cb_friendly_colors()
    # wavelengths = [985]  # single wavelength
    wavelengths = np.arange(950, 1000, 5)
    for wl in wavelengths:
        df_plot = df[df["Wavelength"] == wl]
        df_plot.plot(x="Flight", y="Mean difference", figsize=(10, 6), legend=False, lw=3)
        mean = df_plot["Mean difference"].mean()
        plt.axhline(mean, label=f"Mean ({mean:.3f})", lw=3, color="#CC6677")
        plt.axhline(y=0, color="#888888", lw=2)
        plt.ylim((-0.01, 0.04))
        plt.xticks(range(len(df_plot["Flight"])), df_plot["Flight"].str.replace("RF","").to_list())
        plt.grid()
        plt.legend()
        plt.xlabel("Flight Number")
        plt.ylabel("Difference (W$\,$m$^{-2}\,$nm$^{-1}$)")
        plt.title(f"Mean difference between VNIR and SWIR at {wl}nm")
        plt.tight_layout()
        plt.savefig(f"{h.get_path('plot', campaign=campaign)}/spectrometer_overlap/{campaign.swapcase()}_mean_diff_VNIR-SWIR_{wl}nm.png", dpi=100)
        plt.show()
        plt.close()
