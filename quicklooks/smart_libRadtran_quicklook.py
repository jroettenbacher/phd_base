#!/usr/bin/env python
"""Plot time series of calibrated SMART data and libRadtran clearsky simulation

*author*: Johannes Röttenbacher
"""

# %% modules
import pylim.helpers as h
import pylim.cirrus_hl as meta
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.WARNING)

# %% user input
campaign = "cirrus-hl"
flight_keys = list(meta.flight_names.keys())[3:] if campaign == "halo-ac3" else list(meta.flight_numbers.keys())[1:]
for flight_key in tqdm(flight_keys):
    flight = meta.flight_names[flight_key] if campaign == "halo-ac3" else flight_key
    date = flight[9:17] if campaign == "halo-ac3" else flight[7:]
    savefig = True

    # %% set paths and filenames
    ql_dir = f"{h.get_path('quicklooks', flight, campaign)}/SMART_wavelengths"
    h.make_dir(ql_dir)
    smart_dir = h.get_path("calibrated", flight, campaign)
    libradtran_dir = h.get_path("libradtran", flight, campaign)
    if campaign == "halo-ac3":
        smart_file = f"HALO-AC3_HALO_SMART_spectral-irradiance-Fdw_{date}_{flight_key}_v1.0.nc"
        libradtran_file = f"HALO-AC3_HALO_libRadtran_clearsky_simulation_smart_spectral_{date}_{flight_key}.nc"
    else:
        flight_nr = meta.flight_numbers[flight_key]
        smart_file = f"CIRRUS-HL_{flight_nr}_{date}_HALO_SMART_spectral-irradiance-Fdw_v1.0.nc"
        libradtran_file = f"CIRRUS-HL_HALO_libRadtran_clearsky_simulation_smart_spectral_{date[:-1]}_{flight_key}.nc"

    # %% read in files
    smart_ds = xr.open_dataset(f"{smart_dir}/{smart_file}")
    sim_ds = xr.open_dataset(f"{libradtran_dir}/{libradtran_file}")

    # %% get time range
    time_range = pd.to_timedelta((smart_ds["time"][-1] - smart_ds["time"][0]).values)

    # %% integrate measurement and simulation over wavelength
    smart_ds["Fdw_int"] = smart_ds["Fdw"].integrate("wavelength")
    smart_ds["Fdw_cor_int"] = smart_ds["Fdw_cor"].integrate("wavelength")
    sim_ds["Fdw_int"] = sim_ds["fdw"].integrate("wavelength")

    # %% filter by roll and pitch angle
    roll_threshold = 2
    pitch_threshold = 3
    condition = (np.absolute(smart_ds["roll"]) < roll_threshold) & (np.absolute(smart_ds["pitch"] < pitch_threshold))
    # keep values where condition is true and replace with nan otherwise
    smart_ds_filtered = smart_ds.where(condition)

    # %% plot time series of integrated measurement and simulation
    data = smart_ds_filtered  # select which data to plot
    h.set_cb_friendly_colors()
    plt.rc('font', size=14)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot("time", "Fdw_int", data=data, label="SMART Measurement")
    ax.plot("time", "Fdw_cor_int", data=data, label="SMART Corrected Measurement")
    ax.plot(sim_ds["time"], sim_ds["Fdw_int"], label="libradtran Simulation", lw=3)
    ax.grid()
    ax.legend()
    h.set_xticks_and_xlabels(ax, time_range)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Irradiance (W$\,$m$^{-2}$)")
    ax.set_title(f"Integrated Downward Irradiance - {flight}")
    plt.tight_layout()

    figname = f"{ql_dir}/{campaign.swapcase()}_HALO_SMART-libRadtran_integrated-Fdw_{date}_{flight_key}.png"
    if savefig:
        plt.savefig(figname, dpi=100)
        log.info(f"Saved {figname}")

    plt.show()
    plt.close()

    # %% plot time series of selected wavelengths
    wavelength = 2000  # select which wavelength to plot
    for wavelength in smart_ds_filtered.wavelength:
        data = smart_ds_filtered.sel(wavelength=wavelength)  # select which data to plot
        data_sim = sim_ds.sel(wavelength=wavelength)
        h.set_cb_friendly_colors()
        plt.rc('font', size=14)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot("time", "Fdw", data=data, label="SMART Measurement")
        ax.plot("time", "Fdw_cor", data=data, label="SMART Corrected Measurement")
        ax.plot("time", "fdw", data=data_sim, label="libradtran Simulation", lw=3)
        ax.grid()
        ax.legend()
        h.set_xticks_and_xlabels(ax, time_range)
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Spectral Irradiance (W$\,$m$^{-2}\,$nm$^{-1}$)")
        ax.set_title(f"Downward Irradiance at {wavelength} nm - {flight}")
        plt.tight_layout()

        figname = f"{ql_dir}/{campaign.swapcase()}_HALO_SMART-libRadtran_Fdw-{wavelength}nm_{date}_{flight_key}.png"
        if savefig:
            plt.savefig(figname, dpi=100)
            log.info(f"Saved {figname}")

        # plt.show()
        plt.close()
