#!/usr/bin/env python
"""Script to do some analysis on SMART data
author: Johannes Röttenbacher
"""
if __name__ == "__main__":
    # %% import libraries
    from pylim import reader
    from pylim import smart, helpers as h
    from pylim.cirrus_hl import lookup
    import os
    import xarray as xr
    import matplotlib.pyplot as plt
    import logging
    log = logging.getLogger("pylim")
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)

    # %% list files
    flight = "Flight_20210625a"
    calibrated_path = h.get_path("calibrated", flight)
    pixel_wl_path = h.get_path("pixel_wl")
    plot_path = f"{h.get_path('plot')}/{flight}"
    h.make_dir(plot_path)
    all_files = os.listdir(calibrated_path)
    fdw_files = [f for f in all_files if "Fdw" in f]
    fup_files = [f for f in all_files if "Fup" in f]

    # %% get pixel to wavelength file
    channel = "VNIR"
    direction = "Fdw"
    pixel_wl = reader.read_pixel_to_wavelength(pixel_wl_path, lookup[f"{direction}_{channel}"])
    # %% read in files
    fdw_file = fdw_files[0] if channel in fdw_files[0] else fdw_files[1]
    fup_file = fup_files[0] if channel in fup_files[0] else fup_files[1]
    fdw = reader.read_smart_cor(calibrated_path, fdw_file)
    fup = reader.read_smart_cor(calibrated_path, fup_file)

    # %% remove values < 0
    fdw_clean = fdw[fdw > 0]
    fup_clean = fup[fup > 0]

    # %% calculate albedo
    albedo = fup_clean / fdw_clean

    # %% select time range and wavelength
    begin = "2021-06-25 11:15"
    end = "2021-06-25 16:15"
    wavelength = 1200
    albedo_sel = albedo[begin:end]
    fdw_sel = fdw_clean[begin:end]
    fup_sel = fup_clean[begin:end]
    pixel_nr, wl = smart.find_pixel(pixel_wl, wavelength)
    # %% plot spectral albedo time series
    fig, ax = plt.subplots(figsize=(6, 4))
    albedo_sel.plot(y=pixel_nr, ax=ax, label="Albedo", c="g")
    ax.set_ylabel("Spectral Albedo")
    ax.set_xlabel("Time (UTC)")
    ax2 = ax.twinx()
    fdw_sel.plot(y=pixel_nr, ax=ax2, label="Fdw")
    fup_sel.plot(y=pixel_nr, ax=ax2, label="Fup")
    ax2.set_ylabel("Irradiance (W$\\,$m$^{-2}\\,$nm$^{-1}$)")
    plt.title(f"Time Series of Spectral Albedo (Fup / Fdw) at {wl} nm")
    ax.grid()
    plt.tight_layout()
    plt.show()
    plt.close()

    # %% plot average albedo
    albedo_avg = albedo_sel.median(axis=1)
    fdw_avg = fdw_sel.median(axis=1)
    fup_avg = fup_sel.median(axis=1)
    fig, ax = plt.subplots(figsize=(6, 4))
    albedo_avg.plot(ax=ax, title=f"{channel} Median Albedo (Fup / Fdw)", label="Albedo", c="g")
    ax.set_ylabel("Albedo")
    ax.set_xlabel("Time (UTC)")
    ax.legend()
    ax2 = ax.twinx()
    fup_avg.plot(ax=ax2, label="Fup")
    fdw_avg.plot(ax=ax2, label="Fdw")
    ax2.set_ylabel("Irradiance (W$\\,$m$^{-2}\\,$nm$^{-1}$)")
    ax2.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()
    plt.close()

    # %% plot spectral albedo
    timestep = "2021-06-25 14:38:27.35"
    wavelengths = pixel_wl["wavelength"]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(wavelengths, albedo.loc[timestep], label="Albedo", c="k")
    ax.set_ylabel("Albedo")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylim((0, 1))
    ax.legend()
    ax2 = ax.twinx()
    ax2.plot(wavelengths, fdw_clean.loc[timestep], label="Fdw")
    ax2.plot(wavelengths, fup_clean.loc[timestep], label="Fup")
    ax2.set_ylabel("Irradiance (W$\\,$m$^{-2}\\,$nm$^{-1}$)")
    ax2.legend()
    ax.grid()
    plt.title(f"{channel} Spectrum and Albedo for {timestep}\nDark Current Corrected and Calibrated")
    plt.tight_layout()
    timestep_name = timestep.replace(' ', '_').replace(':', '-')
    figname = f"{plot_path}/{timestep_name}_{channel}_spectrum_albedo.png"
    plt.savefig(figname, dpi=100)
    log.info(f"Saved {figname}")
    plt.show()
    plt.close()

    # %% read in BAHAMAS
    bahamas_dir = h.get_path("bahamas", flight)
    bahamas_file = [f for f in os.listdir(bahamas_dir) if f.endswith(".nc")]

    bahamas = xr.open_dataset(f"{bahamas_dir}/{bahamas_file}")
    bahamas["H"].plot()
    plt.show()
    plt.close()
