#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 01-03-2023

Results of icecloud sensitivity simulations with libRadtran
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^




.. |plot-path| replace:: ./docs/figures/icecloud

"""

if __name__ == "__main__":
    # %% import modules
    import pylim.helpers as h
    import pylim.halo_ac3 as meta
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt

    cbc = h.get_cb_friendly_colors()

    # %% set paths
    campaign = "halo-ac3"
    key = "RF17"
    flight = meta.flight_names[key]
    date = flight[9:17]

    plot_path = f"{h.get_path('plot', campaign)}/icecloud_sensitivity_study"
    fig_path = "./docs/figures/icecloud"
    libradtran_path = h.get_path("libradtran_exp", flight, campaign)
    libradtran_file = f"HALO-AC3_HALO_libRadtran_simulation_icecloud_{date}_{key}.nc"
    smart_path = h.get_path("calibrated", flight, campaign)
    smart_file = f"HALO-AC3_HALO_SMART_spectral-irradiance-Fdw_{date}_{key}_v1.0.nc"

    # %% plotting meta
    h.set_cb_friendly_colors()
    plt.rc("font", size=12)
    cloud_top = 7500  # m
    cloud_base = 6500  # m

# %% read in libradtran and SMART file
    ds = xr.open_dataset(f"{libradtran_path}/{libradtran_file}")
    smart_ds = xr.open_dataset(f"{smart_path}/{smart_file}")

# %% plot all combinations of IWC and re_eff_ice
    ds_plot = ds
    nr_iwc = ds.iwc.shape[0]
    _, ax = plt.subplots(figsize=h.figsize_wide)
    for i in range(ds.re_ice.shape[0]):
        x = np.repeat(ds.re_ice[i].values, nr_iwc)
        ax.plot(x, ds.iwc, "o", ls="", markersize=10)
    ax.set_xlim(0, 70)
    ax.set_xticks(np.arange(0, 70, 10))
    ax.set_yscale("log")
    ax.grid()
    ax.set_title("All combinations of IWC and r$_{eff, ice}$")
    ax.set_xlabel(r"Ice Effective Radius ($\mu$m)")
    ax.set_ylabel(r"Ice Water Content (g$\,$m$^{-3}$)")
    plt.tight_layout()
    figname = f"{plot_path}/icecloud_sensitivity_study_iwc-re_ice_combinations.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% integrate spectral simulations
    ds["eglo_int"] = ds.eglo.integrate("wavelength")

# %% calculate difference in spectra above and below cloud
    ds["eglo_diff"] = ds.eglo.sel(altitude=cloud_top) - ds.eglo.sel(altitude=cloud_base)

# %% plot spectral difference between above and below
    ds_plot = ds["eglo_diff"]
    g = ds_plot.isel(time=0, drop=True).plot(x="wavelength", hue="iwc", col="re_ice", col_wrap=3, add_legend=True)
    for i, ax in enumerate(g.axes.flat):
        ax.grid()
        ax.set_title(r"r$_{eff, ice}$ = " + f"{ds.re_ice[i].values:.0f}")
        ax.get_legend_handles_labels()
        if i > 2:
            ax.set_xlabel("Wavelength (nm)")
    g.axes.flat[0].set_ylabel(r"Irradiance (W$\,$m$^{-2}$)")
    g.axes.flat[3].set_ylabel(r"Irradiance (W$\,$m$^{-2}$)")
    g.figlegend.set_title(r"IWC (g$\,$m$^{-3}$)")
    g.figlegend.set_bbox_to_anchor((0.99, 0.77))
    plt.suptitle("Difference between Spectral Solar Downward Irradiance Above and Below Cloud")
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    figname = f"{plot_path}/libradtran_icecloud_study_spectral_difference_above-below.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

