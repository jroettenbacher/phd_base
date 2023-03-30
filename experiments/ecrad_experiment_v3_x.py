#!/usr/bin/env python
"""Evaluation of the influence of the overlap decorrelation length parameter

**Problem**: The parameterization of cloud overlap in ecRad, the so-called cloud overlap decorrelation length, can only be given in the Fortran namelist but changes with latitude.
It should be avoided to create a new namelist for each time step of the along track simulation.

Here the influence of the overlap decorrelation length on the RF17 case study of single layer Arctic cirrus is investigated.
Two simulations are performed using the maximum and minimum calculated overlap decorrelation length after :cite:t:`Shonk2010` along the flight path.
See `/projekt_agmwend/data/HALO-AC3/09_IFS_ECMWF/20220411/20220411_decorrelation_length.csv` for values in km.
Input file from dropsonde location at 11:01 UTC: `ecrad_input_standard_39660.0_sod_v1.nc`

* ``IFS_namelist_jr_20220411_v3.1.nam``: for flight HALO-AC3_20220411_HALO_RF17 with Fu-IFS ice model and `overlap_decorr_length = 1028 m`
* ``IFS_namelist_jr_20220411_v3.2.nam``: for flight HALO-AC3_20220411_HALO_RF17 with Fu-IFS ice model and `overlap_decorr_length = 450 m`

Results
^^^^^^^^

The actual calculated overlap decorrelation length for the location at 11:01 UTC is 482.19 m.

.. figure:: figures/experiment_v3.x/HALO-AC3_20220411_HALO_RF17_overlap_decorrelation_length.png

    Overlap decorrelation length along the flight track of RF17.
    The minimum can be seen at the most northerly point while the maximum is located at the end and beginning of the flight in Kiruna.

.. figure:: figures/experiment_v3.x/HALO-AC3_20220411_HALO_RF17_ecrad_v3.1-v3.2_difference.png

    ecRad broadband irradiance difference between v3.1 and v3.2 over model half levels.

Looking at the differences between the two simulations we can see differences in the solar downward flux of over :math:`1.25\\,\\text{W}\\,\\text{m}^{-2}`.
The values used to calculate the difference are the means of the 33 columns simulated.
However, the plot would not look any different even when only one column is selected.

The difference starts to increase with the simulated cloud top showing the influence of the cloud overlap decorrelation length even for single layer clouds.
Thus, it is important to choose a realistic value for the case study when HALO flew first above and then below the cirrus.
One option is to calculate the cloud overlap parameter :math:`\\alpha` according to Eq. 2.3 of the ecRad documentation:

.. math::

    \\alpha_{i+\\frac{1}{2}} = \\frac{C_{\\text{rand}} - C}{C_{\\text{max}} -C}

with :math:`i` being the model level, :math:`C` being the true combined cloud cover of :math:`i` and :math:`i+1` and
:math:`C_{\\text{rand}}` and :math:`C_{\\text{max}}` the combined cloud covers assuming random and maximum overlap respectively.
See Eq. 2.4 and Eq. 2.5 of the ecRad Documentation for details.
This would make the parameterization of the cloud overlap via the cloud overlap decorrelation length obsolete and cloud overlap could be provided for each timestep with the input netCDF file.
However, :math:`C` cannot be known from model data alone and thus the overlap parameter cannot be calculated.

The other option is to use the mean cloud overlap decorrelation length calculated for the case study period.

| Case study period: 2022-04-11 10:48:47 - 2022-04-11 12:00:32
| Mean cloud overlap decorrelation length for case study period: **482.87 m**

*author*: Johannes Röttenbacher
"""

if __name__ == "__main__":
    # %% import modules
    import pylim.helpers as h
    import pylim.halo_ac3 as meta
    import ac3airborne
    from ac3airborne.tools import flightphase
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    cm = 1 / 2.54
    cb_colors = h.get_cb_friendly_colors()

    # %% set paths
    campaign = "halo-ac3"
    halo_key = "RF18"
    halo_flight = meta.flight_names[halo_key]
    date = halo_flight[9:17]

    plot_path = f"{h.get_path('plot', halo_flight, campaign)}/{halo_flight}/experiment_v3.x"
    h.make_dir(plot_path)
    ecrad_path = f"{h.get_path('ecrad', campaign=campaign)}/{date}/ecrad_output"
    ifs_path = f"{h.get_path('ifs', campaign=campaign)}/{date}"
    ecrad_v3_1 = "ecrad_output_standard_39660.0_sod_v3.1.nc"
    ecrad_v3_2 = "ecrad_output_standard_39660.0_sod_v3.2.nc"
    overlap_file = f"{date}_decorrelation_length.csv"

    # %% get flight segments for case study period
    segmentation = ac3airborne.get_flight_segments()["HALO-AC3"]["HALO"][f"HALO-AC3_HALO_{halo_key}"]
    segments = flightphase.FlightPhaseFile(segmentation)
    above_cloud, below_cloud = dict(), dict()
    if halo_key == "RF17":
        above_cloud["start"] = segments.select("name", "high level 7")[0]["start"]
        above_cloud["end"] = segments.select("name", "high level 8")[0]["end"]
        below_cloud["start"] = segments.select("name", "high level 9")[0]["start"]
        below_cloud["end"] = segments.select("name", "high level 10")[0]["end"]
        above_slice = slice(above_cloud["start"], above_cloud["end"])
        below_slice = slice(below_cloud["start"], below_cloud["end"])
    else:
        above_cloud["start"] = segments.select("name", "polygon pattern 1")[0]["start"]
        above_cloud["end"] = segments.select("name", "polygon pattern 1")[0]["parts"][-1]["start"]
        below_cloud["start"] = segments.select("name", "polygon pattern 2")[0]["start"]
        below_cloud["end"] = segments.select("name", "polygon pattern 2")[0]["end"]
        above_slice = slice(above_cloud["start"], above_cloud["end"])
        below_slice = slice(below_cloud["start"], below_cloud["end"])
        case_slice = slice(above_cloud["start"], below_cloud["end"])

    # %% read in ecrad data and overlap decorr file
    ecrad_ds_v31 = xr.open_dataset(f"{ecrad_path}/{ecrad_v3_1}")
    ecrad_ds_v32 = xr.open_dataset(f"{ecrad_path}/{ecrad_v3_2}")
    # mean or std over columns
    ecrad_ds_v31 = ecrad_ds_v31.mean(dim="column")  # std(dim="column")
    ecrad_ds_v32 = ecrad_ds_v32.mean(dim="column")  # std(dim="column")
    # select only center column
    # ecrad_ds_v31 = ecrad_ds_v31.sel(column=16)  # std(dim="column")
    # ecrad_ds_v32 = ecrad_ds_v32.sel(column=16)  # std(dim="column")
    # convert from km to m
    overlap_decorr_length = pd.read_csv(f"{ifs_path}/{overlap_file}", index_col="time", parse_dates=True) * 1000

    # %% print actual overlap at 11:01
    print(f"Actual overlap: {overlap_decorr_length.loc['2022-04-11 11:01'].values[0]:.2f} m")

    # %% plot overlap along flight path
    _, ax = plt.subplots()
    ax.plot(overlap_decorr_length)
    ax.axvline(above_cloud["start"], label="Start above cloud section", color=cb_colors[3])
    ax.axvline(below_cloud["end"], label="End below cloud section", color=cb_colors[4])
    h.set_xticks_and_xlabels(ax, overlap_decorr_length.index[-1] - overlap_decorr_length.index[0])
    ax.set_title("Overlap decorrelation length along flight path of RF17")
    ax.set_xlabel(f"Time (HH:MM {date[-2:]} April 2022 UTC)")
    ax.set_ylabel("Overlap decorrelation length (m)")
    ax.grid()
    ax.legend()
    plt.savefig(f"{plot_path}/{halo_flight}_overlap_decorrelation_length.png")
    plt.show()
    plt.close()

    # %% plotting variables
    h.set_cb_friendly_colors()
    plt.rc("font", size=12, family="serif")

    # %% prepare dataset for plotting
    ecrad_plot_v31 = ecrad_ds_v31
    ecrad_plot_v32 = ecrad_ds_v32

    # %% plot ecRad broadband fluxes
    labels = dict(flux_dn_sw=r"$F_{\downarrow, solar}$", flux_dn_lw=r"$F_{\downarrow, terrestrial}$",
                  flux_up_sw=r"$F_{\uparrow, solar}$", flux_up_lw=r"$F_{\uparrow, terrestrial}$")
    plt.rc("font", size=12)
    fig, ax = plt.subplots(figsize=(22 * cm, 11 * cm))
    for var in ["flux_dn_sw", "flux_dn_lw", "flux_up_sw", "flux_up_lw"]:
        ax.plot(ecrad_plot_v31[var], ecrad_plot_v31[var].half_level, label=f"v3.1 {labels[var]}")
        ax.plot(ecrad_plot_v32[var], ecrad_plot_v32[var].half_level, label=f"v3.2 {labels[var]}")

    ax.grid()
    ax.legend(bbox_to_anchor=(0.05, 0), loc="lower left", bbox_transform=fig.transFigure, ncol=4)
    ax.set_xlabel("Broadband Irradiance (W$\,$m$^{-2}$)")
    ax.set_ylabel("Model Half Level")
    ax.set_title("ecRad broadband irradiance comparison")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)
    figname = f"{plot_path}/{halo_flight}_ecRad_bb_irradiance_comparison.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

    # %% plot scatterplot of broadband irradiance
    titles = ["Solar Downward Irradiance", "Terrestrial Downward Irradiance", "Solar Upward Irradiance",
              "Terrestrial Upward Irradiance"]
    lims = [(150, 270), (0, 200), (120, 180), (180, 250)]
    # lims = ((0, 500),) * 4
    ecrad_vars = ["flux_dn_sw", "flux_dn_lw", "flux_up_sw", "flux_up_lw"]
    for i, x in enumerate(ecrad_vars):
        _, ax = plt.subplots(figsize=(12 * cm, 12 * cm))
        ax.scatter(ecrad_plot_v31[x], ecrad_plot_v32[x], c=cb_colors[3])
        rmse = np.sqrt(np.mean((ecrad_plot_v31[x] - ecrad_plot_v32[x])**2))
        ax.axline((0, 0), slope=1, color="k", lw=2, transform=ax.transAxes)
        ax.set_ylim(lims[i])
        ax.set_xlim(lims[i])
        ticks = ax.get_yticks() if i == 0 else ax.get_xticks()
        ax.set_yticks(ticks)
        ax.set_xticks(ticks)
        ax.set_aspect('equal')
        ax.set_xlabel("ecRad v3.1 Irradiance (W$\,$m$^{-2}$)")
        ax.set_ylabel("ecRad v3.2 Irradiance (W$\,$m$^{-2}$)")
        ax.set_title(f"{titles[i]}")
        ax.grid()
        ax.text(0.01, 0.95, f"# points: {sum(~np.isnan(ecrad_plot_v31[x])):.0f}\n"
                            f"RMSE: {rmse.values:.4f}" + " W$\,$m$^{-2}$",
                horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
        plt.tight_layout()
        plt.savefig(f"{plot_path}/{halo_flight}_ecrad_{x}_v3.1_vs_v3.2_scatter.png", dpi=300)
        plt.show()
        plt.close()

    # %% plot difference between v3.1 and v3.2
    for x in ecrad_vars:
        difference = ecrad_plot_v31[x] - ecrad_plot_v32[x]
        _, ax = plt.subplots()
        ax.plot(difference)
        ax.set_title(f"v3.1 - v3.2 {x}")
        ax.set_xlabel("Model Half Level")
        ax.set_ylabel("Broadband Irradiance (W$\,$m$^{-2}$)")
        ax.grid()
        ax.text(0.01, 0.95, f"Mean: {difference.mean():.2f}" + " W$\,$m$^{-2}$", ha="left", va="top", transform=ax.transAxes)
        plt.tight_layout()
        plt.savefig(f"{plot_path}/{halo_flight}_ecrad_{x}_v3.1-v3.2_difference.png")
        plt.show()
        plt.close()
    # %% plot difference between v3.1 and v3.2 in one plot
    _, ax = plt.subplots()
    for i, x in enumerate(ecrad_vars):
        difference = ecrad_plot_v31[x] - ecrad_plot_v32[x]
        ax.plot(difference, label=x)
        ax.text(0.01, 0.65 - i * 0.05, f"Mean {x}: {difference.mean():.2f}" + " W$\,$m$^{-2}$",
                ha="left", va="top", transform=ax.transAxes)
    ax.legend(loc=2)
    ax.set_title("ecRad broadband irradiance difference v3.1 - v3.2")
    ax.set_xlabel("Model Half Level")
    ax.set_ylabel("Broadband Irradiance (W$\,$m$^{-2}$)")
    ax.grid()
    plt.tight_layout()
    plt.savefig(f"{plot_path}/{halo_flight}_ecrad_v3.1-v3.2_difference.png")
    plt.show()
    plt.close()

    # %% mean overlap decor length for case study
    mean_odl = overlap_decorr_length[above_cloud["start"]:below_cloud["end"]].mean()
    print(f"Case study period: {above_cloud['start']} - {below_cloud['end']}"
          f"\nMean overlap decorrelation length for case study period: {mean_odl.values[0]:.2f} m")
