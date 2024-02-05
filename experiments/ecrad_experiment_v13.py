#!/usr/bin/env python
"""
| *author:* Johannes Röttenbacher
| *created:* 15-04-2023

Set the albedo during the whole flight to open ocean (diffuse: 0.06, direct: Taylor et al. 1996) or scale it to 0.99 (maximum albedo) and analyze the impact on the offset between simulation and measurement during the below cloud section of RF17.

* ``IFS_namelist_jr_20220411_v13.nam``: for flight RF17 with Fu-IFS ice model setting albedo to open ocean (input version v5)
* ``IFS_namelist_jr_20220411_v13.1.nam``: for flight RF17 with Fu-IFS ice model setting albedo to 0.99 (input version v5.1)
* ``IFS_namelist_jr_20220411_v13.2.nam``: for flight RF17 with Fu-IFS ice model setting albedo BACARDI measurement from below cloud section (input version v5.2)
* ``IFS_namelist_jr_20220411_v15.1.nam:``: for RF17 with Fu-IFS ice model using O1280 IFS data (input version v6.1) (**reference simulation**)

**Problem statement:** A clear offset can be observed in the solar downward irradiance below the cloud between ecRad and BACARDI with ecRad showing lower values than BACARDI.
One idea is that multiple scattering from the sea ice surface to the cloud and back down plays is a process not captured by the model.
By setting the albedo to open ocean we can eliminate this multiple backscattering between cloud and surface.
Comparing this experiment with the standard experiment can show us the potential impact of multiple scattering.
We also run an experiment where we scale the albedo to 0.99 to see how much more downward irradiance we can observe that way.

**Extension:** We extend the analysis and use the scaled measured below cloud albedo from BACARDI as input.

We will focus on the above and below cloud section in the far north.
The corresponding spectral surface albedo as used in the IFS can be seen in :numref:`surface-albedo-cs`.

.. _surface-albedo-cs:

.. figure:: figures/experiment_v13/HALO-AC3_20220411_HALO_RF17_sw_albedo_along_track_v15.1.png

    Short wave albedo along track above and below cloud for all six spectral bands after :cite:t:`ebert1993`.

At first, we look at the difference in solar upward and downward irradiance between v15.1 (IFS albedo after :cite:t:`ebert1993`) and v13 (ocean albedo).

.. figure:: figures/experiment_v13/HALO-AC3_20220411_HALO_RF17_ecrad_diff_flux_up_sw_along_track.png

    Difference in solar upward irradiance between v15.1 and v13.

.. figure:: figures/experiment_v13/HALO-AC3_20220411_HALO_RF17_ecrad_diff_flux_dn_sw_along_track.png

    Difference in solar downward irradiance between v15.1 and v13.

We can see an unsurprising substantial difference in upward irradiance which then propagates to a smaller but still relevant difference in downward irradiance.
This is especially pronounced for the thicker section of the cirrus at around 11:15 UTC.

An interesting side note: Although the surface albedo is now set to an open ocean value the emissivity and skin temperature are still the same.
Thus, there is only a minor change in the terrestrial upward irradiance.

So how much difference does multiple scattering between the surface and cloud make?
For this we can take a look at the histogramm of differences and some statistics.

.. figure:: figures/experiment_v13/HALO-AC3_20220411_HALO_RF17_ecrad_diff_flux_dn_sw_hist.png

    Histogram of differences between v15.1 and v13.

We see that a lot of values are rather small.
They correspond to the area above the cloud where only the atmosphere causes some minor scattering.
The median and mean of the distribution, however, are around :math:`10\\,Wm^{-2}`, which is quite substantial.

So albedo does obviously have a major influence on the downward irradiance in this scenario.
The next question now is, whether we can reduce the bias by increasing the surface albedo?
For this we take a look at experiment v13.1 with a scaled albedo of 0.99.
The albedo is scaled in such a way that the maximum albedo in the first short wave albedo band is set to 0.99 and the following bands are scaled according to the relative differences between the original short wave albedo bands.
See the script for details.

.. figure:: figures/experiment_v13/HALO-AC3_20220411_HALO_RF17_ecrad_diff1_flux_dn_sw_along_track.png

    Difference in solar downward irradiance between v15.1 and v13.1 (albedo = 0.99).

By scaling the albedo to an unrealistic value of 0.99 we get a maximum of :math:`0.45\\,Wm^{-2}` difference in solar downward irradiance.
Comparing the spectral albedo of each experiment in :numref:`spectral-albedo-all-experiments` we can also see, that the standard albedo for the scene is already high.
So increasing it does not seem to be a sensible idea.

.. _spectral-albedo-all-experiments:

.. figure:: figures/experiment_v13/HALO-AC3_20220411_HALO_RF17_IFS_sw_albedo_spectrum_1100.png

    Spectral albedo for all four experiments at one timestep below cloud.

However, what happens if we use the measured broadband albedo from BACARDI for the below cloud simulation which is lower than the one in the IFS but not as low as for open ocean?

.. figure:: figures/experiment_v13/HALO-AC3_20220411_HALO_RF17_ecrad_diff2_flux_dn_sw_along_track.png

    Difference in solar downward irradiance between v15.1 and v13.2 (albedo from BACARDI).

Looking at the difference in solar downward irradiance we can see that it is still a positive difference meaning the predicted downward irradiance is still smaller compared to the IFS run.
The comparison with the measurements also shows a worse match compared to v15.1.

.. figure:: figures/experiment_v13/HALO-AC3_20220411_HALO_RF17_ecrad_flux_dn_sw_below_cloud_boxplot.png


From all this we can conclude that **the albedo does not seem to be the major problem** in this scene.
Or at least, we cannot tweak it in any reasonable way to improve the simulations.

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
    from matplotlib import colors
    import pandas as pd
    import seaborn as sns
    from tqdm import tqdm
    import cmasher as cmr

    cbc = h.get_cb_friendly_colors("petroff_6")
    h.set_cb_friendly_colors("petroff_6")
    plt.rc("font", size=12)

# %% set paths
    campaign = "halo-ac3"
    key = "RF17"
    flight = meta.flight_names[key]
    date = flight[9:17]

    plot_path = f"{h.get_path('plot', flight, campaign)}/{flight}/experiment_v13"
    fig_path = "./docs/figures/experiment_v13"
    h.make_dir(plot_path)
    ecrad_path = f"{h.get_path('ecrad', campaign=campaign)}/{date}"
    bacardi_path = h.get_path("bacardi", flight, campaign)
    bacardi_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1_JR.nc"
    bahamas_path = h.get_path("bahamas", flight, campaign)
    bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{key}_v1_1Min.nc"
    data_path = h.get_path('plot', flight, campaign)
    stats_file = "statistics.csv"

# %% get flight segments for case study period
    segmentation = ac3airborne.get_flight_segments()["HALO-AC3"]["HALO"][f"HALO-AC3_HALO_{key}"]
    segments = flightphase.FlightPhaseFile(segmentation)
    above_cloud, below_cloud = dict(), dict()
    if key == "RF17":
        above_cloud["start"] = segments.select("name", "high level 7")[0]["start"]
        above_cloud["end"] = segments.select("name", "high level 8")[0]["end"]
        below_cloud["start"] = segments.select("name", "high level 9")[0]["start"]
        below_cloud["end"] = segments.select("name", "high level 10")[0]["end"]
        above_slice = slice(above_cloud["start"], above_cloud["end"])
        below_slice = slice(below_cloud["start"], below_cloud["end"])
        case_slice = slice(pd.to_datetime("2022-04-11 10:30"), pd.to_datetime("2022-04-11 12:29"))
    else:
        above_cloud["start"] = segments.select("name", "polygon pattern 1")[0]["start"]
        above_cloud["end"] = segments.select("name", "polygon pattern 1")[0]["parts"][-1]["start"]
        below_cloud["start"] = segments.select("name", "polygon pattern 2")[0]["start"]
        below_cloud["end"] = segments.select("name", "polygon pattern 2")[0]["end"]
        above_slice = slice(above_cloud["start"], above_cloud["end"])
        below_slice = slice(below_cloud["start"], below_cloud["end"])
        case_slice = slice(above_cloud["start"], below_cloud["end"])

    time_extend_cs = below_cloud["end"] - above_cloud["start"]  # time extend for case study

# %% read in bahamas data
    ins = xr.open_dataset(f"{bahamas_path}/{bahamas_file}")

# %% read in BACARDI data
    bacardi_ds = xr.open_dataset(f"{bacardi_path}/{bacardi_file}").sel(time=case_slice)

# %% read in ecrad data
    ecrad_dict = dict()
    for v in ["v15.1", "v13", "v13.1", "v13.2"]:
        # use center column data
        ds = xr.open_dataset(f"{ecrad_path}/ecrad_merged_inout_{date}_{v}.nc").isel(column=0)
        # select above and below cloud time
        ds = ds.sel(time=case_slice)
        ds = ds.assign_coords({"sw_albedo_band": range(1, 7)})
        ecrad_dict[v] = ds.copy()

# %% read in statistics
    stats_df = pd.read_csv(f"{data_path}/{stats_file}")
    # select only relevant versions

# %% prepare metadata for comparing ecRad and BACARDI
    titles = ["Solar Downward Irradiance", "Terrestrial Downward Irradiance", "Solar Upward Irradiance",
              "Terrestrial Upward Irradiance"]
    names = ["Fdw_solar", "Fdw_terrestrial", "Fup_solar", "Fup_terrestrial"]
    bacardi_vars = ["F_down_solar", "F_down_terrestrial", "F_up_solar", "F_up_terrestrial"]
    ecrad_vars = ["flux_dn_sw", "flux_dn_lw", "flux_up_sw", "flux_up_lw"]

# %% set plotting options
    var = "flux_dn_sw"
    v = "diff2"
    band = None

# %% prepare data set for plotting
    band_str = f"_band{band}" if band is not None else ""
    # kwarg dicts
    alphas = dict()
    ct_fontsize = dict()
    ct_lines = dict(ciwc=[1, 5, 10, 15], cswc=[1, 5, 10, 15], q_ice=[1, 5, 10, 15], clwc=[1, 5, 10, 15],
                    iwc=[1, 5, 10, 15], flux_dn_sw=[-5, -1, 0, 1, 5, 10, 15, 20], flux_up_sw=[0, 25, 50, 75, 100, 125, 150])
    linewidths = dict()
    robust = dict(iwc=False)
    cb_ticks = dict()
    vmaxs = dict()
    vmins = dict(iwp=0)
    xlabels = {"v15.1": "v15.1", "v13": "v13", "diff": "Difference v15.1 - v13"}

    # set kwargs
    alpha = alphas[var] if var in alphas else 1
    cmap = h.cmaps[var] if var in h.cmaps else cmr.rainforest
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(color="white")
    ct_fs = ct_fontsize[var] if var in ct_fontsize else 8
    lines = ct_lines[var] if var in ct_lines else None
    lw = linewidths[var] if var in linewidths else 1
    norm = h.norms[var] if var in h.norms else None
    robust = robust[var] if var in robust else True
    ticks = cb_ticks[var] if var in cb_ticks else None
    if norm is None:
        vmax = vmaxs[var] if var in vmaxs else None
        vmin = vmins[var] if var in vmins else None
    else:
        vmax, vmin = None, None

    if "diff" in v:
        cmap = cmr.fusion_r
        norm = colors.TwoSlopeNorm(vcenter=0)

    # prepare ecrad dataset for plotting
    sf = h.scale_factors[var] if var in h.scale_factors else 1
    if v == "diff":
        # calculate difference between simulations
        ds = ecrad_dict["v13"]
        ecrad_ds_diff = ecrad_dict["v15.1"][var] - ds[var]
        ecrad_plot = ecrad_ds_diff.where((ds[var] != 0) | (~np.isnan(ds[var]))) * sf
    elif v == "diff1":
        # calculate difference between simulations
        ds = ecrad_dict["v13.1"]
        ecrad_ds_diff = ecrad_dict["v15.1"][var] - ds[var]
        ecrad_plot = ecrad_ds_diff.where((ds[var] != 0) | (~np.isnan(ds[var]))) * sf
    elif v == "diff2":
        # calculate difference between simulations
        ds = ecrad_dict["v13.2"]
        ecrad_ds_diff = ecrad_dict["v15.1"][var] - ds[var]
        ecrad_plot = ecrad_ds_diff.where((ds[var] != 0) | (~np.isnan(ds[var]))) * sf
    else:
        ds = ecrad_dict[v]
        ecrad_plot = ds[var] * sf

    # add new z axis mean pressure altitude
    if "half_level" in ecrad_plot.dims:
        new_z = ds["press_height_hl"].mean(dim="time") / 1000
    else:
        new_z = ds["press_height_full"].mean(dim="time") / 1000

    ecrad_plot_new_z = list()
    for t in tqdm(ecrad_plot.time, desc="New Z-Axis"):
        tmp_plot = ecrad_plot.sel(time=t)
        if "half_level" in tmp_plot.dims:
            tmp_plot = tmp_plot.assign_coords(
                half_level=ds["press_height_hl"].sel(time=t, drop=True).to_numpy() / 1000)
            tmp_plot = tmp_plot.rename(half_level="height")

        else:
            tmp_plot = tmp_plot.assign_coords(
                level=ds["press_height_full"].sel(time=t, drop=True).to_numpy() / 1000)
            tmp_plot = tmp_plot.rename(level="height")

        tmp_plot = tmp_plot.interp(height=new_z.to_numpy())
        ecrad_plot_new_z.append(tmp_plot)

    ecrad_plot = xr.concat(ecrad_plot_new_z, dim="time")
    # filter very low to_numpy()
    ecrad_plot = ecrad_plot.where(np.abs(ecrad_plot) > 0.001)

    # select time height slice
    time_sel = case_slice
    height_sel = slice(13, 0)
    if len(ecrad_plot.dims) > 2:
        dim3 = "band_sw"
        dim3 = dim3 if dim3 in ecrad_plot.dims else None
        ecrad_plot = ecrad_plot.sel({"time": time_sel, "height": height_sel, f"{dim3}": band})
    else:
        ecrad_plot = ecrad_plot.sel(time=time_sel, height=height_sel)

    time_extend = pd.to_timedelta((ecrad_plot.time[-1] - ecrad_plot.time[0]).to_numpy())

# %% plot 2D IFS variables along flight track
    _, ax = plt.subplots(figsize=h.figsize_wide, layout="constrained")
    # ecrad 2D field
    ecrad_plot.plot(x="time", y="height", cmap=cmap, ax=ax, robust=robust, vmin=vmin, vmax=vmax, alpha=alpha, norm=norm,
                    cbar_kwargs={"pad": 0.04, "label": f"{h.cbarlabels[var]} ({h.plot_units[var]})",
                                 "ticks": ticks})
    if lines is not None:
        # add contour lines
        ct = ax.contour(ecrad_plot.time, ecrad_plot.height, ecrad_plot.to_numpy().T, levels=lines, linestyles="--",
                        colors="k",
                        linewidths=lw)
        ct.clabel(fontsize=ct_fs, inline=1, inline_spacing=0, fmt='%i', rightside_up=True, use_clabeltext=True)

    ax.set_title(f"{key} IFS/ecRad input/output along Flight Track - {v}")
    ax.set_ylabel("Altitude (km)")
    ax.set_xlabel("Time (UTC)")
    h.set_xticks_and_xlabels(ax, time_extend)
    figname = f"{flight}_ecrad_{v}_{var}{band_str}_along_track.png"
    plt.savefig(f"{plot_path}/{figname}", dpi=300)
    plt.savefig(f"{fig_path}/{figname}", dpi=300)
    plt.show()
    plt.close()

# %% plot histogram
    xlabels = dict(diff="difference v15.1 - v13", diff1="difference v15.1 - v13.1", diff2="difference v15.1 - v13.2")
    xlabel = xlabels[v] if v in xlabels else v
    flat_array = ecrad_plot.to_numpy().flatten()
    mean = np.mean(flat_array)
    median = np.median(flat_array)
    _, ax = plt.subplots(figsize=h.figsize_wide, layout="constrained")
    hist = ax.hist(flat_array, bins=20)
    ax.set(xlabel=f"{h.cbarlabels[var]} {xlabel} ({h.plot_units[var]})",
           ylabel="Number of occurrence")
    ax.grid()
    ax.set_yscale("log")
    ax.text(0.75, 0.9, f"Mean: {mean:.2f}" + "W$\,$m$^{-2}$" + f"\nMedian: {median:.2f}" + "W$\,$m$^{-2}$",
            ha='left', va='top', transform=ax.transAxes,
            bbox=dict(fc='white', ec='black', alpha=0.8, boxstyle='round'))
    figname = f"{flight}_ecrad_{v}_{var}{band_str}_hist.png"
    plt.savefig(f"{fig_path}/{figname}", dpi=300)
    plt.savefig(f"{plot_path}/{figname}", dpi=300)
    plt.show()
    plt.close()

# %% plot timeseries
    var = "sw_albedo"
    v = "v15.1"
    band = None
    if v == "diff":
        ecrad_plot = ecrad_dict["v15.1"][var] - ecrad_dict["v13"][var]
    else:
        ecrad_plot = ecrad_dict[v][var]

    _, ax = plt.subplots(figsize=h.figsize_wide)
    ecrad_plot.plot(x="time", cmap=cmr.rainforest, vmin=0, vmax=1, cbar_kwargs={"label": "SW albedo"})
    ax.grid()
    h.set_xticks_and_xlabels(ax, time_extend_cs)
    yticklabels = list(x[1] for x in h.ci_bands)
    yticklabels.insert(0, 0)
    ax.set(xlabel="Time (UTC)", ylabel="SW albedo band ($\mu$m)",
           title=f"{key} IFS/ecRad input/output along Flight Track - {v}",
           yticks=range(0, 7), yticklabels=yticklabels)
    figname = f"{plot_path}/{flight}_sw_albedo_along_track_{v}.png"
    plt.savefig(figname, bbox_inches="tight", dpi=300)
    figname = f"{fig_path}/{flight}_sw_albedo_along_track_{v}.png"
    plt.savefig(figname, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()

# %% plot albedo spectrum from below cloud
    _, ax = plt.subplots(figsize=h.figsize_wide)
    for v in ["v15.1", "v13", "v13.1", "v13.2"]:
        ds_plot = ecrad_dict[v]["sw_albedo"].sel(time="2022-04-11 11:00", method="nearest")
        ax.plot(ds_plot.to_numpy(), lw=3, label=v, marker="X", ms=12)
    ax.grid()
    ax.legend()
    xticklabels = list(x[1] for x in h.ci_bands)
    ax.set(xlabel="Wavelength ($\mu$m)", ylabel="Spectral albedo", title="Spectral surface albedo at 2022-04-11 11:00 UTC",
           xticks=range(0, 6), xticklabels=xticklabels)
    figname = f"{plot_path}/{flight}_IFS_sw_albedo_spectrum_1100.png"
    plt.savefig(figname, bbox_inches="tight", dpi=300)
    figname = f"{fig_path}/{flight}_IFS_sw_albedo_spectrum_1100.png"
    plt.savefig(figname, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()

# %% prepare data for box plot
    values = list()
    for v in ["v15.1", "v13", "v13.1", "v13.2"]:
        values.append(ecrad_dict[v].flux_dn_sw
                      .isel(half_level=ecrad_dict[v].aircraft_level)
                      .sel(time=below_slice)
                      .to_numpy())

    df1 = pd.DataFrame({"IFS (v15.1)": values[0], "Open Ocean (v13)": values[1],
                        "Maximum (v13.1)": values[2], "Measured (v13.2)": values[3]})
    df2 = (bacardi_ds["F_down_solar_diff"]
           .sel(time=below_slice)
           .dropna("time")
           .to_pandas()
           .reset_index(drop=True))
    df2 = pd.DataFrame(df2)
    df2.columns = ["BACARDI"]
    df1["label"] = "ecRad"
    df2["label"] = "BACARDI"

    df1.to_csv(f"C:/Users/Johannes/Documents/Doktor/manuscripts/_arctic_cirrus/figures/{flight}_boxplot_data.csv",
              index=False)

    df = pd.concat([df2, df1])

# %% plot boxplots of solar downward irradiance below cloud for each experiment
    plt.rc("font", size=12)
    _, ax = plt.subplots(figsize=h.figsize_wide, layout="constrained")
    sns.boxplot(df, notch=True, ax=ax)
    ax.set_ylabel("Solar downward irradiance (W$\,$m$^{-2}$)")
    ax.grid()
    figname = f"{flight}_ecrad_flux_dn_sw_below_cloud_boxplot.png"
    plt.savefig(f"{plot_path}/{figname}", dpi=300)
    plt.savefig(f"{fig_path}/{figname}", dpi=300)
    plt.show()
    plt.close()

# %% plot PDF of below cloud variables
    var = "F_down_solar_diff"
    ecrad_var = "flux_dn_sw"
    v = "v15.1"
    bacardi_plot = bacardi_ds[var].sel(time=below_slice)
    ecrad_plot = (ecrad_dict[v][ecrad_var]
                  .isel(half_level=ecrad_dict[v].aircraft_level)
                  .sel(time=below_slice))
    binsize = 10
    bins = np.arange(np.round(bacardi_plot.min() - 10),
                     np.round(bacardi_plot.max() + 10),
                     binsize)
    plt.rc("font", size=12)
    _, ax = plt.subplots(figsize=h.figsize_wide, layout="constrained")

    sns.histplot(bacardi_plot, label="BACARDI", stat="density", bins=bins,
                 element="step", ax=ax,)
    sns.histplot(ecrad_plot.to_numpy().flatten(), label=f"ecRad {v}",
                 stat="density", element="step", bins=bins, ax=ax)

    ax.grid()
    ax.legend()
    ax.set(xlabel=f"{h.cbarlabels[ecrad_var]} ({h.plot_units[ecrad_var]})")
    figname = f"{fig_path}/{flight}_{ecrad_var}_bacardi_vs_ecrad_{v}.png"
    # plt.savefig(figname, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close()
