#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 05.02.2024

Here we investigate the impact the switch to the direct sea ice albedo according to :cite:t:`ebert1993` has during the case study section of RF 17.
For this we only compare the two reference simulations using the old (v15.1_old_ci_albedo) and the new (v15.1) way of calculating the ``sw_albedo_direct`` variable.

At first, we look at the ``sw_albedo_direct`` variable used in the simulations and the difference between them.

.. figure:: figures/new_direct_sea_ice_albedo/sw_albedo_direct_old_vs_new.png

    Old and new ``sw_albedo_direct`` variable and the difference between them.

We can see up to 10% higher values using the new ``sw_albedo_direct``.
This also has an impact on the simulated irradiance, which is shown below.

.. figure:: figures/new_direct_sea_ice_albedo/HALO-AC3_20220411_HALO_RF17_ecrad_diff_flux_up_sw_along_track.png

    Difference in solar upward flux between old and new ``sw_albedo_direct``.

The influence on the solar transmissivity, however, is rather small as can be seen when comparing the PDFs of the below cloud values.

.. figure:: figures/new_direct_sea_ice_albedo/HALO-AC3_HALO_RF17_RF18_bacardi_ecrad_transmissivity_below_cloud_PDF_ci_albedo_exp.png

    PDFs of solar transmissivity below cloud for the two cases.

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

    plot_path = f"{h.get_path('plot', flight, campaign)}/new_direct_sea_ice_albedo"
    fig_path = "./docs/figures/new_direct_sea_ice_albedo"
    h.make_dir(plot_path)
    ecrad_path = f"{h.get_path('ecrad', campaign=campaign)}/{date}"

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

# %% read in ecrad data
    ecrad_dict = dict()
    for v in ["v15.1", "v15.1_old_ci_albedo"]:
        # use center column data
        ds = xr.open_dataset(f"{ecrad_path}/ecrad_merged_inout_{date}_{v}.nc").isel(column=0)
        # select above and below cloud time
        ds = ds.sel(time=case_slice)
        ds = ds.assign_coords({"sw_albedo_band": range(1, 7)})
        ecrad_dict[v] = ds.copy()

# %% plot new and old sw_albedo_direct
    _, axs = plt.subplots(1, 3, figsize=h.figsize_wide, layout="constrained")
    # old sw_albedo_direct
    ds_old = ecrad_dict["v15.1_old_ci_albedo"]["sw_albedo_direct"]
    ds_old.plot(x="time", ax=axs[0], add_colorbar=False, vmin=0, vmax=1,
                cmap=cmr.rainforest)
    ds_new = ecrad_dict["v15.1"]["sw_albedo_direct"]
    ds_new.plot(x="time", ax=axs[1], vmin=0, vmax=1, cmap=cmr.rainforest,
                cbar_kwargs={"label": "Direct shortwave albedo"})
    axs[1].set(title="New")
    (ds_old - ds_new).plot(x="time", ax=axs[2], cmap=cmr.fusion_r,
                           norm=colors.TwoSlopeNorm(vcenter=0),
                           cbar_kwargs={"label": "Difference"})
    axs[2].set(title="Old - New")
    for ax in axs:
        h.set_xticks_and_xlabels(ax, pd.Timedelta(2, "hr"))
        ax.set(xlabel="Time (UTC)", ylabel="")

    axs[0].set(title="Old",
               ylabel="Shortwave albedo band")

    figname = f"sw_albedo_direct_old_vs_new.png"
    plt.savefig(f"{plot_path}/{figname}", dpi=300)
    plt.savefig(f"{fig_path}/{figname}", dpi=300)
    plt.show()
    plt.close()

# %% set plotting options
    var = "flux_up_sw"
    v = "diff"
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
        ds = ecrad_dict["v15.1"]
        ecrad_ds_diff = ecrad_dict["v15.1_old_ci_albedo"][var] - ds[var]
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
    plt.savefig(f"{fig_path}/{figname}", dpi=300)
    plt.savefig(f"{plot_path}/{figname}", dpi=300)
    plt.show()
    plt.close()

# %% plot histogram
    xlabels = dict(diff="difference Old - New")
    xlabel = xlabels[v] if v in xlabels else v
    flat_array = ecrad_plot.to_numpy().flatten()
    mean = np.nanmean(flat_array)
    median = np.nanmedian(flat_array)
    _, ax = plt.subplots(figsize=h.figsize_wide)
    hist = ax.hist(flat_array, bins=20)
    ax.set(xlabel=f"{h.cbarlabels[var]} {xlabel} ({h.plot_units[var]})",
           ylabel="Number of occurrence")
    ax.grid()
    ax.set_yscale("log")
    ax.text(0.75, 0.9, f"Mean: {mean:.2f}" + "W$\,$m$^{-2}$" + f"\nMedian: {median:.2f}" + "W$\,$m$^{-2}$",
            ha='left', va='top', transform=ax.transAxes,
            bbox=dict(fc='white', ec='black', alpha=0.8, boxstyle='round'))
    plt.tight_layout()
    figname = f"{flight}_ecrad_{v}_{var}{band_str}_hist.png"
    plt.savefig(f"{plot_path}/{figname}", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig_path}/{figname}", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# %% plot PDF of solar transmissivity of BACARDI and ecRad at flight altitude for both setting
    transmissivity_stats = list()
    plt.rc("font", size=7.5)
    ylims = (0, 28)
    binsize = 0.01
    xlabel = "Solar Transmissivity"
    _, ax = plt.subplots(figsize=(18 * h.cm, 14 * h.cm), layout="constrained")
    bins = np.arange(0.5, 1.0, binsize)

    for i, version in enumerate(["v15.1", "v15.1_old_ci_albedo"]):
        ecrad_ds = ecrad_dict[version]
        height_sel = ecrad_ds["aircraft_level"]
        ecrad_plot = ecrad_ds[f"transmissivity_sw_above_cloud"].isel(half_level=height_sel)

        # actual plotting
        sns.histplot(ecrad_plot.to_numpy().flatten(), label=version, stat="density", element="step",
                     kde=False, bins=bins, ax=ax, color=cbc[i])
        # add mean
        ax.axvline(ecrad_plot.mean(), color=cbc[i], lw=3, ls="--")


    ax.plot([], ls="--", color="k", label="Mean")  # label for means
    ax.legend(loc=6)
    ax.grid()
    ax.set(xlabel="Solar Transmissivity",
           ylabel="Probability density function")

    figname = f"{fig_path}/HALO-AC3_HALO_RF17_RF18_bacardi_ecrad_transmissivity_below_cloud_PDF_ci_albedo_exp.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

