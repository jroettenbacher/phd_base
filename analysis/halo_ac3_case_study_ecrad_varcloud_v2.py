#!/usr/bin/env python
"""
| *author:* Johannes Röttenbacher
| *created:* 12-07-2023

Comparison between using the VarCloud retrieval for *IWC* and |re-ice| or the forecasted values from the IFS as input for an ecRad simulation using the Fu-IFS ice optic parameterization.
In addition, an experiment is run where the above cloud retrieved IWC and |re-ice| is used as input for the below cloud section.
This allows for a comparison between measurements and simulations.
The assumption is that the cloud does not change in the 30 minutes it takes to reach the below cloud section.

* ``IFS_namelist_jr_20220411_v15.nam``: for flight RF17 with Fu-IFS ice model using IFS data from its original O1280 grid (input version v6)
* ``IFS_namelist_jr_20220411_v16.nam``: for flight RF17 with Fu-IFS ice model using O1280 IFS data and varcloud retrieval for ciwc and re_ice input for the below cloud section (input version v7)
* ``IFS_namelist_jr_20220411_v17.nam``: for flight RF17 with Fu-IFS ice model using O1280 IFS data and varcloud retrieval for ciwc and re_ice input (input version v8)

.. figure:: figures/varcloud/HALO-AC3_20220411_HALO_RF17_ecrad_v16_iwc_along_track.png

    Retrieved ice water content from VarCloud retrieval interpolated to IFS full level pressure altitudes.

Results
^^^^^^^^

"""

if __name__ == "__main__":
# %% import modules
    import pylim.helpers as h
    import pylim.halo_ac3 as meta
    from pylim import ecrad
    import ac3airborne
    from ac3airborne.tools import flightphase
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import colors
    import pandas as pd
    import os
    from tqdm import tqdm
    import cmasher as cmr

    # plotting variables
    cbc = h.get_cb_friendly_colors()
    h.set_cb_friendly_colors()
    plt.rc("font", size=12)

# %% set paths
    campaign = "halo-ac3"
    key = "RF17"
    flight = meta.flight_names[key]
    date = flight[9:17]

    plot_path = f"{h.get_path('plot', flight, campaign)}/{flight}/varcloud"
    fig_path = "./docs/figures/varcloud"
    h.make_dir(plot_path)
    ecrad_path = f"{h.get_path('ecrad', campaign=campaign)}/{date}/"
    ifs_path = f"{h.get_path('ifs', campaign=campaign)}/{date}"
    varcloud_path = h.get_path("varcloud", flight, campaign)
    libradtran_path = h.get_path("libradtran_exp", flight, campaign)
    bacardi_path = h.get_path("bacardi", flight, campaign)
    varcloud_file = [f for f in os.listdir(varcloud_path) if "nc" in f][0]
    bacardi_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1_1s.nc"
    bacardi_std = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1_std_1s.nc"
    libradtran_file = f"HALO-AC3_HALO_libRadtran_simulation_varcloud_1min_{date}_{key}.nc"
    ecrad_versions = ["v15", "v16", "v17", "v18"]

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
    # above cloud time with thin cirrus below
    sel_time = slice(above_cloud["start"], pd.to_datetime("2022-04-11 11:04"))

# %% read in data
    varcloud = xr.open_dataset(f"{varcloud_path}/{varcloud_file}").swap_dims(time="Time", height="Height").rename(
        Time="time")
    varcloud = varcloud.rename(Varcloud_Cloud_Ice_Water_Content="iwc", Varcloud_Cloud_Ice_Effective_Radius="re_ice")
    varcloud = varcloud.sel(time=sel_time).resample(time="1Min").asfreq()

    bacardi_ds = xr.open_dataset(f"{bacardi_path}/{bacardi_file}")  # read in BACARDI data
    bacardi_std = xr.open_dataset(f"{bacardi_path}/{bacardi_std}")
    bacardi_ds_res = bacardi_ds.resample(time="1Min").mean()

    libradtran_ds = xr.open_dataset(f"{libradtran_path}/{libradtran_file}")  # read in libradtran data
    libradtran_ds["altitude"] = libradtran_ds.altitude / 1000
    libradtran_ds = libradtran_ds.rename(CIWD="iwc")

# %% read in ecrad data and add more variables to each data set
    ecrad_dict = dict()
    cloud_properties = ["iwc", "iwp", "re_ice", "re_liquid", "q_liquid", "q_ice", "ciwc", "cswc", "clwc", "crwc"]
    above_cloud_hl, below_cloud_hl = 87.5, 103.5
    for v in tqdm(ecrad_versions):
        # merged input and output file with additional variables
        ecrad_file = f"ecrad_merged_inout_{date}_{v}.nc"
        ds = xr.open_dataset(f"{ecrad_path}/{ecrad_file}")
        # select only closest column to flight path
        if "column" in ds.dims:
            ds = ds.sel(column=1, drop=True)
        # calculate spectral absorption by cloud, above cloud - below cloud spectrum
        for var in ["spectral_flux_dn_sw", "spectral_flux_dn_lw", "spectral_flux_up_sw", "spectral_flux_up_lw"]:
            ds_tmp = ds[var]
            ds[f"{var}_diff"] = ds_tmp.sel(half_level=above_cloud_hl) - ds_tmp.sel(half_level=below_cloud_hl)
        # filter cloud properties with cloud_fraction
        for p in cloud_properties:
            ds[p] = ds[p].where(ds.cloud_fraction > 0)

        ecrad_dict[v] = ds.copy()

# %% calculate difference between IFS run varcloud runs
    ds_fu = ecrad_dict["v15"]
    for k in ["v16", "v17"]:
        ds = ecrad_dict[k].copy()
        ds_diff = ds_fu - ds
        ecrad_dict[f"{k}_diff"] = ds_diff.copy()

# %% get model level of flight altitude for half and full level
    level_da = ecrad.get_model_level_of_altitude(bacardi_ds.alt, ecrad_dict["v15"], "level")
    hlevel_da = ecrad.get_model_level_of_altitude(bacardi_ds.alt, ecrad_dict["v15"], "half_level")

# %% create selection array for all above cloud sections
    end_ascend = pd.to_datetime(segments.select("name", "high level 1")[0]["start"])
    end_ascend2 = pd.to_datetime(segments.select("name", "high level 11")[0]["start"])
    t = pd.to_datetime(bacardi_ds_res.time)
    above_sel = xr.DataArray(((t > end_ascend) & (t < above_cloud["end"])) | (t > end_ascend2),
                             coords={"time": bacardi_ds_res.time})
# %% plotting preparation
    var_dict = dict(flux_dn_sw="F_down_solar", flux_dn_lw="F_down_terrestrial",
                    flux_up_sw="F_up_solar", flux_up_lw="F_up_terrestrial")
    version_labels = dict(v15="Fu-IFS", v16="VarCloud Below Cloud", v17="VarCloud", v18="Baran2016", v19="Yi",
                          v18_diff="Fu-IFS-Baran2016", v19_diff="Fu-IFS-Yi")

# %% set plotting options
    v_ref = "v15"
    var = "cloud_fraction"
    v = "v15"
    band = None

# %% prepare data set for plotting
    band_str = f"_band{band}" if band is not None else ""

    # kwarg dicts
    alphas = dict()
    ct_fontsize = dict()
    ct_lines = dict(ciwc=[1, 5, 10, 15], cswc=[1, 5, 10, 15], q_ice=[1, 5, 10, 15], clwc=[1, 5, 10, 15],
                    iwc=[1, 5, 10, 15])
    linewidths = dict()
    robust = dict(iwc=False)
    cb_ticks = dict()
    vmaxs = dict()
    vmins = dict(iwp=0)
    xlabels = dict(v15="v15", v16="v16", v17="v17", v16_diff="Difference v15 - v16", v17_diff="Difference v15 - v17")

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

    # prepare ecrad dataset for plotting
    sf = h.scale_factors[var] if var in h.scale_factors else 1
    if "diff" in v:
        cmap = cmr.fusion
        norm = colors.TwoSlopeNorm(vcenter=0)
        # calculate difference between simulations
        ds = ecrad_dict[v[:3]].copy()
        ecrad_ds_diff = ecrad_dict[v_ref][var] - ds[var]
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
    # filter very low values
    ecrad_plot = ecrad_plot.where(np.abs(ecrad_plot) > 0)

    # select time height slice
    time_sel = case_slice if v == "v16" else sel_time
    if len(ecrad_plot.dims) > 2:
        dim3 = "band_sw"
        dim3 = dim3 if dim3 in ecrad_plot.dims else None
        ecrad_plot = ecrad_plot.sel({"time": time_sel, "height": slice(13, 0), f"{dim3}": band})
    else:
        ecrad_plot = ecrad_plot.sel(time=time_sel, height=slice(13, 0))

    time_extend = pd.to_timedelta((ecrad_plot.time[-1] - ecrad_plot.time[0]).to_numpy())

# %% plot 2D IFS variables along flight track
    _, ax = plt.subplots(figsize=h.figsize_wide)
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
    plt.tight_layout()
    figname = f"{plot_path}/{flight}_ecrad_{v}_{var}{band_str}_along_track.png"
    plt.savefig(figname, dpi=300, bbox_inches="tight")
    # figname = f"{fig_path}/{flight}_ecrad_{v}_{var}{band_str}_along_track.png"
    # plt.savefig(figname, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# %% plot histogram
    flat_array = ecrad_plot.to_numpy().flatten()
    _, ax = plt.subplots(figsize=h.figsize_wide)
    hist = ax.hist(flat_array, bins=20)
    ax.set(xlabel=f"{h.cbarlabels[var]} {xlabels[v]} ({h.plot_units[var]})",
           ylabel="Number of Occurrence")
    ax.grid()
    ax.set_yscale("log")
    plt.tight_layout()
    figname = f"{plot_path}/{flight}_ecrad_{v}_{var}{band_str}_hist.png"
    plt.savefig(figname, dpi=300, bbox_inches="tight")
    # figname = f"{fig_path}/{flight}_ecrad_{v}_{var}{band_str}_hist.png"
    # plt.savefig(figname, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# %% plot vertical integrated data
    ecrad_1d = ecrad_plot.sum(dim="height")
    _, ax = plt.subplots(figsize=h.figsize_wide)
    # ecrad 1D profile
    ecrad_1d.plot(x="time", ax=ax, ls="", marker="o")

    ax.set(title=f"{key} ecRad along Flight Track - {v}", ylabel=f"{h.cbarlabels[var]}{band_str}", xlabel="Time (UTC)",
           yscale="linear")
    h.set_xticks_and_xlabels(ax, time_extend)
    ax.grid()
    plt.tight_layout()
    # figname = f"{plot_path}/{flight}_ecrad_{v}_{var}{band_str}_integrated.png"
    # plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot histograms of variable
    var = "iwc"
    all_bins = dict(cre_sw=np.arange(-25, 1, 0.5), cre_lw=np.arange(0, 50), cre_total=np.arange(-2, 28),
                reflectivity_sw=np.arange(0.675, 0.85, 0.01),
                transmissivity_sw=np.arange(0.8, 1.01, 0.01), transmissivity_lw=np.arange(1, 1.51, 0.01),
                flux_dn_sw=np.arange(105, 270, 5), flux_up_sw=np.arange(85, 190, 3),
                flux_dn_lw=np.arange(25, 200, 5), flux_up_lw=np.arange(165, 245, 5),
                g_mean=np.arange(0.78, 0.9, 0.005), od_int=np.arange(0, 3, 0.05), scat_od_int=np.arange(0, 3, 0.05),
                    iwc=np.arange(0, 5, 0.25), re_ice=np.arange(10, 100, 2))
    bins = all_bins[var] if var in all_bins else None
    time_sel = above_slice
    plot1 = (ecrad_dict["v15"][var].sel(time=time_sel).to_numpy() * 1e6).flatten()
    plot2 = (ecrad_dict["v17"][var].sel(time=time_sel).to_numpy() * 1e6).flatten()
    t1, t2 = time_sel.start, time_sel.stop
    binsize = bins[1] - bins[0]
    _, ax = plt.subplots(figsize=h.figsize_wide)
    ax.hist(plot1, bins=bins, label="IFS", histtype="step", lw=3)
    ax.hist(plot2, bins=bins, label="VarCloud", histtype="step", lw=3)
    ax.legend()
    ax.text(0.8, 0.7, f"Binsize: {binsize} {h.plot_units[var]}", transform=ax.transAxes,
            bbox=dict(boxstyle="round", fc="white"))
    ax.grid()
    ax.set(xlabel=f"{h.cbarlabels[var]} ({h.plot_units[var]})", ylabel="Number of Occurrence",
           title=f"Histogram of {h.cbarlabels[var]} between {t1:%H:%M} and {t2:%H:%M} UTC")
    figname = f"{plot_path}/{flight}_ecrad_v15_v17_{var}_psd.png"
    plt.savefig(figname, dpi=300, bbox_inches="tight")
    # figname = f"{fig_path}/{flight}_ecrad_v1_v8_re_ice_psd.png"
    # plt.savefig(figname, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# %% plot BACARDI and ecrad along track
    ds10, ds1 = ecrad_dict["v16"], ecrad_dict["v15"]
    ds10_res = ds10.resample(time="1Min").first()
    time_sel = slice(ds10.time[0], ds10.time[-1])
    hl_sel = hlevel_da.sel(time=ds10.time, method="nearest").assign_coords(time=ds10.time)
    ecrad_plot = ds10["flux_dn_sw"].isel(half_level=hl_sel)
    ecrad_plot1 = ds1["flux_dn_sw"].isel(half_level=hlevel_da).sel(time=time_sel)
    ecrad_plot2 = ds10_res["flux_dn_sw"].isel(half_level=hlevel_da.sel(time=ds10_res.time))
    bacardi_plot = bacardi_ds["F_down_solar"].sel(time=time_sel)
    std_plot = bacardi_std["F_down_solar"].sel(time=time_sel).to_numpy()

    _, ax = plt.subplots(figsize=h.figsize_wide)
    ax.errorbar(bacardi_plot.time.to_numpy(), bacardi_plot, yerr=std_plot, label="$F_{\downarrow , solar}$ BACARDI")
    ax.plot(ecrad_plot.time, ecrad_plot, label="$F_{\downarrow , solar}$ ecRad Varcloud")
    ax.plot(ecrad_plot2.time, ecrad_plot2, marker="o", label="$F_{\downarrow , solar}$ ecRad Varcloud 1Min")
    ax.plot(ecrad_plot1.time, ecrad_plot1, marker="o", label="$F_{\downarrow , solar}$ ecRad")
    ax.legend()
    ax.grid()
    h.set_xticks_and_xlabels(ax, pd.to_timedelta((ds10.time[-1] - ds10.time[0]).to_numpy()))
    ax.set(xlabel="Time (UTC)", ylabel=f"Broadband Irradiance ({h.plot_units['flux_dn_sw']})",
           title=f"Below Cloud comparison of BACARDI and ecRad - {key}")
    plt.tight_layout()
    figname = f"{plot_path}/{flight}_ecrad_varcloud_BACARDI_F_down_solar_below_cloud.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot histogram comparison
    x, y = bacardi_plot.to_numpy().flatten(), ecrad_plot.to_numpy().flatten()

    _, ax = plt.subplots(figsize=h.figsize_wide)
    ax.hist([x, y], bins=20, histtype="step", lw=2,
            label=[r"$F_{\downarrow , solar}$ BACARDI", r"$F_{\downarrow , solar}$ ecRad Varcloud"])
    ax.set(xlabel=f"{h.cbarlabels['flux_dn_sw']} ({h.plot_units['flux_dn_sw']})",
           ylabel="Number of Occurrence",
           title=f"Histogram of below cloud measurements/simulation along flight track\n"
                 f"{key} - {pd.to_datetime(ecrad_plot.time[0].to_numpy()):%H:%M} "
                 f"- {pd.to_datetime(ecrad_plot.time[-1].to_numpy()):%H:%M} UTC")
    ax.legend()

    plt.tight_layout()
    figname = f"{plot_path}/{flight}_ecrad_BACARDI_F_down_solar_histogram_below_cloud.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot histogram of differences
    xy = (bacardi_plot - ecrad_plot).to_numpy().flatten()
    _, ax = plt.subplots(figsize=h.figsize_wide)
    ax.hist(xy, bins=30, histtype="step", lw=2)
    ax.set(xlabel=f"{h.cbarlabels['flux_dn_sw']} ({h.plot_units['flux_dn_sw']})",
           ylabel="Number of Occurrence",
           title=f"Histogram of Difference between measurements and simulation along flight track\n"
                 f"{key} - {pd.to_datetime(ecrad_plot.time[0].to_numpy()):%H:%M} "
                 f"- {pd.to_datetime(ecrad_plot.time[-1].to_numpy()):%H:%M} UTC")

    plt.tight_layout()
    figname = f"{plot_path}/{flight}_BACARDI-ecrad_F_down_solar_histogram_below_cloud.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot scatterplot of below cloud measurements
    labels = dict(F_down_solar=r"$F_{\downarrow, solar}$", F_down_terrestrial=r"$F_{\downarrow, terrestrial}$",
                  F_up_solar=r"$F_{\uparrow, solar}$", F_up_terrestrial=r"$F_{\uparrow, terrestrial}$")

    # prepare metadata for comparing ecRad and BACARDI
    titles = ["Solar Downward Irradiance", "Terrestrial Downward Irradiance", "Solar Upward Irradiance",
              "Terrestrial Upward Irradiance"]
    names = ["Fdw_solar", "Fdw_terrestrial", "Fup_solar", "Fup_terrestrial"]
    bacardi_vars = ["F_down_solar", "F_down_terrestrial", "F_up_solar", "F_up_terrestrial"]
    ecrad_vars = ["flux_dn_sw", "flux_dn_lw", "flux_up_sw", "flux_up_lw"]
    ds10 = ecrad_dict["v16"]
    time_sel = slice(ds10.time[100], ds10.time[-1])
    hl_sel = hlevel_da.sel(time=ds10.time, method="nearest").assign_coords(time=ds10.time)
    bacardi_plot = bacardi_ds.sel(time=time_sel)
    ecrad_plot = ds10.isel(half_level=hl_sel).sel(time=time_sel)
    lims = [(150, 240), (80, 110), (110, 160), (210, 220)]
    for (i, x), y in zip(enumerate(bacardi_vars), ecrad_vars):
        rmse = np.sqrt(np.mean((ecrad_plot[y] - bacardi_plot[x]) ** 2))
        bias = np.mean((ecrad_plot[y] - bacardi_plot[x]))
        _, ax = plt.subplots(figsize=h.figsize_equal)
        ax.scatter(bacardi_plot[x], ecrad_plot[y], c=cbc[3])
        ax.axline((0, 0), slope=1, color="k", lw=2, transform=ax.transAxes)
        ax.set_ylim(lims[i])
        ax.set_xlim(lims[i])
        ticks = ax.get_yticks() if i == 0 else ax.get_xticks()
        ax.set_yticks(ticks)
        ax.set_xticks(ticks)
        ax.set_aspect('equal')
        ax.set_xlabel("BACARDI Irradiance (W$\,$m$^{-2}$)")
        ax.set_ylabel("ecRad Irradiance (W$\,$m$^{-2}$)")
        ax.set_title(f"{titles[i]}\nbelow cloud")
        ax.grid()
        ax.text(0.025, 0.95, f"# points: {sum(~np.isnan(bacardi_plot[x])):.0f}\n"
                             f"RMSE: {rmse.values:.2f}" + " W$\,$m$^{-2}$\n"
                                                          f"Bias: {bias.values:.2f}" + " W$\,$m$^{-2}$",
                ha='left', va='top', transform=ax.transAxes,
                bbox=dict(fc='white', ec='black', alpha=0.8, boxstyle='round'))
        plt.tight_layout()
        plt.savefig(f"{plot_path}/{flight}_{names[i]}_bacardi_vs_ecrad_varcloud_scatter_below_cloud.png", dpi=300)
        plt.show()
        plt.close()

# %% plot scatterplot of above cloud measurements whole flight
    labels = dict(F_down_solar=r"$F_{\downarrow, solar}$", F_down_terrestrial=r"$F_{\downarrow, terrestrial}$",
                  F_up_solar=r"$F_{\uparrow, solar}$", F_up_terrestrial=r"$F_{\uparrow, terrestrial}$")
    # prepare metadata for comparing ecRad and BACARDI
    titles = ["Solar Downward Irradiance", "Terrestrial Downward Irradiance", "Solar Upward Irradiance",
              "Terrestrial Upward Irradiance"]
    names = ["Fdw_solar", "Fdw_terrestrial", "Fup_solar", "Fup_terrestrial"]
    bacardi_vars = ["F_down_solar", "F_down_terrestrial", "F_up_solar", "F_up_terrestrial"]
    ecrad_vars = ["flux_dn_sw", "flux_dn_lw", "flux_up_sw", "flux_up_lw"]

    ecrad_ds = ecrad_dict["v15"].where(above_sel)
    bacardi_plot = bacardi_ds_res.where(above_sel)
    ecrad_plot = ecrad_ds.isel(half_level=hlevel_da)
    lims = [(200, 600), (25, 35), (150, 200), (175, 195)]
    for (i, x), y in zip(enumerate(bacardi_vars), ecrad_vars):
        rmse = np.sqrt(np.mean((ecrad_plot[y] - bacardi_plot[x]) ** 2))
        bias = np.mean((ecrad_plot[y] - bacardi_plot[x]))
        _, ax = plt.subplots(figsize=h.figsize_equal)
        ax.scatter(bacardi_plot[x], ecrad_plot[y], c=bacardi_plot.time)
        ax.axline((0, 0), slope=1, color="k", lw=2, transform=ax.transAxes)
        ax.set_ylim(lims[i])
        ax.set_xlim(lims[i])
        ticks = ax.get_yticks() if i == 0 else ax.get_xticks()
        ax.set_yticks(ticks)
        ax.set_xticks(ticks)
        ax.set_aspect('equal')
        ax.set_xlabel("BACARDI Irradiance (W$\,$m$^{-2}$)")
        ax.set_ylabel("ecRad Irradiance (W$\,$m$^{-2}$)")
        ax.set_title(f"{titles[i]}\nabove cloud")
        ax.grid()
        ax.text(0.025, 0.95, f"# points: {sum(~np.isnan(bacardi_plot[x])):.0f}\n"
                             f"RMSE: {rmse.values:.2f}" + " W$\,$m$^{-2}$\n"
                                                          f"Bias: {bias.values:.2f}" + " W$\,$m$^{-2}$",
                ha='left', va='top', transform=ax.transAxes,
                bbox=dict(fc='white', ec='black', alpha=0.8, boxstyle='round'))
        plt.tight_layout()
        # plt.savefig(f"{plot_path}/{flight}_{names[i]}_bacardi_vs_ecrad_varcloud_scatter_above_cloud.png", dpi=300)
        plt.show()
        plt.close()

# %% plot BACARDI and ecRad at flight altitude along track
    var = "flux_dn_sw"
    bacardi_plot = bacardi_ds_res[var_dict[var]]

    _, ax = plt.subplots(figsize=h.figsize_wide)

    ax.plot(bacardi_plot.time, bacardi_plot, label="BACARDI")

    for v in ["v15", "v17", "v18"]:
        ecrad_plot = ecrad_dict[v][var].isel(half_level=hlevel_da.sel(time=ecrad_dict[v].time))
        ax.plot(ecrad_plot.time, ecrad_plot, label=version_labels[v])

    ax.set(xlabel="Time (UTC)", ylabel=f"Broadband irradiance ({h.plot_units[var]})")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()
    plt.close()

# %% plot IFS variables

