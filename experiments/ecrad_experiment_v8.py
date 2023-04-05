#!/usr/bin/env python
"""
| *author:* Johannes Röttenbacher
| *created:* 24-03-2023

Comparison between using the VarCloud retrieval for *IWC* and |re-ice| or the forecasted values from the IFS as input for an ecRad simulation using the Fu-IFS ice optic parameterization.

* ``IFS_namelist_jr_20220411_v1.nam``: for flight HALO-AC3_20220411_HALO_RF17 with Fu-IFS ice model
* ``IFS_namelist_jr_20220411_v8.nam``: for flight HALO-AC3_20220411_HALO_RF17 with Fu-IFS ice model and VarCloud input

Focus is on the above cloud section in the high north before the below cloud section where no VarCloud retrieval is available anymore.

.. figure:: figures/experiment_v8/HALO-AC3_20220411_HALO_RF17_ecrad_v8_iwc_along_track.png

    Retrieved ice water content from VarCloud retrieval interpolated to IFS full level pressure altitudes.

Results
^^^^^^^^

.. figure:: figures/experiment_v8/HALO-AC3_20220411_HALO_RF17_ecrad_diff_iwc_along_track.png

    Difference between IWC IFS and IWC VarCloud.

.. figure:: figures/experiment_v8/HALO-AC3_20220411_HALO_RF17_ecrad_diff_re_ice_along_track.png

    Difference between |re-ice| IFS and |re-ice| VarCloud.

From the two figures above it can be seen that the VarCloud retrieval produces a higher IWC and larger |re-ice|.
The histograms show the same picture.

.. figure:: figures/experiment_v8/HALO-AC3_20220411_HALO_RF17_ecrad_diff_iwc_hist.png

    Histogramm of difference between IWC IFS and IWC VarCloud.

.. figure:: figures/experiment_v8/HALO-AC3_20220411_HALO_RF17_ecrad_v1_v8_re_ice_psd.png

    Histogramm of |re-ice| used in the IFS run and the VarCloud run.

The differences in microphysical properties also translate to differences in optical properties of the cloud such as higher total optical depth (integrated over all bands) in the IFS run.

.. figure:: figures/experiment_v8/HALO-AC3_20220411_HALO_RF17_ecrad_diff_od_int_along_track.png

    Difference in total optical depth integrated over all shortwave bands between the IFS run and the VarCloud run.

Finally, due to larger |re-ice| the ice cloud from the retrieval is less absorbing/reflecting than the IFS forecasted ice cloud.

.. figure:: figures/experiment_v8/HALO-AC3_20220411_HALO_RF17_ecrad_diff_flux_dn_sw_along_track.png

    Difference in broadband solar downward irradiance between IFS and VarCloud.

.. figure:: figures/experiment_v8/HALO-AC3_20220411_HALO_RF17_ecrad_diff_flux_dn_sw_hist.png

    Histogramm of difference in broadband solar downward irradiance between IFS and VarCloud.

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
    from metpy.calc import density
    from metpy.units import units
    import matplotlib.pyplot as plt
    from matplotlib import colors
    import pandas as pd
    import os
    from tqdm import tqdm
    import cmasher as cmr

    cm = 1 / 2.54
    cb_colors = h.get_cb_friendly_colors()
    # %% plotting variables
    h.set_cb_friendly_colors()
    plt.rc("font", size=12)

    # %% set paths
    campaign = "halo-ac3"
    key = "RF17"
    flight = meta.flight_names[key]
    date = flight[9:17]

    plot_path = f"{h.get_path('plot', flight, campaign)}/{flight}/experiment_v8"
    fig_path = "./docs/figures/experiment_v8"
    h.make_dir(plot_path)
    ecrad_path = f"{h.get_path('ecrad', campaign=campaign)}/{date}/"
    ifs_path = f"{h.get_path('ifs', campaign=campaign)}/{date}"
    varcloud_path = h.get_path("varcloud", flight, campaign)
    libradtran_path = h.get_path("libradtran_exp", flight, campaign)
    ecrad_input = f"ecrad_merged_input_{date}_v2.nc"
    ecrad_v8 = f"ecrad_merged_output_{date}_v8.nc"
    ecrad_v1 = f"ecrad_merged_inout_{date}_v1.nc"
    varcloud_file = [f for f in os.listdir(varcloud_path) if "nc" in f][0]
    libradtran_file = f"HALO-AC3_HALO_libRadtran_simulation_varcloud_1min_{date}_{key}.nc"

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

    # %% read in varcloud data
    varcloud = xr.open_dataset(f"{varcloud_path}/{varcloud_file}").swap_dims(time="Time", height="Height").rename(
        Time="time")
    varcloud = varcloud.rename(Varcloud_Cloud_Ice_Water_Content="iwc")
    libradtran_ds = xr.open_dataset(f"{libradtran_path}/{libradtran_file}")  # read in libradtran data
    libradtran_ds["altitude"] = libradtran_ds.altitude / 1000
    libradtran_ds = libradtran_ds.rename(CIWD="iwc")

    # %% read in ecrad data
    ecrad_ds_input = xr.open_dataset(f"{ecrad_path}/{ecrad_input}")
    ecrad_ds_v8 = xr.open_dataset(f"{ecrad_path}/{ecrad_v8}")
    ecrad_ds_v1 = xr.open_dataset(f"{ecrad_path}/{ecrad_v1}")
    # select only center column from version 1, version 8 has only one column
    ecrad_ds_v1 = ecrad_ds_v1.sel(column=16)
    # merge input und v8
    ecrad_ds_v8 = xr.merge([ecrad_ds_v8, ecrad_ds_input])
    # replace default values with nan
    ecrad_ds_v1["re_ice"] = ecrad_ds_v1.re_ice.where(ecrad_ds_v1.re_ice != 5.19616e-05, np.nan)
    ecrad_ds_v1["re_liquid"] = ecrad_ds_v1.re_liquid.where(ecrad_ds_v1.re_liquid != 4.000001e-06, np.nan)
    ecrad_ds_v8["re_ice"] = ecrad_ds_v8.re_ice.where(ecrad_ds_v8.re_ice != 5.19616e-05, np.nan)
    ecrad_ds_v8["re_liquid"] = ecrad_ds_v8.re_liquid.where(ecrad_ds_v8.re_liquid != 4.000001e-06, np.nan)

    # select time range of v8
    ecrad_ds_v1 = ecrad_ds_v1.sel(time=ecrad_ds_v8.time)

    # %% compute cloud ice water path
    factor = ecrad_ds_v1.pressure_hl.diff(dim="half_level").to_numpy() / (
            9.80665 * ecrad_ds_v1.cloud_fraction.to_numpy())
    ecrad_ds_v1["iwp"] = (["time", "level"], factor * ecrad_ds_v1.ciwc.to_numpy())
    ecrad_ds_v1["iwp"] = ecrad_ds_v1.iwp.where(ecrad_ds_v1.iwp != np.inf, np.nan)
    factor = ecrad_ds_v8.pressure_hl.diff(dim="half_level").to_numpy() / (
            9.80665 * ecrad_ds_v8.cloud_fraction.to_numpy())
    ecrad_ds_v8["iwp"] = (["time", "level"], factor * ecrad_ds_v8.ciwc.to_numpy())
    ecrad_ds_v8["iwp"] = ecrad_ds_v8.iwp.where(ecrad_ds_v8.iwp != np.inf, np.nan)

    # %% convert kg/kg to kg/m³
    air_density = density(ecrad_ds_v1.pressure_full * units.Pa, ecrad_ds_v1.t * units.K, ecrad_ds_v1.q * units("kg/kg"))
    ecrad_ds_v1["iwc"] = ecrad_ds_v1["q_ice"] * units("kg/kg") * air_density

    air_density = density(ecrad_ds_v8.pressure_full * units.Pa, ecrad_ds_v8.t * units.K, ecrad_ds_v8.q * units("kg/kg"))
    ecrad_ds_v8["iwc"] = ecrad_ds_v8["q_ice"] * units("kg/kg") * air_density

    # %% add optical properties to data sets
    ice_optics_fu = ecrad.calc_ice_optics_fu_sw(ecrad_ds_v1["iwp"], ecrad_ds_v1.re_ice)
    ecrad_ds_v1["od"] = ice_optics_fu[0]
    ecrad_ds_v1["scat_od"] = ice_optics_fu[1]
    ecrad_ds_v1["g"] = ice_optics_fu[2]
    ecrad_ds_v1["band_sw"] = range(1, 15)
    ecrad_ds_v1["band_lw"] = range(1, 17)
    ecrad_ds_v1["absorption"] = ecrad_ds_v1["od"] - ecrad_ds_v1["scat_od"]
    ecrad_ds_v1["od_mean"] = ecrad_ds_v1["od"].mean(dim="band_sw")
    ecrad_ds_v1["scat_od_mean"] = ecrad_ds_v1["scat_od"].mean(dim="band_sw")
    ecrad_ds_v1["g_mean"] = ecrad_ds_v1["g"].mean(dim="band_sw")
    ecrad_ds_v1["scat_od_int"] = ecrad_ds_v1["scat_od"].integrate(coord="band_sw")
    ecrad_ds_v1["od_int"] = ecrad_ds_v1["od"].integrate(coord="band_sw")
    ecrad_ds_v1["absorption_int"] = ecrad_ds_v1["absorption"].integrate(coord="band_sw")
    # version8
    ice_optics_fu = ecrad.calc_ice_optics_fu_sw(ecrad_ds_v8["iwp"], ecrad_ds_v8.re_ice)
    ecrad_ds_v8["od"] = ice_optics_fu[0]
    ecrad_ds_v8["scat_od"] = ice_optics_fu[1]
    ecrad_ds_v8["g"] = ice_optics_fu[2]
    ecrad_ds_v8["band_sw"] = range(1, 15)
    ecrad_ds_v8["band_lw"] = range(1, 17)
    ecrad_ds_v8["absorption"] = ecrad_ds_v8["od"] - ecrad_ds_v8["scat_od"]
    ecrad_ds_v8["od_mean"] = ecrad_ds_v8["od"].mean(dim="band_sw")
    ecrad_ds_v8["scat_od_mean"] = ecrad_ds_v8["scat_od"].mean(dim="band_sw")
    ecrad_ds_v8["g_mean"] = ecrad_ds_v8["g"].mean(dim="band_sw")
    ecrad_ds_v8["scat_od_int"] = ecrad_ds_v8["scat_od"].integrate(coord="band_sw")
    ecrad_ds_v8["od_int"] = ecrad_ds_v8["od"].integrate(coord="band_sw")
    ecrad_ds_v8["absorption_int"] = ecrad_ds_v8["absorption"].integrate(coord="band_sw")

    # %% calculate other optical parameters
    ecrad_ds_v1["reflectivity_sw"] = ecrad_ds_v1.flux_up_sw / ecrad_ds_v1.flux_dn_sw
    ecrad_ds_v8["reflectivity_sw"] = ecrad_ds_v8.flux_up_sw / ecrad_ds_v8.flux_dn_sw
    libradtran_ds["eglo_int"] = libradtran_ds["eglo"].integrate(coord="wavelength")
    libradtran_ds["eup_int"] = libradtran_ds["eup"].integrate(coord="wavelength")
    ecrad_ds_v8["flux_dn_sw_int"] = ecrad_ds_v8["spectral_flux_dn_sw"].sel(band_sw=slice(3, 11)).integrate(
        coord="band_sw")

    # %% set plotting options
    var = "od_int"
    v = "v1"
    band = None
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

    if v == "diff":
        cmap = cmr.fusion
        norm = colors.TwoSlopeNorm(vcenter=0)

    # prepare ecrad dataset for plotting
    sf = h.scale_factors[var] if var in h.scale_factors else 1
    if v == "v8":
        ecrad_plot = ecrad_ds_v8[var] * sf
    elif v == "v1":
        ecrad_plot = ecrad_ds_v1[var] * sf
    else:
        # calculate difference between simulations
        ecrad_ds_diff = ecrad_ds_v1[var] - ecrad_ds_v8[var]
        ecrad_plot = ecrad_ds_diff.where((ecrad_ds_v8[var] != 0) | (~np.isnan(ecrad_ds_v8[var]))) * sf

    # add new z axis mean pressure altitude
    if "half_level" in ecrad_plot.dims:
        new_z = ecrad_ds_v8["press_height_hl"].mean(dim="time") / 1000
    else:
        new_z = ecrad_ds_v8["press_height_full"].mean(dim="time") / 1000

    ecrad_plot_new_z = list()
    for t in tqdm(ecrad_plot.time, desc="New Z-Axis"):
        tmp_plot = ecrad_plot.sel(time=t)
        if "half_level" in tmp_plot.dims:
            tmp_plot = tmp_plot.assign_coords(
                half_level=ecrad_ds_v8["press_height_hl"].sel(time=t, drop=True).to_numpy() / 1000)
            tmp_plot = tmp_plot.rename(half_level="height")

        else:
            tmp_plot = tmp_plot.assign_coords(
                level=ecrad_ds_v8["press_height_full"].sel(time=t, drop=True).to_numpy() / 1000)
            tmp_plot = tmp_plot.rename(level="height")

        tmp_plot = tmp_plot.interp(height=new_z.to_numpy())
        ecrad_plot_new_z.append(tmp_plot)

    ecrad_plot = xr.concat(ecrad_plot_new_z, dim="time")
    # filter very low to_numpy()
    ecrad_plot = ecrad_plot.where(np.abs(ecrad_plot) > 0.001)

    # select time height slice
    time_sel = above_slice
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
    xlabel = "Difference v1 - v8" if v == "diff" else v
    flat_array = ecrad_plot.to_numpy().flatten()
    _, ax = plt.subplots(figsize=h.figsize_wide)
    hist = ax.hist(flat_array, bins=20)
    ax.set(xlabel=f"{h.cbarlabels[var]} {xlabel} ({h.plot_units[var]})",
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

    # %% plot varcloud data
    v = "iwc"
    var_plot = varcloud[v].sel(time=above_slice) * 1e6
    var_plot.plot(x="time", cmap=h.cmaps[v], figsize=h.figsize_wide)
    plt.ylabel("Altitude (m)")
    plt.show()
    plt.close()

    # %% calculate histogram of particle size
    time_sel = above_slice
    plot_v1 = (ecrad_ds_v1.re_ice.sel(time=time_sel).to_numpy() * 1e6).flatten()
    plot_v8 = (ecrad_ds_v8.re_ice.sel(time=time_sel).to_numpy() * 1e6).flatten()
    t1, t2 = time_sel.start, time_sel.stop
    binsize = 2
    bins = np.arange(10, 110, binsize)
    _, ax = plt.subplots(figsize=h.figsize_wide)
    hist = ax.hist(plot_v1, bins=bins, label="IFS", histtype="step", lw=3)
    ax.hist(plot_v8, bins=bins, label="VarCloud", histtype="step", lw=3)
    ax.legend()
    ax.text(0.8, 0.7, f"Binsize: {binsize} $\mu$m", transform=ax.transAxes, bbox=dict(boxstyle="round", fc="white"))
    ax.grid()
    ax.set(xlabel=r"Ice effective radius ($\mu$m)", ylabel="Number of Occurrence",
           title=f"Histogram of Particle Size between {t1:%H:%M} and {t2:%H:%M} UTC")
    figname = f"{plot_path}/{flight}_ecrad_v1_v8_re_ice_psd.png"
    plt.savefig(figname, dpi=300, bbox_inches="tight")
    figname = f"{fig_path}/{flight}_ecrad_v1_v8_re_ice_psd.png"
    plt.savefig(figname, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # %% calculate factor difference between IWC IFS and VarCloud
    v8, v1 = ecrad_ds_v8.sel(time=above_slice), ecrad_ds_v1.sel(time=above_slice)
    factor = (v1.iwc / v8.iwc).where(~np.isnan(v8.iwc) & (v8.iwc != 0)).sel(level=slice(75, 138))
    # factor = factor / np.nanmax(factor)
    factor.plot(x="time", robust=True)
    plt.show()
    plt.close()

    # %% plot libradtran simulation
    var = "iwc"
    cmap = h.cmaps[var] if var in h.cmaps else cmr.rainforest
    plot_ds = libradtran_ds[var] * 1e3
    plot_ds = plot_ds.where(plot_ds != 0)
    te = pd.to_timedelta((plot_ds.time[-1] - plot_ds.time[0]).to_numpy())

    _, ax = plt.subplots(figsize=h.figsize_wide)
    plot_ds.plot(x="time", cmap=cmap, ax=ax,
                 cbar_kwargs={"pad": 0.04, "label": f"{h.cbarlabels[var]} ({h.plot_units[var]})"})
    ax.set(xlabel="Time (UTC)", ylabel="Altitude (km)", title="libRadtran varcloud simulation")
    h.set_xticks_and_xlabels(ax, te)
    plt.tight_layout()

    figname = f"{plot_path}/{flight}_libradtran_{var}_along_track.png"
    plt.savefig(figname, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # %% interpolate libradtran to ecrad_plot height axis
    libradtran_inp = libradtran_ds.interp(altitude=np.flip(ecrad_plot.height))

    # %% calculate difference between ecrad and libradtran
    diff_plot = ecrad_plot - libradtran_inp["eglo_int"]

    # %% plot difference between ecRad and libradtran along track
    _, ax = plt.subplots(figsize=h.figsize_wide)
    diff_plot.plot(x="time", ax=ax,
                   cbar_kwargs={"label": "Solar Downward Irradiance (W$\,$m$^{-2}$)"})
    ax.set(xlabel="Time (UTC)", ylabel="Altitude (km)",
           title="Difference between ecRad varcloud and libRadtran varcloud")
    h.set_xticks_and_xlabels(ax, te)
    plt.tight_layout()
    plt.show()
    plt.close()

    # %% plot histogramm of difference between ecrad and libradtran
    _, ax = plt.subplots(figsize=h.figsize_wide)
    plot_array = diff_plot.to_numpy().flatten()
    ax.hist(plot_array, bins=20)
    ax.set(xlabel=f"Difference in Solar Downward Irradiance ({h.plot_units['flux_dn_sw']})",
           ylabel="Number of Occurrence",
           title="Histogram of Difference between ecRad varcloud and libRadtran varcloud")
    plt.show()
    plt.close()
