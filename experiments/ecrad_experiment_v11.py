#!/usr/bin/env python
"""
| *author:* Johannes Röttenbacher
| *created:* 05-04-2023

Analyze the impact of using only Cloud Ice Water Content (ciwc) as ice mass mixing ratio (|q-ice|) instead of summing up  ciwc and Cloud Snow Water Content (cswc) for the |q-ice|.

* ``IFS_namelist_jr_20220411_v1.nam``: for flight RF17 with Fu-IFS ice model
* ``IFS_namelist_jr_20220411_v11.nam``: for flight RF17 with Fu-IFS ice model and ciwc as q_ice

**Problem statement:** Using the Varcloud retrieval as input for IWC (converted to |q-ice|) seemed to explain the overestimation of optical depth in the IFS.
However, RF18 did not show the same bias in downward solar irradiance below the cloud as RF17.
One possible reason to explain this inconsistency could be the differing cswc between the two flights which is not reflected in the Varcloud retrieval.
The cswc also extends the clouds to lower altitudes.
Thus, we investigate the impact of removing it from the simulation.

Focus is on the case study in the high north.
First let's look at the difference in |q-ice| for v1 where :math:`q_{ice} = ciwc + cswc` and v11 where :math:`q_{ice} = ciwc`.

.. figure:: figures/experiment_v11/HALO-AC3_20220411_HALO_RF17_ecrad_v1_q_ice_along_track.png

    Ice mass mixing ration for v1.

.. figure:: figures/experiment_v11/HALO-AC3_20220411_HALO_RF17_ecrad_v11_q_ice_along_track.png

    Ice mass mixing ration for v11.

The interesting question is now if ecRad also sees a cloud with the extent of |q-ice| in v1.
It could well be that it doesn't because there is no |re-ice| simulated there.
Or to be more precise only a fill value (5.196162e-05) is set.

.. figure:: figures/experiment_v11/HALO-AC3_20220411_HALO_RF17_ecrad_v1_re_ice_along_track.png

    Ice effective radius for v1. White areas are filled with a fill value for the simulation.

Let's take a look at the difference in downward solar irradiance for the case study period (v1 - v11).

.. _fluxdnsw-diff:

.. figure:: figures/experiment_v11/HALO-AC3_20220411_HALO_RF17_ecrad_diff_flux_dn_sw_along_track.png

    Difference in downward solar irradiance between v1 and v11.

The difference here seems to come from the |q-ice| difference in the core of the cloud (see :numref:`qice-diff`).

.. _qice-diff:

.. figure:: figures/experiment_v11/HALO-AC3_20220411_HALO_RF17_ecrad_diff_q_ice_along_track.png

    Difference in |q-ice| between v1 and v11.

So although the cswc increases the size of the cloud it does not affect the simulation outside the cloud since there is no |re-ice| simulated there.
It, thus, only increases |q-ice| inside the actual cloud which leads to higher optical depth of the cloud and thus more scattering/absorption.
This can be seen in the reduced downward solar irradiance below the cloud in :numref:`fluxdnsw-diff`.

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
    from tqdm import tqdm
    import cmasher as cmr

    cm = 1 / 2.54
    cb_colors = h.get_cb_friendly_colors()
    h.set_cb_friendly_colors()
    plt.rc("font", size=12)

# %% set paths
    campaign = "halo-ac3"
    key = "RF18"
    flight = meta.flight_names[key]
    date = flight[9:17]

    plot_path = f"{h.get_path('plot', flight, campaign)}/{flight}/experiment_v11"
    fig_path = "./docs/figures/experiment_v11"
    h.make_dir(plot_path)
    ecrad_path = f"{h.get_path('ecrad', campaign=campaign)}/{date}/"
    ifs_path = f"{h.get_path('ifs', campaign=campaign)}/{date}"
    ecrad_input = f"ecrad_merged_input_{date}_v4.nc"
    ecrad_v1 = f"ecrad_merged_inout_{date}_v1_mean.nc"
    ecrad_v11 = f"ecrad_merged_output_{date}_v11_mean.nc"

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
    ecrad_ds_input = xr.open_dataset(f"{ecrad_path}/{ecrad_input}")
    ecrad_ds_v1 = xr.open_dataset(f"{ecrad_path}/{ecrad_v1}")
    ecrad_ds_v1 = ecrad.calculate_pressure_height(ecrad_ds_v1)
    ecrad_ds_v11 = xr.open_dataset(f"{ecrad_path}/{ecrad_v11}")
    ecrad_dict = dict(v1=ecrad_ds_v1, v11=ecrad_ds_v11)

# %% modify data sets
    for k in ecrad_dict:
        ds = ecrad_dict[k].copy()
        # replace default values with nan
        ds["re_ice"] = ds.re_ice.where(ds.re_ice != 5.196162e-05, np.nan)
        ds["re_liquid"] = ds.re_liquid.where(ds.re_liquid != 4.000001e-06, np.nan)

        # compute cloud ice water path
        factor = ds.pressure_hl.diff(dim="half_level").to_numpy() / (9.80665 * ds.cloud_fraction.to_numpy())
        ds["iwp"] = (["time", "level"], factor * ds.ciwc.to_numpy())
        ds["iwp"] = ds.iwp.where(ds.iwp != np.inf, np.nan)

        # convert kg/kg to kg/m³
        air_density = density(ds.pressure_full * units.Pa, ds.t * units.K, ds.q * units("kg/kg"))
        ds["iwc"] = ds["q_ice"] * units("kg/kg") * air_density

        # add optical properties to data sets
        ice_optics_fu = ecrad.calc_ice_optics_fu_sw(ds["iwp"], ds.re_ice)
        ds["od"] = ice_optics_fu[0]
        ds["scat_od"] = ice_optics_fu[1]
        ds["g"] = ice_optics_fu[2]
        ds["band_sw"] = range(1, 15)
        ds["band_lw"] = range(1, 17)
        ds["absorption"] = ds["od"] - ds["scat_od"]
        ds["od_mean"] = ds["od"].mean(dim="band_sw")
        ds["scat_od_mean"] = ds["scat_od"].mean(dim="band_sw")
        ds["g_mean"] = ds["g"].mean(dim="band_sw")
        ds["scat_od_int"] = ds["scat_od"].integrate(coord="band_sw")
        ds["od_int"] = ds["od"].integrate(coord="band_sw")
        ds["absorption_int"] = ds["absorption"].integrate(coord="band_sw")

        # calculate other optical parameters
        ds["reflectivity_sw"] = ds.flux_up_sw / ds.flux_dn_sw

        ecrad_dict[k] = ds.copy()

# %% set plotting options
    var = "q_ice"
    v = "v11"
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
    xlabels = dict(v1="v1", v11="v11", diff="Difference v1 - v11")

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
    if v == "diff":
        # calculate difference between simulations
        ds = ecrad_dict["v11"]
        ecrad_ds_diff = ecrad_dict["v1"][var] - ds[var]
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
    xlabel = "Difference v1 - v11" if v == "diff" else v
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

    # %% plot histogram of particle size
    time_sel = above_slice
    plot_v1 = (ecrad_dict["v1"].re_ice.sel(time=time_sel).to_numpy() * 1e6).flatten()
    plot_v11 = (ecrad_dict["v11"].re_ice.sel(time=time_sel).to_numpy() * 1e6).flatten()
    t1, t2 = time_sel.start, time_sel.stop
    binsize = 2
    bins = np.arange(10, 110, binsize)
    _, ax = plt.subplots(figsize=h.figsize_wide)
    hist = ax.hist(plot_v1, bins=bins, label="IFS", histtype="step", lw=3)
    ax.hist(plot_v11, bins=bins, label="VarCloud", histtype="step", lw=3)
    ax.legend()
    ax.text(0.8, 0.7, f"Binsize: {binsize} $\mu$m", transform=ax.transAxes, bbox=dict(boxstyle="round", fc="white"))
    ax.grid()
    ax.set(xlabel=r"Ice effective radius ($\mu$m)", ylabel="Number of Occurrence",
           title=f"Histogram of Particle Size between {t1:%H:%M} and {t2:%H:%M} UTC")
    figname = f"{plot_path}/{flight}_ecrad_v1_v11_re_ice_histogram.png"
    plt.savefig(figname, dpi=300, bbox_inches="tight")
    figname = f"{fig_path}/{flight}_ecrad_v1_v8_re_ice_psd.png"
    plt.savefig(figname, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
