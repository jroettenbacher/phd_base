#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 13.10.2023

Compare standard simulation (input v6, namelist v15) before and after implementing the CAMS trace gas climatology.

Input files:

- ``ecrad_merged_inout_20220411_v15_old.nc`` -> fixed trace gases, ozone sonde
- ``ecrad_merged_inout_20220411_v15.nc`` -> CAMS climatology trace gases

First, let's compare the mean profiles of trace gases in the case study region.

.. figure:: ./figures/trace_gases/profile_comparison.png

    Mean profiles of trace gases in the case study region. Fixed values are replicated over height.

Apart from O3 and CO2 all trace gases decrease with altitude in the CAMS climatology whereas the constant values are also high in the upper atmosphere.
For CO2 there is not much difference apart from the values at ground.
O3 does show a difference in the upper atomsphere where there are no more sonde measurements and the values are set to 0 in the old run.
In the region where the sonde is available the climatology shows a good match.
It would thus be better to use the climatology as it extends the O3 profile through the whole atmosphere.
A mixture of both could also be possible.

Let's look at the impact of the differet sources of trace gases on the radiative fluxes.
We will look at the difference of new - old, so v15 - v15_old.

.. figure:: ./figures/trace_gases/HALO-AC3_20220411_HALO_RF17_ecrad_diff_flux_dn_sw_along_track.png

    Difference in solar downward irradiance between the simulations using the CAMS climatology (v15) and the ones using the fixed values (v15_old).

Especially above cloud we see a substantial difference of up to :math:`-12\,Wm^{-2}`.

.. figure:: ./figures/trace_gases/HALO-AC3_20220411_HALO_RF17_ecrad_diff_flux_dn_sw_hist.png

    Hisotgram of differences of solar downward irradiance between the simulations using the CAMS climatology (v15) and the ones using the fixed values (v15_old).

Looking at the histogram of differences we see a mean of :math:`-10\,Wm^{-2}` and a minumum at :math:`-7\,Wm^{-2}`.
From this we can conclude that more realistic profiles of trace gases in the simulations lead to less solar downward irradiance.
This is probably due to more scattering in the upper atmosphere.
The main cause for this is probably the ozone (O3) profile.
More sensitivity studies are needed to pinpoint this, however.


"""
if __name__ == "__main__":
# %% import modules
    import pylim.helpers as h
    import pylim.halo_ac3 as meta
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import colors
    import pandas as pd
    from tqdm import tqdm
    import cmasher as cmr

    cbc = h.get_cb_friendly_colors()
    h.set_cb_friendly_colors()
    plt.rc("font", size=12)

# %% set paths
    campaign = "halo-ac3"
    key = "RF17"
    flight = meta.flight_names[key]
    date = flight[9:17]

    plot_path = f"{h.get_path('plot', flight, campaign)}/{flight}/trace_gases"
    fig_path = "./docs/figures/trace_gases"
    h.make_dir(plot_path)
    h.make_dir(fig_path)
    ecrad_path = f"{h.get_path('ecrad', campaign=campaign)}/{date}"

# %% read in ecrad data
    ecrad_dict = dict()
    for v in ["v15", "v15_old"]:
        # use center column data
        ds = xr.open_dataset(f"{ecrad_path}/ecrad_merged_inout_{date}_{v}.nc").isel(column=0)
        ecrad_dict[v] = ds.copy()

    case_slice = slice(pd.Timestamp("2022-04-11 10:30"), pd.Timestamp("2022-04-11 13:00"))

# %% set plotting options
    var = "flux_dn_sw"
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
    xlabels = dict(v15="v15", v15_old="v15_old", diff="Difference v15 - v15_old")

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
        cmap = cmr.fusion
        norm = colors.TwoSlopeNorm(vcenter=0)

    # prepare ecrad dataset for plotting
    sf = h.scale_factors[var] if var in h.scale_factors else 1
    if v == "diff":
        # calculate difference between simulations
        ds = ecrad_dict["v15_old"]
        ecrad_ds_diff = ecrad_dict["v15"][var] - ds[var]
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
    figname = f"{fig_path}/{flight}_ecrad_{v}_{var}{band_str}_along_track.png"
    plt.savefig(figname, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# %% plot histogram
    xlabels = dict(diff="difference v15 - v15_old")
    xlabel = xlabels[v] if v in xlabels else v
    flat_array = ecrad_plot.to_numpy().flatten()
    mean = np.mean(flat_array)
    median = np.median(flat_array)
    _, ax = plt.subplots(figsize=h.figsize_wide)
    hist = ax.hist(flat_array, bins=20)
    ax.set(xlabel=f"{h.cbarlabels[var]} {xlabel} ({h.plot_units[var]})",
           ylabel="Number of occurrence")
    ax.grid()
    ax.set_yscale("log")
    ax.text(0.75, 0.9, f"Mean: {mean:.1f} " + "W$\,$m$^{-2}$" + f"\nMedian: {median:.1f} " + "W$\,$m$^{-2}$",
            ha='left', va='top', transform=ax.transAxes,
            bbox=dict(fc='white', ec='black', alpha=0.8, boxstyle='round'))
    plt.tight_layout()
    figname = f"{plot_path}/{flight}_ecrad_{v}_{var}{band_str}_hist.png"
    plt.savefig(figname, dpi=300, bbox_inches="tight")
    figname = f"{fig_path}/{flight}_ecrad_{v}_{var}{band_str}_hist.png"
    plt.savefig(figname, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# %% plot profiles of trace gases
    version_label = {"v15_old": "fixed trace gases", "v15": "CAMS climatology"}
    _, axs = plt.subplots(1, 6, figsize=h.figsize_wide)
    for version in ["v15", "v15_old"]:
        plot_ds = ecrad_dict[version].sel(time=case_slice).mean(dim="time")
        ls = "-" if version == "v15" else "--"
        for i, tg in enumerate(["cfc11_vmr", "cfc12_vmr", "ch4_vmr", "co2_vmr", "n2o_vmr", "o3_vmr"]):
            ax = axs[i]
            if plot_ds[tg].shape == ():
                plot_ds[tg] = xr.DataArray(np.repeat(plot_ds[tg].to_numpy(),
                                                     len(plot_ds["press_height_full"])),
                                           dims=["level"])
            ax.plot(plot_ds[tg] * 1e9, plot_ds["press_height_full"] / 1000, label=version_label[version],
                    ls=ls, marker=".")
            yticklabels = ax.get_yticklabels() if i == 0 else []
            ax.set(xlabel="VMR (ppb)", title=tg[:-4], yticklabels=yticklabels)
            ax.grid(True)


    axs[0].set(ylabel="Pressure height (km)")
    plt.tight_layout()
    axs[-1].legend(bbox_to_anchor=(0.5, 0), loc="lower center",
                    bbox_transform=_.transFigure, ncol=2)
    plt.subplots_adjust(bottom=0.2)
    figname = f"{plot_path}/profile_comparison.png"
    plt.savefig(figname, dpi=300)
    figname = f"{fig_path}/profile_comparison.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()
