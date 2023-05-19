#!/usr/bin/env python
"""
| *author:* Johannes Röttenbacher
| *created:* 15-04-2023

Set the albedo during the whole flight to 0.06 (open ocean) or 0.99 (maximum albedo) and analyze the impact on the offset between simulation and measurement during the below cloud section in RF17.

* ``IFS_namelist_jr_20220411_v13.nam``: for flight RF17 with Fu-IFS ice model setting albedo to open ocean (input version v5)
* ``IFS_namelist_jr_20220411_v13.1.nam``: for flight RF17 with Fu-IFS ice model setting albedo to 0.99 (input version v5.1)

**Problem statement:** A clear offset can be observed in the solar downward irradiance below the cloud between ecRad and BACARDI with ecRad showing lower values than BACARDI.
One idea is that multiple scattering from the sea ice surface to the cloud and back down plays a role here.
By setting the albedo to open ocean (0.06) we can eliminate this multiple backscattering between cloud and surface.
Comparing this experiment with the standard experiment can show us the potential impact of multiple scattering.
We also run a experiment where we set the albedo to 0.99 to see how much more downward irradiance we can observe that way.

We will focus on the above and below cloud section in the far north.
The corresponding spectral surface albedo can be seen in :numref:`surface-albedo-cs`.

.. _surface-albedo-cs:

.. figure:: figures/experiment_v13/HALO-AC3_20220411_HALO_RF17_sw_albedo_along_track_v1.png

    Short wave albedo along track above and below cloud for all six spectral bands after :cite:t:`Ebert1992`.

At first, we look at the difference in solar upward and downward irradiance between v1 (IFS albedo after :cite:t:`Ebert1992`) and v13 (ocean albedo 0.06).

.. figure:: figures/experiment_v13/HALO-AC3_20220411_HALO_RF17_ecrad_diff_flux_up_sw_along_track.png

    Difference in solar upward irradiance between v1 and v13.

.. figure:: figures/experiment_v13/HALO-AC3_20220411_HALO_RF17_ecrad_diff_flux_dn_sw_along_track.png

    Difference in solar downward irradiance between v1 and v13.

We can see an unsurprising substantial difference in upward irradiance which then propagates to a smaller but still relevant difference in downward irradiance.
This is especially pronounced for the thicker section of the cirrus at around 11:15 UTC.

Looking at this from a more statistical point of view we can see the bias between simulation and measurement increase by about :math:`10\,Wm^{-2}`.

.. figure:: figures/experiment_v13/HALO-AC3_20220411_HALO_RF17_Fdw_solar_bacardi_vs_ecrad_scatter_below_cloud_v1.png

    Scatterplot of along track difference between ecRad and BACARDI for v1.

.. figure:: figures/experiment_v13/HALO-AC3_20220411_HALO_RF17_Fdw_solar_bacardi_vs_ecrad_scatter_below_cloud_v13.png

    Scatterplot of along track difference between ecRad and BACARDI for v13.

An interesting side note: Although the surface albedo is now set to an open ocean value the emissivity and skin temperature are still the same.
Thus, there is only a minor change in the terrestrial upward irradiance.

So how much difference does multiple scattering between the surface and cloud make?
For this we can take a look at the histogramm of differences and some statistics.

.. figure:: figures/experiment_v13/HALO-AC3_20220411_HALO_RF17_ecrad_diff_flux_dn_sw_hist.png

    Histogram of differences between v1 and v13.

We see that a lot of values are rather small.
They correspond to the area above the cloud where only the atmosphere causes some minor scattering.
The median and mean of the distribution, however, are around :math:`10\\,Wm^{-2}`, which is quite substantial.

So albedo does obviously have a major influence on the downward irradiance in this scenario.
The next question now is, whether we can reduce the bias by increasing the surface albedo?
For this we take a look at experiment v13.1 with an albedo of 0.99.

.. figure:: figures/experiment_v13/HALO-AC3_20220411_HALO_RF17_ecrad_diff1_flux_dn_sw_along_track.png

    Difference in solar downward irradiance between v1 and v13.1 (albedo = 0.99).

By increasing the albedo to an unrealistic value of 0.99 we get a maximum of :math:`3\\,Wm^{-2}` difference in solar downward irradiance.
Comparing the spectral albedo of each experiment in :numref:`spectral-albedo-all-experiments` we can also see, that the standard albedo for the scene is already high.
So increasing it does not seem to be a sensible idea.

.. _spectral-albedo-all-experiments:

.. figure:: figures/experiment_v13/HALO-AC3_20220411_HALO_RF17_IFS_sw_albedo_spectrum_1100.png

    Spectral albedo for all three experiments at one timestep below cloud.

From all this we can conclude that **the albedo does not seem to be the major problem** in this scene.

"""

if __name__ == "__main__":
# %% import modules
    import pylim.helpers as h
    import pylim.halo_ac3 as meta
    from pylim import ecrad
    import ac3airborne
    from ac3airborne.tools import flightphase
    import os
    import xarray as xr
    import numpy as np
    from metpy.calc import density
    from metpy.units import units
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

    plot_path = f"{h.get_path('plot', flight, campaign)}/{flight}/experiment_v13"
    fig_path = "./docs/figures/experiment_v13"
    h.make_dir(plot_path)
    ecrad_path = f"{h.get_path('ecrad', campaign=campaign)}/{date}"
    bacardi_path = h.get_path("bacardi", flight, campaign)
    bacardi_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1_1s.nc"
    bahamas_path = h.get_path("bahamas", flight, campaign)
    bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{key}_v1_1s.nc"

    # set up metadata for access to HALO-AC3 cloud
    kwds = {'simplecache': dict(same_names=True)}
    credentials = {"user": os.environ.get("AC3_CLOUD_USER"), "password": os.environ.get("AC3_CLOUD_PASSWORD")}
    cat = ac3airborne.get_intake_catalog()["HALO-AC3"]["HALO"]

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

# %% read in data from HALO-AC3 cloud
    ins = xr.open_dataset(f"{bahamas_path}/{bahamas_file}")

# %% read in BACARDI data
    bacardi_ds = xr.open_dataset(f"{bacardi_path}/{bacardi_file}")
    bacardi_ds_res = bacardi_ds.resample(time="1Min").first().sel(time=case_slice)

# %% read in ecrad data
    ecrad_dict = dict()
    for v in ["v1", "v13", "v13.1"]:
        # use mean over columns data
        ds = xr.open_dataset(f"{ecrad_path}/ecrad_merged_inout_{date}_{v}_mean.nc")
        # select above and below cloud time
        ds = ds.sel(time=case_slice)
        ecrad_dict[v] = ds.copy()

# %% modify data sets
    for k in ecrad_dict:
        ds = ecrad_dict[k].copy()
        # add coordinate values
        ds = ds.assign_coords({"band_sw": range(1, 15), "band_lw": range(1, 17), "sw_albedo_band": range(1, 7)})
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

# %% get height level of actual flight altitude in ecRad model, this determines only the index of the level
    aircraft_height_da, height_level_da = dict(), dict()
    for v in ["v1", "v13"]:
        ds = ecrad_dict[v]
        bahamas_tmp = ins.sel(time=ds.time, method="nearest")
        ecrad_timesteps = len(ds.time)
        aircraft_height_level = np.zeros(ecrad_timesteps)

        for i in tqdm(range(ecrad_timesteps)):
            aircraft_height_level[i] = h.arg_nearest(ds["pressure_hl"][i, :].values, bahamas_tmp.PS[i].values * 100)

        aircraft_height_level = aircraft_height_level.astype(int)
        height_level_da[v] = xr.DataArray(aircraft_height_level, dims=["time"], coords={"time": ds.time})
        aircraft_height = ds["pressure_hl"].isel(half_level=height_level_da[v])
        aircraft_height_da[v] = xr.DataArray(aircraft_height, dims=["time"], coords={"time": ds.time},
                                             name="aircraft_height", attrs={"unit": "Pa"})

# %% prepare metadata for comparing ecRad and BACARDI
    titles = ["Solar Downward Irradiance", "Terrestrial Downward Irradiance", "Solar Upward Irradiance",
              "Terrestrial Upward Irradiance"]
    names = ["Fdw_solar", "Fdw_terrestrial", "Fup_solar", "Fup_terrestrial"]
    bacardi_vars = ["F_down_solar", "F_down_terrestrial", "F_up_solar", "F_up_terrestrial"]
    ecrad_vars = ["flux_dn_sw", "flux_dn_lw", "flux_up_sw", "flux_up_lw"]

# %% set plotting options
    var = "flux_dn_sw"
    v = "diff1"
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

    if "diff" in v:
        cmap = cmr.fusion
        norm = colors.TwoSlopeNorm(vcenter=0)

    # prepare ecrad dataset for plotting
    sf = h.scale_factors[var] if var in h.scale_factors else 1
    if v == "diff":
        # calculate difference between simulations
        ds = ecrad_dict["v13"]
        ecrad_ds_diff = ecrad_dict["v1"][var] - ds[var]
        ecrad_plot = ecrad_ds_diff.where((ds[var] != 0) | (~np.isnan(ds[var]))) * sf
    elif v == "diff1":
        # calculate difference between simulations
        ds = ecrad_dict["v13.1"]
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
    xlabels = dict(diff="difference v1 - v13", diff1="difference v1 - v13.1")
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
    ax.text(0.75, 0.9, f"Mean: {mean:.2f}" + "W$\,$m$^{-2}$" + f"\nMedian: {median:.2f}" + "W$\,$m$^{-2}$",
            ha='left', va='top', transform=ax.transAxes,
            bbox=dict(fc='white', ec='black', alpha=0.8, boxstyle='round'))
    plt.tight_layout()
    figname = f"{plot_path}/{flight}_ecrad_{v}_{var}{band_str}_hist.png"
    plt.savefig(figname, dpi=300, bbox_inches="tight")
    figname = f"{fig_path}/{flight}_ecrad_{v}_{var}{band_str}_hist.png"
    plt.savefig(figname, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# %% plot timeseries
    var = "sw_albedo"
    v = "v1"
    band = None
    if v == "diff":
        ecrad_plot = ecrad_dict["v1"][var] - ecrad_dict["v13"][var]
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
    for v in ["v1", "v13", "v13.1"]:
        ds_plot = ecrad_dict[v]["sw_albedo"].sel(time="2022-04-11 11:00")
        ax.plot(ds_plot.to_numpy(), lw=3, label=v)
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

# %% plot scatterplot of below cloud measurements
    v = "v13"
    bacardi_plot = bacardi_ds_res.sel(time=below_slice)
    ecrad_plot = ecrad_dict[v].isel(half_level=height_level_da[v]).sel(time=below_slice)
    plt.rc("font", size=12)
    lims = dict(v1=[(120, 240), (80, 130), (95, 170), (210, 220)], v13=[(110, 240), (80, 150), (0, 200), (210, 220)])
    lims = lims[v]
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
        ax.set_xlabel("BACARDI irradiance (W$\,$m$^{-2}$)")
        ax.set_ylabel("ecRad irradiance (W$\,$m$^{-2}$)")
        ax.set_title(f"{titles[i]}\nbelow cloud")
        ax.grid()
        ax.text(0.025, 0.95, f"# points: {sum(~np.isnan(bacardi_plot[x])):.0f}\n"
                             f"RMSE: {rmse.values:.2f}" + " W$\,$m$^{-2}$\n"
                                                          f"Bias: {bias.to_numpy():.2f}" + " W$\,$m$^{-2}$",
                ha='left', va='top', transform=ax.transAxes,
                bbox=dict(fc='white', ec='black', alpha=0.8, boxstyle='round'))
        plt.tight_layout()
        figname = f"{plot_path}/{flight}_{names[i]}_bacardi_vs_ecrad_scatter_below_cloud_{v}.png"
        plt.savefig(figname, bbox_inches="tight", dpi=300)
        figname = f"{fig_path}/{flight}_{names[i]}_bacardi_vs_ecrad_scatter_below_cloud_{v}.png"
        plt.savefig(figname, bbox_inches="tight", dpi=300)
        plt.show()
        plt.close()
