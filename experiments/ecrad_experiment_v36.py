#!/usr/bin/env python
"""
| *author:* Johannes Röttenbacher
| *created:* 19-01-2024

Set the fractional standard deviation to 0.
This will remove any subgrid scale variability of IWC.
Thus, the cloud is homogeneous in each grid cell.
Using the VarCloud input for these simulations would then have the effect of representing the measured inhomogeneity of the cloud field.

Namelists used:

* ``IFS_namelist_jr_20220411_v36.nam``: for RF17 with **Fu-IFS** ice model using O1280 IFS, **varcloud** retrieval for ciwc and re_ice input **below cloud**, turned fractional standard deviation to 0 (measure for inhomogeneity) (input version **v7**)
* ``IFS_namelist_jr_20220411_v37.nam``: for RF17 with **Yi2013** ice model using O1280 IFS, **varcloud** retrieval for ciwc and re_ice input **below cloud**, turned fractional standard deviation to 0 (measure for inhomogeneity) (input version **v7**)
* ``IFS_namelist_jr_20220411_v38.nam``: for RF17 with **Baran2016** ice model using O1280 IFS, **varcloud** retrieval for ciwc and re_ice input **below cloud**, turned fractional standard deviation to 0 (measure for inhomogeneity) (input version **v7**)
* ``IFS_namelist_jr_20220412_v36.nam``: for RF17 with **Fu-IFS** ice model using O1280 IFS, **varcloud** retrieval for ciwc and re_ice input **below cloud**, turned fractional standard deviation to 0 (measure for inhomogeneity) (input version **v7**)
* ``IFS_namelist_jr_20220412_v37.nam``: for RF17 with **Yi2013** ice model using O1280 IFS, **varcloud** retrieval for ciwc and re_ice input **below cloud**, turned fractional standard deviation to 0 (measure for inhomogeneity) (input version **v7**)
* ``IFS_namelist_jr_20220412_v38.nam``: for RF17 with **Baran2016** ice model using O1280 IFS, **varcloud** retrieval for ciwc and re_ice input **below cloud**, turned fractional standard deviation to 0 (measure for inhomogeneity) (input version **v7**)

We will compare these simulations with the original VarCloud simulations (v16, v28, v20).
First we start with the solar downward irradiance to see at which altitude changes start to appear.

.. figure:: figures/experiment_v36/RF17_ecrad_diff_flux_dn_sw_along_track.png

    Difference in short wave downward irradiance along track for RF 17.

.. figure:: figures/experiment_v36/RF18_ecrad_diff_flux_dn_sw_along_track.png

    Difference in short wave downward irradiance along track for RF 18.

As expected the changes only start to appear at cloud top.
Another comparison can be done using the upward solar irradiance.

.. figure:: figures/experiment_v36/RF17_ecrad_diff_flux_up_sw_along_track.png

    Difference in short wave upward irradiance along track for RF 17.

.. figure:: figures/experiment_v36/RF18_ecrad_diff_flux_up_sw_along_track.png

    Difference in short wave upward irradiance along track for RF 18.

We can also take a look on the effect the reduced variability has on the solar transmissivity below cloud.

.. figure:: figures/experiment_v36/HALO-AC3_HALO_RF17_RF18_bacardi_ecrad_transmissivity_below_cloud_PDF_ice_optics.png

    Solar transmissivity below cloud for RF 17 and RF 18 using the three different ice optic parameterizations Fu-IFS, Yi213 and Baran2016 (from left to right).

Although only minute, a small shift to lower transmissivities can be seen.
All in all, the change to a fractional standard deviation of 0 leads to a minor change of the radiative fluxes.
Turning the artificially introduced inhomogeneity off, is a better representation of the actual cloud field.
Therefore, these new versions are used in the paper and the thesis.

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
    import matplotlib.pyplot as plt
    from matplotlib import colors
    import pandas as pd
    import seaborn as sns
    from tqdm import tqdm
    import cmasher as cmr

    cbc = h.get_cb_friendly_colors()
    h.set_cb_friendly_colors()
    plt.rc("font", size=12)

# %% set paths
    campaign = "halo-ac3"
    plot_path = "C:/Users/Johannes/PycharmProjects/phd_base/docs/figures/experiment_v36"
    keys = ["RF17", "RF18"]
    ecrad_versions = ["v15.1", "v16", "v28", "v20", "v36", "v37", "v38"]

# %% read in data
    (
        bahamas_ds,
        bacardi_ds,
        bacardi_ds_res,
        ecrad_dicts,
        varcloud_ds,
        above_clouds,
        below_clouds,
        slices,
        ecrad_orgs,
        ifs_ds
    ) = (dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict(), dict())

    for key in keys:
        flight = meta.flight_names[key]
        date = flight[9:17]
        bacardi_path = h.get_path("bacardi", flight, campaign)
        bahamas_path = h.get_path("bahamas", flight, campaign)
        libradtran_path = h.get_path("libradtran", flight, campaign)
        libradtran_exp_path = h.get_path("libradtran_exp", flight, campaign)
        ifs_path = f"{h.get_path('ifs', flight, campaign)}/{date}"
        ecrad_path = f"{h.get_path('ecrad', flight, campaign)}/{date}"
        varcloud_path = h.get_path("varcloud", flight, campaign)

        # filenames
        bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{key}_v1_JR.nc"
        bacardi_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1_JR.nc"
        libradtran_bb_solar_si = f"HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_solar_si_{date}_{key}.nc"
        libradtran_bb_thermal_si = f"HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_thermal_si_{date}_{key}.nc"
        ifs_file = f"ifs_{date}_00_ml_O1280_processed.nc"
        varcloud_file = [f for f in os.listdir(varcloud_path) if f.endswith(".nc")][0]

        # read in aircraft data
        bahamas_ds[key] = xr.open_dataset(f"{bahamas_path}/{bahamas_file}")
        bacardi = xr.open_dataset(f"{bacardi_path}/{bacardi_file}")

        # read in ifs data
        ifs_ds[key] = xr.open_dataset(f"{ifs_path}/{ifs_file}")

        # read in varcloud data
        varcloud = xr.open_dataset(f"{varcloud_path}/{varcloud_file}")
        varcloud = varcloud.swap_dims(time="Time", height="Height").rename(Time="time", Height="height")
        varcloud = varcloud.rename(Varcloud_Cloud_Ice_Water_Content="iwc", Varcloud_Cloud_Ice_Effective_Radius="re_ice")
        varcloud_ds[key] = varcloud

        # read in libRadtran simulation
        bb_sim_solar_si = xr.open_dataset(f"{libradtran_exp_path}/{libradtran_bb_solar_si}")
        bb_sim_thermal_si = xr.open_dataset(f"{libradtran_exp_path}/{libradtran_bb_thermal_si}")
        # interpolate simualtion onto BACARDI time
        bb_sim_solar_si_inp = bb_sim_solar_si.interp(time=bacardi.time)
        bb_sim_thermal_si_inp = bb_sim_thermal_si.interp(time=bacardi.time)

        # calculate transmissivity BACARDI/libRadtran
        bacardi["F_down_solar_sim_si"] = bb_sim_solar_si_inp.fdw
        bacardi["F_down_terrestrial_sim_si"] = bb_sim_thermal_si_inp.edn
        bacardi["F_up_terrestrial_sim_si"] = bb_sim_thermal_si_inp.eup
        bacardi["transmissivity_solar"] = bacardi["F_down_solar"] / bb_sim_solar_si_inp.fdw
        bacardi["transmissivity_terrestrial"] = bacardi["F_down_terrestrial"] / bb_sim_thermal_si_inp.edn
        # calculate reflectivity
        bacardi["reflectivity_solar"] = bacardi["F_up_solar"] / bacardi["F_down_solar"]
        bacardi["reflectivity_terrestrial"] = bacardi["F_up_terrestrial"] / bacardi["F_down_terrestrial"]

        # calculate radiative effect from BACARDI and libRadtran sea ice simulation
        bacardi["F_net_solar"] = bacardi["F_down_solar"] - bacardi["F_up_solar"]
        bacardi["F_net_terrestrial"] = bacardi["F_down_terrestrial"] - bacardi["F_up_terrestrial"]
        bacardi["F_net_solar_sim"] = bb_sim_solar_si_inp.fdw - bb_sim_solar_si_inp.eup
        bacardi["F_net_terrestrial_sim"] = bb_sim_thermal_si_inp.edn - bb_sim_thermal_si_inp.eup
        bacardi["CRE_solar"] = bacardi["F_net_solar"] - bacardi["F_net_solar_sim"]
        bacardi["CRE_terrestrial"] = bacardi["F_net_terrestrial"] - bacardi["F_net_terrestrial_sim"]
        bacardi["CRE_total"] = bacardi["CRE_solar"] + bacardi["CRE_terrestrial"]
        bacardi["F_down_solar_error"] = np.abs(bacardi["F_down_solar"] * 0.03)
        # normalize downward irradiance for cos SZA
        for var in ["F_down_solar", "F_down_solar_diff"]:
            bacardi[f"{var}_norm"] = bacardi[var] / np.cos(np.deg2rad(bacardi["sza"]))
        # filter data for motion flag
        bacardi_org = bacardi.copy()
        bacardi = bacardi.where(bacardi.motion_flag)
        # overwrite variables which do not need to be filtered with original values
        for var in ["alt", "lat", "lon", "sza", "saa", "attitude_flag", "segment_flag", "motion_flag"]:
            bacardi[var] = bacardi_org[var]

        # read in resampled BACARDI data
        bacardi_res = xr.open_dataset(f"{bacardi_path}/{bacardi_file.replace('.nc', '_1Min.nc')}")
        # normalize downward irradiance for cos SZA
        for var in ["F_down_solar", "F_down_solar_diff"]:
            bacardi_res[f"{var}_norm"] = bacardi_res[var] / np.cos(np.deg2rad(bacardi_res["sza"]))
        bacardi_ds_res[key] = bacardi_res.copy()

        # read in ecrad data
        ecrad_dict, ecrad_org = dict(), dict()

        for k in ecrad_versions:
            ds = xr.open_dataset(f"{ecrad_path}/ecrad_merged_inout_{date}_{k}.nc")
            ecrad_org[k] = ds.copy(deep=True)
            # select only center column for direct comparisons
            ds = ds.sel(column=0, drop=True) if "column" in ds.dims else ds
            ecrad_dict[k] = ds.copy(deep=True)

        ecrad_dicts[key] = ecrad_dict
        ecrad_orgs[key] = ecrad_org

        # interpolate standard ecRad simulation onto BACARDI time
        bacardi["ecrad_fdw"] = ecrad_dict["v16"].flux_dn_sw_clear.interp(time=bacardi.time,
                                                                         kwargs={"fill_value": "extrapolate"})
        # calculate transmissivity using ecRad at TOA and above cloud
        bacardi["transmissivity_TOA"] = bacardi["F_down_solar"] / bacardi["ecrad_fdw"].isel(half_level=0)
        bacardi["transmissivity_above_cloud"] = bacardi["F_down_solar"] / bacardi["ecrad_fdw"].isel(half_level=73)
        # normalize transmissivity by cosine of solar zenith angle
        for var in ["transmissivity_TOA", "transmissivity_above_cloud"]:
            bacardi[f"{var}_norm"] = bacardi[var] / np.cos(np.deg2rad(bacardi["sza"]))
        bacardi_ds[key] = bacardi

        # get flight segmentation and select below and above cloud section
        fl_segments = ac3airborne.get_flight_segments()["HALO-AC3"]["HALO"][f"HALO-AC3_HALO_{key}"]
        segments = flightphase.FlightPhaseFile(fl_segments)
        above_cloud, below_cloud = dict(), dict()
        if key == "RF17":
            above_cloud["start"] = segments.select("name", "high level 7")[0]["start"]
            above_cloud["end"] = segments.select("name", "high level 8")[0]["end"]
            below_cloud["start"] = segments.select("name", "high level 9")[0]["start"]
            below_cloud["end"] = segments.select("name", "high level 10")[0]["end"]
            above_slice = slice(above_cloud["start"], above_cloud["end"])
            below_slice = slice(pd.to_datetime("2022-04-11 11:35"), below_cloud["end"])
            case_slice = slice(above_cloud["start"], below_cloud["end"])
        else:
            above_cloud["start"] = segments.select("name", "polygon pattern 1")[0]["start"]
            above_cloud["end"] = segments.select("name", "polygon pattern 1")[0]["parts"][-1]["start"]
            below_cloud["start"] = segments.select("name", "polygon pattern 2")[0]["start"]
            below_cloud["end"] = segments.select("name", "polygon pattern 2")[0]["end"]
            above_slice = slice(above_cloud["start"], above_cloud["end"])
            below_slice = slice(below_cloud["start"], below_cloud["end"])
            case_slice = slice(above_cloud["start"], below_cloud["end"])

        above_clouds[key] = above_cloud
        below_clouds[key] = below_cloud
        slices[key] = dict(case=case_slice, above=above_slice, below=below_slice)

# %% set plotting options
    var = "flux_up_sw"
    v = "diff"
    band = None
    key = "RF17"

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
    xlabels = {"v16": "v16", "v36": "v36", "diff": "Difference v16 - v36"}

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
        ds = ecrad_dicts[key]["v36"]
        ecrad_ds_diff = ecrad_dicts[key]["v16"][var] - ds[var]
        ecrad_plot = ecrad_ds_diff.where((ds[var] != 0) | (~np.isnan(ds[var]))) * sf
    else:
        ds = ecrad_dicts[key][v]
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
    time_sel = slices[key]["case"]
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

    ax.set_title(f"{key} IFS/ecRad input/output along Flight Track - {xlabels[v]}")
    ax.set_ylabel("Altitude (km)")
    ax.set_xlabel("Time (UTC)")
    h.set_xticks_and_xlabels(ax, time_extend)

    figname = f"{plot_path}/{key}_ecrad_{v}_{var}{band_str}_along_track.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot PDF of solar transmissivity of BACARDI and ecRad at flight altitude for both setting
    transmissivity_stats = list()
    plt.rc("font", size=7.5)
    label = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"]]
    ylims = (0, 28)
    binsize = 0.01
    xlabel = "Solar Transmissivity"
    _, axs = plt.subplots(2, 3, figsize=(18 * h.cm, 14 * h.cm), layout="constrained")
    for i, key in enumerate(keys):
        ax = axs[i]
        l = label[i]
        bacardi_sel = bacardi_ds[key].sel(time=slices[key]["below"])
        bacardi_plot = bacardi_sel[f"transmissivity_above_cloud"]
        bins = np.arange(0.5, 1.0, binsize)
        # BACARDI histogram
        bacardi_hist = np.histogram(bacardi_plot, density=True, bins=bins)

        # save statistics
        transmissivity_stats.append((key, "BACARDI", "Mean", bacardi_plot.mean().to_numpy()))
        transmissivity_stats.append((key, "BACARDI", "Median", bacardi_plot.median().to_numpy()))

        for ii, vs in enumerate(zip(ecrad_versions[1:4], ecrad_versions[4:])):
            a = ax[ii]
            # plot BACARDI data
            sns.histplot(bacardi_plot, label="BACARDI", ax=a, stat="density", kde=False, bins=bins, element="step")
            a.axvline(bacardi_plot.mean(), color=cbc[0], lw=3, ls="--")
            for iii, v in enumerate(vs):
                ecrad_ds = ecrad_orgs[key][v].sel(time=slices[key]["below"])
                height_sel = ecrad_ds["aircraft_level"]
                ecrad_plot = ecrad_ds[f"transmissivity_sw_above_cloud"].isel(half_level=height_sel)

                # save statistics
                transmissivity_stats.append((key, v, "Mean", ecrad_plot.mean().to_numpy()))
                transmissivity_stats.append((key, v, "Median", ecrad_plot.median().to_numpy()))
                # actual plotting
                sns.histplot(ecrad_plot.to_numpy().flatten(), label=v, stat="density", element="step",
                             kde=False, bins=bins, ax=a, color=cbc[ii + 1 + iii])
                # add mean
                a.axvline(ecrad_plot.mean(), color=cbc[ii + 1 + iii], lw=3, ls="--")

            a.plot([], ls="--", color="k", label="Mean")  # label for means
            a.set(ylabel="",
                  ylim=ylims,
                  xlim=(0.45, 1)
                  )
            handles, labels = a.get_legend_handles_labels()
            order = [1, 2, 0, 3]
            handles = [handles[idx] for idx in order]
            labels = [labels[idx] for idx in order]
            if key == "RF17":
                a.legend(handles, labels, loc=6)
            a.text(
                    0.02,
                    0.95,
                    f"{l[ii]}",
                    ha="left",
                    va="top",
                    transform=a.transAxes,
                )
            a.grid()

        ax[0].set(ylabel="Probability density function")
        ax[1].set(title=f"{key.replace('1', ' 1')} (n = {len(ecrad_plot.to_numpy().flatten()):.0f})")
        if key == "RF18":
            ax[1].set(xlabel=xlabel)

    figname = f"{plot_path}/HALO-AC3_HALO_RF17_RF18_bacardi_ecrad_transmissivity_below_cloud_PDF_ice_optics.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()
