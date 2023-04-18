#!/usr/bin/env python
"""
| *author:* Johannes Röttenbacher
| *created:* 14-04-2023

Enable 3D parameterizations in the ecRad simulation of RF17 and investigate the impact compared to the normal run and
the VarCloud run in the case study region.

* ``IFS_namelist_jr_20220411_v1.nam``: for flight RF17 with Fu-IFS ice model
* ``IFS_namelist_jr_20220411_v5.nam``: for flight RF17 with Fu-IFS ice model and 3D parameterizations enabled
* ``IFS_namelist_jr_20220411_v6.nam``: for flight RF17 with Baran2016 ice model and 3D parameterizations enabled
* ``IFS_namelist_jr_20220411_v7.nam``: for flight RF17 with Baran2016 ice model
* ``IFS_namelist_jr_20220411_v10.nam``: for flight RF17 with Fu-IFS ice model and VarCloud input

The idea behind the 3D parameterization in SPARTACUS (the solver in ecRad) is to simulate the radiation transport through cloud sides and the entrapment of radiation.
It can be turned on by setting the namelist options:

    | do_3d_effects = true,
    | n_regions = 3,
    | do_lw_side_emissivity = true,
    | sw_entrapment_name = "Explicit",

This has been done using the Fu-IFS ice model (v5) and the Baran2016 ice model (v6).
The first comparison with Fu-IFS without 3D effects (v1) shows that more radiation is absorbed within the cloud.

.. figure:: figures/3d_analysis/HALO-AC3_20220411_HALO_RF17_ecrad_diffv5_flux_dn_sw_along_track.png

    Difference between Fu-IFS ecRad run without 3D effect parameterization (v1) and with (v5).

One peculiar thing to notice is the single column which shows higher downward irradiance in the 3D case.
For the Baran2016 comparison a similar picture can be seen but values seem to be a bit less extreme.
The histograms (not shown) reinforce this picture.

.. figure:: figures/3d_analysis/HALO-AC3_20220411_HALO_RF17_ecrad_diffv6_flux_dn_sw_along_track.png

    Difference between Baran2016 ecRad run without 3D effect parameterization (v7) and with (v6).

Comparing these results to the fact that ecRad is underestimating the transmissivity of the cloud enabling the 3D parameterizations does not improve the performance of ecRad.
In contrast, it increases the optical depth of the cloud further.
As a first result of this it can be said that for Arctic cirrus the 3D parameterizations do not improve the model.
This might have to do with the too small ice crystals assumed in the model.

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

    # plotting variables
    cm = h.cm
    cbc = h.get_cb_friendly_colors()
    h.set_cb_friendly_colors()
    plt.rc("font", size=12)

# %% set paths
    campaign = "halo-ac3"
    key = "RF17"
    flight = meta.flight_names[key]
    date = flight[9:17]

    plot_path = f"{h.get_path('plot', flight, campaign)}/{flight}/ecrad_case_study"
    fig_path = "./docs/figures/3d_analysis"
    h.make_dir(plot_path)
    ecrad_path = f"{h.get_path('ecrad', campaign=campaign)}/{date}/"
    ifs_path = f"{h.get_path('ifs', campaign=campaign)}/{date}"
    varcloud_path = h.get_path("varcloud", flight, campaign)
    libradtran_path = h.get_path("libradtran_exp", flight, campaign)
    bacardi_path = h.get_path("bacardi", flight, campaign)
    ecrad_inputv1 = f"ecrad_merged_input_{date}_v1.nc"
    ecrad_inputv3 = f"ecrad_merged_input_{date}_v3.nc"
    ecrad_v10 = f"ecrad_merged_output_{date}_v10.nc"
    varcloud_file = [f for f in os.listdir(varcloud_path) if "nc" in f][0]
    bacardi_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1_1s.nc"
    bacard_std = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1_std_1s.nc"
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
    # above cloud time with thin cirrus below
    sel_time = slice(above_cloud["start"], pd.to_datetime("2022-04-11 11:04"))

# %% read in data
    varcloud = xr.open_dataset(f"{varcloud_path}/{varcloud_file}").swap_dims(time="Time", height="Height").rename(
        Time="time")
    varcloud = varcloud.rename(Varcloud_Cloud_Ice_Water_Content="iwc", Varcloud_Cloud_Ice_Effective_Radius="re_ice")
    varcloud = varcloud.sel(time=sel_time).resample(time="1Min").asfreq()

    bacardi_ds = xr.open_dataset(f"{bacardi_path}/{bacardi_file}")  # read in BACARDI data
    bacardi_std = xr.open_dataset(f"{bacardi_path}/{bacard_std}")
    bacardi_ds_res = bacardi_ds.resample(time="1Min").mean()

    libradtran_ds = xr.open_dataset(f"{libradtran_path}/{libradtran_file}")  # read in libradtran data
    libradtran_ds["altitude"] = libradtran_ds.altitude / 1000
    libradtran_ds = libradtran_ds.rename(CIWD="iwc")

# %% read in ecrad data
    ecrad_ds_v10 = xr.open_dataset(f"{ecrad_path}/{ecrad_v10}")
    ecrad_ds_inputv1 = xr.open_dataset(f"{ecrad_path}/{ecrad_inputv1}")
    ecrad_ds_inputv3 = xr.open_dataset(f"{ecrad_path}/{ecrad_inputv3}")
    # select only center column from version 1, version 8/10 has only one column, select above and below cloud time
    ecrad_ds_inputv1 = ecrad_ds_inputv1.sel(column=16, time=case_slice)
    # add pressure height to dataset
    ecrad_ds_inputv1 = ecrad.calculate_pressure_height(ecrad_ds_inputv1)
    # merge input and output
    ecrad_ds_v10 = xr.merge([ecrad_ds_v10, ecrad_ds_inputv3])

    # put all data sets in a dictionary
    ecrad_dict = dict()
    for v in ["v1", "v5", "v6", "v7"]:
        ds = xr.open_dataset(f"{ecrad_path}/ecrad_merged_output_{date}_{v}.nc")
        # select only center column from version 1, version 8/10 has only one column, select above and below cloud time
        ds = ds.sel(column=16, time=case_slice)
        # merge input and output
        ds = xr.merge([ds, ecrad_ds_inputv1])
        ecrad_dict[v] = ds.copy()

    ecrad_dict["v10"] = ecrad_ds_v10

# %% modify data sets
    for v in ecrad_dict:
        ds = ecrad_dict[v].copy()
        # replace default values with nan
        ds["re_ice"] = ds.re_ice.where(ds.re_ice != 5.19616e-05, np.nan)
        ds["re_liquid"] = ds.re_liquid.where(ds.re_liquid != 4.000001e-06, np.nan)
        # compute cloud ice water path
        factor = ds.pressure_hl.diff(dim="half_level").to_numpy() / (
                9.80665 * ds.cloud_fraction.to_numpy())
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

        ecrad_dict[v] = ds.copy()

# %% get height level of actual flight altitude in ecRad model on half levels
    ins_tmp = bacardi_ds.resample(time="1Min").asfreq().sel(time=case_slice)
    ds = ecrad_dict["v1"]
    ecrad_timesteps = len(ds.time)
    aircraft_height_level = np.zeros(ecrad_timesteps)

    for i in tqdm(range(ecrad_timesteps)):
        aircraft_height_level[i] = h.arg_nearest(ds["press_height_hl"][i, :].values, ins_tmp.alt[i].values)

    aircraft_height_level = aircraft_height_level.astype(int)
    height_level_da = xr.DataArray(aircraft_height_level, dims=["time"], coords={"time": ds.time})
    aircraft_height = ds["press_height_hl"].isel(half_level=height_level_da)

# %% get height level of actual flight altitude in ecRad model on full levels
    aircraft_height_level_full = np.zeros(ecrad_timesteps)

    for i in tqdm(range(ecrad_timesteps)):
        aircraft_height_level_full[i] = h.arg_nearest(ds["press_height_full"][i, :].values, ins_tmp.alt[i].values)

    aircraft_height_level_full = aircraft_height_level_full.astype(int)
    height_level_da_full = xr.DataArray(aircraft_height_level_full, dims=["time"], coords={"time": ds.time})
    aircraft_height_full = ds["press_height_full"].isel(level=height_level_da_full)

# %% set plotting options
    var = "flux_up_lw"
    v = "diffv5"
    band = None

# %% prepare data set for plotting
    band_str = f"_band{band}" if band is not None else ""

    # kwarg dicts
    alphas = dict()
    ct_fontsize = dict()
    ct_lines = dict(ciwc=[1, 5, 10, 15], cswc=[1, 5, 10, 15], q_ice=[1, 5, 10, 15], clwc=[1, 5, 10, 15],
                    iwc=[1, 5, 10, 15])
    linewidths = dict()
    robust = dict(iwc=False, cloud_fraction=False)
    cb_ticks = dict()
    vmaxs = dict(cloud_fraction=1)
    vmins = dict(iwp=0, cloud_fraction=0)
    xlabels = dict(v1="v1", v5="v5", v6="v6", v7="v7", v10="v10",
                   diffv5="Difference v1 - v5", diffv6="Difference v7 - v6", diffv7="Difference v1 - v7",
                   diffv10="Difference v5 - v10")

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

        version = v[4:]  # get version from v string
        # calculate difference between simulations
        ds = ecrad_dict[version].copy()
        if version == "v6":
            ecrad_ds_diff = ecrad_dict["v7"][var] - ds[var]
        else:
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


