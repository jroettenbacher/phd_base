#!/usr/bin/env python
"""
| *author:* Johannes Röttenbacher
| *created:* 06-07-2023

Comparison between using the original resolution IFS data O1280 and the interpolated F1280 grid.

* ``IFS_namelist_jr_20220411_v1.nam``: for flight RF17 with Fu-IFS ice model
* ``IFS_namelist_jr_20220411_v15.nam``: for flight RF17 with Fu-IFS ice model and IFS model input on O1280 grid

.. figure:: figures/experiment_v15/HALO-AC3_20220411_HALO_RF17_IFS_skin_temperature_grid_comparison.png

    Ice water content from the IFS along flight track showing the closest grid point at each minutely time step.

Results
^^^^^^^^

.. figure:: figures/experiment_v15/HALO-AC3_20220411_HALO_RF17_ecrad_diff_iwc_along_track.png

    Difference between IWC IFS and IWC VarCloud.

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
    import cartopy.crs as ccrs
    import pandas as pd
    from tqdm import tqdm
    import cmasher as cmr
    from collections import Counter

    # plotting variables
    cbc = h.get_cb_friendly_colors()
    h.set_cb_friendly_colors()
    plt.rc("font", size=12)

# %% set paths
    campaign = "halo-ac3"
    key = "RF17"
    flight = meta.flight_names[key]
    date = flight[9:17]

    plot_path = f"{h.get_path('plot', flight, campaign)}/{flight}/experiment_v15"
    fig_path = "./docs/figures/experiment_v15"
    h.make_dir(plot_path)
    ecrad_path = f"{h.get_path('ecrad', campaign=campaign)}/{date}/"
    ifs_path = f"{h.get_path('ifs', campaign=campaign)}/{date}"
    ins_path = h.get_path("bahamas", flight, campaign)
    bacardi_path = h.get_path("bacardi", flight, campaign)
    ecrad_v1 = f"ecrad_merged_inout_{date}_v1.nc"
    ecrad_v15 = f"ecrad_merged_inout_{date}_v15.nc"
    ins = xr.open_dataset(f"{ins_path}/HALO-AC3_HALO_BAHAMAS_20220411_RF17_v1_1s.nc")
    bacardi_ds = xr.open_dataset(f"{bacardi_path}/HALO-AC3_HALO_BACARDI_BroadbandFluxes_20220411_RF17_R1_1s.nc")

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

# %% read in ecrad data
    ecrad_ds_v1 = xr.open_dataset(f"{ecrad_path}/{ecrad_v1}")
    ecrad_ds_v15 = xr.open_dataset(f"{ecrad_path}/{ecrad_v15}")

    # put all data sets in a dictionary
    ecrad_dict = dict(v1=ecrad_ds_v1, v15=ecrad_ds_v15)

# %% read in ifs data
    ifs_o1280 = xr.open_dataset(f"{ifs_path}/ifs_20220411_00_ml_O1280_processed.nc")
    ifs_f1280 = xr.open_dataset(f"{ifs_path}/ifs_20220411_00_ml_F1280_processed.nc")
    ifs_o1280 = ifs_o1280.set_index(rgrid=["lat", "lon"])

# %% plot IFS data and grid points in case study region on O1280 and F1280 grid
    ts = "2022-04-11 11:00"
    var = "skin_temperature"
    lat_filter, lon_filter = 84.5, ifs_f1280.lon.min()
    extent = [-38, 30, 84.5, 90] #[-60, 30, 84.5, 90]
    ds_plot1 = ifs_o1280[var].sel(time=ts)
    loc_filter = (ds_plot1.lat > lat_filter) & (ds_plot1.lon > lon_filter)
    lat_sel = ds_plot1.lat.where(loc_filter , drop=True).to_numpy()
    lon_sel = ds_plot1.lon.where(loc_filter, drop=True).to_numpy()
    lat_lon_sel = [(x, y) for x,y in zip(lat_sel, lon_sel)]
    ds_plot1 = ds_plot1.sel(rgrid=lat_lon_sel)
    ds_plot2 = ifs_f1280[var].where(ifs_f1280.lat > lat_filter, drop=True).sel(time=ts)
    x, y, z = ds_plot1.lon.to_numpy(), ds_plot1.lat.to_numpy(), ds_plot1.to_numpy()
    # z = ds_plot1.unstack(["rgrid"])
    crs_data = ccrs.PlateCarree()
    crs_plot = ccrs.NorthPolarStereo()

    _, axs = plt.subplots(1, 2, figsize=[16, 5], subplot_kw=dict(projection=crs_plot))
    # O1280
    ax = axs[0]
    ax.set_extent(extent, crs=crs_data)
    ax.tricontour(x,y,z, levels=14, linewidths=0.5, colors='k', transform=crs_data)
    ax.coastlines()
    ax.plot(ins.IRS_LON, ins.IRS_LAT, transform=crs_data, lw=3, color=cbc[3], label="HALO flight path")
    cntr2 = ax.tricontourf(x, y, z, levels=14, cmap=h.cmaps["t"], transform=crs_data)

    _.colorbar(cntr2, ax=ax, label="Skin temperature (K)")
    ax.plot(ds_plot1.lon, ds_plot1.lat, 'k.', ms=2, transform=crs_data)
    gl = ax.gridlines(crs=crs_data, linewidth=1, color='black', alpha=0.5, linestyle='--', draw_labels=True)
    gl.top_labels = False
    gl.left_labels = False
    gl.right_labels = True
    gl.xlines = True

    ax.legend(loc=2)
    ax.set_title("IFS skin temperature on original O1280 grid")

    # F1280
    ax = axs[1]
    x, y = np.meshgrid(ds_plot2.lon, ds_plot2.lat)
    z = ds_plot2
    ax.set_extent(extent, crs=crs_data)
    ax.contour(x, y, z, levels=14, linewidths=0.5, colors='k', transform=crs_data)
    ax.coastlines()
    ax.plot(ins.IRS_LON, ins.IRS_LAT, transform=crs_data, lw=3, color=cbc[3], label="HALO flight path")
    cntr2 = ax.contourf(x, y, z, levels=14, cmap=h.cmaps["t"], transform=crs_data)

    _.colorbar(cntr2, ax=ax, label="Skin temperature (K)")
    ax.plot(x, y, 'k.', ms=1, transform=crs_data)
    gl = ax.gridlines(crs=crs_data, linewidth=1, color='black', alpha=0.5, linestyle='--', draw_labels=True)
    gl.top_labels = False
    gl.left_labels = False
    gl.right_labels = True
    gl.xlines = True

    ax.legend(loc=2)
    ax.set_title("IFS skin temperature on interpolated F1280 grid")
    plt.tight_layout()
    figname = f"{plot_path}/HALO-AC3_{date}_{key}_IFS_{var}_grid_comparison.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% check if point is closest to aircraft position
# ds = ecrad_dict["v15"]
# latlon = bac.sel(time=ds.time[1:])[["IRS_LAT", "IRS_LON"]]
# plt.plot(latlon.IRS_LON, latlon.IRS_LAT, ".k")
# plt.plot(ds.lon, ds.lat, ".r")
# plt.show()
# plt.close()

# %% modify data sets
    for v in ecrad_dict:
        ds = ecrad_dict[v].copy()
        # select only center column, select above and below cloud time
        ds = ds.sel(column=16) if "column" in ds.dims else ds
        ds = ds.sel(time=slice(above_cloud["start"], below_cloud["end"]))
        if "press_height_full" not in ds:
            ds = ecrad.calculate_pressure_height(ds)
        # replace default values with nan
        ds["re_ice"] = ds.re_ice.where(np.round(ds.re_ice, 10) != 5.19616e-05, np.nan)
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
    ins_tmp = ins.sel(time=case_slice)
    ds = ecrad_dict["v15"]
    ecrad_timesteps = len(ds.time)
    aircraft_height_level = np.zeros(ecrad_timesteps)

    for i in tqdm(range(ecrad_timesteps)):
        aircraft_height_level[i] = h.arg_nearest(ds["press_height_hl"][i, :].values, ins_tmp.IRS_ALT[i].values)

    aircraft_height_level = aircraft_height_level.astype(int)
    height_level_da = xr.DataArray(aircraft_height_level, dims=["time"], coords={"time": ds.time})
    aircraft_height = ds["press_height_hl"].isel(half_level=height_level_da)

# %% get height level of actual flight altitude in ecRad model on full levels
    aircraft_height_level_full = np.zeros(ecrad_timesteps)

    for i in tqdm(range(ecrad_timesteps)):
        aircraft_height_level_full[i] = h.arg_nearest(ds["press_height_full"][i, :].values, ins_tmp.IRS_ALT[i].values)

    aircraft_height_level_full = aircraft_height_level_full.astype(int)
    height_level_da_full = xr.DataArray(aircraft_height_level_full, dims=["time"], coords={"time": ds.time})
    aircraft_height_full = ds["press_height_full"].isel(level=height_level_da_full)

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
                    iwc=[1, 5, 10, 15])
    linewidths = dict()
    robust = dict(iwc=False)
    cb_ticks = dict()
    vmaxs = dict()
    vmins = dict(iwp=0)
    xlabels = dict(v1="v1", v8="v8", v10="v10", diffv8="Difference v1 - v8", diffv10="Difference v1 - v10")

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
        ds = ecrad_dict["v15"].copy()
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
    time_sel = case_slice if v == "v10" else sel_time
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



# %% plot map with highlighted points were a simulation has been performed
    ds_plot = ecrad_dict["v15"][["lat", "lon", "skin_temperature"]]
    lat_lon_idx = list()
    for lats, lons in zip(ds_plot.lat.to_numpy(), ds_plot.lon.to_numpy()):
        lat_lon_idx = lat_lon_idx + [(lat, lon) for lat, lon in zip(lats, lons)]

    lat_lon_idx = list(Counter(lat_lon_idx).keys())
    lat = [lat_lon_idx[i][0] for i in range(len(lat_lon_idx))]
    lon = [lat_lon_idx[i][1] for i in range(len(lat_lon_idx))]
    _, ax = plt.subplots(subplot_kw=dict(projection=crs_plot))
    ax.set_extent(extent)
    ax.coastlines()
    gl = ax.gridlines(crs=crs_data, linewidth=1, color='black', alpha=0.5, linestyle='--', draw_labels=True)
    ax.plot(ins.IRS_LON, ins.IRS_LAT, transform=crs_data, lw=3, color=cbc[3], label="HALO flight path")
    ax.plot(lon, lat, ".k", ms=1, transform=crs_data)
    plt.show()
    plt.close()

# %% plot histogram of particle sizes
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
    # figname = f"{fig_path}/{flight}_ecrad_v1_v8_re_ice_psd.png"
    # plt.savefig(figname, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# %% plot BACARDI and ecrad v10 along track
    ds10, ds1 = ecrad_dict["v10"], ecrad_dict["v1"]
    ds10_res = ds10.resample(time="1Min").first()
    time_sel = slice(ds10.time[0], ds10.time[-1])
    hl_sel = height_level_da.sel(time=ds10.time, method="nearest").assign_coords(time=ds10.time)
    ecrad_plot = ds10["flux_dn_sw"].isel(half_level=hl_sel)
    ecrad_plot1 = ds1["flux_dn_sw"].isel(half_level=height_level_da).sel(time=time_sel)
    ecrad_plot2 = ds10_res["flux_dn_sw"].isel(half_level=height_level_da.sel(time=ds10_res.time))
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
    ds10 = ecrad_dict["v10"]
    time_sel = slice(ds10.time[100], ds10.time[-1])
    hl_sel = height_level_da.sel(time=ds10.time, method="nearest").assign_coords(time=ds10.time)
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

# %% plot scatterplot of above cloud measurements
    labels = dict(F_down_solar=r"$F_{\downarrow, solar}$", F_down_terrestrial=r"$F_{\downarrow, terrestrial}$",
              F_up_solar=r"$F_{\uparrow, solar}$", F_up_terrestrial=r"$F_{\uparrow, terrestrial}$")
    # prepare metadata for comparing ecRad and BACARDI
    titles = ["Solar Downward Irradiance", "Terrestrial Downward Irradiance", "Solar Upward Irradiance",
              "Terrestrial Upward Irradiance"]
    names = ["Fdw_solar", "Fdw_terrestrial", "Fup_solar", "Fup_terrestrial"]
    bacardi_vars = ["F_down_solar", "F_down_terrestrial", "F_up_solar", "F_up_terrestrial"]
    ecrad_vars = ["flux_dn_sw", "flux_dn_lw", "flux_up_sw", "flux_up_lw"]

    time_sel = above_slice
    ecrad_ds = xr.open_dataset(f"{ecrad_path}/{ecrad_v8}").sel(time=time_sel)
    hl_sel = height_level_da.sel(time=ecrad_ds.time, method="nearest")
    bacardi_plot = bacardi_ds_res.sel(time=time_sel)
    ecrad_plot = ecrad_ds.isel(half_level=hl_sel)
    lims = [(200, 270), (25, 35), (150, 200), (175, 195)]
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
