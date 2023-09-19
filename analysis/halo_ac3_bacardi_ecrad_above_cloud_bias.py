#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 15.09.2023

RF17 show a substantial positive bias for the solar downward irradiance above cloud which should not be there.
RF18 does not show this bias.
Here we look at each flight to investigate this bias.
"""
# %% import modules
import pylim.helpers as h
import pylim.halo_ac3 as meta
from pylim import ecrad, reader
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import colors
import cmasher as cm
from tqdm import tqdm

h.set_cb_friendly_colors()
cbc = h.get_cb_friendly_colors()

# %% set paths
campaign = "halo-ac3"
plot_path = f"{h.get_path('plot', campaign=campaign)}/ecrad_f_down_bias"
keys = [f"RF{i:02}" for i in range(3, 19)]
keys.pop(4)  # remove RF07
keys.pop(1)  # remove RF04

# %% read in data
(
    bahamas_ds,
    bahamas_ds_res,
    bacardi_ds,
    bacardi_ds_res,
    ecrad_dicts,
    ecrad_orgs,
) = (dict(), dict(), dict(), dict(), dict(), dict())

for key in keys:
    flight = meta.flight_names[key]
    date = flight[9:17]
    bacardi_path = h.get_path("bacardi", flight, campaign)
    bahamas_path = h.get_path("bahamas", flight, campaign)
    ecrad_path = f"{h.get_path('ecrad', flight, campaign)}/{date}"

    # filenames
    bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{key}_v1.nc"
    bacardi_file = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1_JR.nc"
    bahamas_file_1min = f"HALO-AC3_HALO_BAHAMAS_{date}_{key}_v1_1Min.nc"
    bacardi_file_1min = f"HALO-AC3_HALO_BACARDI_BroadbandFluxes_{date}_{key}_R1_JR_1Min.nc"

    # read in aircraft data
    bahamas_ds[key] = reader.read_bahamas(f"{bahamas_path}/{bahamas_file}")
    bacardi = xr.open_dataset(f"{bacardi_path}/{bacardi_file}")
    bahamas_ds_res[key] = xr.open_dataset(f"{bahamas_path}/{bahamas_file_1min}")
    bacardi_res = xr.open_dataset(f"{bacardi_path}/{bacardi_file_1min}")
    bacardi_ds_res[key] = bacardi_res

    # filter data for motion flag - original resolution
    bacardi_org = bacardi.copy()
    bacardi_filtered = bacardi.where(bacardi.motion_flag)
    # overwrite variables which do not need to be filtered with original values
    for var in ["alt", "lat", "lon", "sza", "saa", "attitude_flag", "segment_flag", "motion_flag"]:
        bacardi_filtered[var] = bacardi_org[var]
    bacardi_ds[key] = bacardi

    # read in ecrad data
    ecrad_versions = ["v15"]
    ecrad_dict, ecrad_org = dict(), dict()

    for k in ecrad_versions:
        ds = xr.open_dataset(f"{ecrad_path}/ecrad_merged_inout_{date}_{k}.nc")

        if "column" in ds.dims:
            n_column = len(ds.column)
            height_sel = ecrad.get_model_level_of_altitude(bacardi_res.alt, ds.sel(column=0), "half_level")
            ecrad_ds = ds.isel(half_level=height_sel, column=slice(0, 10))
            bacardi_res = bacardi_res.expand_dims(dict(column=np.arange(0, 10))).copy()

            ds["flux_dn_sw_diff"] = ecrad_ds.flux_dn_direct_sw - bacardi_res.F_down_solar
            ds["spread"] = xr.DataArray(
                np.array(
                    [
                        ds["flux_dn_sw_diff"].min(dim="column").to_numpy(),
                        ds["flux_dn_sw_diff"].max(dim="column").to_numpy(),
                    ]
                ),
                coords=dict(x=[0, 1], time=ecrad_ds.time),
            )
            ds["flux_dn_sw_std"] = ds["flux_dn_direct_sw"].std(dim="column")

            ecrad_org[k] = ds.copy(deep=True)
            if k == "v1":
                ds = ds.sel(column=16,
                            drop=True)  # select center column which corresponds to grid cell closest to aircraft
            else:
                # other versions have their nearest points selected via
                # kdTree, thus the first column should be the closest
                ds = ds.sel(column=0, drop=True)
        else:
            # height_sel = ecrad.get_model_level_of_altitude(bacardi_res.sel(column=0).alt, ds, "half_level")
            # ecrad_ds = ds.isel(half_level=height_sel)
            # ds["flux_dn_sw_diff"] = ecrad_ds.flux_dn_sw - bacardi_res.F_down_solar
            # ecrad_org[k] = ds.copy(deep=True)
            pass

        ds["tiwp"] = ds.iwp.where(ds.iwp != np.inf, np.nan).sum(dim="level")
        for var in ["flux_dn_sw", "flux_dn_direct_sw"]:
            ds[f"{var}_norm"] = ds[var] / ds["cos_solar_zenith_angle"]

        ecrad_dict[k] = ds.isel(half_level=height_sel).copy()  # select only flight altitude from ecRad files


    ecrad_dicts[key] = ecrad_dict
    ecrad_orgs[key] = ecrad_org


# %% plot scatter plot of above cloud measurements and simulations
plt.rc("font", size=7)
for key in tqdm(keys):
    _, ax = plt.subplots(figsize=(16 * h.cm, 9 * h.cm))

    above_sel = bahamas_ds_res[key].IRS_ALT > (bahamas_ds_res[key].IRS_ALT.median() - 500)
    bacardi_res = bacardi_ds_res[key]
    bacardi_plot = bacardi_res.where(bacardi_res.alt > (bacardi_res.alt.median() - 500))
    ecrad_ds = ecrad_dicts[key]["v15"]
    ecrad_plot = ecrad_ds.flux_dn_direct_sw.where(above_sel)

    # actual plotting
    rmse = np.sqrt(np.mean((ecrad_plot - bacardi_plot["F_down_solar"]) ** 2)).to_numpy()
    bias = np.nanmean((ecrad_plot - bacardi_plot["F_down_solar"]).to_numpy())
    ax.scatter(bacardi_plot.F_down_solar_diff, ecrad_plot, color=cbc[3])
    ax.axline((0, 0), slope=1, color="k", lw=2, transform=ax.transAxes)
    ax.set(
        aspect="equal",
        title=meta.flight_names[key],
        xlabel="BACARDI irradiance (W$\,$m$^{-2}$)",
        ylabel="ecRad irradiance (W$\,$m$^{-2}$)",
        xlim=(100, 700),
        ylim=(100, 700),
    )
    ax.grid()
    ax.text(
        0.025,
        0.95,
        f"n: {sum(~np.isnan(bacardi_plot['F_down_solar'])):.0f}\n"
        f"RMSE: {rmse:.0f} {h.plot_units['flux_dn_sw']}\n"
        f"Bias: {bias:.0f} {h.plot_units['flux_dn_sw']}",
        ha="left",
        va="top",
        transform=ax.transAxes,
        bbox=dict(fc="white", ec="black", alpha=0.8, boxstyle="round"),
    )

    plt.tight_layout()

    figname = f"{plot_path}/HALO-AC3_HALO_{key}_bacardi_ecrad_f_down_solar_above_cloud_all.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot timeseries of ecRad and BACARDI and color with difference between the two
for key in tqdm(keys):
    norm = colors.TwoSlopeNorm(vcenter=0)
    _, ax = plt.subplots(figsize=(16 * h.cm, 9 * h.cm))

    above_sel = bahamas_ds_res[key].IRS_ALT > (bahamas_ds_res[key].IRS_ALT.median() - 500)
    bacardi_res = bacardi_ds_res[key]
    bacardi_plot = bacardi_res.where(bacardi_res.alt > (bacardi_res.alt.median() - 500))
    ecrad_ds = ecrad_dicts[key]["v15"]
    ecrad_plot = ecrad_ds.flux_dn_direct_sw.where(above_sel)

    # actual plotting
    difference = ecrad_plot - bacardi_plot["F_down_solar"]
    ecrad_plot = ecrad_plot.where(~np.isnan(difference), drop=True)
    bacardi_plot = bacardi_plot.where(~np.isnan(difference), drop=True)
    difference = difference.where(~np.isnan(difference), drop=True)
    rmse = np.sqrt(np.mean(difference ** 2)).to_numpy()
    bias = np.nanmean(difference.to_numpy())

    sc = ax.scatter(bacardi_plot.time, bacardi_plot.F_down_solar_diff,
                    c=difference, cmap=cm.pride, norm=norm,
                    marker=".", ls="-",
                    label="BACARDI")
    ax.plot(ecrad_plot.time, ecrad_plot, marker=".", ms=2.5, ls="", label="ecRad")
    plt.colorbar(sc, label="ecRad - BACARDI (W$\,$m$^{-2}$)")
    t_extend = pd.to_timedelta((bacardi_plot.time[-1] - bacardi_plot.time[0]).to_numpy())
    h.set_xticks_and_xlabels(ax, time_extend=t_extend)
    ax.set(
        title=meta.flight_names[key],
        xlabel="Time (UTC)",
        ylabel="Solar downward irradiance (W$\,$m$^{-2}$)",
    )
    ax.legend()
    ax.grid()
    ax.text(
        0.025,
        0.5,
        f"n: {sum(~np.isnan(bacardi_plot['F_down_solar'])):.0f}\n"
        f"RMSE: {rmse:.0f} {h.plot_units['flux_dn_sw']}\n"
        f"Bias: {bias:.0f} {h.plot_units['flux_dn_sw']}",
        ha="left",
        va="top",
        transform=ax.transAxes,
        bbox=dict(fc="white", ec="black", alpha=0.8, boxstyle="round"),
    )
    plt.tight_layout()

    figname = f"{plot_path}/HALO-AC3_HALO_{key}_bacardi_ecrad_f_down_solar_timeseries_above_cloud_all.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot difference between ecRad and BACARDI against heading in a polar plot
plt.rc("font", size=9)
for key in keys:
    norm = colors.TwoSlopeNorm(vcenter=0)
    _, ax = plt.subplots(figsize=(16 * h.cm, 16 * h.cm), subplot_kw={'projection': 'polar'})

    above_sel = bahamas_ds_res[key].IRS_ALT > (bahamas_ds_res[key].IRS_ALT.median() - 500)
    bacardi_res = bacardi_ds_res[key]
    bacardi_plot = bacardi_res.where(bacardi_res.alt > (bacardi_res.alt.median() - 500))
    ecrad_ds = ecrad_dicts[key]["v15"]
    ecrad_plot = ecrad_ds.flux_dn_direct_sw.where(above_sel)

    difference = ecrad_plot - bacardi_plot["F_down_solar"]
    ecrad_plot = ecrad_plot.where(~np.isnan(difference), drop=True)
    bacardi_plot = bacardi_plot.where(~np.isnan(difference), drop=True)
    difference = difference.where(~np.isnan(difference), drop=True)
    rmse = np.sqrt(np.mean(difference ** 2)).to_numpy()
    bias = np.nanmean(difference.to_numpy())
    bahamas_plot = np.deg2rad(bahamas_ds[key].IRS_HDG.sel(time=difference.time))

    # actual plotting
    plot = ax.scatter(bahamas_plot, difference, marker=".",
                      c=difference, cmap=cm.pride, norm=norm)
    plt.colorbar(plot, label="ecRad - BACARDI (W$\,$m$^{-2}$)", shrink=0.7)
    ax.set(
        title=f"{meta.flight_names[key]} (> {(bacardi_res.alt.median() - 500) / 1000:.1f} km)",
        xlabel="Heading (deg)",
        theta_offset=np.pi/2,
        theta_direction=-1
    )
    ax.grid(True)
    plt.tight_layout()

    figname = f"{plot_path}/HALO-AC3_HALO_{key}_bacardi-ecrad_f_down_solar_vs_heading_above_cloud_all.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot difference between ecRad and BACARDI against heading in one polar plot
plt.rc("font", size=9)
norm = colors.TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20)
_, ax = plt.subplots(figsize=(16 * h.cm, 16 * h.cm), subplot_kw={'projection': 'polar'})
for key in keys:
    above_sel = (bahamas_ds[key].IRS_ALT > (bahamas_ds_res[key].IRS_ALT.median() - 500))
    bacardi_res = bacardi_ds_res[key]
    bacardi_plot = bacardi_res.where(bacardi_res.alt > (bacardi_res.alt.median() - 500))
    ecrad_ds = ecrad_dicts[key]["v15"]
    ecrad_plot = ecrad_ds.flux_dn_direct_sw.where(above_sel)

    difference = ecrad_plot - bacardi_plot["F_down_solar"]
    ecrad_plot = ecrad_plot.where(~np.isnan(difference), drop=True)
    bacardi_plot = bacardi_plot.where(~np.isnan(difference), drop=True)
    difference = difference.where(~np.isnan(difference), drop=True)
    rmse = np.sqrt(np.mean(difference ** 2)).to_numpy()
    bias = np.nanmean(difference.to_numpy())
    bahamas_plot = np.deg2rad(bahamas_ds[key].IRS_HDG.sel(time=difference.time))

    # actual plotting
    plot = ax.scatter(bahamas_plot, difference, marker=".",
                      c=difference, cmap=cm.pride, norm=norm)

plt.colorbar(plot, label="ecRad - BACARDI (W$\,$m$^{-2}$)", shrink=0.7)

ax.set(
    title=f"All Flights (> Median Altitude - 0.5 km)",
    xlabel="Heading (deg)",
    theta_offset=np.pi/2,
    theta_direction=-1
)
ax.grid(True)
plt.tight_layout()

figname = f"{plot_path}/HALO-AC3_HALO_all_bacardi-ecrad_f_down_solar_vs_heading_above_cloud_all.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot difference between libRadtran and BACARDI against heading in one polar plot
plt.rc("font", size=9)
norm = colors.TwoSlopeNorm(vmin=-10, vcenter=0, vmax=20)
_, ax = plt.subplots(figsize=(16 * h.cm, 16 * h.cm), subplot_kw={'projection': 'polar'})
for key in keys:
    above_sel = (bahamas_ds[key].IRS_ALT > (bahamas_ds_res[key].IRS_ALT.median() - 500))
    bacardi_res = bacardi_ds[key]
    bacardi_plot = bacardi_res.where(bacardi_res.alt > (bacardi_res.alt.median() - 500))

    difference = bacardi_plot["F_down_solar_sim"] - bacardi_plot["F_down_solar"]
    bacardi_plot = bacardi_plot.where(~np.isnan(difference), drop=True)
    difference = difference.where(~np.isnan(difference), drop=True)
    rmse = np.sqrt(np.mean(difference ** 2)).to_numpy()
    bias = np.nanmean(difference.to_numpy())
    bahamas_plot = np.deg2rad(bahamas_ds[key].IRS_HDG.sel(time=difference.time))

    # actual plotting
    plot = ax.scatter(bahamas_plot, difference, marker=".",
                      c=difference, cmap=cm.pride, norm=norm)

plt.colorbar(plot, label="libRadtran - BACARDI (W$\,$m$^{-2}$)", shrink=0.7)

ax.set(
    title=f"All Flights (> 10 km)",
    xlabel="Heading (deg)",
    theta_offset=np.pi/2,
    theta_direction=-1
)
ax.grid(True)
plt.tight_layout()

figname = f"{plot_path}/HALO-AC3_HALO_all_bacardi-libradtran_f_down_solar_vs_heading_above_cloud_all.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot difference between ecRad and BACARDI against heading in one polar plot and color by flight
plt.rc("font", size=9)
cols = cm.take_cmap_colors("tab20", 20)
_, ax = plt.subplots(figsize=(16 * h.cm, 16 * h.cm), subplot_kw={'projection': 'polar'})
for i, key in enumerate(keys):
    above_sel = bahamas_ds_res[key].IRS_ALT > (bahamas_ds_res[key].IRS_ALT.median() - 500)
    bacardi_res = bacardi_ds_res[key]
    bacardi_plot = bacardi_res["F_down_solar"].where(above_sel)
    ecrad_ds = ecrad_dicts[key]["v15"]
    ecrad_plot = ecrad_ds.flux_dn_direct_sw.where(above_sel)

    difference = ecrad_plot - bacardi_plot
    ecrad_plot = ecrad_plot.where(~np.isnan(difference), drop=True)
    bacardi_plot = bacardi_plot.where(~np.isnan(difference), drop=True)
    difference = difference.where(~np.isnan(difference), drop=True)
    rmse = np.sqrt(np.mean(difference ** 2)).to_numpy()
    bias = np.nanmean(difference.to_numpy())
    bahamas_plot = np.deg2rad(bahamas_ds[key].IRS_HDG.sel(time=difference.time))

    # actual plotting
    plot = ax.scatter(bahamas_plot, difference, marker=".", label=key, color=cols[i])

angle = np.deg2rad(0)
ax.legend(loc="lower left",
          bbox_to_anchor=(.56 + np.cos(angle)/2, .5 + np.sin(angle)/2))
ax.set(
    title=f"All Flights (> Median Altitude - 0.5 km)",
    xlabel="Heading (deg)",
    theta_offset=np.pi/2,
    theta_direction=-1
)
ax.grid(True)
plt.tight_layout()

figname = f"{plot_path}/HALO-AC3_HALO_all_bacardi-ecrad_f_down_solar_vs_heading_above_cloud_all_labeled.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

# %% plot difference between ecRad and BACARDI against latitude in one polar plot
plt.rc("font", size=9)
cols = cm.take_cmap_colors("tab20", 20)
_, ax = plt.subplots(figsize=(16 * h.cm, 9 * h.cm))
for i, key in enumerate(keys):
    above_sel = (bahamas_ds[key].IRS_ALT > (bahamas_ds_res[key].IRS_ALT.median() - 500))
    bacardi_res = bacardi_ds_res[key]
    bacardi_plot = bacardi_res.where(bacardi_res.alt > (bacardi_res.alt.median() - 500))
    ecrad_ds = ecrad_dicts[key]["v15"]
    ecrad_plot = ecrad_ds.flux_dn_direct_sw.where(above_sel)

    difference = ecrad_plot - bacardi_plot["F_down_solar"]
    ecrad_plot = ecrad_plot.where(~np.isnan(difference), drop=True)
    bacardi_plot = bacardi_plot.where(~np.isnan(difference), drop=True)
    difference = difference.where(~np.isnan(difference), drop=True)
    rmse = np.sqrt(np.mean(difference ** 2)).to_numpy()
    bias = np.nanmean(difference.to_numpy())
    bahamas_plot = bahamas_ds[key].IRS_LAT.sel(time=difference.time)

    # actual plotting
    plot = ax.scatter(bahamas_plot, difference, marker=".", color=cols[i], label=key)

ax.set(
    title=f"All Flights (> Median Altitude - 0.5 km)",
    xlabel="Latitude (deg)",
    ylabel="ecRad - BACARDI (W$\,$m$^{-2}$)"
)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
ax.grid(True)
plt.tight_layout()

figname = f"{plot_path}/HALO-AC3_HALO_all_bacardi-ecrad_f_down_solar_vs_latitude_above_cloud_all.png"
plt.savefig(figname, dpi=300)
plt.show()
plt.close()

