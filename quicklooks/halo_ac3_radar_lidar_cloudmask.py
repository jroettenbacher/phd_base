#!/usr/bin/env python
"""
| *author:* Johannes Röttenbacher
| *created*: 20-03-2023

Produces a radar lidar cloudmask for |haloac3|

**Input**:

- flight key (e.g RF17)
- save figure flag

"""

if __name__ == "__main__":
    # %% import modules and set paths
    import pylim.helpers as h
    import pylim.halo_ac3 as meta
    import ac3airborne
    import sys
    sys.path.append('./larda')
    from larda.pyLARDA.spec2mom_limrad94 import despeckle
    import os
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import colors
    import pandas as pd
    from tqdm import tqdm

    cbc = h.get_cb_friendly_colors()
    cm = 1 / 2.54

    # %% user input
    campaign = "halo-ac3"
    halo_key = "RF17"
    flight = meta.flight_names[halo_key]
    date = flight[9:17]
    savefig = False

    # %% set up metadata for access to HALO-AC3 cloud
    kwds = {'simplecache': dict(same_names=True)}
    credentials = {"user": os.environ.get("AC3_CLOUD_USER"), "password": os.environ.get("AC3_CLOUD_PASSWORD")}
    cat = ac3airborne.get_intake_catalog()["HALO-AC3"]["HALO"]

    # %% set paths
    plot_path = f"{h.get_path('plot', flight, campaign)}/radar_lidar_flag"
    radar_path = h.get_path("hamp_mira", flight, campaign)
    radar_file = "radar_20220411_v1.6.nc"
    lidar_path = h.get_path("wales", flight, campaign)
    bsrgl_file = "HALO-AC3_HALO_WALES_bsrgl_20220411_RF17_V1.nc"
    bahamas_path = h.get_path("bahamas", flight, campaign)
    bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{halo_key}_v1_1s.nc"

    # %% read in data from HALO-AC3 cloud
    dropsonde_ds = cat["DROPSONDES_GRIDDED"][f"HALO-AC3_HALO_{halo_key}"](storage_options=kwds, **credentials).to_dask()
    dropsonde_ds["alt"] = dropsonde_ds.alt / 1000  # convert altitude to km

    # %% read in bahamas data
    bahamas_ds = xr.open_dataset(f"{bahamas_path}/{bahamas_file}")

    # %% plotting meta
    figsize_wide = (24 * cm, 12 * cm)
    time_extend = pd.to_timedelta((bahamas_ds.time[-1] - bahamas_ds.time[0]).values)  # get time extend for x-axis labeling

    # %% read in radar data
    radar_ds = xr.open_dataset(f"{radar_path}/{radar_file}")
    # filter -888 values
    radar_ds["dBZg"] = radar_ds.dBZg.where(np.isnan(radar_ds.radar_flag) & ~radar_ds.dBZg.isin(-888))

    # %% create radar mask and despeckle radar data
    radar_ds["mask"] = ~np.isnan(radar_ds["dBZg"])
    radar_mask = ~radar_ds["mask"].values
    for n in tqdm(range(2)):
        # despeckle 2 times
        radar_mask = despeckle(radar_mask, 50)  # despeckle again
        # plt.pcolormesh(radar_mask.T)
        # plt.title(n + 1)
        # plt.savefig(f"{plot_path}/tmp/radar_despeckle_{n + 1}.png")
        # plt.close()

    radar_ds["spklmask"] = (["time", "height"], radar_mask)

    # %% use despeckle the reverse way to fill signal gaps in radar data and add it as a mask
    radar_mask = ~radar_ds["spklmask"].values
    n = 0
    for n in tqdm(range(17)):
        # fill gaps 17 times
        radar_mask = despeckle(radar_mask, 50)  # fill gaps again
        plt.pcolormesh(radar_mask.T)
        plt.title(n + 1)
        plt.savefig(f"{plot_path}/tmp/radar_fill_gaps_{n + 1}.png")
        plt.close()

    radar_ds["fill_mask"] = (["time", "height"], radar_mask)

    # %% read in lidar data
    lidar_ds = xr.open_dataset(f"{lidar_path}/{bsrgl_file}")  # sensitive to clouds
    lidar_ds_res = xr.open_dataset(f"{lidar_path}/{bsrgl_file.replace('.nc', '_1s.nc')}").reset_coords("altitude")
    # lidar_ds_res = lidar_ds.resample(time="1S").asfreq()
    # lidar_ds_res.to_netcdf(f"{lidar_path}/{bsrgl_file.replace('.nc', '_1s_.nc')}")
    # convert lidar data to radar convention: [time, height], ground = 0m
    lidar_ds_res = lidar_ds_res.rename(range="height").transpose("time", "height")
    lidar_height = lidar_ds_res.height
    # flip height coordinate
    lidar_ds_res = lidar_ds_res.assign_coords(height=np.flip(lidar_height)).isel(height=slice(None, None, -1))

    # %% despeckle lidar data and add a speckle mask to the data set
    lidar_ds_res["mask"] = ~(lidar_ds_res.backscatter_ratio > 1.1)
    lidar_mask = lidar_ds_res["mask"].values
    n = 0
    for n in tqdm(range(17)):
        # despeckle 17 times
        lidar_mask = despeckle(lidar_mask, 50)  # despeckle again
        # plt.pcolormesh(lidar_mask.T)
        # plt.title(n + 1)
        # plt.savefig(f"{plot_path}/tmp/despeckle_{n + 1}.png")
        # plt.close()

    lidar_ds_res["spklmask"] = (["time", "height"], lidar_mask)

    # %% use despeckle the reverse way to fill signal gaps in lidar data and add it as a mask
    lidar_mask = ~lidar_ds_res["spklmask"].values
    n = 0
    for n in tqdm(range(17)):
        # fill gaps 17 times
        lidar_mask = despeckle(lidar_mask, 50)  # fill gaps again
        # plt.pcolormesh(lidar_mask.T)
        # plt.title(n + 1)
        # plt.savefig(f"{plot_path}/tmp/fill_gaps_{n + 1}.png")
        # plt.close()

    lidar_ds_res["fill_mask"] = (["time", "height"], lidar_mask)

    # %% despeckle fill mask
    lidar_mask = ~lidar_ds_res["fill_mask"].values
    for n in tqdm(range(8)):
        # plt.pcolormesh(lidar_mask.T)
        # plt.title(n + 1)
        # plt.savefig(f"{plot_path}/tmp/fill_gaps_despeckled_{n + 1}.png")
        # plt.close()
        lidar_mask = despeckle(lidar_mask, 50)  # despeckle again

    lidar_ds_res["fill_mask"] = (["time", "height"], ~lidar_mask)

    # %% interpolate lidar data onto radar range resolution
    new_range = radar_ds.height.values
    lidar_ds_res_r = lidar_ds_res.interp(height=new_range)

    # %% plot filtered 1s lidar data whole flight as used for cloud mask
    for mask_value in [1.1]:  # , 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]:
            lidar_plot = lidar_ds_res.backscatter_ratio.where(lidar_ds_res.backscatter_ratio > mask_value)
            _, ax = plt.subplots(figsize=figsize_wide)
            lidar_plot.plot(x="time", y="height", robust=True, cmap="plasma", ax=ax)
            ax.plot(bahamas_ds.time, bahamas_ds["IRS_ALT"], color="k", label="HALO altitude")
            ax.legend(loc=2)
            h.set_xticks_and_xlabels(ax, time_extend)
            ax.set_xlabel("Time (UTC)")
            ax.set_ylabel("Altitude (m)")
            ax.set_title(
                f"{halo_key} WALES backscatter ratio 532 nm low sensitivity\nresampled to 1s, masked with {mask_value}")
            plt.tight_layout()
            figname = f"{plot_path}/{flight}_WALES_backscatter_ratio_532_ls_1s_cloud_mask_{mask_value}.png"
            plt.savefig(figname, dpi=300)
            plt.show()
            plt.close()

    # %% plot despeckled lidar data
    lidar_plot = lidar_ds_res.backscatter_ratio.where(~lidar_ds_res["mask"]).where(~lidar_ds_res["spklmask"])
    _, ax = plt.subplots(figsize=figsize_wide)
    lidar_plot.plot(x="time", y="height", robust=True, cmap="plasma", ax=ax)
    ax.plot(bahamas_ds.time, bahamas_ds["IRS_ALT"], color="k", label="HALO altitude")
    ax.legend(loc=2)
    h.set_xticks_and_xlabels(ax, time_extend)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title(
        f"{halo_key} WALES backscatter ratio 532 nm low sensitivity\nresampled to 1s, masked with 1.1, despeckled")
    plt.tight_layout()
    figname = f"{plot_path}/{flight}_WALES_backscatter_ratio_532_ls_1s_despeckled.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

    # %% plot interpolated lidar data
    lidar_mask = (lidar_ds_res_r["mask"] & lidar_ds_res_r["spklmask"])
    lidar_plot = lidar_ds_res_r.backscatter_ratio.where(~lidar_mask)
    _, ax = plt.subplots(figsize=figsize_wide)
    lidar_plot.plot(x="time", y="height", robust=True, cmap="plasma", ax=ax)
    ax.plot(bahamas_ds.time, bahamas_ds["IRS_ALT"], color="k", label="HALO altitude")
    ax.legend(loc=2)
    h.set_xticks_and_xlabels(ax, time_extend)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title(
        f"{halo_key} WALES backscatter ratio 532 nm low sensitivity\nresampled to 1s, masked with 1.1, despeckled, interpolated")
    plt.tight_layout()
    figname = f"{plot_path}/{flight}_WALES_backscatter_ratio_532_ls_1s_30m.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

    # %% combine radar and lidar mask
    lidar_mask = lidar_ds_res_r["fill_mask"]
    lidar_mask = lidar_mask.where(lidar_mask == 0, 2)
    radar_lidar_mask = radar_ds["mask"] + lidar_mask

    # %% plot combined radar and lidar mask
    plot_ds = radar_lidar_mask
    clabel = [x[0] for x in h._CLABEL["detection_status"]]
    cbar = [x[1] for x in h._CLABEL["detection_status"]]
    clabel = list([clabel[-1], clabel[5], clabel[1], clabel[3]])
    cbar = list([cbar[-1], cbar[5], cbar[1], cbar[3]])
    cmap = colors.ListedColormap(cbar)
    _, ax = plt.subplots(figsize=figsize_wide)
    pcm = plot_ds.plot(x="time", y="height", cmap=cmap, vmin=-0.5, vmax=len(cbar) - 0.5)
    pcm.colorbar.set_ticks(np.arange(len(clabel)), labels=clabel)
    ax.plot(bahamas_ds.time, bahamas_ds["IRS_ALT"], color="k", label="HALO altitude")
    ax.legend(loc=2)
    h.set_xticks_and_xlabels(ax, time_extend)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title(f"{halo_key} combined radar lidar mask")
    plt.tight_layout()
    figname = f"{plot_path}/{flight}_radar_lidar_mask.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

    # %% plot combined radar and lidar mask with dropsonde locations
    plot_ds = radar_lidar_mask
    clabel = [x[0] for x in h._CLABEL["detection_status"]]
    cbar = [x[1] for x in h._CLABEL["detection_status"]]
    clabel = list([clabel[-1], clabel[5], clabel[1], clabel[3]])
    cbar = list([cbar[-1], cbar[5], cbar[1], cbar[3]])
    cmap = colors.ListedColormap(cbar)
    _, ax = plt.subplots(figsize=figsize_wide)
    pcm = plot_ds.plot(x="time", y="height", cmap=cmap, vmin=-0.5, vmax=len(cbar) - 0.5)
    pcm.colorbar.set_ticks(np.arange(len(clabel)), labels=clabel)
    ax.plot(bahamas_ds.time, bahamas_ds["IRS_ALT"], color="k", label="HALO altitude")
    ax.vlines(dropsonde_ds.launch_time.values, 0, 1, transform=ax.get_xaxis_transform(), color=cbc[1],
              label="Dropsonde")
    ax.legend(loc=2)
    h.set_xticks_and_xlabels(ax, time_extend)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title(f"{halo_key} combined radar lidar mask")
    plt.tight_layout()
    figname = f"{plot_path}/{flight}_radar_lidar_mask_dropsondes.png"
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

