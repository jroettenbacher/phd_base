#!/usr/bin/env python
"""Plot and save SMART quicklooks of dark current corrected and calibrated measurements for one flight

*author*: Johannes Röttenbacher
"""

if __name__ == "__main__":
    # %% import modules
    import pylim.helpers as h
    import os
    import xarray as xr
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    import cartopy
    import cartopy.crs as ccrs
    cm = 1 / 2.54  # conversion factor for centimeter to inch

    # %% set paths
    campaign = "cirrus-hl"
    prop = "Fup"  # Fdw or Fup
    if campaign == "halo-ac3":
        from pylim import halo_ac3 as meta
    else:
        from pylim import cirrus_hl as meta
    flight_keys = list(meta.flight_names.values()) if campaign == "halo-ac3" else list(meta.flight_numbers.keys())
    ql_path = h.get_path("all", instrument="quicklooks", campaign=campaign)
    flights = flight_keys[3:-1] if campaign == "halo-ac3" else flight_keys[1:]
    # flights = ["HALO-AC3_20220411_HALO_RF17"]  # single flight mode
    for flight in flights:
        wavelengths = [422, 532, 648, 858, 1240, 1640]  # five wavelengths to plot individually
        calibrated_path = h.get_path("calibrated", flight, campaign)  # path to calibrated nc files
        plot_path = calibrated_path  # output path of plot

        # %% get metadata
        flight_no = flight[-4:] if campaign == "halo-ac3" else meta.flight_numbers[flight]
        date = flight[9:17] if campaign == "halo-ac3" else flight[7:16]

        # %% read in calibrated files
        if campaign == "halo-ac3":
            file = f"{campaign.swapcase()}_HALO_SMART_spectral-irradiance-{prop}_{date}_{flight_no}_v1.0.nc"
        else:
            file = f"{campaign.swapcase()}_{flight_no}_{date}_HALO_SMART_spectral-irradiance-{prop}_v1.0.nc"
        filepath = os.path.join(calibrated_path, file)
        ds = xr.open_dataset(filepath)
        F_cor = ds[f"{prop}_cor"]  # extract corrected F
        time_range = pd.to_timedelta((F_cor.time[-1] - F_cor.time[0]).values)  # get time range for time axis formatting

        # %% set plotting aesthetics
        plt.rcdefaults()
        h.set_cb_friendly_colors()
        font = {"size": 14, "family": "serif"}
        plt.rc("font", **font)

        # %% calculate statistics
        F_int = F_cor.integrate("wavelength")
        F_int_flat = F_int.values.flatten()  # flatten 2D array for statistics
        F_int_flat = F_int_flat[~np.isnan(F_int_flat)]  # drop nans for boxplot
        Fmin, Fmax, Fmean, Fmedian, Fstd = F_int.min(), F_int.max(), F_int.mean(), F_int.median(), F_int.std()
        stats_text = f"Statistics \n(W$\,$m$^{{-2}}$)\nMin: {Fmin:.2f}\nMax: {Fmax:.2f}\nMean: {Fmean:.2f}\nMedian: {Fmedian:.2f}" \
                     f"\nStd: {Fstd:.2f}"

        fig = plt.figure(figsize=(25 * cm, 45 * cm), constrained_layout=True)
        gs0 = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[3, 1])
        gs01 = gs0[0, 0].subgridspec(4, 1, height_ratios=[1, 3, 1, 1])
        gs02 = gs0[0, 1].subgridspec(4, 1, height_ratios=[1, 4, 1, 1])

        # 5 wavelength plot - first subrow, first column
        ax1 = fig.add_subplot(gs01[0])
        for wavelength in wavelengths:
            F_cor.sel(wavelength=wavelength).plot(ax=ax1, x="time", label=f"{wavelength}$\,$nm")
        ax1.set_xlabel("")
        ax1.set_ylabel("Irradiance \n(W$\,$m$^{-2}\,$nm$^{-1}$)")
        ax1.set_title("")
        ax1.grid()
        h.set_xticks_and_xlabels(ax1, time_range)

        # wavelength-time plot - second subrow, first column
        ax = fig.add_subplot(gs01[1])
        F_cor.plot(ax=ax, x="time", robust=True, cmap="inferno",
                   cbar_kwargs={"location": "bottom", "label": "Irradiance (W$\,$m$^{-2}\,$nm$^{-1}$)", "pad": 0.01})
        ax.set_xlabel("")
        ax.set_ylabel("Wavelength (nm)")
        h.set_xticks_and_xlabels(ax, time_range)

        # sza and saa - third subrow, first column
        ax = fig.add_subplot(gs01[2])
        ds.sza.plot(ax=ax, label="SZA")
        ax2 = ax.twinx()
        ds.saa.plot(ax=ax2, label="SAA", color="#CC6677")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax2.set_ylabel("")
        ax.grid()
        h.set_xticks_and_xlabels(ax, time_range)
        # add legend
        handles, _ = ax.get_legend_handles_labels()
        handles2, _ = ax2.get_legend_handles_labels()
        handles.append(handles2[0])
        ax.legend(handles=handles)
        # color y axis
        ax2.spines["left"].set_color("#88CCEE")
        ax2.spines["right"].set_color("#CC6677")
        ax2.spines["left"].set_linewidth(3)
        ax2.spines["right"].set_linewidth(3)
        ax.tick_params(colors="#88CCEE", axis="y", which="both")
        ax2.tick_params(colors="#CC6677", axis="y", which="both", direction="in", pad=-30)

        # roll angle with stabbi flag - fourth subrow, first column
        ax = fig.add_subplot(gs01[3])
        ds["roll"].plot(ax=ax, x="time", color="#999933")
        if prop == "Fdw":
            stabbi = ds.stabilization_flag
            stabbi_working = stabbi.where(stabbi == 0, drop=True) + 2
            stabbi_not_working = stabbi.where(stabbi == 1, drop=True) - 1.05 + 2
            stabbi_off = stabbi.where(stabbi == 2, drop=True) - 0.1
            if campaign == "halo-ac3":
                if flight in meta.stabilized_flights and flight not in meta.stabbi_offs:
                    stabbi_working.plot(ls="", marker="s", markersize=2, label="working", color="#44AA99")
                    stabbi_not_working.plot(ls="", marker="s", markersize=2, label="not working", color="#882255")
                elif flight in meta.stabilized_flights and flight in meta.stabbi_offs:
                    stabbi_working.plot(ls="", marker="s", markersize=2, label="working", color="#44AA99")
                    stabbi_not_working.plot(ls="", marker="s", markersize=2, label="not working", color="#882255")
                    stabbi_off.plot(ls="", marker="s", markersize=2, label="off", color="#888888")
            else:
                if flight in meta.stabilized_flights:
                    stabbi_working.plot(ls="", marker="s", markersize=2, label="working", color="#44AA99")
                    stabbi_not_working.plot(ls="", marker="s", markersize=2, label="not working", color="#882255")

        handles_roll, labels_roll = ax.get_legend_handles_labels()
        h.set_xticks_and_xlabels(ax, time_range)
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Roll Angle (°)")
        ax.grid()
        ax.set_ylim([-2, 2.02])

        # legend for 5 wavelength plot - first subrow, second column
        ax = fig.add_subplot(gs02[0])
        ax.axis("off")
        handles, labels = ax1.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=[0.2, 1.1], ncol=1)

        # boxplot - second subrow, second column
        ax = fig.add_subplot(gs02[1])
        ax.boxplot(F_int_flat, vert=True, labels=[""], widths=0.6)
        ax.set_ylabel("Integrated Irradiance (W$\,$m$^{-2}$)")
        ax.grid()

        # textbox with statistics - third subrow, second column
        ax = fig.add_subplot(gs02[2])
        ax.axis("off")  # hide axis
        ax.text(0.5, 1, stats_text, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)

        # legend for roll angle and Stabbi
        ax = fig.add_subplot(gs02[3])
        ax.axis("off")
        if flight in meta.stabilized_flights and prop == "Fdw":
            ax.legend(handles=handles_roll, labels=labels_roll, markerscale=6, title="Stabilization",
                      loc='upper center', bbox_to_anchor=[0.2, 1.1])
        elif flight in meta.unstabilized_flights and prop == "Fdw":
            ax.text(0.5, 1, "Stabilization\nwas turned off!", horizontalalignment='center', verticalalignment='top',
                    transform=ax.transAxes)
        else:
            ax.text(0.5, 1, "No Stabilization\navailable!", horizontalalignment='center', verticalalignment='top',
                    transform=ax.transAxes)

        # map of flight track - second row, both columns
        data_crs = ccrs.PlateCarree()
        if campaign == "halo-ac3":
            name_1, name_2 = "Kiruna", "Longyearbyen"
            x_1, y_1 = meta.coordinates[name_1]
            x_2, y_2 = meta.coordinates[name_2]
        else:
            name_1 = "EDMO"
            name_2 = meta.stop_over_locations[flight] if flight in meta.stop_over_locations else None
            x_1, y_1 = meta.coordinates[name_1]
            x_2, y_2 = meta.coordinates[name_2] if name_2 is not None else (x_1, y_1)
        # get extent of map plot
        pad = 2
        llcrnlat = ds.lat.min(skipna=True) - pad
        llcrnlon = ds.lon.min(skipna=True) - pad
        urcrnlat = ds.lat.max(skipna=True) + pad
        urcrnlon = ds.lon.max(skipna=True) + pad
        extent = [llcrnlon, urcrnlon, llcrnlat, urcrnlat]
        if campaign == "halo-ac3":
            ax = fig.add_subplot(gs0[1, :], projection=ccrs.NorthPolarStereo())
        else:
            ax = fig.add_subplot(gs0[1, :], projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.add_feature(cartopy.feature.BORDERS)
        ax.set_extent(extent, crs=data_crs)
        if campaign == "halo-ac":
            gl = ax.gridlines(crs=data_crs, draw_labels=True, x_inline=True, y_inline=True)
        else:
            gl = ax.gridlines(crs=data_crs, draw_labels=True, x_inline=False, y_inline=False)
            gl.top_labels = False
        # plot flight track
        points = ax.scatter(ds["lon"], ds["lat"], s=2, c="#6699CC", transform=data_crs)
        # add point for Kiruna and Longyearbyen or EDMO and second airport
        ax.plot(x_1, y_1, 'ok', transform=data_crs)
        ax.text(x_1 + 0.1, y_1 + 0.1, name_1, fontsize=8, transform=data_crs,
                path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])
        ax.plot(x_2, y_2, 'ok', transform=data_crs)
        ax.text(x_2 + 0.1, y_2 + 0.1, name_2, fontsize=8, transform=data_crs,
                path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])

        direction = "downward" if prop == "Fdw" else "upward"
        fig.suptitle(f"SMART {direction} Irradiance for {flight}")
        if campaign == "halo-ac3":
            figname = f"{campaign.swapcase()}_SMART_calibrated-{prop}_quicklook_{date}_{flight_no}.png"
        else:
            figname = f"{campaign.swapcase()}_{flight_no}_{date}_HALO_SMART_calibrated-{prop}_quicklook.png"
        plt.savefig(f"{plot_path}/{figname}", dpi=300)
        plt.savefig(f"{ql_path}/{figname}", dpi=300)
        print(f"Saved {figname}")
        plt.show()
        plt.close()
