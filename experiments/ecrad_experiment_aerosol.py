#!/usr/bin/env python
"""
| *author:* Johannes Röttenbacher
| *created:* 12-10-2023

Comparison between ecRad simulation without and with CAMS monthly aerosol and trace gases.

* ``IFS_namelist_jr_20220411_v15.1.nam``: for flight HALO-AC3_20220411_HALO_RF17 with Fu-IFS ice model
* ``IFS_namelist_jr_20220411_v30.1.nam``: for flight HALO-AC3_20220411_HALO_RF17 with Fu-IFS ice model and aersosols from monthly CAMS climatology

Turning on the aerosol in ecRad leads to slightly more extinction throughout the atmosphere as can be seen in the difference between v30.1 and v15.1 below.

.. figure:: figures/experiment_aerosol/HALO-AC3_20220411_HALO_RF17_ecrad_diff_flux_dn_sw_along_track.png

    Difference in solar downward irradiance between v30.1 (with aerosol) and v15.1 (reference).

The solar upward irradiance shows an interesting pattern, where there is more upward irradiance over open ocean in v30.1 and less over sea ice.
This can be explained by a higher concentration of sea salt over the open ocean leading to a slightly increased reflectivity.

.. figure:: figures/experiment_aerosol/HALO-AC3_20220411_HALO_RF17_ecrad_diff_flux_up_sw_along_track.png

    Difference in solar upward irradiance between v30.1 (with aerosol) and v15.1 (reference).

"""

if __name__ == '__main__':
# %% import modules
    import pylim.helpers as h
    import pylim.halo_ac3 as meta
    import ac3airborne
    from ac3airborne.tools import flightphase
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import colors
    import pandas as pd
    from tqdm import tqdm
    import cmasher as cmr

    # plotting variables
    cbc = h.get_cb_friendly_colors()
    h.set_cb_friendly_colors()
    plt.rc('font', size=12)

# %% set paths
    campaign = 'halo-ac3'
    key = 'RF17'
    ecrad_versions = ['v15.1', 'v30.1']
    flight = meta.flight_names[key]
    date = flight[9:17]

    plot_path = f'{h.get_path('plot', flight, campaign)}/{flight}/experiment_aerosol'
    fig_path = './docs/figures/experiment_aerosol'
    h.make_dir(plot_path)
    h.make_dir(fig_path)
    ecrad_path = f'{h.get_path('ecrad', campaign=campaign)}/{date}'
    bahamas_path = h.get_path('bahamas', flight, campaign)

# %% get flight segments for case study period
    segmentation = ac3airborne.get_flight_segments()['HALO-AC3']['HALO'][f'HALO-AC3_HALO_{key}']
    segments = flightphase.FlightPhaseFile(segmentation)
    above_cloud, below_cloud = dict(), dict()
    if key == 'RF17':
        above_cloud['start'] = segments.select('name', 'high level 7')[0]['start']
        above_cloud['end'] = segments.select('name', 'high level 8')[0]['end']
        below_cloud['start'] = segments.select('name', 'high level 9')[0]['start']
        below_cloud['end'] = segments.select('name', 'high level 10')[0]['end']
        above_slice = slice(above_cloud['start'], above_cloud['end'])
        below_slice = slice(below_cloud['start'], below_cloud['end'])
        case_slice = slice(pd.to_datetime('2022-04-11 10:30'), pd.to_datetime('2022-04-11 12:29'))
    else:
        above_cloud['start'] = segments.select('name', 'polygon pattern 1')[0]['start']
        above_cloud['end'] = segments.select('name', 'polygon pattern 1')[0]['parts'][-1]['start']
        below_cloud['start'] = segments.select('name', 'polygon pattern 2')[0]['start']
        below_cloud['end'] = segments.select('name', 'polygon pattern 2')[0]['end']
        above_slice = slice(above_cloud['start'], above_cloud['end'])
        below_slice = slice(below_cloud['start'], below_cloud['end'])
        case_slice = slice(above_cloud['start'], below_cloud['end'])

    time_extend_cs = below_cloud['end'] - above_cloud['start']  # time extend for case study

# %% read in bahamas data
    bahamas = xr.open_dataset(f'{bahamas_path}/HALO-AC3_HALO_BAHAMAS_{date}_{key}_v1_1Min.nc')

# %% read in ecrad data
    ecrad_dict, ecrad_org = dict(), dict()

    for k in ecrad_versions:
        ds = xr.open_dataset(f'{ecrad_path}/ecrad_merged_inout_{date}_{k}.nc')
        ecrad_org[k] = ds.copy(deep=True)
        ds = ds.sel(column=0, drop=True)
        ecrad_dict[k] = ds.copy()

# %% set plotting options
    var = 'flux_up_sw'
    v = 'diff'
    band = None
    aer_type = slice(1, 11)

# %% prepare data set for plotting
    band_str = f'_band{band}' if band is not None else ''

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
    cmap.set_bad(color='white')
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

    if v == 'diff':
        cmap = cmr.fusion
        norm = colors.TwoSlopeNorm(vcenter=0)

    # prepare ecrad dataset for plotting
    sf = h.scale_factors[var] if var in h.scale_factors else 1
    if v == 'diff':
        # calculate difference between simulations
        ds = ecrad_dict['v15.1']
        ecrad_ds_diff = ecrad_dict['v30.1'][var] - ds[var]
        ecrad_plot = ecrad_ds_diff.where((ds[var] != 0) | (~np.isnan(ds[var]))) * sf
    else:
        ecrad_plot = ecrad_dict[v][var] * sf

    # add new z axis mean pressure altitude
    if 'half_level' in ecrad_plot.dims:
        new_z = ecrad_dict['v15.1']['press_height_hl'].mean(dim='time') / 1000
    else:
        new_z = ecrad_dict['v15.1']['press_height_full'].mean(dim='time') / 1000

    ecrad_plot_new_z = list()
    for t in tqdm(ecrad_plot.time, desc='New Z-Axis'):
        tmp_plot = ecrad_plot.sel(time=t)
        if 'half_level' in tmp_plot.dims:
            tmp_plot = tmp_plot.assign_coords(
                half_level=ecrad_dict['v15.1']['press_height_hl'].sel(time=t, drop=True).to_numpy() / 1000)
            tmp_plot = tmp_plot.rename(half_level='height')

        else:
            tmp_plot = tmp_plot.assign_coords(
                level=ecrad_dict['v15.1']['press_height_full'].sel(time=t, drop=True).to_numpy() / 1000)
            tmp_plot = tmp_plot.rename(level='height')

        tmp_plot = tmp_plot.interp(height=new_z.to_numpy())
        ecrad_plot_new_z.append(tmp_plot)

    ecrad_plot = xr.concat(ecrad_plot_new_z, dim='time')
    # filter very low to_numpy()
    # ecrad_plot = ecrad_plot.where(np.abs(ecrad_plot) > 0.001)

    # select time height slice
    time_sel = slice(pd.Timestamp('2022-04-11 08:00'), pd.Timestamp('2022-04-11 17:00'))
    if 'band_sw' in ecrad_plot.dims:
        dim3 = 'band_sw'
        dim3 = dim3 if dim3 in ecrad_plot.dims else None
        ecrad_plot = ecrad_plot.sel({'time': time_sel, 'height': slice(13, 0), f'{dim3}': band})
    elif 'aer_type' in ecrad_plot.dims:
        dim3 = 'aer_type'
        dim3 = dim3 if dim3 in ecrad_plot.dims else None
        ecrad_plot = ecrad_plot.sel({'time': time_sel, 'height': slice(13, 0), f'{dim3}': aer_type})
        if len(ecrad_plot[dim3]) > 1:
            ecrad_plot = ecrad_plot.sum(dim=dim3)
    else:
        ecrad_plot = ecrad_plot.sel(time=time_sel, height=slice(13, 0))

    time_extend = pd.to_timedelta((ecrad_plot.time[-1] - ecrad_plot.time[0]).to_numpy())

# %% plot 2D IFS variables along flight track
    _, ax = plt.subplots(figsize=h.figsize_wide, layout='constrained')
    # ecrad 2D field
    ecrad_plot.plot(x='time', y='height', cmap=cmap, ax=ax, robust=robust,
                    vmin=vmin, vmax=vmax, alpha=alpha, norm=norm,
                    cbar_kwargs={'pad': 0.04, #'label': f'{h.cbarlabels[var]} ({h.plot_units[var]})',
                                 'ticks': ticks})
    if lines is not None:
        # add contour lines
        ct = ax.contour(ecrad_plot.time, ecrad_plot.height, ecrad_plot.to_numpy().T, levels=lines, linestyles='--',
                        colors='k',
                        linewidths=lw)
        ct.clabel(fontsize=ct_fs, inline=1, inline_spacing=0, fmt='%i', rightside_up=True, use_clabeltext=True)

    ax.set_title(f'{key} IFS/ecRad input/output along Flight Track - {v}')
    ax.set_ylabel('Altitude (km)')
    ax.set_xlabel('Time (UTC)')
    h.set_xticks_and_xlabels(ax, time_extend)
    figname = f'{plot_path}/{flight}_ecrad_{v}_{var}{band_str}_along_track.png'
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    figname = f'{fig_path}/{flight}_ecrad_{v}_{var}{band_str}_along_track.png'
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# %% plot histogram
    xlabel = 'Difference v30.1 - v15.1' if v == 'diff' else v
    flat_array = ecrad_plot.to_numpy().flatten()
    _, ax = plt.subplots(figsize=h.figsize_wide, layout='constrained')
    hist = ax.hist(flat_array, bins=20)
    ax.set(xlabel=f'{h.cbarlabels[var]} {xlabel} ({h.plot_units[var]})',
           ylabel='Number of Occurrence')
    ax.grid()
    ax.set_yscale('log')
    figname = f'{plot_path}/{flight}_ecrad_{v}_{var}{band_str}_hist.png'
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    figname = f'{fig_path}/{flight}_ecrad_{v}_{var}{band_str}_hist.png'
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

