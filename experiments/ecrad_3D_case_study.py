#!/usr/bin/env python
"""
| *author:* Johannes Röttenbacher
| *created:* 14-04-2023

Enable 3D parameterizations in the ecRad simulation of RF17 and investigate the impact compared to the normal run.

* ``IFS_namelist_jr_20220411_v15.1.nam``: for flight RF17 with Fu-IFS ice model and filtered for low clouds
* ``IFS_namelist_jr_20220411_v22.1.nam``: for flight RF17 with Fu-IFS ice model and low clouds filtered and 3D parameterizations enabled
* ``IFS_namelist_jr_20220411_v18.1.nam``: for flight RF17 with Baran2016 ice model and low clouds filtered
* ``IFS_namelist_jr_20220411_v24.1.nam``: for flight RF17 with Baran2016 ice model and low clouds filtered and 3D parameterizations enabled

The idea behind the 3D parameterization in SPARTACUS (the solver in ecRad) is to simulate the radiation transport
through cloud sides and the entrapment of radiation.
It can be turned on by setting the namelist options:

    | do_3d_effects = true,
    | n_regions = 3,
    | do_lw_side_emissivity = true,
    | sw_entrapment_name = "Explicit",

This has been done using the Fu-IFS ice model (v22.1) and the Baran2016 ice model (v24.1).
The first comparison with Fu-IFS without 3D effects (v15.1) shows that around :math:`1.5\,W\,m^{-2}` less irradiance is absorbed within the cloud.

.. figure:: figures/3d_analysis/HALO-AC3_20220411_HALO_RF17_ecrad_diffv22_flux_dn_sw_along_track.png

    Difference between Fu-IFS ecRad run without 3D effect parameterization (v15.1) and with (v22.1).

For the Baran2016 comparison a similar picture can be seen but values seem to be a bit smaller.
The histograms (not shown) reinforce this picture.

.. figure:: figures/3d_analysis/HALO-AC3_20220411_HALO_RF17_ecrad_diffv24_flux_dn_sw_along_track.png

    Difference between Baran2016 ecRad run without 3D effect parameterization (v18.1) and with (v24.1).

Comparing these results to the fact that ecRad is underestimating the transmissivity of the cloud enabling the 3D
parameterizations does improve the performance of ecRad slightly.
This is also shown when comparing the solar transmissivity.

.. figure:: figures/3d_analysis/HALO-AC3_20220411_HALO_RF17_ecrad_diffv22_transmissivity_sw_along_track.png

    Difference in solar transmissivity between v15.1 and v22.1.

.. figure:: figures/3d_analysis/HALO-AC3_20220411_HALO_RF17_ecrad_diffv24_transmissivity_sw_along_track.png

    Difference in solar transmissivity between v18.1 and v24.1.

The absolute difference caused by the 3-D parameterization is below 0.1.
However, it is a change in the right direction and can thus be seen as an improvement of the model.

"""

if __name__ == "__main__":
# %% import modules
    import os
    import pylim.helpers as h
    import pylim.halo_ac3 as meta
    import dill
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
    ecrad_versions = ['v15.1', 'v22.1', 'v18.1', 'v24.1']
    flight = meta.flight_names[key]
    date = flight[9:17]

    save_path = h.get_path('plot', flight, campaign)
    plot_path = f'{h.get_path('plot', flight, campaign)}/3d_analysis'
    fig_path = './docs/figures/3d_analysis'
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)
    ecrad_path = f'{h.get_path('ecrad', campaign=campaign)}/{date}'
    bahamas_path = h.get_path('bahamas', flight, campaign)

# %% get flight segments for case study period
    loaded_objects = list()
    filenames = [f'{key}_slices.pkl', f'{key}_above_cloud.pkl', f'{key}_below_cloud.pkl']
    for filename in filenames:
        with open(f'{save_path}/{filename}', 'rb') as f:
            loaded_objects.append(dill.load(f))

    slices = loaded_objects[0]

# %% read in ecrad data
    ecrad_dict, ecrad_org = dict(), dict()

    for k in ecrad_versions:
        ds = xr.open_dataset(f'{ecrad_path}/ecrad_merged_inout_{date}_{k}.nc')
        ds['flux_net_sw'] = ds['flux_dn_sw'] - ds['flux_up_sw']
        ecrad_org[k] = ds.copy(deep=True)
        ds = ds.sel(column=0, drop=True)
        ecrad_dict[k] = ds.copy()


# %% set plotting options
    var = "transmissivity_sw"
    v = "diffv24"
    band = None

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
    xlabels = dict(diffv22="Difference v15.1 - v22.1",
                   diffv24="Difference v18.1 - v24.1")

    # set kwargs
    alpha = alphas[var] if var in alphas else 1
    cmap = h.cmaps[var] if var in h.cmaps else cmr.rainforest_r
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

    if 'diff' in v:
        cmap = cmr.fusion
        norm = colors.TwoSlopeNorm(vcenter=0)

    # prepare ecrad dataset for plotting
    sf = h.scale_factors[var] if var in h.scale_factors else 1
    if v == 'diffv22':
        # calculate difference between simulations
        ds = ecrad_dict['v15.1']
        ecrad_ds_diff = ecrad_dict['v22.1'][var] - ds[var]
        ecrad_plot = ecrad_ds_diff.where((ds[var] != 0) | (~np.isnan(ds[var]))) * sf
    elif v == 'diffv24':
        # calculate difference between simulations
        ds = ecrad_dict['v18.1']
        ecrad_ds_diff = ecrad_dict['v24.1'][var] - ds[var]
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
    time_sel = slices['case'] # slice(pd.Timestamp(f'{date} 10:00'), pd.Timestamp(f'{date} 13:00'))
    if 'band_sw' in ecrad_plot.dims:
        dim3 = 'band_sw'
        dim3 = dim3 if dim3 in ecrad_plot.dims else None
        ecrad_plot = ecrad_plot.sel({'time': time_sel, 'height': slice(13, 0), f'{dim3}': band})
    else:
        ecrad_plot = ecrad_plot.sel(time=time_sel, height=slice(13, 0))

    time_extend = pd.to_timedelta((ecrad_plot.time[-1] - ecrad_plot.time[0]).to_numpy())

# %% plot 2D IFS variables along flight track
    _, ax = plt.subplots(figsize=h.figsize_wide, layout='constrained')
    # ecrad 2D field
    ecrad_plot.plot(x='time', y='height', cmap=cmap, ax=ax, robust=robust,
                    vmin=vmin, vmax=vmax, alpha=alpha, norm=norm,
                    cbar_kwargs={'pad': 0.04,
                                 'label': f'{h.cbarlabels[var]} ({h.plot_units[var]})',
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
    plt.savefig(figname, dpi=300)
    figname = f'{fig_path}/{flight}_ecrad_{v}_{var}{band_str}_along_track.png'
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# %% plot histogram
    xlabel = xlabels[v] if v in xlabels else v
    flat_array = ecrad_plot.to_numpy().flatten()
    _, ax = plt.subplots(figsize=h.figsize_wide,
                         layout='constrained')
    hist = ax.hist(flat_array, bins=20)
    ax.set(xlabel=f"{h.cbarlabels[var]} {xlabel} ({h.plot_units[var]})",
           ylabel="Number of Occurrence")
    ax.grid()
    ax.set_yscale("log")
    figname = f"{plot_path}/{flight}_ecrad_{v}_{var}{band_str}_hist.png"
    plt.savefig(figname, dpi=300)
    # figname = f"{fig_path}/{flight}_ecrad_{v}_{var}{band_str}_hist.png"
    # plt.savefig(figname, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
