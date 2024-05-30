#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 27.05.2024

In the ecRad namelist the longwave cloud scattering can be turned off and on (`do_lw_cloud_scattering`), with on being the default according to the documentation.
Here we look at the difference this option makes for the Fu-IFS reference simulation (v15.1).
To make sure that this does not affect the short wave calculations, we start with the net short wave flux difference between the new (with lw scattering) and the old simulations.

.. figure:: figures/lw_cloud_scattering/HALO-AC3_20220411_HALO_RF17_ecrad_diff_flux_net_sw_along_track.png

    Difference in net short wave irradiance between the new (with lw scattering) and the old simulations.

As we can see, there is no difference in the shortwave flux.

We now move on to the difference in longwave downward irradiance.

.. figure:: figures/lw_cloud_scattering/HALO-AC3_20220411_HALO_RF17_ecrad_diff_flux_dn_lw_along_track.png

    Difference in longwave downward irradiance along flight track between simulations with longwave cloud scattering and without.

It can be seen that the longwave cloud scattering causes an increased downward irradiance.
Thus, more irradiance is scattered by the cloud than absorbed, which is to be expected from this option.
The main emitter in the longwave is the surface.
Therefore, the longwave upward irradiance should be reduced above the cloud.
This can be seen in the next figure.

.. figure:: figures/lw_cloud_scattering/HALO-AC3_20220411_HALO_RF17_ecrad_diff_flux_up_lw_along_track.png

    Difference in longwave upward irradiance along flight track between simulations with longwave cloud scattering and without.

The net longwave flux is generally negative for cirrus cloud scenes.
Knowing this the positive difference in net longwave flux shown below means that the net longwave flux with the longwave cloud scattering is less negative.
More scattering means less absorption but also less upward irradiance and thus less cooling.
**Thus, simulations including longwave scattering show less terrestrial cooling for cirrus clouds.**
As it is the default option, **longwave cloud scattering will be turned on** in all simulations.

.. figure:: figures/lw_cloud_scattering/HALO-AC3_20220411_HALO_RF17_ecrad_diff_flux_net_lw_along_track.png

    Difference in net longwave irradiance along flight track between simulations with longwave cloud scattering and without.

"""

if __name__ == '__main__':
# %% import modules
    import pylim.helpers as h
    import pylim.halo_ac3 as meta
    import cmasher as cmr
    import dill
    from matplotlib import colors
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import pandas as pd
    import seaborn as sns
    from tqdm import tqdm
    import xarray as xr

    cbc = h.get_cb_friendly_colors('petroff_6')
    h.set_cb_friendly_colors('petroff_6')
    plt.rc('font', size=12)

# %% set paths
    campaign = 'halo-ac3'
    key = 'RF17'
    flight = meta.flight_names[key]
    date = flight[9:17]

    plot_path = f'{h.get_path("plot", flight, campaign)}/lw_cloud_scattering'
    fig_path = './docs/figures/lw_cloud_scattering'
    save_path = h.get_path('plot', flight, campaign)
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)
    ecrad_path = f'{h.get_path("ecrad", campaign=campaign)}/{date}'

# %% get flight segments for case study period
    loaded_objects = list()
    filenames = [f'{key}_slices.pkl', f'{key}_above_cloud.pkl', f'{key}_below_cloud.pkl']
    for filename in filenames:
        with open(f'{save_path}/{filename}', 'rb') as f:
            loaded_objects.append(dill.load(f))

    slices = loaded_objects[0]
    above_clouds = loaded_objects[1]
    below_clouds = loaded_objects[2]

# %% read in ecrad data
    ecrad_dict = dict()
    for v in ['v15.1', 'v15.1_old_lw_scat']:
        # use center column data
        ds = xr.open_dataset(f'{ecrad_path}/ecrad_merged_inout_{date}_{v}.nc').isel(column=0)
        # select above and below cloud time
        ds = ds.sel(time=slices['case'])
        ds = ds.assign_coords({'sw_albedo_band': range(1, 7)})
        ecrad_dict[v] = ds.copy()

# %% set plotting options
    var = 'flux_net_lw'
    v = 'diff'
    band = None

# %% prepare data set for plotting
    band_str = f'_band{band}' if band is not None else ''
    cbarlabel = f'{h.cbarlabels[var]} ({h.plot_units[var]})' if var in h.cbarlabels else var
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

    if 'diff' in v:
        cmap = cmr.fusion_r
        norm = colors.TwoSlopeNorm(vcenter=0)

    # prepare ecrad dataset for plotting
    sf = h.scale_factors[var] if var in h.scale_factors else 1
    if v == 'diff':
        # calculate difference between simulations
        ds = ecrad_dict['v15.1_old_lw_scat']
        ecrad_ds_diff = ecrad_dict['v15.1'][var] - ds[var]
        ecrad_plot = ecrad_ds_diff.where((ds[var] != 0) | (~np.isnan(ds[var]))) * sf
    else:
        ds = ecrad_dict[v]
        ecrad_plot = ds[var] * sf

    # add new z axis mean pressure altitude
    if 'half_level' in ecrad_plot.dims:
        new_z = ds['press_height_hl'].mean(dim='time') / 1000
    else:
        new_z = ds['press_height_full'].mean(dim='time') / 1000

    ecrad_plot_new_z = list()
    for t in tqdm(ecrad_plot.time, desc='New Z-Axis'):
        tmp_plot = ecrad_plot.sel(time=t)
        if 'half_level' in tmp_plot.dims:
            tmp_plot = tmp_plot.assign_coords(
                half_level=ds['press_height_hl'].sel(time=t, drop=True).to_numpy() / 1000)
            tmp_plot = tmp_plot.rename(half_level='height')

        else:
            tmp_plot = tmp_plot.assign_coords(
                level=ds['press_height_full'].sel(time=t, drop=True).to_numpy() / 1000)
            tmp_plot = tmp_plot.rename(level='height')

        tmp_plot = tmp_plot.interp(height=new_z.to_numpy())
        ecrad_plot_new_z.append(tmp_plot)

    ecrad_plot = xr.concat(ecrad_plot_new_z, dim='time')
    # filter very low to_numpy()
    ecrad_plot = ecrad_plot.where(np.abs(ecrad_plot) > 0.001)

    # select time height slice
    time_sel = slices['case']
    height_sel = slice(13, 0)
    if len(ecrad_plot.dims) > 2:
        dim3 = 'band_sw'
        dim3 = dim3 if dim3 in ecrad_plot.dims else None
        ecrad_plot = ecrad_plot.sel({'time': time_sel, 'height': height_sel, f'{dim3}': band})
    else:
        ecrad_plot = ecrad_plot.sel(time=time_sel, height=height_sel)

    time_extend = pd.to_timedelta((ecrad_plot.time[-1] - ecrad_plot.time[0]).to_numpy())

# %% plot 2D IFS variables along flight track
    _, ax = plt.subplots(figsize=h.figsize_wide, layout='constrained')
    # ecrad 2D field
    ecrad_plot.plot(x='time', y='height', cmap=cmap, ax=ax,
                    robust=robust, vmin=vmin, vmax=vmax, alpha=alpha, norm=norm,
                    cbar_kwargs={'pad': 0.04,
                                 'label': cbarlabel,
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
    figname = f'{flight}_ecrad_{v}_{var}{band_str}_along_track.png'
    plt.savefig(f'{fig_path}/{figname}', dpi=300)
    plt.savefig(f'{plot_path}/{figname}', dpi=300)
    plt.show()
    plt.close()
