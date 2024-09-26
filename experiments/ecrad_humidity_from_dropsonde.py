#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 25.09.2024

ecRad predicts way higher downward terrestrial irradiance below cloud during both
RF17 and RF18 of HALO-(AC)³. As the cloud base in the IFS is actually higher than
retrieved by the lidar and radar measurements, a too high cloud base temperature
cannot explain this missmatch. The dropsonde soundings show the closest IFS grid
cell to have quite different values compared to the measured ones. Water vapor
is one of the key greenhouse gases in the troposphere and as such also one of the
most important emitters of terrestrial irradiance.

In this experiment I investigate the difference in terrestrial downward irradiance
between the ecRad simulations using the IFS specific humidity and the
simulations using the specific humidity measurements performed by the dropsondes.
For this the dropsonde measurement is interpolated onto the IFS full level pressure
height.
The levels above 10 km, which is the highest measurement point of the dropsonde,
are filled with the original IFS specific humidity.

The IFS data is always replaced by the closest radiosonde data with no interpolation
in space or time.
To avoid results influenced by the mislocation of the dropsonde, only the grid
cells, which are closest to the dropsonde locations are analyzed, at first.

**Results**

.. figure:: ./figures/humidity_from_dropsondes/HALO-AC3_RF17_flux_dn_lw_IFS_vs_dropsonde.png

    Difference in downwelling terrestrial irradiance between IFS dropsonde run
    and reference simulation (v15.1) for profiles at the respective dropsonde
    locations during RF17.

The first dropsonde is located at the start of the cirrus structure,
where it was very thin.
At the below cloud altitude no difference between simulation and measurement can
be observed.
However, at cloud top just below 8 km an overestimation of the thermal-infrared
downward irradiance of 4 Wm-2 can be seen.

At the second dropsonde location, which was launched in the center of the cirrus,
the difference around cloud top is reduced to 2 Wm-2.
Below the cloud the difference increases to 4 Wm-2 at flight altitude and to a
maximum of 6 Wm-2 just below that.

.. figure:: ./figures/humidity_from_dropsondes/HALO-AC3_RF18_flux_dn_lw_IFS_vs_dropsonde.png

    Difference in downwelling terrestrial irradiance between IFS dropsonde run
    and reference simulation (v15.1) for profiles at the respective dropsonde
    locations during RF18.

For the five dropsondes launched during RF18 the differences are smaller compared
to RF17.
Nonetheless, the same trend can be seen as for the first dropsonde of RF17.
The difference at the below cloud flight altitude is close to 0 Wm-2,
while the difference at cloud top (8 km) is larger with up to 2.5 Wm-2.

From the above it can be concluded that a wrong atmospheric humidity profile
close to the cloud in the IFS can lead to large differences in the terrestrial
downward irradiance.

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
    from tqdm import tqdm
    import xarray as xr

    cbc = h.get_cb_friendly_colors('petroff_6')
    h.set_cb_friendly_colors('petroff_6')
    plt.rc('font', size=12)

# %% set paths and read in data
    campaign = 'halo-ac3'
    keys = ['RF17', 'RF18']
    plot_path = f'{h.get_path("plot", campaign=campaign)}/humidity_from_dropsondes'
    fig_path = './docs/figures/humidity_from_dropsondes'
    dropsonde_path = f'{h.get_path('all', campaign=campaign, instrument='dropsondes')}/Level_3'
    dropsonde_file = 'merged_HALO_P5_beta_v2.nc'
    dds = xr.open_dataset(f'{dropsonde_path}/{dropsonde_file}')
    os.makedirs(plot_path, exist_ok=True)
    os.makedirs(fig_path, exist_ok=True)

    (dropsonde_ds, slices, above_clouds,
     below_clouds, ecrad_dicts) = (dict(), dict(), dict(), dict(), dict())
    for key in keys:
        flight = meta.flight_names[key]
        date = flight[9:17]
        dropsonde_ds[key] = dds.where(
            dds.launch_time.dt.date == pd.to_datetime(date).date(),
            drop=True)

        save_path = h.get_path('plot', flight, campaign)
        ecrad_path = f'{h.get_path("ecrad", campaign=campaign)}/{date}'

        # get flight segments for case study period
        loaded_objects = list()
        filenames = [f'{key}_slices.pkl', f'{key}_above_cloud.pkl', f'{key}_below_cloud.pkl']
        for filename in filenames:
            with open(f'{save_path}/{filename}', 'rb') as f:
                loaded_objects.append(dill.load(f))

        slices[key] = loaded_objects[0]
        above_clouds[key] = loaded_objects[1]
        below_clouds[key] = loaded_objects[2]

        # read in ecrad data
        ecrad_dict = dict()
        for v in ['v15.1', 'v43.1']:
            # use center column data
            ds = xr.open_dataset(f'{ecrad_path}/ecrad_merged_inout_{date}_{v}.nc')
            # select the closest column if possible
            if 'column' in ds.dims:
                ds = ds.sel(column=0)
            # select above and below cloud time
            ds = ds.sel(time=slices[key]['case'])
            ds = ds.assign_coords({'sw_albedo_band': range(1, 7)})
            ecrad_dict[v] = ds.copy()

        ecrad_dicts[key] = ecrad_dict.copy()

# %% fill up nan values in specific humidity from dropsonde measurements
    key = 'RF17'
    ds_ds = dropsonde_ds[key]
    dropsonde_q = ds_ds.q.interpolate_na(dim='alt',
                                         method='linear',
                                         fill_value='extrapolate')
    # replace sonde_id with launch time for easier selection
    dropsonde_q = (dropsonde_q
                   .assign_coords(sonde_id=dropsonde_q.launch_time)
                   .rename(sonde_id='time')
                   .sortby('time'))

# %% plot original dropsonde data and original and new IFS data
    for t in dropsonde_q.launch_time:
        ds = ecrad_dicts[key]['v15.1'].sel(time=t, method='nearest')
        q_ifs_org = ds.q * 1e3
        q_ifs_new = ecrad_dicts[key]['v43.1'].q.sel(time=t, method='nearest') * 1e3
        q_ds_org = (ds_ds.q
                    .assign_coords(sonde_id=ds_ds.launch_time)
                    .rename(sonde_id='time')
                    .sortby('time')
                    .sel(time=t, method='nearest') * 1e3)

        _, ax = plt.subplots(layout='constrained')
        (q_ifs_org
         .assign_coords(level=ds.press_height_full)
         .plot(y='level', label=f'IFS q original', ax=ax))
        (q_ifs_new
         .assign_coords(level=ds.press_height_full)
         .plot(y='level', label='IFS q new', ax=ax))
        q_ds_org.plot(y='alt', label='Dropsonde q original', ax=ax)

        ax.set(
            title=f'Dropsonde launch time: '
                  f'{pd.to_datetime(t.to_numpy()):%Y-%m-%d %H:%M}',
            xlabel='Specific humidity (g$\\,$kg$^{-1}$)',
            ylabel='Height (m)',
            ylim=(0, 12000)
        )
        ax.grid()
        ax.legend()
        plt.show()
        plt.close()

# %% set plotting options
    key = 'RF17'
    var = 'flux_dn_lw'
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
        ds = ecrad_dicts[key]['v43.1']
        ecrad_ds_diff = ecrad_dicts[key]['v15.1'][var] - ds[var]
        ecrad_plot = ecrad_ds_diff.where((ds[var] != 0) | (~np.isnan(ds[var]))) * sf
    else:
        ds = ecrad_dicts[key][v]
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
    time_sel = slices[key]['case']
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
    c = (ds.cloud_fraction
         .assign_coords(level=ds.press_height_full / 1e3)
         .plot.contour(x='time', levels=5, cmap='grey'))
    ax.clabel(c, c.levels)

    ax.set_title(f'{key} IFS/ecRad input/output along Flight Track - {v}')
    ax.set_ylabel('Altitude (km)')
    ax.set_xlabel('Time (UTC)')
    h.set_xticks_and_xlabels(ax, time_extend)
    figname = f'{flight}_ecrad_{v}_{var}{band_str}_along_track.png'
    plt.savefig(f'{fig_path}/{figname}', dpi=300)
    plt.savefig(f'{plot_path}/{figname}', dpi=300)
    plt.show()
    plt.close()

# %% plot profile comparison at dropsonde location
    key = 'RF17'
    var = 'flux_dn_lw'
    ds_times = dropsonde_ds[key].launch_time.sortby('launch_time').to_numpy()
    for t in ds_times:
        ds_new = ecrad_dicts[key]['v43.1'].sel(time=t, method='nearest')
        ds_old = ecrad_dicts[key]['v15.1'].sel(time=t, method='nearest')
        _, ax = plt.subplots(figsize=h.figsize_wide, layout='constrained')
        (ds_new[var]
         .assign_coords(half_level=ds_new.press_height_hl / 1e3)
         .plot(y='half_level', label='v43.1', ax=ax))
        (ds_old[var]
         .assign_coords(half_level=ds_old.press_height_hl / 1e3)
         .plot(y='half_level', ax=ax, label='v15.1'))
        ax.legend()
        ax.grid()
        ax.set(
            title=f'Dropsonde launch time: {pd.to_datetime(t):%Y-%m-%d %H:%M}',
            xlabel=f'Downwelling longwave flux ({h.plot_units[var]})',
            ylabel='Height (km)',
            xlim=0,
            ylim=(0, 10)
        )
        plt.show()
        plt.close()

# %% plot difference in profile at dropsonde locations RF 17
    key = 'RF17'
    var = 'flux_dn_lw'
    ds_times = dropsonde_ds[key].launch_time.sortby('launch_time').to_numpy()
    ds_times = ds_times[2:4] if key == 'RF17' else ds_times[11:]
    fig, axs = plt.subplots(1, len(ds_times),
                          figsize=h.figsize_wide,
                          layout='constrained')
    for i, t in enumerate(ds_times):
        ax = axs[i]
        ds_new = ecrad_dicts[key]['v43.1'].sel(time=t, method='nearest')
        ds_old = ecrad_dicts[key]['v15.1'].sel(time=t, method='nearest')
        ds_plot = ds_new - ds_old
        (ds_plot[var]
         .assign_coords(half_level=ds_new.press_height_hl / 1e3)
         .plot(y='half_level', ax=ax))
        ax.axhline((ds_new['press_height_hl'] / 1e3)
                   .isel(half_level=98),
                   color='k',
                   label='Below cloud\nflight altitude'
                   )
        ax.grid()
        ax.set(
            title=f'{pd.to_datetime(t):%Y-%m-%d %H:%M} UTC',
            xlabel='',
            ylabel='',
            xlim=(-7, 2),
            ylim=(0, 10)
        )

    axs[0].set(
        ylabel='Height (km)'
    )
    axs[0].legend()
    fig.supxlabel(f'Difference in downwelling terrestrial irradiance ({h.plot_units[var]})',
                  size=12)
    figname = f'HALO-AC3_{key}_flux_dn_lw_IFS_vs_dropsonde.png'
    plt.savefig(f'{plot_path}/{figname}', dpi=300)
    plt.savefig(f'{fig_path}/{figname}', dpi=300)
    plt.show()
    plt.close()

# %% plot difference in profile at dropsonde locations RF 18
    key = 'RF18'
    var = 'flux_dn_lw'
    ds_times = dropsonde_ds[key].launch_time.sortby('launch_time').to_numpy()
    ds_times = ds_times[2:4] if key == 'RF17' else ds_times[11:]
    fig, axs = plt.subplots(1, len(ds_times),
                            figsize=h.figsize_wide,
                            layout='constrained')
    for i, t in enumerate(ds_times):
        ax = axs[i]
        ds_new = ecrad_dicts[key]['v43.1'].sel(time=t, method='nearest')
        ds_old = ecrad_dicts[key]['v15.1'].sel(time=t, method='nearest')
        ds_plot = ds_new - ds_old
        (ds_plot[var]
         .assign_coords(half_level=ds_new.press_height_hl / 1e3)
         .plot(y='half_level', ax=ax))
        ax.axhline((ds_new['press_height_hl'] / 1e3)
                   .isel(half_level=98),
                   color='k',
                   )
        ax.grid()
        ax.set(
            title=f'{pd.to_datetime(t):%H:%M} UTC',
            xlabel='',
            ylabel='',
            ylim=(0, 10),
            xlim=(-2.5, 2.5)
        )
    axs[0].set(
        ylabel='Height (km)',
    )
    axs[-1].plot([], ls='-', c='k', label='Below cloud flight altitude')
    fig.legend(loc='lower left', bbox_to_anchor=(0.69, -0.01))
    fig.supxlabel(f'Difference in downwelling terrestrial irradiance ({h.plot_units[var]})'
                  f'                      ',
                  size=12)
    figname = f'HALO-AC3_{key}_flux_dn_lw_IFS_vs_dropsonde.png'
    plt.savefig(f'{plot_path}/{figname}', dpi=300)
    plt.savefig(f'{fig_path}/{figname}', dpi=300)
    plt.show()
    plt.close()
