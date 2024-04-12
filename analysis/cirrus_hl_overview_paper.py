#!/usr/bin/env python
"""Plot for CIRRUS-HL overview paper

- define staircase pattern
- average spectra over height during staircase pattern
- plot flight track through IFS output (ciwc, clwc)
- plot evolution of SMART spectra through staircase pattern

*author*: Johannes Röttenbacher
"""

# %% module import
import pylim.ecrad
import pylim.helpers as h
from pylim import reader
from pylim import meteorological_formulas as met
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import xarray as xr
import pandas as pd
import cmasher as cmr
import logging
from tqdm import tqdm

log = logging.getLogger('pylim')
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

cbc = h.get_cb_friendly_colors('petroff_8')

# %% set up paths and meta data
date = 20210629
flight = f'Flight_{date}a'
campaign = 'cirrus-hl'
version = 'v15'
smart_path = h.get_path('calibrated', flight, campaign)
bahamas_path = h.get_path('bahamas', flight, campaign)
wales_path = f'{h.get_path("wales", flight, campaign)}'
ecrad_path = f'{h.get_path("ecrad", campaign=campaign)}/{date}'
ifs_path = f'{h.get_path("ifs", campaign=campaign)}/{date}'
plot_path = 'C:/Users/Johannes/Documents/Doktor/manuscripts/2022_CIRRUS-HL_overview'
# filenames
smart_fdw_file = f'CIRRUS-HL_F05_20210629a_HALO_SMART_spectral-irradiance-Fdw_v1.0.nc'
smart_fup_file = f'CIRRUS-HL_F05_20210629a_HALO_SMART_spectral-irradiance-Fup_v1.0.nc'
bahamas_file = 'CIRRUSHL_F05_20210629a_ADLR_BAHAMAS_v1.nc'
ecrad_input_file = f'ecrad_merged_inout_{date}_{version}_mean.nc'
overlap_decorr_file = f'{date}_decorrelation_length.csv'
g_file = 'CIRRUS-HL_RF05_BAMS_asymmetry.txt'
phips_image = 'PHIPS_images/BAMS_collage.png'
start_dt = pd.Timestamp(2021, 6, 29, 9, 30)
end_dt = pd.Timestamp(2021, 6, 29, 12, 30)

# %% read in data
smart_fdw = xr.open_dataset(f'{smart_path}/{smart_fdw_file}')
smart_fup = xr.open_dataset(f'{smart_path}/{smart_fup_file}')
bahamas = reader.read_bahamas(f'{bahamas_path}/{bahamas_file}')
ecrad_input = xr.open_dataset(f'{ecrad_path}/{ecrad_input_file}')
g = pd.read_csv(f'{plot_path}/{g_file}', parse_dates=[0, 1])
oldl = pd.read_csv(f'{ifs_path}/{overlap_decorr_file}', index_col='time', parse_dates=True)
phips = mpimg.imread(f'{plot_path}/{phips_image}')

# %% select only relevant times
time_sel = slice(start_dt, end_dt)
wavelength_sel = slice(400, 2150)
smart_fdw = smart_fdw.sel(time=time_sel, wavelength=wavelength_sel)
smart_fup = smart_fup.sel(time=time_sel, wavelength=wavelength_sel)
bahamas_sel = bahamas.sel(time=time_sel)
ecrad_input = ecrad_input.sel(time=time_sel)

g["PlotTime"] = g[["StartTime", "EndTime"]].mean(axis=1)

# %% print mean overlap decorrelation length for case study period
mean_oldl = oldl.loc[start_dt:end_dt].mean().to_numpy() * 1000
print(f'Mean overlap decorrelation length for case study: '
      f'{mean_oldl[0]:.2f} m')

# %% get height level of actual flight altitude in ecRad model, this determines only the index of the level
ecrad_input = pylim.ecrad.calculate_pressure_height(ecrad_input)
bahamas_tmp = bahamas.sel(time=ecrad_input.time)
height_level_da = pylim.ecrad.get_model_level_of_altitude(bahamas_tmp.IRS_ALT, ecrad_input, 'level')
aircraft_height = [ecrad_input['press_height_hl'].isel(half_level=i, time=100).values for i in height_level_da]
aircraft_height_da = xr.DataArray(aircraft_height, dims=['time'], coords={'time': ecrad_input.time})

# %% define staircase sections according to flight levels
derivative = bahamas_sel.H.diff(dim='time')
threshold = 0.5
idx = np.where(np.abs(derivative) < threshold)[0]
# draw a quicklook
bahamas_sel.H[idx].plot(marker='.')
# bahamas_sel.H[ids.flatten()].plot(marker='.')
# plt.plot(time_diff.astype(float) * 1e-9)
plt.show()
plt.close()
# select times with constant altitude
times = bahamas_sel.time[idx]
# find indices where the difference to the next timestep is greater 1 second -> change of altitude
time_diff = (times.to_numpy()[1:] - times.to_numpy()[:-1])
ids = np.argwhere(np.diff(times) > pd.Timedelta(2, 's'))
ids2 = ids + 1  # add 1 to get the indices at the start of the next section
ids = np.insert(ids, 0, 0)  # add the start of the first section
ids = np.append(ids, ids2)  # combine all indices
ids.sort()  # sort them
ids = np.unique(ids)  # delete doubles
times_sel = times[ids]  # select only the start and end times
# get start and end times
start_dts = times_sel[2:17:2]
end_dts = times_sel[3:18:2]
# export times for Anna
start_dts.name = 'start'
end_dts.name = 'end'
# start_dts.to_netcdf(f'{plot_path}/start_dts.nc')
# end_dts.to_netcdf(f'{plot_path}/end_dts.nc')
df_export = pd.DataFrame({'start': times_sel[2:17:2],
                          'end': times_sel[3:18:2]})
df_export.index += 1
df_export.to_csv(f'{plot_path}/section_times.txt', index_label='section')
log.info('Defined Sections')

# %% calculate albedo from SMART measurements and ecRad simulations
albedo = smart_fup.Fup_cor.interp_like(smart_fdw.wavelength) / smart_fdw.Fdw_cor

# %% set some variables for the coming plots
# labels = ['Section 1 (FL260, 8.3$\,$km)', 'Section 2 (FL280, 8.7$\,$km)', 'Section 3 (FL300, 9.3$\,$km)',
#           'Section 4 (FL320, 10$\,$km)', 'Section 5 (FL340, 10.6$\,$km)', 'Section 6 (FL360, 11.2$\,$km)',
#           'Section 7 (FL390, 12.2$\,$km)']
labels = ['Section 1', 'Section 2', 'Section 3',
          'Section 4', 'Section 5', 'Section 6',
          'Section 7']

# %% plot SMART spectra for each flight section Fdw
sections = dict()
for i, (st, et) in enumerate(zip(start_dts, end_dts)):
    sections[f'mean_spectra_{i}'] = smart_fdw.Fdw_cor.sel(time=slice(st, et)).mean(dim='time')

h.set_cb_friendly_colors('petroff_8')
plt.rc('font', size=11)
fig, ax = plt.subplots(figsize=h.figsize_wide, layout='constrained')
# SMART measurement
for section, label in zip(sections, labels):
    sections[section].plot(ax=ax, label=label, linewidth=1)

# aesthetics
ax.legend(loc='upper right', ncol=4)
ax.set_xlim((300, 2100))
ax.set_ylim((0, 1.65))
ax.set_title(f'Mean Downward Irradiance measured by SMART')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Spectral Irradiance $(\mathrm{W\,m}^{-2}\mathrm{\,nm}^{-1})$')
ax.grid()
figname = f'{plot_path}/cirrus-hl_smart_fdw_spectra_sections_{flight}_overview.png'
plt.savefig(figname, dpi=300)
log.info(f'Saved {figname}')
plt.show()
plt.close()

# %% plot SMART spectra for each flight section Fup
sections = dict()
for i, (st, et) in enumerate(zip(start_dts, end_dts)):
    sections[f'mean_spectra_{i}'] = smart_fup.Fup_cor.sel(time=slice(st, et)).mean(dim='time')

h.set_cb_friendly_colors('petroff_8')
plt.rc('font', size=11)
fig, ax = plt.subplots(figsize=h.figsize_wide, layout='constrained')
# banded SMART measurement
for section, label in zip(sections, labels):
    sections[section].plot(ax=ax, label=label, linewidth=1)

# aesthetics
ax.legend(loc='upper right',
          ncol=4)
ax.set_title(f'Mean Upward Irradiance measured by SMART')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Spectral Irradiance $(\mathrm{W\,m}^{-2}\mathrm{\,nm}^{-1})$')
ax.grid()
figname = f'{plot_path}/cirrus-hl_smart_fup_spectra_sections_{flight}_overview.png'
plt.savefig(figname, dpi=300)
log.info(f'Saved {figname}')
plt.show()
plt.close()

# %% plot SMART spectra for each flight section Fup and Fdw
sections, sections_fdw = dict(), dict()
for i, (st, et) in enumerate(zip(start_dts, end_dts)):
    sections[f'mean_spectra_{i}'] = smart_fup.Fup_cor.sel(time=slice(st, et)).mean(dim='time')
    sections_fdw[f'mean_spectra_{i}'] = smart_fdw.Fdw_cor.sel(time=slice(st, et)).mean(dim='time')

h.set_cb_friendly_colors()
plt.rc('font', size=5)
fig, axs = plt.subplots(nrows=2, figsize=(10 * h.cm, 6 * h.cm))
# Fdw SMART measurement
for section, label in zip(sections_fdw, labels):
    sections_fdw[section].plot(ax=axs[0], label=label, lw=0.7)
# Fup SMART measurements
for section, label in zip(sections, labels):
    sections[section].plot(ax=axs[1], lw=0.7)

# plot only section 1, 2 and 7
# Fdw
# sections_fdw[f'mean_spectra_0'].plot(ax=axs[0], label=labels[0], linewidth=4)
# sections_fdw[f'mean_spectra_1'].plot(ax=axs[0], label=labels[1], linewidth=4)
# sections_fdw[f'mean_spectra_6'].plot(ax=axs[0], label=labels[6], color='#44AA99', linewidth=4)
# Fup
# sections[f'mean_spectra_0'].plot(ax=axs[1], linewidth=4)
# sections[f'mean_spectra_1'].plot(ax=axs[1], linewidth=4)
# sections[f'mean_spectra_6'].plot(ax=axs[1], color='#44AA99', linewidth=4)

# aesthetics
fig.legend(bbox_to_anchor=(0.5, 0), loc='lower center', bbox_transform=fig.transFigure, ncol=3)  # legend below plot
# axs[0].legend(loc=1)  # legend in upper right corner
# axs[0].set_title(f'Mean Irradiance measured by SMART')
axs[0].set_xlabel('')
axs[0].set_ylabel('Spectral Downward \nIrradiance \n$(\mathrm{W\,m}^{-2}\mathrm{\,nm}^{-1})$')
axs[1].set_xlabel('Wavelength (nm)')
axs[1].set_ylabel('Spectral Upward \nIrradiance \n$(\mathrm{W\,m}^{-2}\mathrm{\,nm}^{-1})$')
for ax in axs:
    ax.grid()
    ax.set_ylim((0, 1.65))
plt.tight_layout()
plt.subplots_adjust(bottom=0.3)
# plt.subplots_adjust(bottom=0.21)
figname = f'{plot_path}/cirrus-hl_smart_spectra_sections_{flight}_overview.png'
# figname = f'{plot_path}/cirrus-hl_smart_spectra_sections127_{flight}.png'
plt.savefig(figname, dpi=300)
log.info(f'Saved {figname}')
plt.show()
plt.close()

# %% plot SMART albedo for each flight section
sections = dict()
for i, (st, et) in enumerate(zip(start_dts, end_dts)):
    sections[f'mean_spectra_{i}'] = albedo.sel(time=slice(st, et)).mean(dim='time')

plt.rcdefaults()
h.set_cb_friendly_colors('petroff_8')
plt.rc('font', size=11)
fig, ax = plt.subplots(figsize=h.figsize_wide, layout='constrained')
# SMART albedo
for section, label in zip(sections, labels):
    sections[section].plot(ax=ax, label=label, linewidth=1)

# plot only section 1, 2 and 7
# albedo
# sections[f'mean_spectra_0'].plot(ax=ax, label=labels[0], linewidth=4)
# sections[f'mean_spectra_1'].plot(ax=ax, label=labels[1], linewidth=4)
# sections[f'mean_spectra_6'].plot(ax=ax, label=labels[6], color='#44AA99', linewidth=4)

# aesthetics
ax.legend(loc='upper right', ncol=4)
ax.set_title(f'Mean Spectral albedo derived from SMART Measurements')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Albedo')
ax.set_ylim((0, 1))
ax.grid()
figname = f'{plot_path}/cirrus-hl_smart_albedo_sections_{flight}_overview.png'
plt.savefig(figname, dpi=200)
log.info(f'Saved {figname}')
plt.show()
plt.close()
# %% plot aircraft track through ecrad input, combine clwc and ciwc and SMART spectra in same figure
units = dict(clwc='g/kg', ciwc='g/kg', q_ice='g/kg', cswc='g/kg', crwc='g/kg', t='K', re_ice='m x $10^{-6}$',
             re_liquid='m x $10^{-6}$')
scale_factor = dict(clwc=1000, ciwc=1000, q_ice=1000, cswc=1000, crwc=1000, t=1, re_ice=1e6, re_liquid=1e6)
colorbarlabel = dict(clwc='Cloud Liquid Water Content', ciwc='Cloud Ice Water Content', q_ice='Ice and Snow Content',
                     cswc='Cloud Snow Water Content', crwc='Cloud Rain Water Content', t='Temperature',
                     re_ice='Effective Ice Particle Radius', re_liquid='Effective Droplet Radius')
variable = 'clwc'
cmaps = dict(t='bwr', clwc=cmr.get_sub_cmap('cmr.flamingo', .25, .9), ciwc=cmr.get_sub_cmap('cmr.freeze', .25, 0.85))
cmap = cmaps[variable] if variable in cmaps else cmr.rainforest
cmap = plt.get_cmap(cmap).copy().reversed()
# cmap.set_under('white')
x_sel = (pd.Timestamp(2021, 6, 29, 9, 44), pd.Timestamp(2021, 6, 29, 12, 10))
ecrad_plot = ecrad_input[variable] * scale_factor[variable]
ecrad_plot = ecrad_plot.assign_coords(level=ecrad_input['press_height'].isel(time=100, drop=True)[1:].values / 1000)
ecrad_plot = ecrad_plot.rename(level='height')
aircraft_height_plot = aircraft_height_da / 1000

plt.rcdefaults()
h.set_cb_friendly_colors()
plt.rc('font', size=6)
plt.rc('lines', linewidth=0.7)
fig, axs = plt.subplots(nrows=2, figsize=(10 * h.cm, 10 * h.cm))
# first row IFS output
# ecrad input clwc
ax = axs[0]
ecrad_plot = ecrad_plot.where(ecrad_plot > 0.001)  # filter very low values
ecrad_plot.plot(x='time', y='height', cmap=cmap, ax=ax, robust=True,
                cbar_kwargs={'pad': 0.04, 'label': f'{colorbarlabel[variable]} ({units[variable]})'})

# ecrad input ciwc
variable = 'ciwc'
cmap = cmaps[variable]
cmap = plt.get_cmap(cmap).copy().reversed()
# cmap.set_under('white')
ecrad_plot = ecrad_input[variable] * scale_factor[variable]
ecrad_plot = ecrad_plot.assign_coords(level=ecrad_input['press_height'].isel(time=100, drop=True)[1:].values / 1000)
ecrad_plot = ecrad_plot.rename(level='height')
ecrad_plot = ecrad_plot.where(ecrad_plot > 0.001)  # filter very low values
ecrad_plot.plot(x='time', y='height', cmap=cmap, ax=ax, robust=True, alpha=0.6, xticks=None,
                cbar_kwargs={'pad': 0.01, 'label': f'{colorbarlabel[variable]} ({units[variable]})'})

# aircraft altitude through model
aircraft_height_plot.plot(x='time', color='k', ax=ax, label='HALO altitude')

# plot sections numbers
for s, x in zip(range(8), start_dts):
    ax.text((x + pd.Timedelta(minutes=2)).values, (aircraft_height_plot.sel(time=x, method='nearest') + 0.5).values,
            s + 1)

ax.text(0.01, 0.75, 'a)', transform=ax.transAxes, size=8)
ax.legend(loc=2)
ax.set_xlim(x_sel)
ax.set_ylim(0, 14)
# for tick in ax.get_xticklabels():
#     tick.set_rotation(0)
h.set_xticks_and_xlabels(ax, x_sel[1] - x_sel[0])
xticks = ax.get_xticks()
ax.set_xticks(xticks)
ax.set_xticklabels(['10:00', '11:00', '12:00'], rotation=0, fontdict={'horizontalalignment': 'center'})
ax.grid()
# ax.set_title('IFS Output along HALO Flight Track 29. June 2021')
ax.set_ylabel('Pressure Height (km)')
ax.set_xlabel('Time (UTC)')

# second row SMART spectra
ax = axs[1]
sections = dict()
for i, (st, et) in enumerate(zip(start_dts, end_dts)):
    sections[f'mean_spectra_{i}'] = albedo.sel(time=slice(st, et)).mean(dim='time')

for section, label in zip(sections, labels):
    sections[section].plot(ax=ax, label=label, linewidth=0.7)

ax.text(0.01, 0.9, 'b)', transform=ax.transAxes, size=8)
# aesthetics
ax.legend(bbox_to_anchor=(0, 0), loc='lower left', bbox_transform=fig.transFigure, ncol=3, fontsize=5)
# ax.legend(loc=1, ncol=2, fontsize=22)
# ax.set_title(f'Mean Spectral albedo derived from SMART Measurements')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Albedo')
ax.set_ylim((0, 1))
ax.grid()
plt.tight_layout()
plt.subplots_adjust(bottom=0.21)
figname = f'{plot_path}/cirrus-hl_IFS_clwc-ciwc_SMART_section_spectra_HALO_alt_{flight}.png'
# plt.savefig(figname, dpi=300)
# log.info(f'Saved {figname}')
plt.show()
plt.close()

# %% prepare ecrad data for plotting
ecrad_plot = dict()
for var in ['clwc', 'ciwc', 'clwp', 'ciwp', 'iwp', 'lwp', 'cloud_fraction']:
    sf = h.scale_factors[var] if var in h.scale_factors else 1
    ds = ecrad_input[var] * sf

    # add new z axis mean pressure altitude
    if "half_level" in ds.dims:
        new_z = ecrad_input["press_height_hl"].mean(dim="time") / 1000
    else:
        new_z = ecrad_input["press_height_full"].mean(dim="time") / 1000

    ds_new_z = list()
    for t in tqdm(ds.time, desc="New Z-Axis"):
        tmp_plot = ds.sel(time=t)
        if "half_level" in tmp_plot.dims:
            tmp_plot = tmp_plot.assign_coords(
                half_level=ecrad_input["press_height_hl"].sel(time=t, drop=True).to_numpy() / 1000)
            tmp_plot = tmp_plot.rename(half_level="height")

        else:
            tmp_plot = tmp_plot.assign_coords(
                level=ecrad_input["press_height_full"].sel(time=t, drop=True).to_numpy() / 1000)
            tmp_plot = tmp_plot.rename(level="height")

        tmp_plot = tmp_plot.interp(height=new_z.to_numpy())
        ds_new_z.append(tmp_plot)

    ds = xr.concat(ds_new_z, dim="time")
    # filter very low to_numpy()
    ds = ds.where(np.abs(ds) > 0.001)

    # select time height slice
    height_sel = slice(15, 0)
    ecrad_plot[var] = ds.sel(height=height_sel)

# %% plot four panel figure for overview paper
plt.rcdefaults()
plt.rc('font', size=7)
h.set_cb_friendly_colors('petroff_8')
cloud_filter = ecrad_plot['cloud_fraction'] > 0
vmax = 200
fig = plt.figure(figsize=(16 * h.cm, 9 * h.cm))

axs = fig.subplot_mosaic([['upper left'],
                          ['middle left'],
                          ['lower left']],
                         height_ratios=[1, 0.5, 0.5],
                         gridspec_kw={
                             'top': 1,
                             'bottom': 0,
                             'left': 0,
                             'right': 0.5,
                             'wspace': 0.1,
                             'hspace': 0.1
                         })
# upper left panel - IFS IWC and LWC forecast
ax = axs['upper left']
cbar = cmr.get_sub_cmap(cmr.flamingo_r, 0.1, 1)
clwc = (ecrad_plot['clwc']
        .where(cloud_filter)
        .plot(x='time', y='height', cmap=cbar, alpha=0.8,
              ax=ax, robust=True, vmin=0.001, vmax=vmax,
              cbar_kwargs={'label': r'Liquid water content (mg$\,$kg$^{-1}$)',
                           'pad': 0.05,
                           'orientation': 'horizontal',
                           'location': 'top',
                           'aspect': 35,
                           }))
clwc.colorbar.ax.xaxis.labelpad = 2

cbar = cmr.get_sub_cmap(cmr.freeze_r, 0.1, 1)
ciwc = (ecrad_plot['ciwc']
        .where(cloud_filter)
        .plot(x='time', y='height', cmap=cbar, alpha=0.8,
              ax=ax, robust=True, vmin=0.001, vmax=vmax,
              cbar_kwargs={'label': r'Ice water content (mg$\,$kg$^{-1}$)',
                           'pad': 0.02,
                           'orientation': 'horizontal',
                           'location': 'top',
                           'aspect': 35,
                           }))
ciwc.colorbar.ax.xaxis.labelpad = 0
ciwc.colorbar.ax.set_xticklabels('')
ciwc.colorbar.ax.tick_params(direction='in')

# aircraft altitude
aircraft_height_plot = aircraft_height_da / 1000
aircraft_height_plot.plot(x='time', label='HALO altitude', c='k', ax=ax)
for i, (st, et) in enumerate(zip(start_dts[:-1], end_dts[:-1])):
    alt_plot = (aircraft_height_plot
                .sel(time=slice(st, et)))
    alt_plot.plot(x='time', ax=ax)
    # add section label at center of each section
    section_dt = (pd.Series(np.stack([st.to_numpy(), et.to_numpy()]))
                  .mean())
    section_alt = alt_plot.mean()
    ax.text(section_dt, section_alt, f'{i + 1}', va='bottom')
ax.legend(loc='upper left')
h.set_xticks_and_xlabels(ax, pd.Timedelta(3, 'hours'))
fig.text(0.01, 1, '(a)')
ax.set(xlabel='',
       xticklabels=[],
       ylabel='Height (km)')
ax.yaxis.set_major_locator(ticker.MultipleLocator(2))

# middle left panel - IWP and LWP
ax = axs['middle left']
iwp = ((ecrad_input['ciwp'])
       .where((~np.isnan(ecrad_input['ciwp']))
              & (ecrad_input['ciwp'].level > height_level_da)
              & (~np.isinf(ecrad_input['ciwp']))
              & (ecrad_input['cloud_fraction'] > 0), 0)
       .integrate(coord='level'))
iwp.plot(x='time', ax=ax, label='IWP', zorder=2)

lwp = ((ecrad_input['clwp'])
       .where((~np.isnan(ecrad_input['clwp']))
              & (ecrad_input['clwp'].level > height_level_da)
              & (~np.isinf(ecrad_input['clwp']))
              & (ecrad_input['cloud_fraction'] > 0), 0)
       .integrate(coord='level'))
color = cmr.take_cmap_colors(cmr.flamingo_r, 1, cmap_range=(0.4, 0.5))
lwp.plot(x='time', ax=ax, label='LWP', c=color[0], zorder=1)

# plot mean per section
for i, (st, et) in enumerate(zip(start_dts[:-1], end_dts[:-1])):
    section_dt = (pd.Series(np.stack([st.to_numpy(), et.to_numpy()]))
                  .mean())
    section_iwp = (iwp
                   .sel(time=slice(st, et))
                   .mean(dim='time'))
    section_lwp = (lwp
                   .sel(time=slice(st, et))
                   .mean(dim='time'))
    ax.plot(section_dt, section_iwp, c='black', ls='',
            marker='.', markersize=6, mfc=cbc[0], zorder=4)
    ax.plot(section_dt, section_lwp, c='black', ls='',
            marker='.', markersize=6, mfc=color[0], zorder=3)

ax.plot([], ls='', marker='.', markersize=4, label="Section mean", c='k')
ax.margins(0, 0.05)
h.set_xticks_and_xlabels(ax, pd.Timedelta(3, 'hours'))
box_props = dict(boxstyle='round', ec='white', fc='white', alpha=0.95)
ax.text(0.02, 0.86, '(b)', transform=ax.transAxes, bbox=box_props)
ax.set(xlabel='',
       xticklabels=[],
       ylabel='Ice/Liquid water\n path (kg$\\,$m$^{-2}$)')
ax.grid()
ax.legend(loc='center left')

# lower left panel - asymmetry parameter
ax = axs['lower left']
ax.errorbar('PlotTime', 'g', yerr='g_std', data=g, markersize=4,
            ls='', fmt='.', capsize=2, label='Measurement')
ax.axhline(0.85, ls='--', label="Cloud droplet", c='k')
ax.margins(0)
h.set_xticks_and_xlabels(ax, pd.Timedelta(3, 'hours'))
ax.grid()
ax.legend(loc='lower left', ncols=2)
ax.text(0.02, 0.86, '(c)', transform=ax.transAxes, bbox=box_props)
ax.set(xlabel='Time (UTC)',
       ylabel='Asymmetry\n parameter $g$',
       xlim=(start_dt, end_dt),
       ylim=(0.5, 0.9),
       yticks=[0.5, 0.6, 0.7, 0.8, 0.9]
       )

axs = fig.subplot_mosaic([['upper right']],
                         gridspec_kw={
                             'top': 1,
                             'bottom': 0.58,
                             'left': 0.57,
                             'right': 1,
                             'wspace': 0.1,
                             'hspace': 0.3
                         }
                         )
# upper right panel - SMART spectral albedo for all sections
ax = axs['upper right']
sections = dict()
for i, (st, et) in enumerate(zip(start_dts, end_dts)):
    sections[f'mean_spectra_{i}'] = albedo.sel(time=slice(st, et)).mean(dim='time')

# SMART albedo
for section, label in zip(sections, labels):
    sections[section].plot(ax=ax, label=label[-1], linewidth=1)

ax.grid()
ax.legend(loc='upper right', ncol=4, title='Flight leg', fontsize=6)
ax.text(0.01, 0.93, '(d)', transform=ax.transAxes)
ax.set(xlabel='Wavelength (nm)',
       ylabel='Albedo',
       ylim=(0, 1))

axs = fig.subplot_mosaic([['lower right']],
                         gridspec_kw={
                             'top': 0.45,
                             'bottom': 0,
                             'left': 0.51,
                             'right': 1,
                             'wspace': 0.1,
                             'hspace': 0.3
                         }
                         )
# lower right panel - PHIPS images
ax = axs['lower right']
ax.imshow(phips)
box_props = dict(boxstyle='round', fc='white')
ax.text(0.015, 0.935, 'Section 2', bbox=box_props, transform=ax.transAxes, size=6)
ax.text(0.517, 0.935, 'Section 3', bbox=box_props, transform=ax.transAxes, size=6)
ax.text(0.015, 0.43, 'Section 4', bbox=box_props, transform=ax.transAxes, size=6)
ax.text(0.517, 0.43, 'Section 5', bbox=box_props, transform=ax.transAxes, size=6)
ax.axis('off')
ax.set_aspect('auto', adjustable='box')
fig.text(0.52, 0.47, '(e)')

plt.savefig(f'{plot_path}/Fig_arctic_cloud.pdf', dpi=300, bbox_inches='tight')
# plt.show()
plt.close()

# %% testing

