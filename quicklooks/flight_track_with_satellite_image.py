#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 06.09.2023

Plot a flight track together with a satellite image from WorldView (www.worldview.earthdata.nasa.gov) or any other
source
"""
# %% import modules
import datetime
import os

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib import patheffects
import pandas as pd
import xarray as xr
from skimage import io

import pylim.halo_ac3 as meta
import pylim.helpers as h

# %% set paths
campaign = 'halo-ac3'
key = 'RF17'
flight = meta.flight_names[key]
date = flight[9:17]
urldate = datetime.datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')

bahamas_path = h.get_path('bahamas', flight, campaign)
dropsonde_path = h.get_path('dropsondes', flight, campaign)
dropsonde_path = f'{dropsonde_path}/Level_1' if key == 'RF17' else f'{dropsonde_path}/Level_2'
bahamas_file = f'{campaign.casefold()}_HALO_BAHAMAS_{date}_{key}_v1_1s.nc'
dropsonde_files = [f for f in os.listdir(dropsonde_path) if f.endswith('.nc')]
plot_path = os.path.join(h.get_path('plot', flight, campaign), 'sat_images')
os.makedirs(plot_path, exist_ok=True)

# read in data
bahamas = xr.open_dataset(f'{bahamas_path}/{bahamas_file}')
lon, lat = bahamas.IRS_LON, bahamas.IRS_LAT

# read in dropsonde data
dropsondes = dict()
for file in dropsonde_files:
    k = file[-11:-5] if key == 'RF17' else file[27:35].replace('_', '')
    dropsondes[k] = xr.open_dataset(f'{dropsonde_path}/{file}')
    if key == 'RF18':
        dropsondes[k]['ta'] = dropsondes[k].ta - 273.15
        dropsondes[k]['rh'] = dropsondes[k].rh * 100

# Construct Arctic Polar Stereographic projection URL
left, right, bottom, top = 0, 2415343, -1500000, 0
layer = "Bands367"  # 'TrueColor'
url = f'https://gibs.earthdata.nasa.gov/wms/epsg3413/best/wms.cgi?\
version=1.3.0&service=WMS&request=GetMap&\
format=image/png&STYLE=default&bbox={left},{bottom},{right},{top}&CRS=EPSG:3413&\
HEIGHT=8192&WIDTH=8192&TIME={urldate}&layers=MODIS_Terra_CorrectedReflectance_{layer}'

# Request image
img = io.imread(url)

# %% plot satellite image with flight track
data_crs = ccrs.PlateCarree()
plot_crs = ccrs.NorthPolarStereo(central_longitude=-45)
extent = (left, right, bottom, top)
_, ax = plt.subplots(figsize=(30 * h.cm, 15 * h.cm),
                     subplot_kw={'projection': plot_crs},
                     layout='constrained')
# satellite
ax.imshow(img, extent=extent, origin='upper')
# bahamas
ax.plot(lon, lat, color='k', transform=data_crs, label='HALO flight track')
# dropsondes
for i, ds in enumerate(dropsondes.values()):
    ds['alt'] = ds.alt / 1000  # convert altitude to km
    launch_time = pd.to_datetime(ds.launch_time.to_numpy()) if key == 'RF17' else pd.to_datetime(ds.time[0].to_numpy())
    x, y = ds.lon.mean().to_numpy(), ds.lat.mean().to_numpy()
    cross = ax.plot(x, y, 'x', color='orangered', markersize=5, transform=data_crs,
                    zorder=450)
    ax.text(x, y, f'{launch_time:%H:%M}', c='k', fontsize=9, transform=data_crs, zorder=500,
            path_effects=[patheffects.withStroke(linewidth=0.5, foreground='white')])
# add legend artist
ax.plot([], label='Dropsonde', ls='', marker='x', color='orangered', markersize=4)

# add Kiruna and Longyearbyen
for location in ['Kiruna']:
    x, y = meta.coordinates[location]
    ax.plot(x, y, color='red', marker='o', ls='', transform=data_crs,)
    ax.text(x+0.5, y+1.25, f'{location}', c='k', fontsize=9, transform=data_crs)

ax.coastlines(color='k', linewidth=1)
gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
gl.top_labels = False
gl.right_labels = False

ax.legend()
ax.set(
    title=f'{key} - MODIS Terra {layer} Corrected Reflectance'
)

plt.show()
plt.close()
