#!/usr/bin/env python
"""Write an albedo file for input to libRadtran

Use the IFS output as input for the libRadtran simulation.

Variables to add to the albedo file:

*author*: Johannes Röttenbacher
"""

# %% import modules
import pylim.helpers as h
import pylim.halo_ac3 as meta
from pylim import reader
import pandas as pd
import xarray as xr
import numpy as np

# %% set paths and filenames
campaign = "halo-ac3"
flight_key = "RF17"
run = '00'  # which run to use (0 or 12 Z)
flight = meta.flight_names[flight_key]
date = flight[9:17]
ifs_dir = f"{h.get_path('ifs', campaign=campaign)}/{date}"
libradtran_dir = h.get_path("libradtran", flight, campaign)
output_path = f"{libradtran_dir}/wkdir/albedo_ifs"
h.make_dir(output_path)
bahamas_dir = h.get_path("bahamas", flight, campaign)
bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{flight_key}_v1_1s.nc"
ifs_srf_file = f"ifs_{date}_{run}_sfc.nc"
time_step = pd.Timedelta("1Min")  # time step at which a new file should be generated

# %% read in BAHAMAS data
# bahamas_ds = reader.read_bahamas(f"{bahamas_dir}/{bahamas_file}")  # for original BAHAMAS resolution
bahamas_ds = xr.open_dataset(f"{bahamas_dir}/{bahamas_file}")  # for 1s BAHAMAS resolution

# %% read in ifs file
data_srf = xr.open_dataset(f"{ifs_dir}/{ifs_srf_file}")

# %% read in IFS albedo parameterization for sea ice (6 spectral bands for each month) and select right month
month_id = int(date[4:6]) - 1
ci_albedo_bands = h.ci_albedo[month_id, :]
# albedo wavelength range (start_1, end_1, start_2, end_2, ...) in nanometer
alb_wavelengths = np.array([185, 250, 251, 440, 441, 690, 691, 1190, 1191, 2380, 2381, 2500])
# set spectral ocean albedo to constant
openocean_albedo_bands = np.repeat(0.6, 6)

# %% loop through bahamas timestamps at given time step
timestamp = pd.to_datetime(bahamas_ds.time[0].values)
while timestamp < bahamas_ds.time[-1].values:
    albedo_file = f"{output_path}/{timestamp:%Y%m%d_%H%M%S}_albedo_from_ifs_{run}Z.dat"
    ins_sel = bahamas_ds.sel(time=timestamp)
    ifs_sel = data_srf.sel(time=timestamp, method="nearest")
    lat_id = h.arg_nearest(data_srf.lat, ins_sel["IRS_LAT"].values)
    lon_id = h.arg_nearest(data_srf.lon, ins_sel["IRS_LON"].values)
    ifs_sel = ifs_sel.isel(lat=lat_id, lon=lon_id)

    sw_alb_bands = ifs_sel.CI.values * ci_albedo_bands + (1. - ifs_sel.CI.values) * openocean_albedo_bands

    sw_alb_for_file_list = []
    for i in range(len(sw_alb_bands)):
        sw_alb_for_file_list.append(sw_alb_bands[i])
        sw_alb_for_file_list.append(sw_alb_bands[i])

    sw_alb_for_file = np.asarray(sw_alb_for_file_list)

    # write albedo file
    print('write albedo file')
    with open(albedo_file, 'w') as f:
        f.write('#wavelength (nm) albedo (0-1)\n')
        for i in range(len(alb_wavelengths)):
            f.write(f"{alb_wavelengths[i]:6d} {sw_alb_for_file[i]:6.2f}\n")

    timestamp = timestamp + time_step
