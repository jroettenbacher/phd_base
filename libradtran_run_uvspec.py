#!/usr/bin/env python
"""Run uvspec
author: Johannes Röttenbacher
"""
# %% module import
import pandas as pd
from smart import get_path
import os
from subprocess import Popen
from tqdm import tqdm
from joblib import cpu_count
from datetime import datetime
from libradtran import get_info_from_libradtran_input
import logging

log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)

# %% set options and get files
flight = "Flight_20210715a"
uvspec_exe = "/opt/libradtran/2.0.4/bin/uvspec"
libradtran_dir = get_path("libradtran", flight)
input_files = [os.path.join(libradtran_dir, "wkdir", f) for f in os.listdir(f"{libradtran_dir}/wkdir")
               if f.endswith(".inp")]
input_files.sort()  # sort input files -> output files will be sorted as well
output_files = [f.replace(".inp", ".out") for f in input_files]
error_logs = [f.replace(".out", ".log") for f in output_files]

# %% call uvspec for one file
# index = 0
# with open(input_files[index], "r") as ifile, open(output_files[index], "w") as ofile, open(error_logs[index], "w") as log:
#     Popen([uvspec_exe], stdin=ifile, stdout=ofile, stderr=log)

# %% call uvspec for all files

processes = set()
max_processes = cpu_count() - 4
for infile, outfile, log_file in zip(tqdm(input_files, desc="libRadtran simulations"), output_files, error_logs):
    with open(infile, "r") as ifile, open(outfile, "w") as ofile, open(log_file, "w") as log:
        processes.add(Popen([uvspec_exe], stdin=ifile, stdout=ofile, stderr=log))
    if len(processes) >= max_processes:
        os.wait()
        processes.difference_update([p for p in processes if p.poll() is not None])

# %% merge output files and write a netCDF file
# read input file and extract header and lat lon from it

latitudes, longitudes, time_stamps = list(), list(), list()

for infile in input_files:
    lat, lon, ts, header = get_info_from_libradtran_input(infile)
    latitudes.append(lat)
    longitudes.append(lon)
    time_stamps.append(ts)

output = pd.concat([pd.read_csv(file, header=None, names=header, sep="\s+") for file in output_files])
output = output.assign(latitude=latitudes)
output = output.assign(longitude=longitudes)
output = output.assign(time=time_stamps)
output = output.set_index(["time"])
# convert mW/m2 to W/m2 (see page 43 of manual)
output["edir"] = output["edir"] / 1000
output["eup"] = output["eup"] / 1000
output["edn"] = output["edn"] / 1000
# convert output altitude to m
output["zout"] = output["zout"] * 1000

attributes = dict(
    comment=f'CIRRUS-HL Campaign, Oberpfaffenhofen, Germany, {flight}',
    contact='PI: m.wendisch@uni-leipzig.de, Data: johannes.roettenbacher@uni-leipzig.de',
    Conventions='CF-1.9',
    history=f'Created {datetime.utcnow():%c} UTC',
    institution='Leipzig Institute for Meteorology, Leipzig University, Stephanstr.3, 04103 Leipzig, Germany',
    references='Emde et al. 2016, 10.5194/gmd-9-1647-2016',
    source='libRadtran 2.0',
    title='Simulated clear sky downward and upward irradiance along flight track',
)

encoding = dict(time=dict(units='seconds since 2021-01-01'))

var_attrs = dict(
    albedo=dict(units="1", long_name="surface albedo", standard_name="surface_albedo"),
    altitude=dict(units="m", long_name="height above mean sea level", standard_name="altitude"),
    edir=dict(units="W m-2", long_name="direct beam irradiance",
              standard_name="direct_downwelling_shortwave_flux_in_air"),
    edn=dict(units="W m-2", long_name="diffuse downward irradiance",
             standard_name="diffuse_downwelling_shortwave_flux_in_air_assuming_clear_sky"),
    eup=dict(units="W m-2", long_name="diffuse upward irradiance",
             standard_name="surface_upwelling_shortwave_flux_in_air_assuming_clear_sky"),
    latitude=dict(units="degrees_north", long_name="latitude", standard_name="latitude"),
    longitude=dict(units="degrees_east", long_name="longitude", standard_name="longitude"),
    sza=dict(units="degree", long_name="solar zenith angle", standard_name="solar_zenith_angle"),
)

ds = output.to_xarray()
ds.attrs = attributes  # assign global attributes
ds = ds.rename({"zout": "altitude"})
# set attributes of each variable
for var in ds:
    ds[var].attrs = var_attrs[var]
nc_filepath = f"/projekt_agmwend/data/Cirrus_HL/01_Flights/{flight}/libRadtran/{flight}_libRadtran_clearsky_smart_bb_simulation.nc"
ds.to_netcdf(nc_filepath, encoding=encoding)
log.info(f"Saved {nc_filepath}")
