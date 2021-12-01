#!/usr/bin/env python
"""Run uvspec
author: Johannes Röttenbacher
"""
if __name__ == "__main__":
    # %% module import
    import pylim.helpers as h
    from pylim.libradtran import get_info_from_libradtran_input
    import pandas as pd
    import os
    from subprocess import Popen
    from tqdm import tqdm
    from joblib import cpu_count
    import datetime as dt
    from pysolar.solar import get_azimuth
    import logging

    log = logging.getLogger("pylim")
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)

    # %% set options and get files
    flight = "Flight_20210629a"
    uvspec_exe = "/opt/libradtran/2.0.3/bin/uvspec"
    libradtran_base_dir = h.get_path("libradtran", flight)
    libradtran_dir = os.path.join(libradtran_base_dir, "wkdir", "smart")  # file where to find input files
    input_files = [os.path.join(libradtran_dir, f) for f in os.listdir(libradtran_dir)
                   if f.endswith(".inp")]
    input_files.sort()  # sort input files -> output files will be sorted as well
    output_files = [f.replace(".inp", ".out") for f in input_files]
    error_logs = [f.replace(".out", ".log") for f in output_files]

    # %% call uvspec for one file
    # index = 0
    # with open(input_files[index], "r") as ifile, open(output_files[index], "w") as ofile, open(error_logs[index], "w") as lfile:
    #     Popen([uvspec_exe], stdin=ifile, stdout=ofile, stderr=lfile)

    # %% call uvspec for all files

    processes = set()
    max_processes = cpu_count() - 4
    for infile, outfile, log_file in zip(tqdm(input_files, desc="libRadtran simulations"), output_files, error_logs):
        with open(infile, "r") as ifile, open(outfile, "w") as ofile, open(log_file, "w") as lfile:
            processes.add(Popen([uvspec_exe], stdin=ifile, stdout=ofile, stderr=lfile))
        if len(processes) >= max_processes:
            os.wait()
            processes.difference_update([p for p in processes if p.poll() is not None])

    # wait for all simulations to finish
    while len(processes) > 0:
        os.wait()
        # this will remove elements of the set which are also in the list
        # the list has only terminated processes in it, p.poll returns a non None value if the process is still running
        processes.difference_update([p for p in processes if p.poll() is not None])

    # %% merge output files and write a netCDF file

    latitudes, longitudes, time_stamps, saa = list(), list(), list(), list()

    log.info("Reading input files and extracting information from it...")
    for infile in tqdm(input_files, desc="Input files"):
        lat, lon, ts, header, wavelengths, integrate_flag = get_info_from_libradtran_input(infile)
        latitudes.append(lat)
        longitudes.append(lon)
        time_stamps.append(ts)
        # convert timestamp to datetime object with timezone information
        dt_ts = ts.to_pydatetime().astimezone(dt.timezone.utc)
        saa.append(get_azimuth(lat, lon, dt_ts))  # calculate solar azimuth angle

    log.info("Merging all output files and adding information from input files...")
    output = pd.concat([pd.read_csv(file, header=None, names=header, sep="\s+")
                        for file in tqdm(output_files, desc="Output files")])
    output = output.assign(latitude=latitudes)
    output = output.assign(longitude=longitudes)
    output = output.assign(time=time_stamps)
    output = output.assign(saa=saa)
    output = output.set_index(["time"])
    # calculate direct fraction
    output["direct_fraction"] = output["edir"] / (output["edir"] + output["edn"])
    # convert mW/m2 to W/m2 (see page 43 of manual)
    output["edir"] = output["edir"] / 1000
    output["eup"] = output["eup"] / 1000
    output["edn"] = output["edn"] / 1000
    # calculate broadband solar clear sky downward
    output["fdw"] = output["edir"] + output["edn"]
    # convert output altitude to m
    output["zout"] = output["zout"] * 1000
    # set up some metadata
    integrate_str = "integrated " if integrate_flag else ""
    wavelenght_str = f"wavelength range {wavelengths[0]} - {wavelengths[1]} nm"

    # set up meta data dictionaries
    var_attrs = dict(
        albedo=dict(units="1", long_name="surface albedo", standard_name="surface_albedo"),
        altitude=dict(units="m", long_name="height above mean sea level", standard_name="altitude"),
        direct_fraction=dict(units="1", long_name="direct fraction of downward irradiance"),
        edir=dict(units="W m-2", long_name=f"{integrate_str}direct beam irradiance",
                  standard_name="direct_downwelling_shortwave_flux_in_air",
                  comment=wavelenght_str),
        edn=dict(units="W m-2", long_name=f"{integrate_str}diffuse downward irradiance",
                 standard_name="diffuse_downwelling_shortwave_flux_in_air_assuming_clear_sky",
                 comment=wavelenght_str),
        eup=dict(units="W m-2", long_name=f"{integrate_str}diffuse upward irradiance",
                 standard_name="surface_upwelling_shortwave_flux_in_air_assuming_clear_sky",
                 comment=wavelenght_str),
        fdw=dict(units="W m-2", longname=f"{integrate_str}total solar downward irradiance",
                 standard_name="solar_irradiance", comment=wavelenght_str),
        latitude=dict(units="degrees_north", long_name="latitude", standard_name="latitude"),
        longitude=dict(units="degrees_east", long_name="longitude", standard_name="longitude"),
        saa=dict(units="degree", long_name="solar azimuth angle", standard_name="soalr_azimuth_angle",
                 comment="clockwise from north"),
        sza=dict(units="degree", long_name="solar zenith angle", standard_name="solar_zenith_angle",
                 comment="0 deg = zenith"),
    )

    # set up global attributes
    attributes = dict(
        comment=f'CIRRUS-HL Campaign, Oberpfaffenhofen, Germany, {flight}',
        contact='PI: m.wendisch@uni-leipzig.de, Data: johannes.roettenbacher@uni-leipzig.de',
        Conventions='CF-1.9',
        history=f'Created {dt.datetime.utcnow():%c} UTC',
        institution='Leipzig Institute for Meteorology, Leipzig University, Stephanstr.3, 04103 Leipzig, Germany',
        references='Emde et al. 2016, 10.5194/gmd-9-1647-2016',
        source='libRadtran 2.0',
        title='Simulated clear sky downward and upward irradiance along flight track',
    )

    encoding = dict(time=dict(units='seconds since 2021-01-01'))

    ds = output.to_xarray()
    ds.attrs = attributes  # assign global attributes
    ds = ds.rename({"zout": "altitude"})
    # set attributes of each variable
    for var in ds:
        ds[var].attrs = var_attrs[var]
    # save file
    nc_filepath = f"/projekt_agmwend/data/Cirrus_HL/01_Flights/{flight}/libRadtran/{flight}_libRadtran_clearsky_simulation_smart.nc"
    ds.to_netcdf(nc_filepath, encoding=encoding)
    log.info(f"Saved {nc_filepath}")
