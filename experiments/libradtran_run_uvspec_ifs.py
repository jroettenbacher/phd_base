#!/usr/bin/env python
"""Run libRadtran simulation (uvspec) for a flight with IFS input

Given the flight and the path to the uvspec executable, this script calls ``uvspec`` for each input file and writes a log and output file.
It does so in parallel, checking how many CPUs are available.
After that the output files are merged into one data frame and information from the input file is added to write one netCDF file.

The script can be run for one flight or for all flights.

**Required User Input:**

* campaign
* flight_key (e.g. "RF17")
* path to uvspec executable
* wavelength, defines the input folder name which is defined in :ref:`processing:libradtran_write_input_file.py`
* name of netCDF file (optional)

**Output:**

* out and log file for each simulation
* log file for script
* netCDF file with simulation in- and output

*author*: Johannes Röttenbacher
"""

if __name__ == "__main__":
    # %% module import
    import pylim.helpers as h
    import pylim.halo_ac3 as meta
    from pylim.libradtran import get_info_from_libradtran_input
    import numpy as np
    import pandas as pd
    import xarray as xr
    import os
    from subprocess import Popen
    from tqdm import tqdm
    from joblib import cpu_count
    import datetime as dt
    from pysolar.solar import get_azimuth

    # %% set options and get files
    campaign = "halo-ac3"
    # uncomment to run for all flights
    # flight_keys = [key for key in meta.flight_names]
    flight_keys = ["RF17"]
    uvspec_exe = "/opt/libradtran/2.0.4/bin/uvspec"
    wavelength = "ifs"  # will be used as directory name and in outfile name (e.g. smart, bacardi, 500-600nm, ...)
    flights = [meta.flight_names[k] for k in flight_keys]
    for flight in flights:
        flight_key = flight[-4:] if campaign == "halo-ac3" else flight
        date = flight[9:17] if campaign == "halo-ac3" else flight[7:15]
        libradtran_base_path = h.get_path("libradtran_exp", campaign=campaign)
        libradtran_path = os.path.join(libradtran_base_path, "wkdir", wavelength)  # file where to find input files
        input_files = [os.path.join(libradtran_path, f) for f in os.listdir(libradtran_path)
                       if f.endswith(".inp")]
        input_files.sort()  # sort input files -> output files will be sorted as well
        output_files = [f.replace(".inp", ".out") for f in input_files]
        error_logs = [f.replace(".out", ".log") for f in output_files]

        # %% setup logging
        try:
            file = __file__
        except NameError:
            file = None
        log = h.setup_logging("./logs", file, flight_key)
        log.info(f"Options Given:\ncampaign: {campaign}\nflight: {flight}\nwavelength: {wavelength}\n"
                 f"uvspec_exe: {uvspec_exe}\nScript started: {dt.datetime.utcnow():%c UTC}")

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

        # %% check if simulation was successful by checking size of each output file, run at least once
        counter = 0  # add a counter to the loop in case rerunning the simulation doesn't fix the problem
        counterbreak = False
        while True:
            counter += 1
            # break the loop after five iterations
            if counter > 5:
                counterbreak = True  # set a variable to signal the while loop was broken due to the counter
                break
            filesizes = list()
            # get size of each output file
            for f in output_files:
                filesizes.append(os.path.getsize(f))
            median_filesize = np.median(filesizes)  # get median filesize to compare all filesizes with
            # if the filesize is smaller than the median add the file to the rerun list
            rerun = [f for f in output_files if os.path.getsize(f) < median_filesize]

            # rerun simulations or break loop
            if len(rerun) > 0:
                rerun_output_files = rerun
                rerun_input_files = [f.replace(".out", ".inp") for f in rerun_output_files]
                rerun_error_logs = [f.replace(".out", ".log") for f in rerun_output_files]
                processes = set()
                max_processes = cpu_count() - 4
                for infile, outfile, log_file in zip(tqdm(rerun_input_files, desc="libRadtran simulations"),
                                                     rerun_output_files, rerun_error_logs):
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
            else:
                break

        if counterbreak:
            log.info(f"The following simulations don't succeed: {rerun}")
            continue

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
        if "lambda" in header:
            # here a spectral simulation has been performed resulting in more than one line per file
            nr_wavelenghts = len(output["lambda"].unique())  # retrieve the number of wavelengths which were simulated
            time_stamps = np.repeat(time_stamps, nr_wavelenghts)
            # retrieve wavelength independent variables from files
            # since the data frames are all concatenated with their original index we can use only the rows with index 0
            zout = output.loc[0, "zout"] * 1000  # convert output altitude to meters
            sza = output.loc[0, "sza"]

        output = output.assign(time=time_stamps)
        output = output.set_index(["time"]) if integrate_flag else output.set_index(["time", "lambda"])
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
            saa=dict(units="degree", long_name="solar azimuth angle", standard_name="solar_azimuth_angle",
                     comment="clockwise from north"),
            sza=dict(units="degree", long_name="solar zenith angle", standard_name="solar_zenith_angle",
                     comment="0 deg = zenith"),
            CLWD=dict(units="g m^-3", long_name="cloud liquid water density",
                      standard_name="mass_concentration_of_cloud_liquid_water_in_air"),
            CIWD=dict(units="g m^-3", long_name="cloud ice water density",
                      standard_name="mass_concentration_of_cloud_ice_water_in_air"),
            p=dict(units="hPa", long_name="atmospheric pressure", standard_name="air_pressure"),
            T=dict(units="K", long_name="air temperature", standard_name="air_temperature")
        )

        # set up global attributes
        # CIRRUS-HL
        attributes = dict(
            title="Simulated clear sky downward and upward irradiance along flight track",
            Conventions="CF-1.9",
            camapign_id=f"{campaign.swapcase()}",
            platform_id="HALO",
            instrument_id="SMART",
            version_id="1",
            comment=f"CIRRUS-HL Campaign, Oberpfaffenhofen, Germany, {flight}",
            contact="PI: m.wendisch@uni-leipzig.de, Data: johannes.roettenbacher@uni-leipzig.de",
            history=f"Created {dt.datetime.utcnow():%c} UTC",
            institution="Leipzig Institute for Meteorology, Leipzig University, Stephanstr.3, 04103 Leipzig, Germany",
            source="libRadtran 2.0",
            references="Emde et al. 2016, 10.5194/gmd-9-1647-2016"
        )
        # HALO-AC3
        global_attrs = dict(
            title="Simulated clear sky downward and upward irradiance along flight track",
            Conventions="CF-1.9",
            campaign_id=f"{campaign.swapcase()}",
            platform_id="HALO",
            instrument_id="SMART",
            version_id="1",
            institution="Leipzig Institute for Meteorology, Leipzig, Germany, Stephanstr.3, 04103 Leipzig, Germany",
            history=f"created {dt.datetime.utcnow():%c} UTC",
            contact="Johannes Röttenbacher, johannes.roettenbacher@uni-leipzig.de",
            PI="André Ehrlich, a.ehrlich@uni-leipzig.de",
            source="libRadtran 2.0.3",
            references="Emde et al. 2016, 10.5194/gmd-9-1647-2016",
        )

        encoding = dict(time=dict(units="seconds since 2017-01-01"))

        ds = output.to_xarray()
        # overwrite zout and sza in the spectral case
        ds = ds.assign(zout=xr.DataArray(zout, coords={"time": ds.time})) if not integrate_flag else ds
        ds = ds.assign(sza=xr.DataArray(sza, coords={"time": ds.time})) if not integrate_flag else ds

        # add the time dependent variables from the input files
        ds = ds.assign(saa=xr.DataArray(saa, coords={"time": ds.time}))
        ds = ds.assign(latitude=xr.DataArray(latitudes, coords={"time": ds.time}))
        ds = ds.assign(longitude=xr.DataArray(longitudes, coords={"time": ds.time}))

        ds.attrs = global_attrs if campaign == "halo-ac3" else attributes  # assign global attributes
        ds = ds.rename({"zout": "altitude"})
        # set attributes of each variable
        for var in ds:
            ds[var].attrs = var_attrs[var]
            encoding[var] = dict(_FillValue=None)  # remove the default _FillValue attribute from each variable
        # save file
        nc_filepath = f"{libradtran_base_path}/{campaign.swapcase()}_HALO_libRadtran_simulation_{wavelength}_{date}_{flight_key}.nc"
        ds.to_netcdf(nc_filepath, encoding=encoding)
        log.info(f"Saved {nc_filepath}")
