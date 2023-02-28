#!/usr/bin/env python
"""Run clearsky broadband libRadtran simulations for comparison with BACARDI measurements.

The input files come from :ref:`processing:libradtran_write_input_file_bacardi.py`.

**Required User Input:**

* campaign
* flights in a list (optional, if not manually specified all flights will be processed)
* solar_flag, run simulation for solar or for thermal infrared wavelengths?

The script will loop through all files and start simulations for all input files it finds (in parallel for one flight).
If the script is called via the command line it creates a `log` folder in the working directory if necessary and saves a log file to that folder.
It also displays a progress bar in the terminal.
After it is done with one flight, it will collect information from the input files and merge all output files in a dataframe to which it appends the information from the input files.
It converts everything to a netCDF file and writes it to disc with a bunch of metadata included.

**Output:**

* out and log file for each simulation
* log file for script
* netCDF file with simulation in- and output

Run like this:

.. code-block:: shell

    python libradtran_run_uvspec_bacardi.py

*author*: Johannes Röttenbacher
"""
if __name__ == "__main__":
    # %% module import
    import pylim.helpers as h
    import pandas as pd
    from pylim.libradtran import get_info_from_libradtran_input
    import pylim.halo_ac3 as meta
    import os
    from subprocess import Popen
    from tqdm import tqdm
    from joblib import cpu_count
    import datetime as dt
    from pysolar.solar import get_azimuth
    import logging

    # %% set options
    campaign = "halo-ac3"
    # get all flights from dictionary
    all_flights = [key for key in meta.transfer_calibs.keys()] if campaign == "cirrus-hl" else list(meta.flight_names.values())
    all_flights = all_flights[18:19]  # select specific flight[s] if needed

    uvspec_exe = "/opt/libradtran/2.0.4/bin/uvspec"
    solar_flag = True
    solar_str = "solar" if solar_flag else "thermal"

    # %% set up logging to console and file when calling script from console
    log = logging.getLogger("pylim")

    # %% run for all flights
    for flight in all_flights:
        log.info(f"Working on {flight}")
        flight_key = flight[-4:] if campaign == "halo-ac3" else flight
        date = flight[9:17] if campaign == "halo-ac3" else flight[7:15]
        # get files
        libradtran_base_dir = h.get_path("libradtran", flight, campaign)
        libradtran_dir = os.path.join(libradtran_base_dir, "wkdir", solar_str)
        input_files = [os.path.join(libradtran_dir, f) for f in os.listdir(libradtran_dir) if f.endswith(".inp")]
        input_files.sort()  # sort input files -> output files will be sorted as well
        output_files = [f.replace(".inp", ".out") for f in input_files]
        error_logs = [f.replace(".out", ".log") for f in output_files]

        # %% setup logging
        try:
            file = __file__
        except NameError:
            file = None
        log = h.setup_logging("./logs", file, flight_key)
        log.info(f"Options Given:\ncampaign: {campaign}\nflight: {flight}\nwavelength: {solar_str}\n"
                 f"uvspec_exe: {uvspec_exe}\nScript started: {dt.datetime.utcnow():%c UTC}")
        # %% call uvspec for all files
        processes = set()
        max_processes = cpu_count() - 4
        tqdm_desc = f"libRadtran simulations {flight}"
        for infile, outfile, log_file in zip(tqdm(input_files, desc=tqdm_desc), output_files, error_logs):
            with open(infile, "r") as ifile, open(outfile, "w") as ofile, open(log_file, "w") as lfile:
                processes.add(Popen([uvspec_exe], stdin=ifile, stdout=ofile, stderr=lfile))
            if len(processes) >= max_processes:
                os.wait()
                processes.difference_update([p for p in processes if p.poll() is not None])

        # wait for all simulations to finish
        while len(processes) > 0:
            # this will remove elements of the set which are also in the list
            # the list has only terminated processes in it, p.poll returns a non None value if the process is still running
            processes.difference_update([p for p in processes if p.poll() is not None])

        # %% check if all simulations created an output and rerun them if not
        file_check = sum([os.path.getsize(file) == 0 for file in output_files])
        # if file size is 0 -> file is empty
        counter = 0  # add a counter to terminate loop if necessary
        try:
            while file_check > 0:
                files_to_rerun = [f for f in input_files if os.path.getsize(f.replace(".inp", ".out")) == 0]
                # rerun simulations
                for infile in tqdm(files_to_rerun, desc="redo libRadtran simulations"):
                    with open(infile, "r") as ifile, \
                            open(infile.replace(".inp", ".out"), "w") as ofile, \
                            open(infile.replace(".inp", ".log"), "w") as lfile:
                        processes.add(Popen([uvspec_exe], stdin=ifile, stdout=ofile, stderr=lfile))
                    if len(processes) >= max_processes:
                        os.wait()
                        processes.difference_update([p for p in processes if p.poll() is not None])

                # wait for all simulations to finish
                while len(processes) > 0:
                    processes.difference_update([p for p in processes if p.poll() is not None])
                # update file_check
                file_check = sum([os.path.getsize(file) == 0 for file in output_files])
                counter += 1
                if counter > 10:
                    raise UserWarning(f"Simulation of {files_to_rerun} does not compute!\nCheck for other errors!")
        except UserWarning as e:
            log.info(f"{e}\nMoving to next flight")
            continue

        # %% merge output files and write a netCDF file
        latitudes, longitudes, time_stamps, saa = list(), list(), list(), list()

        # read input files and extract information from it
        for infile in input_files:
            lat, lon, ts, header, wavelengths, integrate_flag, zout = get_info_from_libradtran_input(infile)
            latitudes.append(lat)
            longitudes.append(lon)
            time_stamps.append(ts)
            # convert timestamp to datetime object with timezone information
            dt_ts = ts.to_pydatetime().astimezone(dt.timezone.utc)
            saa.append(get_azimuth(lat, lon, dt_ts))  # calculate solar azimuth angle

        # merge all output files and add information from input files
        output = pd.concat([pd.read_csv(file, header=None, names=header, sep="\s+") for file in output_files])
        output = output.assign(latitude=latitudes)
        output = output.assign(longitude=longitudes)
        output = output.assign(time=time_stamps)
        output = output.assign(saa=saa)
        output = output.set_index(["time"])
        # calculate direct fraction
        output["direct_fraction"] = output["edir"] / (output["edir"] + output["edn"])
        if solar_flag:
            # convert mW/m2 to W/m2 (see page 43 of manual)
            output["edir"] = output["edir"] / 1000
            output["eup"] = output["eup"] / 1000
            output["edn"] = output["edn"] / 1000
            # calculate solar broadband clear sky downward irradiance
            output["fdw"] = output["edir"] + output["edn"]

        # convert output altitude to m
        output["zout"] = output["zout"] * 1000
        # set up some metadata
        integrate_str = "integrated " if integrate_flag else ""
        wavelenght_str = f"wavelength range {wavelengths[0]} - {wavelengths[1]} nm"

        # set up meta data dictionaries for solar (shortwave) flux
        var_attrs_solar = dict(
            albedo=dict(units="1", long_name="surface albedo", standard_name="surface_albedo"),
            altitude=dict(units="m", long_name="height above mean sea level", standard_name="altitude"),
            direct_fraction=dict(units="1", long_name="direct fraction of downward irradiance", comment=wavelenght_str),
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

        # set up meta data dictionaries for terrestrial (longwave) flux
        var_attrs_terrestrial = dict(
            albedo=dict(units="1", long_name="surface albedo", standard_name="surface_albedo"),
            altitude=dict(units="m", long_name="height above mean sea level", standard_name="altitude"),
            direct_fraction=dict(units="1", long_name="direct fraction of downward irradiance", comment=wavelenght_str),
            edir=dict(units="W m-2", long_name=f"{integrate_str}direct beam irradiance",
                      standard_name="direct_downwelling_longwave_flux_in_air",
                      comment=wavelenght_str),
            edn=dict(units="W m-2", long_name=f"{integrate_str}downward irradiance",
                     standard_name="downwelling_longwave_flux_in_air_assuming_clear_sky",
                     comment=wavelenght_str),
            eup=dict(units="W m-2", long_name=f"{integrate_str}upward irradiance",
                     standard_name="surface_upwelling_longwave_flux_in_air_assuming_clear_sky",
                     comment=wavelenght_str),
            latitude=dict(units="degrees_north", long_name="latitude", standard_name="latitude"),
            longitude=dict(units="degrees_east", long_name="longitude", standard_name="longitude"),
            saa=dict(units="degree", long_name="solar azimuth angle", standard_name="soalr_azimuth_angle",
                     comment="clockwise from north"),
            sza=dict(units="degree", long_name="solar zenith angle", standard_name="solar_zenith_angle",
                     comment="0 deg = zenith"),
        )

        # set up global attributes
        attributes = dict(
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
            source="libRadtran 2.0.4",
            references="Emde et al. 2016, 10.5194/gmd-9-1647-2016",
        )

        encoding = dict(time=dict(units='seconds since 2021-01-01'))

        ds = output.to_xarray()  # convert dataframe to dataset
        ds.attrs = attributes  # assign global attributes
        ds = ds.rename({"zout": "altitude"})
        # set attributes of each variable
        var_attrs = var_attrs_solar if solar_flag else var_attrs_terrestrial
        for var in ds:
            ds[var].attrs = var_attrs[var]
        # save file
        nc_filepath = f"{libradtran_base_dir}/{campaign.swapcase()}_HALO_libRadtran_bb_clearsky_simulation_{solar_str}_{date}_{flight_key}.nc"
        ds.to_netcdf(nc_filepath, encoding=encoding)
        log.info(f"Saved {nc_filepath}")
