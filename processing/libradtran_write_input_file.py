#!/usr/bin/env python
"""Create an input file for libRadtran

* behind some options you find the page number of the manual, where the option is explained in more detail
* set options to "None" if you don't want to use them
* Variables which start with "_" are for internal use only and will not be used as a option for the input file.

author: Johannes Röttenbacher
"""
if __name__ == "__main__":
    # %% module import
    import pylim.helpers as h
    from pylim import reader
    from pylim.libradtran import find_closest_radiosonde_station
    import os
    import datetime
    import numpy as np
    import pandas as pd
    from pysolar.solar import get_altitude
    from global_land_mask import globe
    import logging

    log = logging.getLogger("pylim")
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.WARNING)

    # %% user input
    flight = "Flight_20210629a"
    time_step = pd.Timedelta(minutes=2)

    # %% set paths
    _base_dir = h.get_path("base")
    _libradtran_dir = h.get_path("libradtran", flight)
    _bahamas_dir = h.get_path("bahamas", flight)
    _bahamas_file = [f for f in os.listdir(_bahamas_dir) if f.endswith(".nc")][0]
    radiosonde_path = f"{_base_dir}/../02_Soundings/RS_for_libradtran"
    solar_source_path = f"{_base_dir}/../00_Tools/05_libradtran"
    input_path = f"{_libradtran_dir}/wkdir/smart"  # where to save the created files
    h.make_dir(input_path)  # create directory

    bahamas_ds = reader.read_bahamas(f"{_bahamas_dir}/{_bahamas_file}")
    timestamp = bahamas_ds.time[0]
    while timestamp < bahamas_ds.time[-1]:
        bahamas_ds_sel = bahamas_ds.sel(time=timestamp)
        lat, lon, alt = bahamas_ds_sel.IRS_LAT.values, bahamas_ds_sel.IRS_LON.values, bahamas_ds_sel.IRS_ALT.values
        is_on_land = globe.is_land(lat, lon)  # check if location is over land
        zout = alt / 1000  # page 127; aircraft altitude in km
        radiosonde_station = find_closest_radiosonde_station(lat, lon)
        # need to create a time zone aware datetime object to calculate the solar azimuth angle
        dt_timestamp = datetime.datetime.fromtimestamp(timestamp.values.astype('O')/1e9, tz=datetime.timezone.utc)
        if is_on_land:
            calc_albedo = 0.2  # set albedo to a fixed value (will be updated)
        else:
            # sun altitude measures from ground up -> 90 - sun altitude = solar zenith angle
            # TODO: replace with calculation that includes atmospheric refraction
            cos_sza = np.cos(np.deg2rad(90 - get_altitude(lat, lon, dt_timestamp)))
            calc_albedo = 0.037 / (1.1 * cos_sza ** 1.4 + 0.15)  # calculate albedo after Taylor et al 1996 for sea surface

        # optionally provide libRadtran with sza (libRadtran calculates it itself as well)
        sza_libradtran = 90 - get_altitude(lat, lon, dt_timestamp, elevation=alt)
        if sza_libradtran > 85:
            log.debug(f"Solar zenith angle for {dt_timestamp} is {sza_libradtran:.2f}!\n"
                      f"Skipping this timestamp and moving on to the next one.")
            # increase timestamp by time_step
            timestamp = timestamp + time_step
            continue

        # %% internal variables
        _input_filename = f"{dt_timestamp:%Y%m%d_%H%M%S}_libRadtran.inp"
        _input_filepath = f"{input_path}/{_input_filename}"

        # %% set options for libRadtran run - atmospheric shell
        atmos_settings = dict(
            albedo=f"{calc_albedo:.4f}",
            altitude=0,  # page 80; ground height above sea level in km (0 for over ocean)
            # atmosphere_file="/opt/libradtran/2.0.4/share/libRadtran/data/atmmod/afglms.dat",  # page 81
            data_files_path="/opt/libradtran/2.0.4/share/libRadtran/data",  # location of internal libRadtran data
            latitude=f"N {lat:.6f}" if lat > 0 else f"S {-lat:.6f}",  # page 96
            longitude=f"E {lon:.6f}" if lon > 0 else f"W {-lon:.6f}",  # BAHAMAS: E = positive, W = negative
            mol_file=None,  # page 104
            mol_modify="O3 300 DU",  # page 105
            radiosonde=f"{radiosonde_path}/{radiosonde_station}/{dt_timestamp:%m%d}_12.dat H2O RH",  # page 114
            time=f"{dt_timestamp:%Y %m %d %H %M %S}",  # page 123
            source=f"solar {solar_source_path}/NewGuey2003_BBR.dat",  # page 119
            # sza=f"{sza_libradtran:.4f}",  # page 122
            # verbose="",  # page 123
            # SMART wavelength range (179.5, 2225), BACARDI solar (290, 3600), BACARDI terrestrial (4000, 100000)
            wavelength="179.5, 2225",
            zout=f"{zout:.3f}",  # page 127; altitude in km above surface altitude
        )

        # set options for libRadtran run - radiative transfer equation solver
        rte_settings = dict(
            rte_solver="fdisort2",
        )

        # set options for libRadtran run - post-processing
        postprocess_settings = dict(
            output_user="sza albedo zout edir edn eup",  # page 109
            output_process="integrate",  # page 108
        )

        # %% write input file
        with open(_input_filepath, "w") as ifile:
            ifile.write(f"# libRadtran input file generated with libradtran_write_input_file.py "
                        f"({datetime.datetime.utcnow():%c UTC})\n")
            for settings, line in zip([atmos_settings, rte_settings, postprocess_settings],
                                      ["Atmospheric", "RTE", "Post Process"]):
                ifile.write(f"\n# {line} Settings\n")
                for key, value in settings.items():
                    if value is not None:
                        ifile.write(f"{key} {value}\n")

        # increase timestamp by time_step
        timestamp = timestamp + time_step
        # end of while
