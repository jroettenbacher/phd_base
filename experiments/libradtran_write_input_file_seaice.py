#!/usr/bin/env python
"""Create an input file for libRadtran simulation to be used in BACARDI processing

* behind some options you find the page number of the manual, where the option is explained in more detail
* set options to "None" if you don't want to use them
* Variables which start with "_" are for internal use
* for HALO-AC3 a fixed radiosonde location (Longyearbyen) is used. Uncomment the find_closest_station line for CIRRUS-HL

author: Johannes Röttenbacher
"""
if __name__ == "__main__":
    # %% module import
    import pylim.helpers as h
    from pylim import reader, solar_position
    import pylim.halo_ac3 as meta
    from pylim.libradtran import find_closest_radiosonde_station
    import os
    import datetime
    import numpy as np
    import pandas as pd
    from global_land_mask import globe
    import logging

    log = logging.getLogger("pylim")
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.DEBUG)

    # %% user input
    campaign = "halo-ac3"
    all_flights = [key for key in meta.transfer_calibs.keys()] if campaign == "cirrus-hl" else list(meta.flight_names.values())
    all_flights = all_flights[18:19]  # select single flight if needed
    time_step = pd.Timedelta(minutes=1)
    solar_flag = True  # True for solar wavelength range, False for terrestrial wavelength range
    use_dropsonde = True if campaign == "halo-ac3" else False

    # %% run for all flights
    for flight in all_flights:
        date = flight[9:17] if campaign == "halo-ac3" else flight[7:15]
        # set paths
        _base_dir = h.get_path("base", campaign=campaign)
        _libradtran_dir = h.get_path("libradtran", flight, campaign)
        _bahamas_dir = h.get_path("bahamas", flight, campaign)
        _bahamas_file = [f for f in os.listdir(_bahamas_dir) if f.endswith(".nc")][0]
        input_path = f"{_libradtran_dir}/wkdir/seaice"
        h.make_dir(input_path)  # create directory
        radiosonde_dir = f"{_base_dir}/../0{1 if campaign == 'halo-ac3' else 2}_soundings/RS_for_libradtran"
        # only needed for HALO-AC3
        dropsonde_path = f"{_base_dir}/../01_soundings/RS_for_libradtran/Dropsondes_HALO/Flight_{date}"
        solar_source_path = f"{_base_dir}/../00_tools/0{5 if campaign == 'cirrus-hl' else 8}_libradtran"

        bahamas_ds = reader.read_bahamas(f"{_bahamas_dir}/{_bahamas_file}")
        timestamp = bahamas_ds.time[0]
        while timestamp < bahamas_ds.time[-1]:
            bahamas_ds_sel = bahamas_ds.sel(time=timestamp)
            lat, lon, alt, pres, temp = bahamas_ds_sel.IRS_LAT.values, bahamas_ds_sel.IRS_LON.values, bahamas_ds_sel.IRS_ALT.values, bahamas_ds_sel.PS.values, bahamas_ds_sel.TS.values
            is_on_land = globe.is_land(lat, lon)  # check if location is over land
            zout = alt / 1000  # page 127; aircraft altitude in km
            # need to create a time zone aware datetime object to calculate the solar azimuth angle
            dt_timestamp = datetime.datetime.fromtimestamp(timestamp.values.astype('O') / 1e9, tz=datetime.timezone.utc)

            # define radiosonde station or dropsonde
            if use_dropsonde:
                dropsonde_files = [f for f in os.listdir(dropsonde_path)]
                dropsonde_files.sort()  # sort dropsonde files
                # read out timestamps from file
                dropsonde_times = [int(f[9:14].replace(" ", "")) for f in dropsonde_files]
                # convert to date time to match with BAHAMAS/INS timestamp
                dropsonde_times = pd.to_datetime(dropsonde_times, unit="s", origin=pd.to_datetime(date))
                # find index of the closest dropsonde to BAHAMAS/INS timestamp
                time_diffs = dropsonde_times - timestamp.values
                idx = np.asarray(np.nonzero(time_diffs == time_diffs.min()))[0][0]
                dropsonde_file = dropsonde_files[idx]
                radiosonde = f"{dropsonde_path}/{dropsonde_file} H2O RH"
            else:
                # radiosonde_station = find_closest_radiosonde_station(lat, lon)
                radiosonde_station = "Longyearbyen_01004"  # standard for HALO-(AC)3
                station_nr = radiosonde_station[-5:]
                if campaign == "halo-ac3":
                    radiosonde = f"{radiosonde_dir}/Radiosonde_for_libradtran_{station_nr}_{dt_timestamp:%Y%m%d}_12.dat H2O RH"
                else:
                    radiosonde = f"{radiosonde_dir}/{radiosonde_station}/{dt_timestamp:%m%d}_12.dat H2O RH"

            # get time in decimal hours
            decimal_hour = dt_timestamp.hour + dt_timestamp.minute / 60 + dt_timestamp.second / 60 / 60
            year, month, day = dt_timestamp.year, dt_timestamp.month, dt_timestamp.day
            if is_on_land:
                calc_albedo = 0.2  # set albedo to a fixed value (will be updated)
            else:
                # calculate cosine of solar zenith angle
                cos_sza = np.cos(np.deg2rad(
                    solar_position.get_sza(decimal_hour, lat, lon, year, month, day, pres, temp - 273.15)))
                # calculate albedo after Taylor et al 1996 for sea surface
                calc_albedo = 0.037 / (1.1 * cos_sza ** 1.4 + 0.15)

            # optionally provide libRadtran with sza (libRadtran calculates it itself as well)
            sza_libradtran = solar_position.get_sza(decimal_hour, lat, lon, year, month, day, pres, temp - 273.15)
            if sza_libradtran > 90:
                log.debug(f"Solar zenith angle for {dt_timestamp} is {sza_libradtran:.2f}!\n"
                          f"Skipping this timestamp and moving on to the next one.")
                # increase timestamp by time_step
                timestamp = timestamp + time_step
                continue

            # %% create input file
            _input_filename = f"{dt_timestamp:%Y%m%d_%H%M%S}_libRadtran.inp"
            _input_filepath = f"{input_path}/{_input_filename}"

            # %% set options for libRadtran run - atmospheric shell
            atmos_settings = dict(
                albedo=0.8,
                altitude=0,  # page 80; ground height above sea level in km (0 for over ocean)
                # atmosphere_file="/opt/libradtran/2.0.4/share/libRadtran/data/atmmod/afglms.dat",  # page 81
                data_files_path="/opt/libradtran/2.0.4/share/libRadtran/data",  # location of internal libRadtran data
                latitude=f"N {lat:.6f}" if lat > 0 else f"S {-lat:.6f}",  # page 96
                longitude=f"E {lon:.6f}" if lon > 0 else f"W {-lon:.6f}",  # BAHAMAS: E = positive, W = negative
                mol_file=None,  # page 104
                mol_modify="O3 300 DU",  # page 105
                radiosonde=radiosonde,  # page 114
                time=f"{dt_timestamp:%Y %m %d %H %M %S}",  # page 123
                source=f"solar {solar_source_path}/NewGuey2003_BBR.dat" if solar_flag else "thermal",  # page 119
                sur_temperature="273.15",  # page 121; set to 0 degC for now
                # sza=f"{sza_libradtran:.4f}",  # page 122
                # verbose="",  # page 123
                # SMART wavelength range (179.5, 2225), BACARDI solar (290, 3600), BACARDI terrestrial (4000, 100000)
                wavelength="290 3600" if solar_flag else "4000 100000",
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
                ifile.write(f"# libRadtran input file generated with {__file__} "
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
