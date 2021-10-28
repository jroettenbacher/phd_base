#!/usr/bin/env python
"""Create an input file for libRadtran simulation to be used in BACARDI processing
* behind some options you find the page number of the manual, where the option is explained in more detail
* set options to "None" if you don't want to use them
* Variables which start with "_" are for internal use
author: Johannes Röttenbacher
"""
# %% module import
import datetime
import logging
import numpy as np
import pandas as pd
from smart import get_path
from helpers import make_dir
from pysolar.solar import get_altitude
from global_land_mask import globe
from bahamas import read_bahamas
from libradtran import find_closest_radiosonde_station
from cirrus_hl import transfer_calibs

log = logging.getLogger()
log.addHandler(logging.StreamHandler())
log.setLevel(logging.DEBUG)

# %% user input
all_flights = [key for key in transfer_calibs.keys()]  # get all flights from dictionary
all_flights = all_flights[21:22]  # select single flight if needed
time_step = pd.Timedelta(minutes=2)
solar_flag = True  # True for solar wavelength range, False for terrestrial wavelength range

# %% run for all flights
for flight in all_flights:
    # set paths
    _base_dir = get_path("base")
    _libradtran_dir = get_path("libradtran", flight)
    input_path = f"{_libradtran_dir}/wkdir/{'solar' if solar_flag else 'thermal'}"
    make_dir(input_path)  # create directory
    radiosonde_dir = f"{_base_dir}/../02_Soundings/RS_for_libradtran"
    solar_source_path = f"{_base_dir}/../00_Tools/05_libradtran/NewGuey2003_BBR.dat"  # set solar source file

    bahamas_ds = read_bahamas(flight)
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
            cos_sza = np.cos(np.deg2rad(90 - get_altitude(lat, lon, dt_timestamp)))
            # calculate albedo after Taylor et al 1996 for sea surface
            calc_albedo = 0.037 / (1.1 * cos_sza ** 1.4 + 0.15)

        # optionally provide libRadtran with sza (libRadtran calculates it itself as well)
        sza_libradtran = 90 - get_altitude(lat, lon, dt_timestamp, elevation=alt)
        if sza_libradtran > 85:
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
            albedo=f"{calc_albedo:.4f}",
            altitude=0,  # page 80; ground height above sea level in km (0 for over ocean)
            # atmosphere_file="/opt/libradtran/2.0.4/share/libRadtran/data/atmmod/afglms.dat",  # page 81
            data_files_path="/opt/libradtran/2.0.4/share/libRadtran/data",  # location of internal libRadtran data
            latitude=f"N {lat:.6f}" if lat > 0 else f"S {-lat:.6f}",  # page 96
            longitude=f"E {lon:.6f}" if lon > 0 else f"W {-lon:.6f}",  # BAHAMAS: E = positive, W = negative
            mol_file=None,  # page 104
            mol_modify="O3 300 DU",  # page 105
            radiosonde=f"{radiosonde_dir}/{radiosonde_station}/{dt_timestamp:%m%d}_12.dat H2O RH",  # page 114
            time=f"{dt_timestamp:%Y %m %d %H %M %S}",  # page 123
            source=f"solar {solar_source_path}" if solar_flag else "thermal",  # page 119
            sur_temperature="293.15",  # page 121; set to 20 degC for now
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
            ifile.write(f"# libRadtran input file generated with libradtran_write_input_file_bacardi.py "
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
