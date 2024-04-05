#!/usr/bin/env python
"""Create an input files along flight track for libRadtran

Here one can set all options needed in the libRadtran input file.
Given the flight and time step, one input file will then be created for every time step with the fitting lat and lon values along the flight track.

**Required User Input:**

* campaign
* flight (e.g. 'Flight_202170715a' or 'HALO-AC3_20220225_HALO_RF01')
* time_step (e.g. 'minutes=1')
* use_smart_ins flag
* use_dropsonde flag (only available for |haloac3|)
* integrate flag
* input_path, this is where the files will be saved to

**Output:**

* log file
* input files for libRadtran simulation along flight track

The idea for this script is to generate a dictionary with all options that should be set in the input file.
New options can be manually added to the dictionary.
The options are linked to the page in the manual where they are described in more detail.
Set options to "None" if you don't want to use them.
Variables which start with "_" are for internal use only and will not be used as an option for the input file.

*author*: Johannes Röttenbacher
"""

if __name__ == "__main__":
    # %% module import
    import pylim.helpers as h
    from pylim import reader, solar_position
    from pylim.libradtran import find_closest_radiosonde_station
    import os
    import datetime
    import numpy as np
    import pandas as pd
    import xarray as xr
    from global_land_mask import globe

    # %% user input
    campaign = "halo-ac3"
    flight = "HALO-AC3_20220225_HALO_RF01"
    date = flight[9:17] if campaign == "halo-ac3" else flight[7:15]
    time_step = pd.Timedelta(minutes=1)  # define time steps of simulations
    use_smart_ins = False  # whether to use the SMART INs system or the BAHAMAS file
    use_dropsonde = True
    integrate = False

# %% setup logging
    log = h.setup_logging("./logs", __file__, flight)
    log.info(f"Options Given:\ncampaign: {campaign}\nflight: {flight}\ntimestep: {time_step}"
             f"\nScript started: {datetime.datetime.now(datetime.UTC):%c UTC}")

    # %% set paths
    _base_dir = h.get_path("base", flight, campaign)
    _libradtran_dir = h.get_path("libradtran", flight, campaign)
    input_path = f"{_libradtran_dir}/wkdir/smart_spectral"  # where to save the created files
    h.make_dir(input_path)  # create directory
    if use_smart_ins:
        _horidata_dir = h.get_path("horidata", flight, campaign)
        _horidata_file = [f for f in os.listdir(_horidata_dir) if f.endswith(".nc")][0]
        ins_ds = xr.open_dataset(f"{_horidata_dir}/{_horidata_file}")
    else:
        _bahamas_dir = h.get_path("bahamas", flight, campaign)
        _bahamas_file = [f for f in os.listdir(_bahamas_dir) if f.endswith(".nc")][0]
        ins_ds = reader.read_bahamas(f"{_bahamas_dir}/{_bahamas_file}")
    radiosonde_path = f"{_base_dir}/../0{1 if campaign == 'halo-ac3' else 2}_soundings/RS_for_libradtran"
    # only needed for HALO-AC3
    dropsonde_path = f"{_base_dir}/../01_soundings/RS_for_libradtran/Dropsondes_HALO/Flight_{date}"
    solar_source_path = f"{_base_dir}/../00_tools/0{5 if campaign == 'cirrus-hl' else 8}_libradtran"

    timestamp = ins_ds.time[0]
    while timestamp < ins_ds.time[-1]:
        ins_ds_sel = ins_ds.sel(time=timestamp)
        if use_smart_ins:
            # no temperature and pressure available use norm atmosphere
            lat, lon, alt, pres, temp = ins_ds_sel.lat.values, ins_ds_sel.lon.values, ins_ds_sel.alt.values, 1013.25, 288.15
        else:
            lat, lon, alt, pres, temp = ins_ds_sel.IRS_LAT.values, ins_ds_sel.IRS_LON.values, ins_ds_sel.IRS_ALT.values, ins_ds_sel.PS.values, ins_ds_sel.TS.values
        is_on_land = globe.is_land(lat, lon)  # check if location is over land
        zout = alt / 1000  # page 127; aircraft altitude in km

        # need to create a time zone aware datetime object to calculate the solar azimuth angle
        dt_timestamp = datetime.datetime.fromtimestamp(timestamp.values.astype('O')/1e9, tz=datetime.timezone.utc)
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
            radiosonde_station = "Longyearbyen_01004" # standard for HALO-(AC)3
            station_nr = radiosonde_station[-5:]
            if campaign == "halo-ac3":
                radiosonde = f"{radiosonde_path}/Radiosonde_for_libradtran_{station_nr}_{dt_timestamp:%Y%m%d}_12.dat H2O RH"
            else:
                radiosonde = f"{radiosonde_path}/{radiosonde_station}/{dt_timestamp:%m%d}_12.dat H2O RH"

        # get time in decimal hours
        decimal_hour = dt_timestamp.hour + dt_timestamp.minute / 60 + dt_timestamp.second / 60 / 60
        year, month, day = dt_timestamp.year, dt_timestamp.month, dt_timestamp.day

        # optionally provide libRadtran with sza (libRadtran calculates it itself as well)
        sza_libradtran = solar_position.get_sza(decimal_hour, lat, lon, year, month, day, pres, temp - 273.15)
        if sza_libradtran > 90:
            log.debug(f"Solar zenith angle for {dt_timestamp} is {sza_libradtran}!\n"
                      f"Setting Albedo to 0.")
            calc_albedo = 0.0
        else:
            if is_on_land:
                calc_albedo = 0.2  # set albedo to a fixed value (will be updated)
            else:
                # calculate cosine of solar zenith angle
                cos_sza = np.cos(np.deg2rad(
                    solar_position.get_sza(decimal_hour, lat, lon, year, month, day, pres, temp - 273.15)))
                # calculate albedo after Taylor et al 1996 for sea surface
                calc_albedo = 0.037 / (1.1 * cos_sza ** 1.4 + 0.15)

        # %% internal variables
        _input_filename = f"{dt_timestamp:%Y%m%d_%H%M%S}_libRadtran.inp"
        _input_filepath = f"{input_path}/{_input_filename}"

        # %% set options for libRadtran run - atmospheric shell
        atmos_settings = dict(
            albedo=f"{calc_albedo:.4f}",
            altitude=0,  # page 80; ground height above sea level in km (0 for over ocean)
            # atmosphere_file="/opt/libradtran/2.0.4/share/libRadtran/data/atmmod/afglms.dat",  # page 81
            data_files_path="/opt/libradtran/2.0.3/share/libRadtran/data",  # location of internal libRadtran data
            latitude=f"N {lat:.6f}" if lat > 0 else f"S {-lat:.6f}",  # page 96
            longitude=f"E {lon:.6f}" if lon > 0 else f"W {-lon:.6f}",  # BAHAMAS: E = positive, W = negative
            mol_file=None,  # page 104
            mol_modify="O3 300 DU",  # page 105
            # radiosonde=f"{radiosonde_path}/{radiosonde_station}/{dt_timestamp:%m%d}_12.dat H2O RH",  # page 114
            radiosonde=radiosonde,  # page 114
            time=f"{dt_timestamp:%Y %m %d %H %M %S}",  # page 123
            source=f"solar {solar_source_path}/NewGuey2003_BBR.dat",  # page 119
            # sza=f"{sza_libradtran:.4f}",  # page 122
            # verbose="",  # page 123
            # SMART wavelength range (179.5, 2225), BACARDI solar (290, 3600), BACARDI terrestrial (4000, 100000)
            wavelength="179.5 2225",
            zout=f"{zout:.3f}",  # page 127; altitude in km above surface altitude
        )

        # set options for libRadtran run - radiative transfer equation solver
        rte_settings = dict(
            rte_solver="fdisort2",
        )

        # set options for libRadtran run - post-processing
        if integrate:
            postprocess_settings = dict(
                output_user="sza albedo zout edir edn eup",  # page 109
                output_process="integrate",  # page 108
            )
        else:
            postprocess_settings = dict(
                output_user="wavelength sza albedo zout edir edn eup",  # page 109
            )
        # %% write input file
        with open(_input_filepath, "w") as ifile:
            ifile.write(f"# libRadtran input file generated with libradtran_write_input_file.py "
                        f"({datetime.datetime.now(datetime.UTC):%c UTC})\n")
            for settings, line in zip([atmos_settings, rte_settings, postprocess_settings],
                                      ["Atmospheric", "RTE", "Post Process"]):
                ifile.write(f"\n# {line} Settings\n")
                for key, value in settings.items():
                    if value is not None:
                        ifile.write(f"{key} {value}\n")

        # increase timestamp by time_step
        timestamp = timestamp + time_step
        # end of while
