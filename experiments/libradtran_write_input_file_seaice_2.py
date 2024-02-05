#!/usr/bin/env python
"""Set up libRadtran for clearsky simulation along flight path for BACARDI with sea ice albedo included up to 5000nm

Input:

- Dropsonde profiles from the flight
- sea ice albedo parameterization from the IFS
- sea ice concentration from IFS

**Required User Input:**

* campaign
* flight_key (e.g. 'RF17')
* time_step (e.g. 'minutes=1')
* use_smart_ins flag
* solar_flag
* integrate flag
* input_path, this is where the files will be saved to be executed by uvspec

**Output:**

* log file
* input files for libRadtran simulation along flight track

The idea for this script is to generate a dictionary with all options that should be set in the input file.
New options can be manually added to the dictionary.
The options are linked to the page in the manual where they are described in more detail.
Set options to "None" if you don't want to use them.

Furthermore, an albedo file is generated for the solar simulation using the sea ice albedo parameterization from :cite:t:`ebert1993` and the open ocean albedo parameterization from :cite:t:`taylor1996`.
To allow simulations in the thermal infrared region an additional albedo band between 2501 and 4500 nm is added.
The sea ice albedo is set to 0 in this band.

For the terrestrial simulation the albedo is set to a constant value depending on surface type over all wavelengths.

The atmosphere is provided by a dropsonde measurement which was converted to the right input format by an IDL script.

*author*: Johannes Röttenbacher
"""
if __name__ == "__main__":
    # %% module import
    import pylim.helpers as h
    import pylim.halo_ac3 as meta
    from pylim import reader, solar_position
    import os
    import datetime
    import numpy as np
    import pandas as pd
    import xarray as xr
    from global_land_mask import globe

    # %% user input
    campaign = "halo-ac3"
    flight_key = "RF17"
    flight = meta.flight_names[flight_key]
    date = flight[9:17] if campaign == "halo-ac3" else flight[7:15]
    month_id = int(date[4:6]) - 1  # get month id for albedo parameterization
    time_step = "1Min"  # define time steps of simulations
    use_smart_ins = False  # whether to use the SMART INs system or the BAHAMAS file
    solar_flag = True
    integrate = True

    # %% set paths
    base_path = h.get_path("base", flight, campaign)
    ifs_path = f"{h.get_path('ifs', campaign=campaign)}/{date}"
    libradtran_path = h.get_path("libradtran_exp", campaign=campaign)
    solar_source_path = f"/opt/libradtran/2.0.4/share/libRadtran/data/solar_flux"
    # where to save the created files
    input_path = f"{libradtran_path}/wkdir/{flight_key}/seaice_2_{'solar' if solar_flag else 'thermal'}"
    albedo_path = f"{input_path}/albedo_files"
    dropsonde_path = f"{base_path}/../01_soundings/RS_for_libradtran/Dropsondes_HALO/Flight_{date}"
    for path in [input_path, albedo_path]:
        h.make_dir(path)  # create directory

    # %% setup logging
    try:
        file = __file__
    except NameError:
        file = None
    log = h.setup_logging("./logs", file, flight)
    log.info(f"Options Given:\ncampaign: {campaign}\nflight: {flight}\ntimestep: {time_step}\nintegrate: {integrate}"
             f"\nuse_smart_ins: {use_smart_ins}\nsolar flag: {solar_flag}"
             f"\nScript started: {datetime.datetime.utcnow():%c UTC}\nwkdir: {input_path}")

    # %% read in INS data
    if use_smart_ins:
        horidata_dir = h.get_path("horidata", flight, campaign)
        horidata_file = [f for f in os.listdir(horidata_dir) if f.endswith(".nc")][0]
        ins_ds = xr.open_dataset(f"{horidata_dir}/{horidata_file}")
    else:
        bahamas_dir = h.get_path("bahamas", flight, campaign)
        bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{flight_key}_v1.nc"
        ins_ds = reader.read_bahamas(f"{bahamas_dir}/{bahamas_file}")

    # %% read in ifs data
    ifs_ds = xr.open_dataset(f"{ifs_path}/ifs_{date}_00_ml_processed.nc")

    # %% write input files for each timestep
    timestamp = ins_ds.time[0]
    time_step = pd.to_timedelta(time_step)
    while timestamp < ins_ds.time[-1]:
        ins_ds_sel = ins_ds.sel(time=timestamp)
        if use_smart_ins:
            # no temperature and pressure available use norm atmosphere
            lat, lon, alt, pres, temp = ins_ds_sel.lat.values, ins_ds_sel.lon.values, ins_ds_sel.alt.values, 1013.25, 288.15
        else:
            lat, lon, alt, pres, temp = ins_ds_sel.IRS_LAT.values, ins_ds_sel.IRS_LON.values, ins_ds_sel.IRS_ALT.values, ins_ds_sel.PS.values, ins_ds_sel.TS.values

        # select closest column of ifs data
        ifs_sel = ifs_ds.sel(time=timestamp, lat=lat, lon=lon, method="nearest")
        is_on_land = globe.is_land(lat, lon)  # check if location is over land
        zout = alt / 1000  # page 127; aircraft altitude in km

        # need to create a time zone aware datetime object to calculate the solar zenith angle
        dt_timestamp = datetime.datetime.fromtimestamp(timestamp.values.astype('O') / 1e9, tz=datetime.timezone.utc)
        # get time in decimal hours
        decimal_hour = dt_timestamp.hour + dt_timestamp.minute / 60 + dt_timestamp.second / 60 / 60
        year, month, day = dt_timestamp.year, dt_timestamp.month, dt_timestamp.day

        # use dropsondes for radiosonde file
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

        # set albedo for terrestrial simulation
        sza_libradtran = solar_position.get_sza(decimal_hour, lat, lon, year, month, day, pres, temp - 273.15)
        if sza_libradtran > 90:
            log.debug(f"Solar zenith angle for {dt_timestamp} is {sza_libradtran}!\n"
                      f"Setting Ocean Albedo to 0.")
            openocean_albedo = 0.0
        else:
            if is_on_land:
                openocean_albedo = 0.2  # set albedo to a fixed value
            else:
                # calculate cosine of solar zenith angle
                cos_sza = np.cos(np.deg2rad(
                    solar_position.get_sza(decimal_hour, lat, lon, year, month, day, pres, temp - 273.15)))
                # calculate albedo after Taylor et al. 1996 for sea surface
                openocean_albedo = 0.037 / (1.1 * cos_sza ** 1.4 + 0.15)

        # %% filepaths
        input_filepath = f"{input_path}/{dt_timestamp:%Y%m%d_%H%M%S}_libRadtran.inp"
        albedo_filepath = f"{albedo_path}/{dt_timestamp:%Y%m%d_%H%M%S}_albedo.inp"

        # %% set options for libRadtran run - atmospheric shell
        atmos_settings = dict(
            atmosphere_file="/opt/libradtran/2.0.4/share/libRadtran/data/atmmod/afglsw.dat",  # page 81
            albedo=openocean_albedo if not solar_flag else None,  # page 78
            albedo_file=albedo_filepath if solar_flag else None,  # page 79
            altitude=0,  # page 80; ground height above sea level in km (0 for over ocean)
            data_files_path="/opt/libradtran/2.0.4/share/libRadtran/data",  # location of internal libRadtran data
            latitude=f"N {lat:.6f}" if lat > 0 else f"S {-lat:.6f}",  # page 96
            longitude=f"E {lon:.6f}" if lon > 0 else f"W {-lon:.6f}",  # BAHAMAS: E = positive, W = negative
            mol_file=None,  # page 104
            number_of_streams="16",  # page 107
            radiosonde=radiosonde,  # page 114
            time=f"{dt_timestamp:%Y %m %d %H %M %S}",  # page 123
            source=f"solar {solar_source_path}/kurudz_1.0nm.dat" if solar_flag else "thermal",  # page 119
            # sza=f"{sza_libradtran:.4f}",  # page 122
            # verbose="",  # page 123
            # SMART wavelength range (179.5, 2225), BACARDI solar (290, 3600), BACARDI terrestrial (4000, 100000)
            wavelength="290 5000" if solar_flag else "4000 100000",
            zout=f"{zout:.3f}",  # page 127; altitude in km above surface altitude
        )

        # set options for libRadtran run - radiative transfer equation solver
        rte_settings = dict(
            rte_solver="disort",  # page 116
        )

        # set options for libRadtran run - post-processing
        if integrate:
            postprocess_settings = dict(
                output_user="sza albedo zout edir edn eup",  # page 109
                output_process="integrate",  # page 108
            )
        else:
            postprocess_settings = dict(
                output_user="lambda sza zout albedo edir edn eup p T",  # page 109
            )

        # %% write input file
        log.debug(f"Writing input file: {input_filepath}")
        with open(input_filepath, "w") as ifile:
            ifile.write(f"# libRadtran input file generated with libradtran_write_input_file_seaice.py "
                        f"({datetime.datetime.utcnow():%c UTC})\n")
            for settings, line in zip([atmos_settings, rte_settings, postprocess_settings],
                                      ["Atmospheric", "RTE", "Post Process"]):
                ifile.write(f"\n# {line} Settings\n")
                for key, value in settings.items():
                    if value is not None:
                        ifile.write(f"{key} {value}\n")

        # %% write albedo file
        if solar_flag:
            # read in IFS albedo parameterization for sea ice (6 spectral bands for each month) and select right month
            ci_albedo_bands = h.ci_albedo[month_id, :]
            # albedo wavelength range (start_1, end_1, start_2, end_2, ...) in nanometer
            alb_wavelengths = np.array([185, 250, 251, 440, 441, 690, 691, 1190, 1191, 2380, 2381, 2500, 2501, 5500])
            # set spectral ocean albedo to constant
            sza_libradtran = solar_position.get_sza(decimal_hour, lat, lon, year, month, day, pres, temp - 273.15)
            if sza_libradtran > 90:
                log.debug(f"Solar zenith angle for {dt_timestamp} is {sza_libradtran}!\n"
                          f"Setting Ocean Albedo to 0.")
                openocean_albedo_bands = np.repeat(0.0, 7)
            else:
                if is_on_land:
                    openocean_albedo_bands = np.repeat(0.2, 7)  # set albedo to a fixed value
                else:
                    # calculate cosine of solar zenith angle
                    cos_sza = np.cos(np.deg2rad(
                        solar_position.get_sza(decimal_hour, lat, lon, year, month, day, pres, temp - 273.15)))
                    # calculate albedo after Taylor et al. 1996 for sea surface
                    openocean_albedo_bands = np.repeat(0.037 / (1.1 * cos_sza ** 1.4 + 0.15), 7)

            # add an albedo band for the thermal infrared
            ci_albedo_bands = np.append(ci_albedo_bands, [0])
            # calculate spectral albedo bands
            sw_alb_bands = ifs_sel.CI.values * ci_albedo_bands + (1. - ifs_sel.CI.values) * openocean_albedo_bands

            sw_alb_for_file_list = []
            for i in range(len(sw_alb_bands)):
                sw_alb_for_file_list.append(sw_alb_bands[i])
                sw_alb_for_file_list.append(sw_alb_bands[i])

            sw_alb_for_file = np.asarray(sw_alb_for_file_list)

            # write albedo file
            log.debug(f'writing albedo file: {albedo_filepath}')
            with open(albedo_filepath, 'w') as f:
                f.write('# wavelength (nm)\talbedo (0-1)\n')
                for i in range(len(alb_wavelengths)):
                    f.write(f"{alb_wavelengths[i]:6d}\t{sw_alb_for_file[i]:6.2f}\n")

        # %% increase timestamp by time_step
        timestamp = timestamp + time_step
        # end of while

