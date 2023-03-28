#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 28-03-2023

Set up libRadtran like ecRad using input from the IFS and defining an ice cloud along track using the VarCloud ouput from Florian Ewald, LMU.

Simulate an ice cloud along the above cloud section RF17 (2022-04-11 10:48:47 - 11:07:14 UTC) with IWC and
:math:`r_{eff, ice}` from the VarCloud retrieval.
Generate output at every height level defined by the VarCloud input.

Input:

- IFS processed input file from :py:mod:`ecrad_read_ifs.py`
- VarCloud output file provided by Florian Ewald

**Required User Input:**

* campaign
* flight_key (e.g. 'RF17')
* time_step (e.g. 'minutes=1')
* use_smart_ins flag
* integrate flag
* input_path, this is where the files will be saved to be executed by uvspec

**Output:**

* log file
* input files for libRadtran simulation along flight track

The idea for this script is to generate a dictionary with all options that should be set in the input file.
New options can be manually added to the dictionary.
The options are linked to the page in the manual where they are described in more detail.
Set options to "None" if you don't want to use them.

Furthermore, an atmosphere, an albedo and an ice cloud file are generated.

Variables to add to the atmosphere files:

1 Altitude above sea level in km |rarr| IFS
2 Pressure in hPa |rarr| IFS
3 Temperature in K |rarr| IFS
4 air density in cm−3 |rarr| IFS
5 Ozone density in cm−3 |rarr| sonde (CAMS)
6 Oxygen density in cm−3 |rarr| constant
7 Water vapour density in cm−3 |rarr| IFS
8 CO2 density in cm−3 |rarr| constant (CAMS)

Variables to add to the albedo files:

- spectral albedo from IFS

Variables to add to the cloud files:

- cloud fraction (cloud fraction file)
- ice water content, ice effective radius (ice cloud file)

"""

if __name__ == "__main__":
    # %% module import
    import pylim.helpers as h
    import pylim.halo_ac3 as meta
    from pylim import reader
    from pylim import meteorological_formulas as met
    from pylim.ecrad import apply_ice_effective_radius, apply_liquid_effective_radius
    import os
    import datetime
    import numpy as np
    import pandas as pd
    import xarray as xr
    from metpy.units import units
    from global_land_mask import globe
    from tqdm import tqdm

    # %% user input
    experiment = "varcloud"
    campaign = "halo-ac3"
    flight_key = "RF17"
    flight = meta.flight_names[flight_key]
    date = flight[9:17] if campaign == "halo-ac3" else flight[7:15]
    month_id = int(date[4:6]) - 1  # get month id for albedo parameterization
    time_step = pd.Timedelta(minutes=1)  # define time steps of simulations
    use_smart_ins = False  # whether to use the SMART INs system or the BAHAMAS file
    integrate = False

    # %% set paths
    base_path = h.get_path("base", flight, campaign)
    ifs_path = f"{h.get_path('ifs', campaign=campaign)}/{date}"
    libradtran_path = h.get_path("libradtran_exp", campaign=campaign)
    varcloud_path = h.get_path("varcloud", flight, campaign)
    varcloud_file = [f for f in os.listdir(varcloud_path) if "nc" in f][0]
    solar_source_path = f"/opt/libradtran/2.0.4/share/libRadtran/data/solar_flux"
    input_path = f"{libradtran_path}/wkdir/{flight_key}/{experiment}"  # where to save the created files
    atmosphere_path = f"{input_path}/atmosphere_files"
    albedo_path = f"{input_path}/albedo_files"
    cloud_path = f"{input_path}/cloud_files"
    for path in [input_path, atmosphere_path, albedo_path, cloud_path]:
        h.make_dir(path)  # create directory

    # %% setup logging
    try:
        file = __file__
    except NameError:
        file = None
    log = h.setup_logging("./logs", file, flight)
    log.info(f"Options Given:\n"
             f"\nexperiment: {experiment}\n"
             f"campaign: {campaign}\n"
             f"flight: {flight}\n"
             f"wkdir: {input_path}\n"
             f"timestep: {time_step}\n"
             f"Script started: {datetime.datetime.utcnow():%c UTC}\n")

    # %% read in INS data
    if use_smart_ins:
        horidata_dir = h.get_path("horidata", flight, campaign)
        horidata_file = [f for f in os.listdir(horidata_dir) if f.endswith(".nc")][0]
        ins_ds = xr.open_dataset(f"{horidata_dir}/{horidata_file}")
    else:
        bahamas_dir = h.get_path("bahamas", flight, campaign)
        bahamas_file = f"HALO-AC3_HALO_BAHAMAS_{date}_{flight_key}_v1.nc"
        ins_ds = reader.read_bahamas(f"{bahamas_dir}/{bahamas_file}")

    # %% read in ifs and varcloud data
    ifs_ds = xr.open_dataset(f"{ifs_path}/ifs_{date}_00_ml_processed.nc")
    varcloud_ds = xr.open_dataset(f"{varcloud_path}/{varcloud_file}").swap_dims(time="Time", height="Height").rename(
        Time="time")
    # only use values with height above 0
    varcloud_ds = varcloud_ds.where(varcloud_ds.Height > 0, drop=True)
    varcloud_ds["Height"] = varcloud_ds.Height / 1000

    # %% define zout
    zout = np.flip(np.arange(0, varcloud_ds.Height.max(), 0.1))
    zout_list = [f"{z:.3f}" for z in zout[::-1]]
    zout_str = " ".join(zout_list)

    # %% interpolate varcloud to regular zout grid
    varcloud_ds = varcloud_ds.interp(Height=zout)

    # %% write input files for each timestep
    timestamps = pd.date_range("2022-04-11 10:48", "2022-04-11 11:08", freq=time_step)
    for timestamp in tqdm(timestamps, desc="Write input files"):
        ins_ds_sel = ins_ds.sel(time=timestamp)
        if use_smart_ins:
            # no temperature and pressure available use norm atmosphere
            lat, lon, alt, pres, temp = ins_ds_sel.lat.values, ins_ds_sel.lon.values, ins_ds_sel.alt.values, 1013.25, 288.15
        else:
            lat, lon, alt, pres, temp = ins_ds_sel.IRS_LAT.values, ins_ds_sel.IRS_LON.values, ins_ds_sel.IRS_ALT.values, ins_ds_sel.PS.values, ins_ds_sel.TS.values

        # select closest column of ifs data
        ifs_sel = ifs_ds.sel(time=timestamp, lat=lat, lon=lon, method="nearest")
        # calculate ice and liquid effective radius
        ifs_sel = apply_liquid_effective_radius(ifs_sel)
        ifs_sel = apply_ice_effective_radius(ifs_sel)
        # interpolate data to half levels
        ifs_sel = ifs_sel.interp(level=ifs_sel.half_level, kwargs={"fill_value": "extrapolate"})
        # drop TOA level
        ifs_sel = ifs_sel.isel(half_level=range(1, len(ifs_sel.half_level)))

        varcloud_sel = varcloud_ds.sel(time=timestamp, method="nearest")

        is_on_land = globe.is_land(lat, lon)  # check if location is over land

        # need to create a time zone aware datetime object to calculate the solar azimuth angle
        try:
            dt_timestamp = datetime.datetime.fromtimestamp(timestamp.values.astype('O') / 1e9, tz=datetime.timezone.utc)
        except AttributeError:
            dt_timestamp = timestamp
        # get time in decimal hours
        decimal_hour = dt_timestamp.hour + dt_timestamp.minute / 60 + dt_timestamp.second / 60 / 60
        year, month, day = dt_timestamp.year, dt_timestamp.month, dt_timestamp.day

        # # set a constant albedo if needed
        # sza_libradtran = solar_position.get_sza(decimal_hour, lat, lon, year, month, day, pres, temp - 273.15)
        # if sza_libradtran > 90:
        #     log.debug(f"Solar zenith angle for {dt_timestamp} is {sza_libradtran}!\n"
        #               f"Setting Albedo to 0.")
        #     calc_albedo = 0.0
        # else:
        #     if is_on_land:
        #         calc_albedo = 0.2  # set albedo to a fixed value
        #     else:
        #         # calculate cosine of solar zenith angle
        #         cos_sza = np.cos(np.deg2rad(
        #             solar_position.get_sza(decimal_hour, lat, lon, year, month, day, pres, temp - 273.15)))
        #         # calculate albedo after Taylor et al. 1996 for sea surface
        #         calc_albedo = 0.037 / (1.1 * cos_sza ** 1.4 + 0.15)

        # %% filepaths
        input_filepath = f"{input_path}/{dt_timestamp:%Y%m%d_%H%M%S}_libRadtran.inp"
        atmosphere_filepath = f"{atmosphere_path}/{dt_timestamp:%Y%m%d_%H%M%S}_atmosphere.inp"
        albedo_filepath = f"{albedo_path}/{dt_timestamp:%Y%m%d_%H%M%S}_albedo.inp"
        cloud_fraction_filepath = f"{cloud_path}/{dt_timestamp:%Y%m%d_%H%M%S}_cloud_fraction.inp"
        ice_cloud_filepath = f"{cloud_path}/{dt_timestamp:%Y%m%d_%H%M%S}_ice_cloud.inp"

        # %% set options for libRadtran run - atmospheric shell
        atmos_settings = dict(
            # albedo=f"{calc_albedo:.4f}" if calc_albedo else None,
            # atmosphere_file="/opt/libradtran/2.0.4/share/libRadtran/data/atmmod/afglms.dat",  # page 81
            atmosphere_file=atmosphere_filepath,
            albedo_file=albedo_filepath,  # page 79
            cloud_fraction_file=cloud_fraction_filepath,  # page 86
            cloud_overlap="off",  # page 86, anything else causes libradtran to quit without throwing an error
            ic_file=f"1D {ice_cloud_filepath}",  # page 90
            ic_properties="fu interpolate",  # page 92
            ic_fu="reff_def on",  # page 90
            # wc_file=f"1D {water_cloud_filepath}",  # page 124
            # wc_properties="mie interpolate",  # page 126
            altitude=0,  # page 80; ground height above sea level in km (0 for over ocean)
            data_files_path="/opt/libradtran/2.0.4/share/libRadtran/data",  # location of internal libRadtran data
            latitude=f"N {lat:.6f}" if lat > 0 else f"S {-lat:.6f}",  # page 96
            longitude=f"E {lon:.6f}" if lon > 0 else f"W {-lon:.6f}",  # BAHAMAS: E = positive, W = negative
            mol_file=None,  # page 104
            number_of_streams="16",  # page 107
            # mol_modify="O3 300 DU",  # page 105
            # radiosonde=radiosonde,  # page 114
            time=f"{dt_timestamp:%Y %m %d %H %M %S}",  # page 123
            source=f"solar {solar_source_path}/kurudz_1.0nm.dat",  # page 119
            # sza=f"{sza_libradtran:.4f}",  # page 122
            # verbose="",  # page 123
            # SMART wavelength range (179.5, 2225), BACARDI solar (290, 3600), BACARDI terrestrial (4000, 100000)
            wavelength="250 2225",  # start with 250 due to fu parameterization
            zout=zout_str,  # page 127; altitude in km above surface altitude
        )

        # set options for libRadtran run - radiative transfer equation solver
        rte_settings = dict(
            rte_solver="disort",  # page 116
        )

        # set options for libRadtran run - post-processing
        if integrate:
            postprocess_settings = dict(
                output_user="sza albedo zout edir edn eup heat enet eglo",  # page 109
                output_process="integrate",  # page 108
            )
        else:
            postprocess_settings = dict(
                output_user="lambda sza zout albedo edir edn eup heat enet eglo p T CLWD CIWD",  # page 109
            )

        # %% write input file
        log.debug(f"Writing input file: {input_filepath}")
        with open(input_filepath, "w") as ifile:
            ifile.write(f"# libRadtran input file generated with {file} "
                        f"({datetime.datetime.utcnow():%c UTC})\n")
            for settings, line in zip([atmos_settings, rte_settings, postprocess_settings],
                                      ["Atmospheric", "RTE", "Post Process"]):
                ifile.write(f"\n# {line} Settings\n")
                for key, value in settings.items():
                    if value is not None:
                        ifile.write(f"{key} {value}\n")

        # %% write atmosphere file
        output_altitudes = zout
        alt_atmos = np.flip(met.barometric_height(ifs_sel.pressure_hl.values.flatten(),
                                                  ifs_sel.temperature_hl.values.flatten()))
        alt_atmos[0] = 80000  # set uppermost layer to 80km
        alt_atmos = alt_atmos / 1000  # convert to km
        n_levels = len(output_altitudes)  # number of levels
        # overwrite half_level coordinate with actual altitude
        ifs = ifs_sel.assign_coords(half_level=alt_atmos)
        # interpolate to output altitude grid
        ifs_inp = ifs.interp(half_level=output_altitudes)
        p_atmos = ifs_inp.pressure_hl.values.flatten() / 100
        t_atmos = ifs_inp.temperature_hl.values.flatten()
        k_b = 1.380649e-23  # Boltzmann constant
        air_density = (p_atmos * units.hPa) / (k_b * (units.J / units.K) * t_atmos * units.K)
        air_density = air_density.to('cm^-3')
        o3_density = (ifs_inp["o3_vmr"].values * air_density).magnitude
        o2_density = (ifs_inp["o2_vmr"].values * air_density).magnitude
        molar_mass_air = 28.949 * units.g / units.mol  # mean Molar mass of air
        molar_mass_h2o = 18 * units.g / units.mol
        h2o_density = (ifs_inp["q"].values.flatten() * molar_mass_air / molar_mass_h2o * air_density).magnitude
        co2_density = (ifs_inp["co2_vmr"].values * air_density).magnitude
        air_density = air_density.magnitude

        log.debug(f"Writing atmosphere file: {atmosphere_filepath}")
        with open(atmosphere_filepath, 'w') as f:
            f.write(
                "# z (km)\tpressure (hPa)\ttemperature (K)\tair density (cm^-3)\to3 density (cm^-3)\to2 density (cm^-3)\th2o density (cm^-3)\tco2 density (cm^-3)\n")
            for i in range(n_levels):
                f.write(
                    f"{output_altitudes[i]:6.3f}\t{p_atmos[i]:10.5f}\t{t_atmos[i]:9.3f}\t{air_density[i]:14E}\t{o3_density[i]:14E}\t{o2_density[i]:14E}\t{h2o_density[i]:14E}\t{co2_density[i]:14E}\n")

        # %% write albedo file
        # read in IFS albedo parameterization for sea ice (6 spectral bands for each month) and select right month
        ci_albedo_bands = h.ci_albedo[month_id, :]
        # albedo wavelength range (start_1, end_1, start_2, end_2, ...) in nanometer
        alb_wavelengths = np.array([185, 250, 251, 440, 441, 690, 691, 1190, 1191, 2380, 2381, 2500])
        # set spectral ocean albedo to constant
        openocean_albedo_bands = np.repeat(0.6, 6)
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

        # %% write cloud files
        cloud_fraction = varcloud_sel["ERA5_Fraction_of_Cloud_Cover"].to_numpy()
        nan_replace = np.isnan(varcloud_sel["Varcloud_Cloud_Ice_Water_Content"])
        iwc = (varcloud_sel["Varcloud_Cloud_Ice_Water_Content"] * 1000).where(~nan_replace, 0).to_numpy()
        re_ice = varcloud_sel["Varcloud_Cloud_Ice_Effective_Radius"] * 1e6
        min_re_ice, max_re_ice = 9.315, 65.120
        max_re_ice_replace = re_ice > max_re_ice
        min_re_ice_replace = re_ice < min_re_ice
        # replace nan with 0, too large values with max_re_ice and too small values with min_re_ice
        re_ice = re_ice.where(~nan_replace, 0).where(~max_re_ice_replace, max_re_ice).where(~min_re_ice_replace, min_re_ice).to_numpy()

        log.debug(f"Writing cloud fraction file: {cloud_fraction_filepath}")
        with open(cloud_fraction_filepath, "w") as f:
            f.write("# z (km)\tCF (0-1)\n")
            for i in range(n_levels):
                f.write(f"{output_altitudes[i]:6.3f}\t{cloud_fraction[i]:4.2f}\n")

        log.debug(f"Writing ice cloud file: {ice_cloud_filepath}")
        with open(ice_cloud_filepath, "w") as f:
            f.write("# Height (km)\tIWC (g/m3)\treff (mum)\n")
            for i in range(n_levels):
                f.write(f"{output_altitudes[i]:6.3f}\t{iwc[i]:10.8f}\t{re_ice[i]:8.5f}\n")

        # end of for
