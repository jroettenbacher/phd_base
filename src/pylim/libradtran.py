#!/usr/bin/env python
"""Functions to process and plot libRadtran simulation files

*author*: Johannes Röttenbacher
"""

# %% module import
from pylim.helpers import get_path
from pylim.cirrus_hl import coordinates, radiosonde_stations
import datetime
import pandas as pd
from geopy.distance import geodesic
import re
from typing import List
import logging

log = logging.getLogger(__name__)


def get_netcdf_global_attributes(campaign: str, flight: str, experiment: str):
    """
    Return a dictionary with globa attributes for a libRadtran simulation

    Args:
        campaign: which campaign (cirrus-hl, halo-ac3)
        flight: which flight
        experiment: which experiment has been run

    Returns: dictionary with global attributes

    """

    global_attributes = {
        "cirrus-hl": dict(
            title=f"Simulated downward and upward irradiance along flight track for experiment {experiment}",
            Conventions="CF-1.9",
            camapign_id=f"{campaign.swapcase()}",
            platform_id="HALO",
            version_id="1",
            comment=f"CIRRUS-HL Campaign, Oberpfaffenhofen, Germany, {flight}",
            contact="PI: m.wendisch@uni-leipzig.de, Data: johannes.roettenbacher@uni-leipzig.de",
            history=f"Created {datetime.datetime.utcnow():%c} UTC",
            institution="Leipzig Institute for Meteorology, Leipzig University, Stephanstr.3, 04103 Leipzig, Germany",
            source="libRadtran 2.0.4",
            references="Emde et al. 2016, 10.5194/gmd-9-1647-2016"
        ),
        "halo-ac3": dict(
            title=f"Simulated downward and upward irradiance along flight track for experiment {experiment}",
            Conventions="CF-1.9",
            campaign_id=f"{campaign.swapcase()}",
            platform_id="HALO",
            version_id="1",
            institution="Leipzig Institute for Meteorology, Leipzig, Germany, Stephanstr.3, 04103 Leipzig, Germany",
            history=f"created {datetime.datetime.utcnow():%c} UTC",
            contact="Johannes Röttenbacher, johannes.roettenbacher@uni-leipzig.de",
            PI="André Ehrlich, a.ehrlich@uni-leipzig.de",
            source="libRadtran 2.0.4",
            references="Emde et al. 2016, 10.5194/gmd-9-1647-2016",
        )
    }

    return global_attributes[campaign]


def get_netcdf_variable_attributes(solar_flag: bool, integrate_str: str, wavelength_str: str):
    """
    Return a dictionary with variable attributes for a libRadtran simulation

    Args:
        solar_flag: whether the simulation was a solar or a terrestrial simulation
        integrate_str: whether the simulations was integrated at the end or not
        wavelength_str: wavelength range of the simulation

    Returns: dictionary with variable attributes

    """
    # set up metadata for general variables which are shared between solar and terrestrial simulations
    general_variables = dict(
        latitude=dict(units="degrees_north", long_name="latitude", standard_name="latitude"),
        longitude=dict(units="degrees_east", long_name="longitude", standard_name="longitude"),
        saa=dict(units="degree", long_name="solar azimuth angle", standard_name="soalr_azimuth_angle",
                 comment="clockwise from north"),
        sza=dict(units="degree", long_name="solar zenith angle", standard_name="solar_zenith_angle",
                 comment="0 deg = zenith"),
        CLWD=dict(units="g m^-3", long_name="cloud liquid water density",
                  standard_name="mass_concentration_of_cloud_liquid_water_in_air"),
        CIWD=dict(units="g m^-3", long_name="cloud ice water density",
                  standard_name="mass_concentration_of_cloud_ice_water_in_air"),
        p=dict(units="hPa", long_name="atmospheric pressure", standard_name="air_pressure"),
        T=dict(units="K", long_name="air temperature", standard_name="air_temperature"),
        wavelength=dict(units="nm", long_name="wavelength", standard_name="radiation_wavelength"),
        re_ice=dict(units="mum", long_name="input ice effective radius"),
        iwc=dict(units="g m^-3", long_name="input ice water content")
    )

    # set up metadata dictionaries for solar (shortwave) flux
    solar_variables = dict(
        albedo=dict(units="1", long_name="surface albedo", standard_name="surface_albedo"),
        altitude=dict(units="m", long_name="height above mean sea level", standard_name="altitude"),
        direct_fraction=dict(units="1", long_name="direct fraction of downward irradiance", comment=wavelength_str),
        edir=dict(units="W m-2", long_name=f"{integrate_str}direct beam irradiance",
                  standard_name="direct_downwelling_shortwave_flux_in_air",
                  comment=wavelength_str),
        edn=dict(units="W m-2", long_name=f"{integrate_str}diffuse downward irradiance",
                 standard_name="diffuse_downwelling_shortwave_flux_in_air_assuming_clear_sky",
                 comment=wavelength_str),
        eup=dict(units="W m-2", long_name=f"{integrate_str}diffuse upward irradiance",
                 standard_name="surface_upwelling_shortwave_flux_in_air_assuming_clear_sky",
                 comment=wavelength_str),
        eglo=dict(units="W m-2", long_name=f"{integrate_str}global solar downward irradiance",
                  standard_name="solar_irradiance", comment=wavelength_str),
        enet=dict(units="W m-2", long_name=f"{integrate_str}net irradiance", comment=wavelength_str),
        heat=dict(units="K day-1", long_name="heating rate"),
    )

    # set up metadata dictionaries for terrestrial (longwave) flux
    terrrestrial_variables = dict(
        albedo=dict(units="1", long_name="surface albedo", standard_name="surface_albedo"),
        altitude=dict(units="m", long_name="height above mean sea level", standard_name="altitude"),
        direct_fraction=dict(units="1", long_name="direct fraction of downward irradiance", comment=wavelength_str),
        edir=dict(units="W m-2", long_name=f"{integrate_str}direct beam irradiance",
                  standard_name="direct_downwelling_longwave_flux_in_air",
                  comment=wavelength_str),
        edn=dict(units="W m-2", long_name=f"{integrate_str}downward irradiance",
                 standard_name="downwelling_longwave_flux_in_air_assuming_clear_sky",
                 comment=wavelength_str),
        eup=dict(units="W m-2", long_name=f"{integrate_str}upward irradiance",
                 standard_name="surface_upwelling_longwave_flux_in_air_assuming_clear_sky",
                 comment=wavelength_str),
    )

    if solar_flag:
        attributes_dict = {**general_variables, **solar_variables}
    else:
        attributes_dict = {**general_variables, **terrrestrial_variables}

    return attributes_dict


def find_closest_radiosonde_station(latitude: float, longitude: float):
    """
    Given longitude and latitude, find the closest radiosonde station from the campaign dictionary

    Args:
        latitude: in decimal degrees (N=positive)
        longitude: in decimal degrees (E=positive)

    Returns: Name of the closest radiosonde station

    """
    distances = dict()
    station_names = [station[:-6] for station in radiosonde_stations]  # read out station names from list
    # loop through stations and save distance in km
    for station_name in station_names:
        lon_lat = coordinates[station_name]
        distances[station_name] = geodesic((lon_lat[1], lon_lat[0]), (latitude, longitude)).km

    min_distance = min(distances.values())  # get minimum distance
    closest_station = [s for s in station_names if distances[s] == min_distance][0]
    closest_station = [s for s in radiosonde_stations if closest_station in s][0]  # get complete station name
    log.info(f"Closest Radiosonde station {closest_station} is {min_distance:.1f} km away.")

    return closest_station


def get_info_from_libradtran_input(filepath: str) -> dict:
    """
    Open a libRadtran input file and read out some information.

    Args:
        filepath: path to file

    Returns: Some variables (latitude, longitude, time, header of output file, wavelength range, integrate flag)

    """
    # define all possible output values
    output_values = ["latitude", "longitude", "time_stamp", "header", "wavelengths", "integrate_flag", "zout",
                     "experiment_settings"]
    output_dict = dict()
    with open(filepath, "r") as ifile:
        lines = ifile.readlines()

    for line in lines:
        if line.startswith("latitude"):
            match = re.search(r"(?P<direction>[NS]) (?P<value>[0-9]+\.?[0-9]*)", line)
            if match.group("direction") == "N":
                output_dict["latitude"] = float(match.group("value"))
            else:
                assert match.group("direction") == "S", "Direction is not 'N' or 'S'! Check input"
                output_dict["latitude"] = -float(match.group("value"))

        if line.startswith("longitude"):
            match = re.search(r"(?P<direction>[EW]) (?P<value>[0-9]+\.?[0-9]*)", line)
            if match.group("direction") == "E":
                output_dict["longitude"] = float(match.group("value"))
            else:
                assert match.group("direction") == "W", "Direction is not 'W' or 'E'! Check input"
                output_dict["longitude"] = -float(match.group("value"))

        if line.startswith("time"):
            output_dict["time_stamp"] = pd.to_datetime(line[5:-1], format="%Y %m %d %H %M %S")

        if line.startswith("output_user"):
            output_dict["header"] = line[12:-1].split()

        if line.startswith("wavelength"):
            output_dict["wavelengths"] = line[11:].split()

        if line.startswith("output_process"):
            output_dict["integrate_flag"] = True if line[15:].strip() == "integrate" else False

        if line.startswith("zout"):
            output_dict["zout"] = line[4:].split()

        if line.startswith("# Experiment"):
            # read out experimental values from input file, split line at : or , and strip of whitespace
            tmp = [x.strip() for x in re.split(r"[:,]", line[1:])]
            label, value = tmp[::2], tmp[1::2]
            experiment_settings = dict()
            for l, v in zip(label, value):
                experiment_settings[l] = v
            output_dict["experiment_settings"] = experiment_settings

        # set output to None if not available in input file
        for label in output_values:
            if label not in output_dict:
                output_dict[label] = None
        # integrate flag can be unset in input file and is set to False
        if output_dict["integrate_flag"] is None:
            output_dict["integrate_flag"] = False

    return output_dict
