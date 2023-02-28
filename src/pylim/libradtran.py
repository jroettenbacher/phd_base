#!/usr/bin/env python
"""Functions to process and plot libRadTran simulation files

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

# %% functions


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
