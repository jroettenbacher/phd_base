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


def get_info_from_libradtran_input(filepath: str) -> (float, float, pd.Timestamp, List[str], List[str], bool):
    """
    Open a libRadtran input file and read out some information.

    Args:
        filepath: path to file

    Returns: Some variables (latitude, longitude, time, header of output file, wavelength range, integrate flag)

    """
    # define all possible output values as None in case the right line is not found
    latitude, longitude, time_stamp, header, wavelengths, integrate_flag, zout = None, None, None, None, None, None, None
    with open(filepath, "r") as ifile:
        lines = ifile.readlines()

    for line in lines:
        if line.startswith("latitude"):
            match = re.search(r"(?P<direction>[NS]) (?P<value>[0-9]+\.?[0-9]*)", line)
            if match.group("direction") == "N":
                latitude = float(match.group("value"))
            else:
                assert match.group("direction") == "S", "Direction is not 'N' or 'S'! Check input"
                latitude = -float(match.group("value"))

        if line.startswith("longitude"):
            match = re.search(r"(?P<direction>[EW]) (?P<value>[0-9]+\.?[0-9]*)", line)
            if match.group("direction") == "E":
                longitude = float(match.group("value"))
            else:
                assert match.group("direction") == "W", "Direction is not 'W' or 'E'! Check input"
                longitude = -float(match.group("value"))

        if line.startswith("time"):
            time_stamp = pd.to_datetime(line[5:-1], format="%Y %m %d %H %M %S")

        if line.startswith("output_user"):
            header = line[12:-1].split()

        if line.startswith("wavelength"):
            wavelengths = line[11:].split()

        if line.startswith("output_process"):
            integrate_flag = True if line[15:].strip() == "integrate" else False
        # there is no output process in a spectral run thus integrate_flag is never updated
        if integrate_flag is None:
            integrate_flag = False

        if line.startswith("zout"):
            zout = line[4:].split()

    return latitude, longitude, time_stamp, header, wavelengths, integrate_flag, zout
