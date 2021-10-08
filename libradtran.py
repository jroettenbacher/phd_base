#!/usr/bin/env python
"""Process and plot libRadTran simulation files
author: Johannes Röttenbacher
"""

# %% module import
import matplotlib.pyplot as plt
from smart import get_path
import datetime
import pandas as pd
import numpy as np
from cirrus_hl import coordinates, radiosonde_stations
from geopy.distance import geodesic
import re
from typing import Tuple, List
import logging

log = logging.getLogger(__name__)

# %% functions


def read_libradtran(flight: str, filename: str) -> pd.DataFrame:
    """
    Read a libRadtran simulation file and add a DateTime Index.
    Args:
        flight: which flight does the simulation belong to (e.g. Flight_20210629a)
        filename: filename

    Returns: DataFrame with libRadtran output data

    """
    file_path = f"{get_path('libradtran', flight)}/{filename}"
    bbr_sim = pd.read_csv(file_path, sep="\s+", skiprows=34)
    date_dt = datetime.datetime.strptime(flight[7:15], "%Y%m%d")
    date_ts = pd.Timestamp(year=date_dt.year, month=date_dt.month, day=date_dt.day)
    bbr_sim["time"] = pd.to_datetime(bbr_sim["sod"], origin=date_ts, unit="s")  # add a datetime column
    bbr_sim = bbr_sim.set_index("time")  # set it as index

    return bbr_sim


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
    log.info(f"Closest Radiosonde station {closest_station} is {min_distance:.3f} km away.")

    return closest_station


def get_info_from_libradtran_input(filepath: str) -> Tuple[float, float, pd.Timestamp(), List[str]]:
    """
    Open a libradtran input file and read out some information.
    Args:
        filepath: path to file

    Returns: Some variables (latitude, longitude, time, header of output file)

    """
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

    return latitude, longitude, time_stamp, header

