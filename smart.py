#!/usr/bin/env python
"""Script for processing and plotting of SMART data
author: Johannes Röttenbacher
"""

import os
import re
import logging
import pandas as pd
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

lookup = dict(ASP06_J3="PGS_5_(ASP-06)", ASP06_J4="VIS_6_(ASP_06)", ASP06_J5="PGS_6_(ASP_06)", ASP06_J6="VIS_7_(ASP_06)",
              ASP07_J3="PGS_4_(ASP_07)", ASP07_J4="VIS_8_(ASP_07)")


def read_smart_raw(path: str, filename: str) -> pd.DataFrame:
    """
    Read raw SMART data files
    Args:
        path: Path where to find file
        filename: Name of file

    Returns: pandas DataFrame with column names and datetime index

    """
    file = os.path.join(path, filename)
    # find date string and channel from file
    match = re.search(r"^(?P<date>\d{4}_\d{2}_\d{2}).*_(?P<channel>[A-Z]{4})", filename)
    try:
        date_str = match.group('date')
        channel = match.group('channel')
    except AttributeError:
        log.info("No date and channel information was found! Check filename!")
        raise

    if channel == "SWIR":
        pixels = list(range(1, 257))  # 256 pixels
    elif channel == "VNIR":
        pixels = list(range(1, 1025))  # 1024 pixels

    header = ["time", "t_int", "shutter"]  # first three columns: Time (hh mm ss.ss), integration time (ms), shutter flag
    header.extend(pixels)

    df = pd.read_csv(file, sep="\t", header=None, names=header)
    datetime_str = date_str + " " + df["time"]
    df = df.set_index(pd.to_datetime(datetime_str, format="%Y_%m_%d %H %M %S.%f")).drop("time", axis=1)
    return df


def read_pixel_to_wavelength(path: str, filename: str) -> pd.DataFrame:
    """

    Args:
        path: Path where to find file
        filename: Name of file

    Returns: pandas DataFrame relating pixel number to wavelength

    """
    path = "C:/Users/Johannes/Documents/Doktor/instruments/SMART/pixel_wl"
    filename = "pixel_wl_PGS_5_(ASP_06).dat"

if __name__ == '__main__':
    # test read in function
    path = "C:/Users/Johannes/Documents/Doktor/campaigns/CIRRUS-HL/SMART/Calib_Lab_20210329/calib_J3_4"
    filename = "2021_03_29_11_15.Fdw_SWIR.dat"
    df = read_smart_raw(path, filename)

    # plot netto counts time series
    wavelength =

    # subtract dark current
