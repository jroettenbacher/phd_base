#!/usr/bin/env python
"""Script for processing and plotting of SMART data
author: Johannes Röttenbacher
"""

import os
import re
import logging
import pandas as pd

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())


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


if __name__ == '__main__':
    # test read in function
    path = "C:/Users/Johannes/Documents/Doktor/campaigns/CIRRUS-HL/SMART/Calib_Lab_20210329/15.4cm"
    filename = "2021_03_29_08_48.Fdw_SWIR.dat"
    df = read_smart_raw(path, filename)