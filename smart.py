#!/usr/bin/env python
"""Script for processing and plotting of SMART data
author: Johannes Röttenbacher
"""

import os
import re
import logging
import pandas as pd
import numpy as np
import functions_jr as jr
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

lookup = dict(ASP06_J3="PGS_5_(ASP_06)", ASP06_J4="VIS_6_(ASP_06)", ASP06_J5="PGS_6_(ASP_06)", ASP06_J6="VIS_7_(ASP_06)",
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

    # first three columns: Time (hh mm ss.ss), integration time (ms), shutter flag
    header = ["time", "t_int", "shutter"]
    header.extend(pixels)

    df = pd.read_csv(file, sep="\t", header=None, names=header)
    datetime_str = date_str + " " + df["time"]
    df = df.set_index(pd.to_datetime(datetime_str, format="%Y_%m_%d %H %M %S.%f")).drop("time", axis=1)
    return df


def read_pixel_to_wavelength(path: str, spectrometer: str) -> pd.DataFrame:
    """

    Args:
        path: Path where to find file
        spectrometer: Which spectrometer to read in, refer to lookup table for possible spectrometers

    Returns: pandas DataFrame relating pixel number to wavelength

    """
    filename = f"pixel_wl_{lookup[spectrometer]}.dat"
    file = os.path.join(path, filename)
    df = pd.read_csv(file, sep="\s+", skiprows=7, header=None, names=["pixel", "wavelength"])
    return df


def find_pixel(df: pd.DataFrame, wavelength: float()) -> int:
    """
    Given the dataframe with the pixel to wavelength mapping, return the pixel closest to the requested wavelength.
    Args:
        df: Dataframe with column pixel and wavelength (from read_pixel_to_wavelength)
        wavelength: which wavelength are you interested in

    Returns: closest pixel number corresponding to the given wavelength

    """
    idx = jr.argnearest(df["wavelength"], wavelength)
    pixel_nr = df["pixel"].loc[idx]
    return pixel_nr


if __name__ == '__main__':
    # test read in functions
    path = "C:/Users/Johannes/Documents/Doktor/campaigns/CIRRUS-HL/SMART/Calib_Lab_20210329/calib_J3_4"
    filename = "2021_03_29_11_15.Fdw_VNIR.dat"
    smart = read_smart_raw(path, filename)

    pixel_wl_path = "C:/Users/Johannes/Documents/Doktor/instruments/SMART/pixel_wl"
    spectrometer = "ASP06_J4"
    pixel_wl = read_pixel_to_wavelength(pixel_wl_path, spectrometer)

    # find pixel closest to given wavelength
    pixel_nr = find_pixel(pixel_wl, 525)

    # plot netto counts time series
    fig, ax = plt.subplots()
    ax.plot(smart[pixel_nr])
    ax.grid()
    ax = jr.set_xticks_and_xlabels(ax, smart.index[-1] - smart.index[0])
    ax.set_xlabel("Time [UTC]")
    ax.set_ylabel("Netto Counts")
    fig.autofmt_xdate()
    plt.show()
    plt.close()
    # subtract dark current
