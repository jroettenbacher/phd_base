#!/usr/bin/env python
"""Script for processing and plotting of SMART data
author: Johannes Röttenbacher
"""

import os
import re
import logging
import sys
import pandas as pd
from typing import Tuple
import numpy as np
import toml
import functions_jr as jr
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

lookup = dict(ASP_06_J3="PGS_5_(ASP_06)", ASP_06_J4="VIS_6_(ASP_06)", ASP_06_J5="PGS_6_(ASP_06)",
              ASP_06_J6="VIS_7_(ASP_06)",ASP_07_J3="PGS_4_(ASP_07)", ASP_07_J4="VIS_8_(ASP_07)")


def read_smart_raw(path: str, filename: str) -> Tuple[pd.DataFrame, str, str]:
    """
    Read raw SMART data files
    Args:
        path: Path where to find file
        filename: Name of file

    Returns: pandas DataFrame with column names and datetime index and the channel of the measurement (SWIR or VNIR)

    """
    file = os.path.join(path, filename)
    # find date string and channel from file
    match = re.search(r"^(?P<date>\d{4}_\d{2}_\d{2}).*.(?P<direction>[FI][a-z]{2})_(?P<channel>[A-Z]{4})", filename)
    try:
        date_str = match.group('date')
        channel = match.group('channel')
        direction = match.group('direction')
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
    return df, channel, direction


def read_pixel_to_wavelength(path: str, spectrometer: str) -> pd.DataFrame:
    """
    Read file which maps each pixel to a certain wavelength for a specified spectrometer.
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
    path = "C:/Users/Johannes/Documents/Doktor/campaigns/CIRRUS-HL/SMART/ASP_06_Calib_Lab_20210329/calib_J3_4"
    filename = "2021_03_29_11_07.Fdw_VNIR.dat"
    # filename = "2021_03_29_11_15.Fdw_SWIR.dat"
    smart, channel, direction = read_smart_raw(path, filename)

    pixel_wl_path = "C:/Users/Johannes/Documents/Doktor/instruments/SMART/pixel_wl"
    spectrometer = "ASP_06_J4"
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

    # calculate dark current depending on channel:
    # SWIR: use measurements during shutter phases
    # VNIR: Option 1: use measurements below 290 nm
    #       Option 2: use dark measurements from calibration
    # input: spectrometer, filename, option
    spectrometer = "ASP_06_J4"
    option = 2
    filename = "2021_03_29_11_07.Fdw_VNIR.dat"
    # filename = "2021_03_29_11_15.Fdw_SWIR.dat"

    if os.getcwd().startswith("C"):
        config = toml.load("config.toml")["cirrus-hl"]["jr_local"]
    else:
        config = toml.load("config.toml")["cirrus-hl"]["lim_server"]

    base_dir = config["base_dir"]
    path = os.path.join(base_dir, config["measurement_data"])
    pixel_wl_path = os.path.join(base_dir, config["pixel_to_wavelength"])
    calib_path = os.path.join(base_dir, config["calib_data"])

    smart, channel, direction = read_smart_raw(path, filename)
    pixel_wl = read_pixel_to_wavelength(pixel_wl_path, spectrometer)
    if channel == "VNIR":
        if option == 1:
            last_dark_pixel = find_pixel(pixel_wl, 290)
            dark_pixels = np.arange(1, last_dark_pixel+1)
            dark_current = smart.loc[:, dark_pixels].mean()
            dark_wls = pixel_wl[pixel_wl["pixel"].isin(dark_pixels)]["wavelength"]
            plt.plot(dark_wls, dark_current, color='k')
            plt.axhline(dark_current.mean(), color="orange", label="Mean")
            plt.grid()
            plt.title(f"Dark Current from Spectrometer {spectrometer}, channel: {channel}")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Netto Counts")
            plt.legend()
            plt.show()
            plt.close()
        else:
            assert option == 2, "Option should be either 1 or 2!"
            # read in cali file
            # get path depending on spectrometer and inlet
            instrument = re.search(r'ASP_\d{2}', spectrometer)[0]
            inlet = re.search(r'J\d{1}', spectrometer)[0]
            # find right folder and right cali
            for f in os.listdir(calib_path):
                try:
                    result = re.search(instrument, f)[0]
                    calib_dir = os.path.join(calib_path, f)
                    break
                except TypeError:
                    continue

            dark_dir = []
            for f in os.listdir(calib_dir):
                try:
                    result = re.search(r'dark', f)[0]
                    dark_dir.append(f)
                except TypeError:
                    continue

            if len(dark_dir) > 1:
                for d in dark_dir:
                    try:
                        result = re.search(inlet[1], d)[0]
                        dark_dir = os.path.join(calib_dir, d)
                    except TypeError:
                        continue

            for file in os.listdir(dark_dir):
                try:
                    result = re.search(f'.*.{direction}_{channel}.dat', file)[0]
                    dark_file = file
                except TypeError:
                    continue
            dark_current, channel, direction = read_smart_raw(dark_dir, dark_file)
            dark_current = dark_current.iloc[:, 2:].mean()
            wls = pixel_wl["wavelength"]
            plt.plot(wls, dark_current, color='k')
            plt.axhline(dark_current.mean(), color="orange", label="Mean")
            plt.grid()
            plt.title(f"Dark Current from Spectrometer {spectrometer}, channel: {channel}")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Netto Counts")
            plt.legend()
            plt.show()
            plt.close()

    elif channel == "SWIR":
        # check if the shutter flag was working: If all values are 1 -> shutter flag is probably not working
        if np.sum(smart.shutter == 1) != smart.shutter.shape[0]:
            dark_current = smart.where(smart.shutter == 0).mean().iloc[2:]
            wls = pixel_wl["wavelength"]
            plt.plot(wls, dark_current, color='k')
            plt.axhline(dark_current.mean(), color="orange", label="Mean")
            plt.grid()
            plt.title(f"Dark Current from Spectrometer {spectrometer}, channel: {channel}")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Netto Counts")
            plt.legend()
            plt.show()
            plt.close()
        else:
            log.debug("Shutter flag is probably wrong")
    else:
        log.warning(f"channel should be either 'VNIR' or 'SWIR' but is {channel}!")
        sys.exit(1)

    # subtract dark current
    if option == 1:
        dark_current = dark_current.mean()
    measurement = smart.where(smart.shutter == 1).mean().iloc[2:]
    measurement_cor = measurement - dark_current
    wls = pixel_wl["wavelength"]
    if option == 1:
        plt.axhline(dark_current, label="Dark Current", color='k')
    else:
        plt.plot(wls, dark_current, label="Dark Current", color='k')
    plt.plot(wls, measurement, label="Measurement")
    plt.plot(wls, measurement_cor, label="Corrected Measurement")
    plt.grid()
    plt.title(f"Spectrometer {spectrometer}, channel: {channel}")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Netto Counts")
    plt.legend()
    plt.show()
    plt.close()
