#!/usr/bin/env python
"""A collection of reader functions for instruments operated on HALO and model output files
author: Johannes Röttenbacher
"""

import os
import xarray as xr
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pylim import helpers as h
from pylim import smart
from pylim import cirrus_hl
import logging
log = logging.getLogger(__name__)


def read_bacardi_raw(filename: str, path: str) -> xr.Dataset:
    """
    Read raw BACARDI data as provided by DLR

    Args:
        filename: name of file
        path: path to file

    Returns: Dataset with BACARDI data and time as dimension

    """
    filepath = os.path.join(path, filename)
    date = re.search(r"\d{8}", filename)[0]
    ds = xr.open_dataset(filepath)
    ds = ds.rename({"TIME": "time"})
    ds = ds.swap_dims({"tid": "time"})  # make time the dimension
    # overwrite time to make a datetime index
    ds = ds.assign({"time": pd.to_datetime(ds.time, unit='ms', origin=pd.to_datetime(date, format="%Y%m%d"))})

    return ds


def read_bahamas(bahamas_path: str) -> xr.Dataset:
    """
    Reader function for netcdf BAHAMAS data as provided by DLR.

    Args:
        bahamas_path: full path of netcdf file

    Returns: xr.DataSet with BAHAMAS data and time as dimension

    """
    ds = xr.open_dataset(bahamas_path)
    ds = ds.swap_dims({"tid": "TIME"})
    ds = ds.rename({"TIME": "time"})

    return ds


def read_lamp_file(plot: bool = True, save_fig: bool = True, save_file: bool = True) -> pd.DataFrame:
    """
    Read in the 1000W lamp specification file interpolated to 1nm steps. Converts W/cm^2 to W/m^2.

    Args:
        plot: plot lamp file?
        save_fig: save figure to standard plot path defined in config.toml?
        save_file: save lamp file to standard calib path deined in config.toml?

    Returns: A data frame with the irradiance in W/m² and the corresponding wavelength in nm

    """
    # set paths
    calib_path = h.get_path("calib")
    plot_path = h.get_path("plot")
    lamp_path = h.get_path("lamp")  # get path to lamp defined in config.toml
    # read in lamp file
    lamp_file = "F1587i01_19.std"
    names = ["Irradiance"]  # column name
    lamp = pd.read_csv(os.path.join(lamp_path, lamp_file), skiprows=1, header=None, names=names)
    lamp["Wavelength"] = np.arange(250, 2501)
    # convert from W/cm^2 to W/m^2; cm = m * 10^-2 => cm^2 = (m * 10^-2)^2 = m^2 * 10^-4 => W*10^4/m^2
    lamp["Irradiance"] = lamp["Irradiance"] * 1e4
    if plot:
        # plot lamp calibration
        lamp.plot(x="Wavelength", y="Irradiance", ylabel="Irradiance (W$\\,$m$^{-2}$)", xlabel="Wavelenght (nm)",
                  legend=False, title="1000W Lamp F-1587 interpolated on 1nm steps")
        plt.grid()
        plt.tight_layout()
        if save_fig:
            plt.savefig(f"{plot_path}/1000W_Lamp_F1587_1nm_19.png", dpi=100)
        plt.show()
        plt.close()
    if save_file:
        # save lamp file in calib folder
        lamp.to_csv(f"{calib_path}/1000W_lamp_F1587_1nm_19.dat", index=False)

    return lamp


def read_smart_raw(path: str, filename: str) -> pd.DataFrame:
    """
    Read raw SMART data files

    Args:
        path: Path where to find file
        filename: Name of file

    Returns: pandas DataFrame with column names and datetime index

    """
    file = os.path.join(path, filename)
    date_str, channel, _ = smart.get_info_from_filename(filename)

    if channel == "SWIR":
        pixels = list(range(1, 257))  # 256 pixels
    elif channel == "VNIR":
        pixels = list(range(1, 1025))  # 1024 pixels
    else:
        raise ValueError("channel has to be 'SWIR' or 'VNIR'!")

    # first three columns: Time (hh mm ss.ss), integration time (ms), shutter flag
    header = ["time", "t_int", "shutter"]
    header.extend(pixels)

    df = pd.read_csv(file, sep="\t", header=None, names=header)
    datetime_str = date_str + " " + df["time"]
    df = df.set_index(pd.to_datetime(datetime_str, format="%Y_%m_%d %H %M %S.%f")).drop("time", axis=1)

    return df


def read_smart_cor(path: str, filename: str) -> pd.DataFrame:
    """
        Read dark current corrected SMART data files

        Args:
            path: Path where to find file
            filename: Name of file

        Returns: pandas DataFrame with column names and datetime index

        """
    file = os.path.join(path, filename)
    df = pd.read_csv(file, sep="\t", index_col="time", parse_dates=True)
    df.columns = pd.to_numeric(df.columns)  # make columns numeric

    return df


def read_pixel_to_wavelength(path: str, channel: str) -> pd.DataFrame:
    """
    Read file which maps each pixel to a certain wavelength for a specified channel.

    Args:
        path: Path where to find file
        channel: For which channel the pixel to wavelength mapping should be read in, refer to lookup table for possible
         channels (e.g. ASP06_J3)

    Returns: pandas DataFrame relating pixel number to wavelength

    """
    filename = f"pixel_wl_{cirrus_hl.lookup[channel]}.dat"
    file = os.path.join(path, filename)
    df = pd.read_csv(file, sep="\s+", skiprows=7, header=None, names=["pixel", "wavelength"])
    # sort df by the wavelength column and reset the index, necessary for the SWIR spectrometers
    # df = df.sort_values(by="wavelength").reset_index(drop=True)

    return df


def read_nav_data(nav_path: str) -> pd.DataFrame:
    """
    Reader function for Navigation data file from the INS used by the stabilization of SMART

    Args:
        nav_path: path to file including filename

    Returns: pandas DataFrame with headers and a DateTimeIndex

    """
    # read out the start time information given in the file
    with open(nav_path) as f:
        time_info = f.readlines()[1]
    start_time = pd.to_datetime(time_info[11:31], format="%m/%d/%Y %H:%M:%S")
    # define the start date of the measurement
    start_date = pd.Timestamp(year=start_time.year, month=start_time.month, day=start_time.day)
    header = ["marker", "seconds", "roll", "pitch", "yaw", "AccS_X", "AccS_Y", "AccS_Z", "OmgS_X", "OmgS_Y", "OmgS_Z"]
    nav = pd.read_csv(nav_path, sep="\s+", skiprows=13, header=None, names=header)
    nav["time"] = pd.to_datetime(nav["seconds"], origin=start_date, unit="s")
    nav = nav.set_index("time")

    return nav


def read_libradtran(flight: str, filename: str) -> pd.DataFrame:
    """
    Read an old libRadtran simulation file generated by 01_dirdiff_BBR_Cirrus_HL_Server_jr.pro and add a DateTime Index.

    Args:
        flight: which flight does the simulation belong to (e.g. Flight_20210629a)
        filename: filename

    Returns: DataFrame with libRadtran output data

    """
    file_path = f"{h.get_path('libradtran', flight)}/{filename}"
    bbr_sim = pd.read_csv(file_path, sep="\s+", skiprows=34)
    date_dt = datetime.datetime.strptime(flight[7:15], "%Y%m%d")
    date_ts = pd.Timestamp(year=date_dt.year, month=date_dt.month, day=date_dt.day)
    bbr_sim["time"] = pd.to_datetime(bbr_sim["sod"], origin=date_ts, unit="s")  # add a datetime column
    bbr_sim = bbr_sim.set_index("time")  # set it as index

    return bbr_sim
