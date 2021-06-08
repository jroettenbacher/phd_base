#!/usr/bin/env python
"""Script for processing and plotting of SMART data
author: Johannes Röttenbacher
"""

import os
import re
import logging
import sys
import pandas as pd
from typing import Tuple, Union
import numpy as np
import toml
import functions_jr as jr
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

# inlet to spectrometer mapping and inlet to direction mapping and measurement to spectrometer mapping
lookup = dict(ASP_06_J3="PGS_5_(ASP_06)", ASP_06_J4="VIS_6_(ASP_06)", ASP_06_J5="PGS_6_(ASP_06)",
              ASP_06_J6="VIS_7_(ASP_06)", ASP_07_J3="PGS_4_(ASP_07)", ASP_07_J4="VIS_8_(ASP_07)",
              J3="dw", J4="dw", J5="up", J6="up",
              Fdw_SWIR="ASP_06_J3", Fdw_VNIR="ASP_06_J4", Fup_SWIR="ASP_06_J5", Fup_VNIR="ASP_06_J6",
              Iup_SWIR="ASP_07_J3", Iup_VNIR="ASP_07_J4")


def get_info_from_filename(filename: str) -> Tuple[str, str, str]:
    """
    Using regular expressions some information from the filename is extracted.

    Args:
        filename: string in the form yyyy_mm_dd_hh_MM.[F/I][up/dw]_[SWIR/VNIR].dat (eg. "2021_03_29_11_07.Fup_SWIR.dat")

    Returns: A date string, the channel and the direction of measurement including the quantity.

    """
    # find date string and channel from file
    match = re.search(r"^(?P<date>\d{4}_\d{2}_\d{2}).*.(?P<direction>[FI][a-z]{2})_(?P<channel>[A-Z]{4})", filename)
    try:
        date_str = match.group('date')
        channel = match.group('channel')
        direction = match.group('direction')
    except AttributeError:
        log.info("No date, channel or direction information was found! Check filename!")
        raise

    return date_str, channel, direction


def read_smart_raw(path: str, filename: str) -> pd.DataFrame:
    """
    Read raw SMART data files

    Args:
        path: Path where to find file
        filename: Name of file

    Returns: pandas DataFrame with column names and datetime index

    """
    file = os.path.join(path, filename)
    date_str, channel, _ = get_info_from_filename(filename)

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


def find_pixel(df: pd.DataFrame, wavelength: float()) -> Tuple[int, float]:
    """
    Given the dataframe with the pixel to wavelength mapping, return the pixel and wavelength closest to the requested
    wavelength.

    Args:
        df: Dataframe with column pixel and wavelength (from read_pixel_to_wavelength)
        wavelength: which wavelength are you interested in

    Returns: closest pixel number and wavelength corresponding to the given wavelength

    """
    assert df["wavelength"].iloc[-1] >= wavelength >= df["wavelength"].iloc[0], "Given wavelength not in data frame!"
    idx = jr.argnearest(df["wavelength"], wavelength)
    pixel_nr, wavelength = df["pixel"].iloc[idx], df["wavelength"].iloc[idx]
    return pixel_nr, wavelength


def set_paths():
    """
    Read paths from the toml file according to the current working directory.

    Returns: Paths to measurements, pixel to wavelength calibration files, spectrometer calibration files,
    processed files and plots.

    """
    if os.getcwd().startswith("C"):
        config = toml.load("config.toml")["cirrus-hl"]["jr_local"]
    else:
        config = toml.load("config.toml")["cirrus-hl"]["lim_server"]

    base_dir = config["base_dir"]
    raw_path = os.path.join(base_dir, config["raw_data"])
    pixel_wl_path = os.path.join(base_dir, config["pixel_to_wavelength"])
    calib_path = os.path.join(base_dir, config["calib_data"])
    data_path = os.path.join(base_dir, config["data"])
    plot_path = os.path.join(base_dir, config["plots"])

    return raw_path, pixel_wl_path, calib_path, data_path, plot_path


def get_path(key: str) -> str:
    """
        Read paths from the toml file according to the current working directory.

        Args:
            key: which path to return, see function for possible values

        Returns: Path to specified data

    """
    if os.getcwd().startswith("C"):
        config = toml.load("config.toml")["cirrus-hl"]["jr_local"]
    else:
        config = toml.load("config.toml")["cirrus-hl"]["lim_server"]

    paths = dict()
    base_dir = config["base_dir"]
    paths["base"] = base_dir
    paths["raw"] = os.path.join(base_dir, config["raw_data"])
    paths["pixel_wl"] = os.path.join(base_dir, config["pixel_to_wavelength"])
    paths["calib"] = os.path.join(base_dir, config["calib_data"])
    paths["data"] = os.path.join(base_dir, config["data"])
    paths["plot"] = os.path.join(base_dir, config["plots"])
    paths["lamp"] = os.path.join(base_dir, config["lamp"])

    return paths[key]


def _plot_dark_current(wavelenghts: Union[pd.Series, list],
                      dark_current: Union[pd.Series, list],
                      spectrometer: str, channel: str, **kwargs):
    """
    Plot the dark current over the wavelengths from the specified spectrometer and channel.

    Args:
        wavelenghts: series or list of wavelengths corresponding with the pixel numbers from read_pixel_to_wavelength
        dark_current: series or list with mean dark current for each pixel
        spectrometer: name of the spectrometer inlet
        channel: VNIR or SWIR
        **kwargs:
            save_fig: whether to save the figure in the current directory (True) or just show it (False, default)

    Returns:

    """
    save_fig = kwargs["save_fig"] if "save_fig" in kwargs else False
    plt.plot(wavelenghts, dark_current, color='k')
    plt.axhline(dark_current.mean(), color="orange", label="Mean")
    plt.grid()
    plt.title(f"Dark Current from Spectrometer {spectrometer}, channel: {channel}")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Netto Counts")
    plt.legend()
    if save_fig:
        plt.savefig(f"dark_current_{spectrometer}_{channel}.png")
    else:
        plt.show()
    plt.close()


def get_dark_current(filename: str, option: int, **kwargs) -> Union[pd.Series, plt.figure]:
    """
    Get the corresponding dark current for the specified measurement file to correct the raw SMART measurement.

    Args:
        filename: filename (e.g. "2021_03_29_11_07.Fup_SWIR.dat")
        option: which option to use for VNIR, 1 or 2
        kwargs:
            plot (bool): show plot or not (default: True)
            path (str): path to file if different from raw file path given in config.toml

    Returns: pandas Series with the mean dark current measurements over time for each pixel and optionally a plot of it

    """
    plot = kwargs["plot"] if "plot" in kwargs else True
    path, pixel_wl_path, calib_path, _, _ = set_paths()
    path = kwargs["path"] if "path" in kwargs else path
    smart = read_smart_raw(path, filename)
    date_str, channel, direction = get_info_from_filename(filename)
    spectrometer = lookup[f"{direction}_{channel}"]
    pixel_wl = read_pixel_to_wavelength(pixel_wl_path, spectrometer)
    # calculate dark current depending on channel:
    # SWIR: use measurements during shutter phases
    # VNIR: Option 1: use measurements below 290 nm
    #       Option 2: use dark measurements from calibration
    if channel == "VNIR":
        if option == 1:
            last_dark_pixel, _ = find_pixel(pixel_wl, 290)
            dark_pixels = np.arange(1, last_dark_pixel + 1)
            dark_current = smart.loc[:, dark_pixels].mean()
            dark_wls = pixel_wl[pixel_wl["pixel"].isin(dark_pixels)]["wavelength"]
            if plot:
                _plot_dark_current(dark_wls, dark_current, spectrometer, channel)
        else:
            assert option == 2, "Option should be either 1 or 2!"
            # read in cali file
            # get path depending on spectrometer and inlet
            instrument = re.search(r'ASP_\d{2}', spectrometer)[0]
            inlet = re.search(r'J\d{1}', spectrometer)[0]
            # find right folder and right cali
            for dirpath, dirs, files in os.walk(calib_path):
                if re.search(f".*{instrument}.*dark.*", dirpath) is not None:
                    d_path, d = os.path.split(dirpath)
                    # check if the date of the calibration matches the date of the file
                    date_check = True if date_str.replace("_", "") in d_path else False
                    # ASP_06 has 2 SWIR and 2 VNIR inlets thus search for the folder for the given inlet
                    if instrument == "ASP_06" and inlet[1] in d:
                        run = True
                    # ASP_07 has only one SWIR and VNIR inlet -> no need to search
                    else:
                        run = True
                    if run and date_check:
                        i = 0
                        for file in files:
                            if re.search(f'.*.{direction}_{channel}.dat', file) is not None:
                                dark_dir, dark_file = dirpath, file
                                log.info(f"Calibration file used:\n{os.path.join(dark_dir, dark_file)}")
                                assert i == 0, f"More than one possible file was found!\n Check {dirpath}!"
                                i += 1


            dark_current = read_smart_raw(dark_dir, dark_file)
            dark_current = dark_current.iloc[:, 2:].mean()
            wls = pixel_wl["wavelength"]
            if plot:
                _plot_dark_current(wls, dark_current, spectrometer, channel)

    elif channel == "SWIR":
        # check if the shutter flag was working: If all values are 1 -> shutter flag is probably not working
        if np.sum(smart.shutter == 1) != smart.shutter.shape[0]:
            dark_current = smart.where(smart.shutter == 0).mean().iloc[2:]
            wls = pixel_wl["wavelength"]
            if plot:
                _plot_dark_current(wls, dark_current, spectrometer, channel)
        else:
            log.debug("Shutter flag is probably wrong")
    else:
        log.warning(f"channel should be either 'VNIR' or 'SWIR' but is {channel}!")
        sys.exit(1)

    return dark_current


def plot_mean_corrected_measurement(filename: str, measurement: Union[pd.Series, list],
                                    measurement_cor: Union[pd.Series, list],
                                    option: int, **kwargs):
    """
    Plot the mean dark current corrected SMART measurement over time together with the raw measurement and the dark
    current.

    Args:
        filename: name of file
        measurement: raw SMART measurements for each pixel averaged over time
        measurement_cor: corrected SMART measurements for each pixel averaged over time
        option: which option was used for VNIR correction
        **kwargs:
            save_fig (bool): save figure to current directory or just show it

    Returns: plot

    """
    save_fig = kwargs["save_fig"] if "save_fig" in kwargs else False
    _, pixel_path, _, _, _ = set_paths()
    date_str, channel, direction = get_info_from_filename(filename)
    spectrometer = lookup[f"{direction}_{channel}"]
    dark_current = get_dark_current(filename, option, plot=False)
    wavelength = read_pixel_to_wavelength(pixel_path, spectrometer)["wavelength"]

    if channel == "VNIR" and option == 1:
        plt.axhline(dark_current, label="Dark Current", color='k')
    else:
        plt.plot(wavelength, dark_current, label="Dark Current", color='k')
    plt.plot(wavelength, measurement, label="Measurement")
    plt.plot(wavelength, measurement_cor, label="Corrected Measurement")
    plt.grid()
    plt.title(f"Mean Corrected SMART Measurement {date_str.replace('_', '-')}\n"
              f"Spectrometer {spectrometer}, Channel: {channel}, Option: {option}")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Netto Counts")
    plt.legend()
    if save_fig:
        plt.savefig(f"{date_str}_corrected_smart_measurement.png")
    else:
        plt.show()
    plt.close()


def correct_smart_dark_current(smart_file: str, option: int, **kwargs) -> pd.Series:
    """
    Correct the raw SMART measurement for the dark current of the spectrometer.
    Only returns data when the shutter was open.

    Args:
        smart_file: filename of file to correct
        option: which option should be used to get the dark current? Only relevant for channel "VNIR".
        kwargs:
            path: path to file if not raw file path as given in config.toml

    Returns: Series with corrected smart measurement

    """
    path, _, _, _, _ = set_paths()
    path = kwargs["path"] if "path" in kwargs else path
    date_str, channel, direction = get_info_from_filename(smart_file)
    smart = read_smart_raw(path, smart_file)
    dark_current = get_dark_current(smart_file, option, plot=False, path=path)

    if channel == "VNIR" and option == 1:
        dark_current = dark_current.mean()
    measurement = smart.where(smart.shutter == 1).iloc[:, 2:]  # only use data when shutter is open
    measurement_cor = measurement - dark_current

    return measurement_cor


def plot_smart_data(filename: str, wavelength: Union[list, str], **kwargs) -> None:
    """
    Plot SMART data in the given file. Either a time average over a range of wavelengths or all wavelengths,
    or a time series of one wavelength.
    TODO: add option to plot multiple files

    Args:
        filename: Standard SMART filename
        wavelength: list with either one or two wavelengths in nm or 'all'
        **kwargs:
            path (str): give path to filename if not default from config.toml
            save_fig: save figure? (default: False)
            plot_path: where to save figure (default: given in config.toml)

    Returns: Shows a figure or saves it to disc.

    """
    raw_path, pixel_wl_path, _, data_path, plot_path = set_paths()
    # read in keyword arguments
    raw_path = kwargs["path"] if "path" in kwargs else raw_path
    save_fig = kwargs["save_fig"] if "save_fig" in kwargs else False
    plot_path = kwargs["plot_path"] if "plot_path" in kwargs else plot_path
    date_str, channel, direction = get_info_from_filename(filename)
    if "cor" in filename:
        smart = read_smart_cor(data_path, filename)
        title = "Corrected for Dark Current"
    else:
        smart = read_smart_raw(raw_path, filename)
        smart = smart.iloc[:, 2:]  # remove columns t_int and shutter
        title = "Raw"
    pixel_wl = read_pixel_to_wavelength(pixel_wl_path, lookup[f"{direction}_{channel}"])
    if len(wavelength) == 2:
        pixel_nr = []
        wl_str = ""
        for wl in wavelength:
            pxl, wavel = find_pixel(pixel_wl, wl)
            pixel_nr.append(pxl)
            wl_str = f"{wl_str}_{wavel:.1f}"
        pixel_nr.sort()  # make sure wavelengths are in ascending order
        smart_sel = smart.loc[:, pixel_nr[0]:pixel_nr[1]]  # select range of wavelengths
        begin_dt, end_dt = smart_sel.index[0], smart_sel.index[-1]  # read out start and end time
        smart_mean = smart_sel.mean(axis=0).to_frame()  # calculate mean over time and return a dataframe
        smart_mean = smart_mean.set_index(pd.to_numeric(smart_mean.index))  # update the index to be numeric
        # join the measurement and pixel to wavelength data frames by pixel
        smart_plot = smart_mean.join(pixel_wl.set_index(pixel_wl["pixel"]))
        smart_plot.plot(x="wavelength", y=0, legend=False, xlabel="Wavelength (nm)", ylabel="Netto Counts",
                        title=f"Time Averaged SMART Measurement {title} {direction} {channel}\n {begin_dt} - {end_dt}")
        plt.grid()
        figname = filename.replace('.dat', f'{wl_str}.png')
    elif len(wavelength) == 1:
        pixel_nr, wl = find_pixel(pixel_wl, wavelength[0])
        smart_sel = smart.loc[:, pixel_nr].to_frame()
        begin_dt, end_dt = smart_sel.index[0], smart_sel.index[-1]
        time_extend = end_dt - begin_dt
        fig, ax = plt.subplots()
        smart_sel.plot(ax=ax, legend=False, xlabel="Time (UTC)", ylabel="Netto Counts",
                       title=f"SMART Time Series {title}\n{wl:.3f} nm {begin_dt:%Y-%m-%d}")
        ax = jr.set_xticks_and_xlabels(ax, time_extend)
        ax.grid()
        figname = filename.replace('.dat', f'_{wl:.1f}nm.png')
    elif wavelength == "all":
        begin_dt, end_dt = smart.index[0], smart.index[-1]
        smart_mean = smart.mean().to_frame()
        smart_mean = smart_mean.set_index(pd.to_numeric(smart_mean.index))
        smart_plot = smart_mean.join(pixel_wl.set_index(pixel_wl["pixel"]))
        smart_plot.plot(x="wavelength", y=0, legend=False, xlabel="Wavelength (nm)", ylabel="Netto Counts",
                        title=f"Time Averaged SMART Measurement {title} {direction} {channel}\n "
                              f"{begin_dt:%Y-%m-%d %H:%M:%S} - {end_dt:%Y-%m-%d %H:%M:%S}")
        plt.grid()
        figname = filename.replace('.dat', f'_{wavelength}.png')
    else:
        raise ValueError("wavelength has to be a list of length 1 or 2 or 'all'!")

    if save_fig:
        plt.savefig(f"{plot_path}/{figname}")
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    # test read in functions
    path = "C:/Users/Johannes/Documents/Doktor/campaigns/CIRRUS-HL/SMART/ASP_06_Calib_Lab_20210329/calib_J3_4"
    filename = "2021_03_29_11_15.Fup_VNIR.dat"
    # filename = "2021_03_29_11_15.Fup_SWIR.dat"
    smart = read_smart_raw(path, filename)

    pixel_wl_path = "C:/Users/Johannes/Documents/Doktor/instruments/SMART/pixel_wl"
    spectrometer = "ASP_06_J4"
    pixel_wl = read_pixel_to_wavelength(pixel_wl_path, spectrometer)

    # find pixel closest to given wavelength
    pixel_nr, wavelength = find_pixel(pixel_wl, 525)

    # input: spectrometer, filename, option
    option = 2
    filename = "2021_03_29_11_07.Fup_VNIR.dat"
    filename = "2021_03_29_11_07.Fup_SWIR.dat"
    dark_current = get_dark_current(filename, option)

    # correct raw measurement with dark current
    # input: smart measurement, option
    option = 2
    filename = "2021_03_29_11_07.Fup_VNIR.dat"
    smart_cor = correct_smart_dark_current(filename, option)

    # plot mean corrected smart measurement
    raw_path, _, _, _, _ = set_paths()
    filename = "2021_03_29_11_07.Fup_VNIR.dat"
    option = 2
    smart = read_smart_raw(raw_path, filename)
    smart_cor = correct_smart_dark_current(filename, option=option)
    measurement = smart.mean().iloc[2:]
    measurement_cor = smart_cor.mean()
    plot_mean_corrected_measurement(filename, measurement, measurement_cor, option)

    # read corrected file
    _, _, _, data_path, _ = set_paths()
    filename = "2021_03_29_11_07.Fup_VNIR_cor.dat"
    smart_cor = read_smart_cor(data_path, filename)

    # plot any smart measurement given a range of wavelengths, one specific one or all
    filename = "2021_03_29_11_07.Fup_VNIR_cor.dat"
    raw_file = "2021_03_29_11_07.Fup_VNIR.dat"
    wavelength = [525]
    plot_smart_data(filename, wavelength)
    plot_smart_data(raw_file, wavelength)

    # working section
    raw_file = "2021_06_04_13_07.Fup_VNIR.dat"
    path = "C:/Users/Johannes/Documents/Doktor/campaigns/CIRRUS-HL/SMART/calib/20210604_transfer_cali_ASP06/dark"
    plot_smart_data(raw_file, "all", path=path)


