#!/usr/bin/env python
"""Script for processing and plotting of SMART data
author: Johannes Röttenbacher
"""

import os
import re
import logging
import sys
import pandas as pd
import xarray as xr
from typing import Tuple, Union
import numpy as np
import toml
import functions_jr as jr
import matplotlib.pyplot as plt
import holoviews as hv
from holoviews import opts
from cirrus_hl import lookup

hv.extension('bokeh')

log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler())


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
    calib_path = get_path("calib")
    plot_path = get_path("plot")
    lamp_path = get_path("lamp")  # get path to lamp defined in config.toml
    # read in lamp file
    lamp_file = "F1587i01_19.std"
    names = ["Irradiance"]  # column name
    lamp = pd.read_csv(os.path.join(lamp_path, lamp_file), skiprows=1, header=None, names=names)
    lamp["Wavelength"] = np.arange(250, 2501)
    # convert from W/cm^2 to W/m^2; cm = m * 10^-2 => cm^2 = (m * 10^-2)^2 = m^2 * 10^-4 => W*10^4/m^2
    lamp["Irradiance"] = lamp["Irradiance"] * 10000
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
    # sort df by the wavelength column and reset the index, necessary for the SWIR spectrometers
    # df = df.sort_values(by="wavelength").reset_index(drop=True)
    return df


def read_nav_data(nav_path: str) -> pd.DataFrame:
    """
    Reader function for Navigation data file from the INS
    Args:
        nav_path: path to file including filename

    Returns: pandas DataFrame with headers and a DateTimeIndex

    """
    # read out the time start time information given in the file
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


def read_bahamas(bahamas_path: str) -> xr.Dataset:
    """
    Reader function for netcdf BAHAMAS data
    Args:
        bahamas_path: full path to netcdf file

    Returns: xr.DataSet with BAHAMAS data and Time as dimension

    """
    ds = xr.open_dataset(bahamas_path)
    ds = ds.swap_dims({"tid": "TIME"})
    return ds


def find_pixel(df: pd.DataFrame, wavelength: float()) -> Tuple[int, float]:
    """
    Given the dataframe with the pixel to wavelength mapping, return the pixel and wavelength closest to the requested
    wavelength.

    Args:
        df: Dataframe with column pixel and wavelength (from read_pixel_to_wavelength)
        wavelength: which wavelength are you interested in

    Returns: closest pixel number and wavelength corresponding to the given wavelength

    """
    # find smallest and greatest wavelength in data frame and check if the given wavelength is in it
    min_wl = df["wavelength"].min()
    max_wl = df["wavelength"].max()
    assert max_wl >= wavelength >= min_wl, "Given wavelength not in data frame!"
    idx = jr.arg_nearest(df["wavelength"], wavelength)
    pixel_nr, wavelength = df["pixel"].iloc[idx], df["wavelength"].iloc[idx]
    return pixel_nr, wavelength


def get_path(key: str, flight: str = None, instrument: str = None) -> str:
    """
        Read paths from the toml file according to the current working directory.

        Args:
            key: which path to return, see function for possible values
            flight: for which flight should the path be provided (eg. Flight_20210625a)
            instrument: if key=all which instrument to generate the path to? (e.g. BAHAMAS)

        Returns: Path to specified data

    """
    wk_dir = os.getcwd()
    if wk_dir.startswith("C"):
        config = toml.load("config.toml")["cirrus-hl"]["jr_local"]
    elif wk_dir.startswith("/mnt"):
        config = toml.load("config.toml")["cirrus-hl"]["jr_ubuntu"]
    else:
        config = toml.load("config.toml")["cirrus-hl"]["lim_server"]

    flight = "" if flight is None else flight
    paths = dict()
    base_dir = config["base_dir"]
    paths["base"] = base_dir
    config.pop("base_dir")
    for k in config:
        paths[k] = os.path.join(base_dir, flight, config[k])
    for k in ["calib", "pixel_wl", "lamp", "panel"]:
        paths[k] = config[k]
    if key == 'all':
        paths['all'] = os.path.join(base_dir, config[key], instrument)

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


def get_dark_current(flight: str, filename: str, option: int, **kwargs) -> Union[pd.Series, plt.figure]:
    """
    Get the corresponding dark current for the specified measurement file to correct the raw SMART measurement.

    Args:
        flight: to which flight does the file belong to? (e.g. Flight_20210707a)
        filename: filename (e.g. "2021_03_29_11_07.Fup_SWIR.dat")
        option: which option to use for VNIR, 1 or 2
        kwargs:
            path (str): path if not standard path from config.toml
            plot (bool): show plot or not (default: True)
            date (str): yyyymmdd, date of transfer calibration with dark current measurement to use

    Returns: pandas Series with the mean dark current measurements over time for each pixel and optionally a plot of it

    """
    plot = kwargs["plot"] if "plot" in kwargs else True
    date = kwargs["date"] if "date" in kwargs else None
    path = get_path("raw", flight)
    path = kwargs["path"] if "path" in kwargs else path
    pixel_wl_path = get_path("pixel_wl")
    calib_path = get_path("calib")
    smart = read_smart_raw(path, filename)
    t_int = int(smart["t_int"][0])  # get integration time
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
            instrument = re.search(r'ASP\d{2}', spectrometer)[0]
            inlet = re.search(r'J\d{1}', spectrometer)[0]
            # find right folder and right cali
            for dirpath, dirs, files in os.walk(calib_path):
                if re.search(f".*{instrument}.*dark.*", dirpath) is not None:
                    d_path, d = os.path.split(dirpath)
                    # check if the date of the calibration matches the date of the file
                    date_check = True if date_str.replace("_", "") in d_path else False
                    # overwrite date check if date is given
                    if date is not None:
                        date_check = True if date in d_path else False
                    # check for the right integration time in folder name
                    t_int_check = str(t_int) in d
                    # ASP06 has 2 SWIR and 2 VNIR inlets thus search for the folder for the given inlet
                    if instrument == "ASP06" and inlet[1] in d:
                        run = True
                    # ASP07 has only one SWIR and VNIR inlet -> no need to search
                    else:
                        run = True
                    if run and date_check and t_int_check:
                        i = 0
                        for file in files:
                            if re.search(f'.*.{direction}_{channel}.dat', file) is not None:
                                dark_dir, dark_file = dirpath, file
                                log.info(f"Calibration file used:\n{os.path.join(dark_dir, dark_file)}")
                                assert i == 0, f"More than one possible file was found!\n Check {dirpath}!"
                                i += 1

            try:
                dark_current = read_smart_raw(dark_dir, dark_file)
                dark_current = dark_current.iloc[:, 2:].mean()
                wls = pixel_wl["wavelength"]
                if plot:
                    _plot_dark_current(wls, dark_current, spectrometer, channel)
            except UnboundLocalError as e:
                raise RuntimeError("No dark current file found for measurement!") from e

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


def plot_mean_corrected_measurement(flight: str, filename: str, measurement: Union[pd.Series, list],
                                    measurement_cor: Union[pd.Series, list],
                                    option: int, **kwargs):
    """
    Plot the mean dark current corrected SMART measurement over time together with the raw measurement and the dark
    current.

    Args:
        flight: to which flight does the file belong to? (e.g. Flight_20210707a)
        filename: name of file
        measurement: raw SMART measurements for each pixel averaged over time
        measurement_cor: corrected SMART measurements for each pixel averaged over time
        option: which option was used for VNIR correction
        **kwargs:
            save_fig (bool): save figure to current directory or just show it

    Returns: plot

    """
    save_fig = kwargs["save_fig"] if "save_fig" in kwargs else False
    pixel_path = get_path("pixel_wl")
    date_str, channel, direction = get_info_from_filename(filename)
    path = get_path("raw", flight)
    spectrometer = lookup[f"{direction}_{channel}"]
    dark_current = get_dark_current(path, filename, option, plot=False)
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


def correct_smart_dark_current(flight: str, smart_file: str, option: int, **kwargs) -> pd.Series:
    """
    Correct the raw SMART measurement for the dark current of the spectrometer.
    Only returns data when the shutter was open.

    Args:
        flight: to which flight does the file belong to? (e.g. Flight_20210707a)
        smart_file: filename of file to correct
        option: which option should be used to get the dark current? Only relevant for channel "VNIR".
        kwargs:
            path: path to file if not raw file path as given in config.toml
            date: date from which the dark current measurement should be used for VNIR (necessary if no transfer
                  calibration was made on a measurement day)

    Returns: Series with corrected smart measurement

    """
    # TODO: do not write empty rows (were shutter is closed)
    path = get_path("raw", flight)
    path = kwargs["path"] if "path" in kwargs else path
    date = kwargs["date"] if "date" in kwargs else None
    date_str, channel, direction = get_info_from_filename(smart_file)
    smart = read_smart_raw(path, smart_file)
    dark_current = get_dark_current(flight, smart_file, option, plot=False, path=path, date=date)

    if channel == "VNIR" and option == 1:
        dark_current = dark_current.mean()
    measurement = smart.where(smart.shutter == 1).iloc[:, 2:]  # only use data when shutter is open
    measurement_cor = measurement - dark_current

    return measurement_cor


def plot_smart_data(flight: str, filename: str, wavelength: Union[list, str], **kwargs) -> plt.axes:
    """
    Plot SMART data in the given file. Either a time average over a range of wavelengths or all wavelengths,
    or a time series of one wavelength. Return an axes object to continue plotting or show it.
    TODO: add option to plot multiple files

    Args:
        flight: to which flight does the file belong to? (e.g. Flight_20210707a)
        filename: Standard SMART filename
        wavelength: list with either one or two wavelengths in nm or 'all'
        **kwargs:
            path (str): give path to filename if not default from config.toml
            save_fig: save figure? (default: False)
            plot_path: where to save figure (default: given in config.toml)

    Returns: Shows a figure or saves it to disc.

    """
    raw_path = get_path("raw", flight)
    pixel_wl_path = get_path("pixel_wl")
    data_path = get_path("data", flight)
    calibrated_path = get_path("calibrated", flight)
    plot_path = get_path("plot")
    # read in keyword arguments
    raw_path = kwargs["path"] if "path" in kwargs else raw_path
    data_path = kwargs["path"] if "path" in kwargs else data_path
    save_fig = kwargs["save_fig"] if "save_fig" in kwargs else False
    plot_path = kwargs["plot_path"] if "plot_path" in kwargs else plot_path
    ax = kwargs["ax"] if "ax" in kwargs else None
    date_str, channel, direction = get_info_from_filename(filename)
    if "calibrated" in filename:
        smart = read_smart_cor(calibrated_path, filename)
        title = "\nCorrected for Dark Current and Calibrated"
        ylabel = "Irradiance (W$\\,$m$^{-2}\\,$nm$^{-1}$)" if "F" in filename else "Radiance (W$\\,$sr$^{-1}\\,$m$^{-2}\\,$nm$^{-1}$)"
    elif "cor" in filename:
        smart = read_smart_cor(data_path, filename)
        title = "Corrected for Dark Current"
        ylabel = "Netto Counts"
    else:
        smart = read_smart_raw(raw_path, filename)
        smart = smart.iloc[:, 2:]  # remove columns t_int and shutter
        title = "Raw"
        ylabel = "Netto Counts"
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
        smart_plot.plot(x="wavelength", y=0, legend=False, xlabel="Wavelength (nm)", ylabel=ylabel,
                        title=f"Time Averaged SMART Measurement {title} {direction} {channel}\n {begin_dt} - {end_dt}")
        figname = filename.replace('.dat', f'{wl_str}.png')
    elif len(wavelength) == 1:
        pixel_nr, wl = find_pixel(pixel_wl, wavelength[0])
        smart_sel = smart.loc[:, pixel_nr].to_frame()
        begin_dt, end_dt = smart_sel.index[0], smart_sel.index[-1]
        time_extend = end_dt - begin_dt
        fig = plt.figure()
        if ax is None:
            ax = fig.add_subplot(111)
        smart_sel.plot(ax=ax, legend=False, xlabel="Time (UTC)", ylabel=ylabel,
                       title=f"SMART Time Series {title} {direction} {channel}\n{wl:.3f} nm {begin_dt:%Y-%m-%d}")
        jr.set_xticks_and_xlabels(ax, time_extend)
        figname = filename.replace('.dat', f'_{wl:.1f}nm.png')
    elif wavelength == "all":
        begin_dt, end_dt = smart.index[0], smart.index[-1]
        smart_mean = smart.mean().to_frame()
        smart_mean = smart_mean.set_index(pd.to_numeric(smart_mean.index))
        smart_plot = smart_mean.join(pixel_wl.set_index(pixel_wl["pixel"]))
        smart_plot.plot(x="wavelength", y=0, legend=False, xlabel="Wavelength (nm)", ylabel=ylabel,
                        title=f"Time Averaged SMART Measurement {title} {direction} {channel}\n "
                              f"{begin_dt:%Y-%m-%d %H:%M:%S} - {end_dt:%Y-%m-%d %H:%M:%S}")
        figname = filename.replace('.dat', f'_{wavelength}.png')
    else:
        raise ValueError("wavelength has to be a list of length 1 or 2 or 'all'!")

    plt.grid()
    plt.tight_layout()
    if save_fig:
        plt.savefig(f"{plot_path}/{figname}", dpi=100)
        log.info(f"Saved {plot_path}/{figname}")
        plt.close()
    else:
        return ax


def plot_smart_spectra(path: str, filename: str, index: int, **kwargs) -> None:
    """
    Plot a spectra from a SMART calibrated measurement file for a given index (time step)
    Args:
        path: where the file can be found
        filename: name of the file (standard SMART filename convention)
        index: which row to plot
        **kwargs:
            save_fig: Save figure to plot path given in config.toml
            plot_path: Where to save plot if not standard plot path

    Returns: Shows and or saves a plot

    """
    save_fig = kwargs["save_fig"] if "save_fig" in kwargs else False
    plot_path = kwargs["plot_path"] if "plot_path" in kwargs else get_path("plot")
    pixel_path = get_path("pixel_wl")
    df = read_smart_cor(path, filename)
    date_str, channel, direction = get_info_from_filename(filename)
    spectrometer = lookup[f"{direction}_{channel}"]
    pixel_wl = read_pixel_to_wavelength(pixel_path, spectrometer)
    max_id = len(df) - 1
    try:
        df_sel = df.iloc[index, :]
    except IndexError as e:
        log.info(f"{e}\nGiven index '{index}' out-of-bounds! Using maximum index '{max_id}'!")
        df_sel = df.iloc[max_id, :]

    time_stamp = df_sel.name  # get time stamp which is selected
    pixel_wl[f"{direction}"] = df_sel.reset_index(drop=True)
    ylabel = "Irradiance (W$\\,$m$^{-2}\\,$nm$^{-1}$)" if "F" in filename else "Radiance (W$\\,$sr$^{-1}\\,$m$^{-2}\\,$nm$^{-1}$)"

    fig, ax = plt.subplots()
    ax.plot("wavelength", f"{direction}", data=pixel_wl, label=f"{direction}")
    ax.set_title(f"SMART Spectra {spectrometer} {direction} {channel} \n {time_stamp:%Y-%m-%d %H:%M:%S}")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(ylabel)
    ax.grid()
    plt.show()
    if save_fig:
        figname = filename.replace(".dat", f"_spectra_{time_stamp:%Y%m%d_%H%M%S}.png")
        figpath = f"{plot_path}/{figname}"
        plt.savefig(figpath, dpi=100)
        log.info(f"Saved {figpath}")

    plt.close()


def plot_complete_smart_spectra(path: str, filename: str, index: int, **kwargs) -> None:
    """
    Plot the complete spectra given by both channels from SMART calibrated measurement files for a given index (time step)
    Args:
        path: where the file can be found
        filename: name of the file (standard SMART filename convention)
        index: which row to plot
        **kwargs:
            save_fig: Save figure to plot path given in config.toml
            plot_path: Where to save plot if not standard plot path

    Returns: Shows and or saves a plot

    """
    save_fig = kwargs["save_fig"] if "save_fig" in kwargs else False
    plot_path = kwargs["plot_path"] if "plot_path" in kwargs else get_path("plot")
    pixel_path = get_path("pixel_wl")
    df1 = read_smart_cor(path, filename)
    date_str, channel, direction = get_info_from_filename(filename)
    if channel == "SWIR":
        channel2 = "VNIR"
    else:
        channel2 = "SWIR"

    filename2 = filename.replace(channel, channel2)
    df2 = read_smart_cor(path, filename2)
    spectrometer1 = lookup[f"{direction}_{channel}"]
    spectrometer2 = lookup[f"{direction}_{channel2}"]
    pixel_wl1 = read_pixel_to_wavelength(pixel_path, spectrometer1)
    pixel_wl2 = read_pixel_to_wavelength(pixel_path, spectrometer2)
    # merge pixel dfs and sort by wavelength
    # pixel_wl = pixel_wl1.append(pixel_wl2, ignore_index=True).sort_values(by="wavelength", ignore_index=True)
    max_id1 = len(df1) - 1
    max_id2 = len(df2) - 1
    try:
        df_sel1 = df1.iloc[index, :]
        df_sel2 = df2.iloc[index, :]
    except IndexError as e:
        log.info(f"{e}\nGiven index '{index}' out-of-bounds! Using maximum index '{max_id1}'!")
        df_sel1 = df1.iloc[max_id1, :]
        df_sel2 = df2.iloc[max_id2, :]

    time_stamp = df_sel1.name  # get time stamp which is selected
    assert time_stamp == df_sel2.name, "Time stamps from VNIR and SWIR are not identical!"

    pixel_wl1[f"{direction}"] = df_sel1.reset_index(drop=True)
    pixel_wl2[f"{direction}"] = df_sel2.reset_index(drop=True)
    ylabel = "Irradiance (W$\\,$m$^{-2}\\,$nm$^{-1}$)" if "F" in filename else "Radiance (W$\\,$sr$^{-1}\\,$m$^{-2}\\,$nm$^{-1}$)"

    fig, ax = plt.subplots()
    ax.plot(pixel_wl1["wavelength"], pixel_wl1[f"{direction}"], label=f"{channel}")
    ax.plot(pixel_wl2["wavelength"], pixel_wl2[f"{direction}"], label=f"{channel2}")
    # ax.plot("wavelength", f"{direction}", data=pixel_wl1, label=f"{channel}")
    # ax.plot("wavelength", f"{direction}", data=pixel_wl2, label=f"{channel2}")
    ax.set_title(f"SMART Spectra {direction} {channel}/{channel2} \n {time_stamp:%Y-%m-%d %H:%M:%S}")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.legend()
    if save_fig:
        figname = filename.replace(".dat", f"_spectra_{time_stamp:%Y%m%d_%H%M%S}.png")
        figname = figname.replace(channel, "VNIR+SWIR")
        figpath = f"{plot_path}/{figname}"
        plt.savefig(figpath, dpi=100)
        log.info(f"Saved {figpath}")

    plt.show()
    plt.close()


def plot_complete_smart_spectra_interactive(path: str, filename: str, index: int) -> hv.Overlay:
    """
    Plot the complete spectra given by both channels from SMART calibrated measurement files for a given index (time step)
    Args:
        path: where the file can be found
        filename: name of the file (standard SMART filename convention)
        index: which row to plot

    Returns: Shows and or saves a plot

    """
    pixel_path = get_path("pixel_wl")
    df1 = read_smart_cor(path, filename)
    date_str, channel, direction = get_info_from_filename(filename)
    if channel == "SWIR":
        channel2 = "VNIR"
    else:
        channel2 = "SWIR"

    filename2 = filename.replace(channel, channel2)
    df2 = read_smart_cor(path, filename2)
    spectrometer1 = lookup[f"{direction}_{channel}"]
    spectrometer2 = lookup[f"{direction}_{channel2}"]
    pixel_wl1 = read_pixel_to_wavelength(pixel_path, spectrometer1)
    pixel_wl2 = read_pixel_to_wavelength(pixel_path, spectrometer2)
    # merge pixel dfs and sort by wavelength
    # pixel_wl = pixel_wl1.append(pixel_wl2, ignore_index=True).sort_values(by="wavelength", ignore_index=True)
    max_id1 = len(df1) - 1
    max_id2 = len(df2) - 1
    try:
        df_sel1 = df1.iloc[index, :]
        df_sel2 = df2.iloc[index, :]
    except IndexError as e:
        log.info(f"{e}\nGiven index '{index}' out-of-bounds! Using maximum index '{max_id1}'!")
        df_sel1 = df1.iloc[max_id1, :]
        df_sel2 = df2.iloc[max_id2, :]

    time_stamp = df_sel1.name  # get time stamp which is selected
    assert time_stamp == df_sel2.name, "Time stamps from VNIR and SWIR are not identical!"

    pixel_wl1[f"{direction}"] = df_sel1.reset_index(drop=True)
    pixel_wl2[f"{direction}"] = df_sel2.reset_index(drop=True)
    ylabel = "Irradiance (W m^-2 nm^-1)" if "F" in filename else "Radiance (W sr^-1 m^-2 nm^-1)"

    curve1 = hv.Curve(pixel_wl1, kdims=[("wavelength", "Wavelenght (nm)")], vdims=[(f"{direction}", ylabel)], label=f"{channel}")
    curve2 = hv.Curve(pixel_wl2, kdims=[("wavelength", "Wavelenght (nm)")], vdims=[(f"{direction}", ylabel)], label=f"{channel2}")
    overlay = curve1 * curve2
    overlay.opts(
        opts.Curve(height=500, width=900, fontsize=12))
    overlay.opts(title=f"SMART Spectra {direction} {channel}/{channel2} {time_stamp:%Y-%m-%d %H:%M:%S.%f}",
                 show_grid=True)
    return overlay


def plot_smart_data_interactive(flight: str, filename: str, wavelength: Union[list, str]) -> hv.Curve:
    """
    Plot SMART data in the given file. Either a time average over a range of wavelengths or all wavelengths,
    or a time series of one wavelength.

    Args:
        flight: to which flight does the file belong to? (e.g. Flight_20210707a)
        filename: Standard SMART filename
        wavelength: list with either one or two wavelengths in nm or 'all'


    Returns: Creates an interactive figure

    """
    raw_path = get_path("raw", flight)
    pixel_wl_path = get_path("pixel_wl")
    data_path = get_path("data", flight)
    calibrated_path = get_path("calibrated", flight)
    date_str, channel, direction = get_info_from_filename(filename)
    # make sure wavelength is a list
    if type(wavelength) != list and wavelength != "all":
        wavelength = [wavelength]
    if "calibrated" in filename:
        df = read_smart_cor(calibrated_path, filename)
        title = "\nCorrected for Dark Current and Calibrated"
        ylabel = "Irradiance (W m^-2 nm^-1)" if "F" in filename else "Radiance (W sr^-1 m^-2 nm^-1)"
    elif "cor" in filename:
        df = read_smart_cor(data_path, filename)
        title = "Corrected for Dark Current"
        ylabel = "Netto Counts"
    else:
        df = read_smart_raw(raw_path, filename)
        df = df.iloc[:, 2:]  # remove columns t_int and shutter
        title = "Raw"
        ylabel = "Netto Counts"
    pixel_wl = read_pixel_to_wavelength(pixel_wl_path, lookup[f"{direction}_{channel}"])
    if len(wavelength) == 2:
        pixel_nr = []
        for wl in wavelength:
            pxl, wavel = find_pixel(pixel_wl, wl)
            pixel_nr.append(pxl)
        pixel_nr.sort()  # make sure wavelengths are in ascending order
        smart_sel = df.loc[:, pixel_nr[0]:pixel_nr[1]]  # select range of wavelengths
        begin_dt, end_dt = smart_sel.index[0], smart_sel.index[-1]  # read out start and end time
        smart_mean = smart_sel.mean(axis=0).to_frame()  # calculate mean over time and return a dataframe
        smart_mean = smart_mean.set_index(pd.to_numeric(smart_mean.index))  # update the index to be numeric
        # join the measurement and pixel to wavelength data frames by pixel
        smart_plot = smart_mean.join(pixel_wl.set_index(pixel_wl["pixel"]))
        smart_plot.columns = ["value", "pixel", "wavelength"]
        curve = hv.Curve(smart_plot, kdims=[("wavelength", "Wavelength (nm)")], vdims=[("value", ylabel)])
        title=f"Time Averaged SMART Measurement {title} {direction} {channel}\n {begin_dt} - {end_dt}"
    elif len(wavelength) == 1:
        pixel_nr, wl = find_pixel(pixel_wl, wavelength[0])
        smart_sel = df.loc[:, pixel_nr].to_frame()
        begin_dt, end_dt = smart_sel.index[0], smart_sel.index[-1]
        time_extend = end_dt - begin_dt
        smart_sel.reset_index(inplace=True)
        smart_sel.columns = ["time", "value"]
        curve = hv.Curve(smart_sel, kdims=[("time", "Time (UTC)")], vdims=[("value", ylabel)])
        title=f"SMART Time Series {title} {direction} {channel}\n{wl:.3f} nm {begin_dt:%Y-%m-%d}"
    #         ax = jr.set_xticks_and_xlabels(ax, time_extend)
    elif wavelength == "all":
        begin_dt, end_dt = df.index[0], df.index[-1]
        smart_mean = df.mean().to_frame()
        smart_mean = smart_mean.set_index(pd.to_numeric(smart_mean.index))
        smart_plot = smart_mean.join(pixel_wl.set_index(pixel_wl["pixel"]))
        smart_plot.columns = ["value", "pixel", "wavelength"]
        curve = hv.Curve(smart_plot, kdims=[("wavelength", "Wavelenght (nm)")], vdims=[("value", ylabel)])
        title=f"Time Averaged SMART Measurement {title} {direction} {channel} {begin_dt:%Y-%m-%d %H:%M:%S} - {end_dt:%Y-%m-%d %H:%M:%S}"
    else:
        raise ValueError("wavelength has to be a list of length 1 or 2 or 'all'!")

    curve.opts(
        opts.Curve(height=500, width=900, fontsize=12))
    curve.opts(title=title, show_grid=True)

    return curve


def plot_calibrated_irradiance_flux(filename: str, wavelength: Union[int, list, str], flight:str) -> hv.Overlay:
    """
    Plot upward and downward irradiance as a time averaged series over the wavelength or as a time series for one
    wavelength.
    Args:
        filename: Standard SMART filename
        wavelength: single or range of wavelength or "all"
        flight: flight folder (flight_xx)

    Returns: holoviews overlay plot with two curves

    """
    # make sure wavelength is a list
    if type(wavelength) != list and wavelength != "all":
        wavelength = [wavelength]
    # get paths and define input path
    calibrated_path, pixel_path = get_path("calibrated"), get_path("pixel_wl")
    inpath = f"{calibrated_path}/{flight}"
    date_str, channel, direction = get_info_from_filename(filename)
    direction2 = "Fdw" if direction == "Fup" else "Fup"  # set opposite direction
    filename2 = filename.replace(direction, direction2)
    # read in both irradiance measurements
    df1 = read_smart_cor(inpath, filename)
    df2 = read_smart_cor(inpath, filename2)
    # get spectrometers from lookup dictionary
    spectro1, spectro2 = lookup[f"{direction}_{channel}"], lookup[f"{direction2}_{channel}"]
    pixel_wl1 = read_pixel_to_wavelength(pixel_path, spectro1)
    pixel_wl2 = read_pixel_to_wavelength(pixel_path, spectro2)
    title = "Corrected for Dark Current and Calibrated"
    ylabel = "Irradiance (W m^-2 nm^-1)"

    if len(wavelength) == 2:
        pixel_nr1 = []
        pixel_nr2 = []
        for wl in wavelength:
            pxl, _ = find_pixel(pixel_wl1, wl)
            pixel_nr1.append(pxl)
            pxl, _ = find_pixel(pixel_wl2, wl)
            pixel_nr2.append(pxl)
        pixel_nr1.sort()  # make sure wavelengths are in ascending order
        pixel_nr2.sort()
        df1_sel = df1.loc[:, pixel_nr1[0]:pixel_nr1[1]]  # select range of wavelengths
        df1_sel.sort()  # sort data frame
        df2_sel = df2.loc[:, pixel_nr2[0]:pixel_nr2[1]]  # select range of wavelengths
        df2_sel.sort()  # sort data frame
        begin_dt, end_dt = df1_sel.index[0], df1_sel.index[-1]  # read out start and end time
        df1_mean = df1_sel.mean(axis=0).to_frame()  # calculate mean over time and return a dataframe
        df1_mean = df1_mean.set_index(pd.to_numeric(df1_mean.index))  # update the index to be numeric
        df2_mean = df2_sel.mean(axis=0).to_frame()  # calculate mean over time and return a dataframe
        df2_mean = df2_mean.set_index(pd.to_numeric(df2_mean.index))  # update the index to be numeric
        # join the measurement and pixel to wavelength data frames by pixel
        df1_plot = df1_mean.join(pixel_wl.set_index(pixel_wl["pixel"]))
        df1_plot.columns = ["value", "pixel", "wavelength"]
        df2_plot = df2_mean.join(pixel_wl.set_index(pixel_wl["pixel"]))
        df2_plot.columns = ["value", "pixel", "wavelength"]
        curve1 = hv.Curve(df1_plot, kdims=[("wavelength", "Wavelength (nm)")], vdims=[("value", ylabel)],
                          label=direction)
        curve2 = hv.Curve(df2_plot, kdims=[("wavelength", "Wavelength (nm)")], vdims=[("value", ylabel)],
                          label=direction2)
        title = f"Time Averaged SMART Measurement {title} {channel}\n {begin_dt} - {end_dt}"
        overlay = curve1 * curve2
    elif len(wavelength) == 1:
        pixel_nr1, wl1 = find_pixel(pixel_wl1, wavelength[0])
        df1_sel = df1.loc[:, pixel_nr1].to_frame()
        pixel_nr2, wl2 = find_pixel(pixel_wl2, wavelength[0])
        df2_sel = df2.loc[:, pixel_nr2].to_frame()
        begin_dt, end_dt = df1_sel.index[0], df1_sel.index[-1]
        df1_sel.reset_index(inplace=True)
        df1_sel.columns = ["time", "value"]
        df2_sel.reset_index(inplace=True)
        df2_sel.columns = ["time", "value"]
        curve1 = hv.Curve(df1_sel, kdims=[("time", "Time (UTC)")], vdims=[("value", ylabel)], label=direction)
        curve2 = hv.Curve(df2_sel, kdims=[("time", "Time (UTC)")], vdims=[("value", ylabel)], label=direction2)
        title = f"SMART Time Series {title} {channel}\n{wl1:.3f} nm {begin_dt:%Y-%m-%d}"
        overlay = curve1 * curve2
    elif wavelength == "all":
        begin_dt, end_dt = df1.index[0], df1.index[-1]
        df1_mean = df1.mean().to_frame()
        df1_mean = df1_mean.set_index(pd.to_numeric(df1_mean.index))
        df1_plot = df1_mean.join(pixel_wl.set_index(pixel_wl["pixel"]))
        df1_plot.columns = ["value", "pixel", "wavelength"]
        df2_mean = df2.mean().to_frame()
        df2_mean = df2_mean.set_index(pd.to_numeric(df2_mean.index))
        df2_plot = df2_mean.join(pixel_wl.set_index(pixel_wl["pixel"]))
        df2_plot.columns = ["value", "pixel", "wavelength"]
        curve1 = hv.Curve(df1_plot, kdims=[("wavelength", "Wavelenght (nm)")], vdims=[("value", ylabel)], label=direction)
        curve2 = hv.Curve(df2_plot, kdims=[("wavelength", "Wavelenght (nm)")], vdims=[("value", ylabel)], label=direction2)
        title = f"Time Averaged SMART Irradiance Measurement {title} {channel} {begin_dt:%Y-%m-%d %H:%M:%S} - {end_dt:%Y-%m-%d %H:%M:%S}"
        overlay = curve1 * curve2
    else:
        raise ValueError("wavelength has to be a list of length 1 or 2 or 'all'!")

    overlay.opts(
        opts.Curve(height=500, width=900, fontsize=12))
    overlay.opts(title=title, show_grid=True)

    return overlay


if __name__ == '__main__':
    # test read in functions
    path = "C:/Users/Johannes/Documents/Doktor/campaigns/CIRRUS-HL/SMART/raw_only/ASP06_transfer_calib_20210616/Tint_500ms"
    filename = "2021_06_16_07_20.Fdw_SWIR.dat"
    # filename = "2021_03_29_11_15.Fup_SWIR.dat"
    smart = read_smart_raw(path, filename)

    pixel_wl_path = get_path("pixel_wl")
    spectrometer = "ASP06_J3"
    pixel_wl = read_pixel_to_wavelength(pixel_wl_path, spectrometer)

    # find pixel closest to given wavelength
    pixel_nr, wavelength = find_pixel(pixel_wl, 525)

    # input: spectrometer, filename, option
    # option = 2
    # filename = "2021_03_29_11_07.Fup_VNIR.dat"
    # filename = "2021_03_29_11_07.Fup_SWIR.dat"
    # dark_current = get_dark_current(filename, option)

    # # correct raw measurement with dark current
    # # input: smart measurement, option
    # option = 2
    # filename = "2021_03_29_11_07.Fup_VNIR.dat"
    # smart_cor = correct_smart_dark_current(filename, option)
    #
    # # plot mean corrected smart measurement
    # raw_path, _, _, _, _ = set_paths()
    # filename = "2021_03_29_11_07.Fup_VNIR.dat"
    # option = 2
    # smart = read_smart_raw(raw_path, filename)
    # smart_cor = correct_smart_dark_current(filename, option=option)
    # measurement = smart.mean().iloc[2:]
    # measurement_cor = smart_cor.mean()
    # plot_mean_corrected_measurement(filename, measurement, measurement_cor, option)
    #
    # # read corrected file
    # _, _, _, data_path, _ = set_paths()
    # filename = "2021_03_29_11_07.Fup_VNIR_cor.dat"
    # smart_cor = read_smart_cor(data_path, filename)
    #
    # # plot any smart measurement given a range of wavelengths, one specific one or all
    # filename = "2021_03_29_11_07.Fup_VNIR_cor.dat"
    # raw_file = "2021_03_29_11_07.Fup_VNIR.dat"
    # wavelength = [525]
    # plot_smart_data(filename, wavelength)
    # plot_smart_data(raw_file, wavelength)
    #
    # # plot SMART spectra
    # file = "2021_06_04_13_40.Fdw_VNIR_cor_calibrated.dat"
    # path = "C:/Users/Johannes/Documents/Doktor/campaigns/CIRRUS-HL/SMART/calibrated_data/flight_00"
    # index = 500
    # plot_smart_spectra(path, file, index)

    # plot a complete spectra from both channels
    filename = "2021_06_21_08_41.Fdw_SWIR_cor_calibrated_norm.dat"
    path = f"{get_path('calibrated')}/flight_00"
    index = 500
    plot_path = "C:/Users/Johannes/Documents/Doktor/campaigns/CIRRUS-HL/SMART/plots/quicklooks/flight_00/spectra"
    plot_complete_smart_spectra(path, filename, index, save_fig=True, plot_path=plot_path)

    # working section
    raw_file = "2021_03_29_11_15.Fdw_SWIR.dat"
    calibrated_file = "2021_06_24_10_28.Fdw_VNIR_cor_calibrated_norm.dat"
    file = "2021_06_25_06_14.Iup_SWIR.dat"
    path = "C:/Users/Johannes/Documents/Doktor/campaigns/CIRRUS-HL/SMART/calib/ASP07_transfer_calib_20210625/dark_300ms"
    plot_smart_data("Flight_20210625a", file, "all", path=path)
    smart = read_smart_raw(path, file)
    fig, ax = plt.subplots()
    smart.iloc[4:, 2:].plot(ax=ax)
    smart.iloc[:, 2:].plot(ax=ax, label="dark")
    (smart.iloc[2, 2:] - smart.iloc[21, 2:]).plot(ax=ax, label="diff")
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()

