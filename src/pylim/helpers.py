#!/usr/bin/env python
"""General helper functions
author: Johannes Röttenbacher
"""
import os
import toml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime
import logging

log = logging.getLogger(__name__)

# ecRad bands in nanometers
ecRad_bands = dict(Band1=(3077, 3846), Band2=(2500, 3076), Band3=(2150, 2500), Band4=(1942, 2150), Band5=(1626, 1941),
                   Band6=(1299, 1625), Band7=(1242, 1298), Band8=(778, 1241), Band9=(625, 777), Band10=(442, 624),
                   Band11=(345, 442), Band12=(263, 344), Band13=(200, 262), Band14=(3846, 12195))


def get_path(key: str, flight: str = None, campaign: str = "cirrus-hl", instrument: str = None) -> str:
    """
        Read paths from the toml file according to the current working directory.

        Args:
            key: which path to return, see config.toml for possible values
            flight: for which flight should the path be provided (eg. Flight_20210625a for CIRRUS-HL or HALO-AC3_FD_00_HALO_Flight_01_20220225 for HALO-AC3)
            campaign: campaign for which the paths should be generated
            instrument: if key=all which instrument to generate the path to? (e.g. BAHAMAS)

        Returns: Path to specified data

    """
    # make sure to search for the config file in the project directory
    wk_dir = os.getcwd()
    log.debug(f"Searching for config.toml in {wk_dir}")
    if wk_dir.startswith("C"):
        config = toml.load(f"{wk_dir}/config.toml")[campaign]["jr_local"]
    elif wk_dir.startswith("/mnt"):
        config = toml.load(f"{wk_dir}/config.toml")[campaign]["jr_ubuntu"]
    else:
        config = toml.load(f"{wk_dir}/config.toml")[campaign]["lim_server"]

    flight = "" if flight is None else flight
    paths = dict()
    base_dir = config.pop("base_dir")
    paths["base"] = base_dir
    for k in config:
        paths[k] = os.path.join(base_dir, flight, config[k])
    for k in ["calib", "pixel_wl", "lamp", "panel"]:
        try:
            paths[k] = config[k]
        except KeyError:
            log.debug(f"Found no path for key: {k} and campaign {campaign}")
            pass
    if key == 'all':
        paths['all'] = os.path.join(base_dir, config[key], instrument)

    return paths[key]


def make_dir(folder: str) -> None:
    """
    Creates folder if it doesn't exist already.

    Args:
        folder: folder name or full path

    Returns: nothing, but creates a new folder if possible

    """
    try:
        os.makedirs(folder)
    except FileExistsError:
        pass


def arg_nearest(array, value):
    """
    Find the index of the nearest value in an array.

    Args:
        array: Input has to be convertible to an ndarray
        value: Value to search for

    Returns: index of closest value

    """
    array = np.asarray(array)
    idx = np.nanargmin(np.abs(array - value))
    return idx


# from pyLARDA.Transformations
def set_xticks_and_xlabels(ax: plt.axis, time_extend: datetime.timedelta) -> plt.axis:
    """This function sets the ticks and labels of the x-axis (only when the x-axis is time in UTC).

    Options:
        -   time_extend > 7 days:               major ticks every 2 day,  minor ticks every 12 hours
        -   7 days > time_extend > 2 days:      major ticks every day, minor ticks every  6 hours
        -   2 days > time_extend > 1 days:      major ticks every 12 hours, minor ticks every  3 hours
        -   1 days > time_extend > 12 hours:    major ticks every 2 hours, minor ticks every  30 minutes
        -   12hours > time_extend > 6 hours:    major ticks every 1 hours, minor ticks every  30 minutes
        -   6 hours > time_extend > 2 hour:     major ticks every hour, minor ticks every  15 minutes
        -   2 hours > time_extend > 15 min:     major ticks every 15 minutes, minor ticks every 5 minutes
        -   15 min > time_extend > 5 min:       major ticks every 15 minutes, minor ticks every 5 minutes
        -   else:                               major ticks every minute, minor ticks every 10 seconds

    Args:
        ax: axis in which the x-ticks and labels have to be set
        time_extend: time difference of t_end - t_start (format datetime.timedelta)

    Returns:
        ax - axis with new ticks and labels
    """

    if time_extend > datetime.timedelta(days=30):
        pass
    elif datetime.timedelta(days=30) > time_extend >= datetime.timedelta(days=7):
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(bymonthday=range(1, 32, 2)))
        ax.xaxis.set_minor_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 12)))
    elif datetime.timedelta(days=7) > time_extend >= datetime.timedelta(days=2):
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0]))
        ax.xaxis.set_minor_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 6)))
    elif datetime.timedelta(days=2) > time_extend >= datetime.timedelta(hours=25):
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d\n%H:%M'))
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 12)))
        ax.xaxis.set_minor_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 3)))
    elif datetime.timedelta(hours=25) > time_extend >= datetime.timedelta(hours=12):
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 2)))
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 30)))
    elif datetime.timedelta(hours=12) > time_extend >= datetime.timedelta(hours=6):
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator())
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 30)))
    elif datetime.timedelta(hours=6) > time_extend >= datetime.timedelta(hours=2):
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=1))
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 15)))
    elif datetime.timedelta(hours=2) > time_extend >= datetime.timedelta(minutes=15):
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 15)))
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 5)))
    elif datetime.timedelta(minutes=15) > time_extend >= datetime.timedelta(minutes=5):
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 15)))
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 5)))
    else:
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 1)))
        ax.xaxis.set_minor_locator(matplotlib.dates.SecondLocator(interval=10))

    return ax


def set_cb_friendly_colors():
    """Set new colorblind friendly color cycle.

    Returns: Modifies the standard pyplot color cycle

    """
    cb_color_cycle = ["#88CCEE", "#CC6677", "#DDCC77", "#117733", "#332288", "#AA4499", "#44AA99", "#999933", "#882255",
                      "#661100", "#6699CC", "#888888"]
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=cb_color_cycle)


def nested_dict_values_iterator(dict_obj: dict):
    """ Loop over all values in a nested dictionary
    See: https://thispointer.com/python-iterate-loop-over-all-nested-dictionary-values/

    Args:
        dict_obj: nested dictionary

    Returns: Each value in a nested dictionary

    """
    # Iterate over all values of dict argument
    for value in dict_obj.values():
        # Check if value is of dict type
        if isinstance(value, dict):
            # If value is dict then iterate over all its values
            for v in nested_dict_values_iterator(value):
                yield v
        else:
            # If value is not dict type then yield the value
            yield value


def nested_dict_pairs_iterator(dict_obj: dict):
    """ Loop over all values in a nested dictionary and return the key, value pair
    See: https://thispointer.com/python-how-to-iterate-over-nested-dictionary-dict-of-dicts/

    Args:
        dict_obj: nested dictionary

    Returns: Each value in a nested dictionary with its key

    """
    # Iterate over all values of dict argument
    for key, value in dict_obj.items():
        # Check if value is of dict type
        if isinstance(value, dict):
            # If value is dict then iterate over all its values
            for pair in nested_dict_pairs_iterator(value):
                yield key, *pair
        else:
            # If value is not dict type then yield the key, value pair
            yield key, value


def setup_logging(dir: str, custom_string: str = None):
    """
    Setup up logging to file if script is called from console. If it is executed inside a console setup logging only
    to console.

    Args:
        dir: Directory where to save logging file. Gets created if it doesn't exist yet.
        custom_string: Custom String to append to logging file name. Logging file always starts with date and the name
                       of the script being called.

    Returns: Logger
    """
    log = logging.getLogger("pylim")
    try:
        log.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        make_dir(dir)
        fh = logging.FileHandler(f'{dir}/{datetime.datetime.utcnow():%Y%m%d}_{__file__[:-3]}_{custom_string}.log')
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s : %(levelname)s - %(message)s', datefmt="%c")
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        # add the handlers to logger
        log.addHandler(ch)
        log.addHandler(fh)
    except NameError:
        # __file__ is undefined if script is executed in console, set a normal logger instead
        log.addHandler(logging.StreamHandler())
        log.setLevel(logging.INFO)

    return log