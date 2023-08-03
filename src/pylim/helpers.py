#!/usr/bin/env python
"""General helper functions and general information

*author*: Johannes Röttenbacher
"""
import os
import shutil
import sys
from itertools import groupby
import toml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import datetime
import logging
import cmasher as cmr
from tqdm import tqdm

log = logging.getLogger(__name__)

cm = 1 / 2.54

# ecRad bands in nanometers
ecRad_bands = dict(Band1=(3077, 3846), Band2=(2500, 3076), Band3=(2150, 2500), Band4=(1942, 2150), Band5=(1626, 1941),
                   Band6=(1299, 1625), Band7=(1242, 1298), Band8=(778, 1241), Band9=(625, 777), Band10=(442, 624),
                   Band11=(345, 442), Band12=(263, 344), Band13=(200, 262), Band14=(3846, 12195))

# sea ice albedo bands micron
ci_bands = [(0.185, 0.25), (0.25, 0.44), (0.44, 0.69), (0.69, 1.19), (1.19, 2.38), (2.38, 4.0)]
# sea ice albedo in 6-spectral intervals for each month
ci_albedo = np.empty((12, 6))
# Sea ice surf. albedo for 0.185-0.25 micron (snow covered; Ebert and Curry, 1993)
ci_albedo[:, 0] = (0.975, 0.975, 0.975, 0.975,
                   0.975, 0.876, 0.778, 0.778,
                   0.975, 0.975, 0.975, 0.975)
# Sea ice surf. albedo for 0.25-0.44 micron (snow covered; Ebert and Curry, 1993)
ci_albedo[:, 1] = (0.975, 0.975, 0.975, 0.975,
                   0.975, 0.876, 0.778, 0.778,
                   0.975, 0.975, 0.975, 0.975)
# Sea ice surf. albedo for 0.44-0.69 micron (snow covered; Ebert and Curry, 1993)
ci_albedo[:, 2] = (0.975, 0.975, 0.975, 0.975,
                   0.975, 0.876, 0.778, 0.778,
                   0.975, 0.975, 0.975, 0.975)
# Sea ice surf. albedo for 0.69-1.19 micron (snow covered; Ebert and Curry, 1993)
ci_albedo[:, 3] = (0.832, 0.832, 0.832, 0.832,
                   0.832, 0.638, 0.443, 0.443,
                   0.832, 0.832, 0.832, 0.832)
# Sea ice surf. albedo for 1.19-2.38 micron (snow covered; Ebert and Curry, 1993)
ci_albedo[:, 4] = (0.250, 0.250, 0.250, 0.250,
                   0.250, 0.153, 0.055, 0.055,
                   0.250, 0.250, 0.250, 0.250)
# Sea ice surf. albedo for 2.38-4.00 microns (snow covered; Ebert and Curry, 1993)
ci_albedo[:, 5] = (0.025, 0.025, 0.025, 0.025,
                   0.025, 0.030, 0.036, 0.036,
                   0.025, 0.025, 0.025, 0.025)

# ozone sonde stations
ozone_files = dict(Flight_20210629a="sc210624.b11",
                   RF17="ny220413.b16", RF18="ny220413.b16")

# plotting metadata
figsize_wide = (24 * cm, 12 * cm)
figsize_equal = (12 * cm, 12 * cm)
plot_units = dict(cloud_fraction="", clwc=r"mg$\,$kg$^{-1}$", ciwc=r"mg$\,$kg$^{-1}$", cswc=r"mg$\,$kg$^{-1}$",
                  q_ice=r"mg$\,$kg$^{-1}$", q_liquid=r"mg$\,$kg$^{-1}$", iwp=r"g$\,$m$^{-2}$", iwc=r"mg$\,$m$^{-3}$",
                  crwc=r"mg$\,$kg$^{-1}$", t="K", q=r"g$\,$kg$^{-1}$", re_ice=r"$\mu$m", re_liquid=r"$\mu$m",
                  heating_rate_sw=r"K$\,$day$^{-1}$", heating_rate_lw=r"K$\,$day$^{-1}$",
                  heating_rate_net=r"K$\,$day$^{-1}$",
                  flux_dn_sw=r"W$\,$m$^{-2}$", flux_dn_lw=r"W$\,$m$^{-2}$", flux_up_sw=r"W$\,$m$^{-2}$",
                  flux_up_lw=r"W$\,$m$^{-2}$",
                  cre_sw=r"W$\,$m$^{-2}$", cre_lw=r"W$\,$m$^{-2}$", cre_total=r"W$\,$m$^{-2}$",
                  transmissivity_sw="", transmissivity_lw="", reflectivity_sw="", reflectivity_lw="",
                  od="", scat_od="", od_mean="", scat_od_mean="", g="", g_mean="", od_int="", scat_od_int="", g_int="",
                  absorption="", absorption_int="",
                  eglo=r"W$\,$m$^{-2}\,$nm$^{-1}$", eglo_int=r"W$\,$m$^{-2}$", eup=r"W$\,$m$^{-2}\,$nm$^{-1}$",
                  eup_int=r"W$\,$m$^{-2}$")

cbarlabels = dict(cloud_fraction="Cloud fraction", clwc="Cloud liquid water content", ciwc="Cloud ice water content",
                  cswc="Cloud snow water content", crwc="Cloud rain water content", t="Temperature",
                  q="Specific humidity", q_ice="Ice mass mixing ratio", q_liquid="Liquid mass mixing ratio",
                  iwp="Ice water path", iwc="Ice water content",
                  re_ice="Ice effective radius", re_liquid="Liquid effective radius",
                  heating_rate_sw="Solar heating rate", heating_rate_lw="Terrestrial heating rate",
                  heating_rate_net="Net heating rate",
                  transmissivity_sw="Solar transmissivity", transmissivity_lw="Terrestrial transmissivity",
                  reflectivity_sw="Solar reflectivity", reflectivity_lw="Terrestrial reflectivity",
                  flux_dn_sw="Downward solar irradiance", flux_up_sw="Upward solar irradiance",
                  flux_dn_lw="Downward terrestrial irradiance", flux_up_lw="Upward terrestrial irradiance",
                  cre_sw="Solar cloud radiative effect", cre_lw="Terrestrial cloud radiative effect",
                  cre_total="Total cloud radiative effect",
                  od=f"Total optical depth", scat_od=f"Scattering optical depth", od_mean=f"Mean total optical depth",
                  scat_od_mean=f"Mean scattering optical depth", g=f"Asymmetry factor", g_mean="Mean asymmetry factor",
                  od_int="Integrated total optical depth", scat_od_int="Integrated scattering optical depth",
                  absorption="Absorption", absorption_int="Integrated absorption",
                  eglo="Spectral global downward irradiance", eglo_int="Global downward irradiance",
                  eup="Spectral diffuse upward irradiance", eup_int="Diffuse upward irradiance")

scale_factors = dict(cloud_fraction=1, clwc=1e6, ciwc=1e6, cswc=1e6, crwc=1e6, t=1, q=1000, re_ice=1e6,
                     re_liquid=1e6, q_ice=1e6, q_liquid=1e6, iwp=1000, iwc=1e6)

cmaps = dict(t=cmr.prinsenvlag_r,
             ciwc=cmr.get_sub_cmap("cmr.freeze", .25, 0.85), cswc=cmr.get_sub_cmap("cmr.freeze", .25, 0.85),
             crwc=cmr.get_sub_cmap("cmr.freeze", .25, 0.85), q_ice=cmr.get_sub_cmap("cmr.freeze", .25, 0.85),
             iwp=cmr.get_sub_cmap("cmr.freeze", .25, 0.85), iwc=cmr.get_sub_cmap("cmr.freeze", .25, 0.85),
             cloud_fraction=cmr.neutral,
             re_ice=cmr.cosmic_r, re_liquid=cmr.cosmic_r,
             heating_rate_sw=cmr.get_sub_cmap("cmr.ember_r", 0, 0.75), heating_rate_lw=cmr.fusion_r,
             heating_rate_net=cmr.fusion_r,
             cre_total=cmr.fusion_r,
             flux_dn_sw=cmr.get_sub_cmap("cmr.torch", 0.2, 1))

norms = dict(t=colors.TwoSlopeNorm(vmin=200, vcenter=235, vmax=280), clwc=colors.LogNorm(),
             heating_rate_lw=colors.TwoSlopeNorm(vmin=-3, vcenter=0, vmax=1.5),
             heating_rate_net=colors.TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2),
             od=colors.LogNorm(vmax=10), od_scat=colors.LogNorm(),
             od_int=colors.LogNorm(vmax=10), scat_od_int=colors.LogNorm())

# plotting dictionaries for BACARDI
bacardi_labels = dict(F_down_solar=r"$F_{\downarrow, solar}$", F_down_terrestrial=r"$F_{\downarrow, terrestrial}$",
                      F_up_solar=r"$F_{\uparrow, solar}$", F_up_terrestrial=r"$F_{\uparrow, terrestrial}$",
                      F_net_solar=r"$F_{net, solar}$", F_net_terrestrial=r"$F_{net, terrestrial}$",
                      CRE_solar=r"CRE$_{solar}$", CRE_terrestrial=r"CRE$_{terrestrial}$",
                      CRE_total=r"CRE$_{total}$")

def get_path(key: str, flight: str = None, campaign: str = "cirrus-hl", instrument: str = None) -> str:
    """
        Read paths from the toml file according to the current working directory.

        Args:
            key: which path to return, see config.toml for possible values
            flight: for which flight should the path be provided (e.g. Flight_20210625a for CIRRUS-HL or HALO-AC3_20220311_HALO_RF01 for HALO-AC3)
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


def delete_folder_contents(folder: str) -> None:
    """
    Deletes all files and subfolders in a folder.
    From: https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder

    Args:
        folder: folder name or full path

    Returns: nothing, but deletes all files in a folder

    """
    for filename in tqdm(os.listdir(folder)):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except OSError as e:
            print(f"Failed to delete {file_path}.\n Reason: {e}")


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
        -   2 hours > time_extend > 30 min:     major ticks every 15 minutes, minor ticks every 5 minutes
        -   30 min > time_extend > 5 min:       major ticks every 5 minutes, minor ticks every 1 minute
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
    elif datetime.timedelta(hours=2) > time_extend >= datetime.timedelta(minutes=45):
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 15)))
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 5)))
    elif datetime.timedelta(minutes=45) > time_extend >= datetime.timedelta(minutes=5):
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 5)))
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 1)))
    else:
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 1)))
        ax.xaxis.set_minor_locator(matplotlib.dates.SecondLocator(interval=10))

    return ax


def set_yticks_and_ylabels(ax: plt.axis, time_extend: datetime.timedelta) -> plt.axis:
    """This function sets the ticks and labels of the y-axis (only when the y-axis is time in UTC).

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
        ax: axis in which the y-ticks and labels have to be set
        time_extend: time difference of t_end - t_start (format datetime.timedelta)

    Returns:
        ax - axis with new ticks and labels
    """

    if time_extend > datetime.timedelta(days=30):
        pass
    elif datetime.timedelta(days=30) > time_extend >= datetime.timedelta(days=7):
        ax.yaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d'))
        ax.yaxis.set_major_locator(matplotlib.dates.DayLocator(bymonthday=range(1, 32, 2)))
        ax.yaxis.set_minor_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 12)))
    elif datetime.timedelta(days=7) > time_extend >= datetime.timedelta(days=2):
        ax.yaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d'))
        ax.yaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0]))
        ax.yaxis.set_minor_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 6)))
    elif datetime.timedelta(days=2) > time_extend >= datetime.timedelta(hours=25):
        ax.yaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d\n%H:%M'))
        ax.yaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 12)))
        ax.yaxis.set_minor_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 3)))
    elif datetime.timedelta(hours=25) > time_extend >= datetime.timedelta(hours=12):
        ax.yaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        ax.yaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 2)))
        ax.yaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 30)))
    elif datetime.timedelta(hours=12) > time_extend >= datetime.timedelta(hours=6):
        ax.yaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        ax.yaxis.set_major_locator(matplotlib.dates.HourLocator())
        ax.yaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 30)))
    elif datetime.timedelta(hours=6) > time_extend >= datetime.timedelta(hours=2):
        ax.yaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        ax.yaxis.set_major_locator(matplotlib.dates.HourLocator(interval=1))
        ax.yaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 15)))
    elif datetime.timedelta(hours=2) > time_extend >= datetime.timedelta(minutes=15):
        ax.yaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        ax.yaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 15)))
        ax.yaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 5)))
    elif datetime.timedelta(minutes=15) > time_extend >= datetime.timedelta(minutes=5):
        ax.yaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        ax.yaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 15)))
        ax.yaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 5)))
    else:
        ax.yaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        ax.yaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 1)))
        ax.yaxis.set_minor_locator(matplotlib.dates.SecondLocator(interval=10))

    return ax


def set_cb_friendly_colors():
    """Set new colorblind friendly color cycle.

    Returns: Modifies the standard pyplot color cycle

    """
    cb_color_cycle = ["#88CCEE", "#CC6677", "#DDCC77", "#117733", "#332288", "#AA4499", "#44AA99", "#999933", "#882255",
                      "#661100", "#6699CC", "#888888"]
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=cb_color_cycle)


def get_cb_friendly_colors():
    """Get colorblind friendly color cycle.

    Returns: List with colorblind friendly colors

    """
    return ["#88CCEE", "#CC6677", "#DDCC77", "#117733", "#332288", "#AA4499", "#44AA99", "#999933", "#882255",
            "#661100", "#6699CC", "#888888"]


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


def setup_logging(dir: str, file: str = None, custom_string: str = None):
    """
    Setup up logging to file if script is called from console. If it is executed inside a console setup logging only
    to console.

    Args:
        dir: Directory where to save logging file. Gets created if it doesn't exist yet.
        file: Name of the file which called the function. Should be given via the __file__ attribute.
        custom_string: Custom String to append to logging file name. Logging file always starts with date and the name
                       of the script being called.

    Returns: Logger
    """
    log = logging.getLogger("pylim")
    # remove existing handlers, necessary when function is called in a loop
    log.handlers = []
    if file is not None:
        file = os.path.basename(file)
        log.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        make_dir(dir)
        fh = logging.FileHandler(f'{dir}/{datetime.datetime.utcnow():%Y%m%d}_{file[:-3]}_{custom_string}.log')
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
    else:
        # __file__ is undefined if script is executed in console, set a normal logger instead
        log.addHandler(logging.StreamHandler())
        log.setLevel(logging.INFO)

    return log


# from pyLARDA.SpectraProcessing
def seconds_to_fstring(time_diff):
    return str(datetime.timedelta(seconds=time_diff))


def read_command_line_args():
    """
    Read out command line arguments and save them to a dictionary. Expects arguments in the form key=value.

    Returns: dictionary with command line arguments as dict[key] = value

    """
    args = dict()
    for arg in sys.argv[1:]:
        if arg.count('=') == 1:
            key, value = arg.split('=')
            args[key] = value

    return args


def set_cdo_path(path: str = "/home/jroettenbacher/.conda/envs/phd_base/bin/cdo"):
    # add cdo path to python environment
    os.environ["CDO"] = path


def generate_specific_rows(filePath, userows=[]):
    """Function for trajectory plotting"""
    with open(filePath) as f:
        for i, line in enumerate(f):
            if i in userows:
                yield line


def make_flag(boolean_array, name: str):
    """
    Make a list of flag values for plotting using a boolean array as input

    Args:
        boolean_array: array like input with True and False
        name: replace True with this string

    Returns: list with as many strings as there are True values in the input array

    """
    array = np.array(boolean_array)  # convert to numpy.ndarray
    return [str(a).replace("True", name) for a in array if a]


def find_bases_tops(mask, rg_list):
    """
    This function finds cloud bases and tops for a provided binary cloud mask.
    Args:
        mask (np.array, dtype=bool) : bool array containing False = signal, True=no-signal
        rg_list (np.ndarray) : array of range values

    Returns:
        cloud_prop (list) : list containing a dict for every time step consisting of cloud bases/top indices, range and width
        cloud_mask (np.array) : integer array, containing +1 for cloud tops, -1 for cloud bases and 0 for fill_value
    """
    cloud_prop = []
    cloud_mask = np.full(mask.shape, 0, dtype=np.int)
    # bug fix: add an emtpy first range gate to detect cloud bases of clouds which start at the first range gate
    mask = np.hstack((np.full_like(mask, fill_value=True)[:, 0:1], mask))
    for iT in range(mask.shape[0]):
        cloud = [(k, sum(1 for j in g)) for k, g in groupby(mask[iT, :])]
        idx_cloud_edges = np.cumsum([prop[1] for prop in cloud])
        bases, tops = idx_cloud_edges[0:][::2][:-1], idx_cloud_edges[1:][::2]
        if tops.size > 0:
            tops = [t - 1 for t in tops]  # reduce top indices by 1 to account for the introduced row
            if tops[-1] == cloud_mask.shape[1]:
                tops[-1] = cloud_mask.shape[1] - 1  # account for python starting counting at 0
        if bases.size > 0:
            bases = [b - 1 for b in bases]  # reduce base indices by 1 to account for the introduced row
        cloud_mask[iT, bases] = -1
        cloud_mask[iT, tops] = +1
        cloud_prop.append({'idx_cb': bases, 'val_cb': rg_list[bases],  # cloud bases
                           'idx_ct': tops, 'val_ct': rg_list[tops],  # cloud tops
                           'width': [ct - cb for ct, cb in zip(rg_list[tops], rg_list[bases])]
                           })
    return cloud_prop, cloud_mask


def longitude_values_for_gaussian_grid(latitudes: np.array,
                                       n_points: np.array,
                                       longitude_boundaries: np.array = None) -> (np.array, np.array):
    """
    Calculate the longitude values for each latitude circle on a reduced Gaussian grid.
    If the longitude boundaries are given only the longitude values within these boundaries are returned.

    The ECMWF uses regular/reduced Gaussian grids to represent their model data.
    These have a fixed number of latitudes between the equator and each pole with either a regular amount of longitude
    points on each latitude ring or in case of a reduced Gaussian grid with a decreasing number of points towards the
    poles on each latitude ring.
    For more information on Gaussian grids as used by the ECMWF see: https://confluence.ecmwf.int/display/FCST/Gaussian+grids

    When retrieving data on a reduced Gaussian grid the exact longitude values are not included in the data set and have
    to be calculated according to the definition of the grid.
    For this the latitude rings (latitudes) and the amount of longitude points on each latitude ring is needed (n_points).
    As one rarely retrieves the whole domain of the model the longitude boundaries are also needed to return the correct
    longitude values.


    Args:
        latitudes: The latitude values of the Gaussian grid starting in the North
        n_points: The number of longitude points on each latitude circle (needs to be of same length as latitudes)
        longitude_boundaries: The longitude boundaries (E, W). E =-90, W =90, N=0, S=-180/180

    Returns: Two arrays with repeating latitude values and the corresponding longitude values

    """
    assert len(latitudes) == len(n_points), "Number of latitudes does not match number of points given!"
    lon_values = [np.linspace(0, 360, num=points, endpoint=False) for points in n_points]
    lon_values_out = np.array([])
    lon_values_list = list()
    for i, lons in enumerate(lon_values):
        all_lons = np.where(lons > 180, (lons+180)%360 - 180, lons)
        assert len(all_lons) == len(np.unique(all_lons)), f"Non unique longitude values found for {i}! Check input!"
        if longitude_boundaries is not None:
            all_lons = all_lons[(all_lons >= longitude_boundaries[0]) & (all_lons <= longitude_boundaries[1])]
            all_lons.sort()
        lon_values_out = np.concatenate([lon_values_out, all_lons])
        lon_values_list.append(all_lons)

    # create list of latitude values as coordinate
    lat_values_out = np.array([])
    for i, lon_array in enumerate(lon_values_list):
        lat = np.repeat(latitudes[i], lon_array.size)
        lat_values_out = np.concatenate([lat_values_out, lat])

    return lat_values_out, lon_values_out


_COLORS = {
    "green": "#3cb371",
    "darkgreen": "#253A24",
    "lightgreen": "#70EB5D",
    "yellowgreen": "#C7FA3A",
    "yellow": "#FFE744",
    "orange": "#ffa500",
    "pink": "#B43757",
    "red": "#F57150",
    "shockred": "#E64A23",
    "seaweed": "#646F5E",
    "seaweed_roll": "#748269",
    "white": "#ffffff",
    "lightblue": "#6CFFEC",
    "blue": "#209FF3",
    "skyblue": "#CDF5F6",
    "darksky": "#76A9AB",
    "darkpurple": "#464AB9",
    "lightpurple": "#6A5ACD",
    "purple": "#BF9AFF",
    "darkgray": "#2f4f4f",
    "lightgray": "#ECECEC",
    "gray": "#d3d3d3",
    "lightbrown": "#CEBC89",
    "lightsteel": "#a0b0bb",
    "steelblue": "#4682b4",
    "mask": "#C8C8C8",
}

_CLABEL = {
    "target_classification": (
        ("_Clear sky", _COLORS["white"]),
        ("Droplets", _COLORS["lightblue"]),
        ("Drizzle or rain", _COLORS["blue"]),
        ("Drizzle & droplets", _COLORS["purple"]),
        ("Ice", _COLORS["lightsteel"]),
        ("Ice & droplets", _COLORS["darkpurple"]),
        ("Melting ice", _COLORS["orange"]),
        ("Melting & droplets", _COLORS["yellowgreen"]),
        ("Aerosols", _COLORS["lightbrown"]),
        ("Insects", _COLORS["shockred"]),
        ("Aerosols & insects", _COLORS["pink"]),
        ("No data", _COLORS["mask"]),
    ),
    "detection_status": (
        ("_Clear sky", _COLORS["white"]),
        ("Lidar only", _COLORS["yellow"]),
        ("Uncorrected atten.", _COLORS["seaweed_roll"]),
        ("Radar & lidar", _COLORS["green"]),
        ("_No radar but unknown atten.", _COLORS["purple"]),
        ("Radar only", _COLORS["lightgreen"]),
        ("_No radar but known atten.", _COLORS["orange"]),
        ("Corrected atten.", _COLORS["skyblue"]),
        ("Clutter", _COLORS["shockred"]),
        ("_Lidar molecular scattering", _COLORS["pink"]),
        ("No data", _COLORS["mask"]),)
}
