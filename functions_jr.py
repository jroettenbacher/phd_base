#!/usr/bin/env python
"""More general useful functions.
author: Johannes Röttenbacher and others
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime


# from pyLARDA.helpers
def argnearest(array, value):
    """find the index of the nearest value in a sorted array
    for example time or range axis

    Args:
        array (np.array): sorted array with values, list will be converted to 1D array
        value: value to find
    Returns:
        index
    """
    if type(array) == list:
        array = np.array(array)
    i = np.searchsorted(array, value) - 1

    if not i == array.shape[0] - 1:
        if np.abs(array[i] - value) > np.abs(array[i + 1] - value):
            i = i + 1
    return i


def make_dir(folder):
    """
    creates folder if it doesn't exist already
    Args:
        folder (str): folder name

    Returns: nothing, but creates a new folder if possible

    """
    try:
        os.makedirs(folder)
    except FileExistsError:
        pass


# from pyLARDA.Transformations
def set_xticks_and_xlabels(
        ax: plt.axis,
        time_extend: datetime.timedelta
) -> plt.axis:
    """This function sets the ticks and labels of the x-axis (only when the x-axis is time in UTC).

    Options:
        -   time_extend > 7 days:               major ticks every 2 day,  minor ticks every 12 hours
        -   7 days > time_extend > 2 days:      major ticks every day, minor ticks every  6 hours
        -   2 days > time_extend > 1 days:      major ticks every 12 hours, minor ticks every  3 hours
        -   1 days > time_extend > 6 hours:     major ticks every 3 hours, minor ticks every  30 minutes
        -   6 hours > time_extend > 1 hour:     major ticks every hour, minor ticks every  15 minutes
        -   else:                               major ticks every 15 minutes, minor ticks every  5 minutes

    Args:
        ax: axis in which the x-ticks and labels have to be set
        time_extend: time difference of t_end - t_start (format datetime.timedelta)

    Returns:
        ax - axis with new ticks and labels
    """

    if time_extend > datetime.timedelta(days=30):
        pass
    elif datetime.timedelta(days=30) > time_extend > datetime.timedelta(days=7):
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(bymonthday=range(1, 32, 2)))
        ax.xaxis.set_minor_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 12)))
    elif datetime.timedelta(days=7) > time_extend > datetime.timedelta(days=2):
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=[0]))
        ax.xaxis.set_minor_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 6)))
    elif datetime.timedelta(days=2) > time_extend > datetime.timedelta(hours=25):
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %d\n%H:%M'))
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 12)))
        ax.xaxis.set_minor_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 3)))
    elif datetime.timedelta(hours=25) > time_extend > datetime.timedelta(hours=6):
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(byhour=range(0, 24, 3)))
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 30)))
    elif datetime.timedelta(hours=6) > time_extend > datetime.timedelta(hours=2):
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(matplotlib.dates.HourLocator(interval=1))
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 15)))
    elif datetime.timedelta(hours=2) > time_extend > datetime.timedelta(minutes=15):
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 30)))
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 5)))
    elif datetime.timedelta(minutes=15) > time_extend > datetime.timedelta(minutes=5):
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 15)))
        ax.xaxis.set_minor_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 5)))
    else:
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 1)))
        ax.xaxis.set_minor_locator(matplotlib.dates.SecondLocator(interval=10))

    return ax
