#!/usr/bin/env python
"""Functions to read in quicklook files and attitude correction

*author*: Johannes Röttenbacher
"""

# %% module import
import pylim.helpers as h
from pylim import reader
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Patch
import pandas as pd
import numpy as np
import os
import re
import logging

log = logging.getLogger(__name__)

# %% functions


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
    # overwrite TIME to make a datetime index
    ds = ds.assign({"time": pd.to_datetime(ds.time, unit='ms', origin=pd.to_datetime(date, format="%Y%m%d"))})

    return ds


def fdw_attitude_correction(fdw, roll, pitch, yaw, sza, saa, fdir, r_off: float = 0, p_off: float = 0):
    """Attitude Correction for downward irradiance.
    Corrects downward irradiance for misalignment of the sensor (deviation from horizontal alignment).

    - only direct fraction of irradiance can be corrected by the equation, therefore a direct fraction (fdir) has to be provided
    - please check correct definition of the attitude angle
    - for differences between the sensor attitude and the attitude given by an INS the offset angles (p_off and r_off) can be defined.

    Args:
        fdw: downward irradiance [W m-2] or [W m-2 nm-1]
        roll: roll angle [deg] - defined positive for left wing up
        pitch: pitch angle [deg] - defined positive for nose down
        yaw: yaw angle [deg] - defined clockwise with North=0°
        sza: solar zenith angle [deg]
        saa: solar azimuth angle [deg] - defined clockwise with North=0°
        r_off: roll offset angle between INS and sensor [deg] - defined positive for left wing up
        p_off: pitch offset angle between INS and sensor [deg] - defined positive for nose down
        fdir: fraction of direct radiation [0..1] (0=pure diffuse, 1=pure direct)

    Returns: corrected downward irradiance [W m-2] or [W m-2 nm-1] and correction factor

    """
    r = np.deg2rad(roll + r_off)
    p = np.deg2rad(pitch + p_off)
    h0 = np.deg2rad(90 - sza)

    factor = np.sin(h0) / \
             (np.cos(h0) * np.sin(r) * np.sin(np.deg2rad(saa - yaw)) +
              np.cos(h0) * np.sin(p) * np.cos(r) * np.cos(np.deg2rad(saa - yaw)) +
              np.sin(h0) * np.cos(p) * np.cos(r))
    try:
        fdw_cor = fdir * fdw * factor + (1 - fdir) * fdw
    except ValueError:
        # fdw and fdir are 2 dimensional, add an empty axis to the factor two make it 2D as well
        fdw_cor = fdir * fdw * factor[:, None] + (1 - fdir) * fdw

    return fdw_cor, factor

