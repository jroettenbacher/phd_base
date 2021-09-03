#!/usr/bin/env python
"""Process and plot libRadTran simulation files
author: Johannes Röttenbacher
"""

# %% module import
import matplotlib.pyplot as plt
from smart import get_path
import datetime
import pandas as pd

# %% reader function


def read_libradtran(flight: str, filename: str) -> pd.DataFrame:
    """
    Read a libRadtran simulation file and add a DateTime Index.
    Args:
        flight: which flight does the simulation belong to (e.g. Flight_20210629a)
        filename: filename

    Returns: DataFrame with libRadtran output data

    """
    file_path = f"{get_path('libradtran', flight)}/{filename}"
    bbr_sim = pd.read_csv(file_path, sep="\s+", skiprows=34)
    date_dt = datetime.datetime.strptime(flight[7:15], "%Y%m%d")
    date_ts = pd.Timestamp(year=date_dt.year, month=date_dt.month, day=date_dt.day)
    bbr_sim["time"] = pd.to_datetime(bbr_sim["sod"], origin=date_ts, unit="s")  # add a datetime column
    bbr_sim = bbr_sim.set_index("time")  # set it as index

    return bbr_sim

