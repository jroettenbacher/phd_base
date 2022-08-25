#!/usr/bin/env python
"""Run uvspec
author: Johannes Röttenbacher
"""
import sys

if __name__ == "__main__":
# %% module import
    import pylim.helpers as h
    from pylim.libradtran import get_info_from_libradtran_input
    from pylim.cirrus_hl import transfer_calibs
    import numpy as np
    import pandas as pd
    import xarray as xr
    import os
    from subprocess import Popen
    from tqdm import tqdm
    from joblib import cpu_count
    import datetime as dt
    from pysolar.solar import get_azimuth

# %% set options and get files
    campaign = "cirrus-hl"
    flight = "Flight_20210628a"
    input_file = ""
    flight_key = flight[-4:] if campaign == "halo-ac3" else flight
    date = flight[9:17] if campaign == "halo-ac3" else flight[7:15]
    wavelength = "smart_spectral"  # will be used as directory name and in outfile name (e.g. smart, bacardi, 500-600nm, ...)
    uvspec_exe = "/opt/libradtran/2.0.3/bin/uvspec"
    libradtran_base_dir = h.get_path("libradtran", flight, campaign)
    libradtran_dir = os.path.join(libradtran_base_dir, "wkdir", wavelength)  # file where to find input files
    input_file = os.path.join(libradtran_dir, input_file)
    output_file = input_file.replace(".inp", ".out")
    error_log = input_file.replace(".out", ".log")

# %% setup logging
    try:
        file = __file__
    except NameError:
        file = None
    log = h.setup_logging("./logs", file, flight_key)
    log.info(f"Options Given:\ncampaign: {campaign}\nflight: {flight}\nfile: {input_file}\nwavelength: {wavelength}\n"
             f"uvspec_exe: {uvspec_exe}\nScript started: {dt.datetime.utcnow():%c UTC}")
# %% call uvspec for one file
    with open(input_file, "r") as ifile, open(output_file, "w") as ofile, open(error_log, "w") as lfile:
        Popen([uvspec_exe], stdin=ifile, stdout=ofile, stderr=lfile)
