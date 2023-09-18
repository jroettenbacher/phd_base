#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 18.09.2023

Convert raw IFS surface and multilevel file from grib to netCDF.
Use cdo to decode grib file and save it to netCDF.
Will not overwrite existing files.

**Required User Input:**

    - date (e.g. 20220313, all)

If date=all runs for all flights.

Run like this:

.. code-block:: shell

    python processing/ecrad_preprocessing.py date=all


"""
if __name__ == "__main__":
    # %% import modules
    import pylim.helpers as h
    import pylim.halo_ac3 as meta
    import os
    from tqdm import tqdm
    from cdo import *
    cdo = Cdo()

    # %% read in command line arguments
    args = h.read_command_line_args()
    date = args["date"] if "date" in args else "20220313"
    if date == "all":
        dates = [flight[9:17] for flight in list(meta.flight_names.values())[5:-3]]
    else:
        dates = [date]

    for date in tqdm(dates):
        path = f"/projekt_agmwend2/data_raw/HALO-AC3_raw_only/09_IFS_ECMWF/{date}"
        for t in ["sfc", "ml"]:
            infile = f"{path}/ifs_{date}_00_{t}_O1280.grb"
            outfile = infile.replace("grb", "nc")
            if not os.path.isfile(outfile):
                cdo.copy(input=infile, output=outfile, options="-t ecmwf -f nc")
