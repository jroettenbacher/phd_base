#!/usr/bin/env python
"""Script to merge minutely dark current corrected measurement files into one file per channel and folder.

**Required User Input:** campaign and flight

**Output:** one file with all dark current corrected measurements for one channel

ASP06 and ASP07 were configured to write minutely files.
This script

* merges the minutely files into one file,
* deletes the minutely files,
* saves the merged files with the first found filename.

By uncommenting some lines you can switch between single flight (campaign mode) and multi flight mode.

*author*: Johannes Röttenbacher
"""
if __name__ == "__main__":
    import pylim.helpers as h
    from pylim.cirrus_hl import transfer_calibs
    import os
    import pandas as pd
    import logging
    from tqdm import tqdm

    log = logging.getLogger("pylim")
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)

    # User input
    campaign = "halo-ac3"
    # uncomment for single flight use
    flights = ["HALO-AC3_20220321_HALO_RF08"]  # set flight folder
    # uncomment for all flight loop
    # flights = list(transfer_calibs.keys())[11:]  # list all flight keys
    for flight in flights:
        data_path = h.get_path("data", flight=flight, campaign=campaign)  # dark current corrected files
        directory = data_path

        channels = ["SWIR", "VNIR"]
        props = ["Fdw", "Fup", "Iup"] if campaign == "cirrus-hl" else ["Fdw"]  # set available properties depending on campaign
        corrected = True  # merge dark current corrected files
        cor = "_cor" if corrected else ""
        for dirpath, dirs, files in os.walk(directory):
            for prop in props:
                for channel in channels:
                    try:
                        filename = [file for file in files if file.endswith(f"{prop}_{channel}{cor}.dat")]
                        filename.sort()
                        if corrected:
                            df = pd.concat(
                                [pd.read_csv(f"{dirpath}/{file}", sep="\t", index_col="time") for file in
                                 tqdm(filename, desc=f"{prop}_{channel}")])
                        else:
                            df = pd.concat(
                                [pd.read_csv(f"{dirpath}/{file}", sep="\t") for file in tqdm(filename,
                                                                                             desc=f"{prop}_{channel}")])
                        # delete all minutely files
                        for file in filename:
                            os.remove(os.path.join(dirpath, file))
                        outname = f"{dirpath}/{filename[0]}"
                        if corrected:
                            df.to_csv(outname, sep="\t")
                        else:
                            df.to_csv(outname, sep="\t", index=False)
                        log.info(f"Saved {outname}")
                    except ValueError:
                        pass
