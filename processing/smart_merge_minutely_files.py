#!/usr/bin/env python
"""
ASP06 and ASP07 were configured to write minutely files during calibration.
This script
* merges the minutely files into one file,
* deletes the minutely files,
* saves the merged file to the first found filename.
author: Johannes Röttenbacher
"""
if __name__ == "__main__":
    import pylim.helpers as h
    import os
    import pandas as pd
    import logging
    from tqdm import tqdm

    log = logging.getLogger("pylim")
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)

    # combine minutely files to one file
    # User input
    campaign = "halo-ac3"
    flight = "HALO-AC3_20220412_HALO_RF18"  # set flight folder
    data_path = h.get_path("data", flight=flight, campaign=campaign)  # dark current corrected files
    directory = data_path

    channels = ["SWIR", "VNIR"]
    props = ["Fdw"]
    corrected = True  # merge dark current corrected files
    cor = "_cor" if corrected else ""
    for dirpath, dirs, files in os.walk(directory):
        for prop in props:
            for channel in channels:
                try:
                    filename = [file for file in files if file.endswith(f"{prop}_{channel}{cor}.dat")]
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
