#!/usr/bin/env python
"""Read out timestamps from GoPro images and write to a text file (run on Ubuntu)

* use exiftool to get timestamps of GoPro images
* write the output to a textfile
* read the textfile line by line and extract the picture number and datetime
* convert to pd.datetime and correct for the time offset to BAHAMAS
* create DataFrame and save to csv

*author*: Johannes Röttenbacher
"""
if __name__ == "__main__":
    # %% module import
    import pylim.helpers as h
    from subprocess import run
    import re
    import pandas as pd

    # %% set paths
    campaign = "halo-ac3"
    flight = "HALO-AC3_20220225_HALO_RF00"
    date = flight[9:17]
    path = f"{h.get_path('gopro', campaign=campaign)}/{flight}"
    # file = f"{path}/{date}_Gopro_0001.jpg"

    # write meta data info to file
    with open(f"{path}/../{flight}_timestamps.txt", "w+") as outfile:
        run(['exiftool', '-m', '-DateTimeOriginal', path], stdout=outfile)

    # %% read the file and create a pandas data frame and write to csv again
    pic_num = list()
    timestamps = list()
    with open(f"{path}/../{flight}_timestamps.txt", "r") as ts:
        for line in ts.readlines():
            if line.startswith("="):
                pic_num.append(re.findall(r"2022\d{4}_Gopro_(?P<number>\d{4}).JPG", line)[0])
            elif line.startswith("Date"):
                timestamps.append(re.findall(r"2022:\d{2}:\d{2} \d{2}:\d{2}:\d{2}", line)[0])
            else:
                pass

    # %% convert timestamps to datetime
    ts_dt = pd.to_datetime(timestamps, format="%Y:%m:%d %H:%M:%S")

    # %% create a pandas data frame and write to csv
    df = pd.DataFrame(dict(number=pic_num), index=ts_dt).to_csv(f"{path}/../{flight}_timestamps.csv",
                                                                index_label="datetime")
    print(f"Done with {flight}")
