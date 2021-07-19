#!/usr/bin/env python
"""Read out timestamps from GoPro images and write to a text file
author: Johannes Röttenbacher
"""

# TODO: correct for time shift in GoPro image
# %% module import
from subprocess import run
import re
import pandas as pd

# %% set paths
date = 20210707
path = f"/mnt/c/Users/Johannes/Documents/Gopro/{date}"
# file = f"{path}/{date}_Gopro_0001.jpg"

# write meta data info to file
with open(f"{path}/../{date}_timestamps.txt", "w+") as outfile:
    run(['exiftool', '-m', '-DateTimeOriginal', path], stdout=outfile)

# %% read the file and create a pandas data frame and write to csv again
pic_num = list()
timestamps = list()
with open(f"{path}/../{date}_timestamps.txt", "r") as ts:
    for line in ts.readlines():
        if line.startswith("="):
            pic_num.append(re.findall(r"2021\d{4}_Gopro_(?P<number>\d{4}).JPG", line)[0])
        elif line.startswith("Date"):
            timestamps.append(re.findall(r"2021:\d{2}:\d{2} \d{2}:\d{2}:\d{2}", line)[0])
        else:
            pass

ts_dt = pd.to_datetime(timestamps, format="%Y:%m:%d %H:%M:%S")
df = pd.DataFrame(dict(number=pic_num), index=ts_dt)\
    .to_csv(f"{path}/../{date}_timestamps.csv", index_label="datetime")

