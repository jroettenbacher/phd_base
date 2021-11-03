#!\usr\bin\env python
"""Add the bahamas map to a picture (run on Ubuntu)
saves to a new directory
author: Johannes Röttenbacher
"""

# %%
import pylim.helpers as h
import os
import pandas as pd
from tqdm import tqdm
from subprocess import Popen

# %% set paths
date = "20210729"
number = "a"
flight = f"Flight_{date}{number}"
gopro_dir = h.get_path('gopro')
gopro_path = f"{gopro_dir}/{date}"
map_path = f"{h.get_path('bahamas', flight)}/plots/time_lapse"
maps = [os.path.join(map_path, f) for f in os.listdir(map_path) if f.endswith(".png")]
map_numbers = pd.read_csv(f"{gopro_dir}/{flight}_timestamps_sel.csv", index_col="datetime", parse_dates=True)
f1, f2 = map_numbers.number.iloc[0], map_numbers.number.iloc[-1]
files = [os.path.join(gopro_path, f) for f in os.listdir(gopro_path) if f.endswith(".JPG")][f1-1:f2-1]
outpath = f"{gopro_dir}/{flight}"
h.make_dir(outpath)
processes = set()
max_processes = 10

# %% add map to the right upper corner of the picture
# for picture, map in zip(files[:1], maps[:1]):  # for testing
for picture, map in zip(tqdm(files, desc="Add Map"), maps):
    outfile = picture.replace(gopro_path, outpath)
    processes.add(Popen(['convert', picture,
                         map, '-geometry', '+3100+0',
                         '-composite', outfile]))
    if len(processes) >= max_processes:
        os.wait()
        processes.difference_update([p for p in processes if p.poll() is not None])


print(f"Done with all files for {flight}")
