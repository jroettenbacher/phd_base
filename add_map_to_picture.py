#!\usr\bin\env python
"""Add the timstamp to a picture (run on Ubuntu)
author: Johannes Röttenbacher
"""

# %%
import os
import pandas as pd
from tqdm import tqdm
from subprocess import Popen
import smart

# %% set paths
bahamas_path = smart.get_path("bahamas")
date = "20210707"
number = "a"
# path = "/mnt/e/CIRRUS-HL/Gopro/20210629"
path = f"/mnt/c/Users/Johannes/Documents/Gopro/{date}"
map_path = f"{bahamas_path}/plots/Flight_{date}{number}/time_lapse"
maps = [os.path.join(map_path, f) for f in os.listdir(map_path) if f.endswith(".png")]
map_numbers = pd.read_csv(f"{path}/../{date}_timestamps_sel.csv", index_col="datetime", parse_dates=True)
f1, f2 = map_numbers.number.iloc[0], map_numbers.number.iloc[-1]
files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".JPG")][f1-1:f2-1]
processes = set()
max_processes = 10

# %% add map to the right upper corner of the picture
for picture, map in zip(tqdm(files, desc="Add Map"), maps):
    processes.add(Popen(['convert', picture, map, '-geometry', '+2900+0', '-composite', picture.replace(".JPG", "_new.JPG")]))
    if len(processes) >= max_processes:
        os.wait()
        processes.difference_update([p for p in processes if p.poll() is not None])


print(f"Done with all files in {path}")
