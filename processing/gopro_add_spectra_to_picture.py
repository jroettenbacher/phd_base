#!\usr\bin\env python
"""Add the smart spectra to a picture (run on Ubuntu)
overwrites picture with map on ir
author: Johannes Röttenbacher
"""

# %%
import pylim.helpers as h
import os
from tqdm import tqdm
from subprocess import Popen

# %% set paths
date = "20210707"
number = "a"
flight = f"Flight_{date}{number}"
gopro_dir = h.get_path('gopro')
gopro_path = f"{gopro_dir}/{flight}"
smart_path = f"{h.get_path('plot')}/time_lapse/{flight}"
files = [os.path.join(gopro_path, f) for f in os.listdir(gopro_path) if f.endswith(".JPG")]
smart_spectra = [os.path.join(smart_path, f) for f in os.listdir(smart_path) if f.endswith(".png")]
processes = set()
max_processes = 10

# %% add spectra to the left upper corner of the picture
for picture, smart_plot in zip(tqdm(files, desc="Add Spectra"), smart_spectra):
    processes.add(Popen(['convert', picture,
                         smart_plot, '-geometry', '+0+0',
                         '-composite', picture]))
    if len(processes) >= max_processes:
        os.wait()
        processes.difference_update([p for p in processes if p.poll() is not None])


print(f"Done with all files for {flight}")
