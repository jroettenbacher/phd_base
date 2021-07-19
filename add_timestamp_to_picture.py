#!\usr\bin\env python
"""Add the timstamp to a picture (run on Ubuntu)
author: Johannes Röttenbacher
"""

# %%
import os
from tqdm import tqdm
from subprocess import run, Popen

# path = "/mnt/e/CIRRUS-HL/Gopro/20210629"
path = "/mnt/c/Users/Johannes/Documents/Gopro/20210712"
correct_time = True
start_file = 0
files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".JPG")][start_file:]
processes = set()
max_processes = 10

# %% update meta data time stamp to set to UTC (GoPro switched to LT on 28.06 due to WiFi connection)
if correct_time:
    utc_correction = 2  # convert local time to UTC
    run(['exiftool', '-m', '-progress', '-overwrite_original', f'-DateTimeOriginal-={utc_correction}', path])

# %% add the time stamp from the exif meta data in the right lower corner
for f in tqdm(files, desc="Convert"):
    processes.add(Popen(['convert', f, '-pointsize', '72', '-annotate', '+3100+2900',
                         '%[exif:DateTimeOriginal] UTC', f]))
    if len(processes) >= max_processes:
        os.wait()
        processes.difference_update([p for p in processes if p.poll() is not None])


print(f"Done with all files in {path}")
