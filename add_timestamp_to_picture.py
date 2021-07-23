#!\usr\bin\env python
"""Add the timstamp to a picture (run on Ubuntu)
corrects the meta data timestamp in an image for a
author: Johannes Röttenbacher
"""

# %%
import os
import datetime
from tqdm import tqdm
from subprocess import run, Popen
from smart import gopro_lt, gopro_offsets

flight = "Flight_20210625"
# file = "/mnt/c/Users/Johannes/Documents/Gopro/20210625_Gopro_0001.JPG"
# path = "/mnt/e/CIRRUS-HL/Gopro/{flight[7:]}"
path = f"/mnt/c/Users/Johannes/Documents/Gopro/{flight[7:]}"
correct_time = True
sync_to_bahamas = True if flight in gopro_offsets else False
LT_to_UTC = gopro_lt[flight] if flight in gopro_lt else False
start_file = 0
files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".JPG")][start_file:]
processes = set()
max_processes = 10

# %% update meta data time stamp to set to UTC
if correct_time:
    # GoPro switched to LT on 28.06 due to WiFi connection
    utc_correction = 2 if LT_to_UTC else 0  # convert local time to UTC
    # GoPro is not synched to BAHAMAS
    bahamas_correction = gopro_offsets[flight] if sync_to_bahamas else 0
    correction = utc_correction + bahamas_correction / 60 / 60  # convert seconds to hours
    # format correction
    delta = datetime.timedelta(hours=abs(correction))
    sign = "-" if correction > 0 else "+"
    cor_str = str(delta)
    run(['exiftool', '-m', '-progress', '-overwrite_original', f'-DateTimeOriginal{sign}={cor_str}', path])

# %% add the time stamp from the exif meta data in the right lower corner
for f in tqdm(files, desc="Convert"):
    processes.add(Popen(['convert', f, '-pointsize', '72', '-annotate', '+3100+2900',
                         '%[exif:DateTimeOriginal] UTC', f]))
    if len(processes) >= max_processes:
        os.wait()
        processes.difference_update([p for p in processes if p.poll() is not None])


print(f"Done with all files in {path}")
