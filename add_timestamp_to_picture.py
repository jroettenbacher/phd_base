#!\usr\bin\env python
"""Add the timstamp to a picture (run on Ubuntu)
author: Johannes Röttenbacher
"""

import os
from tqdm import tqdm

path = "/mnt/c/Users/Johannes/Documents/Gopro/20210710"
files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".JPG")]

for f in tqdm(files):
    os.system(f"convert {f} -pointsize 72 -fill black -annotate +3200+2900  %[exif:DateTimeOriginal] {f}")
