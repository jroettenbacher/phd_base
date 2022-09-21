#!/usr/bin/env python
"""Copy all files from the GoPro SD Card into one folder

**Required User Input:**

* date
* src, source directory
* dst_dir, destination directory

*author*: Johannes Roettenbacher
"""
if __name__ == "__main__":
    import pylim.helpers as h
    import os
    import shutil
    from tqdm import tqdm

    date = "20220412"
    src = "D:/DCIM"
    dst_dir = f"F:/HALO-AC3_raw_only/04_GoPro/{date}"
    h.make_dir(dst_dir)
    for src_dir, names, files in os.walk(src):
        for file in tqdm(files, desc=src_dir):
            src_file = os.path.join(src_dir, file)
            dst_file = os.path.join(dst_dir, file)
            shutil.copy(src_file, dst_dir)
