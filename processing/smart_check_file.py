#!/usr/bin/env python
"""Check if the raw files are good to read
author: Johannes Roettenbacher
"""

import os
import smart

flight = "flight_01"
raw_path = smart.get_path("raw")
inpath = f"{raw_path}/{flight}"

for file in os.listdir(inpath):
    try:
        df = smart.read_smart_raw(inpath, file)
    except:
        os.remove(f"{inpath}/{file}")
        print(f"Deleted {file}")
