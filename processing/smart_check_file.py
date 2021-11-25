#!/usr/bin/env python
"""Check if the raw files are good to read
author: Johannes Roettenbacher
"""
if __name__ == "__main__":
    import pylim.helpers as h
    from pylim import reader
    import os

    flight = "Flight_20210625a"
    raw_path = h.get_path("raw", flight)
    inpath = f"{raw_path}/{flight}"

    for file in os.listdir(inpath):
        try:
            df = reader.read_smart_raw(inpath, file)
        except:
            os.remove(f"{inpath}/{file}")
            print(f"Deleted {file}")
