#!/usr/bin/env python
"""Given a SMART input file write a well documented netCDF file.
Two options:

    1. One spectrometer = one file
    2. One flight = one file

Option 2 might result in a huge file but would be easier to distribute.
With option 1 one could still merge all single files quite easily with xarray.
Go with option 1 for now.

The netCDF file could be writen as a standard output from smart_calibrate_measurement.py or as a separate step in this
 script. Start with this script and write a function that can then be used in the calibration script.
author: Johannes Röttenbacher
"""
if __name__ == "__main__":
    import pylim.helpers as h
    from pylim import reader

    flight = "Flight_20210624a"