#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 19.07.2024

Convert all NASA ames files from the cloud combination probe (CCP) to netCDF using the nappy command line tool.
https://github.com/cedadev/nappy

Run on Linux!
"""
import os
import subprocess
from datetime import datetime


def convert_ames_to_netcdf(input_folder):
    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".ames"):
            # Construct full file paths
            input_filepath = os.path.join(input_folder, filename)

            # Extract the date from the filename
            try:
                parts = filename.split('_')
                date_str = parts[2]
                date_obj = datetime.strptime(date_str, "%Y%m%d")
                time_arg = f"seconds since {date_obj.strftime('%Y-%m-%d')} 00:00:00"
            except (IndexError, ValueError):
                print(f"Failed to extract date from filename: {filename}")
                continue

            # Construct the command
            command = [
                'na2nc',
                '-i', input_filepath,
                '-t', time_arg
            ]

            # Run the command
            try:
                subprocess.run(command, check=True)
                print(f"Successfully converted {filename} to NetCDF")
            except subprocess.CalledProcessError as e:
                print(f"Error converting {filename}: {e}")


if __name__ == "__main__":
    input_folder = "/mnt/e/CIRRUS-HL/01_Flights/all/CCP"  # Update this path to your input folder
    convert_ames_to_netcdf(input_folder)
