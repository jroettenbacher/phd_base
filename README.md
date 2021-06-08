# PhD Base Project
*author: Johannes Röttenbacher*

Here I code all the stuff I need for my PhD.

## 1. SMART

These are functions in relation with the SMART instrument.
"_" denotes internal functions which are called inside other functions.
They are grouped together in `smart.py`.
There is also a lookup dictionary to look up which spectrometer belongs to which inlet, which inlet is measuring up or 
downward radiance and which filename part belongs to which channel.
The functions are explained in their docstring and they can be tested using the main frame.
You can:
* read in raw and processed SMART data
* read in the pixel to wavelength calibration file for the given spectrometer 
* find the closest pixel and wavelength to any given wavelength for the given wavelength calibration file
* get information (date, measured property, channel) from the filename
* set the paths according to the config.toml file
* get the dark current for a specified measurement file with either option 1 or 2 and optionally plot it
* correct the raw measurement by the dark current
* plot the mean corrected measurement
* plot smart data either for one wavelength over time or for a range of or all wavelengths

## 2. merge_minutely_files.py
Script to merge minutely ASP07 files into one file per channel and folder.

## 3. write_corrected_smart_file.py
Script to correct a directory of raw smart measurements for the dark current.
Set the input and output paths in `config.toml`.
### FAQ
* What do negative measurements in the dark current mean?
