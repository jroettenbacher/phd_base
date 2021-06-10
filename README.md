# PhD Base Project
*author: Johannes Röttenbacher*

Here I code all the stuff I need for my PhD.

## 1. SMART

These scripts work with the SMART calibration and measurement files to generate calibrated measurement files and
quicklooks.
General functions are in `smart.py` and are used in other processing scripts.
In `config.toml` one can find the paths where the scripts expect to find files and where they will save the files to.
As a first step the minutely files from ASP07 are merged to one file per folder and channel.
Then the files are corrected for the dark current and saved with the new ending `*_cor.dat`.
The usual workflow is then to start with the laboratory calibrations and calculate the laboratory calibration factor 
`c_lab` of each spectrometer. 
This is saved to a file in the calib folder.
Then the transfer calibrations are related to the laboratory calibration and another file is saved to calib.
Finally, the measurement files are corrected for the dark current and calibrated with the transfer calibration.

### 1.1 smart.py
These are functions in relation with the SMART instrument.
"_" denotes internal functions which are called inside other functions.
There is also a lookup dictionary to look up which spectrometer belongs to which inlet, which inlet is measuring up or 
downward irradiance and which filename part belongs to which channel.
The functions are explained in their docstring and they can be tested using the main frame.
You can:
* read in raw and processed SMART data
* read in the pixel to wavelength calibration file for the given spectrometer
* read in the lamp standard file  
* find the closest pixel and wavelength to any given wavelength for the given wavelength calibration file
* get information (date, measured property, channel) from the filename
* set the paths according to the config.toml file 
* get the path to a specified folder
* get the dark current for a specified measurement file with either option 1 or 2 and optionally plot it
* correct the raw measurement by the dark current
* plot the mean corrected measurement
* plot smart data either for one wavelength over time or for a range of or all wavelengths

### 1.2 smart_merge_minutely_files.py
Script to merge minutely ASP07 files into one file per channel and folder.

**Required User Input:** 
* directory where to find given folder
* folder which to loop through

### 1.3 smart_write_dark_current_corrected_file.py
Script to correct a directory of raw smart measurements for the dark current.
Set the input and output paths in `config.toml`.

**Required User Input:**
* flight folder in raw_path

### 1.4 smart_calib_lab_ASP06
Creates a lab calibration file with the lamp measurements and the calibrated Ulli transfer measurements.
Needs to be run once for each channel (SWIR and VNIR).

**Required User Input:**
* channel which to run (SWIR or VNIR)
* base directory of lab calibration -> should be found in calib folder

### FAQ
* What do negative measurements in the dark current mean?
