# PhD Base Project

*author: Johannes Röttenbacher*

Here I code all the stuff I need for my PhD.

## 1. SMART

These scripts work with the SMART calibration and measurement files to generate calibrated measurement files and
quicklooks. 
General functions are in `smart.py` and are used in other processing scripts. 
In `config.toml` one can find
the paths where the scripts expect to find files and where they will save the files to.

**Folder Structure**

All SMART files should be in one folder with the following subfolders:

* `calib`: raw calibration measurements in subfolders and processed calibration files on the top level.
* `data_calibrated`: dark current corrected and calibrated measurement files in flight folders
* `data_cor`: dark current corrected measurement files in flight folders
* `lamp_F1587`: calibration lamp file
* `panel_34816`: reflectance panel file for each spectrometer
* `pixel_wl`: pixel to wavelength files for each spectrometer
* `plots`: plots and quicklooks
* `raw`: merged raw measurement files in flight folders
* `raw_only`: raw measurement files as written by ASP06/07, do not work on those files, but copy them into `raw`

**Workflow**

After the raw files are copied from ASP06/07 into `raw_only` and `raw` the minutely files are corrected for the dark 
current and saved with the new ending `*_cor.dat` in `data_cor`.
Then the minutely files are merged to one file per folder and channel.
The usual workflow is then to start with the laboratory calibrations, correct them for the dark current and calculate 
the laboratory calibration factor `c_lab` of each spectrometer.
This is saved to a file in the `calib` folder and only needs to be done once for each spectrometer.
Then the transfer calibrations are corrected for the dark current and related to the laboratory calibration and another file is saved to `calib`. 
Finally, the measurement files are corrected for the dark current and calibrated with the transfer calibration.

### 1.1 smart.py

These are functions in relation with the SMART instrument.
"_" denotes internal functions which are called inside other functions. There is also a lookup dictionary to look up
which spectrometer belongs to which inlet, which inlet is measuring up or downward irradiance and which filename part
belongs to which channel. The functions are explained in their docstring and they can be tested using the main frame.
You can:

* read in raw and processed SMART data
* read in the pixel to wavelength calibration file for the given spectrometer
* read in the lamp standard file
* find the closest pixel and wavelength to any given wavelength for the given wavelength calibration file
* get information (date, measured property, channel) from the filename
* set the paths according to the `config.toml` file
* get the path to a specified key defined in `config.toml`
* get the dark current for a specified measurement file with either option 1 or 2 and optionally plot it
* correct the raw measurement by the dark current
* plot the mean corrected measurement
* plot smart data either for one wavelength over time or for a range of or all wavelengths


### 1.2 smart_write_dark_current_corrected_file.py

Script to correct a directory of raw smart measurements for the dark current. Set the input and output paths
in `config.toml`.
Comment in the for loop to correct the calibration files.

**Required User Input:**

* flight folder in raw_path

### 1.3 smart_merge_minutely_files.py

Script to merge minutely dark current corrected measurement files into one file per channel and folder.
Deletes minutely files.

**Required User Input:**

* directory where to find given folder
* folder which to loop through

### 1.4 smart_calib_lab_ASP06

Calculates the lab calibration factor `c_lab` (unit: W/m^2/count).
Creates a lab calibration file with the irradiance measurements from the lamp and the calibrated Ulli transfer measurements. 
Needs to be run once for each spectrometer.

**Required User Input:**

* channel which to run (SWIR or VNIR)
* folder pair (spectrometer pair) (0 or 1)
* base directory of lab calibration -> should be found in calib folder
* whether to normalize the measurement by the integration time or not

### 1.5 smart_calib_lab_ASP07

Calculates the lab calibration factor `c_lab` (unit: W/sr/m^2/count).
Creates a lab calibration file with the radiance measurements from the reflectance panel and the calibrated Ulli transfer measurements. 
Needs to be run once for each channel (SWIR and VNIR).

**Required User Input:**

* channel which to run (SWIR or VNIR)
* base directory of lab calibration -> should be found in calib folder
* whether to normalize the measurement by the integration time or not

### 1.6 smart_calib_transfer

Calculates the field calibration factor `c_field`.
Creates a transfer calibration file with the radiance/irradiance measurements from the lab and the calibrated measurements from the field.

**Required User Input:**
* transfer calib folder -> should be found in `calib` folder
* integration time of transfer calibration (T_int) in ms
* whether to normalize the measurement by the integration time or not

### 1.7 smart_calibrate_measurment

Reads in dark current corrected measurement file and corresponding transfer calibration to calibrate measurement files.

**Required User Input:**

* flight folder found in `data_cor`
* integration time of ASP06 and ASP07 measurements (check raw measurement files to get integration times)
* whether to normalize measurements or not (use normalized calibration factor, necessary if no calibration with the same integration time was made)
* give date of transfer calib to use for calibrating measurement if not same as measurement date

### FAQ

* What do negative measurements in the dark current mean?

Answer: The conversion of the analog signal to a digital can lead to this.
