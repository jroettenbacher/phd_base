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

SMART data is organized by flight. 
Each flight folder has one `SMART` folder with the following subfolders:

* `data_calibrated`: dark current corrected and calibrated measurement files
* `data_cor`: dark current corrected measurement files
* `raw`: raw measurement files

  In the calibration folder each calibration is saved in its own folder.
  Each calibration is used to generate one calibration file, which is corrected for dark current.

  A few more folders needed are:

* `raw_only`: raw measurement files as written by ASP06/07, do not work on those files, but copy them into `raw`
* `lamp_F1587`: calibration lamp file
* `panel_34816`: reflectance panel file for each spectrometer
* `pixel_wl`: pixel to wavelength files for each spectrometer
* `plots`: plots and quicklooks

**Workflow**

There are two workflows:
1. Calibration files
2. Measurement files

Both workflows start with the correction of the dark current. 
After the raw files are copied from ASP06/07 into `raw_only` and `raw` the minutely files are corrected for the dark
current and saved with the new ending `*_cor.dat` in `data_cor`.
Then the minutely files are merged to one file per folder and channel.

**Calibration files**

Use `smart_process_transfer_calib.py` or `smart_process_lab_calib.py` to correct the calibration files for the 
dark current and merge the minutely files.
Then run `smart_calib_lab_ASP06/07.py` for the lab calibrations or `smart_calib_transfer.py` for the transfer 
calibration.
Each script returns a file in the `calib` folder with the calibration factor.

**Measurement files**

Use `smart_write_dark_currented_corrected_file.py` to correct one flight for the dark current.
Merge the resulting minutely files with `smart_merge_minutely_files.py`.
Finally calibrate the measurement with `smart_calibrate_measurment.py`.
The resulting calibrated files are saved in the `data_calibrated` folder.


### 1.1 smart.py

These are functions in relation with the SMART instrument.
"_" denotes internal functions which are called inside other functions. There is also a lookup dictionary to look up
which spectrometer belongs to which inlet, which inlet is measuring up or downward irradiance and which filename part
belongs to which channel. Another dictionary relates each measurement with the date of the transfer calibration for that
measurement. The functions are explained in their docstring and they can be tested using the main frame.
You can:

* read in raw and processed SMART data
* read in the pixel to wavelength calibration file for the given spectrometer
* read in the lamp standard file
* find the closest pixel and wavelength to any given wavelength for the given wavelength calibration file
* get information (date, measured property, channel) from the filename
* get the path to a specified key defined in `config.toml`
* get the dark current for a specified measurement file with either option 1 or 2 and optionally plot it
* correct the raw measurement by the dark current
* plot the mean corrected measurement
* plot smart data either for one wavelength over time or for a range of or all wavelengths
* use the holoviews functions to create a dynamic map for interactive quicklooks in a jupyter notebook

### 1.2 smart_process_lab_calib.py

**TODO:** make it work for 2021-03-19/29

Script to correct the lab calibration files for the dark current and merge the minutely corrected files into one file.

**Required User Input:**
* calibration folder

### 1.3 smart_process_transfer_calib.py

Script to correct SMART transfer calibration measurement for dark current and save it to a new file and merge 
the minutely files.

**Required User Input:**
* calibration folder


### 1.4 smart_write_dark_current_corrected_file.py

Script to correct a directory of raw smart measurements for the dark current. Set the input and output paths
in `config.toml`.
Comment in the for loop to correct the calibration files.

**Required User Input:**

* flight folder in raw_path

### 1.5 smart_merge_minutely_files.py

Script to merge minutely dark current corrected measurement files into one file per channel and folder.
Deletes minutely files.

**Required User Input:**

* directory where to find given folder
* folder which to loop through

### 1.6 smart_calib_lab_ASP06.py

Calculates the lab calibration factor `c_lab` (unit: W/m^2/count).
Creates a lab calibration file with the irradiance measurements from the lamp and the calibrated Ulli transfer measurements. 
Needs to be run once for each spectrometer.

**Required User Input:**

* channel which to run (SWIR or VNIR)
* folder pair (spectrometer pair) (0 or 1)
* base directory of lab calibration -> should be found in calib folder
* whether to normalize the measurement by the integration time or not

### 1.7 smart_calib_lab_ASP07.py

Calculates the lab calibration factor `c_lab` (unit: W/sr/m^2/count).
Creates a lab calibration file with the radiance measurements from the reflectance panel and the calibrated Ulli transfer measurements. 
Needs to be run once for each channel (SWIR and VNIR).

**Required User Input:**

* channel which to run (SWIR or VNIR)
* base directory of lab calibration -> should be found in calib folder
* whether to normalize the measurement by the integration time or not

### 1.8 smart_calib_transfer.py

Calculates the field calibration factor `c_field`.
Creates a transfer calibration file with the radiance/irradiance measurements from the lab and the calibrated measurements from the field.

**Required User Input:**
* transfer calib folder -> should be found in `calib` folder
* integration time of transfer calibration (T_int) in ms
* whether to normalize the measurement by the integration time or not

### 1.9 smart_calibrate_measurment.py

Reads in dark current corrected measurement file and corresponding transfer calibration to calibrate measurement files.

**Required User Input:**

* flight folder found in `data_cor`
* integration time of ASP06 and ASP07 measurements (check raw measurement files to get integration times)
* whether to normalize measurements or not (use normalized calibration factor, necessary if no calibration with the same integration time was made)
* give date of transfer calib to use for calibrating measurement if not same as measurement date

### FAQ

* What do negative measurements in the dark current mean?

Answer: The conversion of the analog signal to a digital can lead to this.

## 2. BACARDI

BACARDI is a broadband radiometer mounted on the bottom and the top of HALO.
The data is initially processed by DLR and then Anna Luebke used the scripts provided by André Ehrlich to process the
data further.

### 2.1 Radiosonde data

In order to simulate the clear sky broadband irradiance along the flight path and calculate the direct and diffuse
fraction radiosonde data is used. 
The data is downloaded from the [University Wyoming website](http://weather.uwyo.edu/upperair/sounding.html) by copying
the HTML site into a text file.
Then an IDL script from Kevin Wolf is used to extract the necessary data for libRadTran. 
It can be found here: `/projekt_agmwend/data/Cirrus_HL/00_Tools/02_Soundings/00_prepare_radiosonde_jr.pro`

## 3. BAHAMAS

These scripts work with the BAHAMAS system from HALO.
BAHAMAS gives in situ and flight data like altitude, temperature, wind speed and other parameters.

## 4. GoPro Time Lapse quicklooks

During the flight a GoPro was attached to the second window on the left side of HALO.
Using the time lapse function a picture was taken every 5 seconds.
Together with BAHAMAS position data (and SMART spectra measurements) a time lapse video is created.
The GoPro was set to UTC time but cannot be synchronized to BAHAMAS.
At one point it reset its internal time to local time, so the meta data for some flights had to be corrected.
See the `README.md` in the GoPro Folder for details.
There a list which tracks the processing status can be found.

Due to the offset from the BAHAMAS time an offset correction has to be applied to each timestamp.

### 3.1 add_timestamp_to_picture.py

Run on Linux (Ubuntu)

**Input:**
* flight (User input)
* correct_time flag (User input)
* filename for a test file (User input)
* path with all GoPro pictures

**Output:**
* overwrites meta data in original file with UTC time from BAHAMAS
* adds a time stamp to the right bottom of the original file

This script reads out the DateTimeOriginal meta data tag of each file and corrects it for the LT to UTC and BAHAMAS offset if necessary.
It overwrites the original meta data tag and places a time stamp to the right bottom of the file.
One can test the time correction by replacing `path` with `file` in `run()` (\~line 38).

### 3.2 write_gopro_timestamps.py

Run on Linux (Ubuntu)

**Input:** 
* flight date (User input)
* GoPro images

**Output:**
* txt file with `exiftool` output
* csv file with datetime from picture meta data and picture number

Reads the metadata time stamps and saves them together with the picture number in a csv file.

### 3.3 plot_maps.py

**Input:**
* date of flight (User input)
* flight number (User input)
* csv file with time stamp and GoPro picture number
* BAHAMAS nc file

**Output:**
* csv file with selected GoPro picture numbers and timestamps which are used for the time lapse video
* map for each GoPro picture

Reads in the BAHAMAS latitude and longitude data and selects only the time steps which correspond with a GoPro picture.
In the `plot_props` dictionary the map layout properties for each flight are defined.
For testing the first four lines of the last cell can be uncommented and the Parallel call can be commented.
It makes sense to run this script on the server to utilize more cores and increase processing speed.

### 3.4 