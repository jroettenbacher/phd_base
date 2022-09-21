# PhD Base Project

*author: Johannes Röttenbacher*

Here I code all the stuff I need for my PhD on Arctic cirrus. 
It includes processing for measurement data gathered by the HALO aircraft.
During the first year a new python package was created which provides useful functions used in the processing and analysis of the data.
The package is called `pylim` and can be found in `src`.
Further there are folders for different purposes:

* `analysis`: case studies or general campaign analysis to answer scientific questions
* `processing`: scripts to generate error/bias corrected, calibrated and shareable/publishable files from raw files
* `quicklooks`: quicklooks of measurements (raw and calibrated) and of calibrations

The data is sorted by flight.
The exact paths are defined in `config.toml` and can be adjusted according to your setup.
More information on how to adjust this file can be found in [Setup](./docs/setup.rst).

Documentation can be found [here](https://jroettenbacher.github.io/phd_base/).


### FAQ

* What do negative measurements in the SMART dark current mean?

Answer: The conversion of the analog signal to a digital can lead to this.

## 2. BACARDI

BACARDI is a broadband radiometer mounted on the bottom and top of HALO. 
The data is initially processed by DLR and then
Anna Luebke used the scripts provided by André Ehrlich and written by Kevin Wolf to process the
data further.
During the processing libRadtran simulations of cloud free conditions are done along the flight track of HALO.
For details of the BACARDI post processing see [2.3 BACARDI processing](#23-bacardi-processing).

**Workflow**

* download and process the radiosonde data (needs to be done once for each station)
* run [libRadtran simulation](#22-libradtran-simulation) for whole flight
* run [BACARDI processing](#23-bacardi-processing)

### 2.1 Radiosonde data

In order to simulate the clear sky broadband irradiance along the flight path and calculate the direct and diffuse
fraction radiosonde data is used. 
The data is downloaded from the [University Wyoming website](http://weather.uwyo.edu/upperair/sounding.html) by copying
the HTML site into a text file.
Data can only be downloaded in monthly chunks.
Then an IDL script from Kevin Wolf is used to extract the necessary data for libRadTran. 
It can be found here: `/projekt_agmwend/data/Cirrus_HL/00_Tools/02_Soundings/00_prepare_radiosonde_jr.pro`

#### 00_prepare_radiosonde_jr.pro

**TODO:**
* Check what radiosonde input is necessary for libRadtran. Simple interpolation yields negative relative humidity at lowest levels. (case: 21.7.21)

**Required User Input:**

* station name and number (select station closest to flight path)
* quicklook flag
* month

**Input:**

* radiosonde file

**Output:**

* daily radiosonde file (12UTC and 00UTC) for libRadtran

Run like this:
```shell
# cd into script folder
cd /projekt_agmwend/data/Cirrus_HL/00_Tools/02_soundings
# start idl
idl
# run script
idl> .r 00_prepare_radiosonde_jr
```

### 2.2 libRadtran simulation

Run libRadtran simulation for solar and terrestrial wavelengths along flight track with specified radiosonde data as 
input.
This is then used in the BACARDI processing.
See libRadtran section for details.

### 2.3 BACARDI processing

For more details see the processing script. From the processing script:

#### Solar downward

1. smooth sensor temperature sensor for electronic noise to avoid implications in temperature dependent corrections. - running mean dt=100 sec
2. correct thermophile signal with Temperature dependence of sensor sensitivity (Kipp&Zonen calibration)
3. correct thermal offset due to fast changing temperatures (DLR paramterization using the derivate of the sensor temperature)
4. apply inertness correction of CMP22 sensors (tau_pyrano=1.20, fcut_pyrano=0.6, rm_length_pyrano=0.5)
5. attitude correction (roll_offset=+0.3, pitch_offset=+2.55)

#### Solar upward

1. smooth sensor temperature sensor for electronic noise to avoid implications in temperature dependent corrections. - running mean dt=100 sec
2. correct thermophile signal with Temperature dependence of sensor sensitivity (Kipp&Zonen calibration)
3. correct thermal offset due to fast changing temperatures (DLR paramterization using the derivate of the sensor temperature)
4. apply inertness correction of CMP22 sensors (tau_pyrano=1.20, fcut_pyrano=0.6, rm_length_pyrano=0.5)

#### terrestrial downward

1. smooth sensor temperature sensor for electronic noise to avoid implications in temperature dependent corrections. - running mean dt=100 sec
2. correct thermophile signal with Temperature dependence of sensor sensitivity (Kipp&Zonen calibration)
3. correct thermal offset due to fast changing temperatures (DLR paramterization using the derivate of the sensor temperature)
4. apply inertness correction of CGR4 sensors (tau_pyrano=2.00, fcut_pyrano=0.5, rm_length_pyrano=2.0)

#### terretrial upward

1. smooth sensor temperature sensor for electronic noise to avoid implications in temperature dependent corrections. - running mean dt=100 sec
2. correct thermophile signal with Temperature dependence of sensor sensitivity (Kipp&Zonen calibration)
3. correct thermal offset due to fast changing temperatures (DLR paramterization using the derivate of the sensor temperature)
4. apply inertness correction of CGR4 sensors (tau_pyrano=2.00, fcut_pyrano=0.5, rm_length_pyrano=2.0)

#### 00_process_bacardi_V20210928.pro

**Required User Input:**

* Flight date
* Flight number (Fxx)

**Input:**

* BACARDI quicklook data from DLR (e.g. `QL-CIRRUS-HL_F15_20210715_ADLR_BACARDI_v1.nc`)
* simulated broadband downward irradiance from libRadtran
* direct diffuse fraction from libRadtran

**Output:**

* corrected BACARDI measurement and libRadtran simulations in netCDF file

Run like this:

```shell
# cd into script folder
cd /projekt_agmwend/data/Cirrus_HL/00_Tools/01_BACARDI/
# start IDL
idl
# run script
idl> .r 00_process_bacardi_V20210903.pro
```

## 3. BAHAMAS

These scripts work with the BAHAMAS system from HALO.
BAHAMAS gives in situ and flight data like altitude, temperature, wind speed and other parameters.

## 4. GoPro Time Lapse quicklooks

During the flight a GoPro was attached to the second window on the left side of HALO.
Using the time-lapse function a picture was taken every 5 seconds.
Together with BAHAMAS position data (and SMART spectra measurements) a time-lapse video is created.
The GoPro was set to UTC time but cannot be synchronized to BAHAMAS.
At one point it reset its internal time to local time, so the metadata for some flights had to be corrected.
See the `README.md` in the GoPro Folder for details.
A list which tracks the processing status can be found there.

Due to the offset from the BAHAMAS time an offset correction has to be applied to each timestamp.

### 4.1 add_timestamp_to_picture.py

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
Run on Linux.

### 4.2 write_gopro_timestamps.py

**Input:** 
* flight date (User input)
* GoPro images

**Output:**
* txt file with `exiftool` output
* csv file with datetime from picture meta data and picture number

Reads the metadata time stamps and saves them together with the picture number in a csv file.
Run on Linux.

### 4.3 plot_maps.py

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

### 4.4 add_map_to_picture.py

**Input:**
* date of flight (User input)
* flight number (User input)
* maps
* map numbers from csv file

**Output:**
* new GoPro picture with map in the upper right corner and timestamp in the lower right corner

Adds the BAHAMAS plot onto the GoPro picture but only for pictures, which were taken in flight, according to the csv file
from `plot_maps.py`. Saves those pictures in a new folder: `Flight_{yyyymmdda/b}`.
Run on Linux.

### 4.5 make_video_from_pictures.sh

**Input:**
* date of flight (User input)
* number of flight (User input)
* framerate [12, 24] (User input)
* start_number, number in filename of first picture in folder (User input)
* GoPro pictures with map and timestamp

**Output:**
* video (slow or fast) of flight from GoPro pictures

Uses ffmpeg to create a stop motion video of the GoPro pictures.
Run on Linux.

Run like this:

```shell
ffmpeg 
```

## 5. libRadtran

[libRadtran](https://doi.org/10.5194/gmd-9-1647-2016) is a radiative transfer model which can model spectral radiative 
fluxes.

### 5.0 libRadtran simulations along flight path

The following scripts use the BAHAMAS data to create libRadtran input files to simulate fluxes along the flightpath.
The two scripts are meant to allow for flexible settings of the simulation.
BACARDI versions of these scripts are available which replace the old IDL scripts.
They are to be used as part of the BACARDI processing. 
Before publishing BACARDI data, the state of the libRadtran input settings should be saved!

**TODO:**
- [ ] use specific total column ozone concentrations from OMI
  * can be downloaded [here](https://disc.gsfc.nasa.gov/datasets/OMTO3G_003/summary?keywords=aura)
  * you need an Earth Data account and [add the application to your profile](https://disc.gsfc.nasa.gov/earthdata-login)
  * checkout the instructions for command line download [here](https://disc.gsfc.nasa.gov/data-access#windows_wget)
- [x] change atmosphere file according to location -> uvspec does this automatically when lat, lon and time are supplied 
  - [x] CIRRUS-HL: use midlatitude summer (afglms.dat) or subarctic summer (afglss.dat)
- [x] use ocean or land albedo according to land sea mask
- [x] include solar zenith angle filter
- [ ] use altitude (ground height above sea level) from a surface map, when over land -> adjust zout definition accordingly
- [ ] use self-made surface_type_map for simulations in the Arctic
- [ ] use sur_temperature for thermal infrared calculations (input from VELOX)
- BACARDI
- [ ] use surface_type_map for BACARDI simulations
- [ ] use surface temperature according to ERA5 reanalysis for BACARDI simulations

#### libradtran_write_input_file.py

Here one can set all options needed in the libRadtran input file.
Given the flight and time step, one input file will then be created for every time step with the fitting lat and lon values
along the flight track.

**Required User Input:**

* flight (e.g. 'Flight_202170715a')
* time_step (e.g. 'minutes=2')

#### libradtran_run_uvspec.py

Given the flight and the path to `uvspec`, this script calls `uvspec` for each input file and writes a log and output file.
It does so in parallel, checking how many CPUs are available.
After that the output files are merged into one data frame and information from the input file is added to write one netCDF file.

**Required User Input:**

* flight (e.g. 'Flight_20210715a')
* path to uvspec.exe
* name of netCDF file (optional)

### 5.1 BACARDI processing

The following two scripts ~~are needed~~ were used in order to prepare the BACARDI processing.
They are superseded by the new python versions of these scripts.
* [libradtran_write_input_file_bacardi.py](#libradtran_write_input_file_bacardi.py)
* [libradtran_run_uvspec_bacardi.py](#libradtran_run_uvspec_bacardi.py)

#### 01_dirdiff_BBR_Cirrus_HL_Server_jr.pro

**Current settings:**
* Albedo from Taylor et al. 1996
* atmosphere file: afglt.dat -> tropical atmosphere

**Required User Input:**

* Flight date
* sonde date (mmdd)
* sounding station (stationname_stationnumber)
* time interval for modelling (time_step)

Run like this:

```shell
# cd into script folder
cd /projekt_agmwend/data/Cirrus_HL/00_Tools/01_BACARDI/
# start IDL
idl
# start logging to a file
idl> journal, 'filename.log'
# run script
idl> .r 01_dirdiff_BBR_Cirrus_HL_Server_jr.pro
# stop logging
idl> journal
```

#### 03_dirdiff_BBR_Cirrus_HL_Server_ter.pro

**Current settings:**
* Albedo from Taylor et al. 1996

**Required User Input:**

* Flight date
* sonde date (mmdd)
* sounding station (stationname_stationnumber)

Run like this:

```shell
# cd into script folder
cd /projekt_agmwend/data/Cirrus_HL/00_Tools/01_BACARDI/
# start IDL
idl
# start logging to a file
idl> journal, 'filename.log'
# run script
idl> .r 03_dirdiff_BBR_Cirrus_HL_Server_ter.pro
# stop logging
idl> journal
```

#### libradtran_write_input_file_bacardi.py

This will write all input files for libRadtran simulations along the flight path.

**Required User Input:**
* flights in a list (optional, if not manually specified all flights will be processed)
* time_step, in what interval should the simulations be done along the flight path?
* solar_flag, write file for simulation of solar or thermal infrared fluxes?

The idea for this script is to generate a dictionary with all options that should be set in the input file.
New options can be manually added to the dictionary. 
The options are linked to the page in the manual where they are described in more detail.
The input files are then saved to a working directory where `libradtran_run_uvspec_bacardi.py` will find them 
and read them in.

Run like this:

```shell
python libradtran_write_input_file_bacardi.py
```

#### libradtran_run_uvspec_bacardi.py

This will run libRadtran simulations with the given input files from `libradtran_write_input_file_bacardi.py`.

**Required User Input:**
* flights in a list (optional, if not manually specified all flights will be processed)
* solar_flag, run simulation for solar or for thermal infrared wavelengths?

The script will loop through all files and start simulations for all input files it finds (in parallel for one flight).
If the script is called via the command line it creates a `log` folder in the working directory if necessary and saves
a log file to that folder.
It also displays a progress bar in the terminal.
After it is done with one flight, it will collect information from the input files and merge all output files in a 
dataframe to which it appends the information from the input files.
It converts everything to a netCDF file and writes it to disc with a bunch of metadata included.

Run like this:

```shell
python libradtran_run_uvspec_bacardi.py
```