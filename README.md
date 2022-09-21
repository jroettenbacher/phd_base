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

