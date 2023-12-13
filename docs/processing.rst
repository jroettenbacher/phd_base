**********
Processing
**********

With data processing one usually refers to the process of converting the raw measurement files as delivered by the instrument PC into error/bias corrected, calibrated and distributable/publishable files.
For this work the netCDF standard is defined as the preferred output format.

Each instrument has different requirements before, after and during the campaign.
In general these scripts are **not** meant for analysing the data to answer specific science questions or for producing quicklooks.
This is handled by the scripts in the folders :ref:`analysis:Analysis` and :ref:`quicklooks:Quicklooks`.

SMART
=====

SMART has to be calibrated in the lab and in the field so there are processing routines for both cases.
Some scripts are designated for postprocessing specific campaign data as sometimes the normal processing does not cover all possible cases which occur during a campaign.
In general scripts can be divided into :ref:`processing:Campaign Scripts` and :ref:`processing:Postprocessing Scripts`.
The campaign scripts work under the assumption that everything worked as intended and are used in the field to quickly genrate calibrated data for quicklooks.
The postprocessing scripts are campaign specific and try to correct for all eventualities and problems that occurred during the campaign.

**Folder Structure**

SMART data is organized by flight.
Each flight folder has one ``SMART`` folder with the following subfolders:

* ``data_calibrated``: dark current corrected and calibrated measurement files
* ``data_cor``: dark current corrected measurement files
* ``raw``: raw measurement files

Each campaign also has a calibration folder.
In the calibration folder each calibration is saved in its own folder.
Each calibration is used to generate one calibration file, which is corrected for dark current and saved in the top level calibration folder.

A few more folders needed are:

* ``raw_only``: raw measurement files as written by ASP06/07, do not work on those files, but copy them into ``raw``
* ``lamp_F1587``: calibration lamp file
* ``panel_34816``: reflectance panel file
* ``pixel_wl``: pixel to wavelength files for each spectrometer

**Workflow**

There are two workflows:

1. Calibration files
2. Measurement files

Both workflows start with the correction of the dark current.
After the raw files are copied from ASP06/07 into ``raw_only`` and ``raw`` the minutely files are corrected for the dark current and saved with the new ending ``*_cor.dat`` in ``data_cor``.
Then the minutely files are merged to one file per folder and channel.

**Calibration files**

Use :ref:`processing:smart_process_transfer_calib.py` or :ref:`processing:smart_process_lab_calib_cirrus_hl.py` to correct the calibration files for the dark current and merge the minutely files.
Then run :ref:`processing:smart_calib_lab_ASP06.py` or :ref:`processing:smart_calib_lab_ASP07.py` for the lab calibrations or :ref:`processing:smart_calib_transfer.py` for the transfer calibration.
Each script returns a file in the ``calib`` folder with the calibration factors.

**Measurement files**

Use :ref:`processing:smart_write_dark_current_corrected_file.py` to correct one flight for the dark current.
Merge the resulting minutely files with :ref:`processing:smart_merge_minutely_files.py`.
Finally, calibrate the measurement with :ref:`processing:smart_calibrate_measurement.py`.
The resulting calibrated files are saved in the ``data_calibrated`` folder.
As a final step the calibrated files can then be converted to netCDF with :ref:`processing:smart_write_ncfile.py`.

Campaign Scripts
-----------------

**Measurement files**

smart_write_dark_current_corrected_file.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.smart_write_dark_current_corrected_file

smart_merge_minutely_files.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.smart_merge_minutely_files

smart_calibrate_measurement.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.smart_calibrate_measurement

smart_write_ncfile.py
^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.smart_write_ncfile

**Calibration files**

smart_process_transfer_calib.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.smart_process_transfer_calib

smart_calib_transfer.py
^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.smart_calib_transfer

Postprocessing Scripts
-----------------------

smart_calib_lab_ASP06.py
^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.smart_calib_lab_ASP06

smart_calib_lab_ASP07.py
^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.smart_calib_lab_ASP07

smart_calib_lab_ASP06_halo_ac3.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.smart_calib_lab_ASP06_halo_ac3

smart_process_lab_calib_cirrus_hl.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.smart_process_lab_calib_cirrus_hl

smart_process_transfer_calib_cirrus_hl.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.smart_process_transfer_calib_cirrus_hl

smart_process_lab_calib_halo_ac3.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.smart_process_lab_calib_halo_ac3

Final calibration
-----------------

These scripts are used for the final calibration of the measurement data.
They are designed to take care of everything necessary given the correct input files.

cirrus_hl_smart_calibration.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.cirrus_hl_smart_calibration

halo_ac3_smart_calibration.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.halo_ac3_smart_calibration

BACARDI
=======

BACARDI is a broadband radiometer mounted on the bottom and top of HALO.
The data is initially processed by DLR and then Anna Luebke used the scripts provided by André Ehrlich and written by Kevin Wolf to process the data further.
During the processing libRadtran simulations of cloud free conditions are done along the flight track of HALO.
For details on the BACARDI post processing see :ref:`processing:BACARDI postprocessing`.

**Workflow**

* download and process the radiosonde data (needs to be done once for each station)
* run libRadtran simulation as explained in :ref:`processing:BACARDI processing` for the whole flight
* run :ref:`processing:BACARDI postprocessing`

Radiosonde data
---------------

In order to simulate the clear sky broadband irradiance along the flight path and calculate the direct and diffuse fraction radiosonde data is used as input for libRadtran.
The data is downloaded from the `University Wyoming website <http://weather.uwyo.edu/upperair/sounding.html>`_ by copying the HTML site into a text file.
Data can only be downloaded in monthly chunks.
Then an IDL script from Kevin Wolf is used to extract the necessary data for libRadTran.
It can be found here: ``/projekt_agmwend/data/Cirrus_HL/00_Tools/02_Soundings/00_prepare_radiosonde_jr.pro``

00_prepare_radiosonde_jr.pro
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**TODO:**

- [ ] Check what radiosonde input is necessary for libRadtran. Simple interpolation yields negative relative humidity at lowest levels. (case: 21.7.21)

**Required User Input:**

* station name and number (select station closest to flight path)
* quicklook flag
* month

**Input:**

* monthly radiosonde file

**Output:**

* daily radiosonde file (12UTC and 00UTC) for libRadtran

Run like this:

.. code-block:: shell

   # cd into script folder
   cd /projekt_agmwend/data/Cirrus_HL/00_Tools/02_soundings
   # start idl
   idl
   # run script
   idl> .r 00_prepare_radiosonde_jr

libRadtran simulation
---------------------

Run libRadtran simulation for solar and terrestrial wavelengths along flight track with specified radiosonde data as input.
This is then used in the BACARDI processing.
See the section on :ref:`processing:BACARDI processing` for details.

BACARDI postprocessing
----------------------

Some values are filtered out during the postprocessing.
We set an aircraft attitude limit in the processing routine, and if the attitude exceeds this threshold, then the data is filtered out.
For example, this would be the case during sharp turns.
The threshold also takes the attitude correction factor into account.
For CIRRUS-HL, if this factor is below 0.25, then we filter out data where the roll or pitch exceeds 8°.
If this factor is above 0.25, then we begin filtering at 5°.
For EUREC4A, the limits were not as strict because the SZAs were usually higher.
Since this is not the case for the Arctic, something stricter was needed.
For more details on other corrections see the :ref:`processing script <processing:00_process_bacardi_V20210928.pro>`. From the processing script:

Solar downward
^^^^^^^^^^^^^^

1. smooth sensor temperature sensor for electronic noise to avoid implications in temperature dependent corrections. - running mean dt=100 sec
2. correct thermophile signal with Temperature dependence of sensor sensitivity (Kipp&Zonen calibration)
3. correct thermal offset due to fast changing temperatures (DLR paramterization using the derivate of the sensor temperature)
4. apply inertness correction of CMP22 sensors (tau_pyrano=1.20, fcut_pyrano=0.6, rm_length_pyrano=0.5)
5. attitude correction (roll_offset=+0.3, pitch_offset=+2.55)

Solar upward
^^^^^^^^^^^^

1. smooth sensor temperature sensor for electronic noise to avoid implications in temperature dependent corrections. - running mean dt=100 sec
2. correct thermophile signal with Temperature dependence of sensor sensitivity (Kipp&Zonen calibration)
3. correct thermal offset due to fast changing temperatures (DLR paramterization using the derivate of the sensor temperature)
4. apply inertness correction of CMP22 sensors (tau_pyrano=1.20, fcut_pyrano=0.6, rm_length_pyrano=0.5)

Terrestrial downward
^^^^^^^^^^^^^^^^^^^^

1. smooth sensor temperature sensor for electronic noise to avoid implications in temperature dependent corrections. - running mean dt=100 sec
2. correct thermophile signal with Temperature dependence of sensor sensitivity (Kipp&Zonen calibration)
3. correct thermal offset due to fast changing temperatures (DLR paramterization using the derivate of the sensor temperature)
4. apply inertness correction of CGR4 sensors (tau_pyrano=2.00, fcut_pyrano=0.5, rm_length_pyrano=2.0)

Terretrial upward
^^^^^^^^^^^^^^^^^

1. smooth sensor temperature sensor for electronic noise to avoid implications in temperature dependent corrections. - running mean dt=100 sec
2. correct thermophile signal with Temperature dependence of sensor sensitivity (Kipp&Zonen calibration)
3. correct thermal offset due to fast changing temperatures (DLR paramterization using the derivate of the sensor temperature)
4. apply inertness correction of CGR4 sensors (tau_pyrano=2.00, fcut_pyrano=0.5, rm_length_pyrano=2.0)

00_process_bacardi_V20210928.pro
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Required User Input:**

* Flight date
* Flight number (Fxx)

**Input:**

* BACARDI quicklook data from DLR (e.g. ``QL-CIRRUS-HL_F15_20210715_ADLR_BACARDI_v1.nc``)
* simulated broadband downward irradiance from libRadtran
* direct diffuse fraction from libRadtran

**Output:**

* corrected BACARDI measurement and libRadtran simulations in netCDF file

Run like this:

.. code-block:: shell

   # cd into script folder
   cd /projekt_agmwend/data/Cirrus_HL/00_Tools/01_BACARDI/
   # start IDL
   idl
   # run script
   idl> .r 00_process_bacardi_V20210903.pro


BAHAMAS
=======

BAHAMAS records meteorological and location data during the flight.
It is mostly used for map plots and information about general flight conditions such as outside temperature, pressure, altitude, speed and so on.
It is processed by DLR and there are quicklook files provided during campaign and quality controlled files after the campaign.


libRadtran
==========

`libRadtran <https://doi.org/10.5194/gmd-9-1647-2016>`_ is a radiative transfer model which can model spectral radiative fluxes.

libRadtran simulations along flight path
----------------------------------------

The following scripts use the BAHAMAS data to create libRadtran input files to simulate fluxes along the flightpath.
The two scripts are meant to allow for flexible settings of the simulation.

BACARDI versions of these scripts are available which replace the old IDL scripts.
They are to be used as part of the BACARDI processing.
Before publishing BACARDI data, the state of the libRadtran input settings should be saved!

SMART versions of the scripts are also available which run a standard SMART setup for campaign purposes.

**TODO:**

- [ ] use specific total column ozone concentrations from OMI

   - can be downloaded here: https://disc.gsfc.nasa.gov/datasets/OMTO3G_003/summary?keywords=aura
   - you need an Earth Data account and `add the application to your profile <https://disc.gsfc.nasa.gov/earthdata-login>`_
   - checkout the `instructions for command line download <https://disc.gsfc.nasa.gov/data-access#windows_wget>`_

- [x] change atmosphere file according to location -> uvspec does this automatically when lat, lon and time are supplied

   - [x] CIRRUS-HL: use midlatitude summer (afglms.dat) or subarctic summer (afglss.dat)

- [x] use ocean or land albedo according to land sea mask
- [x] include solar zenith angle filter
- [ ] use altitude (ground height above sea level) from a surface map, when over land |rarr| adjust zout definition accordingly
- [ ] use self-made surface_type_map for simulations in the Arctic
- [ ] use sur_temperature for thermal infrared calculations (input from VELOX)
- BACARDI
- [ ] use surface_type_map for BACARDI simulations
- [ ] use surface temperature according to ERA5 reanalysis for BACARDI simulations

libradtran_write_input_file.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.libradtran_write_input_file

libradtran_run_uvspec.py
^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.libradtran_run_uvspec


BACARDI processing
------------------

The following two scripts were used in order to prepare the BACARDI processing.
They are superseded by the new python versions of these scripts.

* :ref:`processing:libradtran_write_input_file_bacardi.py`
* :ref:`processing:libradtran_run_uvspec_bacardi.py`

01_dirdiff_BBR_Cirrus_HL_Server_jr.pro
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Current settings:**

* Albedo from Taylor et al. 1996
* atmosphere file: afglt.dat -> tropical atmosphere

**Required User Input:**

* Flight date
* sonde date (mmdd)
* sounding station (stationname_stationnumber)
* time interval for modelling (time_step)

Run like this:

.. code-block:: shell

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


03_dirdiff_BBR_Cirrus_HL_Server_ter.pro
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Current settings:**

* Albedo from Taylor et al. 1996

**Required User Input:**

* Flight date
* sonde date (mmdd)
* sounding station (stationname_stationnumber)

Run like this:

.. code-block:: shell

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

libradtran_write_input_file_bacardi.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.libradtran_write_input_file_bacardi

libradtran_run_uvspec_bacardi.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.libradtran_run_uvspec_bacardi

SMART processing
----------------

For the calibration of SMART the incoming direct radiation needs to be corrected for the cosine response of the inlet.
In order to get the direct fraction of incoming radiation a clearsky simulation is necessary.
For this purpose the scripts :ref:`processing:libradtran_write_input_file_smart.py` and :ref:`processing:libradtran_run_uvspec.py` are used.
To be able to repeat the simulation the state of the first script which produced the input files for the simulation should not be changed.

libradtran_write_input_file_smart.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.libradtran_write_input_file_smart

ecRad
=====

`ecRad <https://confluence.ecmwf.int/display/ECRAD>`_ is the radiation scheme used in the ECMWF's IFS numerical weather prediction model.
For my PhD we are comparing measured radiative fluxes with simulated fluxes along the flight track.
For this we run ecRad version 1.5.0 in an offline mode and adjust input parameters.
Those experiments are documented in :ref:`experiments:Experiments`.
Here the general processing with ecRad is described.

**Notes**

*Questions:*

* What is the difference between flux_dn_direct_sw and flux_dn_sw in ecrad output? |rarr| The second includes diffuse radiation.
* What does gpoint_sw stand for?


General Notes on setting up ecRad
---------------------------------

- To avoid a floating point error when running ecrad, run ``create_practical.sh`` from the ecrad ``practical`` folder in the directory of the ecRad executable once. Somehow the data link is needed to avoid this error. (for version 1.4.1)
- changing the verbosity in the namelist files causes an floating point error (for version 1.4.1)
- decided to use ecRad version 1.5.0 for PhD

*Thoughts on the solar zenith angle:*

The ultimate goal is to compare irradiances measured by aircraft with the ones simulated by ecRad.
A big influence on these irradiances in the Arctic is the solar zenith angle or the cosine thereof.
In principle we search for the closest IFS grid point and use this data as input to ecRad.
However, for the calculation of the solar zenith angle we should not use the latitude and longitude value of the grid point as these are probably slightly off of the values recorded by the aircraft.
This slight offset in position between aircraft and grid point can cause a big difference in solar zenith angle and thus in the simulated irradiances.
To avoid additional uncertainty we therefore calculate the solar zenith angle using the latitude and longitude value of the aircraft.
As we compare minutely simulations to minutely averages of the measured data, we also take the minutely mean of the BAHAMAS data to get the aircraft's location.

**Folder Structure**

.. code-block:: shell

   ├── 0X_ecRad
   │   ├── yyyymmdd
   │   │   ├── ecrad_input
   │   │   ├── ecrad_output
   │   │   ├── ecrad_merged
   │   │   ├── radiative_properties_vX
   │   │   ├── IFS_namelists.nam
   │   │   └── ncfiles.nc


Workflow with ecRad
-------------------

| IFS raw output + navigation data from aircraft
| |rarr| create ecRad input files which correspond to the columns (+10 surrounding ones) which the aircraft passed during flight
| |rarr| run ecrad for aircraft track

| SMART/BACARDI measurements during flight + ecRad output files for aircraft track
| |rarr| compare upward and downward (spectral/banded) irradiance

#. Download IFS/CAMS data for campaign |rarr| :ref:`processing:IFS/CAMS Download`
#. Run :ref:`processing:IFS Preprocessing` to convert grib to nc files
#. Run :ref:`processing:ecrad_read_ifs.py` with the options as you want them to be (see script for details)
#. Run :ref:`processing:ecrad_cams_preprocessing.py` to prepare CAMS data
#. Update namelist in the ``{yyyymmdd}`` folder with the decorrelation length |rarr| choose one value which is representative for the period you want to study
#. Run one of :ref:`processing:ecrad_write_input_files_vx.py`
#. Run :ref:`processing:ecrad_execute_IFS.sh` with options which runs ecRad for each matching file in ``ecrad_input`` and then runs the following processing steps

    #. Run :ref:`processing:ecrad_merge_radiative_properties.py` to generate one merged radiative properties file from the single files given by the ecRad simulation
    #. Run :ref:`processing:ecrad_merge_files.py` to generate merged input and output files for and from the ecRad simulation
    #. Run :ref:`processing:ecrad_processing.py` to generate one merged file from input and output files for and from the ecRad simulation with additional variables

In general one can either vary the input to ecRad or the given namelist.
For this purpose different input versions can be/were created using modified copies of :py:mod:`ecrad_write_input_files.py`.
They can be found in the ``experiments`` folder.
An overview of which input versions should be run with which namelist versions can be found in the following table.
The version numbers reflect the process in which experiments were thought of or conducted.
With version 5 we switched from the interpolated regular lat lon grid (F1280) to the original grid resolution of the IFS which is a octahedral reduced gaussian grid (O1280).
The namelists mainly differ in the chosen ice optic parameterization (*Fu-IFS*, *Baran2016*, *Yi2013*) and whether the 3D parameterizations are turned on or not.
The output file names of the simulations only differ in the version string (e.g. *..._v16.nc*) reflecting the namelist version.
Thus, many namelists have the same settings but only have different experiment names and the difference comes due to the input.
This repetition was chosen to have a better overview of the different combinations of input version and namelist version.

=============   ==============================  =================
Input version   Namelist version                Short description
=============   ==============================  =================
1               1, 2, 3.1, 3.2, 4, 5, 6, 7, 12  Original along track data from F1280 IFS output
2               8, 9                            Use VarCloud retrieval as iwc and |re-ice| input along flight track
3               10                              Use VarCloud retrieval for below cloud simulation
4               11                              Replace q_ice=sum(ciwc, cswc) with q_ice=ciwc
5               13                              Set albedo to open ocean (0.06)
5.1             13.1                            Set albedo to 0.99
5.2             13.2                            Set albedo to BACARDI measurement below cloud
6               15, 18, 19, 22, 24              Along track data from O1280 IFS output (used instead of v1)
6.1             15.1, 18.1, 19.1, 22.1, 24.1, 30, 31, 32    Along track data from O1280 IFS output (used instead of v1) filtered for low clouds
7               16, 20, 26, 27, 28, 33, 34, 35              As v3 but with O1280 IFS output
7.1             16.1, 20.1, 26.1, 27.1, 28.1    As v3 but with O1280 IFS output using re_ice from Sun & Rikus
8               17, 21, 23, 25, 29              As v2 but with O1280 IFS output
9               14                              Turn on aerosol and use CAMS data for it
=============   ==============================  =================

IFS/CAMS Download
^^^^^^^^^^^^^^^^^

To download IFS/CAMS data from the ECMWF servers we got a user account there.
For details on how to access and download data there please see the internal Strahlungs Wiki.
These are the download scripts used and run on the ECMWF server:

IFS download script: :py:mod:`processing.halo_ac3_ifs_download_from_ecmwf.sh`

CAMS download script: :py:mod:`processing.halo_ac3_cams_downlaod_from_ecmwf.sh` (deprecated since 13.10.2023)

The CAMS trace gas climatology was implemented in the ecRad input files on 13.10.2023 (see analysis of impact in :ref:`trace-gases`).
Following the usage in the IFS the files provided at https://confluence.ecmwf.int/display/ECRAD were used in :py:mod:`ecrad_cams_preprocessing.py`.

For the CAMS aerosol climatology another file is available at https://sites.ecmwf.int/data/cams/aerosol_radiation_climatology/.

Another option would be to download the monthly mean CAMS files from the Copernicus Atmospheric Data Store (`ADS <https://ads.atmosphere.copernicus.eu>`_) and use these files.
The script :py:mod:`processing.download_cams_data.py` downloads the CAMS aerosol and trace gas climatology and saves them to seperate files.

IFS Preprocessing
^^^^^^^^^^^^^^^^^^^^^^

IFS data comes in grib format.
To convert it to netcdf and rename the parameters according to the ECMWF codes run

.. code-block:: shell

   cdo --eccodes -f nc copy infile.grb outfile.nc

on each file.
Or run the python script :py:mod:`processing.ecrad_preprocessing.py` (currently only working for IFS files):

.. automodule:: processing.ecrad_preprocessing

**CAMS files (deprecated since 13.10.2023)**

*This is not needed since there is a monthly mean product available on the ADS!*

We want to get yearly monthly means from the CAMS reanalysis.
For this we download 3-hourly data and preprocess it on the ECMWF server to avoid downloading a huge amount of data.

CAMS preprocessing script: :py:mod:`processing.halo_ac3_preprocess_cams_data_on_ecmwf.py`

The CAMS monthly files are not so big.
Thus, you can merge them into one big file and run the above command only on one file.
The merging and conversion to netCDF will take a while though.
The download script for CAMS data generates yearly folders for better structure in case more than two months are downloaded.
You can move all files into one folder by calling the following command in the CAMS folder and then merge the files with cdo:

.. code-block:: shell

    mv --target-directory=. 20*/20*.grb
    cdo mergetime 20*.grb cams_ml_halo_ac3.grb

ecrad_read_ifs.py
^^^^^^^^^^^^^^^^^
.. automodule:: processing.ecrad_read_ifs

ecrad_cams_preprocessing.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.ecrad_cams_preprocessing

ecrad_write_input_files_vx.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.ecrad_write_input_files

ecrad_execute_IFS.sh
^^^^^^^^^^^^^^^^^^^^

This script loops through all input files and runs ecrad with the setup given in ``IFS_namelist_jr_{date}_{version}.nam``.

**Attention (version 1.4.1):** ecRad has to be run without full paths for the input and output nc file.
Only the namelist has to be given with its full path.
The namelist has to be in the same folder as the input files and the output files have to be written in the same folder.


The date defines the input path which is generally ``/projekt_agmwend/data/{campaign}/{ecrad_folder}/ecrad_input/yyyymmdd/``.
It then writes the output to the given output path, one output file per input file.
The ``radiative_properties.nc`` file which is optionally generated in each run depending on the namelist is renamed and moved to a separate folder to avoid overwriting the file.

**Input:**

* ecrad input files
* IFS namelist

**Required User Input:**

* -t: use the time interpolated data (default: False)
* -d yyyymmdd: give the date to be processed
* -v v1: select which namelist version (experimental setup) to use (see :ref:`experiments:ecRad namelists and experiments` for details on version)
* -i v1: select which input version to use

**Output:**

* ecrad output files
* ``radiative_properties.nc`` moved to a separate folder and renamed according to input file (optional)

**Run like this:**

This will write all output to the console and to the specified file.

.. code-block:: shell

   . ./ecrad_execute_IFS.sh [-t] [-d yyyymmdd] [-v v1] [-i v1] 2>&1 | tee ./log/today_ecrad_yyyymmdd.log


ecrad_execute_IFS_single.sh
^^^^^^^^^^^^^^^^^^^^^^^^^^^

As above but runs only one file which has to be defined in the script.

ecrad_merge_radiative_properties.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.ecrad_merge_radiative_properties

ecrad_merge_files.py
^^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.ecrad_merge_files

ecrad_processing.py
^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.ecrad_processing


GoPro Time Lapse quicklooks
============================

During the flight a GoPro was attached to one of the windows of HALO.
Using the time-lapse function a picture was taken every 5 seconds.
Together with BAHAMAS position data (and SMART spectra measurements) a time-lapse video is created.
The GoPro was set to UTC time but cannot be synchronized to BAHAMAS.
Thus, a foto of the BAHAMAS time server is taken at the start of each recording to determine the offset of the camera from the fotos metadata.

During CIRRUS-HL the camera reset its internal time to local time, so the metadata for some flights had to be corrected for that as well.
See the ``README.md`` in the CIRRUS-HL GoPro folder for details.
A list which tracks the processing status can be found there.
For |haloac3| this table is part of the ``processing_diary.md`` which can be found in the upper level folder ``HALO-AC3``.

gopro_copy_files_to_disk.py
---------------------------
.. automodule:: processing.gopro_copy_files_to_disk

gopro_add_timestamp_to_picture.py
---------------------------------
.. automodule:: processing.gopro_add_timestamp_to_picture

gopro_write_timestamps.py
-------------------------
.. automodule:: processing.gopro_write_timestamps

gopro_plot_maps.py
------------------
.. automodule:: processing.gopro_plot_maps

gopro_add_map_to_picture.py
---------------------------
.. automodule:: processing.gopro_add_map_to_picture


gopro_make_video_from_pictures.sh
---------------------------------

Uses ffmpeg to create a stop-motion video of the GoPro pictures.

**Run on Linux!**

**Required User Input:**

* flight
* base directory of GoPro images
* framerate [12, 24]
* start_number, number in filename of first picture in folder

**Output:** video (slow or fast) of flight from GoPro pictures

Other
=====

Instrument independent processing scripts.

halo_calculate_attitude_correction.py
-------------------------------------
.. automodule:: processing.halo_calculate_attitude_correction

ifs_calculate_along_track_stats.py
----------------------------------
.. automodule:: processing.ifs_calculate_along_track_stats