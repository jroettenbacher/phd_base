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

`ecRad <https://confluence.ecmwf.int/display/ECRAD>_ is the radiation scheme used in the ECMWF's IFS numerical weather prediction model.
For my PhD we are comparing measured radiative fluxes with simulated fluxes along the flight track.
For this we run ecRad in a offline mode and adjust input parameters.
Those experiments are documented in :ref:`experiments`.
Here the general processing of ecRad is described.

General Notes on setting up ecRad
---------------------------------

- To avoid a floating point error when running ecrad, run ``create_practical.sh`` from the ecrad ``practical`` folder in the directory of the ecRad executable once.
Somehow the data link is needed to avoid this error.

Workflow with ecRad
-------------------

IFS raw output as downloaded by Jan + navigation data from aircraft
    -> create ecRad input files which correspond to the columns which the aircraft passed during flight\
    -> run ecrad for aircraft track \
SMART measurements during flight + ecRad output files for aircraft track \
    -> compare upward and downward irradiance

1. Download IFS data for campaign (TODO: Ask Hanno for instructions)
2. Run :ref:`processing:IFS preprocessing` to convert grib to nc files
3. Decide which flight to work on -> set date in `read_ifs.py`
4. Run `read_ifs.py` with the options `step` and `t_interp` as you want them to be (see Scripts)
5. Update namelist in the `ecrad_input/{yyyymmdd}` folder with the decorrelation length
6. Run `execute_IFS.sh` which runs ecrad for each file in `ecrad_input` (maybe set verbosity level lower to avoid cluttering your screen)

IFS preprocessing
^^^^^^^^^^^^^^^^^

IFS data comes in grib format.
To convert it to netcdf and rename the parameters according to the ecmwf codes run
.. code-block:: shell

   cdo -t ecmwf -f nc copy infile.grb outfile.nc

on each file.

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
