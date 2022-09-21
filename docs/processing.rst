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

GoPro Time Lapse quicklooks
============================

During the flight a GoPro was attached to one of the windows of HALO.
Using the time-lapse function a picture was taken every 5 seconds.
Together with BAHAMAS position data (and SMART spectra measurements) a time-lapse video is created.
The GoPro was set to UTC time but cannot be synchronized to BAHAMAS.
Thus, a foto of the BAHAMAS time server is taken at the start of each recording to determine the offset of the camera from the fotos metadata.

During CIRRUS-HL the camera reset its internal time to local time, so the metadata for some flights had to be corrected for that as well.
See the `README.md` in the CIRRUS-HL GoPro folder for details.
A list which tracks the processing status can be found there.
For |haloac3| this table is part of the ``processing_diary.md`` which can be found in the upper level folder ``HALO-AC3``.

gopro_copy_files_to_disc.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.gopro_copy_files_to_disc

add_timestamp_to_picture.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.add_timestamp_to_picture

write_gopro_timestamps.py
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.write_gopro_timestamps

gopro_plot_maps.py
^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.gopro_plot_maps

add_map_to_picture.py
^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.add_map_to_picture


gopro_make_video_from_pictures.sh
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Uses ffmpeg to create a stop-motion video of the GoPro pictures.

**Run on Linux**

**Required User Input:**

* flight
* base directory of GoPro images
* framerate [12, 24]
* start_number, number in filename of first picture in folder


**Output:** video (slow or fast) of flight from GoPro pictures
