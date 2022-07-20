Processing
===========

With data processing one usually refers to the process of converting the raw measurement files as delivered by the instrument PC into error/bias corrected, calibrated and distributable/publishable files.
For this work the netCDF standard is defined as the preferred output format.

Each instrument has different requirements before, after and during the campaign.
In general these scripts are **not** meant for analysing the data to answer specific science questions or for producing quicklooks.
This is handled by the scripts in the folders :ref:`analysis:Analysis` and :ref:`quicklooks:Quicklooks`.

SMART
------

SMART has to be calibrated in the lab and in the field so there are processing routines for both cases.
Some scripts are designated for postprocessing specific campaign data as sometimes the normal processing does not cover all possible cases which occur during a campaign.

**Folder Structure**

SMART data is organized by flight.
Each flight folder has one `SMART` folder with the following subfolders:

* `data_calibrated`: dark current corrected and calibrated measurement files
* `data_cor`: dark current corrected measurement files
* `raw`: raw measurement files

Each campaign also has a calibration folder.
In the calibration folder each calibration is saved in its own folder.
Each calibration is used to generate one calibration file, which is corrected for dark current and saved in the top level calibration folder.

A few more folders needed are:

* `raw_only`: raw measurement files as written by ASP06/07, do not work on those files, but copy them into `raw`
* `lamp_F1587`: calibration lamp file
* `panel_34816`: reflectance panel file
* `pixel_wl`: pixel to wavelength files for each spectrometer

**Workflow**

There are two workflows:
1. Calibration files
2. Measurement files

Both workflows start with the correction of the dark current.
After the raw files are copied from ASP06/07 into `raw_only` and `raw` the minutely files are corrected for the dark current and saved with the new ending `*_cor.dat` in `data_cor`.
Then the minutely files are merged to one file per folder and channel.

**Calibration files**

Use :py:mod:`smart_process_transfer_calib.py` or :py:mod:`smart_process_lab_calib.py` to correct the calibration files for the dark current and merge the minutely files.
Then run :py:mod:`smart_calib_lab_ASP06/07.py` for the lab calibrations or :py:mod:`smart_calib_transfer.py` for the transfer calibration.
Each script returns a file in the `calib` folder with the calibration factors.

**Measurement files**

Use :py:mod:`smart_write_dark_currented_corrected_file.py` to correct one flight for the dark current.
Merge the resulting minutely files with :py:mod:`smart_merge_minutely_files.py`.
Finally, calibrate the measurement with :py:mod:`smart_calibrate_measurment.py`.
The resulting calibrated files are saved in the `data_calibrated` folder.
As a final step the calibrated files can then be converted to netCDF with :py:mod:`smart_write_ncfile.py`.

smart_calib_lab_ASP06.py
^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.smart_calib_lab_ASP06

smart_process_transfer_calib_cirrus_hl.py
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. automodule:: processing.smart_process_transfer_calib_cirrus_hl
