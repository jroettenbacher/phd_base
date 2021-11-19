Processing
===========

With data processing one usually refers to the process of converting the raw measurement files as delivered by the instrument PC into error/bias corrected, calibrated and distributable/publishable files.
For this work the netCDF standard is defined as the preferred output format.

Each instrument has different requirements before, after and during the campaign.
In general these scripts are not meant for analysing the data to answer specific science questions or for producing quicklooks.
This is handled by the scripts in the folders :ref:`analysis:Analysis` and :ref:`quicklooks:Quicklooks`.

SMART
------

SMART has to be calibrated in the lab and in the field so there are processing routines for both cases.

smart_calib_lab_ASP06.py
^^^^^^^^^^^^^^^^^^^^^^^^
