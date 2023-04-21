***********
Experiments
***********

Here different model setups and experiments are documented and discussed.
A lot of knobs can be turned in models.
This chapter is meant to track the different model setup used and their results.

ecRad Namelists and Experiments
===============================

Namelists can be found in the corresponding date folder in the ecrad folder for each campaign (see ``config.toml``).

* ``IFS_namelist.nam``: Namelist as used by Kevin
* ``IFS_namelist_jr.nam``: Adjusted namelist for first runs by Johannes
* ``IFS_namelist_v2.nam``: Namelist with spectral short wave albedo enabled
* ``IFS_namelist_hanno_org.nam``: Used by Hanno

**CIRRUS-HL 2021-06-29**

* ``IFS_namelist_jr_20210629a_v1.nam``: for flight 20210629a with Fu-IFS ice model
* ``IFS_namelist_jr_20210629a_v2.nam``: for flight 20210629a with Baran2017 ice model
* ``IFS_namelist_jr_20210629a_v3.nam``: for flight 20210629a with Baran2016 ice model
* ``IFS_namelist_jr_20210629a_v4.nam``: for flight 20210629a with Yi2013 ice model

|haloac3| **2022-04-11**

* ``IFS_namelist_jr_20220411_v1.nam``: for flight RF17 with Fu-IFS ice model
* ``IFS_namelist_jr_20220411_v2.nam``: for flight RF17 with Baran2017 ice model
* ``IFS_namelist_jr_20220411_v4.nam``: for flight RF17 with Yi ice model
* ``IFS_namelist_jr_20220411_v5.nam``: for flight RF17 with Fu-IFS ice model and 3D parameterizations enabled
* ``IFS_namelist_jr_20220411_v6.nam``: for flight RF17 with Baran2016 ice model and 3D parameterizations enabled
* ``IFS_namelist_jr_20220411_v7.nam``: for flight RF17 with Baran2016 ice model
* ``IFS_namelist_jr_20220411_v8.nam``: for flight RF17 with Fu-IFS ice model using varcloud retrieval for ciwc and re_ice input (input version v2)
* ``IFS_namelist_jr_20220411_v9.nam``: for flight RF17 with Baran2016 ice model using varcloud retrieval for ciwc and re_ice input (input version v2)
* ``IFS_namelist_jr_20220411_v10.nam``: for flight RF17 with Fu-IFS ice model using varcloud retrieval for ciwc and re_ice (input version v3)
* ``IFS_namelist_jr_20220411_v11.nam``: for flight RF17 with Fu-IFS ice model using ciwc as q_ice instead of sum(ciwc, cswc) (input version v4)

|haloac3| **2022-04-12**

* ``IFS_namelist_jr_20220412_v1.nam``: for flight RF18 with Fu-IFS ice model
* ``IFS_namelist_jr_20220412_v8.nam``: for flight RF18 with Fu-IFS ice model using varcloud retrieval for ciwc and re_ice input
* ``IFS_namelist_jr_20220412_v11.nam``: for flight RF18 with Fu-IFS ice model using ciwc as q_ice instead of sum(ciwc, cswc)


Overlap decorrelation length experiment
---------------------------------------

*Script:* :py:mod:`ecrad_experiment_v3_x.py`

.. automodule:: experiments.ecrad_experiment_v3_x

Varcloud retrieval input experiment
-----------------------------------

*Script:* :py:mod:`experiments.ecrad_write_input_files_v2.py`

.. automodule:: experiments.ecrad_write_input_files_v2

*Script:* :py:mod:`experiments.ecrad_experiment_v8.py`

.. automodule:: experiments.ecrad_experiment_v8

Ice Mass Mixing Ratio Experiment
--------------------------------

*Script:* :py:mod:`experiments.ecrad_write_input_files_v4.py`

.. automodule:: experiments.ecrad_write_input_files_v4

*Script:* :py:mod:`experiments.ecrad_experiment_v11.py`

.. automodule:: experiments.ecrad_experiment_v11


ecRad Setups
============

Setup ala Hanno
---------------

* manually enter decorrelation length in namelist depending on latitude (output from :ref:`processing:ecrad_read_ifs.py`)
* solar constant: distance sun-earth from `heavens above website <https://www.heavens-above.com/sun.aspx?lat=0&lng=0&loc=Unspecified&alt=0&tz=UCT>`_, calculate solar constant in Excel ({campaign}_solar_constant.xlsx)
* use ozone data from ozone sondes |rarr| http://www.ndaccdemo.org/

Standard IFS setup for |haloac3|
--------------------------------

* namelists: ``IFS_namelist_jr_20220411_v1.nam``
* manually entered decorrelation length
* solar constant given
* ozone data from ozone sonde
* aerosol disabled


libRadtran Setups and Experiments
=================================

|haloac3| BACARDI broadband simulation for BACARDI nc file
----------------------------------------------------------

- use Longyearbyen radiosonde for vertical profile of relative humidity
- surface temperature |rarr| unclear, ask Anna
   - set to 293.15K in script but might have been different for Anna's simulation
- for details see: :py:mod:`processing.libradtran_write_input_file_bacardi.py`
- filename: `HALO-AC3_HALO_libRadtran_bb_clearsky_simulation_[solar/terrestrial]_[yyyymmdd]_RF[xx].nc`

|haloac3| BACARDI below cloud RF17
----------------------------------

*Script:* :py:mod:`experiments.libradtran_write_input_file_below_cloud.py`
*Folder:* ``exp0``

Simulate a clearsky situation above/below the cirrus while HALO was below/above it.
This can be used to compare the above and below cloud simulation at the same time to derive the atmospheric absorption.
Using this the actual influence of the cirrus can be derived.

- use dropsonde profiles as input
- use the IFS sea ice albedo parameterization (TODO)

|haloac3| BACARDI/SMART clear sky simulation with sea ice
---------------------------------------------------------

*Scripts:*

- :py:mod:`experiments.libradtran_write_input_file_seaice.py`
- :py:mod:`experiments.libradtran_run_uvspec_seaice.py`

*Folders:*

- ``seaice_smart``: The first run of this experiment was done for the wavelength range 250 - 2225 nm on accident
- ``seaice_solar``, ``seaice_thermal``

.. automodule:: experiments.libradtran_write_input_file_seaice

|haloac3| BACARDI/SMART clear sky simulation with sea ice up to 5000nm
----------------------------------------------------------------------

*Scripts:*

- :py:mod:`experiments.libradtran_write_input_file_seaice_2.py`
- :py:mod:`experiments.libradtran_run_uvspec_seaice.py`

*Folders:*

- ``seaice_2_solar``

*Notes:*

The same execution script is used as for the previous sea ice experiment.
Only ``libradtran_dir`` in line 40 and ``nc_filepath`` in line 208 are adjusted.

.. automodule:: experiments.libradtran_write_input_file_seaice_2

|haloac3| Icecloud over sea ice experiment
-------------------------------------------

**Name:** iceloud

*Scripts:*

- :py:mod:`experiments.libradtran_write_input_file_icecloud.py`
- :py:mod:`experiments.libradtran_run_uvspec_experiment.py`
- :py:mod:`experiments.libradtran_icecloud_sensitivity_study.py`

*Folders:*

- ``icecloud``

Icecloud Setup
^^^^^^^^^^^^^^

.. automodule:: experiments.libradtran_write_input_file_icecloud

.. automodule:: experiments.libradtran_icecloud_sensitivity_study

|haloac3| Icecloud along flight track for RF17
-----------------------------------------------

**Name:** icecloud2

*Scripts:*

- :py:mod:`experiments.libradtran_write_input_file_icecloud2.py`
- :py:mod:`experiments.libradtran_run_uvspec_experiment.py`

*Folders:*

- ``icelcoud2``

Icecloud 2 Setup
^^^^^^^^^^^^^^^^

.. automodule:: experiments.libradtran_write_input_file_icecloud2

|haloac3| Varcloud simulation above cloud
------------------------------------------

**Name:** varcloud

*Scripts:*

- :py:mod:`experiments.libradtran_write_input_file_varcloud.py`
- :py:mod:`experiments.libradtran_run_uvspec_experiment.py`

*Folders:*

- ``varcloud``

Varcloud Setup
^^^^^^^^^^^^^^^^

.. automodule:: experiments.libradtran_write_input_file_varcloud