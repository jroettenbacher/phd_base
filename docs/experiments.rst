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
* ``IFS_namelist_jr_20210629a_v2.nam``: for flight 20210629a with Baran2017 ice model (deprecated)
* ``IFS_namelist_jr_20210629a_v3.nam``: for flight 20210629a with Baran2016 ice model
* ``IFS_namelist_jr_20210629a_v4.nam``: for flight 20210629a with Yi2013 ice model

|haloac3| **2022-04-11**

* vxx.1: uses input version vx.1 instead of vx
* v6.1: with low clouds filtered
* v6.2: with low clouds filtered and cosine dependence of ice effective radius removed
* v7.1: instead of using the VarCloud retrieved re_ice use Sun & Rikus to calculate it from the VarCloud IWC
* v7.2: instead of using the VarCloud retrieved re_ice use Sun & Rikus to calculate it from the VarCloud IWC and remove cosine dependence of minimum re_ice

.. csv-table:: ecRad version overview
   :file: files/ecrad_version_overview.csv
   :header-rows: 1

=============   ==============================================  ===========================================================================================
Input version   Namelist version                                Short description
=============   ==============================================  ===========================================================================================
1               1, 2, 3.1, 3.2, 4, 5, 6, 7, 12                  Original along track data from F1280 IFS output
2               8, 9                                            Use VarCloud retrieval as iwc and |re-ice| input along flight track
3               10                                              Use VarCloud retrieval for below cloud simulation
4               11                                              Replace q_ice=sum(ciwc, cswc) with q_ice=ciwc
5               13                                              Set albedo to open ocean (0.06)
5.1             13.1                                            Set albedo to 0.99
5.2             13.2                                            Set albedo to BACARDI measurement below cloud
6               15, 18, 19, 22, 24                              Along track data from O1280 IFS output (used instead of v1)
6.1             15.1, 18.1, 19.1, 22.1, 24.1, 30.1, 31.1, 32.1  As above but filtered for low clouds
6.2             39.2, 40.2                                      As above but with latitude set to 0 to remove cosine dependence of the ice effective radius
7               16, 20, 26, 27, 28, 33, 34, 35, 36, 37, 38      As v3 but with O1280 IFS output
7.1             16.1, 20.1, 26.1, 27.1, 28.1                    As above but using re_ice from Sun & Rikus
7.2             41.2, 42.2                                      As above but with latitude set to 0 to remove cosine dependence of the ice effective radius
8               17, 21, 23, 25, 29                              As v2 but with O1280 IFS output
9               14                                              Turn on aerosol and use CAMS data for it
=============   ==============================================  ===========================================================================================

* ``IFS_namelist_jr_20220411_v1.nam``: for RF17 with **Fu-IFS** ice model
* ``IFS_namelist_jr_20220411_v2.nam``: for RF17 with **Baran2017** ice model (deprecated)
* ``IFS_namelist_jr_20220411_v3.1.nam``: for RF17 with Fu-IFS ice model and `overlap_decorr_length = 1028 m`
* ``IFS_namelist_jr_20220411_v3.2.nam``: for RF17 with Fu-IFS ice model and `overlap_decorr_length = 450 m`
* ``IFS_namelist_jr_20220411_v4.nam``: for RF17 with **Yi2013** ice model
* ``IFS_namelist_jr_20220411_v5.nam``: for RF17 with **Fu-IFS** ice model and **3D** parameterizations enabled
* ``IFS_namelist_jr_20220411_v6.nam``: for RF17 with **Baran2016** ice model and **3D** parameterizations enabled
* ``IFS_namelist_jr_20220411_v7.nam``: for RF17 with **Baran2016** ice model
* ``IFS_namelist_jr_20220411_v8.nam``: for RF17 with **Fu-IFS** ice model using **VarCloud** retrieval for q_ice and re_ice input (input version v2)
* ``IFS_namelist_jr_20220411_v9.nam``: for RF17 with **Baran2016** ice model using **VarCloud** retrieval for q_ice and re_ice input (input version v2)
* ``IFS_namelist_jr_20220411_v10.nam``: for RF17 with **Fu-IFS** ice model using **VarCloud** retrieval for q_ice and re_ice for the **below cloud** section as well (input version v3)
* ``IFS_namelist_jr_20220411_v11.nam``: for RF17 with **Fu-IFS** ice model using ciwc as q_ice instead of sum(ciwc, cswc) (input version v4)
* ``IFS_namelist_jr_20220411_v12.nam``: for RF17 with **Fu-IFS** ice model using general cloud optics
* ``IFS_namelist_jr_20220411_v13.nam``: for RF17 with **Fu-IFS** ice model setting albedo to open ocean (input version v5)
* ``IFS_namelist_jr_20220411_v13.1.nam``: for RF17 with **Fu-IFS** ice model setting albedo to 0.99 (input version v5.1)
* ``IFS_namelist_jr_20220411_v13.2.nam``: for RF17 with **Fu-IFS** ice model setting albedo to BACARDI measurements belwo cloud (input version v5.2)
* ``IFS_namelist_jr_20220411_v14.nam``: for RF17 with **Fu-IFS** ice model including aerosol in run (TBD, input version v9)
* ``IFS_namelist_jr_20220411_v15.nam``: for RF17 with **Fu-IFS** ice model using O1280 IFS data (input version **v6**)
* ``IFS_namelist_jr_20220411_v16.nam``: for RF17 with **Fu-IFS** ice model using O1280 IFS data and **VarCloud** retrieval for q_ice and re_ice input for the **below cloud** section (input version **v7**)
* ``IFS_namelist_jr_20220411_v17.nam``: for RF17 with **Fu-IFS** ice model using O1280 IFS data and **VarCloud** retrieval for q_ice and re_ice input (input version **v8**)
* ``IFS_namelist_jr_20220411_v18.nam``: for RF17 with **Baran2016** ice model using O1280 IFS data (input version **v6**)
* ``IFS_namelist_jr_20220411_v19.nam``: for RF17 with **Yi2013** ice model using O1280 IFS data (input version **v6**)
* ``IFS_namelist_jr_20220411_v20.nam``: for RF17 with **Baran2016** ice model using O1280 IFS data and the **VarCloud** retrieval for q_ice and re_ice input for the **below cloud** section (input version **v7**)
* ``IFS_namelist_jr_20220411_v21.nam``: for RF17 with **Baran2016** ice model using O1280 IFS data and the **VarCloud** retrieval for q_ice and re_ice input (input version **v8**)
* ``IFS_namelist_jr_20220411_v22.nam``: for RF17 with **Fu-IFS** ice model using O1280 IFS data and **3D** parameterizations enabled (input version **v6**)
* ``IFS_namelist_jr_20220411_v23.nam``: for RF17 with **Fu-IFS** ice model using O1280 IFS data and **VarCloud** retrieval for q_ice and re_ice input and **3D** parameterizations enabled (input version **v8**)
* ``IFS_namelist_jr_20220411_v24.nam``: for RF17 with **Baran2016** ice model using O1280 IFS and **3D** parameterizations enabled (input version **v6**)
* ``IFS_namelist_jr_20220411_v25.nam``: for RF17 with **Baran2016** ice model using O1280 IFS and **VarCloud** retrieval for q_ice and re_ice input and **3D** parameterizations enabled (input version **v8**)
* ``IFS_namelist_jr_20220411_v26.nam``: for RF17 with **Fu-IFS** ice model using O1280 IFS data and **VarCloud** retrieval for q_ice and re_ice input **below cloud** and **3D** parameterizations enabled (input version **v7**)
* ``IFS_namelist_jr_20220411_v27.nam``: for RF17 with **Baran2016** ice model using O1280 IFS and **VarCloud** retrieval for q_ice and re_ice input **below cloud** and **3D** parameterizations enabled (input version **v7**)
* ``IFS_namelist_jr_20220411_v28.nam``: for RF17 with **Yi2013** ice model using O1280 IFS and **VarCloud** retrieval for q_ice and re_ice input **below cloud** (input version **v7**)
* ``IFS_namelist_jr_20220411_v29.nam``: for RF17 with **Yi2013** ice model using O1280 IFS and **VarCloud** retrieval for q_ice and re_ice input (input version **v8**)
* ``IFS_namelist_jr_20220411_v30.nam``: for RF17 with **Fu-IFS** ice model using O1280 IFS and **aerosols** turned on (input version **v6.1**)
* ``IFS_namelist_jr_20220411_v31.nam``: for RF17 with **Yi2013** ice model using O1280 IFS and **aerosols** turned on (input version **v6.1**)
* ``IFS_namelist_jr_20220411_v32.nam``: for RF17 with **Baran2016** ice model using O1280 IFS and **aerosols** turned on (input version **v6.1**)
* ``IFS_namelist_jr_20220411_v33.nam``: for RF17 with **Fu-IFS** ice model using O1280 IFS, **VarCloud** retrieval for q_ice and re_ice input and **aerosols** turned on (input version **v7**)
* ``IFS_namelist_jr_20220411_v34.nam``: for RF17 with **Yi2013** ice model using O1280 IFS, **VarCloud** retrieval for q_ice and re_ice input and **aerosols** turned on (input version **v7**)
* ``IFS_namelist_jr_20220411_v35.nam``: for RF17 with **Baran2016** ice model using O1280 IFS, **VarCloud** retrieval for q_ice and re_ice input and **aerosols** turned on (input version **v7**)
* ``IFS_namelist_jr_20220411_v36.nam``: for RF17 with **Fu-IFS** ice model using O1280 IFS, **VarCloud** retrieval for q_ice and re_ice input **below cloud**, turned fractional standard deviation to 0 (measure for inhomogeneity) (input version **v7**)
* ``IFS_namelist_jr_20220411_v37.nam``: for RF17 with **Yi2013** ice model using O1280 IFS, **VarCloud** retrieval for q_ice and re_ice input **below cloud**, turned fractional standard deviation to 0 (measure for inhomogeneity) (input version **v7**)
* ``IFS_namelist_jr_20220411_v38.nam``: for RF17 with **Baran2016** ice model using O1280 IFS, **VarCloud** retrieval for q_ice and re_ice input **below cloud**, turned fractional standard deviation to 0 (measure for inhomogeneity) (input version **v7**)
* ``IFS_namelist_jr_20220411_v39.2.nam``: for RF18 with **Fu-IFS** ice model using O1280 IFS, turned of cosine dependence of minimum re_ice (input version **v6.2**)
* ``IFS_namelist_jr_20220411_v40.2.nam``: for RF18 with **Yi2013** ice model using O1280 IFS, turned of cosine dependence of minimum re_ice (input version **v6.2**)
* ``IFS_namelist_jr_20220411_v41.2.nam``: for RF18 with **Fu-IFS** ice model using O1280 IFS and **VarCloud** retrieval for q_ice input for the **below cloud** section, re_ice is calculated with Sun & Rikus from VarCloud IWC but with turned of cosine dependence of minimum re_ice (input version **v7.2**)
* ``IFS_namelist_jr_20220411_v42.2.nam``: for RF18 with **Yi2013** ice model using O1280 IFS and **VarCloud** retrieval for q_ice input for the **below cloud** section, re_ice is calculated with Sun & Rikus from VarCloud IWC but with turned of cosine dependence of minimum re_ice (input version **v7.2**)

|haloac3| **2022-04-12**

* vxx.1: uses input version vx.1 instead of vx
* v6.1: with low clouds filtered
* v7.1: instead of using the VarCloud retrieved re_ice use Sun & Rikus to calculate it

* ``IFS_namelist_jr_20220412_v1.nam``: for RF18 with **Fu-IFS** ice model
* ``IFS_namelist_jr_20220412_v8.nam``: for RF18 with **Fu-IFS** ice model using **VarCloud** retrieval for q_ice and re_ice input
* ``IFS_namelist_jr_20220412_v11.nam``: for RF18 with **Fu-IFS** ice model using ciwc as q_ice instead of sum(ciwc, cswc)
* ``IFS_namelist_jr_20220412_v15.nam``: for RF18 with **Fu-IFS** ice model using O1280 IFS data (input version **v6**)
* ``IFS_namelist_jr_20220412_v16.nam``: for RF18 with **Fu-IFS** ice model using O1280 IFS data and **VarCloud** retrieval for q_ice and re_ice input for the **below cloud** section (input version **v7**)
* ``IFS_namelist_jr_20220412_v17.nam``: for RF18 with **Fu-IFS** ice model using O1280 IFS data and **VarCloud** retrieval for q_ice and re_ice input (input version **v8**)
* ``IFS_namelist_jr_20220412_v18.nam``: for RF18 with **Baran2016** ice model using O1280 IFS data (input version **v6**)
* ``IFS_namelist_jr_20220412_v19.nam``: for RF18 with **Yi2013** ice model using O1280 IFS data (input version **v6**)
* ``IFS_namelist_jr_20220412_v20.nam``: for RF18 with **Baran2016** ice model using O1280 IFS data and the **VarCloud** retrieval for q_ice and re_ice input for the **below cloud** section (input version **v7**)
* ``IFS_namelist_jr_20220412_v21.nam``: for RF18 with **Baran2016** ice model using O1280 IFS data and the **VarCloud** retrieval for q_ice and re_ice input (input version **v8**)
* ``IFS_namelist_jr_20220412_v22.nam``: for RF18 with **Fu-IFS** ice model using O1280 IFS data and **3D** parameterizations enabled (input version **v6**)
* ``IFS_namelist_jr_20220412_v23.nam``: for RF18 with **Fu-IFS** ice model using O1280 IFS data and **VarCloud** retrieval for q_ice and re_ice input and **3D** parameterizations enabled (input version **v8**)
* ``IFS_namelist_jr_20220412_v24.nam``: for RF18 with **Baran2016** ice model using O1280 IFS and **3D** parameterizations enabled (input version **v6**)
* ``IFS_namelist_jr_20220412_v25.nam``: for RF18 with **Baran2016** ice model using O1280 IFS and **VarCloud** retrieval for q_ice and re_ice input and **3D** parameterizations enabled (input version **v8**)
* ``IFS_namelist_jr_20220412_v26.nam``: for RF18 with **Fu-IFS** ice model using O1280 IFS data and **VarCloud** retrieval for q_ice and re_ice input **below cloud** and **3D** parameterizations enabled (input version **v7**)
* ``IFS_namelist_jr_20220412_v27.nam``: for RF18 with **Baran2016** ice model using O1280 IFS and **VarCloud** retrieval for q_ice and re_ice input **below cloud** and **3D** parameterizations enabled (input version **v7**)
* ``IFS_namelist_jr_20220412_v28.nam``: for RF18 with **Yi2013** ice model using O1280 IFS and **VarCloud** retrieval for q_ice and re_ice input **below cloud** (input version **v7**)
* ``IFS_namelist_jr_20220412_v29.nam``: for RF18 with **Yi2013** ice model using O1280 IFS and **VarCloud** retrieval for q_ice and re_ice input (input version **v8**)
* ``IFS_namelist_jr_20220412_v30.nam``: for RF18 with **Fu-IFS** ice model using O1280 IFS and **aerosols** turned on (input version **v6.1**)
* ``IFS_namelist_jr_20220412_v31.nam``: for RF18 with **Yi2013** ice model using O1280 IFS and **aerosols** turned on (input version **v6.1**)
* ``IFS_namelist_jr_20220412_v32.nam``: for RF18 with **Baran2016** ice model using O1280 IFS and **aerosols** turned on (input version **v6.1**)
* ``IFS_namelist_jr_20220412_v33.nam``: for RF18 with **Fu-IFS** ice model using O1280 IFS, **VarCloud** retrieval for q_ice and re_ice input and **aerosols** turned on (input version **v7**)
* ``IFS_namelist_jr_20220412_v34.nam``: for RF18 with **Yi2013** ice model using O1280 IFS, **VarCloud** retrieval for q_ice and re_ice input and **aerosols** turned on (input version **v7**)
* ``IFS_namelist_jr_20220412_v35.nam``: for RF18 with **Baran2016** ice model using O1280 IFS, **VarCloud** retrieval for q_ice and re_ice input and **aerosols** turned on (input version **v7**)
* ``IFS_namelist_jr_20220412_v36.nam``: for RF18 with **Fu-IFS** ice model using O1280 IFS, **VarCloud** retrieval for q_ice and re_ice input **below cloud**, turned fractional standard deviation to 0 (measure for inhomogeneity) (input version **v7**)
* ``IFS_namelist_jr_20220412_v37.nam``: for RF18 with **Yi2013** ice model using O1280 IFS, **VarCloud** retrieval for q_ice and re_ice input **below cloud**, turned fractional standard deviation to 0 (measure for inhomogeneity) (input version **v7**)
* ``IFS_namelist_jr_20220412_v38.nam``: for RF18 with **Baran2016** ice model using O1280 IFS, **VarCloud** retrieval for q_ice and re_ice input **below cloud**, turned fractional standard deviation to 0 (measure for inhomogeneity) (input version **v7**)
* ``IFS_namelist_jr_20220412_v39.2.nam``: for RF18 with **Fu-IFS** ice model using O1280 IFS, turned of cosine dependence of minimum re_ice (input version **v6.2**)
* ``IFS_namelist_jr_20220412_v40.2.nam``: for RF18 with **Yi2013** ice model using O1280 IFS, turned of cosine dependence of minimum re_ice (input version **v6.2**)
* ``IFS_namelist_jr_20220412_v41.2.nam``: for RF18 with **Fu-IFS** ice model using O1280 IFS and **VarCloud** retrieval for q_ice input for the **below cloud** section, re_ice is calculated with Sun & Rikus from VarCloud IWC but with turned of cosine dependence of minimum re_ice (input version **v7.2**)
* ``IFS_namelist_jr_20220412_v42.2.nam``: for RF18 with **Yi2013** ice model using O1280 IFS and **VarCloud** retrieval for q_ice input for the **below cloud** section, re_ice is calculated with Sun & Rikus from VarCloud IWC but with turned of cosine dependence of minimum re_ice (input version **v7.2**)

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

Fixed Albedo Experiment
-----------------------

**Input files for ecRad**

*Script:* :py:mod:`experiments.ecrad_write_input_files_v5.py`

.. automodule:: experiments.ecrad_write_input_files_v5

*Script:* :py:mod:`experiments.ecrad_write_input_files_v5_1.py`

.. automodule:: experiments.ecrad_write_input_files_v5_1

*Script:* :py:mod:`experiments.ecrad_write_input_files_v5_2.py`

.. automodule:: experiments.ecrad_write_input_files_v5_2

**Analysis**

*Script:* :py:mod:`experiments.ecrad_experiment_v13.py`

.. automodule:: experiments.ecrad_experiment_v13


.. _trace-gases:

Trace gas comparison
--------------------

*Script:* :py:mod:`experiments.ecrad_trace_gases.py`

.. automodule:: experiments.ecrad_trace_gases

Inhomogeneity test (fractional_std)
-----------------------------------

*Script:* :py:mod:`experiments.ecrad_experiment_v36.py`

.. automodule:: experiments.ecrad_experiment_v36


Direct sea ice albdeo
---------------------

*Script:* :py:mod:`experiments.ecrad_new_direct_sea_ice_albedo.py`

.. automodule:: experiments.ecrad_new_direct_sea_ice_albedo


CAMS aerosol climatology
------------------------

*Script:* :py:mod:`experiments.ecrad_experiment_aerosol.py`

.. automodule:: experiments.ecrad_experiment_aerosol

3-D effects parameterization
----------------------------

*Script:* :py:mod:`experiments.ecrad_3D_case_study.py`

.. automodule:: experiments.ecrad_3D_case_study


ecRad Setups
============

Setup ala Hanno
---------------

* manually enter decorrelation length in namelist depending on latitude (output from :ref:`processing:ecrad_read_ifs.py`)
* solar constant: distance sun-earth from `heavens above website <https://www.heavens-above.com/sun.aspx?lat=0&lng=0&loc=Unspecified&alt=0&tz=UCT>`_, calculate solar constant in Excel ({campaign}_solar_constant.xlsx)
* use ozone data from ozone sondes |rarr| http://www.ndaccdemo.org/

Standard IFS setup for |haloac3|
--------------------------------

* namelists: ``IFS_namelist_jr_20220411_v15.nam``, ...
* manually entered mean decorrelation length for case study region
* solar constant given
* ozone data from trace gas climatology
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