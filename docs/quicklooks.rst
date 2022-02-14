Quicklooks
==========

Especially during a campaign it is important to get a quick look into the measured data.
Usually those quicklooks are provided by each instrument operator or team shortly after the flight.
For |haloac3| a hackathon was held to prepare those quicklooks for each instrument to get a set of comparable plots for each flight.

Another idea was to generate quicklook files which are mostly unprocessed subsets of the original measurement data, but can be shared and used by other groups easily to prepare combined quicklooks or to correct their measurements.

From LIM side there are quicklooks provided for:

- BAHAMAS (see :ref:`bahamas-ql`)
- HALO-SMART system (see :ref:`smart-ql`)

For HALO-SMART there are also two quicklook nc files generated:

- HALO-SMART data quicklook file with selected wavelengths and broadband data (see :ref:`smart-ql-nc`)
- HALO-SMART INS quicklook file with attitude angles and solar position (see :ref:`smart-ins-ql-nc`)


.. _bahamas-ql:

bahamas_quicklook_halo_ac3.py
------------------------------
.. automodule:: quicklooks.bahamas_quicklook_halo_ac3


.. _smart-ql:

halo_smart_quicklook_halo_ac3.py
---------------------------------
TBD

.. _smart-ql-nc:

smart_write_ncfile_ql.py
-------------------------
.. automodule:: quicklooks.smart_write_ncfile_ql

.. _smart-ins-ql-nc:

smart_write_INS_ql_file.py
---------------------------
.. automodule:: quicklooks.smart_write_INS_ql_file
