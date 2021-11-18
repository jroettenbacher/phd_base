Setup
=====

Get it running locally on your machine
--------------------------------------

For now (November 2021) the easiest way to use this is to ask Johannes to become collaborator on his git project and then either work within his repository on your own branch or create a folder in the cloned repository which you can track with your own git.
A possible structure could look like this::

   ├── phd_base  # source code managed by Johannes`s github account
   │   ├── analysis
   │   ├── docs
   │   ├── processing
   │   ├── src
   │   │   └──  pylim  # actual python module
   │   ├── quicklooks
   │   ├── your_folder  # source code managed by your git
   │   ├── config.toml
   │   ├── LICENSE
   │   ├── README.md
   │   ├── requirements.txt

Installing pylim
----------------
:py:mod:`pylim` is a full module complete with a :file:`pyproject.toml` and a :file:`setup.cfg`.
In order to be able to import it into your environment you have to either `install it from source <https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#installing-from-source>`_ or build a distribution and then install it.
Both ways work but the first one should be easier.
These are the ways it works for me on Windows10.

Windows10 - Install from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Open an anaconda prompt, move to your project directory (e.g. phd_base) and activate your working environment.

.. code-block:: console

   cd C:\Users\USERNAME\PyCharmProjects\phd_base
   conda activate phd_base

Then install pylim from source with:

.. code-block:: console

   python -m pip install .

Windows10 - Build distribution and install from it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Make sure you have the newest version of :py:mod:`build` installed:

.. code-block:: console

   conda install build
   conda update build

From your project directory (e.g. phd_base) you can then call:

.. code-block:: console

   python -m build

You will see a lot of output and hopefully this line::

   Successfully built pylim-0.1.0.tar.gz and pylim-0.1.0-py3-none-any.whl

For more information see `Generating distribution archives <https://packaging.python.org/tutorials/packaging-projects/#generating-distribution-archives>`_.

You will now have a new ``dist`` in your project directory with the two mentioned files above.
To install :py:mod:`pylim` from your anaconda prompt call:

.. code-block:: console

   python -m pip install .\dist\pylim-0.1.0-py3-none-any.whl

Whatever way you chose, you should be able to import :py:mod:`pylim` now:

.. code-block:: python

   import pylim.helpers as h

Data structure
--------------

HALO campaign data is organized by flight in the ``01_Flights`` folder, so that every flight has its own folder with subfolders for each instrument in it::

   ├── 01_Flights
   │   ├── all
   │   │   ├── BACARDI
   │   │   ├── BAHAMAS
   │   │   └── horidata
   │   ├── Flight_20210624a
   │   │   ├── BACARDI
   │   │   ├── BAHAMAS
   │   │   └── libRadtran
   │   ├── Flight_20210625a
   │   │   ├── BACARDI
   │   │   ├── BAHAMAS
   │   │   ├── horidata
   │   │   ├── libRadtran
   │   │   ├── quicklooks
   │   │   └── SMART
   ...

In order to be able to work across all flights an additional folder can be found called ``all``.
This folder contains one folder for each instrument which holds all data for the whole campaign.

This data is stored on the server but can also be stored locally.
To access it without needing to worry about changing the paths every time one switches from the server to local data, the function :py:func:`pylim.helpers.get_path` is used together with ``config.toml`` to generate to correct paths.
In the configuration toml file the path to each instrument can be defined either as a absolute path or -to allow for easy path creation- relative to the base directory and the flight folder.
Providing :py:func:`pylim.helpes.get_path` with the instrument key (e.g. "smart") and the flight (e.g. "Flight_20210625a") the correct path will then be created according to the current working directory.
:py:func:`pylim.helpes.get_path` also accepts a campaign keyword as well to switch between different campaigns.

There are two ways of setting up path to your local data source:

1. Edit the existing paths under ``jr_local`` or ``jr_unbuntu`` depending on whether you are using Windows or Linux.
2. Create a new campaign which defines the paths as you need them.

The second options is kind of hacky but would allow everyone to use the same config file.
However, I don't see a merit in that so number 1 would be the preferred option.
Every user should have their own toml file.
