Setup
=====

Get it running locally on your machine
--------------------------------------

For now (November 2021) the easiest way to use this is to fork this project and install pylim in your python environment.
If you want to work with the analysis and quicklook scripts you can just work in the cloned repository and add your files as you wish.
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
If you only want to work with the functions provided by pylim and don't care for the other scripts too much, the easiest way of installing :py:mod:`pylim` is to download the :file:`.tar.gz` file from the ``dist`` folder.
The :file:`.tar.gz` file holds the complete package and can be installed like the self build distribution (see :ref:`install-from-dist`).

If you forked/copied the whole repository you can also `install it from source <https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#installing-from-source>`_ or build a distribution and then install it.
All ways work but the first one should be the easiest.
These are the ways it works for me on Windows10 and Unix.

Windows10/Unix - Install from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Open an anaconda prompt (Windows10) or your bash shell (Unix), move to your project directory (e.g. phd_base) and activate your working environment.

.. code-block:: console

   cd C:\Users\USERNAME\PyCharmProjects\phd_base
   conda activate phd_base

Then install pylim from source with:

.. code-block:: console

   python -m pip install .

.. _install-from-dist:

Windows10/Unix - (Build distribution and) install from it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you already have the :file:`tar.gz` file downloaded you do not need to build the distribution anymore.
Just skip to the installation.
Otherwise start here:

Make sure you have the newest version of :py:mod:`build` installed:

.. code-block:: console

   conda install build
   conda update build

From your project directory (e.g. phd_base) you can then call:

.. code-block:: console

   python -m build

You will see a lot of output and hopefully this line::

   Successfully built pylim-|version|.tar.gz and pylim-|version|-py3-none-any.whl

For more information see `Generating distribution archives <https://packaging.python.org/tutorials/packaging-projects/#generating-distribution-archives>`_.

You will now have a new ``dist`` folder in your project directory with the two mentioned files above.
To install :py:mod:`pylim` from your anaconda prompt call:

.. code-block:: console

   python -m pip install .\dist\pylim-|version|.tar.gz

Using the ``.tar.gz`` file will delete any old installations of pylim.
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
To access it without needing to worry about changing the paths every time one switches from the server to local data, the function :py:func:`pylim.helpers.get_path` is used together with ``config.toml`` to generate the correct paths.
In the configuration toml file the path to each instrument can be defined either as a absolute path or -to allow for easy path creation- relative to the base directory and the flight folder.
Providing :py:func:`pylim.helpers.get_path` with the instrument key (e.g. "smart") and the flight (e.g. "Flight_20210625a") the correct path will then be created according to the current working directory.
:py:func:`pylim.helpers.get_path` also accepts a campaign keyword as well to switch between different campaigns.

.. attention::
   The ``config.toml`` file has to be in the current working directory of the python console. So when you run a script in a different folder (like :file:`processing`) be sure to copy your most recent ``config.toml`` to that folder as well. Or you change into the directory with the ``config.toml`` using :py:func:`os.chdir`.


There are two ways of setting up paths to your local data source:

1. Edit the existing paths under ``jr_local`` or ``jr_ubuntu`` depending on whether you are using Windows or Linux.
2. Create a new campaign which defines the paths as you need them.

The second options is kind of hacky but would allow everyone to use the same config file.
However, I don't see a merit in that so number 1 would be the preferred option.
Every user should have their own toml file.
