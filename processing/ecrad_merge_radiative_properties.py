#!\usr\bin\env python
"""
| *author*: Johannes Röttenbacher
| *created*: 21-04-2023

**Merge radiative properties nc files created by ecRad**

This script takes all radiative property files created by ecRad as additional output and merges them.
Each radiative property file has its time stamp in the file name (seconds of day) and can potentially have only a few columns in it.
How many columns are saved in one radiative property file depends on the ``n_blocksize`` namelist option.

It can be run via the command line and accepts several keyword arguments.

**Run like this:**

.. code-block:: shell

    python ecrad_merge_radiative_properties.py base_dir="./data_jr" date=yyyymmdd version=v1

This would merge all radiative property files which can be found in ``{base_dir}/{date}/radiative_properties_{version}/``.

**User Input:**

* date (yyyymmdd)
* version (vx, default:v1)
* base_dir (directory, default: ecrad directory for halo-ac3 campaign)

**Output:**

* log file
* intermediate merged files in ``{base_dir}/radiative_properties_merged/``
* final merged file: ``{base_dir}/radiative_properties_merged_{yyyymmdd}_{version}.nc``

"""

# %% functions (a leading underscore (_) denotes functions which are used by other functions)


def _merge_single_files(files: list,  date: str, outpath: str, version: str = "v1", ending: str = ""):
    """
    Merge single nc files

    Args:
        files: list of files names to merge
        date: yyyymmdd
        outpath: where to save file
        version: namelist version
        ending (optional): add a custom ending to the filename

    Returns: writes a new netCDF file to disk with encoded time and dropped one sized dimensions (eg. column)

    """
    h.make_dir(outpath)
    date = str(date)
    date_dt = datetime.datetime.strptime(date, "%Y%m%d")
    outfile = f"{outpath}/radiative_properties_merged_{date}_{version}{ending}.nc"
    if not os.path.isfile(outfile):
        # get time stamps from filenames via regular expression
        pattern = r".*_(?P<sod>\d{,5}\.\d{1}).*"  # match maximum of 5 digits followed by one decimal
        sod = [float(re.match(pattern, file).group('sod')) for file in files]

        ds = xr.open_mfdataset(files, combine="nested", concat_dim="time")
        ds = ds.assign_coords(dict(time=sod))
        ds = ds.squeeze()  # remove one sized dimensions (eg. column)
        # assign attributes to time
        ds["time"] = ds["time"].assign_attrs(
            {'units': f'seconds since {date_dt:%Y-%m-%d}', 'long_name': 'seconds since midnight UTC'})
        ds.to_netcdf(outfile, format="NETCDF4_CLASSIC")
    else:
        log.info(f"{outfile} already exists. Not overwritten.")


def _set_stepsize(nr_iter: int) -> int:
    """
    Define stepsize for a for loop to reduce memory load

    Args:
        nr_iter: number of total iterations

    Returns: stepsize between 10 and 100

    """
    stepsize = 0
    div = 1000
    while stepsize < 10:
        stepsize = nr_iter // div
        div = int(div / 10)

    return stepsize


def merge_radiative_properties_files(base_dir: str, date: str, version: str = "v1"):
    """
    Merge radiaitve property files in chunks to reduce memory load.

    Args:
        base_dir: base directory where to find radiative_properties_{version} folder and where to save file to
        date: date in yyyymmdd
        version: namelist version

    Returns: Writes several netCDF files

    """
    date = str(date)
    date_dt = datetime.datetime.strptime(date, "%Y%m%d")
    reg_ex = f"radiative_properties_*.nc"

    files = sorted(glob.glob(os.path.join(base_dir, f"radiative_properties_{version}", reg_ex)))
    nr_files = len(files)
    log.info(f"Number of files to merge: {nr_files}")
    stepsize = _set_stepsize(nr_files)
    log.info(f"Stepsize for premerging: {stepsize}")
    outfile = f"{base_dir}/radiative_properties_merged_{date}_{version}.nc"

    # merge single files if outfile doesn't exist yet
    if not os.path.isfile(outfile):
        for i in tqdm(range(0, nr_files, stepsize), desc="Merging single radiative properties files"):
            _merge_single_files(files[i:i + stepsize], date, outpath=f"{base_dir}/radiative_properties_merged",
                                version=version, ending=f"_{i:05}")

    # merge merged files if outfile doesn't exist yet
    if not os.path.isfile(outfile):
        merged_files = sorted(glob.glob(f"{base_dir}/radiative_properties_merged/radiative_properties_merged_{date}_{version}*.nc"))
        nr_merged_files = len(merged_files)
        log.info(f"Merged files to merge: {nr_merged_files}")
        while nr_merged_files > 50:
            stepsize = _set_stepsize(nr_merged_files)
            for i in tqdm(range(0, nr_merged_files, stepsize), desc="Merging merged files"):
                files_to_merge = merged_files[i:i + stepsize]
                ds = xr.open_mfdataset(files_to_merge, combine="nested", concat_dim="time")
                tmp_out = files_to_merge[0].replace(".nc", f"_{i:04}.nc")
                ds.to_netcdf(tmp_out, format="NETCDF4_CLASSIC",
                             encoding=dict(time=dict(units=f"seconds since {date_dt:%Y-%m-%d}")))
                # remove merged files to avoid counting them again
                [os.remove(f) for f in files_to_merge]

            merged_files = sorted(glob.glob(f"{base_dir}/radiative_properties_merged/radiative_properties_merged_{date}_{version}*.nc"))
            nr_merged_files = len(merged_files)

        # final merge of files
        ds = xr.open_mfdataset(merged_files, combine="nested", concat_dim="time")
        ds.to_netcdf(outfile, format="NETCDF4_CLASSIC",
                     encoding=dict(time=dict(units=f"seconds since {date_dt:%Y-%m-%d}")))
        log.info(f"Saved {outfile}")
    else:
        log.info(f"{outfile} already exists. Not overwritten!")


# %% run script
if __name__ == "__main__":
    # %% import modules
    import pylim.helpers as h
    import xarray as xr
    import os
    import time
    import glob
    import re
    import datetime
    from tqdm import tqdm

    start = time.time()
    # read in command line arguments or set defaults
    args = h.read_command_line_args()
    date = args["date"] if "date" in args else None
    if date is None:
        raise ValueError("'date' needs to be given!")
    version = args["version"] if "version" in args else "v1"
    base_dir = args["base_dir"] if "base_dir" in args else h.get_path("ecrad", campaign="halo-ac3")

    # setup logging
    try:
        file = __file__
    except NameError:
        file = None
    log = h.setup_logging("./logs", file, f"{date}_{version}")
    log.info(f"The following options have been passed:\n"
             f"date: {date}\n"
             f"version: {version}\n"
             f"base_dir: {base_dir}\n")
    # create input path according to given base_dir and date
    inpath = os.path.join(base_dir, str(date))

    # %% merge radiative property files
    log.info(f"Merging radiative property files")
    merge_radiative_properties_files(inpath, date, version)

    # %% remove all intermediate merged files in radiative_properties_merged
    h.delete_folder_contents(f"{inpath}/radiative_properties_merged")

    log.info(f"Done with merge_radiative_properties in: {h.seconds_to_fstring(time.time() - start)} [h:mm:ss]")