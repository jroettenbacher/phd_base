#!\usr\bin\env python
"""Bundle postprocessing steps for ecRad input and output files

This script takes all in- and outfiles for and from ecRad and merges them together on a given time axis which is constructed from the file names.
That should rapidly increase further work with ecRad data.
It merges stepwise to reduce IO.
In a selectable step the merged input and output files can also be merged.

It can be run via the command line and accepts several keyword arguments.

**Run like this:**

.. code-block:: shell

    python ecrad_merge_files.py io_flag=input t_interp=False base_dir="./data_jr" date=yyyymmdd merge_io=T

This would merge all ecrad input files which are not time interpolated and can be found in ``{base_dir}/{date}/ecrad_input/``.
After that the script would try to merge the merged in- and outfiles into one file.
If only ``merge_io`` is given only this would happen:

.. code-block:: shell

    # merge merged in- and outfiles
    python ecrad_merge_files.py merge_io=T date=yyyymmdd

Usually one would first call it to merge all input files and then a second time to merge all output files and merge them with the merged input files.

**User Input:**

* io_flag (input, output or None, default: None)
* date (yyyymmdd)
* version (vx, default:v1)
* t_interp (True or False, default: False)
* base_dir (directory, default: ecrad directory for halo-ac3 campaign)
* merge_io (T, optional)

**Output:**

* log file
* intermediate merged files in ``{base_dir}/ecrad_merged/``
* final merged file: ``{base_dir}/ecrad_merged_(input/output)_{yyyymmdd}(_inp).nc``
* possibly ``{base_dir}/ecrad_merged_inout_{yyyymmdd}(_inp).nc``

*author*: Johannes Röttenbacher
"""

# %% functions (a leading underscore (_) denotes functions which are used by other functions)


def _merge_ecrad_files(files: list,  date: str, outpath: str, version: str = "v1", ending: str = ""):
    """
    Merge single ecRad files as returned by read_ifs.py and ecrad itself

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
    substr1 = "input" if "input" in files[0] else "output"
    substr2 = f"_inp_{version}" if f"_inp_{version}.nc" in files[0] else f"_{version}"
    outfile = f"{outpath}/ecrad_merged_{substr1}_{date}{substr2}{ending}.nc"
    if not os.path.isfile(outfile):
        # get time stamps from filenames via regular expression
        pattern = r".*_(?P<sod>\d{,5}\.\d{1}).*"  # match maximum of 5 digits followed by one decimal
        sod_ecrad = [float(re.match(pattern, file).group('sod')) for file in files]

        ecrad = xr.open_mfdataset(files, combine="nested", concat_dim="time")
        ecrad = ecrad.assign_coords(dict(time=sod_ecrad))
        ecrad = ecrad.squeeze()  # remove one sized dimensions (eg. column)
        # assign attributes to time
        ecrad["time"] = ecrad["time"].assign_attrs(
            {'units': f'seconds since {date_dt:%Y-%m-%d}', 'long_name': 'seconds since midnight UTC'})
        ecrad.to_netcdf(outfile, format="NETCDF4_CLASSIC")
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


def merge_ecrad_files(base_dir: str, _type: str, date: str, t_interp: bool = False, version: str = "v1"):
    """
    Merge ecrad input or output files in chunks to reduce memory load.

    Args:
        base_dir: base directory where to find ecrad_input/ecrad_output folder and where to save file to
        _type: "input" or "output"
        date: date in yyyymmdd
        t_interp: use time interpolated files or not
        version: namelist version

    Returns: Writes several netCDF files

    """
    date = str(date)
    date_dt = datetime.datetime.strptime(date, "%Y%m%d")
    reg_ex = f"*_inp_{version}.nc" if t_interp else f"*_sod_{version}.nc"

    ecrad_files = sorted(glob.glob(os.path.join(base_dir, f"ecrad_{_type}", f"ecrad_{_type}_{reg_ex}")))
    nr_files = len(ecrad_files)
    log.info(f"Number of files to merge: {nr_files}")
    stepsize = _set_stepsize(nr_files)
    log.info(f"Stepsize for premerging: {stepsize}")
    ending = f"_inp_{version}" if t_interp else f"_{version}"
    outfile = f"{base_dir}/ecrad_merged_{_type}_{date}{ending}.nc"

    # merge single files if outfile doesn't exist yet
    if not os.path.isfile(outfile):
        for i in tqdm(range(0, nr_files, stepsize), desc="Merging single ecRad files"):
            _merge_ecrad_files(ecrad_files[i:i + stepsize], date, outpath=f"{base_dir}/ecrad_merged", version=version,
                               ending=f"_{i:05}")

    # merge merged files if outfile doesn't exist yet
    if not os.path.isfile(outfile):
        merged_files = sorted(glob.glob(f"{base_dir}/ecrad_merged/ecrad_merged_{_type}_{date}{ending}*.nc"))
        nr_merged_files = len(merged_files)
        log.info(f"Merged files to merge: {nr_merged_files}")
        while nr_merged_files > 50:
            stepsize = _set_stepsize(nr_merged_files)
            for i in tqdm(range(0, nr_merged_files, stepsize), desc="Merging merged files"):
                files_to_merge = merged_files[i:i + stepsize]
                ds = xr.open_mfdataset(files_to_merge)
                tmp_out = files_to_merge[0].replace(".nc", f"_{i:04}.nc")
                ds.to_netcdf(tmp_out, format="NETCDF4_CLASSIC",
                             encoding=dict(time=dict(units=f"seconds since {date_dt:%Y-%m-%d}")))
                # remove merged files to avoid counting them again
                [os.remove(f) for f in files_to_merge]

            merged_files = sorted(glob.glob(f"{base_dir}/ecrad_merged/ecrad_merged_{_type}_{date}{ending}*.nc"))
            nr_merged_files = len(merged_files)

        # final merge of files
        ds = xr.open_mfdataset(merged_files, combine="nested", concat_dim="time")
        ds.to_netcdf(outfile, format="NETCDF4_CLASSIC",
                     encoding=dict(time=dict(units=f"seconds since {date_dt:%Y-%m-%d}")))
        log.info(f"{outfile} saved")
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
    io_flag = str(args["io_flag"]) if "io_flag" in args else None
    date = args["date"] if "date" in args else None
    if date is None:
        raise ValueError("'date' needs to be given!")
    campaign = args["campaign"] if "campaign" in args else "halo-ac3"
    version = args["version"] if "version" in args else "v1"
    i_version = args["i_version"] if "i_version" in args else "v1"
    t_interp = h.strtobool(args["t_interp"]) if "t_interp" in args else False
    base_dir = args["base_dir"] if "base_dir" in args else h.get_path("ecrad", campaign=campaign)
    merge_io = h.strtobool(args["merge_io"]) if "merge_io" in args else False
    # setup logging
    __file__ = None if "__file__" not in locals() else __file__
    log = h.setup_logging("./logs", __file__, f"{io_flag}_tinp-{t_interp}_{date}")
    log.info(f"The following options have been passed:\nio_flag: {io_flag}\nt_interp: {t_interp}\nversion: {version}\n"
             f"i_version: {i_version}\nbase_dir: {base_dir}\ndate: {date}\nmerge_io: {merge_io}")
    # create input path according to given base_dir and date
    inpath = os.path.join(base_dir, date)
    if io_flag is not None:
        # merge ecrad files (input or output files)
        log.info(f"Merging ecRad {io_flag} files")
        merge_ecrad_files(inpath, io_flag, date, t_interp, version)

    # %% read in ecrad merged in and out file and merge them
    ending = f"_inp_{version}" if t_interp else f"_{version}"
    i_ending = f"_inp_{i_version}" if t_interp else f"_{i_version}"
    outfile = f"{inpath}/ecrad_merged_inout_{date}{ending}.nc"
    if merge_io and not os.path.isfile(outfile):
        ifile = f"ecrad_merged_input_{date}{i_ending}.nc"
        ofile = f"ecrad_merged_output_{date}{ending}.nc"
        log.info(f"Merging {ofile} with {ifile}")
        ecrad_in = xr.open_dataset(f"{inpath}/{ifile}")
        ecrad_out = xr.open_dataset(f"{inpath}/{ofile}")
        ecrad = xr.merge([ecrad_out, ecrad_in])#, compat="override")
        ecrad.to_netcdf(outfile, format="NETCDF4_CLASSIC")
        log.info(f"Saved {outfile}")
        if "column" in ecrad.dims:
            log.info("Take mean over column dimension")
            ecrad.mean(dim="column").to_netcdf(outfile.replace(".nc", "_mean.nc"), format="NETCDF4_CLASSIC")
            log.info(f"Saved {outfile.replace('.nc', '_mean.nc')}")

    # %% remove all intermediate merged files in ecrad_merged
    h.delete_folder_contents(f"{inpath}/ecrad_merged")

    log.info(f"Done with ecrad_merge_files in: {h.seconds_to_fstring(time.time() - start)} [h:mm:ss]")