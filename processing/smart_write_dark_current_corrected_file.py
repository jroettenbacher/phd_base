#!/usr/bin/env python
"""Script to correct all SMART measurements from one flight for the dark current and save them to new files

Set the input and output paths in ``config.toml``.

**Required User Input**: campaign and flight(s)

**Output**: dark current corrected smart measurements

You can uncomment some lines to change the behavior of the script.

- run for all flights of a campaign
- run for one file
- correct a selection of files in a for loop and skip uncorrectable files

The default behavior is to run for one flight and execute everything in parallel.
This is the campaign mode.


*author*: Johannes Roettenbacher
"""


def make_dark_cur_cor_file(flight: str, file: str, inpath: str, transfer_cali_date: str, outdir: str) -> None:
    """
    Function to write a dark current corrected file to the given input file
    Args:
        flight: to which flight does the file belong to? (e.g. Flight_20210707a)
        file: Standard SMART file name
        inpath: input directory
        transfer_cali_date: date of transfer calibration to use for the dark current measurements ("" or yyyymmdd)
        outdir: output directory

    Returns: Writes dark current corrected file to outdir

    """
    log.info(f"Working on {file}")
    campaign = "halo-ac3" if flight.startswith("HALO-AC3") else "cirrus-hl"
    if len(transfer_cali_date) > 0:
        smart_cor = smart.correct_smart_dark_current(flight, file, campaign=campaign, option=3, path=inpath,
                                                     date=transfer_cali_date)
    else:
        smart_cor = smart.correct_smart_dark_current(flight, file, campaign=campaign, option=3, path=inpath)
    outfile = f"{outdir}/{file.replace('.dat', '_cor.dat')}"
    smart_cor.to_csv(outfile, sep="\t", float_format="%.0f")
    log.info(f"Saved {outfile}")


if __name__ == "__main__":
    import pylim.helpers as h
    from pylim import smart
    from pylim.halo_ac3 import transfer_calibs
    import os
    import logging
    from tqdm import tqdm
    from joblib import Parallel, delayed, cpu_count

    log = logging.getLogger(__name__)
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)

    # User input
    campaign = "halo-ac3"
    # uncomment for single flight use
    flights = ["HALO-AC3_20220321_HALO_RF08"]  # which flight do the files in raw belong to?
    # uncomment for all flights
    # flights = list(transfer_calibs.keys())  # get all flight keys for loop
    for flight in flights:
        flight_key = flight[-4:] if campaign == "halo-ac3" else flight
        transfer_cali_date = transfer_calibs[flight_key]

        # Set paths in config.toml
        inpath = h.get_path("raw", flight, campaign=campaign)
        outdir = h.get_path("data", flight, campaign=campaign)
        h.make_dir(outdir)
        # create list of input files and add a progress bar to it
        files = tqdm([file for file in os.listdir(inpath) if os.path.isfile(os.path.join(inpath, file))])
        files_debug = [file for file in os.listdir(inpath) if os.path.isfile(os.path.join(inpath, file))]

        # test one file
        # file = files_debug[1]
        # make_dark_cur_cor_file(flight, file, inpath, transfer_cali_date, outdir)
        # for file in files_debug:
        #     try:
        #         make_dark_cur_cor_file(flight, file, inpath, transfer_cali_date, outdir)
        #     except:
        #         print(f"{file} not corrected.")
        #         os.remove(f"{inpath}/{file}")
        #         pass
        # run job in parallel
        Parallel(n_jobs=cpu_count()-2)(delayed(make_dark_cur_cor_file)(flight, file, inpath, transfer_cali_date, outdir)
                                       for file in files)

        print(f"Done with smart_write_dark_current_corrected_file.py for {flight}")
