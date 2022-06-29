#!/usr/bin/env python
"""Script to correct all SMART measurements from one flight for the dark current and save them to new files

**input**: raw smart measurements

**output**: dark current corrected smart measurements

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
    from pylim.cirrus_hl import transfer_calibs, smart_lookup
    import os
    import logging
    from tqdm import tqdm
    from joblib import Parallel, delayed, cpu_count

    log = logging.getLogger(__name__)
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)

    # User input
    campaign = "cirrus-hl"
    flight = "Flight_20210629a"  # which flight do the files in raw belong to?
    flight_key = flight[-4:] if campaign == "halo-ac3" else flight
    # date of transfer cali with dark current measurements to use for VNIR, set to "" if not needed
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
    #     make_dark_cur_cor_file(flight, file, inpath, transfer_cali_date, outdir)
    # run job in parallel
    Parallel(n_jobs=cpu_count()-2)(delayed(make_dark_cur_cor_file)(flight, file, inpath, transfer_cali_date, outdir)
                                   for file in files)

    print(f"Done with smart_write_dark_current_corrected_file.py for {flight}")
