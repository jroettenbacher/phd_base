#!/usr/bin/env python
"""Correct the lab calibration files for the dark current and merge the minutely files

**author**: Johannes Röttenbacher
"""
if __name__ == "__main__":
    import pylim.helpers as h
    from pylim import reader, smart
    import os
    import pandas as pd
    import logging

    log = logging.getLogger("pylim")
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.DEBUG)

    # User input
    folder = "ASP06_Calib_Lab_20210329"
    # folder = "ASP07_Calib_Lab_20210809"
    date = folder[-8:]  # extract date from folder name
    if date == "20210809":
        dark_file_J34 = "2021_08_09_10_51.Fdw_VNIR.dat"
        dark_file_J56 = "2021_08_09_10_43.Fup_VNIR.dat"
    elif date == "20210329":
        dark_file_J34 = "2021_03_29_11_19.Fdw_VNIR.dat"
        dark_file_J56 = "2021_03_29_11_09.Fup_VNIR.dat"

    # Set paths in config.toml
    calib_path = h.get_path("calib")

    log.info("Merging VNIR dark measurement files before correcting the calib files")
    properties = ["Iup", "Fup", "Fdw"]
    for dirpath, dirs, files in os.walk(os.path.join(calib_path, folder)):
        log.info(f"Working on {dirpath}")
        if "dark" in dirpath:
            for prop in properties:
                try:
                    vnir_dark_files = [f for f in files if f.endswith(f"{prop}_VNIR.dat")]
                    df = pd.concat([pd.read_csv(f"{dirpath}/{file}", sep="\t", header=None) for file in vnir_dark_files])
                    # delete all minutely files
                    for file in vnir_dark_files:
                        os.remove(os.path.join(dirpath, file))
                        log.info(f"Deleted {dirpath}/{file}")
                    outname = f"{dirpath}/{vnir_dark_files[0]}"
                    df.to_csv(outname, sep="\t", index=False, header=False)
                    log.info(f"Saved {outname}")
                except ValueError as e:
                    log.info(f"'{e}' encountered in. -> Moving on")
                    pass

    log.info("Correcting all calibration measurement files for the dark current")
    for dirpath, dirs, files in os.walk(os.path.join(calib_path, folder)):
        log.info(f"Working on {dirpath}")
        for file in files:
            if file.endswith("SWIR.dat"):
                log.info(f"Working on {dirpath}/{file}")
                smart_cor = smart.correct_smart_dark_current("", file, option=3, path=dirpath)

            if file.endswith("VNIR.dat"):
                log.info(f"Working on {dirpath}/{file}")
                if "ASP06" in dirpath:
                    if "J3_4" in dirpath:
                        dark_filepath = f"{calib_path}/{folder}/Ulli_trans_J3_4_dark/{dark_file_J34}"
                    elif "J5_6" in dirpath:
                        dark_filepath = f"{calib_path}/{folder}/Ulli_trans_J5_6_dark/{dark_file_J56}"
                    else:
                        log.info("Wrong dirpath!")
                        continue

                else:
                    dark_filepath = f"{calib_path}/{folder}/dark_200ms/2021_08_09_06_57.Iup_VNIR.dat"

                smart_cor = smart.correct_smart_dark_current("", file, option=3, path=dirpath, dark_filepath=dark_filepath)

            # write corrected smart measurement to new dat file
            outname = f"{dirpath}/{file.replace('.dat', '_cor.dat')}"
            smart_cor.to_csv(outname, sep="\t", float_format="%.0f")
            log.info(f"Saved {outname}")

    log.info("Merging minutely corrected files to one file")
    channels = ["SWIR", "VNIR"]
    for dirpath, dirs, files in os.walk(os.path.join(calib_path, folder)):
        log.info(f"Working on {dirpath}")
        for channel in channels:
            for prop in properties:
                try:
                    filename = [file for file in files if file.endswith(f"{prop}_{channel}_cor.dat")]
                    df = pd.concat([pd.read_csv(f"{dirpath}/{file}", sep="\t", index_col="time") for file in filename])
                    # delete all minutely files
                    for file in filename:
                        os.remove(os.path.join(dirpath, file))
                        log.info(f"Deleted {dirpath}/{file}")
                    outname = f"{dirpath}/{filename[0]}"
                    df.to_csv(outname, sep="\t")
                    log.info(f"Saved {outname}")
                except ValueError:
                    pass
