#!/usr/bin/env python
"""Calibrate measurement files with the transfer calibration

Reads in dark current corrected measurement file and corresponding transfer calibration to calibrate measurement files.

**Required User Input:**

* campaign
* flight folder
* integration time of ASP06 and ASP07 measurements (check raw measurement files to get integration times)
* whether to normalize measurements or not (use normalized calibration factor, necessary if no calibration with the same integration time was made)

**Output:** Calibrated SMART measurement files in .dat format

*author*: Johannes Roettenbacher
"""
if __name__ == "__main__":
    # %% import modules and set paths
    import pylim.helpers as h
    from pylim import smart, reader
    from pylim.halo_ac3 import smart_lookup, transfer_calibs
    import os
    import numpy as np
    import pandas as pd
    import logging

    log = logging.getLogger("pylim")
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)

    # %% set user given parameters
    campaign = "halo-ac3"
    flight = "HALO-AC3_20220412_HALO_RF18"  # set flight folder
    flight_key = flight[-4:] if campaign == "halo-ac3" else flight
    t_int_asp06 = 300  # give integration time of field measurement for ASP06
    t_int_asp07 = 300  # give integration time of field measurement for ASP07
    normalize = True  # normalize counts with integration time
    # give date of transfer calib to use for calibrating measurement if not same as measurement date else set to ""
    transfer_date = transfer_calibs[flight_key]
    date = f"{transfer_date[:4]}_{transfer_date[4:6]}_{transfer_date[6:]}"  # reformat date to match file name

    # %% set paths
    norm = "_norm" if normalize else ""
    calib_path = h.get_path("calib", campaign=campaign)
    data_path = h.get_path("data", flight, campaign=campaign)
    calibrated_path = h.get_path("calibrated", flight, campaign=campaign)
    inpath = data_path
    outpath = calibrated_path
    h.make_dir(outpath)  # create outpath if necessary

    # %% read in dark current corrected measurement files
    files = [f for f in os.listdir(inpath)]
    for file in files:
        date_str, channel, direction = smart.get_info_from_filename(file)
        date_str = date if len(date) > 0 else date_str  # overwrite date_str if date is given
        spectrometer = smart_lookup[f"{direction}_{channel}"]
        t_int = t_int_asp06 if "ASP06" in spectrometer else t_int_asp07  # select relevant integration time
        measurement = reader.read_smart_cor(data_path, file)
        # measurement[measurement.values < 0] = 0  # set negative values to 0

        # %% read in matching transfer calibration file from same day or from given day with matching t_int
        cali_file = f"{calib_path}/{date_str}_{spectrometer}_{direction}_{channel}_{t_int}ms_transfer_calib{norm}.dat"
        log.info(f"Calibration file used:\n {cali_file}")
        cali = pd.read_csv(cali_file)
        # convert to long format
        m_long = measurement.melt(var_name="pixel", value_name="counts", ignore_index=False)
        if normalize:
            m_long["counts"] = m_long["counts"] / t_int

        # merge field calibration factor to long df on pixel column
        df = m_long.join(cali.set_index(cali.pixel)["c_field"], on="pixel")
        df[direction] = df["counts"] * df["c_field"]  # calculate calibrated radiance/irradiance

        # %% save wide format calibrated measurement
        df = df[~np.isnan(df.index)]  # remove rows where the index is nan
        df_out = df.pivot(columns="pixel", values=direction)  # convert to wide format (row=time, column=pixel)
        outname = "_calibrated_norm.dat" if normalize else "_calibrated.dat"
        outfile = f"{outpath}/{file.replace('.dat', outname)}"
        df_out.to_csv(outfile, sep="\t")
        log.info(f"Saved {outfile}")
