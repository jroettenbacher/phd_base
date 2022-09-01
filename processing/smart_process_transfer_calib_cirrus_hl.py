#!/usr/bin/env python
"""Script to correct SMART SWIR transfer calibration measurement from the CIRRUS-HL campaign for dark current and save it to a new file and merge the minutely files

**Input**: raw SMART transfer calibration measurements

**Output**: dark current corrected and merged SMART measurements

This script needs to be run **after** :py:mod:`processing.smart_process_transfer_calib.py` and **before** :py:mod:`processing.smart_calib_transfer.py`.

During the campaign the LabView program controlling the shutter on the SWIR spectrometers of ASP06 somehow decided to change the way it works.
Usually every SWIR measurement would start with four dark current measurements (=shutter closed).
That way one can use the first measurements to correct the file for the dark current.
That behaviour changed starting with the transfer calibration on the 22. July 2021.
Now the shutter flag was still set to 0 (=closed) for the first four measurements but the shutter was not actually closed.
However, once the second file would be written after one minute of measurements the shutter would actually close.
Thus, only the first measurement file was affected by this behaviour and this is therefore only a problem for the transfer calibrations, where one would only record one file for each spectrometer.
This weird behaviour was detected during the laboratory calibration of ASP06 for |haloac3| and is now accounted for in the LabView program.

The result of this is that the field calibration factors for the ASP06 SWIR spectrometers are wrong starting on the 22. July 2021.
For the calculation of the final field calibration factors the laboratory calibration which was done after the campaign is used.

.. figure:: figures/SMART_calib_factors_Fdw_SWIR.png

    Evolution of the field calibration factor for ASP06 Fdw SWIR channel.

.. figure:: figures/SMART_calib_factors_Fup_SWIR.png

    Evolution of the field calibration factor for ASP06 Fup SWIR channel.

In order to fix this wrong correction of the dark current in the SWIR files the dark current measurements, which were routinely done during the transfer calibrations, can be used.
Instead of using the first four measurements the mean of the dark current measurement is used to correct the transfer calibration measurements for the dark current.

After calculating the new field calibration factors with the newly corrected SWIR measurements for all dates after the 22. July it was discovered that the 29. June and the 11. July also show a significantly different field calibration factor for Fup SWIR.

.. figure:: figures/SMART_calib_factors_Fup_SWIR_new.png

    Evolution of the field calibration factor after new correction of the dark current for 22. - 30. July for ASP06 Fup SWIR channel.

**Transfer Calib Fup SWIR 29. June**

Looking at the raw measurement from the calibration on 29. June shows that there was a dark current measurement at the beginning, however it seems that the shutter only opened slowly.
Usually a jump in the counts should be happening, here only a steady increase in the counts is happening.

.. figure:: figures/SMART_transfer_calib_raw_Fup_SWIR_20210629.png

    All wavelengths from the transfer calib measurement of ASP06 Fup SWIR channel.

Using this information the part where the counts gradually increase is cut out from the dark current corrected file before the field calibration factor is calculated with :py:mod:`processing.smart_calib_transfer.py` .
However, this also does not seem to yield a reasonable field calibration factor from that transfer calibration.
The most reasonable explanation seems to be that the SWIR spectrometer was unstable during the calibration.
**In that case it is best to discard this transfer calibration and use another one for the 29. June 2021.**

**Transfer Calib Fup SWIR 11. July**

.. figure:: figures/SMART_transfer_calib_raw_Fup_SWIR_20210711.png

    All wavelengths from the transfer calib measurement of ASP06 Fup SWIR channel.

The first row of the file (2021_07_11_14_19.Fup_SWIR.dat) is deleted as its timestamp lies 12 seconds before the next measurement (see the raw file).
Looking at the plot after that first correction was made still shows some weird behaviour.
At first the counts decrease during the dark current measurement, then they jump up to a plateau where they increase slightly for a few measurements and then jump down again and slowly decrease until stable conditions are reached roughly at the minute mark (14:20 UTC).
Some wavelengths even decrease back to the level of the dark current measurement, hinting at the possibility that the shutter was not working perfectly.

Looking at the corresponding dark current measurement file shows, that the dark current dropped significantly after the first couple of measurements.
Thus, to correct the transfer calibration measurement of Fup SWIR only the dark current measurements **before** the drop in counts at 14:21:03.78 UTC are used to calculate the mean dark current and then correct the calibration measurement by that.
The corresponding rows are deleted in the dark current measurement file (everything before 14:21:03.78).

After the dark current correction the rows exhibiting the described weird behaviour are then deleted (everything before before 14:19:50.6).

**Transfer Calib Fup SWIR 16. July**

.. figure:: figures/SMART_calib_factors_after_Fup_SWIR_new2.png

    Evolution of the field calibration factor for ASP06 Fup SWIR channel after the corrections.

After correcting the 11. July the 16. July also shows up as a rather high calibration factor.
Looking into the dark current corrected data reveals that the first pixels show only negative values, hinting at a bad dark current measurement at the beginning of the file.
**Thus, the transfer calibration from the 16. July is also discarded.**

Quicklooks generated from the final calibrated files show that the SWIR data is way out of bounds which can be traced back to a very high calibration factor starting on the 21.07.21.
Thus, for flights after that date the transfer calibration from 20.07.2021 is used for calibration.

The finally used transfer calibrations can be found in the :code:`transfer_calibs` dictionary in `cirrus_hl.py <https://github.com/jroettenbacher/phd_base/blob/0604f52c525b8db555486e8328e4fa6595d02485/src/pylim/cirrus_hl.py#L26>`_ .

*author*: Johannes Röttenbacher
"""
if __name__ == "__main__":
# %% import modules
    import pylim.helpers as h
    from pylim import smart, reader
    import os
    import pandas as pd
    import logging

    log = logging.getLogger("pylim")
    log.addHandler(logging.StreamHandler())
    log.setLevel(logging.INFO)

# %% User input
    campaign = "cirrus-hl"
    # Set paths in config.toml
    calib_path = h.get_path("calib", campaign=campaign)
    # list all erroneous transfer calib dates
    dates = ["20210711", "20210722", "20210725", "20210729", "20210730"]
    folders = [f"ASP06_transfer_calib_{d}" for d in dates]
    # folders = [folders[0]]  # select a single date to run

# %% loop through all folders where the SWIR files need to be corrected by their dark current measurement
    for folder in folders:
        dark_dir = f"{calib_path}/{folder}/dark_300ms"

        # %% merge SWIR dark measurement files before correcting the calib files
        props = ["Fup", "Fdw"]
        for dirpath, dirs, files in os.walk(os.path.join(calib_path, folder)):
            log.info(f"Working on {dirpath}")
            if "dark" in dirpath:
                for prop in props:
                    try:
                        swir_dark_files = [f for f in files if f.endswith(f"{prop}_SWIR.dat")]
                        df = pd.concat([pd.read_csv(f"{dirpath}/{file}", sep="\t", header=None) for file in swir_dark_files])
                        # delete all minutely files
                        for file in swir_dark_files:
                            os.remove(os.path.join(dirpath, file))
                            log.info(f"Deleted {dirpath}/{file}")
                        outname = f"{dirpath}/{swir_dark_files[0]}"
                        df.to_csv(outname, sep="\t", index=False, header=False)
                        log.info(f"Saved {outname}")
                    except ValueError:
                        pass

        # %% correct all SWIR calibration measurement files for the dark current
        for dirpath, dirs, files in os.walk(os.path.join(calib_path, folder)):
            log.info(f"Working on {dirpath}")
            for file in files:
                if file.endswith("SWIR.dat"):
                    log.info(f"Working on {dirpath}/{file}")
                    date_str, channel, direction = smart.get_info_from_filename(file)
                    dark_file = [f for f in os.listdir(dark_dir) if f.endswith(f"{direction}_{channel}.dat")][0]
                    log.info(f"Using dark file: {dark_file}")
                    # read in dark current measurement, drop t_int and shutter column and take mean over time
                    dark_current = reader.read_smart_raw(dark_dir, dark_file).iloc[:, 2:].mean()
                    # read in measurement file
                    measurement = reader.read_smart_raw(dirpath, file)
                    measurement = measurement.where(measurement.shutter == 1).iloc[:, 2:]  # only use data when shutter is open
                    smart_cor = measurement - dark_current
                    outname = f"{dirpath}/{file.replace('.dat', '_cor.dat')}"
                    smart_cor.to_csv(outname, sep="\t", float_format="%.0f")
                    log.info(f"Saved {outname}")

        # %% merge minutely corrected files to one file
        channels = ["SWIR"]
        for dirpath, dirs, files in os.walk(os.path.join(calib_path, folder)):
            log.info(f"Working on {dirpath}")
            for channel in channels:
                for prop in props:
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

# %% cut out the gradual increase in counts in the dark current corrected file from 29. June 202
    folder = "ASP06_transfer_calib_20210629/Tint_300ms"
    file = "2021_06_29_04_36.Fup_SWIR_cor.dat"
    df = reader.read_smart_cor(f"{calib_path}/{folder}", file)
    cut_time = pd.to_datetime("2021-06-29T04:36:18.54")
    df = df[df.index >= cut_time]
    # save the modified data frame
    df.to_csv(f"{calib_path}/{folder}/{file}", sep="\t")
    log.info(f"Saved {calib_path}/{folder}/{file}")

# %% cut out the weird behaviour in the dark current corrected file from 11. July 2021
    folder = "ASP06_transfer_calib_20210711/Tint_300ms"
    file = "2021_07_11_14_19.Fup_SWIR_cor.dat"
    df = reader.read_smart_cor(f"{calib_path}/{folder}", file)
    cut_time = pd.to_datetime("2021-07-11T14:19:50.6")
    df = df[df.index >= cut_time]
    # save the modified data frame
    df.to_csv(f"{calib_path}/{folder}/{file}", sep="\t")
    log.info(f"Saved {calib_path}/{folder}/{file}")
