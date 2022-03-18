#!/usr/bin/env python
"""Create a 1Hz INS file for quicklook processing directly after the flight

**Input:** SMART IMS and GPS data

**Output:** 1Hz INS netCDF file

*author*: Johannes Röttenbacher
"""

if __name__ == "__main__":
    # %% module import
    import os
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import re

    # %% user input
    campaign = "halo-ac3"
    flight = "HALO-AC3_20220201_P5_RF00"
    flight_key = flight[-4:] if campaign == "halo-ac3" else flight
    date = flight[9:17]

    # %% set paths
    base_path = f"E:/{campaign.swapcase()}/02_Flights/{flight}"
    outpath = f"E:/{campaign.swapcase()}/02_Flights/{flight}/horidata"
    try:
        os.makedirs(outpath)
    except FileExistsError:
        pass
    hori_path = f"{base_path}/horidata"
    hori_files = [f for f in os.listdir(hori_path) if f.endswith("nav")]
    date_from_file = pd.to_datetime(re.findall(r"\d{8}", hori_files[0])[0])

    # %% read in IMS and GPS data
    with open(f"{hori_path}/{hori_files[0]}") as f:
        colnames = f.readline()  # read first line of file
    colnames = re.findall(r"[a-zA-Z]+", colnames)  # extract column names with regex
    nav = pd.read_csv(f"{hori_path}/{hori_files[0]}", sep="\s+", skipinitialspace=True, comment="#", names=colnames)
    time_index = pd.to_datetime(nav.Time * 3600, unit="s", origin=date_from_file)
    nav.set_index(time_index, inplace=True)
    nav.index.name = "time"

# %% drop rows where the time index is not increasing as long as needed
    rows_to_drop = np.asarray(np.nonzero(np.diff(nav.index) <= pd.Timedelta(seconds=0)))[0]
    while len(rows_to_drop) > 1:
        nav = nav.drop(index=nav.iloc[rows_to_drop+1].index)
        rows_to_drop = np.asarray(np.nonzero(np.diff(nav.index) <= pd.Timedelta(seconds=0)))[0]

    # %% resample ims data to 1 Hz
    nav_1Hz = nav.resample("1s").asfreq()  # create a dataframe with a 1Hz index
    # reindex original dataframe and use the nearest values for the full seconds
    nav = nav.reindex_like(nav_1Hz, method="nearest")

    # %% rename columns
    df = nav.rename(columns=dict(Time="decimal_hours", Latitude="lat", Longitude="lon", Altitude="alt", Velocity="vel",
                                 Pitch="pitch", Roll="roll", Yaw="yaw", SZA="sza", SAA="saa"))

    # %% convert to xarray
    ds = df.to_xarray()

    var_attrs = dict(
        decimal_hours=dict(
            long_name='Decimal Hours since start of day',
            units='s',
            comment='Time as recorded by the GPS'),
        lat=dict(
            long_name='latitude',
            units='degrees_north',
            comment='GPS latitude'),
        lon=dict(
            long_name='longitude',
            units='degrees_east',
            comment='GPS longitude'),
        alt=dict(
            long_name='altitude',
            units='m',
            comment='Aircraft altitude'),
        vel=dict(
            long_name="Ground speed",
            unit="m s^-1",
            comment="Ground speed measured by GPS"),
        pitch=dict(
            long_name='Pitch angle',
            units='deg',
            comment='Pitch angle: positive = nose up'),
        roll=dict(
            long_name='Roll angle',
            units='deg',
            comment='Roll angle: positive = left wing up'),
        yaw=dict(
            long_name='Yaw angle',
            units='deg',
            comment='0 = East, 90 = North, 180 = West, -90 = South, range: -180 to 180'),
        sza=dict(
            long_name='Solar zenith angle',
            units='deg',
            comment='Solar zenith angle calculated with refraction correction (pressure: 1013hPa, temperature: 15°C)'),
        saa=dict(
            long_name='Solar azimuth angle',
            units='deg',
            comment='South = 180, range: 0 to 360')
    )
    # %% create variable and global attributes

    global_attrs = dict(
        title="Raw attitude angles and GPS position data from the Polar5-SMART IMS system",
        campaign_id=f"{campaign.swapcase()}",
        platform_id="Polar5",
        instrument_id="SMART",
        version_id="quicklook",
        description="1Hz data from the Polar5-SMART IMS system prepared for quicklook creation",
        institution="Leipzig Institute for Meteorology, Leipzig, Germany",
        history=f"created {datetime.strftime(datetime.utcnow(), '%c UTC')}",
        contact="Johannes Röttenbacher, johannes.roettenbacher@uni-leipzig.de",
        PI="Evi Jäkel, evi.jaekel@uni-leipzig.de"
    )

    # set units according to campaign
    if campaign == "cirrus-hl":
        units = "seconds since 2017-01-01 00:00:00 UTC"
    elif campaign == "halo-ac3":
        units = "seconds since 2017-01-01 00:00:00 UTC"
    elif campaign == "eurec4a":
        units = "seconds since 2020-01-01 00:00:00 UTC"
    else:
        raise ValueError(f"Campaign {campaign} unknown!")

    encoding = dict(
        time=dict(units=units, _FillValue=None),
    )
    for var in var_attrs:
        encoding[var] = dict(_FillValue=None)

    # %% assign meta data
    for var in ds:
        ds[var].attrs = var_attrs[var]

    ds.attrs = global_attrs

    # %% create ncfile
    outfile = f"HALO-AC3_P5_gps_ins_{date}_{flight_key}.nc"
    out = os.path.join(outpath, outfile)
    ds.to_netcdf(out, format="NETCDF4_CLASSIC", encoding=encoding)
    print(f"Saved {out}")
