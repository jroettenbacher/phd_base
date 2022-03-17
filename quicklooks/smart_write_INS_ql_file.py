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
    import pylim.helpers as h
    from pylim import reader, solar_position
    from pylim.halo_ac3 import take_offs_landings
    from datetime import datetime
    from tqdm import tqdm

    # %% user input
    campaign = "halo-ac3"
    flight = "HALO-AC3_20220316_HALO_RF06"
    flight_key = flight[-4:] if campaign == "halo-ac3" else flight
    date = flight[9:17]
    # get flight start and end
    to, td = take_offs_landings[flight_key][0], take_offs_landings[flight_key][1]

    # %% set paths
    base_path = f"E:/{campaign.swapcase()}/02_Flights/{flight}"
    outpath = f"E:/{campaign.swapcase()}/02_Flights/{flight}/horidata"
    h.make_dir(outpath)
    hori_path = f"{base_path}/horidata"
    hori_files = os.listdir(hori_path)
    nav_files = [f for f in hori_files if "Nav_IMS" in f]
    nav_filepaths = [f"{hori_path}/{f}" for f in nav_files]
    gps_files = [f for f in hori_files if "Nav_GPSPos" in f]
    gps_filepaths = [f"{hori_path}/{f}" for f in gps_files]
    vel_files = [f for f in hori_files if "Nav_GPSVel" in f]
    vel_filepaths = [f"{hori_path}/{f}" for f in vel_files]

    # %% read in IMS and GPS data
    ims = pd.concat([reader.read_nav_data(f) for f in nav_filepaths])
    gps = pd.concat([reader.read_ins_gps_pos(f) for f in gps_filepaths])
    vel = pd.concat([reader.read_ins_gps_vel(f) for f in vel_filepaths])

    # %% resample ims data to 1 Hz
    ims_1Hz = ims.resample("1s").asfreq()  # create a dataframe with a 1Hz index
    # reindex original dataframe and use the nearest values for the full seconds
    ims = ims.reindex_like(ims_1Hz, method="nearest")

    # %% merge dataframes
    df = ims.merge(gps, how="inner", on="time")
    df = df.merge(vel, how="inner", on="time")

    # %% select only relevant columns
    df = df.loc[:, ["seconds_y", "roll", "pitch", "yaw", "lat", "lon", "alt", "v_east", "v_north", "v_up"]]

    # %% rename columns
    df = df.rename(columns=dict(seconds_y="seconds"))

    # %% IMS pitch is opposite to BAHAMAS pitch, switch signs so that they follow the same convention
    df["pitch"] = -df["pitch"]

    # %% calculate velocity from northerly and easterly part
    df["vel"] = np.sqrt(df["v_east"]**2 + df["v_north"]**2)

    # %% calculate solar zenith and azimuth angle
    dezimal_hours = df["seconds"] / 60 / 60
    year, month, day = df.index.year.values, df.index.month.values, df.index.day.values
    sza = list()
    saa = list()
    for i in tqdm(range(len(year))):
        sza.extend(solar_position.get_sza(dezimal_hours[i], df["lat"][i], df["lon"][i], year[i], month[i], day[i], 1013,
                                          15).flatten())
        # TODO: check SZA function why it returns sometimes an array sometimes a float
        saa.append(solar_position.get_saa(dezimal_hours[i], df["lat"][i], df["lon"][i], year[i], month[i], day[i]))

    # %% add to dataframe
    df["sza"], df["saa"] = sza, saa

    # %% convert to xarray
    ds = df.to_xarray()
    # select only time between TO and TD
    ds = ds.sel(time=slice(to, td))

    # %% create variable and global attributes
    var_attrs = dict(
        seconds=dict(
            long_name='Seconds since start of day',
            units='s',
            comment='Time as recorded by the GPS'),
        roll=dict(
            long_name='Roll angle',
            units='deg',
            comment='Roll angle: positive = left wing up'),
        pitch=dict(
            long_name='Pitch angle',
            units='deg',
            comment='Pitch angle: positive = nose up'),
        yaw=dict(
            long_name='Yaw angle',
            units='deg',
            comment='0 = East, 90 = North, 180 = West, -90 = South, range: -180 to 180'),
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
        sza=dict(
            long_name='Solar zenith angle',
            units='deg',
            comment='Solar zenith angle calculated with refraction correction (pressure: 1013hPa, temperature: 15°C)'),
        saa=dict(
            long_name='Solar azimuth angle',
            units='deg',
            comment='South = 180, range: 0 to 360'),
        v_east=dict(
            long_name="Eastward velocity",
            unit="m s^-1",
            comment="Eastward velocity component as derived from the GPS sensor."),
        v_north=dict(
            long_name="Northward velocity",
            unit="m s^-1",
            comment="Northward velocity component as derived from the GPS sensor."),
        v_up=dict(
            long_name="Upward velocity",
            unit="m s^-1",
            comment="Vertical velocity as derived from the GPS sensor."),
        vel=dict(
            long_name="Ground speed",
            unit="m s^-1",
            comment="Ground speed calculated from northward and eastward velocity component: vel = sqrt(v_north^2 + v_east^2)")
    )

    global_attrs = dict(
        title="Raw attitude angles and GPS position data from the HALO-SMART IMS system",
        campaign_id=f"{campaign.swapcase()}",
        platform_id="HALO",
        instrument_id="SMART",
        version_id="quicklook",
        description="1Hz data from the HALO-SMART IMS system prepared for quicklook creation",
        institution="Leipzig Institute for Meteorology, Leipzig, Germany",
        history=f"created {datetime.strftime(datetime.utcnow(), '%c UTC')}",
        contact="Johannes Röttenbacher, johannes.roettenbacher@uni-leipzig.de",
        PI="André Ehrlich, a.ehrlich@uni-leipzig.de"
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
    outfile = f"HALO-AC3_HALO_gps_ins_{date}_{flight_key}.nc"
    out = os.path.join(outpath, outfile)
    ds.to_netcdf(out, format="NETCDF4_CLASSIC", encoding=encoding)
    print(f"Saved {out}")
