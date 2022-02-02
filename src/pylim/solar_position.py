#!/usr/bin/env python
"""Functions calculating the solar position

**author:** Hanno Müller, Johannes Röttenbacher
"""
import numpy as np


def dec(julian, year):
    """
    Declination calculated after Michalsky, J.  1988.
    **Reference**: Michalsky, J.  1988. The Astronomical Almanac's algorithm for approximate solar position (1950-2050).  Solar Energy 40 (3), pp. 227-235.

    Args:
        julian: julian day (day of year) as calculated with julian2.pro
        year: The year (e.g., 2020)

    Returns:
        declination in (deg)

    Examples:
        >>> dec(303, 2020)
        -13.52916587749051

    """
    rad = np.pi / 180.

    # get the current julian date (actually add 2,400,000 for jd)
    delta = year - 1949.
    leap = int(delta / 4.)
    julian_zero = 32916.5 + delta * 365. + leap + julian
    # 1st no. is mid. 0 jan 1949 minus 2.4e6 leap=leap days since 1949
    # the last yr of century is not leap yr unless divisible by 400
    if ((year % 100.) == 0.0) and ((year % 400.) != 0.0): julian_zero = julian_zero - 1.

    # calculate ecliptic coordinates
    time = julian_zero - 51545.0

    # force mean longitude between 0 and 360 degs
    mnlong = 280.460 + .9856474 * time
    mnlong = (mnlong % 360.)
    if mnlong <= 0.: mnlong = mnlong + 360.

    # mean anomaly in radians between 0 and 2*pi
    mnanom = 357.528 + .9856003 * time
    mnanom = (mnanom % 360.)
    if mnanom <= 0.: mnanom = mnanom + 360.
    mnanom = mnanom * rad

    # compute the ecliptic longitude and obliquity of ecliptic in radians
    eclong = mnlong + 1.915 * np.sin(mnanom) + .020 * np.sin(2. * mnanom)
    eclong = (eclong % 360.)
    if eclong <= 0.: eclong = eclong + 360.
    oblqec = 23.439 - .0000004 * time
    eclong = eclong * rad
    oblqec = oblqec * rad

    dec = np.arcsin(np.sin(oblqec) * np.sin(eclong)) / rad

    return dec


def get_saa(t0, lat, lon, year, month, day):
    """
    Calculates the solar azimuth angles.
    Requires function julian_day,local_time,dec
    Args:
        t0: Time in UTC (in decimal hours, i.e. 10.5 for 10h30min)
        lat: Latitude (North positive)
        lon: Longitude (East positive)
        year: The year  (e.g. 2020)
        month: The month (1-12)
        day: The day (1-31)

    Returns:
        The solar azimuth angle (deg)

    Examples:
        >>> azimuth = get_saa(11, 51.34, 12.376, 2022, 2, 2)
        >>> azimuth
        173.75177751099122
    """
    julian = julian_day(year, month, day, t0)
    t_loc = local_time(julian, t0, lon)
    tau = (12. - t_loc) * 15.
    height = np.arcsin(
        np.cos(np.pi / 180. * lat) * np.cos(np.pi / 180. * dec(julian, year)) * np.cos(np.pi / 180. * tau) + np.sin(
            np.pi / 180. * lat) * np.sin(np.pi / 180. * dec(julian, year)))
    azimuth = (np.sin(height) * np.sin(np.pi / 180. * lat) - np.sin(np.pi / 180. * dec(julian, year))) / (
            np.cos(height) * np.cos(np.pi / 180. * lat))
    # check that the azimuth is below else set it to 1
    azimuth = azimuth if azimuth < 1 else 1
    azimuth = np.arccos(azimuth)
    # check if it is before local noon, if yes switch sign of azimuth
    azimuth = azimuth if t_loc > 12 else -azimuth
    # tl12 = np.where(t_loc < 12.)
    # if len(tl12[0]) > 0: azimuth[tl12[0]] = -azimuth[tl12[0]]
    azimuth += np.pi

    azimuth = azimuth * 180. / np.pi

    return azimuth


def get_sza(t0, lat, lon, year, month, day, pres, temp):
    """
    Calculates the solar zenith angle
    Requires function julian_day,local_time,dec,refract

    Copyright by Sebastian Schmidt

    changes by Andre Ehrlich:
    - new declination parametrization after Michalsky, J.  1988. The Astronomical Almanac's algorithm for approximate solar position (1950-2050).  Solar Energy 40 (3), pp. 227-235.
    - refraction correction algorithm from Meeus,1991
    - check: for SZA >95 no refraction correction

    Args:
        t0: Time in UTC (in decimal hours, i.e. 10.5 for 10h30min)
        lat: Latitude (North positive)
        lon: Longitude (East positive)
        year: The year  (e.g. 2020)
        month: The month (1-12)
        day: The day (1-31)
        pres: Surface Pressure in hPa (for refraction correction)
        temp: Surface Temperature in deg C (for refraction correction)

    Returns:
        The solar zenith angle (deg)

    """
    julian = julian_day(year, month, day, t0)
    t_loc = local_time(julian, t0, lon)
    tau = (12. - t_loc) * 15.
    height = np.arcsin(
        np.cos(np.pi / 180. * lat) * np.cos(np.pi / 180. * dec(julian, year)) * np.cos(np.pi / 180. * tau) + np.sin(
            np.pi / 180. * lat) * np.sin(np.pi / 180. * dec(julian, year)))
    # correction for refraction
    # refcor = fltarr(n_elements(height))
    refcor = np.zeros(height.size)

    h1 = np.where(height >= -0.087)  # corrects only for angles > -5�
    # if len(h1[0]) > 0: refcor[h1[0]]=refract(height[h1[0]],pres,temp)
    if len(h1[0]) > 0: refcor = refract(height, pres, temp)

    sza = 90. - (height + refcor) * 180. / np.pi

    return sza


def julian_day(year, month, day, t0):
    """
    Returns the day of the year (julian_day) for a given date. Jan 1 = 1, Feb 1 = 32 etc.
    Considers leap years and the time of day.

    Args:
        year: The year (e.g. 2020)
        month: The month (1-12)
        day: The day (1-31)
        t0: Time in UTC (dezimal hours)

    Returns:
        A value between 1 and 366, corresponding to the day of the year of year/month/day.

    Examples:
        >>> jd = julian_day(2020, 2, 14, 6.8)
        >>> jd
        45.28333333333333


    """
    year = int(year)
    month = int(month)
    day = int(day)
    # leap year -> s=1:
    s = 0
    if year % 4 == 0: s = 1
    if s == 1 and year % 100 == 0 and year % 400 != 0: s = 0
    if month == 1: jd = day
    if month == 2: jd = day + 31
    if month == 3: jd = day + 31 + 28 + s
    if month == 4: jd = day + 31 + 28 + s + 31
    if month == 5: jd = day + 31 + 28 + s + 31 + 30
    if month == 6: jd = day + 31 + 28 + s + 31 + 30 + 31
    if month == 7: jd = day + 31 + 28 + s + 31 + 30 + 31 + 30
    if month == 8: jd = day + 31 + 28 + s + 31 + 30 + 31 + 30 + 31
    if month == 9: jd = day + 31 + 28 + s + 31 + 30 + 31 + 30 + 31 + 31
    if month == 10: jd = day + 31 + 28 + s + 31 + 30 + 31 + 30 + 31 + 31 + 30
    if month == 11: jd = day + 31 + 28 + s + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31
    if month == 12: jd = day + 31 + 28 + s + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30
    #                      J  (F+s)M  A  M  J  J  A  S  O  N
    jd = jd + t0 / 24.  # consider time of day (important for the hourly change of declination which can be large in spring/autumn)

    return jd


def local_time(julian, t0, lon):
    """
    Calculates the local solar time by accounting for
    - longitude
    - time equation (two versions are implemented, one commented, both are almost identical)

    Args:
        julian: julian day (day of year) as calculated with julian2.pro
        t0: Time in UTC (in decimal hours, i.e. 10.5 for 10h30min)
        lon: Longitude (East positive)

    Returns:
        Local solar time (in decimal hours, i.e. 10.5 for 10h30min)

    Examples:
        >>> solar_local_time = local_time(265, 5.2, 14)
        >>> solar_local_time
        6.2521447402113886

    """
    mean_local_time_min = t0 * 60. + 4. * lon
    fac = 0.0132 * 0.5 + 7.3525 * np.cos(2. * np.pi * julian / 365. + 1.4989) + 9.9359 * np.cos(
        2. * 2. * np.pi * julian / 365. + 1.9006) + 0.3387 * np.cos(3. * 2. * np.pi * julian / 365. + 1.8360)
    corrected_local_time = (mean_local_time_min + fac) / 60.

    return corrected_local_time


def refract(elv, pres, temp):
    """
    Correction of the solar elevation angle for refraction
    **Reference**: J.Meeus, Astronomical Algorithms, 1991, pp102

    Args:
        elv: solar elevation (rad)
        pres: station pressure (hPa)
        temp: station temperature (°C)

    Returns:
        Refraction correction offset (rad) which needs to be applied after-wards: corrected_elv = elv + refcor

    Examples:
        >>> refcor = refract(0.6, 850, 5)
        >>> refcor
        0.0003679487426608174

    """
    h = elv / np.pi * 180.
    refcor = 1.02 / np.tan((h + 10.3 / (h + 5.11)) * 2 * np.pi / 360.0) * (pres / 1010.0) * 283.0 / (273.0 + temp)
    refcor = refcor / 60.0 * np.pi / 180.0

    return refcor
