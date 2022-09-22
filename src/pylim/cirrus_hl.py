#!/usr/bin/env python
"""General information about the CIRRUS-HL campaign

- lookup dictionary to look up which spectrometer belongs to which inlet, which inlet is measuring up or downward irradiance and which filename part belongs to which channel
- dictionary relating flight with flight number
- BACARDI offsets
- dictionary which relates each measurement with the date of the transfer calibration for that measurement.
- take off and landing times according to BAHAMAS
- gopro local time and BAHAMAS time offset
- stop over locations for each double flight
- coordinates of airports
- radiosonde stations
- specific flight sections
- flight hours
- ozone files

*author*: Johannes Röttenbacher
"""
import numpy as np
import pandas as pd
from pandas import Timestamp as Ts
from pandas import Timedelta as Td

# inlet to spectrometer mapping and inlet to direction mapping and measurement to spectrometer mapping
smart_lookup = dict(ASP06_J3="PGS_5_(ASP_06)", ASP06_J4="VIS_6_(ASP_06)", ASP06_J5="PGS_6_(ASP_06)",
                    ASP06_J6="VIS_7_(ASP_06)", ASP07_J3="PGS_4_(ASP_07)", ASP07_J4="VIS_8_(ASP_07)",
                    J3="dw", J4="dw", J5="up", J6="up",
                    Fdw_SWIR="ASP06_J3", Fdw_VNIR="ASP06_J4", Fup_SWIR="ASP06_J5", Fup_VNIR="ASP06_J6",
                    Iup_SWIR="ASP07_J3", Iup_VNIR="ASP07_J4",
                    Fdw="VN05", Fup="VN11",  # inlet names
                    irradiance_standard="FEL-1587"  # irradiance standard used for calibration
                    )

flight_numbers = dict(Flight_20210624a="F01", Flight_20210625a="F02", Flight_20210626a="F03", Flight_20210628a="F04",
                      Flight_20210629a="F05", Flight_20210629b="F06", Flight_20210701a="F07", Flight_20210705a="F08",
                      Flight_20210705b="F09", Flight_20210707a="F10", Flight_20210707b="F11", Flight_20210708a="F12",
                      Flight_20210712a="F13", Flight_20210712b="F14", Flight_20210713a="F15", Flight_20210715a="F16",
                      Flight_20210715b="F17", Flight_20210719a="F18", Flight_20210719b="F19", Flight_20210721a="F20",
                      Flight_20210721b="F21", Flight_20210723a="F22", Flight_20210728a="F23", Flight_20210729a="F24")
# BACARDI pitch and roll offset of upward facing sensor
roll_offset = 0.3
pitch_offset = 2.55

# transfer calibrations to each flight
transfer_calibs = dict(Flight_20210624a="20210625", Flight_20210625a="20210625", Flight_20210626a="20210627",
                       Flight_20210628a="20210627", Flight_20210629a="20210630", Flight_20210629b="20210630",
                       Flight_20210701a="20210702", Flight_20210705a="20210706", Flight_20210705b="20210706",
                       Flight_20210707a="20210708", Flight_20210707b="20210708", Flight_20210708a="20210708",
                       Flight_20210712a="20210711", Flight_20210712b="20210711", Flight_20210713a="20210713",
                       Flight_20210715a="20210714", Flight_20210715b="20210714", Flight_20210719a="20210720",
                       Flight_20210719b="20210720", Flight_20210721a="20210720", Flight_20210721b="20210720",
                       Flight_20210723a="20210720", Flight_20210728a="20210720", Flight_20210729a="20210720")

take_offs_landings = dict(Flight_20210624a=(Ts(2021, 6, 24, 11, 58, 59), Ts(2021, 6, 24, 15, 29, 6)),
                          Flight_20210625a=(Ts(2021, 6, 25, 11, 3, 20), Ts(2021, 6, 25, 16, 50, 58)),
                          Flight_20210626a=(Ts(2021, 6, 26, 7, 59, 40), Ts(2021, 6, 26, 15, 27, 35)),
                          Flight_20210628a=(Ts(2021, 6, 28, 8, 7, 6), Ts(2021, 6, 28, 15, 40, 54)),
                          Flight_20210629a=(Ts(2021, 6, 29, 7, 4, 42), Ts(2021, 6, 29, 14, 58, 19)),
                          Flight_20210629b=(Ts(2021, 6, 29, 15, 57, 30), Ts(2021, 6, 29, 18, 47, 39)),
                          Flight_20210701a=(Ts(2021, 7, 1, 10, 3, 40), Ts(2021, 7, 1, 15, 10, 56)),
                          Flight_20210705a=(Ts(2021, 7, 5, 6, 21, 54), Ts(2021, 7, 5, 10, 37, 36)),
                          Flight_20210705b=(Ts(2021, 7, 5, 11, 36, 12), Ts(2021, 7, 5, 18, 42, 0)),
                          Flight_20210707a=(Ts(2021, 7, 7, 6, 29, 42), Ts(2021, 7, 7, 12, 46, 15)),
                          Flight_20210707b=(Ts(2021, 7, 7, 13, 50, 52), Ts(2021, 7, 7, 18, 38, 36)),
                          Flight_20210708a=(Ts(2021, 7, 8, 13, 12, 39), Ts(2021, 7, 8, 18, 3, 58)),
                          Flight_20210712a=(Ts(2021, 7, 12, 6, 38, 25), Ts(2021, 7, 12, 11, 36, 29)),
                          Flight_20210712b=(Ts(2021, 7, 12, 12, 27, 10), Ts(2021, 7, 12, 18, 45, 28)),
                          Flight_20210713a=(Ts(2021, 7, 13, 13, 7, 3), Ts(2021, 7, 13, 18, 21, 41)),
                          Flight_20210715a=(Ts(2021, 7, 15, 6, 23, 55), Ts(2021, 7, 15, 10, 30, 56)),
                          Flight_20210715b=(Ts(2021, 7, 15, 12, 2, 19), Ts(2021, 7, 15, 18, 18, 10)),
                          Flight_20210719a=(Ts(2021, 7, 19, 6, 21, 5), Ts(2021, 7, 19, 12, 43, 24)),
                          Flight_20210719b=(Ts(2021, 7, 19, 13, 47, 4), Ts(2021, 7, 19, 18, 14, 8)),
                          Flight_20210721a=(Ts(2021, 7, 21, 5, 55, 57), Ts(2021, 7, 21, 10, 56, 10)),
                          Flight_20210721b=(Ts(2021, 7, 21, 11, 56, 19), Ts(2021, 7, 21, 17, 5, 19)),
                          Flight_20210723a=(Ts(2021, 7, 23, 14, 27, 58), Ts(2021, 7, 23, 22, 30, 19)),
                          Flight_20210728a=(Ts(2021, 7, 28, 11, 5, 27), Ts(2021, 7, 28, 17, 54, 52)),
                          Flight_20210729a=(Ts(2021, 7, 29, 11, 45, 48), Ts(2021, 7, 29, 16, 2, 5)))

# GoPro time stamp in LT
gopro_lt = dict(Flight_20210627=True, Flight_20210628=True, Flight_20210629=True, Flight_20210701=True,
                Flight_20210703=True, Flight_20210705=True, Flight_20210707=True, Flight_20210708=True,
                Flight_20210710=True, Flight_20210712=True, Flight_20210713=True)

# GoPro time offsets from bahamas in seconds
gopro_offsets = dict(Flight_20210625=-1, Flight_20210626=6, Flight_20210628=4, Flight_20210705=47, Flight_20210707=59,
                     Flight_20210708=66, Flight_20210712=90, Flight_20210713=98, Flight_20210719=0, Flight_20210721=-1,
                     Flight_20210723=14, Flight_20210728=14)

# stop over locations for each flight
stop_over_locations = dict(Flight_20210629a="Bergen", Flight_20210629b="Bergen",
                           Flight_20210705a="Kiruna", Flight_20210705b="Kiruna",
                           Flight_20210707a="Keflavik", Flight_20210707b="Keflavik",
                           Flight_20210712a="Keflavik", Flight_20210712b="Keflavik",
                           Flight_20210715a="Santiago", Flight_20210715b="Santiago",
                           Flight_20210719a="Bergen", Flight_20210719b="Bergen",
                           Flight_20210721a="Santiago", Flight_20210721b="Santiago")

# coordinates for map plots (lon, lat)
coordinates = dict(EDMO=(11.28, 48.08), Keflavik=(-22.6307, 63.976), Kiruna=(20.336, 67.821), Santiago=(-8.418, 42.898),
                   Bergen=(5.218, 60.293), Torshavn=(-6.76, 62.01), Muenchen_Oberschleissheim=(11.55, 48.25),
                   Meiningen=(10.38, 50.56), Lerwick=(-1.18, 60.13), Ittoqqortoormiit=(-21.95, 70.48),
                   Tasiilaq=(-37.63, 65.60), Norderney=(7.15, 53.71))

# radiosonde stations
radiosonde_stations = ["Torshavn_06011", "Muenchen_Oberschleissheim_10868", "Meiningen_10548", "Lerwick_03005",
                       "Ittoqqortoormiit_04339", "Tasiilaq_04360", "Norderney_10113"]

# above, below and in cloud intervals for each flight
flight_sections = dict(
    Flight_20210629a=dict(
        above=[(Ts(2021, 6, 29, 11, 54), Ts(2021, 6, 29, 12, 5)), (Ts(2021, 6, 19, 12, 36), Ts(2021, 6, 29, 14, 20))],
        below=[(Ts(2021, 6, 29, 10, 10), Ts(2021, 6, 29, 10, 14))],
        inside=[(Ts(2021, 6, 29, 10, 15), Ts(2021, 6, 29, 11, 54)), (Ts(2021, 6, 29, 12, 8), Ts(2021, 6, 29, 12, 25))],
        staircase=dict(
            start_dts=pd.to_datetime(np.array(['2021-06-29T09:43:41.700000000', '2021-06-29T10:10:52.000000000',
                                               '2021-06-29T10:25:24.100000000', '2021-06-29T10:47:18.400000000',
                                               '2021-06-29T11:08:56.200000000', '2021-06-29T11:32:42.300000000',
                                               '2021-06-29T11:56:42.700000000'])),
            end_dts=pd.to_datetime(np.array(['2021-06-29T10:09:42.700000000', '2021-06-29T10:23:56.800000000',
                                             '2021-06-29T10:45:49.400000000', '2021-06-29T11:07:27.200000000',
                                             '2021-06-29T11:31:14.100000000', '2021-06-29T11:54:31.600000000',
                                             '2021-06-29T12:05:04.700000000'])))),
    Flight_20210719a=dict(
        above=[(Ts(2021, 7, 19, 10, 20), Ts(2021, 7, 19, 11, 42))],
        below=[(Ts(2021, 7, 19, 8, 25), Ts(2021, 7, 19, 8, 33)), (Ts(2021, 7, 19, 8, 43), Ts(2021, 7, 19, 8, 48))],
        inside=[(Ts(2021, 7, 19, 8, 35), Ts(2021, 7, 19, 8, 42)), (Ts(2021, 7, 19, 8, 48), Ts(2021, 7, 19, 10, 9))]
    ))

# flight hours
flight_hours = [Td("05:47:00"), Td("07:27:00"), Td("07:33:00"), Td("07:53:00"), Td("02:50:00"), Td("05:07:00"),
                Td("04:15:00"), Td("07:05:00"), Td("06:16:00"), Td("04:47:00"), Td("04:51:00"), Td("04:58:00"),
                Td("06:18:00"), Td("05:14:00"), Td("04:07:00"), Td("06:15:00"), Td("06:22:00"), Td("04:27:00"),
                Td("05:00:00"), Td("05:08:00"), Td("08:02:00"), Td("06:49:00"), Td("04:16:00")]

# ozone sonde stations
ozone_files = dict(Flight_20210629a="sc210624.b11")