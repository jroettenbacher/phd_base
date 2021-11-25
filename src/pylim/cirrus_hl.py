#!/usr/bin/env python
"""General information about the CIRRUS-HL campaign
author: Johannes Röttenbacher
"""

from pandas import Timestamp as Ts
from pandas import Timedelta as Td

# inlet to spectrometer mapping and inlet to direction mapping and measurement to spectrometer mapping
lookup = dict(ASP06_J3="PGS_5_(ASP_06)", ASP06_J4="VIS_6_(ASP_06)", ASP06_J5="PGS_6_(ASP_06)",
              ASP06_J6="VIS_7_(ASP_06)", ASP07_J3="PGS_4_(ASP_07)", ASP07_J4="VIS_8_(ASP_07)",
              J3="dw", J4="dw", J5="up", J6="up",
              Fdw_SWIR="ASP06_J3", Fdw_VNIR="ASP06_J4", Fup_SWIR="ASP06_J5", Fup_VNIR="ASP06_J6",
              Iup_SWIR="ASP07_J3", Iup_VNIR="ASP07_J4",
              Fdw="VN05", Fup="VN11",  # inlet names
              irradiance_standard="FEL-1587"  # irradiance standard used for calibration
              )

# BACARDI pitch and roll offset of upward facing sensor
roll_offset = 0.3
pitch_offset = 2.55

# transfer calibrations to each flight
transfer_calibs = dict(Flight_20210624a="20210616", Flight_20210625a="20210625", Flight_20210626a="20210627",
                       Flight_20210628a="20210629", Flight_20210629a="20210630", Flight_20210629b="20210630",
                       Flight_20210701a="20210702", Flight_20210705a="20210706", Flight_20210705b="20210706",
                       Flight_20210707a="20210708", Flight_20210707b="20210708", Flight_20210708a="20210708",
                       Flight_20210712a="20210711", Flight_20210712b="20210711", Flight_20210713a="20210713",
                       Flight_20210715a="20210714", Flight_20210715b="20210714", Flight_20210719a="20210720",
                       Flight_20210719b="20210720", Flight_20210721a="20210722", Flight_20210721b="20210722",
                       Flight_20210723a="20210725", Flight_20210728a="20210729", Flight_20210729a="20210730")

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
                   Tasiilaq=(-37.63, 65.60))

# radiosonde stations
radiosonde_stations = ["Torshavn_06011", "Muenchen_Oberschleissheim_10868", "Meiningen_10548", "Lerwick_03005",
                       "Ittoqqortoormiit_04339", "Tasiilaq_04360"]

# above, below and in cloud intervals for each flight
flight_sections = dict(
    Flight_20210629a=dict(
        above=[(Ts(2021, 6, 29, 11, 54), Ts(2021, 6, 29, 12, 5)), (Ts(2021, 6, 19, 12, 36), Ts(2021, 6, 29, 14, 20))],
        below=[(Ts(2021, 6, 29, 10, 10), Ts(2021, 6, 29, 10, 14))],
        inside=[(Ts(2021, 6, 29, 10, 15), Ts(2021, 6, 29, 11, 54)), (Ts(2021, 6, 29, 12, 8), Ts(2021, 6, 29, 12, 25))]),
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
