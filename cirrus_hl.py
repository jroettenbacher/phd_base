#!/usr/bin/env python
"""Collects information about the cirrus-hl campaign
author: Johannes Röttenbacher
"""
# inlet to spectrometer mapping and inlet to direction mapping and measurement to spectrometer mapping
lookup = dict(ASP06_J3="PGS_5_(ASP_06)", ASP06_J4="VIS_6_(ASP_06)", ASP06_J5="PGS_6_(ASP_06)",
              ASP06_J6="VIS_7_(ASP_06)", ASP07_J3="PGS_4_(ASP_07)", ASP07_J4="VIS_8_(ASP_07)",
              J3="dw", J4="dw", J5="up", J6="up",
              Fdw_SWIR="ASP06_J3", Fdw_VNIR="ASP06_J4", Fup_SWIR="ASP06_J5", Fup_VNIR="ASP06_J6",
              Iup_SWIR="ASP07_J3", Iup_VNIR="ASP07_J4")

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
                   Bergen=(5.218, 60.293), Torshavn=(-6.76, 62.01), München_Oberschleissheim=(11.55, 48.25),
                   Meiningen=(10.38, 50.56), Lerwick=(-1.18, 60.13), Ittoqqortoormiit=(-21.95, 70.48),
                   Tasiilaq=(-37.63, 65.60))