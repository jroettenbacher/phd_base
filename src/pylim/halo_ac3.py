#!/usr/bin/env python
"""Background information and lookup dictionaries for the HALO-AC3 campaign

author: Johannes Röttenbacher
"""

from pandas import Timestamp as Ts

smart_lookup = dict(ASP06_J5="PGS_6_(ASP_06)", ASP06_J6="VIS_7_(ASP_06)",  # spectrometers
                    J5="dw", J6="dw",  # direction of channel
                    Fdw_SWIR="ASP06_J5", Fdw_VNIR="ASP06_J6",  # direction, property of channel
                    Fdw="VN11",  # inlet name
                    irradiance_standard="FEL-1587"  # irradiance standard used for calibration
                    )

# coordinates for map plots (lon, lat)
coordinates = dict(EDMO=(11.28, 48.08), Keflavik=(-22.6307, 63.976), Kiruna=(20.336, 67.821), Santiago=(-8.418, 42.898),
                   Bergen=(5.218, 60.293), Torshavn=(-6.76, 62.01), Muenchen_Oberschleissheim=(11.55, 48.25),
                   Meiningen=(10.38, 50.56), Lerwick=(-1.18, 60.13), Ittoqqortoormiit=(-21.95, 70.48),
                   Tasiilaq=(-37.63, 65.60), Leipzig=(12.39, 51.33), Jülich=(6.37, 50.92),
                   Longyearbyen=(15.47, 78.25), Norderney=(7.15, 53.71))

# radiosonde stations
radiosonde_stations = ["Torshavn_06011", "Muenchen_Oberschleissheim_10868", "Meiningen_10548", "Lerwick_03005",
                       "Ittoqqortoormiit_04339", "Tasiilaq_04360", "Norderney_10113"]

# transfer calibrations to each flight
transfer_calibs = dict(EMV="20220222", RF00="20220222")

take_offs_landings = dict(EMV=(Ts(2022, 2, 21, 10, 25), Ts(2022, 2, 21, 13, 42)),
                          RF00=(Ts(2022, 2, 25, 7, 30), Ts(2022, 2, 25, 12, 13)))

# GoPro time offset to BAHAMAS
gopro_offsets = dict(EMV=-2, RF00=24)
