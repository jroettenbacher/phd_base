#!/usr/bin/env python
"""Background information and lookup dictionaries for the HALO-AC3 campaign

author: Johannes Röttenbacher
"""

from pandas import Timestamp as Ts

smart_lookup = dict(ASP06_J3="PGS_5_(ASP_06)", ASP06_J4="VIS_6_(ASP_06)", ASP06_J5="PGS_6_(ASP_06)",
                    ASP06_J6="VIS_7_(ASP_06)",  # spectrometers
                    J3="dw", J4="dw",  # direction of channel
                    # TODO: Change J5 and J6 to J3 and J4 -> look out for breaking stuff
                    # The problem is the lab calibration
                    Fdw_SWIR="ASP06_J3", Fdw_VNIR="ASP06_J4",  # direction, property of channel
                    Fdw="VN05",  # inlet name
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
transfer_calibs = dict(EMV="20220222", RF00="20220222", RF01="20220313", RF02="20220313", RF03="20220314",
                       RF04="20220315", RF05="20220316", RF06="20220318", RF07="20220320", RF08="20220321",
                       RF09="20220328", RF10="20220329", RF11="20220330", RF12="20220401", RF13="20220403",
                       RF14="20220405")

take_offs_landings = dict(EMV=(Ts(2022, 2, 21, 10, 25), Ts(2022, 2, 21, 13, 42)),
                          RF00=(Ts(2022, 2, 25, 7, 31, 6), Ts(2022, 2, 25, 12, 12, 27)),
                          RF01=(Ts(2022, 3, 11, 13, 19, 28), Ts(2022, 3, 11, 16, 22, 31)),
                          RF02=(Ts(2022, 3, 12, 8, 22, 6), Ts(2022, 3, 12, 16, 44, 7)),
                          RF03=(Ts(2022, 3, 13, 8, 4, 31), Ts(2022, 3, 13, 16, 52, 50)),
                          RF04=(Ts(2022, 3, 14, 8, 45, 34), Ts(2022, 3, 14, 17, 19, 8)),
                          RF05=(Ts(2022, 3, 15, 9, 4, 50), Ts(2022, 3, 15, 17, 46, 40)),
                          RF06=(Ts(2022, 3, 16, 8, 58, 18), Ts(2022, 3, 16, 18, 27, 20)),
                          RF07=(Ts(2022, 3, 20, 7, 58, 48), Ts(2022, 3, 20, 17, 7, 5)),
                          RF08=(Ts(2022, 3, 21, 8, 48, 45), Ts(2022, 3, 21, 16, 35, 38)),
                          RF09=(Ts(2022, 3, 28, 8, 36, 38), Ts(2022, 3, 28, 16, 6, 6)),
                          RF10=(Ts(2022, 3, 29, 7, 53, 9), Ts(2022, 3, 29, 16, 27, 6)),
                          RF11=(Ts(2022, 3, 30, 7, 56, 42), Ts(2022, 3, 30, 16, 20, 55)),
                          RF12=(Ts(2022, 4, 1, 7, 29, 57), Ts(2022, 4, 1, 15, 40, 34)),
                          RF13=(Ts(2022, 4, 4, 7, 25, 58), Ts(2022, 4, 4, 16, 18, 50)),
                          RF14=(Ts(2022, 4, 7, 8, 32, 57), Ts(2022, 4, 7, 16, 14, 40)))

# GoPro time offset to BAHAMAS
gopro_offsets = dict(EMV=-2, RF00=24, RF01=0, RF02=123, RF03=127, RF04=0, RF05=21, RF06=-2, RF07=26, RF08=28, RF09=75,
                     RF10=79, RF11=85, RF12=97, RF13=5, RF14=24)
