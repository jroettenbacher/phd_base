#!/usr/bin/env python
"""General meteorological formulas

*author*: Johannes Röttenbacher
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Union


def relative_humidity_water_to_relative_humidity_ice(relative_humidity_water: Union[float, np.ndarray, list],
                                                     temperature: Union[float, np.ndarray, list],
                                                     version: str = "huang"):
    """
    Convert the relative humidity over water to relative humidity over ice using either the formulas by
    :cite:t:`Huang2018` or by :cite:t:`Alduchov1996`.

    .. math::

        RH_{ice} = \\frac{RH_{water} * e_{s,w}}{e_{s,i}}

    with :math:`e_{s,w}` the saturation vapor pressure of water and :math:`e_{s,i}` the saturation vapor pressure of ice:

    .. math::

        e_{s,w} = \\frac{\\exp\\left(34.494 - \\frac{4924.99}{t + 237.1}\\right)}{(t + 105)^{1.157}} (t > 0°C)

        e_{s,i} = \\frac{\\exp\\left(43.494 - \\frac{6545.8}{t + 278}\\right)}{(t + 868)^{2}} (t \\le 0°C)

    with :math:`t` being the temperature in °C. When version is ``alduchov`` the following equations are used:

    .. math::

        e_{s,w} = 6.1094 * \\exp\\left(\\frac{17.625 * t}{243.04 + t}\\right)

        e_{s,i} = 6.1121 * \\exp\\left(\\frac{22.587 * t}{273.86 + t}\\right)

    with :math:`t` the temperature in °C.

    Args:
        relative_humidity_water: ambient relative humidity in percent
        temperature: ambient temperature in °C
        version: which formulas to use for calculating the saturation vapor pressure of water and ice ('huang' or 'alduchov')

    Returns: relative humidity over ice

    """
    if version == "alduchov":
        saturation_vapour_pressure_water = 6.1094 * np.exp(17.625 * temperature / (243.04 + temperature))
        saturation_vapour_pressure_ice = 6.1121 * np.exp(22.587 * temperature / (273.86 + temperature))
    else:
        saturation_vapour_pressure_water = np.exp(34.494 - (4924.99 / (temperature + 237.1))) / (
                    temperature + 105) ** 1.57
        saturation_vapour_pressure_ice = np.exp(43.494 - (6545.8 / (temperature + 278))) / (temperature + 868) ** 2

    relative_humidity_ice = relative_humidity_water * saturation_vapour_pressure_water / saturation_vapour_pressure_ice

    return relative_humidity_ice


def barometric_height_old(p_high, p_low, T_high):
    g_geo = 9.81  # earth acceleration in m/s^2
    R = 8.314  # universelle Gaskonstante
    molar_mass_air = 0.02896  # kg/mol
    delta_h = -np.log(p_high / p_low) * R * T_high / (molar_mass_air * g_geo)
    return delta_h


def barometric_height(pressure_profile, temperature_profile):
    """
    Calculate the barometric height from a pressure and temperature profile.

    .. math::

        \\delta h = \\frac{\\log{\frac{p(h_1)}{p(h_0)}} * R * T(h_1)}{M * g}

    Args:
        pressure_profile: pressure profile (Pa)
        temperature_profile: temperature profile (K)

    Returns: barometric height (m)

    @author: Hanno Müller, Johannes Röttenbacher
    """
    assert len(pressure_profile) == len(temperature_profile), \
        "Pressure and Temperature profile have to be of same length!"
    g_geo = 9.81  # earth acceleration in m/s^2
    R = 8.314  # universal gas constant
    molar_mass_air = 0.02896  # kg/mol
    # check if pressure profile is ascending (surface at index 0)
    if pressure_profile[0] > pressure_profile[-1]:
        pass
    else:
        pressure_profile = np.flip(pressure_profile)

    # check if temperature profile is ascending (surface at index 0)
    if temperature_profile[0] > temperature_profile[-1]:
        pass
    else:
        temperature_profile = np.flip(temperature_profile)

    levels = len(pressure_profile)
    barometric_height = np.zeros(levels)
    p_low = pressure_profile[0]  # set p_low to surface pressure for first iteration
    for j in range(1, levels):
        p_high = pressure_profile[j]
        t_high = temperature_profile[j]
        delta_h = -np.log(p_high / p_low) * R * t_high / (molar_mass_air * g_geo)
        barometric_height[j] = barometric_height[j - 1] + delta_h
        p_low = pressure_profile[j]

    # replace top of atmosphere value (inf) with nan
    barometric_height = np.where(barometric_height == np.inf, np.nan, barometric_height)
    return barometric_height


def barometric_height_simple(pressure):
    """
    Calculate the barometric height assuming a constant temperature of 0°C

    Args:
        pressure: pressure profile (Pa)

    Returns: barometric_height

    """
    q_air = 1.292  # dry air density at 0°C in kg/m3
    g_geo = 9.81  # earth acceleration in m/s^2
    # check if array is ascending from surface to top of atmosphere
    if pressure[0] > pressure[-1]:
        surface_pressure = pressure[0]
    else:
        pressure = np.flip(pressure)
        surface_pressure = pressure[0]
    barometric_height = -surface_pressure * np.log(pressure / surface_pressure) / (q_air * g_geo)
    # replace TOA height (calculated as infinity) with nan
    barometric_height[barometric_height == np.inf] = np.nan

    return barometric_height


if __name__ == "__main__":
    pressure_half = np.array([0.00000000e+00, 2.00036502e+00, 3.10224104e+00, 4.66608381e+00,
                              6.82797718e+00, 9.74696636e+00, 1.36054239e+01, 1.86089306e+01,
                              2.49857178e+01, 3.29857101e+01, 4.28792419e+01, 5.49554634e+01,
                              6.95205765e+01, 8.68958817e+01, 1.07415741e+02, 1.31425507e+02,
                              1.59279404e+02, 1.91338562e+02, 2.27968948e+02, 2.69539581e+02,
                              3.16420746e+02, 3.68982361e+02, 4.27592499e+02, 4.92616028e+02,
                              5.64413452e+02, 6.43339905e+02, 7.29744141e+02, 8.23967834e+02,
                              9.26344910e+02, 1.03720117e+03, 1.15685364e+03, 1.28561035e+03,
                              1.42377014e+03, 1.57162292e+03, 1.72944897e+03, 1.89751929e+03,
                              2.07609595e+03, 2.26543164e+03, 2.46577051e+03, 2.67734814e+03,
                              2.90039136e+03, 3.13511938e+03, 3.38174365e+03, 3.64046826e+03,
                              3.91149048e+03, 4.19493066e+03, 4.49081738e+03, 4.79914941e+03,
                              5.11989502e+03, 5.45299072e+03, 5.79834473e+03, 6.15607422e+03,
                              6.52694678e+03, 6.91187061e+03, 7.31187304e+03, 7.72810139e+03,
                              8.16183637e+03, 8.61453271e+03, 9.08781047e+03, 9.58292831e+03,
                              1.01006815e+04, 1.06418845e+04, 1.12073701e+04, 1.17979909e+04,
                              1.24146231e+04, 1.30581674e+04, 1.37295223e+04, 1.44296190e+04,
                              1.51594137e+04, 1.59198552e+04, 1.67119243e+04, 1.75366231e+04,
                              1.83949558e+04, 1.92879478e+04, 2.02166434e+04, 2.11820925e+04,
                              2.21853650e+04, 2.32275472e+04, 2.43097385e+04, 2.54330433e+04,
                              2.65985942e+04, 2.78075270e+04, 2.90609874e+04, 3.03601470e+04,
                              3.17061728e+04, 3.31002532e+04, 3.45435863e+04, 3.60373793e+04,
                              3.75828546e+04, 3.91812344e+04, 4.08337637e+04, 4.25416791e+04,
                              4.43062400e+04, 4.61287056e+04, 4.80103531e+04, 4.99524554e+04,
                              5.19512909e+04, 5.39971732e+04, 5.60759427e+04, 5.81777296e+04,
                              6.02922922e+04, 6.24092042e+04, 6.45180831e+04, 6.66087664e+04,
                              6.86715311e+04, 7.06972280e+04, 7.26774463e+04, 7.46046457e+04,
                              7.64722106e+04, 7.82745138e+04, 8.00069724e+04, 8.16659859e+04,
                              8.32489411e+04, 8.47541778e+04, 8.61808668e+04, 8.75289750e+04,
                              8.87991768e+04, 8.99927474e+04, 9.11114692e+04, 9.21575697e+04,
                              9.31336048e+04, 9.40424091e+04, 9.48869959e+04, 9.56705218e+04,
                              9.63962206e+04, 9.70673438e+04, 9.76871191e+04, 9.82587361e+04,
                              9.87853159e+04, 9.92698744e+04, 9.97153078e+04, 1.00124391e+05,
                              1.00499770e+05, 1.00843945e+05, 1.01159256e+05, 1.01447925e+05,
                              1.01712062e+05, 1.01953680e+05])
    temperature_half = np.array([196.17694092, 201.40683631, 205.77407659, 209.69290417,
                                 215.18804762, 224.70515674, 239.62107908, 253.48715003,
                                 260.76468316, 266.56703579, 272.19733915, 275.99084954,
                                 278.18932111, 279.59909037, 281.13585504, 279.84737073,
                                 274.79752344, 270.90699415, 268.58975561, 265.54942241,
                                 261.85652346, 258.25080959, 253.84112023, 248.21520521,
                                 244.26001714, 242.39257771, 240.87704963, 238.95997646,
                                 237.00596294, 235.49064744, 234.7742646, 234.37706688,
                                 233.02942684, 231.102146, 229.93968816, 229.42357187,
                                 228.98211629, 228.58466477, 228.02236696, 227.58871118,
                                 227.52062781, 227.41269591, 227.36565461, 227.55814335,
                                 227.83813869, 228.01022473, 228.16288009, 228.50897199,
                                 228.78288898, 228.65908774, 228.2926687, 228.13172126,
                                 228.40948917, 228.90150978, 229.17019413, 229.04995671,
                                 228.77499157, 228.65572399, 228.81564267, 229.1123165,
                                 229.39787394, 229.60247572, 229.6953589, 229.71883424,
                                 229.77129607, 229.91671325, 230.09982177, 230.28120973,
                                 230.51645064, 230.81505553, 231.11827729, 231.39082662,
                                 231.65619322, 231.82700061, 231.85056775, 231.80312025,
                                 231.62201592, 231.32407089, 230.87611508, 229.87908428,
                                 228.31838619, 226.68121558, 225.17473224, 223.93250971,
                                 223.06728503, 222.57664404, 222.69944018, 223.55664063,
                                 224.88072837, 226.40566633, 228.06836655, 229.88666956,
                                 231.8267085, 233.86827167, 235.94517621, 237.93157341,
                                 239.82533873, 241.66794826, 243.40350236, 244.96791598,
                                 246.38663828, 247.73728547, 249.07883826, 250.40834186,
                                 251.69327222, 252.92193625, 254.07749592, 255.14140998,
                                 256.10996523, 256.96492583, 257.67200387, 258.17842959,
                                 258.49984674, 258.70327237, 258.8506826, 259.02602872,
                                 259.28609687, 259.58250006, 260.08193421, 260.77866427,
                                 261.52518146, 262.28001606, 262.92771229, 263.48446416,
                                 264.00193651, 264.48909901, 264.94803007, 265.37910789,
                                 265.77652654, 266.14457721, 266.48686509, 266.80185783,
                                 267.09266898, 267.36279344, 267.61413022, 267.85496194,
                                 268.11645932, 273.37158203])

    nhalf_levels = 138
    press_height_flip = np.zeros(nhalf_levels)  # aufsteigend

    p_atmos_half_flip = np.flip(pressure_half)  # aufsteigend
    t_atmos_half_flip = np.flip(temperature_half)  # aufsteigend

    press_height_flip[0] = 0.
    # calculate complex altitude profile
    for j in range(1, len(press_height_flip)):
        delta_h = barometric_height_old(p_atmos_half_flip[j], p_atmos_half_flip[j - 1], t_atmos_half_flip[j])
        press_height_flip[j] = delta_h + press_height_flip[j - 1]
    press_height = np.flip(press_height_flip)  # absteigend
    # calculate simple altitude profile
    press_height_simple = np.flip(barometric_height_simple(p_atmos_half_flip))
    # JR function
    press_height_jr = np.flip(barometric_height(p_atmos_half_flip, t_atmos_half_flip))

    plt.plot(press_height, label="complex")
    plt.plot(press_height_simple, label="simple")
    plt.plot(press_height_jr, label="JR version")
    plt.legend()
    plt.show()
    plt.close()

    plt.plot(press_height - press_height_simple)
    plt.show()
    plt.close()
