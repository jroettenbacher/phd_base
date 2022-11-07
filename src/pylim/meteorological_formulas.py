#!/usr/bin/env python
"""General meteorological formulas

*author*: Johannes Röttenbacher
"""
import numpy as np
from typing import Union


def relative_humidity_water_to_relative_humidity_ice(relative_humidity_water: Union[float, np.ndarray, list],
                                                     temperature: Union[float, np.ndarray, list],
                                                     version: str = "huang"):
    """
    Convert the relative humidity over water to relative humidity over ice using either the formulas by
    :cite:t`Huang2019` or by :cite:t:`Alduchov1996`.

    .. math::

        RH_{ice} = \\frac{RH_{water} * e_{s,w}}{e_{s,i}}

    with :math:`e_{s,w}` the saturation vapor pressure of water and :math:`e_{s,i}` the saturation vapor pressure of ice:

    .. math::

        e_{s,w} = \\frac{\\exp{34.494 - \\frac{4924.99}{t + 237.1}{(t + 105)^{1.157}} (t > 0°C)

        e_{s,i} = \\frac{\\exp{43.494 - \\frac{6545.8}{t + 278}{(t + 868)^{2}} (t <= 0°C)

    with :math:`t` being the temperature in °C. When version is 'alduchov' the following equations are used:

    .. math::

        e_{s,w} = 6.1094 * \\exp{\\frac{17.625 * t}{243.04 + t}}

        e_{s,i} = 6.1121 * \\exp{\\frac{22.587 * t}{273.86 + t}}

    Args:
        relative_humidity_water: ambient relative humidity in percent
        temperature: ambient temperature in Celsius
        version: which formulas to use for calculating the saturation vapor pressure of water and ice ('huang' or 'alduchov')

    Returns: relative humidity over ice

    """
    if version == "alduchov":
        saturation_vapour_pressure_water = 6.1094 * np.exp(17.625 * temperature / (243.04 + temperature))
        saturation_vapour_pressure_ice = 6.1121 * np.exp(22.587 * temperature / (273.86 + temperature))
    else:
        saturation_vapour_pressure_water = np.exp(34.494 - (4924.99 / (temperature + 237.1))) / (temperature + 105)**1.157
        saturation_vapour_pressure_ice = np.exp(43.494 - (6545.8 / (temperature + 278))) / (temperature + 868)**2

    relative_humidity_ice = relative_humidity_water * saturation_vapour_pressure_water / saturation_vapour_pressure_ice

    return relative_humidity_ice
