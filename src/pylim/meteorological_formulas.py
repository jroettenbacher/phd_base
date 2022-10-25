#!/usr/bin/env python
"""General meteorological formulas

*author*: Johannes Röttenbacher
"""
import numpy as np
from typing import Union


def relative_humidity_water_to_relative_humidity_ice(relative_humidity_water: Union[float, np.ndarray, list],
                                                     temperature: Union[float, np.ndarray, list]):
    """
    Convert the relative humidity over water to relative humidity over ice.
    .. math::

        RH_{ice} = frac{RH_{water} * e_{s,w}}{e_{s,i}}

    Args:
        relative_humidity_water: ambient relative humidity in percent
        temperature: ambient temperature in Celsius

    Returns: relative humidity over ice

    """
    saturation_vapour_pressure_water = 6.1094 * np.exp(17.625 * temperature / (243.04 + temperature))
    saturation_vapour_pressure_ice = 6.1121 * np.exp(22.587 * temperature / (273.86 + temperature))
    relative_humidity_ice = relative_humidity_water * saturation_vapour_pressure_water / saturation_vapour_pressure_ice

    return relative_humidity_ice
