#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 23.07.2024

Plots for introduction

"""

# %% import modules
import os

import matplotlib.pyplot as plt
import numpy as np

import pylim.helpers as h

# %% set paths
plot_path = 'C:/Users/Johannes/Documents/Doktor/manuscripts/_thesis/figure'

# %% plot hom het sketch
relative_humidity = np.arange(100, 180, 5)
temperatures = np.flip(np.linspace(-60, 0.1, len(relative_humidity)))

_, ax = plt.subplots(1, 1, figsize=(15 * h.cm, 6 * h.cm), layout='constrained')
ax.set(
    xlabel='Relative humidity over ice (%)',
    ylabel='Temperature (°C)',
    xlim=(100, 180),
    ylim=(0, -60),
)
ax.axhline(-38, color='k', ls='--')
ax.axvline(140, color='k', ls='--')
plt.savefig(f'{plot_path}/01_hom_het_plot.svg')
plt.show()
plt.close()
