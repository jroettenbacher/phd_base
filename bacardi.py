#!/usr/bin/env python
"""Script for processing and plotting BACARDI data
author: Johannes Röttenbacher
"""

# %% module import
from smart import get_path
import xarray as xr

# %% set paths
flight = "Flight_20210629a"
bacardi_path = get_path("bacardi", flight)

# %% read in bacardi data
filename = "CIRRUS_HL_F05_20210629a_ADLR_BACARDI_BroadbandFluxes_R0.nc"
ds = xr.open_dataset(f"{bacardi_path}/{filename}")
ds

