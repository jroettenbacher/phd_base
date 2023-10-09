#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 20.06.2023

Download monthly mean CAMS data from the Copernicus Atmosphere store
"""

import pylim.helpers as h
import cdsapi

c = cdsapi.Client()

cams_path = h.get_path("cams", campaign="halo-ac3")
year = 2019

# aerosol and ozone
c.retrieve(
    "cams-global-reanalysis-eac4-monthly",
    {
        "format": "netcdf",
        "variable": [
            "dust_aerosol_0.03-0.55um_mixing_ratio", "dust_aerosol_0.55-0.9um_mixing_ratio", "dust_aerosol_0.9-20um_mixing_ratio",
            "hydrophilic_black_carbon_aerosol_mixing_ratio", "hydrophilic_organic_matter_aerosol_mixing_ratio", "hydrophobic_black_carbon_aerosol_mixing_ratio",
            "hydrophobic_organic_matter_aerosol_mixing_ratio", "ozone", "sea_salt_aerosol_0.03-0.5um_mixing_ratio",
            "sea_salt_aerosol_0.5-5um_mixing_ratio", "sea_salt_aerosol_5-20um_mixing_ratio", "sulphate_aerosol_mixing_ratio",
        ],
        "pressure_level": [
            "1", "2", "3",
            "5", "7", "10",
            "20", "30", "50",
            "70", "100", "150",
            "200", "250", "300",
            "400", "500", "600",
            "700", "800", "850",
            "900", "925", "950",
            "1000",
        ],
        "year": f"{year}",
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
        ],
        "product_type": "monthly_mean",
        "area": [
            90, -40, 60,
            40,
        ],
    },
    f"{cams_path}/cams_eac4_global_reanalysis_mm_{year}_pl.nc")


# methane and CO2
c.retrieve(
    "cams-global-ghg-reanalysis-egg4-monthly",
    {
        "variable": [
            "carbon_dioxide", "methane",
        ],
        "pressure_level": [
            "1", "2", "3",
            "5", "7", "10",
            "20", "30", "50",
            "70", "100", "150",
            "200", "250", "300",
            "400", "500", "600",
            "700", "800", "850",
            "900", "925", "950",
            "1000",
        ],
        "year": f"{year}",
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
        ],
        "product_type": "monthly_mean",
        "area": [
            90, -40, 60,
            40,
        ],
        "format": "netcdf",
    },
    f"{cams_path}/cams_global_ghg_reanalysis_mm_{year}_pl.nc")
