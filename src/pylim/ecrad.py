#!/usr/bin/env python
"""Functions translated from FORTRAN code in ecRad

*author*: Johannes Röttenbacher
"""

import pylim.meteorological_formulas as met
import numpy as np
import xarray as xr
from tqdm import tqdm
import logging
import importlib_resources as pkg_resources

log = logging.getLogger(__name__)


def ice_effective_radius(PPRESSURE, PTEMPERATURE, PCLOUD_FRAC, PQ_ICE, PQ_SNOW, PLAT):
    """
    From ice_effective_radius.F90 from the ecRad source code (https://github.com/ecmwf-ifs/ecrad).

    (C) Copyright 2016- ECMWF.

    This software is licensed under the terms of the Apache Licence Version 2.0
    which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

    In applying this licence, ECMWF does not waive the privileges and immunities
    granted to it by virtue of its status as an intergovernmental organisation
    nor does it submit to any jurisdiction.

    PURPOSE
    -------
        Calculate effective radius of ice clouds

    AUTHOR
    ------
        Robin Hogan, ECMWF (using code extracted from radlswr.F90)
        Original: 2016-02-24

    MODIFICATIONS
    -------------
        2022-09-22  J. Röttenbacher translated Sun and Rikus part to python3

    -------------------------------------------------------------------

    Ice effective radius = f(T,IWC) from Sun and Rikus (1999), revised by Sun (2001)

    Args:
        PPRESSURE: (Pa)
        PTEMPERATURE: (K)
        PCLOUD_FRAC: (kg/kg)
        PQ_ICE: (kg/kg)
        PQ_SNOW: (kg/kg)
        PLAT: (degrees)

    Returns: ice effective radius in micrometer

    """
    # constants
    RRE2DE = 0.64952  # from suecrad.f90
    RMIN_ICE = 60  # min ice radius (um)
    RTT = 273.15  # temperature of fusion of water (K)
    RD = 287  # J kg-1 K-1

    ZDEFAULT_RE_UM = 80 * RRE2DE
    ZMIN_DIAMETER_UM = 20 + (RMIN_ICE - 20) * np.cos(
        (PLAT * np.pi / 180))  # Ice effective radius varies with latitude, smaller at poles

    if (PCLOUD_FRAC > 0.001) and (PQ_ICE + PQ_SNOW > 0):  # Consider only cloudy regions
        ZAIR_DENSITY_GM3 = 1000 * PPRESSURE / (RD * PTEMPERATURE)
        ZIWC_INCLOUD_GM3 = ZAIR_DENSITY_GM3 * (PQ_ICE + PQ_SNOW) / PCLOUD_FRAC
        PTEMPERATURE_C = PTEMPERATURE - RTT
        #  Sun, 2001(corrected from Sun & Rikus, 1999)
        ZAIWC = 45.8966 * ZIWC_INCLOUD_GM3 ** (0.2214)
        ZBIWC = 0.7957 * ZIWC_INCLOUD_GM3 ** (0.2535)
        ZDIAMETER_UM = (1.2351 + 0.0105 * PTEMPERATURE_C) * (ZAIWC + ZBIWC * (PTEMPERATURE - 83.15))
        ZDIAMETER_UM = np.min([np.max([ZDIAMETER_UM, ZMIN_DIAMETER_UM]), 155])
        PRE_UM = ZDIAMETER_UM * RRE2DE
    else:
        PRE_UM = ZDEFAULT_RE_UM

    return PRE_UM * 1e-6


def liquid_effective_radius(PPRESSURE, PTEMPERATURE, PCLOUD_FRAC, PQ_LIQ, PQ_RAIN):
    """
    From liquid_effective_radius.F90 from the ecRad source code (https://github.com/ecmwf-ifs/ecrad).

    (C) Copyright 2015- ECMWF.

    This software is licensed under the terms of the Apache Licence Version 2.0
    which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

    In applying this licence, ECMWF does not waive the privileges and immunities
    granted to it by virtue of its status as an intergovernmental organisation
    nor does it submit to any jurisdiction.

    PURPOSE
    -------
        Calculate effective radius of liquid clouds

    AUTHOR
    ------
        Robin Hogan, ECMWF (using code extracted from radlswr.F90)
        Original: 2015-09-24

    MODIFICATIONS
    -------------
        2022-09-22 J. Röttenbacher translated Martin et al. (JAS 1994) part to python3

    -------------------------------------------------------------------

    Args:
        PPRESSURE: (Pa)
        PTEMPERATURE: (K)
        PCLOUD_FRAC: (kg/kg)
        PQ_LIQ: (kg/kg)
        PQ_RAIN: (kg/kg)

    Returns: liquid effective radius in micrometer after Martin et al. (JAS 1994)

    """
    # constants
    PP_MIN_RE_UM = 4  # min radius
    PP_MAX_RE_UM = 30  # max radius
    PCCN_SEA = 50  # my personal assumption!!!   better estimation???????
    # PCCN_SEA = 20# for the liquid water section from flight 20160926a between 14.5 and 15.1 UTC, derived from modis tau and reff using my cdnc retrieval
    ZCCN = PCCN_SEA
    ZSPECTRAL_DISPERSION = 0.77
    R_DRY = 287  # J kg - 1 K - 1
    REPSCW = 1.E-12
    REPLOG = 1.E-12

    ZNTOT_CM3 = -1.15 * 10E-3 * ZCCN * ZCCN + 0.963 * ZCCN + 5.3  # JR: this is the ocean case
    ZRATIO = (0.222 / ZSPECTRAL_DISPERSION) ** (0.333)

    if (PCLOUD_FRAC > 0.001) and (PQ_LIQ + PQ_RAIN > 0):  # Consider only cloudy regions
        ZAIR_DENSITY_GM3 = 1000 * PPRESSURE / (R_DRY * PTEMPERATURE)
        # In - cloud mean water contents found by dividing by cloud fraction
        ZLWC_GM3 = ZAIR_DENSITY_GM3 * PQ_LIQ / PCLOUD_FRAC
        ZRWC_GM3 = ZAIR_DENSITY_GM3 * PQ_RAIN / PCLOUD_FRAC
        if ZLWC_GM3 > REPSCW:
            ZRAIN_RATIO = ZRWC_GM3 / ZLWC_GM3
            ZWOOD_FACTOR = ((1 + ZRAIN_RATIO) ** 0.666) / (1 + 0.2 * ZRATIO * ZRAIN_RATIO)
        else:
            ZWOOD_FACTOR = 1

        ZRE_CUBED = (3 * (ZLWC_GM3 + ZRWC_GM3)) / (4 * np.pi * ZNTOT_CM3 * ZSPECTRAL_DISPERSION)
        if ZRE_CUBED > REPLOG:
            PRE_UM = ZWOOD_FACTOR * 100 * np.exp(0.333 * np.log(ZRE_CUBED))
            # make sure calculated radius is within boarders
            if PRE_UM < PP_MIN_RE_UM:
                PRE_UM = PP_MIN_RE_UM
            if PRE_UM > PP_MAX_RE_UM:
                PRE_UM = PP_MAX_RE_UM
        else:
            PRE_UM = PP_MIN_RE_UM
    else:
        # when cloud fraction or liquid + rain water content too low to consider this as a cloud
        PRE_UM = PP_MIN_RE_UM

    return PRE_UM * 1e-6


def calc_ice_optics_baran2017(bands: str, ice_wp, qi, temperature):
    """
    Compute ice-particle scattering properties using a parameterization as a function of ice water mixing ratio
    and temperature.

    From radiation_ice_optics_baran2017.F90 from the ecRad source code (https://github.com/ecmwf-ifs/ecrad).

    (C) Copyright 2017- ECMWF.

    This software is licensed under the terms of the Apache Licence Version 2.0
    which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

    In applying this licence, ECMWF does not waive the privileges and immunities
    granted to it by virtue of its status as an intergovernmental organisation
    nor does it submit to any jurisdiction.

    Author:  Robin Hogan
    Email:   r.j.hogan@ecmwf.int

    Modifications
      2023-03-06  J. Röttenbacher  Translated to python3

    Args:
        bands: 'sw' or 'lw', shortwave or longwave bands
        ice_wp: Ice water path (kg m-2)
        qi: Mixing ratio (kg kg-1)
        temperature: Temperature (K)

    Returns:
        od: Total optical depth
        scat_od: Scattering optical depth
        g: Asymmetry factor

    """
    # read in coefficients from netcdf file
    filename = pkg_resources.files("pylim.data").joinpath("baran2017_ice_scattering_rrtm.nc")
    ds = xr.open_dataset(filename)
    nb = len(ds[f"band_{bands}"])  # number of bands
    coeff_gen = ds["coeff_gen"].values  # General coefficients
    coeff = ds[f"coeff_{bands}"]  # Band-specific coefficients
    # Modified ice mixing ratio, and the same raised to an appropriate power
    qi_mod = qi * np.exp(coeff_gen[0] * (temperature - coeff_gen[1]))
    qi_mod_od = qi_mod ** coeff_gen[2]
    qi_mod_ssa = qi_mod ** coeff_gen[3]
    qi_mod_g = qi_mod ** coeff_gen[4]

    od = ice_wp * (coeff[0:nb, 0] + coeff[0:nb, 1] / (1.0 + qi_mod_od * coeff[0:nb, 2]))
    scat_od = od * (coeff[0:nb, 3] + coeff[0:nb, 4] / (1.0 + qi_mod_ssa * coeff[0:nb, 5]))
    g = coeff[0:nb, 6] + coeff[0:nb, 7] / (1.0 + qi_mod_g * coeff[0:nb, 8])

    return od, scat_od, g


def calc_ice_optics_fu_sw(ice_wp, re):
    """
    Compute shortwave ice-particle scattering properties using Fu (1996) parameterization.
    The asymmetry factor in band 14 goes larger than one for re > 100.8 um, so we cap re at 100 um.
    Asymmetry factor is capped at just less than 1 because if it is exactly 1 then delta-Eddington scaling leads to a
    zero scattering optical depth and then division by zero.

    From radiation_ice_optics_fu.F90 from the ecRad source code (https://github.com/ecmwf-ifs/ecrad).

    (C) Copyright 2014- ECMWF.

    This software is licensed under the terms of the Apache Licence Version 2.0
    which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

    In applying this licence, ECMWF does not waive the privileges and immunities
    granted to it by virtue of its status as an intergovernmental organisation
    nor does it submit to any jurisdiction.

    Author:  Robin Hogan
    Email:   r.j.hogan@ecmwf.int

    Modifications
      2020-08-10  R. Hogan  Bounded re to be <= 100um and g to be < 1.0
      2023-03-06  J. Röttenbacher  Translated to python3

    Args:
        ice_wp: Ice water path (kg m-2)
        re: Effective radius (m)

    Returns:

    """
    MaxAsymmetryFactor = 1.0 - 10.0 * np.finfo(dtype="float32").eps
    MaxEffectiveRadius = 100.0e-6  # metres
    # read in coefficients from netCDF file
    filename = pkg_resources.files("pylim.data").joinpath("fu_ice_scattering_rrtm_new.nc")
    ds = xr.open_dataset(filename)
    coeff = ds.coeff_sw1
    nb = 14  # number of shortwave bands
    od = [0.0] * nb  # Total optical depth
    scat_od = [0.0] * nb  # scattering optical depth
    g = [0.0] * nb  # asymmetry factor

    # cap effective radius at MaxEffectiveRadius and keep nan values
    replace_values = np.isnan(re) | (~np.isnan(re) & (re < MaxEffectiveRadius))
    re = re.where(replace_values, MaxEffectiveRadius)  # cap effective radius
    # Convert to effective diameter using the relationship in the IFS
    de_um = re * (1.0e6 / 0.64952)  # Fu's effective diameter (microns)
    inv_de_um = 1.0 / de_um  # and its inverse
    iwp_gm_2 = ice_wp * 1000.0  # Ice water path in g m-2

    for jb in range(nb):
        od[jb] = iwp_gm_2 * (coeff[jb, 0] + coeff[jb, 1] * inv_de_um)
        scat_od[jb] = od[jb] * (
                1.0 - (coeff[jb, 2] + de_um * (coeff[jb, 3] + de_um * (coeff[jb, 4] + de_um * coeff[jb, 5]))))
        g_tmp = coeff[jb, 6] + de_um * (coeff[jb, 7] + de_um * (coeff[jb, 8] + de_um * coeff[jb, 9]))
        replace_values = np.isnan(g_tmp) | (~np.isnan(g_tmp) & (g_tmp < MaxAsymmetryFactor))
        g[jb] = g_tmp.where(replace_values, MaxAsymmetryFactor)

    # convert to dataset with new dimension band_sw
    od = xr.concat(od, "band_sw")
    scat_od = xr.concat(scat_od, "band_sw")
    g = xr.concat(g, "band_sw")

    return od, scat_od, g


def calc_ice_optics_fu_lw(ice_wp, re):
    """
    Compute longwave ice-particle scattering properties using Fu et al. (1998) parameterization.

    From radiation_ice_optics_fu.F90 from the ecRad source code (https://github.com/ecmwf-ifs/ecrad).

    (C) Copyright 2014- ECMWF.

    This software is licensed under the terms of the Apache Licence Version 2.0
    which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

    In applying this licence, ECMWF does not waive the privileges and immunities
    granted to it by virtue of its status as an intergovernmental organisation
    nor does it submit to any jurisdiction.

    Author:  Robin Hogan
    Email:   r.j.hogan@ecmwf.int

    Modifications
      2020-08-10  R. Hogan  Bounded re to be <= 100um and g to be < 1.0
      2023-03-06  J. Röttenbacher  Translated to python3

    Args:
        ice_wp: Ice water path (kg m-2)
        re: Effective radius (m)

    Returns:

    """
    MaxAsymmetryFactor = 1.0 - 10.0 * np.finfo(1.0).eps
    MaxEffectiveRadius = 100.0e-6  # metres
    # read in coefficients from netCDF file
    filename = pkg_resources.files("pylim.data").joinpath("fu_ice_scattering_rrtm_new.nc")
    ds = xr.open_dataset(filename)
    coeff = ds.coeff_lw1
    nb = 16  # number of longwave bands
    od = [0.0] * nb  # Total optical depth
    scat_od = [0.0] * nb  # scattering optical depth
    g = [0.0] * nb  # asymmetry factor

    # cap effective radius at MaxEffectiveRadius and keep nan values
    replace_values = np.isnan(re) | (~np.isnan(re) & (re < MaxEffectiveRadius))
    re = re.where(replace_values, MaxEffectiveRadius)  # cap effective radius
    # Convert to effective diameter using the relationship in the IFS
    de_um = min(re, MaxEffectiveRadius) * (1.0e6 / 0.64952)  # Fu's effective diameter (microns)
    inv_de_um = 1.0 / de_um  # and its inverse
    iwp_gm_2 = ice_wp * 1000.0  # Ice water path in g m-2

    for jb in range(nb):
        od[jb] = iwp_gm_2 * (coeff[jb, 0] + inv_de_um * (coeff[jb, 1] + inv_de_um * coeff[jb, 2]))
        scat_od[jb] = od[jb] - iwp_gm_2 * inv_de_um * (
                coeff[jb, 3] + de_um * (coeff[jb, 4] + de_um * (coeff[jb, 5] + de_um * coeff[jb, 6])))
        g_tmp = coeff[jb, 7] + de_um * (coeff[jb, 8] + de_um * (coeff[jb, 9] + de_um * coeff[jb, 10]))
        replace_values = np.isnan(g_tmp) | (~np.isnan(g_tmp) & (g_tmp < MaxAsymmetryFactor))
        g[jb] = g_tmp.where(replace_values, MaxAsymmetryFactor)

    # convert to dataset with new dimension band_lw
    od = xr.concat(od, "band_lw")
    scat_od = xr.concat(scat_od, "band_lw")
    g = xr.concat(g, "band_lw")

    return od, scat_od, g


def calc_ice_optics_yi(bands: str, ice_wp, re):
    """
    Compute shortwave ice-particle scattering properties using Yi et al. (2013) parameterization.

    From radiation_ice_optics_yi.F90 from the ecRad source code (https://github.com/ecmwf-ifs/ecrad).

    (C) Copyright 2017- ECMWF.

    This software is licensed under the terms of the Apache Licence Version 2.0
    which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

    In applying this licence, ECMWF does not waive the privileges and immunities
    granted to it by virtue of its status as an intergovernmental organisation
    nor does it submit to any jurisdiction.

    Authors:  Mark Fielding and Robin Hogan
    Email:   r.j.hogan@ecmwf.int

    The reference for this ice optics parameterization is:
    Yi, B., P. Yang, B.A. Baum, T. L'Ecuyer, L. Oreopoulos, E.J. Mlawer, A.J. Heymsfield, and K. Liou, 2013:
    Influence of Ice Particle Surface Roughening on the Global Cloud Radiative Effect. J. Atmos. Sci., 70, 2794–2807,
    https://doi.org/10.1175/JAS-D-13-020.1

    Modifications
      2023-03-06  J. Röttenbacher  Translated to python3

    Args:
        bands: 'sw' or 'lw', shortwave or longwave bands
        ice_wp: Ice water path (kg m-2)
        re: effective radius (m)

    Returns:
        od: Total optical depth
        scat_od: Scattering optical depth
        g: Asymmetry factor

    """
    NSingleCoeffs = 23
    # read in coefficients from netcdf file
    filename = pkg_resources.files("pylim.data").joinpath("yi_ice_scattering_rrtm_new.nc")
    ds = xr.open_dataset(filename)
    nb = len(ds[f"band_{bands}"])  # number of bands
    lu_scale = 0.2
    lu_offset = 1.0
    coeff = ds[f"coeff_{bands}1"]  # Band-specific coefficients
    # Convert to effective diameter using the relationship in the IFS
    # de_um = re * (1.0e6 / 0.64952)
    de_um = re * 2.0e6

    # limit de_um to validity of LUT
    replace_values = np.isnan(de_um) | (~np.isnan(de_um) & (de_um < 119.99))
    de_um = de_um.where(replace_values, 119.99)  # avoid greater than or equal to 120 um
    replace_values = np.isnan(de_um) | (~np.isnan(de_um) & (de_um > 10))
    de_um = de_um.where(replace_values, 10.0)  # avoid smaller 20 um

    iwp_gm_2 = ice_wp * 1000.0  # convert to g/m2
    lu_idx = np.floor(de_um * lu_scale - lu_offset)  # generate look up indices
    wts_2 = (de_um * lu_scale - lu_offset) - lu_idx
    wts_1 = 1.0 - wts_2

    od_list, scat_od, g = list(), list(), list()  # define lists to append each band to
    for i in range(nb):
        c = coeff[i, :].values  # retrieve coefficients
        c1, c2, c3, c4, c5, c6 = np.copy(lu_idx), np.copy(lu_idx), np.copy(lu_idx), np.copy(lu_idx), np.copy(lu_idx), np.copy(lu_idx)
        # loop through coefficients
        for key, value in enumerate(c):
            k = key + 1  # look up indices start at 1
            c1[lu_idx == k] = value
            # use try and except to catch IndexError
            try:
                c2[lu_idx == k] = c[k]
                c3[lu_idx == k] = c[key + NSingleCoeffs]
                c4[lu_idx == k] = c[key + NSingleCoeffs + 1]
                c5[lu_idx == k] = c[key + 2 * NSingleCoeffs]
                c6[lu_idx == k] = c[key + 2 * NSingleCoeffs + 1]
            except IndexError:
                pass

        od = (0.001 * iwp_gm_2 * (wts_1 * c1 + wts_2 * c2))
        od_list.append(od)
        scat_od.append(od * (wts_1 * c3 + wts_2 * c4))
        g.append(wts_1 * c5 + wts_2 * c6)

    od = xr.concat(od_list, dim=f"band_{bands}")
    scat_od = xr.concat(scat_od, dim=f"band_{bands}")
    g = xr.concat(g, dim=f"band_{bands}")

    return od, scat_od, g


def calc_pressure(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculate the pressure at half and full hybrid model level.
    See https://confluence.ecmwf.int/display/CKB/ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height

    Args:
        ds: DataSet as provided from the IFS output

    Returns: DataSet with pressure at half and full model level

    """
    # calculate pressure at half and mid-level, this will add the single level for surface pressure to its dimension
    # remove it by selecting only this level and dropping the coordinate, which is now not an index anymore
    ph = ds["hyai"] + np.exp(ds["lnsp"]) * ds["hybi"]
    # pf = ds["hyam"] + np.e ** ds["lnsp"] * ds["hybm"]
    # use code as suggested by confluence article
    # difference in the lowest levels < 0.05 Pa (new - old)
    pf = list()
    for hybrid in range(len(ph) - 1):
        pf.append((ph.isel(nhyi=hybrid + 1) + ph.isel(nhyi=hybrid)) / 2.0)
    pf = xr.concat(pf, 'lev')
    ds["pressure_hl"] = ph.sel(lev_2=1, drop=True).astype("float32")  # assign as a new variable
    ds["pressure_full"] = pf.sel(lev_2=1, drop=True).astype("float32")

    return ds


def calculate_pressure_height(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculate the pressure height for the half and full model levels

    Args:
        ds: Dataset with temperature and pressure on model half and full levels (``pressure_hl``, ``temperature_hl``,
            ``pressure_full``, ``t``)

    Returns: Dataset with two new variables ``pressure_height_hl`` and ``pressure_height_full`` in meters

    """
    # calculate pressure height of model half levels
    vertical_profiles = ds[["pressure_hl", "temperature_hl"]]
    p_array_list = list()
    # ignore division by zero error
    with np.errstate(divide='ignore'):
        for time in tqdm(vertical_profiles.time, desc="Half Levels"):
            tmp = vertical_profiles.sel(time=time, drop=True)
            press_height = met.barometric_height(tmp["pressure_hl"], tmp["temperature_hl"])
            p_array = xr.DataArray(data=press_height[None, :], dims=["time", "half_level"],
                                   coords={"half_level": (["half_level"], np.flip(tmp.half_level.values)),
                                           "time": np.array([time.values])},
                                   name="pressure_height")
            p_array_list.append(p_array)

    ds["press_height_hl"] = xr.merge(p_array_list).pressure_height
    # replace nan (TOA) with 80km
    ds["press_height_hl"] = ds["press_height_hl"].where(~np.isnan(ds["press_height_hl"]), 80000)

    # calculate pressure height of model full levels
    vertical_profiles = ds[["pressure_full", "t"]]
    p_array_list = list()
    for time in tqdm(vertical_profiles.time, desc="Full Levels"):
        tmp = vertical_profiles.sel(time=time, drop=True)
        press_height = met.barometric_height(tmp["pressure_full"], tmp["t"])
        p_array = xr.DataArray(data=press_height[None, :], dims=["time", "level"],
                               coords={"level": (["level"], np.flip(tmp.level.values)),
                                       "time": np.array([time.values])},
                               name="pressure_height")
        p_array_list.append(p_array)

    ds["press_height_full"] = xr.merge(p_array_list).pressure_height
    # replace nan (TOA) with 80km
    ds["press_height_full"] = ds["press_height_full"].where(~np.isnan(ds["press_height_full"]), 80000)

    return ds


def apply_ice_effective_radius(ds: xr.Dataset) -> xr.Dataset:
    """Apply ice effective radius function over a whole dataset

    Args:
        ds: IFS DataSet

    Returns: DataSet with new variable re_ice containing the ice effective radius for each point

    """
    # check if time, lat and lon are dimensions in data set and adjust chunking accordingly
    if "time" in ds.dims and "lat" in ds.dims and "lon" in ds.dims:
        chunk_dict = {"time": 1, "level": 10, "lat": 10, "lon": 10}
    elif "lat" in ds.dims and "lon" in ds.dims:
        chunk_dict = {"level": 10, "lat": 10, "lon": 10}
    elif "time" not in ds.dims and "lat" not in ds.dims and "lon" not in ds.dims:
        chunk_dict = {"level": 10}
    else:
        raise ValueError(f"Input dimensions {ds.dims} do not match any combination of expected dimensions")

    ieffr = xr.apply_ufunc(
        ice_effective_radius,  # first the function
        # now arguments in the order expected by 'ice_effective_radius'
        ds.pressure_full.chunk(chunk_dict),
        ds.t.chunk(chunk_dict),  # as above
        ds.cloud_fraction.chunk(chunk_dict),
        ds.ciwc.chunk(chunk_dict),
        ds.cswc.chunk(chunk_dict),
        ds.lat,  # as above
        input_core_dims=[[], [], [], [], [], []],  # list with one entry per arg
        vectorize=True,  # loop over non-core dims
        dask="parallelized",
        output_dtypes=[ds.t.dtype]
    )
    ds["re_ice"] = ieffr.compute()
    ds["re_ice"].attrs = dict(units="m")

    return ds


def apply_liquid_effective_radius(ds: xr.Dataset) -> xr.Dataset:
    """Apply liquid effective radius function over a whole dataset

    Args:
        ds: IFS DataSet

    Returns: DataSet with new variable re_ice containing the ice effective radius for each point

    """
    # check if time, lat and lon are dimensions in data set and adjust chunking accordingly
    if "time" in ds.dims and "lat" in ds.dims and "lon" in ds.dims:
        chunk_dict = {"time": 1, "level": 10, "lat": 10, "lon": 10}
    elif "lat" in ds.dims and "lon" in ds.dims:
        chunk_dict = {"level": 10, "lat": 10, "lon": 10}
    elif "time" not in ds.dims and "lat" not in ds.dims and "lon" not in ds.dims:
        chunk_dict = {"level": 10}
    else:
        raise ValueError(f"Input dimensions {ds.dims} do not match any combination of expected dimensions")

    leffr = xr.apply_ufunc(
        liquid_effective_radius,  # first the function
        # now arguments in the order expected by 'liquid_effective_radius'
        ds.pressure_full.chunk(chunk_dict),
        ds.t.chunk(chunk_dict),  # as above
        ds.cloud_fraction.chunk(chunk_dict),
        ds.clwc.chunk(chunk_dict),
        ds.crwc.chunk(chunk_dict),
        input_core_dims=[[], [], [], [], []],  # list with one entry per arg
        vectorize=True,  # loop over non-core dims
        dask="parallelized",
        output_dtypes=[ds.t.dtype]
    )
    ds["re_liquid"] = leffr.compute()
    ds["re_liquid"].attrs = dict(units="m")

    return ds


def cloud_overlap_decorr_len(latitude: float, scheme: int):
    """Implementation of the overlap decorrelation length parameter according to Shonk et al. 2010 (https://doi.org/10.1002/qj.647)

    Args:
        latitude: Latitude of IFS grid cell
        scheme: Which scheme to apply?

    Returns: decorr_len_edges_km, decorr_len_water_km, decorr_len_ratio

    """
    # sin_lat = np.sin(latitude * np.pi / 180)

    if scheme == 0:
        print('nicht gut')
        raise ValueError(f"Use another value for scheme!")

    elif scheme == 1:  # operational in IFS (Shonk et al. 2010, Part I)
        abs_lat_deg = np.abs(latitude)
        decorr_len_edges_km = 2.899 - 0.02759 * abs_lat_deg

    elif scheme == 2:
        cos_lat = np.cos(latitude * np.pi / 180)
        decorr_len_edges_km = 0.75 + 2.149 * cos_lat * cos_lat

    else:
        raise ValueError(f"Wrong value for scheme: {scheme}")

    decorr_len_water_km = decorr_len_edges_km * 0.5
    decorr_len_ratio = 0.5

    return decorr_len_edges_km, decorr_len_water_km, decorr_len_ratio


if __name__ == '__main__':
    # testing
    ifs_ml = xr.open_dataset("data_jr/ifs_raw_output/ifs_20170525_00_ml.nc")
    ifs_ml = calc_pressure(ifs_ml)
    # select only one time step to reduce computational load for now
    ifs_ml = ifs_ml.sel({"time": "2017-05-25T00:00:00"})
    # %% test effective radius functions with single values
    ds = ifs_ml.isel({"lev": 114, "lat": 129, "lon": 378})
    ieffr_sel = ice_effective_radius(ds.pressure_full, ds.t, ds.cc, ds.ciwc, ds.cswc, ds.lat)
    leffr_sel = liquid_effective_radius(ds.pressure_full, ds.t, ds.cc, ds.clwc, ds.crwc)

    ieffr = xr.apply_ufunc(
        ice_effective_radius,  # first the function
        # now arguments in the order expected by 'ice_effective_radius'
        ifs_ml.pf.chunk({"lev": 10, "lat": 10, "lon": 10}),
        ifs_ml.t.chunk({"lev": 10, "lat": 10, "lon": 10}),  # as above
        ifs_ml.cc.chunk({"lev": 10, "lat": 10, "lon": 10}),
        ifs_ml.ciwc.chunk({"lev": 10, "lat": 10, "lon": 10}),
        ifs_ml.cswc.chunk({"lev": 10, "lat": 10, "lon": 10}),
        ifs_ml.lat.chunk({"lat": 10}),  # as above
        input_core_dims=[[], [], [], [], [], []],  # list with one entry per arg
        vectorize=True,  # loop over non-core dims
        dask="parallelized",
        output_dtypes=[ifs_ml.t.dtype]
    )

    ieffr_test = ieffr.isel({"lev": 114, "lat": 129, "lon": 378}).compute()
    np.testing.assert_allclose(ieffr_test.values, ieffr_sel)  # check if both functions return the same result
    # no error is thrown -> same result

    leffr = xr.apply_ufunc(
        liquid_effective_radius,  # first the function
        # now arguments in the order expected by 'liquid_effective_radius'
        ifs_ml.pf.chunk({"lev": 10, "lat": 10, "lon": 10}),
        ifs_ml.t.chunk({"lev": 10, "lat": 10, "lon": 10}),  # as above
        ifs_ml.cc.chunk({"lev": 10, "lat": 10, "lon": 10}),
        ifs_ml.clwc.chunk({"lev": 10, "lat": 10, "lon": 10}),
        ifs_ml.crwc.chunk({"lev": 10, "lat": 10, "lon": 10}),
        input_core_dims=[[], [], [], [], []],  # list with one entry per arg
        vectorize=True,  # loop over non-core dims
        dask="parallelized",
        output_dtypes=[ifs_ml.t.dtype]
    )

    leffr_test = leffr.isel({"lev": 114, "lat": 129, "lon": 378}).compute()
    np.testing.assert_allclose(leffr_test.values, leffr_sel.values)  # check if both functions return the same result


    # no error is thrown -> same result
    # %%

    def liquid_effective_radius(PPRESSURE, PTEMPERATURE, PCLOUD_FRAC, PQ_LIQ, PQ_RAIN, PLAND_FRAC, PCCN_SEA,
                                PCCN_LAND, **kwargs):
        # analyze CCN climatology for CCN concentration
        LCCNO = kwargs["LCCNO"] if "LCCNO" in kwargs else True  # ocean
        LCCNL = kwargs["LCCNL"] if "LCCNL" in kwargs else True  # land

        # constants
        PP_MIN_RE_UM = 4  # min radius
        PP_MAX_RE_UM = 30  # max radius
        RCCNSEA = 50  # default CCN concentration for ocean (Martin et al. 1994)
        RCCNLND = 250  # default CCN concentration for land TODO: check for literature value
        # PCCN_SEA = 20# for the liquid water section from flight 20160926a between 14.5 and 15.1 UTC, derived from modis tau and reff using my cdnc retrieval
        R_DRY = 287  # J kg - 1 K - 1
        REPSCW = 1.E-12
        REPLOG = 1.E-12

        # First compute the cloud droplet concentration
        if PLAND_FRAC < 0.5:
            # Sea case
            if LCCNO:
                ZCCN = PCCN_SEA
            else:
                ZCCN = RCCNSEA

            ZSPECTRAL_DISPERSION = 0.77
            # Cloud droplet concentration in cm - 3(activated CCN) over ocean
            ZNTOT_CM3 = -1.15E-03 * ZCCN * ZCCN + 0.963 * ZCCN + 5.30
        else:
            # Land case
            if LCCNL:
                ZCCN = PCCN_LAND
            else:
                ZCCN = RCCNLND

            ZSPECTRAL_DISPERSION = 0.69
            # Cloud droplet concentration in cm - 3(activated CCN) over land
            ZNTOT_CM3 = -2.10E-04 * ZCCN * ZCCN + 0.568 * ZCCN - 27.9

        ZRATIO = (0.222 / ZSPECTRAL_DISPERSION) ** 0.333

        ###########################################################################################
        # ZNTOT_CM3 = -1.15 * 10E-3 * ZCCN * ZCCN + 0.963 * ZCCN + 5.3  # JR: this is the ocean case
        # ZRATIO = (0.222 / ZSPECTRAL_DISPERSION) ** (0.333)

        if (PCLOUD_FRAC > 0.001) and (PQ_LIQ + PQ_RAIN > 0):  # Consider only cloudy regions
            ZAIR_DENSITY_GM3 = 1000 * PPRESSURE / (R_DRY * PTEMPERATURE)
            # In - cloud mean water contents found by dividing by cloud fraction
            ZLWC_GM3 = ZAIR_DENSITY_GM3 * PQ_LIQ / PCLOUD_FRAC
            ZRWC_GM3 = ZAIR_DENSITY_GM3 * PQ_RAIN / PCLOUD_FRAC
            # Wood's (2000, eq. 19) adjustment to Martin et al's parameterization
            if ZLWC_GM3 > REPSCW:
                ZRAIN_RATIO = ZRWC_GM3 / ZLWC_GM3
                ZWOOD_FACTOR = ((1 + ZRAIN_RATIO) ** 0.666) / (1 + 0.2 * ZRATIO * ZRAIN_RATIO)
            else:
                ZWOOD_FACTOR = 1

            # gm - 3 and cm - 3 units cancel out with density of water 10 ^ 6 / (1000 * 1000);
            # need a factor of 10 ^ 6 to convert to microns and cubed root is factor of 100 which appears in
            # equation below
            ZRE_CUBED = (3 * (ZLWC_GM3 + ZRWC_GM3)) / (4 * np.pi * ZNTOT_CM3 * ZSPECTRAL_DISPERSION)
            if ZRE_CUBED > REPLOG:
                PRE_UM = ZWOOD_FACTOR * 100 * np.exp(0.333 * np.log(ZRE_CUBED))
                # make sure calculated radius is within boarders
                if PRE_UM < PP_MIN_RE_UM:
                    PRE_UM = PP_MIN_RE_UM
                if PRE_UM > PP_MAX_RE_UM:
                    PRE_UM = PP_MAX_RE_UM
            else:
                PRE_UM = PP_MIN_RE_UM
        else:
            # when cloud fraction or liquid + rain water content too low to consider this as a cloud
            PRE_UM = PP_MIN_RE_UM

        return PRE_UM


    ifs_sfc = xr.open_dataset("data_jr/ifs_raw_output/ifs_20170525_00_sfc.nc")
    ifs_sfc = ifs_sfc.sel({"time": "2017-05-25T00:00:00"})
    lat, lon = 129, 378
    ds = ifs_ml.isel({"lev": 114, "lat": lat, "lon": lon})
    land_frac = ifs_sfc.LSM.isel({"lat": lat, "lon": lon})
    leffr_sel = liquid_effective_radius(ds.pf, ds.t, ds.cc, ds.clwc, ds.crwc, land_frac)

    # %%
    # for final run
    # ifs_ml = apply_ice_effective_radius(ifs_ml)
    # ifs_ml = apply_liquid_effective_radius(ifs_ml)
    # ifs_ml.to_netcdf("./data_jr/ifs_20170525_00_ml_v01.nc", format="NETCDF4")
