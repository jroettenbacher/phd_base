#!/usr/bin/env python
"""Functions translated from FORTRAN code in ecRad

*author*: Johannes Röttenbacher
"""

import numpy as np
import xarray as xr
import logging

log = logging.getLogger(__name__)


def ice_effective_radius(PPRESSURE, PTEMPERATURE, PCLOUD_FRAC, PQ_ICE, PQ_SNOW, PLAT):
    # constants
    RRE2DE = 0.64952  # from suecrad.f90
    RMIN_ICE = 60  # min ice radius
    RTT = 273.15  # temperature of fusion of water
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


def calc_pressure(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculate the pressure at half and full hybrid model level
    Args:
        ds: DataSet as provided from the IFS output

    Returns: DataSet with pressure at half and full model level

    """
    # calculate pressure at half and mid-level, this will add the single level for surface pressure to its dimension
    # remove it by selecting only this level and dropping the coordinate, which is now not an index anymore
    ph = ds["hyai"] + np.exp(ds["lnsp"]) * ds["hybi"]
    pf = list()
    for hybrid in range(len(ph) - 1):
        pf.append((ph.isel(nhyi=hybrid + 1) + ph.isel(nhyi=hybrid)) / 2.0)
    pf = xr.concat(pf, 'lev')
    ds["pressure_hl"] = ph.sel(lev_2=1, drop=True).astype("float32")  # assign as a new variable
    ds["pressure_full"] = pf.sel(lev_2=1, drop=True).astype("float32")

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
        chunk_dict = {"level": 10,"lat": 10, "lon": 10}
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
    ds["re_liquid"].attrs = dict(unit="m")

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
