#!/usr/bin/env python
"""Plot an interactive figure for easier flight segmentation

*author*: Johannes Röttenbacher
"""

# %% library import
import pylim.helpers as h
import os
import holoviews as hv
from holoviews import opts
import ac3airborne
from metpy.units import units
from metpy.constants import Rd, g
import numpy as np
import xarray as xr

hv.extension('bokeh')

# %% functions
# Values according to the 1976 U.S. Standard atmosphere [NOAA1976]_.
# List of tuples (height, temperature, pressure, temperature gradient)
_STANDARD_ATMOSPHERE = [
    (0 * units.km, 288.15 * units.K, 101325 * units.Pa, 0.0065 * units.K / units.m),
    (11 * units.km, 216.65 * units.K, 22632.1 * units.Pa, 0 * units.K / units.m),
    (20 * units.km, 216.65 * units.K, 5474.89 * units.Pa, -0.001 * units.K / units.m),
    (32 * units.km, 228.65 * units.K, 868.019 * units.Pa, -0.0028 * units.K / units.m),
    (47 * units.km, 270.65 * units.K, 110.906 * units.Pa, 0 * units.K / units.m),
    (51 * units.km, 270.65 * units.K, 66.9389 * units.Pa, 0.0028 * units.K / units.m),
    (71 * units.km, 214.65 * units.K, 3.95642 * units.Pa, float("NaN") * units.K / units.m)
]
_HEIGHT, _TEMPERATURE, _PRESSURE, _TEMPERATURE_GRADIENT = 0, 1, 2, 3


def pressure2flightlevel(pressure):
    r"""
    Conversion of pressure to height (hft) with
    hydrostatic equation, according to the profile of the 1976 U.S. Standard atmosphere [NOAA1976]_.
    Reference:
        H. Kraus, Die Atmosphaere der Erde, Springer, 2001, 470pp., Sections II.1.4. and II.6.1.2.
    Parameters
    ----------
    pressure : `pint.Quantity` or `xarray.DataArray`
        Atmospheric pressure
    Returns
    -------
    `pint.Quantity` or `xarray.DataArray`
        Corresponding height value(s) (hft)
    Notes
    -----
    .. math:: Z = \begin{cases}
              Z_0 + \frac{T_0 - T_0 \cdot \exp\left(\frac{\Gamma \cdot R}{g\cdot\log(\frac{p}{p0})}\right)}{\Gamma}
              &\Gamma \neq 0\\
              Z_0 - \frac{R \cdot T_0}{g \cdot \log(\frac{p}{p_0})} &\text{else}
              \end{cases}
    """
    is_array = hasattr(pressure.magnitude, "__len__")
    if not is_array:
        pressure = [pressure.magnitude] * pressure.units

    # Initialize the return array.
    z = np.full_like(pressure, np.nan) * units.hft

    for i, ((z0, t0, p0, gamma), (z1, t1, p1, _)) in enumerate(zip(_STANDARD_ATMOSPHERE[:-1],
                                                                   _STANDARD_ATMOSPHERE[1:])):
        p1 = _STANDARD_ATMOSPHERE[i + 1][_PRESSURE]
        indices = (pressure > p1) & (pressure <= p0)
        if i == 0:
            indices |= (pressure >= p0)
        if gamma != 0:
            z[indices] = z0 + 1. / gamma * (t0 - t0 * np.exp(gamma * Rd / g * np.log(pressure[indices] / p0)))
        else:
            z[indices] = z0 - (Rd * t0) / g * np.log(pressure[indices] / p0)

    if np.isnan(z).any():
        raise ValueError("flight level to pressure conversion not "
                         "implemented for z > 71km")

    return z if is_array else z[0]

# %% Define flight and paths
campaign="halo-ac3"
date = "20220328"
flight_key = "RF09"
flight = f"HALO-AC3_{date}_HALO_{flight_key}"
ql_path = h.get_path("quicklooks", flight, campaign)
ims_path = h.get_path("horidata", flight, campaign)
h.make_dir(ql_path)
output_format = "html"  # png or html, html gives an interactive plot
cat = ac3airborne.get_intake_catalog()
kwds = {"storage_options": {'simplecache': dict(cache_storage=ims_path, same_names=True)},
        "user": os.environ.get("AC3_CLOUD_USER"), "password": os.environ.get("AC3_CLOUD_PASSWORD")}

# %% calculate flight level
bahamas = cat["HALO-AC3"]["HALO"]["BAHAMAS"][f"HALO-AC3_HALO_{flight_key}"](**kwds).to_dask()
bahamas = bahamas.swap_dims({"tid": "TIME"})
bahamas = bahamas.rename({"TIME": "time"})
ims = cat["HALO-AC3"]["HALO"]["GPS_INS"][f"HALO-AC3_HALO_{flight_key}"](**kwds).to_dask()
bahamas = bahamas.reindex_like(ims, method="nearest")
flv = pressure2flightlevel(bahamas.PS.values * units.hPa)
p_bot, p_top = 101315, 12045
flv_limits = pressure2flightlevel([p_bot, p_top] * units.Pa)
_pres_maj = np.concatenate([np.arange(top * 10, top, -top) for top in (10000, 1000, 100, 10)] + [[10]])
_pres_min = np.concatenate([np.arange(top * 10, top, -top // 10) for top in (10000, 1000, 100, 10)] + [[10]])

# %% Holoviews Dashboard of INS quicklook
ims["flv"] = xr.DataArray(data=flv.magnitude, dims=("time"))
hv_ds = hv.Dataset(ims)
# annotate data
roll = hv.Dimension("roll", label="Roll Angle", unit="deg")
# pitch = hv.Dimension("pitch", label="Pitch Angle", unit="deg")
yaw = hv.Dimension("yaw", label="Yaw Angle", unit="deg")
time = hv.Dimension("time", label="Time", unit="UTC")
alt = hv.Dimension("flv", label="Flight Level", unit="hectofoot")
layout = hv.Curve(hv_ds, time, roll) + hv.Curve(hv_ds, time, yaw) + hv.Curve(hv_ds, time, alt)
layout.opts(
    opts.Curve(responsive=True, height=300, tools=["hover"], show_grid=True,
               fontsize={"title": 16, "labels": 14, "xticks": 12, "yticks": 12, "legend": 12})
)
layout.opts(title=f"{flight} SMART INS Measurements")
layout.cols(1)
figname = f"{ql_path}/HALO-AC3_HALO_SMART_GPS-INS-fs_{date}_{flight_key}.{output_format}"
hv.save(layout, figname)
print(f"Saved {figname}")


# %% create a unique identifier for each dropsonde and write it to a file
import hashlib
import glob
dropsonde_path = f"{h.get_path('dropsondes', flight, campaign)}/.."
with open(f"{dropsonde_path}/hashfile.txt", "w") as hashfile:
    hashfile.write("dropsonde, hash\n")
    ds_level0 = glob.glob(f"{dropsonde_path}/Level_0/D*[0-9].*")
    for file in ds_level0:
        ds = open(file, "rb")
        string = ds.read()
        hashfile.write(f"{os.path.basename(file)}, ")
        hashfile.write(hashlib.sha1(string).hexdigest())
        hashfile.write("\n")
