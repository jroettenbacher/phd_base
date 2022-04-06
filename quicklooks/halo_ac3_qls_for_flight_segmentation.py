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

hv.extension('bokeh')

# %% Define flight and paths
campaign="halo-ac3"
date = "20220404"
flight_key = "RF13"
flight = f"HALO-AC3_{date}_HALO_{flight_key}"
ql_path = h.get_path("quicklooks", flight, campaign)
ims_path = h.get_path("horidata", flight, campaign)
h.make_dir(ql_path)
output_format = "html"  # png or html, html gives an interactive plot
cat = ac3airborne.get_intake_catalog()
kwds = {"storage_options": {'simplecache': dict(cache_storage=ims_path, same_names=True)},
        "user": os.environ.get("AC3_CLOUD_USER"), "password": os.environ.get("AC3_CLOUD_PASSWORD")}

# %% Holoviews Dashboard of INS quicklook

ims = cat["HALO-AC3"]["HALO"]["GPS_INS"][f"HALO-AC3_HALO_{flight_key}"](**kwds).to_dask()
hv_ds = hv.Dataset(ims)
# annotate data
roll = hv.Dimension("roll", label="Roll Angle", unit="deg")
pitch = hv.Dimension("pitch", label="Pitch Angle", unit="deg")
yaw = hv.Dimension("yaw", label="Yaw Angle", unit="deg")
time = hv.Dimension("time", label="Time", unit="UTC")
alt = hv.Dimension("alt", label="Altitude", unit="m")
layout = hv.Curve(hv_ds, time, roll) * hv.Curve(hv_ds, time, pitch) + hv.Curve(hv_ds, time, yaw) \
    + hv.Curve(hv_ds, time, alt)
layout.opts(
    opts.Curve(responsive=True, height=300, tools=["hover"], show_grid=True,
               fontsize={"title": 16, "labels": 14, "xticks": 12, "yticks": 12, "legend": 12})
)
layout.opts(title=f"{flight} SMART INS Measurements")
layout.cols(1)
figname = f"{ql_path}/HALO-AC3_HALO_SMART_GPS-INS-fs_{date}_{flight_key}.{output_format}"
hv.save(layout, figname)
print(f"Saved {figname}")
