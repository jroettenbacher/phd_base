#!/usr/bin/env python
"""Plot for Tina for TAC-5

- BACARDI Timeseries

*author*: Johannes Röttenbacher
"""

# %% import modules
from pylim import helpers as h
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

# %% set options
flight = "Flight_20210723a"

# %% set paths
bacardi_dir = h.get_path("bacardi", flight)
plot_path = f"{h.get_path('plot', flight=flight)}/{flight}"
h.make_dir(plot_path)

# %% read in data
bacardi = xr.open_dataset(f"{bacardi_dir}/CIRRUS-HL_F22_20210723a_ADLR_BACARDI_BroadbandFluxes_v1.1.nc")

# %% time series plot of BACARDI solar and terrestrial irradiance Up and Downward
h.set_cb_friendly_colors()
plt.rc("font", family="serif", size=20)
plt.rc('lines', linewidth=3)

fig, ax = plt.subplots(figsize=(14, 4))
# solar radiation
bacardi.F_up_solar.plot(x="time", label=r"F$_\uparrow$ BACARDI", ax=ax, c="#6699CC", ls="-", linewidth=5)
bacardi.F_down_solar.plot(x="time", label=r"F$_\downarrow$ BACARDI", ax=ax, c="#117733", ls="-", linewidth=5)

# terrestrial radiation
bacardi.F_up_terrestrial.plot(x="time", label=r"F$_\uparrow$ BACARDI", ax=ax, c="#CC6677", ls="-", linewidth=5)
bacardi.F_down_terrestrial.plot(x="time", label=r"F$_\downarrow$ BACARDI", ax=ax, c="#f89c20", ls="-", linewidth=5)
ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Broadband \nIrradiance (W$\,$m$^{-2}$)")
h.set_xticks_and_xlabels(ax, pd.to_timedelta((bacardi.time[-1] - bacardi.time[0]).values))
ax.grid()
handles, labels = ax.get_legend_handles_labels()
legend_column_headers = ["Solar", "Terrestrial"]
handles.insert(0, Patch(color='none', label=legend_column_headers[0]))
handles.insert(3, Patch(color='none', label=legend_column_headers[1]))
# add dummy legend entries to get the right amount of rows per column
handles.append(Patch(color='none', label=""))
handles.append(Patch(color='none', label=""))
ax.legend(handles=handles, loc=1, ncol=3, fontsize=16)
plt.subplots_adjust(bottom=0.28)
plt.tight_layout()
# plt.show()
figname = f"{plot_path}/{flight}_bacardi_broadband_irradiance.png"
plt.savefig(figname, dpi=100)
print(f"Saved {figname}")
plt.close()
