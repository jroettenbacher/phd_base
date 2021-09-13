#!/usr/bin/env python
"""Script for processing and plotting BACARDI data
author: Johannes Röttenbacher
"""

# %% module import
from smart import get_path
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Patch
from functions_jr import set_xticks_and_xlabels
from libradtran import read_libradtran
import pandas as pd

# %% set paths
flight = "Flight_20210719a"
bacardi_path = get_path("bacardi", flight)
ql_path = bacardi_path

# %% read in bacardi data
filename = dict(Flight_20210629a="CIRRUS_HL_F05_20210629a_ADLR_BACARDI_BroadbandFluxes_R0.nc",
                Flight_20210719a="CIRRUS_HL_F18_20210719a_ADLR_BACARDI_BroadbandFluxes_R0.nc")
ds = xr.open_dataset(f"{bacardi_path}/{filename[flight]}")

# %% read in libRadtran simulations
libradtran_file = f"BBR_Fdn_clear_sky_{flight}_R0_ds_high.dat"
libradtran_file_ter = f"BBR_Fdn_clear_sky_{flight}_R0_ds_high_ter.dat"
bbr_sim = read_libradtran(flight, libradtran_file)
bbr_sim_ter = read_libradtran(flight, libradtran_file_ter)

# %% BACARDI and libRadtran quicklooks
plt.rc('font', size=14)
plt.rc('lines', linewidth=3)
plt.rc('font', family="serif")
# x_sel = (pd.Timestamp(2021, 6, 29, 10), pd.Timestamp(2021, 6, 29, 12, 15))
fig, ax = plt.subplots(figsize=(9, 5.5))
# solar radiation
ds.F_up_solar.plot(x="time", label=r"$F_{\uparrow}$ BACARDI", ax=ax, c="#6699CC", ls="-")
ds.F_down_solar.plot(x="time", label=r"$F_{\downarrow}$ BACARDI", ax=ax, c="#117733", ls="-")
bbr_sim.plot(y="F_up", ax=ax, label=r"$F_{\uparrow}$ libRadtran", c="#6699CC", ls="--",
             path_effects=[patheffects.withStroke(linewidth=6, foreground="k")])
bbr_sim.plot(y="F_dw", ax=ax, ylabel=r"Broadband irradiance (W$\,$m$^{-2}$)", label=r"$F_{\downarrow}$ libRadtran",
             c="#117733", ls="--", path_effects=[patheffects.withStroke(linewidth=6, foreground="k")])
# terrestrial radiation
ds.F_up_terrestrial.plot(x="time", label=r"$F_{\uparrow}$ BACARDI", ax=ax, c="#CC6677", ls="-")
ds.F_down_terrestrial.plot(x="time", label=r"$F_{\downarrow}$ BACARDI", ax=ax, c="#f89c20", ls="-")
bbr_sim_ter.plot(y="F_up", ax=ax, label=r"$F_{\uparrow}$ libRadtran", c="#CC6677", ls="--",
                 path_effects=[patheffects.withStroke(linewidth=6, foreground="k")])
bbr_sim_ter.plot(y="F_dw", ax=ax, ylabel=r"Broadband irradiance (W$\,$m$^{-2}$)", label=r"$F_{\downarrow}$ libRadtran",
                 c="#f89c20", ls="--", path_effects=[patheffects.withStroke(linewidth=6, foreground="k")])
ax.set_xlabel(r"Time (UTC)")
set_xticks_and_xlabels(ax, pd.to_timedelta((ds.time[-1]-ds.time[0]).values))
ax.grid()
handles, labels = ax.get_legend_handles_labels()
legend_column_headers = ["Solar", "Terrestrial"]
handles.insert(0, Patch(color='none', label=legend_column_headers[0]))
handles.insert(5, Patch(color='none', label=legend_column_headers[1]))
# add dummy legend entries to get the right amount of rows per column
# handles.append(Patch(color='none', label=""))
# handles.append(Patch(color='none', label=""))
ax.legend(handles=handles, bbox_to_anchor=(0.5, 0), loc="lower center", ncol=2, bbox_transform=fig.transFigure)
plt.subplots_adjust(bottom=0.45)
plt.tight_layout()
# plt.show()
plt.savefig(f"{ql_path}/CIRRUS_HL_{flight}_bacardi_libradtran_broadband_irradiance.png", dpi=100)
plt.close()

