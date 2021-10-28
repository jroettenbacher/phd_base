#!/usr/bin/env python
"""BACARDI processing as in 00_process_bacardi_V20210928.pro

author: Johannes Röttenbacher
"""

# %% module import
from bacardi import read_bacardi_raw
from smart import get_path
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# %% INPUT / OPTIONS

# fast or full deconvolution
fast_decon = 1  # 0=fast deconvolution, 1=full deconvolution
#  do or do no attitude correction on F_dw solar
correct_attitude = 1
#  Fdw roll and pitch offsets from radiation square 25. June 2021 CIRRUS-HL
#  if you want to play around with these you can uncomment the filename down at the netcdf part
#  to have those offsets included in the filename
roll_offset = -0.15
pitch_offset = 2.85

# %% read BACARDI data
flight = "Flight_20210624a"
bacardi_path = get_path("bacardi", flight)
libradtran_path = get_path("libradtran", flight)
libradtran_file_solar = [f for f in os.listdir(libradtran_path) if "clearsky_bb_simulation_solar" in f][0]
bacardi_ql_file = [f for f in os.listdir(bacardi_path) if "QL" in f][0]
bacardi = read_bacardi_raw(bacardi_ql_file, bacardi_path)

# %% smooth sensor temperature - there is some noise which might influence later temperature dependent corrections
bacardi["F_RBLTR_smooth"] = bacardi["F_RBLTR"].rolling(time=1000, min_periods=1).mean()
bacardi["F_RBSTR_smooth"] = bacardi["F_RBSTR"].rolling(time=1000, min_periods=1).mean()
bacardi["F_RTLTR_smooth"] = bacardi["F_RTLTR"].rolling(time=1000, min_periods=1).mean()
bacardi["F_RTSTR_smooth"] = bacardi["F_RTSTR"].rolling(time=1000, min_periods=1).mean()

# %% plot smoothed and unsmoothed sensor reference temperatures
# fig, axs = plt.subplots(nrows=4, figsize=(12, 9))
# for i, var in enumerate(["F_RBLTR", "F_RBSTR", "F_RTLTR", "F_RTSTR"]):
#     bacardi[var].plot(ax=axs[i], label=var)
#     bacardi[f"{var}_smooth"].plot(ax=axs[i], label=f"{var}_smooth")
#     axs[i].set_xlabel("")
#     axs[i].set_ylabel("Sensor reference\n temperature (K)")
#     axs[i].grid()
#     axs[i].legend()
# axs[-1].set_xlabel("Time (UTC)")
# plt.tight_layout()
# plt.show()
# plt.close()

# %% read libRadtran simulation
libradtran_solar = xr.open_dataset(f"{libradtran_path}/{libradtran_file_solar}")

# %% OWN Application of Sensitivity correction
# the following were last updated on 02.07.2021 with the latest calibration information

# Fdw_solar CMP22 SN 190615
Fdw_sol_sens = 9.01  # uV / (W m^-2)
Fdw_sol_sens_temp_T = np.array([-100, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50])
Fdw_sol_sens_temp_D = np.array([-1.78, -1.78, -0.72, -0.16, 0.13, 0.11, 0.13, 0, 0.1, 0.2, 0.33]) / 100 + 1

# linear interpolation!!! much safer then polynomial fit for extrapolation...
Fdw_sol_sens_temp_corr_fac = np.interp(bacardi["F_RTSTR_smooth"] - 273.15, Fdw_sol_sens_temp_T, Fdw_sol_sens_temp_D)
bacardi["Fdw_sol_sens_temp_corr_fac"] = xr.DataArray(Fdw_sol_sens_temp_corr_fac, dims="time")
bacardi = bacardi.reset_coords(names="Fdw_sol_sens_temp_corr_fac")
bacardi["F_down_solar"] = bacardi["F_RTSTH"] / bacardi["Fdw_sol_sens_temp_corr_fac"]

# Fup_solar CMP22 SN 190613
Fup_sol_sens = 9.11  # uV / (W m ^ -2)
Fup_sol_sens_temp_T = np.array([-100., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50.])
Fup_sol_sens_temp_D = np.array([-2.42, -2.42, -1.2, -0.46, -0.15, -0.15, -0.1, 0., 0.04, 0.32, 0.47]) / 100 + 1

# linear interpolation!!! much safer then polynomial fit for extrapolation...
Fup_sol_sens_temp_corr_fac = np.interp(bacardi["F_RBSTR"] - 273.15, Fup_sol_sens_temp_T, Fup_sol_sens_temp_D)
bacardi["Fup_sol_sens_temp_corr_fac"] = xr.DataArray(Fup_sol_sens_temp_corr_fac, dims="time")
bacardi["F_up_solar"] = bacardi["F_RBSTH"] / bacardi["Fup_sol_sens_temp_corr_fac"]

# Fdw_terr CGR4 SN 190316

Fdw_ter_sens = 9.12  # uV / (W m ^ -2)# uncertainty + -0.44 uV / (W m ^ -2)
Fdw_ter_sens_temp_T = np.array([-100.0, -40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
Fdw_ter_sens_temp_D = np.array([0.98, 0.98, 0.66, 0.23, -0.04, -0.11, -0.09, 0, 0.01, 0, -0.11]) / 100 + 1

# linear interpolation!!! much safer then polynomial fit for extrapolation...
Fdw_ter_sens_temp_corr_fac = np.interp(bacardi["F_RTLTR"] - 273.15, Fdw_ter_sens_temp_T, Fdw_ter_sens_temp_D)
bacardi["Fdw_ter_sens_temp_corr_fac"] = xr.DataArray(Fdw_ter_sens_temp_corr_fac, dims="time")
bacardi["F_down_terrestrial"] = bacardi["F_RTLTH"] / bacardi["Fdw_ter_sens_temp_corr_fac"]

# Fup_terr CGR4 SN 190315

Fup_ter_sens = 9.76  # uV / (W m ^ -2)  # uncertainty + -0.46 uV / (W m ^ -2)
Fup_ter_sens_temp_T = np.array([-100., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50.])
Fup_ter_sens_temp_D = np.array([0.47, 0.47, 0.14, -0.23, -0.40, -0.39, -0.19, 0, 0.1, 0.17, 0.11]) / 100. + 1.

# linear interpolation!!! much safer then polynomial fit for extrapolation...
Fup_ter_sens_temp_corr_fac = np.interp(bacardi["F_RBLTR"] - 273.15, Fup_ter_sens_temp_T, Fup_ter_sens_temp_D)
bacardi["Fup_ter_sens_temp_corr_fac"] = xr.DataArray(Fup_ter_sens_temp_corr_fac, dims="time")
bacardi["F_up_terrestrial"] = bacardi["F_RBLTH"] / bacardi["Fup_ter_sens_temp_corr_fac"]

# %% plot temperature correction factor and corrected stuff
fig, axs = plt.subplots(nrows=3, figsize=(12, 9))
axs[0].plot(Fdw_sol_sens_temp_T, Fdw_sol_sens_temp_D)
axs[0].set_xlabel("Temperature (deg C)")
axs[0].set_ylabel("D")
bacardi["Fdw_sol_sens_temp_corr_fac"].plot(ax=axs[1])
bacardi["F_RTSTH"].plot(ax=axs[2], label="uncorrected")
bacardi["F_down_solar"].plot(ax=axs[2], label="corrected")
axs[0].grid()
axs[1].grid()
axs[2].grid()
axs[2].legend()
plt.tight_layout()
plt.show()
plt.close()

# %% OWN Application of thermal offset correction
# Fdw_solar   CMP22 SN 190615

Fdw_sol_thermal_factor = 230.0  # (W m^-2) / (K s-1)

Fdw_sol_dT = np.gradient(bacardi["F_RTSTR_smooth"], 0.1)
bacardi["F_RTSTR_dt"] = xr.DataArray(Fdw_sol_dT, dims="time")
bacardi["F_RTSTR_dt"] = bacardi["F_RTSTR_dt"].rolling(time=600, min_periods=1).mean()
bacardi["Fdw_sol_thermal_offset"] = bacardi["F_RTSTR_dt"] * Fdw_sol_thermal_factor

bacardi["F_down_solar"] = bacardi["F_down_solar"] - bacardi["Fdw_sol_thermal_offset"]

# Fup_solar   CMP22 SN 190613

Fup_sol_thermal_factor = 453.1  # (W m^-2) / (K s-1)

Fup_sol_dT = np.gradient(bacardi["F_RBSTR_smooth"], 0.1)
bacardi["F_RBSTR_dt"] = xr.DataArray(Fup_sol_dT, dims="time")
bacardi["F_RBSTR_dt"] = bacardi["F_RBSTR_dt"].rolling(time=600, min_periods=1).mean()
bacardi["Fup_sol_thermal_offset"] = bacardi["F_RBSTR_dt"] * Fup_sol_thermal_factor

bacardi["F_up_solar"] = bacardi["F_up_solar"] - bacardi["Fup_sol_thermal_offset"]

# Fdw_terr   CGR4 SN 190316

Fdw_ter_thermal_factor = -528.7  # (W m^-2) / (K s-1)

Fdw_ter_dT = np.gradient(bacardi["F_RTLTR_smooth"], 0.1)
bacardi["F_RTLTR_dt"] = xr.DataArray(Fdw_ter_dT, dims="time")
bacardi["F_RTLTR_dt"] = bacardi["F_RTLTR_dt"].rolling(time=600, min_periods=1).mean()
bacardi["Fdw_ter_thermal_offset"] = bacardi["F_RTLTR_dt"] * Fdw_ter_thermal_factor

bacardi["F_down_terrestrial"] = bacardi["F_down_terrestrial"] - bacardi["Fdw_ter_thermal_offset"]

# Fup_terr   CGR4 SN 190315

Fup_ter_thermal_factor = -547.6  # (W m^-2) / (K s-1)

Fup_ter_dT = np.gradient(bacardi["F_RBLTR_smooth"], 0.1)
bacardi["F_RBLTR_dt"] = xr.DataArray(Fup_ter_dT, dims="time")
bacardi["F_RBLTR_dt"] = bacardi["F_RBLTR_dt"].rolling(time=600, min_periods=1).mean()
bacardi["Fup_ter_thermal_offset"] = bacardi["F_RBLTR_dt"] * Fup_ter_thermal_factor

bacardi["F_up_terrestrial"] = bacardi["F_up_terrestrial"] - bacardi["Fup_ter_thermal_offset"]

# %% Inertia correction
tau_pyrano = 1.20
tau_pyrgeo = 3.3

fcut_pyrano = 0.6
fcut_pyrgeo = 0.5

rm_length_pyrano = 0.5
rm_length_pyrgeo = 2.0

if (fast_decon eq 0) then begin  # apply fast deconvolution
Fdw_sol_decon = decon_rt_fast(Fdw_sol, time_bacardi_sod, tau_pyrano, fcut_pyrano, rm_length_pyrano, 0.1)
Fup_sol_decon = decon_rt_fast(Fup_sol, time_bacardi_sod, tau_pyrano, fcut_pyrano, rm_length_pyrano, 0.1)
Fdw_ter_decon = decon_rt_fast(Fdw_ter, time_bacardi_sod, tau_pyrgeo, fcut_pyrgeo, rm_length_pyrgeo, 0.1)
Fup_ter_decon = decon_rt_fast(Fup_ter, time_bacardi_sod, tau_pyrgeo, fcut_pyrgeo, rm_length_pyrgeo, 0.1)

endif else begin  # apply full deconvolution
Fdw_sol_decon = decon_rt(Fdw_sol, time_bacardi_sod, tau_pyrano, fcut_pyrano, rm_length_pyrano, 0.1)
Fup_sol_decon = decon_rt(Fup_sol, time_bacardi_sod, tau_pyrano, fcut_pyrano, rm_length_pyrano, 0.1)
Fdw_ter_decon = decon_rt(Fdw_ter, time_bacardi_sod, tau_pyrgeo, fcut_pyrgeo, rm_length_pyrgeo, 0.1)
Fup_ter_decon = decon_rt(Fup_ter, time_bacardi_sod, tau_pyrgeo, fcut_pyrgeo, rm_length_pyrgeo, 0.1)
endelse
#  Final processing of Pyrgeometer - add thermal emission

# apply attitude correction for Pyranometer F_dw

#  Quality check data

#  Cloud flag(cloud above)

#  Write Output NC - File
