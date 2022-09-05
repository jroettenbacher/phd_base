#!/usr/bin/env python
"""
==================================================
Functions for processing and plotting BACARDI data
==================================================

--------------
Postprocessing
--------------

Postprocessing of the BACARDI data is done by the DLR and the LIM.
For details on the postprocessing see the IDL postprocessing script.
Some values are filtered out during the postprocessing.
We set an aircraft attitude limit in the processing routine, and if the attitude exceeds this threshold, then the data is filtered out.
For example, this would be the case during sharp turns.
The threshold also takes the attitude correction factor into account.
For CIRRUS-HL, if this factor is below 0.25, then we filter out data where the roll or pitch exceeds 8°.
If this factor is above 0.25, then we begin filtering at 5°.
For EUREC4A, the limits were not as strict because the SZAs were usually higher.
Since this is not the case for the Arctic, something stricter was needed.

**author**: Johannes Röttenbacher
"""

# %% module import
import pylim.helpers as h
from pylim import reader
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Patch
import pandas as pd
import numpy as np
import os
import re
import logging

log = logging.getLogger(__name__)

# %% functions


def read_bacardi_raw(filename: str, path: str) -> xr.Dataset:
    """
    Read raw BACARDI data as provided by DLR

    Args:
        filename: name of file
        path: path to file

    Returns: Dataset with BACARDI data and time as dimension

    """
    filepath = os.path.join(path, filename)
    date = re.search(r"\d{8}", filename)[0]
    ds = xr.open_dataset(filepath)
    ds = ds.rename({"TIME": "time"})
    ds = ds.swap_dims({"tid": "time"})  # make time the dimension
    # overwrite TIME to make a datetime index
    ds = ds.assign({"time": pd.to_datetime(ds.time, unit='ms', origin=pd.to_datetime(date, format="%Y%m%d"))})

    return ds


def fdw_attitude_correction(fdw, roll, pitch, yaw, sza, saa, fdir, r_off: float = 0, p_off: float = 0):
    """Attitude Correction for downward irradiance.
    Corrects downward irradiance for misalignment of the sensor (deviation from horizontal alignment).

    - only direct fraction of irradiance can be corrected by the equation, therefore a direct fraction (fdir) has to be provided
    - please check correct definition of the attitude angle
    - for differences between the sensor attitude and the attitude given by an INS the offset angles (p_off and r_off) can be defined.

    Args:
        fdw: downward irradiance [W m-2] or [W m-2 nm-1]
        roll: roll angle [deg] - defined positive for left wing up
        pitch: pitch angle [deg] - defined positive for nose down
        yaw: yaw angle [deg] - defined clockwise with North=0°
        sza: solar zenith angle [deg]
        saa: solar azimuth angle [deg] - defined clockwise with North=0°
        r_off: roll offset angle between INS and sensor [deg] - defined positive for left wing up
        p_off: pitch offset angle between INS and sensor [deg] - defined positive for nose down
        fdir: fraction of direct radiation [0..1] (0=pure diffuse, 1=pure direct)

    Returns: corrected downward irradiance [W m-2] or [W m-2 nm-1]

    """
    r = np.deg2rad(roll + r_off)
    p = np.deg2rad(pitch + p_off)
    h0 = np.deg2rad(90 - sza)

    factor = np.sin(h0) / \
             (np.cos(h0) * np.sin(r) * np.sin(np.deg2rad(saa - yaw)) +
              np.cos(h0) * np.sin(p) * np.cos(r) * np.cos(np.deg2rad(saa - yaw)) +
              np.sin(h0) * np.cos(p) * np.cos(r))
    try:
        fdw_cor = fdir * fdw * factor + (1 - fdir) * fdw
    except ValueError:
        # fdw and fdir are 2 dimensional, add an empty axis to the factor two make it 2D as well
        fdw_cor = fdir * fdw * factor[:, None] + (1 - fdir) * fdw

    return fdw_cor, factor


def decon_rt(xdatacon, xtime, xrsp_time, xfcut, xrm_length, xdt, NO_RM=False, xfsigma=False,
             show_spectra="show_spectra", spectra_out="spectra"):
    """
    Deconvolution of a time series smoothed due to the respons time of the sensor

    - Deconvolution is applied via the convolution theorem using fourier transformation
    - see Numerical Recipies 13.1
    - additional filters are applied to reduce the influence of sensor noise
        - cuting off the fourier series at noise level
        - additional running mean filter (rectangular window)

    Args:
        xtime      ... has to be equidistant [s]
        xdatacon   ... data series as measured
        xrsp_time  ... 1/e response time of the sensor [s]
        xfcut      ... Cut off frequenzy for noise filtering in [Hz]
        xrm_length ... Window size of the running mean filter in [s]
        xdt        ... Time step between two measurements [s]

    Returns: xdata      ... deconvoluted data

    Optional output:
        spectra_out ... The name of the variable to receive the fourier coefficients calculated within the routine.
            The output is an array containing following parameter
            [0] ... frequency
            [1] ... fourier coeff. of original data
            [2] ... fourier coeff. of deconvolutetd data
            [3] ... fourier coeff. of deconvolutetd data + cut if freq. applied
            [4] ... fourier coeff. of deconvolutetd data + cut if freq. + running mean applied
            [5] ... fourier coeff. of convolution function

    PARAMETERS:
        /NO_RM ...  If set, the running mean filter is not applied
        /show_spectra ... can be specified to give a plot of the power spectra as calculated within the routine
            "ENTER" hast to be pressed to continue the calculation after the plot opened.
        SIGMA=*.** ... choose to apply the Lanczos sigma factor reducing the Gibbs-Phenomenon
            value given to SIGMA=*.**  is the max. frequency xfsigma for which Sigma-Approx. is applied
            ==> SIGMA is somehow redundant as the running mean makes nothing else than sigma approximation...

    """

    xdatacon = xdatacon.flatten()
    xtime = xtime.flatten()

    pi = np.pi

    xnt = len(xtime)

    if (xnt % 2) == 1:
        xdatacon = xdatacon[1:]
    if (xnt % 2) == 1:
        xtime = xtime[1:]
    if (xnt % 2) == 1:
        xnt = xnt - 1

    # ==============================================================================
    # create convolution function
    # ==============================================================================

    xfcon_all = 1 / (xrsp_time * np.exp((np.linspace(0, xnt - 1, num=xnt) * xdt) / xrsp_time))

    xt99 = xrsp_time * np.log(1 / 0.00001)  # time when contribution is less then 99.999 %
    ximax = int(xt99 / xdt)

    if ximax % 2 == 1:
        ximax = ximax + 1

    xzero = np.zeros(ximax)
    xfcon = np.append(xfcon_all, xzero)

    # ==============================================================================
    # Fouriertrafo of time series and convolution function
    # ==============================================================================

    # add zeros to avoid wrap problem

    xzero = np.full(ximax, xdatacon[0])
    xdataconzero = np.append(xdatacon, xzero)

    # fourier transformation
    xftdatacon = np.fft.fft(xdataconzero)
    ##xftfcon1=fft(xfcon,/double)*(xnt+ximax)*xdt#/(pi/2./xrsp_time)   #scalierung funftioniert hier nicht richtig

    xftfcon = np.fft.fft(xfcon)
    xftfcon = xftfcon / xftfcon[0]

    xntzero = xnt + ximax
    xfreq = np.linspace(0, (xntzero / 2) - 1, num=xntzero) / xdt / xntzero

    # =======================================================================================
    # Calculate fourier transform of boxcar function (running mean window) analytically
    # =======================================================================================

    # "xrm_length/xrm_length"=1  but keep it as it is part of the equations, once in the fourier transform
    #  and once in the normation of the weighting (boxcar function not 1 but 1/xrm_length)
    xftrm = xrm_length / xrm_length * np.sin(pi * xrm_length * xfreq) / (pi * xrm_length * xfreq)
    xftrm[0] = 1
    xftrm = np.append(xftrm, np.flip(xftrm))

    # ==============================================================================
    # De-Convolution of fourier coefficients
    # ==============================================================================

    xftdata0 = xftdatacon / xftfcon
    xftdata1 = xftdata0

    # filter noise of sensor by cutting off back transformation
    xifcut = np.where(xfreq > xfcut)[0][0]
    xftdata1[xifcut:(xntzero - xifcut)] = np.complex(0, 0)

    if NO_RM:
        xftdata = xftdata1
    else:
        xftdata = xftdata1 * xftrm

    # ==============================================================================
    # Backtransformation into time series
    # ==============================================================================

    # not implemented yet
    # apply Sigma-Approximation to minimize Gibbs-Phenomenon
    # if xfsigma:
    ##x=indgen(100)/10.-5.
    ##a=1
    ##
    ##lanczos=sin(!pi*x)/(!pi*x)*sin(!pi*x/a)/(!pi*x/a)
    ##lanczos(50)=1
    ##
    ##plot,x,lanczos
    ##

    # xisigma = np.where(xfreq > xfsigma)[0][0]

    # if xisigma != -1: n2=xisigma

    # sigma = pi * ((indgen(n2) + 1) * 1d) / n2
    # xftdata(1: n2)=xftdata(1: n2)*sin(sigma) / sigma
    # xftdata(xntzero - n2 + 1: *)=xftdata(xntzero - n2 + 1: *)*sin(reverse(sigma)) / reverse(sigma)
    # else:
    #     xftdata = xftdata

    # ==========================================

    xdatazero = np.real(np.fft.ifft(xftdata))
    xdata = xdatazero[0:(xnt - 1)]

    return xdata


if __name__ == '__main__':
    # %% set paths
    flight = "Flight_20210719a"
    bacardi_path = h.get_path("bacardi", flight)
    ql_path = bacardi_path

    # %% read in bacardi data
    filename = dict(Flight_20210629a="CIRRUS_HL_F05_20210629a_ADLR_BACARDI_BroadbandFluxes_R0.nc",
                    Flight_20210719a="CIRRUS_HL_F18_20210719a_ADLR_BACARDI_BroadbandFluxes_R0.nc")
    ds = xr.open_dataset(f"{bacardi_path}/{filename[flight]}")

    # %% read in libRadtran simulations
    libradtran_file = f"BBR_Fdn_clear_sky_{flight}_R0_ds_high.dat"
    libradtran_file_ter = f"BBR_Fdn_clear_sky_{flight}_R0_ds_high_ter.dat"
    bbr_sim = reader.read_libradtran(flight, libradtran_file)
    bbr_sim_ter = reader.read_libradtran(flight, libradtran_file_ter)

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
    bbr_sim_ter.plot(y="F_dw", ax=ax, ylabel=r"Broadband irradiance (W$\,$m$^{-2}$)",
                     label=r"$F_{\downarrow}$ libRadtran",
                     c="#f89c20", ls="--", path_effects=[patheffects.withStroke(linewidth=6, foreground="k")])
    ax.set_xlabel(r"Time (UTC)")
    h.set_xticks_and_xlabels(ax, pd.to_timedelta((ds.time[-1] - ds.time[0]).values))
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
