#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 23-03-2023

As v2 but change the time of the input files to match the below cloud section in order to compare ecRad with Varcloud input with BACARDI measurements.
This will remove the influence of the changing solar zenith angle.

**Required User Input:**

All options can be set in the script or given as command line key=value pairs.
The first possible option is the default.

* key (RF17), flight key
* t_interp (False), interpolate time or use the closest time step
* init_time (00, 12, yesterday), initialization time of the IFS model run

**Output:**

* well documented ecRad input file in netCDF format for each time step with retrieved *CIWC* and :math:`re_{eff, ice}`

"""

if __name__ == "__main__":
    # %% module import
    import pylim.helpers as h
    import pylim.solar_position as sp
    from pylim.ecrad import apply_liquid_effective_radius, calculate_pressure_height
    from pylim import reader
    import ac3airborne
    from ac3airborne.tools import flightphase
    from metpy.calc import density
    from metpy.units import units
    import numpy as np
    import xarray as xr
    import os
    import pandas as pd
    from datetime import datetime
    from tqdm import tqdm
    import time
    from distutils.util import strtobool

    start = time.time()
    # %% read in command line arguments
    campaign = "halo-ac3"
    version = "v3"
    args = h.read_command_line_args()
    key = args["key"] if "key" in args else "RF17"
    # set interpolate flag
    t_interp = strtobool(args["t_interp"]) if "t_interp" in args else False  # interpolate between timesteps?
    init_time = args["init"] if "init" in args else "00"
    if campaign == "halo-ac3":
        import pylim.halo_ac3 as meta

        flight = meta.flight_names[key]
        date = flight[9:17]
    else:
        import pylim.cirrus_hl as meta

        flight = key
        date = flight[7:15]
    dt_day = datetime.strptime(date, '%Y%m%d')  # convert date to date time for further use
    # setup logging
    try:
        file = __file__
    except NameError:
        file = None
    log = h.setup_logging("./logs", file, key)
    # print options to user
    log.info(f"Options set: \ncampaign: {campaign}\nflight: {flight}\ndate: {date}"
             f"\ninit time: {init_time}\nt_interp: {t_interp}")

    # %% get flight segments for case study period
    segmentation = ac3airborne.get_flight_segments()["HALO-AC3"]["HALO"][f"HALO-AC3_HALO_{key}"]
    segments = flightphase.FlightPhaseFile(segmentation)
    above_cloud, below_cloud = dict(), dict()
    if key == "RF17":
        above_cloud["start"] = segments.select("name", "high level 7")[0]["start"]
        above_cloud["end"] = segments.select("name", "high level 8")[0]["end"]
        below_cloud["start"] = segments.select("name", "high level 9")[0]["start"]
        below_cloud["end"] = segments.select("name", "high level 10")[0]["end"]
        above_slice = slice(above_cloud["start"], above_cloud["end"])
        below_slice = slice(below_cloud["start"], below_cloud["end"])
        case_slice = slice(pd.to_datetime("2022-04-11 10:30"), pd.to_datetime("2022-04-11 12:29"))
    else:
        above_cloud["start"] = segments.select("name", "polygon pattern 1")[0]["start"]
        above_cloud["end"] = segments.select("name", "polygon pattern 1")[0]["parts"][-1]["start"]
        below_cloud["start"] = segments.select("name", "polygon pattern 2")[0]["start"]
        below_cloud["end"] = segments.select("name", "polygon pattern 2")[0]["end"]
        above_slice = slice(above_cloud["start"], above_cloud["end"])
        below_slice = slice(below_cloud["start"], below_cloud["end"])
        case_slice = slice(above_cloud["start"], below_cloud["end"])

    time_extend_cs = below_cloud["end"] - above_cloud["start"]  # time extend for case study

    # %% set paths
    path_ifs_output = os.path.join(h.get_path("ifs", campaign=campaign), date)
    path_ecrad = os.path.join(h.get_path("ecrad", campaign=campaign), date, "ecrad_input")
    path_varcloud = h.get_path("varcloud", flight, campaign)
    file_varcloud = [f for f in os.listdir(path_varcloud) if f.endswith(".nc")][0]
    path_bahamas = h.get_path("bahamas", flight, campaign)
    file_bahamas = f"HALO-AC3_HALO_BAHAMAS_{date}_{key}_v1.nc"
    # create output path
    h.make_dir(path_ecrad)

    # %% read in file from read_ifs
    if init_time == "yesterday":
        ifs_date = int(date) - 1
        init_time = 12
    else:
        ifs_date = date

    data_ml = xr.open_dataset(f"{path_ifs_output}/ifs_{ifs_date}_{init_time}_ml_processed.nc")
    varcloud_ds = xr.open_dataset(f"{path_varcloud}/{file_varcloud}").swap_dims(time="Time", height="Height").rename(
        Time="time")
    bahamas_ds = reader.read_bahamas(f"{path_bahamas}/{file_bahamas}")

    # %% select only case study time which features the cloud that HALO also underpassed
    sel_time = slice(pd.to_datetime("2022-04-11 10:49"), pd.to_datetime("2022-04-11 11:04"))
    fake_time = pd.date_range("2022-04-11 11:35", "2022-04-11 11:50", freq="1s")

    # %% resample varcloud data to minutely resolution
    new_index = pd.date_range(str(varcloud_ds.time[0].astype('datetime64[s]').to_numpy()),
                              str(varcloud_ds.time[-1].astype('datetime64[s]').to_numpy()), freq="1s")
    varcloud_ds = varcloud_ds.reindex(time=new_index, method="bfill")
    varcloud_ds = varcloud_ds.sel(time=sel_time)

    # %% select lat and lon closest to flightpath
    lats, lons, times = varcloud_ds.Latitude, varcloud_ds.Longitude, varcloud_ds.time
    if t_interp:
        ds_sel = data_ml.sel(lat=lats[0], lon=lons[0], method="nearest").reset_coords(["lat", "lon"])
        ds_sel = ds_sel.interp(time=times[0])
        for i in tqdm(range(1, len(lats)), desc="Select closest IFS data"):
            tmp = data_ml.sel(lat=lats[i], lon=lons[i], method="nearest").reset_coords(["lat", "lon"])
            tmp = tmp.interp(time=times[i])
            ds_sel = xr.concat([ds_sel, tmp], dim="time")
        ending = "inp"

    else:
        ds_sel = data_ml.sel(lat=lats[0], lon=lons[0], time=times[0], method="nearest").reset_coords(["lat", "lon"])
        ds_sel["time"] = times[0]
        for i in tqdm(range(1, len(lats))):
            tmp = data_ml.sel(lat=lats[i], lon=lons[i], time=times[i], method="nearest").reset_coords(["lat", "lon"])
            tmp["time"] = times[i]
            ds_sel = xr.concat([ds_sel, tmp], dim="time")
        ending = ""

    # %% calculate pressure height for model levels if needed
    if "press_height_full" in ds_sel.data_vars:
        log.debug("Pressure Height already in dataset. Moving on.")
        pass
    else:
        ds_sel = calculate_pressure_height(ds_sel)

    # %% loop through time steps and write one file per time step
    for i in tqdm(range(len(varcloud_ds.time)), desc="Time loop"):
        dsi_ml_out = ds_sel.isel(time=i).reset_coords("time")
        varcloud_sel = varcloud_ds.isel(time=i)
        # interpolate varcloud height to model height
        varcloud_sel = varcloud_sel.interp(Height=dsi_ml_out.press_height_full).reset_coords(["time", "Height"])
        # convert kg/m3 to kg/kg
        air_density = density(dsi_ml_out.pressure_full * units.Pa, dsi_ml_out.t * units.K,
                              dsi_ml_out.q * units("kg/kg"))
        q_ice = varcloud_sel["Varcloud_Cloud_Ice_Water_Content"] * units("kg/m3") / air_density
        # overwrite ice water content
        dsi_ml_out["q_ice"] = q_ice.metpy.dequantify().where(~np.isnan(q_ice), 0)
        t = fake_time[i]

        # add cos_sza for the grid point using only model data
        sod = t.hour * 3600 + t.minute * 60 + t.second
        p_surf_nearest = dsi_ml_out.pressure_hl.isel(half_level=137).to_numpy() / 100  # hPa
        t_surf_nearest = dsi_ml_out.temperature_hl.isel(half_level=137).to_numpy() - 273.15  # degree Celsius
        ypos = bahamas_ds.IRS_LAT.sel(time=t).to_numpy()
        xpos = bahamas_ds.IRS_LON.sel(time=t).to_numpy()
        sza = sp.get_sza(sod / 3600, ypos, xpos, dt_day.year, dt_day.month, dt_day.day, p_surf_nearest, t_surf_nearest)
        cos_sza = np.cos(sza / 180. * np.pi)

        dsi_ml_out["cos_solar_zenith_angle"] = xr.DataArray(cos_sza,
                                                            attrs=dict(unit="1",
                                                                       long_name="Cosine of the solar zenith angle"))

        # assign effective radius
        re_ice = varcloud_sel["Varcloud_Cloud_Ice_Effective_Radius"]
        dsi_ml_out["re_ice"] = re_ice.where(~np.isnan(re_ice), 51.9616 * 1e-6)  # replace nan with default value
        dsi_ml_out = apply_liquid_effective_radius(dsi_ml_out)
        dsi_ml_out = dsi_ml_out.expand_dims("column", axis=0)
        # remove column dim from dimensionless variables
        for var in ["co2_vmr", "n2o_vmr", "ch4_vmr", "o2_vmr", "cfc11_vmr", "cfc12_vmr", "time"]:
            dsi_ml_out[var] = dsi_ml_out[var].isel(column=0)
        n_column = dsi_ml_out.dims["column"]  # get number of columns
        dsi_ml_out["column"] = np.arange(1, n_column + 1)
        dsi_ml_out = dsi_ml_out.astype(np.float32)  # change type from double to float32

        dsi_ml_out.to_netcdf(
            path=f"{path_ecrad}/ecrad_input_standard_{sod:7.1f}_sod{ending}_{version}.nc",
            format='NETCDF4_CLASSIC')

    log.info(f"Done with date {date}: {pd.to_timedelta((time.time() - start), unit='second')} (hr:min:sec)")