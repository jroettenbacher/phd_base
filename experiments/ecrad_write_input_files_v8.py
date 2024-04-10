#!/usr/bin/env python
"""
| *author*: Johannes Röttenbacher
| *created*: 23-03-2023

Use a processed IFS output file and the Varcloud retrieval from Florian Ewald, LMU and generate one ecRad input file for each time step along the flight path of HALO.
Instead of the *CIWC* and :math:`r_{eff, ice}` from the IFS use the retrieved variables from the Varcloud lidar/radar retrieval.

**Required User Input:**

All options can be set in the script or given as command line key=value pairs.
The first possible option is the default.

* campaign, (default: 'halo-ac3')
* key, flight key (default: 'RF17')
* t_interp, interpolate IFS data in time? (default: False)
* init_time, initalization time of IFS run (00, 12, yesterday)
* o3_source, which ozone concentration to use? (one of '47r1', 'ifs', 'constant', 'sonde')
* trace_gas_source, which trace gas concentrations to use? (one of '47r1', 'constant')
* aerosol_source, which aersol concentrations to use? (one of '47r1', 'ADS')

**Output:**

* well documented ecRad input file in netCDF format for each time step with retrieved *CIWC* and :math:`re_{eff, ice}`

"""
if __name__ == "__main__":
    # %% module import
    import pylim.helpers as h
    import pylim.solar_position as sp
    import pylim.meteorological_formulas as met
    from pylim.ecrad import apply_liquid_effective_radius
    from metpy.calc import density, mixing_ratio_from_specific_humidity
    from metpy.units import units
    import numpy as np
    import xarray as xr
    from sklearn.neighbors import BallTree
    import os
    import pandas as pd
    from datetime import datetime
    from tqdm import tqdm
    import time

    start = time.time()
    # %% read in command line arguments
    version = "v8"
    args = h.read_command_line_args()
    campaign = args["campaign"] if "campaign" in args else "halo-ac3"
    key = args["key"] if "key" in args else "RF17"
    t_interp = h.strtobool(args["t_interp"]) if "t_interp" in args else False
    init_time = args["init"] if "init" in args else "00"
    o3_source = args["o3_source"] if "o3_source" in args else "47r1"
    trace_gas_source = args["trace_gas_source"] if "trace_gas_source" in args else "47r1"
    aerosol_source = args["aerosol_source"] if "aerosol_source" in args else "47r1"

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
    __file__ = None if '__file__' not in locals() else __file__
    log = h.setup_logging('./logs', __file__, key)
    # print options to user
    log.info(f"Options set: \ncampaign: {campaign}\nkey: {key}\nflight: {flight}\ndate: {date}\n"
             f"init time: {init_time}\nt_interp: {t_interp}\nversion: {version}\n"
             f"O3 source: {o3_source}\nTrace gas source: {trace_gas_source}\n"
             f"Aerosol source: {aerosol_source}\n")

    # %% set paths
    ifs_path = os.path.join(h.get_path("ifs", campaign=campaign), date)
    ecrad_path = os.path.join(h.get_path("ecrad", campaign=campaign), date, "ecrad_input")
    varcloud_path = h.get_path("varcloud", flight, campaign)
    varcloud_file = [f for f in os.listdir(varcloud_path) if f.endswith(".nc")][0]
    cams_path = h.get_path("cams", campaign=campaign)
    trace_gas_file = f"trace_gas_mm_climatology_2020_{trace_gas_source}_{date}.nc"
    aerosol_file = f"aerosol_mm_climatology_2020_{aerosol_source}_{date}.nc"
    # create output path
    h.make_dir(ecrad_path)

    # %% read in intermediate files from read_ifs
    if init_time == "yesterday":
        ifs_date = int(date) - 1
        init_time = 12
    else:
        ifs_date = date

    data_ml = xr.open_dataset(f"{ifs_path}/ifs_{ifs_date}_{init_time}_ml_O1280_processed.nc")
    data_ml = data_ml.set_index(rgrid=["lat", "lon"])
    varcloud_ds = (xr.open_dataset(f"{varcloud_path}/{varcloud_file}")
                   .swap_dims(time="Time", height="Height")
                   .rename(Time="time"))
    # %% read in trace gas and aerosol data
    trace_gas = xr.open_dataset(f"{cams_path}/{trace_gas_file}")
    aerosol = xr.open_dataset(f"{cams_path}/{aerosol_file}")

    # %% resample varcloud data to minutely resolution
    varcloud_ds = varcloud_ds.resample(time="1min").mean()

    # %% select lat and lon closest to flightpath
    lats, lons, times = varcloud_ds.Latitude, varcloud_ds.Longitude, varcloud_ds.time
    ifs_lat_lon = np.column_stack((data_ml.lat, data_ml.lon))
    ifs_tree = BallTree(np.deg2rad(ifs_lat_lon), metric="haversine")  # build the kd tree for nearest neighbour look up
    # generate an array with lat, lon values from the flight position
    points = np.deg2rad(np.column_stack((lats.to_numpy(), lons.to_numpy())))
    dist, idxs = ifs_tree.query(points, k=1)  # query the tree
    closest_latlons = ifs_lat_lon[idxs.flatten()]
    # a sphere with radius 1 is assumed so multiplying by Earth's radius gives the distance in km
    distances = dist.flatten() * 6371

    # %% loop through time steps and write one file per time step
    idx = len(times)
    for i in tqdm(range(idx), desc="Time loop"):
        latlon_sel = (closest_latlons[i][0], closest_latlons[i][1])
        t = times[i]
        ds = data_ml.sel(rgrid=latlon_sel).reset_coords(["rgrid", "lat", "lon"]).drop_vars("rgrid").drop_dims(
            "reduced_points")
        if t_interp:
            ds = ds.interp(time=times[i]).reset_coords("time")  # interpolate to time step
            ending = "_inp"
        else:
            ds = ds.sel(time=t, method="nearest").reset_coords("time")  # select closest time step
            ending = ""

        varcloud_sel = varcloud_ds.sel(time=t)
        # interpolate varcloud height to model height
        varcloud_sel = varcloud_sel.interp(Height=ds.press_height_full).reset_coords(["time", "Height"])
        # convert kg/m3 to kg/kg
        mixing_ratio = mixing_ratio_from_specific_humidity(ds["q"] * units("kg/kg"))
        air_density = density(ds.pressure_full * units.Pa, ds.t * units.K, mixing_ratio)
        q_ice = varcloud_sel["Varcloud_Cloud_Ice_Water_Content"] * units("kg/m3") / air_density
        # overwrite ice water content
        ds["q_ice"] = q_ice.metpy.dequantify().where(~np.isnan(q_ice), 0)

        # add cos_sza for the grid point using model data for the thermodynamics and aircraft data for the location
        sod = (t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second).to_numpy()
        p_surf_nearest = ds.pressure_hl.isel(half_level=137).to_numpy() / 100  # hPa
        t_surf_nearest = ds.temperature_hl.isel(half_level=137).to_numpy() - 273.15  # degree Celsius
        ypos = varcloud_sel.Latitude.to_numpy()
        xpos = varcloud_sel.Longitude.to_numpy()
        sza = sp.get_sza(sod / 3600, ypos, xpos, dt_day.year, dt_day.month, dt_day.day, p_surf_nearest, t_surf_nearest)
        cos_sza = np.cos(sza / 180. * np.pi)

        ds["cos_solar_zenith_angle"] = xr.DataArray(cos_sza,
                                                    attrs=dict(unit="1",
                                                               long_name="Cosine of the solar zenith angle"))

        # add sw_albedo_direct to account for direct reflection of solar incoming radiation above ocean
        sw_albedo_bands = list()
        open_ocean_albedo_taylor = met.calculate_open_ocean_albedo_taylor(cos_sza)
        for ii in range(h.ci_albedo.shape[1]):
            sw_albedo_bands.append(ds.ci * h.ci_albedo[3, ii]
                                   + (1. - ds.ci) * open_ocean_albedo_taylor)

        sw_albedo_direct = xr.concat(sw_albedo_bands, dim="sw_albedo_band")
        sw_albedo_direct.attrs = dict(unit=1, long_name="Banded direct short wave albedo")
        ds["sw_albedo_direct"] = sw_albedo_direct

        # interpolate trace gas data onto ifs full pressure levels
        new_pressure = ds.pressure_full.to_numpy()
        tg = (trace_gas
              .sel(time=t, method="nearest")
              .interp(level=new_pressure,
                      kwargs={"fill_value": 0})
              .reset_coords("time", drop=True))

        # read out trace gases from trace gas file
        tg_vars = ["cfc11_vmr", "cfc12_vmr", "ch4_vmr", "co2_vmr", "n2o_vmr", "o3_vmr"]
        for var in tg_vars:
            ds[var] = tg[var].assign_coords(level=ds.level)

        # overwrite the trace gases with the variables corresponding to trace_gas_source
        if trace_gas_source == "constant":
            for var in tg_vars:
                ds[var] = ds[f"{var}_{trace_gas_source}"]
        # overwrite ozone according to o3_source
        if o3_source != "47r1":
            ds["o3_vmr"] = ds[f"o3_vmr_{o3_source}"]

        # interpolate aerosol dataset to ifs full pressure levels
        # and turn it into a data array with one new dimension: aer_type
        aerosol_mmr = (aerosol
                       .sel(time=t, method="nearest")
                       .assign(level=(aerosol
                                      .sel(time=t, method="nearest")["full_level_pressure"]
                                      .to_numpy()))
                       .interp(level=new_pressure,
                               kwargs={"fill_value": 0})
                       .drop_vars(["half_level_pressure", "full_level_pressure",
                                   "half_level_delta_pressure"])
                       .to_array(dim="aer_type")
                       .assign_coords(aer_type=np.arange(1, 12),
                                      level=ds.level)
                       .reset_coords("time", drop=True))
        ds["aerosol_mmr"] = aerosol_mmr

        # assign effective radius
        re_ice = varcloud_sel["Varcloud_Cloud_Ice_Effective_Radius"]
        ds["re_ice"] = re_ice.where(~np.isnan(re_ice), 51.9616 * 1e-6)  # replace nan with default value
        ds = apply_liquid_effective_radius(ds)
        ds = ds.expand_dims("column", axis=0)
        # remove column dim from dimensionless variables
        for var in ["o2_vmr", "time"]:
            ds[var] = ds[var].isel(column=0)
        n_column = ds.dims["column"]  # get number of columns
        ds["column"] = np.arange(n_column)

        # add distance to aircraft location
        ds["distance"] = xr.DataArray(np.expand_dims(distances[i], axis=0), dims="column",
                                      attrs=dict(long_name="distance", units="km",
                                                 description="Haversine distance to aircraft location"))

        ds = ds.transpose("column", ...)  # move column to the first dimension
        ds = ds.astype("float32")  # change type from double to float32

        ds.to_netcdf(
            path=f"{ecrad_path}/ecrad_input_standard_{sod:7.1f}_sod{ending}_v8.nc",
            format='NETCDF4_CLASSIC')

    log.info(f"Done with date {date}: {pd.to_timedelta((time.time() - start), unit='second')} (hr:min:sec)")
