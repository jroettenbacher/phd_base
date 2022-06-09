

import numpy as np
import xarray as xr
from scipy import spatial
import os
import ac3airborne
import pandas as pd
from read_sea_ice import read as read_sea_ice


"""
Script calculates the sea ice concentration along flight track for every flight 
during all airborne campaigns
"""


def transform_coordinates(coords):
    """ Transform coordinates from geodetic to cartesian

    Keyword arguments:
    coords - a set of lan/lon coordinates (e.g. a tuple or 
             an array of tuples)
    """
    # WGS 84 reference coordinate system parameters
    A = 6378.137 # major axis [km]   
    E2 = 6.69437999014e-3 # eccentricity squared    

    coords = np.asarray(coords).astype('float64')

    # is coords a tuple? Convert it to an one-element array of tuples
    if coords.ndim == 1:
        coords = np.array([coords])

    # convert to radiants
    lat_rad = np.radians(coords[:,0])
    lon_rad = np.radians(coords[:,1]) 

    # convert to cartesian coordinates
    r_n = A / (np.sqrt(1 - E2 * (np.sin(lat_rad) ** 2)))
    x = r_n * np.cos(lat_rad) * np.cos(lon_rad)
    y = r_n * np.cos(lat_rad) * np.sin(lon_rad)
    z = r_n * (1 - E2) * np.sin(lat_rad)

    return np.column_stack((x, y, z))


if __name__ == '__main__':
    
    # interpolate missing gps information: necessary to get exact times over surfaces
    interpolated = False
    
    # read meta data
    meta = ac3airborne.get_flight_segments()
    cat = ac3airborne.get_intake_catalog()
    
    ac3cloud_username = os.environ['AC3_USER']
    ac3cloud_password = os.environ['AC3_PASSWORD']
    credentials = dict(user=ac3cloud_username, password=ac3cloud_password)
    
    kwds = {'simplecache': dict(
        cache_storage='/net/secaire/nrisse/data/halo-ac3/',
        same_names=True,
    )}
    
    #%% calculate sea ice on path for each flight
    for mission in meta.keys():
        
        if mission == 'PAMARCMiP':
            continue
    
        for platform in meta[mission].keys():
            for flight_id, flight in meta[mission][platform].items():
                
                if flight_id in ['HALO-AC3_HALO_RF00', 'HALO-AC3_HALO_RF01']:  # test flight and transfer flight
                    continue
                
                print(flight_id)
                
                # read sea ice
                path = '/data/obs/campaigns/'+flight['mission'].lower()+'/auxiliary/sea_ice/daily_grid/'
                ds_sic = read_sea_ice(date=flight['date'].strftime('%Y%m%d'),
                                      path=path)
                
                try:
                    ds_gps = cat[mission][platform]['GPS_INS'][flight_id](
                        storage_options=kwds, **credentials).read()
                    
                except TypeError:
                    ds_gps = cat[mission][platform]['GPS_INS'][flight_id](
                        storage_options=kwds).read()
                
                # drop some variables
                keep_vars = ['lon', 'lat', 'alt']
                ds_gps = ds_gps[keep_vars]
                
                # remove duplicates in gps data
                ds_gps = ds_gps.sel(time=~ds_gps.indexes['time'].duplicated())
                
                # drop where lon or lat is nan
                ds_gps = ds_gps.sel(time=(~np.isnan(ds_gps.lat)) | (~np.isnan(ds_gps.lon)))
                
                if interpolated:
                    
                    # make homogeneous 1 s timestamp
                    ds_gps_int = xr.Dataset()
                    ds_gps_int.coords['time'] = pd.date_range(flight['takeoff'], flight['landing'], freq='1S')
                    ds_gps_int = xr.merge([ds_gps_int, ds_gps], join='left')
                    
                    # replace nans in taking the values from the nearest time step
                    # caution, this is only done to fill gaps and evaluate the surface type later
                    ds_gps_int = ds_gps_int.interpolate_na(dim='time', method='nearest')
                
                    ds_gps = ds_gps_int
                    
                # KDtree from sea ice data
                coords = np.column_stack((ds_sic['lat'].values.ravel(), ds_sic['lon'].values.ravel()))
                ground_pixel_tree = spatial.cKDTree(transform_coordinates(coords), leafsize=20)
                
                # get sea ice along Polar5 track
                lon_corr = ds_gps.lon.values
                lon_corr[lon_corr < 0] += 360
                distance, indices1d = ground_pixel_tree.query(transform_coordinates(np.array([ds_gps.lat.values, lon_corr]).T))
                indices2d = np.unravel_index(indices1d, ds_sic['lat'].values.shape)
                
                # get sea ice with the new index
                ds_gps['sic'] = (('time'), ds_sic.sic.values.flatten()[indices1d])
                ds_gps['sic'].attrs = dict(standard_name='sea_ice_concentration', long_name='sea ice concentration', units='percent', source='asi-AMSR2 n6250 v5.4 by University of Bremen')                
                
                # save as nc file
                intpl = ''
                if interpolated:
                    intpl = '_interpolated'
                outfile = './data/'+flight['mission'].lower()+'/'+flight['mission']+'_'+flight['platform']+'_sea_ice_along_track_'+flight['date'].strftime('%Y%m%d')+'_'+flight['name']+'.nc'
                ds_gps.to_netcdf(outfile)
