"""
Script calculates the sea ice concentration along flight track for every flight
during all airborne campaigns

Adapted from Nils Risse
"""
import numpy as np
import xarray as xr
from scipy import spatial


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
    
    #%% calculate sea ice on path for each flight
    inpath = "C:/Users/Johannes/Documents/tmp"
    outpath = "C:/Users/Johannes/Documents/tmp"
    date = "20220417"
    flight_nr = "RF17"
    # read sea ice
    ds_sic = xr.open_dataset(f"{inpath}/asi-AMSR2-n6250-{date}-v5.4.nc")
    ds_gps = xr.open_dataset(f"{inpath}/HALO-AC3_HALO_gps_ins_{date}_{flight_nr}.nc")

    # drop some variables
    keep_vars = ['lon', 'lat', 'alt']
    ds_gps = ds_gps[keep_vars]

    # KDtree from sea ice data
    coords = np.column_stack((ds_sic['lat'].values.ravel(), ds_sic['lon'].values.ravel()))
    ground_pixel_tree = spatial.cKDTree(transform_coordinates(coords), leafsize=20)

    # get sea ice along track
    distance, indices1d = ground_pixel_tree.query(transform_coordinates(np.array([ds_gps.lat.values, lon_corr]).T))
    indices2d = np.unravel_index(indices1d, ds_sic['lat'].values.shape)

    # get sea ice with the new index
    ds_gps['sic'] = (('time'), ds_sic.seaice.values.flatten()[indices1d])
    ds_gps['sic'].attrs = dict(standard_name='sea_ice_concentration', long_name='sea ice concentration',
                               units='percent', source='asi-AMSR2 n6250 v5.4 by University of Bremen')

    # save as nc file
    outfile = f"{outpath}/HALO-AC3_HALO_sic_{date}_{flight_nr}.nc"
    ds_gps.to_netcdf(outfile)
