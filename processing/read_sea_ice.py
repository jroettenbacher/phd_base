

from osgeo import gdal
import xarray as xr


def read(date, path='/media/nrisse/data/university/master/thesis/data/sea_ice/'):
    """
    Read hdf files of sea ice concentration from University of Bremen
    
    Input
    -------
    date:  date (yyyymmdd) for which sea ice should be retrieved
    
    Returns
    -------
    None.

    """
    
    lon = gdal.Open('HDF4_SDS:UNKNOWN:"'+path+'LongitudeLatitudeGrid-n6250-Arctic.hdf":0').ReadAsArray()
    lat = gdal.Open('HDF4_SDS:UNKNOWN:"'+path+'LongitudeLatitudeGrid-n6250-Arctic.hdf":1').ReadAsArray()
    sic = gdal.Open(path+'asi-AMSR2-n6250-'+date+'-v5.4.hdf').ReadAsArray()
    
    # combineas xarray dataset
    ds = xr.Dataset()
    ds.coords['lon'] = (('x', 'y'), lon)
    ds.coords['lat'] = (('x', 'y'), lat)
    ds['sic'] = (('x', 'y'), sic)
    
    # close dataset
    del sic
    
    return ds
