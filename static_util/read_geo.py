#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------#
# Function to read GIS tiff file
# 
#
# @author: ricardofaria
# Modified by Dongqi Lin (dongqi.lin@pg.canterbury.ac.nz)
#--------------------------------------------------------------------------------#



import numpy as np



def readgeotiff(file):
    
    from osgeo import gdal
    
    ds = gdal.Open(str(file), gdal.GA_ReadOnly)

    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    
    width = ds.RasterXSize
    height = ds.RasterYSize
    

    pos = ds.GetGeoTransform()
    x_min = pos[0]
    y_min = pos[3] + width*pos[4] + height*pos[5] 
    x_max = pos[0] + width*pos[1] + height*pos[2]
    y_max = pos[3] 
    
    lat = np.linspace(y_min, y_max, height)
    lon = np.linspace(x_min, x_max, width)
    
    data_array = ds.ReadAsArray().astype(np.float)
    data_array[data_array==nodata] = -9999
    return(data_array[::-1, :], lat, lon)
    

    
