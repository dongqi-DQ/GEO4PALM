#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------#
# Function to read GIS tiff file
# 
# @author: Jiawei Zhang
#--------------------------------------------------------------------------------#


import numpy as np
import rioxarray as rxr


def readgeotiff(file):
    fill_value = -9999
    ds=rxr.open_rasterio(file,masked=True)
    lat = ds.y.values
    lon = ds.x.values
    ds=ds.where(~(ds.isnull() | (ds<=fill_value)),fill_value).squeeze(drop=True)
    array_final=np.flip(ds.values,axis=0)
    lat = np.flip(lat)
    return(array_final, lat, lon)


    
