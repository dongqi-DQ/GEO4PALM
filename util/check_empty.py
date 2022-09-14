#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------#
# check and create empty tif files 
# @ author: Dongqi Lin, Jiawei Zhang
#--------------------------------------------------------------------------------#
import rioxarray as rxr
from rasterio.crs import CRS
import rasterio
import numpy as np
from pyproj import Proj,transform

def check_empty(infile, centlat, centlon, dem_tif, config_proj):
    
    with rxr.open_rasterio(infile) as ds_geo:
        west = ds_geo.x.min()
        east = ds_geo.x.max()
        south = ds_geo.y.min()
        north = ds_geo.y.max()
    
    if ds_geo.rio.crs== CRS.from_string(config_proj):
        if west<centlat<east and south<centlon<north:
            print("empty.tif is valid")
        else:
            create_empty(infile, dem_tif)
    else:
        inProj = Proj(init=config_proj)
        outProj = Proj(init=str(ds_geo.rio.crs))

        tif_centx,tif_centy = transform(inProj,outProj,centlon,centlat)
        if west<tif_centx[0]<east and south<tif_centy[0]<north:
            print("empty.tif is valid")
        else:
            create_empty(infile, dem_tif)
            
    return(infile)

def create_empty(empty_tif, dem_tif):
    with rxr.open_rasterio(dem_tif) as ds_geo:
        empty_tmp_out =  ds_geo.where( ds_geo.isnull(), np.nan)
        empty_tmp_out.rio.to_raster(empty_tif)
    print("new empty.tif created")
