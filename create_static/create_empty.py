#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------#
# create empty tif files based on input tif
# How to use:
# python create_empty.py [input tif file]
# example:
# python create_empty.py ./raw_static/chch_dem_10m_EPSG_2193.tif
# @ author: Dongqi Lin, Jiawei Zhang
#--------------------------------------------------------------------------------#
import rioxarray as rxr
from rasterio.crs import CRS
import rasterio
import numpy as np
import sys

infile = sys.argv[1]
with rxr.open_rasterio(infile) as ds_geo:
    empty_tmp_out =  ds_geo.where( ds_geo.isnull(), np.nan)
    empty_tmp_out.rio.to_raster("./raw_static/empty.tif")
print("empty.tif created")
