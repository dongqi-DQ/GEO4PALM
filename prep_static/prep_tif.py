#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------#
# Script to reproject and resample tif files
#
# How to use:
# python prep_tif.py [infile] [out EPSG projection] [outfile prefix] [resolution list] [resample_class] 
# Example:
#
# python prep_tif.py chch_dem_1m.tif 2193 chch_dem 10,20 nearest
# 
# Default resample calss is nearest
# for more details of resample class options please refer to rasterio documentation 
# https://rasterio.readthedocs.io/en/stable/api/rasterio.enums.html#rasterio.enums.Resampling
#
# @originial script by: Jiawei Zhang
# @modified by: Dongqi Lin
#--------------------------------------------------------------------------------#

import rioxarray as rxr
from rasterio.crs import CRS
import rasterio
from rasterio.enums import Resampling
import sys
from tqdm import tqdm

resample_class = {"nearest": 0,
                  "bilinear": 1,
                  "cubic": 2,
                  "cubic_spline": 3,
                  "lanczos": 4,
                  "average": 5,
                  "mode": 6,
                  "gauss": 7,
                  "max": 8,
                  "min": 9,
                  "med":10,
                  "q1": 11,
                  "q2": 12,
                  "sum": 13,
                  "rms": 14,
}

tif_path = "./tiff/"
tif_infile = sys.argv[1] 
crs_output =  CRS.from_epsg(int(sys.argv[2])) 
prefix = sys.argv[3]

ds_geo = rxr.open_rasterio(tif_path + tif_infile)

with rasterio.open(tif_path+tif_infile) as src:
    tif_crs = src.crs

resolution_list = [int(x) for x in sys.argv[4].split(',')]

try:
    resample_num = resample_class[sys.argv[5]]
    print(resample_num)
except:
    print("no resample class give, use nearest")
    resample_num = 0

# identify whether reprojection is needed
if tif_crs==crs_output:
    ds_geo_tmp = ds_geo
else:
    ds_geo_tmp = ds_geo.rio.reproject(crs_output)

# resample to desired resolution 
for res in tqdm(resolution_list, position=0, leave=True):
    tif_outfile_name = str(res) + "m_"+str(crs_output).replace(':','_')+".tif"
    ds_geo_out = ds_geo_tmp.rio.reproject(crs_output, res, resampling=resample_num)
    ds_geo_out.rio.to_raster(tif_path + prefix+"_" + tif_outfile_name)
