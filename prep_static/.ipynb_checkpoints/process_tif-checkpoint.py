#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------#
# Script to reproject and resample tif files
#
# How to use:
# python process_tif.py [infile] [out EPSG projection] [outfile prefix] [resolution list]  
# Example:
#
# python process_tif.py chch_dem_1m.tif 2193 chch_dem 10,20
#
# @originial script by: Jiawei Zhang
# @modified by: Dongqi Lin
#--------------------------------------------------------------------------------#

import rioxarray as rxr
from rasterio.crs import CRS
import rasterio
import sys
from tqdm import tqdm

tif_inpath = "./tiff/"
tif_infile = sys.argv[1] 
tif_outpath = "./processed/"
crs_output =  CRS.from_epsg(int(sys.argv[2])) 
prefix = sys.argv[3]

ds_org = rxr.open_rasterio(tif_inpath + tif_infile)
with rasterio.open(tif_inpath+tif_infile) as src:
    tif_crs = src.crs

resolution_list = [int(x) for x in sys.argv[4].split(',')]

# identify whether reprojection is needed
if tif_crs==crs_output:
    ds_org_tmp = ds_org
else:
    ds_org_tmp = ds_org.rio.reproject(crs_output)

# resample to desired resolution 
for res in tqdm(resolution_list, position=0, leave=True):
    tif_outfile_name = str(res) + "m_"+str(crs_output).replace(':','_')+".tif"
    ds_org_out = ds_org_tmp.rio.reproject(crs_output, res)
    ds_org_out.rio.to_raster(tif_outpath + prefix+"_" + tif_outfile_name)
