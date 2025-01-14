#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------#
# Script to convert shp files to tif files
#
# How to use:
# python shp2tif.py [case_name] [shp file folder] [variable_name] 
# 
# Note:
# - shp file folder needs to be inside the case_name/INPUT directory
#   make sure to put all relavent files inside tiff folder, inlcuding shx,xml etc.
#
# - variable_name is the field in shp file that the user desires to convert to tif file
#
# @author: Dongqi Lin, Jiawei Zhang
# --------------------------------------------------------------------------------#



import rioxarray as rxr
from rasterio.crs import CRS
from rasterio.enums import Resampling
import sys
sys.path.append('.')
import ast
import configparser
import geopandas as gpd
from geocube.api.core import make_geocube
from shapely.geometry import Polygon
from glob import glob
from util.loc_dom import convert_wgs_to_utm
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
# read namelist
settings_cfg = configparser.ConfigParser(inline_comment_prefixes='#')
config = configparser.RawConfigParser()
prefix = sys.argv[1]
namelist =  f"./JOBS/{prefix}/INPUT/config.static-{prefix}"
config.read(namelist)
## [case]
case_name =  ast.literal_eval(config.get("case", "case_name"))[0]
origin_time = ast.literal_eval(config.get("case", "origin_time"))[0]
# local projection (unit: m)
config_proj = ast.literal_eval(config.get("case", "config_proj"))[0]
# use WGS84 (EPSG:4326) for centlat/centlon
default_proj = ast.literal_eval(config.get("case", "default_proj"))[0] 
## [domain]
centlat = ast.literal_eval(config.get("domain", "centlat"))[0]
centlon = ast.literal_eval(config.get("domain", "centlon"))[0]

dx = ast.literal_eval(config.get("domain", "dx"))

## check if UTM projection is given
if len(config_proj)==0:
    config_proj_code = convert_wgs_to_utm(centlon, centlat)
    config_proj = f"EPSG:{config_proj_code}"
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
# read shp file location
shp_path = sys.argv[2]

shp_file = glob(f"./JOBS/{prefix}/INPUT/"+shp_path + "/*.shp")[0]
# variable name to convert to tif from shp file
var_name = sys.argv[3]

res = int(min(dx))
# output tif file name

crs_output = CRS.from_string(config_proj)
tif_out_name = f"{var_name}_{res}m_"+config_proj.replace(":","")+"_shp2tif.tif"
# read shp file
tmp_dataset = gpd.read_file(shp_file)
tmp_dataset_reproj =  tmp_dataset.to_crs(config_proj)
geo_grid = make_geocube(
    vector_data=tmp_dataset_reproj,
    measurements=[var_name],
    resolution=(res, res),
) 
#geo_grid = geo_grid[var_name]
geo_grid = geo_grid.reindex(y=geo_grid.y[::-1])
tmp_dataset_output = geo_grid.rio.reproject(crs_output)
tmp_dataset_output.rio.to_raster(f"./JOBS/{prefix}/INPUT/{tif_out_name}")
print(f"tif file {tif_out_name} saved")
