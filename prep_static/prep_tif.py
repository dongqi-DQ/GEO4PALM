#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------#
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
# --------------------------------------------------------------------------------#

import rioxarray as rxr
from rasterio.crs import CRS
from rasterio.enums import Resampling
import sys
from tqdm import tqdm
import ast
import configparser
import geopandas as gpd
from geocube.api.core import make_geocube

# read namelist
settings_cfg = configparser.ConfigParser(inline_comment_prefixes="#")
config = configparser.RawConfigParser()
config.read(sys.argv[1])

tif_path = ast.literal_eval(config.get("input", "tif_path"))[0]
dem_infile = ast.literal_eval(config.get("input", "dem"))[0]
other_input_keylist = [
    x for x in list(dict(config.items("input")).keys()) if x not in ["dem", "tif_path"]
]
crs_output = CRS.from_epsg(int(ast.literal_eval(config.get("output", "crs"))[0]))
prefix = ast.literal_eval(config.get("output", "prefix"))[0]

ds_geo = rxr.open_rasterio(tif_path + dem_infile)

resolution_list = [int(x) for x in ast.literal_eval(config.get("output", "out_res"))]

resample_method_list = [
    Resampling[rs] for rs in ast.literal_eval(config.get("output", "resampling"))
]


# identify whether reprojection is needed
if ds_geo.rio.crs == crs_output:
    ds_geo_tmp = ds_geo
else:
    ds_geo_tmp = ds_geo.rio.reproject(crs_output)

# resample to desired resolution
for res, rs_method in tqdm(
    zip(resolution_list, resample_method_list), position=0, leave=True
):
    tif_outfile_name = str(res) + "m_" + str(crs_output).replace(":", "_") + ".tif"
    ds_geo_out = ds_geo_tmp.rio.reproject(crs_output, res, resampling=rs_method)
    ds_geo_out.rio.to_raster(tif_path + prefix + "_" + tif_outfile_name)
    
    # reample and reproject for other information like lu,bld, etc.
    for other_input in other_input_keylist:
        tmp_infile = ast.literal_eval(config.get("input", other_input))[0]
        if tmp_infile.endswith(".tif"):
            tmp_dataset = rxr.open_rasterio(tif_path + tmp_infile)
            tmp_dataset_reproj = tmp_dataset.rio.reproject_match(ds_geo_out)
            tmp_dataset_reproj.rio.to_raster(
                tif_path
                + prefix
                + "_"
                + other_input
                + "_"
                + str(res)
                + "m_"
                + str(crs_output).replace(":", "_")
                + ".tif"
            )
        elif tmp_infile.endswith(".shp"):

            tmp_dataset = gpd.read_file(tif_path + tmp_infile)
            var_name = ast.literal_eval(config.get("input", other_input))[1]
            geo_grid = make_geocube(
                vector_data=tmp_dataset,
                measurements=[var_name],
                resolution=(res/4, res/4),
            ) #divide res by 4 to make the reproject more accurate, can be removed if it's too slow.
            geo_grid = geo_grid[var_name]
            geo_grid = geo_grid.reindex(y=geo_grid.y[::-1])
            tmp_dataset_reproj = geo_grid.rio.reproject_match(ds_geo_out)
            tmp_dataset_reproj.rio.to_raster(
                tif_path
                + prefix
                + "_"
                + other_input
                + "_"
                + str(res)
                + "m_"
                + str(crs_output).replace(":", "_")
                + ".tif"
            )
