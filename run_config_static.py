#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------#
# This is the main run script
# How to run:
# python run_config_static.py case_name
# 
# User input files should be in JOBS/case_name/INPUT/
#
# @author: Dongqi Lin, Jiawei Zhang 
#--------------------------------------------------------------------------------#

import requests
import math
import getpass, pprint, time, os, cgi, json
import geopandas as gpd
import osmnx as ox
import rasterio
from geocube.api.core import make_geocube
from datetime import datetime, timezone
import pandas as pd
from pyproj import Proj, Transformer, CRS
from shapely.geometry import box, Point
from pathlib import Path
import os
import pickle
import numpy as np
import pandas as pd
import osmnx as ox
import rasterio
from geocube.api.core import make_geocube
from util.get_osm import *
from util.get_sst import download_sst
from util.get_geo_nasa import *
from util.loc_dom import convert_wgs_to_utm,domain_location, domain_nest
from util.create_static import *
from util.pre_process_tif import *
import configparser
import ast
import sys
from glob import glob
from urllib.request import urlretrieve
import warnings
## supress warnings
## switch to other actions if needed
warnings.filterwarnings("ignore")

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
# read namelist
settings_cfg = configparser.ConfigParser(inline_comment_prefixes='#')
config = configparser.RawConfigParser()
prefix = sys.argv[1]
namelist =  f"./JOBS/{prefix}/INPUT/namelist.static-{prefix}"
config.read(namelist)
## [case]
case_name =  ast.literal_eval(config.get("case", "case_name"))[0]
origin_time = ast.literal_eval(config.get("case", "origin_time"))[0]
# local projection (unit: m)
config_proj = ast.literal_eval(config.get("case", "config_proj"))[0]
# use WGS84 (EPSG:4326) for centlat/centlon
default_proj = ast.literal_eval(config.get("case", "default_proj"))[0] 

## [domain configuration]
ndomain = ast.literal_eval(config.get("domain", "ndomain"))[0]
centlat = ast.literal_eval(config.get("domain", "centlat"))[0]
centlon = ast.literal_eval(config.get("domain", "centlon"))[0]
dx = ast.literal_eval(config.get("domain", "dx"))
dy = ast.literal_eval(config.get("domain", "dy"))
dz = ast.literal_eval(config.get("domain", "dz"))
nx = ast.literal_eval(config.get("domain", "nx"))
ny = ast.literal_eval(config.get("domain", "ny"))
nz = ast.literal_eval(config.get("domain", "nz"))
z_origin = ast.literal_eval(config.get("domain", "z_origin"))
ll_x = ast.literal_eval(config.get("domain", "ll_x"))
ll_y = ast.literal_eval(config.get("domain", "ll_y"))

## [required tif files]
sst = ast.literal_eval(config.get("geotif", "sst"))
dem = ast.literal_eval(config.get("geotif", "dem"))
lu = ast.literal_eval(config.get("geotif", "lu"))

dem_start_date = ast.literal_eval(config.get("geotif", "dem_start_date"))[0]
dem_end_date = ast.literal_eval(config.get("geotif", "dem_end_date"))[0]

lu_start_date = ast.literal_eval(config.get("geotif", "lu_start_date"))[0]
lu_end_date = ast.literal_eval(config.get("geotif", "lu_end_date"))[0]


## [tif files for urban canopy]
bldh = ast.literal_eval(config.get("urban", "bldh"))
bldid = ast.literal_eval(config.get("urban", "bldid"))
pavement = ast.literal_eval(config.get("urban", "pavement"))
street = ast.literal_eval(config.get("urban", "street"))

## [tif files for plant canopy]
sfch = ast.literal_eval(config.get("plant", "sfch"))



# specify the directory of tif files
# users can provide their own tif files
# otherwise will download from NASA or OSM
static_tif_path = f'./JOBS/{case_name}/INPUT/'
output_path = static_tif_path.replace("INPUT","OUTPUT")
tmp_path = static_tif_path.replace("INPUT","TMP")
## create folders for temporary tif files and final netcdf outputs
if not os.path.exists(tmp_path):
    print("Create tmp folder")
    os.makedirs(tmp_path)
if not os.path.exists(output_path):
    print("Create output folder")
    os.makedirs(output_path)
    
## check if UTM projection is given
if len(config_proj)==0:
    print("UTM projection not given, identifying...")
    config_proj_code = convert_wgs_to_utm(centlon, centlat)
    config_proj = f"EPSG:{config_proj_code}"
    print(config_proj)
## these dictionanries only pass keys 
tif_geotif_dict = dict(config.items('geotif'))
tif_urban_dict = dict(config.items('urban'))
tif_plant_dict = dict(config.items('plant'))

for i in range(0,ndomain):
    if i == 0:
        case_name_d01 = case_name+"_N01"
        dom_cfg_d01 = {'origin_time': origin_time,
                    'centlat': centlat,  
                    'centlon': centlon,
                    'dx': dx[i],
                    'dy': dy[i],
                    'dz': dz[i],
                    'nx': nx[i],
                    'ny': ny[i],
                    'nz': nz[i],
                    'z_origin': z_origin[i],
                    }
        
        tif_dict_d01 = {}
        for keys in tif_geotif_dict.keys():
            tif_dict_d01[keys] = ast.literal_eval(config.get("geotif", keys))[i]
        for keys in tif_urban_dict.keys():
            tif_dict_d01[keys] = ast.literal_eval(config.get("urban", keys))[i]
        for keys in tif_plant_dict.keys():
            tif_dict_d01[keys] = ast.literal_eval(config.get("plant", keys))[i]
        # configure domain location information
        dom_cfg_d01 = domain_location(default_proj, config_proj,  dom_cfg_d01)
        # generate static driver 
#         dom_cfg_d01 = generate_palm_static(case_name_d01, config_proj, tif_proj, dom_cfg_d01, tif_dict_d01)
        
        #--------------------------------------------------------------------------------#
        ## first check if we need to download data from online sources
        ## SST data
        if tif_dict_d01["sst"]=="online":
            download_sst(case_name, origin_time, static_tif_path)
        ## DEM and land use (currently only for NASA online data sets)
        ## prepare dictionaries
        geodata_name_dict = {}
        output_format_dict  = {}
        start_date_dict = {}
        end_date_dict = {}
        if tif_dict_d01["dem"]=="online":
            geodata_name_dict["DEM"] = ["SRTMGL1_NC.003",]
            output_format_dict["DEM"] = "geotiff"
            # NASA DEM data only available for these dates
            start_date_dict["DEM"] = dem_start_date
            end_date_dict["DEM"] = dem_end_date
        if tif_dict_d01["lu"]=="online":
            # https://lpdaac.usgs.gov/documents/101/MCD12_User_Guide_V6.pdf
            geodata_name_dict["Land_Use"] = ["MCD12Q1.006",]
            output_format_dict["Land_Use"] = "geotiff"
            # User to choose the start/end date
            start_date_dict["Land_Use"] = lu_start_date
            end_date_dict["Land_Use"] = lu_end_date
        ## download data for NASA AρρEEARS API only
        if tif_dict_d01["dem"]=="online" or tif_dict_d01["lu"]=="online":
            area_radius = np.max([dx[i]*nx[i], dy[i]*ny[0]])/2 # units=metre
            default_buffer_ratio = 1.2 # used to multiply area_radius avoid areas becoming smaller than required after reproject
            api = 'https://appeears.earthdatacloud.nasa.gov/api/'  # Set the AρρEEARS API to a variable
            task_type = 'area'   # this is the only type used in this script
            # check if the files are already there
            if len(glob(static_tif_path+case_name+"_DEM_*"))>0 or len(glob(static_tif_path+case_name+"_Land_Use_*"))>0:
                # asking if need to download data
                if input("Data directories exist, do you wish to continue download? [y/N]") == "y":
                    download_nasa_main(api, geodata_name_dict, centlon, centlat, area_radius, default_proj, task_type,\
                           default_buffer_ratio, start_date_dict,end_date_dict, output_format_dict,case_name,static_tif_path)
            else:
                download_nasa_main(api, geodata_name_dict, centlon, centlat, area_radius, default_proj, task_type,\
                           default_buffer_ratio, start_date_dict,end_date_dict, output_format_dict,case_name,static_tif_path)
        if tif_dict_d01["bldh"]=="online" or tif_dict_d01["bldid"]=="online":
            get_osm_building(centlat, centlon, area_radius, static_tif_path, case_name, i)
        if tif_dict_d01["pavement"]=="online":
            get_osm_street(centlat, centlon, area_radius, static_tif_path, case_name, i)
    ## for child domains 
    else:    
        #--------------------------------------------------------------------------------#
        # downloading data for nested domains
        #--------------------------------------------------------------------------------#
        dom_cfg_nest = {'origin_time': origin_time,
                    'dx': dx[i],
                    'dy': dy[i],
                    'dz': dz[i],
                    'nx': nx[i],
                    'ny': ny[i],
                    'nz': nz[i],
                    'z_origin': z_origin[i],
                    }
        ll_x_nest, ll_y_nest = ll_x[i], ll_y[i]

        dom_cfg_nest = domain_nest(config_proj, dom_cfg_d01['west'], dom_cfg_d01['south'], ll_x_nest, ll_y_nest,dom_cfg_nest)

        tif_dict_nest = {}
        for keys in tif_urban_dict.keys():
            tif_dict_nest[keys] = ast.literal_eval(config.get("urban", keys))[i]
        for keys in tif_plant_dict.keys():
            tif_dict_nest[keys] = ast.literal_eval(config.get("plant", keys))[i]
        area_radius_nest = np.max([dx[i]*nx[i], dy[i]*ny[i]])/2 # units=metre

        if tif_dict_nest["bldh"]=="online" or tif_dict_nest["bldid"]=="online":
                get_osm_building(dom_cfg_nest["centlat"], dom_cfg_nest["centlon"], area_radius_nest, static_tif_path, case_name, i)
        if tif_dict_nest["pavement"]=="online":
                get_osm_street(dom_cfg_nest["centlat"], dom_cfg_nest["centlon"], area_radius_nest, static_tif_path, case_name, i)
        # save domain configuration for later - creating static drivers
        with open(f'{tmp_path}{case_name}_cfg_N0{i+1}.pickle', 'wb') as dicts:
            pickle.dump(dom_cfg_nest, dicts, protocol=pickle.HIGHEST_PROTOCOL)
            
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
# preprocess tif files 
process_all(prefix)

#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
# create static driver 
for i in range(0,ndomain):
    if i==0:
        static_driver_file = output_path + case_name + f'_static_N{i+1:02d}'
        if not os.path.exists(static_driver_file):
            dom_cfg_d01 = generate_palm_static(case_name,tmp_path, i, config_proj, dom_cfg_d01)
    else:
        static_driver_file = output_path + case_name + f'_static_N{i+1:02d}'
        if not os.path.exists(static_driver_file):
            with open(f'{tmp_path}{case_name}_cfg_N0{i+1}.pickle', 'rb') as dicts:
                 dom_cfg_nest = pickle.load(dicts)
            dom_cfg_nest = generate_palm_static(case_name,tmp_path, i, config_proj, dom_cfg_nest)
