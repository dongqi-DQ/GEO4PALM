#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------#
# Run this script to configure and create static drivers for PALM
# Edit namelist.static for PALM domain configuration
# cfg files are saved for future references
# 
# @author: Dongqi Lin, Jiawei Zhang 
#--------------------------------------------------------------------------------#

import numpy as np
import pandas as pd
from pyproj import Proj
from static_util.loc_dom import domain_location, domain_nest, write_cfg
from create_static import *
import configparser
import ast
from pyproj import Proj, transform

import warnings
## supress warnings
## switch to other actions if needed
warnings.filterwarnings("ignore")
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#

# use WGS84 (EPSG:4326) for centlat/centlon 
config_proj = 'EPSG:4326'
# local projection (unit: m)
tif_proj = 'EPSG:2193'

# read namelist
settings_cfg = configparser.ConfigParser(inline_comment_prefixes='#')
config = configparser.RawConfigParser()
config.read('namelist.static')
case_names =  ast.literal_eval(config.get("case", "case_name"))
origin_time = ast.literal_eval(config.get("case", "origin_time"))[0]

ndomain = ast.literal_eval(config.get("domain", "ndomain"))[0]
centlat = ast.literal_eval(config.get("domain", "centlat"))
centlon = ast.literal_eval(config.get("domain", "centlon"))
dx = ast.literal_eval(config.get("domain", "dx"))
dy = ast.literal_eval(config.get("domain", "dy"))
dz = ast.literal_eval(config.get("domain", "dz"))
nx = ast.literal_eval(config.get("domain", "nx"))
ny = ast.literal_eval(config.get("domain", "ny"))
nz = ast.literal_eval(config.get("domain", "nz"))
z_origin = ast.literal_eval(config.get("domain", "z_origin"))
ll_x = ast.literal_eval(config.get("domain", "ll_x"))
ll_y = ast.literal_eval(config.get("domain", "ll_y"))

# specify the directory of tif files
static_tif_path = './raw_static/'

tif_input_dict = dict(config.items('tif'))
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#

# start generating static drivers
for i in range(0,ndomain):
    if i == 0:
        case_name_d01 = case_names[i]
        dom_cfg_d01 = {'origin_time': origin_time,
                    'centlat': centlat[i],  
                    'centlon': centlon[i],
                    'dx': dx[i],
                    'dy': dy[i],
                    'dz': dz[i],
                    'nx': nx[i],
                    'ny': ny[i],
                    'nz': nz[i],
                    'z_origin': z_origin[i],
                    }
        
        tif_dict_d01 = {}
        # if tif file not given use "empty.tif"
        for keys in tif_input_dict.keys():
            tif_dict_d01[keys] = ast.literal_eval(config.get("tif", keys))[i]
            if len(tif_dict_d01[keys]) ==0:
                tif_dict_d01[keys] = static_tif_path + "empty.tif"
            else:
                tif_dict_d01[keys] = static_tif_path + tif_dict_d01[keys]
        # configure domain location information
        dom_cfg_d01 = domain_location(config_proj, tif_proj, dom_cfg_d01)
        # generate static driver 
        dom_cfg_d01 = generate_palm_static(case_name_d01, config_proj, tif_proj, dom_cfg_d01, tif_dict_d01)
        # write cfg files for future reference
        write_cfg(case_name_d01, dom_cfg_d01)
        
    else:
        #--------------------------------------------------------------------------------#
        # generating static drivers for nested domains
        #--------------------------------------------------------------------------------#
        case_name_nest = case_names[i]
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
        tif_dict_nest = {}
        # if not tif file not specified/provided, use "empty.tif"
        for keys in tif_input_dict.keys():
            try:
                tif_dict_nest[keys] = ast.literal_eval(config.get("tif", keys))[i]
            except:
                tif_dict_nest[keys] = ''
            if len(tif_dict_nest[keys]) ==0:
                tif_dict_nest[keys] = static_tif_path + 'empty.tif'
            else:
                tif_dict_nest[keys] = static_tif_path + tif_dict_nest[keys]
        # calculate nested domain location
        dom_cfg_nest = domain_nest(tif_proj, dom_cfg_d01['west'], dom_cfg_d01['south'], ll_x_nest, ll_y_nest,dom_cfg_nest)                  
        
        # generate static driver for nested domain
        dom_cfg_nest = generate_palm_static(case_name_nest, tif_proj, tif_proj, dom_cfg_nest, tif_dict_nest)
        # write cfg for future reference
        write_cfg(case_name_nest, dom_cfg_nest)


