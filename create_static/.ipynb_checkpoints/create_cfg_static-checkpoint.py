#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------#
# This script is the cfg file generator for WRF-PALM coupler
# This file requires users provide:
#     latitude and longitude of domain centre
#     domain size and resolution (dx, dy, dz, nx, ny, nz)
#
# To do list:
#     add vertical streching feature
#
# @author: Dongqi Lin (dongqi.lin@pg.canterbury.ac.nz)
#--------------------------------------------------------------------------------#

import numpy as np
import pandas as pd
from pyproj import Proj
import utm
from static_util.loc_dom import domain_location, domain_nest, write_cfg
from create_static import *
import configparser
import ast
from pyproj import Proj, transform

config_proj = 'EPSG:4326'#this is only used to indicate the projection of which the configuration file's centlan lon is
tif_proj = 'epsg:2193'

settings_cfg = configparser.ConfigParser(inline_comment_prefixes='#')
config = configparser.RawConfigParser()
config.read('namelist.static')
case_names =  ast.literal_eval(config.get("case", "case_name"))

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



static_dir = 'raw_static/'

tif_input_dict = dict(config.items('tif'))

tif_names = {'dem' : 'dem_',
            'bldh': 'bldh_',
            'bldid': 'bldid_',
            'lu': 'landcover_',
            'tree': 'tree_',
            'road': 'road_',
           'water': 'CHCH_WATER_',
            'street': 'street_'
            }


for i in range(0,ndomain):
    if i == 0:
        case_name_d01 = case_names[i]
        dom_cfg_d01 = {'centlat': centlat[i],  
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

        for keys in tif_input_dict.keys():
            tif_dict_d01[keys] = ast.literal_eval(config.get("tif", keys))[i]
            if len(tif_dict_d01[keys]) ==0:
                tif_dict_d01[keys] = static_dir + tif_names[keys] + f'{int(dx[i])}m_lc.tif'
            else:
                tif_dict_d01[keys] = static_dir + tif_dict_d01[keys]
                
#         zone = utm.from_latlon(dom_cfg_d01['centlat'],dom_cfg_d01['centlon'])[2]
#         myProj = Proj("+proj=utm +zone=" + str(zone) + " +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
        dom_cfg_d01 = domain_location(config_proj, tif_proj, dom_cfg_d01)
        dom_cfg_d01 = generate_palm_static(case_name_d01, config_proj, tif_proj, dom_cfg_d01, tif_dict_d01)
        write_cfg(case_name_d01, dom_cfg_d01)
        
    else:
        case_name_nest = case_names[i]
        dom_cfg_nest = {
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

        for keys in tif_input_dict.keys():
            try:
                tif_dict_nest[keys] = ast.literal_eval(config.get("tif", keys))[i]
            except:
                tif_dict_nest[keys] = ''
            if len(tif_dict_nest[keys]) ==0:
                tif_dict_nest[keys] = static_dir + tif_names[keys] + f'{int(dx[i])}m_lc.tif'
            else:
                tif_dict_nest[keys] = static_dir + tif_dict_nest[keys]
        dom_cfg_nest['centlat'], dom_cfg_nest['centlon'] = domain_nest(dom_cfg_d01['west'], dom_cfg_d01['south'], 
                                                                        ll_x_nest, ll_y_nest, 
                                                              dom_cfg_nest['dx'], dom_cfg_nest['dy'], 
                                                              dom_cfg_nest['nx'], dom_cfg_nest['ny'])
        
        # for inner domains, use tif_proj as the original to directly use local cord to 
        # calculate centlat centlon etc.
        dom_cfg_nest = domain_location(tif_proj, tif_proj, dom_cfg_nest)
        dom_cfg_nest = generate_palm_static(case_name_nest, tif_proj, tif_proj, dom_cfg_nest, tif_dict_nest)
        write_cfg(case_name_nest, dom_cfg_nest)
            







