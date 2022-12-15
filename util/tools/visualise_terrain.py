#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------#
# Run this script to visualise the PALM domains on Stamen Terrain using folium
# based on namelist
# How to run:
# python visualise_terrain.py [case_name]
# @author: Dongqi Lin
#--------------------------------------------------------------------------------#
import numpy as np 
import os
import six
import sys
## make file directories recognisable
sys.path.append('.')
from util.loc_dom import convert_wgs_to_utm,domain_location, domain_nest
import folium
import pandas as pd
import configparser
import ast
import webbrowser
import pathlib
import warnings
## supress warnings
## switch to other actions if needed
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')

    
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
# read namelist
settings_cfg = configparser.ConfigParser(inline_comment_prefixes='#')
config = configparser.RawConfigParser()
prefix = sys.argv[1]
namelist =  f"./JOBS/{prefix}/INPUT/namelist.static-{prefix}"
config.read(namelist)
case_name =  ast.literal_eval(config.get("case", "case_name"))[0]
origin_time = ast.literal_eval(config.get("case", "origin_time"))[0]
# use WGS84 (EPSG:4326) for centlat/centlon
default_proj = ast.literal_eval(config.get("case", "default_proj"))[0]
# local projection (unit: m)
config_proj = ast.literal_eval(config.get("case", "config_proj"))[0]
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
## check if UTM projection is given
if len(config_proj)==0:
    config_proj_code = convert_wgs_to_utm(centlon, centlat)
    config_proj = f"EPSG:{config_proj_code}"
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
m = folium.Map(location=[centlat, centlon], zoom_start=10, tiles="Stamen Terrain")



# start generating static drivers
for i in range(0,ndomain):
    if i == 0:
        case_name_d01 = case_name+f"_N{i:02d}"
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
        
        # configure domain location information
        dom_cfg_d01 = domain_location(default_proj, config_proj, dom_cfg_d01)
        locations = [[dom_cfg_d01['lat_s'],dom_cfg_d01['lon_w']], # lower left 
                     [dom_cfg_d01['lat_n'],dom_cfg_d01['lon_w']], # upper left 
                     [dom_cfg_d01['lat_n'],dom_cfg_d01['lon_e']], # upper right
                     [dom_cfg_d01['lat_s'],dom_cfg_d01['lon_e']], # lower right
                     [dom_cfg_d01['lat_s'],dom_cfg_d01['lon_w']] # connect back
                    ]
        box = folium.PolyLine(locations, weight=3, color='red', opacity=0.8).add_to(m) 
        
    else:
        #--------------------------------------------------------------------------------#
        # generating static drivers for nested domains
        #--------------------------------------------------------------------------------#
        case_name_nest = case_name+f"_N{i:02d}"
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
        # calculate nested domain location
        dom_cfg_nest = domain_nest(config_proj, dom_cfg_d01['west'], dom_cfg_d01['south'], ll_x_nest, ll_y_nest,dom_cfg_nest)                  
        nest_locations = [[dom_cfg_nest['lat_s'],dom_cfg_nest['lon_w']], # lower left 
                     [dom_cfg_nest['lat_n'],dom_cfg_nest['lon_w']], # upper left 
                     [dom_cfg_nest['lat_n'],dom_cfg_nest['lon_e']], # upper right
                     [dom_cfg_nest['lat_s'],dom_cfg_nest['lon_e']], # lower right
                     [dom_cfg_nest['lat_s'],dom_cfg_nest['lon_w']] # connect back
                         ] 
        nest_box = folium.PolyLine(nest_locations, weight=3, color='red', opacity=0.8).add_to(m) 

    

        

## save as html file
html_file = 'visual_folium.html'
m.save(html_file)
## open in webbrowser
new = 2
html_path =str(pathlib.Path(html_file).resolve())
webbrowser.open('file:///'+html_path,new=new)
