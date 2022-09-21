#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------#
# Run this script to visualise the PALM domains on OpenStreet Map
# based on namelist
# How to run:
# python visualise_domains.py [case_name]
# @Author: Dongqi Lin
#--------------------------------------------------------------------------------#
import numpy as np 
import matplotlib as mpl        
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import os
import six
import sys
## make file directories recognisable
sys.path.append('.')
from util.loc_dom import convert_wgs_to_utm,domain_location, domain_nest

from PIL import Image
import pandas as pd
import configparser
import ast
import warnings
## supress warnings
## switch to other actions if needed
warnings.filterwarnings("ignore")

def get_map_image(self, tile):
    if six.PY3:
        from urllib.request import urlopen, Request
    else:
        from urllib2 import urlopen
    url = self._image_url(tile)  # added by H.C. Winsemius
    req = Request(url) # added by H.C. Winsemius
    req.add_header('User-agent', 'your bot 0.1')
    # fh = urlopen(url)  # removed by H.C. Winsemius
    fh = urlopen(req)
    im_data = six.BytesIO(fh.read())
    fh.close()
    img = Image.open(im_data)

    img = img.convert(self.desired_tile_form)

    return img, self.tileextent(tile), 'lower'

def def_extent(lon_0, lon_1, lat_0, lat_1, buffer):
    left = lon_0 - buffer
    right = lon_1 + buffer
    north = lat_1 + buffer -0.02
    south = lat_0 - buffer +0.02
    return ([left, right, north, south])

def plot_rectangle(ax, lon_0, lon_1, lat_0, lat_1):
    ax.add_patch(mpl.patches.Rectangle((lon_0,lat_0),lon_1-lon_0, lat_1-lat_0,fill=None, lw =3, edgecolor='red', zorder=10, transform=ccrs.PlateCarree()))

    
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
plt.figure(figsize=(9,9))
cimgt.GoogleWTS.get_image = get_map_image
request = cimgt.OSM()
ax = plt.axes(projection=request.crs)

# start generating static drivers
for i in range(0,ndomain):
    if i == 0:
        case_name_d01 = case_name+f"_N{i:02d}"
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
        
        # configure domain location information
        dom_cfg_d01 = domain_location(default_proj, config_proj, dom_cfg_d01)
        extent = def_extent(dom_cfg_d01['lon_w'], dom_cfg_d01['lon_e'], dom_cfg_d01['lat_s'], dom_cfg_d01['lat_n'], 0.05)
        plot_rectangle(ax, dom_cfg_d01['lon_w'], dom_cfg_d01['lon_e'], dom_cfg_d01['lat_s'], dom_cfg_d01['lat_n'])
        
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
        plot_rectangle(ax, dom_cfg_nest['lon_w'], dom_cfg_nest['lon_e'], dom_cfg_nest['lat_s'], dom_cfg_nest['lat_n'])


    

        


ax.set_extent(extent)

ax.add_image(request, 12)
plt.show()
