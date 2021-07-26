#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------#
# Run this script to visualise the PALM domains on OpenStreet Map
# This scripts will look for cfg files based on case names given in the namelist
# @author: Dongqi Lin
#--------------------------------------------------------------------------------#
import numpy as np 
import matplotlib as mpl        
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import six
from PIL import Image
import pandas as pd
import configparser
import ast


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

def read_cfg(cfg_file):
    cfg = pd.read_csv(cfg_file)
    print("Reading "+cfg_file)
    lat_1 = cfg.lat_n.values[0]
    lat_0 = cfg.lat_s.values[0]
    lon_1 = cfg.lon_e.values[0]
    lon_0 = cfg.lon_w.values[0]
    return ([lon_0, lon_1, lat_0, lat_1])

def def_extent(lon_0, lon_1, lat_0, lat_1, buffer):
    left = lon_0 - buffer
    right = lon_1 + buffer
    north = lat_1 + buffer -0.02
    south = lat_0 - buffer +0.02
    return ([left, right, north, south])

def plot_rectangle(ax, lon_0, lon_1, lat_0, lat_1):
    ax.add_patch(mpl.patches.Rectangle((lon_0,lat_0),lon_1-lon_0, lat_1-lat_0,fill=None, lw =3, edgecolor='red', zorder=10, transform=ccrs.PlateCarree()))

# read namelist
settings_cfg = configparser.ConfigParser(inline_comment_prefixes='#')
config = configparser.RawConfigParser()
config.read('namelist.static')
cfg_path = "./cfg_input/"


case_names =  ast.literal_eval(config.get("case", "case_name"))


plt.figure(figsize=(9,9))
cimgt.GoogleWTS.get_image = get_map_image
request = cimgt.OSM()
ax = plt.axes(projection=request.crs)

for idx, names in enumerate(case_names):
    if idx == 0:
        lat_lon_d01 = read_cfg(cfg_path+names+".cfg")
        extent = def_extent(lat_lon_d01[0], lat_lon_d01[1], lat_lon_d01[2], lat_lon_d01[3], 0.05)
        plot_rectangle(ax, lat_lon_d01[0], lat_lon_d01[1], lat_lon_d01[2], lat_lon_d01[3])
    else:
        lat_lon_nest = read_cfg(cfg_path+names+".cfg")
        plot_rectangle(ax, lat_lon_nest[0], lat_lon_nest[1], lat_lon_nest[2], lat_lon_nest[3])



ax.set_extent(extent)

ax.add_image(request, 12)
plt.show()