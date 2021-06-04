#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:50:23 2021

@author: dli84
"""
import numpy as np 
import matplotlib as mpl        
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import six
from PIL import Image
import pandas as pd

def new_get_image(self, tile):
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
    print("Reading CFG file")
    lat_1 = cfg.north.values[0]
    lat_0 = cfg.south.values[0]
    lon_1 = cfg.east.values[0]
    lon_0 = cfg.west.values[0]
    return ([lon_0, lon_1, lat_0, lat_1])

def def_extent(lon_0, lon_1, lat_0, lat_1, buffer):
    left = lon_0 - buffer
    right = lon_1 + buffer
    north = lat_1 + buffer -0.02
    south = lat_0 - buffer +0.02
    return ([left, right, north, south])

def plot_rectangle(ax, lon_0, lon_1, lat_0, lat_1):
    ax.add_patch(mpl.patches.Rectangle((lon_0,lat_0),lon_1-lon_0, lat_1-lat_0,fill=None, lw =3, edgecolor='red', zorder=10, transform=ccrs.PlateCarree()))


cfg_d01 = '/home/dli84/Documents/PALM/create_static/cfg_input/mapm_test_80m.cfg'
cfg_d02 = '/home/dli84/Documents/PALM/create_static/cfg_input/mapm_test_40m.cfg'
cfg_d03 = '/home/dli84/Documents/PALM/create_static/cfg_input/mapm_test_20m.cfg'
cfg_d04 = '/home/dli84/Documents/PALM/create_static/cfg_input/mapm_test_10m.cfg'


lat_lon_d01 = read_cfg(cfg_d01)
lat_lon_d02 = read_cfg(cfg_d02)
lat_lon_d03 = read_cfg(cfg_d03)
lat_lon_d04 = read_cfg(cfg_d04)

extent = def_extent(lat_lon_d01[0], lat_lon_d01[1], lat_lon_d01[2], lat_lon_d01[3], 0.05)

cimgt.GoogleWTS.get_image = new_get_image
request = cimgt.OSM()
# extent = [170, 174, -41, -45]
plt.figure(figsize=(9,9))

ax = plt.axes(projection=request.crs)
plot_rectangle(ax, lat_lon_d01[0], lat_lon_d01[1], lat_lon_d01[2], lat_lon_d01[3])
plot_rectangle(ax, lat_lon_d02[0], lat_lon_d02[1], lat_lon_d02[2], lat_lon_d02[3])
plot_rectangle(ax, lat_lon_d03[0], lat_lon_d03[1], lat_lon_d03[2], lat_lon_d03[3])
plot_rectangle(ax, lat_lon_d04[0], lat_lon_d04[1], lat_lon_d04[2], lat_lon_d04[3])

# ax.add_patch(mpl.patches.Rectangle((171,-42),1,1,fill=None, lw =3, edgecolor='red', zorder=10, transform=ccrs.PlateCarree()))
# ax.plot(-88.5,41.5,'x', transform=ccrs.PlateCarree())
ax.set_extent(extent)

ax.add_image(request, 12)#, interpolation='spline36')
plt.show()