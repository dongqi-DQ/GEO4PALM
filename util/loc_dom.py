#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------#
# Fucntions to:
# - identify UTM zone
# - find domain location
# - calculate coordinates for domain nesting
# 
# @author: Dongqi Lin, Jiawei Zhang 
#--------------------------------------------------------------------------------#
import warnings
## supress warnings
## switch to other actions if needed
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=FutureWarning)

import pandas as pd
from pyproj import Proj,Transformer
import math



def convert_wgs_to_utm(lon, lat):
    '''
    Function to identify UTM projection code
    https://stackoverflow.com/a/40140326/4556479
    '''
    utm_band = str((math.floor((lon + 180) / 6 ) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0'+utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
    else:
        epsg_code = '327' + utm_band
    return epsg_code

def domain_location(default_projection,config_projection, dom_dict):
    '''
    Identify west, east, south, and north boundaries of doamin based on lat/lon at
    domain centre 
    '''
    centlon = dom_dict['centlon']
    centlat = dom_dict['centlat']
    nx = dom_dict['nx']
    ny = dom_dict['ny']
    dx = dom_dict['dx']
    dy = dom_dict['dy']
    if default_projection == config_projection:
        # basically means the centlan and centlon is already in the local (m) coordinate
        # don't do convertion in this time
        tif_centx =  centlon
        tif_centy =  centlat
    else:
    # change lat/lon to UTM to calculate the coordinates of the domain

        inProj = Proj(init=default_projection)
        outProj = Proj(init=config_projection)
        t = Transformer.from_proj(inProj,outProj, always_xy=True)
        tif_centx,tif_centy = t.transform(centlon,centlat)
    
    tif_west, tif_east = tif_centx- (nx-1)*dx/2, tif_centx+(nx-1)*dx/2
    tif_north, tif_south = tif_centy+(ny-1)*dy/2, tif_centy-(ny-1)*dy/2
    
    # transform back to latitude/logintude to save in cfg for future reference
    config_proj = Proj(config_projection)
    wgs_proj = Proj('EPSG:4326')
    t = Transformer.from_proj(config_proj, wgs_proj, always_xy=True)
    lon_w, lat_s = t.transform(tif_west,tif_south)
    lon_e, lat_n = t.transform(tif_east,tif_north)
    
    dom_dict['west'] = tif_west
    dom_dict['east'] = tif_east
    dom_dict['south'] = tif_south
    dom_dict['north'] = tif_north
    dom_dict['lon_w'] = lon_w
    dom_dict['lon_e'] = lon_e
    dom_dict['lat_s'] = lat_s
    dom_dict['lat_n'] = lat_n
    return dom_dict

    
def domain_nest(config_projection, west, south, llx, lly, dom_dict):
    '''

    Parameters
    ----------
    config_projection: desired projection with unit in m
    west : longitude of west (left) boundary
    south : latitude of south boundary
    llx : distance between nest domains at lower left corner (x-axis)
    lly : distance between nest domains at lower left corner (y-axis)
    nest domain dictionary: dx, dy, nx, ny
    
    Returns
    -------
    domain dictionary containing all domain geo-info:
    latitudes and longitudes at PALM nested domain centre 
    boundaries in desired projection with unit in m
    '''
    nx = dom_dict['nx']
    ny = dom_dict['ny']
    dx = dom_dict['dx']
    dy = dom_dict['dy']
    
    nest_west = west + llx
    nest_east = west + llx + dx*(nx-1)
    nest_south = south + lly
    nest_north = south + lly + dy*(ny-1)

    nest_cent_lon = nest_west + dx*(nx-1)/2.0
    nest_cent_lat = nest_south + dy*(ny-1)/2.0
    # transform back to latitude/logintude to save in cfg for future reference
    config_proj = Proj(config_projection)
    wgs_proj = Proj('EPSG:4326')
    t = Transformer.from_proj(config_proj, wgs_proj, always_xy=True)
    cent_lon, cent_lat = t.transform(nest_cent_lon,nest_cent_lat)
    lon_w, lat_s = t.transform(nest_west,nest_south)
    lon_e, lat_n = t.transform(nest_east,nest_north)
    
    dom_dict['centlat'] = cent_lat
    dom_dict['centlon'] = cent_lon
    dom_dict['west'] = nest_west
    dom_dict['east'] = nest_east
    dom_dict['south'] = nest_south
    dom_dict['north'] = nest_north
    dom_dict['lon_w'] = lon_w
    dom_dict['lon_e'] = lon_e
    dom_dict['lat_s'] = lat_s
    dom_dict['lat_n'] = lat_n
    return(dom_dict)



