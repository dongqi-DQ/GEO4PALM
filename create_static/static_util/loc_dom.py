#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------#
# Locate domain centre 
# write configuration file
# calculate coordinates for domain nesting
#
# @author: Dongqi Lin, Jiawei Zhang 
#--------------------------------------------------------------------------------#


import pandas as pd
from pyproj import Proj,transform


def domain_location(config_projection,tif_projection, dom_dict):
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
    if config_projection == tif_projection:
        # basically means the centlan and centlon is already in the local (m) coordinate
        # don't do convertion in this time
        tif_centx =  centlon
        tif_centy =  centlat
    else:
    # change lat/lon to UTM to calculate the coordinates of the domain

        inProj = Proj(init=config_projection)
        outProj = Proj(init=tif_projection)

        tif_centx,tif_centy = transform(inProj,outProj,centlon,centlat)
    
    tif_west, tif_east = tif_centx-nx*dx/2, tif_centx+nx*dx/2
    tif_north, tif_south = tif_centy+ny*dy/2, tif_centy-ny*dy/2
    
    # transform back to latitude/logintude to save in cfg for future reference
    tifProj = Proj(init=tif_projection)
    wgsProj = Proj(init='EPSG:4326')
    lon_w, lat_s = transform(tifProj,wgsProj,tif_west,tif_south)
    lon_e, lat_n = transform(tifProj,wgsProj,tif_east,tif_north)
    
    dom_dict['west'] = tif_west
    dom_dict['east'] = tif_east
    dom_dict['south'] = tif_south
    dom_dict['north'] = tif_north
    dom_dict['lon_w'] = lon_w
    dom_dict['lon_e'] = lon_e
    dom_dict['lat_s'] = lat_s
    dom_dict['lat_n'] = lat_n
    return dom_dict

def write_cfg(case_name, dom_dict):
    cfg = pd.DataFrame() 
    for names, values in dom_dict.items():
        cfg[names] = [values]
    cfg.to_csv('./cfg_files/'+ case_name + '.cfg', index=None)
    print('cfg file is ready: '+case_name)
    
def domain_nest(tif_projection, west, south, llx, lly, dom_dict):
    '''

    Parameters
    ----------
    tif_projection: desired projection with unit in m
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
    nest_east = west + llx + dx*nx
    nest_south = south + lly
    nest_north = south + lly + dy*ny

    nest_cent_lon = nest_west + dx*nx/2
    nest_cent_lat = nest_south + dy*ny/2
    # transform back to latitude/logintude to save in cfg for future reference
    tifProj = Proj(init=tif_projection)
    wgsProj = Proj(init='EPSG:4326')
    cent_lon, cent_lat = transform(tifProj,wgsProj,nest_cent_lon,nest_cent_lat)
    lon_w, lat_s = transform(tifProj,wgsProj,nest_west,nest_south)
    lon_e, lat_n = transform(tifProj,wgsProj,nest_east,nest_north)
    
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



