#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------#
# This script contains functions to run create_cfg.py
#
#
# @author: Dongqi Lin (dongqi.lin@pg.canterbury.ac.nz)
#--------------------------------------------------------------------------------#


import pandas as pd
from pyproj import Proj
import utm

def domain_location(myProj, dom_dict):
    centlon = dom_dict['centlon']
    centlat = dom_dict['centlat']
    nx = dom_dict['nx']
    ny = dom_dict['ny']
    dx = dom_dict['dx']
    dy = dom_dict['dy']
    # change lat/lon to UTM to calculate the coordinates of the domain
    utm_centx, utm_centy = myProj(centlon,centlat)
    
    utm_left, utm_right = utm_centx-nx*dx/2, utm_centx+nx*dx/2
    utm_north, utm_south = utm_centy+ny*dy/2, utm_centy-ny*dy/2
    
    # change UTM to lat/lon to locate the domain in the WRF output
    west, north = myProj(utm_left,utm_north,inverse=True)
    east, south = myProj(utm_right,utm_south,inverse=True)
    dom_dict['west'] = west
    dom_dict['east'] = east
    dom_dict['south'] = south
    dom_dict['north'] = north
    return dom_dict

def write_cfg(case_name, dom_dict):
    cfg = pd.DataFrame() 
    for names, values in dom_dict.items():
        cfg[names] = [values]
    cfg.to_csv('cfg_input/'+ case_name + '.cfg', index=None)
    print('cfg file is ready: '+case_name)
    
def domain_nest(west, south, llx, lly, dx, dy, nx, ny):
    '''

    Parameters
    ----------
    west : longitude of west (left) boundary
    south : latitude of south boundary
    llx : distance between nest domains at lower left corner (x-axis)
    lly : distance between nest domains at lower left corner (y-axis)
    dx : PALM nested domain horizontal grid spacing along x-axis
    dy : PALM nested domain horizontal grid spacing along y-axis
    nx : PALM nested domain horizontal number of grids along x-axis
    ny : PALM nested domain horizontal number of grids along y-axi

    Returns
    -------
    latitudes and longitudes at PALM nested domain centre

    '''
    zone = utm.from_latlon(south, west)[2]
    myProj = Proj("+proj=utm +zone=" + str(zone) + ", +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    utm_west, utm_south = myProj(west, south)
    utm_cent_ew_d02, utm_cent_sn_d02 = utm_west+llx+(dx*nx/2), utm_south+lly+(dy*ny/2)
    cent_lon_d02, cent_lat_d02 = myProj(utm_cent_ew_d02, utm_cent_sn_d02, inverse=True)
    return(cent_lat_d02, cent_lon_d02)