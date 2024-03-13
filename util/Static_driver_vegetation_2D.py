#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------#
# Read geotiff files and convert to array for static drives 
# 
# @author:  Eva Bendix Nielsen
#--------------------------------------------------------------------------------#

import xarray as xr
import rioxarray
import numpy as np
import pandas as pd
import rasterio
import os
import matplotlib.pyplot as plt
import numpy.ma as ma
import gc

from util.canopy_generator import generate_single_tree_lad, process_patch 
from util.pre_process_tif import *
from util.loc_dom import *
from util.read_geo import *
from util.nearest import nearest


def extract_tiff(array, lat,lon, dom_dict,west_loc,south_loc,north_loc,proj_str,case_name, idomain, save_tif=False):
    # west_loc is the local coordinate (m) location of westest of the domain
    # south_loc and north_loc is the local coordinate (m) location of southest/northest
    # proj_str is the projection string
    
    nx = dom_dict['nx']
    ny = dom_dict['ny']
     # find the nearest index
    xmin, west_idx  = nearest(lon,west_loc)
    south_idx       = nearest(lat,south_loc)[1]
    ymax            = nearest(lat,north_loc)[0]
    ## debugging messages
    # print("south",south_loc,"west", west_loc)
    # print("lon_min",lon[0],"lon_max",lon[-1])
    # print("lat_min",lat[0],"lat_max",lat[-1])
    array_palm = array[south_idx:south_idx+ny,west_idx:west_idx+nx]
    xsize = lon[-1]-lon[-2]
    ysize = lat[-1]-lat[-2]
    
    return array_palm

def generate_palm_static_LAD(case_name, tmp_path, idomain, config_proj, dom_dict):
    output_path = tmp_path.replace("TMP", "OUTPUT")
    static_driver_file = output_path + case_name + f'_static_N{idomain+1:02d}'
    # read domain info
    origin_time = dom_dict['origin_time']
    centlon = dom_dict['centlon']
    centlat = dom_dict['centlat']
    nx = dom_dict['nx']
    ny = dom_dict['ny']
    nz = dom_dict['nz']
    dx = dom_dict['dx']
    dy = dom_dict['dy']
    dz = dom_dict['dz']
    z_origin = dom_dict['z_origin']
    y = np.arange(dy/2,dy*(ny+0.5),dy)
    x = np.arange(dx/2,dx*(nx+0.5),dx)
    z = np.arange(dz/2, dz*nz, dz)
    tif_left = dom_dict['west']
    tif_right = dom_dict['east']
    tif_north = dom_dict['north']
    tif_south = dom_dict['south']
    ### leaf area index parameters
    tree_lai_max = dom_dict['tree_lai_max']      # default value 5.0
    lad_max_height = dom_dict['lad_max_height']  # default value 0.4
    
    
    # assign tiff file names
    dem_tif = f"{tmp_path}{case_name}_DEM_N{idomain+1:02d}.tif"
    print(dem_tif)
    
    lai_tif = dem_tif.replace("DEM","LAI")
    patch_type_tif = dem_tif.replace("DEM","patch_type")
    vegetation_height_tif = dem_tif.replace("DEM","vegetation_height")
    single_tree_height_tif = dem_tif.replace("DEM","tree_height")
    tree_crown_diameter_tif =  dem_tif.replace("DEM","tree_crown_diameter")
    tree_trunk_diameter_tif =  dem_tif.replace("DEM","tree_trunk_diameter")
    tree_crown_shape_tif =  dem_tif.replace("DEM","crown_shape")
    #tree_type_tif =  dem_tif.replace("DEM","tree_height")
    

    # Read topography file
    dem, lat, lon = readgeotiff(dem_tif)
    dem[dem < 0] = 0
    ## debugging message
    # print("@@@@dem",dem.shape)


    # extract topography to match PALM domain
    zt = extract_tiff(dem, lat, lon, dom_dict, tif_left,tif_south,tif_north, config_proj,case_name, idomain,save_tif=True)

    zt = zt - z_origin
    zt[zt < .5] = 0
    ## debugging message
    # print("@@@@zt",zt.shape)
    del dem, lat, lon

    print('Number of grid points x, y = ' + str(zt.shape[1]) + ', ' + str(zt.shape[0]))
    
    
    
    #Vegetation properties 
    print('Reading LAI')
    if not os.path.exists(lai_tif):
        LAI = np.zeros_like(zt)
        LAI[:] = np.nan
    else:
        LAI_geo, lat, lon = readgeotiff(lai_tif)
        LAI= extract_tiff(LAI_geo, lat, lon, dom_dict, tif_left,tif_south,tif_north,config_proj,case_name, idomain)
        del LAI_geo, lat, lon
        gc.collect()
    
    
    
    print('Reading patch_type')
    if not os.path.exists(patch_type_tif):
        patch_type = np.zeros_like(zt)
        patch_type[:] = np.nan
    else:
        patch_type_geo, lat, lon = readgeotiff(patch_type_tif)
        patch_type = extract_tiff(patch_type_geo, lat, lon, dom_dict, tif_left,tif_south,tif_north,config_proj,case_name, idomain)
        del patch_type_geo, lat, lon
        gc.collect()

        
    print('Reading vegetation_height')
    if not os.path.exists(vegetation_height_tif):
        vegetation_height = np.zeros_like(zt)
        vegetation_height[:] = np.nan
    else:
        vegetation_height_geo, lat, lon = readgeotiff(vegetation_height_tif)
        vegetation_height = extract_tiff(vegetation_height_geo, lat, lon, dom_dict, tif_left,tif_south,tif_north,config_proj,case_name, idomain)
        #vegetation_height[vegetation_height == NaN] = 0 
        del vegetation_height_geo, lat, lon
        gc.collect()
        
        
    print('Reading single tree_height') # maximum height of tree 
    if not os.path.exists(single_tree_height_tif):
        single_tree_height = np.zeros_like(zt)
        single_tree_height[:] = np.nan
    else:
        single_tree_height_geo, lat, lon = readgeotiff(single_tree_height_tif)
        single_tree_height = extract_tiff(single_tree_height_geo, lat, lon, dom_dict, tif_left,tif_south,tif_north,config_proj,case_name, idomain)
        del single_tree_height_geo, lat, lon
        gc.collect()
        
        

    print('Reading tree_crown_diameter')
    if not os.path.exists(tree_crown_diameter_tif):
        tree_crown_diameter = np.zeros_like(zt)
        tree_crown_diameter[:] = np.nan
    else:
        tree_crown_diameter_geo, lat, lon = readgeotiff(tree_crown_diameter_tif)
        tree_crown_diameter = extract_tiff(tree_crown_diameter_geo, lat, lon, dom_dict, tif_left,tif_south,tif_north,config_proj,case_name, idomain)
        del tree_crown_diameter_geo, lat, lon
        gc.collect()
        

        
    print('Reading tree_trunk_diameter')
    if not os.path.exists(tree_trunk_diameter_tif):
        tree_trunk_diameter = np.zeros_like(zt)
        tree_trunk_diameter[:] = np.nan
    else:
        tree_trunk_diameter_geo, lat, lon = readgeotiff(tree_trunk_diameter_tif)
        tree_trunk_diameter = extract_tiff(tree_trunk_diameter_geo, lat, lon, dom_dict, tif_left,tif_south,tif_north,config_proj,case_name, idomain)
        del tree_trunk_diameter_geo, lat, lon
        gc.collect()   
        
    
    print('Reading tree_crown_shape')
    if not os.path.exists(tree_crown_shape_tif):
        tree_crown_shape = np.zeros_like(zt)
        tree_crown_shape[:] = np.nan
    else:
        tree_crown_shape_geo, lat, lon = readgeotiff(tree_crown_shape_tif)
        tree_crown_shape = extract_tiff(tree_crown_shape_geo, lat, lon, dom_dict, tif_left,tif_south,tif_north,config_proj,case_name, idomain)
        del tree_crown_shape_geo, lat, lon
        gc.collect()   
        
 
    
    
    
    
    
    # Create an array with the same shape as the original but filled with 'default'
    tree_type_array = np.full(tree_trunk_diameter.shape, 0, dtype=object)

    tree_type_array[tree_trunk_diameter ==-9999]=-9999

    single_tree_height=ma.masked_where((single_tree_height==0)|(single_tree_height<0), single_tree_height, copy = False)
    tree_crown_diameter=ma.masked_where((tree_crown_diameter==0)|(tree_crown_diameter<0),tree_crown_diameter, copy = False)
    vegetation_height=ma.masked_where((vegetation_height==0)|(vegetation_height<0), vegetation_height, copy = False)
    tree_trunk_diameter=ma.masked_where((tree_trunk_diameter==0)|(tree_trunk_diameter<0), tree_trunk_diameter, copy = False)
    tree_type_array=ma.masked_where((tree_type_array==0)|(tree_type_array<0), tree_type_array, copy = False)
    LAI=ma.masked_where((LAI==0)|(LAI<1), LAI, copy = False)
    tree_crown_shape=ma.masked_where((tree_crown_shape<0), tree_crown_shape, copy = False)
    patch_type = ma.masked_where((patch_type<0), patch_type, copy = False)
    
   
    
    print('generate_single_tree')
    lad,bad,tree_ids,tree_type, zlad = generate_single_tree_lad(x,y,dz, single_tree_height, vegetation_height, tree_type_array, 
                                                            single_tree_height,tree_crown_diameter, tree_trunk_diameter, LAI, 'summer',0, ma.masked, tree_crown_shape)
    
    
    
    print('generate patch')
    lad_patch, patch_id,patch_types, patch_nz,status = process_patch(dz, vegetation_height, patch_type, patch_type, np.max(vegetation_height), LAI, 5.0, 3.0)
    

    
        
        
    return lad, bad, tree_ids, tree_type, lad_patch, patch_id,patch_types






def process_single_tree(i, j, dx, dz,
                        tree_type, tree_shape, tree_height, tree_lai, tree_dia, trunk_dia,
                        season, lai_tree_lower_threshold, remove_low_lai_tree,
                        low_lai_counter, mod_counter):

    # Set some parameters
    sphere_extinction = 0.6
    cone_extinction = 0.2
    ml_n_low = 0.5
    ml_n_high = 6.0

    # Populate look up table for tree species and their properties
    # #0 species name
    # #1 Tree shapes were manually lookep up.
    # #2 Crown h/w ratio - missing
    # #3 Crown diameter based on Berlin tree statistics
    # #4 Tree height based on Berlin tree statistics
    # #5 Tree LAI summer - missing
    # #6 Tree LAI winter - missing
    # #7 Height of lad maximum - missing
    # #8 Ratio LAD/BAD - missing
    # #9 Trunk diameter at breast height from Berlin
    default_trees = []
    default_trees.append(Tree("Default",         1.0, 1.0,  4.0, 12.0, 3.0, 0.8, 0.6, 0.025, 0.35))
    default_trees.append(Tree("Abies",           3.0, 1.0,  4.0, 12.0, 3.0, 0.8, 0.6, 0.025, 0.80))
    default_trees.append(Tree("Acer",            1.0, 1.0,  7.0, 12.0, 3.0, 0.8, 0.6, 0.025, 0.80))
    default_trees.append(Tree("Aesculus",        1.0, 1.0,  7.0, 12.0, 3.0, 0.8, 0.6, 0.025, 1.00))
    default_trees.append(Tree("Ailanthus",       1.0, 1.0,  8.5, 13.5, 3.0, 0.8, 0.6, 0.025, 1.30))
    default_trees.append(Tree("Alnus",           3.0, 1.0,  6.0, 16.0, 3.0, 0.8, 0.6, 0.025, 1.20))
    default_trees.append(Tree("Amelanchier",     1.0, 1.0,  3.0,  4.0, 3.0, 0.8, 0.6, 0.025, 1.20))
    default_trees.append(Tree("Betula",          1.0, 1.0,  6.0, 14.0, 3.0, 0.8, 0.6, 0.025, 0.30))
    default_trees.append(Tree("Buxus",           1.0, 1.0,  4.0,  4.0, 3.0, 0.8, 0.6, 0.025, 0.90))
    default_trees.append(Tree("Calocedrus",      3.0, 1.0,  5.0, 10.0, 3.0, 0.8, 0.6, 0.025, 0.50))
    default_trees.append(Tree("Caragana",        1.0, 1.0,  3.5,  6.0, 3.0, 0.8, 0.6, 0.025, 0.90))
    default_trees.append(Tree("Carpinus",        1.0, 1.0,  6.0, 10.0, 3.0, 0.8, 0.6, 0.025, 0.70))
    default_trees.append(Tree("Carya",           1.0, 1.0,  5.0, 17.0, 3.0, 0.8, 0.6, 0.025, 0.80))
    default_trees.append(Tree("Castanea",        1.0, 1.0,  4.5,  7.0, 3.0, 0.8, 0.6, 0.025, 0.80))
    default_trees.append(Tree("Catalpa",         1.0, 1.0,  5.5,  6.5, 3.0, 0.8, 0.6, 0.025, 0.70))
    default_trees.append(Tree("Cedrus",          1.0, 1.0,  8.0, 13.0, 3.0, 0.8, 0.6, 0.025, 0.80))
    default_trees.append(Tree("Celtis",          1.0, 1.0,  6.0,  9.0, 3.0, 0.8, 0.6, 0.025, 0.80))
    default_trees.append(Tree("Cercidiphyllum",  1.0, 1.0,  3.0,  6.5, 3.0, 0.8, 0.6, 0.025, 0.80))
    default_trees.append(Tree("Cercis",          1.0, 1.0,  2.5,  7.5, 3.0, 0.8, 0.6, 0.025, 0.90))
    default_trees.append(Tree("Chamaecyparis",   5.0, 1.0,  3.5,  9.0, 3.0, 0.8, 0.6, 0.025, 0.70))
    default_trees.append(Tree("Cladrastis",      1.0, 1.0,  5.0, 10.0, 3.0, 0.8, 0.6, 0.025, 0.80))
    default_trees.append(Tree("Cornus",          1.0, 1.0,  4.5,  6.5, 3.0, 0.8, 0.6, 0.025, 1.20))
    default_trees.append(Tree("Corylus",         1.0, 1.0,  5.0,  9.0, 3.0, 0.8, 0.6, 0.025, 0.40))
    default_trees.append(Tree("Cotinus",         1.0, 1.0,  4.0,  4.0, 3.0, 0.8, 0.6, 0.025, 0.70))
    default_trees.append(Tree("Crataegus",       3.0, 1.0,  3.5,  6.0, 3.0, 0.8, 0.6, 0.025, 1.40))
    default_trees.append(Tree("Cryptomeria",     3.0, 1.0,  5.0, 10.0, 3.0, 0.8, 0.6, 0.025, 0.50))
    default_trees.append(Tree("Cupressocyparis", 3.0, 1.0,  3.0,  8.0, 3.0, 0.8, 0.6, 0.025, 0.40))
    default_trees.append(Tree("Cupressus",       3.0, 1.0,  5.0,  7.0, 3.0, 0.8, 0.6, 0.025, 0.40))
    default_trees.append(Tree("Cydonia",         1.0, 1.0,  2.0,  3.0, 3.0, 0.8, 0.6, 0.025, 0.90))
    default_trees.append(Tree("Davidia",         1.0, 1.0, 10.0, 14.0, 3.0, 0.8, 0.6, 0.025, 0.40))
    default_trees.append(Tree("Elaeagnus",       1.0, 1.0,  6.5,  6.0, 3.0, 0.8, 0.6, 0.025, 1.20))
    default_trees.append(Tree("Euodia",          1.0, 1.0,  4.5,  6.0, 3.0, 0.8, 0.6, 0.025, 0.90))
    default_trees.append(Tree("Euonymus",        1.0, 1.0,  4.5,  6.0, 3.0, 0.8, 0.6, 0.025, 0.60))
    default_trees.append(Tree("Fagus",           1.0, 1.0, 10.0, 12.5, 3.0, 0.8, 0.6, 0.025, 0.50))
    default_trees.append(Tree("Fraxinus",        1.0, 1.0,  5.5, 10.5, 3.0, 0.8, 0.6, 0.025, 1.60))
    default_trees.append(Tree("Ginkgo",          3.0, 1.0,  4.0,  8.5, 3.0, 0.8, 0.6, 0.025, 0.80))
    default_trees.append(Tree("Gleditsia",       1.0, 1.0,  6.5, 10.5, 3.0, 0.8, 0.6, 0.025, 0.60))
    default_trees.append(Tree("Gymnocladus",     1.0, 1.0,  5.5, 10.0, 3.0, 0.8, 0.6, 0.025, 0.80))
    default_trees.append(Tree("Hippophae",       1.0, 1.0,  9.5,  8.5, 3.0, 0.8, 0.6, 0.025, 0.80))
    default_trees.append(Tree("Ilex",            1.0, 1.0,  4.0,  7.5, 3.0, 0.8, 0.6, 0.025, 0.80))
    default_trees.append(Tree("Juglans",         1.0, 1.0,  7.0,  9.0, 3.0, 0.8, 0.6, 0.025, 0.50))
    default_trees.append(Tree("Juniperus",       5.0, 1.0,  3.0,  7.0, 3.0, 0.8, 0.6, 0.025, 0.90))
    default_trees.append(Tree("Koelreuteria",    1.0, 1.0,  3.5,  5.5, 3.0, 0.8, 0.6, 0.025, 0.50))
    default_trees.append(Tree("Laburnum",        1.0, 1.0,  3.0,  6.0, 3.0, 0.8, 0.6, 0.025, 0.60))
    default_trees.append(Tree("Larix",           3.0, 1.0,  7.0, 16.5, 3.0, 0.8, 0.6, 0.025, 0.60))
    default_trees.append(Tree("Ligustrum",       1.0, 1.0,  3.0,  6.0, 3.0, 0.8, 0.6, 0.025, 1.10))
    default_trees.append(Tree("Liquidambar",     3.0, 1.0,  3.0,  7.0, 3.0, 0.8, 0.6, 0.025, 0.30))
    default_trees.append(Tree("Liriodendron",    3.0, 1.0,  4.5,  9.5, 3.0, 0.8, 0.6, 0.025, 0.50))
    default_trees.append(Tree("Lonicera",        1.0, 1.0,  7.0,  9.0, 3.0, 0.8, 0.6, 0.025, 0.70))
    default_trees.append(Tree("Magnolia",        1.0, 1.0,  3.0,  5.0, 3.0, 0.8, 0.6, 0.025, 0.60))
    default_trees.append(Tree("Malus",           1.0, 1.0,  4.5,  5.0, 3.0, 0.8, 0.6, 0.025, 0.30))
    default_trees.append(Tree("Metasequoia",     5.0, 1.0,  4.5, 12.0, 3.0, 0.8, 0.6, 0.025, 0.50))
    default_trees.append(Tree("Morus",           1.0, 1.0,  7.5, 11.5, 3.0, 0.8, 0.6, 0.025, 1.00))
    default_trees.append(Tree("Ostrya",          1.0, 1.0,  2.0,  6.0, 3.0, 0.8, 0.6, 0.025, 1.00))
    default_trees.append(Tree("Parrotia",        1.0, 1.0,  7.0,  7.0, 3.0, 0.8, 0.6, 0.025, 0.30))
    default_trees.append(Tree("Paulownia",       1.0, 1.0,  4.0,  8.0, 3.0, 0.8, 0.6, 0.025, 0.40))
    default_trees.append(Tree("Phellodendron",   1.0, 1.0, 13.5, 13.5, 3.0, 0.8, 0.6, 0.025, 0.50))
    default_trees.append(Tree("Picea",           3.0, 1.0,  3.0, 13.0, 3.0, 0.8, 0.6, 0.025, 0.90))
    default_trees.append(Tree("Pinus",           3.0, 1.0,  6.0, 16.0, 3.0, 0.8, 0.6, 0.025, 0.80))
    default_trees.append(Tree("Platanus",        1.0, 1.0, 10.0, 14.5, 3.0, 0.8, 0.6, 0.025, 1.10))
    default_trees.append(Tree("Populus",         1.0, 1.0,  9.0, 20.0, 3.0, 0.8, 0.6, 0.025, 1.40))
    default_trees.append(Tree("Prunus",          1.0, 1.0,  5.0,  7.0, 3.0, 0.8, 0.6, 0.025, 1.60))
    default_trees.append(Tree("Pseudotsuga",     3.0, 1.0,  6.0, 17.5, 3.0, 0.8, 0.6, 0.025, 0.70))
    default_trees.append(Tree("Ptelea",          1.0, 1.0,  5.0,  4.0, 3.0, 0.8, 0.6, 0.025, 1.10))
    default_trees.append(Tree("Pterocaria",      1.0, 1.0, 10.0, 12.0, 3.0, 0.8, 0.6, 0.025, 0.50))
    default_trees.append(Tree("Pterocarya",      1.0, 1.0, 11.5, 14.5, 3.0, 0.8, 0.6, 0.025, 1.60))
    default_trees.append(Tree("Pyrus",           3.0, 1.0,  3.0,  6.0, 3.0, 0.8, 0.6, 0.025, 1.80))
    default_trees.append(Tree("Quercus",         1.0, 1.0,  8.0, 14.0, 3.1, 0.1, 0.6, 0.025, 0.40))
    default_trees.append(Tree("Rhamnus",         1.0, 1.0,  4.5,  4.5, 3.0, 0.8, 0.6, 0.025, 1.30))
    default_trees.append(Tree("Rhus",            1.0, 1.0,  7.0,  5.5, 3.0, 0.8, 0.6, 0.025, 0.50))
    default_trees.append(Tree("Robinia",         1.0, 1.0,  4.5, 13.5, 3.0, 0.8, 0.6, 0.025, 0.50))
    default_trees.append(Tree("Salix",           1.0, 1.0,  7.0, 14.0, 3.0, 0.8, 0.6, 0.025, 1.10))
    default_trees.append(Tree("Sambucus",        1.0, 1.0,  8.0,  6.0, 3.0, 0.8, 0.6, 0.025, 1.40))
    default_trees.append(Tree("Sasa",            1.0, 1.0, 10.0, 25.0, 3.0, 0.8, 0.6, 0.025, 0.60))
    default_trees.append(Tree("Sequoiadendron",  5.0, 1.0,  5.5, 10.5, 3.0, 0.8, 0.6, 0.025, 1.60))
    default_trees.append(Tree("Sophora",         1.0, 1.0,  7.5, 10.0, 3.0, 0.8, 0.6, 0.025, 1.40))
    default_trees.append(Tree("Sorbus",          1.0, 1.0,  4.0,  7.0, 3.0, 0.8, 0.6, 0.025, 1.10))
    default_trees.append(Tree("Syringa",         1.0, 1.0,  4.5,  5.0, 3.0, 0.8, 0.6, 0.025, 0.60))
    default_trees.append(Tree("Tamarix",         1.0, 1.0,  6.0,  7.0, 3.0, 0.8, 0.6, 0.025, 0.50))
    default_trees.append(Tree("Taxodium",        5.0, 1.0,  6.0, 16.5, 3.0, 0.8, 0.6, 0.025, 0.60))
    default_trees.append(Tree("Taxus",           2.0, 1.0,  5.0,  7.5, 3.0, 0.8, 0.6, 0.025, 1.50))
    default_trees.append(Tree("Thuja",           3.0, 1.0,  3.5,  9.0, 3.0, 0.8, 0.6, 0.025, 0.70))
    default_trees.append(Tree("Tilia",           3.0, 1.0,  7.0, 12.5, 3.0, 0.8, 0.6, 0.025, 0.70))
    default_trees.append(Tree("Tsuga",           3.0, 1.0,  6.0, 10.5, 3.0, 0.8, 0.6, 0.025, 1.10))
    default_trees.append(Tree("Ulmus",           1.0, 1.0,  7.5, 14.0, 3.0, 0.8, 0.6, 0.025, 0.80))
    default_trees.append(Tree("Zelkova",         1.0, 1.0,  4.0,  5.5, 3.0, 0.8, 0.6, 0.025, 1.20))
    default_trees.append(Tree("Zenobia",         1.0, 1.0,  5.0,  5.0, 3.0, 0.8, 0.6, 0.025, 0.40))

    # Check for missing data in the input and set default values if needed
    if tree_type is ma.masked:
        tree_type = int(0)
    else:
        tree_type = int(tree_type)

    if tree_shape is ma.masked:
        tree_shape = default_trees[tree_type].shape

    if tree_height is ma.masked:
        tree_height = default_trees[tree_type].height

    if tree_lai is ma.masked:
        if season == "summer":
            tree_lai = default_trees[tree_type].lai_summer
        else:
            tree_lai = default_trees[tree_type].lai_winter

    if tree_dia is ma.masked:
        tree_dia = default_trees[tree_type].diameter

    if trunk_dia is ma.masked:
        trunk_dia = default_trees[tree_type].dbh

    # Check tree_lai
    # Tree LAI lower then threshold?
    if tree_lai < lai_tree_lower_threshold:
        # Deal with low lai tree
        mod_counter = mod_counter + 1
        if remove_low_lai_tree:
            # Skip this tree
            print("Removed tree with LAI = ", "%0.3f" % tree_lai, " at (", i, ", ", j, ").", sep="")
            return None, None, None, None, None, low_lai_counter, mod_counter, 1
        else:
            # Use type specific default
            if season == "summer":
                tree_lai = default_trees[tree_type].lai_summer
            else:
                tree_lai = default_trees[tree_type].lai_winter
            print("Adjusted tree to LAI = ", "%0.3f" % tree_lai, " at (", i, ", ", j, ").", sep="")

    # Warn about a tree with lower LAI than we would expect in winter
    if tree_lai < default_trees[tree_type].lai_winter:
        low_lai_counter = low_lai_counter + 1
        print("Found tree with LAI = ", "%0.3f" % tree_lai,
              " (tree type specific default winter LAI of ",
              "%0.2f" % default_trees[tree_type].lai_winter, ")",
              " at (", i, ", ", j, ").", sep="")

    # Assign values that are not defined as user input from lookup table
    tree_ratio = default_trees[tree_type].ratio
    lad_max_height = default_trees[tree_type].lad_max_height
    bad_scale = default_trees[tree_type].bad_scale

    print("Tree input parameters:")
    print("----------------------")
    print("type:           " + str(default_trees[tree_type].species) )
    print("height:         " + str(tree_height))
    print("lai:            " + str(tree_lai))
    print("crown diameter: " + str(tree_dia))
    print("trunk diameter: " + str(trunk_dia))
    print("shape: " + str(tree_shape))
    print("height/width: " + str(tree_ratio))

    # Calculate crown height and height of the crown center
    crown_height = tree_ratio * tree_dia
    if crown_height > tree_height:
        crown_height = tree_height

    crown_center = tree_height - crown_height * 0.5

    # Calculate height of maximum LAD
    z_lad_max = lad_max_height * tree_height

    # Calculate the maximum LAD after Lalic and Mihailovic (2004)
    lad_max_part_1 = integrate.quad(
        lambda z: ((tree_height - z_lad_max) / (tree_height - z))**ml_n_high * np.exp(
            ml_n_high * (1.0 - (tree_height - z_lad_max) / (tree_height - z))), 0.0, z_lad_max)
    lad_max_part_2 = integrate.quad(
        lambda z: ((tree_height - z_lad_max) / (tree_height - z))**ml_n_low * np.exp(
            ml_n_low * (1.0 - (tree_height - z_lad_max) / (tree_height - z))), z_lad_max,
        tree_height)

    lad_max = tree_lai / (lad_max_part_1[0] + lad_max_part_2[0])

    # Define position of tree and its output domain
    nx = int(tree_dia / dx) + 2
    nz = int(tree_height / dz) + 2

    # Add one grid point if diameter is an odd value
    if (tree_dia % 2.0) != 0.0:
        nx = nx + 1

    # Create local domain of the tree's LAD
    x = np.arange(0, nx * dx, dx)
    x[:] = x[:] - 0.5 * dx
    y = x

    z = np.arange(0, nz * dz, dz)
    z[1:] = z[1:] - 0.5 * dz

    # Define center of the tree position inside the local LAD domain
    tree_location_x = x[int(nx / 2)]
    tree_location_y = y[int(nx / 2)]

    # Calculate LAD profile after Lalic and Mihailovic (2004). Will be later used for normalization
    lad_profile = np.arange(0, nz, 1.0)
    lad_profile[:] = 0.0

    for k in range(1, nz - 1):
        if (z[k] > 0.0) & (z[k] < z_lad_max):
            n = ml_n_high
        else:
            n = ml_n_low

        lad_profile[k] = lad_max * ((tree_height - z_lad_max) / (tree_height - z[k]))**n * np.exp(
            n * (1.0 - (tree_height - z_lad_max) / (tree_height - z[k])))

    # Create lad array and populate according to the specific tree shape. This is still
    # experimental
    lad_loc = ma.ones((nz, nx, nx))
    lad_loc[:, :, :] = ma.masked
    bad_loc = ma.copy(lad_loc)

    # For very small trees, no LAD is calculated
    if tree_height <= (0.5 * dz):
        print("    Shallow tree found. Action: ignore.")
        return lad_loc, bad_loc, x, y, z, low_lai_counter, mod_counter, 1

    # Branch for spheres and ellipsoids. A symmetric LAD sphere is created assuming an LAD
    # extinction towards the center of the tree, representing the effect of sunlight extinction
    # and thus less leaf mass inside the tree crown. Extinction coefficients are experimental.
    if tree_shape == 1:
        for i in range(0, nx):
            for j in range(0, nx):
                for k in range(0, nz):
                    r_test = np.sqrt((x[i] - tree_location_x)**2 / (tree_dia * 0.5)**2 + (
                                y[j] - tree_location_y)**2 / (tree_dia * 0.5)**2 + (
                                                 z[k] - crown_center)**2 / (crown_height * 0.5)**(
                                         2))
                    if r_test <= 1.0:
                        lad_loc[k, j, i] = lad_max * np.exp(- sphere_extinction * (1.0 - r_test))
                    else:
                        lad_loc[k, j, i] = ma.masked

                if ma.any(~lad_loc.mask[:, j, i]):
                    lad_loc[0, j, i] = 0.0

    # Branch for cylinder shapes
    if tree_shape == 2:
        k_min = int((crown_center - crown_height * 0.5) / dz)
        k_max = int((crown_center + crown_height * 0.5) / dz)
        for i in range(0, nx):
            for j in range(0, nx):
                for k in range(k_min, k_max):
                    r_test = np.sqrt((x[i] - tree_location_x)**2 / (tree_dia * 0.5)**2 + (
                                y[j] - tree_location_y)**2 / (tree_dia * 0.5)**2)
                    if r_test <= 1.0:
                        r_test3 = np.sqrt((z[k] - crown_center)**2 / (crown_height * 0.5)**2)
                        lad_loc[k, j, i] = lad_max * np.exp(
                            - sphere_extinction * (1.0 - max(r_test, r_test3)))
                    else:
                        lad_loc[k, j, i] = ma.masked

                if ma.any(~lad_loc.mask[:, j, i]):
                    lad_loc[0, j, i] = 0.0

    # Branch for cone shapes
    if tree_shape == 3:
        k_min = int((crown_center - crown_height * 0.5) / dz)
        k_max = int((crown_center + crown_height * 0.5) / dz)
        for i in range(0, nx):
            for j in range(0, nx):
                for k in range(k_min, k_max):
                    k_rel = k - k_min
                    r_test = (x[i] - tree_location_x)**2 + (y[j] - tree_location_y)**2 - (
                            (tree_dia * 0.5)**2 / crown_height**2) * (
                                         z[k_rel] - crown_height)**2
                    if r_test <= 0.0:
                        r_test2 = np.sqrt((x[i] - tree_location_x)**2 / (tree_dia * 0.5)**2 + (
                                    y[j] - tree_location_y)**2 / (tree_dia * 0.5)**2)
                        r_test3 = np.sqrt((z[k] - crown_center)**2 / (crown_height * 0.5)**2)
                        lad_loc[k, j, i] = lad_max * np.exp(
                            - cone_extinction * (1.0 - max((r_test + 1.0), r_test2, r_test3)))
                    else:
                        lad_loc[k, j, i] = ma.masked

                if ma.any(~lad_loc.mask[:, j, i]):
                    lad_loc[0, j, i] = 0.0

    # Branch for inverted cone shapes. TODO: what is r_test2 and r_test3 used for? Debugging needed!
    if tree_shape == 4:
        k_min = int((crown_center - crown_height * 0.5) / dz)
        k_max = int((crown_center + crown_height * 0.5) / dz)
        for i in range(0, nx):
            for j in range(0, nx):
                for k in range(k_min, k_max):
                    k_rel = k_max - k
                    r_test = (x[i] - tree_location_x)**2 + (y[j] - tree_location_y)**2 - (
                            (tree_dia * 0.5)**2 / crown_height**2) * (
                                         z[k_rel] - crown_height)**2
                    if r_test <= 0.0:
                        r_test2 = np.sqrt((x[i] - tree_location_x)**2 / (tree_dia * 0.5)**2 + (
                                    y[j] - tree_location_y)**2 / (tree_dia * 0.5)**2)
                        r_test3 = np.sqrt((z[k] - crown_center)**2 / (crown_height * 0.5)**2)
                        lad_loc[k, j, i] = lad_max * np.exp(- cone_extinction * (- r_test))
                    else:
                        lad_loc[k, j, i] = ma.masked

                if ma.any(~lad_loc.mask[:, j, i]):
                    lad_loc[0, j, i] = 0.0

    # Branch for paraboloid shapes
    if tree_shape == 5:
        k_min = int((crown_center - crown_height * 0.5) / dz)
        k_max = int((crown_center + crown_height * 0.5) / dz)
        for i in range(0, nx):
            for j in range(0, nx):
                for k in range(k_min, k_max):
                    k_rel = k - k_min
                    r_test = ((x[i] - tree_location_x)**2 + (y[j] - tree_location_y)**(
                        2)) * crown_height / (tree_dia * 0.5)**2 - z[k_rel]
                    if r_test <= 0.0:
                        lad_loc[k, j, i] = lad_max * np.exp(- cone_extinction * (- r_test))
                    else:
                        lad_loc[k, j, i] = ma.masked

                if ma.any(~lad_loc.mask[:, j, i]):
                    lad_loc[0, j, i] = 0.0

    # Branch for inverted paraboloid shapes
    if tree_shape == 6:
        k_min = int((crown_center - crown_height * 0.5) / dz)
        k_max = int((crown_center + crown_height * 0.5) / dz)
        for i in range(0, nx):
            for j in range(0, nx):
                for k in range(k_min, k_max):
                    k_rel = k_max - k
                    r_test = ((x[i] - tree_location_x)**2 + (y[j] - tree_location_y)**(
                        2)) * crown_height / (tree_dia * 0.5)**2 - z[k_rel]
                    if r_test <= 0.0:
                        lad_loc[k, j, i] = lad_max * np.exp(- cone_extinction * (- r_test))
                    else:
                        lad_loc[k, j, i] = ma.masked

                if ma.any(~lad_loc.mask[:, j, i]):
                    lad_loc[0, j, i] = 0.0


    # Create BAD array and populate. TODO: revise as low LAD inside the foliage does not result
    # in low BAD values.
    bad_loc = (1.0 - (lad_loc / (ma.max(lad_loc) + 0.01))) * 0.1

    # Overwrite grid cells that are occupied by the tree trunk
    radius = trunk_dia * 0.5
    for i in range(0, nx):
        for j in range(0, nx):
            for k in range(0, nz):
                if z[k] <= crown_center:
                    r_test = np.sqrt((x[i] - tree_location_x)**2 + (y[j] - tree_location_y)**2)
                    if r_test == 0.0:
                        if trunk_dia <= dx:
                            bad_loc[k, j, i] = radius**2 * 3.14159265359
                        else:
                            # WORKAROUND: divide remaining circle area over the 8 surrounding
                            # valid_pixels
                            bad_loc[k, j - 1:j + 2, i - 1:i + 2] = radius**2 * 3.14159265359 / 8.0
                            # for the central pixel fill the pixel
                            bad_loc[k, j, i] = dx**2
                    # elif ( r_test <= radius ):
                    # TODO: calculate circle segment of grid points cut by the grid

    return lad_loc, bad_loc, x, y, z, low_lai_counter, mod_counter, 0
