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
from glob import glob
import matplotlib.pyplot as plt
import numpy.ma as ma
import numpy as np
from scipy.interpolate import interp1d
import re

from util.pre_process_tif import *
from util.loc_dom import *
from util.read_geo import *
from util.create_static import*

def generate_static_LAD_3D(case_name, tmp_path, idomain, config_proj, dom_dict):
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
    dem_tif = f"{tmp_path}{case_name}_DEM.tif"
    print(dem_tif)
   
    #sfch_tif = dem_tif.replace("DEM","SFCH")
    bldh_tif = dem_tif.replace("DEM","BLDH")


    vegetation_type_tif = dem_tif.replace("DEM","vegetation_type")
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
    
    print('Reading building height')
    if not os.path.exists(bldh_tif):
        bldh = np.zeros_like(zt)
        bldh[:] = np.nan
    else:
        bldh_geo, lat, lon = readgeotiff(bldh_tif)
        bldh = extract_tiff(bldh_geo, lat, lon, dom_dict, tif_left,tif_south,tif_north,config_proj,case_name, idomain)
        del bldh_geo, lat, lon
        gc.collect()
    
    
    print('Reading vegetation_type')
    if not os.path.exists(vegetation_type_tif):
        vegetation_type = np.zeros_like(zt)
        vegetation_type[:] = np.nan
    else:
        vegetation_type_geo, lat, lon = readgeotiff(vegetation_type_tif)
        vegetation_type = extract_tiff(vegetation_type_geo, lat, lon, dom_dict, tif_left,tif_south,tif_north,config_proj,case_name, idomain)
        del vegetation_type_geo, lat, lon
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
        
    

 
    
    
    
    
    
    # Create an array with the same shape as the original but filled with '0'
    tree_type_array = np.full(tree_trunk_diameter.shape, 0, dtype=object)

    tree_type_array[tree_trunk_diameter ==-9999]=-9999

    single_tree_height=ma.masked_where((single_tree_height==0)|(single_tree_height<0), single_tree_height, copy = False)
    tree_crown_diameter=ma.masked_where((tree_crown_diameter==0)|(tree_crown_diameter<0),tree_crown_diameter, copy = False)
    vegetation_height=ma.masked_where((vegetation_height==0)|(vegetation_height<0), vegetation_height, copy = False)
    tree_trunk_diameter=ma.masked_where((tree_trunk_diameter==0)|(tree_trunk_diameter<0), tree_trunk_diameter, copy = False)
    tree_type_array=ma.masked_where((tree_type_array==0)|(tree_type_array<0), tree_type_array, copy = False)
    vegetation_type = ma.masked_where((vegetation_type<0), vegetation_type, copy = False)
    
    bldh = ma.masked_where((bldh<0), bldh, copy = False)
   
    
    # LAD files (1 tif file for each m) 
    file_list = sorted(glob(os.path.join(tmp_path, "LAD_layer_*.tif")), key=sort_key)
    layer_arrays=[]

    for file in file_list:
        array_geo, lat, lon = readgeotiff(file)
        array_palm = extract_tiff(array_geo, lat, lon, dom_dict, tif_left,tif_south,tif_north,config_proj,case_name, idomain)
    
        layer_arrays.append(array_palm)

    # Convert list of arrays to a single stacked array along a new dimension
    final_array = np.stack(layer_arrays, axis=0)
    
    final_array_intp=interpolate_along_z(final_array, dz)

    # Expand dimensions of the 2D mask to match the 3D array shape
    expanded_mask = (np.sum(final_array_intp, axis=0) == 0) | (np.sum(final_array_intp, axis=0) < 0 | (bldh>0))
    expanded_mask = np.expand_dims(expanded_mask, axis=0)  # shape will become (1, 324, 324)
    expanded_mask = np.repeat(expanded_mask, final_array_intp.shape[0], axis=0)  # repeat along the first axis, shape will become (9, 324, 324)

    # Now, apply the mask
    LAD_array = np.ma.masked_where(expanded_mask, final_array_intp)
    
    
    
    #Use a modified version generate_single_tree_lad

    print('generate_single_tree')

    lad,bad,tree_ids,tree_type, zlad= generate_single_tree_LAD_3D(x, y, dz, vegetation_height, tree_type_array, single_tree_height,tree_crown_diameter,
                             tree_trunk_diameter, LAD_array, 0,
                             ma.masked)
    
    
    
    
     #Use a modified version generate_patch

    print('generate patch')    
    
    lad_patch, patch_id,patch_types, patch_nz,status = process_patch_LAD_3D(dz, vegetation_height, vegetation_type, vegetation_type, np.max(vegetation_height), LAD_array, 5.0, 3.0)
    

    
        
        
    return lad, bad, tree_ids, tree_type, lad_patch, patch_id,patch_types




def generate_single_tree_LAD_3D(x, y, dz, patch_height, tree_type, tree_height, tree_dia,
                             trunk_dia, LAD, lai_tree_lower_threshold,
                             remove_low_lai_tree):
    
    """
    Modified from canopy_generator.py
    
    Inputs 3D LAD profile
    Returns LAD, BAD and tree ID 


    """
    # Step 1: Create arrays for storing the data
    max_canopy_height = ma.max(patch_height)

    zlad = np.arange(0, math.floor(max_canopy_height / dz) * dz + 2 * dz, dz)
    zlad[1:] = zlad[1:] - 0.5 * dz

    lad = LAD[0:len(zlad),:,:]

    bad = ma.ones((len(zlad), len(y), len(x)))
    bad[:, :, :] = ma.masked

    ids = ma.ones((len(zlad), len(y), len(x)))
    ids[:, :, :] = ma.masked

    types = ma.ones((len(zlad), len(y), len(x)))
    types[:, :, :] = np.byte(-127)

    # Calculating the number of trees in the arrays and a boolean array storing the location of
    # trees which is used for convenience in the following loop environment
    number_of_trees_array = ma.where(
        ~tree_type.mask.flatten() | ~trunk_dia.mask.flatten(),
        1.0, ma.masked)
    number_of_trees = len(number_of_trees_array[number_of_trees_array == 1.0])
    dx = x[1] - x[0]

    valid_pixels = ma.where(~tree_type.mask | ~trunk_dia.mask,
                            True, False)


    # For each tree, create a small 3d array containing the LAD field for the individual tree
    print("Start generating " + str(number_of_trees) + " trees...")
    print('test')
    print(number_of_trees_array)
    tree_id_counter = 0
    if number_of_trees > 0:
        low_lai_count = 0
        mod_count = 0
        for i in range(0, len(x)):
            for j in range(0, len(y)):
                if valid_pixels[j, i]:
                    tree_id_counter = tree_id_counter + 1

                    print("   Processing tree No " +  str(tree_id_counter) + " ...", end="")
                    lad_loc, bad_loc, x_loc, y_loc, z_loc, low_lai_count, mod_count, status = \
                        process_single_tree_LAD_3D(i, j, dx, dz, LAD, tree_height[j,i],  tree_dia[j,i],
                                            tree_type[j, i], trunk_dia[j, i], lai_tree_lower_threshold, remove_low_lai_tree,
                                            low_lai_count, mod_count)

                    if status == 0 and ma.any(~lad_loc.mask):
                        # Calculate the position of the local 3d tree array within the full
                        # domain in order to achieve correct mapping and cutting off at the edges
                        # of the full domain
                        #This following part has not been changed

                        lad_loc_nx = int(len(x_loc) / 2)
                        lad_loc_ny = int(len(y_loc) / 2)
                        lad_loc_nz = int(len(z_loc))

                        odd_x = int(len(x_loc) % 2)
                        odd_y = int(len(y_loc) % 2)

                        ind_l_x = max(0, (i - lad_loc_nx))
                        ind_l_y = max(0, (j - lad_loc_ny))
                        ind_r_x = min(len(x) - 1, i + lad_loc_nx - 1 + odd_x)
                        ind_r_y = min(len(y) - 1, j + lad_loc_ny - 1 + odd_y)

                        out_l_x = ind_l_x - (i - lad_loc_nx)
                        out_l_y = ind_l_y - (j - lad_loc_ny)
                        out_r_x = len(x_loc) - 1 + ind_r_x - (i + lad_loc_nx - 1 + odd_x)
                        out_r_y = len(y_loc) - 1 + ind_r_y - (j + lad_loc_ny - 1 + odd_y)

                        lad[0:lad_loc_nz, ind_l_y:ind_r_y + 1, ind_l_x:ind_r_x + 1] = ma.where(
                            ~lad_loc.mask[0:lad_loc_nz, out_l_y:out_r_y + 1, out_l_x:out_r_x + 1],
                            lad_loc[0:lad_loc_nz, out_l_y:out_r_y + 1, out_l_x:out_r_x + 1],
                            lad[0:lad_loc_nz, ind_l_y:ind_r_y + 1, ind_l_x:ind_r_x + 1])
                        bad[0:lad_loc_nz, ind_l_y:ind_r_y + 1, ind_l_x:ind_r_x + 1] = ma.where(
                            ~bad_loc.mask[0:lad_loc_nz, out_l_y:out_r_y + 1, out_l_x:out_r_x + 1],
                            bad_loc[0:lad_loc_nz, out_l_y:out_r_y + 1, out_l_x:out_r_x + 1],
                            bad[0:lad_loc_nz, ind_l_y:ind_r_y + 1, ind_l_x:ind_r_x + 1])
                        ids[0:lad_loc_nz, ind_l_y:ind_r_y + 1, ind_l_x:ind_r_x + 1] = ma.where(
                            ~lad_loc.mask[0:lad_loc_nz, out_l_y:out_r_y + 1, out_l_x:out_r_x + 1],
                            tree_id_counter,
                            ids[0:lad_loc_nz, ind_l_y:ind_r_y + 1, ind_l_x:ind_r_x + 1])
                        types[0:lad_loc_nz, ind_l_y:ind_r_y + 1, ind_l_x:ind_r_x + 1] = ma.where(
                            ~lad_loc.mask[0:lad_loc_nz, out_l_y:out_r_y + 1, out_l_x:out_r_x + 1],
                            np.byte(tree_type[j, i]),
                            types[0:lad_loc_nz, ind_l_y:ind_r_y + 1, ind_l_x:ind_r_x + 1])

                    #                  if ( status == 0 ):
                    #                     status_char = " ok."
                    #                  else:
                    #                     status_char = " skipped."
                    #                  print(status_char)

                    del lad_loc, x_loc, y_loc, z_loc, status
        if mod_count > 0:
            if remove_low_lai_tree:
                print("Removed", mod_count, "trees due to low LAI.")
            else:
                print("Adjusted LAI of", mod_count, "trees.")
        if low_lai_count > 0:
            print("Warning: Found", low_lai_count, "trees with LAI lower then the",
                  "tree type specific default winter LAI.",
                  "Consider adjusting lai_tree_lower_threshold and remove_low_lai_tree.")
    return lad, bad, ids, types, zlad



def process_single_tree_LAD_3D(i, j, dx, dz, LAD, tree_height, tree_dia,
                        tree_type, trunk_dia, lai_tree_lower_threshold, remove_low_lai_tree,
                        low_lai_counter, mod_counter):

    """
    Modified from canopy_generator.py
    
    Inputs 3D LAD profile
    Returns BAD and tree ID 


    """
    
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
    

    # Check for missing data in the input and set default values if needed
    if tree_type is ma.masked:
        tree_type = int(0)
    else:
        tree_type = int(tree_type)

    if trunk_dia is ma.masked:
        trunk_dia = default_trees[tree_type].dbh
        
    if tree_height is ma.masked:
        tree_height = int(0)
    else:
        tree_height = int(tree_height)

    # Check tree_lai
    # Tree LAI lower then threshold?
    if sum(LAD[:,j,i]) < lai_tree_lower_threshold:
        # Deal with low lai tree
        mod_counter = mod_counter + 1
        if remove_low_lai_tree:
            # Skip this tree
            print("Removed tree with LAI = ", "%0.3f", " at (", i, ", ", j, ").", sep="")
            return None, None, None, None, None, low_lai_counter, mod_counter, 1
        else:
            print("Adjusted tree to LAI = ", "%0.3f", " at (", i, ", ", j, ").", sep="")

    # Assign values that are not defined as user input from lookup table
    #tree_ratio = default_trees[tree_type].ratio
    #lad_max_height = default_trees[tree_type].lad_max_height
    #bad_scale = default_trees[tree_type].bad_scale

    print("Tree input parameters:")
    print("----------------------")
    print("type:           " + str(default_trees[tree_type].species) )
    print("trunk diameter: " + str(trunk_dia))


    # Calculate height of maximum LAD
    #z_lad_max = lad_max_height * tree_height


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
    
    crown_height = 0.025 * tree_dia      #0.025 is Ratio LAD/BAD - missing spo used value from look up table
    if crown_height > tree_height:
        crown_height = tree_height

    crown_center = tree_height - crown_height * 0.5

    lad_loc = LAD
    bad_loc = ma.copy(lad_loc)

    # For very small trees, no LAD is calculated
    if tree_height <= (0.5 * dz):
        print("    Shallow tree found. Action: ignore.")
        lad_loc[:, x, y] = ma.masked
        return lad_loc, bad_loc, x, y, z, low_lai_counter, mod_counter, 1


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





def interpolate_along_z(data, dz):
    """
    Interpolate a 3D array along the z-axis.

    Parameters:
    - data: 3D numpy array of shape (z, y, x)
    - dz: New z-spacing

    Returns:
    - Interpolated 3D array
    """
    # Original z-coordinates
    z_old = np.arange(data.shape[0])

    # New z-coordinates
    z_new = np.arange(0, data.shape[0], dz)

    # Placeholder for the interpolated data
    interpolated_data = np.empty((len(z_new), data.shape[1], data.shape[2]))

    # Loop over the x and y dimensions and interpolate along z
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            f = interp1d(z_old, data[:, i, j], kind='linear', fill_value="extrapolate")
            interpolated_data[:, i, j] = f(z_new)

    return interpolated_data




# Extract the number from the filename
def sort_key(filename):
    match = re.search(r'layer_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0





def process_patch_LAD_3D(dz, patch_height, patch_type_2d, vegetation_type, max_height_lad, LAD, alpha, beta):
    """
    Modified from canopy_generator.py
    
    Inputs 3D LAD profile


    """
    phdz = patch_height[:, :] / dz
    pch_index = ma.where(patch_height.mask, int(-1), phdz.astype(int) + 1)
    ma.masked_equal(pch_index, 0, copy=False)
    pch_index = ma.where(pch_index == -1, 0, pch_index)

    max_canopy_height = max(ma.max(patch_height), max_height_lad)

    z = np.arange(0, math.floor(max_canopy_height / dz) * dz + 2 * dz, dz)

    z[1:] = z[1:] - 0.5 * dz

    nz = len(z)
    
    ny = len(patch_height[:, 0])
    nx = len(patch_height[0, :])
    
    
    lad_loc = LAD[0:nz,:,:]

    print(lad_loc.shape)

    patch_id_2d = ma.where(lad_loc.mask[0, :, :], 0, 1)
    patch_id_3d = ma.where(lad_loc.mask, 0, 1)
    patch_type_3d = ma.empty((nz, ny, nx))

    for k in range(0, nz):
        patch_id_3d[k, :, :] = ma.where((patch_id_2d != 0) & ~lad_loc.mask[k, :, :],
                                     patch_id_2d, ma.masked)
        patch_type_3d[k, :, :] = ma.where((patch_id_2d != 0) & ~lad_loc.mask[k, :, :],
                                       patch_type_2d, ma.masked)

    return lad_loc, patch_id_3d, patch_type_3d, nz, 0




# CLASS TREE
#
# Default tree geometrical parameters:
#
# species: name of the tree type
#
# shape: defines the general shape of the tree and can be one of the following types:
# 1.0 sphere or ellipsoid
# 2.0 cylinder
# 3.0 cone
# 4.0 inverted cone
# 5.0 paraboloid (rounded cone)
# 6.0 inverted paraboloid (invertes rounded cone)
#
# ratio:  ratio of maximum crown height to the maximum crown diameter
# diameter: default crown diameter (m)
# height:   default total height of the tree including trunk (m)
# lai_summer: default leaf area index fully leafed
# lai_winter: default winter-teim leaf area index
# lad_max: default maximum leaf area density (m2/m3)
# lad_max_height: default height where the leaf area density is maximum relative to total tree
#                 height
# bad_scale: ratio of basal area in the crown area to the leaf area
# dbh: default trunk diameter at breast height (1.4 m) (m)
#
class Tree:
    def __init__(self, species, shape, ratio, diameter, height, lai_summer, lai_winter,
                 lad_max_height, bad_scale, dbh):
        self.species = species
        self.shape = shape
        self.ratio = ratio
        self.diameter = diameter
        self.height = height
        self.lai_summer = lai_summer
        self.lai_winter = lai_winter
        self.lad_max_height = lad_max_height
        self.bad_scale = bad_scale
        self.dbh = dbh
