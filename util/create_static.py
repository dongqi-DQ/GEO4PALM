#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------#
# Read geotiff files and convert to PALM static file
# 
# @author: Dongqi Lin, Jiawei Zhang, Eva Bendix Nielsen
#--------------------------------------------------------------------------------#
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=FutureWarning)
import numpy as np
import xarray as xr
import time
import pandas as pd
import gc
from netCDF4 import Dataset
from pyproj import Proj, transform
import scipy.integrate as integrate
from scipy.ndimage.measurements import label
from util.read_geo import readgeotiff
from util.nearest import nearest
from util.palm_lu import lu2palm, get_albedo
from util.get_sst import nearest_sst, get_nearest_sst 
from util.Static_driver_vegetation_2D import *
from util.Static_driver_vegetation_3D import *
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=UserWarning)
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.crs import CRS
    from rasterio import logging
    log = logging.getLogger()
    log.setLevel(logging.ERROR)
import os


    
def array_to_raster(array,xmin,ymax,xsize,ysize,proj_str,output_filename):
    ## create reference tif file with original crs for each domain.
    arr = np.flip(array,axis=0)
    xmin = xmin-xsize/2
    ymax = ymax +ysize/2
    transform = from_origin(xmin, ymax, xsize, ysize)
    new_dataset = rasterio.open(output_filename, 'w', driver='GTiff',
                                height = arr.shape[0], width = arr.shape[1],
                                count=1, dtype=str(arr.dtype),
                                crs=CRS.from_string(proj_str),
                                transform=transform)
    new_dataset.write(arr, 1)
    new_dataset.close()

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
    
    if save_tif == True:
        #save tif file of each cropped domain
        array_to_raster(array_palm,xmin,ymax,xsize,ysize,proj_str,"./JOBS/"+case_name+"/OUTPUT/"+case_name+f"_static_dem_N{idomain+1:02d}.tif")

    return array_palm


def make_3d_from_2d(array_2d,x,y,dz):
    # PALM csd function
    # create 3d arrays and z coordinate for 3d variables
    # e.g. lad, buildings_3d
    k_tmp = np.arange(0,max(array_2d.flatten())+dz*2,dz)
 
    k_tmp[1:] = k_tmp[1:] - dz * 0.5
    array_3d = np.ones((len(k_tmp),len(y),len(x)))
  
    for l in range(0,len(x)):
        for m in range(0,len(y)):
            for k in range(0,len(k_tmp)):
                if k_tmp[k] > array_2d[m,l]:
                    array_3d[k,m,l] = 0

    return array_3d.astype(np.byte), k_tmp

def generate_palm_static(case_name, tmp_path, idomain, config_proj, dom_dict):
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
    ## land use look up table
    lu_csv_file = dom_dict["lu_csv_file"]
    
    ### leaf area index parameters
    lad_mode = dom_dict['lad_mode']
    tree_lai_max = dom_dict['tree_lai_max']      # default value 5.0
    lad_max_height = dom_dict['lad_max_height']  # default value 0.4
    
    ## water input file options
    water_temperature_file = dom_dict['water_temperature_file']
    
    ### settings
    water_temperature_default = dom_dict['water_temperature_default']
    bldh_dummy = dom_dict['bldh_dummy']
    tree_height_filter = dom_dict['tree_height_filter']
    
    # assign tiff file names
    dem_tif = f"{tmp_path}{case_name}_DEM_N{idomain+1:02d}.tif"
    lu_tif = dem_tif.replace("DEM","LU")
    bldh_tif = dem_tif.replace("DEM","BLDH")
    bldid_tif = dem_tif.replace("DEM","BLDID")   
    sfch_tif = dem_tif.replace("DEM","SFCH")
    pavement_tif = dem_tif.replace("DEM","pavement")
    street_tif = dem_tif.replace("DEM","street")

    lai_tif = dem_tif.replace("DEM","LAI")
    patch_type_tif = dem_tif.replace("DEM","patch_type")
    vegetation_height_tif = dem_tif.replace("DEM","vegetation_height")
    single_tree_height_tif = dem_tif.replace("DEM","tree_height")
    tree_crown_diameter_tif =  dem_tif.replace("DEM","tree_crown_diameter")
    tree_trunk_diameter_tif =  dem_tif.replace("DEM","tree_trunk_diameter")
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
    

    
    # Read land use files
    n_surface_fraction = 3
    
    print('Reading land use data')
    if not os.path.exists(lu_tif):
        lu = np.zeros_like(zt)
        lu[:] = np.nan
    else:
        lu_geo, lat, lon = readgeotiff(lu_tif)
        lu = extract_tiff(lu_geo, lat, lon, dom_dict, tif_left,tif_south,tif_north,config_proj,case_name, idomain)
        del lu_geo, lat, lon
        gc.collect()
    
    print('Reading building ID')
    if not os.path.exists(bldid_tif):
        bldid = np.zeros_like(zt)
        bldid[:] = np.nan
    else:
        bldid_geo, lat, lon = readgeotiff(bldid_tif)
        bldid = extract_tiff(bldid_geo, lat, lon, dom_dict, tif_left,tif_south,tif_north,config_proj,case_name, idomain)
        del bldid_geo, lat, lon
        gc.collect()
    
    print('Reading building height')
    if not os.path.exists(bldh_tif):
        bldh = np.zeros_like(zt)
        bldh[:] = np.nan
    else:
        bldh_geo, lat, lon = readgeotiff(bldh_tif)
        bldh = extract_tiff(bldh_geo, lat, lon, dom_dict, tif_left,tif_south,tif_north,config_proj,case_name, idomain)
        del bldh_geo, lat, lon
        gc.collect()
    
    print('Reading surface height for vegetation height')
    if not os.path.exists(sfch_tif):
        sfch_tmp = np.zeros_like(zt)
        sfch_tmp[:] = np.nan
    else:
        sfch_geo, lat, lon = readgeotiff(sfch_tif)
        sfch_tmp = extract_tiff(sfch_geo, lat, lon, dom_dict, tif_left,tif_south,tif_north,config_proj,case_name, idomain)
        del sfch_geo, lat, lon
        gc.collect()
    
    print('Reading pavement')
    
    if not os.path.exists(pavement_tif):
        pavement = np.zeros_like(zt)
        pavement[:] = np.nan
    else:
        pavement_geo, lat, lon = readgeotiff(pavement_tif)
        pavement = extract_tiff(pavement_geo, lat, lon, dom_dict, tif_left,tif_south,tif_north,config_proj,case_name, idomain)
        del pavement_geo, lat, lon    
        gc.collect()
    
    print('Reading street')
    if not os.path.exists(street_tif):
        street = np.zeros_like(zt)
        street[:] = np.nan
    else:
        street_geo, lat, lon = readgeotiff(street_tif)
        street = extract_tiff(street_geo, lat, lon, dom_dict, tif_left,tif_south,tif_north,config_proj,case_name, idomain)
        del street_geo, lat, lon    
        gc.collect()

    # process water type
    water_lu = lu2palm(lu, 'water', lu_csv_file)
    water_type =  np.array([[cell if cell>0 else -9999 for cell in row] for row in water_lu])
    
    # set up water temperature
    # other parameters stay the same as default
    print("Processing water temperature")
    water_tif = dem_tif.replace("DEM","WATER_T")
    # create array for water_pars
    water_pars = np.zeros((7,water_type.shape[0],water_type.shape[1]))
    
    if water_temperature_file == "online":
        static_tif_path = tmp_path.replace("TMP","INPUT")
        water_temperature = get_nearest_sst(centlat, centlon, case_name, static_tif_path)
        water_pars[0,:,:][water_type>0] = water_temperature
    elif os.path.exists(water_tif):
        water_geo, lat, lon = readgeotiff(water_tif)
        water_data = extract_tiff(water_geo, lat, lon, dom_dict, tif_left,tif_south,tif_north,config_proj,case_name)
        del water_geo, lat, lon    
        gc.collect()
        water_temperature = water_data
        ## use input water temperature to locate water bodies
        water_type[water_temperature>0] = 2
        ## apply NaNs
        water_temperature[water_type<0] = -9999.0
        water_pars[0,:,:] = water_temperature[:,:]
    else:
        water_pars[0,:,:][water_type>0] = water_temperature_default
    
    water_pars[water_pars<=0] = -9999.0
    
    
    # process pavement
    pavement_lu = lu2palm(lu, 'pavement', lu_csv_file)
    pavement_type =  np.array([[cell if cell>0 else -9999 for cell in row] for row in pavement])
    
    # if pavement tif input is provided
    if os.path.exists(pavement_tif):
        pavement_type[pavement_lu>0] = 3
    else:
        pavement_type[:,:] = -9999
    pavement_type[water_type>0] = -9999
    

    
    # process street
    street_type =  np.array([[cell if cell>0 else -9999 for cell in row] for row in street])
    street_type[pavement_lu>0] = 17
    street_type[water_type>0] = -9999
    
    # process building height
    building_height =  np.array([[cell if cell>=0 else -9999 for cell in row] for row in bldh])
    building_height[water_type>0] = -9999
    building_height[pavement_type > 0 ] = -9999
    
    building_id  =  np.array([[cell if cell>0 else -9999 for cell in row] for row in bldid])
    building_height[(building_id>0) &(building_height==0) ] = bldh_dummy
    building_id[building_height==-9999] = -9999
    
    building_type =  np.array([[3 if cell>0 else -9999 for cell in row] for row in building_id])
    
    ## match building height with building ID
    building_height[building_id==-9999] = -9999
    
    # if the buildings input is provided
    if os.path.exists(bldh_tif):
        buildings_3d, zbld = make_3d_from_2d(building_height, x, y, dz)
    
    # process vegetation
    vegetation_type = lu2palm(lu, 'vegetation', lu_csv_file)
    vegetation_type[water_type>0] = -9999
    vegetation_type[pavement_type>0] = -9999
    vegetation_type[building_type>0] = -9999

    # process bare land
    bare_land = lu2palm(lu, 'building', lu_csv_file)
    bare_land[building_type>0] = -9999 
    bare_land[pavement_type>0] = -9999 
    bare_land[water_type>0] = -9999 
    bare_land[vegetation_type>0] = -9999
    # if the buildings input is provided
    if os.path.exists(bldh_tif):
        vegetation_type[bare_land>0] = 1 # bare land 
    else:
        urban_type = 18                   # vegetation type to represent roughness length of urban buildup
        vegetation_type[bare_land>0] = urban_type  
        
    # process soil
    soil_type = np.zeros_like(lu)
    soil_type[:] = -9999
    soil_type = lu2palm(lu, 'soil', lu_csv_file)
    soil_type[building_type>0] = -9999 
    soil_type[pavement_type>0] = 1
    soil_type[water_type>0] = -9999
    for i in range(soil_type.shape[1]):
        for j in range(soil_type.shape[0]):
            if vegetation_type[j, i] == 1 and soil_type[j, i] < 0:
                soil_type[j, i] = 2
            if vegetation_type[j, i] > 1 and soil_type[j, i] < 0:
                soil_type[j, i] = 3
    # All area without land use information (when nesting with coarse resolution)
    # set as grass land
    for idx in range(soil_type.shape[1]):
        for idy in range(soil_type.shape[0]):
            if vegetation_type[idy, idx] < 0 and pavement_type[idy, idx] < 0 and water_type[idy, idx] < 0 and building_type[idy, idx] < 0:
                vegetation_type[idy, idx] = 3
                soil_type[idy, idx] = 3
                
    ############
    ##  LAD   ##
    ############
    # MODE 1:
    # if only surface height input is provided
    # calculate leaf area density (LAD) 
    
    if lad_mode == 1 and os.path.exists(sfch_tif):
        # process lad
        # remove cars and some other "noise"
            #tree_height = np.copy(sfch_tmp)   #tree height changed to DSM
            tree_height = vegetation_height
            tree_height[tree_height<=1.5] = -9999
        # remove single points in the domain
            n_thresh = 1
            labeled_array, num_features = label(tree_height)
            binc = np.bincount(labeled_array.ravel())
            noise_idx = np.where(binc <= n_thresh)
            shp = tree_height.shape
            mask = np.in1d(labeled_array, noise_idx).reshape(shp)
            tree_height[mask] = -9999
        # generate 3d variables for LAD        
            tree_3d, zlad = make_3d_from_2d(tree_height, x, y, dz)
            tree_3d = tree_3d.astype(np.float)
            tree_3d[tree_3d==0] = -9999
        # specify parameters for LAD calculation
         #   tree_lai_max = 5
         #   lad_max_height = 0.4
            scale_height=np.nanquantile(np.where(tree_height>0,tree_height,np.nan),0.9) # use the top 10% quantile to scale the lai  
            ml_n_low = 0.5
            ml_n_high = 6.0

            print("Calculating LAD")
            for idx in range(0, zt.shape[1]):
                for idy in range(0, zt.shape[0]):
                    if tree_height[idy, idx] <= 0:
                        continue
                    else:
                        #  Calculate height of maximum LAD
                        tree_lai_scaled = tree_lai_max* tree_height[idy, idx]/scale_height
                        z_lad_max = lad_max_height * tree_height[idy, idx]

                        #  Calculate the maximum LAD after Lalic and Mihailovic (2004)
                        lad_max_part_1 = integrate.quad(lambda z: ( ( tree_height[idy, idx]- z_lad_max ) / ( tree_height[idy, idx] - z ) ) ** (ml_n_high) * np.exp( ml_n_high * (1.0 - ( tree_height[idy, idx] - z_lad_max ) / ( tree_height[idy, idx] - z ) ) ), 0.0, z_lad_max)
                        lad_max_part_2 = integrate.quad(lambda z: ( ( tree_height[idy, idx] - z_lad_max ) / ( tree_height[idy, idx] - z ) ) ** (ml_n_low) * np.exp( ml_n_low * (1.0 - ( tree_height[idy, idx] - z_lad_max ) / ( tree_height[idy, idx] - z ) ) ), z_lad_max, tree_height[idy, idx])

                        lad_max = tree_lai_scaled / (lad_max_part_1[0] + lad_max_part_2[0])

                        lad_profile     =  np.zeros_like(zlad)
                        for k in range(1,len(lad_profile)):
                            if zlad[k] > 0.0 and zlad[k] < z_lad_max:
                                n = ml_n_high
                            else:
                                n = ml_n_low
                            lad_profile[k] =  lad_max * ( ( tree_height[idy, idx] - z_lad_max ) / ( tree_height[idy, idx] - zlad[k] ) ) ** (n) * np.exp( n * (1.0 - ( tree_height[idy, idx] - z_lad_max ) / ( tree_height[idy, idx] - zlad[k] ) ) )
                        tree_3d[:, idy, idx] = lad_profile[:]
            ####### end of LAD calculation
            # convert nans to -9999
            tree_3d[np.isnan(tree_3d)] = -9999
            tree_3d[tree_3d<=0] = -9999
            tree_3d[0][tree_height==0] = -9999
    ############
    ##  LAD   ##
    ############
    # MODE 2:
    # If 2D properties are provided
    if lad_mode == 2 and os.path.exists(lai_tif):

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
            print("Use land use classification for vegetation type")
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
            vegetation_height[np.isnan(vegetation_height)] = 0 
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
        if not os.path.exists(tree_crown_diameter_tif):
            tree_trunk_diameter = np.zeros_like(zt)
            tree_trunk_diameter[:] = np.nan
        else:
            tree_trunk_diameter_geo, lat, lon = readgeotiff(tree_trunk_diameter_tif)
            tree_trunk_diameter = extract_tiff(tree_trunk_diameter_geo, lat, lon, dom_dict, tif_left,tif_south,tif_north,config_proj,case_name, idomain)
            del tree_trunk_diameter_geo, lat, lon
            gc.collect()
        
        tree_3d, zlad = make_3d_from_2d(vegetation_height, x, y, dz)

        ## now generate LAD
        lad_3d, bad_3d, tree_ids_3d, tree_type_3d, lad_patch, patch_id,patch_types = generate_palm_static_LAD(case_name, tmp_path, idomain, config_proj, dom_dict)
    
    # check overlaps
    for idx in range(0, zt.shape[1]):
        for idy in range(0, zt.shape[0]):
            if vegetation_type[idy, idx] < 0 and  water_type[idy, idx] < 0 and pavement_type[idy, idx]< 0:
                vegetation_type[idy, idx] = 1
                soil_type[idy, idx] = 2
    
    # set up fillvalues
    vegetation_type[building_type>0] = -9999
    vegetation_type[vegetation_type == -9999.0] = -127.0
    pavement_type[pavement_type == -9999.0] = -127.0
    street_type[street_type == -9999.0] = -127.0
    building_type[building_type == -9999.0] = -127.0
    water_type[water_type == -9999.0] = -127.0
    soil_type[soil_type == -9999.0] = -127.0
    soil_type[soil_type == 0] = -127.0
    
    # calculate albedo type for vegetation, pavements, buildings, water
    albedo_type = np.zeros_like(vegetation_type)
    albedo_pavement = get_albedo(pavement_type, "pavement")
    albedo_vegetation = get_albedo(vegetation_type, "vegetation")
    albedo_water = get_albedo(water_type, "water")
    # if building data are provided
    if os.path.exists(bldh_tif):
        albedo_building = get_albedo(building_type, "building")
    else:
        bare_land[vegetation_type!=urban_type] = -9999.0
        albedo_building = get_albedo(bare_land, "building")
    
    albedo_vegetation[albedo_building>0] = 0
    albedo_type = albedo_pavement + albedo_vegetation + albedo_building + albedo_water
    albedo_type[albedo_type[:,:]<=0] = -127.0
    
    
    # process surface fraction
    surface_fraction = np.array([np.zeros_like(lu), np.zeros_like(lu), np.zeros_like(lu)])
    surface_fraction[:] = 0
    surface_fraction[0][vegetation_type > 0] = 1
    surface_fraction[1][pavement_type > 0] = 1
    surface_fraction[2][water_type > 0] = 1
    
    # output netcdf file
    print(f"Generating netcdf static driver for domain N{idomain+1:02d}")
    
    nc_output = xr.Dataset()
    nc_output.attrs['description'] = 'PALM static driver - containing geospatial data from OSM, NZLCDB, DEM, DSM etc.'
    nc_output.attrs['history'] = 'Created at' + time.ctime(time.time())
    nc_output.attrs['source'] = 'multiple source'
    nc_output.attrs['origin_lat'] = np.float32(centlat)
    nc_output.attrs['origin_lon'] = np.float32(centlon)
    nc_output.attrs['origin_z'] = z_origin
    nc_output.attrs['origin_x'] = tif_left
    nc_output.attrs['origin_y'] = tif_south
    nc_output.attrs['rotation_angle'] = float(0)
    nc_output.attrs['origin_time'] = origin_time
    nc_output.attrs['Conventions'] = 'CF-1.7'
    if os.path.exists(bldh_tif):
         nc_output['z'] = xr.DataArray(zbld.astype(np.float32), dims=['z'], attrs={'units': 'm', 'axis': 'z'})
    if os.path.exists(sfch_tif) or os.path.exists(lai_tif):
        nc_output['zlad'] = xr.DataArray(zlad.astype(np.float32), dims=['zlad'], attrs={'units': 'm', 'axis': 'zlad'})
    nc_output.to_netcdf(static_driver_file, mode='w', format='NETCDF4')
    
    nc_output['x'] = xr.DataArray(x.astype(np.float32), dims=['x'], attrs={'units': 'm', 'axis': 'x','_FillValue': -9999,
                                                                           'long_name': 'distance to origin in x-direction'})
    nc_output['y'] = xr.DataArray(y.astype(np.float32), dims=['y'], attrs={'units': 'm', 'axis': 'y','_FillValue': -9999,
                                                                           'long_name': 'distance to origin in y-direction'})
    
    nc_output['zt'] = xr.DataArray(zt, dims=['y', 'x'],
                                     attrs={'units': 'm', 'long_name': 'terrain_height'})
    
    for var in nc_output.data_vars:
        encoding = {var: {'dtype': 'float32', '_FillValue': -9999, 'zlib': True}}
        nc_output[var].to_netcdf(static_driver_file, encoding=encoding, mode='a')

    
    nc_output = Dataset(static_driver_file, "a", format="NETCDF4")
    
    nc_output.createDimension('nsurface_fraction', n_surface_fraction)
    nc_nsurface_fraction = nc_output.createVariable('nsurface_fraction', np.int32, 'nsurface_fraction')
    nc_vegetation = nc_output.createVariable('vegetation_type', np.byte, ('y', 'x'), fill_value=np.byte(-127), zlib=True)
    nc_pavement = nc_output.createVariable('pavement_type', np.byte, ('y', 'x'), fill_value=np.byte(-127), zlib=True)
    nc_water = nc_output.createVariable('water_type', np.byte, ('y', 'x'), fill_value=np.byte(-127), zlib=True)
    nc_albedo = nc_output.createVariable('albedo_type', np.byte, ('y', 'x'), fill_value=np.byte(-127), zlib=True)
    nc_soil = nc_output.createVariable('soil_type', np.byte, ('y', 'x'), fill_value=np.byte(-127), zlib=True)
    nc_surface_fraction = nc_output.createVariable('surface_fraction', np.float32, ('nsurface_fraction', 'y', 'x'), fill_value=-9999.0, zlib=True)     
    nc_output.createDimension('nwater_pars', 7)
    nc_nwater_pars = nc_output.createVariable('nwater_pars', np.int32, 'nwater_pars')
    nc_water_pars = nc_output.createVariable('water_pars', np.float32, ('nwater_pars', 'y', 'x'), fill_value=-9999.0, zlib=True)
    

    nc_vegetation.long_name = 'vegetation_type_classification'
    nc_vegetation.units = ''

    nc_pavement.long_name = 'pavement_type_classification'
    nc_pavement.res_orig = np.float32(dx)
    nc_pavement.units = ''
    
    nc_water.long_name = 'water_type_classification'
    nc_water.res_orig = np.float32(dx)
    nc_water.units = ''
    
    nc_albedo.long_name = 'albedo type classification'
    nc_albedo.res_orig = np.float32(dx)
    nc_albedo.units = ''

    nc_soil.long_name = 'soil_type_classification'
    nc_soil.res_orig = np.float32(dx)
    nc_soil.units = ''

    nc_surface_fraction.long_name = 'surface_fraction'
    nc_surface_fraction.res_orig = np.float32(dx)
    nc_surface_fraction.units = ''
    
    nc_water_pars.long_name = 'water_parameters'
    nc_water_pars.res_orig = np.float32(dx)

    
    nc_vegetation[:] = vegetation_type
    nc_pavement[:] = pavement_type
    nc_water[:] = water_type
    nc_albedo[:] = albedo_type
    nc_soil[:] = soil_type
    nc_surface_fraction[:] = surface_fraction
    nc_nsurface_fraction[:] = np.arange(0, n_surface_fraction)
    nc_nwater_pars[:] =  np.arange(0,7,1)
    nc_water_pars[:] = water_pars
    
    ## if surface height is provided
    if lad_mode == 1 and os.path.exists(sfch_tif):
        
        nc_lad = nc_output.createVariable('lad', np.float32, ('zlad', 'y', 'x'), fill_value=-9999.0, zlib=True)
        nc_lad.long_name = 'leaf_area_density'
        nc_lad.res_orig = np.float32(dx)
        nc_lad.units = 'm2 m-3'
        nc_lad[:] = tree_3d
        
        nc_output.createDimension('nvegetation_pars', 12)
        nc_nveg_pars = nc_output.createVariable('nvegetation_pars', np.int32, 'nvegetation_pars')
        nc_vegetation_pars = nc_output.createVariable('vegetation_pars', np.float32, ('nvegetation_pars', 'y', 'x'), fill_value=-9999.0, zlib=True)
        vegetation_pars = np.zeros((12,vegetation_type.shape[0],vegetation_type.shape[1]))
        lai_sum = tree_3d.copy()
        lai_sum[lai_sum<0] =0
        lai_sum = np.nansum(lai_sum, 0)
        
        lai_sum[lai_sum<0] = -9999.0
        vegetation_pars[1,:,:] = lai_sum
        vegetation_pars[vegetation_pars<0] = -9999.0
        nc_nveg_pars[:] =  np.arange(0,12,1)
        nc_vegetation_pars[:] = vegetation_pars
        
        
    if lad_mode == 2 and os.path.exists(lai_tif):
        nc_lad = nc_output.createVariable('lad', np.float32, ('zlad', 'y', 'x'), fill_value=-9999.0, zlib=True)
        nc_lad.long_name = 'leaf_area_density'
        nc_lad.res_orig = np.float32(dx)
        nc_lad.units = 'm2 m-3'
        # vegetation patch
        lad_patch_arr = lad_patch.filled(0)
        # single tree
        lad_arr = lad_3d.filled(0)
        # remove patch if tree is there
        lad_patch_arr[lad_arr>0] = 0
        # sum
        tree_3d = lad_patch_arr + lad_arr
        tree_3d[np.isnan(tree_3d)] = -9999
        tree_3d[tree_3d<=0] = -9999
        tree_3d[0][vegetation_height==0] = -9999
        nc_lad[:-1,:,:] = tree_3d[:,:,:]
        
        nc_output.createDimension('nvegetation_pars', 12)
        nc_nveg_pars = nc_output.createVariable('nvegetation_pars', np.int32, 'nvegetation_pars')
        nc_vegetation_pars = nc_output.createVariable('vegetation_pars', np.float32, ('nvegetation_pars', 'y', 'x'), fill_value=-9999.0, zlib=True)
        vegetation_pars = np.zeros((12,vegetation_type.shape[0],vegetation_type.shape[1]))

        lai_sum = tree_3d.copy()
        lai_sum[lai_sum<0] =0
        lai_sum = np.nansum(lai_sum, 0)
        
        lai_sum[lai_sum<=0] = -9999.0
        vegetation_pars[:,:,:] = -9999.0
        vegetation_pars[1,:,:] = lai_sum
        vegetation_pars[vegetation_pars<0] = -9999.0
        nc_nveg_pars[:] =  np.arange(0,12,1)
        nc_vegetation_pars[:] = vegetation_pars
        
        nc_bad = nc_output.createVariable('bad', np.float32, ('zlad', 'y', 'x'), fill_value=-9999.0, zlib=True)
        nc_bad.long_name = 'basal_area_density'
        nc_bad.res_orig = np.float32(dx)
        nc_bad.units = 'm2 m-3'
        bad_out = bad_3d.filled(-9999.0)
        nc_bad[:-1,:,:] = bad_out[:,:,:]

        nc_treeid = nc_output.createVariable('tree_id', np.float32, ('zlad', 'y', 'x'), fill_value=-9999.0, zlib=True)
        nc_treeid.long_name = 'tree_id'
        nc_treeid.res_orig = np.float32(dx)
        nc_treeid.units = '1'
        tree_id = tree_ids_3d.filled(-9999.0)
        ## make a new output array
        tree_id_out = np.copy(tree_3d)
        tree_id_out[tree_id_out>=0] = -1
        tree_id_out = tree_id + tree_id_out
        tree_id_out[tree_id_out<=-9999.0] = -9999.0
        nc_treeid[:-1,:,:] = tree_id_out[:,:,:]
        
    if os.path.exists(street_tif):
        nc_street = nc_output.createVariable('street_type', np.byte, ('y', 'x'), fill_value=np.byte(-127), zlib=True)
        nc_street.long_name = 'pavement_type_classification'
        nc_street.res_orig = np.float32(dx)
        nc_street.units = ''
        nc_street[:] = street_type

    if os.path.exists(bldh_tif):
        nc_building = nc_output.createVariable('building_type', np.byte, ('y', 'x'), fill_value=np.byte(-127), zlib=True)

        nc_buildings_2d = nc_output.createVariable('buildings_2d', np.float32, ('y', 'x'), fill_value=-9999.0,zlib=True)

        nc_building_id = nc_output.createVariable('building_id', np.int32, ('y', 'x'), fill_value=int(-9999), zlib=True)

        nc_buildings_3d = nc_output.createVariable('buildings_3d', np.byte, ('z', 'y', 'x'), fill_value=np.byte(-127), zlib=True)
        
        nc_building.long_name = 'building_type_classification'
        nc_building.res_orig = np.float32(dx)
        nc_building.units = ''
        
        nc_buildings_2d.long_name = 'building_height'
        nc_buildings_2d.res_orig = np.float32(dx)
        nc_buildings_2d.lod = np.int32(1)
        nc_buildings_2d.units = 'm'

        nc_building_id.long_name = 'ID of single building'
        nc_building_id.res_orig = np.float32(dx)
        nc_building_id.units = ''

        nc_buildings_3d.long_name = 'builidng structure in 3D'
        nc_buildings_3d.lod = np.int32(2)
        
        nc_building[:] = building_type
        nc_buildings_2d[:] = building_height
        nc_building_id[:] = building_id
        nc_buildings_3d[:] = buildings_3d
        
    nc_output.close()

    print('Process finished! Static driver is created with following specs in centered grids: \n [nx, ny, nz] = ' + str(
        [nx - 1, ny - 1, nz]) + ' \n [dx, dy, dz] = ' + str([dx, dy, dz]))

    # check if is evern or odd for multi processing
    # read the static_topo back
    with xr.open_dataset(static_driver_file) as ncdata:
        print('Checking topo dimensions: ' + str(ncdata.zt.shape))

    return(dom_dict)

