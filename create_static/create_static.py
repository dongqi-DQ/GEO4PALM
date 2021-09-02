#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------#
# Read geotiff files and convert to PALM static file
# 
# @authors: Dongqi Lin, Jiawei Zhang
#--------------------------------------------------------------------------------#

import numpy as np
import xarray as xr
import time
import pandas as pd
import gc
from netCDF4 import Dataset
from pyproj import Proj, transform
import scipy.integrate as integrate
from scipy.ndimage.measurements import label
from static_util.read_geo import readgeotiff
from static_util.nearest import nearest
from static_util.palm_lu import lu2palm, get_albedo
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

def nearest_sst(sst,idx_lat,idx_lon):
    # find nearest available SST if no SST available at centlat, centlon
    lat,lon = np.nonzero(sst)
    min_idx = ((lat - idx_lat)**2 + (lon - idx_lon)**2).argmin()
    return sst[lat[min_idx], lon[min_idx]]

def get_sst(centlat, centlon, origin_time):
    # get nearest sst to be used as water temperatuer in water_pars
    sst_file = "./raw_static/2019_monthly_SST.nc"
    month = origin_time[5:7]
    with xr.open_dataset(sst_file) as ds_sst:
        sst = ds_sst.sel(month=int(month))["sst"]
        sst_lat = ds_sst["latitude"]
        sst_lon = ds_sst["longitude"]
        sst.data[np.isnan(sst.data)] = 0
    _, idx_lat = nearest(sst_lat, centlat)
    _, idx_lon = nearest(sst_lon, centlon)
    
    if sst[idx_lat,idx_lon].data>0:
        water_temperature = sst[idx_lat,idx_lon].data
    else:
        water_temperature = nearest_sst(sst.data, idx_lat, idx_lon)
    
    return water_temperature
    
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

def extract_tiff(array, lat,lon, dom_dict,west_loc,south_loc,north_loc,proj_str,case_name, save_tif=False):
    # west_loc is the local coordinate (m) location of westest of the domain
    # south_loc and north_loc is the local coordinate (m) location of southest/northest
    # proj_str is the projection string
    
    nx = dom_dict['nx']
    ny = dom_dict['ny']
    
     # find the nearest index
    xmin, west_idx  = nearest(lon,west_loc)
    south_idx       = nearest(lat,south_loc)[1]
    ymax            = nearest(lat,north_loc)[0]
    print("south",south_loc,"west", west_loc)
    print("lon_min",lon[0],"lon_max",lon[-1])
    print("lat_min",lat[0],"lat_max",lat[-1])
    array_palm = array[south_idx:south_idx+ny,west_idx:west_idx+nx]
    xsize = lon[-1]-lon[-2]
    ysize = lat[-1]-lat[-2]
    
    if save_tif == True:
        #save tif file of each cropped domain
        array_to_raster(array_palm,xmin,ymax,xsize,ysize,proj_str,"./static_files/"+case_name+"_static.tif")

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

def generate_palm_static(case_name, config_projection,tif_projection, dom_dict, tif_dict):
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
    z = np.arange(dz/2, dz*nz, dz) + z_origin
    tif_left = dom_dict['west']
    tif_right = dom_dict['east']
    tif_north = dom_dict['north']
    tif_south = dom_dict['south']
    
    # assign tiff file names
    dem_tif = tif_dict['dem']
    bldh_tif = tif_dict['bldh']
    bldid_tif = tif_dict['bldid']
    lu_tif = tif_dict['lu']
    sfch_tif = tif_dict['sfch']
    pavement_tif = tif_dict['pavement']
    street_tif = tif_dict['street']

    # Read topography file
    dem, lat, lon = readgeotiff(dem_tif)
    dem[dem < 0] = 0
    print("@@@@dem",dem.shape)


    # extract topography to match PALM domain
    zt = extract_tiff(dem, lat, lon, dom_dict, tif_left,tif_south,tif_north,tif_projection,case_name,save_tif=True)
    zt[zt < .5] = 0
    print("@@@@zt",zt.shape)
    del dem, lat, lon

    print('Number of grid points x, y = ' + str(zt.shape[1]) + ', ' + str(zt.shape[0]))
    
    # find the lowest terrain height
    zt_min = np.nanmin(zt)
    dom_dict['z_origin'] = zt_min
    
    # Read land use files
    n_surface_fraction = 3
    
    print('read LCDB')
    lu_geo, lat, lon = readgeotiff(lu_tif)
    lu = extract_tiff(lu_geo, lat, lon, dom_dict, tif_left,tif_south,tif_north,tif_projection,case_name)
    del lu_geo, lat, lon
    gc.collect()
    
    print('read building ID')
    bldid_geo, lat, lon = readgeotiff(bldid_tif)
    bldid = extract_tiff(bldid_geo, lat, lon, dom_dict, tif_left,tif_south,tif_north,tif_projection,case_name)
    del bldid_geo, lat, lon
    gc.collect()
    
    print('read building height')
    bldh_geo, lat, lon = readgeotiff(bldh_tif)
    bldh = extract_tiff(bldh_geo, lat, lon, dom_dict, tif_left,tif_south,tif_north,tif_projection,case_name)
    del bldh_geo, lat, lon
    gc.collect()
    
    print('read surface height (excluding buildings)')
    sfch_geo, lat, lon = readgeotiff(sfch_tif)
    sfch_tmp = extract_tiff(sfch_geo, lat, lon, dom_dict, tif_left,tif_south,tif_north,tif_projection,case_name)
    del sfch_geo, lat, lon
    gc.collect()
    
    print('read pavement')
    pavement_geo, lat, lon = readgeotiff(pavement_tif)
    pavement = extract_tiff(pavement_geo, lat, lon, dom_dict, tif_left,tif_south,tif_north,tif_projection,case_name)
    del pavement_geo, lat, lon    
    gc.collect()
    
    print('read street')    
    street_geo, lat, lon = readgeotiff(street_tif)
    street = extract_tiff(street_geo, lat, lon, dom_dict, tif_left,tif_south,tif_north,tif_projection,case_name)
    del street_geo, lat, lon    
    gc.collect()
    
    # process water type
    water_lu = lu2palm(lu, 'water')
    water_type =  np.array([[cell if cell>0 else -9999 for cell in row] for row in water_lu])
    
    # process pavement
    pavement_lu = lu2palm(lu, 'pavement')
    pavement_type =  np.array([[cell if cell>0 else -9999 for cell in row] for row in pavement])
    # if pavement tif input is not empty
    if "empty" not in bldh_tif:
        pavement_type[pavement_lu>0] = 3
    else:
        pavement_type[:,:] = -9999
    pavement_type[water_type>0] = -9999
    

    
    # process street
    street_type =  np.array([[cell if cell>0 else -9999 for cell in row] for row in street])
    street_type[pavement_lu>0] = 17
    street_type[water_type>0] = -9999
    
    # process building height
    building_height =  np.array([[cell if cell>0 else -9999 for cell in row] for row in bldh])
    building_height[water_type>0] = -9999
    building_height[pavement_type > 0 ] = -9999
    
    building_id  =  np.array([[cell if cell>0 else -9999 for cell in row] for row in bldid])
    building_id[building_height==-9999] = -9999
    
    building_type =  np.array([[3 if cell>0 else -9999 for cell in row] for row in building_id])
    # if the buildings input is not empty
    if "empty" not in bldh_tif:
        buildings_3d, zbld = make_3d_from_2d(building_height, x, y, dz)
    
    # process vegetation
    vegetation_type = lu2palm(lu, 'vegetation')
    vegetation_type[water_type>0] = -9999
    vegetation_type[pavement_type>0] = -9999
    vegetation_type[building_type>0] = -9999

    # process bare land
    bare_land = lu2palm(lu, 'building')
    bare_land[building_type>0] = -9999 
    bare_land[pavement_type>0] = -9999 
    bare_land[water_type>0] = -9999 
    bare_land[vegetation_type>0] = -9999
    if "empty" not in bldh_tif:
        vegetation_type[bare_land>0] = 1 # bare land 
    else:
        urban_type = 18                   # vegetation type to represent roughness length of urban buildup
        vegetation_type[bare_land>0] = urban_type  
        
    # process soil
    soil_type = np.zeros_like(lu)
    soil_type[:] = -9999
    soil_type = lu2palm(lu, 'soil')
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
                
    # if surface height input is not empty
    # calculate leaf area density (LAD)
    if "empty" not in sfch_tif:
    # process lad
    # remove cars and some other "noise"
        tree_height = np.copy(sfch_tmp)
        tree_height[tree_height<=3] = -9999
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
        tree_lai = 5
        lad_max_height = 0.025
        ml_n_low = 0.5
        ml_n_high = 6.0
        
        print("Calculating LAD")
        for idx in range(0, zt.shape[1]):
            for idy in range(0, zt.shape[0]):
                if tree_height[idy, idx] <= 0:
                    continue
                else:
                    #  Calculate height of maximum LAD
                    z_lad_max = lad_max_height * tree_height[idy, idx]

                    #  Calculate the maximum LAD after Lalic and Mihailovic (2004)
                    lad_max_part_1 = integrate.quad(lambda z: ( ( tree_height[idy, idx]- z_lad_max ) / ( tree_height[idy, idx] - z ) ) ** (ml_n_high) * np.exp( ml_n_high * (1.0 - ( tree_height[idy, idx] - z_lad_max ) / ( tree_height[idy, idx] - z ) ) ), 0.0, z_lad_max)
                    lad_max_part_2 = integrate.quad(lambda z: ( ( tree_height[idy, idx] - z_lad_max ) / ( tree_height[idy, idx] - z ) ) ** (ml_n_low) * np.exp( ml_n_low * (1.0 - ( tree_height[idy, idx] - z_lad_max ) / ( tree_height[idy, idx] - z ) ) ), z_lad_max, tree_height[idy, idx])

                    lad_max = tree_lai / (lad_max_part_1[0] + lad_max_part_2[0])

                    lad_profile     =  np.zeros_like(zlad)
                    for k in range(1,len(lad_profile)-1):
                        if z[k] > 0.0 and z[k] < z_lad_max:
                            n = ml_n_high
                        else:
                            n = ml_n_low
                        lad_profile[k] =  lad_max * ( ( tree_height[idy, idx] - z_lad_max ) / ( tree_height[idy, idx] - z[k] ) ) ** (n) * np.exp( n * (1.0 - ( tree_height[idy, idx] - z_lad_max ) / ( tree_height[idy, idx] - z[k] ) ) )
                    tree_3d[:, idy, idx] = lad_profile[:]
        ####### end of LAD calculation
        # convert nans to -9999
        tree_3d[np.isnan(tree_3d)] = -9999
        tree_3d[tree_3d<=0] = -9999
        tree_3d[0][tree_height==0] = -9999
    
    
    # check overlaps
    for idx in range(0, zt.shape[1]):
        for idy in range(0, zt.shape[0]):
            if vegetation_type[idy, idx] < 0 and  water_type[idy, idx] < 0 and pavement_type[idy, idx]< 0:
                vegetation_type[idy, idx] = 1
                soil_type[idy, idx] = 2
    
    # set up fillvalues
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
    if "empty" not in bldh_tif:
        albedo_building = get_albedo(building_type, "building")
    else:
        bare_land[vegetation_type!=urban_type] = -9999.0
        albedo_building = get_albedo(bare_land, "building")
    
    albedo_vegetation[albedo_building>0] = 0
    albedo_type = albedo_pavement + albedo_vegetation + albedo_building + albedo_water
    albedo_type[albedo_type[:,:]<=0] = -127.0
    
    # set up water temperature
    # other parameters stay the same as default
    water_temperature = get_sst(centlat, centlon, origin_time)
    water_pars = np.zeros((7,water_type.shape[0],water_type.shape[1]))
    water_pars[0,:,:][water_type>0] = water_temperature
    water_pars[water_pars<=0] = -9999.0
    
    
    # process surface fraction
    surface_fraction = np.array([np.zeros_like(lu), np.zeros_like(lu), np.zeros_like(lu)])
    surface_fraction[:] = 0
    surface_fraction[0][vegetation_type > 0] = 1
    surface_fraction[1][pavement_type > 0] = 1
    surface_fraction[2][water_type > 0] = 1
    
    # output netcdf file
    nc_output = xr.Dataset()
    nc_output.attrs['description'] = 'PALM static driver - containing geospatial data from OSM, NZLCDB, DEM, DSM etc.'
    nc_output.attrs['history'] = 'Created at' + time.ctime(time.time())
    nc_output.attrs['source'] = 'multiple source'
    nc_output.attrs['origin_lat'] = np.float32(centlat)
    nc_output.attrs['origin_lon'] = np.float32(centlon)
    nc_output.attrs['origin_z'] = z_origin
    nc_output.attrs['origin_x'] = tif_left
    nc_output.attrs['origin_y'] = tif_south
    nc_output.attrs['rotation_angle'] = np.float(0)
    nc_output.attrs['origin_time'] = origin_time
    nc_output.attrs['Conventions'] = 'CF-1.7'
    if "empty" not in bldh_tif:
         nc_output['z'] = xr.DataArray(zbld.astype(np.float32), dims=['z'], attrs={'units': 'm', 'axis': 'z'})
    if "empty" not in sfch_tif:
        nc_output['zlad'] = xr.DataArray(zlad.astype(np.float32), dims=['zlad'], attrs={'units': 'm', 'axis': 'zlad'})
    nc_output.to_netcdf('static_files/' + case_name + '_static', mode='w', format='NETCDF4')
    
    nc_output['x'] = xr.DataArray(x.astype(np.float32), dims=['x'], attrs={'units': 'm', 'axis': 'x','_FillValue': -9999,
                                                                           'long_name': 'distance to origin in x-direction'})
    nc_output['y'] = xr.DataArray(y.astype(np.float32), dims=['y'], attrs={'units': 'm', 'axis': 'y','_FillValue': -9999,
                                                                           'long_name': 'distance to origin in y-direction'})
    
    nc_output['zt'] = xr.DataArray(zt, dims=['y', 'x'],
                                     attrs={'units': 'm', 'long_name': 'terrain_height'})
    
    for var in nc_output.data_vars:
        encoding = {var: {'dtype': 'float32', '_FillValue': -9999, 'zlib': True}}
        nc_output[var].to_netcdf('./static_files/' + case_name + '_static', encoding=encoding, mode='a')

    
    nc_output = Dataset('./static_files/' + case_name + '_static', "a", format="NETCDF4")
    
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
    
    if "empty" not in sfch_tif:
        nc_lad = nc_output.createVariable('lad', np.float32, ('zlad', 'y', 'x'), fill_value=-9999.0, zlib=True)
        nc_lad.long_name = 'leaf_area_density'
        nc_lad.res_orig = np.float32(dx)
        nc_lad.units = 'm2 m-3'
        nc_lad[:] = tree_3d
        
    if "empty" not in street_tif:
        nc_street = nc_output.createVariable('street_type', np.byte, ('y', 'x'), fill_value=np.byte(-127), zlib=True)
        nc_street.long_name = 'pavement_type_classification'
        nc_street.res_orig = np.float32(dx)
        nc_street.units = ''
        nc_street[:] = street_type

    if "empty" not in bldh_tif:
        nc_building = nc_output.createVariable('building_type', np.byte, ('y', 'x'), fill_value=np.byte(-127), zlib=True)

        nc_buildings_2d = nc_output.createVariable('buildings_2d', np.float32, ('y', 'x'), fill_value=-9999.0,zlib=True)

        nc_building_id = nc_output.createVariable('building_id', np.int32, ('y', 'x'), fill_value=np.int(-9999), zlib=True)

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
    with xr.open_dataset('./static_files/' + case_name + '_static') as ncdata:
        print('Checking topo dimensions: ' + str(ncdata.zt.shape))

    return(dom_dict)

