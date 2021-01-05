#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------#
# Read geotiff files and convert to PALM static file
# Input tiff must be in WGS84 projection 
# @author: Dongqi Lin (dongqi.lin@pg.canterbury.ac.nz)
#--------------------------------------------------------------------------------#

import numpy as np
import xarray as xr
import time
import pandas as pd
import utm
import gc
from netCDF4 import Dataset
from pyproj import Proj
import scipy.integrate as integrate
from static_util.read_geo import readgeotiff
from static_util.interp_array import interp_array_1d, interp_array_2d
from static_util.nearest import nearest
from static_util.palm_lu import lu2palm
  


def extract_tiff(array, lat, lon, dom_dict):
    nx = dom_dict['nx']
    ny = dom_dict['ny']
    west = dom_dict['west']
    east = dom_dict['east']
    north = dom_dict['north']
    south = dom_dict['south']
     # change projection from WGS84 to UTM
    if (-180 <= lon[0] <= 180) or (-90 <= lat[0] <= 90):

        # subset
        north_lat, north_idx = nearest(lat, north)
        south_lat, south_idx = nearest(lat, south)
        east_lon, east_idx = nearest(lon, east)
        west_lon, west_idx = nearest(lon, west)
        array = array[south_idx:north_idx, west_idx:east_idx]
        lat, lon = lat[south_idx:north_idx], lon[west_idx:east_idx]
    
    array_palm = interp_array_2d(array, int(nx), int(ny), 'nearest')
    # check aray size
#    if array_palm.shape[1] - 1 != nx:
#        array_palm = array_palm[:, 0:array_palm.shape[1] - 1]
#    if array_palm.shape[0] - 1 != ny:
#        array_palm = array_palm[0:array_palm.shape[0] - 1, :]
    return array_palm

def make_3d_from_2d(array_2d,x,y,dz):
    # PALM csd function
    k_tmp = np.arange(0,max(array_2d.flatten())+dz*2,dz)
 
    k_tmp[1:] = k_tmp[1:] - dz * 0.5
    array_3d = np.ones((len(k_tmp),len(y),len(x)))
  
    for l in range(0,len(x)):
        for m in range(0,len(y)):
            for k in range(0,len(k_tmp)):
                if k_tmp[k] > array_2d[m,l]:
                    array_3d[k,m,l] = 0

    return array_3d.astype(np.byte), k_tmp


def generate_palm_static(case_name, myProj, dom_dict, tif_dict):
    # read domain info
    centlon = dom_dict['centlon']
    centlat = dom_dict['centlat']
    nx = dom_dict['nx']
    ny = dom_dict['ny']
    nz = dom_dict['nz']
    dx = dom_dict['dx']
    dy = dom_dict['dy']
    dz = dom_dict['dz']
    z_origin = dom_dict['z_origin']
    y = np.arange(dy/2,dy*ny+0.5,dy)
    x = np.arange(dx/2,dx*nx+0.5,dx)
    z = np.arange(dz/2, dz*nz, dz) + z_origin
    west = dom_dict['west']
    east = dom_dict['east']
    north = dom_dict['north']
    south = dom_dict['south']
    utm_midx, utm_midy = myProj(centlon, centlat)
    utm_left, utm_right = utm_midx - nx * dx / 2, utm_midx + nx * dx / 2
    utm_north, utm_south = utm_midy + ny * dy / 2, utm_midy - ny * dy / 2

    # read tiff file names
    
    dem_tif = tif_dict['dem']
    bldh_tif = tif_dict['bldh']
    bldid_tif = tif_dict['bldid']
    lu_tif = tif_dict['lu']
    tree_tif = tif_dict['tree']
    water_tif = tif_dict['water']
    road_tif = tif_dict['road']
    street_tif = tif_dict['street']

    # Read topography file
    dem, lat, lon = readgeotiff(dem_tif)
    dem[dem < 0] = 0

    # extract topography to match PALM domain
    zt = extract_tiff(dem, lat, lon, dom_dict)
    zt[zt < .5] = 0
    del dem, lat, lon


    print('Number of grid points x, y = ' + str(zt.shape[1]) + ', ' + str(zt.shape[0]))
    
    # claculating lat/lon and UTM fields
    x_lon = np.linspace(west,east,nx) 
    y_lat = np.linspace(south,north,ny)

    #  prepare lat/lon output
    lon2d = np.empty((zt.shape[0], zt.shape[1]))
    lat2d = np.empty((zt.shape[0], zt.shape[1]))

    #  prepare UTM output
    n_utm2d = np.empty((zt.shape[0], zt.shape[1]))
    e_utm2d = np.empty((zt.shape[0], zt.shape[1]))


    e_utm = np.linspace(utm_left, utm_right,nx)
    n_utm = np.linspace(utm_south, utm_north,ny)
    for i in range(0,e_utm.shape[0]):
        for j in range(0,n_utm.shape[0]):
            x_lon[i], y_lat[j] = myProj(e_utm[i], n_utm[j], inverse=True)
    
    
    for j in range(zt.shape[0]):
        lon2d[j, :] = x_lon[:]
        e_utm2d[j, :] = e_utm[:]

    for i in range(zt.shape[1]):
        lat2d[:, i] = y_lat[:]
        n_utm2d[:, i] = n_utm[:]
    
    zt_min = np.nanmin(zt)
    dom_dict['z_origin'] = zt_min
    
    # Read land use files
#    vegetation_type_specs = pd.read_csv('vegetation_type_specs.csv', sep=';')
    n_surface_fraction = 3
    
    print('read LCDB')
    lu_raw, lat, lon = readgeotiff(lu_tif)
    lu = extract_tiff(lu_raw, lat, lon, dom_dict)
    del lu_raw, lat, lon
    gc.collect()
    
    print('read building ID')
    bldid_raw, lat, lon = readgeotiff(bldid_tif)
    bldid = extract_tiff(bldid_raw, lat, lon, dom_dict)
    del bldid_raw, lat, lon
    gc.collect()
    
    print('read building height')
    bldh_raw, lat, lon = readgeotiff(bldh_tif)
    bldh = extract_tiff(bldh_raw, lat, lon, dom_dict)
    del bldh_raw, lat, lon
    gc.collect()
    
    print('read tree height')
    tree_raw, lat, lon = readgeotiff(tree_tif)
    tree_tmp = extract_tiff(tree_raw, lat, lon, dom_dict)
    del tree_raw, lat, lon
    gc.collect()
    
    print('read water')
    water_raw, lat, lon = readgeotiff(water_tif)
    water = extract_tiff(water_raw, lat, lon, dom_dict)
    del water_raw, lat, lon
    gc.collect()
    
    print('read pavement')
    road_raw, lat, lon = readgeotiff(road_tif)
    road = extract_tiff(road_raw, lat, lon, dom_dict)
    del road_raw, lat, lon    
    gc.collect()
    
    print('read street')    
    street_raw, lat, lon = readgeotiff(street_tif)
    street = extract_tiff(street_raw, lat, lon, dom_dict)
    del street_raw, lat, lon    
    gc.collect()
    
    # process water type
    water_type = np.zeros_like(water)    
    water_type[water>0] = 2
    water_type[water<0] = -9999
    
    # process pavement
    pavement_lu = lu2palm(lu, 'pavement')
    pavement_type =  np.array([[cell if cell>0 else -9999 for cell in row] for row in road])
    pavement_type[pavement_lu>0] = 3
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
    vegetation_type[bare_land>0] = 1
    
    # porcess soil
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
                
    

    # process lad
    # remove cars and some other "noise"
    tree = np.copy(tree_tmp)
    tree[tree<=3] = 0
    
    for idx in range(1, tree.shape[1]-1):
        for idy in range(1, tree.shape[0]-1):
            if tree_tmp[idy, idx] > 3:
                if tree_tmp[idy+1, idx] <= 3 and tree_tmp[idy-1, idx] <= 3:
                    tree[idy,idx] = 0
                if tree_tmp[idy, idx+1] <= 3 and tree_tmp[idy, idx-1] <= 3:
                    tree[idy,idx] = 0

    tree_3d, zlad = make_3d_from_2d(tree, x, y, dz)
    tree_3d = tree_3d.astype(np.float)
    tree_3d[tree_3d==0] = -9999
    # give some parameters
    tree_lai = 5
    lad_max_height = 0.025
    ml_n_low = 0.5
    ml_n_high = 6.0
    print("Calculating LAD")
    for idx in range(0, tree.shape[1]):
        for idy in range(0, tree.shape[0]):
            if tree[idy, idx] == 0:
                continue
            else:
                #  Calculate height of maximum LAD
                z_lad_max = lad_max_height * tree[idy, idx]
                   
                #  Calculate the maximum LAD after Lalic and Mihailovic (2004)
                lad_max_part_1 = integrate.quad(lambda z: ( ( tree[idy, idx]- z_lad_max ) / ( tree[idy, idx] - z ) ) ** (ml_n_high) * np.exp( ml_n_high * (1.0 - ( tree[idy, idx] - z_lad_max ) / ( tree[idy, idx] - z ) ) ), 0.0, z_lad_max)
                lad_max_part_2 = integrate.quad(lambda z: ( ( tree[idy, idx] - z_lad_max ) / ( tree[idy, idx] - z ) ) ** (ml_n_low) * np.exp( ml_n_low * (1.0 - ( tree[idy, idx] - z_lad_max ) / ( tree[idy, idx] - z ) ) ), z_lad_max, tree[idy, idx])
                   
                lad_max = tree_lai / (lad_max_part_1[0] + lad_max_part_2[0])
                
                lad_profile     =  np.zeros_like(zlad)
                for k in range(1,len(lad_profile)-1):
                    if z[k] > 0.0 and z[k] < z_lad_max:
                        n = ml_n_high
                    else:
                        n = ml_n_low
                    lad_profile[k] =  lad_max * ( ( tree[idy, idx] - z_lad_max ) / ( tree[idy, idx] - z[k] ) ) ** (n) * np.exp( n * (1.0 - ( tree[idy, idx] - z_lad_max ) / ( tree[idy, idx] - z[k] ) ) )
                tree_3d[:, idy, idx] = lad_profile[:]
    tree_3d[np.isnan(tree_3d)] = -9999
    tree_3d[0][tree==0] = -9999
    tree_3d[-1][tree_3d[-1]==0] = -9999
    
    for idx in range(0, tree.shape[1]):
        for idy in range(0, tree.shape[0]):
            if vegetation_type[idy, idx] < 0 and  water_type[idy, idx] < 0 and pavement_type[idy, idx]< 0:
                vegetation_type[idy, idx] = 1
                soil_type[idy, idx] = 2
    
    vegetation_type[vegetation_type == -9999.0] = -127.0
    pavement_type[pavement_type == -9999.0] = -127.0
    street_type[street_type == -9999.0] = -127.0
    building_type[building_type == -9999.0] = -127.0
    water_type[water_type == -9999.0] = -127.0
    # soil_type[building_type > 0] = -9999.0
    soil_type[soil_type == -9999.0] = -127.0
    soil_type[soil_type == 0] = -127.0
    
    
   
    # process surface fraction
    surface_fraction = np.array([np.zeros_like(lu), np.zeros_like(lu), np.zeros_like(lu)])
    surface_fraction[:] = 0
    surface_fraction[0][vegetation_type > 0] = 1
    surface_fraction[1][pavement_type > 0] = 1
    surface_fraction[2][water_type > 0] = 1
    
    nc_output = xr.Dataset()
    nc_output.attrs['description'] = 'PALM static driver - containing geospatial data from OSM, NZLCDB, DEM, DSM etc.'
    nc_output.attrs['history'] = 'Created at' + time.ctime(time.time())
    nc_output.attrs['source'] = 'multiple source'
    nc_output.attrs['origin_lat'] = np.float32((south + north) / 2)
    nc_output.attrs['origin_lon'] = np.float32((east + west) / 2)
    nc_output.attrs['origin_z'] = z_origin
    nc_output.attrs['origin_x'] = utm_left
    nc_output.attrs['origin_y'] = utm_south
    nc_output.attrs['rotation_angle'] = np.float(0)
    nc_output.attrs['origin_time'] = '2019-06-16 00:00:00 +00'
    nc_output['z'] = xr.DataArray(zbld.astype(np.float32), dims=['z'], attrs={'units': 'm', 'axis': 'z'})
    nc_output['zlad'] = xr.DataArray(zlad.astype(np.float32), dims=['zlad'], attrs={'units': 'm', 'axis': 'zlad'})
    nc_output.to_netcdf('static_files/' + case_name + '_static', mode='w', format='NETCDF4')
    
    nc_output['x'] = xr.DataArray(x.astype(np.float32), dims=['x'], attrs={'units': 'm', 'axis': 'x','_FillValue': -9999,
                                                                           'long_name': 'distance to origin in x-direction'})
    nc_output['y'] = xr.DataArray(y.astype(np.float32), dims=['y'], attrs={'units': 'm', 'axis': 'y','_FillValue': -9999,
                                                                           'long_name': 'distance to origin in y-direction'})
    nc_output['lat'] = xr.DataArray(lat2d, dims=['y', 'x'],
                                           attrs={'units': 'degrees_north', 'long_name': 'latitude'})
    nc_output['lon'] = xr.DataArray(lon2d, dims=['y', 'x'],
                                           attrs={'units': 'degrees_east', 'long_name': 'longitude'})
    nc_output['E_UTM'] = xr.DataArray(e_utm2d, dims=['y', 'x'],
                                           attrs={'units': 'm', 'long_name': 'easting',
                                                  'standard_name': 'projection_x_coorindate'})
    nc_output['N_UTM'] = xr.DataArray(n_utm2d, dims=['y', 'x'],
                                     attrs={'units': 'm', 'long_name': 'northing',
                                            'standard_name': 'projection_y_coorindate'})
    nc_output['zt'] = xr.DataArray(zt, dims=['y', 'x'],
                                     attrs={'units': 'm', 'long_name': 'terrain_height'})
    
    for var in nc_output.data_vars:
        encoding = {var: {'dtype': 'float32', '_FillValue': -9999, 'zlib': True}}
        nc_output[var].to_netcdf('static_files/' + case_name + '_static', encoding=encoding, mode='a')

    
    nc_output = Dataset('static_files/' + case_name + '_static', "a", format="NETCDF4")
    
    nc_output.createDimension('nsurface_fraction', n_surface_fraction)
    nc_nsurface_fraction = nc_output.createVariable('nsurface_fraction', np.int32, 'nsurface_fraction')
    
    nc_lad = nc_output.createVariable('lad', np.float32, ('zlad', 'y', 'x'), fill_value=-9999.0, zlib=True)
    nc_vegetation = nc_output.createVariable('vegetation_type', np.byte, ('y', 'x'), fill_value=np.byte(-127), zlib=True)
    nc_pavement = nc_output.createVariable('pavement_type', np.byte, ('y', 'x'), fill_value=np.byte(-127), zlib=True)
    nc_street = nc_output.createVariable('street_type', np.byte, ('y', 'x'), fill_value=np.byte(-127), zlib=True)

    nc_building = nc_output.createVariable('building_type', np.byte, ('y', 'x'), fill_value=np.byte(-127), zlib=True)
    nc_water = nc_output.createVariable('water_type', np.byte, ('y', 'x'), fill_value=np.byte(-127), zlib=True)
    nc_soil = nc_output.createVariable('soil_type', np.byte, ('y', 'x'), fill_value=np.byte(-127), zlib=True)
    nc_surface_fraction = nc_output.createVariable('surface_fraction', np.float32, ('nsurface_fraction', 'y', 'x'),
                                                   fill_value=-9999.0, zlib=True)
    nc_buildings_2d = nc_output.createVariable('buildings_2d', np.float32, ('y', 'x'), fill_value=-9999.0,
                                               zlib=True)
    nc_building_id = nc_output.createVariable('building_id', np.int32, ('y', 'x'), fill_value=np.int(-9999), zlib=True)

    nc_buildings_3d = nc_output.createVariable('buildings_3d', np.byte, ('z', 'y', 'x'), fill_value=np.byte(-127), zlib=True)
    

    
    nc_lad.long_name = 'leaf_area_density'
    nc_lad.res_orig = np.float32(dx)
    nc_lad.units = 'm2 m-3'

    nc_vegetation.long_name = 'vegetation_type_classification'
    nc_vegetation.units = ''

    nc_pavement.long_name = 'pavement_type_classification'
    nc_pavement.res_orig = np.float32(dx)
    nc_pavement.units = ''
    
    nc_street.long_name = 'pavement_type_classification'
    nc_street.res_orig = np.float32(dx)
    nc_street.units = ''


    nc_building.long_name = 'building_type_classification'
    nc_building.res_orig = np.float32(dx)
    nc_building.units = ''

    nc_water.long_name = 'water_type_classification'
    nc_water.res_orig = np.float32(dx)
    nc_water.units = ''

    nc_soil.long_name = 'soil_type_classification'
    nc_soil.res_orig = np.float32(dx)
    nc_soil.units = ''

    nc_surface_fraction.long_name = 'surface_fraction'
    nc_surface_fraction.res_orig = np.float32(dx)
    nc_surface_fraction.units = ''

    nc_buildings_2d.long_name = 'building_height'
    nc_buildings_2d.res_orig = np.float32(dx)
    nc_buildings_2d.lod = np.int32(1)
    nc_buildings_2d.units = 'm'

    nc_building_id.long_name = 'ID of single building'
    nc_building_id.res_orig = np.float32(dx)
    nc_building_id.units = ''
    
    nc_buildings_3d.long_name = 'builidng structure in 3D'
    nc_buildings_3d.lod = np.int32(2)

    

    

    nc_lad[:] = tree_3d
    nc_vegetation[:] = vegetation_type
    nc_pavement[:] = pavement_type
    nc_street[:] = street_type
    nc_building[:] = building_type
    nc_water[:] = water_type
    nc_soil[:] = soil_type
    nc_surface_fraction[:] = surface_fraction
    nc_buildings_2d[:] = building_height
    nc_building_id[:] = building_id
    nc_buildings_3d[:] = buildings_3d
    nc_nsurface_fraction[:] = np.arange(0, n_surface_fraction)
#    topo = nc_output.variables['zt'][:]
#    topo[water_type[:, :] > 0] = 0
#    nc_output.variables['zt'][:] = topo
    
    nc_output.close()





    print('Process finished! Topo is created with following specs in centered grids: \n [nx, ny, nz] = ' + str(
        [nx - 1, ny - 1, nz]) + ' \n [dx, dy, dz] = ' + str([dx, dy, dz]))

    # check if is evern or odd for multi processing
    # read the static_topo back
    ncdata = xr.open_dataset('static_files/' + case_name + '_static')

    print('Checking topo dimensions: ' + str(ncdata.zt.shape))
    return(dom_dict)

