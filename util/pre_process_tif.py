#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --------------------------------------------------------------------------------#
# Script to reproject and resample tif files
# @autho: Dongqi Lin, Jiawei Zhang
# --------------------------------------------------------------------------------#
import rioxarray as rxr
from rasterio.crs import CRS
from rasterio.enums import Resampling
from util.loc_dom import convert_wgs_to_utm
import sys
import ast
import configparser
import geopandas as gpd
from geocube.api.core import make_geocube
from shapely.geometry import Polygon
import os
from glob import glob
from rasterio.enums import Resampling
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")



def process_tif(tif_file, tif_type, config_proj, case_name, tmp_path, idomain, dx, method):
    '''
    Function to process tif files
    '''
    ds_geo = rxr.open_rasterio(tif_file)
    crs_output = CRS.from_string(config_proj)
    # Identify whether reprojection is needed
    if ds_geo.rio.crs == crs_output:
        ds_geo_tmp = ds_geo
    else:
        ds_geo_tmp = ds_geo.rio.reproject(crs_output)
    out_file = f"{tmp_path}{case_name}_{tif_type}_N{idomain+1:02d}.tif"
    if not os.path.exists(out_file):
        print(f"Processing {tif_type} tif file for Domain N{idomain+1:02d}")
        ds_geo_out = ds_geo_tmp.rio.reproject(crs_output, dx, resampling=Resampling[method])
        # match projection with DEM
        if tif_type!="DEM":
            ds_dem = rxr.open_rasterio(out_file.replace(tif_type,"DEM"))
            ds_geo_out = ds_geo_out.rio.reproject_match(ds_dem)
        ds_geo_out.rio.to_raster(out_file)
        print(f"{tif_type} tif file processed to {tmp_path}")
    else:
        print(f"{tif_type} tif file for Domain N{idomain+1:02d} exists")
        
def process_osm_building(bld_file, config_proj, case_name, tmp_path, idomain, dx, dy):
    '''
    Function to process OSM building info
    '''
    bldh_out_file = f"{tmp_path}{case_name}_BLDH_N{idomain+1:02d}.tif"
    if not os.path.exists(bldh_out_file):
        print(f"Processing building tif file for Domain N{idomain+1:02d}")
        gpd_file = gpd.read_file(bld_file)
        gpd_file = gpd_file.assign(new_height=gpd_file["osmid"])
        gpd_file["osmid"] = gpd_file["osmid"].astype('float32')
        gpd_file["new_height"] = gpd_file["new_height"].astype('float32')
        if "level" not in gpd_file.keys():
            gpd_file = gpd_file.assign(level=gpd_file["osmid"])
            gpd_file["level"] = np.nan
        if "height" not in gpd_file.keys():
            gpd_file = gpd_file.assign(height=gpd_file["osmid"])
            gpd_file["height"] = np.nan
        ## calculate building height from OSM data
        ## note that if OSM do not have height data, the height will be 0 m
        for i in range(0,len(gpd_file["height"])):
            if type(gpd_file.loc[i,"height"]) is not type(None):
                try:
                    gpd_file.loc[i,"new_height"]  = float(gpd_file.loc[i,"height"])
                except:
                    # in case units are included
                    gpd_file.loc[i,"new_height"]  = float(gpd_file.loc[i,"height"][:-1])
            elif type(gpd_file.loc[i,"level"]) is not type(None):
                gpd_file.loc[i,"new_height"] = float(gpd_file.loc[i,"level"])*3
            else:
                # if no building height is given then set as 3 m
                gpd_file.loc[i,"new_height"] = 3 
        # make building height geocube
        bldh_geogrid = make_geocube(vector_data=gpd_file, measurements=["new_height"], resolution = (dx, dy), output_crs=config_proj)
        # make building ID geocube
        osmid_geogrid = make_geocube(vector_data=gpd_file, measurements=["osmid"], resolution = (dx, dy), output_crs=config_proj)

        # match projection with DEM
        ds_dem = rxr.open_rasterio(bldh_out_file.replace("BLDH","DEM"))
        bldh_geogrid = bldh_geogrid.reindex(y=bldh_geogrid.y[::-1]).rio.reproject_match(ds_dem)
        osmid_geogrid = osmid_geogrid.reindex(y=osmid_geogrid.y[::-1]).rio.reproject_match(ds_dem)
        # save files 
        bldh_geogrid.rio.to_raster(bldh_out_file)
        osmid_geogrid.rio.to_raster(bldh_out_file.replace("BLDH","BLDID"))
        print(f"Building tif files processed to {tmp_path}")
    else:
        print(f"Building tif file for Domain N{idomain+1:02d} exists")
        
def process_osm_pavement_street(osm_file, tif_type, config_proj, case_name, tmp_path, idomain, dx, dy):
    '''
    Function to process OSM street/pavement info
    '''
    osm_out_file = f"{tmp_path}{case_name}_{tif_type}_N{idomain+1:02d}.tif"
    if not os.path.exists(osm_out_file):
        print(f"Processing {tif_type} tif file for Domain N{idomain+1:02d}")
        gdf = gpd.read_file(osm_file)
        # read OSM PALM street/pavement type convertion lookup table
        df_palm = pd.read_csv('./util/OSM2PALM.txt')
        # read PALM classification to dictionarys
        palm_dict = {}
        for i in range(0,df_palm.shape[0]):
            name = df_palm['OSM_road_type'][i]
            palm_dict[name] = (df_palm['buffer_width'][i], df_palm['PALM_pavement_type'][i], df_palm['street_type'][i])
        gdf["buffer"]=0
        gdf['pavement'] = np.nan
        gdf['street'] = np.nan
        for name, (buffer, pavement, street) in palm_dict.items():
            gdf.loc[gdf["highway"].eq(name), "buffer"] = buffer
            gdf.loc[gdf["highway"].eq(name), "pavement"] = pavement
            gdf.loc[gdf["highway"].eq(name), "street"] = street
        # convert to UTM projection
        utm_gdf = gdf.to_crs(config_proj)
        utm_gdf['geometry'] = utm_gdf.apply(lambda x: x.geometry.buffer(x.buffer), axis=1)
        # read DEM tif to match projection
        ds_dem = rxr.open_rasterio(osm_out_file.replace(tif_type,"DEM"))
        osm_grid = make_geocube(vector_data=utm_gdf, measurements=[tif_type], resolution = (dx, dy), output_crs=config_proj)
        osm_grid = osm_grid.reindex(y=osm_grid.y[::-1]).rio.reproject_match(ds_dem)
        osm_grid.rio.to_raster(osm_out_file)
        print(f"{tif_type} tif files processed to {tmp_path}")
    else:
        print(f"{tif_type} tif file for Domain N{idomain+1:02d} exists")
#--------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------#
# main processing 
def process_all(prefix):
    # read namelist
    settings_cfg = configparser.ConfigParser(inline_comment_prefixes='#')
    config = configparser.RawConfigParser()
    namelist =  f"./JOBS/{prefix}/INPUT/namelist.static-{prefix}"
    config.read(namelist)
    ## [case]
    case_name =  ast.literal_eval(config.get("case", "case_name"))[0]
    origin_time = ast.literal_eval(config.get("case", "origin_time"))[0]
    # local projection (unit: m)
    config_proj = ast.literal_eval(config.get("case", "config_proj"))[0]
    # use WGS84 (EPSG:4326) for centlat/centlon
    default_proj = ast.literal_eval(config.get("case", "default_proj"))[0] 

    ## [domain configuration]
    ndomain = ast.literal_eval(config.get("domain", "ndomain"))[0]
    centlat = ast.literal_eval(config.get("domain", "centlat"))[0]
    centlon = ast.literal_eval(config.get("domain", "centlon"))[0]
    dx = ast.literal_eval(config.get("domain", "dx"))
    dy = ast.literal_eval(config.get("domain", "dy"))
    dz = ast.literal_eval(config.get("domain", "dz"))
    nx = ast.literal_eval(config.get("domain", "nx"))
    ny = ast.literal_eval(config.get("domain", "ny"))
    nz = ast.literal_eval(config.get("domain", "nz"))
    z_origin = ast.literal_eval(config.get("domain", "z_origin"))
    ll_x = ast.literal_eval(config.get("domain", "ll_x"))
    ll_y = ast.literal_eval(config.get("domain", "ll_y"))
    ## check if UTM projection is given
    if len(config_proj)==0:
        config_proj_code = convert_wgs_to_utm(centlon, centlat)
        config_proj = f"EPSG:{config_proj_code}"
    ## [required tif files]
    sst = ast.literal_eval(config.get("geotif", "sst"))
    dem = ast.literal_eval(config.get("geotif", "dem"))
    lu = ast.literal_eval(config.get("geotif", "lu"))
    resample_method = ast.literal_eval(config.get("geotif", "resample_method"))

    ## [tif files for urban canopy]
    bldh = ast.literal_eval(config.get("urban", "bldh"))
    bldid = ast.literal_eval(config.get("urban", "bldid"))
    pavement = ast.literal_eval(config.get("urban", "pavement"))
    street = ast.literal_eval(config.get("urban", "street"))

    ## [tif files for plant canopy]
    sfch = ast.literal_eval(config.get("plant", "sfch"))

    # specify the directory of tif files
    # users can provide their own tif files
    # otherwise will download from NASA or OSM
    static_tif_path = f'./JOBS/{case_name}/INPUT/'
    output_path = static_tif_path.replace("INPUT","OUTPUT")
    tmp_path = static_tif_path.replace("INPUT","TMP")
    ## create folders for temporary tif files and final netcdf outputs
    if not os.path.exists(tmp_path):
        print("Create tmp folder")
        os.makedirs(tmp_path)
    if not os.path.exists(output_path):
        print("Create output folder")
        os.makedirs(output_path)
    ## these dictionanries only pass keys 
    tif_geotif_dict = dict(config.items('geotif'))
    tif_urban_dict = dict(config.items('urban'))
    tif_plant_dict = dict(config.items('plant'))
    for i in range(0,ndomain):
        ## DEM
        if dem[i] == "online":
            dem_file = glob(f"{static_tif_path}{case_name}_DEM*/*DEM*.tif")[0]
        else:
            dem_file = dem[i]
        process_tif(dem_file, "DEM", config_proj, case_name, tmp_path, i, dx[i], resample_method[i])
        ## Land Use
        if lu[i] == "online":
            lu_file = glob(f"{static_tif_path}{case_name}_Land_Use*/*LC_Type*.tif")[0]
        else:
            lu_file = lu[i]
        process_tif(lu_file, "LU", config_proj, case_name, tmp_path, i, dx[i], resample_method[i])
        ## water temperature (if provided by user)
        if sst[i] != "online" and sst[i]!="":
            sst_file = sst[i]
            process_tif(sst_file, "SST", config_proj, case_name, tmp_path, i, dx[i], resample_method[i])
        # OSM buildings
        if bldh[i]=="online":
            bld_file = f"{static_tif_path}{case_name}_osm_building_N{i+1:02d}.gpkg"
            process_osm_building(bld_file, config_proj, case_name, tmp_path, i, dx[i], dy[i])
        elif bldh[i]!="online" and bldh[i]!="":
            bld_file = bldh[i]
            process_tif(bld_file, "BLDH", config_proj, case_name, tmp_path, i, dx[i], "nearest")
        # building ID - if not from OSM
        if bldid[i]!="online" and bldid[i]!="":
            bldid_file = bldid[i]
            process_tif(bldid_file, "BLDID", config_proj, case_name, tmp_path, i, dx[i], "nearest")
        # OSM pavement type 
        if pavement[i] == "online":
            pavement_file = f"{static_tif_path}{case_name}_osm_street_N{i+1:02d}.gpkg"
            process_osm_pavement_street(pavement_file, "pavement", config_proj, case_name, tmp_path, i, dx[i], dy[i])
        elif pavement[i]!="online" and pavement[i]!="":
            pavement_file = pavement[i]
            process_tif(pavement_file, "pavement", config_proj, case_name, tmp_path, i, dx[i], "nearest")
        # OSM street type
        if street[i] == "online":
            street_file = f"{static_tif_path}{case_name}_osm_street_N{i+1:02d}.gpkg"
            process_osm_pavement_street(street_file, "street", config_proj, case_name, tmp_path, i, dx[i], dy[i])
        elif street[i]!="online" and street[i]!="":
            street_file = street[i]
            process_tif(street_file, "street", config_proj, case_name, tmp_path, i, dx[i], "nearest")
        # Surface height - for trees
        if sfch[i]!="online" and sfch[i]!="":
            sfch_file = sfch[i]
            process_tif(sfch_file, "SFCH", config_proj, case_name, tmp_path, i, dx[i], "nearest")
            
if __name__ == "__main__":
    process_all(sys.argv[1])
