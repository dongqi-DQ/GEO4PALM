#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------#
# Script for downloading data from OpenStreetMap using osmnx
# - building data
# - street/pavement data
#
# @author: Dongqi Lin
#--------------------------------------------------------------------------------#
import osmnx as ox
import rasterio
from geocube.api.core import make_geocube
import os 
import sys
import warnings
## supress warnings
## switch to other actions if needed
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')


def get_osm_building(centlat, centlon, area_radius, static_tif_path, case_name, idomain):
    out_file = f"{static_tif_path}{case_name}_osm_building_N{idomain+1:02d}.gpkg"
    if not os.path.exists(out_file):
        print("Retrieving OSM building data...")
        bld = ox.geometries.geometries_from_point((centlat,centlon), tags={"building":True}, dist=area_radius)
        if bld.size==0:
            print(idomain)
            sys.exit("No OSM data retrievied. Please check your configuration.")
        gdf_save = bld.applymap(lambda x: str(x) if isinstance(x, list) else x)
        bld_list = ["geometry", "height", "level"]
        for keys in gdf_save.keys():
            if keys not in bld_list:
                gdf_save = gdf_save.drop(labels=keys, axis=1) 
        gdf_save.to_file(out_file, driver="GPKG")
        print(f"OSM building footprint file downloaded to {static_tif_path}")
    else:
        print("OSM building footprint file exists")

def get_osm_street(centlat, centlon, area_radius, static_tif_path, case_name, idomain):
    out_file = f"{static_tif_path}{case_name}_osm_street_N{idomain+1:02d}.gpkg"
    if not os.path.exists(out_file):
        cf = '["highway"]'
        G = ox.graph_from_point((centlat, centlon), dist=area_radius, dist_type='bbox', network_type="all_private", custom_filter=cf)
        if G.size==0:
            sys.exit("No OSM data retrievied. Please check your configuration.")
        gdf = ox.graph_to_gdfs(G,nodes=False)
        # list of keys to keep
        street_list = ["geometry", "osmid", "highway"]
        for keys in gdf.keys():
            if keys not in street_list:
                gdf = gdf.drop(labels=keys, axis=1)
        for i in range(0,len(gdf["highway"])):
            if type(gdf["highway"].iloc[i]) is list:
                gdf["highway"].iloc[i] = gdf["highway"].iloc[i][0]
            if type(gdf["osmid"].iloc[i]) is list:
                gdf["osmid"].iloc[i] = gdf["osmid"].iloc[i][0]
        gdf.to_file(out_file, driver="GPKG")
        print(f"OSM street file downloaded to {static_tif_path}")
    else:
        print("OSM street file exists")
        