#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------#
# Script for downloading data from ESA WorldCover 
# - Land use data (10 m resolution)
#
# @author: Dongqi Lin
#--------------------------------------------------------------------------------#
from shapely.geometry import Polygon 
from terracatalogueclient import Catalogue 
import getpass
import os
import rioxarray as rxr
from rioxarray import merge
import rasterio 
from glob import glob
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')

def merge_esa_tif(files,static_tif_path):
    elements = []
    for val in files:
        rds = rxr.open_rasterio(val)
        elements.append(rds)
    merged = merge.merge_arrays(elements, nodata=255)
    merged.rio.to_raster(f"{static_tif_path}ESA_WorldCover_merged_10m.tif",windowed=True)


def download_esa_main(static_tif_path, west, east, south, north):
    '''
    west - left bound longitude
    east - right bound longitude
    south - bottom bound longitude
    north - top bound longitude
    '''
    buffer = 0.2
    ### Authenticate to the Terrascope platform (registration required) 
    # create catalogue object and authenticate interactively with a browser 
    user_credential = False
    while not user_credential:
        username = input('Enter ESA Login Username: \n(Register here if you haven\'t, '\
                             'https://esa-worldcover.org/en'\
                             '\nUsername:') 

        password = getpass.getpass(prompt = 'Enter ESA Login Password: ') 


        # authenticate with username and password 
        catalogue = Catalogue().authenticate_non_interactive(username, password) 

        ### Filter catalogue 
        # search for all products in the WorldCover collection 
        # products = catalogue.get_products("urn:eop:VITO:ESA_WorldCover_10m_2020_V1") 

        # or filter to a desired geometry, by providing it as an argument to get_products 
        # xmin, ymin, xmax, ymax
        bounds = ( west-buffer, south-buffer, east+buffer, north+buffer)#(3, 33, 15, 45) 
        geometry = Polygon.from_bounds(*bounds) 
        products = catalogue.get_products("urn:eop:VITO:ESA_WorldCover_10m_2020_V1", geometry=geometry) 

        ### Download 
        # download the products to the given directory 
        directory = f"{static_tif_path}ESA_cache/"
        print("start downloading ESA WorldCover data")
        try:
            catalogue.download_products(products, directory) 
            user_credential = True
        except:
            print("User credential not valid. Please re-enter.")
    print("ESA WorldCover data download done")
    
    ### merge downloaeded data
    files = glob(directory+"ESA*/*Map.tif")
    print("merge ESA data")
    merge_esa_tif(files,static_tif_path)
    print("Done")

def check_esa_download(static_tif_path, west, east, south, north):
    esa_tif_file = f"{static_tif_path}ESA_WorldCover_merged_10m.tif"
    if os.path.exists(esa_tif_file):
        while True:
            user_input = input("ESA tif file exists, continue download? [y/N]")
            if user_input.lower() == "y":
                download_esa_main(static_tif_path, west, east, south, north)
                break
            elif user_input.lower()=="n":
                break
            else:
                print('Please answer y or n')
                continue
    else:
        download_esa_main(static_tif_path, west, east, south, north)
    
if __name__ == "__main__":
    path = input("Please enter download path: ")
    xmin = input("xmin: ")
    ymin = input("ymin: ")
    xmax = input("xmax: ")
    ymax = input("ymax: ")
    check_esa_download(path, xmin, xmax, ymin, ymax)