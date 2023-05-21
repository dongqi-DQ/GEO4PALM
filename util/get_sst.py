#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------#
# Script for downloading SST data from opendap
# Here we use UKMO daily SST data (availability: 2006 to present)
# 
# @author: Dongqi Lin
#--------------------------------------------------------------------------------#
import urllib
from urllib import request, parse
from urllib.request import urlretrieve
from http.cookiejar import CookieJar
import json
import getpass
import netrc
from datetime import datetime
import os
import numpy as np
import xarray as xr
from util.nearest import nearest
from netCDF4 import Dataset
import dask
import requests

#Allows us to visualize the dask progress for parallel operations
from dask.diagnostics import ProgressBar
ProgressBar().register()

import warnings
## supress warnings
## switch to other actions if needed
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')




def setup_earthdata_login_auth(endpoint):
    """
    Set up the request library so that it authenticates against the given Earthdata Login
    endpoint and is able to track cookies between requests.  This looks in the .netrc file
    first and if no credentials are found, it prompts for them.
    Valid endpoints include:
        urs.earthdata.nasa.gov - Earthdata Login production
    """
    try:
        username, _, password = netrc.netrc().authenticators(endpoint)
    except (FileNotFoundError, TypeError):
        # FileNotFound = There's no .netrc file
        # TypeError = The endpoint isn't in the netrc file, causing the above to try unpacking None
        print('Please provide your Earthdata Login credentials to allow data access')
        print('Your credentials will only be passed to %s and will not be exposed in GEO4PALM' % (endpoint))
        username = input('Username:')
        password = getpass.getpass()

    
    manager = request.HTTPPasswordMgrWithDefaultRealm()
    manager.add_password(None, endpoint, username, password)
    auth = request.HTTPBasicAuthHandler(manager)

    jar = CookieJar()
    processor = request.HTTPCookieProcessor(jar)
    opener = request.build_opener(auth, processor)
    request.install_opener(opener)


def download_sst(case_name, origin_time, static_tif_path):
    ## a new function to download SST
    ## further optimisation may be needed
    print("Retrieving SST data from OPeNDAP")
    dst = f"{static_tif_path}{case_name}-SST.nc"

    ## check if SST file is there
    if not os.path.exists(dst):
        print(f"Downloading SST file to {static_tif_path}...")
         ## get identification info
        edl="urs.earthdata.nasa.gov"

        setup_earthdata_login_auth(edl)
        #CMR Link to use
        #https://cmr.earthdata.nasa.gov/search/granules.umm_json?collection_concept_id=C1625128926-GHRC_CLOUD&temporal=2019-01-01T10:00:00Z,2019-12-31T23:59:59Z
        sst_dt = origin_time
        r = requests.get(f'https://cmr.earthdata.nasa.gov/search/granules.umm_json?collection_concept_id=C1996881146-POCLOUD&temporal={sst_dt[:10]}T{sst_dt[11:19]}Z,{sst_dt[:10]}T{sst_dt[11:19]}Z&pageSize=365')
        response_body = r.json()

        for itm in response_body['items']:
            for urls in itm['umm']['RelatedUrls']:
                if urls['URL'].endswith(".nc") and urls['URL'].startswith("https"):
                    data_url = urls['URL']
        urlretrieve(data_url, dst)
        print(f"SST file downloaded to {static_tif_path}")
    else:
         while True:
            user_input = input("SST data exists, do you wish to continue download? [y/N]")
            if user_input.lower() == "y":
                print(f"Downloading SST file to {static_tif_path}...")
                         ## get identification info
                edl="urs.earthdata.nasa.gov"

                setup_earthdata_login_auth(edl)
                #CMR Link to use
                #https://cmr.earthdata.nasa.gov/search/granules.umm_json?collection_concept_id=C1625128926-GHRC_CLOUD&temporal=2019-01-01T10:00:00Z,2019-12-31T23:59:59Z
                sst_dt = origin_time
                r = requests.get(f'https://cmr.earthdata.nasa.gov/search/granules.umm_json?collection_concept_id=C1996881146-POCLOUD&temporal={sst_dt[:10]}T{sst_dt[11:19]}Z,{sst_dt[:10]}T{sst_dt[11:19]}Z&pageSize=365')
                response_body = r.json()

                for itm in response_body['items']:
                    for urls in itm['umm']['RelatedUrls']:
                        if urls['URL'].endswith(".nc") and urls['URL'].startswith("https"):
                            data_url = urls['URL']
                urlretrieve(data_url, dst)
                print(f"SST file downloaded to {static_tif_path}")
                break
            elif user_input.lower().lower() == "n":
                break
            else:
                print('Please answer y or n')
                continue

def nearest_sst(sst,idx_lat,idx_lon):
    # find nearest available SST if no SST available at centlat, centlon
    lat,lon = np.nonzero(sst)
    min_idx = ((lat - idx_lat)**2 + (lon - idx_lon)**2).argmin()
    return sst[lat[min_idx], lon[min_idx]]

def get_nearest_sst(centlat, centlon, case_name, static_tif_path):
    # get nearest sst to be used as water temperatuer in water_pars
    sst_file = f"{static_tif_path}{case_name}-SST.nc"
    with xr.open_dataset(sst_file, engine="netcdf4") as ds_sst:
            sst = ds_sst["analysed_sst"].isel(time=0)
            sst_lat = ds_sst["lat"]
            sst_lon = ds_sst["lon"]
            sst.data[np.isnan(sst.data)] = 0
            _, idx_lat = nearest(sst_lat, centlat)
            _, idx_lon = nearest(sst_lon, centlon)
    if sst[idx_lat,idx_lon].data>0:
        water_temperature = sst[idx_lat,idx_lon].data
    else:
        water_temperature = nearest_sst(sst.data, idx_lat, idx_lon)
    
    return water_temperature

########
## Disabled function due to retirement of PO.DAAC OPeNDAP
# def download_sst(case_name, origin_time, static_tif_path):
#     print("Retrieving SST data from OPeNDAP")
#     opendap_url = "https://podaac-opendap.jpl.nasa.gov/opendap/allData/ghrsst/data/L4/GLOB/UKMO/OSTIA"
#     year = origin_time[:4]
#     date = origin_time[:10].replace("-","")
#     # convert date to julian day of the year
#     julian_doy = datetime.strptime(origin_time[:10],"%Y-%m-%d").timetuple().tm_yday
#     # convert to 3 digit string for downloading
#     julian_str = f"{julian_doy:03d}"
#     url = f"{opendap_url}/{year}/{julian_str}/{date}-UKMO-L4HRfnd-GLOB-v01-fv02-OSTIA.nc.bz2"
#     dst = f"{static_tif_path}{case_name}-SST.nc"
#     # check if SST file is there
#     if not os.path.exists(dst):
#         urlretrieve(url, dst)
#         print(f"SST file downloaded to {static_tif_path}")
#     else:
#         print("SST file exists")
########