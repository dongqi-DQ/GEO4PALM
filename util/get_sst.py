#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------#
# Script for downloading SST data from opendap
# Here we use UKMO daily SST data (availability: 2006 to present)
# 
# @author: Dongqi Lin
#--------------------------------------------------------------------------------#
from urllib.request import urlretrieve
from datetime import datetime
import os
import numpy as np
import xarray as xr
from util.nearest import nearest
import warnings
## supress warnings
## switch to other actions if needed
warnings.filterwarnings("ignore")
def download_sst(case_name, origin_time, static_tif_path):
    print("Retrieving SST data from OPeNDAP")
    opendap_url = "https://podaac-opendap.jpl.nasa.gov/opendap/allData/ghrsst/data/L4/GLOB/UKMO/OSTIA"
    year = origin_time[:4]
    date = origin_time[:10].replace("-","")
    # convert date to julian day of the year
    julian_doy = datetime.strptime(origin_time[:10],"%Y-%m-%d").timetuple().tm_yday
    # convert to 3 digit string for downloading
    julian_str = f"{julian_doy:03d}"
    url = f"{opendap_url}/{year}/{julian_str}/{date}-UKMO-L4HRfnd-GLOB-v01-fv02-OSTIA.nc.bz2"
    dst = f"{static_tif_path}{case_name}-SST.nc"
    # check if SST file is there
    if not os.path.exists(dst):
        urlretrieve(url, dst)
        print(f"SST file downloaded to {static_tif_path}")
    else:
        print("SST file exist")

def nearest_sst(sst,idx_lat,idx_lon):
    # find nearest available SST if no SST available at centlat, centlon
    lat,lon = np.nonzero(sst)
    min_idx = ((lat - idx_lat)**2 + (lon - idx_lon)**2).argmin()
    return sst[lat[min_idx], lon[min_idx]]

def get_nearest_sst(centlat, centlon, case_name, static_tif_path):
    # get nearest sst to be used as water temperatuer in water_pars
    sst_file = f"{static_tif_path}{case_name}-SST.nc"
    with xr.open_dataset(sst_file) as ds_sst:
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