#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------#
# Convert static driver for flat terrains and precursor runs (pcr)
# How to use:
# python static_to_flat.py [static_file] [nx,ny]
# Example:
# python static_to_flat.py JOBS/case_name/OUTPUT/static_driver_N01 64,196
# Remark:
# - If no nx and ny are given, then no pcr static driver will be created
# - soil and vegetation types can be changed to desired types in tif_to_flat function
# @author: Jiawei Zhang, Dongqi Lin 
#--------------------------------------------------------------------------------#

import numpy as np
import xarray as xr
import sys

def tif_to_flat(infile, soil_type=3, vegetation_type=3, pavement_type=np.nan, water_type=np.nan):
    # the static driver should not include any urban variables, e.g. buildings or streets
    # users need to either create static driver without urban information 
    # or remove the buildings/streets from the static driver first
    ds = xr.open_dataset(infile)
    zero_var_list = ["zt"]
    other_var_dict = {
        "soil_type": soil_type,
        "vegetation_type": vegetation_type,
        "pavement_type": pavement_type,
        "water_type": water_type,
    }
    ds = ds.drop(["albedo_type","water_pars"])

    for var_name in zero_var_list:
        ds[var_name][:] = 0
    for var_key in other_var_dict:
        ds[var_key][:] = other_var_dict[var_key]

    if ~np.isnan(vegetation_type):
        ds["surface_fraction"][0, :, :] = 1
    else:
        ds["surface_fraction"][0, :, :] = 0
    if ~np.isnan(pavement_type):
        ds["surface_fraction"][1, :, :] = 1
    else:
        ds["surface_fraction"][1, :, :] = 0
    if ~np.isnan(water_type):
        ds["surface_fraction"][2, :, :] = 1
    else:
        ds["surface_fraction"][2, :, :] = 0
    ds.to_netcdf(infile + "_flat")
    print("flat static created")


def get_pcr_static(infile, nx, ny):
    # get pcr flat terrain from flat static driver
    ds = xr.open_dataset(infile)
    ds_pcr = ds.isel(x=range(0, nx), y=range(0, ny))
    ds_pcr.to_netcdf(infile + "_pcr")
    print("pcr static created")

infile = sys.argv[1]


tif_to_flat(infile)
try:
    nx, ny = [int(x) for x in sys.argv[2].split(',')]
    get_pcr_static(infile+"_flat", nx, ny)
except:
    print("No precursor run information is given")
