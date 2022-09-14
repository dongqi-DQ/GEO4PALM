#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------#
# Function to convert land cover / land use maps to PALM (v6.0) classification
# Users are required to define the classification conversion in a csv file 
# (see raw_static/nzlcdb_2_PALM_num.csv for more information)
# Conversion depends on the original maps
# @author: Dongqi Lin, Jiawei Zhang
# Acknowledgement: Some parts of the land use classification function 
#                  is based on Ricardo Faria's WRF2PALM tools. 
#--------------------------------------------------------------------------------#
import pandas as pd
import numpy as np

def lu2palm(array, classification):


    '''
    The classification can be:
        'vegetation', 'pavement', 'building', 'water', 'soil', etc.
    '''
    
    tab = pd.read_csv('./util/lu_2_PALM_num.csv')
    array_out = np.zeros_like(array)
    if classification == 'vegetation':
        for l in range(0,tab.shape[0]):
            array_out[array[:,:] == tab.iloc[l,0]] = tab.iloc[l,1]
    elif classification == 'pavement':
        for l in range(0,tab.shape[0]):
            array_out[array[:,:] == tab.iloc[l,0]] = tab.iloc[l,2]
    elif classification == 'building':
        for l in range(0,tab.shape[0]):
            array_out[array[:,:] == tab.iloc[l,0]] = tab.iloc[l,3]
    elif classification == 'water':
        for l in range(0,tab.shape[0]):
            array_out[array[:,:] == tab.iloc[l,0]] = tab.iloc[l,4]
        
    elif classification == 'soil':
        for l in range(0,tab.shape[0]):
            array_out[array[:,:] == tab.iloc[l,0]] = tab.iloc[l,5]
   
    # check NaNs for water
    if classification == 'water':
        array_out[np.isnan(array)] = 2
    else:
        array_out[np.isnan(array_out)] = -9999.0
        array_out[np.isnan(array)] = -9999.0
    # remove zeros
    array_out[array_out[:,:]==0] = -9999.0
    return(array_out)

def get_albedo(array, classification):
    '''
    The classification can be:
        'vegetation', 'pavement', 'building', 'water', 'soil', etc.
    Note that for all building types, albedo type is 33
              for all water types, albedo type is 1
    This may change in future versions of PALM.
    '''
    
    array_out = np.zeros_like(array)
    
    if classification == 'vegetation':
        tab = pd.read_csv('./util/vegetation_to_albedo.csv')
        for l in range(0,tab.shape[0]):
            array_out[array[:,:] == tab.iloc[l,0]] = tab.iloc[l,1]
    elif classification == 'pavement':
        tab = pd.read_csv('./util/pavement_to_albedo.csv')
        for l in range(0,tab.shape[0]):
            array_out[array[:,:] == tab.iloc[l,0]] = tab.iloc[l,1]
    elif classification == 'building':
        array_out[array[:,:] > 0] = 33

    # check NaNs for water
    if classification == 'water':
        array_out[array>0] = 1
    else:
        array_out[np.isnan(array_out)] = -9999.0
        array_out[np.isnan(array)] = -9999.0
    
    # where no types are given, keep albedo type as 0 for combination
    array_out[array_out[:,:]<=0] = 0.0


    return array_out


