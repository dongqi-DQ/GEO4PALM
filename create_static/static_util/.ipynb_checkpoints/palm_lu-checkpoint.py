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


def lu2palm(array, classification):


    '''
    The classification can be:
        'vegetation', 'pavement', 'building', 'water', 'soil', etc.
    '''
    
    import pandas as pd
    import numpy as np
    
    tab = pd.read_csv('raw_static/nzlcdb_2_PALM_num.csv')
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


    if classification == 'water':
        array_out[np.isnan(array)] = 2
    else:
        array_out[np.isnan(array_out)] = -9999.0
        array_out[np.isnan(array)] = -9999.0
    
    array_out[array_out[:,:]==0] = -9999.0
    return(array_out)

