#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#--------------------------------------------------------------------------------#
# Function to find nearest index and value in an array 
# 
#
# @author: ricardofaria
# Modified by Dongqi Lin (dongqi.lin@pg.canterbury.ac.nz)
#--------------------------------------------------------------------------------#


def nearest(array, value):
    
    '''
    
    find nearest index value and index in array.
    
    nearest(array, value) 
    
    return(nearest_value, nearest_index)
    
    '''
    
    import numpy as np
    
    nearest_index = np.where(np.abs(array-value) == np.nanmin(np.abs(array-value)))
    nearest_index = int(nearest_index[0])
    nearest_value= array[nearest_index]
    
    return(nearest_value, nearest_index)
    
