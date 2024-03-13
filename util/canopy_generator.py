#!/usr/bin/env python3
# ------------------------------------------------------------------------------ #
# This file is part of the PALM model system.
#
# PALM is free software: you can redistribute it and/or modify it under the terms
# of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# PALM is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# PALM. If not, see <http://www.gnu.org/licenses/>.
#
# Copyright 1997-2021  Leibniz Universitaet Hannover
# ------------------------------------------------------------------------------ #
#
# Description:
# ------------
# Canopy generator routines for creating 3D leaf and basal area densities for
# single trees and tree patches
#
# @Author Bjoern Maronga (maronga@muk.uni-hannover.de)
# ------------------------------------------------------------------------------ #
import numpy as np
import numpy.ma as ma
import math
import scipy.integrate as integrate
import scipy.ndimage as nd


def generate_single_tree_lad(x, y, dz, max_tree_height, max_patch_height, tree_type, tree_height,
                             tree_dia, trunk_dia, tree_lai, season, lai_tree_lower_threshold,
                             remove_low_lai_tree, tree_shape):
    # Step 1: Create arrays for storing the data
    max_canopy_height = ma.max(ma.append(max_tree_height, max_patch_height))

    zlad = np.arange(0, math.floor(max_canopy_height / dz) * dz + 2 * dz, dz)
    zlad[1:] = zlad[1:] - 0.5 * dz

    lad = ma.ones((len(zlad), len(y), len(x)))
    lad[:, :, :] = ma.masked

    bad = ma.ones((len(zlad), len(y), len(x)))
    bad[:, :, :] = ma.masked

    ids = ma.ones((len(zlad), len(y), len(x)))
    ids[:, :, :] = ma.masked

    types = ma.ones((len(zlad), len(y), len(x)))
    types[:, :, :] = np.byte(-127)

    # Calculating the number of trees in the arrays and a boolean array storing the location of
    # trees which is used for convenience in the following loop environment
    number_of_trees_array = ma.where(
        ~tree_type.mask.flatten() | ~tree_dia.mask.flatten() | ~trunk_dia.mask.flatten(),
        1.0, ma.masked)
    number_of_trees = len(number_of_trees_array[number_of_trees_array == 1.0])
    dx = x[1] - x[0]

    valid_pixels = ma.where(~tree_type.mask | ~tree_dia.mask | ~trunk_dia.mask,
                            True, False)


    # For each tree, create a small 3d array containing the LAD field for the individual tree
    print("Start generating " + str(number_of_trees) + " trees...")
    print('test')
    print(number_of_trees_array)
    tree_id_counter = 0
    if number_of_trees > 0:
        low_lai_count = 0
        mod_count = 0
        for i in range(0, len(x)):
            for j in range(0, len(y)):
                if valid_pixels[j, i]:
                    tree_id_counter = tree_id_counter + 1

                    print("   Processing tree No " +  str(tree_id_counter) + " ...", end="")
                    lad_loc, bad_loc, x_loc, y_loc, z_loc, low_lai_count, mod_count, status = \
                        process_single_tree(i, j, dx, dz,
                                            tree_type[j, i], tree_shape[j, i], tree_height[j, i],
                                            tree_lai[j, i], tree_dia[j, i], trunk_dia[j, i],
                                            season, lai_tree_lower_threshold, remove_low_lai_tree,
                                            low_lai_count, mod_count)

                    if status == 0 and ma.any(~lad_loc.mask):
                        # Calculate the position of the local 3d tree array within the full
                        # domain in order to achieve correct mapping and cutting off at the edges
                        # of the full domain
                        lad_loc_nx = int(len(x_loc) / 2)
                        lad_loc_ny = int(len(y_loc) / 2)
                        lad_loc_nz = int(len(z_loc))

                        odd_x = int(len(x_loc) % 2)
                        odd_y = int(len(y_loc) % 2)

                        ind_l_x = max(0, (i - lad_loc_nx))
                        ind_l_y = max(0, (j - lad_loc_ny))
                        ind_r_x = min(len(x) - 1, i + lad_loc_nx - 1 + odd_x)
                        ind_r_y = min(len(y) - 1, j + lad_loc_ny - 1 + odd_y)

                        out_l_x = ind_l_x - (i - lad_loc_nx)
                        out_l_y = ind_l_y - (j - lad_loc_ny)
                        out_r_x = len(x_loc) - 1 + ind_r_x - (i + lad_loc_nx - 1 + odd_x)
                        out_r_y = len(y_loc) - 1 + ind_r_y - (j + lad_loc_ny - 1 + odd_y)

                        lad[0:lad_loc_nz, ind_l_y:ind_r_y + 1, ind_l_x:ind_r_x + 1] = ma.where(
                            ~lad_loc.mask[0:lad_loc_nz, out_l_y:out_r_y + 1, out_l_x:out_r_x + 1],
                            lad_loc[0:lad_loc_nz, out_l_y:out_r_y + 1, out_l_x:out_r_x + 1],
                            lad[0:lad_loc_nz, ind_l_y:ind_r_y + 1, ind_l_x:ind_r_x + 1])
                        bad[0:lad_loc_nz, ind_l_y:ind_r_y + 1, ind_l_x:ind_r_x + 1] = ma.where(
                            ~bad_loc.mask[0:lad_loc_nz, out_l_y:out_r_y + 1, out_l_x:out_r_x + 1],
                            bad_loc[0:lad_loc_nz, out_l_y:out_r_y + 1, out_l_x:out_r_x + 1],
                            bad[0:lad_loc_nz, ind_l_y:ind_r_y + 1, ind_l_x:ind_r_x + 1])
                        ids[0:lad_loc_nz, ind_l_y:ind_r_y + 1, ind_l_x:ind_r_x + 1] = ma.where(
                            ~lad_loc.mask[0:lad_loc_nz, out_l_y:out_r_y + 1, out_l_x:out_r_x + 1],
                            tree_id_counter,
                            ids[0:lad_loc_nz, ind_l_y:ind_r_y + 1, ind_l_x:ind_r_x + 1])
                        types[0:lad_loc_nz, ind_l_y:ind_r_y + 1, ind_l_x:ind_r_x + 1] = ma.where(
                            ~lad_loc.mask[0:lad_loc_nz, out_l_y:out_r_y + 1, out_l_x:out_r_x + 1],
                            np.byte(tree_type[j, i]),
                            types[0:lad_loc_nz, ind_l_y:ind_r_y + 1, ind_l_x:ind_r_x + 1])

                    #                  if ( status == 0 ):
                    #                     status_char = " ok."
                    #                  else:
                    #                     status_char = " skipped."
                    #                  print(status_char)

                    del lad_loc, x_loc, y_loc, z_loc, status
        if mod_count > 0:
            if remove_low_lai_tree:
                print("Removed", mod_count, "trees due to low LAI.")
            else:
                print("Adjusted LAI of", mod_count, "trees.")
        if low_lai_count > 0:
            print("Warning: Found", low_lai_count, "trees with LAI lower then the",
                  "tree type specific default winter LAI.",
                  "Consider adjusting lai_tree_lower_threshold and remove_low_lai_tree.")
    return lad, bad, ids, types, zlad


def process_single_tree(i, j, dx, dz,
                        tree_type, tree_shape, tree_height, tree_lai, tree_dia, trunk_dia,
                        season, lai_tree_lower_threshold, remove_low_lai_tree,
                        low_lai_counter, mod_counter):

    # Set some parameters
    sphere_extinction = 0.6
    cone_extinction = 0.2
    ml_n_low = 0.5
    ml_n_high = 6.0

    # Populate look up table for tree species and their properties
    # #0 species name
    # #1 Tree shapes were manually lookep up.
    # #2 Crown h/w ratio - missing
    # #3 Crown diameter based on Berlin tree statistics
    # #4 Tree height based on Berlin tree statistics
    # #5 Tree LAI summer - missing
    # #6 Tree LAI winter - missing
    # #7 Height of lad maximum - missing
    # #8 Ratio LAD/BAD - missing
    # #9 Trunk diameter at breast height from Berlin
    default_trees = []
    default_trees.append(Tree("Default",         1.0, 1.0,  4.0, 12.0, 3.0, 0.8, 0.6, 0.025, 0.35))
    default_trees.append(Tree("Abies",           3.0, 1.0,  4.0, 12.0, 3.0, 0.8, 0.6, 0.025, 0.80))
    default_trees.append(Tree("Acer",            1.0, 1.0,  7.0, 12.0, 3.0, 0.8, 0.6, 0.025, 0.80))
    default_trees.append(Tree("Aesculus",        1.0, 1.0,  7.0, 12.0, 3.0, 0.8, 0.6, 0.025, 1.00))
    default_trees.append(Tree("Ailanthus",       1.0, 1.0,  8.5, 13.5, 3.0, 0.8, 0.6, 0.025, 1.30))
    default_trees.append(Tree("Alnus",           3.0, 1.0,  6.0, 16.0, 3.0, 0.8, 0.6, 0.025, 1.20))
    default_trees.append(Tree("Amelanchier",     1.0, 1.0,  3.0,  4.0, 3.0, 0.8, 0.6, 0.025, 1.20))
    default_trees.append(Tree("Betula",          1.0, 1.0,  6.0, 14.0, 3.0, 0.8, 0.6, 0.025, 0.30))
    default_trees.append(Tree("Buxus",           1.0, 1.0,  4.0,  4.0, 3.0, 0.8, 0.6, 0.025, 0.90))
    default_trees.append(Tree("Calocedrus",      3.0, 1.0,  5.0, 10.0, 3.0, 0.8, 0.6, 0.025, 0.50))
    default_trees.append(Tree("Caragana",        1.0, 1.0,  3.5,  6.0, 3.0, 0.8, 0.6, 0.025, 0.90))
    default_trees.append(Tree("Carpinus",        1.0, 1.0,  6.0, 10.0, 3.0, 0.8, 0.6, 0.025, 0.70))
    default_trees.append(Tree("Carya",           1.0, 1.0,  5.0, 17.0, 3.0, 0.8, 0.6, 0.025, 0.80))
    default_trees.append(Tree("Castanea",        1.0, 1.0,  4.5,  7.0, 3.0, 0.8, 0.6, 0.025, 0.80))
    default_trees.append(Tree("Catalpa",         1.0, 1.0,  5.5,  6.5, 3.0, 0.8, 0.6, 0.025, 0.70))
    default_trees.append(Tree("Cedrus",          1.0, 1.0,  8.0, 13.0, 3.0, 0.8, 0.6, 0.025, 0.80))
    default_trees.append(Tree("Celtis",          1.0, 1.0,  6.0,  9.0, 3.0, 0.8, 0.6, 0.025, 0.80))
    default_trees.append(Tree("Cercidiphyllum",  1.0, 1.0,  3.0,  6.5, 3.0, 0.8, 0.6, 0.025, 0.80))
    default_trees.append(Tree("Cercis",          1.0, 1.0,  2.5,  7.5, 3.0, 0.8, 0.6, 0.025, 0.90))
    default_trees.append(Tree("Chamaecyparis",   5.0, 1.0,  3.5,  9.0, 3.0, 0.8, 0.6, 0.025, 0.70))
    default_trees.append(Tree("Cladrastis",      1.0, 1.0,  5.0, 10.0, 3.0, 0.8, 0.6, 0.025, 0.80))
    default_trees.append(Tree("Cornus",          1.0, 1.0,  4.5,  6.5, 3.0, 0.8, 0.6, 0.025, 1.20))
    default_trees.append(Tree("Corylus",         1.0, 1.0,  5.0,  9.0, 3.0, 0.8, 0.6, 0.025, 0.40))
    default_trees.append(Tree("Cotinus",         1.0, 1.0,  4.0,  4.0, 3.0, 0.8, 0.6, 0.025, 0.70))
    default_trees.append(Tree("Crataegus",       3.0, 1.0,  3.5,  6.0, 3.0, 0.8, 0.6, 0.025, 1.40))
    default_trees.append(Tree("Cryptomeria",     3.0, 1.0,  5.0, 10.0, 3.0, 0.8, 0.6, 0.025, 0.50))
    default_trees.append(Tree("Cupressocyparis", 3.0, 1.0,  3.0,  8.0, 3.0, 0.8, 0.6, 0.025, 0.40))
    default_trees.append(Tree("Cupressus",       3.0, 1.0,  5.0,  7.0, 3.0, 0.8, 0.6, 0.025, 0.40))
    default_trees.append(Tree("Cydonia",         1.0, 1.0,  2.0,  3.0, 3.0, 0.8, 0.6, 0.025, 0.90))
    default_trees.append(Tree("Davidia",         1.0, 1.0, 10.0, 14.0, 3.0, 0.8, 0.6, 0.025, 0.40))
    default_trees.append(Tree("Elaeagnus",       1.0, 1.0,  6.5,  6.0, 3.0, 0.8, 0.6, 0.025, 1.20))
    default_trees.append(Tree("Euodia",          1.0, 1.0,  4.5,  6.0, 3.0, 0.8, 0.6, 0.025, 0.90))
    default_trees.append(Tree("Euonymus",        1.0, 1.0,  4.5,  6.0, 3.0, 0.8, 0.6, 0.025, 0.60))
    default_trees.append(Tree("Fagus",           1.0, 1.0, 10.0, 12.5, 3.0, 0.8, 0.6, 0.025, 0.50))
    default_trees.append(Tree("Fraxinus",        1.0, 1.0,  5.5, 10.5, 3.0, 0.8, 0.6, 0.025, 1.60))
    default_trees.append(Tree("Ginkgo",          3.0, 1.0,  4.0,  8.5, 3.0, 0.8, 0.6, 0.025, 0.80))
    default_trees.append(Tree("Gleditsia",       1.0, 1.0,  6.5, 10.5, 3.0, 0.8, 0.6, 0.025, 0.60))
    default_trees.append(Tree("Gymnocladus",     1.0, 1.0,  5.5, 10.0, 3.0, 0.8, 0.6, 0.025, 0.80))
    default_trees.append(Tree("Hippophae",       1.0, 1.0,  9.5,  8.5, 3.0, 0.8, 0.6, 0.025, 0.80))
    default_trees.append(Tree("Ilex",            1.0, 1.0,  4.0,  7.5, 3.0, 0.8, 0.6, 0.025, 0.80))
    default_trees.append(Tree("Juglans",         1.0, 1.0,  7.0,  9.0, 3.0, 0.8, 0.6, 0.025, 0.50))
    default_trees.append(Tree("Juniperus",       5.0, 1.0,  3.0,  7.0, 3.0, 0.8, 0.6, 0.025, 0.90))
    default_trees.append(Tree("Koelreuteria",    1.0, 1.0,  3.5,  5.5, 3.0, 0.8, 0.6, 0.025, 0.50))
    default_trees.append(Tree("Laburnum",        1.0, 1.0,  3.0,  6.0, 3.0, 0.8, 0.6, 0.025, 0.60))
    default_trees.append(Tree("Larix",           3.0, 1.0,  7.0, 16.5, 3.0, 0.8, 0.6, 0.025, 0.60))
    default_trees.append(Tree("Ligustrum",       1.0, 1.0,  3.0,  6.0, 3.0, 0.8, 0.6, 0.025, 1.10))
    default_trees.append(Tree("Liquidambar",     3.0, 1.0,  3.0,  7.0, 3.0, 0.8, 0.6, 0.025, 0.30))
    default_trees.append(Tree("Liriodendron",    3.0, 1.0,  4.5,  9.5, 3.0, 0.8, 0.6, 0.025, 0.50))
    default_trees.append(Tree("Lonicera",        1.0, 1.0,  7.0,  9.0, 3.0, 0.8, 0.6, 0.025, 0.70))
    default_trees.append(Tree("Magnolia",        1.0, 1.0,  3.0,  5.0, 3.0, 0.8, 0.6, 0.025, 0.60))
    default_trees.append(Tree("Malus",           1.0, 1.0,  4.5,  5.0, 3.0, 0.8, 0.6, 0.025, 0.30))
    default_trees.append(Tree("Metasequoia",     5.0, 1.0,  4.5, 12.0, 3.0, 0.8, 0.6, 0.025, 0.50))
    default_trees.append(Tree("Morus",           1.0, 1.0,  7.5, 11.5, 3.0, 0.8, 0.6, 0.025, 1.00))
    default_trees.append(Tree("Ostrya",          1.0, 1.0,  2.0,  6.0, 3.0, 0.8, 0.6, 0.025, 1.00))
    default_trees.append(Tree("Parrotia",        1.0, 1.0,  7.0,  7.0, 3.0, 0.8, 0.6, 0.025, 0.30))
    default_trees.append(Tree("Paulownia",       1.0, 1.0,  4.0,  8.0, 3.0, 0.8, 0.6, 0.025, 0.40))
    default_trees.append(Tree("Phellodendron",   1.0, 1.0, 13.5, 13.5, 3.0, 0.8, 0.6, 0.025, 0.50))
    default_trees.append(Tree("Picea",           3.0, 1.0,  3.0, 13.0, 3.0, 0.8, 0.6, 0.025, 0.90))
    default_trees.append(Tree("Pinus",           3.0, 1.0,  6.0, 16.0, 3.0, 0.8, 0.6, 0.025, 0.80))
    default_trees.append(Tree("Platanus",        1.0, 1.0, 10.0, 14.5, 3.0, 0.8, 0.6, 0.025, 1.10))
    default_trees.append(Tree("Populus",         1.0, 1.0,  9.0, 20.0, 3.0, 0.8, 0.6, 0.025, 1.40))
    default_trees.append(Tree("Prunus",          1.0, 1.0,  5.0,  7.0, 3.0, 0.8, 0.6, 0.025, 1.60))
    default_trees.append(Tree("Pseudotsuga",     3.0, 1.0,  6.0, 17.5, 3.0, 0.8, 0.6, 0.025, 0.70))
    default_trees.append(Tree("Ptelea",          1.0, 1.0,  5.0,  4.0, 3.0, 0.8, 0.6, 0.025, 1.10))
    default_trees.append(Tree("Pterocaria",      1.0, 1.0, 10.0, 12.0, 3.0, 0.8, 0.6, 0.025, 0.50))
    default_trees.append(Tree("Pterocarya",      1.0, 1.0, 11.5, 14.5, 3.0, 0.8, 0.6, 0.025, 1.60))
    default_trees.append(Tree("Pyrus",           3.0, 1.0,  3.0,  6.0, 3.0, 0.8, 0.6, 0.025, 1.80))
    default_trees.append(Tree("Quercus",         1.0, 1.0,  8.0, 14.0, 3.1, 0.1, 0.6, 0.025, 0.40))
    default_trees.append(Tree("Rhamnus",         1.0, 1.0,  4.5,  4.5, 3.0, 0.8, 0.6, 0.025, 1.30))
    default_trees.append(Tree("Rhus",            1.0, 1.0,  7.0,  5.5, 3.0, 0.8, 0.6, 0.025, 0.50))
    default_trees.append(Tree("Robinia",         1.0, 1.0,  4.5, 13.5, 3.0, 0.8, 0.6, 0.025, 0.50))
    default_trees.append(Tree("Salix",           1.0, 1.0,  7.0, 14.0, 3.0, 0.8, 0.6, 0.025, 1.10))
    default_trees.append(Tree("Sambucus",        1.0, 1.0,  8.0,  6.0, 3.0, 0.8, 0.6, 0.025, 1.40))
    default_trees.append(Tree("Sasa",            1.0, 1.0, 10.0, 25.0, 3.0, 0.8, 0.6, 0.025, 0.60))
    default_trees.append(Tree("Sequoiadendron",  5.0, 1.0,  5.5, 10.5, 3.0, 0.8, 0.6, 0.025, 1.60))
    default_trees.append(Tree("Sophora",         1.0, 1.0,  7.5, 10.0, 3.0, 0.8, 0.6, 0.025, 1.40))
    default_trees.append(Tree("Sorbus",          1.0, 1.0,  4.0,  7.0, 3.0, 0.8, 0.6, 0.025, 1.10))
    default_trees.append(Tree("Syringa",         1.0, 1.0,  4.5,  5.0, 3.0, 0.8, 0.6, 0.025, 0.60))
    default_trees.append(Tree("Tamarix",         1.0, 1.0,  6.0,  7.0, 3.0, 0.8, 0.6, 0.025, 0.50))
    default_trees.append(Tree("Taxodium",        5.0, 1.0,  6.0, 16.5, 3.0, 0.8, 0.6, 0.025, 0.60))
    default_trees.append(Tree("Taxus",           2.0, 1.0,  5.0,  7.5, 3.0, 0.8, 0.6, 0.025, 1.50))
    default_trees.append(Tree("Thuja",           3.0, 1.0,  3.5,  9.0, 3.0, 0.8, 0.6, 0.025, 0.70))
    default_trees.append(Tree("Tilia",           3.0, 1.0,  7.0, 12.5, 3.0, 0.8, 0.6, 0.025, 0.70))
    default_trees.append(Tree("Tsuga",           3.0, 1.0,  6.0, 10.5, 3.0, 0.8, 0.6, 0.025, 1.10))
    default_trees.append(Tree("Ulmus",           1.0, 1.0,  7.5, 14.0, 3.0, 0.8, 0.6, 0.025, 0.80))
    default_trees.append(Tree("Zelkova",         1.0, 1.0,  4.0,  5.5, 3.0, 0.8, 0.6, 0.025, 1.20))
    default_trees.append(Tree("Zenobia",         1.0, 1.0,  5.0,  5.0, 3.0, 0.8, 0.6, 0.025, 0.40))

    # Check for missing data in the input and set default values if needed
    if tree_type is ma.masked:
        tree_type = int(0)
    else:
        tree_type = int(tree_type)

    if tree_shape is ma.masked:
        tree_shape = default_trees[tree_type].shape

    if tree_height is ma.masked:
        tree_height = default_trees[tree_type].height

    if tree_lai is ma.masked:
        if season == "summer":
            tree_lai = default_trees[tree_type].lai_summer
        else:
            tree_lai = default_trees[tree_type].lai_winter

    if tree_dia is ma.masked:
        tree_dia = default_trees[tree_type].diameter

    if trunk_dia is ma.masked:
        trunk_dia = default_trees[tree_type].dbh

    # Check tree_lai
    # Tree LAI lower then threshold?
    if tree_lai < lai_tree_lower_threshold:
        # Deal with low lai tree
        mod_counter = mod_counter + 1
        if remove_low_lai_tree:
            # Skip this tree
            print("Removed tree with LAI = ", "%0.3f" % tree_lai, " at (", i, ", ", j, ").", sep="")
            return None, None, None, None, None, low_lai_counter, mod_counter, 1
        else:
            # Use type specific default
            if season == "summer":
                tree_lai = default_trees[tree_type].lai_summer
            else:
                tree_lai = default_trees[tree_type].lai_winter
            print("Adjusted tree to LAI = ", "%0.3f" % tree_lai, " at (", i, ", ", j, ").", sep="")

    # Warn about a tree with lower LAI than we would expect in winter
    if tree_lai < default_trees[tree_type].lai_winter:
        low_lai_counter = low_lai_counter + 1
        print("Found tree with LAI = ", "%0.3f" % tree_lai,
              " (tree type specific default winter LAI of ",
              "%0.2f" % default_trees[tree_type].lai_winter, ")",
              " at (", i, ", ", j, ").", sep="")

    # Assign values that are not defined as user input from lookup table
    tree_ratio = default_trees[tree_type].ratio
    lad_max_height = default_trees[tree_type].lad_max_height
    bad_scale = default_trees[tree_type].bad_scale

    print("Tree input parameters:")
    print("----------------------")
    print("type:           " + str(default_trees[tree_type].species) )
    print("height:         " + str(tree_height))
    print("lai:            " + str(tree_lai))
    print("crown diameter: " + str(tree_dia))
    print("trunk diameter: " + str(trunk_dia))
    print("shape: " + str(tree_shape))
    print("height/width: " + str(tree_ratio))

    # Calculate crown height and height of the crown center
    crown_height = tree_ratio * tree_dia
    if crown_height > tree_height:
        crown_height = tree_height

    crown_center = tree_height - crown_height * 0.5

    # Calculate height of maximum LAD
    z_lad_max = lad_max_height * tree_height

    # Calculate the maximum LAD after Lalic and Mihailovic (2004)
    lad_max_part_1 = integrate.quad(
        lambda z: ((tree_height - z_lad_max) / (tree_height - z))**ml_n_high * np.exp(
            ml_n_high * (1.0 - (tree_height - z_lad_max) / (tree_height - z))), 0.0, z_lad_max)
    lad_max_part_2 = integrate.quad(
        lambda z: ((tree_height - z_lad_max) / (tree_height - z))**ml_n_low * np.exp(
            ml_n_low * (1.0 - (tree_height - z_lad_max) / (tree_height - z))), z_lad_max,
        tree_height)

    lad_max = tree_lai / (lad_max_part_1[0] + lad_max_part_2[0])

    # Define position of tree and its output domain
    nx = int(tree_dia / dx) + 2
    nz = int(tree_height / dz) + 2

    # Add one grid point if diameter is an odd value
    if (tree_dia % 2.0) != 0.0:
        nx = nx + 1

    # Create local domain of the tree's LAD
    x = np.arange(0, nx * dx, dx)
    x[:] = x[:] - 0.5 * dx
    y = x

    z = np.arange(0, nz * dz, dz)
    z[1:] = z[1:] - 0.5 * dz

    # Define center of the tree position inside the local LAD domain
    tree_location_x = x[int(nx / 2)]
    tree_location_y = y[int(nx / 2)]

    # Calculate LAD profile after Lalic and Mihailovic (2004). Will be later used for normalization
    lad_profile = np.arange(0, nz, 1.0)
    lad_profile[:] = 0.0

    for k in range(1, nz - 1):
        if (z[k] > 0.0) & (z[k] < z_lad_max):
            n = ml_n_high
        else:
            n = ml_n_low

        lad_profile[k] = lad_max * ((tree_height - z_lad_max) / (tree_height - z[k]))**n * np.exp(
            n * (1.0 - (tree_height - z_lad_max) / (tree_height - z[k])))

    # Create lad array and populate according to the specific tree shape. This is still
    # experimental
    lad_loc = ma.ones((nz, nx, nx))
    lad_loc[:, :, :] = ma.masked
    bad_loc = ma.copy(lad_loc)

    # For very small trees, no LAD is calculated
    if tree_height <= (0.5 * dz):
        print("    Shallow tree found. Action: ignore.")
        return lad_loc, bad_loc, x, y, z, low_lai_counter, mod_counter, 1

    # Branch for spheres and ellipsoids. A symmetric LAD sphere is created assuming an LAD
    # extinction towards the center of the tree, representing the effect of sunlight extinction
    # and thus less leaf mass inside the tree crown. Extinction coefficients are experimental.
    if tree_shape == 1:
        for i in range(0, nx):
            for j in range(0, nx):
                for k in range(0, nz):
                    r_test = np.sqrt((x[i] - tree_location_x)**2 / (tree_dia * 0.5)**2 + (
                                y[j] - tree_location_y)**2 / (tree_dia * 0.5)**2 + (
                                                 z[k] - crown_center)**2 / (crown_height * 0.5)**(
                                         2))
                    if r_test <= 1.0:
                        lad_loc[k, j, i] = lad_max * np.exp(- sphere_extinction * (1.0 - r_test))
                    else:
                        lad_loc[k, j, i] = ma.masked

                if ma.any(~lad_loc.mask[:, j, i]):
                    lad_loc[0, j, i] = 0.0

    # Branch for cylinder shapes
    if tree_shape == 2:
        k_min = int((crown_center - crown_height * 0.5) / dz)
        k_max = int((crown_center + crown_height * 0.5) / dz)
        for i in range(0, nx):
            for j in range(0, nx):
                for k in range(k_min, k_max):
                    r_test = np.sqrt((x[i] - tree_location_x)**2 / (tree_dia * 0.5)**2 + (
                                y[j] - tree_location_y)**2 / (tree_dia * 0.5)**2)
                    if r_test <= 1.0:
                        r_test3 = np.sqrt((z[k] - crown_center)**2 / (crown_height * 0.5)**2)
                        lad_loc[k, j, i] = lad_max * np.exp(
                            - sphere_extinction * (1.0 - max(r_test, r_test3)))
                    else:
                        lad_loc[k, j, i] = ma.masked

                if ma.any(~lad_loc.mask[:, j, i]):
                    lad_loc[0, j, i] = 0.0

    # Branch for cone shapes
    if tree_shape == 3:
        k_min = int((crown_center - crown_height * 0.5) / dz)
        k_max = int((crown_center + crown_height * 0.5) / dz)
        for i in range(0, nx):
            for j in range(0, nx):
                for k in range(k_min, k_max):
                    k_rel = k - k_min
                    r_test = (x[i] - tree_location_x)**2 + (y[j] - tree_location_y)**2 - (
                            (tree_dia * 0.5)**2 / crown_height**2) * (
                                         z[k_rel] - crown_height)**2
                    if r_test <= 0.0:
                        r_test2 = np.sqrt((x[i] - tree_location_x)**2 / (tree_dia * 0.5)**2 + (
                                    y[j] - tree_location_y)**2 / (tree_dia * 0.5)**2)
                        r_test3 = np.sqrt((z[k] - crown_center)**2 / (crown_height * 0.5)**2)
                        lad_loc[k, j, i] = lad_max * np.exp(
                            - cone_extinction * (1.0 - max((r_test + 1.0), r_test2, r_test3)))
                    else:
                        lad_loc[k, j, i] = ma.masked

                if ma.any(~lad_loc.mask[:, j, i]):
                    lad_loc[0, j, i] = 0.0

    # Branch for inverted cone shapes. TODO: what is r_test2 and r_test3 used for? Debugging needed!
    if tree_shape == 4:
        k_min = int((crown_center - crown_height * 0.5) / dz)
        k_max = int((crown_center + crown_height * 0.5) / dz)
        for i in range(0, nx):
            for j in range(0, nx):
                for k in range(k_min, k_max):
                    k_rel = k_max - k
                    r_test = (x[i] - tree_location_x)**2 + (y[j] - tree_location_y)**2 - (
                            (tree_dia * 0.5)**2 / crown_height**2) * (
                                         z[k_rel] - crown_height)**2
                    if r_test <= 0.0:
                        r_test2 = np.sqrt((x[i] - tree_location_x)**2 / (tree_dia * 0.5)**2 + (
                                    y[j] - tree_location_y)**2 / (tree_dia * 0.5)**2)
                        r_test3 = np.sqrt((z[k] - crown_center)**2 / (crown_height * 0.5)**2)
                        lad_loc[k, j, i] = lad_max * np.exp(- cone_extinction * (- r_test))
                    else:
                        lad_loc[k, j, i] = ma.masked

                if ma.any(~lad_loc.mask[:, j, i]):
                    lad_loc[0, j, i] = 0.0

    # Branch for paraboloid shapes
    if tree_shape == 5:
        k_min = int((crown_center - crown_height * 0.5) / dz)
        k_max = int((crown_center + crown_height * 0.5) / dz)
        for i in range(0, nx):
            for j in range(0, nx):
                for k in range(k_min, k_max):
                    k_rel = k - k_min
                    r_test = ((x[i] - tree_location_x)**2 + (y[j] - tree_location_y)**(
                        2)) * crown_height / (tree_dia * 0.5)**2 - z[k_rel]
                    if r_test <= 0.0:
                        lad_loc[k, j, i] = lad_max * np.exp(- cone_extinction * (- r_test))
                    else:
                        lad_loc[k, j, i] = ma.masked

                if ma.any(~lad_loc.mask[:, j, i]):
                    lad_loc[0, j, i] = 0.0

    # Branch for inverted paraboloid shapes
    if tree_shape == 6:
        k_min = int((crown_center - crown_height * 0.5) / dz)
        k_max = int((crown_center + crown_height * 0.5) / dz)
        for i in range(0, nx):
            for j in range(0, nx):
                for k in range(k_min, k_max):
                    k_rel = k_max - k
                    r_test = ((x[i] - tree_location_x)**2 + (y[j] - tree_location_y)**(
                        2)) * crown_height / (tree_dia * 0.5)**2 - z[k_rel]
                    if r_test <= 0.0:
                        lad_loc[k, j, i] = lad_max * np.exp(- cone_extinction * (- r_test))
                    else:
                        lad_loc[k, j, i] = ma.masked

                if ma.any(~lad_loc.mask[:, j, i]):
                    lad_loc[0, j, i] = 0.0

    # Normalize the LAD profile so that the vertically integrated Lalic and Mihailovic (2004) is
    # reproduced by the LAD array. Deactivated for now.
    # for i in range(0,nx):
    # for j in range(0,nx):
    # lad_clean = np.where(lad_loc[:,j,i] == fillvalues["tree_data"],0.0,lad_loc[:,j,i])
    # lai_from_int = integrate.simps (lad_clean, z)
    # print(lai_from_int)
    # for k in range(0,nz):
    # if ( np.any(lad_loc[k,j,i] > 0.0) ):
    # lad_loc[k,j,i] = np.where(
    #     (lad_loc[k,j,i] != fillvalues["tree_data"]),
    #     lad_loc[k,j,i] / lai_from_int * lad_profile[k],
    #     lad_loc[k,j,i]
    #     )

    # Create BAD array and populate. TODO: revise as low LAD inside the foliage does not result
    # in low BAD values.
    bad_loc = (1.0 - (lad_loc / (ma.max(lad_loc) + 0.01))) * 0.1

    # Overwrite grid cells that are occupied by the tree trunk
    radius = trunk_dia * 0.5
    for i in range(0, nx):
        for j in range(0, nx):
            for k in range(0, nz):
                if z[k] <= crown_center:
                    r_test = np.sqrt((x[i] - tree_location_x)**2 + (y[j] - tree_location_y)**2)
                    if r_test == 0.0:
                        if trunk_dia <= dx:
                            bad_loc[k, j, i] = radius**2 * 3.14159265359
                        else:
                            # WORKAROUND: divide remaining circle area over the 8 surrounding
                            # valid_pixels
                            bad_loc[k, j - 1:j + 2, i - 1:i + 2] = radius**2 * 3.14159265359 / 8.0
                            # for the central pixel fill the pixel
                            bad_loc[k, j, i] = dx**2
                    # elif ( r_test <= radius ):
                    # TODO: calculate circle segment of grid points cut by the grid

    return lad_loc, bad_loc, x, y, z, low_lai_counter, mod_counter, 0


def process_patch(dz, patch_height, patch_type_2d, vegetation_type, max_height_lad, patch_lai, alpha, beta):

    phdz = patch_height[:, :] / dz
    pch_index = ma.where(patch_height.mask, int(-1), phdz.astype(int) + 1)
    ma.masked_equal(pch_index, 0, copy=False)
    pch_index = ma.where(pch_index == -1, 0, pch_index)

    max_canopy_height = max(ma.max(patch_height), max_height_lad)

    z = np.arange(0, math.floor(max_canopy_height / dz) * dz + 2 * dz, dz)

    z[1:] = z[1:] - 0.5 * dz

    nz = len(z)
    ny = len(patch_height[:, 0])
    nx = len(patch_height[0, :])

    pre_lad = ma.zeros(nz)
    lad_loc = ma.empty((nz, ny, nx))
    lad_loc.mask = True

    for i in range(0, nx):
        for j in range(0, ny):
            int_bpdf = 0.0
            if patch_height[j, i] >= (0.5 * dz):
                for k in range(1, pch_index[j, i]):
                    int_bpdf = int_bpdf + (((z[k] / patch_height[j, i])**(alpha - 1)) * (
                                (1.0 - (z[k] / patch_height[j, i]))**(beta - 1)) * (
                                                       dz / patch_height[j, i]))

                for k in range(1, pch_index[j, i]):
                    pre_lad[k] = patch_lai[j, i] * (
                                ((dz * k / patch_height[j, i])**(alpha - 1.0)) * (
                                    (1.0 - (dz * k / patch_height[j, i]))**(
                                        beta - 1.0)) / int_bpdf) / patch_height[j, i]

                lad_loc[0, j, i] = pre_lad[0]

                for k in range(0, pch_index[j, i]):
                    lad_loc[k, j, i] = 0.5 * (pre_lad[k - 1] + pre_lad[k])

    patch_id_2d = ma.where(lad_loc.mask[0, :, :], 0, 1)
    patch_id_3d = ma.where(lad_loc.mask, 0, 1)
    patch_type_3d = ma.empty((nz, ny, nx))

    for k in range(0, nz):
        patch_id_3d[k, :, :] = ma.where((patch_id_2d != 0) & ~lad_loc.mask[k, :, :],
                                     patch_id_2d, ma.masked)
        patch_type_3d[k, :, :] = ma.where((patch_id_2d != 0) & ~lad_loc.mask[k, :, :],
                                       patch_type_2d, ma.masked)

    return lad_loc, patch_id_3d, patch_type_3d, nz, 0


# CLASS TREE
#
# Default tree geometrical parameters:
#
# species: name of the tree type
#
# shape: defines the general shape of the tree and can be one of the following types:
# 1.0 sphere or ellipsoid
# 2.0 cylinder
# 3.0 cone
# 4.0 inverted cone
# 5.0 paraboloid (rounded cone)
# 6.0 inverted paraboloid (invertes rounded cone)
#
# ratio:  ratio of maximum crown height to the maximum crown diameter
# diameter: default crown diameter (m)
# height:   default total height of the tree including trunk (m)
# lai_summer: default leaf area index fully leafed
# lai_winter: default winter-teim leaf area index
# lad_max: default maximum leaf area density (m2/m3)
# lad_max_height: default height where the leaf area density is maximum relative to total tree
#                 height
# bad_scale: ratio of basal area in the crown area to the leaf area
# dbh: default trunk diameter at breast height (1.4 m) (m)
#
class Tree:
    def __init__(self, species, shape, ratio, diameter, height, lai_summer, lai_winter,
                 lad_max_height, bad_scale, dbh):
        self.species = species
        self.shape = shape
        self.ratio = ratio
        self.diameter = diameter
        self.height = height
        self.lai_summer = lai_summer
        self.lai_winter = lai_winter
        self.lad_max_height = lad_max_height
        self.bad_scale = bad_scale
        self.dbh = dbh
