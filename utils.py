#!/usr/bin/env python                                                          #
#                                                                              #
# Autor: Michela Negro, GSFC/CRESST/UMBC                                       #
#        Niccolo' Cibrario, Torino University                                  #
# This program is free software; you can redistribute it and/or modify         #
# it under the terms of the GNU General Public License as published by         #
# the Free Software Foundation; either version 3 of the License, or            #
# (at your option) any later version.                                          #
#                                                                              #
#------------------------------------------------------------------------------#


"""general parsing functions.
"""

import os
import numpy as np
from matplotlib import pyplot as plt

def parse_grb_files(grb_file):
    """
    Parsing of the GRB*.npy files.

    file : str
        npy file for a GRB.
    """
    
    f = np.load(grb_file)
    
    return f


def load_grb_images(file_folder_path):
    """
    Parsing of the GRB*.npy files.

    file_folder_path : str
        file path to GRB*.npy files.
    """
    
    grb_images_ = []
    for filename in os.listdir(file_folder_path):
        try:
            f = parse_grb_files(file_folder_path+filename)
            grb_images_.append(f)
        except:
            pass
    return grb_images_

	

if __name__ == '__main__': 
    """ test module
    """
    fpath = 'testdata/'
    grbs_ = load_grb_images(fpath)
    print(len(grbs_), grbs_[0].shape)
	
