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
    
    Parameters
    ----------
    file : str
        npy file for a GRB.
    
    Returns
    -------
    array 
       set of 12 waterfall images: (12, N, M)
    """
    
    f = np.load(grb_file)
    
    return f


def load_grb_images(file_folder_path):
    """
    Parsing of the GRB*.npy files.
    
    
    Parameters
    ----------
    file_folder_path : str
        file path to GRB*.npy files.
        
    Returns
    -------
    array
        set of images.
    """
    
    grb_images_ = []
    for filename in os.listdir(file_folder_path):
        try:
            f = parse_grb_files(file_folder_path+filename)
            grb_images_.append(f)
        except:
            pass
    return grb_images_


def resize_image(image):
    """
    

    Parameters
    ----------
    image : (N, M) array 
        one waterfall image.

    Returns
    -------
    new_image : (N', M') array
        reshaped waterfall image.

    """
    
    print('Old size:', image.shape)
    new_image = image[:, ::10]
    print('New size:', new_image.shape)
    
    return new_image


def resize_images(grb):
    """
    Function to resize the input images.

    Parameters
    ----------
    grb : array
        set of 12 NxM impages.

    Returns
    -------
    array
        set of new reshaped N'xM' images.

    """
    
    new_grb = []
    for i, image in enumerate(grb):
        new_image = resize_image(image)
        new_grb.append(new_image)
        
    return np.array(new_grb)


def visualize(npy_file, tobeopen=True):
    """
    Plot a given GRB*.npy file.
    
    Parameters
    ----------
    npy_file : str
          .npy files containing the 12 imaages of the GRB waterfalls.
    tobeopen: bool
          if False, assume that the npy file is already open (the input is the 
                 set of images)
          
    """
    
    titles = 	['long_hard', 'long_norm', 'long_soft', 'long_blackbody',
                 'med_hard', 'med_norm', 'med_soft', 'med_blackbody',
                 'short_hard', 'short_norm', 'short_soft', 'short_blackbody']
    
    if tobeopen == True:
        grb = parse_grb_files(npy_file)
    else:
        grb = npy_file
    print('File shape:', grb.shape)

    for i, image in enumerate(grb):
        plt.figure()
        plt.title(titles[i], size=16)
        plt.imshow(image, origin='upper', aspect='auto')
        plt.colorbar()
        plt.xticks(color='w')
        plt.yticks(color='w')
        plt.tick_params(bottom = False)
        plt.tick_params(left = False)
        
    plt.show()

	

	

if __name__ == '__main__': 
    """ test module
    """
    fpath = 'testdata/'
    grbs_ = load_grb_images(fpath)
    print(len(grbs_), grbs_[0].shape)
	
