# SmartWaterfalls


    #------------------------------------------------------------------------------#
    #                                                                              #
    # Autors: Michela Negro, GSFC/CRESST/UMBC                                      #
    #         Eric Burns, LSU                                                      #
    #         Nicolò Cibrario, University of Torino                                #
    #         + anyone who will contribute                                         #
    #                                                                              #
    # This program is free software; you can use it and/or modify                  #
    # it under the terms of the GNU General Public License as published by         #
    # the Free Software Foundation; either version 3 of the License, or            #
    # (at your option) any later version.                                          #
    #                                                                              #
    #------------------------------------------------------------------------------#


Required python packages
------------------------

* pytorch
* pycuda


What's in the data files:
-------------------------
The image arrays for ~1400 GRBs. Each GRB has one file. 
The shape of these files are (12, 8, 9376). The first index for 12 gives you the selection from:
['long_hard',
 'long_norm',
 'long_soft',
 'long_blackbody',
 'med_hard',
 'med_norm',
 'med_soft',
 'med_blackbody',
 'short_hard',
 'short_norm',
 'short_soft',
 'short_blackbody']

The values in these arrays are for a scaled value of a loglikelihood ratio which should be between 0 and 1. What sets the 0 scale is a logLR value of 10^-6, which should handle nans, data issues, etc. The max 
value of 1 is set to the highest logLR of that burst. I think this is a reasonable starting point but we may want to consider tweaks there. The structure of the files is described below (slightly modified from 
earlier exchanges)
 
These are the combinations that arise from 4 different spectral templates (hard, norm, soft, blackbody) and the timescales / window considered (long, med, short). The next index of 8 gives you the row 
(timescale considered, e.g. 16.384 or 1.024). The last index steps through the time grid, giving the values used in color plots. The last index goes for 9376. 

The short / med / long steps do not have identical spaces. I have added 0s to make the arrays the same size as the maximal one

