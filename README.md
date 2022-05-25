# SmartWaterfalls


    #------------------------------------------------------------------------------#
    #                                                                              #
    # Autors: Michela Negro, GSFC/CRESST/UMBC                                      #
    #         Eric Burns, LSU                                                      #
    #         Nicolò Cibrario, University of Torino                                #
    #                                                                              #
    # This program is free software; you can redistribute it and/or modify         #
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
The shape of these files are (12, 8, 9376). 
The first index for 12 gives you the selection from:

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

These are the combinations that arise from 4 different spectral templates (hard, norm, soft, blackbody) and the timescales / window considered (long, med, short). Forgive the naming convention, its the field standard.

