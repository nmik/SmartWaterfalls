# SmartWaterfalls

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

