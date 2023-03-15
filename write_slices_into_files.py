# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 21:06:29 2023

@author: giriprasad.1
"""

import h5py

FILES = ['../../../Downloads/Microstructures_all_features/Microstructure_00_new_all_features.dream3d',
         '../../../Downloads/Microstructures_all_features/Microstructure_01_new_all_features.dream3d',
         '../../../Downloads/Microstructures_all_features/Microstructure_02_new_all_features.dream3d',
         '../../../Downloads/Microstructures_all_features/Microstructure_03_new_all_features.dream3d',
         '../../../Downloads/Microstructures_all_features/Microstructure_04_new_all_features.dream3d',
         '../../../Downloads/Microstructures_all_features/Microstructure_05_new_all_features.dream3d',
         '../../../Downloads/Microstructures_all_features/Microstructure_06_new_all_features.dream3d',
         '../../../Downloads/Microstructures_all_features/Microstructure_07_new_all_features.dream3d',
         '../../../Downloads/Microstructures_all_features/Microstructure_08_new_all_features.dream3d',
         '../../../Downloads/Microstructures_all_features/Microstructure_09_new_all_features.dream3d']


for i,file in enumerate(FILES):
    
    for slice_index in range(1,4):

        dest_file = f'../../../Downloads/Microstructures_all_features/Microstructure_0{i}_new_all_features_slice_{slice_index}.dream3d'
        
        if slice_index == 1:
            
            slice_string = ''
            
        else:
            
            slice_string = '_' + str(slice_index)
        
        with h5py.File(dest_file,'w') as f_dest:
            
            with h5py.File(file,'r+') as f_src:
                
                # create new files based on slice number 
                f_dest.create_group("DataContainers/")
        
                f_src.copy(f_src[f"DataContainers/SliceDataContainer{slice_string}/"],f_dest["DataContainers/"])
                
                # rename
                if int(slice_index) > 1:
                    
                    f_dest["DataContainers/SliceDataContainer"] = f_dest[f"DataContainers/SliceDataContainer{slice_string}/"]
                    
                    del f_dest[f"DataContainers/SliceDataContainer{slice_string}/"]
