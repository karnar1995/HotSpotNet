# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 08:41:18 2023

@author: giriprasad.1
"""


import pandas as pd
import numpy as np
import random
from torch.autograd import Variable 
import torch
from utils.utils import normalize, normalize_all, mhi, animation
import h5py
import cv2
import os
import matplotlib.pyplot as plt

PATCH_SIZE = 64

BATCH_SIZE = 100
TARGET_ROW = 64
TARGET_COL = 64
TIMESTEP = 10

"""
# create a dataset with actual and
# predicted values
d = {'Actual': np.arange(0, 20, 2)*np.sin(2),
     'Predicted': np.arange(0, 20, 2)*np.cos(2)}
  
# convert the data to pandas dataframe
data = pd.DataFrame(data=d)
  
# create a weights array based on 
# the importance
y_weights = np.arange(2, 4, 0.2)
y_weights = np.array([100]*10) 

# calculate the squared difference
diff = (data['Actual']-data['Predicted'])**2
  
# compute the weighted mean square error
weighted_mean_sq_error = np.sum(diff * y_weights) / np.sum(y_weights)

print(weighted_mean_sq_error)
print(np.square(np.subtract(data['Actual'],data['Predicted'])).mean())


print(np.ones(shape=(100,32,32)))
"""

"""
TRAIN_FILES = ['../../../Downloads/Initial Microstructure/Threshold 0.5%/PolycrystalSynthetic_all_features.dream3d',
         '../../../Downloads/New Microstructure Threshold 0.5%/PolycrystalSynthetic_size_01_slab_20_all_features.dream3d',
         '../../../Downloads/New Microstructure Threshold 0.5%/PolycrystalSynthetic_size_01_slab_20_rolled_all_features.dream3d',
         '../../../Downloads/New Microstructure Threshold 0.5%/PolycrystalSynthetic_size_01_slab_20_rolled_texture_01_all_features.dream3d',
         '../../../Downloads/New Microstructure Threshold 0.5%/Microstructure_01_all_features.dream3d',
         '../../../Downloads/New Microstructure Threshold 0.5%/Microstructure_02_all_features.dream3d',
         '../../../Downloads/New Microstructure Threshold 0.5%/Microstructure_03_all_features.dream3d',
         '../../../Downloads/New Microstructure Threshold 0.5%/Microstructure_04_all_features.dream3d',
         '../../../Downloads/New Microstructure Threshold 0.5%/Microstructure_05_all_features.dream3d',
         '../../../Downloads/New Microstructure Threshold 0.5%/Microstructure_06_all_features.dream3d',
         '../../../Downloads/New Microstructure Threshold 0.5%/Microstructure_07_all_features.dream3d']

for file in TRAIN_FILES:
    with h5py.File(file,'r+') as f:
        
        
        euler_angles = f[f'DataContainers/SliceDataContainer/CellData/EulerAngles'][:]
        euler_angles = np.reshape(euler_angles, (1024, 1024, 3)).astype(np.float32)
        #euler_angles = normalize(euler_angles)
        
        
        von_mises_actual = f[f'DataContainers/SliceDataContainer/CellData/Von_Miss_{TIMESTEP}'][:]
        von_mises_actual = np.reshape(von_mises_actual, (1024, 1024, 1)).astype(np.float32)
        von_mises_actual = normalize(von_mises_actual)
        
        
        
        print(np.max(euler_angles))
        print(np.max(von_mises_actual))
        
        plt.figure()
        plt.imshow(von_mises_actual)
        plt.title('Original stress field')
        
        
        histogram, bin_edges = np.histogram(euler_angles[:,:,2], bins=256)
        
        plt.figure()
        plt.title("Euler Angles Histogram")
        plt.xlabel("Euler Angle Value")
        plt.ylabel("Count")
        
        plt.plot(bin_edges[0:-1], histogram) 
        
        histogram, bin_edges = np.histogram(von_mises_actual, bins=256)
        
        plt.figure()
        plt.title("Stress Histogram")
        plt.xlabel("Stress Value")
        plt.ylabel("Count")
        # <- named arguments do not work here

        plt.plot(bin_edges[0:-1], histogram) 
"""


"""
TRAIN_FILES = []

PATCH_SIZE = 64


for i in range(10):
    for slice_index in range(1,4):
        
            TRAIN_FILES.append(f'../../../Downloads/Microstructures_all_features/Microstructure_0{i}_new_all_features_slice_{slice_index}.dream3d')
            



for file in TRAIN_FILES:
    with h5py.File(file,'r+') as f:
        
        
        von_mises_actual = f['DataContainers/SliceDataContainer/CellData/EulerAngles'][:]
        von_mises_actual = np.reshape(von_mises_actual, (1024, 1024, 3))
        von_mises_actual = normalize(von_mises_actual)
        
        ipf = f['DataContainers/SliceDataContainer/CellData/IPFColor'][:]
        ipf = np.reshape(ipf, (1024, 1024, 3))
        
        ipf = ipf[PATCH_SIZE//2:1024-PATCH_SIZE//2,PATCH_SIZE//2:1024-PATCH_SIZE//2]
        
        #plt.figure()
        #plt.imshow(ipf)
        
        plt.figure()
        plt.imshow(von_mises_actual)
"""


TRAIN_FILES = ['../../../Downloads/Microstructures_all_features/Microstructure_08_new_all_features_slice_2.dream3d']


for file in TRAIN_FILES:
    with h5py.File(file,'r+') as f:
        
        von_mises_time = np.zeros(shape=(1024-PATCH_SIZE,1024-PATCH_SIZE,20))
        count = 0
        for i in range(5,101,5):
            
            von_mises_actual = f[f'DataContainers/SliceDataContainer/CellData/Von_Miss_{i}'][:]
            von_mises_actual = np.reshape(von_mises_actual, (1024, 1024))
            von_mises_actual = von_mises_actual[PATCH_SIZE//2:1024-PATCH_SIZE//2,PATCH_SIZE//2:1024-PATCH_SIZE//2]
            von_mises_time[:,:,count] = von_mises_actual
    
            count += 1


MAX = np.max(von_mises_time)
MIN = np.min(von_mises_time)            

diff = []
for i in range(19):
    
    diff.append(np.mean(von_mises_time[:,:,i+1]) - np.mean(von_mises_time[:,:,i]))

mean_diff = np.mean(diff)
std_diff = np.std(diff)

mhi_matrix = mhi(von_mises_time, mean_diff+2.0*std_diff)


animation(von_mises_time,MIN,MAX,'Von Mises Evolution',20,1,2,True)
animation(mhi_matrix,0,20,'MHI',20, 1, 2)

"""
TRAIN_FILES = ['../../../Downloads/Microstructures_all_features/Microstructure_00_new_all_features_slice_2.dream3d']


for file in TRAIN_FILES:
    with h5py.File(file,'r+') as f:
        
        video_name = 'video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video = cv2.VideoWriter(video_name, fourcc, 1.0, (1024, 1024), isColor=False)

        for i in range(5,101,5):
            
            von_mises_actual = f[f'DataContainers/SliceDataContainer/CellData/Euler_{i}'][:]
            von_mises_actual = np.reshape(von_mises_actual, (1024, 1024, 3))
            #von_mises_actual = np.rollaxis(von_mises_actual,0,1)
            
            video.write(von_mises_actual)

cv2.destroyAllWindows()
video.release()
"""

"""
import matplotlib.animation as animation


TRAIN_FILES = ['../../../Downloads/Microstructures_all_features/Microstructure_00_new_all_features_slice_2.dream3d']


for file in TRAIN_FILES:
    with h5py.File(file,'r+') as f:
        
        img = [] # some array of images
        frames = [] # for storing the generated images
        fig = plt.figure()
        
        for i in range(5,101,5):
            
                von_mises_actual = f[f'DataContainers/SliceDataContainer/CellData/Von_Miss_{i}'][:]
                von_mises_actual = np.reshape(von_mises_actual, (1024, 1024, 1))

                frames.append([plt.imshow(von_mises_actual, animated=True)])
        
        ani = animation.ArtistAnimation(fig, frames, interval=1, blit=True,
                                       repeat_delay=1000)
        #ani.save('movie.avi')
        plt.show()
"""






"""
flattened_array = 
index_sorted = np.argsort(von_mises_actual.flatten())
sorted_stress_values = von_mises_actual.flatten()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights_array = np.ones(shape=(BATCH_SIZE,TARGET_ROW//2,TARGET_COL//2))

LOSS_WEIGHTS = Variable(torch.Tensor(weights_array)).to(device)
print(LOSS_WEIGHTS.shape)
"""

