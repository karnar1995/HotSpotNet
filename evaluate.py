# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 02:52:48 2023

@author: giriprasad.1
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 17:51:48 2022

@author: giriprasad.1
"""

import torch
from torch.utils.data import DataLoader
from datasets.make_dataset import MakeDataset, GetDataset
from models.HotSpotNet import HotSpotNet_64
from utils.utils import normalize
import h5py
import time
import numpy as np

TIMESTEP = 10

NORMALIZE = True

BLUR_OUTPUT = True
CROP_OUTPUT = True

if BLUR_OUTPUT:
    BLUR_KERNEL_SIZE = 11
    BLUR_SIGMA = 1
else:
    BLUR_KERNEL_SIZE = None
    BLUR_SIGMA = None
    
PATCH_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.0001
TARGET_ROW = 64
TARGET_COL = 64
RANDOMIZE = False 
NUM_RAND_PATCHES = 20000
RANDOM_NUMBER_RANGE = 400000

STRIDE = 8 # defauly: 8

# TODO: Use os to load files
TRAIN_FILES = ['../../../Downloads/Microstructures_all_features/Microstructure_03_new_all_features_slice_3.dream3d']

#TEST_FILES = ['../../../Downloads/Microstructures_all_features/Microstructure_01_new_all_features_slice_3.dream3d']

def main():
        
    max_values = np.load('MAX_VALUES.npy')
    min_values = np.load('MIN_VALUES.npy')
    
    for file in TRAIN_FILES:
        with h5py.File(file,'r+') as f:
            von_mises_actual = f[f'DataContainers/SliceDataContainer/CellData/Von_Miss_{TIMESTEP}'][:]
            von_mises_actual = np.reshape(von_mises_actual, (1024, 1024, 1)).astype(np.float32)
            
            for c in range(von_mises_actual.shape[2]):
                
                von_mises_actual[:,:,c] = (von_mises_actual[:,:,c] - min_values[c])/(max_values[c] - min_values[c])
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # parameters to GPU
    net_Adam = HotSpotNet_64().to(device)
        
    md = MakeDataset(files=TRAIN_FILES,
                     randomize=RANDOMIZE,
                     random_number=0,
                     blur=BLUR_OUTPUT,
                     crop=CROP_OUTPUT,
                     normalize = NORMALIZE,
                     blur_kernel_size=BLUR_KERNEL_SIZE,
                     blur_sigma=BLUR_SIGMA,
                     patch_size=PATCH_SIZE, 
                     num_rand_patches=NUM_RAND_PATCHES, 
                     target_row=TARGET_ROW,
                     target_col=TARGET_COL,
                     stride=STRIDE,
                     timestep=TIMESTEP)
        
    X_test,Y_test = md[:]
    
    tdataset = GetDataset(X_test,Y_test,PATCH_SIZE,transform=None)
    test_loader = DataLoader(dataset=tdataset,batch_size=BATCH_SIZE,
                              shuffle=False)
    
    net_Adam.load_state_dict(torch.load(f'model_save_files/hotspotnet_{PATCH_SIZE}_blur_{BLUR_OUTPUT}_timestep_{TIMESTEP}'))
    net_Adam.eval()
    
    X_test,Y_test = tdataset[:]
    
    print(X_test.shape,Y_test.shape)
    
    y_hat = []
    y_t = []
    
    with torch.no_grad():
        start_time = time.time() 
        for data in test_loader:
            inputs, labels = data 
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = net_Adam(inputs)
            
            for i in range(len(outputs)):
                y_hat.append(outputs[i])
                y_t.append(labels[i])
            
        end_time = time.time()
        for i in range(len(y_hat)):
    
            y_hat[i] = y_hat[i].cpu().numpy()
            y_t[i] = y_t[i].cpu().numpy()
        
    print('Total Prediction Time: ', end_time - start_time)
    
    
    von_mises_predict = np.zeros(shape=(1024,1024))
    pixel_count = np.zeros(shape=(1024,1024))
    

    n = STRIDE

    count = 0;
    for i in range(0,1024-PATCH_SIZE+1,n):
        for j in range(0,1024-PATCH_SIZE+1,n):
            
            von_mises_predict[i+(PATCH_SIZE//2-16):i+(PATCH_SIZE//2+16),j+(PATCH_SIZE//2-16):j+(PATCH_SIZE//2+16)] = von_mises_predict[i+(PATCH_SIZE//2-16):i+(PATCH_SIZE//2+16),j+(PATCH_SIZE//2-16):j+(PATCH_SIZE//2+16)] + y_hat[count]
            pixel_count[i+(PATCH_SIZE//2-16):i+(PATCH_SIZE//2+16),j+(PATCH_SIZE//2-16):j+(PATCH_SIZE//2+16)] = pixel_count[i+(PATCH_SIZE//2-16):i+(PATCH_SIZE//2+16),j+(PATCH_SIZE//2-16):j+(PATCH_SIZE//2+16)] + 1
            count = count+1
    
    von_mises_predict = von_mises_predict/pixel_count
    
    np.save(f'datasets/test/predicted_whole_image_{PATCH_SIZE}_timestep_{TIMESTEP}', von_mises_predict)
    np.save(f'datasets/test/true_whole_image_{PATCH_SIZE}_timestep_{TIMESTEP}', von_mises_actual)
    
    np.save(f'datasets/test/predicted_test_images_{PATCH_SIZE}_timestep_{TIMESTEP}', y_hat)
    np.save(f'datasets/test/true_test_images_{PATCH_SIZE}_timestep_{TIMESTEP}', y_t)
    
    
if __name__ == '__main__':
    
    main()