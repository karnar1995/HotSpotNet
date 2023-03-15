# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 17:52:07 2022

@author: giriprasad.1
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import cv2

from utils.utils import normalize, normalize_all, extract_patches_to_array


class MakeDataset():
    
    def __init__(self, files, randomize, random_number, blur=False, crop=True, normalize = True,
                 blur_kernel_size=7, blur_sigma=1,
                 patch_size=256, num_rand_patches=2000, 
                 target_row=64, target_col=64, stride=1, timestep = 100):
        
        self.timestep = timestep
        
        self.patch_size = patch_size
        self.randomize = randomize
        self.normalize = normalize
        self.num_rand_patches = num_rand_patches
        self.blur = blur
        self.crop = crop
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.stride = stride
        
        self.target_row = target_row
        self.target_col = target_col
        self.target_crop_row = self.target_row//2
        self.target_crop_col = self.target_col//2
        
        self.random_number = random_number
        
        self.files = files
        self.EulerAngles = {}
        self.F7 = {}
        self.Directional_modulus = {}
        self.Von_Mises_Stress = {}
        self.Schmids = {}
        self.Misorientations = {}
        self.BoundaryCells = {}
        self.SlipSystems = {}
        self.mPrime = {}
        self.F1 = {}
        
        self.merged_data = {}
        
        self.X = []
        self.Y = []
        
        self.X_train = None
        self.Y_train = None
        self.X_val = None
        self.Y_val = None
        
    def _read_data(self):
        
        for i,file in enumerate(self.files):
            with h5py.File(file,'r+') as f:
                
                print(file)
                self.EulerAngles[i] = f[
                    'DataContainers/SliceDataContainer/CellData/EulerAngles'][:]
                self.F7[i] = f[
                    'DataContainers/SliceDataContainer/CellData/F7_0'][:]
                self.Directional_modulus[i] = f[
                    'DataContainers/SliceDataContainer/CellData/Directional_Modulus_0'][:]
                self.Schmids[i] = f[
                    'DataContainers/SliceDataContainer/CellData/Schmids_0'][:]
                self.Von_Mises_Stress[i] = f[
                    f'DataContainers/SliceDataContainer/CellData/Von_Miss_{self.timestep}'][:]
                self.Misorientations[i] = f[
                    'DataContainers/SliceDataContainer/CellData/KernelAverageMisorientationsArrayName_0'][:]
                self.BoundaryCells[i] = f[
                    'DataContainers/SliceDataContainer/CellData/BoundaryCells'][:]
                self.SlipSystems[i] = f[
                    'DataContainers/SliceDataContainer/CellData/SlipSystems_0'][:]
                self.mPrime[i] = f[
                    'DataContainers/SliceDataContainer/CellData/mPrime_0'][:]
                self.F1[i] = f[
                    'DataContainers/SliceDataContainer/CellData/F1_0'][:]
            
         
    def _reshape_data(self):
        
        for i in range(len(self.files)):
            
             self.EulerAngles[i] = np.reshape(self.EulerAngles[i], 
                                              (1024, 1024, 3)).astype(np.float32)
             self.F7[i] = np.reshape(self.F7[i], 
                                     (1024, 1024, 1)).astype(np.float32)
             self.Directional_modulus[i] = np.reshape(self.Directional_modulus[i], 
                                                      (1024, 1024, 1)).astype(np.float32)
             self.Schmids[i] = np.reshape(self.Schmids[i], 
                                          (1024, 1024, 1)).astype(np.float32)
             self.Von_Mises_Stress[i] = np.reshape(self.Von_Mises_Stress[i], 
                                                      (1024, 1024, 1)).astype(np.float32)
             self.Misorientations[i] = np.reshape(self.Misorientations[i], 
                                                  (1024, 1024, 1)).astype(np.float32)
             self.BoundaryCells[i] = np.reshape(self.BoundaryCells[i], 
                                                (1024, 1024, 1)).astype(np.float32)
             self.SlipSystems[i] = np.reshape(self.SlipSystems[i], 
                                              (1024, 1024, 1)).astype(np.float32)
             self.mPrime[i] = np.reshape(self.mPrime[i], 
                                         (1024, 1024, 1)).astype(np.float32)
             self.F1[i] = np.reshape(self.F1[i], 
                                     (1024, 1024, 1)).astype(np.float32)
    
    def _merge(self):
        
        for i in range(len(self.EulerAngles)):
            c1, c2, c3 = cv2.split(self.EulerAngles[i])
            self.merged_data[i] = cv2.merge((c1,c2,c3,
                                             self.Schmids[i],
                                             self.Misorientations[i],
                                             self.BoundaryCells[i],
                                             self.Directional_modulus[i],
                                             self.F7[i],
                                             self.SlipSystems[i],
                                             self.mPrime[i],
                                             self.F1[i]
                                             ))
         
    def _normalize(self, method='minmax'):
        
        for i in range(len(self.merged_data)):
    
            #self.merged_data[i] = normalize(self.merged_data[i], method=method)
            self.Von_Mises_Stress[i] = normalize(self.Von_Mises_Stress[i], method=method)

    def _normalize_all(self, method='minmax'):
        
        #self.merged_data = normalize_all(self.merged_data, method=method)
        self.Von_Mises_Stress = normalize_all(self.Von_Mises_Stress, method=method)    

    def _get_train_test(self):
        
        self._read_data()
        self._reshape_data()
        self._merge()
        
        if self.normalize:
            self._normalize_all(method='minmax')
        
        #Gaussian 
        if self.blur:

            for i in range(len(self.Von_Mises_Stress)):
                
                self.Von_Mises_Stress[i] = np.reshape(cv2.GaussianBlur(self.Von_Mises_Stress[i],(self.blur_kernel_size,self.blur_kernel_size),cv2.BORDER_DEFAULT),
                                                      (1024, 1024, 1))
                
                
        for i in range(len(self.merged_data)):
            train_list = extract_patches_to_array(
                randomize=self.randomize,
                patch_size=self.patch_size, 
                img=self.merged_data[i], 
                random_number=self.random_number, 
                num_rand_patches=self.num_rand_patches,
                stride=self.stride)
            self.X.append(train_list)
            del train_list
        
        self.y = []
        for i in range(len(self.Von_Mises_Stress)):
            predict_list = extract_patches_to_array(
                randomize=self.randomize,
                patch_size=self.patch_size, 
                img=self.Von_Mises_Stress[i], 
                random_number=self.random_number, 
                num_rand_patches=self.num_rand_patches,
                stride=self.stride)
            self.y.append(predict_list)
            del predict_list
        
        self.X = [item for sublist in self.X for item in sublist]
        self.y = [item for sublist in self.y for item in sublist]
        
        self.X = np.asarray(self.X)
        self.y = np.asarray(self.y)
        
        self.y = np.reshape(self.y,(len(self.y),self.y[0].shape[0],self.y[0].shape[1]))
        
        #Downsample if needed and take the center cropped image as output
        if self.crop:
            
            self.Y = np.empty([len(self.y), self.target_crop_row, self.target_crop_col])
            for i in range(len(self.y)):
                temp = cv2.resize(self.y[i], (self.target_row, self.target_col), interpolation=cv2.INTER_AREA)
                self.Y[i] = temp[self.target_crop_row//2:(self.target_crop_row//2)+self.target_crop_row,
                                      self.target_crop_col//2:(self.target_crop_col//2)+self.target_crop_col]
        else:
            self.Y=self.y.copy()
        
        del self.y
        
        return self.X,self.Y
    
    def __getitem__(self, index):
        
        x,y = self._get_train_test()
        return x[index],y[index]
    
    def __len__(self):
        return len(self.X)
    
    
    def split_train_test(self, test_size=0.2):
        
       self.X, self.Y = self._get_train_test()
       
       self.X_train, self.X_val, self.Y_train, self.Y_val = \
           train_test_split(self.X, self.Y, test_size=test_size, shuffle=True)
       
       del self.X, self.Y
       
       return self.X_train, self.X_val, self.Y_train, self.Y_val
   
class GetDataset(Dataset):

    def __init__(self, X, Y, patch_size=256, transform=None):
        self.len = len(X)
        self.transform = transform
        self.patch_size = patch_size
        self.x_data = X
        self.x_data = torch.FloatTensor(self.x_data).view(-1,X.shape[3],self.patch_size,self.patch_size)
        self.y_data = torch.from_numpy(Y)
        self.y_data = self.y_data.float()
        

    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index]
        if self.transform:
            x=self.transform(x)
        return x,y
    
    def __len__(self):
        return self.len