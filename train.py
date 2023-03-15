# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 17:51:48 2022

@author: giriprasad.1
"""

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from datasets.make_dataset import MakeDataset, GetDataset
from models.HotSpotNet import HotSpotNet_64
from train_optim import Trainer
import time
import numpy as np

TIMESTEP = 10

WEIGHTS = False
NORMALIZE = True

BLUR_OUTPUT = True 
CROP_OUTPUT = True # Takes center cropped image as y value

if BLUR_OUTPUT:
    BLUR_KERNEL_SIZE = 11
    BLUR_SIGMA = 1
else:
    BLUR_KERNEL_SIZE = None
    BLUR_SIGMA = None
    
PATCH_SIZE = 64 # default: 64
BATCH_SIZE = 32 # default: 32
EPOCHS = 100 # default: 20
LR = 0.0001 # default: 0.0001 learning rate
WD = 0.0001 # default: 0.0001 weight decay
TARGET_ROW = 64 # default: 64 Output resize row size if CROP_OUTPUT is True
TARGET_COL = 64 # default: 64 Output resize col size if CROP_OUTPUT is True
RANDOMIZE = True # default: True
NUM_RAND_PATCHES = 10000 # default: 20000
RANDOM_NUMBER_RANGE = 20000 # default: 200000

STRIDE = 1 # default: 1

TRAIN_TEST_SPLIT_SIZE = 0.3 # default: 0.3

TRAIN_FILES = []
TEST_FILES = []

# TODO: Use os to load files
for i in range(10):
    for slice_index in range(2,4):
        
        if slice_index == 3:
            
            TEST_FILES.append(f'../../../Downloads/Microstructures_all_features/Microstructure_0{i}_new_all_features_slice_{slice_index}.dream3d')
            
        else:
                
            TRAIN_FILES.append(f'../../../Downloads/Microstructures_all_features/Microstructure_0{i}_new_all_features_slice_{slice_index}.dream3d')


train_losses = []
val_losses = []
test_losses = []

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # parameters to GPU
    net_Adam = HotSpotNet_64().to(device)
    
    print(f'Total sets of random samples - {RANDOM_NUMBER_RANGE//NUM_RAND_PATCHES}')
    
    sample_number = 1
    for random_number in range(0, RANDOM_NUMBER_RANGE, NUM_RAND_PATCHES):
        
        print(f'######## Random Samples Set - {sample_number} ########')
        train_md = MakeDataset(files=TRAIN_FILES, 
                         randomize=RANDOMIZE,
                         random_number=random_number,
                         blur=BLUR_OUTPUT,
                         crop=CROP_OUTPUT,
                         normalize=NORMALIZE,
                         blur_kernel_size=BLUR_KERNEL_SIZE,
                         blur_sigma=BLUR_SIGMA,
                         patch_size=PATCH_SIZE, 
                         num_rand_patches=NUM_RAND_PATCHES, 
                         target_row=TARGET_ROW,
                         target_col=TARGET_COL,
                         stride=STRIDE,
                         timestep=TIMESTEP)
        
        X_train,X_val,Y_train,Y_val = train_md.split_train_test(test_size=TRAIN_TEST_SPLIT_SIZE)
        
        # Test dataset not part of training or validation
        
        test_md = MakeDataset(files=TEST_FILES, 
                         randomize=RANDOMIZE,
                         random_number=random_number,
                         blur=BLUR_OUTPUT,
                         crop=CROP_OUTPUT,
                         normalize=NORMALIZE,
                         blur_kernel_size=BLUR_KERNEL_SIZE,
                         blur_sigma=BLUR_SIGMA,
                         patch_size=PATCH_SIZE, 
                         num_rand_patches=NUM_RAND_PATCHES//2, 
                         target_row=TARGET_ROW,
                         target_col=TARGET_COL,
                         stride=STRIDE,
                         timestep=TIMESTEP)
        
        X_test,Y_test = test_md[:]
        
        print("Train shape:", X_train.shape, Y_train.shape,
              "Val shape:", X_val.shape, Y_val.shape, 
              "Test shape:", X_test.shape, Y_test.shape)
        # Train Dataset
        
        xdataset = GetDataset(X_train,Y_train,PATCH_SIZE,transform=None)
        train_loader = DataLoader(dataset=xdataset,batch_size=BATCH_SIZE,
                                  shuffle=True)
        
        vdataset = GetDataset(X_val,Y_val,PATCH_SIZE,transform=None)
        val_loader = DataLoader(dataset=vdataset,batch_size=BATCH_SIZE,
                              shuffle=True)
        
        tdataset = GetDataset(X_test,Y_test,PATCH_SIZE,transform=None)
        test_loader = DataLoader(dataset=tdataset,batch_size=BATCH_SIZE,
                              shuffle=True)
        
        if WEIGHTS:
            LOSS_WEIGHTS = Variable(torch.Tensor(np.ones(shape=(BATCH_SIZE,TARGET_ROW//2,TARGET_COL//2)))).to(device)
        else:
            LOSS_WEIGHTS = Variable(torch.Tensor(np.ones(shape=(BATCH_SIZE,TARGET_ROW//2,TARGET_COL//2)))).to(device)
        
        trainer = Trainer(net_Adam, train_loader, val_loader, test_loader, device, loss_weights=None,  
                     epoch=EPOCHS, learning_rate=LR, weight_decay=WD)
        
        net_Adam, train_loss, val_loss, test_loss = trainer.train()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        
        sample_number += 1
        torch.save(net_Adam.state_dict(), f'model_save_files/hotspotnet_{PATCH_SIZE}_blur_{BLUR_OUTPUT}_timestep_{TIMESTEP}')
    
    net_Adam.eval()
    y_hat = []
    y_t = []
    
    with torch.no_grad():
        start_time = time.time() 
        for data in val_loader:
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
    
    np.save(f'datasets/val/predicted_{PATCH_SIZE}_val_blur_{BLUR_OUTPUT}_timestep_{TIMESTEP}', y_hat)
    np.save(f'datasets/val/true_{PATCH_SIZE}_val_blur_{BLUR_OUTPUT}_timestep_{TIMESTEP}', y_t)
    
    np.save(f'losses/train_losses_{PATCH_SIZE}_blur_{BLUR_OUTPUT}_timestep_{TIMESTEP}', train_losses)
    np.save(f'losses/val_losses_{PATCH_SIZE}_blur_{BLUR_OUTPUT}_timestep_{TIMESTEP}', val_losses)
    np.save(f'losses/val_losses_{PATCH_SIZE}_blur_{BLUR_OUTPUT}_timestep_{TIMESTEP}', test_losses)
    
if __name__ == '__main__':
    
    main()