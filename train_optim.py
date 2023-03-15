# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 14:43:36 2022

@author: giriprasad.1
"""

#!/usr/bin/python3

import torch
import torch.nn as nn


class Trainer():


    def __init__(self, model, train_loader, val_loader, test_loader, device, loss_weights, 
                 epoch=25, learning_rate=0.0001, weight_decay=0.0001):

        self.device = device
        self.loss_weights = loss_weights
        self.epoch = epoch
        
        if loss_weights is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = self.weighted_mse_loss
        
        self.model = model
        
        # Choose optimizer
        self.opt_Adam = torch.optim.Adam(self.model.parameters(),lr=learning_rate, weight_decay=weight_decay)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.Adaml = []
        
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
       
    def train(self):


         for epoch in range(self.epoch):
             
             running_loss_train = 0.0
             running_loss_val = 0.0
             running_loss_test = 0.0
             
             self.model.train()
             for i, data in enumerate(self.train_loader,0):
                 
                  inputs,labels = data
                  inputs = inputs.to(self.device)
                  labels = labels.to(self.device)
            
                  self.opt_Adam.zero_grad()
  
                  outputs = self.model(inputs)
                  if self.loss_weights is not None:
                      loss = self.criterion(outputs, labels, self.loss_weights)
                  else:
                      loss = self.criterion(outputs, labels)
                  loss.backward()
                  self.opt_Adam.step()
  
                  running_loss_train += loss.item()*inputs.size(0)
        
             self.model.eval()
             for i, data in enumerate(self.val_loader,0):
                  inputs,labels = data
                  inputs = inputs.to(self.device)
                  labels = labels.to(self.device)
        
                  outputs = self.model(inputs)
                  if self.loss_weights is not None:
                      loss = self.criterion(outputs, labels, self.loss_weights)
                  else:
                      loss = self.criterion(outputs, labels)
   
                  running_loss_val += loss.item()*inputs.size(0)  
                  
             if self.test_loader is not None:
                 for i, data in enumerate(self.test_loader,0):
                      inputs,labels = data
                      inputs = inputs.to(self.device)
                      labels = labels.to(self.device)
            
                      outputs = self.model(inputs)
                      if self.loss_weights is not None:
                          loss = self.criterion(outputs, labels, self.loss_weights)
                      else:
                          loss = self.criterion(outputs, labels)
       
                      running_loss_test += loss.item()*inputs.size(0)
                  
             running_loss_train = running_loss_train/len(self.train_loader.sampler)
             running_loss_val = running_loss_val/len(self.val_loader.sampler)
             
             if self.test_loader is not None:
                 running_loss_test = running_loss_test/len(self.test_loader.sampler)
                 
             self.train_losses.append(running_loss_train)
             self.val_losses.append(running_loss_val)
             self.test_losses.append(running_loss_test)
        
             print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTest Loss: {:.6f}'
                   .format(epoch, running_loss_train, running_loss_val, running_loss_test))
        
         print('Finished Training')
         
         return self.model, self.train_losses, self.val_losses, self.test_losses
     
    def weighted_mse_loss(self, input, target, loss_weights):
        
        return (loss_weights * (input - target) ** 2).sum() / loss_weights.sum()
     
        
    
         

   