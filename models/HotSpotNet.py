# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 17:52:23 2022

@author: giriprasad.1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class HotSpotNet_256(nn.Module):
    
    def __init__(self):
        super(HotSpotNet_256, self).__init__()
        
        
        self.conv1 = nn.Conv2d(11, 64, 3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2)
        self.conv3 = nn.Conv2d(128, 64,3)

        self.deconv1 = nn.ConvTranspose2d(64, 128, 4, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 32, 3, stride=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, 7, stride=3)
        
        self.drop1 = nn.Dropout(p=0.2)
        
        self.pool1 = nn.MaxPool2d(3, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 1)
       
        self.fc1 = nn.Linear(48*48, 2000)
        self.fc2 = nn.Linear(2000, 64*64)
        
  
    def forward(self, x):
        
        x = self.pool1(F.leaky_relu(self.conv1(x)))
        x = self.drop1(self.pool2(F.leaky_relu(self.conv2(x))))
        x = self.drop1(self.pool2(F.leaky_relu(self.conv3(x))))
        
        
        #x = self.pool(F.leaky_relu(self.conv5(x)))
        #x = self.b_n5(x)
        #x = self.pool(F.leaky_relu(self.conv6(x)))
        #x = self.b_n6(x)
        #x = self.pool(F.leaky_relu(self.conv7(x)))
        #x = self.b_n7(x)
        
        x = x.view(-1, 48*48)
        
        x = F.leaky_relu(self.fc1(x))
        x = self.drop1(F.leaky_relu(self.fc2(x)))
        
        x = x.view(-1, 64, 8, 8)
        
        x = self.drop1(F.leaky_relu(self.deconv1(x)))
        x = self.drop1(F.leaky_relu(self.deconv2(x)))
        x = torch.tanh(self.deconv3(x))
        
        x = x.view(-1, 64, 64)
        
        
        #x = F.relu(self.fc1(x))
        #x = self.fc2(x)
        
        return x


class HotSpotNet_128_original(nn.Module):
    
    
    def __init__(self):
        super(HotSpotNet_128_original, self).__init__()
        
        self.conv1 = nn.Conv2d(11, 64, 8, stride=8)
        self.conv2 = nn.Conv2d(64, 16, 3, stride=1)

        self.deconv1 = nn.ConvTranspose2d(16, 64, 2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(64, 8, 3, stride=1)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 10, stride=2)

        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 1)

        
  
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.tanh(self.deconv3(x))
        
        x = x.view(-1, 32, 32)

        #x = F.relu(self.fc1(x))
        #x = self.fc2(x)
        
        return x

class HotSpotNet_128(nn.Module):
    
    
    def __init__(self):
        super(HotSpotNet_128, self).__init__()
        
        self.conv1 = nn.Conv2d(11, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2)
        self.conv3 = nn.Conv2d(128, 64,3, stride=2)

        self.deconv1 = nn.ConvTranspose2d(64, 128, 5)
        self.deconv2 = nn.ConvTranspose2d(128, 32, 3, stride=2)
        self.deconv3 = nn.ConvTranspose2d(32, 1, 8)
        
        self.drop1 = nn.Dropout(p=0.01)
        
        self.pool1 = nn.MaxPool2d(3, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 1)
       
        self.fc1 = nn.Linear(48*48, 2000)
        self.fc2 = nn.Linear(2000, 64*64)
        
  
    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.conv1(x)))
        x = self.drop1(self.pool2(F.leaky_relu(self.conv2(x))))
        x = self.drop1(self.pool3(F.leaky_relu(self.conv3(x))))
        
        #x = self.pool(F.leaky_relu(self.conv5(x)))
        #x = self.b_n5(x)
        #x = self.pool(F.leaky_relu(self.conv6(x)))
        #x = self.b_n6(x)
        #x = self.pool(F.leaky_relu(self.conv7(x)))
        #x = self.b_n7(x)
        
        x = x.view(-1, 48*48)
        
        x = F.leaky_relu(self.fc1(x))
        x = self.drop1(F.leaky_relu(self.fc2(x)))
        
        x = x.view(-1, 64, 8, 8)
        
        x = self.drop1(F.leaky_relu(self.deconv1(x)))
        x = self.drop1(F.leaky_relu(self.deconv2(x)))
        x = torch.tanh(self.deconv3(x))
        
        x = x.view(-1, 32, 32)
        
        #x = F.relu(self.fc1(x))
        #x = self.fc2(x)
        
        return x
    
class HotSpotNet_64_up(nn.Module):
    
    
    def __init__(self):
        super(HotSpotNet_64_up, self).__init__()
        
        self.conv1 = nn.Conv2d(11, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 64, 3)
        
        self.conv4 = nn.Conv2d(64, 128, 3,padding=1)
        self.conv5 = nn.Conv2d(128, 32, 3,padding=1)
        self.conv6 = nn.Conv2d(32, 1, 3,padding=1)
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
  
        self.drop1 = nn.Dropout(p=0.2)
        
        self.b_n1 = nn.BatchNorm2d(32)
        self.b_n2 = nn.BatchNorm2d(64)
        self.b_n3 = nn.BatchNorm2d(128)
        
        self.pool1 = nn.MaxPool2d(3, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 1)
       
        self.fc1 = nn.Linear(48*48, 2000)
        self.fc2 = nn.Linear(2000, 64*64)
        
  
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.drop1(self.pool2(F.relu(self.conv2(x))))
        x = self.drop1(self.pool2(F.relu(self.conv3(x))))
        
        #x = self.pool(F.leaky_relu(self.conv5(x)))
        #x = self.b_n5(x)
        #x = self.pool(F.leaky_relu(self.conv6(x)))
        #x = self.b_n6(x)
        #x = self.pool(F.leaky_relu(self.conv7(x)))
        #x = self.b_n7(x)
        
        x = x.view(-1, 48*48)
        
        x = F.relu(self.fc1(x))
        x = self.drop1(F.relu(self.fc2(x)))
        
        x = x.view(-1, 64, 8, 8)
        
        x = self.up(x)
        x = self.drop1(F.relu(self.conv4(x)))
        x = self.up(x)
        x = self.drop1(F.relu(self.conv5(x)))
        x = torch.tanh(self.conv6(x))
        
        x = x.view(-1, 32, 32)
        
        #x = F.relu(self.fc1(x))
        #x = self.fc2(x)
        
        return x

class HotSpotNet_64(nn.Module):
    
    
    def __init__(self):
        super(HotSpotNet_64, self).__init__()
        
        self.conv1 = nn.Conv2d(11, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 64, 3)

        self.deconv1 = nn.ConvTranspose2d(64, 128, 5)
        self.deconv2 = nn.ConvTranspose2d(128, 32, 3, stride=2)
        self.deconv3 = nn.ConvTranspose2d(32, 1, 8)
        
        self.drop1 = nn.Dropout(p=0.2)
        
        self.b_n1 = nn.BatchNorm2d(32)
        self.b_n2 = nn.BatchNorm2d(64)
        self.b_n3 = nn.BatchNorm2d(128)
        
        self.pool1 = nn.MaxPool2d(3, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 1)
       
        self.fc1 = nn.Linear(48*48, 2000)
        self.fc2 = nn.Linear(2000, 64*64)
        
  
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.drop1(self.pool2(F.relu(self.conv2(x))))
        x = self.drop1(self.pool2(F.relu(self.conv3(x))))
        
        #x = self.pool(F.leaky_relu(self.conv5(x)))
        #x = self.b_n5(x)
        #x = self.pool(F.leaky_relu(self.conv6(x)))
        #x = self.b_n6(x)
        #x = self.pool(F.leaky_relu(self.conv7(x)))
        #x = self.b_n7(x)
        
        x = x.view(-1, 48*48)
        
        x = F.relu(self.fc1(x))
        x = self.drop1(F.relu(self.fc2(x)))
        
        x = x.view(-1, 64, 8, 8)
        
        x = self.drop1(F.relu(self.deconv1(x)))
        x = self.drop1(F.relu(self.deconv2(x)))
        x = torch.tanh(self.deconv3(x))
        
        x = x.view(-1, 32, 32)
        
        #x = F.relu(self.fc1(x))
        #x = self.fc2(x)
        
        return x
    
class HotSpotNet_64_new(nn.Module):
    
    
    def __init__(self):
        super(HotSpotNet_64_new, self).__init__()
        
        self.conv1 = nn.Conv2d(11, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)

        self.deconv1 = nn.ConvTranspose2d(128, 128, 4, padding=1, stride=2)
        self.deconv2 = nn.ConvTranspose2d(256, 32, 4, padding=1, stride=2)
        self.deconv3 = nn.ConvTranspose2d(96, 1, 3, padding=1)
        
        self.drop1 = nn.Dropout(p=0.2)
        
        self.b_n1 = nn.BatchNorm2d(32)
        self.b_n2 = nn.BatchNorm2d(64)
        self.b_n3 = nn.BatchNorm2d(128)
        
        self.pool1 = nn.MaxPool2d(3, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 1)
       
        self.fc1 = nn.Linear(64*64, 5000)
        self.fc2 = nn.Linear(5000, 64*64)
        
  
    def forward(self, x):
        
        skip_connections = []
        x = self.pool2(F.relu(self.conv1(x)))
        x = self.b_n2(x)
        skip_connections.append(x)
        #print('Conv1 pool1', x.shape)
        x = self.drop1(self.pool2(F.relu(self.conv2(x))))
        skip_connections.append(x)
        #print('Conv2 pool2', x.shape)
        x = self.drop1(self.pool2(F.relu(self.conv3(x))))
        skip_connections.append(x)
        #print('Conv3 pool3', x.shape)
        
        #x = self.pool(F.leaky_relu(self.conv5(x)))
        #x = self.b_n5(x)
        #x = self.pool(F.leaky_relu(self.conv6(x)))
        #x = self.b_n6(x)
        #x = self.pool(F.leaky_relu(self.conv7(x)))
        #x = self.b_n7(x)
        
        x = x.view(-1, 64*64)
        
        x = F.relu(self.fc1(x))
        x = self.drop1(F.relu(self.fc2(x)))
        
        x = x.view(-1, 64, 8, 8)
        
        #print(skip_connections[-1].shape, skip_connections[-2].shape, skip_connections[-3].shape)
        x = torch.cat([x, skip_connections[-1]], dim=1)
        x = self.drop1(F.relu(self.deconv1(x)))
        #print('Deconv1', x.shape)
        
        x = torch.cat([x, skip_connections[-2]], dim=1)
        x = self.drop1(F.relu(self.deconv2(x)))
        #print('Deconv2', x.shape)
        
        x = torch.cat([x, skip_connections[-3]], dim=1)
        x = torch.tanh(self.deconv3(x))
        #print('Deconv3', x.shape)
        
        
        x = x.view(-1, 32, 32)
        
        #x = F.relu(self.fc1(x))
        #x = self.fc2(x)
        
        return x
    
class HotSpotNet_64_test(nn.Module):
    
    
    def __init__(self):
        super(HotSpotNet_64_test, self).__init__()
        
        self.conv1 = nn.Conv2d(11, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)

        self.deconv1 = nn.ConvTranspose2d(64, 128, 4, padding=1, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 32, 4, padding=1, stride=2)
        self.deconv3 = nn.ConvTranspose2d(32, 1, 3, padding=1)
        
        self.drop1 = nn.Dropout(p=0.2)
        
        self.b_n1 = nn.BatchNorm2d(32)
        self.b_n2 = nn.BatchNorm2d(64)
        self.b_n3 = nn.BatchNorm2d(128)
        
        self.pool1 = nn.MaxPool2d(3, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 1)
       
        self.fc1 = nn.Linear(64*64, 5000)
        self.fc2 = nn.Linear(5000, 64*64)
        
  
    def forward(self, x):
        x = self.pool2(F.relu(self.conv1(x)))
        x = self.b_n2(x)
        x = self.drop1(self.pool2(F.relu(self.conv2(x))))
        x = self.drop1(self.pool2(F.relu(self.conv3(x))))
        
        #x = self.pool(F.leaky_relu(self.conv5(x)))
        #x = self.b_n5(x)
        #x = self.pool(F.leaky_relu(self.conv6(x)))
        #x = self.b_n6(x)
        #x = self.pool(F.leaky_relu(self.conv7(x)))
        #x = self.b_n7(x)
        
        x = x.view(-1, 64*64)
        
        x = F.relu(self.fc1(x))
        x = self.drop1(F.relu(self.fc2(x)))
        
        x = x.view(-1, 64, 8, 8)
        
        x = self.drop1(F.relu(self.deconv1(x)))
        x = self.drop1(F.relu(self.deconv2(x)))
        x = torch.tanh(self.deconv3(x))
        
        x = x.view(-1, 32, 32)
        
        #x = F.relu(self.fc1(x))
        #x = self.fc2(x)
        
        return x
    
class HotSpotNet_uncomp_64(nn.Module):
    def __init__(self):
        super(HotSpotNet_uncomp_64, self).__init__()
        
        self.conv1 = nn.Conv2d(11, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, 3, padding=1)

        self.deconv1 = nn.ConvTranspose2d(64, 128, 3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 32, 3, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, 3, padding=1)
        
        self.drop1 = nn.Dropout(p=0.2)
        
        self.pool1 = nn.MaxPool2d(3, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 1)
       
        #self.fc1 = nn.Linear(64*4096, 2000)
        #self.fc2 = nn.Linear(2000, 64*4096)
        
  
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.drop1(F.leaky_relu(self.conv2(x)))
        x = self.drop1(F.leaky_relu(self.conv3(x)))
        
        #x = self.pool(F.leaky_relu(self.conv5(x)))
        #x = self.b_n5(x)
        #x = self.pool(F.leaky_relu(self.conv6(x)))
        #x = self.b_n6(x)
        #x = self.pool(F.leaky_relu(self.conv7(x)))
        #x = self.b_n7(x)
        
        """
        x = x.view(-1, 64*4096)
        
        x = F.leaky_relu(self.fc1(x))
        x = self.drop1(F.leaky_relu(self.fc2(x)))
        
        x = x.view(-1, 64, 64, 64)
        """
        
        x = self.drop1(self.pool2(F.leaky_relu(self.deconv1(x))))
        x = self.drop1(F.leaky_relu(self.deconv2(x)))
        x = torch.tanh((self.deconv3(x)))
        
        x = x.view(-1, 32, 32)
        
        #x = F.relu(self.fc1(x))
        #x = self.fc2(x)
        
        return x