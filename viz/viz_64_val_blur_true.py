# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 22:23:08 2022

@author: giriprasad.1
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

y_hat = np.load('../datasets/val/predicted_64_val_blur_True.npy')
y_t = np.load('../datasets/val/true_64_val_blur_True.npy')

train_losses = np.load('../losses/train_losses_64_blur_True.npy')
val_losses = np.load('../losses/val_losses_64_blur_True.npy')

train_losses = [item for sublist in train_losses for item in sublist]
val_losses = [item for sublist in val_losses for item in sublist]


plt.figure()
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(frameon=False)


fig,ax = plt.subplots()

index = random.randrange(0,len(y_t))
print(index)
colormap = 'viridis'

y_t_blurred = cv2.GaussianBlur(y_t[index],(11,11),sigmaX=3)

fig.set_size_inches(15,15)
plt.subplot(2,2,1)
plt.imshow(y_t[index],colormap)
plt.title('Original stress field')

plt.subplot(2,2,2)
plt.imshow(y_hat[index],colormap)
plt.title('Predicted stress field')
