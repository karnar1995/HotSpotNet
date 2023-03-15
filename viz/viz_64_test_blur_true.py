# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 20:29:56 2023

@author: giriprasad.1
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import cv2

PATCH_SIZE = 64

TIMESTEP = 10

print(np.load('../MIN_VALUES.npy'))
print(np.load('../MAX_VALUES.npy'))

y_hat = np.load(f'../datasets/test/predicted_whole_image_64_timestep_{TIMESTEP}.npy')
y_t = np.load(f'../datasets/test/true_whole_image_64_timestep_{TIMESTEP}.npy')


fig,ax = plt.subplots()
colormap = 'viridis'

y_t_blurred = cv2.GaussianBlur(y_t,(11,11),cv2.BORDER_DEFAULT)

fig.set_size_inches(15,15)
plt.subplot(2,2,1)
plt.imshow(y_t_blurred[PATCH_SIZE//2:1024-PATCH_SIZE//2,PATCH_SIZE//2:1024-PATCH_SIZE//2],colormap)
plt.title('Original stress field')

plt.subplot(2,2,2)
plt.imshow(y_hat[PATCH_SIZE//2:1024-PATCH_SIZE//2,PATCH_SIZE//2:1024-PATCH_SIZE//2],colormap)
plt.title('Predicted stress field')

fig,ax = plt.subplots(figsize=(6,6))
colormap = 'viridis'

histogram, bin_edges = np.histogram(y_t_blurred[PATCH_SIZE//2:1024-PATCH_SIZE//2,PATCH_SIZE//2:1024-PATCH_SIZE//2], bins=256)

plt.subplot(2,2,1)
plt.plot(bin_edges[0:-1], histogram) 
plt.title("Original Stress Histogram")
plt.xlim(0,1)

plt.xticks(fontsize=7)
plt.yticks(fontsize=7)

plt.xlabel("Stress Value",fontsize=6)
plt.ylabel("Count",fontsize=6)

histogram, bin_edges = np.histogram(y_hat[PATCH_SIZE//2:1024-PATCH_SIZE//2,PATCH_SIZE//2:1024-PATCH_SIZE//2], bins=256)

plt.subplot(2,2,2)
plt.plot(bin_edges[0:-1], histogram) 
plt.title("Predicted Stress Histogram")
plt.xlim(0,1)

plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.xlabel("Stress Value", fontsize=6)

plt.figure()

plt.scatter(y_t_blurred[PATCH_SIZE//2:1024-PATCH_SIZE//2,PATCH_SIZE//2:1024-PATCH_SIZE//2], y_hat[PATCH_SIZE//2:1024-PATCH_SIZE//2,PATCH_SIZE//2:1024-PATCH_SIZE//2], 
                c="yellow",
            linewidths = 2,
            marker =".",
            edgecolor ="green",
            cmap="viridis",
            s = 200)

plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("original")
plt.ylabel("predicted")



"""
histogram, bin_edges = np.histogram(y_t_blurred[PATCH_SIZE//2:1024-PATCH_SIZE//2,PATCH_SIZE//2:1024-PATCH_SIZE//2], bins=256)
plt.figure()
plt.plot(bin_edges[0:-1], histogram) 
plt.title("Original Stress Histogram (unnormalized")



plt.figure()
a = plt.scatter(y_t, y_hat, c ="yellow",
            linewidths = 2,
            marker =".",
            edgecolor ="red",
            s = 200)

plt.xlabel("original")
plt.ylabel("predicted")

plt.legend('32x32')
"""

"""
# create the histogram
histogram, bin_edges = np.histogram(y_t, bins=256, range=(0, 1))

plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixel count")
plt.xlim([0.0, 1.0])  # <- named arguments do not work here

plt.plot(bin_edges[0:-1], histogram) 


# create the histogram
histogram, bin_edges = np.histogram(y_hat, bins=256, range=(0, 1))

plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixel count")
plt.xlim([0.0, 1.0])  # <- named arguments do not work here

plt.plot(bin_edges[0:-1], histogram) 

"""

"""

y_hat = np.load(f'../datasets/test/predicted_test_images_64_timestep_{TIMESTEP}.npy')
y_t = np.load(f'../datasets/test/true_test_images_64_timestep_{TIMESTEP}.npy')

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

MSE = np.square(np.subtract(y_t,y_hat)).mean()
print(MSE)
"""