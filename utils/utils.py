# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 17:52:46 2022

@author: giriprasad.1
"""

import random
import pywt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from itertools import product

def normalize(image, method='minmax'):
    
    img = image.copy()
    if method == 'meanstd':
        for c in range(img.shape[2]):
            mean = img[:,:,c].mean()
            std = img[:,:,c].std()
            img[:,:,c] = (img[:,:,c]-mean)/std
    elif method == 'minmax':
        for c in range(img.shape[2]):
            MIN = img[:,:,c].min()
            MAX = img[:,:,c].max()
            if MAX == MIN:
                return img*0
            img[:,:,c] = (img[:,:,c]-MIN)/(MAX - MIN)
    return img


def normalize_all(images, method='minmax'):
    
    img = images.copy()
    img = np.array(list(img.values()))
    
    min_values = []
    max_values = []
    
    mean_values = []
    std_values = []
    
    print(img.shape)

    if method == 'meanstd':
        
        for c in range(img.shape[3]):
        
            mean = np.mean(img[:,:,:,c])
            std = np.std(img[:,:,:,c])
            
            print(mean,std)
            
            mean_values.append(mean)
            std_values.append(std)
            
            for i in range(len(img)):
                
                img[i,:,:,c] = (img[i,:,:,c]-mean)/std
        
        np.save('MEAN_VALUES', mean_values)
        np.save('STD_VALUES', std_values)
            
    elif method == 'minmax':
        
        for c in range(img.shape[3]):
            
            MAX = np.amax(img[:,:,:,c])
            MIN = np.amin(img[:,:,:,c])
            
            print(MIN, MAX)
            
            max_values.append(MAX)
            min_values.append(MIN)
            
            for i in range(len(img)):
    
                img[i,:,:,c] = (img[i,:,:,c]-MIN)/(MAX - MIN)
        
        #np.save('MAX_VALUES', max_values)
        #np.save('MIN_VALUES', min_values)
        
    return img


def random_coordinates(image_shape, patch_size=128, seed=2000): 
    random.seed(seed)
    cord_x_01 = random.randrange(0, image_shape-patch_size)
    cord_y_02 = random.randrange(0, image_shape-patch_size)
    
    return(cord_x_01, cord_y_02)

def random_pairs(coords_list, index):
    
    cord_x_01, cord_y_02 = coords_list[index]
    return(cord_x_01, cord_y_02)
    
def extract_patches_to_array(patch_size, img, stride=1, randomize=True, 
                             random_number=2000, num_rand_patches=1000):
    patch_array = []
    
    # If randomize, pick random coordinates in the image
    if randomize:
        number_list = np.arange(0, img.shape[0]-patch_size)
        coords = [p for p in product(number_list, repeat=2)]
        random.seed(2022)
        random.shuffle(coords)
        
        r = 0
        while r < num_rand_patches:
            i, j = random_pairs(coords_list=coords, index=r+random_number)
            patch_image = img[i:i+patch_size,j:j+patch_size,:]
            patch_array.append(patch_image)
            r += 1
        return patch_array
    
    # Else take all patches with defined stride going col by col, row by row
    else:
        for i in range(0,img.shape[0]-patch_size+1,stride):
            for j in range(0,img.shape[1]-patch_size+1,stride):
                patch_image = img[i:i+patch_size,j:j+patch_size,:]
                patch_array.append(patch_image)
        return patch_array
    
def wavelet_transform(wave2, transform='haar', level=4, k=0.005, plot=True, random_index_to_plot = 10):

    coeffs = pywt.wavedec2(wave2, wavelet=transform, level=level)
    
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs, axes=[1,2])
    
    csort = np.sort(np.abs(coeff_arr.reshape(-1)))
    
    thresh = csort[int(np.floor((1-k)*len(csort)))]
    ind = np.abs(coeff_arr) > thresh
    cfilt = coeff_arr * ind
    
    print(cfilt.shape)
    coeffs_filt = pywt.array_to_coeffs(cfilt, coeff_slices, output_format='wavedec2')
    
    decoeffs = pywt.waverec2(coeffs_filt,wavelet=transform)
    
    if plot:
        plt.figure()
        plt.imshow(decoeffs[random_index_to_plot])
        plt.axis('off')
        plt.rcParams['figure.figsize'] = [8,8]
        plt.title('k = ' + str(k))
        plt.show()
    
    return coeffs_filt


def mhi(images, threshold):
    

    abs_data = np.zeros(shape=(images.shape[0],images.shape[1],images.shape[2]))
    mhi = np.zeros(shape=(images.shape[0],images.shape[1],images.shape[2]))
    tau = images.shape[2];
    
    for i in range(images.shape[2]-1):
        img_prev = images[:,:,i]; 
        img_next = images[:,:,i+1]; 

        abs_im = abs(img_next - img_prev);
        
        abs_im = np.where(abs_im>threshold,1,0)
        
        abs_data[:,:,i+1] = abs_im;
    
    
    for i in range(images.shape[2]-1):
        
        data_next = abs_data[:,:,i+1];
        
        mhi_image_prev = mhi[:,:,i];
        mhi_image_next = mhi[:,:,i+1];
    
        for j in range(data_next.shape[0]):
            for k in range(data_next.shape[1]):
                
                if data_next[j,k] == 1:
                    mhi_image_next[j,k] = tau;

                elif mhi_image_prev[j,k] != 0:             
                        mhi_image_next[j,k] = mhi_image_prev[j,k] - 1;
    
                else:
                    mhi_image_next[j,k] = 0;

    
        mhi[:,:,i+1] = mhi_image_next;
    
    return mhi
    
def animation(images, vmin, vmax, filename, frames, interval, fps, scatter=False):
    
    # Create the figure and axes
    fig, ax = plt.subplots()

    # Create the initial image
    im = ax.imshow(np.random.rand(images.shape[0], images.shape[1]), vmin=vmin, vmax=vmax, cmap='coolwarm')
    fig.colorbar(im)
        
    # Define the animation function
    def animate(i):
        # Generate a new random image
        image = images[:,:,i]
        # Update the data for the image plot
        im.set_data(image)
        ax.set_title(filename + f' (timestep {(i+1)*5})')
        
        if scatter:
            
            x, y = np.unravel_index(image.argmax(), image.shape)
            
            ax.scatter(x, y, c="r", s=6**2, marker="+") # plot markers
            ax.text(x, y, str(i), color="r", fontsize=5) # plot markers
            
        return im
    
    
    # Create the animation object
    anim = FuncAnimation(fig, animate, frames=frames, interval=interval)
    
    # Save the animation as an MP4 video file
    writergif = PillowWriter(fps=fps)
    anim.save(f'{filename}.gif',writer=writergif)