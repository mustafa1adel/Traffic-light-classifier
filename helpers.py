# Helper functions

import os
import glob # library for loading images from a directory
import matplotlib.image as mpimg
import cv2
import numpy as np
import matplotlib.pyplot as plt


# This function loads in images and their labels and places them in a list
# The list contains all images and their associated labels
# For example, after data is loaded, im_list[0][:] will be the first image-label pair in the list
def load_dataset(image_dir):
    
    # Populate this empty image list
    im_list = []
    image_types = ["red", "yellow", "green"]
    
    # Iterate through each color folder
    for im_type in image_types:
        
        # Iterate through each image file in each image_type folder
        # glob reads in any image with the extension "image_dir/im_type/*"
        for file in glob.glob(os.path.join(image_dir, im_type, "*")):
            
            # Read in the image
            im = mpimg.imread(file)
            
            # Check if the image exists/if it's been correctly read-in
            if not im is None:
                # Append the image, and it's type (red, green, yellow) to the image list
                im_list.append((im, im_type))

    return im_list

def show_hist_rgb_img(rgb_image):
    """
    plot histogram of H, S and V channel for the RGB input image,
    compared with the H, S and V images.
    """
    r_channel = rgb_image[:,:,0]
    g_channel = rgb_image[:,:,1]
    b_channel = rgb_image[:,:,2]
    
    r_hist = np.histogram(r_channel, bins = 32, range=(0, 256))
    g_hist = np.histogram(g_channel, bins = 32, range=(0, 256))
    b_hist = np.histogram(b_channel, bins = 32, range=(0, 256))
    
    
    # Generating bin centers
    bin_edges = r_hist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2

    
    f, ax = plt.subplots(2, 3, figsize=(20, 10))

    ax[0,0].bar(bin_centers, r_hist[0])
    ax[0,0].set_xticks(range(0,256,25))
    ax[0,0].set_title('Red Histogram')

    ax[0,1].bar(bin_centers, g_hist[0])
    ax[0,1].set_xticks(range(0,256,25))
    ax[0,1].set_title('Green Histogram')

    ax[0,2].bar(bin_centers, b_hist[0])
    ax[0,2].set_xticks(range(0,256,25))
    ax[0,2].set_title('Blue Histogram')

    ax[1,0].imshow(r_channel, 'gray')
    ax[1,0].set_title('Red Channel')
    ax[1,0].set_axis_off()

    ax[1,1].imshow(g_channel, 'gray')
    ax[1,1].set_title('Green Channel')
    ax[1,1].set_axis_off()

    ax[1,2].imshow(b_channel, 'gray')
    ax[1,2].set_title('Blue Channel')
    ax[1,2].set_axis_off()


    
def show_hist_hsv_img(rgb_inp_img):
    """
    plot histogram of H, S and V channel for the RGB input image,
    compared with the H, S and V images.
    """

    # Convert to HSV
    hsv = cv2.cvtColor(rgb_inp_img, cv2.COLOR_RGB2HSV)
    
    # HSV channels
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]

    h_hist = np.histogram(h, bins=32, range=(0, 256))
    s_hist = np.histogram(s, bins=32, range=(0, 256))
    v_hist = np.histogram(v, bins=32, range=(0, 256))

    # Generating bin centers
    bin_edges = h_hist[1]
    bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2

    f, ax = plt.subplots(2, 3, figsize=(20, 10))

    ax[0,0].bar(bin_centers, h_hist[0])
    ax[0,0].set_xticks(range(0,256,25))
    ax[0,0].set_title('H Histogram')

    ax[0,1].bar(bin_centers, s_hist[0])
    ax[0,1].set_xticks(range(0,256,25))
    ax[0,1].set_title('S Histogram')

    ax[0,2].bar(bin_centers, v_hist[0])
    ax[0,2].set_xticks(range(0,256,25))
    ax[0,2].set_title('V Histogram')

    ax[1,0].imshow(h, 'gray')
    ax[1,0].set_title('H Channel')
    ax[1,0].set_axis_off()

    ax[1,1].imshow(s, 'gray')
    ax[1,1].set_title('S Channel')
    ax[1,1].set_axis_off()

    ax[1,2].imshow(v, 'gray')
    ax[1,2].set_title('V Channel')
    ax[1,2].set_axis_off()

