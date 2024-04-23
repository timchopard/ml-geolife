""" from helper import tif_helper
"""
import cv2 
import numpy as np

def tif_helper(filepath:str):
    image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    min_val = image.min()

    print(min_val)
    
    if min_val > 0:
        return image 
    
    previous_value = (image * (image != min_val)).mean()
    dims = image.shape

    for ii in range(dims[0]):
        for jj in range(dims[1]):
            if image[ii, jj] == min_val:
                image[ii, jj] = previous_value
            previous_value = image[ii, jj]

    # if filepath

    return image