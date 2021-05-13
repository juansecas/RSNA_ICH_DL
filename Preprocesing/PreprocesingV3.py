#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:44:16 2019

@author: sebastian
"""
import numpy as np
import pydicom
import cv2
import matplotlib.pyplot as plt
#from albumentations import Compose, CenterCrop, ShiftScaleRotate, RandomBrightnessContrast, HorizontalFlip
from math import log
from scipy import ndimage

def normalize(x):
    x_max = x.max()
    x_min = x.min()
    if x_max != x_min:
        z = (x - x_min) / (x_max - x_min)
        return z
    return np.zeros(x.shape)

def refine_label(label_mask):
    label_mask = label_mask.astype(np.bool)
    # Fill hole
    label_mask = ndimage.binary_fill_holes(label_mask)
    # Get largest connected component
    label_im, nb_labels = ndimage.label(label_mask)
    sizes = ndimage.sum(label_mask, label_im, range(nb_labels + 1))
    mask_size = sizes < max(sizes)
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0
    labels = np.unique(label_im)
    label_mask = np.searchsorted(labels, label_im)
    return label_mask


def autocropmin(image, threshold=0, kernsel_size = 10):
        
    img = image.copy()
    SIZE = img.shape[0]
    imgfilt = ndimage.minimum_filter(img.max((2)), size=kernsel_size)
    rows = np.where(np.max(imgfilt, 0) > threshold)[0]
    cols = np.where(np.max(imgfilt, 1) > threshold)[0]
    row1, row2 = rows[0], rows[-1] + 1
    col1, col2 = cols[0], cols[-1] + 1
    row1, col1 = max(0, row1-kernsel_size), max(0, col1-kernsel_size)
    row2, col2 = min(SIZE, row2+kernsel_size), min(SIZE, col2+kernsel_size)
    image = image[col1: col2, row1: row2]
    #logger.info(image.shape)
    sqside = max(image.shape)
    imageout = np.zeros((sqside, sqside, 3), dtype = 'uint8')
    imageout[:image.shape[0], :image.shape[1],:] = image.copy()
    return imageout


def pre_preocessing(image, pad_size=(512, 512)):
    # Convert to [0, 255]
    # image = (image-image.min()) / (image.max() - image.min())
    # image= image*255
    image[image < 0] = 0
    # Remove unwanted region
    mask = image > 0
    mask = refine_label(mask)
    image = image * mask
    # Center crop and pad to size
    # mask = image>0
    # min_H_s, max_H_e, min_W_s, max_W_e = cut_edge(mask, 32)
    # image = image[min_H_s: max_H_e, min_W_s:max_W_e]
    # Pad to size
    H, W = image.shape
    pad_H, pad_W = pad_size[0], pad_size[1]
    pad_H0 = max((pad_H - H) // 2, 0)
    pad_H1 = max(pad_H - H - pad_H0, 0)
    pad_W0 = max((pad_W - W) // 2, 0)
    pad_W1 = max(pad_W - W - pad_W0, 0)
    image = np.pad(image, [(pad_H0, pad_H1), (pad_W0, pad_W1)], mode='constant', constant_values=0)
    return image


def apply_window(dicom, center, width, desired_size=(224,224)):
    intercept, slope = dicom.RescaleIntercept, dicom.RescaleSlope
    image = dicom.pixel_array * slope + intercept
    
    image = image.copy()
    min_value = center - width // 2
    max_value = center + width // 2
    image[image < min_value] = min_value
    image[image > max_value] = max_value
#    image = cv2.resize(image, desired_size[:2], interpolation=cv2.INTER_CUBIC)
 #   image = autocropmin(image)
    return image

def apply_window_policy(image, desired_size=(224,224)):
        
    image1 = apply_window(image, 40, 80, desired_size) # brain
    image2 = apply_window(image, 80, 200, desired_size) # subdural
    image3 = apply_window(image, 40, 380, desired_size) # bone
    image1 = pre_preocessing(image1, pad_size= desired_size)
    image2 = pre_preocessing(image2, pad_size= desired_size)
    image2 = pre_preocessing(image3, pad_size= desired_size)
    image1 = normalize(image1)
    image2 = normalize(image2)
    image3 = normalize(image3)
    image = np.array([
        image1,
        image2,
        image3,
    ]).transpose(1,2,0)

    return image






def read(path, desired_size):
    """Will be used in DataGenerator"""
    
    dcm = pydicom.dcmread(path)
    
#    try:
    img = apply_window_policy(dcm, desired_size)
    img = (255*img).astype(np.uint8)
    #img = autocropmin(img)
    #img = autocropmin(img, threshold=0, kernsel_size = 10)
    #img = cv2.resize(img, desired_size[:2], interpolation=cv2.INTER_CUBIC)
#    except:
#        img = np.zeros(desired_size)
    
    return img

image2 = read("/home/sebastian/Descargas/ID_000039fa0.dcm", desired_size=(224,224, 3))
plt.imshow(image2[:,:,0], 'gray')
#plt.imshow(image2[:,:,0], 'gray')

#image = (255*image).astype(np.uint8)
#cv2.imwrite('imagen_prueba'+'.jpg', image)
#image2 = autocropmin(image)
#
#plt.imshow(image2)
