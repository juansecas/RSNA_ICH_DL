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

def _normalize(x):
    x_max = x.max()
    x_min = x.min()
    if x_max != x_min:
        z = (x - x_min) / (x_max - x_min)
        return z
    return np.zeros(x.shape)

def crop(image):
    image2 = image[60:440,60:440]
    return image2

def sigmoid_window(img, window_center, window_width, U=1.0, eps=(1.0 / 255.0), desired_size=(256, 256)):
    intercept, slope = img.RescaleIntercept, img.RescaleSlope
    img = img.pixel_array * slope + intercept
    
    # resizing already to save computation
    img = crop(img)
    img = cv2.resize(img, desired_size[:2], interpolation=cv2.INTER_CUBIC)
    
    ue = log((U / eps) - 1.0)
    W = (2 / window_width) * ue
    b = ((-2 * window_center) / window_width) * ue
    z = W * img + b
    img = U / (1 + np.power(np.e, -1.0 * z))
    
    img = _normalize(img)
    
    return img

def sigmoid_bsb_window(img, desired_size):
    brain_img = sigmoid_window(img, 40, 80, desired_size=desired_size)
    subdural_img = sigmoid_window(img, 80, 200, desired_size=desired_size)
    bone_img = sigmoid_window(img, 40, 380, desired_size=desired_size)
    
    bsb_img = np.zeros((brain_img.shape[0], brain_img.shape[1], 3))
    bsb_img[:, :, 0] = brain_img
    bsb_img[:, :, 1] = subdural_img
    bsb_img[:, :, 2] = bone_img
    return bsb_img





def read(path, desired_size):
    """Will be used in DataGenerator"""
    
    dcm = pydicom.dcmread(path)
    
    try:
        img = sigmoid_bsb_window(dcm, desired_size)
    except:
        img = np.zeros(desired_size)
    
    return img

#image = read("C:\\Users\\Wael\\Documents\\sebas\\stage_1_train_images\\ID_6064dd119.dcm", desired_size=(224,224, 3))
#plt.imshow(image)
