#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:19:51 2019

@author: sebastian
"""

import numpy as np
import pandas as pd
import collections
from datetime import datetime
from math import ceil, floor

from Preprocesing import read
import tensorflow as tf
import keras

import sys

from keras_applications.resnet import ResNet50
from keras.applications import vgg16

from sklearn.model_selection import ShuffleSplit
from keras.layers import Dense,Dropout,Flatten,Convolution2D,MaxPooling2D, BatchNormalization
from keras.utils.np_utils import to_categorical 
from sklearn.metrics import f1_score, recall_score,precision_score, accuracy_score
from efficientnet.keras import EfficientNetB0
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint


train_images_dir = 'C:\\Users\\Wael\\Documents\\sebas\\stage_1_train_images\\'
test_images_dir = 'C:\\Users\Wael\\Documents\\sebas\\stage_1_test_images\\'

class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, labels=None, batch_size=1, img_size=(512, 512, 3), 
                 img_dir=train_images_dir, *args, **kwargs):

        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.on_epoch_end()

    def __len__(self):
        return int(ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indices]
        
        if self.labels is not None:
            X, Y = self.__data_generation(list_IDs_temp)
            return X, Y
        else:
            X = self.__data_generation(list_IDs_temp)
            return X
        
    def on_epoch_end(self):
        
        
        if self.labels is not None: # for training phase we undersample and shuffle
            # keep probability of any=0 and any=1
            keep_prob = self.labels.iloc[:, 0].map({0: 0.4, 1: 0.8})
            keep = (keep_prob > np.random.rand(len(keep_prob)))
            self.indices = np.arange(len(self.list_IDs))[keep]
            np.random.shuffle(self.indices)
        else:
            self.indices = np.arange(len(self.list_IDs))

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.img_size))
        
        if self.labels is not None: # training phase
            Y = np.empty((self.batch_size, 6), dtype=np.float32)
        
            for i, ID in enumerate(list_IDs_temp):
                X[i,] = read(self.img_dir+ID+".dcm", self.img_size)
                Y[i,] = self.labels.loc[ID].values
        
            return X, Y
        
        else: # test phase
            for i, ID in enumerate(list_IDs_temp):
                X[i,] = read(self.img_dir+ID+".dcm", self.img_size)
            
            return X

class MyDeepModel:
    
    def __init__(self, engine, input_dims, batch_size=5, num_epochs=4, learning_rate=1e-3, 
                 decay_rate=1.0, decay_steps=1, weights="imagenet", verbose=1):
        
        self.engine = engine
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.weights = weights
        self.verbose = verbose
        self._build()

    def _build(self):
        
#        inputs = keras.layers.Input((*self.input_dims, 1))
#        x = keras.layers.Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), name="initial_conv2d")(inputs)
#        x = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='initial_bn')(x)
#        x = keras.layers.Activation('relu', name='initial_relu')(x)
#             
        engine = EfficientNetB0(include_top=False, input_shape=(*self.input_dims[:2], 3),
                                    classes=6)
        
        x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(engine.output)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(keras.backend.int_shape(x)[1], activation="relu", name="dense_hidden_1")(x)
        x = keras.layers.Dropout(0.1)(x)
        out = keras.layers.Dense(6, activation="sigmoid", name='dense_output')(x)
        
        self.model = keras.models.Model(inputs=engine.input, outputs=out)
                                   
        self.model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['acc', 'mse'])
        
    

    def fit(self, train_df, valid_df):
        
        # callbacks
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=self.verbose, patience=100)
        mc = ModelCheckpoint('best_model_VGG.h5', monitor='val_loss', mode='min', verbose=self.verbose, save_best_only=True)
        
        self.model.fit_generator(
            DataGenerator(
                train_df.index, 
                train_df, 
                self.batch_size, 
                self.input_dims, 
                train_images_dir
            ),
            epochs=self.num_epochs,
            verbose=self.verbose,
#            use_multiprocessing=True,
            callbacks=[es, mc],
            validation_data= DataGenerator(
                valid_df.index, 
                valid_df, 
                self.batch_size, 
                self.input_dims, 
                train_images_dir
            ),
            workers=4
        )
            
    def predict(self, test_df):
        
        self.model.predict_generator(
                DataGenerator(
                test_df.index, 
                self.batch_size, 
                self.input_dims, 
                test_images_dir
            ),
            workers=4,
            verbose=self.verbose
        )
        
        
    
    def save(self, path):
        self.model.save_weights(path)
    
    def load(self, path):
        self.model.load_weights(path)
        
from Read_csv import read_testset, read_trainset
#
test_df = read_testset(filename = "stage_1_sample_submission.csv")
df = read_trainset(filename = "stage_1_train.csv")



# train set (90%) and validation set (10%)
ss = ShuffleSplit(n_splits=10, test_size=0.1, random_state=42).split(df.index)

# lets go for the second fold instead of the first one
next(ss)
train_idx, valid_idx = next(ss)

# obtain model
model = MyDeepModel(engine=ResNet50, input_dims=(224, 224, 3), batch_size=32, learning_rate=5e-4, 
                    num_epochs=30, decay_rate=0.8, decay_steps=1, weights="imagenet", verbose=1)

# obtain test + validation predictions (history.test_predictions, history.valid_predictions)
history = model.fit(df.iloc[train_idx], df.iloc[valid_idx])
pred = model.predict(test_df)
#%%
test_df.iloc[:, :] = np.average(history.test_predictions, axis=0, weights=[2**i for i in range(len(history.test_predictions))])

test_df = test_df.stack().reset_index()

test_df.insert(loc=0, column='ID', value=test_df['Image'].astype(str) + "_" + test_df['Diagnosis'])

test_df = test_df.drop(["Image", "Diagnosis"], axis=1)

test_df.to_csv('submission_Efficient.csv', index=False)