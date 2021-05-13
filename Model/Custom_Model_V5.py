#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:19:51 2019

@author: sebastian
"""

import numpy as np
from math import ceil, floor

from PreprocesingV3 import read
import tensorflow as tf
import keras
import pandas as pd

import sys

from keras_applications.resnet import ResNet50
from keras.applications import vgg16

from sklearn.model_selection import ShuffleSplit, KFold, StratifiedKFold
from keras.layers import Dense,Dropout,Flatten,Convolution2D,MaxPooling2D, BatchNormalization
from keras.utils.np_utils import to_categorical 
from sklearn.metrics import f1_score, recall_score,precision_score, accuracy_score
from efficientnet.keras import EfficientNetB2
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


test_images_dir = 'stage_2_train/'
train_images_dir = 'stage_1_train_images/'

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
            keep_prob = self.labels.iloc[:, 0].map({0: 1, 1: 1})
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
        
        
class TestGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, batch_size=1, img_size=(512, 512, 3), 
                 img_dir=train_images_dir, *args, **kwargs):

        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.on_epoch_end()

    def __len__(self):
        return int(ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indices]
        
        X = self.__data_generation(list_IDs_temp)
        return X
        
    def on_epoch_end(self):
        self.indices = np.arange(len(self.list_IDs))

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.img_size))
                
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = read(self.img_dir+ID+".dcm", self.img_size)
        
        return X
        
        
from keras import backend as K

def weighted_log_loss(y_true, y_pred):
    """
    Can be used as the loss function in model.compile()
    ---------------------------------------------------
    """
    
    class_weights = np.array([2., 1., 1., 1., 1., 1.])
    
    eps = K.epsilon()
    
    y_pred = K.clip(y_pred, eps, 1.0-eps)

    out = -(         y_true  * K.log(      y_pred) * class_weights
            + (1.0 - y_true) * K.log(1.0 - y_pred) * class_weights)
    
    return K.mean(out, axis=-1)


def _normalized_weighted_average(arr, weights=None):
    """
    A simple Keras implementation that mimics that of 
    numpy.average(), specifically for this competition
    """
    
    if weights is not None:
        scl = K.sum(weights)
        weights = K.expand_dims(weights, axis=1)
        return K.sum(K.dot(arr, weights), axis=1) / scl
    return K.mean(arr, axis=1)


def weighted_loss(y_true, y_pred):
    """
    Will be used as the metric in model.compile()
    ---------------------------------------------
    
    Similar to the custom loss function 'weighted_log_loss()' above
    but with normalized weights, which should be very similar 
    to the official competition metric:
        https://www.kaggle.com/kambarakun/lb-probe-weights-n-of-positives-scoring
    and hence:
        sklearn.metrics.log_loss with sample weights
    """
    
    class_weights = K.variable([2., 1., 1., 1., 1., 1.])
    
    eps = K.epsilon()
    
    y_pred = K.clip(y_pred, eps, 1.0-eps)

    loss = -(        y_true  * K.log(      y_pred)
            + (1.0 - y_true) * K.log(1.0 - y_pred))
    
    loss_samples = _normalized_weighted_average(loss, class_weights)
    
    return K.mean(loss_samples)


def weighted_log_loss_metric(trues, preds):
    """
    Will be used to calculate the log loss 
    of the validation set in PredictionCheckpoint()
    ------------------------------------------
    """
    class_weights = [2., 1., 1., 1., 1., 1.]
    
    epsilon = 1e-7
    
    preds = np.clip(preds, epsilon, 1-epsilon)
    loss = trues * np.log(preds) + (1 - trues) * np.log(1 - preds)
    loss_samples = np.average(loss, axis=1, weights=class_weights)

    return - loss_samples.mean()

class PredictionCheckpoint(keras.callbacks.Callback):
    
    def __init__(self, test_df, valid_df, 
                 test_images_dir=test_images_dir, 
                 valid_images_dir=train_images_dir, 
                 batch_size=32, input_size=(224, 224, 3)):
        
        self.test_df = test_df
        self.valid_df = valid_df
        self.test_images_dir = test_images_dir
        self.valid_images_dir = valid_images_dir
        self.batch_size = batch_size
        self.input_size = input_size
        
    def on_train_begin(self, logs={}):
        self.test_predictions = []
        self.valid_predictions = []
        
    def on_epoch_end(self,batch, logs={}):
        self.test_predictions.append(
            self.model.predict_generator(
                DataGenerator(self.test_df.index, None, self.batch_size, self.input_size, self.test_images_dir), verbose=1)[:len(self.test_df)])

#        self.valid_predictions.append(
#            self.model.predict_generator(
#                DataGenerator(self.valid_df.index, None, self.batch_size, self.input_size, self.valid_images_dir), verbose=2)[:len(self.valid_df)])
#        
#        print("validation loss: %.4f" %
#               weighted_log_loss_metric(self.valid_df.values, 
#                                    np.average(self.valid_predictions, axis=0, 
#                                               weights=[2**i for i in range(len(self.valid_predictions))])))
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
        engine = EfficientNetB2(include_top=False, input_shape=(*self.input_dims[:2], 3),
                                    classes=1)
        x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(engine.output)
        out = keras.layers.Dense(1, activation="sigmoid", name='dense_output')(x)
        self.model = keras.models.Model(inputs=engine.input, outputs=out)
        self.model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(), metrics=[weighted_loss,'mse','acc'])
    

    def fit_and_predict(self, train_df, valid_df, test_df):
        
        # callbacks
        pred_history = PredictionCheckpoint(test_df, valid_df, input_size=self.input_dims)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=self.verbose, patience=5)
        #checkpointer = keras.callbacks.ModelCheckpoint(filepath='%s-{epoch:02d}.hdf5' % self.engine.__name__, verbose=1, save_weights_only=True, save_best_only=False)
        scheduler = keras.callbacks.LearningRateScheduler(lambda epoch: self.learning_rate * pow(self.decay_rate, floor(epoch / self.decay_steps)))
        
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
 #           use_multiprocessing=True,
            validation_data= DataGenerator(
                valid_df.index, 
                valid_df, 
                self.batch_size, 
                self.input_dims, 
                train_images_dir
            ),
            workers=4,
            callbacks=[es,pred_history, scheduler]
        )
        
        return pred_history
        
            
    def mymodel_predict(self, test_df):
        
        self.model.predict_generator(
            TestGenerator(
                test_df.index,
                None,
                self.batch_size, 
                self.input_dims, 
                train_images_dir
            ),
#            use_multiprocessing=True,
            workers=4,
            verbose=self.verbose
        )
        
        
        
    
    def save(self, path):
        self.model.save_weights(path)
    
    def load(self, path):
        self.model.load_weights(path)
        
#from Read_csv import read_testset, read_trainset

df = pd.read_csv('/home/sebastian/RNSA/TTS/complete/train_data_0.csv', index_col=0)
test_df = pd.read_csv('/home/sebastian/RNSA/TTS/complete/test_data_0.csv', index_col=0)


# # train set (80%) and validation set (20%)
# ss = ShuffleSplit(n_splits=10, test_size=0.1).split(df.index)

# # lets go for the second fold instead of the first one
# next(ss)
# train_idx, valid_idx = next(ss)


# ss = ShuffleSplit(n_splits=10, test_size=0.3).split(test_df.index)

# # lets go for the second fold instead of the first one
# next(ss)
# t, test_idx = next(ss)

kf = KFold(n_splits = 5)
                         
skf = StratifiedKFold(n_split = 5, random_state = 7, shuffle = True)

save_dir = '/saved_models/'
fold_var = 1 

for train_index, test_index in kf.split(df.index, df):
    
    # obtain model
    model = MyDeepModel(engine=ResNet50, input_dims=(224, 224, 3), batch_size=32, learning_rate=5e-4, 
                        num_epochs=10, decay_rate=0.8, decay_steps=1, weights="ModelV5_Weights.h5", verbose=1)
    
    # obtain test + validation predictions (history.test_predictions, history.valid_predictions)
    
    #history = model.fit_and_predict(df.iloc[train_idx], df.iloc[valid_idx], df.iloc[test_idx])
    
    history = model.fit_and_predict(df.iloc[train_idx], df.iloc[valid_idx], test_df.iloc[test_idx])

    pred_df = test_df.iloc[test_idx].reset_index(col_level=1)
    pred_df.set_index(('','Image'),inplace=True)

    pred_df.iloc[:,:] = np.average(history.test_predictions, axis=0, weights=[2**i for i in range(len(history.test_predictions))])

#
#test_df.insert(loc=0, column='ID', value=test_df['Image'].astype(str) + "_" + test_df['Diagnosis'])
#
#test_df = test_df.drop(["Image", "Diagnosis"], axis=1)

    pred_df.to_csv('prediction_Efficient-0.csv', index=True)

    model.save('ModelV5_Weights.h5')
