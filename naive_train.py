#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 13:54:54 2018

@author: Amillo
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from my_utils import melody2matrix, softmax


#-------------------------- HELPER FUNCTIONS ------------------------------------# 

def get_training_melody(solo,seq_length = 36):
    training_data = melody2matrix(solo)
    
    #add a column of zeros to be first input of the first training sequence:
    zero_column = np.zeros( (training_data.shape[0],1) )
    training_data = np.concatenate( (zero_column,training_data), axis=1 )
    
    #if you don't want rhythm:
    #training_data = training_data[:129,:]
    
    #notes as row vectors:
    training_data = np.transpose(training_data)
    notes_total , note_feature_size = training_data.shape
    
    stride = int(seq_length/2)
    seq_total = notes_total//stride-1
    
    X = np.empty(shape = (seq_total,seq_length-1,note_feature_size) )
    Y = np.empty(shape = (seq_total,seq_length-1,note_feature_size) )
    for i in range(seq_total):
        X[i,:,:] = training_data[ i*stride : i*stride+seq_length-1 , :]
        Y[i,:,:] = training_data[ i*stride + 1 : i*stride+seq_length, :]
        
    shuffled_order = [j for j in range(X.shape[0])]
    np.random.shuffle(shuffled_order)
    X_train = np.empty( shape = (seq_total,seq_length-1,note_feature_size) )
    Y_train = np.empty( shape = (seq_total,seq_length-1,note_feature_size) )
    for i in range( len(shuffled_order) ):
        X_train[i,:,:] = X[shuffled_order[i],:,:]
        Y_train[i,:,:] = Y[shuffled_order[i],:,:]
          
    return X_train , Y_train


def melody_model(X_train , Y_train):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(16, input_shape = (X_train.shape[1:]), activation = 'relu' , \
                                    return_sequences=True))
    model.add(tf.keras.layers.LSTM(16, activation = 'relu', return_sequences=True))
    model.add(tf.keras.layers.Dense(Y_train.shape[2], activation='tanh'))
    opt = tf.keras.optimizers.Adam(lr = .001)
    model.compile(loss='binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
    
    return model


#-------------------------- Public functions ---------------------------------#
    
def train_melody(solo, num_epochs):
    X_train , Y_train = get_training_melody(solo)
    model = melody_model(X_train , Y_train)
    tf.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,  
          write_graph=True, write_images=True)
    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    model.fit(X_train, Y_train, epochs = num_epochs,callbacks=[tbCallBack])
    
    return model
    
def predict_example(model, solo):
    X_train , Y_train = get_training_melody(solo)
    idx = np.random.int(0, X_train.shape[0])
    example = X_train[idx, :, :]
    ground_truth = Y_train[idx, :, :]  
    example = A.reshape(1,A.shape[0],A.shape[1])
    prediction = model.predict(example)
    
    #notes as column vectors:
    prediction = np.transpose(prediction[0,:,:])
    ground_truth = np.transpose(ground_truth[0,:,:])
       
    return prediction , ground_truth

'''
Obtains probability distributions for pitch and duration from predictions
'''
def get_prob_dist(prediction):
    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            if prediction[i,j]<0:
                prediction[i,j]=-1000
    melody_prediction = softmax(prediction[:129,:])
    rhythm_prediction = softmax(prediction[129:,:])
    
    return melody_prob , rhythm_prob
    