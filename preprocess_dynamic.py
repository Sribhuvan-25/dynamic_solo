#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 12:49:32 2018

@author: Rodrigo Castro
"""

from music21 import *
import numpy as np
from my_utils import parse_midi
import torch
from torch.autograd import Variable
import random


#-------------------------- HELPER FUNCTIONS ------------------------------------# 


#Splits the solo into smaller training windows
#make sure the piece is in 4/4
def split_solo(midi_data , window_size):
        
    #getting the total number of measures of the solo:
    totalMeasures = int(midi_data.quarterLength / 4)
    if window_size == 'all':
        windowSize = totalMeasures        
    else:
        windowSize = window_size 
    
    totalWindows = int(totalMeasures - windowSize + 1)
    
    print('The solo contains '+str(totalMeasures)+' measures. '\
          'We will divide the solo into '+str(totalWindows)+' smaller (sliding) windows '\
          'of '+str(windowSize)+' measures each.')
    
    #get training windows:
    trainingWindows = []                                                        #list of a collection of mini stream.Score
    for i in range(0,totalWindows):
        beginWindow = i + 1
        endWindow = beginWindow+windowSize - 1
        print('Fetching training window '+str(beginWindow)+'/'+str(totalWindows)+'...')
        trainingWindows.append(midi_data.measures(beginWindow,endWindow))
              
    #Transposing to all keys up and down      
    transposedTrainingWindows = []
    for intervalo in range(0,12):
        if intervalo == 0:
            continue
        print('transposing all training windows '+str(intervalo)+' half steps...')
        for elemento in trainingWindows:
            transposedTrainingWindows.append(elemento.transpose(intervalo))
      
    allTrainingWindows = trainingWindows + transposedTrainingWindows

    return allTrainingWindows


#Converts each window to multi-hot encodings
def encode_windows(allTrainingWindows):
    chord_embedding_size = 24                                                   #24x1  0-11:root, 12-23: chord notes
    note_embedding_size = 225                                                   #0-127: midi pitch, 128: rest?, 129-201 :note length
    
    count = 0
    m = len(allTrainingWindows)
    melodyTrain = []                                                           
    progressionTrain = []                                                        
    for window in allTrainingWindows:
        melody,progression = window.getElementsByClass('Part')
        melodyMatrix = np.zeros([note_embedding_size,0])
        progressionMatrix = np.zeros([chord_embedding_size,0])
        for nota in melody.recurse().getElementsByClass(['Note','Rest']):
            note_vect = np.zeros([note_embedding_size,1])           
            if nota.isRest:
                height = 128
            else:
                height = nota.pitch.midi
            note_vect[height] = 1
            length = min(12*float(nota.quarterLength),96)                       #mapping durations to integers 0,1,2,3,4,6,...; max duration=96 (2 measures)
            duration_idx = int(length)                                          #mapping the integers above to 0,1,2,3,4,5,6,7,8,etc
            note_vect[129+duration_idx] = 1                                     #the duration encoding begins at entry 129
            melodyMatrix = np.append(melodyMatrix,note_vect,axis=1)
        melodyTrain.append(melodyMatrix)
            
        for acorde in progression.recurse().getElementsByClass(chord.Chord):
            chord_vect = np.zeros([chord_embedding_size,1])
            chord_length = int(acorde.quarterLength)                            #all chords in the corpus should have an integer length
            chord_offset = int(acorde.offset)          
            root_idx = acorde.root().midi % 12
            chord_vect[root_idx] = 1
            notes=[p.midi for p in acorde.pitches]
            for i in range(len(notes)):
                idx = notes[i]%12
                chord_vect[12 + idx] = 1
            for j in range(chord_length):   
                progressionMatrix = np.append(progressionMatrix,chord_vect,axis=1)
        progressionTrain.append(progressionMatrix)
        count += 1 
        if count%100 == 0:
            print(str(count)+' windows of '+str(m)+' have been encoded...')
                    
    return melodyTrain , progressionTrain
                

#---------------------------- PUBLIC FUNCTIONS ---------------------------------#


#Solo data in Numpy format (notes and chords in column form):
def preprocess_solo(solo , window_size ):
    midi_data = parse_midi(solo)
    allTrainingWindows = split_solo(midi_data , window_size)
    melodyTrain , progressionTrain = encode_windows(allTrainingWindows)
    AllTrain = list( zip(melodyTrain,progressionTrain) )
    random.shuffle(AllTrain)
    melodyTrain , progressionTrain = zip(*AllTrain)
    
    return melodyTrain, progressionTrain


#Solo data in Pytorch format (notes and chords in row form):
def get_data(solo , window_size):
    melodyTrain, progressionTrain = preprocess_solo(solo , window_size )

    progressionTrain = np.stack(progressionTrain)
    event_data = Variable(torch.from_numpy(progressionTrain))
    E = event_data.transpose(1,2)
    E = E.type(torch.FloatTensor)
    
    S=[]
    for window in melodyTrain:
        window = np.stack(window)
        window = Variable(torch.from_numpy(window))
        window = window.transpose(0,1)
        window = window.type(torch.FloatTensor)
        S.append(window)
   
    return E , S


# select window_size = 'all' to make the whole solo a single training example
def save_data( solo , window_size , filename):  
    E , S =  get_data(solo , window_size )
    Training_data = [ E , S ]
    torch.save(Training_data, filename)