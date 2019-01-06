#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 14:06:02 2018

@author: Amillo
"""

from music21 import *
import numpy as np

#-------------------------- HELPER FUNCTIONS ------------------------------------# 

'''
Parse the corpus and fix potential problems
'''
def parse_midi(solo):
    midi_data = converter.parse(solo)
   
    '''
    Making sure the last rest at the end of the solo is accounted for
    '''   
    #split into solo part and progression part:
    soloPart, progressionPart = midi_data.getElementsByClass('Part')
    
    #get lengths:
    initial_soloLength = soloPart.quarterLength
    progressionLength = progressionPart.quarterLength
    
    if initial_soloLength < progressionLength:
        lastRestLength = progressionLength - initial_soloLength
        lastRest = note.Rest()
        lastRest.quarterLength = lastRestLength
        midi_data[0].append(lastRest)
    
    if initial_soloLength > progressionLength:
        print('Solo part is longer than chord progression')
    
    assert midi_data[0].quarterLength == midi_data[1].quarterLength
            
    '''
    Making sure the solo contains only supported note lengths 
            
    durationTypes = set()     
    
    for nota in midi_data.recurse().getElementsByClass(note.Note):
        durationTypes.add(str(nota.quarterLength))
    
    #durations supported in the training model        
    allDurationTypes = {'0.0','0.25','0.5','0.75','1.0','1.25','1.5',\
                      '1.75','1/3','10/3','2.0','2.5','2.75','2/3',\
                      '3.0','3.5','4.0','4.5','4/3','5.0','5.5','5/3',\
                      '7/3','8/3'}
    
    if durationTypes.issubset(allDurationTypes):
        print('All note lengths in the solo are supported.')
    else:
        print('The solo contains some note lengths not supported. Clean it first.')
    '''
       
    #get rid of unsupported durations:
        #pending
        
    '''
    Correcting | Cmaj7 |  Rest | where there should be | Cmaj7 |  %  |
    '''    
    chordSequence = midi_data[1].recurse().getElementsByClass(['Chord','Rest'])
    n=len(chordSequence)
    
    replacingChords = {}                                                        #dictionary of chords that will replace the rests
    if chordSequence[0].isRest:                                                 #I made sure the solos in the corpus do not start without an initial chord, but just in case
            print('The beginning of the progression does not have a chord!')     
    else:
        for i in range(1,n):
            if chordSequence[i].isRest:
                location = chordSequence[i].offset                                #position where we will put the replacing chord
                length = chordSequence[i].quarterLength                           #length of the replacing chord
                chordNotes = list(chordSequence[i-1].pitches)                     #the replacing chord should have the same notes as the previous
                midi_data[1].remove(chordSequence[i])                             #remove the rest
                replacingChords['chord{0}'.format(i)] = chord.Chord(chordNotes)   #creating replacing chord
                replacingChords['chord{0}'.format(i)].quarterLength = length
                midi_data[1].insert(location,replacingChords['chord{0}'.format(i)])  #insert replacing chord

    return midi_data


'''
Converts a two-hot vector into a note or rest
'''
def vect2note(vector):
    note_embedding_size = 225 
    assert np.shape(vector) == (note_embedding_size,)
    duration = int(np.argwhere(vector[129:]))/12.0
    if vector[128] == 1:
        nota = note.Rest()
        nota.quarterLength = duration
    else:
        height = int(np.argwhere(vector[:128]))
        nota = note.Note()
        nota.pitch.midi = height
        nota.quarterLength = duration
    
    return nota


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

#-------------------------- PUPLIC FUNCTIONS ---------------------------------#
    
def melody2matrix(solo):
    midi_data = parse_midi(solo)
    #Indices: 0-127: midi pitch, 128: rest?, 129-225 :note length
    note_embedding_size = 225  
    melody,progression = midi_data.getElementsByClass('Part')
    melodyMatrix = np.zeros([note_embedding_size,0])
    for nota in melody.recurse().getElementsByClass(['Note','Rest']):
        note_vect = np.zeros([note_embedding_size,1])           
        if nota.isRest:
            height = 128
        else:
            height = nota.pitch.midi
        note_vect[height] = 1
        length = min(12*float(nota.quarterLength),96)               #mapping durations to integers 0,1,2,3,4,6,...; max duration=96 (2 measures)
        duration_idx = int(length)                                  #mapping the integers above to 0,1,2,3,4,5,6,7,8,etc
        note_vect[129+duration_idx] = 1                            #the duration encoding begins at entry 129
        melodyMatrix = np.append(melodyMatrix,note_vect,axis=1)
        
    return melodyMatrix
 

def matrix2melody(melodyMatrix):
    m,n = melodyMatrix.shape
    melodyStream=stream.Stream()
    #melodyStream.timeSignature = meter.TimeSignature('4/4')
    #melodyStream.keySignature = key.Key('C')
    for i in range(m):
        vector = melodyMatrix[:,i]
        nota = vect2note(vector)
        melodyStream.append(nota)
        
    return melodyStream