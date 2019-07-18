#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 14:06:02 2018

@author: Rodrigo Castro
"""

from music21 import *
import numpy as np



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
    Correcting | Cmaj7 |  Rest | where it should be | Cmaj7 |  %  |
    '''    
    chordSequence = midi_data[1].recurse().getElementsByClass(['Chord','Rest'])
    n=len(chordSequence)
    
    #dictionary of chords that will replace the rests:
    replacingChords = {}    

    #I made sure the solos in the corpus do not start without an initial chord, but just in case:                                                    
    if chordSequence[0].isRest:                                                 
            print('The beginning of the progression does not have a chord!')     
    else:
        for i in range(1,n):
            if chordSequence[i].isRest:
                location = chordSequence[i].offset                                
                length = chordSequence[i].quarterLength                           
                chordNotes = list(chordSequence[i-1].pitches)                     
                midi_data[1].remove(chordSequence[i])                             
                replacingChords['chord{0}'.format(i)] = chord.Chord(chordNotes)   
                replacingChords['chord{0}'.format(i)].quarterLength = length
                midi_data[1].insert(location,replacingChords['chord{0}'.format(i)])  

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


    
def melody2matrix(solo):
    midi_data = parse_midi(solo)
    #Indices: 0-127: midi pitch, 128: rest?, 129-225 :note length
    note_embedding_size = 225  
    duration_LCD = 12
    max_note_length = 96
    melody,progression = midi_data.getElementsByClass('Part')
    melodyMatrix = np.zeros([note_embedding_size,0])
    for nota in melody.recurse().getElementsByClass(['Note','Rest']):
        note_vect = np.zeros([note_embedding_size,1])           
        if nota.isRest:
            height = 128
        else:
            height = nota.pitch.midi
        note_vect[height] = 1
        length = min(duration_LCD*float(nota.quarterLength),max_note_length)               
        duration_idx = int(length)                                  
        note_vect[129+duration_idx] = 1                            
        melodyMatrix = np.append(melodyMatrix,note_vect,axis=1)
        
    return melodyMatrix
 

def matrix2melody(melodyMatrix):
    m,n = melodyMatrix.shape
    melodyStream = stream.Stream()
    
    #if you wnat to impose a time and key signature add:
    #melodyStream.timeSignature = meter.TimeSignature('4/4')
    #melodyStream.keySignature = key.Key('C')
    
    for i in range(m):
        vector = melodyMatrix[:,i]
        nota = vect2note(vector)
        melodyStream.append(nota)
        
    return melodyStream

'''
Obtain probability distributions for pitch and duration from predictions
'''
def get_prob_dist(prediction):
    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            if prediction[i,j]<0:
                prediction[i,j]=-1000
    melody_prediction = softmax(prediction[:129,:])
    rhythm_prediction = softmax(prediction[129:,:])
    
    return melody_prediction , rhythm_prediction