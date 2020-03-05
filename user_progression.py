#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:58:57 2019

@author: Rodrigo Castro
"""

import numpy as np
import torch


def enter_progression(event_emb_size, beats_per_measure=4):
    
    chord_vect_size = event_emb_size
    progression_matrix  = np.zeros([0, chord_vect_size])
    progression_symbols = []
    chord_list = []
    
    print('How many bars long is your progression? (enter an integer): ')
    bars = int(input()) 
    total_beats = beats_per_measure * bars    
    for beat in range(0, total_beats):                              
        progression_symbols.append('?')
    
    progression_display = ''    
    progression_display_length = total_beats + bars + 1
    temp_display = progression_symbols[:]    
    for i in range(0, progression_display_length, 5):                                      
        temp_display.insert(i,'|')        
                                                  
    for i in range(len(temp_display)):
        progression_display += temp_display[i] + ' '
        
    print('Progression status: ' + progression_display )
    
    counter = 0
    while True:
        chord_vector = np.zeros([1, chord_vect_size])
        
        print('Enter a chord:')
        chord_name = str(input())
        
        print('How many beats of that chord?')
        chord_duration = int(input())
        total_duration = counter + chord_duration
        if total_duration > total_beats:
            break
        else:
            chord = harmony.ChordSymbol(chord_name)
            chord.quarterLength = chord_duration
            chord_list.append(chord)
           
            root_idx       = chord.root().midi % 12
            chord_pitches  = [p.midi for p in chord.pitches]
            chord_vector[0,root_idx] = 1
            for i, chord_pitch in enumerate(chord_pitches):
                chord_pitch_idx = chord_pitch % 12
                chord_vector[0, 12 + chord_pitch_idx] = 1
            for j in range(chord_duration):
                progression_matrix = np.append(progression_matrix, chord_vector, axis=0)
                
            for i in range(counter, total_duration):
                progression_symbols[i] = chord_name              
            temp_display = progression_symbols[:] 
            
            for i in range(0, progression_display_length, 5):                                      
                temp_display.insert(i,'|')
                
            progression_display = ''
            for i in range(len(temp_display)):
                progression_display += temp_display[i] + ' '
                
            print('Progression: ' + progression_display)
            counter = total_duration
            if counter == total_beats:
                break
            else:
                continue
            
    chord_matrix = torch.from_numpy(progression_matrix)
    chord_matrix = chord_matrix.type(torch.FloatTensor)
        
    return progression_display , chord_matrix , chord_list


#progression, chord_matrix, chord_list = enter_progression(event_emb_size, beats_per_measure=4)