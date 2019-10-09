# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:02:10 2019

@author: Rodrigo Castro
"""

import os
from music21 import *
from fractions import Fraction
from random import shuffle
import numpy as np
import torch
from torch.autograd import Variable


#---------------------------------- HELPER FUNCTIONS --------------------------------------# 

def createSoloDict(directory_in_str):
    
    print('Creating a dictonary of all solos...')
    directory = os.fsencode(directory_in_str)
    solo_dict = {}
    for file in os.listdir(directory):
         filename = os.fsdecode(file)
         if filename.endswith('.xml') or filename.endswith('.musicxml'): 
             solo_name            = os.path.splitext(filename)[0]
             path                 = str(directory_in_str + '/' + filename)
             parsed_solo          = converter.parse(path)
             solo_dict[solo_name] = parsed_solo
             continue
         else:
             continue
    
    return solo_dict


def inspect_corpus(solo_dict, beats_per_measure=4):
    
    print('GETTING SOME INFORMATION ABOUT YOUR CORPUS: ')
        
    all_solo_durations = []
    total_measures     = 0
    all_note_durations = set()
    soloist_range      = set()
    never_plays        = set()
    tonalities         = []
    tonality_count     = set()
    beat_types         = set()
    for key in solo_dict.keys():        
        score                   = solo_dict[key]
        score_duration_beats    = score.quarterLength
        score_duration_measures = score_duration_beats / beats_per_measure
        total_measures         += int(score_duration_measures)
        tonality                = score.analyze('key')
        
        all_solo_durations.append(score_duration_measures)
        tonalities.append(tonality)
        
        #print(str(key) +' is ' + str(int(score_duration_measures)) + ' measures long;')
        
        if score_duration_beats % beats_per_measure != 0:
            print(f'-Warning: {key} has a measure that is not {beats_per_measure} beats long:')
            for measure_idx in range (1 , int(score_duration_measures + 1) ):
                if score.measure(measure_idx, indicesNotNumbers = True).quarterLength % beats_per_measure != 0:
                    print(f'The problem is in measure # {measure_idx + 1}')
        if len(score.getElementsByClass('Part')) > 1 :
            print(f'*Warning: it seems that there is more than one instrument part in the file {key}')
        
        for pitch in score.pitches:
            soloist_range.add(pitch.midi)
            
        for item in score.recurse().getElementsByClass(['Note','Rest']).stripTies():
            all_note_durations.add(Fraction(item.quarterLength)) 
            beat_types.add(Fraction(item.offset%1))
            if item.quarterLength == Fraction(0, 1):
                print(f'There is a note with no length: {(key,item,item.activeSite,item.offset)}')
        
    min_pitch, max_pitch       = int(min(soloist_range)), int(max(soloist_range))    
    min_duration, max_duration = int(min(all_solo_durations)), int(max(all_solo_durations))
    
    for pitch in range(min_pitch, max_pitch + 1):
        if (pitch in soloist_range) == False:
            never_plays.add(pitch)
            
    durations_list = list(all_note_durations)
    durations_list.sort()
    
    for tone in tonalities:
        counter = tonalities.count(tone)
        tonality_count.add((counter,tone))
    
    print(f'*There are {int(len(solo_dict))} files in the corpus.')
    print(f'*The shortest solo in the corpus has {min_duration} measures.')
    print(f'*The longest solo in the corpus has {max_duration} measures.')
    print(f'*There are in total {total_measures} measures of music.')
    print(f'*The lowest midi pitch played in the corpus is {min_pitch}, while the highest midi pitch is' + \
          f'{max_pitch}. However, the soloist(s) never played the midi pitches {never_plays}.')
    print(f'*There are {len(durations_list)} different note/rest durations in the corpus.')
    #print(tonality_count)
    #print(durations_list)
    #print(beat_types)
   
    return min_pitch, max_pitch, durations_list

    
def clean_chords(solo_dict, beats_per_measure=4):
    print('Fixing measures with no explicit chord symbol assigned...')
    for key in solo_dict.keys():        
        score = solo_dict[key]
        harmony.realizeChordSymbolDurations(score)
        score_duration_beats    = int(score.quarterLength)
        score_duration_measures = int(score_duration_beats / beats_per_measure)
        for measure_idx in range(0, score_duration_measures):
            measure_chords = score.measure(measure_idx, indicesNotNumbers=True).recurse().getElementsByClass('ChordSymbol')
            if [c for c in measure_chords] == []:
                #print('Fixing chord in measure ' + str(measure_idx+1) + ' in ' + str(key))
                last_chord      = score.measure(measure_idx-1, indicesNotNumbers=True).recurse().getElementsByClass('ChordSymbol')[-1]
                last_chord_name = last_chord.figure
                try:
                    missing_chord = harmony.ChordSymbol(last_chord_name)
                except:
                    missing_chord_root       = last_chord.root().name
                    missing_chord_pitches    = [p.name for p in last_chord.pitches]
                    missing_chord            = harmony.ChordSymbol(root=missing_chord_root)
                    missing_chord.pitchNames = missing_chord_pitches                                      
                missing_chord.quarterLength = beats_per_measure
                
                score.parts[0].measure(measure_idx, indicesNotNumbers=True).insert(0, missing_chord)
                if last_chord.quarterLength % beats_per_measure == 0:
                    score.parts[0].measure(measure_idx-1, indicesNotNumbers=True).getElementsByClass('ChordSymbol')[-1].quarterLength \
                    = beats_per_measure
                else:
                    score.parts[0].measure(measure_idx-1, indicesNotNumbers=True).getElementsByClass('ChordSymbol')[-1].quarterLength \
                    = last_chord.quarterLength % beats_per_measure
                
                
def parse_dict(solo_dict, durations_list,min_pitch, max_pitch, window_size=4, \
               beats_per_measure=4, transpose=False):     
    all_windows = []
    dict_len = len(solo_dict)  
    for key_idx, key in enumerate(solo_dict.keys()):                
        score                   = solo_dict[key]
        score_duration_measures = len(score.recurse().getElementsByClass('Measure'))
        last_window_idx         = score_duration_measures - window_size
        print(f'Splitting {key}, score {key_idx+1}/{dict_len}')      
        for window_idx in range(0, last_window_idx):
            window = score.measures( window_idx , window_idx + window_size , indicesNotNumbers=True)
            if window.quarterLength != beats_per_measure*window_size :
                 print(f'Window {(key,window_idx)} has quarter length {window.quarterLength}')
                
            all_windows.append(window)
            if transpose==True:
                print(f'Processing and transposing window {window_idx+1}/{last_window_idx}')
                for interval in range(-5,7):
                    if interval == 0:
                        continue  
                    #print('Transposing ' + str(interval) + ' half steps...')
                    transposed_window = window.transpose(interval)
                    min_pitch += -5
                    max_pitch += 6
                    all_windows.append(transposed_window)                            
    shuffle(all_windows)
    
    num_windows          = len(all_windows)
    chord_vect_size      = 24
    pitch_vect_size      = max_pitch - min_pitch + 2   #include rest as a pitch
    duration_vect_size   = len(durations_list)
    note_vect_size       = pitch_vect_size + duration_vect_size
    progression_matrices = []
    melody_matrices      = []
    for count, window in enumerate(all_windows):
        print('Encoding window ' + str(count+1) + '/' + str(num_windows))           
        harmony.realizeChordSymbolDurations(window)
        window_chords = window.recurse().getElementsByClass('ChordSymbol')
        window_notes  = window.recurse().getElementsByClass(['Note','Rest']).stripTies()
        
        progression_matrix = np.zeros([0, chord_vect_size])        
        for chord in window_chords:
            chord_vector   = np.zeros([1, chord_vect_size])
            chord_duration = int(chord.quarterLength)
            root_idx       = chord.root().midi % 12
            chord_pitches    = [p.midi for p in chord.pitches]
            chord_vector[0,root_idx] = 1
            for i, chord_pitch in enumerate(chord_pitches):
                chord_pitch_idx = chord_pitch % 12
                chord_vector[0, 12 + chord_pitch_idx] = 1
            for j in range(chord_duration):
                progression_matrix = np.append(progression_matrix, chord_vector, axis=0)
        progression_matrices.append(progression_matrix)
        
        melody_matrix = np.zeros([0, note_vect_size])
        for note in window_notes:
            pitch_vector    = np.zeros([1, pitch_vect_size])
            duration_vector = np.zeros([1, duration_vect_size])
            if note.isRest:
                pitch_idx = pitch_vect_size - 1
            else:
                pitch_idx = note.pitch.midi - min_pitch
            pitch_vector[0, pitch_idx] = 1
            duration = Fraction(note.quarterLength)
            if duration in durations_list:
                duration_idx = durations_list.index(duration)
            else:
                raise ValueError('The duration ' + str(duration) + ' is not in durations_list!')
            duration_vector[0, duration_idx] = 1
            note_vector   = np.append(pitch_vector, duration_vector, axis=1)
            melody_matrix = np.append(melody_matrix, note_vector, axis=0)
        melody_matrices.append(melody_matrix)
    
        
        #check for problems in the length of the chord progressions:
        '''
        for idx,window in enumerate(progression_matrices):
            if window.shape != (16,24):
                print(idx,window.shape)
                for chord in all_windows[idx].recurse().getElementsByClass('ChordSymbol'):
                    print(chord.quarterLength,chord)
        '''
        
        
        #check for problems in the length of the note sequences:
        '''
        for i,window in enumerate(all_windows):
            window_notes  = window.parts[0].recurse().getElementsByClass(['Note','Rest']).stripTies()
            if window_notes.quarterLength != 16:
                print(i,window_notes.quarterLength)
        '''
                
    return progression_matrices, melody_matrices


def matrices2tensors(progression_matrices, melody_matrices):
    event_data = np.stack(progression_matrices)
    E = Variable(torch.from_numpy(event_data))
    E = E.type(torch.FloatTensor)

    S=[]
    for window in melody_matrices:
        window = np.stack(window)
        window = Variable(torch.from_numpy(window))
        window = window.type(torch.FloatTensor)
        S.append(window)
        
    return E, S, durations_list

    
    
#------------------------------- PUBLIC FUNCTION -------------------------------------#
    

def build_dataset( directory_in_str, filename, beats_per_measure=4, transpose=False):
    solo_dict = createSoloDict(directory_in_str)
    min_pitch , max_pitch , durations_list = inspect_corpus(solo_dict)
    clean_chords(solo_dict, beats_per_measure)
    progression_matrices, melody_matrices = \
    parse_dict(solo_dict, durations_list,min_pitch, max_pitch, window_size=4, \
               beats_per_measure=4,transpose=False)
    E, S, durations_list = matrices2tensors(progression_matrices, melody_windows)
    Training_data = [ E, S, durations_list, min_pitch , max_pitch]
    torch.save(Training_data, filename)
    
#build_dataset('Divided_solos', 'Parker_Dataset.pt', beats_per_measure=4)
#build_dataset('Divided_solos', 'Parker_Dataset_allKeys.pt', beats_per_measure=4, transpose=True)
    