#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 19:19:09 2018

@author: Rodrigo Castro
"""

import numpy as np

#-------------------------- HELPER FUNCTIONS ------------------------------------# 


#Create a dictionary of roots and chords notes for "all" chords symbols:
def chord_dict():
    #Use only sharps, no flats.
    NOTES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

    #Qualities: C=CM,Cm=Cmin,Cmaj7,Cm7=Cmin7,C7,Cm7b5,Cdim,C6,Caug=C+.       
    CHORDS={ 'C':[0,4,7], 'CM':[0,4,7], 'Cm':[0,3,7],\
    'Cmin':[0,3,7], 'Cmaj7':[0,4,7,11], 'Cm7':[0,3,7,10], 'Cmin7':[0,3,7,10],\
    'C7':[0,4,7,10], 'Cm7b5':[0,3,6,10],\
    'Cdim':[0,3,6,9], 'C6':[0,4,7,9], 'Caug':[0,4,8], 'C+':[0,4,8] }
    
    #transposing each chord to all keys.
    chord=CHORDS['C']
    for i in range(1,12):
        next_chord=np.add(chord,1)%12
        chord=next_chord
        CHORDS[(NOTES[i]).format(i)]=chord 
        
    chord=CHORDS['Cm']
    for i in range(1,12):
        next_chord=np.add(chord,1)%12
        chord=next_chord
        CHORDS[(NOTES[i]+'m').format(i)]=chord
        
    chord=CHORDS['Cmaj7']
    for i in range(1,12):
        next_chord=np.add(chord,1)%12
        chord=next_chord
        CHORDS[(NOTES[i]+"maj7").format(i)]=chord
        
    chord=CHORDS['Cm7']
    for i in range(1,12):
        next_chord=np.add(chord,1)%12
        chord=next_chord
        CHORDS[(NOTES[i]+'m7').format(i)]=chord
        
    chord=CHORDS['C7']
    for i in range(1,12):
        next_chord=np.add(chord,1)%12
        chord=next_chord
        CHORDS[(NOTES[i]+'7').format(i)]=chord
        
    chord=CHORDS['Cm7b5']
    for i in range(1,12):
        next_chord=np.add(chord,1)%12
        chord=next_chord
        CHORDS[(NOTES[i]+'m7b5').format(i)]=chord
        
    chord=CHORDS['Cdim']
    for i in range(1,12):
        next_chord=np.add(chord,1)%12
        chord=next_chord
        CHORDS[(NOTES[i]+'dim').format(i)]=chord
        
    chord=CHORDS['CM']
    for i in range(1,12):
        next_chord=np.add(chord,1)%12
        chord=next_chord
        CHORDS[(NOTES[i]+'M').format(i)]=chord
        
    chord=CHORDS['Cmin']
    for i in range(1,12):
        next_chord=np.add(chord,1)%12
        chord=next_chord
        CHORDS[(NOTES[i]+'min').format(i)]=chord
        
    chord=CHORDS['Cmin7']
    for i in range(1,12):
        next_chord=np.add(chord,1)%12
        chord=next_chord
        CHORDS[(NOTES[i]+'min7').format(i)]=chord
        
    chord=CHORDS['C6']
    for i in range(1,12):
        next_chord=np.add(chord,1)%12
        chord=next_chord
        CHORDS[(NOTES[i]+'6').format(i)]=chord
 
    chord=CHORDS['Caug']
    for i in range(1,12):
        next_chord=np.add(chord,1)%12
        chord=next_chord
        CHORDS[(NOTES[i]+'aug').format(i)]=chord
        
    return CHORDS


#Getting the chords in the dictionary in a multi-hot vector form (root and chord notes have 1's)
def chord_vect_dict():
    CHORDS = chord_dict()    
    CHORD_VECT = {}   
    for key,value in CHORDS.items():
        chord_vect = np.zeros([24,1])
        chord_vect[ value[0] ] = 1
        for i in range(len(value)):
            chord_vect[ value[i] + 12 ] = 1
        CHORD_VECT[key] = chord_vect    
    
    return CHORD_VECT


#Any other chord to vector:
def chord2vec(chord):
    chord=sorted(chord)
    vect=np.zeros([24,1])
    root=chord[0]%12
    vect[root]=1
    n=len(chord)
    for i in range(n):
        vect[chord[i]%12+12]=1
    
    return vect
    

def identify_chord(chord):
    CHORD_DICT = chord_vect_dict()
    vect=chord2vec(chord)
    count=0
    vectors=CHORD_DICT.values()
    for value in vectors:
        a=np.allclose(vect,value)
        count+=1*a
    if count>0:
        print('I know that chord')
    else:
        print('I don\'t know that chord')
        
        
#----------------------------------- PUBLIC FUNCTIONS -------------------------------------#


def user_progression():
    CHORD_VECT = chord_vect_dict()  
    chord_embedding_size = 24 
    
    print('How many bars long is your progression? (enter an integer):')
    bars = int(input()) 
    total_beats = bars*4
    chord_matrix = np.zeros([chord_embedding_size, total_beats]) 
    
    #list that will contain the chord symbols corresponting to each beat (without the symbols |'s):                                      
    p = [] 
    #fill in with ?'s:                                                       
    for i in range(0,total_beats):                              
        p.append('?')
        
    #length of the vector containing also the |'s:  
    m = total_beats + bars + 1                                        
    p_temp = p[:] 
    
    #insert |'s every 4 symbols:
    for i in range(0, m, 5):                                      
        p_temp.insert(i,'|')
    
    #string that will display the progression:    
    progression = ''                                              
    for i in range(len(p_temp)):
        progression += p_temp[i] + ' '
        
    print('Progression: ' + progression)
        
    counter=0
    while True:
        if bars == 0:
            print('The progression has to be at least 1 bar long. Try again.')
            break
        else:
            print('Enter a chord:')
            symbol_input = str(input())
            if str(symbol_input) not in CHORD_VECT.keys():
                print ('I don\'t know that notation')
                break
            else:
                chord = CHORD_VECT[symbol_input]      
                print('How many beats of that chord?')
                beats_input = int(input())
                t = counter+beats_input
                if t > total_beats:
                    print('My friend, you said you wanted ' + str(bars) \
                          + ' bars, your progression was longer. Try again.')
                    break      
                else:
                    for i in range(counter, t):
                            chord_matrix[:,i] = np.transpose(chord)
                    for i in range(counter, t):
                        p[i] = symbol_input              
                    p_temp = p[:]  
                    
                    #insert |'s every 4 symbols:
                    for i in range(0, m, 5):                                      
                        p_temp.insert(i,'|')
                    progression = ''
                    for i in range(len(p_temp)):
                        progression += p_temp[i] + ' '
                        
                    print('Progression: '+progression)
                    counter = t
                    if counter == total_beats:
                        break
                    else:
                        continue

    return progression , chord_matrix
