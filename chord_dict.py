#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 07:46:58 2018

@author: Rodrigo Castro
"""

import numpy as np

#It creates a dictionary of roots and chords notes for "all" chords symbols.
def chord_dict():
    #Use only sharps, no flats.
    NOTES=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

    #Qualities: C=CM,Cm=Cmin,Cmaj7,Cm7=Cmin7,C7,Cm7b5,Cdim,C6,Caug=C+.       
    CHORDS={'C':[0,4,7],'CM':[0,4,7], 'Cm':[0,3,7],\
    'Cmin':[0,3,7], 'Cmaj7':[0,4,7,11],'Cm7':[0,3,7,10],'Cmin7':[0,3,7,10],\
    'C7':[0,4,7,10],'Cm7b5':[0,3,6,10],\
    'Cdim':[0,3,6,9],'C6':[0,4,7,9],'Caug':[0,4,8],'C+':[0,4,8]}
    
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

'''
Getting the chords in the dictionary in a multi-hot vector form,
root and chord notes have 1's.
'''
def chord_vect_dict():
    CHORDS=chord_dict()    
    CHORD_VECT={}   
    for key,value in CHORDS.items():
        chord_vect=np.zeros([24,1])
        chord_vect[value[0]]=1
        for i in range(len(value)):
            chord_vect[value[i]+12]=1
        CHORD_VECT[key]=chord_vect    
    
    return CHORD_VECT