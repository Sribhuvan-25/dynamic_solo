#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 08:19:53 2018

@author: Amillo
"""

import numpy as np
import chord_dict

def chord2vec(chord):
    chord=sorted(chord)
    vect=np.zeros([24,1])
    root=chord[0]%12
    vect[root]=1
    n=len(chord)
    for i in range(n):
        vect[chord[i]%12+12]=1
    
    return vect
    
CHORD_DICT=CHORD_DICT_Simple.chord_vect_dict()

def identify_chord(chord):
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
        
    
