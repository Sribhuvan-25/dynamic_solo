#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 19:19:09 2018

@author: Amillo
"""
import numpy as np
import CHORD_DICT_Simple

def main():
    CHORD_VECT = CHORD_DICT_Simple.chord_vect_dict()
    
    chord_embedding_size=24 #24x1
    
    print('How many bars long is your progression? (enter an integer):')
    bars = int(input()) 
    total_beats = bars*4
    chord_matrix = np.zeros([chord_embedding_size,total_beats]) #matrix that will contain the chord embeddings
                                           
    p=[]                                                        #list that will contain the chord symbols corresponting to each beat (without the symbols |'s)
    for i in range(0,total_beats):                              #fill in with ?'s
        p.append('?')
       
    m=total_beats+bars+1                                        #length of the vector containing also the |'s
    p_temp=p[:]    
    for i in range(0,m,5):                                      #insert |'s every 4 symbols
        p_temp.insert(i,'|')
        
    progression=''                                              #string that will display the progression
    for i in range(len(p_temp)):
        progression+=p_temp[i]+' '
        
    print('Progression: '+progression)
        
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
                chord=CHORD_VECT[symbol_input]      
                print('For how many beats of that chord?')
                beats_input = int(input())
                t=counter+beats_input
                if t > total_beats:
                    print('My friend, you said you wanted '+str(bars)+' bars, your progression was longer. Try again.')
                    break      
                else:
                    for i in range(counter,t):
                            chord_matrix[:,i]=np.transpose(chord)
                    for i in range(counter,t):
                        p[i]=symbol_input              
                    p_temp=p[:]   
                    for i in range(0,m,5):                                      #insert |'s every 4 symbols
                        p_temp.insert(i,'|')
                    progression=''
                    for i in range(len(p_temp)):
                        progression+=p_temp[i]+' '
                        
                    print('Progression: '+progression)
                    counter=t
                    if counter==total_beats:
                        break
                    else:
                        continue

    return progression,chord_matrix
  
if __name__  == '__main__':                
    chord_progression,chord_embeddings=main()