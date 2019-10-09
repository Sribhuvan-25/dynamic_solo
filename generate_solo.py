# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 16:08:39 2019

@author: Rodrigo Castro
"""

import numpy as np
import torch
import torch.nn.functional as F
from music21 import *
from user_progression import user_progression

#---------------------------------- HELPER FUNCTIONS --------------------------------------# 
    
def net_predict(e):
    
    W_ze , W_zz , b_z ,\
    W_update_ze , W_update_zz , b_update_z ,\
    W_forget_ze , W_forget_zz , b_forget_z ,\
    W_output_ze , W_output_zz , b_output_z ,\
    W_Zz , W_ZZ , W_Zs , b_Z,\
    W_update_ZZ , W_update_Zs , b_update_Z ,\
    W_forget_ZZ , W_forget_Zs , b_forget_Z ,\
    W_output_ZZ , W_output_Zs , b_output_Z ,\
    W_yZ , b_y ,\
    = net_parameters
    
    z_initial_hidden_state  = torch.zeros(1,z_size)
    Z_initial_hidden_state  = torch.zeros(1,Z_size)
    initial_memory_cell_z   = torch.zeros(1,z_size)
    initial_memory_cell_Z   = torch.zeros(1,Z_size)

    event_steps = e.size(0)
    
    z            = torch.zeros( event_steps , z_size )     
    z_prev       = z_initial_hidden_state
    cell_z_prev  = initial_memory_cell_z    
    for i in reversed(range(0,event_steps)):
        event = torch.unsqueeze(e[i,:],0)
        
        pre_cell_z_step  = torch.tanh( torch.mm( z_prev , W_zz ) + torch.mm( event , W_ze ) + b_z )
        update_z         = torch.sigmoid( torch.mm( z_prev , W_update_zz ) + torch.mm( event , W_update_ze ) + b_update_z )
        forget_z         = torch.sigmoid( torch.mm( z_prev , W_forget_zz ) + torch.mm( event , W_forget_ze ) + b_forget_z )
        output_z         = torch.sigmoid( torch.mm( z_prev , W_output_zz ) + torch.mm( event , W_output_ze ) + b_output_z )
        cell_z_next      = torch.mul( update_z , pre_cell_z_step ) + torch.mul( forget_z , cell_z_prev )
        z_next           = torch.mul( output_z , torch.tanh( cell_z_next ) )
        
        z[i,:]       = z_next   
        cell_z_prev  = cell_z_next
        z_prev       = z_next

    print('Predicting solo...')    
    Z_prev               = Z_initial_hidden_state
    cell_Z_prev          = initial_memory_cell_Z
    signal_prev          = torch.zeros( 1 , signal_emb_size )
    prediction_list      = []
    raw_prediction_list  = []
    melody_duration      = 0 
    while melody_duration <= float(event_steps-1):
        melody_duration += float(torch.mm( signal_prev , durations_vector ))
        dynamic_idx      = int(melody_duration)
        if dynamic_idx >= float(event_steps):
            break
        conditioning_hidden  = torch.unsqueeze(z[dynamic_idx,:], 0)
    
        pre_cell_Z_step  = torch.tanh( torch.mm( Z_prev , W_ZZ ) + torch.mm( signal_prev , W_Zs ) + torch.mm(conditioning_hidden , W_Zz) + b_Z )
        update_Z         = torch.sigmoid( torch.mm( Z_prev , W_update_ZZ ) + torch.mm( signal_prev , W_update_Zs ) + b_update_Z )
        forget_Z         = torch.sigmoid( torch.mm( Z_prev , W_forget_ZZ ) + torch.mm( signal_prev , W_forget_Zs ) + b_forget_Z )
        output_Z         = torch.sigmoid( torch.mm( Z_prev , W_output_ZZ ) + torch.mm( signal_prev , W_output_Zs ) + b_output_Z )
        cell_Z_next      = torch.mul( update_Z , pre_cell_Z_step ) + torch.mul( forget_Z , cell_Z_prev )
        Z_next           = torch.mul( output_Z , torch.tanh( cell_Z_next ) )
        Y_hat_pre        = torch.mm(Z_next, W_yZ ) + b_y
        Y_hat_pitch      = F.softmax( Y_hat_pre[:,0:rythym_idx_ini], dim = 1 )
        Y_hat_rhythm     = F.softmax( Y_hat_pre[:,rythym_idx_ini:], dim = 1 )
        
        raw_y_hat        = torch.cat ((Y_hat_pitch,Y_hat_rhythm) , dim = 1)
        raw_prediction_list.append(raw_y_hat)        
        
        #note_max , note_argmax = Y_hat_pitch.max(1)
        prob_dist = torch.distributions.Categorical(Y_hat_pitch)
        note_argmax = int(prob_dist.sample())
        
        rhythm_max , rhythm_argmax = Y_hat_rhythm.max(1)
        y_hat = torch.zeros(Y_hat_pre.size())
        y_hat[0, int(note_argmax)] = 1  
        y_hat[0, int(rythym_idx_ini+int(rhythm_argmax))] = 1                    
        prediction_list.append(y_hat)
    
        cell_Z_prev  = cell_Z_next
        Z_prev       = Z_next
        signal_prev  = y_hat
        print( str(melody_duration) + ' beats generated')
    
    prediction = torch.cat(prediction_list)
    raw_prediction = torch.cat(raw_prediction_list)
    
    return prediction , raw_prediction


#Converts a two-hot vector into a note or rest:
def vect2note(vector):
    note_embedding_size = signal_emb_size 
    assert np.shape(vector) == (note_embedding_size,)
    duration_idx = int(np.argwhere(vector[rythym_idx_ini:]))
    duration = durations_list[duration_idx]
    
    if vector[rythym_idx_ini-1] == 1:
        nota = note.Rest()
        nota.quarterLength = duration
    else:
        height = int(np.argwhere(vector[:rythym_idx_ini-1])) + min_pitch
        nota = note.Note()
        nota.pitch.midi = height
        nota.quarterLength = duration
    
    return nota


def matrix2melody(melodyMatrix):
    m,n = melodyMatrix.shape
    melodyStream = stream.Stream()
    
    #To impose a time and key signature:
    melodyStream.timeSignature = meter.TimeSignature('4/4')
    melodyStream.keySignature = key.Key('C')
    
    for i in range(n):
        vector = melodyMatrix[:,i]
        nota = vect2note(vector)
        melodyStream.append(nota)
        
    return melodyStream


def predict_new():
    print('Let\'s generate a new solo.')
    progression , chord_matrix = user_progression()
    chord_matrix = torch.from_numpy(chord_matrix)
    chord_matrix = chord_matrix.transpose(0,1)
    chord_matrix = chord_matrix.type(torch.FloatTensor)
    solo_prediction, raw_prediction = net_predict(chord_matrix)
    solo_prediction = solo_prediction.transpose(0,1)
    raw_prediction = raw_prediction.transpose(0,1)
    solo_prediction = solo_prediction.numpy()
    raw_prediction = raw_prediction.detach().numpy()
      
    solo = matrix2melody(solo_prediction)    
    solo.show('text')
    
    return solo, solo_prediction , raw_prediction


#----------------------------------------------------------------------------------------------#
    
#rythym_idx_ini = ?         
#net_parameters, durations_list, z_size, Z_size, event_emb_size, signal_emb_size = torch.load('model_settings.pt')    
#solo, solo_prediction , raw_prediction = predict_new()
#solo.write('xml', 'generated_solo.xml')
    

