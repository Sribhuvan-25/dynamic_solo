#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 19:16:35 2019

@author: Rodrigo Castro
"""

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from music21 import *
import matplotlib.pyplot as plt
from user_progression import user_progression


'''
Form of the data [ E , S ]:
E is a tensor containing all of the sequences of conditioning events (the chord progression of each note sequence)
Each event should be a row vector
The shape of E should be [# event sequences, # events in each example , event embedding size]
S is the list of tensors each epresenting a signal sequence (note sequences)
The signal sequence S[i] is conditioned to the event sequence E[i,:,:]
Each signal should be a row vector
i-th signal sequence length = S[i].size(0)
z_size denotes the hidden layer size for event sequence E
Z_size denotes the hidden layer size for signal sequence S
'''


def dimensions(E,S): 
    
    num_event_examples, num_events , event_emb_size  = E.shape    
    num_seq_examples = len(S)
    signal_emb_size = S[0].size(1)
    
    dims = [num_event_examples, num_events , event_emb_size, num_seq_examples, signal_emb_size ]
    
    return dims


def create_parameters():
    
    W_ze = Variable(torch.randn( event_emb_size , z_size ), requires_grad = True)
    W_zz = Variable(torch.randn( z_size , z_size ), requires_grad = True)
    b_z = Variable(torch.randn( 1 , z_size ), requires_grad = True)
    
    W_update_ze = Variable(torch.randn( event_emb_size , z_size ), requires_grad = True)
    W_update_zz = Variable(torch.randn( z_size , z_size ), requires_grad = True)
    b_update_z = Variable(torch.randn( 1 , z_size ), requires_grad = True)
    
    W_forget_ze = Variable(torch.randn( event_emb_size , z_size ), requires_grad = True)
    W_forget_zz = Variable(torch.randn( z_size , z_size ), requires_grad = True)
    b_forget_z = Variable(torch.randn( 1 , z_size ), requires_grad = True)
    
    W_output_ze = Variable(torch.randn( event_emb_size , z_size ), requires_grad = True)
    W_output_zz = Variable(torch.randn( z_size , z_size ), requires_grad = True)
    b_output_z = Variable(torch.randn( 1 , z_size ), requires_grad = True)
    
    W_Zz = Variable(torch.randn( z_size , Z_size ), requires_grad = True)
    W_ZZ = Variable(torch.randn( Z_size , Z_size ), requires_grad = True)
    W_Zs = Variable(torch.randn( signal_emb_size , Z_size ), requires_grad = True)
    b_Z = Variable(torch.randn( 1 , Z_size ), requires_grad = True)
    
    W_update_ZZ = Variable(torch.randn( Z_size , Z_size ), requires_grad = True)
    W_update_Zs = Variable(torch.randn( signal_emb_size , Z_size ), requires_grad = True)
    b_update_Z = Variable(torch.randn( 1 , Z_size ), requires_grad = True)
    
    W_forget_ZZ = Variable(torch.randn( Z_size , Z_size ), requires_grad = True)
    W_forget_Zs = Variable(torch.randn( signal_emb_size , Z_size ), requires_grad = True)
    b_forget_Z = Variable(torch.randn( 1 , Z_size ), requires_grad = True)
    
    W_output_ZZ = Variable(torch.randn( Z_size , Z_size ), requires_grad = True)
    W_output_Zs = Variable(torch.randn( signal_emb_size , Z_size ), requires_grad = True)
    b_output_Z = Variable(torch.randn( 1 , Z_size ), requires_grad = True)
    
    W_yZ = Variable(torch.randn(  Z_size , signal_emb_size ), requires_grad = True)
    b_y = Variable(torch.randn( 1 , signal_emb_size ), requires_grad = True)
    
    
    net_parameters = [ W_ze , W_zz , b_z ,\
                      W_update_ze , W_update_zz , b_update_z ,\
                      W_forget_ze , W_forget_zz , b_forget_z ,\
                      W_output_ze , W_output_zz , b_output_z ,\
                      W_Zz , W_ZZ , W_Zs , b_Z,\
                      W_update_ZZ , W_update_Zs , b_update_Z ,\
                      W_forget_ZZ , W_forget_Zs , b_forget_Z ,\
                      W_output_ZZ , W_output_Zs , b_output_Z ,\
                      W_yZ , b_y ]
    
    return net_parameters


def count_parameters(parameters):
    count = 0
    for item in net_parameters:
        count += item.size(0)*item.size(1)
        
    return count


def get_durations_vector( signal_emb_size, idx_ini, idx_fin, subdivision):
    
    durations_vector = torch.zeros(signal_emb_size,1)   
    for i in range( idx_ini, idx_fin+1 ):
        durations_vector[i] = i - idx_ini + 1
        
    durations_vector = durations_vector/subdivision
    
    return durations_vector


def net_train(e,s):
    #e is one sequence of e.size(1) events
    #s is one sequence of s.size(1) signals
    
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

        
    signal_steps = s.size(0)
        
    Z                    = torch.zeros(signal_steps, Z_size)       
    Z_prev               = Z_initial_hidden_state
    cell_Z_prev          = initial_memory_cell_Z
    signal_prev          = torch.zeros( 1 , signal_emb_size )
    dynamic_idx          = 0
    for i in range(0,signal_steps):
        
        dynamic_idx         += int( torch.mm( signal_prev , durations_vector ) )
        conditioning_hidden  = torch.unsqueeze(z[dynamic_idx,:], 0)
    
        pre_cell_Z_step  = torch.tanh( torch.mm( Z_prev , W_ZZ ) + torch.mm( signal_prev , W_Zs ) + torch.mm(conditioning_hidden , W_Zz) + b_Z )
        update_Z         = torch.sigmoid( torch.mm( Z_prev , W_update_ZZ ) + torch.mm( signal_prev , W_update_Zs ) + b_update_Z )
        forget_Z         = torch.sigmoid( torch.mm( Z_prev , W_forget_ZZ ) + torch.mm( signal_prev , W_forget_Zs ) + b_forget_Z )
        output_Z         = torch.sigmoid( torch.mm( Z_prev , W_output_ZZ ) + torch.mm( signal_prev , W_output_Zs ) + b_output_Z )
        cell_Z_next      = torch.mul( update_Z , pre_cell_Z_step ) + torch.mul( forget_Z , cell_Z_prev )
        Z_next           = torch.mul( output_Z , torch.tanh( cell_Z_next ) )
        
        Z[i,:]       = Z_next  
        cell_Z_prev  = cell_Z_next
        Z_prev       = Z_next
        signal_prev = torch.unsqueeze(s[i,:], 0)
    
    y_hat_pre     = torch.mm(Z, W_yZ ) + b_y
    y_hat_pitch   = F.softmax( y_hat_pre[:,0:129] , dim=1 )
    y_hat_rhythm  = F.softmax( y_hat_pre[:,129:] , dim=1 )
    y_hat         = torch.cat((y_hat_pitch , y_hat_rhythm ), dim = 1)
    
    return y_hat_pre

#stochastic GD:
def train_parameters_stoch( loss_func, optimizer ):
    
    #stochastic:
    J_hist=[]
    for i in range(1,epochs+1):
        for j in range(num_seq_examples):
            e = E[j,:,:]
            s = S[j]
            y_hat = net_train(e,s)
            J = loss_func(y_hat,s)
            optimizer.zero_grad()
            J.backward()
            optimizer.step()
            if j%50 == 0:
               print('Epoch: '+str(i)+', examples trained: ' + str(j) + ', Cost: ' + str(J))
            J_hist.append(J)        
        #print('Epoch ' + str(i) + ', Cost: ' + str(J))
    
    plt.plot(J_hist)
    plt.xlabel('Gradient steps')
    vert_label=plt.ylabel('Loss')
    vert_label.set_rotation(0)


def loss_sum():
    SUM = torch.Tensor([0])
    for j in range(num_seq_examples):
        e = E[j,:,:]
        s = S[j]
        y_hat = net_train(e,s)
        J = loss_func(y_hat,s)
        SUM = SUM + J
        if j%50 == 0:
             print('Examples processed: ' + str(j))
    SUM = SUM/num_seq_examples
             
    return SUM
    
#Batch GD:    
def train_parameters_batch( loss_func, optimizer ):
    
    J_hist=[]
    for i in range(1,epochs+1):
        J = loss_sum()
        optimizer.zero_grad()
        J.backward()
        optimizer.step()
        J_hist.append(J)        
        print('Epoch ' + str(i) + ', Cost: ' + str(J))
    
    plt.plot(J_hist)
    plt.xlabel('Epochs')
    vert_label=plt.ylabel('Loss')
    vert_label.set_rotation(0)


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
    while melody_duration <= float(event_steps-1) :
        melody_duration += float(torch.mm( signal_prev , durations_vector ))
        dynamic_idx      = int(melody_duration)
        if dynamic_idx > event_steps-1:
            break
        conditioning_hidden  = torch.unsqueeze(z[dynamic_idx,:], 0)
    
        pre_cell_Z_step  = torch.tanh( torch.mm( Z_prev , W_ZZ ) + torch.mm( signal_prev , W_Zs ) + torch.mm(conditioning_hidden , W_Zz) + b_Z )
        update_Z         = torch.sigmoid( torch.mm( Z_prev , W_update_ZZ ) + torch.mm( signal_prev , W_update_Zs ) + b_update_Z )
        forget_Z         = torch.sigmoid( torch.mm( Z_prev , W_forget_ZZ ) + torch.mm( signal_prev , W_forget_Zs ) + b_forget_Z )
        output_Z         = torch.sigmoid( torch.mm( Z_prev , W_output_ZZ ) + torch.mm( signal_prev , W_output_Zs ) + b_output_Z )
        cell_Z_next      = torch.mul( update_Z , pre_cell_Z_step ) + torch.mul( forget_Z , cell_Z_prev )
        Z_next           = torch.mul( output_Z , torch.tanh( cell_Z_next ) )
        Y_hat_pre        = torch.mm(Z_next, W_yZ ) + b_y
        Y_hat_pitch      = F.softmax( Y_hat_pre[:,0:129], dim = 1 )
        Y_hat_rhythm     = F.softmax( Y_hat_pre[:,129:], dim = 1 )
        
        raw_y_hat        = torch.cat ((Y_hat_pitch,Y_hat_rhythm) , dim = 1)
        raw_prediction_list.append(raw_y_hat)        
        
        note_max , note_argmax = Y_hat_pitch.max(1)
        rhythm_max , rhythm_argmax = Y_hat_rhythm.max(1)
        y_hat = torch.zeros(Y_hat_pre.size())
        y_hat[0, int(note_argmax)] = 1  
        y_hat[0, int(129+int(rhythm_argmax))] = 1                    
        prediction_list.append(y_hat)
    
        cell_Z_prev  = cell_Z_next
        Z_prev       = Z_next
        signal_prev  = y_hat
        print( str(melody_duration) + ' beats generated')
    
    prediction = torch.cat(prediction_list)
    raw_prediction = torch.cat(raw_prediction_list)
    
    return prediction , raw_prediction


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


#------------------------------------- UNDER CONSTRUCTION --------------------------------------------#

torch.manual_seed(12)

E , S = torch.load('Dataset_window16.pt')


z_size = 16       #hidden layer dimension of event LSTM
Z_size = 48       #hidden layer dimension of signal LSTM
 
LR = 0.05
epochs = 1000
WeightDecay = 0 #1e-6
Momentum = 0.5

#loss_func = torch.nn.MSELoss()    #if used, return y_hat instead of y_hat_pre
loss_func = torch.nn.BCEWithLogitsLoss()

#Try with only one small training example:
#E = E[0,0:8,:]
#E = E.reshape(1,8,24)
#S[0] = S[0][0:16,:]

num_event_examples, num_events , event_emb_size, num_seq_examples, signal_emb_size = dimensions(E,S)
durations_vector = get_durations_vector( signal_emb_size, 129, signal_emb_size-1 , 12)
net_parameters = create_parameters()
num_parameters = count_parameters(net_parameters)
print('There are ' + str(num_parameters) + ' parameters to train.')
optimizer = torch.optim.RMSprop(net_parameters,lr=LR, alpha=0.99, eps=1e-6, weight_decay = WeightDecay, momentum = Momentum, centered=True)
#train_parameters_stoch( loss_func, optimizer )
#train_parameters_batch( loss_func, optimizer )


#Testing the trained net:
#solo, solo_prediction , raw_prediction = predict_new()

#Importing parameters from file and testing:
#net_parameters = torch.load('net_parameters_cpu.pt')
#solo, solo_prediction , raw_prediction = predict_new()