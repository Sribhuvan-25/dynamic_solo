#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 19:16:35 2019

@author: Rodrigo Castro
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


#---------------------------------------- UNDER CONSTRUCTION ----------------------------------------------#

'''
Form of the data:
E is a tensor containing all of the sequences of conditioning chords, which we will call events
Each event should be a row vector
The shape of E should be [# event sequences, # events in each example , event embedding size]
S is the list of tensors each epresenting a sequence of notes that we will call signals
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


def get_durations_vector( signal_emb_size, idx_ini, idx_fin, subdivision):
    
    durations_vector = torch.zeros(signal_emb_size,1)  
    for i in range( idx_ini, idx_fin+1 ):
        durations_vector[i] = i - idx_ini
        
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
    
    pre_cell_z     = torch.zeros( event_steps , z_size )
    gate_update_z  = torch.zeros( event_steps , z_size )
    gate_forget_z  = torch.zeros( event_steps , z_size )
    gate_output_z  = torch.zeros( event_steps , z_size )
    cell_z         = torch.zeros( event_steps , z_size )
    z              = torch.zeros( event_steps , z_size )
      
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
        
        pre_cell_z[i,:]     = pre_cell_z_step 
        gate_update_z[i,:]  = update_z
        gate_forget_z[i,:]  = forget_z
        gate_output_z[i,:]  = output_z
        cell_z[i,:]         = cell_z_next
        z[i,:]              = z_next
    
        cell_z_prev  = cell_z_next
        z_prev       = z_next

        
    signal_steps = s.size(0)
    
    pre_cell_Z     = torch.zeros(signal_steps, Z_size)
    gate_update_Z  = torch.zeros(signal_steps, Z_size)
    gate_forget_Z  = torch.zeros(signal_steps, Z_size)
    gate_output_Z  = torch.zeros(signal_steps, Z_size)
    cell_Z         = torch.zeros(signal_steps, Z_size)    
    Z              = torch.zeros(signal_steps, Z_size)
        
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

        pre_cell_Z[i,:]     = pre_cell_Z_step 
        gate_update_Z[i,:]  = update_Z
        gate_forget_Z[i,:]  = forget_Z
        gate_output_Z[i,:]  = output_Z
        cell_Z[i,:]         = cell_Z_next
        Z[i,:]              = Z_next
    
        cell_Z_prev  = cell_Z_next
        Z_prev       = Z_next
        signal_prev = torch.unsqueeze(s[i,:], 0)
    
    y_hat = F.softmax( torch.mm(Z, W_yZ ) + b_y , dim=1 )
    
    return y_hat


def train_parameters():
    
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(net_parameters,lr=LR, alpha=0.99, eps=1e-6, weight_decay=1e-6, momentum=0, centered=True)
    
    J_hist=[]
    for i in range(epochs):
        for j in range(num_seq_examples):
            e = E[j,:,:]
            s = S[j]
            y_hat = net_train(e,s)
            J = loss_func(y_hat,s)
            optimizer.zero_grad()
            J.backward()
            optimizer.step()
        J_hist.append(J)
        print(J)
    
    plt.plot(J_hist)
    plt.xlabel('iterations')
    vert_label=plt.ylabel('Loss')
    vert_label.set_rotation(0)
    
    return net_parameters

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
    
    pre_cell_z     = torch.zeros( event_steps , z_size )
    gate_update_z  = torch.zeros( event_steps , z_size )
    gate_forget_z  = torch.zeros( event_steps , z_size )
    gate_output_z  = torch.zeros( event_steps , z_size )
    cell_z         = torch.zeros( event_steps , z_size )
    z              = torch.zeros( event_steps , z_size )
      
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
        
        pre_cell_z[i,:]     = pre_cell_z_step 
        gate_update_z[i,:]  = update_z
        gate_forget_z[i,:]  = forget_z
        gate_output_z[i,:]  = output_z
        cell_z[i,:]         = cell_z_next
        z[i,:]              = z_next
    
        cell_z_prev  = cell_z_next
        z_prev       = z_next

        
    Z_prev               = Z_initial_hidden_state
    cell_Z_prev          = initial_memory_cell_Z
    signal_prev          = torch.zeros( 1 , signal_emb_size )
    prediction_list      = []
    dynamic_idx          = 0
    while dynamic_idx < event_steps-1 :
        
        dynamic_idx         += int( torch.mm( signal_prev , durations_vector ) )
        if dynamic_idx > event_steps-1:
            break
        conditioning_hidden  = torch.unsqueeze(z[dynamic_idx,:], 0)
    
        pre_cell_Z_step  = torch.tanh( torch.mm( Z_prev , W_ZZ ) + torch.mm( signal_prev , W_Zs ) + torch.mm(conditioning_hidden , W_Zz) + b_Z )
        update_Z         = torch.sigmoid( torch.mm( Z_prev , W_update_ZZ ) + torch.mm( signal_prev , W_update_Zs ) + b_update_Z )
        forget_Z         = torch.sigmoid( torch.mm( Z_prev , W_forget_ZZ ) + torch.mm( signal_prev , W_forget_Zs ) + b_forget_Z )
        output_Z         = torch.sigmoid( torch.mm( Z_prev , W_output_ZZ ) + torch.mm( signal_prev , W_output_Zs ) + b_output_Z )
        cell_Z_next      = torch.mul( update_Z , pre_cell_Z_step ) + torch.mul( forget_Z , cell_Z_prev )
        Z_next           = torch.mul( output_Z , torch.tanh( cell_Z_next ) )
        y_hat            = torch.round( F.softmax( torch.mm(Z_next, W_yZ ) + b_y , dim =1 ) )
        
        prediction_list.append(y_hat)
    
        cell_Z_prev  = cell_Z_next
        Z_prev       = Z_next
        signal_prev  = y_hat 
    
    prediction = torch.cat(prediction_list)
    
    return prediction



#---------------------------------------- UNDER CONSTRUCTION ------------------------------------------#

#E,S = ?

'''
torch.manual_seed(123)

LR = 0.02
epochs = 1500
z_size = 8       #hidden layer dimension of event LSTM
Z_size = 8       #hidden layer dimension of signal LSTM


num_event_examples, num_events , event_emb_size, num_seq_examples, signal_emb_size = dimensions(E,S)
durations_vector = get_durations_vector(225,129,224,12)
net_parameters = create_parameters()
net_parameters = train_parameters()
'''


