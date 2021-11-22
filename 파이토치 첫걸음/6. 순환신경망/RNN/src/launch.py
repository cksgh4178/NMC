# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 16:50:18 2021

@author: USER
"""

import os
import sys
import torch
import torch.nn as nn
import wandb

base_dir = os.getcwd()
sys.path.append(base_dir)

from utils import string_to_onehot, onehot_to_word
from RNN import rnn

# wandb setting
wandb.init(project = 'pytorch_start', entity='cksgh4178')
wandb.run.name = 'rnn'

# data
string = 'hello pytorch. how long can a rnn cell remember?'
chars = 'abcdefghijklmnopqrstuvwxyz ?!.,:;01' # 마지막 두 개가 start, end 표지
char_list = [i for i in chars]
n_letters = len(char_list)

# settings
n_hidden = 35
lr = 0.01
epochs = 1000

model = rnn(n_letters, n_hidden, n_letters)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

# train
onehot = torch.from_numpy(string_to_onehot(string, n_letters, char_list)).type_as(torch.FloatTensor())

for i in range(epochs):
    optimizer.zero_grad()
    total_loss = 0
    hidden = model.init_hidden()
    
    for j in range(onehot.size(0)-1):
        input_ = onehot[j:j+1,:]
        target = onehot[j+1]
        
        output, hidden = model(input_, hidden)
        loss = criterion(output.view(-1), target.view(-1))
        total_loss += loss
        input_ = output
   
    total_loss.backward()
    optimizer.step()
    
    wandb.log({'loss':total_loss.item()})
    
# test
start = torch.zeros(1, n_letters)
start[:, -2] = 1

with torch.no_grad():
    hidden = model.init_hidden()
    input_ = start
    output_string = ''
    for i in range(len(string)):
        output, hidden = model(input_, hidden)
        output_string += onehot_to_word(output.data, char_list)
        input_ = output

print(output_string)
        


