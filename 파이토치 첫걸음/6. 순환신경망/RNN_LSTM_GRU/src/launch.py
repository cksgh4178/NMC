# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 19:46:32 2021

@author: USER
"""

import os
import sys
import string
import unidecode
import torch
import torch.nn as nn
base_dir = os.path.dirname(os.getcwd())
sys.path.append(base_dir)
from src.utils import random_training_set
from src.RNN import rnn
from src.GRU import gru
from src.LSTM import lstm

# Data prepare
all_characters = string.printable
n_characters = len(all_characters)

file = unidecode.unidecode(open('../data/input.txt').read())
file_len = len(file)

# common hyperparameters
epochs = 2000
chunk_len = 200
hidden_size = 100
batch_size = 1
num_layers = 1
embedding_size = 70
lr = 0.002

# model setting
model = rnn(n_characters, embedding_size, hidden_size, n_characters, 2)
# model = gru(n_characters, embedding_size, hidden_size, n_characters, 2)
# model = lstm(n_characters, embedding_size, hidden_size, n_characters, 2)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# train
for i in range(epochs):
    inp, target = random_training_set(file, file_len, chunk_len, all_characters)
    hidden,cell = model.init_hidden()
    loss = torch.tensor([0]).type(torch.FloatTensor)
    optimizer.zero_grad()
    for j in range(chunk_len-1):
        x  = inp[j]
        y_ = target[j].unsqueeze(0).type(torch.LongTensor)
        y,hidden,cell = model(x,hidden,cell)
        loss += criterion(y,y_)
loss.backward()
optimizer.step()