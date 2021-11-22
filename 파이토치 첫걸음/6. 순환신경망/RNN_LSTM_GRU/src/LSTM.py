# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 19:32:21 2021

@author: USER
"""

import torch
import torch.nn as nn

class lstm(nn.Module):
    
    def __init__(self, input_size, embedding_size, hidden_size, output_size, batch_size, num_layers=1):
        super(lstm, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.encoder = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
        
    def forward(self, input, hidden, cell):
        out = self.encoder(input.view(-1, 1))
        out, (hidden, cell) = self.lstm(out,(hidden, cell))
        out = self.decoder(out.view(self.batch_size, -1))
        return out, hidden, cell
    
    def init_hidden(self):
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        cell = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        return hidden, cell