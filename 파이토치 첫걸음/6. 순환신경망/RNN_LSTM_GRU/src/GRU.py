# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 19:32:17 2021

@author: USER
"""

import torch
import torch.nn as nn

class gru(nn.Module):
    
    def __init__(self, input_size, embedding_size, hidden_size, output_size, batch_size, num_layers=1):
        super(gru, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.encoder = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
        
    def forward(self, input, hidden):
        out = self.encoder(input.view(-1, 1))
        out, hidden = self.gru(out)
        out = self.decoder(out.view(self.batch_size, -1))
        return out, hidden
    
    def init_hidden(self):
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        return hidden