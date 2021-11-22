# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 16:27:28 2021

@author: USER
"""

import torch
import torch.nn as nn

class rnn(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super(rnn, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # RNN cell 직접 구성
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.act_fn = nn.Tanh()
        
    def forward(self, input, hidden):
        hidden = self.act_fn(self.i2h(input)+self.h2h(hidden))
        output = self.h2o(hidden)
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)