# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 13:54:22 2021

@author: USER
"""

import torch.nn as nn

class cnn(nn.Module):
    def __init__(self, batch_size):
        super(cnn, self).__init__()
        self.batch_size = batch_size
        
        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
            )
        
        self.fc_layer = nn.Sequential(
            nn.Linear(64*3*3, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
            )
        
    def forward(self, x):
        out = self.conv_layer(x)
        out = out.view(self.batch_size, -1)
        out = self.fc_layer(out)
        return out