# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 14:39:23 2021

@author: USER
"""

import os
import sys
import torch
import torch.nn as nn
import wandb

base_dir = os.path.dirname(os.getcwd())
sys.path.append(base_dir)
from src.data_utils import dataloaders
from src.model import cnn


# settings
batch_size = 256
learning_rate = 1e-3
epoch = 10
download = False
num_workers = 0
drop_last = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# wandb settings
wandb.init(project="pytorch_start", entity="cksgh4178")
wandb.run.name = 'cnn'

# model
model = cnn(batch_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

train_loader, test_loader = dataloaders(download, batch_size, num_workers, drop_last)

# trian
for i in range(epoch):
    loss = 0.
    for _, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        optimizer.zero_grad()
        err = criterion(pred, y)
        loss += err.item()
        err.backward()
        optimizer.step()
    
    loss /= len(train_loader)
    wandb.log({'train_loss':loss})
    
# test
total = 0.
correct = 0.
with torch.no_grad():
    for _, (x, y) in enumerate(test_loader):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        _, index = torch.max(pred, 1)
        total += y.size(0)
        correct += (index == y).sum().float()
    wandb.log({'acc':100*correct / total})
    