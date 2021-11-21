# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 14:22:42 2021

@author: USER
"""


import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def dataloaders(download, batch_size, num_workers, drop_last):
      mnist_train = dset.MNIST('../data', train=True, transform=transforms.ToTensor(),
                               target_transform=None, download=download)
      mnist_test = dset.MNIST('../data', train=False, transform=transforms.ToTensor(),
                              target_transform=None, download=download)
      
      train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, drop_last=drop_last)
      test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, drop_last=drop_last)
      return train_loader, test_loader