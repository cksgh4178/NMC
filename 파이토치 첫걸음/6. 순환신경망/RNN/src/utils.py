# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 16:55:54 2021

@author: USER
"""

import torch
import numpy as np

def string_to_onehot(string, shape, char_list):
    start = np.zeros(shape=shape, dtype=int)
    end = np.zeros(shape=shape, dtype=int)
    start[-2] = 1
    end[-1] = 1
    for i in string:
        idx = char_list.index(i)
        zero = np.zeros(shape=shape, dtype=int)
        zero[idx] = 1
        start = np.vstack([start, zero])
    output = np.vstack([start, end])
    return output

def onehot_to_word(vector, char_list):
    onehot = torch.Tensor.numpy(vector)
    return char_list[onehot.argmax()]