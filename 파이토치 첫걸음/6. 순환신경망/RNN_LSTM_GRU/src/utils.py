# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 19:31:40 2021

@author: USER
"""

import random
import torch

def random_chunk(file, file_len, chunk_len):
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    return file[start_index:end_index]

def char_tensor(string, all_characters):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return tensor

def random_training_set(file, file_len, chunk_len, all_characters):
    chunk = random_chunk(file, file_len, chunk_len)
    inp = char_tensor(chunk[:-1], all_characters)
    target = char_tensor(chunk[1:], all_characters)
    return inp,target