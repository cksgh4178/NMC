# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 11:45:22 2021

@author: bsclab
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt 

path = "C:\\Users\\bsclab\\Dropbox (개인용)\\항만생산성\\PythonCode\\Step.12\\"
Data = pd.read_csv(path + 'Model_Data.csv')

data_pre = Data[['WORK_EQT_NO','SCHEDULE_NO','EMP_NO','HATCH_NO','L','U','TWIN','F','M','EMP_CHANGE','WK_ID_CHANGE',
                 'QC_Delay','R_ETW', 'BAY_CHANGE']]

scaler = MinMaxScaler()

x = data_pre.drop(['R_ETW', 'BAY_CHANGE'],axis=1).to_numpy()
y = data_pre['R_ETW'].to_numpy().reshape((-1, 1))

scaler.fit(x)
x = scaler.transform(x)

scaler.fit(y)
y = scaler.transform(y)

class TensorData(Dataset):

    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index] 

    def __len__(self):
        return self.len

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

trainsets = TensorData(x, y)
trainloader = torch.utils.data.DataLoader(trainsets, batch_size=512, shuffle=True)

testsets = TensorData(x_test, y_test)
testloader = torch.utils.data.DataLoader(testsets, batch_size=512, shuffle=False)

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__() # 모델 연산 정의
        self.fc1 = nn.Linear(12, 50, bias=True) # 입력층(12) -> 은닉층1(50)으로 가는 연산
        self.fc2 = nn.Linear(50, 30, bias=True) # 은닉층1(50) -> 은닉층2(30)으로 가는 연산
        self.fc3 = nn.Linear(30, 1, bias=True) # 은닉층2(30) -> 출력층(1)으로 가는 연산
        self.dropout = nn.Dropout(0.2) # 연산이 될 때마다 20%의 비율로 랜덤하게 노드를 없앤다.

    def forward(self, x): # 모델 연산의 순서를 정의
        x = F.relu(self.fc1(x)) # Linear 계산 후 활성화 함수 ReLU를 적용한다.  
        x = self.dropout(F.relu(self.fc2(x))) # 은닉층2에서 드랍아웃을 적용한다.(즉, 30개의 20%인 6개의 노드가 계산에서 제외된다.)
        x = F.relu(self.fc3(x)) # Linear 계산 후 활성화 함수 ReLU를 적용한다.  
      
        return x
    
model = Regressor()
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_ = [] # loss를 저장할 리스트.
n = len(trainloader)

for epoch in range(500):

  running_loss = 0.0 

  for i, data in enumerate(trainloader, 0): 
    
    inputs, values = data 

    optimizer.zero_grad() 

    outputs = model(inputs) 
    loss = criterion(outputs, values) 
    loss.backward() 
    optimizer.step() 

    running_loss += loss.item() 
  
  loss_.append(running_loss/n) 
  
def evaluation(dataloader):

  predictions = torch.tensor([], dtype=torch.float) 
  actual = torch.tensor([], dtype=torch.float) 

  with torch.no_grad():
    model.eval() 

    for data in dataloader:
      inputs, values = data
      outputs = model(inputs)

      predictions = torch.cat((predictions, outputs), 0) 
      actual = torch.cat((actual, values), 0) 

  predictions = predictions.numpy() # 넘파이 배열로 변경.
  actual = actual.numpy() # 넘파이 배열로 변경.
  mae = mean_absolute_error(predictions, actual) # sklearn을 이용해 MAE를 계산.

  return mae

train_mae = evaluation(trainloader) 
test_mae = evaluation(testloader)

print(f'train MAE:{train_mae}')
print(f'test MAE:{test_mae}')

target = [[0.006099]]
scaler.inverse_transform(target)
