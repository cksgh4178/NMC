```python
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

data_pre = Data[['HATCH_NO','L','U','TWIN','F','M','EMP_CHANGE','WK_ID_CHANGE',
                 'QC_Delay','R_ETW', 'BAY_CHANGE','BAY_MOVE']]

scaler = MinMaxScaler()

x = data_pre.drop(['R_ETW', 'BAY_MOVE', 'QC_Delay'],axis=1)
y = data_pre['R_ETW']

#scaler.fit(x)
#x = scaler.transform(x)

#scaler.fit(y)
#y = scaler.transform(y)

class TensorData(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.len = self.y_data.shape[0]
    
    def __getitem__(self, index):
    
        return self.x_data[index], self.y_data[index] 
    
    def __len__(self):
        return self.len

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

x_train = torch.from_numpy(x_train.astype(float).values)
y_train = torch.from_numpy(y_train.astype(float).values)

x_test = torch.from_numpy(x_test.astype(float).values)
y_test = torch.from_numpy(y_test.astype(float).values)

trainsets = TensorData(x_train, y_train)
trainloader = torch.utils.data.DataLoader(trainsets, batch_size=512, shuffle=True)

testsets = TensorData(x_test, y_test)
testloader = torch.utils.data.DataLoader(testsets, batch_size=512, shuffle=False)

use_cuda = torch.cuda.is_available()

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__() # 모델 연산 정의
        self.fc1 = nn.Linear(9, 512, bias=True) # 입력층(9) -> 은닉층1(50)으로 가는 연산
        self.fc2 = nn.Linear(512, 256, bias=True) # 은닉층1(50) -> 은닉층2(30)으로 가는 연산
        self.fc3 = nn.Linear(256, 128, bias=True)
        self.fc4 = nn.Linear(128, 64, bias=True)
        self.fc5 = nn.Linear(64, 8, bias=True) # 은닉층2(30) -> 출력층(1)으로 가는 연산
        self.fc6 = nn.Linear(8, 1, bias = True)
        self.dropout = nn.Dropout(0.2) # 연산이 될 때마다 20%의 비율로 랜덤하게 노드를 없앤다.
        self.hidden = nn.Sequential(
            self.fc1,
            nn.Tanh(),
            self.fc2,
            nn.Tanh(),
            self.fc3,
            nn.Tanh(),
            self.fc4,
            nn.Tanh(),
            self.fc5,
            nn.Tanh(),
            self.fc6
        )
        if use_cuda:
            self.hidden = self.hidden.cuda()

    def forward(self, x): # 모델 연산의 순서를 정의
        x = F.relu(self.fc1(x)) # Linear 계산 후 활성화 함수 ReLU를 적용한다.  
        x = self.dropout(F.relu(self.fc2(x))) # 은닉층2에서 드랍아웃을 적용한다.(즉, 30개의 20%인 6개의 노드가 계산에서 제외된다.)
        x = F.relu(self.fc3(x)) # Linear 계산 후 활성화 함수 ReLU를 적용한다.
        x = self.dropout(F.relu(self.fc4(x))) # 은닉층4에서 드랍아웃을 적용한다.(즉, 30개의 20%인 6개의 노드가 계산에서 제외된다.)
        x = F.relu(self.fc6(self.fc5(x))) # Linear 계산 후 활성화 함수 ReLU를 적용한다.
        
        return x

model = Regressor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

train_loss_list = [] # loss를 저장할 리스트.
test_loss_list = []

n = len(trainloader)

for epoch in range(1000):

  train_loss_summary = 0.0 

  for i, data in enumerate(trainloader, 0): 
    
    x_train, y_train = x_train['x_train'], y_train['y_train']
    if use_cuda:
        x_train, y_train = x_train.cuda(), y_train.cuda()
    optimizer.zero_grad() 
    train_pred = model(x_train) 
    train_loss = criterion(train_pred, y_train) 
    train_loss.backward() 
    optimizer.step() 
    
    train_loss_summary += train_loss
    
    if (i+1) % 15 == 0:
        with torch.no_grad():
            test_loss_summary = 0.0
            for j, data in enumerate(testloader, 0):
                x_test, y_test = x_test['x_test'], y_test['y_test']
                if use_cuda:
                    x_test, y_test = x_test.cuda(), y_test.cuda()
                test_pred = model(x_test)
                test_loss = criterion(test_pred, y_test)
                test_loss_summary += test_loss
                
        train_loss_list.append((train_loss_summary/15)**(1/2))
        test_loss_list.append((test_loss_summary/len(testloader))**(1/2))
        test_loss_summary = 0.0
        
        ### IndexError: too many indices for tensor of dimension 2
```