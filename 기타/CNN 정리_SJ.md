# 1. CNN 등장 배경

- `CNN`이 나오기 이전, 이미지 인식은 2차원으로 된 이미지(채널까지 포함해서 3차원)를 1차원배열로 바꾼 뒤 `FC(Fully Connected)`신경망으로 학습시키는 방법을 사용했음.

  - 아래와 같이 이미지의 형상은 고려하지 않고, raw data를 직접 처리하기 때문에 많은 양의 학습데이터가 필요하고 학습시간 역시 길어짐(연산량 증가)

  ![img](https://wikidocs.net/images/page/60324/%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C.png)

  - 또한, 이미지가 회전하거나 움직이면 새로운 입력으로 데이터를 처리해야 함. **즉, 이미지의 특성을 이해하지 못하고 단순 1차원 데이터로 보고 학습 함.**

- 위와 같은 `FC(Fully Connected)`신경망으로 학습시키는 방법은 이미지의 특징을 추출하고 학습하는 데 비효율적이며 정확도도 떨어짐.

- 이러한`FC(Fully Connected)`신경망의 단점을 보완하여 **이미지의 공간적 특징을 유지한 채 학습할 수 있는 모델이 `CNN`.**

# 2. CNN의 특징 및 관련 용어 정리

## 2.1 Convolution

- Convolution의 사전적 정의는 '합성곱'

- 입력데이터에 도장을 찍어 유용한 특성만을 드러나게 하는 것으로 비유할 수 있음.

  - 아래 그림과 같이 5X5 크기의 이미지와 3X3 크기의`Filter`를 가정하면, `Filter`가 이동하며(몇 칸씩 이동할지에 대한 값을 `stride`라고 함) 원본 이미지 데이터에 가중치를 곱하고 곱해진 값 중 특정 값을 추출(`pooling`)하여 특징으로 사용 함.

  ![Picture3](https://user-images.githubusercontent.com/15958325/58845860-ca23ed00-86b7-11e9-805f-ef5c8adcab9f.png)

- 합성곱의 과정에서 `Filter`의 구성에 따라 추출하게되는 이미지의 특징이 달라짐.

## 2.2 Filter(Kernel) 

- 이미지의 특징을 뽑아낼 수 있게 해주는 도구.
- 필터 하나가 하나의 특징을 출력한다고 생각하면 될 거 같음.
- CNN은 신경망에서 **학습을 통해 자동으로 적합한 필터를 학습**한다는 것이 특징.
- 즉, CNN에서는 필터가 가중치 역할.

## 2.3 Channel

- 하나의 컬러 이미지는 red channel, green channel, blue channel로 구성돼 있음(RGB).
  - 컬러이미지와 같은 Multi Channel의 경우, input data의 channel 수와 filter의 channel수는 동일해야 함.
  - ouput은 channel 수와 관계없이 filter의 개수만큼 나옴.

![Picture1](https://user-images.githubusercontent.com/15958325/58845631-d5c2e400-86b6-11e9-87ae-3e82cd8da0c0.png)

- 보통은 연산량을 줄이기 위해 전처리에서 이미지를 흑백(하나의 채널)으로 변환하여 처리함.

  ![Picture4](https://user-images.githubusercontent.com/15958325/58845636-d8253e00-86b6-11e9-80a7-cdbc61739b6f.png)

  ​

## 2.4 Padding

- `Convolution` 레이어에서는`filter` 와 `stride` 의 작용으로 `Feature map`의 크기는 입력데이터보다 작아질 수 밖에 없음.
  - 특히 가장 자리 정보들이 지속적으로 손실 됨.
- 가장 자리 정보들을 잃지 않고,  `Feature map`이 계속해서 작아지는 것을 방지하기 위해 사용하는 것이  `padding`.
- zero padding은 아래 그림과 같이 이미지를 0으로 둘러싼다. 단순히 0을 덧붙였기 때문에 특징추출에 영향을 미치지 않음.
  - same padding는 입력과 출력의 크기가 같도록  `padding`하는 것을 말함.

![Picture7](https://user-images.githubusercontent.com/15958325/58846398-ff313f00-86b9-11e9-8268-7989df7d38f2.png)

## 2.5 Pooling

-   `padding`과정을 통해 원본 이미지의 크기를 계속해서 유지한 채 `FC(Fully Connected)`로 넘어가게 된다면 기존의 문제였던 연산량 문제를 해결할 수 없음.
- 특징을 잘 추출하면서, 적당히 크기도 줄이는 것이 중요함. 그 역할을 하는 것이`pooling`.
  - Max pooling
  - Averge pooling
  - Min pooling
- CNN에서는 주로 Max Pooling을 사용(이는 뉴런이 가장 큰 신호에 반응하는 것과 유사하며, 실제로 노이즈가 감소하고 속도가 빨라지며 이미지 분별력이 좋아진다고 함).
- Pooling layer를 통과하면 이미지 행렬의 크기는 감소하나 channel 수는 변하지 않음.

# 3. 코드 구현

1. 패키지 import 및 데이터 mnist 데이터 로드

- train data: 60,000개, test data: 10,000개
- batach size: 256

```{.python}
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.image as img

batch_size = 256
learning_rate = 0.0001
num_epoch= 10

mnist_train = dset.MNIST("./", train = True, transform=transforms.ToTensor(),
                         target_transform=None, download=True)

mnist_test = dset.MNIST("./", train = False, transform=transforms.ToTensor(),
                         target_transform=None, download=True)

train_loader = torch.utils.data.DataLoader(mnist_train, batch_size = batch_size,
                                           shuffle=True, num_workers = 0, drop_last = True)

test_loader = torch.utils.data.DataLoader(mnist_test, batch_size = batch_size,
                                           shuffle=False, num_workers = 0, drop_last = True)
```

2. CNN 모델 정의

```{.python}
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1,16,5),
            nn.ReLU(),
            nn.Conv2d(16,32,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
            )
        
        self.fc_layer = nn.Sequential(
            nn.Linear(64*3*3,100),
            nn.ReLU(),
            nn.Linear(100,10)
            )
    def forward(self,x):
        out = self.layer(x)
        out = out.view(batch_size,-1)
        out = self.fc_layer(out)
        return out        
```

![img](https://postfiles.pstatic.net/MjAyMTExMDZfNTUg/MDAxNjM2MTgwMjI1NjE4.JESw5gE9dXDdqpGuHXSAoLPvofIA1XWj-l1WcegRBqcg.V4JEcBkC3bjFsNG5g0rQJG1KV7Hl74OUzxll7Vl39ncg.PNG.shout_sg/image.png?type=w773)

3. 모델 학습

- 분류문제이므로, loss function으로 cross-entropy 사용
- 옵티마이저는 Adam 사용.
- 역전파 통해 학습.

```{.python}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

loss_arr =[]
for i in range(num_epoch):
    for j,[image,label] in enumerate(train_loader):
        x = image.to(device) 
        y_ = label.to(device)
        
        optimizer.zero_grad()     
        output = model.forward(x) 
        loss = loss_func(output,y_)
        loss.backward()
        optimizer.step()
        
        if j % 1000 ==0:
            print(loss)
            loss_arr.append(loss.cpu().detach().numpy())
```

4. 모델 테스트

```{.python}
correct = 0
total = 0

with torch.no_grad():
    for image, label in test_loader:
        x = image.to(device)
        y_ = label.to(device)
        
        output = model.forward(x) # 출력 사이즈는 [10]       
        _,output_index = torch.max(output,1) #10개 중 가장 값이 가장 큰 클래스를 output으로.  
        
        total += label.size(0)
        correct += (output_index ==y_).sum().float()
        
    print("Accuracy of Text Data: {}".format(100*correct/total)

>> Accuracy of Text Data: 98.36738586425781
```

