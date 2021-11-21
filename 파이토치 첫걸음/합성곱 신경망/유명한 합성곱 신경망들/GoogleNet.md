# GoogleNet

* **Inception module** 을 가지고 있는 것이 핵심
* 전체적인 구조
![그림](https://postfiles.pstatic.net/MjAxODEwMTNfMTI0/MDAxNTM5NDEyNjIyMzg0.y1z76GN3mP2yfiYyU2lgI5emhqY10EFrcPDFo1B3bBIg.6i7avVtaDGneOB6MmNon1mVTYiW1EBxTd9PNiSzebS8g.PNG.siniphia/googlenet.PNG?type=w773)

* 그림에서 Max pooling 이후에 갈라졌다가 다시 모이는 부분이 인셉션 모듈
* 인셉션 모듈의 구조
![그림](https://postfiles.pstatic.net/MjAxODEwMTNfMTUw/MDAxNTM5NDE1OTI0MzA0.ZtZsSsT7q4iZE6-vArx_8BqIh02ORKHnmQVp58ltzWIg.lJZtNv4fBpAlK-uedyZTkWSwGmadB_JWjOlGUrJg6Kkg.PNG.siniphia/image.png?type=w773)
* 인센셥 모듈은 이전 레이어의 출력에 다양한 크기의 필터를 적용한 다음 그걸 하나로 합치는 형태
* 필터의 크기가 다르다는 것은 각각 다른 연산 범위를 갖는다는 의미
* 1 X 1 합성곱 연산을 통해 모델의 연산량을 줄이는 것이 핵심
	* 1 X 1 합성곱 연산은 결국 fc_layer와 동일함 
* 1 X 1 합성곱 연산으로 연산량을 줄여 깊은 모델을 구성하는 것이 가능하지만 모델이 깊어 학습이 잘되지 않는 문제는 보조 분류기 (auxiliary classifier)를 통해 해결
![그림](https://miro.medium.com/max/550/1*htr2D6tKh3JMS7Acy4BDTw.png)

```python
import torch
import torch.nn as nn

class inception_module(nn.Module):
    def __init__(self, in_dim, out_dim1, mid_dim3, out_dim3, mid_dim5, out_dim5, pool):
        super(inception_module, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim1, 1,1),
            nn.ReLu()
            )
        self.conv_1_3 = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim3, 1,1),
            nn.ReLU(),
            nn.Conv2d(mid_dim3, out_dim3, 3,1,1),
            nn.ReLU()
            )
        self.conv_1_5 = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim5, 1,1),
            nn.ReLU(),
            nn.Conv2d(mid_dim5, out_dim5, 5,1,2),
            nn.ReLU()
            )
        self.max_3_1 = nn.Sequential(
            nn.MaxMaxPool2(3,1,1),
            nn.Conv2d(in_dim, pool, 1,1),
            nn.ReLU()
            )
    
    def forward(self, x):
        out_1 = self.conv1(x)
        out_2 = self.conv_1_3(x)
        out_3 = self.conv_1_5(x)
        out_4 = self.max_3_1(x)
        out = torch.cat([out_1, out_2, out_3, out_4],1)
        return out

class GoogleNet(nn.Module):
    def __init__(self, base_dim, batch_size, num_classes=2):
        super(GoogleNet, self).__init__()
        self.batch_size = batch_size
        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, base_dim, 7,2,3),
            nn.MaxPool2d(3,2,1),
            nn.Conv2d(base_dim, base_dim*3, 3,3,1,1),
            nn.MaxPool2d(3,2,1)
            )
        self.layer_2 = nn.Sequential(
            inception_module(base_dim*3, 64, 96, 128, 16, 32, 32),
            inception_module(base_dim*4, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3,2,1)
            )
        self.layer_3 = nn.Sequential(
            inception_module(480, 192, 96, 208, 16, 48, 64),
            inception_module(512, 160, 112, 224, 24, 64, 64),
            inception_module(512, 128, 128, 256, 24, 48, 64),
            inception_module(512, 112, 144, 288, 32, 64, 64),
            inception_module(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3,2,1)
            )
        self.layer_4 = nn.Sequential(
            inception_module(832, 256, 160, 320, 32, 128, 128),
            inception_module(832, 384, 192, 384, 48, 128, 128),
            nn.AvgPool2d(7,1)
            )
        self.layer_5 = nn.Dropout2d(0.4)
        self.fc_layer = nn.Linear(1024, 1000)
        
    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = out.view(self.batch_size, -1)
        out = self.fc_layer(out)
        return out
```