# ResNet
* 모델의 깊이가 깊을수록 학습이 잘되지만 일정 깊이 이상이 되면 오히려 성능이 떨어지는 문제 발생
* 이 문제를 해결하기 위해 **잔차 학습(Residual Learning)** 이라는 방법을 제시하고 이를 구현한 모델이 ResNet
* 특정 위치에서 입력이 들어왔을 때 합성곱 연산을 통한 결과와 입력으로 들어온 결과를 더해서 다음 레이어에 전달
	* 잔차 학습 블록은 이전 단계에서 뽑았던 특성들을 변형시키지 않고 그대로 더해서 전달하기 때문에 손실이 줄어들지 않고 학습이 가능

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbFPOry%2FbtqzR2En9ry%2F2DTETgT1BkCrW74hKQCsrk%2Fimg.png" width = "500" height = "350">

* ResNet 모델과 다른 모델 구조 비교

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbdQ7nn%2FbtqzVCKyKVV%2F5nkGhNvCqK9BcIgasYRxH0%2Fimg.jpg" width = "400" height = "800">

* ResNet 상에서 실선은 해상도가 바뀌지 않는 경우, 점선은 다운샘플링으로 인해 해상도가 바뀌는 경우
* GoogleNet의 1 X 1 합성곱 연산을 통한 연산량을 줄이는 기법이 여기서도 적용됨
	* 조금 다른 형태로 변경하여 적용하여 Bottleneck 블록이라는 이름을 붙임

<img src="https://t1.daumcdn.net/cfile/tistory/9907E0375CB8A11F0F" width = "600" height = "300">

```python
import torch
import torch.nn as nn

def conv_block1(in_dim, out_dim, act_fn, stride=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride),
        act_fn
        )
    return model

def conv_block3(in_dim, out_dim, act_fn):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        act_fn
        )
    return model

class BottleNeck(nn.Module):
    
    def __init__(self,in_dim, mid_dim, out_dim, act_fn, down=False):
        super(BottleNeck, self).__init__()
        self.act_fn = act_fn
        self.down = down
        
        if self.down:
            self.layer = nn.Sequential(
                conv_block1(in_dim, mid_dim, act_fn, 2),
                conv_block3(mid_dim, mid_dim, act_fn),
                conv_block3(mid_dim, out_dim, act_fn)
                )
            self.downsample = nn.Conv2d(in_dim, out_dim, 1, 2)
        else:
            self.layer = nn.Sequential(
                conv_block1(in_dim, mid_dim, act_fn),
                conv_block3(mid_dim, mid_dim, act_fn),
                conv_block1(mid_dim, out_dim, act_fn)
                )
            self.dim_equalizer = nn.Conv2d(in_dim, out_dim, kernel_size=1)
            
    def forward(self, x):
        if self.down:
            downsample = self.downsample(x)
            out = self.layer(x)
            out = out + downsample
        else:
            out = self.layer(x)
            if x.size() is not out.size():
                x = self.dim_equalizer(x)
            out = out + x
            return out

class ResNet(nn.Module):
    
    def __init__(self, base_dim, batch_size, num_classes=2):
        super(ResNet, self).__init__()
        self.batch_size = batch_size
        self.act_fn = nn.ReLU()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, base_dim, 7, 2, 3),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1)
            )
        self.layer_2 = nn.Sequential(
            BottleNeck(base_dim, base_dim, base_dim*4, self.act_fn),
            BottleNeck(base_dim*4, base_dim, base_dim*4, self.act_fn),
            BottleNeck(base_dim*4, base_dim, base_dim*4, self.act_fn, down=True)
            )
        self.layer_3 = nn.Sequential(
            BottleNeck(base_dim*4, base_dim*2, base_dim*8, self.act_fn),
            BottleNeck(base_dim*8, base_dim*2, base_dim*8, self.act_fn),
            BottleNeck(base_dim*8, base_dim*2, base_dim*8, self.act_fn),
            BottleNeck(base_dim*8, base_dim*2, base_dim*8, self.act_fn, down = True)
            )
        self.layer_4 = nn.Sequential(
            BottleNeck(base_dim*8, base_dim*4, base_dim*16, self.act_fn),
            BottleNeck(base_dim*16, base_dim*4, base_dim*16, self.act_fn),
            BottleNeck(base_dim*16, base_dim*4, base_dim*16, self.act_fn),
            BottleNeck(base_dim*16, base_dim*4, base_dim*16, self.act_fn),
            BottleNeck(base_dim*16, base_dim*4, base_dim*16, self.act_fn),
            BottleNeck(base_dim*16, base_dim*4, base_dim*16, self.act_fn, down = True)
            )
        self.layer_5 = nn.Sequential(
            BottleNeck(base_dim*16, base_dim*8, base_dim*32, self.act_fn),
            BottleNeck(base_dim*32, base_dim*8, base_dim*32, self.act_fn),
            BottleNeck(base_dim*32, base_dim*8, base_dim*32, self.act_fn)
            )
        self.avgpool = nn.AvgPool2d(7,1)
        self.fc_layer = nn.Linear(base_dim*32, num_classes)
    
    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.avgpool(out)
        out = out.view(self.batch_size, -1)
        out = self.fc_layer(out)
        return out 
```
