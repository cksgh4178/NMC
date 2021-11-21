# VGG-16
* 16개의 layer를 가진 합성곱 신경망 네트워크
* 파이토치 공식 구현보다 간단한 구현
* 파이토치 공식 구현은 [공식 구현](https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py)에서 확인 가능

```python
def conv_2block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
        )
    return model

def conv_3block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
        )
    return model

class VGG_16(nn.Module):
    def __init__(self, base_dim, num_classes = 2):
        super(VGG_16, self).__init__()
        self.feature = nn.Sequential(
            conv_2block(3, base_dim),
            conv_2block(base_dim, 2*base_dim),
            conv_3block(2*base_dim, 4*base_dim),
            conv_3block(4*base_dim, 8*base_dim),
            conv_3block(8*base_dim, 8*base_dim),           
          )
        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim*7*7, 100),
            nn.ReLU(True),
            nn.Linear(100, 20),
            nn.ReLU(True),
            nn.Linear(20, num_classes)
            )
```

* VGG모델은 깊이의 영향만을 알기 위해 3 X 3 합성곱, 맥스 풀링 만 사용한 구조(커널 사이즈 전부 고정)
* 앞의 블록 함수들은 코드 반복을 줄이기 위한 함수 (Conv layer가 몇 개 이어지냐에 따라 2, 3으로 나뉨)
* 공식 구현에 따르면 모델의 레이어를 11 ~ 19부터 자유롭게 조절 가능 
* 현재 코드는 16개의 레이어만을 다루고 있어 모델을 새로 짜야 다른 깊이가 가능함
* VGG 모델 구성 (D열이 VGG-16)
![image](https://user-images.githubusercontent.com/23060537/142754184-b348d689-f02d-4f92-b960-61b283b4b778.png)
