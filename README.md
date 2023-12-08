# Stacked Denoising AutoEncoder using MNIST datasets

[MMORPG에서 비정상 행위를 AI로 탐지한 넷마블](https://youtu.be/2Kxnufo54UU?si=nWyloU729j5ajAJC) 영상에서 영감을 얻어 MNIST 데이터셋을 사용하여 Stacked Denoising AutoEncoder 모델을 구현했습니다.  
정상 데이터로 훈련된 오토인코더 모델에 비정상 데이터가 입력으로 주어지는 경우 비정상적인 Loss 값이 도출되는 원리를 이용해 이상 탐지에 활용할 수 있습니다.  
넷마블의 경우 게임에서 발생하는 로그 데이터를 활용하여 전체 데이터를 학습하고 비정상 행위를 탐지하는 방식으로 모델을 활용하고 있습니다.  

## Noise가 추가된 이미지 데이터 생성하기
이미지 분야에서 기본적인 노이즈 중 하나인 가우시안 노이즈(Gaussian noise)를 추가했습니다.  
가우시안 노이즈는 보통 이미지의 전송 과정에서 나타나는데, 이미지를 압축하여 보낸 후 복구하는 과정에서 원래의 픽셀값이 아닌 오차가 생기는 것입니다.  
  
본 구현에서는 평균이 0이고 표준편차가 0.1인 정규(가우시안) 분포에서 랜덤한 값들을 생성하여 노이즈를 추가했습니다.  
게임 데이터를 사용하는 경우 환경에 따라, 여러 요인에 따라 발생하는 이상값(노이즈)들이 존재할 것이기 때문에 이를 제어함으로써 비정상 행위 탐지 성능을 높일 수 있습니다.  

## Stacked AutoEncoder

<img src="https://github.com/cku7808/Stacked-denoising-AE/assets/66200628/749a18b6-6105-4f29-b118-6c6270aa1c28" width=400 height=400>

Stacked AutoEncoder는 여러 개의 오토인코더를 쌓아 올린 구조, Basic AutoEncdoer에서 Hidden Layer를 깊게 만든 구조입니다.  
해당 구현에서는 MNIST datasets를 사용하기 때문에 Convolution layer를 사용했습니다.

## Training & Test
학습 환경은 다음과 같습니다.  
`Epochs : 100`  
`Optimizer : SGD / Adam`  
`Loss Function : MSE`  
`Learning Rate : 0.00001`  
두 옵티마이저의 성능 비교를 위해서 random seed를 0으로 고정했습니다.

#### Adam Optimizer

<img src="https://github.com/cku7808/Stacked-denoising-AE/assets/66200628/c3037ba8-eba4-4f24-8909-9c1cdd5aca41" width="400" height="300">
<img src="https://github.com/cku7808/Stacked-denoising-AE/assets/66200628/873124f5-fe88-45e9-a457-20afcf9134fc" width="600" height="250">
Test Loss : 0.0010



#### SGD Optimizer

<img src="https://github.com/cku7808/Stacked-denoising-AE/assets/66200628/61876b64-539b-4d73-9d03-5e0f2295fb10" width="400" height="300">
<img src="https://github.com/cku7808/Stacked-denoising-AE/assets/66200628/1fdcf1ee-3e25-4daa-96af-142fd717cfd6" width="600" height="250">
Test Loss : 0.0026  
  
Adam과 SGD 모두 Loss가 Epoch이 증가함에 따라 잘 감소하는 모습을 보이며 원본 복원 결과 또한 원본과 매우 유사하게 잘 복원한 모습입니다.
그 중에서도 Test Loss가 0.001로 더 작은 것으로 볼 때 Adam Optimizer를 사용해 학습했을 때의 결과가 더 좋음을 알 수 있었습니다.
