# Stacked Denoising AutoEncoder using MNIST datasets

[MMORPG에서 비정상 행위를 AI로 탐지한 넷마블](https://youtu.be/2Kxnufo54UU?si=nWyloU729j5ajAJC) 영상에서 영감을 얻어 동일하게 MNIST 데이터셋을 사용하되 최종 모델로 사용된 Stacked Denoising AutoEncoder 모델을 구현했습니다. 

## Noise가 추가된 이미지 데이터 생성하기
이미지 분야에서 기본적인 노이즈 중 가우시안 노이즈(Gaussian noise)를 추가했습니다.
가우시안 노이즈는 보통 이미지의 전송 과정에서 나타나는데, 이미지를 압축하여 보낸 후 복구하는 과정에서 원래의 픽셀값이 아닌 오차가 생기는 것입니다.

본 구현에서는 평균이 0이고 표준편차가 0.1인 정규 분포에서 랜덤한 값들을 선택하여 노이즈를 추가했습니다.

![output](https://github.com/cku7808/Stacked-denoising-AE/assets/66200628/40ce495f-8bba-4209-b8aa-65c01e0b0667)

## Stacked AutoEncoder
Denoising AutoEncoder와 Basic AutoEncoder의 
