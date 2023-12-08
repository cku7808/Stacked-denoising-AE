import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from Stacked_Denoising_AutoEncoder import StackedDenoisingAE

train_data = datasets.MNIST("dataset", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST("dataset", train=False, download=True, transform=transforms.ToTensor())

device = "cuda" if torch.cuda.is_available() else "cpu"
model = StackedDenoisingAE(1,64)
model = model.to(device)

batch_size = 64
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

# train starts
epochs = 100
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
for epoch in range(epochs):
    optimizer.zero_grad()

    for data in tqdm(train_loader, total=len(train_loader)):
        x, _ = data
        x = x.to(torch.float).to(device)
        noise = torch.randn_like(x) * 0.1  # 노이즈를 GPU로 올리기
        noise = noise.to(torch.float).to(device)
        noised_x = x + noise
        noised_x = torch.clamp(noised_x, 0, 1).to(torch.float).to(device)  # [0, 1] 범위로 클리핑 및 GPU로 올리기

        out_x = model(noised_x)
        loss = criterion(out_x, x)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# test starts
model.eval()
test_loss = 0
for i, data in enumerate(test_loader):
    x, _ = data
    x = x.to(device)  # GPU로 이동

    noise = torch.randn_like(x) * 0.1  # x의 형상을 하는 평균0, 표준편차1인 정규분포에 0.1을 곱해 표준편차를 0.1로 만들기
    noise = noise.to(torch.float).to(device)
    noised_x = x + noise
    noised_x = torch.clamp(noised_x, 0, 1).to(torch.float).to(device)

    out_x = model(noised_x)  # 모델 결과를 CPU로 옮기기
    test_loss += criterion(x,out_x)
    test_loss = test_loss/len(test_loader)
print("Test Loss :",test_loss)