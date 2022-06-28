import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn import functional as F
from utils import *
from autoencoder_test import *
import sys


model = autoencoder(in_channels=12)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
EPOCHS=1
file_folder_path='./testdata/'



dataset=load_grb_images(file_folder_path)
#del dataset[41] #It has 7500 columns instead of 9376
del dataset[3:]

dataset=torch.tensor(np.array(dataset))
target=torch.ones(len(dataset))
trainset=TensorDataset(target,dataset)
trainloader=DataLoader(trainset,batch_size=8,shuffle=False)


loss_=[]
for epoch in range(EPOCHS):
    loss, cnt = 0, 0
    for target_,data_ in trainloader:
        optimizer.zero_grad()
        outputs = model(data_.type(torch.FloatTensor))
        outputs=outputs.type(torch.DoubleTensor)
        train_loss = criterion(outputs, data_)
        train_loss.backward()
        optimizer.step()
        breakpoint()
        loss += train_loss.item()
    loss = loss / len(trainloader)
    loss_.append(loss)

    if epoch%2 == 0:
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, EPOCHS, loss))
    
print('----------------------------')
print('Building LOSS plots...')
plt.figure()
plt.plot(range(EPOCHS), loss_, 'o--', label='Train Loss')
plt.xlabel('EPOCH')
plt.ylabel('LOSS')
plt.yscale('log')
plt.show()
plt.close()






