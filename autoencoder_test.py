import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn import functional as F
import sys

# DATASET CLASS TO AVOID USING TARGETS

class UnlabeledTensorDataset(Dataset):
    def __init__(self,data):
        self.data = data
        
    def __getitem__(self, index):
        x = self.data[index]
        return x
    
    def __len__(self):
        return len(self.data)


#AUTOENCODER

class autoencoder(nn.Module):
    def __init__(self, in_channels, n_e=10, xlen=9376,h1=10000):
        super(autoencoder, self).__init__()

        k2=int((xlen%8)/2.+3)

        self.conv1=nn.Conv2d(in_channels=in_channels, out_channels=in_channels*2, kernel_size=4, stride=2, padding=0)
        self.maxpool=nn.MaxPool2d(kernel_size=(2,k2), stride=(1,2), padding=0,return_indices=True)
        self.conv2=nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels*4, kernel_size=(2,3), stride=2, padding=0)
        self.linear1=nn.Linear(int((xlen/8 - (1+k2)/4)*in_channels*4), h1)
        self.linear2=nn.Linear(h1, n_e)
        self.linear3=nn.Linear(n_e, h1)
        self.linear4=nn.Linear(h1, int((xlen/8 - (1+k2)/4)*in_channels*4))
        self.unflatten=nn.Unflatten(1, (in_channels*4,1,int((xlen/8 - (1+k2)/4))))
        self.deconv1=nn.ConvTranspose2d(in_channels=in_channels*4, out_channels=in_channels*2, kernel_size=(2,3), stride=2, padding=0)
        self.unpool=nn.MaxUnpool2d(kernel_size=(2,k2), stride=(1,2), padding=0)
        self.deconv2=nn.ConvTranspose2d(in_channels=in_channels*2, out_channels=in_channels, kernel_size=4, stride=2, padding=0)


    def encoder(self,x):
        x=F.relu(self.conv1(x))
        #breakpoint()
        x,index=self.maxpool(x)
        x=F.relu(self.conv2(x))
        x=torch.flatten(x,1)
        x=self.linear2(self.linear1(x))
        return x,index

    def decoder(self,x,index):
        x=self.linear4(self.linear3(x))
        x=self.unflatten(x)
        #breakpoint()
        x=self.deconv1(x)
        x=self.unpool(x,index)
        x=self.deconv2(x)
        return x


    def forward(self, x):
        x,index = self.encoder(x)
        x = self.decoder(x,index)
        return x








