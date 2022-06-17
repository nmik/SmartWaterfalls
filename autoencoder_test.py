import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn import datasets
from sklearn import metrics
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from torch.nn import functional as F

n_cl=2
model_K = KMeans(n_clusters=n_cl,n_init=50,max_iter=1000)
model_H = AgglomerativeClustering(n_clusters=n_cl, linkage='ward')


#AUTOENCODER

class autoencoder(nn.Module):
    def __init__(self, in_channels, n_e=10, xlen=9376,h1=10000):
        super(autoencoder, self).__init__()

        self.conv1=nn.Conv2d(in_channels=in_channels, out_channels=in_channels*2, kernel_size=4, stride=2, padding=0)
        self.maxpool=nn.MaxPool2d(kernel_size=(2,3), stride=(1,2), padding=0) #Return indices??
        self.conv2=nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels*4, kernel_size=(2,3), stride=2, padding=0)
        self.linear1=nn.Linear(6*xlen-in_channels*4, h1)
        self.linear2=nn.Linear(h1, n_e)
        self.linear3=nn.Linear(n_e, h1)
        self.linear4=nn.Linear(h1, 6*xlen-in_channels*4)
        self.deconv1=nn.ConvTranspose2d(in_channels=in_channels*4, out_channels=in_channels*2, kernel_size=(2,3), stride=2, padding=0)
        self.unpool=nn.MaxUnpool2d(kernel_size=(2,3), stride=(1,2), padding=0)
        self.deconv2=nn.ConvTranspose2d(in_channels=in_channels*2, out_channels=in_channels, kernel_size=4, stride=2, padding=0)


    def encoder(self,x):
        x=F.relu(self.conv1(x))
        #print(len(x[1][1][0]))
        x=self.maxpool(x)
        #print(len(x[1][1][0]))
        x=F.relu(self.conv2(x))
        #print(len(x[1][1][0]))
        x=torch.flatten(x,1)
        #print(len(x[0]))
        return self.linear2(self.linear1(x))

    def decoder(self,x):
        x=self.linear4(self.linear3(x))

        x=torch.reshape(x, (12*4,1171)) #Needs to be changed
        x=self.deconv1(x)
        x=self.unpool(x)
        x=self.deconv2(x)
        return x


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x








