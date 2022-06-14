#%matplotlib qt
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.integrate import simps
import torch
from torch import nn, optim
import variables_var
import binning
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
    def __init__(self, in_channels, n_e, x=9376,h1=10000):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            #8x9376(x12)
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels*2, kernel_size=4, stride=2, padding=0),
            #3x4687(x24)
            nn.MaxPool2d(kernel_size=(2,3), stride=(1,2), padding=0), #Return indices??
            #2x2343(x24)
            nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels*4, kernel_size=(2,3), stride=2, padding=0),
            # 1x1171(x48)
            #Flattening?
            nn.Linear(4*x-in_channels*4, h1),
            nn.Linear(h1, n_e)

        )
        self.decoder = nn.Sequential(
            nn.Linear(n_e, h1),
            nn.Linear(h1, 4*x-in_channels*4),
            #De-flattening?
            nn.ConvTranspose2d(in_channels=in_channels*4, out_channels=in_channels*2, kernel_size=(2,3), stride=2, padding=0)
            nn.MaxUnpool2d(kernel_size=(2,3), stride=(1,2), padding=0)
            nn.ConvTranspose2d(in_channels=in_channels*2, out_channels=in_channels, kernel_size=4, stride=2, padding=0)
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x








