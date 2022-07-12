import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import argparse
import torch
from torch import nn, optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from utils import *
from autoencoder_test import *
import sys


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--datafolder", type=str, required=True,
                    help = "Path to folder with the .npy files")
parser.add_argument("-e", "--epochs", type=int, default=1,
                    help = "Number of epochs for the training")
args = parser.parse_args()


# --------------------------- Data loading  -----------------------------
file_folder_path=args.datafolder
RESHAPE = True

dataset=load_grb_images(file_folder_path)

if RESHAPE:
    count = 0
    badevents_ = []
    dataset_reshaped = []
    for i, grb in enumerate(dataset):
        if grb.shape[2] != 9376:
            print('Nooooooooooooooooooooooooooooo', i, grb.shape[2])
            count += 1
            badevents_.append(i)
        else:
            new_grb = resize_images(grb)
            dataset_reshaped.append(new_grb)
    dataset = dataset_reshaped
    print('Found bad events', count)
    print(np.array(os.listdir(file_folder_path))[np.array(badevents_)])
else:
    del dataset[2:]
    
print('Image x size:', dataset[0][0].shape[1])
# -------------------------------------------------------------------


# --------------------------- NN  param -----------------------------
model = autoencoder(in_channels=12, n_e=10, xlen=dataset[0][0].shape[1], h1=10000)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
EPOCHS=args.epochs
# -------------------------------------------------------------------

dataset=torch.tensor(np.array(dataset))
trainset=UnlabeledTensorDataset(dataset)
trainloader=DataLoader(trainset,batch_size=8,shuffle=False)

loss_=[]
for epoch in range(EPOCHS):
    loss, cnt = 0, 0
    for data_ in trainloader:
        optimizer.zero_grad()
        outputs = model(data_.type(torch.FloatTensor))
        outputs = outputs.type(torch.DoubleTensor)
        train_loss = criterion(outputs, data_)
        # breakpoint()
        train_loss.backward()
        optimizer.step() #Long step
        # breakpoint()
        loss += train_loss.item()
    loss = loss / len(trainloader)
    loss_.append(loss)

    if epoch%2 == 0:
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, EPOCHS, loss))

# Number of total parameters

#pytorch_total_params = sum(p.numel() for p in model.parameters())
#print(pytorch_total_params)


#Output image (first layer)

with torch.no_grad(): 
    # outputs = outputs.type(torch.FloatTensor)
    trainloader_1 = DataLoader(trainset)
    for data_ in trainloader:
        outputs = model(data_.type(torch.FloatTensor))
    tr=transforms.ToPILImage()
    plt.figure()                                   
    plt.title('Long hard')                                                                               
    plt.imshow(tr(outputs[0][0]),aspect='auto',cmap='viridis')
    plt.savefig('output_test_image_0')
    plt.figure()     
    plt.title('Mid hard')                                                                                                              
    plt.imshow(tr(outputs[0][4]),aspect='auto',cmap='viridis')
    plt.savefig('output_test_image_1')
    plt.figure()    
    plt.title('Short hard')                                                                                                               
    plt.imshow(tr(outputs[0][8]),aspect='auto',cmap='viridis')
    plt.savefig('output_test_image_2')
    
print('----------------------------')
print('Building LOSS plots...')
plt.figure()
plt.plot(range(EPOCHS), loss_, 'o--', label='Train Loss')
plt.xlabel('EPOCH')
plt.ylabel('LOSS')
plt.show()
plt.close()






