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
            count += 1
            badevents_.append(i)
        else:
            new_grb = resize_images(grb)
            dataset_reshaped.append(new_grb)
    dataset = dataset_reshaped
    print('Found bad events', count)
    if count != 0:
        print(np.array(os.listdir(file_folder_path))[np.array(badevents_)])
else:
    count = 0
    badevents_ = []
    for i, grb in enumerate(dataset):
        if grb.shape[2] != 9376:
            count += 1
            badevents_.append(i)
        else:
            pass
    print('Found bad events', count)
    if count != 0:
        print(np.array(os.listdir(file_folder_path))[np.array(badevents_)])

print('Image x size:', dataset[0][0].shape[1])
# -------------------------------------------------------------------


# --------------------------- NN  param -----------------------------
model = autoencoder(in_channels=12, n_e=3, xlen=dataset[0][0].shape[1], h1=512)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
EPOCHS=args.epochs
# -------------------------------------------------------------------

dataset=torch.tensor(np.array(dataset))
trainset=UnlabeledTensorDataset(dataset)
trainloader=DataLoader(trainset,batch_size=16,shuffle=False)

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
    for i,data_ in enumerate(trainloader):
        l_space=model.encoder(data_.type(torch.FloatTensor))[0]
        if i == 0:
            l_space_vector=l_space
        else:
            l_space_vector=torch.cat((l_space_vector,l_space),dim=0)


    outputs = model(data_.type(torch.FloatTensor))
    tr=transforms.ToPILImage()
    f, (ax1,ax2) = plt.subplots(2, 1, sharex=True, figsize=(8,8))                                  
    ax1.set_title('Long hard reconstructed')                                                                               
    ax1.imshow(tr(outputs[0][0]),aspect='auto',cmap='BuPu')
    ax2.set_title('Long hard true')                                                                               
    ax2.imshow(tr(data_.type(torch.FloatTensor)[0][0]),aspect='auto',cmap='BuPu')
    plt.savefig('output_test_image_0')

    f, (ax1,ax2) = plt.subplots(2, 1, sharex=True, figsize=(8,8))                                  
    ax1.set_title('Mid hard reconstructed')                                                                               
    ax1.imshow(tr(outputs[0][4]),aspect='auto',cmap='BuPu')
    ax2.set_title('Mid hard true')                                                                               
    ax2.imshow(tr(data_.type(torch.FloatTensor)[0][4]),aspect='auto',cmap='BuPu')
    plt.savefig('output_test_image_1')

    f, (ax1,ax2) = plt.subplots(2, 1, sharex=True, figsize=(8,8))                                  
    ax1.set_title('Short hard reconstructed')                                                                               
    ax1.imshow(tr(outputs[0][8]),aspect='auto',cmap='BuPu')
    ax2.set_title('Short hard true.png')                                                                               
    ax2.imshow(tr(data_.type(torch.FloatTensor)[0][8]),aspect='auto',cmap='BuPu')
    plt.savefig('output_test_image_2.png')
    
print('----------------------------')
print('Building LOSS plots...')
plt.figure()
plt.plot(range(EPOCHS), loss_, 'o--', label='Train Loss')
plt.xlabel('EPOCH')
plt.ylabel('LOSS')
plt.savefig('Loss.png')
plt.show()
plt.close()

print('----------------------------')
print('Building latent space distribution plot...')
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(l_space_vector[:,0], l_space_vector[:,1], l_space_vector[:,2])
plt.show()
plt.close()






