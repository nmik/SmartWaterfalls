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
    def __init__(self, **kwargs):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(kwargs["input_shape"], 16),
            nn.Linear(16, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3,16),
            #nn.ReLU(True), 
            nn.Linear(16, kwargs["input_shape"]), 
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#VARIATIONAL AUTOENCODER

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(len(VARIABLES)-2, 16)
        self.fc21 = nn.Linear(16, 3)
        self.fc22 = nn.Linear(16, 3)
        self.fc3 = nn.Linear(3, 16)
        self.fc4 = nn.Linear(16, len(VARIABLES)-2)

    def encoder(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encoder(x.view(-1, len(VARIABLES)-2))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, len(VARIABLES)-2), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD




#DATI PER LA FASE DI TRAINING

print('Importando il file contenente i dati...')
df=pd.read_csv('Dati/df_var.csv',index_col=0)
df_eval=pd.read_csv('Dati/df_eval_var.csv',index_col=0)
df_test=pd.read_csv('Dati/df_test_var.csv',index_col=0)

df=pd.concat([df,df_eval,df_test], ignore_index=True)

print('Selezionando gli eventi con una certa energia per il train...')
df=binning.SelectEnergyPandas(emin,emax,df)
df=binning.SelectZPandas(zmin,zmax,df)

# Da usare per togliere protoni
indexvalue=df[(df['target']==t_p)].index
print(len(indexvalue))
indexvalue=indexvalue[:26353]
df.drop(indexvalue,inplace=True)

# Da usare per togliere elettroni
indexvalue2=df[(df['target']==t_e)].index
print(len(indexvalue2))
indexvalue2=indexvalue2[:13245]
df.drop(indexvalue2,inplace=True)


# NORMALIZZAZIONE
df,maxx,minn=variables_var.NormalizationTrain(df,VARIABLES)



#Prepariamo i dataset e dataloader
target=torch.tensor(df['target'].values)
df.drop(['EvtJointEnergy','target','Tkr1ZDir'], axis='columns', inplace=True)
data=torch.tensor(df.values)
trainset=TensorDataset(target,data)
trainloader=DataLoader(trainset,batch_size=1,shuffle=False)


df.drop(['Energy','ZDir'], axis='columns', inplace=True)
data=torch.tensor(df.values)
trainset=TensorDataset(target,data)
trainloader=DataLoader(trainset,batch_size=32,shuffle=False)



nvar=len(VARIABLES)-2

model = autoencoder(input_shape=nvar)
#model = VAE()

optimizer = optim.Adam(model.parameters(), lr=5e-3)

# mean-squared error loss
criterion = nn.MSELoss()


EPOCHS=15


loss_ = []
for epoch in range(EPOCHS):
    loss, cnt = 0, 0
    for target_,data_ in trainloader:

        batch_features = data_.view(-1, nvar)
        batch_features = batch_features.float()
        
        optimizer.zero_grad()

        outputs = model(batch_features)
        train_loss = criterion(outputs, batch_features)

        #recon_batch, mu, logvar = model(batch_features)
        #train_loss = loss_function(recon_batch, batch_features, mu, logvar)

        train_loss.backward()
        
        # perform parameter update based on current gradients
        optimizer.step()
        
        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()
    
    # compute the epoch training loss
    loss = loss / len(trainloader)
    loss_.append(loss)
    
    # display the epoch training loss
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


enc_output = model.encoder(data.float()).detach().numpy()
#dec_output= model.decoder(enc_output)


#dec_output=model(data.float())

#print(dec_output)
#print(data)

#np.save('enc_output.npy', enc_output)

#enc1, enc2 = model.encoder(data.float())
#enc_output  = enc1.detach().numpy()

xaxis=[]
yaxis=[]
zaxis=[]

for i in range(len(target)):
	xaxis.append(enc_output[i][0])
	yaxis.append(enc_output[i][1])
	zaxis.append(enc_output[i][2])

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
m=ax.scatter(xaxis, yaxis, zaxis, c=target, alpha=0.1)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
legend1=ax.legend(*m.legend_elements(),title='Classes')
ax.add_artist(legend1)
plt.title("Autoencoder")
plt.figure()
plt.show()
plt.close()



model_H.fit(enc_output)
predictions_H=model_H.labels_
#np.savetxt("centroidi.csv", centroidi, delimiter=",")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


scatter=ax.scatter(xaxis, yaxis, zaxis, c=predictions_H, alpha=0.1)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
legend1=ax.legend(*scatter.legend_elements(),title='Classes')
ax.add_artist(legend1)
plt.title("Autoencoder + Hierarchical")
plt.figure()
plt.show()
plt.close()

ARI_H=metrics.adjusted_rand_score(target,predictions_H)
print("ARI_H: ", ARI_H)




