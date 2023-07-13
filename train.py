#
#  An example of meteonet dataloader usage to train a U-Net [Bouget et al, 2021]
#
from glob import glob
from tqdm import tqdm
from loader.meteonet import MeteonetDataset
from loader.samplers import meteonet_random_oversampler, meteonet_sequential_sampler
from torch.utils.data import DataLoader
import os, torch

## user parameters
input_len    = 12
time_horizon = 6
stride       = input_len
thresholds   = [0.1, 1, 2.5]  # series of thresholds (unit: mm/h)
thresholds   = [100*k/12 for k in thresholds] #  unit: CRF over 5 minutes in 1/100 of mm (as meteonet data)
oversampling = 0.8 # oversampling of the last class

train_files = glob('data/rainmaps/y201[67]-*')
val_files   = glob('data/rainmaps/y2018-*')

# datasets
train_ds = MeteonetDataset( train_files, input_len, input_len + time_horizon, stride, cached='data/train.npz', tqdm=tqdm)
val_ds   = MeteonetDataset( val_files, input_len, input_len + time_horizon, stride, cached='data/val.npz', tqdm=tqdm)

device = torch.device('cuda')

# samplers for dataloaders
train_sampler = meteonet_random_oversampler( train_ds, thresholds[-1], oversampling)
val_sampler = meteonet_sequential_sampler( val_ds)

# dataloaders
train_loader = DataLoader(train_ds, batch_size = 32, sampler=train_sampler, num_workers=8, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size = 32, sampler=val_sampler, num_workers=8, pin_memory=True) 


## Model & training procedure
from trainer import Trainer
import torch.nn as nn
from torch.optim import Adam
from model.unet import UNet
from loader.utilities import map_to_classes

loss = nn.BCEWithLogitsLoss()
loss.to(device)

model = UNet(n_channels = input_len, n_classes = len(thresholds), bilinear = True)
model.to(device)

optimizer = Adam(model.parameters(), lr=0.01, weight_decay=1e-8)

epochs = 100

os.system("mkdir -p runs")

def get_xy( data):
    return data['inputs'], map_to_classes(data['target'], thresholds)


train_losses = []
val_losses = []

def calculate_TPFPFN( pred, true):
    """ BCNM*BCNM
    """
    diff = 2*pred - true
    return torch.cat((torch.sum(diff==1,dim=(0,2,3)), # TP
                      torch.sum(diff==2,dim=(0,2,3)), # FP
                      torch.sum(diff==-1,dim=(0,2,3)  # FN
                    ))).reshape(3,-1)

def calculate_scores( TPFPFN):
    TP, FP, FN = TPFPFN
    precis = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*precis*recall/(precis+recall) # if precis>0 or recall>0 else np.nan
    return precis,recall,f1

for epoch in range(40):
    # one epoch    
    train_loss = 0
    for batch in tqdm(train_loader):
        x,y = get_xy(batch)
        x,y = x.to(device), y.to(device)

        y_hat = model(x)

        l = loss(y_hat, y)
        train_loss += l.item()

        optimizer.zero_grad()
        l.backward()
        nn.utils.clip_grad_value_(model.parameters(), 0.1)
        optimizer.step()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    print(f'epoch {epoch+1} {train_loss=}')

    if epoch % 5 == 4: # validation toutes les 5 epochs
        val_loss = 0
        TPFPFN_pred = 0
        TPFPFN_pers = 0        
        for batch in tqdm(val_loader):
            x,y = get_xy(batch)
            x,y = x.to(device), y.to(device)
            with torch.no_grad():
                y_hat = model(x)
            
            l = loss(y_hat, y)
            val_loss += l.item()
            TPFPFN_pred += calculate_TPFPFN(torch.sigmoid(y_hat)>.5, y)
            TPFPFN_pers += calculate_TPFPFN(map_to_classes(batch['persistence'], thresholds).to(device), y)
            
        precis_pred, recall_pred, f1_pred =  calculate_scores( TPFPFN_pred)
        precis_pers, recall_pers, f1_pers =  calculate_scores( TPFPFN_pers)

            
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f'epoch {epoch+1} {val_loss=}')

torch.save(model.state_dict, "model.pt")

import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.plot(train_losses)
plt.subplot(1,2,2)
plt.plot(val_losses)
plt.show()

#trainer = Trainer( model, loss, solver, get_xy, 'runs', device, 0.1)
#train_loss, valid_loss = trainer.fit( train_loader, val_loader, epochs)



exit(0)
import torch
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm


# model
device = 'cpu'


# optimisation




# train procedure
for epoch in range(epochs):
#    if epoch in lr.keys():
#        print('***:',lr[epoch], wd[epoch])
#        optimizer = optim.Adam(net.parameters(), lr=lr[epoch], weight_decay=wd[epoch])
            
    net.train()
    epoch_loss = 0
    n_batch = len(train_loader)

    with tqdm(total=n_batch, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
        for batch in train_loader:
            imgs = batch['inputs']  #BC(temp)HW
            true_imgs = batch['target']   #BClsHW
            assert imgs.shape[1] == net.n_channels, \
                f'Network has been defined with {net.n_channels} input channels, ' \
                f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'
                   
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_imgs = true_imgs.to(device=device, dtype=torch.float32)
                
            imgs_pred = net(imgs)   #BClsHW
            
            loss = criterion(imgs_pred, true_imgs)
            epoch_loss += loss.item()
            pbar.set_postfix(**{'loss (epoch)': epoch_loss})
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()
            pbar.update()
