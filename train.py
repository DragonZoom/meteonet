#
#  An example of meteonet dataloader usage to train a U-Net [Bouget et al, 2021]
#

import os
from loader.utilities import load_params
from loader.meteonet import MeteonetDataset

os.system('mkdir -p params')

input_len = 12
time_horizon = 6
thresholds_in_mmh = [0, 0.1, 1, 2.5] 
thresholds_in_cent_mm = [100*k/12 for k in thresholds_in_mmh] # CRF sur 5 minutes en 1/100 de mm comme les données MétéoNet.

experiment = "params/experiment1.npz"

if not os.path.isfile(experiment):
    from loader.utilities import by_year, split_date, save_params, create_params

    # define train/val/test sets and normalisation parameters
    rainmaps_dir = "data/rainmaps"
    
    train_years  = [2016, 2017]   # two years for the train set
    val_years    = [2018]        # one year for the validation set

    train_files = []
    for y in train_years:
        train_files += sorted(filter(lambda f: by_year(f,y), rainmap_files), key=lambda f: split_date(f))
        
    val_files = []
    for y in val_years:
        val_files += sorted(filter(lambda f: by_year(f,y), rainmap_files), key=lambda f: split_date(f))

    print('compute normalisation parameters...')
    params = create_params( train_files, val_files, thresholds_in_cent_mm)
    print(f'writing {experiment} ...')
    save_params(params, experiment)
else:
    params = load_params(experiment)


device = 'cpu'

train = MeteonetDataset( params, 'train', input_len, input_len + time_horizon, input_len, thresholds_in_cent_mm)
val   = MeteonetDataset( params, 'val',  input_len, input_len + time_horizon, input_len, thresholds_in_cent_mm)

if False:
    """ this test successfully passes"""
    print("Test 1...")
    from tqdm import tqdm
    pbar = tqdm(train, unit=" files")
    for t in pbar:
        pass
    print("pass train set")


# oversampling !
from torch.utils.data import DataLoader

train_loader = DataLoader(train, batch_size = 32) #ne marche pas sur le mac ???? : , num_workers=8, pin_memory=True) #, sampler=train_sampler,
val_loader   = DataLoader(val, batch_size = 32)

if False:
    print("Test 2...")
    from tqdm import tqdm
    print(len(train),len(train_loader))
    pbar = tqdm(train_loader, unit=" 32-file")
    for t in pbar:
        pass
    print('pass train dataloader')

## Model & training procedure
from trainer import Trainer
import torch.nn as nn
from torch.optim import Adam
from model.unet import UNet

logit = nn.BCEWithLogitsLoss()
logit.to(device)
unet = UNet(n_channels = input_len, n_classes = len(thresholds_in_mmh)-1, bilinear = True)
#net.to(device)
adam = Adam(unet.parameters(), lr=0.01, weight_decay=1e-8)

epochs = 100

os.system("mkdir -p runs")

def get_xy( data):
    return data['inputs'], data['target']

trainer = Trainer( unet, logit, adam, get_xy, 'runs', device, 0.1)
train_loss, valid_loss = trainer.fit( train_loader, val_loader, epochs)

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
