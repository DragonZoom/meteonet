#
#  An example of meteonet dataloader usage to train a U-Net 
#  [1] Bouget et al, 2021, https://www.mdpi.com/2072-4292/13/2/246#B24-remotesensing-13-00246

from loader.meteonet import MeteonetDataset
from loader.samplers import meteonet_random_oversampler, meteonet_sequential_sampler
from torch.utils.data import DataLoader
import os, torch
from loader.filesets import bouget21
from datetime import datetime
from platform import processor # for M1/M2 support
from tqdm import tqdm

## user parameters
input_len    = 12
time_horizon = 6
stride       = input_len
thresholds   = [0.1, 1, 2.5]  # series of thresholds (unit: mm/h)
thresholds   = [100*k/12 for k in thresholds] #  unit: CRF over 5 minutes in 1/100 of mm (as meteonet data)
oversampling = 0.9 # oversampling of the last class

model_size   = 8 # to do
wind_dir     = 'data/windmaps'  # or None if you don't want assimilate wind maps

epochs       = 20
batch_size   = 128
lr           = {0:8e-4, 4:1e-4}
wd           = {0:1e-5 ,4:5e-5} # 1e-8
clip_grad    = 0.1

snapshot_step = 5

if torch.backends.cuda.is_built() and torch.cuda.is_available():
    device   = 'cuda'
elif torch.backends.mps.is_built():
    device    = 'mps'
else:
    device   = 'cpu'

num_workers  = 8 if processor() != 'arm' else 0 # no multithreadings on M1/M2 :(

if os.path.isfile( 'data/.full_dataset'):
    dataset_size = 'full'
elif os.path.isfile( 'data/.reduced_dataset'):
    datadset_size = 'reduced'
else:
    print( 'No dataset found. Please download one with download-meteonet-*.sh scripts.')
    exit (1)

print(f"""
Data params:
   {dataset_size = }
   {input_len = } (history of {12*5} minutes)
   {time_horizon = } (nowcasting at {time_horizon*5} minutes)
   {stride = }
   model = Unet classif
   model_size = ?
   {wind_dir = }
   {len(thresholds)} classes ({thresholds=})
   
Train params:
   {epochs = } 
   {batch_size = }
   {lr = }
   {wd = }
   {clip_grad = }

Others params:
   {device = }
   {snapshot_step = }
   {num_workers = }
""")

# split in validation/test sets according to Section 4.1 from [1]
train_files, val_files, _ = bouget21('data/rainmaps')

# datasets
train_ds = MeteonetDataset( train_files, input_len, input_len + time_horizon, stride, wind_dir=wind_dir, cached='data/train.npz', tqdm=tqdm)
val_ds   = MeteonetDataset( val_files, input_len, input_len + time_horizon, stride, wind_dir=wind_dir, cached='data/val.npz', tqdm=tqdm)
val_ds.norm_factors = train_ds.norm_factors

device = torch.device(device)

# samplers for dataloaders
train_sampler = meteonet_random_oversampler( train_ds, thresholds[-1], oversampling)
val_sampler   = meteonet_sequential_sampler( val_ds)

# dataloaders
train_loader = DataLoader(train_ds, batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=True)


print(f"""
size of train files/items/batch
     {len(train_files)} {len(train_ds)} {len(train_loader)}
size of  files/items/batch
     {len(val_files)} {len(val_ds)} {len(val_loader)}
""")


## Model & training procedure
from models.unet import UNet
if wind_dir:
    model = UNet(n_channels = input_len*3, n_classes = len(thresholds), bilinear = True)
else:
    model = UNet(n_channels = input_len, n_classes = len(thresholds), bilinear = True)
    
from trainer import train_meteonet_classif

scores, rundir = train_meteonet_classif( train_loader, val_loader, model, thresholds, epochs, lr, wd, snapshot_step, device = device)

import torch
torch.save( { 'input_len': input_len, 'time_horizon': time_horizon, 'stride': stride,
              'thresholds': thresholds, 'oversampling': oversampling, model_size: model_size,
              'batch_size': batch_size,
              'lr': lr, 'wd': wd, epochs: epochs,
              'database': dataset_size,
              'wind_dir': wind_dir},
            os.path.join(rundir,'hyperparams.pt'))
