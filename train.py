#
#  An example of meteonet dataloader usage to train a U-Net 
#  [1] Bouget et al, 2021, https://www.mdpi.com/2072-4292/13/2/246#B24-remotesensing-13-00246

from glob import glob
from tqdm import tqdm
from loader.meteonet import MeteonetDataset
from loader.samplers import meteonet_random_oversampler, meteonet_sequential_sampler
from torch.utils.data import DataLoader
import os, torch
from loader.filesets import filesets_bouget21
from datetime import datetime
from platform import processor # for M1/M2 support

## user parameters
input_len    = 12
time_horizon = 6
stride       = input_len
thresholds   = [0.1, 1, 2.5]  # series of thresholds (unit: mm/h)
thresholds   = [100*k/12 for k in thresholds] #  unit: CRF over 5 minutes in 1/100 of mm (as meteonet data)
oversampling = 0.9 # oversampling of the last class

modelsize    = 8 # to do

epochs       = 20
batch_size   = 256
lr           = {0:8e-4, 4:1e-4}
wd           = {0:1e-5 ,4:5e-5} # 1e-8
clip_grad    = 0.1

snapshot_step  = 5

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
train_files, val_files, _ = filesets_bouget21('data/rainmaps')

# datasets
train_ds = MeteonetDataset( train_files, input_len, input_len + time_horizon, stride, cached=f'data/train.npz', tqdm=tqdm)
val_ds   = MeteonetDataset( val_files, input_len, input_len + time_horizon, stride, cached=f'data/val.npz', tqdm=tqdm)
val_ds.norm_factor = train_ds.norm_factor

device = torch.device(device)

# samplers for dataloaders
train_sampler = meteonet_random_oversampler( train_ds, thresholds[-1], oversampling)
val_sampler   = meteonet_sequential_sampler( val_ds)

# dataloaders
train_loader = DataLoader(train_ds, batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=True)

## Model & training procedure
import torch.nn as nn
from torch.optim import Adam
from models.unet import UNet
from loader.utilities import map_to_classes, calculate_CT, calculate_BS
from torch.utils.tensorboard import SummaryWriter

loss = nn.BCEWithLogitsLoss()
loss.to(device)

model = UNet(n_channels = input_len, n_classes = len(thresholds), bilinear = True)
model.to(device)

def get_xy( data):
    return data['inputs'], map_to_classes(data['target'], thresholds)

train_losses = []
val_losses = []
val_f1 = []
val_bias = []
val_ts = []

print(f"""
size of train files/items/batch
     {len(train_files)} {len(train_ds)} {len(train_loader)}
size of  files/items/batch
     {len(val_files)} {len(val_ds)} {len(val_loader)}
""")

print('eval persistence...')
# eval persistence
CT_pers = 0
for batch in tqdm(val_loader):
    CT_pers += calculate_CT( map_to_classes(batch['persistence'], thresholds),
                             map_to_classes(batch['target'], thresholds))
f1_pers, bias_pers, ts_pers = calculate_BS( CT_pers, ['F1','BIAS','TS'])

print('start training...')

rundir = f'runs/{datetime.now()}'
os.system(f'mkdir -p "{rundir}"')
writer = SummaryWriter(log_dir=rundir)

for epoch in range(epochs):
    if epoch in lr:
        optimizer = Adam(model.parameters(), lr=lr[epoch], weight_decay=wd[epoch])
        
    model.train()  
    train_loss = 0
    for batch in tqdm(train_loader):
        x,y = get_xy(batch)
        x,y = x.to(device), y.to(device)

        y_hat = model(x)

        l = loss(y_hat, y)
        train_loss += l.item()

        optimizer.zero_grad()
        l.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_grad)
        optimizer.step()
    train_loss /= len(train_loader)*batch_size
    train_losses.append(train_loss)
    print(f'epoch {epoch+1} {train_loss=}')

    model.eval()
    val_loss = 0
    CT_pred = 0
    for batch in tqdm(val_loader):
        x,y = get_xy(batch)
        x,y = x.to(device), y.to(device)
        with torch.no_grad():
            y_hat = model(x)
            
        l = loss(y_hat, y)
        val_loss += l.item()
        CT_pred += calculate_CT(torch.sigmoid(y_hat)>.5, y)
            
    f1_pred, bias, ts =  calculate_BS( CT_pred, ['F1','BIAS','TS'])
            
    val_loss /= len(val_loader)*batch_size
    val_losses.append(val_loss)
        
    val_f1.append(f1_pred)
    val_bias.append(bias)
    val_ts.append(ts)

    print(f'epoch {epoch+1} {val_loss=} {f1_pred=} {f1_pers=}')

    writer.add_scalar('train', train_loss, epoch)
    writer.add_scalar('val', val_loss, epoch)
    for c in range(len(thresholds)):
        writer.add_scalar(f'F1_C{c+1}', f1_pred[c], epoch)
        writer.add_scalar(f'TS_C{c+1}', ts[c], epoch)
        writer.add_scalar(f'BIAS_C{c+1}', bias[c], epoch)
    
    if epoch % snapshot_step == snapshot_step-1:
        torch.save(model.state_dict, f'{rundir}/model_epoch_{epoch}.pt')
    
os.system(f'rm -f lastrun; ln -sf "{rundir}" lastrun')

scores = {'train_losses': train_losses, 'val_losses': val_losses,
          'val_f1': val_f1, 'f1_pers': f1_pers ,
          'val_bias': val_bias, 'bias_pers': bias_pers,
          'val_ts': val_ts, 'ts_pers': ts_pers
          }

print( f'Optimisation is over. Scores and weights are saved in {rundir}')
torch.save(model.state_dict(), rundir+"/model_last_epoch.pt")
torch.save(scores, rundir+"/scores.pt")
