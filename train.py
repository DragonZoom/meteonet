#
#  An example of meteonet dataloader usage to train a U-Net [Bouget et al, 2021]
#
from glob import glob
from tqdm import tqdm
from loader.meteonet import MeteonetDataset
from loader.samplers import meteonet_random_oversampler, meteonet_sequential_sampler
from torch.utils.data import DataLoader
import os, torch
from datetime import datetime

## user parameters
input_len    = 12
time_horizon = 6
stride       = input_len
thresholds   = [0.1, 1, 2.5]  # series of thresholds (unit: mm/h)
thresholds   = [100*k/12 for k in thresholds] #  unit: CRF over 5 minutes in 1/100 of mm (as meteonet data)
oversampling = 0.8 # oversampling of the last class

modelsize    = 8 # to do

epochs       = 100
batch_size   = 32
lr           = 0.01
wd           = 1e-8
clip_grad    = 0.1

val_step     = 1

if os.path.isfile( 'data/rainmaps/y2018-M2-d5-h18-m55.npz'):
    datadir = 'data'
elif os.path.isfile( 'meteonet/rainmaps/y2018-M11-d9-h0-m0.npz'):
    datadir = 'meteonet'
else:
    print( 'No dataset found. Please download one with download-meteonet-*.sh scripts.')
    exit (1)

print(f"""
Data params:
   {datadir = }
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
   {val_step = }
""")

train_files = glob(f'{datadir}/rainmaps/y201[67]-*')
val_files   = glob(f'{datadir}/rainmaps/y2018-*')

# datasets
train_ds = MeteonetDataset( train_files, input_len, input_len + time_horizon, stride, cached='data/train.npz', tqdm=tqdm)
val_ds   = MeteonetDataset( val_files, input_len, input_len + time_horizon, stride, cached='data/val.npz', tqdm=tqdm)

device = torch.device('cuda')

# samplers for dataloaders
train_sampler = meteonet_random_oversampler( train_ds, thresholds[-1], oversampling)
val_sampler = meteonet_sequential_sampler( val_ds)

# dataloaders
train_loader = DataLoader(train_ds, batch_size, sampler=train_sampler, num_workers=8, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size, sampler=val_sampler, num_workers=8, pin_memory=True) 


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

optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)

def get_xy( data):
    return data['inputs'], map_to_classes(data['target'], thresholds)

def calculate_TPFPFN( pred, true):
    """ BCNM*BCNM
    """
    diff = 2*pred - true
    return torch.cat((torch.sum(diff==1,dim=(0,2,3)), # TP
                      torch.sum(diff==2,dim=(0,2,3)), # FP
                      torch.sum(diff==-1,dim=(0,2,3)  # FN
                    ))).reshape(3,-1)
train_losses = []
val_losses = []
val_f1_pred = []

def calculate_scores( TPFPFN):
    TP, FP, FN = TPFPFN
    precis = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*precis*recall/(precis+recall) # if precis>0 or recall>0 else np.nan
    return precis,recall,f1

# eval persistence
TPFPFN_pers = 0
print('eval persistence...')
for batch in tqdm(val_loader):
    _,y = get_xy(batch)
    TPFPFN_pers += calculate_TPFPFN(map_to_classes(batch['persistence'], thresholds),
                                    map_to_classes(batch['target'], thresholds))
_, _, f1_pers =  calculate_scores( TPFPFN_pers)
f1_pers = list(f1_pers.numpy())

print('start training...')
for epoch in range(epochs):
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
        nn.utils.clip_grad_value_(model.parameters(), clip_grad)
        optimizer.step()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    print(f'epoch {epoch+1} {train_loss=}')
    
    if epoch%val_step == val_step-1: # validation toutes les val_step epochs
        val_loss = 0
        TPFPFN_pred = 0
        for batch in tqdm(val_loader):
            x,y = get_xy(batch)
            x,y = x.to(device), y.to(device)
            with torch.no_grad():
                y_hat = model(x)
            
            l = loss(y_hat, y)
            val_loss += l.item()
            TPFPFN_pred += calculate_TPFPFN(torch.sigmoid(y_hat)>.5, y)
            
        _, _, f1_pred =  calculate_scores( TPFPFN_pred)
            
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        f1_pred = f1_pred.cpu().numpy()
        val_f1_pred.append(f1_pred)
        f1_pred = list(f1_pred)
        print(f'epoch {epoch+1} {val_loss=} {f1_pred=} {f1_pers=}')


rundir = f'runs/{datetime.now()}'
os.system(f'mkdir -p "{rundir}"')

scores = {'train_losses': train_losses, 'val_losses': val_losses,
          'val_f1_pred': val_f1_pred, 'f1_pers': f1_pers }

print( f'Optimisation over, scores and weights are saved in {rundir}')
torch.save(model.state_dict, rundir+"/model.pt")
torch.save(scores, rundir+"/scores.pt")


if False:
    import matplotlib.pyplot as plt
    plt.subplot(1,2,1)
    plt.plot(train_losses)
    plt.subplot(1,2,2)
    plt.plot(val_losses)
    plt.show()

#trainer = Trainer( model, loss, solver, get_xy, 'runs', device, 0.1)
#train_loss, valid_loss = trainer.fit( train_loader, val_loader, epochs)



