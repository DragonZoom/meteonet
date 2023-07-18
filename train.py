#
#  An example of meteonet dataloader usage to train a U-Net 
#  [1] Bouget et al, 2021, https://www.mdpi.com/2072-4292/13/2/246#B24-remotesensing-13-00246

from glob import glob
from tqdm import tqdm
from loader.meteonet import MeteonetDataset
from loader.samplers import meteonet_random_oversampler, meteonet_sequential_sampler
from torch.utils.data import DataLoader
import os, torch
from datetime import datetime
from loader.utilities import split_date

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

val_step     = 1

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
   {val_step = }
""")

train_files = glob(f'data/rainmaps/y201[67]-*')
val_test_files = glob(f'data/rainmaps/y2018-*')

# split in validation/test sets according to Section 4.1 from [1]
val_files = []
test_files = []
for f in sorted(val_test_files, key=lambda f:split_date(f)):
    year, month, day, hour, _ = split_date(f)
    yday = datetime(year, month, day).timetuple().tm_yday - 1
    if (yday // 7) % 2 == 0: # odd week
        val_files.append(f)
    else:
        if not (yday % 7 == 0 and hour == 0): # ignore the first hour of the first day of even weeks
            test_files.append(f)

# datasets
train_ds = MeteonetDataset( train_files, input_len, input_len + time_horizon, stride, cached=f'data/train.npz', tqdm=tqdm)
val_ds   = MeteonetDataset( val_files, input_len, input_len + time_horizon, stride, cached=f'data/val.npz', tqdm=tqdm)

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
from models.unet import UNet
from loader.utilities import map_to_classes

loss = nn.BCEWithLogitsLoss()
loss.to(device)

model = UNet(n_channels = input_len, n_classes = len(thresholds), bilinear = True)
model.to(device)

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


print(f"""
size of train files/items/batch
     {len(train_files)} {len(train_ds)} {len(train_loader)}
size of  files/items/batch
     {len(val_files)} {len(val_ds)} {len(val_loader)}
""")

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
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    print(f'epoch {epoch+1} {train_loss=}')

    model.eval()
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
os.system(f'ln -sf "{rundir}" lastrun')

scores = {'train_losses': train_losses, 'val_losses': val_losses,
          'val_f1_pred': val_f1_pred, 'f1_pers': f1_pers }

print( f'Optimisation over, scores and weights are saved in {rundir}')
torch.save(model.state_dict(), rundir+"/model.pt")
torch.save(scores, rundir+"/scores.pt")


if True:
    import matplotlib.pyplot as plt
    plt.subplot(1,2,1)
    plt.plot(train_losses)
    plt.subplot(1,2,2)
    plt.plot(val_losses)
    plt.show()

#trainer = Trainer( model, loss, solver, get_xy, 'runs', device, 0.1)
#train_loss, valid_loss = trainer.fit( train_loader, val_loader, epochs)



