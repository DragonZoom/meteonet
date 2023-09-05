#
#  An example of meteonet dataloader usage to train in a regresionn manner 
#  

import argparse, torch, os
from platform import processor # for M1/M2 support

parser = argparse.ArgumentParser( prog='train-reg', description='Traning a UNet for Meteonet nowforecast')

parser.add_argument( '-dd', '--data-dir', type=str, help='Directory containing data', dest='data_dir', default='data')
parser.add_argument( '-wd', '--wind-dir', type=str, help='Directory containing the wind data', dest='wind_dir', default=None)
parser.add_argument( '-t', '--thresholds', type=float, nargs='+', help='Rainmap thresholds in mm/h (used by binary metrics)', dest='thresholds', default=[0.1,1,2.5])
parser.add_argument( '-m', '--model', type=str, help='Model to train', dest='model', default='Unet.py') 
parser.add_argument( '-Rd', '--run-dir', type=str, help='Directory to save logs and checkpoints', dest='run_dir', default="runs")
parser.add_argument( '-e', '--epochs', type=int,help='Number of epochs', dest='epochs', default=20)
parser.add_argument( '-b', '--batch-size', type=int, help='Batch size', dest='batch_size', default=128)
parser.add_argument( '-lr', '--learning-rate', type=str, nargs='+', help='LR_WD format is: epoch:Learning rate, weight decay', dest='lr_wd', default=['0:8e-4,1e-5', '4:1e-4,5e-5'])
parser.add_argument( '-nw', '--num-workers', type=int,  help='Numbers of workers for Cuda', dest='num_workers', default = 8 if processor() != 'arm' else 0 ) # no multithreadings on M1/M2 :(
parser.add_argument( '-o', '--oversampling', type=float, help='Oversampling percentage of last class', dest='oversampling', default=0.9)
parser.add_argument( '-ss', '--snapshot-step', type=int, help='', dest='snapshot_step', default=5)

#parser.add_argument('-f', '--load', dest='load', type=str, default=False, help='Load model from a .pth file')
#parser.add_argument( '-gs', '--global-step-start', metavar='gstp', type=int, default=0,
#                     help='Number of the last global step of loaded model', dest='glb_step_start')
#parser.add_argument( '-es', '--epoch-start', metavar='es', type=int, default=0,
#                     help='Number of last epoch of the loaded model', dest='epoch_start')

args = parser.parse_args()

## user parameters
input_len    = 12
time_horizon = 6
stride       = input_len
clip_grad    = 0.1

thresholds   = [100*k/12 for k in args.thresholds] #  unit: CRF over 5 minutes in 1/100 of mm (as meteonet data)
model_size   = 8 # to do
lr_wd=dict()
for a in args.lr_wd:
    k,u=a.split(':')
    a,b=u.split(',')
    lr_wd[int(k)]=float(a),float(b)

if torch.backends.cuda.is_built() and torch.cuda.is_available():
    device   = 'cuda'
elif torch.backends.mps.is_built():
    device    = 'mps'
else:
    device   = 'cpu'

print(f"""
Data params:
   {input_len = } (history of {12*5} minutes)
   {time_horizon = } (nowcasting at {time_horizon*5} minutes)
   {stride = }
   model = Unet regression
   model_size = ?
   {args.data_dir = }
   {args.wind_dir = }
   {len(thresholds)} classes ({thresholds=})
   
Train params:
   {args.epochs = } 
   {args.batch_size = }
   {lr_wd = }
   {clip_grad = }

Others params:
   {device = }
   {args.snapshot_step = }
   {args.num_workers = }
   {args.run_dir = }
""")

device = torch.device(device)

from loader.filesets import bouget21
from loader.meteonet import MeteonetDataset
from loader.samplers import meteonet_random_oversampler, meteonet_sequential_sampler
from torch.utils.data import DataLoader
from tqdm import tqdm
from os.path import join

# split in validation/test sets according to Section 4.1 from [1]
train_files, val_files, _ = bouget21(join(args.data_dir, 'rainmaps'))

# datasets
indexes = [join(args.data_dir,'train.npz'), join(args.data_dir,'val.npz')]
train_ds = MeteonetDataset( train_files, input_len, input_len + time_horizon, stride, wind_dir=args.wind_dir, cached=indexes[0], tqdm=tqdm)
val_ds   = MeteonetDataset( val_files, input_len, input_len + time_horizon, stride, wind_dir=args.wind_dir, cached=indexes[1], tqdm=tqdm)
val_ds.norm_factors = train_ds.norm_factors

# samplers for dataloaders
train_sampler = meteonet_random_oversampler( train_ds, thresholds[-1], args.oversampling)
val_sampler   = meteonet_sequential_sampler( val_ds)

# dataloaders
train_loader = DataLoader(train_ds, args.batch_size, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
val_loader   = DataLoader(val_ds, args.batch_size, sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)


print(f"""
size of train files/items/batch
     {len(train_files)} {len(train_ds)} {len(train_loader)}
size of  files/items/batch
     {len(val_files)} {len(val_ds)} {len(val_loader)}
""")


## Model & training procedure
from models.unet import UNet
from trainers.regression import train_meteonet_regression
from datetime import datetime

if args.wind_dir:
    model = UNet(n_channels = input_len*3, n_classes = 1, bilinear = True)
else:
    model = UNet(n_channels = input_len, n_classes = 1, bilinear = True)
    
#try:

rundir = join(args.run_dir,f'{datetime.now()}')
print(f'run files will be recorded in directory {rundir}')
os.system(f'mkdir -p "{rundir}"')
scores = train_meteonet_regression( train_loader, val_loader, model, thresholds, args.epochs, lr_wd, args.snapshot_step, rundir=rundir, device = device)

hyperparams = { 'input_len': input_len,
                'time_horizon': time_horizon,
                'stride': stride, 
                'thresholds': thresholds,
                'batch_size': args.batch_size,
                # 'clip_grad': clip_grad,
                'epochs': args.epochs,
                'lr_wd': lr_wd,
                'oversampling': args.oversampling,
                # 'model_size': model_size,
                'data_dir': args.data_dir,
                'dataset_indexes': indexes,
                'wind_dir': args.wind_dir
               }

os.system(f'rm -f lastrun; ln -sf "{rundir}" lastrun')

#except KeyboardInterrupt:     # suspend
#        try:
#            sys.exit(0)
#        except SystemExit:
#            os._exit(0)


import torch
torch.save( { 'hyperparams': hyperparams, 'scores': scores}, join(rundir,'run_info.pt'))





    
    
