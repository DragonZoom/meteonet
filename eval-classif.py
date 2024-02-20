# Evaluation of the model on the test set

import torch, pandas as pd
from tqdm import tqdm
from meteonet.loader import MeteonetDataset
from meteonet.samplers import meteonet_sequential_sampler
from torch.utils.data import DataLoader
from meteonet.filesets import bouget21
from meteonet.utilities import map_to_classes, calculate_CT, calculate_BS
from platform import processor, system as sysname
import os, argparse

parser = argparse.ArgumentParser( prog='eval-Unet', description='evaluation of last run on the test set')
parser.add_argument('-rd', '--rundir', default='lastrun', type=str, help='a run directory')
parser.add_argument('-w', '--weights', default='model_last_epoch.pt', type=str, help='name of weights file')
args = parser.parse_args()

weights_path = os.path.join(args.rundir, args.weights)

hyperparams = torch.load(os.path.join(args.rundir,'run_info.pt'))['hyperparams']
print(hyperparams)
input_len = hyperparams['input_len']
time_horizon = hyperparams['time_horizon']
stride = hyperparams['stride']
batch_size = hyperparams['batch_size']
thresholds = hyperparams['thresholds']
wind_dir = hyperparams['wind_dir'] 

num_workers = 0 if processor() == 'arm' and sysname() == 'Darwin' else 8

train_files, _, test_files = bouget21('data/rainmaps')

train_ds = MeteonetDataset( train_files, input_len, input_len + time_horizon, stride, cached='data/train.npz', wind_dir=wind_dir, tqdm=tqdm)
test_ds  = MeteonetDataset( test_files, input_len, input_len + time_horizon, stride, cached='data/test.npz', wind_dir=wind_dir, tqdm=tqdm)
test_ds.norm_factors = train_ds.norm_factors

test_sampler = meteonet_sequential_sampler( test_ds)
test_loader  = DataLoader(test_ds, batch_size, sampler=test_sampler, num_workers=num_workers, pin_memory=True)

if torch.backends.cuda.is_built() and torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_built():
    device = 'mps'
else:
    device = 'cpu'

from models.unet import UNet

if wind_dir:
    model = UNet(n_channels = 3*input_len, n_classes = len(thresholds), bilinear = True)
else:
    model = UNet(n_channels = input_len, n_classes = len(thresholds), bilinear = True)
model.load_state_dict(torch.load( weights_path,  map_location=torch.device('cpu')))
model.to(device)


print("Evaluation on test set")
model.eval()
CT_pred = 0
CT_pers = 0
for batch in tqdm(test_loader):
    x = batch['inputs']
    y = map_to_classes( batch['target'], thresholds)
    p = map_to_classes( batch['persistence'], thresholds)

    x,y,p = x.to(device), y.to(device), p.to(device)
    with torch.no_grad():
        y_hat = model(x)
            
    CT_pred += calculate_CT(torch.sigmoid(y_hat)>.5, y)
    CT_pers += calculate_CT(p, y)            

import numpy as np
score_names = ['Pres/POD', 'Recall/Success Ratio', 'F1', 'TS/CSI', 'Bias', 'HSS', 'FAR', 'ETS', 'ORSS']

print('*** Scores for prediction ***')
print(pd.DataFrame( calculate_BS( CT_pred, ['Precision', 'Recall', 'F1', 'TS', 'BIAS', 'HSS', 'FAR', 'ETS', 'ORSS']),
              columns=['C1','C2','C3'], index=score_names))

print('\n\n*** Scores for persistence ***')
print(pd.DataFrame( calculate_BS( CT_pers, ['Precision', 'Recall', 'F1', 'TS', 'BIAS', 'HSS', 'FAR', 'ETS', 'ORSS']),
              columns=['C1','C2','C3'], index=score_names))

