#

import torch, pandas as pd
from tqdm import tqdm
from loader.meteonet import MeteonetDataset
from loader.samplers import meteonet_sequential_sampler
from torch.utils.data import DataLoader
from loader.filesets import filesets_bouget21
from loader.utilities import map_to_classes, calculate_CT, calculate_BS

# parametres a sauver lors du run
input_len = 12
time_horizon = 6
stride = 12
batch_size = 256
weights_path = 'weights/model_dom_30m_rain.pt'
thresholds   = [0.1, 1, 2.5]  # series of thresholds (unit: mm/h)
thresholds   = [100*k/12 for k in thresholds] #  unit: CRF over 5 minutes in 1/100 of mm (as meteonet data)
num_workers = 0

train_files, _, test_files = filesets_bouget21('data/rainmaps')

train_ds = MeteonetDataset( train_files, input_len, input_len + time_horizon, stride, cached=f'data/train.npz', tqdm=tqdm)
test_ds  = MeteonetDataset( test_files, input_len, input_len + time_horizon, stride, cached=f'data/test.npz', tqdm=tqdm)
test_ds.norm_factor = train_ds.norm_factor

test_sampler   = meteonet_sequential_sampler( test_ds)
test_loader   = DataLoader(test_ds, batch_size, sampler=test_sampler, num_workers=num_workers, pin_memory=True)

if torch.backends.cuda.is_built() and torch.cuda.is_available():
    device   = 'cuda'
elif torch.backends.mps.is_built():
    device    = 'mps'
else:
    device   = 'cpu'

from models.unet import UNet

model = UNet(n_channels = input_len, n_classes = len(thresholds), bilinear = True)
model.load_state_dict(torch.load( weights_path,  map_location=torch.device('cpu')))
model.to(device)


# test evaluation on test set  TO DO
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

