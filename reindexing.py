# This script forces reindexing databases

from glob import glob
from loader.meteonet import MeteonetDataset
from tqdm import tqdm
import os
from loader.filesets import bouget21

if os.path.isfile('data/.reduced_dataset'):
    print('Indexing reduced dataset, please wait...')
elif os.path.isfile('data/.full_dataset'):
    print('Indexing full dataset, please wait...')
else:
    print('No dataset found. Please download one with download-meteonet-*.sh scripts.')
    exit(1)

os.system('rm -f data/{train,val,test}.npz')

train, val, test = bouget21( 'data/rainmaps')
ds = MeteonetDataset( train, 12, 18, 12, cached='data/train.npz', wind_dir='data/windmaps', tqdm=tqdm)
print(ds.norm_factors)
ds = MeteonetDataset( val, 12, 18, 12, cached='data/val.npz', wind_dir='data/windmaps', tqdm=tqdm)
print(ds.norm_factors)
ds = MeteonetDataset( test, 12, 18, 12, cached='data/test.npz', wind_dir='data/windmaps', tqdm=tqdm)
print(ds.norm_factors)

    
