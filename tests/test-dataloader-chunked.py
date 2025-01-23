from meteonet.loader import MeteonetDatasetChunked
from meteonet.samplers import meteonet_sequential_sampler, meteonet_random_oversampler
from tqdm import tqdm
from glob import glob
from os.path import isfile, isdir
from torch.utils.data import DataLoader
from platform import processor # for M1/M2 support


data_folder = 'data-chunked'
if isdir(data_folder):
    print('reduced dataset')
else:
    print('No dataset found. Please run script-data-chunked.py first')
    exit(1)

train = MeteonetDatasetChunked( data_folder, 'train', 12, 18, 12, target_is_one_map=False) 
print(f"found {len(train.params['missing_dates'])} missing dates")

print("Time to iterate the dataset ...")
for d in tqdm(train, unit=' items'): pass
print(f"Total loads: {train.loader.n_loaded}, Total missing: {train.loader.n_missed} ({train.loader.n_missed/train.loader.n_loaded*100:.2f}%)")

print("Time to iterate the batched dataset ...")
sampler = meteonet_random_oversampler( train, 20.83)
train = DataLoader( train, 128, sampler=sampler, num_workers=8 if processor() != 'arm' else 0)
for a in tqdm(train, unit= ' batches'): pass

