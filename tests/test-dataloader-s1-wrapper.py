from meteonet.loader import MeteonetDatasetChunked, DatsetWrapperFsrSecondStage
from meteonet.samplers import meteonet_sequential_sampler, meteonet_random_oversampler
from tqdm import tqdm
from os.path import isdir
from torch.utils.data import DataLoader
from platform import processor # for M1/M2 support

COMPRESSED = False
data_folder = f'data-chunked{'-z' if COMPRESSED else ''}'
if isdir(data_folder):
    print('reduced dataset')
else:
    print('No dataset found. Please run script-data-chunked.py first')
    exit(1)

data_type = 'val'
train = MeteonetDatasetChunked( data_folder, data_type, 12, 18, 12, target_is_one_map=True, compressed=COMPRESSED, use_wind=True) 
print(f"found {len(train.params['missing_dates'])} missing dates")
train_wrapp = DatsetWrapperFsrSecondStage(f'cache/first_stage_predictions/{data_type}', train, 'cpu')

print("Time to iterate the dataset ...")
for d in tqdm(train_wrapp, unit=' items'): pass
print(f"Total loads: {train.loader.n_loaded}, Total missing: {train.loader.n_missed} ({train.loader.n_missed/train.loader.n_loaded*100:.2f}%)")

print("Time to iterate the batched dataset ...")
sampler = meteonet_random_oversampler( train, 20.83)
train = DataLoader( train_wrapp, 128, sampler=sampler, num_workers=8 if processor() != 'arm' else 0)
for a in tqdm(train_wrapp, unit= ' batches'): pass

