from loader.meteonet import MeteonetDataset
from loader.samplers import meteonet_sequential_sampler
from tqdm import tqdm
from glob import glob
from os.path import isfile
from torch.utils.data import DataLoader
from platform import processor # for M1/M2 support

if isfile('data/.reduced_dataset'):
    print('reduced dataset')
elif isfile('data/.full_dataset'):
    print('full dataset')
else:
    print('No dataset found. Please download one with download-meteonet.sh script.')
    exit(1)

files = glob( "data/rainmaps/y201[67]-*.npz")

# from loader.utilities import get_files, next_date, split_date
# files = sorted(files, key=lambda f:split_date(f))
# curdate = basename(files[0])
# for f in files[1:]:
#    nextdate = next_date(curdate)
#    tmpdate = curdate
#    curdate = basename(f)
#    if  nextdate != curdate:
#        print( f'{tmpdate} missing')

print(f"Time to read and indexing {len(files)} files with windmaps...")
train = MeteonetDataset( files, 12, 18, 12, tqdm=tqdm, wind_dir='data/windmaps') 
print(f"found {len(train.params['missing_dates'])} missing dates")

print("Time to iterate the dataset ...")
for d in tqdm(train, unit=' items'): pass

print("Time to iterate the batched dataset ...")
sampler = meteonet_sequential_sampler( train)
train = DataLoader( train, 32, sampler=sampler, num_workers=8 if processor() != 'arm' else 0)
for a in tqdm(train, unit= ' batches'): pass
