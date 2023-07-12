from loader.meteonet import MeteonetDataset
from loader.sampler import items_to_oversample, meteonet_sequential_sampler
from loader.utilities import split_date
from tqdm import tqdm
from glob import glob
from os.path import basename

from torch.utils.data import DataLoader

files = sorted(glob( "data/rainmaps/*.npz"), key=lambda f:split_date(f))[:10000]

# from loader.utilities import get_files, next_date, split_date
# files = sorted(files, key=lambda f:split_date(f))
# curdate = basename(files[0])
# for f in files[1:]:
#    nextdate = next_date(curdate)
#    tmpdate = curdate
#    curdate = basename(f)
#    if  nextdate != curdate:
#        print( f'{tmpdate} missing')

print(f"Time to read {len(files)} files...")
train = MeteonetDataset( files, 12, 18, 12, tqdm=tqdm)

print("Time to iterate the dataset ...")
#print( f"Found {len(train.missing_dates)} Missing files: ", train.get_missing_dates(iterate=True))
sampler = meteonet_sequential_sampler( train, tqdm)

print("Time to iterate the batched dataset ...")
train = DataLoader( train, 32, sampler=sampler, pin_memory = True)
for a in tqdm(train, unit= ' batches'): pass

