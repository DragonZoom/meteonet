from loader.meteonet import MeteonetDataset, MeteonetTime
from loader.utilities import get_files, next_date
from tqdm import tqdm

from os.path import basename

from torch.utils.data import DataLoader

files = get_files( "data/rainmaps/*.npz")

test_files = MeteonetTime(files)
print( f"Time to check {len(test_files)} files ...")
test_files.timeit()

test_load = MeteonetTime(files, load=True)
print( f"Time to load {len(test_load)} files ...")
test_load.timeit()

# curdate = basename(files[0])
# for f in files[1:]:
#    nextdate = next_date(curdate)
#    tmpdate = curdate
#    curdate = basename(f)
#    if  nextdate != curdate:
#        print( f'{tmpdate} missing')

params = { 'train': files, 'max': 1.0 }

print("Time to iterate the dataset ...")
train = MeteonetDataset( params, 'train', 12, 18, 12)
for a in tqdm(train, unit=' files'): pass

print( f"Found {len(train.missing_dates)} Missing files: ", train.missing_dates)

print("Time to iterate the batched dataset ...")
train = DataLoader( train, 32)
for a in tqdm(train, unit= ' batches'): pass

