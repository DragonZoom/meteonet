# This script prepares the train and validation datasets, computes normalisation
# and class flags for each image
# configuration is written in file data_dir/configs.npz

from glob import glob
from tqdm import tqdm
from os.path import join
from loader.utilities import by_year, split_date, load_map, load_dict, map_to_classes
import numpy as np

### parameters to set up by the user 
data_dir     = "data"
rainmaps_dir = join(data_dir,"rainmaps") 
windmaps_dir = "data/windmaps"  # TODO

train_years = [2016, 2017]  # two years for the train set
val_years   = [2018]        # one year for the validation set

thresholds_in_mmh     = [0, 0.1, 1, 2.5]  # unit: mm/h 
### end of parameters


## TODO: test_set : gets somes dates from 2018
thresholds_in_cent_mm = [100*k/12 for k in thresholds_in_mmh] # CRF over 5 minutes as 1/100 of mm (as Meteonet data).
rainmap_files = glob(join(rainmaps_dir,'*.npz'))

train_files = []
for y in train_years:
    train_files += sorted(filter(lambda f: by_year(f,y), rainmap_files), key=lambda f: split_date(f))

val_files = []
for y in val_years:
    val_files += sorted(filter(lambda f: by_year(f,y), rainmap_files), key=lambda f: split_date(f))


print('compute normalisation parameters...')
imin, imax = 1000, 0
tq = tqdm(train_files, unit=" file")
data = np.empty( (len(tq),3))
l = len(thresholds_in_cent_mm)
for i,file in enumerate(tq):
    map = load_map(file)
    amin = map.min()
    if amin != -1:
        amax = map.max()
        if amin < imin: imin = amin
        if amax > imax: imax = amax
        classes = map_to_classes( map, thresholds_in_cent_mm).reshape((l,-1))
        data[i] = 1*(np.sum(classes[1:],axis=1)>0)
    else:
        data[i] = -1*np.ones((1,l-1))

stats = {'min': imin, 'max':imax, 'train': train_files, 'val': val_files, 'thresholds': thresholds_in_cent_mm, 'data': data}
np.savez_compressed( join(data_dir,"config.npz"),  stats)

# Vérification
d = load_dict(join(data_dir,"config.npz"))
assert d['min'] == imin
assert d['max'] == amax
assert d['train'][0] == train_files[0]
assert d['val'][0] == val_files[0]
assert d['thresholds'] == thresholds_in_cent_mm
assert d['data'][0] == data[0]

# TODO: print general statistics on database
