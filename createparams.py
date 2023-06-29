# This script prepares the train and validation datasets, computes normalisation
# and class flags for each image
# configuration is written in the file "config.npz"

from glob import glob
from os.path import join
from loader.utilities import by_year, split_date, load_map, load_params, save_params, create_params

### parameters to set up by the user 
data_dir     = "data"
rainmaps_dir = join(data_dir,"rainmaps") 
windmaps_dir = "data/windmaps"  # TODO

train_years  = [2016, 2017]  # two years for the train set
val_years    = [2018]        # one year for the validation set

thresholds_in_mmh = [0, 0.1, 1, 2.5]  # unit: mm/h 
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
params = create_params( train_files, val_files, thresholds_in_cent_mm)
# print(params)
print('writing params.npz ...')
save_params(params, "params.npz")

# TODO: print general statistics on database
