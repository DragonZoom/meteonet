# plot meteonet rainmaps (todo: scores and inference)

from loader.meteonet import MeteonetDataset
from loader.plots import plot_meteonet_rainmaps
from loader.filesets import bouget21

import numpy as np
from os.path import isfile
from tqdm import tqdm

if isfile('data/.reduced_dataset'):
    print('reduced dataset')
elif isfile('data/.full_dataset'):
    print('full dataset')
else:
    print('No dataset found. Please download one with download-meteonet-*.sh scripts.')
    exit(1)

_, val_files, _ = bouget21("data/rainmaps")

from data.constants import *
coord = np.load(f'data/radar_coords_NW.npz',allow_pickle=True)

lon = coord['lons'][lat_extract_start:lat_extract_end, lon_extract_start:lon_extract_end]
lat = coord['lats'][lat_extract_start:lat_extract_end, lon_extract_start:lon_extract_end]

val_ds = MeteonetDataset( val_files, 12, 18, 12, wind_dir='data/windmaps', cached='data/val.npz', tqdm=tqdm)
val_date = 2018,3,12,3,5

plot_meteonet_rainmaps( val_ds, val_date, lon, lat, zone,'Rainmaps with Meteonet style')


