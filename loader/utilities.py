# Meteonet files utilities

from os.path import basename, isfile, join, dirname
import numpy as np
from tqdm import tqdm
from glob import glob
import torch

def split_date(filename):
    year,month,day,hour,minute = [k[1:] for k in basename(filename).split(".")[0].split("-")]
    return int(year),int(month),int(day),int(hour),int(minute)

def load_map(npz):
    # ->Array[int16] 
    return list(np.load(npz).values())[0]

def map_to_classes( rainmap, thresholds):
    # Array * List[float] -> Array[bool]
    # return np.array([rainmap >= th for th in thresholds])
    # version torch tensor
    return torch.cat([1.*(rainmap >= th).unsqueeze(1) for th in thresholds], dim=1)

def next_date(filename, ext='npz'):
    """ determine the next file according to its name """
    year,month,day,hour,minute = [int(k[1:]) for k in filename.split('.')[0].split('-')]
    Months=[0,31,28,31,30,31,30,31,31,30,31,30,31]
    if year == 2016: Months[2] = 29 # bissextile year
    if minute == 55:
        minute=0
        if hour == 23:
            hour=0
            if day==Months[month]:
                day=1
                if month==12:
                    month=1
                    year+=1
                else: month += 1
            else: day += 1
        else: hour += 1
    else: minute += 5
    return f'y{year}-M{month}-d{day}-h{hour}-m{minute}.{ext}'

def get_item_by_date(ds, date):
    """ utility: returns the item in which date belongs to """
    files = ds.params['files']
    items = ds.params['items']
        
    y,M,d,h,m = date
    f = f'{dirname(files[0])}/y{y}-M{M}-d{d}-h{h}-m{m}.npz'
    if not f in files:
        print('date not available')
        return None
    idx, _ = np.where(items[:,:-1]==files.index(f))
    return idx[0]
