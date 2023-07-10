# Meteonet files utilities

from os.path import basename, isfile, join
import numpy as np
from tqdm import tqdm
from glob import glob

def by_year(filename, year):
    return filename.find(f'y{year}-') != -1
def by_month(filename, month):
    return filename.find(f'M{month}-') != -1
def by_day(filename, day):
    return filename.find(f'd{day}-') != -1
def by_hour(filename, hour):
    return filename.find(f'h{hour}-') != -1
def by_minute(filename, minute):
    return filename.find(f'm{minute}.') != -1
def split_date(filename):
    year,month,day,hour,minute = [k[1:] for k in basename(filename).split(".")[0].split("-")]
    return int(year),int(month),int(day),int(hour),int(minute)
def have_wind(rainmap):
    return isfile(rainmap.replace('rainmaps',join('windmaps','U')))
def get_wind(rainmap):
    return rainmap.replace('rainmaps',join('windmaps','U')), rainmap.replace('rainmaps',join('windmaps','V'))
def load_map(npz):
    # ->Array[int16] 
    return list(np.load(npz).values())[0]
def load_params(npz):
    """ load param file created by function .... """
    obj = np.load(npz,allow_pickle=True)
    return obj['arr_0'].reshape(-1)[0]
def get_files(dir):
    return sorted( glob(dir), key=lambda f:split_date(f))
def save_params(params, file):
    np.savez_compressed( file, params)

def map_to_classes( rainmap, thresholds):
    # Array * List[float] -> Array[bool]
    threshs = thresholds[1:] # class 0 is not a class
    return np.array([rainmap >= th for th in threshs])

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
#    return 'y'+str(year)+'-M'+str(month)+'-d'+str(day)+'-h'+str(hour)+'-m'+str(minute)+'.'+ext

def create_params(rainmap_train, rainmap_val, thresholds):
    """ determine normalisation paramerts from a train set
        and various parameters useful for the Meteonet dataloader
        * difference with Vincent: maps having missing values (-1) are kept because they contain valuable signal
        * V1: only rainmaps
    """
    tq = tqdm( rainmap_train, unit=" files")
    l = len(thresholds)
    data = np.empty( (len(tq),l-1))
    imin, imax = 1000, 0
    for i,file in enumerate(tq):
        map = load_map(file)
        amax = map.max()
        if amax > imax: imax = amax
        classes = map_to_classes( map, thresholds).reshape((l-1,-1))
        data[i] = 1*(np.sum(classes,axis=1)>0)
    return { 'max':imax, 'train': rainmap_train, 'val': rainmap_val,
             'thresholds': thresholds, 'data': data}
