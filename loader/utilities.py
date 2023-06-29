# Meteonet files utilities

from os.path import basename, isfile, join
import numpy as np

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
    return list(np.load(npz).values())[0]

def load_dict(npz):
    obj = np.load(npz,allow_pickle=True)
    return obj['arr_0'].reshape(-1)[0]

def map_to_classes( map, thresholds):
    # image -> 3 images seuillées (le code de Vincent est vraiment étrange)
    threshs = thresholds[1:] # class 0 is not a class
    nb_class = len(threshs) 
    n,m = map.shape
    result = np.expand_dims( map, axis=0)
    for th in threshs:
        result =  np.concatenate((result, np.expand_dims(map >= th, axis=0)), axis=0)
    return result

