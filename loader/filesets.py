from glob import glob
from loader.utilities import split_date
from datetime import datetime
from os.path import join

def bouget21( rainmaps_dir):
    """ 
    Split in train/validation/test sets according to Section 4.1 from Bouget et al, 2021
    """
    test_files = []
    val_files = []

    for f in sorted( glob(join(rainmaps_dir,"y2018-*.npz")), key=lambda f:split_date(f)):
        year, month, day, hour, _ = split_date(f)
        yday = datetime(year, month, day).timetuple().tm_yday - 1
        if (yday // 7) % 2 == 0: # odd week
            val_files.append(f)
        else:
            if not (yday % 7 == 0 and hour == 0): # ignore the first hour of the first day of even weeks
                test_files.append(f)
    return glob(join(rainmaps_dir,"y201[67]-*.npz")), val_files, test_files
