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
    # version torch tensor, require a 4-D tensor
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

def calculate_CT( pred, true):
    """ Calculate Contigenycy Table (ie. TP, TN, FP, FN)
    """
    n = pred.shape[0]*pred.shape[2]*pred.shape[3]
    diff = 2*pred - 1*true
    TP = torch.sum(diff==1,dim=(0,2,3)).unsqueeze(0)
    FP = torch.sum(diff==2,dim=(0,2,3)).unsqueeze(0)
    FN = torch.sum(diff==-1,dim=(0,2,3)).unsqueeze(0)
    TN = n - TP - FP - FN
    return torch.cat((TP, TN, FP, FN),dim=0)

def calculate_BS(CT, scores):
    """ Calculate various binary scores from the contingency table 
        scores: list of scores to calculates. Possible values are:
         'Precision', 'Recall', 'F1','TS','BIAS','HSS','FAR','Accuracy','ETS','ORSS'

        See Bouget et al, 2021 and Kumar et al, 2020
    """
    TP, TN, FP, FN = CT.to('cpu')

    scores_values = []

    precis = TP/(TP+FP)  # c'est le POD d'Aniss
    recall = TP/(TP+FN)
    n = TP+TN+FP+FN

    for sc in scores:
        if sc == 'Precision':
            scores_values.append(precis) # POD
        elif sc == 'Recall':
            scores_values.append(recall)
        elif sc == 'F1':
            scores_values.append(2*precis*recall/(precis+recall+1e-10))
        elif sc == 'TS':
            scores_values.append(TP/(TP+FP+FN))  # CSI, FMS 
        elif sc == 'BIAS':
            scores_values.append((TP+FP)/(TP+FN))  
        elif sc == 'HSS':  # alternative definition, see Hogan 2009
            n = TP+TN+FP+FN 
            rc = ((TP+FP)*(TP+FN) + (TN+FP)*(TN+FN))/n
            scores_values.append( (TP+TN-rc)/(n-rc))
        elif sc == 'FAR':
            scores_values.append( FP / (FP+TN))
        elif sc == 'Accuracy': 
            scores_values.append( (TP+TN) / n)
        elif sc == 'ETS':
            TP_r = (TP+FN)*(TP+FP) / n
            scores_values.append( (TP-TP_r)/(TP+FP+FN-TP_r))
        elif sc == 'ORSS':
            scores_values.append( (TP*TN - FP*FN)/(TP*TN + FP*FN))
        else:
            print(f'score {sc} unkown')

    return np.array([scv.numpy() for scv in scores_values])

