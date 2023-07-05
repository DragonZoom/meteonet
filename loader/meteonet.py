# Meteonet dataloader
import torch
import numpy as np
from torch.utils.data import Dataset
from loader.utilities import next_date, load_map, map_to_classes
from os.path import dirname, basename, join, isfile
import time

class MeteonetDataset(Dataset):
    """ A class to load Meteonet data
        V1 limitations:
         - no windmaps
         - no log transformation, to discuss

       stratégie: on vérifie que les dates sont cohérentes. Quand une date manque, on utilise une carte de valeurs nulles.
       les dates manquantes sont reportées dans la variables missing_date de la classe.
    
    """
    def __init__(self, params, tset="train", input_len = 12, target_pos = 18, stride = 12, thresholds = False, log=False):
        """
        params: a dictionnary containing sorted lists of train and set files
                and normalisation parameters, created by initparams() function
        tset: 'train' or 'val'
        input_len: number of maps to read as input of the model
                   recommended: 12 (stands for 1 hour)
        target_pos: position of target starting from the first map read as input
                    recommended: 18 = 12+6 (prevision horizon time at 6 = 30 minutes)
        stride: offset between each input sequence
                recommended: input_len (for no overlapping)
        thresholds: if non False, a list of thresholds for a classification task
        """
        self.input_len = input_len
        self.target_pos = target_pos
        self.stride = stride
        self.params = params
        self.thresh = thresholds
        self.files = params[tset]
        self.maps = []
        self.missing_dates = set()
        self.log = log

        self.dirname = dirname(self.files[0])
        self.extname = basename(self.files[0]).split('.')[1] # pas de . dans les noms de fichiers.
        self.nm = load_map(self.files[0]).shape

    def __len__(self):
        return (len(self.files) - self.target_pos)// self.stride

    def read(self, file):
        # log ??
        rainmap = torch.unsqueeze(torch.tensor(load_map(file), dtype=torch.float32), dim=0)
        # normalisation (map between 0 and 1)
        rainmap = (rainmap + 1)/self.params['max']
        return rainmap
    
    def __getitem__(self, i):
        j = i*self.stride

        curr_date = basename(self.files[j])
        maps = self.read(self.files[j])

        n,m = self.nm
        
        # check if the next input_len-1 files have correct dates
        num_obs = self.input_len - 1
        k = 1
        while num_obs:
            next_available_date = basename(self.files[j+k])
            curr_date = next_date(curr_date, self.extname)
            num_obs -= 1

            if curr_date != next_available_date:
                # this date is not available, we use a 0-map
                maps = torch.cat((maps,torch.zeros(1,n,m,dtype=torch.float32)), dim=0)
                if self.log: print( f"warning: {i}/{len(self)}  {curr_date} is missing")
                self.missing_dates.add( curr_date)
            else:
                maps = torch.cat((maps,self.read(self.files[j+k])), dim=0)
                k += 1
            
        # get the target date
        k = self.input_len
        while k < self.target_pos:
            curr_date = next_date(curr_date, self.extname)
            k += 1
            
        target_file = join(self.dirname, curr_date)

        if not isfile(target_file):
            if self.log: print( f"warning (as target): {i}/{len(self)} {curr_date} is missing")
            self.missing_dates.add( curr_date) 
            target = np.zeros((n,m),dtype=torch.float32)            
        else:            
            target = load_map(target_file)

        return { 'inputs': maps,
                 'target': torch.from_numpy(map_to_classes(target, self.thresh)) if self.thresh else torch.from_numpy(target),
                 'name': target_file }
    
class MeteonetTime(Dataset):
    """ A class to check the time cost of loading 200000 files """
    def __init__(self, files, load = False):
        self.files = files
        self.load = load
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        file = self.files[i]
        if self.load:
            return torch.unsqueeze(torch.tensor(load_map(file), dtype=torch.float32), dim=0)
        return isfile(file)
    def timeit(self):
        tic = time.perf_counter()
        for a in self: pass
        tac =  time.perf_counter()
        print(f"{tac - tic:0.4f} seconds")
