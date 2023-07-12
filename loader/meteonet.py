# Meteonet dataloader
import torch
import numpy as np
from torch.utils.data import Dataset
from loader.utilities import next_date, load_map, map_to_classes, split_date
from os.path import dirname, basename, join, isfile
import time

epsilon = 1e-3 # As Antonia

class MeteonetDataset(Dataset):
    """ A class to load Meteonet data
        V1 limitations:
         - no windmaps
         - no multiple targets
         - if some dates are missing, iterations could be inconsistent.
         - no character '.' in filename excepted extension
       stratégie: on vérifie que les dates sont cohérentes. Quand une date manque, getitem() retourne None
       les dates manquantes sont reportées dans la variable missing_date de la classe.
    """
    def __init__(self, files, input_len = 12, target_pos = 18, stride = 12, cached=False, logging=False, tqdm=False):
        """
        files: list of files (will be sorted)
        tset: 'train' or 'val'
        input_len: number of maps to read as input of the model
                   recommended: 12 (stands for 1 hour)
        target_pos: position of target starting from the first map read as input
                    recommended: 18 = 12+6 (prevision horizon time at 6 = 30 minutes)
        stride: offset between each input sequence
                recommended: input_len (for no overlapping)
        thresholds: if non False, a list of thresholds for a classification task
        """
        files = sorted(files, key=lambda f:split_date(f))
        recalculate = True
        if cached and isfile(cached):
            obj = np.load(cached,allow_pickle=True)
            params = obj['arr_0'].reshape(-1)[0]
            if params['input_len'] == input_len and  params['stride'] == stride and params['files'] == files:
                recalculate = False
            
        if recalculate:
            if logging: print('parameters changed, or cached file not found')
            params = {'files': files,
                      'input_len': input_len, 'target_pos': target_pos, 'stride': stride,
                      'max': 0}
            for f in tqdm(files, unit=' files') if tqdm else files:
                mapr = load_map(f)
                maxi = mapr.max()
                if maxi > params['max']:
                    params['max'] = maxi
            if cached:
                np.savez_compressed( cached, params)
        
        self.logging = logging
        self.tqdm = tqdm
        self.norm_factor = np.log(1+params['max'])
        self.params = params
        self.missing_dates = set()

        self.dirname = dirname(files[0])
        self.extname = basename(files[0]).split('.')[1] # limitation: pas de . dans les noms de fichiers

    def get_missing_dates( self, iterate=False):
        if iterate:
            for d in self.tqdm(self, unit=' files') if self.tqdm else self: pass
        return sorted( self.missing_dates, key=lambda f:split_date(f))

    def __len__(self):
        ## question pour Anastase: on pourrait aussi avoir une fenêtre glissante sur les inputs_len ??
        ## le code de Vincent ne le prévoit pas: les inputs ne se superposent pas.
        ### Cette formule est exacte si il ne manque pas de dates, sinon, c'est approximatif et
        ### c'est compliqué à calculer (il faudrait le faire dans le init).
        ### Ce n'est pas trop grave je pense.
        ### Le plus simple a faire est de compléter les dates manquantes avec des fichiers à valeurs nulles,
        ### le sampler a pour charge de filtrer ensuite ces données. Le plus simple sera d'avoir un script qui
        ### le fasse au moment de la récupération des données.
        return (len(self.params['files']) - self.params['target_pos'] + self.params['input_len']) // self.params['stride']

    def read(self, file):
        if self.logging: print(f'load input {file}')
        ## Anastase: tensor() ou Tensor() ?
        rainmap = torch.Tensor(load_map(file)).unsqueeze(0)
        # normalisation (map between 0 and 1)
        rainmap = torch.log(rainmap + 1 + epsilon)/self.norm_factor
        return rainmap
    
    def __getitem__(self, i):
        j = i*self.params['stride']

        curr_date = basename(self.params['files'][j])
        maps = self.read(self.params['files'][j])
        to_ignore = False
        
        # check if the next input_len-1 files have correct dates
        num_obs = self.params['input_len'] - 1
        k = 1 
        while num_obs:
            next_available_date = basename(self.params['files'][j+k])
            curr_date = next_date(curr_date, self.extname)
            num_obs -= 1

            if curr_date != next_available_date:
                # curr_date is not available
                if self.logging: print( f"warning: {i}/{len(self)}  {curr_date} is missing")
                self.missing_dates.add( curr_date)
                to_ignore = True
            else:
                maps = torch.cat((maps,self.read(self.params['files'][j+k])), dim=0)
                k += 1
            
        # get the target date
        k = self.params['input_len']
        while k < self.params['target_pos']:
            curr_date = next_date(curr_date, self.extname)
            k += 1
            
        target_file = join(self.dirname, curr_date)

        if not isfile(target_file):
            if self.logging: print( f"warning (as target): {i}/{len(self)} {curr_date} is missing")
            self.missing_dates.add( curr_date) 
            to_ignore = True
        else:
            if self.logging: print(f'load target {target_file}')
            # Anastase: Tensor() ou tensor() ?
            target = torch.Tensor(load_map(target_file))

        if to_ignore: return None # Ces valeurs seront filtrées par le sampler
        return { 'inputs': maps,
                 'target': target,
                 'name': target_file }
    
class MeteonetTime(Dataset):
    """ A class to check the time cost of loading 200000 files, obsolet """
    def __init__(self, files, load = False):
        self.files = files
        self.load = load
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        file = self.files[i]
        if self.load:
            return torch.Tensor(load_map(file))
        return isfile(file)
    def timeit(self):
        tic = time.perf_counter()
        for a in self: pass
        tac =  time.perf_counter()
        print(f"{tac - tic:0.4f} seconds")
