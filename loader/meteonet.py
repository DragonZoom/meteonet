# Meteonet dataloader
# (c) 2023 D.Bereziat, Sorbonne Université

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
        Difference with the vanilla dataloader
         - maps having missing values are no more ignored

      Stratégie:
         l'indexation a lieu pendant l'instantiation de la classe.
         on peut la sauvegarder dans un cache.
         il n'y a plus de calcul dans getitem(), ce dernier se contente de
         charger directement les fichiers grâce à l'index.
         Quand une date manque, getitem() retourne None
         les dates manquantes sont consultats dans la variable missing_date de la classe.
    """
    def __init__(self, files, input_len = 12, target_pos = 18, stride = 12, cached=False, tqdm=False, logging=False):
        """
        files: list of files (will be sorted)
        input_len: number of maps to read as input of the model
                   recommended: 12 (stands for 1 hour)
        target_pos: position of target starting from the first map read as input
                    recommended: 18 = 12+6 (prevision horizon time at 6 = 30 minutes)
        stride: offset between each input sequence
                recommended: input_len (for no overlapping)
        cached: store dataset indexation in a cache file
        """
        files = sorted(files, key=lambda f:split_date(f))
        recalculate = True
        if cached and isfile(cached):
            obj = np.load(cached,allow_pickle=True)
            params = obj['arr_0'].reshape(-1)[0]
            if params['input_len'] == input_len and  params['stride'] == stride and \
               params['target_pos'] == target_pos and params['files'] == files:
                recalculate = False

        if recalculate:
            if tqdm: print('parameters changed, or cached file not found: indexing dataset, please wait...')
            params = {'files': files,
                      'input_len': input_len, 'target_pos': target_pos, 'stride': stride}
            maxs = []
            for f in tqdm(files, unit=' files') if tqdm else files:
                maxs.append( load_map(f).max())                
            params['maxs'] = np.array(maxs)

            l = len(files)
            items = []
            dname = dirname(files[0])
            missing_dates = []
            for j in range(0, len(files), stride):
                item = [j]
                curr_date = basename( files[j])
                # check if the next input_len-1 files have correct dates
                j += 1
                num_obs = input_len - 1
                while num_obs and j<l:
                    next_available_date = basename(files[j])
                    curr_date = next_date(curr_date)
                    if curr_date == next_available_date:
                        item.append(j)
                        j += 1
                    else:
                        item.append(-1)
                        missing_dates.append(curr_date)
                    num_obs -= 1
                if j == l: break

                # get the target date
                pos = target_pos - input_len
                while pos:
                    curr_date = next_date(curr_date)
                    pos -= 1
                target_file = join(dname, curr_date)

                jend = min(j+target_pos, l)
                if target_file in files[j:jend]:
                    item.append( j+files[j:jend].index(target_file))
                else:
                    item.append(-1)
                    missing_dates.append(basename(target_file))
                items.append(item)

            params['items'] = np.array(items)
            params['missing_dates'] = missing_dates
            if cached:
                np.savez_compressed( cached, params)
        
        self.logging = logging
        self.tqdm = tqdm
        self.norm_factor = np.log(1+params['maxs'].max())
        self.params = params
        self.do_not_read_map = False
        
    def __len__(self):
        ## Question pour Anastase: on pourrait aussi avoir une fenêtre glissante sur les inputs_len ??
        ## le code de Vincent ne le prévoit pas: les inputs ne se superposent pas.
        return self.params['items'].shape[0]

    def read(self, idx):
        if self.do_not_read_map:
            return torch.zeros((1,1)), torch.zeros((1,1))
        file = self.params['files'][idx]
        ## Anastase: tensor() ou Tensor() ?
        rainmap = torch.Tensor(load_map(file))
        return torch.log(rainmap.unsqueeze(0) + 1 + epsilon)/self.norm_factor, rainmap

    def __getitem__(self, i):
        item = self.params['items'][i]

        # on peut changer ça, notamment uniquement si la target vaut -1.
        # if item[-1] == -1:
        if item.min() == -1:
            return None

        maps, _ = self.read(item[0])
        for idx in item[1:self.params['input_len']]:
            rmap, persistence = self.read(idx)
            maps = torch.cat((maps, rmap), dim=0)

        target_file = self.params['files'][item[-1]]
        
        return {
            'inputs': maps,
            'target': torch.Tensor(load_map(target_file)) if not self.do_not_read_map else torch.zeros(1),
            'target_name': target_file,
            'persistence': persistence
        }

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
