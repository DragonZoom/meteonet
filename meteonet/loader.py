# Meteonet dataloader
# (c) 2023 D.Bereziat, Sorbonne Université

import torch
import numpy as np
from torch.utils.data import Dataset
from meteonet.utilities import next_date, load_map, map_to_classes, split_date
from os.path import dirname, basename, join, isfile
import time

epsilon = 1e-3 # As Antonia

class MeteonetDataset(Dataset):
    """ A class to load Meteonet data
        Limitations:
         - no multiple targets
        Difference with the vanilla dataloader
         - maps having missing values are no more ignored

      Stratégie:
         l'indexation a lieu pendant l'instantiation de la classe.
         on peut la sauvegarder dans un cache
         il n'y a plus de calcul dans getitem(), ce dernier se contente de
         charger directement les fichiers grâce à l'index.
         Quand une date manque, getitem() retourne None
         les dates manquantes sont consultats dans la variable missing_date de la classe.
    """
    def __init__(self, rainmaps, input_len = 12, target_pos = 18, stride = 12,
                 wind_dir=None, cached=None, tqdm=None, logging=False):
        """
        files: list of file name paths (will be sorted),

        windir: directory where wind maps are stored
                should contain two subdirectories, U/ and V/,

        input_len: number of maps to read as input of the model
                   recommended: 12 (stands for 1 hour),

        target_pos: position of target starting from the first map read as input
                    recommended: 18 = 12+6 (prevision horizon time at 6 = 30 minutes),

        stride: offset between each input sequence
                recommended: input_len (for no overlapping),

        cached: store dataset indexation in a cache file
        """
        files = sorted(rainmaps, key=lambda f:split_date(f))
        recalculate = True
        if cached and isfile(cached):
            obj = np.load(cached,allow_pickle=True)
            params = obj['arr_0'].reshape(-1)[0]
            if params['input_len'] == input_len and  params['stride'] == stride and \
               params['target_pos'] == target_pos and params['files'] == files and \
               (wind_dir == None or params['wind_dir'] == wind_dir):
                recalculate = False

        if recalculate:
            if tqdm: print('parameters changed, or cached file not found: indexing dataset, please wait...')
            params = {'files': files,
                      'input_len': input_len, 'target_pos': target_pos, 'stride': stride,
                      'wind_dir': wind_dir}
            maxs = []
            Umean = Vmean = 0.
            Uvar = Vvar = 0.
            size = 0.
            has_wind = []
            for f in tqdm(files, unit=' files') if tqdm else files:
                maxs.append( load_map(f).max())
                if wind_dir:
                    Upath = join(wind_dir,'U',basename(f))
                    Vpath = join(wind_dir,'V',basename(f))
                    if isfile(Upath) and isfile(Vpath):
                        U = np.array(load_map(Upath), dtype=float)
                        Umean += U.sum()
                        Uvar += (U**2).sum()
                        V = load_map(Vpath)*1.
                        Vmean += V.sum()
                        Vvar += (V**2).sum()
                        size += U.size
                        has_wind.append(True)
                    else:
                        has_wind.append(False)
            if wind_dir:
                Umean /= size
                Vmean /= size
                params['U_moments'] = Umean, np.sqrt(Uvar/size - Umean**2)
                params['V_moments'] = Vmean, np.sqrt(Vvar/size - Vmean**2)    

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
            params['has_wind'] = np.array(has_wind)
            params['missing_dates'] = missing_dates
            if cached:
                np.savez_compressed( cached, params)

        self.logging = logging
        self.tqdm = tqdm
        self.params = params
        self.norm_factors = [np.log(1+params['maxs'].max())]
        self.use_wind = False
                
        if wind_dir:
            for v in params['U_moments']: self.norm_factors.append(v)
            for v in params['V_moments']: self.norm_factors.append(v)
            self.use_wind = True

        self.do_not_read_map = False # for performances tests. could be removed
        
    def __len__(self):
        ## Question pour Anastase: on pourrait aussi avoir une fenêtre glissante sur les inputs_len ??
        ## le code de Vincent ne le prévoit pas: les inputs ne se superposent pas.
        return self.params['items'].shape[0]

    def read(self, idx):
        if self.do_not_read_map:
            return torch.zeros((1,1)), torch.zeros((1,1))
        ## Anastase: tensor() ou Tensor() ?
        rainmap = torch.Tensor(load_map(self.params['files'][idx]))
        return torch.log(rainmap.unsqueeze(0) + 1 + epsilon)/self.norm_factors[0], rainmap

    def read_wind(self, idx):
        if self.do_not_read_map:
            return torch.zeros((1,1)), torch.zeros((1,1))
        m,s = self.norm_factors[1], self.norm_factors[2]
        U = torch.Tensor(load_map( join(self.params['wind_dir'],'U',basename(self.params['files'][idx])))-m)/s
        m,s = self.norm_factors[3], self.norm_factors[4]        
        V = torch.Tensor(load_map( join(self.params['wind_dir'],'V',basename(self.params['files'][idx])))-m)/s
        return U.unsqueeze(0),V.unsqueeze(0)

    def __getitem__(self, i):
        item = self.params['items'][i]

        # Anastase: 
        # on peut changer ça, notamment uniquement si la target vaut -1.
        # rappel: un index à -1 indique une date manquante.
        # ces dates pourrait être remplacées par des images nulles,
        # avec un minimum de deux non nulles sans doute pour inférer la dynamique
        # if item[-1] == -1:
        if item.min() == -1:
            return None

        maps, _ = self.read(item[0])
        for idx in item[1:-1]:
            rmap, persistence = self.read(idx)
            maps = torch.cat((maps, rmap), dim=0)
        if self.use_wind:
            if not self.params['has_wind'][item[:-1]].all():
                return None
            Umaps, Vmaps = self.read_wind(item[0])
            for idx in item[1:-1]:                
                U,V = self.read_wind(idx)
                Umaps = torch.cat((Umaps, U), dim=0)
                Vmaps = torch.cat((Vmaps, V), dim=0)
            maps = torch.cat((maps, Umaps, Vmaps), dim=0)
        
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
